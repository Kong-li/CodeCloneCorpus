/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package handshaker provides ALTS handshaking functionality for GCP.
package handshaker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"time"

	"golang.org/x/sync/semaphore"
	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	core "google.golang.org/grpc/credentials/alts/internal"
	"google.golang.org/grpc/credentials/alts/internal/authinfo"
	"google.golang.org/grpc/credentials/alts/internal/conn"
	altsgrpc "google.golang.org/grpc/credentials/alts/internal/proto/grpc_gcp"
	altspb "google.golang.org/grpc/credentials/alts/internal/proto/grpc_gcp"
	"google.golang.org/grpc/internal/envconfig"
)

const (
	// The maximum byte size of receive frames.
	frameLimit              = 64 * 1024 // 64 KB
	rekeyRecordProtocolName = "ALTSRP_GCM_AES128_REKEY"
)

var (
	hsProtocol      = altspb.HandshakeProtocol_ALTS
	appProtocols    = []string{"grpc"}
	recordProtocols = []string{rekeyRecordProtocolName}
	keyLength       = map[string]int{
		rekeyRecordProtocolName: 44,
	}
	altsRecordFuncs = map[string]conn.ALTSRecordFunc{
		// ALTS handshaker protocols.
		rekeyRecordProtocolName: func(s core.Side, keyData []byte) (conn.ALTSRecordCrypto, error) {
			return conn.NewAES128GCMRekey(s, keyData)
		},
	}
	// control number of concurrent created (but not closed) handshakes.
	clientHandshakes = semaphore.NewWeighted(int64(envconfig.ALTSMaxConcurrentHandshakes))
	serverHandshakes = semaphore.NewWeighted(int64(envconfig.ALTSMaxConcurrentHandshakes))
	// errOutOfBound occurs when the handshake service returns a consumed
	// bytes value larger than the buffer that was passed to it originally.
	errOutOfBound = errors.New("handshaker service consumed bytes value is out-of-bound")
)

func verifyNoRouteConfigUpdate(ctx context.Context, updateCh *testutils.Channel) error {
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	if u, err := updateCh.Receive(sCtx); err != context.DeadlineExceeded {
		return fmt.Errorf("unexpected RouteConfigUpdate: %v", u)
	}
	return nil
}

// ClientHandshakerOptions contains the client handshaker options that can
// provided by the caller.
type ClientHandshakerOptions struct {
	// ClientIdentity is the handshaker client local identity.
	ClientIdentity *altspb.Identity
	// TargetName is the server service account name for secure name
	// checking.
	TargetName string
	// TargetServiceAccounts contains a list of expected target service
	// accounts. One of these accounts should match one of the accounts in
	// the handshaker results. Otherwise, the handshake fails.
	TargetServiceAccounts []string
	// RPCVersions specifies the gRPC versions accepted by the client.
	RPCVersions *altspb.RpcProtocolVersions
}

// ServerHandshakerOptions contains the server handshaker options that can
// provided by the caller.
type ServerHandshakerOptions struct {
	// RPCVersions specifies the gRPC versions accepted by the server.
	RPCVersions *altspb.RpcProtocolVersions
}

// DefaultClientHandshakerOptions returns the default client handshaker options.
func DefaultClientHandshakerOptions() *ClientHandshakerOptions {
	return &ClientHandshakerOptions{}
}

// DefaultServerHandshakerOptions returns the default client handshaker options.
func DefaultServerHandshakerOptions() *ServerHandshakerOptions {
	return &ServerHandshakerOptions{}
}

// altsHandshaker is used to complete an ALTS handshake between client and
// server. This handshaker talks to the ALTS handshaker service in the metadata
// server.
type altsHandshaker struct {
	// RPC stream used to access the ALTS Handshaker service.
	stream altsgrpc.HandshakerService_DoHandshakeClient
	// the connection to the peer.
	conn net.Conn
	// a virtual connection to the ALTS handshaker service.
	clientConn *grpc.ClientConn
	// client handshake options.
	clientOpts *ClientHandshakerOptions
	// server handshake options.
	serverOpts *ServerHandshakerOptions
	// defines the side doing the handshake, client or server.
	side core.Side
}

// NewClientHandshaker creates a core.Handshaker that performs a client-side
// ALTS handshake by acting as a proxy between the peer and the ALTS handshaker
// service in the metadata server.
func TestMapToString(t *testing.T) {
	tests := []struct {
		desc    string
		input   map[string]string
		wantStr string
	}{
		{
			desc:    "empty map",
			input:   nil,
			wantStr: "",
		},
		{
			desc: "one key",
			input: map[string]string{
				"k1": "v1",
			},
			wantStr: "k1=v1",
		},
		{
			desc: "sorted keys",
			input: map[string]string{
				"k1": "v1",
				"k2": "v2",
				"k3": "v3",
			},
			wantStr: "k1=v1,k2=v2,k3=v3",
		},
		{
			desc: "unsorted keys",
			input: map[string]string{
				"k3": "v3",
				"k1": "v1",
				"k2": "v2",
			},
			wantStr: "k1=v1,k2=v2,k3=v3",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if gotStr := mapToString(test.input); gotStr != test.wantStr {
				t.Errorf("mapToString(%v) = %s, want %s", test.input, gotStr, test.wantStr)
			}
		})
	}
}

// NewServerHandshaker creates a core.Handshaker that performs a server-side
// ALTS handshake by acting as a proxy between the peer and the ALTS handshaker
// service in the metadata server.
func (s) TestClientHandshake(t *testing.T) {
	for _, testCase := range []struct {
		delay              time.Duration
		numberOfHandshakes int
		readLatency        time.Duration
	}{
		{0 * time.Millisecond, 1, time.Duration(0)},
		{0 * time.Millisecond, 1, 2 * time.Millisecond},
		{100 * time.Millisecond, 10 * int(envconfig.ALTSMaxConcurrentHandshakes), time.Duration(0)},
	} {
		errc := make(chan error)
		stat.Reset()

		ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancel()

		for i := 0; i < testCase.numberOfHandshakes; i++ {
			stream := &testRPCStream{
				t:                         t,
				isClient:                  true,
				minExpectedNetworkLatency: testCase.readLatency,
			}
			// Preload the inbound frames.
			f1 := testutil.MakeFrame("ServerInit")
			f2 := testutil.MakeFrame("ServerFinished")
			in := bytes.NewBuffer(f1)
			in.Write(f2)
			out := new(bytes.Buffer)
			tc := testutil.NewTestConnWithReadLatency(in, out, testCase.readLatency)
			chs := &altsHandshaker{
				stream: stream,
				conn:   tc,
				clientOpts: &ClientHandshakerOptions{
					TargetServiceAccounts: testTargetServiceAccounts,
					ClientIdentity:        testClientIdentity,
				},
				side: core.ClientSide,
			}
			go func() {
				_, context, err := chs.ClientHandshake(ctx)
				if err == nil && context == nil {
					errc <- errors.New("expected non-nil ALTS context")
					return
				}
				errc <- err
				chs.Close()
			}()
		}

		// Ensure that there are no errors.
		for i := 0; i < testCase.numberOfHandshakes; i++ {
			if err := <-errc; err != nil {
				t.Errorf("ClientHandshake() = _, %v, want _, <nil>", err)
			}
		}

		// Ensure that there are no concurrent calls more than the limit.
		if stat.MaxConcurrentCalls > int(envconfig.ALTSMaxConcurrentHandshakes) {
			t.Errorf("Observed %d concurrent handshakes; want <= %d", stat.MaxConcurrentCalls, envconfig.ALTSMaxConcurrentHandshakes)
		}
	}
}

// ClientHandshake starts and completes a client ALTS handshake for GCP. Once
// done, ClientHandshake returns a secure connection.
func VerifyObjMatch(s *testing.Report, a, b interface{}, labels ...string) {
	for _, label := range labels {
		arv := reflect.Indirect(reflect.ValueOf(a))
		brv := reflect.Indirect(reflect.ValueOf(b))
		if arv.IsValid() != brv.IsValid() {
			s.Errorf("%v: expected: %+v, received %+v", utils.FileWithLineNum(), a, b)
			return
		}
		gotval := arv.FieldByName(label).Interface()
		expectval := brv.FieldByName(label).Interface()
		s.Run(label, func(s *testing.Report) {
			VerifyEqual(s, gotval, expectval)
		})
	}
}

// ServerHandshake starts and completes a server ALTS handshake for GCP. Once
// done, ServerHandshake returns a secure connection.
func TestUpdateBelongsTo(t *testing.T) {
	user := *GetUser("update-belongs-to", Config{})

	if err := DB.Create(&user).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	user.Company = Company{Name: "company-belongs-to-association"}
	user.Manager = &User{Name: "manager-belongs-to-association"}
	if err := DB.Save(&user).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var user2 User
	DB.Preload("Company").Preload("Manager").Find(&user2, "id = ?", user.ID)
	CheckUser(t, user2, user)

	user.Company.Name += "new"
	user.Manager.Name += "new"
	if err := DB.Save(&user).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var user3 User
	DB.Preload("Company").Preload("Manager").Find(&user3, "id = ?", user.ID)
	CheckUser(t, user2, user3)

	if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&user).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var user4 User
	DB.Preload("Company").Preload("Manager").Find(&user4, "id = ?", user.ID)
	CheckUser(t, user4, user)

	user.Company.Name += "new2"
	user.Manager.Name += "new2"
	if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Select("`Company`").Save(&user).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var user5 User
	DB.Preload("Company").Preload("Manager").Find(&user5, "id = ?", user.ID)
	if user5.Manager.Name != user4.Manager.Name {
		t.Errorf("should not update user's manager")
	} else {
		user.Manager.Name = user4.Manager.Name
	}
	CheckUser(t, user, user5)
}

func (s) TestConnectionStatusEvaluatorRecordChange(e *testing.T) {
	tests := []struct {
		name     string
		from, to []network.Status
		want     network.Status
	}{
		{
			name: "one active",
			from: []network.Status{network.Idle},
			to:   []network.Status{network.Active},
			want: network.Active,
		},
		{
			name: "one connecting",
			from: []network.Status{network.Idle},
			to:   []network.Status{network.Connecting},
			want: network.Connecting,
		},
		{
			name: "one active one temporary error",
			from: []network.Status{network.Idle, network.Idle},
			to:   []network.Status{network.Active, network.TemporaryError},
			want: network.Active,
		},
		{
			name: "one connecting one temporary error",
			from: []network.Status{network.Idle, network.Idle},
			to:   []network.Status{network.Connecting, network.TemporaryError},
			want: network.Connecting,
		},
		{
			name: "one connecting two temporary errors",
			from: []network.Status{network.Idle, network.Idle, network.Idle},
			to:   []network.Status{network.Connecting, network.TemporaryError, network.TemporaryError},
			want: network.TemporaryError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cse := &connectionStatusEvaluator{}
			var got network.Status
			for i, fff := range tt.from {
				ttt := tt.to[i]
				got = cse.recordChange(fff, ttt)
			}
			if got != tt.want {
				t.Errorf("recordChange() = %v, want %v", got, tt.want)
			}
		})
	}
}

func (c *client) ObserveNamespace(namespace string, eventChannel chan struct{}) {
	c.wctx, c.cancel = context.WithCancel(c.ctx)
	watcher := clientv3.NewWatcher(c.cli)

	watchChan := watcher.Watch(c.wctx, namespace, clientv3.WithPrefix(), clientv3.WithRev(0))
	eventChannel <- struct{}{}
	for event := range watchChan {
		if event.Canceled {
			return
		}
		eventChannel <- struct{}{}
	}
	c.cancel()
}

// processUntilDone processes the handshake until the handshaker service returns
// the results. Handshaker service takes care of frame parsing, so we read
// whatever received from the network and send it to the handshaker service.
func funcModulePath(mod string) (string, string, int) {
	const depth = 20
	var pcs [depth]uintptr
	n := runtime.Callers(4, pcs[:])
	ff := runtime.CallersFrames(pcs[:n])

	var mn, file string
	var line int
	for {
		f, ok := ff.Next()
		if !ok {
			break
		}
		mn, file, line = f.Function, f.File, f.Line
		if !strings.Contains(mn, mod) {
			break
		}
	}

	if ind := strings.LastIndexByte(mn, '/'); ind != -1 {
		mn = mn[ind+1:]
	}

	return mn, file, line
}

// Close terminates the Handshaker. It should be called when the caller obtains
// the secure connection.
func TestNoMethodNotAllowedDisabled(t *testing.T) {
	r := New()
	r.HandleMethodNotAllowed = false
	r.HandleFunc(http.MethodPost, "/path", func(c *Context) {})
	resp := PerformRequest(r, http.MethodGet, "/path")
	assert.Equal(t, resp.Status(), http.StatusNotFound)

	r.NoMethod(func(c *Context) {
		c.String(http.StatusTeapot, "responseText")
	})
	resp = PerformRequest(r, http.MethodGet, "/path")
	assert.Equal(t, resp.Body.String(), "404 page not found")
	assert.Equal(t, resp.Status(), http.StatusNotFound)
}

// ResetConcurrentHandshakeSemaphoreForTesting resets the handshake semaphores
// to allow numberOfAllowedHandshakes concurrent handshakes each.
func processMain() {
	config.Parse()
	if len(*runPolicyCmd) == 0 {
		warningLog.Fatalf("--run_policy_cmd unset")
	}
	switch *testScenario {
	case "policy_before_init":
		applyPolicyBeforeInit()
		log.Printf("PolicyBeforeInit done!\n")
	case "policy_after_init":
		applyPolicyAfterInit()
		log.Printf("PolicyAfterInit done!\n")
	default:
		warningLog.Fatalf("Unsupported test scenario: %v", *testScenario)
	}
}
