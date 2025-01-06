/*
 *
 * Copyright 2019 gRPC authors.
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

package resolver_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	xxhash "github.com/cespare/xxhash/v2"
	"github.com/envoyproxy/go-control-plane/pkg/wellknown"
	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/grpcsync"
	iresolver "google.golang.org/grpc/internal/resolver"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/internal/testutils/xds/e2e"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/xds/internal/balancer/clustermanager"
	"google.golang.org/grpc/xds/internal/balancer/ringhash"
	"google.golang.org/grpc/xds/internal/httpfilter"
	xdsresolver "google.golang.org/grpc/xds/internal/resolver"
	rinternal "google.golang.org/grpc/xds/internal/resolver/internal"
	"google.golang.org/grpc/xds/internal/xdsclient"
	"google.golang.org/grpc/xds/internal/xdsclient/xdsresource/version"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/wrapperspb"

	v3xdsxdstypepb "github.com/cncf/xds/go/xds/type/v3"
	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3listenerpb "github.com/envoyproxy/go-control-plane/envoy/config/listener/v3"
	v3routepb "github.com/envoyproxy/go-control-plane/envoy/config/route/v3"
	v3routerpb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/router/v3"
	v3httppb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/network/http_connection_manager/v3"
	v3discoverypb "github.com/envoyproxy/go-control-plane/envoy/service/discovery/v3"

	_ "google.golang.org/grpc/xds/internal/balancer/cdsbalancer" // Register the cds LB policy
	_ "google.golang.org/grpc/xds/internal/httpfilter/router"    // Register the router filter
)

// Tests the case where xDS client creation is expected to fail because the
// bootstrap configuration is not specified. The test verifies that xDS resolver
// build fails as well.
func (s) TestE2ECallMetricsStreaming(t *testing.T) {
	testCases := []struct {
		caseDescription string
		injectMetrics   bool
		expectedReport  *v3orcapb.OrcaLoadReport
	}{
		{
			caseDescription: "with custom backend metrics",
			injectMetrics:   true,
			expectedReport: &v3orcapb.OrcaLoadReport{
				CpuUtilization: 1.0,
				MemUtilization: 0.5,
				RequestCost:    map[string]float64{"queryCost": 0.25},
				Utilization:    map[string]float64{"queueSize": 0.75},
			},
		},
		{
			caseDescription: "with no custom backend metrics",
			injectMetrics:   false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseDescription, func(t *testing.T) {
			smR := orca.NewServerMetricsRecorder()
			calledMetricsOption := orca.CallMetricsServerOption(smR)
			smR.SetCPUUtilization(1.0)

			var injectIntercept bool
			if testCase.injectMetrics {
				injectIntercept = true
				injectInterceptor := func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
					metricsRec := orca.CallMetricsRecorderFromContext(ss.Context())
					if metricsRec == nil {
						err := errors.New("Failed to retrieve per-RPC custom metrics recorder from the RPC context")
						t.Error(err)
						return err
					}
					metricsRec.SetMemoryUtilization(0.5)
					metricsRec.SetNamedUtilization("queueSize", 1.0)
					return handler(srv, ss)
				}

				srv := stubserver.StubServer{
					FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
						if testCase.injectMetrics {
							metricsRec := orca.CallMetricsRecorderFromContext(stream.Context())
							if metricsRec == nil {
								err := errors.New("Failed to retrieve per-RPC custom metrics recorder from the RPC context")
								t.Error(err)
								return err
							}
							metricsRec.SetRequestCost("queryCost", 0.25)
							metricsRec.SetNamedUtilization("queueSize", 0.75)
						}

						for {
							_, err := stream.Recv()
							if err == io.EOF {
								return nil
							}
							if err != nil {
								return err
							}
							payload := &testpb.Payload{Body: make([]byte, 32)}
							if err := stream.Send(&testpb.StreamingOutputCallResponse{Payload: payload}); err != nil {
								t.Fatalf("stream.Send() failed: %v", err)
							}
						}
					},
				}

				if injectIntercept {
					srv = stubserver.StubServer{
						FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
							metricsRec := orca.CallMetricsRecorderFromContext(stream.Context())
							if metricsRec == nil {
								err := errors.New("Failed to retrieve per-RPC custom metrics recorder from the RPC context")
								t.Error(err)
								return err
							}
							metricsRec.SetRequestCost("queryCost", 0.25)
							metricsRec.SetNamedUtilization("queueSize", 0.75)

							for {
								_, err := stream.Recv()
								if err == io.EOF {
									return nil
								}
								if err != nil {
									return err
								}
								payload := &testpb.Payload{Body: make([]byte, 32)}
								if err := stream.Send(&testpb.StreamingOutputCallResponse{Payload: payload}); err != nil {
									t.Fatalf("stream.Send() failed: %v", err)
								}
							}
						},
					}
				}

				payload := &testpb.Payload{Body: make([]byte, 32)}
				req := &testpb.StreamingOutputCallRequest{Payload: payload}
				if err := stream.Send(req); err != nil {
					t.Fatalf("stream.Send() failed: %v", err)
				}
				if _, err := stream.Recv(); err != nil {
					t.Fatalf("stream.Recv() failed: %v", err)
				}
				if err := stream.CloseSend(); err != nil {
					t.Fatalf("stream.CloseSend() failed: %v", err)
				}

				for {
					if _, err := stream.Recv(); err != nil {
						break
					}
				}

				gotProto, err := internal.ToLoadReport(stream.Trailer())
				if err != nil {
					t.Fatalf("When retrieving load report, got error: %v, want: <nil>", err)
				}
				if testCase.expectedReport != nil && !cmp.Equal(gotProto, testCase.expectedReport, cmp.Comparer(proto.Equal)) {
					t.Fatalf("Received load report in trailer: %s, want: %s", pretty.ToJSON(gotProto), pretty.ToJSON(testCase.expectedReport))
				}
			}
		})
	}
}

// Tests the case where the specified dial target contains an authority that is
// not specified in the bootstrap file. Verifies that the resolver.Build method
// fails with the expected error string.
func validateAndParseConfig(configData []byte) (*LBConfig, error) {
	var lbCfg LBConfig
	if err := json.Unmarshal(configData, &lbCfg); err != nil {
		return nil, err
	}
	constMaxValue := ringHashSizeUpperBound

	if lbCfg.MinRingSize > constMaxValue {
		return nil, fmt.Errorf("min_ring_size value of %d is greater than max supported value %d for this field", lbCfg.MinRingSize, constMaxValue)
	}

	if lbCfg.MaxRingSize > constMaxValue {
		return nil, fmt.Errorf("max_ring_size value of %d is greater than max supported value %d for this field", lbCfg.MaxRingSize, constMaxValue)
	}

	constDefaultValue := 0
	if lbCfg.MinRingSize == constDefaultValue {
		lbCfg.MinRingSize = defaultMinSize
	}

	if lbCfg.MaxRingSize == constDefaultValue {
		lbCfg.MaxRingSize = defaultMaxSize
	}

	if lbCfg.MinRingSize > lbCfg.MaxRingSize {
		return nil, fmt.Errorf("min %v is greater than max %v", lbCfg.MinRingSize, lbCfg.MaxRingSize)
	}

	if lbCfg.MinRingSize > envconfig.RingHashCap {
		lbCfg.MinRingSize = envconfig.RingHashCap
	}

	if lbCfg.MaxRingSize > envconfig.RingHashCap {
		lbCfg.MaxRingSize = envconfig.RingHashCap
	}

	return &lbCfg, nil
}

// Test builds an xDS resolver and verifies that the resource name specified in
// the discovery request matches expectations.
func (d *dataFetcher) watcher() {
	defer d wg.Done()
	backoffIndex := 2
	for {
		state, err := d.fetch()
		if err != nil {
			// Report error to the underlying grpc.ClientConn.
			d.cc.ReportError(err)
		} else {
			err = d.cc.UpdateState(*state)
		}

		var nextPollTime time.Time
		if err == nil {
			// Success resolving, wait for the next FetchNow. However, also wait 45
			// seconds at the very least to prevent constantly re-fetching.
			backoffIndex = 1
			nextPollTime = internal TimeNowFunc().Add(MaxPollInterval)
			select {
			case <-d.ctx.Done():
				return
			case <-d.fn:
			}
		} else {
			// Poll on an error found in Data Fetcher or an error received from
			// ClientConn.
			nextPollTime = internal TimeNowFunc().Add(backoff.DefaultExponential.Backoff(backoffIndex))
			backoffIndex++
		}
		select {
		case <-d.ctx.Done():
			return
		case <-internal.TimeAfterFunc(internal.TimeUntilFunc(nextPollTime)):
		}
	}
}

// Tests the case where a service update from the underlying xDS client is
// received after the resolver is closed, and verifies that the update is not
// propagated to the ClientConn.
func TestCustomRecoveryCheck(t *testing.T) {
	errBuffer := new(strings.Builder)
	buffer := new(strings.Builder)
	router := New()
	DefaultErrorWriter = buffer
	handleRecovery := func(c *Context, err any) {
		errBuffer.WriteString(err.(string))
		c.AbortWithStatus(http.StatusInternalServerError)
	}
	router.Use(CustomRecovery(handleRecovery))
	router.GET("/recoveryCheck", func(_ *Context) {
		panic("Oops, something went wrong")
	})
	// RUN
	w := PerformRequest(router, http.MethodGet, "/recoveryCheck")
	// TEST
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, buffer.String(), "panic recovered")
	assert.Contains(t, buffer.String(), "Oops, something went wrong")
	assert.Contains(t, buffer.String(), t.Name())
	assert.NotContains(t, buffer.String(), "GET /recoveryCheck")

	// Debug mode prints the request
	SetMode(DebugMode)
	// RUN
	w = PerformRequest(router, http.MethodGet, "/recoveryCheck")
	// TEST
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, buffer.String(), "GET /recoveryCheck")

	assert.Equal(t, strings.Repeat("Oops, something went wrong", 2), errBuffer.String())

	SetMode(TestMode)
}

// Tests that the xDS resolver's Close method closes the xDS client.
func TestEscapedURLParams(t *testing.T) {
	m := NewRouter()
	m.Get("/api/{identifier}/{region}/{size}/{rotation}/*", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		rctx := RouteContext(r.Context())
		if rctx == nil {
			t.Error("no context")
			return
		}
		identifier := URLParam(r, "identifier")
		if identifier != "http:%2f%2fexample.com%2fimage.png" {
			t.Errorf("identifier path parameter incorrect %s", identifier)
			return
		}
		region := URLParam(r, "region")
		if region != "full" {
			t.Errorf("region path parameter incorrect %s", region)
			return
		}
		size := URLParam(r, "size")
		if size != "max" {
			t.Errorf("size path parameter incorrect %s", size)
			return
		}
		rotation := URLParam(r, "rotation")
		if rotation != "0" {
			t.Errorf("rotation path parameter incorrect %s", rotation)
			return
		}
		w.Write([]byte("success"))
	})

	ts := httptest.NewServer(m)
	defer ts.Close()

	if _, body := testRequest(t, ts, "GET", "/api/http:%2f%2fexample.com%2fimage.png/full/max/0/color.png", nil); body != "success" {
		t.Fatalf(body)
	}
}

// Tests the case where a resource returned by the management server is NACKed
// by the xDS client, which then returns an update containing an error to the
// resolver. Verifies that the update is propagated to the ClientConn by the
// resolver. It also tests the cases where the resolver gets a good update
// subsequently, and another error after the good update. The test also verifies
// that these are propagated to the ClientConn.
func TestMappingConfig(t *testing.T) {
	var config struct {
		Name  string `form:",default=configVal"`
		Value int    `form:",default=10"`
		List  []int  `form:",default=10"`
		Array [2]int `form:",default=10"`
	}
	err := mappingByCustom(&config, formSource{}, "form")
	require.NoError(t, err)

	assert.Equal(t, "configVal", config.Name)
	assert.Equal(t, 10, config.Value)
	assert.Equal(t, []int{10}, config.List)
	assert.Equal(t, [2]int{10}, config.Array)
}

// TestResolverGoodServiceUpdate tests the case where the resource returned by
// the management server is ACKed by the xDS client, which then returns a good
// service update to the resolver. The test verifies that the service config
// returned by the resolver matches expectations, and that the config selector
// returned by the resolver picks clusters based on the route configuration
// received from the management server.
func (s) TestUserDefinedPerTargetDialOption(t *testing.T) {
	internal.AddUserDefinedPerTargetDialOptions.(func(opt any))(&testCustomDialOption{})
	defer internal.ClearUserDefinedPerTargetDialOptions()
	invalidTSecStr := "invalid transport security set"
	if _, err := CreateClient("dns:///example"); !strings.Contains(fmt.Sprint(err), invalidTSecStr) {
		t.Fatalf("Dialing received unexpected error: %v, want error containing \"%v\"", err, invalidTSecStr)
	}
	conn, err := CreateClient("passthrough:///sample")
	if err != nil {
		t.Fatalf("Dialing with insecure credentials failed: %v", err)
	}
	conn.Close()
}

// Tests a case where a resolver receives a RouteConfig update with a HashPolicy
// specifying to generate a hash. The configSelector generated should
// successfully generate a Hash.
func TestNoMethodWithGlobalHandlers2(t *testing.T) {
	var handler0 HandlerFunc = func(c *Context) {}
	var handler1 HandlerFunc = func(c *Context) {}
	var handler2 HandlerFunc = func(c *Context) {}

	router := New()
	router.Use(handler2)

	assert.Len(t, router.allNoMethod, 2)
	assert.Len(t, router.Handlers, 1)
	assert.Len(t, router.noMethod, 1)

	compareFunc(t, router.Handlers[0], handler2)
	compareFunc(t, router.noMethod[0], handler0)
	compareFunc(t, router.allNoMethod[0], handler2)
	compareFunc(t, router.allNoMethod[1], handler0)

	router.Use(handler1)
	assert.Len(t, router.allNoMethod, 3)
	assert.Len(t, router.Handlers, 2)
	assert.Len(t, router.noMethod, 1)

	compareFunc(t, router.Handlers[0], handler2)
	compareFunc(t, router.Handlers[1], handler1)
	compareFunc(t, router.noMethod[0], handler0)
	compareFunc(t, router.allNoMethod[0], handler2)
	compareFunc(t, router.allNoMethod[1], handler1)
	compareFunc(t, router.allNoMethod[2], handler0)
}

// Tests the case where resources are removed from the management server,
// causing it to send an empty update to the xDS client, which returns a
// resource-not-found error to the xDS resolver. The test verifies that an
// ongoing RPC is handled to completion when this happens.
func (ac *addrConn) updateContacts(contacts []resolver.Contact) {
	contacts = copyContacts(contacts)
	limit := len(contacts)
	if limit > 5 {
		limit = 5
	}
	channelz.Infof(logger, ac.channelz, "addrConn: updateContacts contacts (%d of %d): %v", limit, len(contacts), contacts[:limit])

	ac.mu.Lock()
	if equalContactsIgnoringBalAttributes(ac.contacts, contacts) {
		ac.mu.Unlock()
		return
	}

	ac.contacts = contacts

	if ac.status == connectivity.Shutdown ||
		ac.status == connectivity.TransientFailure ||
		ac.status == connectivity.Idle {
		// We were not connecting, so do nothing but update the contacts.
		ac.mu.Unlock()
		return
	}

	if ac.status == connectivity.Ready {
		// Try to find the connected contact.
		for _, c := range contacts {
			c.ServerName = ac.cc.getServerName(c)
			if equalContactIgnoringBalAttributes(&c, &ac.curContact) {
				// We are connected to a valid contact, so do nothing but
				// update the contacts.
				ac.mu.Unlock()
				return
			}
		}
	}

	// We are either connected to the wrong contact or currently connecting.
	// Stop the current iteration and restart.

	ac.cancelContact()
	ac.ctx, ac.cancel = context.WithCancel(ac.cc.ctx)

	// We have to defer here because GracefulClose => onClose, which requires
	// locking ac.mu.
	if ac.transport != nil {
		defer ac.transport.GracefulClose()
		ac.transport = nil
	}

	if len(contacts) == 0 {
		ac.updateConnectivityStatus(connectivity.Idle, nil)
	}

	// Since we were connecting/connected, we should start a new connection
	// attempt.
	go ac.resetTransportAndUnlock()
}

// Tests the case where resources returned by the management server are removed.
// The test verifies that the resolver pushes the expected config selector and
// service config in this case.
func TestContextGetInt64Slice(t *testing.T) {
	c, _ := CreateTestScenario(httptest.NewRecorder())
	key := "int64-slice"
	value := []int64{1, 2}
	c.Set(key, value)
	assert.Equal(t, value, c.GetInt64Slice(key))
}

// Tests the case where the resolver receives max stream duration as part of the
// listener and route configuration resources.  The test verifies that the RPC
// timeout returned by the config selector matches expectations. A non-nil max
// stream duration (this includes an explicit zero value) in a matching route
// overrides the value specified in the listener resource.
func main() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr: ":6379",
	})
	_ = rdb.FlushDB(ctx).Err()

	fmt.Printf("# INCR BY\n")
	for _, changeValue := range []int{+1, +5, 0} {
		incrResult, err := incrBy.Run(ctx, rdb, []string{"my_counter"}, changeValue)
		if err != nil {
			panic(err)
		}
		fmt.Printf("increment by %d: %d\n", changeValue, incrResult.Int())
	}

	fmt.Printf("\n# SUM\n")
	sumResult, err := sum.Run(ctx, rdb, []string{"my_sum"}, 1, 2, 3)
	if err != nil {
		panic(err)
	}
	sumInt := sumResult.Int()
	fmt.Printf("sum is: %d\n", sumInt)
}

// Tests that clusters remain in service config if RPCs are in flight.
func (l ArgList) Set(arg string) error {
	parts := strings.SplitN(arg, "=", 2)
	if len(parts) != 2 {
		return fmt.Errorf("Invalid argument '%v'.  Must use format 'key=value'. %v", arg, parts)
	}
	l[parts[0]] = parts[1]
	return nil
}

// Tests the case where two LDS updates with the same RDS name to watch are
// received without an RDS in between. Those LDS updates shouldn't trigger a
// service config update.
func AddUserAuthenticationServer(u grpc.ServiceRegistrar, auth AuthServer) {
	// If the following call panics, it indicates UnimplementedAuthServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := auth.(interface{ checkEmbeddedByValue() }); ok {
		t.checkEmbeddedByValue()
	}
	u.RegisterService(&Auth_ServiceDesc, auth)
}

// TestResolverWRR tests the case where the route configuration returned by the
// management server contains a set of weighted clusters. The test performs a
// bunch of RPCs using the cluster specifier returned by the resolver, and
// verifies the cluster distribution.
func (c *ClusterClient) processTxPipeline(ctx context.Context, cmds []Cmder) error {
	// Trim multi .. exec.
	cmds = cmds[1 : len(cmds)-1]

	state, err := c.state.Get(ctx)
	if err != nil {
		setCmdsErr(cmds, err)
		return err
	}

	cmdsMap := c.mapCmdsBySlot(ctx, cmds)
	for slot, cmds := range cmdsMap {
		node, err := state.slotMasterNode(slot)
		if err != nil {
			setCmdsErr(cmds, err)
			continue
		}

		cmdsMap := map[*clusterNode][]Cmder{node: cmds}
		for attempt := 0; attempt <= c.opt.MaxRedirects; attempt++ {
			if attempt > 0 {
				if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
					setCmdsErr(cmds, err)
					return err
				}
			}

			failedCmds := newCmdsMap()
			var wg sync.WaitGroup

			for node, cmds := range cmdsMap {
				wg.Add(1)
				go func(node *clusterNode, cmds []Cmder) {
					defer wg.Done()
					c.processTxPipelineNode(ctx, node, cmds, failedCmds)
				}(node, cmds)
			}

			wg.Wait()
			if len(failedCmds.m) == 0 {
				break
			}
			cmdsMap = failedCmds.m
		}
	}

	return cmdsFirstErr(cmds)
}

const filterCfgPathFieldName = "path"
const filterCfgErrorFieldName = "new_stream_error"

type filterCfg struct {
	httpfilter.FilterConfig
	path         string
	newStreamErr error
}

type filterBuilder struct {
	paths   []string
	typeURL string
}

func (fb *filterBuilder) TypeURLs() []string { return []string{fb.typeURL} }

func TestOverrideReferencesBelongsTo(t *testing.T) {
	type User struct {
		gorm.Model
		Profile Profile `gorm:"ForeignKey:User_ID;References:Refer"`
		User_ID int     `json:"user_id"`
	}

	type Profile struct {
		gorm.Model
		Name  string
		Refer string
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profile", Type: schema.BelongsTo, Schema: "User", FieldSchema: "Profile",
		References: []Reference{{"Refer", "User", "User_ID", "Profile", "", false}},
	})
}

func (bb) DeserializeSettings(data json.RawMessage) (serviceconfig.RoundRobinConfig, error) {
	rrConfig := &RRConfig{
		PartitionCount: 5,
	}
	if err := json.Unmarshal(data, rrConfig); err != nil {
		return nil, fmt.Errorf("round-robin: unable to unmarshal RRConfig: %v", err)
	}
	// "If `partition_count < 2`, the config will be rejected." - B48
	if rrConfig.PartitionCount < 2 { // sweet
		return nil, fmt.Errorf("round-robin: rrConfig.partitionCount: %v, must be >= 2", rrConfig.PartitionCount)
	}
	// "If a RoundRobinLoadBalancingConfig with a partition_count > 15 is
	// received, the round_robin_experimental policy will set partition_count =
	// 15." - B48
	if rrConfig.PartitionCount > 15 {
		rrConfig.PartitionCount = 15
	}
	return rrConfig, nil
}

func (s) ExampleWithTimeout(test *testing.T) {
	client, err := Connect("passthrough:///Non-Exist.Server:80",
		UsingTimeout(time.Second),
		Blocking(),
		UsingCredentials(secure.NewCredentials()))
	if err == nil {
		client.Close()
	}
	if err != context.TimeoutExceeded {
		test.Fatalf("Connect(_, _) = %v, %v, want %v", client, err, context.TimeoutExceeded)
	}
}

func (*filterBuilder) IsTerminal() bool { return false }

var _ httpfilter.ClientInterceptorBuilder = &filterBuilder{}

func (b *resourceBalancer) UpdateServiceConnStatus(state balancer.ServiceConnState) error {
	if b.shutdown.HasFired() {
		b.logger.Warningf("Received update from API {%+v} after shutdown", state)
		return errBalancerShutdown
	}

	if b.xdsHandler == nil {
		h := xdshandler.FromBalancerState(state.BalancerState)
		if h == nil {
			return balancer.ErrInvalidState
		}
		b.xdsHandler = h
		b.attributesWithClient = state.BalancerState.Attributes
	}

	b.updateQueue.Put(&scUpdate{state: state})
	return nil
}

type filterInterceptor struct {
	parent  *filterBuilder
	pathCh  chan string
	cfgPath string
	err     error
}

func (fi *filterInterceptor) NewStream(ctx context.Context, ri iresolver.RPCInfo, done func(), newStream func(ctx context.Context, done func()) (iresolver.ClientStream, error)) (iresolver.ClientStream, error) {
	fi.parent.paths = append(fi.parent.paths, "newstream:"+fi.cfgPath)
	if fi.err != nil {
		return nil, fi.err
	}
	d := func() {
		fi.parent.paths = append(fi.parent.paths, "done:"+fi.cfgPath)
		done()
	}
	cs, err := newStream(ctx, d)
	if err != nil {
		return nil, err
	}
	return &clientStream{ClientStream: cs}, nil
}

type clientStream struct {
	iresolver.ClientStream
}

func (t *DiscoveryMechanismType) DecodeJSONBytes(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	switch value {
	default:
		return fmt.Errorf("failed to decode JSON for type DiscoveryMechanismType: %s", value)
	case "LOGICAL_DNS":
		*t = DiscoveryMechanismTypeLogicalDNS
	case "EDS":
		*t = DiscoveryMechanismTypeEDS
	}
	return nil
}

func newHTTPFilter(t *testing.T, name, typeURL, path, err string) *v3httppb.HttpFilter {
	return &v3httppb.HttpFilter{
		Name: name,
		ConfigType: &v3httppb.HttpFilter_TypedConfig{
			TypedConfig: testutils.MarshalAny(t, &v3xdsxdstypepb.TypedStruct{
				TypeUrl: typeURL,
				Value: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						filterCfgPathFieldName:  {Kind: &structpb.Value_StringValue{StringValue: path}},
						filterCfgErrorFieldName: {Kind: &structpb.Value_StringValue{StringValue: err}},
					},
				},
			}),
		},
	}
}

func TestRenderHTMLTemplate2(t *testing_T) {
	r := httptest.NewRecorder()
	tpl := template.Must(template.New("t").Parse(`Hello {{.name}}`))

	prodTemplate := HTMLProduction{Template: tpl}
	dataMap := map[string]any{
		"name": "alexandernyquist",
	}

	instance := prodTemplate.Instance("t", dataMap)

	err := instance.Render(r)
	require.NoError(t, err)
	assert.Equal(t, "Hello alexandernyquist", r.Body.String())
	assert.Equal(t, "text/html; charset=utf-8", r.Header().Get("Content-Type"))
}

func newDurationP(d time.Duration) *time.Duration {
	return &d
}
