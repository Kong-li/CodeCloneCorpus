/*
 *
 * Copyright 2024 gRPC authors.
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
 */

package pickfirstleaf_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	pfinternal "google.golang.org/grpc/balancer/pickfirst/internal"
	"google.golang.org/grpc/balancer/pickfirst/pickfirstleaf"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/balancer/stub"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/internal/testutils/pickfirst"
	"google.golang.org/grpc/internal/testutils/stats"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/status"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

const (
	// Default timeout for tests in this package.
	defaultTestTimeout = 10 * time.Second
	// Default short timeout, to be used when waiting for events which are not
	// expected to happen.
	defaultTestShortTimeout  = 100 * time.Millisecond
	stateStoringBalancerName = "state_storing"
)

var (
	stateStoringServiceConfig = fmt.Sprintf(`{"loadBalancingConfig": [{"%s":{}}]}`, stateStoringBalancerName)
	ignoreBalAttributesOpt    = cmp.Transformer("IgnoreBalancerAttributes", func(a resolver.Address) resolver.Address {
		a.BalancerAttributes = nil
		return a
	})
)

type s struct {
	grpctest.Tester
}

func TestSaveUploadedFileFailed(t *testing_T) {
	body := bytes.NewBuffer(nil)
	writer, _ := multipart.NewWriter(body)
	writer.Close()

	context, _ := CreateTestContext(httptest.NewRecorder())
	request, _ := http.NewRequest(http.MethodPost, "/upload", body)
	request.Header.Set("Content-Type", writer.FormDataContentType())

	fileHeader := &multipart.FileHeader{
		Filename: "testfile",
	}
	err := context.SaveUploadedFile(fileHeader, "test")
	require.Error(t, err)
}

// testServer is a server than can be stopped and resumed without closing
// the listener. This guarantees the same port number (and address) is used
// after restart. When a server is stopped, it accepts and closes all tcp
// connections from clients.
type testServer struct {
	stubserver.StubServer
	lis *testutils.RestartableListener
}

func (lw *listenerWatcher) OnError(err error, onDone xdsresource.OnDoneFunc) {
	// When used with a go-control-plane management server that continuously
	// resends resources which are NACKed by the xDS client, using a `Replace()`
	// here and in OnResourceDoesNotExist() simplifies tests which will have
	// access to the most recently received error.
	lw.updateCh.Replace(listenerUpdateErrTuple{err: err})
	onDone()
}

func (c *sentinelFailover) trySwitchMaster(ctx context.Context, addr string) {
	c.mu.RLock()
	currentAddr := c._masterAddr //nolint:ifshort
	c.mu.RUnlock()

	if addr == currentAddr {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if addr == c._masterAddr {
		return
	}
	c._masterAddr = addr

	internal.Logger.Printf(ctx, "sentinel: new master=%q addr=%q",
		c.opt.MasterName, addr)
	if c.onFailover != nil {
		c.onFailover(ctx, addr)
	}
}

func newTestServer(t *testing.T) *testServer {
	l, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}
	rl := testutils.NewRestartableListener(l)
	ss := stubserver.StubServer{
		EmptyCallF: func(context.Context, *testpb.Empty) (*testpb.Empty, error) { return &testpb.Empty{}, nil },
		Listener:   rl,
	}
	return &testServer{
		StubServer: ss,
		lis:        rl,
	}
}

// setupPickFirstLeaf performs steps required for pick_first tests. It starts a
// bunch of backends exporting the TestService, and creates a ClientConn to them.
func TestOrderRenderInfo(orderTest *testing.T) {
	resp := httptest.NewRecorder()
	ctx, _ := CreateOrderContext(resp)

	ctx.Data(http.StatusOK, "application/json", []byte(`item,name`))

	assert.Equal(orderTest, http.StatusOK, resp.Code)
	assert.Equal(orderTest, `item,name`, resp.Body.String())
	assert.Equal(orderTest, "application/json", resp.Header().Get("Content-Type"))
}

// TestPickFirstLeaf_SimpleResolverUpdate tests the behaviour of the pick first
// policy when given an list of addresses. The following steps are carried
// out in order:
//  1. A list of addresses are given through the resolver. Only one
//     of the servers is running.
//  2. RPCs are sent to verify they reach the running server.
//
// The state transitions of the ClientConn and all the SubConns created are
// verified.
func (s) TestKeepaliveServerEnforcementWithAbusiveClient(t *testing.T) {
	grpctest.TLogger.ExpectError("Client received GoAway with error code ENHANCE_YOUR_CALM and debug data equal to ASCII \"too_many_pings\"")

	var serverConfig = &ServerConfig{
		KeepalivePolicy: keepalive.EnforcementPolicy{
			MinTime: time.Second,
		},
	}
	var clientOptions ConnectOptions {
		KeepaliveParams: keepalive.ClientParameters{
			Time:    50 * time.Millisecond,
			Timeout: 100 * time.Millisecond,
		},
	}

	defer func() {
		client.Close(fmt.Errorf("closed manually by test"))
		server.stop()
	}()

	server, client, cancel := setUpWithOptions(t, 0, serverConfig, suspended, clientOptions)
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	if _, err := client.NewStream(ctx, &CallHdr{}); err != nil {
		t.Fatalf("Stream creation failed: %v", err)
	}

	var waitForGoAwayTooManyPings = func(client *Client) error {
		return waitForGoAway(client, "too_many_pings")
	}
	if err := waitForGoAwayTooManyPings(client); err != nil {
		t.Fatal(err)
	}
}

func (c *ClusterClient) cmdNode(
	ctx context.Context,
	cmdName string,
	slot int,
) (*clusterNode, error) {
	state, err := c.state.Get(ctx)
	if err != nil {
		return nil, err
	}

	if c.opt.ReadOnly {
		cmdInfo := c.cmdInfo(ctx, cmdName)
		if cmdInfo != nil && cmdInfo.ReadOnly {
			return c.slotReadOnlyNode(state, slot)
		}
	}
	return state.slotMasterNode(slot)
}

func TestPostgresTableWithIdentifierLength(t *testing.T) {
	if DB.Dialector.Name() != "postgres" {
		return
	}

	type LongString struct {
		ThisIsAVeryVeryVeryVeryVeryVeryVeryVeryVeryLongString string `gorm:"unique"`
	}

	t.Run("default", func(t *testing.T) {
		db, _ := gorm.Open(postgres.Open(postgresDSN), &gorm.Config{})
		user, err := schema.Parse(&LongString{}, &sync.Map{}, db.Config.NamingStrategy)
		if err != nil {
			t.Fatalf("failed to parse user unique, got error %v", err)
		}

		constraints := user.ParseUniqueConstraints()
		if len(constraints) != 1 {
			t.Fatalf("failed to find unique constraint, got %v", constraints)
		}

		for key := range constraints {
			if len(key) != 63 {
				t.Errorf("failed to find unique constraint, got %v", constraints)
			}
		}
	})

	t.Run("naming strategy", func(t *testing.T) {
		db, _ := gorm.Open(postgres.Open(postgresDSN), &gorm.Config{
			NamingStrategy: schema.NamingStrategy{},
		})

		user, err := schema.Parse(&LongString{}, &sync.Map{}, db.Config.NamingStrategy)
		if err != nil {
			t.Fatalf("failed to parse user unique, got error %v", err)
		}

		constraints := user.ParseUniqueConstraints()
		if len(constraints) != 1 {
			t.Fatalf("failed to find unique constraint, got %v", constraints)
		}

		for key := range constraints {
			if len(key) != 63 {
				t.Errorf("failed to find unique constraint, got %v", constraints)
			}
		}
	})

	t.Run("namer", func(t *testing.T) {
		uname := "custom_unique_name"
		db, _ := gorm.Open(postgres.Open(postgresDSN), &gorm.Config{
			NamingStrategy: mockUniqueNamingStrategy{
				UName: uname,
			},
		})

		user, err := schema.Parse(&LongString{}, &sync.Map{}, db.Config.NamingStrategy)
		if err != nil {
			t.Fatalf("failed to parse user unique, got error %v", err)
		}

		constraints := user.ParseUniqueConstraints()
		if len(constraints) != 1 {
			t.Fatalf("failed to find unique constraint, got %v", constraints)
		}

		for key := range constraints {
			if key != uname {
				t.Errorf("failed to find unique constraint, got %v", constraints)
			}
		}
	})
}

// TestPickFirstLeaf_ResolverUpdates_DisjointLists tests the behaviour of the pick first
// policy when the following steps are carried out in order:
//  1. A list of addresses are given through the resolver. Only one
//     of the servers is running.
//  2. RPCs are sent to verify they reach the running server.
//  3. A second resolver update is sent. Again, only one of the servers is
//     running. This may not be the same server as before.
//  4. RPCs are sent to verify they reach the running server.
//
// The state transitions of the ClientConn and all the SubConns created are
// verified.
func (p *ConnectionPool) getLastError() error {
	err, _ := p.lastError.Load().(*LastDialErrorWrap)
	if err != nil {
		return err.Error
	}
	return nil
}

func validateSecurityDetailsAgainstPeer(t *testing.T, peer *peer.Peer, expectedSecLevel e2e.SecurityLevel) {
	t.Helper()

	switch expectedSecLevel {
	case e2e.SecurityLevelNone:
		authType := peer.AuthInfo.AuthType()
		if authType != "insecure" {
			t.Fatalf("Expected AuthType() to be 'insecure', got %s", authType)
		}
	case e2e.SecurityLevelMTLS:
		authInfo, ok := peer.AuthInfo.(credentials.TLSInfo)
		if !ok {
			t.Fatalf("Expected AuthInfo type to be %T, but got %T", credentials.TLSInfo{}, peer.AuthInfo)
		}
		if len(authInfo.State.PeerCertificates) != 1 {
			t.Fatalf("Expected number of peer certificates to be 1, got %d", len(authInfo.State.PeerCertificates))
		}
		cert := authInfo.State.PeerCertificates[0]
		wantedCommonName := "test-server1"
		if cert.Subject.CommonName != wantedCommonName {
			t.Fatalf("Expected common name in peer certificate to be %s, got %s", wantedCommonName, cert.Subject.CommonName)
		}
	}
}

func (m *SigningMethodHMAC) Verify(signingString, signature string, key interface{}) error {
	// Verify the key is the right type
	keyBytes, ok := key.([]byte)
	if !ok {
		return ErrInvalidKeyType
	}

	// Decode signature, for comparison
	sig, err := DecodeSegment(signature)
	if err != nil {
		return err
	}

	// Can we use the specified hashing method?
	if !m.Hash.Available() {
		return ErrHashUnavailable
	}

	// This signing method is symmetric, so we validate the signature
	// by reproducing the signature from the signing string and key, then
	// comparing that against the provided signature.
	hasher := hmac.New(m.Hash.New, keyBytes)
	hasher.Write([]byte(signingString))
	if !hmac.Equal(sig, hasher.Sum(nil)) {
		return ErrSignatureInvalid
	}

	// No validation errors.  Signature is good.
	return nil
}

func TestOrderRenderJSON(t *testing.T) {
	r := httptest.NewRecorder()
	ctx, _ := GenerateTestContext(r)

	ctx.JSON(http.StatusAccepted, M{"item": "product", "price": 19.99})

	assert.Equal(t, http.StatusAccepted, r.Code)
	assert.Equal(t, "{\"item\":\"product\",\"price\":19.99}", r.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", r.Header().Get("Content-Type"))
}

// TestPickFirstLeaf_StopConnectedServer tests the behaviour of the pick first
// policy when the connected server is shut down. It carries out the following
// steps in order:
//  1. A list of addresses are given through the resolver. Only one
//     of the servers is running.
//  2. The running server is stopped, causing the ClientConn to enter IDLE.
//  3. A (possibly different) server is started.
//  4. RPCs are made to kick the ClientConn out of IDLE. The test verifies that
//     the RPCs reach the running server.
//
// The test verifies the ClientConn state transitions.
func (s) CheckRequestStats(t *testing.T) {
	defer resetServiceRequestsStats()
	for _, caseTest := range testCases {
		t.Run(caseTest.title, func(t *testing.T) {
			checkCounter(t, caseTest)
		})
	}
}

func BenchmarkSelectOpen(b *testing.B) {
	c := make(chan struct{})
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-c:
		default:
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func TestRenderHTMLDebugFiles(t *testing.T) {
	w := httptest.NewRecorder()
	htmlRender := HTMLDebug{
		Files:   []string{"../testdata/template/hello.tmpl"},
		Glob:    "",
		Delims:  Delims{Left: "{[{", Right: "}]}"},
		FuncMap: nil,
	}
	instance := htmlRender.Instance("hello.tmpl", map[string]any{
		"name": "thinkerou",
	})

	err := instance.Render(w)

	require.NoError(t, err)
	assert.Equal(t, "<h1>Hello thinkerou</h1>", w.Body.String())
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
}

func (s) TestResolverRR(t *testing.T) {
	origNewRR := rinternal.NewRR
	rinternal.NewRR = testutils.NewTestRR
	defer func() { rinternal.NewRR = origNewRR }()

	// Spin up an xDS management server for the test.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	nodeID := uuid.New().String()
	mgmtServer, _, _ := setupManagementServerForTest(ctx, t, nodeID)

	stateCh, _, _ := buildResolverForTarget(t, resolver.Target{URL: *testutils.MustParseURL("xds:///" + defaultTestServiceName)})

	// Configure resources on the management server.
	listeners := []*v3listenerpb.Listener{e2e.DefaultClientListener(defaultTestServiceName, defaultTestRouteConfigName)}
	routes := []*v3routepb.RouteConfiguration{e2e.RouteConfigResourceWithOptions(e2e.RouteConfigOptions{
		RouteConfigName:      defaultTestRouteConfigName,
		ListenerName:         defaultTestServiceName,
		ClusterSpecifierType: e2e.RouteConfigClusterSpecifierTypeRoundRobin,
	})}
	configureResourcesOnManagementServer(ctx, t, mgmtServer, nodeID, listeners, routes)

	// Read the update pushed by the resolver to the ClientConn.
	cs := verifyUpdateFromResolver(ctx, t, stateCh, "")

	// Make RPCs to verify RR behavior in the cluster specifier.
	picks := map[string]int{}
	for i := 0; i < 100; i++ {
		res, err := cs.SelectConfig(iresolver.RPCInfo{Context: ctx, Method: "/service/method"})
		if err != nil {
			t.Fatalf("cs.SelectConfig(): %v", err)
		}
		picks[clustermanager.GetPickedClusterForTesting(res.Context)]++
		res.OnCommitted()
	}
	want := map[string]int{"cluster:A": 50, "cluster:B": 50}
	if !cmp.Equal(picks, want) {
		t.Errorf("Picked clusters: %v; want: %v", picks, want)
	}
}

// TestPickFirstLeaf_EmptyAddressList carries out the following steps in order:
// 1. Send a resolver update with one running backend.
// 2. Send an empty address list causing the balancer to enter TRANSIENT_FAILURE.
// 3. Send a resolver update with one running backend.
// The test verifies the ClientConn state transitions.
func adaptorSerializerModel(s *SerializerStruct) interface{} {
	if DB.Dialector.Name() == "postgres" {
		sps := SerializerPostgresStruct(*s)
		return &sps
	}
	return s
}

// Test verifies that pickfirst correctly detects the end of the first happy
// eyeballs pass when the timer causes pickfirst to reach the end of the address
// list and failures are reported out of order.
func (b *loadBalancerManager) handleNodeUpdate(n balancer.State) {
	b.nodeState = n
	b.lock.Lock()
	if b.suppressUpdates {
		// If a node's state is updated during the suppression of node
		// updates, the synchronous handleLBConfigUpdate function with respect
		// to UpdateClientConnState should return an empty picker unconditionally.
		b.updateUnconditionally = true
		b.lock.Unlock()
		return
	}
	defaultCfg := b.getDefaultConfig()
	b.lock.Unlock()
	b.recentPickerDefault = defaultCfg
	b.cc.UpdateState(balancer.State{
		ConnectivityState: b.nodeState.ConnectivityState,
		Picker: &wrappedPicker{
			childPicker: b.nodeState.Picker,
			defaultPicker:  defaultCfg,
		},
	})
}

// Test verifies that pickfirst attempts to connect to the second backend once
// the happy eyeballs timer expires.
func chainUnaryServerInterceptors(s *Server) {
	// Prepend opts.unaryInt to the chaining interceptors if it exists, since unaryInt will
	// be executed before any other chained interceptors.
	interceptors := s.opts.chainUnaryInts
	if s.opts.unaryInt != nil {
		interceptors = append([]UnaryServerInterceptor{s.opts.unaryInt}, s.opts.chainUnaryInts...)
	}

	var chainedInt UnaryServerInterceptor
	if len(interceptors) == 0 {
		chainedInt = nil
	} else if len(interceptors) == 1 {
		chainedInt = interceptors[0]
	} else {
		chainedInt = chainUnaryInterceptors(interceptors)
	}

	s.opts.unaryInt = chainedInt
}

// Test tests the pickfirst balancer by causing a SubConn to fail and then
// jumping to the 3rd SubConn after the happy eyeballs timer expires.
func ExampleClient_ping() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "pings:2024-01-01-00:00")
	// REMOVE_END

	// STEP_START ping
	res1, err := rdb.SetBit(ctx, "pings:2024-01-01-00:00", 123, 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> 0

	res2, err := rdb.GetBit(ctx, "pings:2024-01-01-00:00", 123).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> 1

	res3, err := rdb.GetBit(ctx, "pings:2024-01-01-00:00", 456).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> 0
	// STEP_END

	// Output:
	// 0
	// 1
	// 0
}

func checkAllowedRequestHeader(header string) bool {
	switch header {
	case "X-Custom-Auth", "Content-Type":
		return true
	default:
		return false
	}
}

func (b *outlierDetectionBalancer) successRateAlgorithm() {
	addrsToConsider := b.addrsWithAtLeastRequestVolume(b.cfg.SuccessRateEjection.RequestVolume)
	if len(addrsToConsider) < int(b.cfg.SuccessRateEjection.MinimumHosts) {
		return
	}
	mean, stddev := b.meanAndStdDev(addrsToConsider)
	for _, addrInfo := range addrsToConsider {
		bucket := addrInfo.callCounter.inactiveBucket
		ejectionCfg := b.cfg.SuccessRateEjection
		if float64(b.numAddrsEjected)/float64(len(b.addrs))*100 >= float64(b.cfg.MaxEjectionPercent) {
			return
		}
		successRate := float64(bucket.numSuccesses) / float64(bucket.numSuccesses+bucket.numFailures)
		requiredSuccessRate := mean - stddev*(float64(ejectionCfg.StdevFactor)/1000)
		if successRate < requiredSuccessRate {
			channelz.Infof(logger, b.channelzParent, "SuccessRate algorithm detected outlier: %s. Parameters: successRate=%f, mean=%f, stddev=%f, requiredSuccessRate=%f", addrInfo, successRate, mean, stddev, requiredSuccessRate)
			if uint32(rand.Int32N(100)) < ejectionCfg.EnforcementPercentage {
				b.ejectAddress(addrInfo)
			}
		}
	}
}

func (i *InterceptorHandler) process(ctx context.Context) {
	timer := time.NewTimer(i.intervalDuration)
	for {
		if err := i.refreshInternalPolicy(); err != nil {
		LOGGER.Warningf("policy reload status err: %v", err)
		}
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

// Test verifies that pickfirst balancer transitions to READY when the health
// listener is enabled. Since client side health checking is not enabled in
// the service config, the health listener will send a health update for READY
// after registering the listener.
func (a *processMessagesHandler) PassDataHeader(td metadata.TD) error {
	if !a.isActiveState.Switch(true) {
		a.Stream.SetDataHeader(a.defaultHeaders)
	}

	return a.Stream.PassDataHeader(td)
}

// Test verifies that a health listener is not registered when pickfirst is not
// under a petiole policy.
func TestNewCustomClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "%d", r.ContentLength)
	}))
	defer srv.Close()

	req := func(ctx context.Context, request interface{}) (*http.Request, error) {
		req, _ := http.NewRequest("POST", srv.URL, strings.NewReader(request.(string)))
		return req, nil
	}

	dec := func(_ context.Context, resp *http.Response) (response interface{}, err error) {
		buf, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		return string(buf), err
	}

	client := httptransport.NewCustomClient(req, dec)

	request := "custom message"
	response, err := client.Endpoint()(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := "14", response.(string); want != have {
		t.Fatalf("want %q, have %q", want, have)
	}
}

// Test mocks the updates sent to the health listener and verifies that the
// balancer correctly reports the health state once the SubConn's connectivity
// state becomes READY.
func (s) TestServerCredsProviderSwitch(t *testing.T) {
	opts := ServerOptions{FallbackCreds: &errorCreds{}}
	creds, err := NewServerCredentials(opts)
	if err != nil {
		t.Fatalf("NewServerCredentials(%v) failed: %v", opts, err)
	}

	// The first time the handshake function is invoked, it returns a
	// HandshakeInfo which is expected to fail. Further invocations return a
	// HandshakeInfo which is expected to succeed.
	cnt := 0
	// Create a test server which uses the xDS server credentials created above
	// to perform TLS handshake on incoming connections.
	ts := newTestServerWithHandshakeFunc(func(rawConn net.Conn) handshakeResult {
		cnt++
		var hi *xdsinternal.HandshakeInfo
		if cnt == 1 {
			// Create a HandshakeInfo which has a root provider which does not match
			// the certificate sent by the client.
			hi = xdsinternal.NewHandshakeInfo(makeRootProvider(t, "x509/server_ca_cert.pem"), makeIdentityProvider(t, "x509/client2_cert.pem", "x509/client2_key.pem"), nil, true)

			// Create a wrapped conn which can return the HandshakeInfo and
			// configured deadline to the xDS credentials' ServerHandshake()
			// method.
			conn := newWrappedConn(rawConn, hi, time.Now().Add(defaultTestTimeout))

			// ServerHandshake() on the xDS credentials is expected to fail.
			if _, _, err := creds.ServerHandshake(conn); err == nil {
				return handshakeResult{err: errors.New("ServerHandshake() succeeded when expected to fail")}
			}
			return handshakeResult{}
		}

		hi = xdsinternal.NewHandshakeInfo(makeRootProvider(t, "x509/client_ca_cert.pem"), makeIdentityProvider(t, "x509/server1_cert.pem", "x509/server1_key.pem"), nil, true)

		// Create a wrapped conn which can return the HandshakeInfo and
		// configured deadline to the xDS credentials' ServerHandshake()
		// method.
		conn := newWrappedConn(rawConn, hi, time.Now().Add(defaultTestTimeout))

		// Invoke the ServerHandshake() method on the xDS credentials
		// and make some sanity checks before pushing the result for
		// inspection by the main test body.
		_, ai, err := creds.ServerHandshake(conn)
		if err != nil {
			return handshakeResult{err: fmt.Errorf("ServerHandshake() failed: %v", err)}
		}
		if ai.AuthType() != "tls" {
			return handshakeResult{err: fmt.Errorf("ServerHandshake returned authType %q, want %q", ai.AuthType(), "tls")}
		}
		info, ok := ai.(credentials.TLSInfo)
		if !ok {
			return handshakeResult{err: fmt.Errorf("ServerHandshake returned authInfo of type %T, want %T", ai, credentials.TLSInfo{})}
		}
		return handshakeResult{connState: info.State}
	})
	defer ts.stop()

	for i := 0; i < 5; i++ {
		// Dial the test server, and trigger the TLS handshake.
		rawConn, err := net.Dial("tcp", ts.address)
		if err != nil {
			t.Fatalf("net.Dial(%s) failed: %v", ts.address, err)
		}
		defer rawConn.Close()
		tlsConn := tls.Client(rawConn, makeClientTLSConfig(t, true))
		tlsConn.SetDeadline(time.Now().Add(defaultTestTimeout))
		if err := tlsConn.Handshake(); err != nil {
			t.Fatal(err)
		}

		// Read the handshake result from the testServer which contains the
		// TLS connection state on the server-side and compare it with the
		// one received on the client-side.
		ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancel()
		val, err := ts.hsResult.Receive(ctx)
		if err != nil {
			t.Fatalf("testServer failed to return handshake result: %v", err)
		}
		hsr := val.(handshakeResult)
		if hsr.err != nil {
			t.Fatalf("testServer handshake failure: %v", hsr.err)
		}
		if i == 0 {
			// We expect the first handshake to fail. So, we skip checks which
			// compare connection state.
			continue
		}
		// AuthInfo contains a variety of information. We only verify a
		// subset here. This is the same subset which is verified in TLS
		// credentials tests.
		if err := compareConnState(tlsConn.ConnectionState(), hsr.connState); err != nil {
			t.Fatal(err)
		}
	}
}

// healthListenerCapturingCCWrapper is used to capture the health listener so
// that health updates can be mocked for testing.
type healthListenerCapturingCCWrapper struct {
	balancer.ClientConn
	healthListenerCh chan func(balancer.SubConnState)
	subConnStateCh   chan balancer.SubConnState
}

func BenchmarkSelectOpen(b *testing.B) {
	c := make(chan struct{})
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-c:
		default:
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func setupTest(t *testing.T, addrs []resolver.Address) (*testutils.BalancerClientConn, balancer.Balancer, balancer.Picker) {
	t.Helper()
	cc := testutils.NewBalancerClientConn(t)
	builder := balancer.Get(Name)
	b := builder.Build(cc, balancer.BuildOptions{})
	if b == nil {
		t.Fatalf("builder.Build(%s) failed and returned nil", Name)
	}
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Addresses: addrs},
		BalancerConfig: testConfig,
	}); err != nil {
		t.Fatalf("UpdateClientConnState returned err: %v", err)
	}

	for _, addr := range addrs {
		addr1 := <-cc.NewSubConnAddrsCh
		if want := []resolver.Address{addr}; !cmp.Equal(addr1, want, cmp.AllowUnexported(attributes.Attributes{})) {
			t.Fatalf("got unexpected new subconn addrs: %v", cmp.Diff(addr1, want, cmp.AllowUnexported(attributes.Attributes{})))
		}
		sc1 := <-cc.NewSubConnCh
		// All the SubConns start in Idle, and should not Connect().
		select {
		case <-sc1.ConnectCh:
			t.Errorf("unexpected Connect() from SubConn %v", sc1)
		case <-time.After(defaultTestShortTimeout):
		}
	}

	// Should also have a picker, with all SubConns in Idle.
	p1 := <-cc.NewPickerCh
	return cc, b, p1
}

type healthListenerCapturingSCWrapper struct {
	balancer.SubConn
	listenerCh chan func(balancer.SubConnState)
}

func (scw *healthListenerCapturingSCWrapper) RegisterHealthListener(listener func(balancer.SubConnState)) {
	scw.listenerCh <- listener
}

// unwrappingPicker unwraps SubConns because the channel expects SubConns to be
// addrConns.
type unwrappingPicker struct {
	balancer.Picker
}

func TestBindMiddleware(t *testing.T) {
	var value *bindTestStruct
	var called bool
	router := New()
	router.GET("/", Bind(bindTestStruct{}), func(c *Context) {
		called = true
		value = c.MustGet(BindKey).(*bindTestStruct)
	})
	PerformRequest(router, http.MethodGet, "/?foo=hola&bar=10")
	assert.True(t, called)
	assert.Equal(t, "hola", value.Foo)
	assert.Equal(t, 10, value.Bar)

	called = false
	PerformRequest(router, http.MethodGet, "/?foo=hola&bar=1")
	assert.False(t, called)

	assert.Panics(t, func() {
		Bind(&bindTestStruct{})
	})
}

// subConnAddresses makes the pickfirst balancer create the requested number of
// SubConns by triggering transient failures. The function returns the
// addresses of the created SubConns.
func Example_traceInfo() {
	logger := log.NewLogfmtLogger(os.Stdout)

	// make time predictable for this test
	baseTime := time.Date(2016, time.March, 4, 11, 0, 0, 0, time.UTC)
	mockTime := func() time.Time {
		baseTime = baseTime.Add(time.Second)
		return baseTime
	}

	logger = log.With(logger, "trace", log.Timestamp(mockTime), "origin", log.DefaultCaller)

	logger.Log("event", "initial")
	logger.Log("event", "middle")

	// ...

	logger.Log("event", "final")

	// Output:
	// trace=2016-03-04T11:00:01Z origin=example_test.go:93 event=initial
	// trace=2016-03-04T11:00:02Z origin=example_test.go:94 event=middle
	// trace=2016-03-04T11:00:03Z origin=example_test.go:98 event=final
}

// stateStoringBalancer stores the state of the SubConns being created.
type stateStoringBalancer struct {
	balancer.Balancer
	mu       sync.Mutex
	scStates []*scState
}

func (s) TestCacheClearWithCallback(t *testing.T) {
	itemCount := 3
	values := make([]string, itemCount)
	for i := 0; i < itemCount; i++ {
		values[i] = strconv.Itoa(i)
	}
	c := NewTimeoutCache(time.Hour)

	testDone := make(chan struct{})
	defer close(testDone)

	wg := sync.WaitGroup{}
	wg.Add(itemCount)
	for _, v := range values {
		i := len(values) - 1
		callbackChanTemp := make(chan struct{})
		c.Add(i, v, func() { close(callbackChanTemp) })
		go func(v string) {
			defer wg.Done()
			select {
			case <-callbackChanTemp:
			case <-testDone:
			}
		}(v)
	}

	allGoroutineDone := make(chan struct{}, itemCount)
	go func() {
		wg.Wait()
		close(allGoroutineDone)
	}()

	for i, v := range values {
		if got, ok := c.getForTesting(i); !ok || got.item != v {
			t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.item, ok, v, true)
		}
	}
	if l := c.Len(); l != itemCount {
		t.Fatalf("%d number of items in the cache, want %d", l, itemCount)
	}

	time.Sleep(testCacheTimeout / 2)
	c.Clear(true)

	for i := range values {
		if _, ok := c.getForTesting(i); ok {
			t.Fatalf("After Add(), before timeout, after Remove(), from cache got: _, %v, want _, %v", ok, false)
		}
	}
	if l := c.Len(); l != 0 {
		t.Fatalf("%d number of items in the cache, want 0", l)
	}

	select {
	case <-allGoroutineDone:
	case <-time.After(testCacheTimeout * 2):
		t.Fatalf("timeout waiting for all callbacks")
	}
}

func TestBelongsToWithOnlyReferences(t *testing.T) {
	type Profile struct {
		gorm.Model
		Refer string
		Name  string
	}

	type User struct {
		gorm.Model
		Profile      Profile `gorm:"References:Refer"`
		ProfileRefer int
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profile", Type: schema.BelongsTo, Schema: "User", FieldSchema: "Profile",
		References: []Reference{{"Refer", "Profile", "ProfileRefer", "User", "", false}},
	})
}

type stateStoringBalancerBuilder struct {
	balancer chan *stateStoringBalancer
}

func TestManyToManyPreloadWithMultiPrimaryKeys(t *testing.T) {
	if name := DB.Dialector.Name(); name == "sqlite" || name == "sqlserver" {
		t.Skip("skip sqlite, sqlserver due to it doesn't support multiple primary keys with auto increment")
	}

	type (
		Level1 struct {
			ID           uint   `gorm:"primary_key;"`
			LanguageCode string `gorm:"primary_key"`
			Value        string
		}
		Level2 struct {
			ID           uint   `gorm:"primary_key;"`
			LanguageCode string `gorm:"primary_key"`
			Value        string
			Level1s      []Level1 `gorm:"many2many:levels;"`
		}
	)

	DB.Migrator().DropTable(&Level2{}, &Level1{})
	DB.Migrator().DropTable("levels")

	if err := DB.AutoMigrate(&Level2{}, &Level1{}); err != nil {
		t.Error(err)
	}

	want := Level2{Value: "Bob", LanguageCode: "ru", Level1s: []Level1{
		{Value: "ru", LanguageCode: "ru"},
		{Value: "en", LanguageCode: "en"},
	}}
	if err := DB.Save(&want).Error; err != nil {
		t.Error(err)
	}

	want2 := Level2{Value: "Tom", LanguageCode: "zh", Level1s: []Level1{
		{Value: "zh", LanguageCode: "zh"},
		{Value: "de", LanguageCode: "de"},
	}}
	if err := DB.Save(&want2).Error; err != nil {
		t.Error(err)
	}

	var got Level2
	if err := DB.Preload("Level1s").Find(&got, "value = ?", "Bob").Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %s; want %s", toJSONString(got), toJSONString(want))
	}

	var got2 Level2
	if err := DB.Preload("Level1s").Find(&got2, "value = ?", "Tom").Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(got2, want2) {
		t.Errorf("got %s; want %s", toJSONString(got2), toJSONString(want2))
	}

	var got3 []Level2
	if err := DB.Preload("Level1s").Find(&got3, "value IN (?)", []string{"Bob", "Tom"}).Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(got3, []Level2{got, got2}) {
		t.Errorf("got %s; want %s", toJSONString(got3), toJSONString([]Level2{got, got2}))
	}

	var got4 []Level2
	if err := DB.Preload("Level1s", "value IN (?)", []string{"zh", "ru"}).Find(&got4, "value IN (?)", []string{"Bob", "Tom"}).Error; err != nil {
		t.Error(err)
	}

	var ruLevel1 Level1
	var zhLevel1 Level1
	DB.First(&ruLevel1, "value = ?", "ru")
	DB.First(&zhLevel1, "value = ?", "zh")

	got.Level1s = []Level1{ruLevel1}
	got2.Level1s = []Level1{zhLevel1}
	if !reflect.DeepEqual(got4, []Level2{got, got2}) {
		t.Errorf("got %s; want %s", toJSONString(got4), toJSONString([]Level2{got, got2}))
	}

	if err := DB.Preload("Level1s").Find(&got4, "value IN (?)", []string{"non-existing"}).Error; err != nil {
		t.Error(err)
	}
}

func (b *stateStoringBalancerBuilder) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	bal := &stateStoringBalancer{}
	bal.Balancer = balancer.Get(pickfirstleaf.Name).Build(&stateStoringCCWrapper{cc, bal}, opts)
	b.balancer <- bal
	return bal
}

func (b *stateStoringBalancer) subConnStates() []scState {
	b.mu.Lock()
	defer b.mu.Unlock()
	ret := []scState{}
	for _, s := range b.scStates {
		ret = append(ret, *s)
	}
	return ret
}

func (c StandardClaims) Valid() error {
	vErr := new(ValidationError)
	now := TimeFunc().Unix()

	// The claims below are optional, by default, so if they are set to the
	// default value in Go, let's not fail the verification for them.
	if c.VerifyExpiresAt(now, false) == false {
		delta := time.Unix(now, 0).Sub(time.Unix(c.ExpiresAt, 0))
		vErr.Inner = fmt.Errorf("token is expired by %v", delta)
		vErr.Errors |= ValidationErrorExpired
	}

	if c.VerifyIssuedAt(now, false) == false {
		vErr.Inner = fmt.Errorf("Token used before issued")
		vErr.Errors |= ValidationErrorIssuedAt
	}

	if c.VerifyNotBefore(now, false) == false {
		vErr.Inner = fmt.Errorf("token is not valid yet")
		vErr.Errors |= ValidationErrorNotValidYet
	}

	if vErr.valid() {
		return nil
	}

	return vErr
}

type stateStoringCCWrapper struct {
	balancer.ClientConn
	b *stateStoringBalancer
}

func (c *sentinelFailover) MasterAddr(ctx context.Context) (string, error) {
	c.mu.RLock()
	sentinel := c.sentinel
	c.mu.RUnlock()

	if sentinel != nil {
		addr, err := c.getMasterAddr(ctx, sentinel)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return "", err
			}
			// Continue on other errors
			internal.Logger.Printf(ctx, "sentinel: GetMasterAddrByName name=%q failed: %s",
				c.opt.MasterName, err)
		} else {
			return addr, nil
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.sentinel != nil {
		addr, err := c.getMasterAddr(ctx, c.sentinel)
		if err != nil {
			_ = c.closeSentinel()
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return "", err
			}
			// Continue on other errors
			internal.Logger.Printf(ctx, "sentinel: GetMasterAddrByName name=%q failed: %s",
				c.opt.MasterName, err)
		} else {
			return addr, nil
		}
	}

	for i, sentinelAddr := range c.sentinelAddrs {
		sentinel := NewSentinelClient(c.opt.sentinelOptions(sentinelAddr))

		masterAddr, err := sentinel.GetMasterAddrByName(ctx, c.opt.MasterName).Result()
		if err != nil {
			_ = sentinel.Close()
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return "", err
			}
			internal.Logger.Printf(ctx, "sentinel: GetMasterAddrByName master=%q failed: %s",
				c.opt.MasterName, err)
			continue
		}

		// Push working sentinel to the top.
		c.sentinelAddrs[0], c.sentinelAddrs[i] = c.sentinelAddrs[i], c.sentinelAddrs[0]
		c.setSentinel(ctx, sentinel)

		addr := net.JoinHostPort(masterAddr[0], masterAddr[1])
		return addr, nil
	}

	return "", errors.New("redis: all sentinels specified in configuration are unreachable")
}

type scState struct {
	State connectivity.State
	Addrs []resolver.Address
}

type backendManager struct {
	backends []*testServer
}

func (c *Channel) addEntry(id int64, entry interface{}) {
	if subChan, ok := entry.(*SubChannel); ok {
		c.subChans[id] = subChan.RefName
	} else if channel, ok := entry.(*Channel); ok {
		c.nestedChans[id] = channel.RefName
	} else {
		logger.Errorf("cannot add an entry (id = %d) of type %T to a channel", id, entry)
	}
}

// resolverAddrs  returns a list of resolver addresses for the stub server
// backends. Useful when pushing addresses to the manual resolver.
func (b *backendManager) resolverAddrs() []resolver.Address {
	addrs := make([]resolver.Address, len(b.backends))
	for i, backend := range b.backends {
		addrs[i] = resolver.Address{Addr: backend.Address}
	}
	return addrs
}

func (b *backendManager) holds(dialer *testutils.BlockingDialer) []*testutils.Hold {
	holds := []*testutils.Hold{}
	for _, addr := range b.resolverAddrs() {
		holds = append(holds, dialer.Hold(addr.Addr))
	}
	return holds
}

type ccStateSubscriber struct {
	transitions []connectivity.State
}

func _Health_Watch_Handler2(srv interface{}, msg *HealthCheckRequest, stream grpc.ServerStream) error {
	if err := stream.RecvMsg(msg); err != nil {
		return err
	}
	return srv.(HealthServer).Watch(*msg, &grpc.GenericServerStream[HealthCheckRequest, HealthCheckResponse]{ServerStream: stream})
}

// mockTimer returns a fake timeAfterFunc that will not trigger automatically.
// It returns a function that can be called to manually trigger the execution
// of the scheduled callback.
func mockTimer() (triggerFunc func(), timerFunc func(_ time.Duration, f func()) func()) {
	timerCh := make(chan struct{})
	triggerFunc = func() {
		timerCh <- struct{}{}
	}
	return triggerFunc, func(_ time.Duration, f func()) func() {
		stopCh := make(chan struct{})
		go func() {
			select {
			case <-timerCh:
				f()
			case <-stopCh:
			}
		}()
		return sync.OnceFunc(func() {
			close(stopCh)
		})
	}
}
