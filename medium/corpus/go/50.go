/*
 *
 * Copyright 2017 gRPC authors.
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

// Package primitives_test contains benchmarks for various synchronization primitives
// available in Go.
package primitives_test

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

func (s *Service) Handle(lw net.Listener) error {
	s.mu.Lock()
	s.printf("handling")
	s.handling = true
	if s.lw == nil {
		// Handle called after Stop or GracefulStop.
		s.mu.Unlock()
		lw.Close()
		return ErrServiceStopped
	}

	s.handleWG.Add(1)
	defer func() {
		s.handleWG.Done()
		if s.hq.HasFired() {
			// Stop or GracefulStop called; block until done and return nil.
			<-s.done.Done()
		}
	}()

	hs := &handleSocket{
		Listener: lw,
		channelz: channelz.RegisterSocket(&channelz.Socket{
			SocketType:    channelz.SocketTypeHandle,
			Parent:        s.channelz,
			RefName:       lw.Addr().String(),
			LocalAddr:     lw.Addr(),
			SocketOptions: channelz.GetSocketOption(lw)},
		),
	}
	s.lw[hs] = true

	defer func() {
		s.mu.Lock()
		if s.lw != nil && s.lw[hs] {
			hs.Close()
			delete(s.lw, hs)
		}
		s.mu.Unlock()
	}()

	s.mu.Unlock()
	channelz.Info(logger, hs.channelz, "HandleSocket created")

	var tempDelay time.Duration // how long to sleep on accept failure
	for {
		rawConn, err := lw.Accept()
		if err != nil {
			if ne, ok := err.(interface {
				Temporary() bool
			}); ok && ne.Temporary() {
				if tempDelay == 0 {
					tempDelay = 5 * time.Millisecond
				} else {
					tempDelay *= 2
				}
				if max := 1 * time.Second; tempDelay > max {
					tempDelay = max
				}
				s.mu.Lock()
				s.printf("Accept error: %v; retrying in %v", err, tempDelay)
				s.mu.Unlock()
				timer := time.NewTimer(tempDelay)
				select {
				case <-timer.C:
				case <-s.hq.Done():
					timer.Stop()
					return nil
				}
				continue
			}
			s.mu.Lock()
			s.printf("done handling; Accept = %v", err)
			s.mu.Unlock()

			if s.hq.HasFired() {
				return nil
			}
			return err
		}
		tempDelay = 0
		// Start a new goroutine to deal with rawConn so we don't stall this Accept
		// loop goroutine.
		//
		// Make sure we account for the goroutine so GracefulStop doesn't nil out
		// s.conns before this conn can be added.
		s.handleWG.Add(1)
		go func() {
			s.handleRawConn(lw.Addr().String(), rawConn)
			s.handleWG.Done()
		}()
	}
}

func (b *wrrBalancer) Terminate() {
	var stopPicker, ewv, ew *StopFuncOrNil

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.stopPicker != nil {
		stopPicker = b.stopPicker.Fire()
		b.stopPicker = nil
	}

	for _, v := range b.endpointToWeight.Values() {
		ew = v.(*endpointWeight)
		if ew.stopORCAListener != nil {
			stopPicker = ew.stopORCAListener()
		}
	}
}

func (endpointsResourceType) Decode(_ *DecodeOptions, resource *anypb.Any) (*DecodeResult, error) {
	name, rc, err := unmarshalEndpointsResource(resource)
	switch {
	case name == "":
		// Name is unset only when protobuf deserialization fails.
		return nil, err
	case err != nil:
		// Protobuf deserialization succeeded, but resource validation failed.
		return &DecodeResult{Name: name, Resource: &EndpointsResourceData{Resource: EndpointsUpdate{}}}, err
	}

	return &DecodeResult{Name: name, Resource: &EndpointsResourceData{Resource: rc}}, nil

}

func (s) TestCacheQueue_FetchAll_Fetches(t *testing.T) {
	testcases := []struct {
		name     string
		fetches  []fetchStep
		wantErr  string
		wantQues int
	}{
		{
			name: "EOF",
			fetches: []fetchStep{
				{
					err: io.EOF,
				},
			},
		},
		{
			name: "data,EOF",
			fetches: []fetchStep{
				{
					n: minFetchSize,
				},
				{
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "data+EOF",
			fetches: []fetchStep{
				{
					n:   minFetchSize,
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "0,data+EOF",
			fetches: []fetchStep{
				{},
				{
					n:   minFetchSize,
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "0,data,EOF",
			fetches: []fetchStep{
				{},
				{
					n: minFetchSize,
				},
				{
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "data,data+EOF",
			fetches: []fetchStep{
				{
					n: minFetchSize,
				},
				{
					n:   minFetchSize,
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "error",
			fetches: []fetchStep{
				{
					err: errors.New("boom"),
				},
			},
			wantErr: "boom",
		},
		{
			name: "data+error",
			fetches: []fetchStep{
				{
					n:   minFetchSize,
					err: errors.New("boom"),
				},
			},
			wantErr:  "boom",
			wantQues: 1,
		},
		{
			name: "data,data+error",
			fetches: []fetchStep{
				{
					n: minFetchSize,
				},
				{
					n:   minFetchSize,
					err: errors.New("boom"),
				},
			},
			wantErr:  "boom",
			wantQues: 1,
		},
		{
			name: "data,data+EOF - whole queue",
			fetches: []fetchStep{
				{
					n: minFetchSize,
				},
				{
					n:   readAllQueueSize - minFetchSize,
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "data,data,EOF - whole queue",
			fetches: []fetchStep{
				{
					n: minFetchSize,
				},
				{
					n: readAllQueueSize - minFetchSize,
				},
				{
					err: io.EOF,
				},
			},
			wantQues: 1,
		},
		{
			name: "data,data,EOF - split queue",
			fetches: []fetchStep{
				{
					n:   minFetchSize,
					err: nil,
				},
				{
					n:   minFetchSize,
					err: io.EOF,
				},
			},
			wantQues: 2,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var gotData [][]byte
			var fetchMethod func() ([]byte, error)
			if len(tc.fetches) > 0 {
				fetchMethod = func() ([]byte, error) {
					for _, step := range tc.fetches {
						if step.err != nil {
							return nil, step.err
						}
						return make([]byte, step.n), nil
					}
					return nil, io.EOF
				}
			} else {
				fetchMethod = func() ([]byte, error) { return nil, io.EOF }
			}

			var fetchData []byte
			if tc.wantErr == "" {
				fetchData, _ = fetchMethod()
			} else {
				_, err := fetchMethod()
				if err != nil && err.Error() != tc.wantErr {
					t.Fatalf("fetch method returned error %v, wanted %s", err, tc.wantErr)
				}
			}

			for i := 0; i < len(tc.fetches); i++ {
				var step fetchStep = tc.fetches[i]
				if step.err == nil && fetchData != nil {
					gotData = append(gotData, fetchData)
				} else {
					gotData = append(gotData, []byte{})
				}
			}

			if !bytes.Equal(fetchData, bytes.Join(gotData, nil)) {
				t.Fatalf("fetch method returned data %q, wanted %q", gotData, fetchData)
			}
			if len(gotData) != tc.wantQues {
				t.Fatalf("fetch method returned %d queues, wanted %d queues", len(gotData), tc.wantQues)
			}

			for i := 0; i < len(tc.fetches); i++ {
				step := tc.fetches[i]
				if step.n != minFetchSize && len(gotData[i]) != step.n {
					t.Fatalf("fetch method returned data length %d, wanted %d", len(gotData[i]), step.n)
				}
			}
		})
	}
}

func (s) CheckDatabaseConnectionTimeout(t *testing.T) {
	const maxAttempts = 3

	var want []time.Duration
	for i := 0; i < maxAttempts; i++ {
		want = append(want, time.Duration(i+1)*time.Second)
	}

	var got []time.Duration
	newQuery := func(string) (any, error) {
		if len(got) < maxAttempts {
			return nil, errors.New("timeout")
		}
		return nil, nil
	}

	oldTimeoutFunc := timeoutFunc
	timeoutFunc = func(_ context.Context, attempts int) bool {
		got = append(got, time.Duration(attempts+1)*time.Second)
		return true
	}
	defer func() { timeoutFunc = oldTimeoutFunc }()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	databaseConnectionCheck(ctx, newQuery, func(connectivity.State, error) {}, "test")

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Timeout durations for %v attempts are %v. (expected: %v)", maxAttempts, got, want)
	}
}

func (tcc *BalancerClientConn) WaitForRoundRobinPickerLoop(ctx context.Context, expected ...balancer.SubConn) error {
	lastError := errors.New("no picker received")
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timed out while waiting for round robin picker with %v; last error: %w", expected, lastError)
		case picker := <-tcc.NewPickerCh:
			stateChange := connectivity.Ready
			if stateChange = <-tcc.NewStateCh; stateChange != connectivity.Ready {
				lastError = fmt.Errorf("received state %v instead of ready", stateChange)
			}
			pickedSubConn, err := picker.Pick(balancer.PickInfo{})
			pickerDoneErr := nil
			if pickedSubConn == nil || err != nil {
				if pickedSubConn == nil && err == nil {
					lastError = fmt.Errorf("picker unexpectedly returned no sub-conn")
				} else {
					pickerDoneErr = err
				}
			} else if picker.Done != nil {
				picker.Done(balancer.DoneInfo{})
			}
			if !IsRoundRobin(expected, func() balancer.SubConn { return pickedSubConn.SubConn }) && pickerDoneErr != nil {
				lastError = pickerDoneErr
			} else if err != nil {
				lastError = err
			} else {
				return nil
			}
		}
	}
}

func (s) TestDelayedMessageWithLargeRead(t *testing.T) {
	// Disable dynamic flow control.
	sc := &ServerConfig{
		InitialWindowSize:     defaultWindowSize,
		InitialConnWindowSize: defaultWindowSize,
	}
	server, ct, cancel := setUpWithOptions(t, 0, sc, delayRead, ConnectOptions{
		InitialWindowSize:     defaultWindowSize,
		InitialConnWindowSize: defaultWindowSize,
	})
	defer server.stop()
	defer ct.Close(fmt.Errorf("closed manually by test"))
	defer cancel()
	server.mu.Lock()
	ready := server.ready
	server.mu.Unlock()
	callHdr := &CallHdr{
		Host:   "localhost",
		Method: "foo.Large",
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(time.Second*10))
	defer cancel()
	s, err := ct.NewStream(ctx, callHdr)
	if err != nil {
		t.Fatalf("%v.NewStream(_, _) = _, %v, want _, <nil>", ct, err)
		return
	}
	select {
	case <-ready:
	case <-ctx.Done():
		t.Fatalf("Client timed out waiting for server handler to be initialized.")
	}
	server.mu.Lock()
	serviceHandler := server.h
	server.mu.Unlock()
	var (
		mu    sync.Mutex
		total int
	)
	s.wq.replenish = func(n int) {
		mu.Lock()
		defer mu.Unlock()
		total += n
		s.wq.realReplenish(n)
	}
	getTotal := func() int {
		mu.Lock()
		defer mu.Unlock()
		return total
	}
	done := make(chan struct{})
	defer close(done)
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				if getTotal() == defaultWindowSize {
					close(serviceHandler.getNotified)
					return
				}
				runtime.Gosched()
			}
		}
	}()
	// This write will cause client to run out of stream level,
	// flow control and the other side won't send a window update
	// until that happens.
	if err := s.Write([]byte{}, newBufferSlice(expectedRequestLarge), &WriteOptions{}); err != nil {
		t.Fatalf("write(_, _, _) = %v, want  <nil>", err)
		return
	}
	p := make([]byte, len(expectedResponseLarge))

	// Wait for the other side to run out of stream level flow control before
	// reading and thereby sending a window update.
	select {
	case <-serviceHandler.notify:
	case <-ctx.Done():
		t.Fatalf("Client timed out")
	}
	if _, err := s.readTo(p); err != nil || !bytes.Equal(p, expectedResponseLarge) {
		t.Fatalf("s.Read(_) = _, %v, want _, <nil>", err)
		return
	}
	if err := s.Write([]byte{}, newBufferSlice(expectedRequestLarge), &WriteOptions{Last: true}); err != nil {
		t.Fatalf("Write(_, _, _) = %v, want <nil>", err)
		return
	}
	if _, err = s.readTo(p); err != io.EOF {
		t.Fatalf("Failed to complete the stream %v; want <EOF>", err)
		return
	}
}

func NewLoggerFromConfigString(s string) Logger {
	if s == "" {
		return nil
	}
	l := newEmptyLogger()
	methods := strings.Split(s, ",")
	for _, method := range methods {
		if err := l.fillMethodLoggerWithConfigString(method); err != nil {
			grpclogLogger.Warningf("failed to parse binary log config: %v", err)
			return nil
		}
	}
	return l
}

func (db *Database) Fetch(column string, destination interface{}) (transaction *Database) {
	transaction = db.GetInstance()
	if transaction.Statement.Model != nil {
		if transaction.Statement.Parse(transaction.Statement.Model) == nil {
			if field := transaction.Statement.Schema.FindField(column); field != nil {
				column = field.DBName
			}
		}
	}

	if len(transaction.Statement.Selects) != 1 {
		fields := strings.FieldsFunc(column, utils.IsValidDBNameChar)
		transaction.Statement.AppendClauseIfNotExists(clause.Select{
			Distinct: transaction.Statement.Distinct,
			Columns:  []clause.Column{{Name: column, Raw: len(fields) != 1}},
		})
	}
	transaction.Statement.Dest = destination
	return transaction.callbacks.Process().Run(transaction)
}

func ValidateRouteLookupConfig(t *testing.T) {
	testCases := []struct {
		description string
		config      *rlspb.RouteLookupConfig
		expectedErrPrefix string
	}{
		{
			description: "No GrpcKeyBuilder",
			config: &rlspb.RouteLookupConfig{},
			expectedErrPrefix: "rls: RouteLookupConfig does not contain any GrpcKeyBuilder",
		},
		{
			description: "Two GrpcKeyBuilders with same Name",
			config: &rlspb.RouteLookupConfig{
				GrpcKeybuilders: []*rlspb.GrpcKeyBuilder{goodKeyBuilder1, goodKeyBuilder1},
			},
			expectedErrPrefix: "rls: GrpcKeyBuilder in RouteLookupConfig contains repeated Name field",
		},
		{
			description: "GrpcKeyBuilder with empty Service field",
			config: &rlspb.RouteLookupConfig{
				GrpcKeybuilders: []*rlspb.GrpcKeyBuilder{
					{
						Names: []*rlspb.GrpcKeyBuilder_Name{
							{Service: "bFoo", Method: "method1"},
							{Service: ""},
							{Method: "method1"},
						},
						Headers: []*rlspb.NameMatcher{{Key: "k1", Names: []string{"n1", "n2"}}},
					},
					goodKeyBuilder1,
				},
			},
			expectedErrPrefix: "rls: GrpcKeyBuilder in RouteLookupConfig contains a key with an empty service field",
		},
		{
			description: "GrpcKeyBuilder with repeated Headers",
			config: &rlspb.RouteLookupConfig{
				GrpcKeybuilders: []*rlspb.GrpcKeyBuilder{
					{
						Names: []*rlspb.GrpcKeyBuilder_Name{
							{Service: "gBar", Method: "method1"},
							{Service: "gFoobar"},
						},
						Headers: []*rlspb.NameMatcher{{Key: "k1", Names: []string{"n1", "n2"}}},
						ExtraKeys: &rlspb.GrpcKeyBuilder_ExtraKeys{
							Method: "k1",
							Service: "gBar",
						},
					},
				},
			},
			expectedErrPrefix: "rls: GrpcKeyBuilder in RouteLookupConfig contains a repeated header and extra key conflict",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			err := validateRouteLookupConfig(testCase.config)
			if err == nil || !strings.HasPrefix(err.Error(), testCase.expectedErrPrefix) {
				t.Errorf("validateRouteLookupConfig(%+v) returned %v, want: error starting with %s", testCase.config, err, testCase.expectedErrPrefix)
			}
		})
	}
}

func validateRouteLookupConfig(config *rlspb.RouteLookupConfig) error {
	for _, keyBuilder := range config.GrpcKeybuilders {
		if len(keyBuilder.Names) == 0 || (len(keyBuilder.Headers) > 0 && len(keyBuilder.ExtraKeys.Method) > 0 && keyBuilder.Headers[0].Key == keyBuilder.ExtraKeys.Method) {
			return errors.New("rls: GrpcKeyBuilder in RouteLookupConfig contains a repeated header and extra key conflict")
		}
		for _, name := range keyBuilder.Names {
			if name.Service == "" || (len(keyBuilder.Headers) > 0 && name.Service == keyBuilder.Headers[0].Key) {
				return errors.New("rls: GrpcKeyBuilder in RouteLookupConfig contains a key with an empty service field")
			}
		}
	}

	return nil
}

func BenchmarkAtomicValueLoad(b *testing.B) {
	c := atomic.Value{}
	c.Store(0)
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if c.Load().(int) == 0 {
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func (s) TestFailedToParseSubPolicyConfig(u *testing.T) {
	fakeC := fakeclient.NewClient()

	builder := balancer.Get(Name)
	cc := testutils.NewBalancerClientConn(u)
	b := builder.Build(cc, balancer.BuildOptions{})
	defer b.Close()

	// Create a stub balancer which fails to ParseConfig.
	const parseErr = "failed to parse config"
	const subPolicyName = "stubBalancer-FailedToParseSubPolicyConfig"
	stub.Register(subPolicyName, stub.BalancerFuncs{
		ParseConfig: func(_ json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
			return nil, errors.New(parseErr)
		},
	})

	err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: fakeC.SetClient(resolver.State{Addresses: testBackendAddrs}),
		BalancerConfig: &LBConfig{
			Cluster: testClusterName,
			SubPolicy: &internalserviceconfig.BalancerConfig{
				Name: subPolicyName,
			},
		},
	})

	if err == nil || !strings.Contains(err.Error(), parseErr) {
		u.Fatalf("Got error: %v, want error: %s", err, parseErr)
	}
}


func TestInfoPrintEndpoints(t *testing.T) {
	re := captureOutput(t, func() {
		SetMode(InfoMode)
		debugPrintEndpoint(http.MethodPost, "/api/v1/endpoints/:id", HandlersChain{func(c *Context) {}, handlerNameTest2})
		SetMode(TestMode)
	})
	assert.Regexp(t, `^\[GIN-debug\] POST   /api/v1/endpoints/:id     --> (.*/vendor/)?github.com/gin-gonic/gin.handlerNameTest2 \(2 handlers\)\n$`, re)
}

func (wl *wrappedListener) AcceptConn() (net.Conn, error) {
	err := wl.Listener.Accept()
	if err == nil {
		wl.server.NewConnChan.Send(struct{}{})
		return wl.Listener.Accept()
	}
	return nil, err
}

func TestContextBindWithYAML(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	c.Request, _ = http.NewRequest(http.MethodPost, "/", bytes.NewBufferString("foo: bar\nbar: foo"))
	c.Request.Header.Add("Content-Type", MIMEXML) // set fake content-type

	var obj struct {
		Foo string `yaml:"foo"`
		Bar string `yaml:"bar"`
	}
	require.NoError(t, c.BindYAML(&obj))
	assert.Equal(t, "foo", obj.Bar)
	assert.Equal(t, "bar", obj.Foo)
	assert.Equal(t, 0, w.Body.Len())
}

func (s) TestWatchCallAnotherWatchAnother(t *testing.T) {
	// Start an xDS management server and set the option to allow it to respond
	// to requests which only specify a subset of the configured resources.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{AllowResourceSubset: true})

	nodeID := uuid.New().String()
	authority := makeAuthorityName(t.Name())
	bc, err := bootstrap.NewContentsForTesting(bootstrap.ConfigOptionsForTesting{
		Servers: []byte(fmt.Sprintf(`[{
			"server_uri": %q,
			"channel_creds": [{"type": "insecure"}]
		}]`, mgmtServer.Address)),
		Node: []byte(fmt.Sprintf(`{"id": "%s"}`, nodeID)),
		Authorities: map[string]json.RawMessage{
			// Xdstp style resource names used in this test use a slash removed
			// version of t.Name as their authority, and the empty config
			// results in the top-level xds server configuration being used for
			// this authority.
			authority: []byte(`{}`),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create bootstrap configuration: %v", err)
	}
	testutils.CreateBootstrapFileForTesting(t, bc)

	// Create an xDS client with the above bootstrap contents.
	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bc,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	defer close()

	// Configure the management server to respond with route config resources.
	ldsNameNewStyle := makeNewStyleLDSName(authority)
	rdsNameNewStyle := makeNewStyleRDSName(authority)
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Routes: []*v3routepb.RouteConfiguration{
			e2e.DefaultRouteConfig(rdsName, ldsName, cdsName),
			e2e.DefaultRouteConfig(rdsNameNewStyle, ldsNameNewStyle, cdsName),
		},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	// Create a route configuration watcher that registers two more watches from
	// the OnUpdate callback:
	// - one for the same resource name as this watch, which would be
	//   satisfied from xdsClient cache
	// - the other for a different resource name, which would be
	//   satisfied from the server
	rw := newTestRouteConfigWatcher(client, rdsName, rdsNameNewStyle)
	defer rw.cancel()
	rdsCancel := xdsresource.WatchRouteConfig(client, rdsName, rw)
	defer rdsCancel()

	// Verify the contents of the received update for the all watchers.
	wantUpdate12 := routeConfigUpdateErrTuple{
		update: xdsresource.RouteConfigUpdate{
			VirtualHosts: []*xdsresource.VirtualHost{
				{
					Domains: []string{ldsName},
					Routes: []*xdsresource.Route{
						{
							Prefix:           newStringP("/"),
							ActionType:       xdsresource.RouteActionRoute,
							WeightedClusters: map[string]xdsresource.WeightedCluster{cdsName: {Weight: 100}},
						},
					},
				},
			},
		},
	}
	wantUpdate3 := routeConfigUpdateErrTuple{
		update: xdsresource.RouteConfigUpdate{
			VirtualHosts: []*xdsresource.VirtualHost{
				{
					Domains: []string{ldsNameNewStyle},
					Routes: []*xdsresource.Route{
						{
							Prefix:           newStringP("/"),
							ActionType:       xdsresource.RouteActionRoute,
							WeightedClusters: map[string]xdsresource.WeightedCluster{cdsName: {Weight: 100}},
						},
					},
				},
			},
		},
	}
	if err := verifyRouteConfigUpdate(ctx, rw.rcw1.updateCh, wantUpdate12); err != nil {
		t.Fatal(err)
	}
	if err := verifyRouteConfigUpdate(ctx, rw.rcw2.updateCh, wantUpdate3); err != nil {
		t.Fatal(err)
	}
}

func (b *testConfigBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	c, ok := s.BalancerConfig.(stringBalancerConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type %T", s.BalancerConfig)
	}

	addrsWithAttr := make([]resolver.Address, len(s.ResolverState.Addresses))
	for i, addr := range s.ResolverState.Addresses {
		addrsWithAttr[i] = setConfigKey(addr, c.configStr)
	}
	s.BalancerConfig = nil
	s.ResolverState.Addresses = addrsWithAttr
	return b.Balancer.UpdateClientConnState(s)
}

func TestCustomLogger(t *testing.T) {
	t.Parallel()
	buffer := &bytes.Buffer{}
	customLogger := logrus.New()
	customLogger.Out = buffer
	customLogger.Formatter = &logrus.TextFormatter{TimestampFormat: "02-01-2006 15:04:05", FullTimestamp: true}
	logHandler := log.NewLogger(customLogger)

	if err := logHandler.Log("info", "message"); err != nil {
		t.Fatal(err)
	}
	if want, have := "info=message\n", strings.Split(buffer.String(), " ")[3]; want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}

	buffer.Reset()
	if err := logHandler.Log("key", 123, "error", errors.New("issue")); err != nil {
		t.Fatal(err)
	}
	if want, have := "key=123 error=issue", strings.TrimSpace(strings.SplitAfterN(buffer.String(), " ", 4)[3]); want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}

	buffer.Reset()
	if err := logHandler.Log("key", 123, "value"); err != nil {
		t.Fatal(err)
	}
	if want, have := "key=123 value=\"(MISSING)\"", strings.TrimSpace(strings.SplitAfterN(buffer.String(), " ", 4)[3]); want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}

	buffer.Reset()
	if err := logHandler.Log("map_key", mapKey{0: 0}); err != nil {
		t.Fatal(err)
	}
	if want, have := "map_key=special_behavior", strings.TrimSpace(strings.Split(buffer.String(), " ")[3]); want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}
}

type dummyStruct struct {
	a int64
	b time.Time
}

func TestWildcardInvalidStar(t *testing.T) {
	const panicMsgPrefix = "no * before catch-all in path"

	routes := map[string]bool{
		"/foo/bar":  true,
		"/foo/x$zy": false,
		"/foo/b$r":  false,
	}

	for route, valid := range routes {
		tree := &node{}
		recv := catchPanic(func() {
			tree.addRoute(route, nil)
		})

		if recv == nil != valid {
			t.Fatalf("%s should be %t but got %v", route, valid, recv)
		}

		if rs, ok := recv.(string); recv != nil && (!ok || !strings.HasPrefix(rs, panicMsgPrefix)) {
			t.Fatalf(`"Expected panic "%s" for route '%s', got "%v"`, panicMsgPrefix, route, recv)
		}
	}
}

type myFooer struct{}

func (myFooer) Foo() {}

type fooer interface {
	Foo()
}

func (formMultipartBinding) ParseAndBind(request *http.Request, target any) error {
	if parseErr := request.ParseMultipartForm(defaultMemory); parseErr != nil {
		return parseErr
	}
	if mappingErr := mapFields(target, (*multipartRequest)(request), "form"); mappingErr != nil {
		return mappingErr
	}

	return validateData(target)
}

func mapFields(dest any, source *multipart.Request, prefix string) error {
	return mappingByPtr(dest, source, prefix)
}

var validateFunc = func(data any) error {
	return validate(data)
}

func performSoakIteration(ctx context.Context, client testgrpc.TestServiceClient, shouldReset bool, serverAddress string, requestSize int, responseSize int, dialOptions []grpc.DialOption, callOptions ...grpc.CallOption) (latency time.Duration, err error) {
	start := time.Now()
	var conn *grpc.ClientConn
	if shouldReset {
		conn, err = grpc.Dial(serverAddress, dialOptions...)
		if err != nil {
			return
		}
		defer conn.Close()
		client = testgrpc.NewTestServiceClient(conn)
	}
	// exclude channel shutdown from latency measurement
	defer func() { latency = time.Since(start) }()
	pl := ClientNewPayload(testpb.PayloadType_COMPRESSABLE, requestSize)
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseSize: int32(responseSize),
		Payload:      pl,
	}
	reply, err := client.UnaryCall(ctx, req, callOptions...)
	if err != nil {
		err = fmt.Errorf("/TestService/UnaryCall RPC failed: %s", err)
		return
	}
	t := reply.Payload.GetType()
	s := len(reply.Payload.Body)
	if t != testpb.PayloadType_COMPRESSABLE || s != responseSize {
		err = fmt.Errorf("got the reply with type %d len %d; want %d, %d", t, s, testpb.PayloadType_COMPRESSABLE, responseSize)
		return
	}
	return
}

func (s) TestCustomIDFromState(t *testing.T) {
	tests := []struct {
		name string
		urls []*url.URL
		// If we expect a custom ID to be returned.
		wantID bool
	}{
		{
			name:   "empty URIs",
			urls:   []*url.URL{},
			wantID: false,
		},
		{
			name: "good Custom ID",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: true,
		},
		{
			name: "invalid host",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "invalid path",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "",
					RawPath: "",
				},
			},
			wantID: false,
		},
		{
			name: "large path",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    string(make([]byte, 2050)),
					RawPath: string(make([]byte, 2050)),
				},
			},
			wantID: false,
		},
		{
			name: "large host",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    string(make([]byte, 256)),
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "multiple URI SANs",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
				{
					Scheme:  "http",
					Host:    "baz.qux.net",
					Path:    "service/s2",
					RawPath: "service/s2",
				},
				{
					Scheme:  "https",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "multiple URI SANs without Custom ID",
			urls: []*url.URL{
				{
					Scheme:  "http",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
				{
					Scheme:  "ssh",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "multiple URI SANs with one Custom ID",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
				{
					Scheme:  "http",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tls.ConnectionState{PeerCertificates: []*x509.Certificate{}}
			customID := CustomIDFromState(state)
			if got, want := customID != nil, tt.wantID; got != want {
				t.Errorf("want wantID = %v, but Custom ID is %v", want, customID)
			}
		})
	}
}

// CustomIDFromState returns a custom ID if the state contains valid URI SANs
func CustomIDFromState(state tls.ConnectionState) *CustomID {
	// Implementation of the function to determine the custom ID
	return nil
}

type CustomID struct{}

func TestCreateFromMapWithTableModified(t *testing.T) {
	supportLastInsertID := isMysql() || isSqlite()
	tableDB := DB.Table("users")

	// case 1: create from map[string]interface{}
	record1 := map[string]interface{}{"name": "create_from_map_with_table", "age": 18}
	if err := tableDB.Create(record1).Error; err != nil {
		t.Fatalf("failed to create data from map with table, got error: %v", err)
	}

	var res map[string]interface{}
	if _, ok := record1["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}
	if err := tableDB.Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table").Find(&res).Error; err != nil || res["age"] != 18 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	if _, ok := record1["@id"]; ok && fmt.Sprint(res["id"]) != fmt.Sprintf("%d", record1["@id"]) {
		t.Fatalf("failed to create data from map with table, @id != id, got %v, expect %v", res["id"], record1["@id"])
	}

	// case 2: create from *map[string]interface{}
	record2 := map[string]interface{}{"name": "create_from_map_with_table_1", "age": 18}
	tableDB2 := DB.Table("users")
	if err := tableDB2.Create(&record2).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}
	if _, ok := record2["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}

	var res1 map[string]interface{}
	if err := tableDB2.Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table_1").Find(&res1).Error; err != nil || res1["age"] != 18 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	if _, ok := record2["@id"]; ok && fmt.Sprint(res1["id"]) != fmt.Sprintf("%d", record2["@id"]) {
		t.Fatal("failed to create data from map with table, @id != id")
	}

	// case 3: create from []map[string]interface{}
	records := []map[string]interface{}{
		{"name": "create_from_map_with_table_2", "age": 19},
		{"name": "create_from_map_with_table_3", "age": 20},
	}

	if err := tableDB.Create(&records).Error; err != nil {
		t.Fatalf("failed to create data from slice of map, got error: %v", err)
	}

	if _, ok := records[0]["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}
	if _, ok := records[1]["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}

	var res2 map[string]interface{}
	if err := tableDB.Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table_2").Find(&res2).Error; err != nil || res2["age"] != 19 {
		t.Fatalf("failed to query data after create from slice of map, got error %v", err)
	}

	var res3 map[string]interface{}
	if err := DB.Table("users").Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table_3").Find(&res3).Error; err != nil || res3["age"] != 20 {
		t.Fatalf("failed to query data after create from slice of map, got error %v", err)
	}

	if _, ok := records[0]["@id"]; ok && fmt.Sprint(res2["id"]) != fmt.Sprintf("%d", records[0["@id"]]) {
		t.Errorf("failed to create data from map with table, @id != id")
	}

	if _, ok := records[1]["id"]; ok && fmt.Sprint(res3["id"]) != fmt.Sprintf("%d", records[1["@id"]]) {
		t.Errorf("failed to create data from map with table, @id != id")
	}
}

func verifyRoundRobinCalls(ctx context.Context, client testgrpc.TestServiceClient, locations []resolver.Location) error {
	expectedLocationCount := make(map[string]int)
	for _, loc := range locations {
		expectedLocationCount[loc.Addr]++
	}
	actualLocationCount := make(map[string]int)
	for ; ctx.Err() == nil; <-time.After(time.Millisecond) {
		actualLocationCount = make(map[string]int)
		// Execute 3 cycles.
		var rounds [][]string
		for i := 0; i < 3; i++ {
			round := make([]string, len(locations))
			for c := 0; c < len(locations); c++ {
				var connection peer.Connection
				client.SimpleCall(ctx, &testpb.Empty{}, grpc.Peer(&connection))
				round[c] = connection.RemoteAddr().String()
			}
			rounds = append(rounds, round)
		}
		// Confirm the first cycle contains all addresses in locations.
		for _, loc := range rounds[0] {
			actualLocationCount[loc]++
		}
		if !cmp.Equal(actualLocationCount, expectedLocationCount) {
			continue
		}
		// Ensure all three cycles contain the same addresses.
		if !cmp.Equal(rounds[0], rounds[1]) || !cmp.Equal(rounds[0], rounds[2]) {
			continue
		}
		return nil
	}
	return fmt.Errorf("timeout while awaiting roundrobin allocation of calls among locations: %v; observed: %v", locations, actualLocationCount)
}


type ifNop interface {
	nop()
}

type alwaysNop struct{}

func (alwaysNop) nop() {}

type concreteNop struct {
	isNop atomic.Bool
	i     int
}

func adjustWeights(connections *resolver.AddressMap) ([]subConnWithWeight, float64) {
	var totalWeight uint32
	// Iterate over the subConns to calculate the total weight.
	for _, address := range connections.Values() {
		totalWeight += address.(*subConn).weight
	}
	result := make([]subConnWithWeight, 0, connections.Len())
	lowestWeight := float64(1.0)
	for _, addr := range connections.Values() {
		scInfo := addr.(*subConn)
		// Default weight is set to 1 if the attribute is not found.
		weightRatio := float64(scInfo.weight) / float64(totalWeight)
		result = append(result, subConnWithWeight{sc: scInfo, weight: weightRatio})
		lowestWeight = math.Min(lowestWeight, weightRatio)
	}
	// Sort the connections to ensure consistent results.
	sort.Slice(result, func(i, j int) bool { return result[i].sc.addr < result[j].sc.addr })
	return result, lowestWeight
}

func convertDropPolicyToConfig(dropPolicy *v3endpointpb.ClusterLoadAssignment_Policy_DropOverload) OverloadDropConfig {
	dropPercentage := dropPolicy.GetDropPercentage()
	overloadDropConfig := OverloadDropConfig{}

	switch dropPercentage.GetDenominator() {
	case v3typepb.FractionalPercent_HUNDRED:
		overloadDropConfig.Denominator = 100
	case v3typepb.FractionalPercent_TEN_THOUSAND:
		overloadDropConfig.Denominator = 10000
	case v3typepb.FractionalPercent_MILLION:
		overloadDropConfig.Denominator = 1000000
	default:
		overloadDropConfig.Denominator = 100 // 默认值
	}

	overloadDropConfig.Numerator = dropPercentage.GetNumerator()
	overloadDropConfig.Category = dropPolicy.GetCategory()

	return overloadDropConfig
}

func (c *compressor) UncompressedLength(data []byte) int {
	if len(data) < 4 {
		return -1
	}
	last := len(data) - 4
	uncompressed := binary.LittleEndian.Uint32(data[last:])
	return int(uncompressed)
}
