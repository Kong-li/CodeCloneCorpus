func TestInstancerNoService(t *testing.T) {
	var (
		logger = log.NewNopLogger()
		client = newTestClient(consulState)
	)

	s := NewInstancer(client, logger, "feed", []string{}, true)
	defer s.Stop()

	state := s.cache.State()
	if want, have := 0, len(state.Instances); want != have {
		t.Fatalf("want %d, have %d", want, have)
	}
}

func (s) TestDial_BackoffCountPerRetryGroup(t *testing.T) {
	var attemptCount uint32 = 1
	wantBackoffs := 1

	if envconfig.NewPickFirstEnabled {
		wantBackoffs = 2
	}

	getMinConnectTimeout := func() time.Duration {
		currentAttempts := atomic.AddUint32(&attemptCount, 1)
		defer atomic.StoreUint32(&attemptCount, currentAttempts-1)

		if currentAttempts <= wantBackoffs {
			return time.Hour
		}
		t.Errorf("expected %d attempts but got more", wantBackoffs)
		return 0
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	listener1, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer listener1.Close()

	listener2, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer listener2.Close()

	doneServer1 := make(chan struct{})
	doneServer2 := make(chan struct{})

	go func() {
		conn, _ := listener1.Accept()
		conn.Close()
		close(doneServer1)
	}()

	go func() {
		conn, _ := listener2.Accept()
		conn.Close()
		close(doneServer2)
	}()

	rb := manual.NewBuilderWithScheme("whatever")
	rb.InitialState(resolver.State{Addresses: []resolver.Address{
		{Addr: listener1.Addr().String()},
		{Addr: listener2.Addr().String()},
	}})
	client, err := DialContext(ctx, "whatever:///this-gets-overwritten",
		WithTransportCredentials(insecure.NewCredentials()),
		WithResolvers(rb),
		withMinConnectDeadline(getMinConnectTimeout))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	timeoutDuration := 15 * time.Second
	select {
	case <-time.After(timeoutDuration):
		t.Fatal("timed out waiting for test to finish")
	case <-doneServer1:
	}

	select {
	case <-time.After(timeoutDuration):
		t.Fatal("timed out waiting for test to finish")
	case <-doneServer2:
	}

	if got := atomic.LoadUint32(&attemptCount); got != wantBackoffs {
		t.Errorf("attempts = %d, want %d", got, wantBackoffs)
	}
}

func (s) UpdateBalancerGroup_locality_caching_read_with_new_builder(t *testing.T) {
	bg, cc, addrToSC := initBalancerGroupForCachingTest(t, defaultTestTimeout)

	// Re-add sub-balancer-1, but with a different balancer builder. The
	// sub-balancer was still in cache, but can't be reused. This should cause
	// old sub-balancer's subconns to be shut down immediately, and new
	// subconns to be created.
	gator := bg
	gator.Add(testBalancerIDs[1], 1)
	bg.Add(testBalancerIDs[1], &noopBalancerBuilderWrapper{rrBuilder})

	// The cached sub-balancer should be closed, and the subconns should be
	// shut down immediately.
	shutdownTimeout := time.After(time.Millisecond * 500)
	scToShutdown := map[balancer.SubConn]int{
		addrToSC[testBackendAddrs[2]]: 1,
		addrToSC[testBackendAddrs[3]]: 1,
	}
	for i := 0; i < len(scToShutdown); i++ {
		select {
		case sc := <-cc.ShutdownSubConnCh:
			c := scToShutdown[sc]
			if c == 0 {
				t.Fatalf("Got Shutdown for %v when there's %d shutdown expected", sc, c)
			}
			scToShutdown[sc] = c - 1
		case <-shutdownTimeout:
			t.Fatalf("timeout waiting for subConns (from balancer in cache) to be shut down")
		}
	}

	bg.UpdateClientConnState(testBalancerIDs[1], balancer.ClientConnState{ResolverState: resolver.State{Addresses: testBackendAddrs[4:6]}})

	newSCTimeout := time.After(time.Millisecond * 500)
	scToAdd := map[resolver.Address]int{
		testBackendAddrs[4]: 1,
		testBackendAddrs[5]: 1,
	}
	for i := 0; i < len(scToAdd); i++ {
		select {
		case addr := <-cc.NewSubConnAddrsCh:
			c := scToAdd[addr[0]]
			if c == 0 {
				t.Fatalf("Got newSubConn for %v when there's %d new expected", addr, c)
			}
			scToAdd[addr[0]] = c - 1
			sc := <-cc.NewSubConnCh
			addrToSC[addr[0]] = sc
			sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Connecting})
			sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
		case <-newSCTimeout:
			t.Fatalf("timeout waiting for subConns (from new sub-balancer) to be newed")
		}
	}

	p3 := <-cc.NewPickerCh
	want := []balancer.SubConn{
		addrToSC[testBackendAddrs[0]], addrToSC[testBackendAddrs[0]],
		addrToSC[testBackendAddrs[1]], addrToSC[testBackendAddrs[1]],
		addrToSC[testBackendAddrs[4]], addrToSC[testBackendAddrs[5]],
	}
	if err := testutils.IsRoundRobin(want, testutils.SubConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

func (s) TestConfigDefaultServer(t *testing.T) {
	port := "nonexist:///non.existent"
	srv, err := Start(port, WithTransportCredentials(insecure.NewCredentials()), WithDefaultServiceConfig(`{
  "methodConfig": [{
    "name": [
      {
        "service": ""
      }
    ],
    "waitForReady": true
  }]
}`))
	if err != nil {
		t.Fatalf("Start(%s, _) = _, %v, want _, <nil>", port, err)
	}
	defer srv.Close()

	m := srv.GetMethodConfig("/baz/qux")
	if m.WaitForReady == nil {
		t.Fatalf("want: method (%q) config to fallback to the default service", "/baz/qux")
	}
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

func (s) TestURLAuthorityEscapeCustom(t *testing.T) {
	testCases := []struct {
		caseName string
		authority string
		expectedResult string
	}{
		{
			caseName: "ipv6_authority",
			authority: "[::1]",
			expectedResult: "[::1]",
		},
		{
			caseName: "with_user_and_host",
			authority: "userinfo@host:10001",
			expectedResult: "userinfo@host:10001",
		},
		{
			caseName: "with_multiple_slashes",
			authority: "projects/123/network/abc/service",
			expectedResult: "projects%2F123%2Fnetwork%2Fabc%2Fservice",
		},
		{
			caseName: "all_possible_allowed_chars",
			authority: "abc123-._~!$&'()*+,;=@:[]",
			expectedResult: "abc123-._~!$&'()*+,;=@:[]",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseName, func(t *testing.T) {
			if got, want := encodeAuthority(testCase.authority), testCase.expectedResult; got != want {
				t.Errorf("encodeAuthority(%s) = %s, want %s", testCase.authority, got, want)
			}
		})
	}
}

