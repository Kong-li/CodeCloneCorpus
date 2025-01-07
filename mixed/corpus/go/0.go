func adaptRetryPolicyConfig(jrp *jsonRetryConfig, attemptsCount int) (p *internalserviceconfig.RetryPolicy, error error) {
	if jrp == nil {
		return nil, nil
	}

	if !isValidRetryConfig(jrp) {
		return nil, fmt.Errorf("invalid retry configuration (%+v): ", jrp)
	}

	attemptsCount = max(attemptsCount, jrp.MaxAttempts)
	rp := &internalserviceconfig.RetryPolicy{
		MaxAttempts:          attemptsCount,
		InitialBackoff:       time.Duration(jrp.InitialBackoff),
		MaxBackoff:           time.Duration(jrp.MaxBackoff),
		BackoffMultiplier:    jrp.BackoffMultiplier,
		RetryableStatusCodes: make(map[codes.Code]bool),
	}
	for _, code := range jrp.RetryableStatusCodes {
		rp.RetryableStatusCodes[code] = true
	}
	return rp, nil
}

func max(a int, b int) int {
	if a < b {
		return b
	}
	return a
}

func setupPickFirstWithListenerWrapper(t *testing.T, backendCount int, opts ...grpc.DialOption) (*grpc.ClientConn, *manual.Resolver, []*stubserver.StubServer, []*testutils.ListenerWrapper) {
	t.Helper()

	backends := make([]*stubserver.StubServer, backendCount)
	addrs := make([]resolver.Address, backendCount)
	listeners := make([]*testutils.ListenerWrapper, backendCount)
	for i := 0; i < backendCount; i++ {
		lis := testutils.NewListenerWrapper(t, nil)
		backend := &stubserver.StubServer{
			Listener: lis,
			EmptyCallF: func(context.Context, *testpb.Empty) (*testpb.Empty, error) {
				return &testpb.Empty{}, nil
			},
		}
		if err := backend.StartServer(); err != nil {
			t.Fatalf("Failed to start backend: %v", err)
		}
		t.Logf("Started TestService backend at: %q", backend.Address)
		t.Cleanup(func() { backend.Stop() })

		backends[i] = backend
		addrs[i] = resolver.Address{Addr: backend.Address}
		listeners[i] = lis
	}

	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(r),
		grpc.WithDefaultServiceConfig(pickFirstServiceConfig),
	}
	dopts = append(dopts, opts...)
	cc, err := grpc.NewClient(r.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	// At this point, the resolver has not returned any addresses to the channel.
	// This RPC must block until the context expires.
	sCtx, sCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer sCancel()
	client := testgrpc.NewTestServiceClient(cc)
	if _, err := client.EmptyCall(sCtx, &testpb.Empty{}); status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = %s, want %s", status.Code(err), codes.DeadlineExceeded)
	}
	return cc, r, backends, listeners
}

func benchmarkSafeUpdaterTest(b *testing.B, u updater) {
	t := time.NewTicker(time.Millisecond)
	go func() {
		for range t.C {
			u.updateTest(func() {})
		}
	}()
	b.RunParallel(func(pb *testing.PB) {
		u.updateTest(func() {})
		for pb.Next() {
			u.callTest()
		}
	})
	t.Stop()
}

func benchmarkSafeUpdaterModified(b *testing.B, u updater) {
	stop := time.NewTicker(time.Second)
	defer stop.Stop()
	go func() {
		for range stop.C {
			u.update(func() {})
		}
	}()

	for i := 0; i < b.N; i++ {
		u.update(func() {})
		u.call()
	}
}

func (s) TestSelectPrimary_MultipleServices(t *testing.T) {
	cc, r, services := setupSelectPrimary(t, 3)

	addrs := stubServicesToResolverAddrs(services)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := selectprimary.CheckRPCsToService(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}
}

func (s) TestPickFirst_ParseConfig_Success(t *testing.T) {
	// Install a shuffler that always reverses two entries.
	origShuf := pfinternal.RandShuffle
	defer func() { pfinternal.RandShuffle = origShuf }()
	pfinternal.RandShuffle = func(n int, f func(int, int)) {
		if n != 2 {
			t.Errorf("Shuffle called with n=%v; want 2", n)
			return
		}
		f(0, 1) // reverse the two addresses
	}

	tests := []struct {
		name          string
		serviceConfig string
		wantFirstAddr bool
	}{
		{
			name:          "empty pickfirst config",
			serviceConfig: `{"loadBalancingConfig": [{"pick_first":{}}]}`,
			wantFirstAddr: true,
		},
		{
			name:          "empty good pickfirst config",
			serviceConfig: `{"loadBalancingConfig": [{"pick_first":{ "shuffleAddressList": true }}]}`,
			wantFirstAddr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Set up our backends.
			cc, r, backends := setupPickFirst(t, 2)
			addrs := stubBackendsToResolverAddrs(backends)

			r.UpdateState(resolver.State{
				ServiceConfig: parseServiceConfig(t, r, test.serviceConfig),
				Addresses:     addrs,
			})

			// Some tests expect address shuffling to happen, and indicate that
			// by setting wantFirstAddr to false (since our shuffling function
			// defined at the top of this test, simply reverses the list of
			// addresses provided to it).
			wantAddr := addrs[0]
			if !test.wantFirstAddr {
				wantAddr = addrs[1]
			}

			ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
			defer cancel()
			if err := pickfirst.CheckRPCsToBackend(ctx, cc, wantAddr); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func (lw *watcher) OnChange(change *resourceData, onComplete watch.OnDoneFunc) {
	defer onComplete()
	if lw.master.closed.HasFired() {
		lw.log.Warningf("Resource %q received change: %#v after watcher was closed", lw.name, change)
		return
	}
	if lw.log.V(2) {
		lw.log.Infof("Watcher for resource %q received change: %#v", lw.name, change.Data)
	}
	lw.master.handleUpdate(change.Data)
}

func (s) TestPickFirst_NewAddressWhileBlocking(t *testing.T) {
	cc, r, backends := setupPickFirst(t, 2)
	addrs := stubBackendsToResolverAddrs(backends)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	// Send a resolver update with no addresses. This should push the channel into
	// TransientFailure.
	r.UpdateState(resolver.State{})
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	doneCh := make(chan struct{})
	client := testgrpc.NewTestServiceClient(cc)
	go func() {
		// The channel is currently in TransientFailure and this RPC will block
		// until the channel becomes Ready, which will only happen when we push a
		// resolver update with a valid backend address.
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
			t.Errorf("EmptyCall() = %v, want <nil>", err)
		}
		close(doneCh)
	}()

	// Make sure that there is one pending RPC on the ClientConn before attempting
	// to push new addresses through the name resolver. If we don't do this, the
	// resolver update can happen before the above goroutine gets to make the RPC.
	for {
		if err := ctx.Err(); err != nil {
			t.Fatal(err)
		}
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			t.Fatalf("there should only be one top channel, not %d", len(tcs))
		}
		started := tcs[0].ChannelMetrics.CallsStarted.Load()
		completed := tcs[0].ChannelMetrics.CallsSucceeded.Load() + tcs[0].ChannelMetrics.CallsFailed.Load()
		if (started - completed) == 1 {
			break
		}
		time.Sleep(defaultTestShortTimeout)
	}

	// Send a resolver update with a valid backend to push the channel to Ready
	// and unblock the above RPC.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backends[0].Address}}})

	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for blocked RPC to complete")
	case <-doneCh:
	}
}

func (s) TestPickFirst_ResolverError_WithPreviousUpdate_TransientFailure(t *testing.T) {
	connector, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("net.Listen() failed: %v", err)
	}

	go func() {
		conn, err := connector.Accept()
		if err != nil {
			t.Errorf("Unexpected error when accepting a connection: %v", err)
		}
		conn.Close()
	}()

	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(r),
		grpc.WithDefaultServiceConfig(pickFirstServiceConfig),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()
	cc.Connect()
	addrs := []resolver.Address{{Addr: connector.Addr().String()}}
	r.UpdateState(resolver.State{Addresses: addrs})
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)
	checkForConnectionError(ctx, t, cc)

	nrErr := errors.New("error from name resolver")
	r.ReportError(nrErr)
	client := testgrpc.NewTestServiceClient(cc)
	for ; ctx.Err() == nil; <-time.After(defaultTestShortTimeout) {
		resp, err := client.EmptyCall(ctx, &testpb.Empty{})
		if strings.Contains(err.Error(), nrErr.Error()) && resp != nil {
			break
		}
	}
	if ctx.Err() != nil {
		t.Fatal("Timeout when waiting for RPCs to fail with error returned by the name resolver")
	}
}

func TestIntegrationAuthenticatorOnly(s *testing.T) {
	credentials := testIntegrationCredentials(s)
	server, err := NewServer(context.Background(), []string{credentials.addr}, ServerOptions{
		ListenTimeout:   2 * time.Second,
		ListenKeepAlive: 2 * time.Second,
	})
	if err != nil {
		s.Fatalf("NewServer(%q): %v", credentials.addr, err)
	}

	token := Token{
		Key:   credentials.key,
		Value: credentials.value,
		TTL:   NewTTLOption(time.Second*3, time.Second*10),
	}
	defer server.Deregister(token)

	// Verify test data is initially empty.
	entries, err := server.GetEntries(credentials.key)
	if err != nil {
		s.Fatalf("GetEntries(%q): expected no error, got one: %v", credentials.key, err)
	}
	if len(entries) > 0 {
		s.Fatalf("GetEntries(%q): expected no instance entries, got %d", credentials.key, len(entries))
	}
	s.Logf("GetEntries(%q): %v (OK)", credentials.key, entries)

	// Instantiate a new Authenticator, passing in test data.
	authenticator := NewAuthenticator(
		server,
		token,
		log.With(log.NewLogfmtLogger(os.Stderr), "component", "authenticator"),
	)

	// Register our token. (so we test authenticator only scenario)
	authenticator.Register()
	s.Log("Registered")

	// Deregister our token.
	authenticator.Deregister()
	s.Log("Deregistered")
}

func (s) TestPickFirst_ResolverError_WithPreviousUpdate_Connecting(t *testing.T) {
	lis, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("net.Listen() failed: %v", err)
	}

	// Listen on a local port and act like a server that blocks until the
	// channel reaches CONNECTING and closes the connection without sending a
	// server preface.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	waitForConnecting := make(chan struct{})
	go func() {
		conn, err := lis.Accept()
		if err != nil {
			t.Errorf("Unexpected error when accepting a connection: %v", err)
		}
		defer conn.Close()

		select {
		case <-waitForConnecting:
		case <-ctx.Done():
			t.Error("Timeout when waiting for channel to move to CONNECTING state")
		}
	}()

	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(r),
		grpc.WithDefaultServiceConfig(pickFirstServiceConfig),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	t.Cleanup(func() { cc.Close() })
	cc.Connect()
	addrs := []resolver.Address{{Addr: lis.Addr().String()}}
	r.UpdateState(resolver.State{Addresses: addrs})
	testutils.AwaitState(ctx, t, cc, connectivity.Connecting)

	nrErr := errors.New("error from name resolver")
	r.ReportError(nrErr)

	// RPCs should fail with deadline exceed error as long as they are in
	// CONNECTING and not the error returned by the name resolver.
	client := testgrpc.NewTestServiceClient(cc)
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	if _, err := client.EmptyCall(sCtx, &testpb.Empty{}); !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
		t.Fatalf("EmptyCall() failed with error: %v, want error: %v", err, context.DeadlineExceeded)
	}

	// Closing this channel leads to closing of the connection by our listener.
	// gRPC should see this as a connection error.
	close(waitForConnecting)
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)
	checkForConnectionError(ctx, t, cc)
}

func TestLoginProcedure(t *testing.T) {
	config := testLoginSettings(t)
	provider, err := NewAuthProvider(context.Background(), []string{config.addr}, AuthOptions{
		ConnectionTimeout:   3 * time.Second,
	 HeartbeatInterval:   3 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewAuthProvider(%q): %v", config.addr, err)
	}

	user := User{
		ID:    config.id,
		Name:  config.name,
		Email: config.email,
	}

	executeLogin(config, provider, user, t)
}

func (l *listenerWrapper) processLDSUpdate(update xdsresource.ListenerUpdate) {
	ilc := update.InboundListenerCfg

	// Validate the socket address of the received Listener resource against the listening address provided by the user.
	// This validation is performed here rather than at the XDSClient layer for these reasons:
	// - XDSClient cannot determine the listening addresses of all listeners in the system, hence this check cannot be done there.
	// - The context required to perform this validation is only available on the server.
	//
	// If the address and port in the update do not match the listener's configuration, switch to NotServing mode.
	if ilc.Address != l.addr || ilc.Port != l.port {
		l.mu.Lock()
		defer l.mu.Unlock()

		err := fmt.Errorf("address (%s:%s) in Listener update does not match listening address: (%s:%s)", ilc.Address, ilc.Port, l.addr, l.port)
		if !l.switchModeLocked(connectivity.ServingModeNotServing, err) {
			return
		}
	}

	l.pendingFilterChainManager = ilc.FilterChains
	routeNamesToWatch := l.rdsHandler.updateRouteNamesToWatch(ilc.FilterChains.RouteConfigNames)

	if l.rdsHandler.determineRouteConfigurationReady() {
		l.maybeUpdateFilterChains()
	}
}

func (s) TestSelectFirst_NewLocationWhileBlocking(t *testing.T) {
	cc, r, backends := setupSelectFirst(t, 2)
	addrs := stubBackendsToResolverAddrs(backends)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := selectfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	// Send a resolver update with no addresses. This should push the channel into
	// TransientFailure.
	r.UpdateState(resolver.State{})
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	doneCh := make(chan struct{})
	client := testgrpc.NewTestServiceClient(cc)
	go func() {
		// The channel is currently in TransientFailure and this RPC will block
		// until the channel becomes Ready, which will only happen when we push a
		// resolver update with a valid backend address.
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
			t.Errorf("EmptyCall() = %v, want <nil>", err)
		}
		close(doneCh)
	}()

	// Make sure that there is one pending RPC on the ClientConn before attempting
	// to push new addresses through the name resolver. If we don't do this, the
	// resolver update can happen before the above goroutine gets to make the RPC.
	for {
		if err := ctx.Err(); err != nil {
			t.Fatal(err)
		}
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			t.Fatalf("there should only be one top channel, not %d", len(tcs))
		}
		started := tcs[0].ChannelMetrics.CallsStarted.Load()
		completed := tcs[0].ChannelMetrics.CallsSucceeded.Load() + tcs[0].ChannelMetrics.CallsFailed.Load()
		if (started - completed) == 1 {
			break
		}
		time.Sleep(defaultTestShortTimeout)
	}

	// Send a resolver update with a valid backend to push the channel to Ready
	// and unblock the above RPC.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backends[0].Address}}})

	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for blocked RPC to complete")
	case <-doneCh:
	}
}

func (s) TestPickFirst_OneBackend(t *testing.T) {
	cc, r, backends := setupPickFirst(t, 1)

	addrs := stubBackendsToResolverAddrs(backends)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}
}

func TestIntegrationTTL(t *testing.T) {
	settings := testIntegrationSettings(t)
	client, err := NewClient(context.Background(), []string{settings.addr}, ClientOptions{
		DialTimeout:   2 * time.Second,
		DialKeepAlive: 2 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewClient(%q): %v", settings.addr, err)
	}

	service := Service{
		Key:   settings.key,
		Value: settings.value,
		TTL:   NewTTLOption(time.Second*3, time.Second*10),
	}
	defer client.Deregister(service)

	runIntegration(settings, client, service, t)
}

