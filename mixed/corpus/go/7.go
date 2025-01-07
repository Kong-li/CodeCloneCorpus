func TestDefaultValidator(t *testing.T) {
	type exampleStruct struct {
		A string `binding:"max=8"`
		B int    `binding:"gt=0"`
	}
	tests := []struct {
		name    string
		v       *defaultValidator
		obj     any
		wantErr bool
	}{
		{"validate nil obj", &defaultValidator{}, nil, false},
		{"validate int obj", &defaultValidator{}, 3, false},
		{"validate struct failed-1", &defaultValidator{}, exampleStruct{A: "123456789", B: 1}, true},
		{"validate struct failed-2", &defaultValidator{}, exampleStruct{A: "12345678", B: 0}, true},
		{"validate struct passed", &defaultValidator{}, exampleStruct{A: "12345678", B: 1}, false},
		{"validate *struct failed-1", &defaultValidator{}, &exampleStruct{A: "123456789", B: 1}, true},
		{"validate *struct failed-2", &defaultValidator{}, &exampleStruct{A: "12345678", B: 0}, true},
		{"validate *struct passed", &defaultValidator{}, &exampleStruct{A: "12345678", B: 1}, false},
		{"validate []struct failed-1", &defaultValidator{}, []exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate []struct failed-2", &defaultValidator{}, []exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate []struct passed", &defaultValidator{}, []exampleStruct{{A: "12345678", B: 1}}, false},
		{"validate []*struct failed-1", &defaultValidator{}, []*exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate []*struct failed-2", &defaultValidator{}, []*exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate []*struct passed", &defaultValidator{}, []*exampleStruct{{A: "12345678", B: 1}}, false},
		{"validate *[]struct failed-1", &defaultValidator{}, &[]exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate *[]struct failed-2", &defaultValidator{}, &[]exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate *[]struct passed", &defaultValidator{}, &[]exampleStruct{{A: "12345678", B: 1}}, false},
		{"validate *[]*struct failed-1", &defaultValidator{}, &[]*exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate *[]*struct failed-2", &defaultValidator{}, &[]*exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate *[]*struct passed", &defaultValidator{}, &[]*exampleStruct{{A: "12345678", B: 1}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.v.ValidateStruct(tt.obj); (err != nil) != tt.wantErr {
				t.Errorf("defaultValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func (s) UpdateBalancerAttributes(t *testing.T) {
	testBackendAddrStrs1 := []string{"localhost:8080"}
	addrs1 := make([]resolver.Address, 1)
	for i := range addrs1 {
		addr := internal.SetLocalityID(resolver.Address{Addr: testBackendAddrStrs1[i]}, internal.LocalityID{Region: "americas"})
		addrs1[i] = addr
	}
	cc, b, _ := setupTest(t, addrs1)

	testBackendAddrStrs2 := []string{"localhost:8080"}
	addrs2 := make([]resolver.Address, 1)
	for i := range addrs2 {
		addr := internal.SetLocalityID(resolver.Address{Addr: testBackendAddrStrs2[i]}, internal.LocalityID{Region: "americas"})
		addrs2[i] = addr
	}
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Addresses: addrs2},
		BalancerConfig: testConfig,
	}); err != nil {
		t.Fatalf("UpdateClientConnState returned err: %v", err)
	}
	select {
	case <-cc.NewSubConnCh:
		t.Fatal("new subConn created for an update with the same addresses")
	default:
		time.Sleep(defaultTestShortTimeout)
	}
}

func verifyAndUnmarshalConfig(data json.RawMessage) (*config, error) {
	var config = config{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("error parsing observability configuration: %v", err)
	}
	samplingRateValid := true
	if config.CloudTrace != nil && (config.CloudTrace.SamplingRate > 1 || config.CloudTrace.SamplingRate < 0) {
		samplingRateValid = false
	}

	if !samplingRateValid || validateLoggingEvents(&config) != nil {
		return nil, fmt.Errorf("error parsing observability configuration: %v", err)
	}
	logger.Infof("Parsed ObservabilityConfig: %+v", &config)
	return &config, nil
}

func (l *DefaultLogFormatter) NewLogEntry(r *http.Request) LogEntry {
	useColor := !l.NoColor
	entry := &defaultLogEntry{
		DefaultLogFormatter: l,
		request:             r,
		buf:                 &bytes.Buffer{},
		useColor:            useColor,
	}

	reqID := GetReqID(r.Context())
	if reqID != "" {
		cW(entry.buf, useColor, nYellow, "[%s] ", reqID)
	}
	cW(entry.buf, useColor, nCyan, "\"")
	cW(entry.buf, useColor, bMagenta, "%s ", r.Method)

	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	cW(entry.buf, useColor, nCyan, "%s://%s%s %s\" ", scheme, r.Host, r.RequestURI, r.Proto)

	entry.buf.WriteString("from ")
	entry.buf.WriteString(r.RemoteAddr)
	entry.buf.WriteString(" - ")

	return entry
}

func (l *StandardLogger) NewLogRecord(req *http.Request) LogRecord {
	enableColor := !l.DisableColor
	logEntry := &genericLogEntry{
		StandardLogger: l,
		httpRequest:    req,
		buffer:         &bytes.Buffer{},
		enableColor:    enableColor,
	}

	requestID := FetchReqID(req.Context())
	if requestID != "" {
		cW(logEntry.buffer, enableColor, nYellow, "[%s] ", requestID)
	}
	cW(logEntry.buffer, enableColor, nCyan, "\"")
	cW(logEntry.buffer, enableColor, bMagenta, "%s ", req.Method)

	httpScheme := "http"
	if req.TLS != nil {
		httpScheme = "https"
	}
	cW(logEntry.buffer, enableColor, nCyan, "%s://%s%s %s\" ", httpScheme, req.Host, req.RequestURI, req.Proto)

	logEntry.buffer.WriteString("from ")
	logEntry.buffer.WriteString(req.RemoteAddr)
	logEntry.buffer.WriteString(" - ")

	return logEntry
}

func (s) TestQueueLatency_Disabled_NoActivity(t *testing.T) {
	closeCh := registerWrappedRoundRobinPolicy(t)

	// Create a ClientConn with idle_timeout set to 0.
	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(r),
		grpc.WithIdleTimeout(0), // Disable idleness.
		grpc.WithDefaultServiceConfig(`{"loadBalancingConfig": [{"round_robin":{}}]}`),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()
	cc.Connect()

	// Start a test backend and push an address update via the resolver.
	backend := stubserver.StartTestService(t, nil)
	defer backend.Stop()
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backend.Address}}})

	// Verify that the ClientConn moves to READY.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	// Verify that the ClientConn stays in READY.
	sCtx, sCancel := context.WithTimeout(ctx, 3*defaultTestShortIdleTimeout)
	defer sCancel()
	testutils.AwaitNoStateChange(sCtx, t, cc, connectivity.Ready)

	// Verify that the LB policy is not closed which is expected to happen when
	// the channel enters IDLE.
	sCtx, sCancel = context.WithTimeout(ctx, defaultTestShortIdleTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-closeCh:
		t.Fatal("LB policy closed when expected not to")
	}
}

func channelzTraceLogFound(ctx context.Context, wantMsg string) error {
	for ctx.Err() == nil {
		tcs, _ := channelz.GetRootChannels(0, 0)
		if l := len(tcs); l != 1 {
			return fmt.Errorf("when looking for channelz trace log with message %q, found %d root channels, want 1", wantMsg, l)
		}
		logs := tcs[0].Logs()
		if logs == nil {
			return fmt.Errorf("when looking for channelz trace log with message %q, no logs events found for root channel", wantMsg)
		}

		for _, e := range logs.Entries {
			if strings.Contains(e.Message, wantMsg) {
				return nil
			}
		}
	}
	return fmt.Errorf("when looking for channelz trace log with message %q, %w", wantMsg, ctx.Err())
}

func (s) TestQueueLength_Enabled_NoWork(t *testing.T) {
	closeCh := registerWrappedRandomPolicy(t)

	// Create a ClientConn with a short idle_timeout.
	q := manual.NewBuilderWithScheme("any_scheme")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(q),
		grpc.WithIdleTimeout(defaultTestShortIdleTimeout),
		grpc.WithDefaultServiceConfig(`{"loadBalancingConfig": [{"random":{}}]}`),
	}
	cc, err := grpc.NewClient(q.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()

	cc.Connect()
	// Start a test backend and push an address update via the resolver.
	lis := testutils.NewListenerWrapper(t, nil)
	backend := stubserver.StartTestService(t, &stubserver.StubServer{Listener: lis})
	defer backend.Stop()
	q.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backend.Address}}})

	// Verify that the ClientConn moves to READY.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	// Retrieve the wrapped conn from the listener.
	v, err := lis.NewConnCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Failed to retrieve conn from test listener: %v", err)
	}
	conn := v.(*testutils.ConnWrapper)

	// Verify that the ClientConn moves to IDLE as there is no activity.
	testutils.AwaitState(ctx, t, cc, connectivity.Idle)

	// Verify idleness related channelz events.
	if err := channelzTraceEventFound(ctx, "entering idle mode"); err != nil {
		t.Fatal(err)
	}

	// Verify that the previously open connection is closed.
	if _, err := conn.CloseCh.Receive(ctx); err != nil {
		t.Fatalf("Failed when waiting for connection to be closed after channel entered IDLE: %v", err)
	}

	// Verify that the LB policy is closed.
	select {
	case <-ctx.Done():
		t.Fatal("Timeout waiting for LB policy to be closed after the channel enters IDLE")
	case <-closeCh:
	}
}

func (s) TestConnectivityEvaluatorRecordStateChange(t *testing.T) {
	testCases := []struct {
		name     string
		initial  []connectivity.State
		final    []connectivity.State
		expected connectivity.State
	}{
		{
			name: "one ready",
			initial: []connectivity.State{connectivity.Idle},
			final:   []connectivity.State{connectivity.Ready},
			expected: connectivity.Ready,
		},
		{
			name: "one connecting",
			initial: []connectivity.State{connectivity.Idle},
			final:   []connectivity.State{connectivity.Connecting},
			expected: connectivity.Connecting,
		},
		{
			name: "one ready one transient failure",
			initial: []connectivity.State{connectivity.Idle, connectivity.Idle},
			final:   []connectivity.State{connectivity.Ready, connectivity.TransientFailure},
			expected: connectivity.Ready,
		},
		{
			name: "one connecting one transient failure",
			initial: []connectivity.State{connectivity.Idle, connectivity.Idle},
			final:   []connectivity.State{connectivity.Connecting, connectivity.TransientFailure},
			expected: connectivity.Connecting,
		},
		{
			name: "one connecting two transient failure",
			initial: []connectivity.State{connectivity.Idle, connectivity.Idle, connectivity.Idle},
			final:   []connectivity.State{connectivity.Connecting, connectivity.TransientFailure, connectivity.TransientFailure},
			expected: connectivity.TransientFailure,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			evaluator := &connectivityEvaluator{}
			var result connectivity.State
			for i, initialState := range tc.initial {
				finalState := tc.final[i]
				result = evaluator.recordTransition(initialState, finalState)
			}
			if result != tc.expected {
				t.Errorf("recordTransition() = %v, want %v", result, tc.expected)
			}
		})
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

func (s) TestThreeSubConnsAffinityModified(t *testing.T) {
	wantAddrs := []resolver.Address{
		{Addr: testBackendAddrStrs[0]},
		{Addr: testBackendAddrStrs[1]},
		{Addr: testBackendAddrStrs[2]},
	}
	testConn, _, picker0 := setupTest(t, wantAddrs)
	ring0 := picker0.(*picker).ring

	firstHash := ring0.items[0].hash
	testHash := firstHash + 1

	sc0 := ring0.items[1].sc.sc.(*testutils.TestSubConn)
	_, err := picker0.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
	if err == nil || err != balancer.ErrNoSubConnAvailable {
		t.Fatalf("first pick returned err %v, want %v", err, balancer.ErrNoSubConnAvailable)
	}
	select {
	case <-sc0.ConnectCh:
	default:
		t.Errorf("timeout waiting for Connect() from SubConn %v", sc0)
	}

	sc0.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	p1 := <-testConn.NewPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p1.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
		if gotSCSt.SubConn != sc0 {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc0)
		}
	}

	sc1 := ring0.items[2].sc.sc.(*testutils.TestSubConn)
	select {
	case <-sc1.ConnectCh:
	default:
		t.Errorf("timeout waiting for Connect() from SubConn %v", sc1)
	}

	sc1.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	p3 := <-testConn.NewPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p3.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
		if gotSCSt.SubConn != sc1 {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc1)
		}
	}

	sc0.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Idle})
	select {
	case <-sc0.ConnectCh:
	default:
		t.Errorf("timeout waiting for Connect() from SubConn %v", sc0)
	}

	sc0.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	p4 := <-testConn.NewPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p4.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
		if gotSCSt.SubConn != sc0 {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc0)
		}
	}
}

