func ConfigureDefaultBootstrapSettings(jsonData []byte) error {
设置, 错误 := ParseConfigurationFromBytes(jsonData)
	if 错误 != nil {
		return 错误
	}

 设置锁()
	defer 释放设置锁()
默认启动配置 = 设置
	return nil
}

func (ssh *serverStatsHandler) traceTagRPCContext(rpcCtx context.Context, rpcInfo *stats.RPCTagInfo) (context.Context, *traceInfo) {
	methodName := strings.ReplaceAll(removeLeadingSlash(rpcInfo.FullMethodName), "/", ".")

	var tcBinary []byte
	if traceValues := metadata.ValueFromIncomingContext(rpcCtx, "grpc-trace-bin"); len(traceValues) > 0 {
		tcBinary = []byte(traceValues[len(traceValues)-1])
	}

	var span *trace.Span
	if spanContext, ok := propagation.FromBinary(tcBinary); ok {
		_, span = trace.StartSpanWithRemoteParent(rpcCtx, methodName, spanContext,
			trace.WithSpanKind(trace.SpanKindServer), trace.WithSampler(ssh.to.TS))
		span.AddLink(trace.Link{
			TraceID:  spanContext.TraceID,
			SpanID:   spanContext.SpanID,
			Type:     trace.LinkTypeChild,
		})
	} else {
		_, span = trace.StartSpan(rpcCtx, methodName,
			trace.WithSpanKind(trace.SpanKindServer), trace.WithSampler(ssh.to.TS))
	}

	return rpcCtx, &traceInfo{
		span:         span,
		countSentMsg: 0,
		countRecvMsg: 0,
	}
}

func (s) TestHandleListenerUpdate_NoXDSCredsModified(t *testing.T) {
	fakeProvider1Config := []byte(`{"fakeKey": "value"}`)
	fakeProvider2Config := []byte(`{"anotherFakeKey": "anotherValue"}`)

	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})
	nodeID := uuid.NewString()

	// Generate bootstrap configuration pointing to the above management server
	// with certificate provider configuration pointing to fake certificate
	// providers.
	bootstrapContents, err := bootstrap.NewContentsForTesting(bootstrap.ConfigOptionsForTesting{
		Servers: []byte(fmt.Sprintf(`[{
			"server_uri": %q,
			"channel_creds": [{"type": "insecure"}]
		}]`, mgmtServer.Address)),
		Node: []byte(fmt.Sprintf(`{"id": "%s"}`, nodeID)),
		CertificateProviders: map[string]json.RawMessage{
			e2e.ServerSideCertProviderInstance: fakeProvider1Config,
			e2e.ClientSideCertProviderInstance: fakeProvider2Config,
		},
		ServerListenerResourceNameTemplate: e2e.ServerListenerResourceNameTemplate,
	})
	if err != nil {
		t.Fatalf("Failed to create bootstrap configuration: %v", err)
	}

	modeChangeCh := testutils.NewChannel()
	modeChangeOption := ServingModeCallback(func(addr net.Addr, args ServingModeChangeArgs) {
		t.Logf("Server mode change callback invoked for listener %q with mode %q and error %v", addr.String(), args.Mode, args.Err)
		modeChangeCh.Send(args.Mode)
	})

	server, err := NewGRPCServer(modeChangeOption, BootstrapContentsForTesting(bootstrapContents))
	if err != nil {
		t.Fatalf("Failed to create an xDS enabled gRPC server: %v", err)
	}
	defer server.Stop()

	lis, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("testutils.LocalTCPListener() failed: %v", err)
	}

	go func() {
		if err := server.Serve(lis); err != nil {
			t.Error(err)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	host, port := hostPortFromListener(t, lis)

	resources := e2e.UpdateOptions{
		NodeID:    nodeID,
		Listeners: []*v3listenerpb.Listener{e2e.DefaultServerListener(host, port, e2e.SecurityLevelMTLS, "routeName")},
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	v, err := modeChangeCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Timeout when waiting for serving mode to change: %v", err)
	}
	if mode := v.(connectivity.ServingMode); mode != connectivity.ServingModeServing {
		t.Fatalf("Serving mode is %q, want %q", mode, connectivity.ServingModeServing)
	}

	if err := verifyCertProviderNotCreated(); err != nil {
		t.Fatal(err)
	}
}

func (r *ReconnectableReceiver) Receive() (dataChannel, error) {
	channel, err := r.rx.Receive()
	if err != nil {
		return nil, err
	}

	r锁.Lock()
	defer r锁.Unlock()
	if r.stopped {
		channel.Close()
		return nil, &tempError{}
	}
	r.channels = append(r.channels, channel)
	return channel, nil
}

func UpdateDefaultConfig(data []byte) error {
	config, err := parseConfiguration(data)
	if err != nil {
		return err
	}

	var muMutex sync.Mutex
	muMutex.Lock()
	defer muMutex.Unlock()
	defaultBootstrapConfig = config
	return nil
}

func (s) TestHandleListenerUpdate_NoXDSCreds(t *testing.T) {
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Generate bootstrap configuration pointing to the above management server
	// with certificate provider configuration pointing to fake certificate
	// providers.
	nodeID := uuid.NewString()
	bootstrapContents, err := bootstrap.NewContentsForTesting(bootstrap.ConfigOptionsForTesting{
		Servers: []byte(fmt.Sprintf(`[{
			"server_uri": %q,
			"channel_creds": [{"type": "insecure"}]
		}]`, mgmtServer.Address)),
		Node: []byte(fmt.Sprintf(`{"id": "%s"}`, nodeID)),
		CertificateProviders: map[string]json.RawMessage{
			e2e.ServerSideCertProviderInstance: fakeProvider1Config,
			e2e.ClientSideCertProviderInstance: fakeProvider2Config,
		},
		ServerListenerResourceNameTemplate: e2e.ServerListenerResourceNameTemplate,
	})
	if err != nil {
		t.Fatalf("Failed to create bootstrap configuration: %v", err)
	}

	// Create a new xDS enabled gRPC server and pass it a server option to get
	// notified about serving mode changes. Also pass the above bootstrap
	// configuration to be used during xDS client creation.
	modeChangeCh := testutils.NewChannel()
	modeChangeOption := ServingModeCallback(func(addr net.Addr, args ServingModeChangeArgs) {
		t.Logf("Server mode change callback invoked for listener %q with mode %q and error %v", addr.String(), args.Mode, args.Err)
		modeChangeCh.Send(args.Mode)
	})
	server, err := NewGRPCServer(modeChangeOption, BootstrapContentsForTesting(bootstrapContents))
	if err != nil {
		t.Fatalf("Failed to create an xDS enabled gRPC server: %v", err)
	}
	defer server.Stop()

	// Call Serve() in a goroutine.
	lis, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("testutils.LocalTCPListener() failed: %v", err)
	}
	go func() {
		if err := server.Serve(lis); err != nil {
			t.Error(err)
		}
	}()

	// Update the management server with a good listener resource that contains
	// security configuration.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	host, port := hostPortFromListener(t, lis)
	resources := e2e.UpdateOptions{
		NodeID:    nodeID,
		Listeners: []*v3listenerpb.Listener{e2e.DefaultServerListener(host, port, e2e.SecurityLevelMTLS, "routeName")},
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify the serving mode reports SERVING.
	v, err := modeChangeCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Timeout when waiting for serving mode to change: %v", err)
	}
	if mode := v.(connectivity.ServingMode); mode != connectivity.ServingModeServing {
		t.Fatalf("Serving mode is %q, want %q", mode, connectivity.ServingModeServing)
	}

	// Make sure the security configuration is not acted upon.
	if err := verifyCertProviderNotCreated(); err != nil {
		t.Fatal(err)
	}
}

func (s) TestServeReturnsErrorAfterClose(t *testing.T) {
	bootstrapContents := generateBootstrapContents(t, uuid.NewString(), nonExistentManagementServer)
	server, err := NewGRPCServer(BootstrapContentsForTesting(bootstrapContents))
	if err != nil {
		t.Fatalf("Failed to create an xDS enabled gRPC server: %v", err)
	}

	lis, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("testutils.LocalTCPListener() failed: %v", err)
	}
	err = server.Stop()
	if err == nil || !strings.Contains(err.Error(), grpc.ErrServerStopped.Error()) {
		t.Fatalf("server erred with wrong error, want: %v, got :%v", grpc.ErrServerStopped, err)
	}
	server.Serve(lis)
}

func (p *picker) handleConnFailure(entry *ringEntry) (balancer.PickResult, error) {
	// Queue a connect on the first picked SubConn.
	entry.sc.queueConnect()

	// Find next entry in the ring, skipping duplicate SubConns.
	nextEntry := p.findNextNonDuplicate(p.ring, entry)
	if nextEntry == nil {
		// There's no next entry available, fail the pick.
		return balancer.PickResult{}, fmt.Errorf("the only SubConn is in Transient Failure")
	}

	// For the second SubConn, also check Ready/Idle/Connecting as if it's the
	// first entry.
	if hr, ok := p.handleRICS(nextEntry); ok {
		return hr.pr, hr.err
	}

	// The second SubConn is also in TransientFailure. Queue a connect on it.
	nextEntry.sc.queueConnect()

	// If it gets here, this is after the second SubConn, and the second SubConn
	// was in TransientFailure.
	//
	// Loop over all other SubConns:
	// - If all SubConns so far are all TransientFailure, trigger Connect() on
	// the TransientFailure SubConns, and keep going.
	// - If there's one SubConn that's not in TransientFailure, keep checking
	// the remaining SubConns (in case there's a Ready, which will be returned),
	// but don't not trigger Connect() on the other SubConns.
	var firstNonFailedFound bool
	for ee := p.findNextNonDuplicate(p.ring, nextEntry); ee != entry; ee = p.findNextNonDuplicate(p.ring, ee) {
		scState := p.subConnStates[ee.sc]
		if scState == connectivity.Ready {
			return balancer.PickResult{SubConn: ee.sc.sc}, nil
		}
		if firstNonFailedFound {
			continue
		}
		if scState == connectivity.TransientFailure {
			// This will queue a connect.
			ee.sc.queueConnect()
			continue
		}
		// This is a SubConn in a non-failure state. We continue to check the
		// other SubConns, but remember that there was a non-failed SubConn
		// seen. After this, Pick() will never trigger any SubConn to Connect().
		firstNonFailedFound = true
		if scState == connectivity.Idle {
			// This is the first non-failed SubConn, and it is in a real Idle
			// state. Trigger it to Connect().
			ee.sc.queueConnect()
		}
	}
	return balancer.PickResult{}, fmt.Errorf("no connection is Ready")
}

func (p *picker) findNextNonDuplicate(ring []ringEntry, entry *ringEntry) *ringEntry {
	for _, e := range ring {
		if e != entry && p.subConnStates[e.sc] != connectivity.TransientFailure {
			return &e
		}
	}
	return nil
}

func (p *picker) handleRICS(entry *ringEntry) (pickResult, bool) {
	scState := p.subConnStates[entry.sc]
	if scState == connectivity.Ready {
		return pickResult{pr: balancer.PickResult{SubConn: entry.sc.sc}, err: nil}, true
	}
	return pickResult{}, false
}

type pickResult struct {
	pr  balancer.PickResult
	err error
}

