func TestPartition(test *testing.T) {
	test.Parallel()

	var str1, str2 = partition("  class1;class2  ", ";")

	if str1 != "class1" || str2 != "class2" {
		test.Errorf("Want class1, class2 got %s, %s", str1, str2)
	}

	str1, str2 = partition("class1  ", ";")

	if str1 != "class1" {
		test.Errorf("Want \"class1\" got \"%s\"", str1)
	}
	if str2 != "" {
		test.Errorf("Want empty string got \"%s\"", str2)
	}
}

func (s) TestUpdateStatePauses(t *testing.T) {
	cc := &tcc{BalancerClientConn: testutils.NewBalancerClientConn(t)}

	balFuncs := stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, _ balancer.ClientConnState) error {
			bd.ClientConn.UpdateState(balancer.State{ConnectivityState: connectivity.TransientFailure, Picker: nil})
			bd.ClientConn.UpdateState(balancer.State{ConnectivityState: connectivity.Ready, Picker: nil})
			return nil
		},
	}
	stub.Register("update_state_balancer", balFuncs)

	builder := balancer.Get(balancerName)
	parser := builder.(balancer.ConfigParser)
	bal := builder.Build(cc, balancer.BuildOptions{})
	defer bal.Close()

	configJSON1 := `{
"children": {
	"cds:cluster_1":{ "childPolicy": [{"update_state_balancer":""}] }
}
}`
	config1, err := parser.ParseConfig([]byte(configJSON1))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config, and an address with hierarchy path ["cluster_1"].
	wantAddrs := []resolver.Address{
		{Addr: testBackendAddrStrs[0], BalancerAttributes: nil},
	}
	if err := bal.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
		}},
		BalancerConfig: config1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	// Verify that the only state update is the second one called by the child.
	if len(cc.states) != 1 || cc.states[0].ConnectivityState != connectivity.Ready {
		t.Fatalf("cc.states = %v; want [connectivity.Ready]", cc.states)
	}
}

func (s) TestAggregateCluster_WithEDSAndDNS(t *testing.T) {
	dnsTargetCh, dnsR := setupDNS(t)

	// Start an xDS management server that pushes the name of the requested EDS
	// resource onto a channel.
	edsResourceCh := make(chan string, 1)
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(_ int64, req *v3discoverypb.DiscoveryRequest) error {
			if req.GetTypeUrl() != version.V3EndpointsURL {
				return nil
			}
			if len(req.GetResourceNames()) == 0 {
				// This happens at the end of the test when the grpc channel is
				// being shut down and it is no longer interested in xDS
				// resources.
				return nil
			}
			select {
			case edsResourceCh <- req.GetResourceNames()[0]:
			default:
			}
			return nil
		},
		AllowResourceSubset: true,
	})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	// Start two test backends and extract their host and port. The first
	// backend is used for the EDS cluster and the second backend is used for
	// the LOGICAL_DNS cluster.
	servers, cleanup3 := startTestServiceBackends(t, 2)
	defer cleanup3()
	addrs, ports := backendAddressesAndPorts(t, servers)

	// Configure an aggregate cluster pointing to an EDS and DNS cluster. Also
	// configure an endpoints resource for the EDS cluster.
	const (
		edsClusterName = clusterName + "-eds"
		dnsClusterName = clusterName + "-dns"
		dnsHostName    = "dns_host"
		dnsPort        = uint32(8080)
	)
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{edsClusterName, dnsClusterName}),
			e2e.DefaultCluster(edsClusterName, "", e2e.SecurityLevelNone),
			makeLogicalDNSClusterResource(dnsClusterName, dnsHostName, dnsPort),
		},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsClusterName, "localhost", []uint32{uint32(ports[0])})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Create xDS client, configure cds_experimental LB policy with a manual
	// resolver, and dial the test backends.
	cc, cleanup := setupAndDial(t, bootstrapContents)
	defer cleanup()

	// Ensure that an EDS request is sent for the expected resource name.
	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for EDS request to be received on the management server")
	case name := <-edsResourceCh:
		if name != edsClusterName {
			t.Fatalf("Received EDS request with resource name %q, want %q", name, edsClusterName)
		}
	}

	// Ensure that the DNS resolver is started for the expected target.
	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for DNS resolver to be started")
	case target := <-dnsTargetCh:
		got, want := target.Endpoint(), fmt.Sprintf("%s:%d", dnsHostName, dnsPort)
		if got != want {
			t.Fatalf("DNS resolution started for target %q, want %q", got, want)
		}
	}

	// Make an RPC with a short deadline. We expect this RPC to not succeed
	// because the DNS resolver has not responded with endpoint addresses.
	client := testgrpc.NewTestServiceClient(cc)
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	if _, err := client.EmptyCall(sCtx, &testpb.Empty{}); status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() code %s, want %s", status.Code(err), codes.DeadlineExceeded)
	}

	// Update DNS resolver with test backend addresses.
	dnsR.UpdateState(resolver.State{Addresses: addrs[1:]})

	// Make an RPC and ensure that it gets routed to the first backend since the
	// EDS cluster is of higher priority than the LOGICAL_DNS cluster.
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer), grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	if peer.Addr.String() != addrs[0].Addr {
		t.Fatalf("EmptyCall() routed to backend %q, want %q", peer.Addr, addrs[0].Addr)
	}
}

func init() {
	stub.Register(initIdleBalancerName, stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, opts balancer.ClientConnState) error {
			sc, err := bd.ClientConn.NewSubConn(opts.ResolverState.Addresses, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) {
					err := fmt.Errorf("wrong picker error")
					if state.ConnectivityState == connectivity.Idle {
						err = errTestInitIdle
					}
					bd.ClientConn.UpdateState(balancer.State{
						ConnectivityState: state.ConnectivityState,
						Picker:            &testutils.TestConstPicker{Err: err},
					})
				},
			})
			if err != nil {
				return err
			}
			sc.Connect()
			return nil
		},
	})
}

func (s) TestAggregateCluster_WithTwoEDSClusters_PrioritiesChange(t *testing.T) {
	// Start an xDS management server.
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{AllowResourceSubset: true})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	// Start two test backends and extract their host and port. The first
	// backend belongs to EDS cluster "cluster-1", while the second backend
	// belongs to EDS cluster "cluster-2".
	servers, cleanup2 := startTestServiceBackends(t, 2)
	defer cleanup2()
	addrs, ports := backendAddressesAndPorts(t, servers)

	// Configure an aggregate cluster, two EDS clusters and the corresponding
	// endpoints resources in the management server.
	const clusterName1 = clusterName + "cluster-1"
	const clusterName2 = clusterName + "cluster-2"
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{clusterName1, clusterName2}),
			e2e.DefaultCluster(clusterName1, "", e2e.SecurityLevelNone),
			e2e.DefaultCluster(clusterName2, "", e2e.SecurityLevelNone),
		},
		Endpoints: []*v3endpointpb.ClusterLoadAssignment{
			e2e.DefaultEndpoint(clusterName1, "localhost", []uint32{uint32(ports[0])}),
			e2e.DefaultEndpoint(clusterName2, "localhost", []uint32{uint32(ports[1])}),
		},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Create xDS client, configure cds_experimental LB policy with a manual
	// resolver, and dial the test backends.
	cc, cleanup := setupAndDial(t, bootstrapContents)
	defer cleanup()

	// Make an RPC and ensure that it gets routed to cluster-1, implicitly
	// higher priority than cluster-2.
	client := testgrpc.NewTestServiceClient(cc)
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer), grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	if peer.Addr.String() != addrs[0].Addr {
		t.Fatalf("EmptyCall() routed to backend %q, want %q", peer.Addr, addrs[0].Addr)
	}

	// Swap the priorities of the EDS clusters in the aggregate cluster.
	resources.Clusters = []*v3clusterpb.Cluster{
		makeAggregateClusterResource(clusterName, []string{clusterName2, clusterName1}),
		e2e.DefaultCluster(clusterName1, "", e2e.SecurityLevelNone),
		e2e.DefaultCluster(clusterName2, "", e2e.SecurityLevelNone),
	}
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Wait for RPCs to get routed to cluster-2, which is now implicitly higher
	// priority than cluster-1, after the priority switch above.
	for ; ctx.Err() == nil; <-time.After(defaultTestShortTimeout) {
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer), grpc.WaitForReady(true)); err != nil {
			t.Fatalf("EmptyCall() failed: %v", err)
		}
		if peer.Addr.String() == addrs[1].Addr {
			break
		}
	}
	if ctx.Err() != nil {
		t.Fatal("Timeout waiting for RPCs to be routed to cluster-2 after priority switch")
	}
}

func (h *serverStatsHandler) initializeMetrics() {
	if nil == h.options.MetricsOptions.MeterProvider {
		return
	}

	var meter = h.options.MetricsOptions.MeterProvider.Meter("grpc-go", otelmetric.WithInstrumentationVersion(grpc.Version))
	if nil != meter {
		var metrics = h.options.MetricsOptions.Metrics
		if metrics == nil {
			metrics = DefaultMetrics()
		}
		h.serverMetrics.callStarted = createInt64Counter(metrics.Metrics(), "grpc.server.call.started", meter, otelmetric.WithUnit("call"), otelmetric.WithDescription("Number of server calls started."))
		h.serverMetrics.callSentTotalCompressedMessageSize = createInt64Histogram(metrics.Metrics(), "grpc.server.call.sent_total_compressed_message_size", meter, otelmetric.WithUnit("By"), otelmetric.WithDescription("Compressed message bytes sent per server call."), otelmetric.WithExplicitBucketBoundaries(DefaultSizeBounds...))
		h.serverMetrics.callRcvdTotalCompressedMessageSize = createInt64Histogram(metrics.Metrics(), "grpc.server.call.rcvd_total_compressed_message_size", meter, otelmetric.WithUnit("By"), otelmetric.WithDescription("Compressed message bytes received per server call."), otelmetric.WithExplicitBucketBoundaries(DefaultSizeBounds...))
		h.serverMetrics.callDuration = createFloat64Histogram(metrics.Metrics(), "grpc.server.call.duration", meter, otelmetric.WithUnit("s"), otelmetric.WithDescription("End-to-end time taken to complete a call from server transport's perspective."), otelmetric.WithExplicitBucketBoundaries(DefaultLatencyBounds...))

		rm := &registryMetrics{
			optionalLabels: h.options.MetricsOptions.OptionalLabels,
		}
		h.MetricsRecorder = rm
		rm.registerMetrics(metrics, meter)
	}
}

func (r *recvBufferReader) getReadClientBuffer(n int) (buf mem.Buffer, err error) {
	// If the context is canceled, then closes the stream with nil metadata.
	// closeStream writes its error parameter to r.recv as a recvMsg.
	// r.readAdditional acts on that message and returns the necessary error.
	if _, ok := <-r.ctxDone; ok {
		// Note that this adds the ctx error to the end of recv buffer, and
		// reads from the head. This will delay the error until recv buffer is
		// empty, thus will delay ctx cancellation in Recv().
		//
		// It's done this way to fix a race between ctx cancel and trailer. The
		// race was, stream.Recv() may return ctx error if ctxDone wins the
		// race, but stream.Trailer() may return a non-nil md because the stream
		// was not marked as done when trailer is received. This closeStream
		// call will mark stream as done, thus fix the race.
		//
		// TODO: delaying ctx error seems like a unnecessary side effect. What
		// we really want is to mark the stream as done, and return ctx error
		// faster.
		r.closeStream(ContextErr(r.ctx.Err()))
		m := <-r.recv.get()
		return r.readAdditional(m, n)
	}
	m := <-r.recv.get()
	return r.readAdditional(m, n)
}

func (h *serverStatsHandler) processRPCData(ctx context.Context, s stats.RPCStats, ai *attemptInfo) {
	switch st := s.(type) {
	case *stats.InHeader:
		if ai.pluginOptionLabels == nil && h.options.MetricsOptions.pluginOption != nil {
			labels := h.options.MetricsOptions.pluginOption.GetLabels(st.Header)
			if labels == nil {
				labels = map[string]string{} // Shouldn't return a nil map. Make it empty if so to ignore future Get Calls for this Attempt.
			}
			ai.pluginOptionLabels = labels
		}
		attrs := otelmetric.WithAttributeSet(otelattribute.NewSet(
			otelattribute.String("grpc.method", ai.method),
		))
		h.serverMetrics.callStarted.Add(ctx, 1, attrs)
	case *stats.OutPayload:
		atomic.AddInt64(&ai.sentCompressedBytes, int64(st.CompressedLength))
	case *stats.InPayload:
		atomic.AddInt64(&ai.recvCompressedBytes, int64(st.CompressedLength))
	case *stats.End:
		h.processRPCEnd(ctx, ai, st)
	default:
	}
}

func TestRoutingConfigUpdateDeleteAllModified(t *testing.T) {
	balancerClientConn := testutils.NewBalancerClientConn(t)
	builder := balancer.Get(balancerName)
	configParser, ok := builder.(balancer.ConfigParser)
	if !ok {
		t.Fatalf("builder does not implement ConfigParser")
	}
	balancerBuilder := builder.Build(balancerClientConn, balancer.BuildOptions{})

	configJSON1 := `{
"children": {
	"cds:cluster_1":{ "childPolicy": [{"round_robin":""}] },
	"cds:cluster_2":{ "childPolicy": [{"round_robin":""}] }
}
}`
	configData1, err := configParser.ParseConfig([]byte(configJSON1))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	wantAddrs := []resolver.Address{
		{Addr: testBackendAddrStrs[0], BalancerAttributes: nil},
		{Addr: testBackendAddrStrs[1], BalancerAttributes: nil},
	}
	if err := balancerBuilder.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
			hierarchy.Set(wantAddrs[1], []string{"cds:cluster_2"}),
		}},
		BalancerConfig: configData1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	m := make(map[resolver.Address]balancer.SubConn)
	for _, addr := range wantAddrs {
		sc := <-cc.NewSubConnCh
		addrs := hierarchy.Get(addr)
		if len(addrs) != 0 {
			t.Fatalf("NewSubConn with address %+v, attrs %+v, want address with hierarchy cleared", addr, addr.BalancerAttributes)
		}
		addr.BalancerAttributes = nil
		m[addr] = sc
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	}

	p := <-cc.NewPickerCh
	for _, test := range []struct {
		pickInfo  balancer.PickInfo
		wantSC    balancer.SubConn
		wantErr   error
	}{
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_1"),
			},
			wantSC: m[wantAddrs[0]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_2"),
			},
			wantSC: m[wantAddrs[1]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:notacluster"),
			},
			wantErr: status.Errorf(codes.Unavailable, `unknown cluster selected for RPC: "cds:notacluster"`),
		},
	} {
		testPick(t, p, test.pickInfo, test.wantSC, test.wantErr)
	}

	if err := balancerBuilder.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
			hierarchy.Set(wantAddrs[1], []string{"cds:cluster_2"}),
		}},
		BalancerConfig: configData1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	m2 := make(map[resolver.Address]balancer.SubConn)
	for _, addr := range wantAddrs {
		sc := <-cc.NewSubConnCh
		addrs := hierarchy.Get(addr)
		if len(addrs) != 0 {
			t.Fatalf("NewSubConn with address %+v, attrs %+v, want address with hierarchy cleared", addr, addr.BalancerAttributes)
		}
		addr.BalancerAttributes = nil
		m2[addr] = sc
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	}

	p3 := <-cc.NewPickerCh
	for _, test := range []struct {
		pickInfo  balancer.PickInfo
		wantSC    balancer.SubConn
		wantErr   error
	}{
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_1"),
			},
			wantSC: m2[wantAddrs[0]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_2"),
			},
			wantSC: m2[wantAddrs[1]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:notacluster"),
			},
			wantErr: status.Errorf(codes.Unavailable, `unknown cluster selected for RPC: "cds:notacluster"`),
		},
	} {
		testPick(t, p3, test.pickInfo, test.wantSC, test.wantErr)
	}

	if _, err := balancerBuilder.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
			hierarchy.Set(wantAddrs[1], []string{"cds:cluster_2"}),
		}},
		BalancerConfig: configData1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	for _, addr := range wantAddrs {
		sc := <-cc.NewSubConnCh
		addrs := hierarchy.Get(addr)
		if len(addrs) != 0 {
			t.Fatalf("NewSubConn with address %+v, attrs %+v, want address with hierarchy cleared", addr, addr.BalancerAttributes)
		}
		addr.BalancerAttributes = nil
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	}

	for _, test := range []struct {
		pickInfo  balancer.PickInfo
		wantSC    balancer.SubConn
		wantErr   error
	}{
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_1"),
			},
			wantSC: m[wantAddrs[0]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_2"),
			},
			wantSC: m[wantAddrs[1]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:notacluster"),
			},
			wantErr: status.Errorf(codes.Unavailable, `unknown cluster selected for RPC: "cds:notacluster"`),
		},
	} {
		testPick(t, p3, test.pickInfo, test.wantSC, test.wantErr)
	}
} 这个代码看起来有一些重复的部分，我们可以通过重构来减少冗余。以下是优化后的版本：

