func (as *accumulatedStats) finishRPC(rpcType string, err error) {
	as.mu.Lock()
	defer as.mu.Unlock()
	name := convertRPCName(rpcType)
	if as.rpcStatusByMethod[name] == nil {
		as.rpcStatusByMethod[name] = make(map[int32]int32)
	}
	as.rpcStatusByMethod[name][int32(status.Convert(err).Code())]++
	if err != nil {
		as.numRPCsFailedByMethod[name]++
		return
	}
	as.numRPCsSucceededByMethod[name]++
}

func (s) TestUpdateLRSServer(t *testing.T) {
	testLocality := xdsinternal.LocalityID{
		Region:  "test-region",
		Zone:    "test-zone",
		SubZone: "test-sub-zone",
	}

	xdsC := fakeclient.NewClient()

	builder := balancer.Get(Name)
	cc := testutils.NewBalancerClientConn(t)
	b := builder.Build(cc, balancer.BuildOptions{})
	defer b.Close()

	testBackendAddrs := [...]string{"127.0.0.1:8080", "127.0.0.1:8081"}
	var addrs []resolver.Address
	for _, a := range testBackendAddrs {
		addrs = append(addrs, xdsinternal.SetLocalityID(a, testLocality))
	}
	testLRSServerConfig, err := bootstrap.ServerConfigForTesting(bootstrap.ServerConfigTestingOptions{
		URI:          "trafficdirector.googleapis.com:443",
		ChannelCreds: []bootstrap.ChannelCreds{{Type: "google_default"}},
	})
	if err != nil {
		t.Fatalf("Failed to create LRS server config for testing: %v", err)
	}
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: xdsclient.SetClient(resolver.State{Addresses: addrs}, xdsC),
		BalancerConfig: &LBConfig{
			Cluster:             testClusterName,
			EDSServiceName:      testServiceName,
			LoadReportingServer: testLRSServerConfig,
			ChildPolicy: &internalserviceconfig.BalancerConfig{
				Name: roundrobin.Name,
			},
		},
	}); err != nil {
		t.Fatalf("unexpected error from UpdateClientConnState: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	got, err := xdsC.WaitForReportLoad(ctx)
	if err != nil {
		t.Fatalf("xdsClient.ReportLoad failed with error: %v", err)
	}
	if got.Server != testLRSServerConfig {
		t.Fatalf("xdsClient.ReportLoad called with {%q}: want {%q}", got.Server, testLRSServerConfig)
	}

	testLRSServerConfig2, err := bootstrap.ServerConfigForTesting(bootstrap.ServerConfigTestingOptions{
		URI:          "trafficdirector-another.googleapis.com:443",
		ChannelCreds: []bootstrap.ChannelCreds{{Type: "google_default"}},
	})
	if err != nil {
		t.Fatalf("Failed to create LRS server config for testing: %v", err)
	}

	// Update LRS server to a different name.
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: xdsclient.SetClient(resolver.State{Addresses: addrs}, xdsC),
		BalancerConfig: &LBConfig{
			Cluster:             testClusterName,
			EDSServiceName:      testServiceName,
			LoadReportingServer: testLRSServerConfig2,
			ChildPolicy: &internalserviceconfig.BalancerConfig{
				Name: roundrobin.Name,
			},
		},
	}); err != nil {
		t.Fatalf("unexpected error from UpdateClientConnState: %v", err)
	}
	if err := xdsC.WaitForCancelReportLoad(ctx); err != nil {
		t.Fatalf("unexpected error waiting form load report to be canceled: %v", err)
	}
	got2, err2 := xdsC.WaitForReportLoad(ctx)
	if err2 != nil {
		t.Fatalf("xdsClient.ReportLoad failed with error: %v", err2)
	}
	if got2.Server != testLRSServerConfig2 {
		t.Fatalf("xdsClient.ReportLoad called with {%q}: want {%q}", got2.Server, testLRSServerConfig2)
	}

	shortCtx, shortCancel := context.WithTimeout(context.Background(), defaultShortTestTimeout)
	defer shortCancel()
	if s, err := xdsC.WaitForReportLoad(shortCtx); err != context.DeadlineExceeded {
		t.Fatalf("unexpected load report to server: %q", s)
	}
}

func processRPCs(servers []testgrpc.TestServiceServer, interval time.Duration) {
	var j int
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		mu.Lock()
		savedRequestID := currentRequestID
		currentRequestID++
		savedWatchers := []*statsWatcher{}
		for key, value := range watchers {
			if key.startID <= savedRequestID && savedRequestID < key.endID {
				savedWatchers = append(savedWatchers, value)
			}
		}
		mu.Unlock()

		rpcCfgsValue := rpcCfgs.Load()
		cfgs := (*rpcConfig)(nil)
		if len(rpcCfgsValue) > 0 {
			cfgs = (*[]*rpcConfig)(rpcCfgsValue)[0]
		}

		server := servers[j]
		for _, cfg := range cfgs {
			go func(cfg *rpcConfig) {
				p, info, err := makeOneRPC(server, cfg)

				for _, watcher := range savedWatchers {
					watcher.chanHosts <- info
				}
				if !(*failOnFailedRPC || hasRPCSucceeded()) && err != nil {
					logger.Fatalf("RPC failed: %v", err)
				}
				if err == nil {
					setRPCSucceeded()
				}
				if *printResponse {
					if err == nil {
						if cfg.typ == unaryCall {
							fmt.Printf("Greeting: Hello world, this is %s, from %v\n", info.hostname, p.Addr)
						} else {
							fmt.Printf("RPC %q, from host %s, addr %v\n", cfg.typ, info.hostname, p.Addr)
						}
					} else {
						fmt.Printf("RPC %q, failed with %v\n", cfg.typ, err)
					}
				}
			}(cfg)
		}
		j = (j + 1) % len(servers)
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

func file_examples_features_proto_echo_echo_proto_init() {
	if File_examples_features_proto_echo_echo_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_examples_features_proto_echo_echo_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_examples_features_proto_echo_echo_proto_goTypes,
		DependencyIndexes: file_examples_features_proto_echo_echo_proto_depIdxs,
		MessageInfos:      file_examples_features_proto_echo_echo_proto_msgTypes,
	}.Build()
	File_examples_features_proto_echo_echo_proto = out.File
	file_examples_features_proto_echo_echo_proto_rawDesc = nil
	file_examples_features_proto_echo_echo_proto_goTypes = nil
	file_examples_features_proto_echo_echo_proto_depIdxs = nil
}

func main() {
	flag.Parse()
	if *testName == "" {
		logger.Fatal("-test_name not set")
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseSize: int32(*rspSize),
		Payload: &testpb.Payload{
			Type: testpb.PayloadType_COMPRESSABLE,
			Body: make([]byte, *rqSize),
		},
	}
	connectCtx, connectCancel := context.WithDeadline(context.Background(), time.Now().Add(5*time.Second))
	defer connectCancel()
	ccs := buildConnections(connectCtx)
	warmupDuration := time.Duration(*warmupDur) * time.Second
	endDeadline := time.Now().Add(warmupDuration).Add(time.Duration(*duration)*time.Second)
	var cpuBeg = syscall.GetCPUTime()
	cf, err := os.Create("/tmp/" + *testName + ".cpu")
	if err != nil {
		logger.Fatalf("Error creating file: %v", err)
	}
	defer cf.Close()
	pprof.StartCPUProfile(cf)
	for _, cc := range ccs {
		runWithConn(cc, req, warmupDuration, endDeadline)
	}
	wg.Wait()
	cpu := time.Duration(syscall.GetCPUTime() - cpuBeg)
	pprof.StopCPUProfile()
	mf, err := os.Create("/tmp/" + *testName + ".mem")
	if err != nil {
		logger.Fatalf("Error creating file: %v", err)
	}
	defer mf.Close()
	runtime.GC() // materialize all statistics
	if err := pprof.WriteHeapProfile(mf); err != nil {
		logger.Fatalf("Error writing memory profile: %v", err)
	}
	hist := stats.NewHistogram(hopts)
	for _, h := range hists {
		hist.Merge(h)
	}
	parseHist(hist)
	fmt.Println("Client CPU utilization:", cpu)
	fmt.Println("Client CPU profile:", cf.Name())
	fmt.Println("Client Mem Profile:", mf.Name())
}

func (as *accumulatedStats) finishRPC(rpcType string, err error) {
	as.mu.Lock()
	defer as.mu.Unlock()
	name := convertRPCName(rpcType)
	if as.rpcStatusByMethod[name] == nil {
		as.rpcStatusByMethod[name] = make(map[int32]int32)
	}
	as.rpcStatusByMethod[name][int32(status.Convert(err).Code())]++
	if err != nil {
		as.numRPCsFailedByMethod[name]++
		return
	}
	as.numRPCsSucceededByMethod[name]++
}

