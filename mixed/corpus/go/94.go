func (s) TestDelegatingResolverNoProxyEnvVarsSet(t *testing.T) {
	hpfe := func(req *http.Request) (*url.URL, error) { return nil, nil }
	originalhpfe := delegatingresolver.HTTPSProxyFromEnvironment
	delegatingresolver.HTTPSProxyFromEnvironment = hpfe
	defer func() {
		delegatingresolver.HTTPSProxyFromEnvironment = originalhpfe
	}()

	const (
		targetTestAddr          = "test.com"
		resolvedTargetTestAddr1 = "1.1.1.1:8080"
		resolvedTargetTestAddr2 = "2.2.2.2:8080"
	)

	// Set up a manual resolver to control the address resolution.
	targetResolver := manual.NewBuilderWithScheme("test")
	target := targetResolver.Scheme() + ":///" + targetTestAddr

	// Create a delegating resolver with no proxy configuration
	tcc, stateCh, _ := createTestResolverClientConn(t)
	if _, err := delegatingresolver.New(resolver.Target{URL: *testutils.MustParseURL(target)}, tcc, resolver.BuildOptions{}, targetResolver, false); err != nil {
		t.Fatalf("Failed to create delegating resolver: %v", err)
	}

	// Update the manual resolver with a test address.
	targetResolver.UpdateState(resolver.State{
		Addresses: []resolver.Address{
			{Addr: resolvedTargetTestAddr1},
			{Addr: resolvedTargetTestAddr2},
		},
		ServiceConfig: &serviceconfig.ParseResult{},
	})

	// Verify that the delegating resolver outputs the same addresses, as returned
	// by the target resolver.
	wantState := resolver.State{
		Addresses: []resolver.Address{
			{Addr: resolvedTargetTestAddr1},
			{Addr: resolvedTargetTestAddr2},
		},
		ServiceConfig: &serviceconfig.ParseResult{},
	}

	var gotState resolver.State
	select {
	case gotState = <-stateCh:
	case <-time.After(defaultTestTimeout):
		t.Fatal("Timeout when waiting for a state update from the delegating resolver")
	}

	if diff := cmp.Diff(gotState, wantState); diff != "" {
		t.Fatalf("Unexpected state from delegating resolver. Diff (-got +want):\n%v", diff)
	}
}

func ExampleUser_zscore() {
	userCtx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(userCtx, "racer_scores")
	// REMOVE_END

	_, err := rdb.ZAdd(userCtx, "racer_scores",
		redis.Z{Member: "Norem", Score: 10},
		redis.Z{Member: "Sam-Bodden", Score: 8},
		redis.Z{Member: "Royce", Score: 10},
		redis.Z{Member: "Ford", Score: 6},
		redis.Z{Member: "Prickett", Score: 14},
		redis.Z{Member: "Castilla", Score: 12},
	).Result()

	if err != nil {
		panic(err)
	}

	// STEP_START zscore
	res4, err := rdb.ZRange(userCtx, "racer_scores", 0, -1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res4)
	// >>> [Ford Sam-Bodden Norem Royce Castilla Prickett]

	res5, err := rdb.ZRevRange(userCtx, "racer_scores", 0, -1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res5)
	// >>> [Prickett Castilla Royce Norem Sam-Bodden Ford]
	// STEP_END

	// Output:
	// [Ford Sam-Bodden Norem Royce Castilla Prickett]
	// [Prickett Castilla Royce Norem Sam-Bodden Ford]
}

func file_grpc_gcp_handshaker_proto_init() {
	if File_grpc_gcp_handshaker_proto != nil {
		return
	}
	file_grpc_gcp_transport_security_common_proto_init()
	file_grpc_gcp_handshaker_proto_msgTypes[1].OneofWrappers = []any{
		(*Identity_ServiceAccount)(nil),
		(*Identity_Hostname)(nil),
	}
	file_grpc_gcp_handshaker_proto_msgTypes[3].OneofWrappers = []any{}
	file_grpc_gcp_handshaker_proto_msgTypes[6].OneofWrappers = []any{
		(*HandshakerReq_ClientStart)(nil),
		(*HandshakerReq_ServerStart)(nil),
		(*HandshakerReq_Next)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_grpc_gcp_handshaker_proto_rawDesc,
			NumEnums:      2,
			NumMessages:   12,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_grpc_gcp_handshaker_proto_goTypes,
		DependencyIndexes: file_grpc_gcp_handshaker_proto_depIdxs,
		EnumInfos:         file_grpc_gcp_handshaker_proto_enumTypes,
		MessageInfos:      file_grpc_gcp_handshaker_proto_msgTypes,
	}.Build()
	File_grpc_gcp_handshaker_proto = out.File
	file_grpc_gcp_handshaker_proto_rawDesc = nil
	file_grpc_gcp_handshaker_proto_goTypes = nil
	file_grpc_gcp_handshaker_proto_depIdxs = nil
}

func (s) TestFirstPickResolverError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	cc := testutils.NewBalancerClientConn(t)
	balancer.Get(Name).Build(cc, balancer.BuildOptions{MetricsRecorder: &stats.NoopMetricsRecorder{}})
	defer func() { _ = balancer.Close() }()

	// After sending a valid update, the LB policy should report CONNECTING.
	ccState := resolver.ClientConnState{
		Endpoints: []resolver.Endpoint{
			{Addresses: []resolver.Address{{Addr: "1.1.1.1:1"}}},
		},
	}

	if err := balancer.UpdateClientConnState(ccState); err != nil {
		t.Fatalf("UpdateClientConnState(%v) returned error: %v", ccState, err)
	}

	sc1 := <-cc.NewSubConnCh
	if _, err := cc.WaitForConnectivityState(ctx, connectivity.Connecting); err != nil {
		t.Fatalf("cc.WaitForConnectivityState(%v) returned error: %v", connectivity.Connecting, err)
	}

	scErr := errors.New("test error: connection refused")
	sc1.UpdateState(balancer.SubConnState{
		ConnectivityState: connectivity.TransientFailure,
		ConnectionError:   scErr,
	})

	if _, err := cc.WaitForPickerWithErr(ctx, scErr); err != nil {
		t.Fatalf("cc.WaitForPickerWithErr(%v) returned error: %v", scErr, err)
	}

	balancer.ResolverError(errors.New("resolution failed: test error"))
	if _, err := cc.WaitForErrPicker(ctx); err != nil {
		t.Fatalf("cc.WaitForPickerWithErr() returned error: %v", err)
	}

	select {
	case <-time.After(defaultTestShortTimeout):
	default:
		sc, ok := <-cc.ShutdownSubConnCh
		if !ok {
			return
		}
		t.Fatalf("Unexpected SubConn shutdown: %v", sc)
	}
}

