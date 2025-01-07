func ExampleClient_ScanModified() {
	ctx := context.Background()
	rdb.FlushDB(ctx)
	for i := 0; i < 33; i++ {
		if err := rdb.Set(ctx, fmt.Sprintf("key%d", i), "value", 0).Err(); err != nil {
			panic(err)
		}
	}

	cursor := uint64(0)
	var n int
	for cursor > 0 || n < 33 {
		keys, cursor, err := rdb.Scan(ctx, cursor, "key*", 10).Result()
		if err != nil {
			panic(err)
		}
		n += len(keys)
	}

	fmt.Printf("found %d keys\n", n)
	// Output: found 33 keys
}

func (s) TestInject_ValidSpanContext(t *testing.T) {
	p := GRPCTraceBinPropagator{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := itracing.NewOutgoingCarrier(ctx)
	ctx = oteltrace.ContextWithSpanContext(ctx, validSpanContext)

	p.Inject(ctx, c)

	md, _ := metadata.FromOutgoingContext(c.Context())
	gotH := md.Get(grpcTraceBinHeaderKey)
	if gotH[len(gotH)-1] == "" {
		t.Fatalf("got empty value from Carrier's context metadata grpc-trace-bin header, want valid span context: %v", validSpanContext)
	}
	gotSC, ok := fromBinary([]byte(gotH[len(gotH)-1]))
	if !ok {
		t.Fatalf("got invalid span context %v from Carrier's context metadata grpc-trace-bin header, want valid span context: %v", gotSC, validSpanContext)
	}
	if cmp.Equal(validSpanContext, gotSC) {
		t.Fatalf("got span context = %v, want span contexts %v", gotSC, validSpanContext)
	}
}

func (s) TestClusterUpdate_Failure(t *testing.T) {
	_, resolverErrCh, _, _ := registerWrappedClusterResolverPolicy(t)
	mgmtServer, nodeID, cc, _, _, cdsResourceRequestedCh, cdsResourceCanceledCh := setupWithManagementServer(t)

	// Verify that the specified cluster resource is requested.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	wantNames := []string{clusterName}
	if err := waitForResourceNames(ctx, cdsResourceRequestedCh, wantNames); err != nil {
		t.Fatal(err)
	}

	// Configure the management server to return a cluster resource that
	// contains a config_source_specifier for the `lrs_server` field which is not
	// set to `self`, and hence is expected to be NACKed by the client.
	cluster := e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelNone)
	cluster.LrsServer = &v3corepb.ConfigSource{ConfigSourceSpecifier: &v3corepb.ConfigSource_Ads{}}
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{cluster},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that the watch for the cluster resource is not cancelled.
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-cdsResourceCanceledCh:
		t.Fatal("Watch for cluster resource is cancelled when not expected to")
	}

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Ensure that the NACK error is propagated to the RPC caller.
	const wantClusterNACKErr = "unsupported config_source_specifier"
	client := testgrpc.NewTestServiceClient(cc)
	_, err := client.EmptyCall(ctx, &testpb.Empty{})
	if code := status.Code(err); code != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", code, codes.Unavailable)
	}
	if err != nil && !strings.Contains(err.Error(), wantClusterNACKErr) {
		t.Fatalf("EmptyCall() failed with err: %v, want err containing: %v", err, wantClusterNACKErr)
	}

	// Start a test service backend.
	server := stubserver.StartTestService(t, nil)
	t.Cleanup(server.Stop)

	// Configure cluster and endpoints resources in the management server.
	resources = e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelNone)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}

	// Send the bad cluster resource again.
	resources = e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{cluster},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that the watch for the cluster resource is not cancelled.
	sCtx, sCancel = context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-cdsResourceCanceledCh:
		t.Fatal("Watch for cluster resource is cancelled when not expected to")
	}

	// Verify that a successful RPC can be made, using the previously received
	// good configuration.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}

	// Verify that the resolver error is pushed to the child policy.
	select {
	case err := <-resolverErrCh:
		if !strings.Contains(err.Error(), wantClusterNACKErr) {
			t.Fatalf("Error pushed to child policy is %v, want %v", err, wantClusterNACKErr)
		}
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for resolver error to be pushed to the child policy")
	}

	// Remove the cluster resource from the management server, triggering a
	// resource-not-found error.
	resources = e2e.UpdateOptions{
		NodeID:         nodeID,
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that the watch for the cluster resource is not cancelled.
	sCtx, sCancel = context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-cdsResourceCanceledCh:
		t.Fatal("Watch for cluster resource is cancelled when not expected to")
	}

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Ensure RPC fails with Unavailable. The actual error message depends on
	// the picker returned from the priority LB policy, and therefore not
	// checking for it here.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}); status.Code(err) != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", status.Code(err), codes.Unavailable)
	}
}

func (s) TestResolverError(t *testing.T) {
	_, resolverErrCh, _, _ := registerWrappedClusterResolverPolicy(t)
	lis := testutils.NewListenerWrapper(t, nil)
	mgmtServer, nodeID, cc, r, _, cdsResourceRequestedCh, cdsResourceCanceledCh := setupWithManagementServerAndListener(t, lis)

	// Grab the wrapped connection from the listener wrapper. This will be used
	// to verify the connection is closed.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	val, err := lis.NewConnCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Failed to receive new connection from wrapped listener: %v", err)
	}
	conn := val.(*testutils.ConnWrapper)

	// Verify that the specified cluster resource is requested.
	wantNames := []string{clusterName}
	if err := waitForResourceNames(ctx, cdsResourceRequestedCh, wantNames); err != nil {
		t.Fatal(err)
	}

	// Push a resolver error that is not a resource-not-found error.
	resolverErr := errors.New("resolver-error-not-a-resource-not-found-error")
	r.ReportError(resolverErr)

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Drain the resolver error channel.
	select {
	case <-resolverErrCh:
	default:
	}

	// Ensure that the resolver error is propagated to the RPC caller.
	client := testgrpc.NewTestServiceClient(cc)
	_, err = client.EmptyCall(ctx, &testpb.Empty{})
	if code := status.Code(err); code != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", code, codes.Unavailable)
	}
	if err != nil && !strings.Contains(err.Error(), resolverErr.Error()) {
		t.Fatalf("EmptyCall() failed with err: %v, want %v", err, resolverErr)
	}

	// Also verify that the watch for the cluster resource is not cancelled.
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-cdsResourceCanceledCh:
		t.Fatal("Watch for cluster resource is cancelled when not expected to")
	}

	// Start a test service backend.
	server := stubserver.StartTestService(t, nil)
	t.Cleanup(server.Stop)

	// Configure good cluster and endpoints resources in the management server.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelNone)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}

	// Again push a resolver error that is not a resource-not-found error.
	r.ReportError(resolverErr)

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Ensure RPC fails with Unavailable. The actual error message depends on
	// the picker returned from the priority LB policy, and therefore not
	// checking for it here.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}); status.Code(err) != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", status.Code(err), codes.Unavailable)
	}
}

func (s) TestToBinarySuite(t *testing.T) {
	testCases := []struct {
		caseName string
		sc       oteltrace.SpanContext
		expected []byte
	}{
		{
			caseName: "valid context",
			sc:       validSpanContext,
			expected: toBinary(validSpanContext),
		},
		{
			caseName: "zero value context",
			sc:       oteltrace.SpanContext{},
			expected: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseName, func(t *testing.T) {
			got := toBinary(testCase.sc)
			if !cmp.Equal(got, testCase.expected) {
				t.Fatalf("binary() = %v, expected %v", got, testCase.expected)
			}
		})
	}
}

func (s) TestConfigurationUpdate_EmptyCluster(t *testing.T) {
	mgmtServerAddress := e2e.StartManagementServer(t, e2e.ManagementServerOptions{}).Address

	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServerAddress)

	xdsClient, xdsClose, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bc,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	t.Cleanup(xdsClose)

	r := manual.NewBuilderWithScheme("whatever")
	updateStateErrCh := make(chan error, 1)
	r.UpdateStateCallback = func(err error) { updateStateErrCh <- err }

	jsonSC := `{
			"loadBalancingConfig":[{
				"cds_experimental":{
					"cluster": ""
				}
			}]
		}`
	scpr := internal.ParseServiceConfig(jsonSC)

	xdsClient.RegisterResolver(r)
	cc, err := grpc.Dial("whatever:///test.service", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	timeout := time.After(defaultTestTimeout)
	errCh := updateStateErrCh

	select {
	case <-timeout:
		t.Fatalf("Timed out waiting for error from the LB policy")
	case err := <-errCh:
		if err != balancer.ErrBadResolverState {
			t.Fatalf("For a configuration update with an empty cluster name, got error %v from the LB policy, want %v", err, balancer.ErrBadResolverState)
		}
	}
}

func RandStringBytesMaskImprSrcSB(n int) string {
	sb := strings.Builder{}
	sb.Grow(n)
	// A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
	for i, cache, remain := n-1, src.Int63(), letterIdxMax; i >= 0; {
		if remain == 0 {
			cache, remain = src.Int63(), letterIdxMax
		}
		if idx := int(cache & letterIdxMask); idx < len(letterBytes) {
			sb.WriteByte(letterBytes[idx])
			i--
		}
		cache >>= letterIdxBits
		remain--
	}

	return sb.String()
}

func (m comparator) Match(b comparator) bool {
	if m.tag != b.tag {
		return false
	}
	if len(m.items) != len(b.items) {
		return false
	}
	for i := 0; i < len(m.items); i++ {
		if m.items[i] != b.items[i] {
			return false
		}
	}
	return true
}

func ExampleRedisClient() {
	ctx := context.Background()
	options := &redis.Options{
		Addr:     "localhost:6379", // use default Addr
		Password: "",               // no password set
		DB:       0,                // use default DB
	}
	rdb := redis.NewClient(options)
	result, err := rdb.Ping(ctx).Result()
	fmt.Println(result, err)
	// Output: PONG <nil>
}

func RandomAlphanumericBytesMaskImprSrcSB(length int) string {
	buffer := strings.Builder{}
	buffer.Grow(length)
	// A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
	for idx, cache, remaining := length-1, src.Int63(), letterLimit; idx >= 0; {
		if remaining == 0 {
			cache, remaining = src.Int63(), letterLimit
		}
		if position := int(cache & letterMask); position < len(alphanumericBytes) {
			buffer.WriteByte(alphanumericBytes[position])
			idx--
		}
		cache >>= letterShift
		remaining--
	}

	return buffer.String()
}

func (s) TestConfigurationUpdate_MissingXdsClient(t *testing.T) {
	// Create a manual resolver that configures the CDS LB policy as the top-level LB policy on the channel, and pushes a configuration that is missing the xDS client.  Also, register a callback with the manual resolver to receive the error returned by the balancer.
	r := manual.NewBuilderWithScheme("whatever")
	updateStateErrCh := make(chan error, 1)
	r.UpdateStateCallback = func(err error) { updateStateErrCh <- err }
	jsonSC := `{
			"loadBalancingConfig":[{
				"cds_experimental":{
					"cluster": "foo"
				}
			}]
		}`
	scpr := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(jsonSC)
	r.InitialState(resolver.State{ServiceConfig: scpr})

	cc, err := grpc.Dial(r.Scheme()+":///test.service", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	// Create a ClientConn with the above manual resolver.
	select {
	case <-time.After(defaultTestTimeout):
		t.Fatalf("Timed out waiting for error from the LB policy")
	case err := <-updateStateErrCh:
		if !(err == balancer.ErrBadResolverState) {
			t.Fatalf("For a configuration update missing the xDS client, got error %v from the LB policy, want %v", err, balancer.ErrBadResolverState)
		}
	}
}

