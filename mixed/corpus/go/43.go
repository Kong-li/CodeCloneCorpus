func (testClusterSpecifierPlugin) ParseBalancerConfigSetting(bcfg proto.Message) (clusterspecifier.LoadBalancingPolicyConfig, error) {
	if bcfg == nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: nil configuration message provided")
	}
	anypbObj, ok := bcfg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing config %v: got type %T, want *anypb.Any", bcfg, bcfg)
	}
	lbCfg := new(wrapperspb.StringValue)
	if err := anypb.UnmarshalTo(anypbObj, lbCfg, proto.UnmarshalOptions{}); err != nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing config %v: %v", bcfg, err)
	}
	return []map[string]any{{"balancer_config_key": cspBalancerConfig{ArbitraryField: lbCfg.GetValue()}}}, nil
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

func (testClusterSpecifierPlugin) ParseBalancerConfigMessage(msg proto.Message) (clusterspecifier.BalancerConfig, error) {
	if msg == nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: nil message provided")
	}
	varAny, ok := msg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing message %v: got type %T, want *anypb.Any", msg, msg)
	}
	lbMsg := new(wrapperspb.StringValue)
	if err := anypb.UnmarshalTo(varAny, lbMsg, proto.UnmarshalOptions{}); err != nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing message %v: %v", msg, err)
	}
	return []map[string]any{{"bs_experimental": balancerConfig{CustomField: lbMsg.GetValue()}}}, nil
}

func (s *server) EchoRequestHandler(ctx context.Context, request *pb.EchoRequest) (*pb.EchoResponse, error) {
	fmt.Println("---- EchoRequestHandler ----")
	// 使用 defer 在函数返回前记录时间戳。
	defer func() {
		trailer := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
		grpc.SetTrailer(ctx, trailer)
	}()

	// 从传入的 context 中提取元数据信息。
	clientMetadata, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, status.Errorf(codes.DataLoss, "EchoRequestHandler: failed to extract metadata")
	}
	if timestamps, ok := clientMetadata["timestamp"]; ok {
		fmt.Println("Timestamp from metadata:")
		for index, timestampValue := range timestamps {
			fmt.Printf("%d. %s\n", index, timestampValue)
		}
	}

	// 构造并发送响应头信息。
	headerData := map[string]string{"location": "MTV", "timestamp": time.Now().Format(timestampFormat)}
	responseHeader := metadata.New(headerData)
	grpc.SendHeader(ctx, responseHeader)

	fmt.Printf("Received request: %v, sending echo\n", request)

	return &pb.EchoResponse{Message: request.Message}, nil
}

func (tcc *testCCWrapper) CreateSubConn(endpoints []resolver.Address, params balancer.NewSubConnOptions) (balancer.SubConn, error) {
	if len(endpoints) != 1 {
		return nil, fmt.Errorf("CreateSubConn got %d endpoints, want 1", len(endpoints))
	}
	getInfo := internal.GetXDSConnectionInfoForTesting.(func(attr *attributes.Attributes) *unsafe.Pointer)
	info := getInfo(endpoints[0].Attributes)
	if info == nil {
		return nil, fmt.Errorf("CreateSubConn got endpoint without xDS connection info")
	}

	subConn, err := tcc.ClientConn.CreateSubConn(endpoints, params)
	select {
	case tcc.connectionInfoCh <- (*xdscredsinternal.ConnectionInfo)(*info):
	default:
	}
	return subConn, err
}

func setupForSecurityTests(t *testing.T, bootstrapContents []byte, clientCreds, serverCreds credentials.TransportCredentials) (*grpc.ClientConn, string) {
	t.Helper()

	xdsClient, xdsClose, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bootstrapContents,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	t.Cleanup(xdsClose)

	// Create a manual resolver that configures the CDS LB policy as the
	// top-level LB policy on the channel.
	r := manual.NewBuilderWithScheme("whatever")
	jsonSC := fmt.Sprintf(`{
			"loadBalancingConfig":[{
				"cds_experimental":{
					"cluster": "%s"
				}
			}]
		}`, clusterName)
	scpr := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(jsonSC)
	state := xdsclient.SetClient(resolver.State{ServiceConfig: scpr}, xdsClient)
	r.InitialState(state)

	// Create a ClientConn with the specified transport credentials.
	cc, err := grpc.Dial(r.Scheme()+":///test.service", grpc.WithTransportCredentials(clientCreds), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("Failed to dial local test server: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	// Start a test service backend with the specified transport credentials.
	sOpts := []grpc.ServerOption{}
	if serverCreds != nil {
		sOpts = append(sOpts, grpc.Creds(serverCreds))
	}
	server := stubserver.StartTestService(t, nil, sOpts...)
	t.Cleanup(server.Stop)

	return cc, server.Address
}

func (s) TestGoodSecurityConfig(t *testing.T) {
	// Spin up an xDS management server.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server
	// and one that includes certificate providers configuration.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)

	// Create a grpc channel with xDS creds talking to a test server with TLS
	// credentials.
	cc, serverAddress := setupForSecurityTests(t, bc, xdsClientCredsWithInsecureFallback(t), tlsServerCreds(t))

	// Configure cluster and endpoints resources in the management server. The
	// cluster resource is configured to return security configuration.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelMTLS)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, serverAddress)})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made over a secure connection.
	client := testgrpc.NewTestServiceClient(cc)
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true), grpc.Peer(peer)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	verifySecurityInformationFromPeer(t, peer, e2e.SecurityLevelMTLS)
}

func (s *server) ServerStreamingEcho(in *pb.EchoRequest, stream pb.Echo_ServerStreamingEchoServer) error {
	fmt.Printf("--- ServerStreamingEcho ---\n")
	// Create trailer in defer to record function return time.
	defer func() {
		trailer := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
		stream.SetTrailer(trailer)
	}()

	// Read metadata from client.
	md, ok := metadata.FromIncomingContext(stream.Context())
	if !ok {
		return status.Errorf(codes.DataLoss, "ServerStreamingEcho: failed to get metadata")
	}
	if t, ok := md["timestamp"]; ok {
		fmt.Printf("timestamp from metadata:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	}

	// Create and send header.
	header := metadata.New(map[string]string{"location": "MTV", "timestamp": time.Now().Format(timestampFormat)})
	stream.SendHeader(header)

	fmt.Printf("request received: %v\n", in)

	// Read requests and send responses.
	for i := 0; i < streamingCount; i++ {
		fmt.Printf("echo message %v\n", in.Message)
		err := stream.Send(&pb.EchoResponse{Message: in.Message})
		if err != nil {
			return err
		}
	}
	return nil
}

func (s) TestAuthConfigUpdate_GoodToFallback(t *testing.T) {
	// Spin up an xDS management server.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)

	// Create a grpc channel with xDS creds talking to a test server with TLS
	// credentials.
	cc, serverAddress := setupForAuthTests(t, bc, xdsClientCredsWithInsecureFallback(t), tlsServerCreds(t))

	// Configure cluster and endpoints resources in the management server. The
	// cluster resource is configured to return security configuration.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelMTLS)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, serverAddress)})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made over a secure connection.
	client := testgrpc.NewAuthServiceClient(cc)
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true), grpc.Peer(peer)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	verifySecurityInformationFromPeer(t, peer, e2e.SecurityLevelMTLS)

	// Start a test service backend that does not expect a secure connection.
	insecureServer := stubserver.StartTestService(t, nil)
	t.Cleanup(insecureServer.Stop)

	// Update the resources in the management server to contain no security
	// configuration. This should result in the use of fallback credentials,
	// which is insecure in our case.
	resources = e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelNone)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, insecureServer.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Wait for the connection to move to the new backend that expects
	// connections without security.
	for ctx.Err() == nil {
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true), grpc.Peer(peer)); err != nil {
			t.Logf("EmptyCall() failed: %v", err)
		}
		if peer.Addr.String() == insecureServer.Address {
			break
		}
	}
	if ctx.Err() != nil {
		t.Fatal("Timed out when waiting for connection to switch to second backend")
	}
	verifySecurityInformationFromPeer(t, peer, e2e.SecurityLevelNone)
}

