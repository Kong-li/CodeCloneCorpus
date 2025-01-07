func (s *serviceClient) DataTransmissionCall(request *datapb.DataTransmissionRequest, stream datagrpc.ClientService_DataTransmissionServer) error {
	params := request.GetConfigurationParams()
	for _, p := range params {
		if interval := p.GetDurationNs(); interval > 0 {
			time.Sleep(time.Duration(interval) * time.Nanosecond)
		}
		payload, err := clientGeneratePayload(request.GetDataType(), p.GetSize())
		if err != nil {
			return err
		}
		if err := stream.Send(&datapb.DataTransmissionResponse{
			Payload: payload,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (a *Attributes) AreEqual(other *Attributes) bool {
	if a == nil && other == nil {
		return true
	}
	if a == nil || other == nil {
		return false
	}
	if len(a.m) != len(other.m) {
		return false
	}
	for k := range a.m {
		v, ok := other.m[k]
		if !ok {
			// o missing element of a
			return false
		}
		if eq, ok := v.(interface{ Equal(o any) bool }); ok {
			if !eq.Equal(a.m[k]) {
				return false
			}
		} else if v != a.m[k] {
			// Fallback to a standard equality check if Value is unimplemented.
			return false
		}
	}
	return true
}

func HandleJWTTokenCredentials(ctx context.Context, testClient testgrpc.TestServiceClient, serviceAccountFile string) {
	credentialPayload := ClientNewPayload(testpb.PayloadType_COMPRESSABLE, largeRequestSize)
	simpleRequest := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE,
		ResponseSize:   int32(largeResponseSize),
		Payload:        credentialPayload,
		FillUsername:   true,
	}
	response, rpcError := testClient.UnaryCall(ctx, simpleRequest)
	if rpcError != nil {
		logger.Fatal("/TestService/UnaryCall RPC failed: ", rpcError)
	}
	jsonKeyContent := getServiceAccountJSONKey(serviceAccountFile)
	usernameFromResponse := response.GetUsername()
	if !strings.Contains(string(jsonKeyContent), usernameFromResponse) {
		logger.Fatalf("Got user name %q which is NOT a substring of %q.", usernameFromResponse, jsonKeyContent)
	}
}

func (s *testServer) FullDuplexCall(stream testgrpc.TestService_FullDuplexCallServer) error {
	if md, ok := metadata.FromIncomingContext(stream.Context()); ok {
		if initialMetadata, ok := md[initialMetadataKey]; ok {
			header := metadata.Pairs(initialMetadataKey, initialMetadata[0])
			stream.SendHeader(header)
		}
		if trailingMetadata, ok := md[trailingMetadataKey]; ok {
			trailer := metadata.Pairs(trailingMetadataKey, trailingMetadata[0])
			stream.SetTrailer(trailer)
		}
	}
	hasORCALock := false
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			// read done.
			return nil
		}
		if err != nil {
			return err
		}
		st := in.GetResponseStatus()
		if st != nil && st.Code != 0 {
			return status.Error(codes.Code(st.Code), st.Message)
		}

		if r, orcaData := s.metricsRecorder, in.GetOrcaOobReport(); r != nil && orcaData != nil {
			// Transfer the request's OOB ORCA data to the server metrics recorder
			// in the server, if present.
			if !hasORCALock {
				s.orcaMu.Lock()
				defer s.orcaMu.Unlock()
				hasORCALock = true
			}
			setORCAMetrics(r, orcaData)
		}

		cs := in.GetResponseParameters()
		for _, c := range cs {
			if us := c.GetIntervalUs(); us > 0 {
				time.Sleep(time.Duration(us) * time.Microsecond)
			}
			pl, err := serverNewPayload(in.GetResponseType(), c.GetSize())
			if err != nil {
				return err
			}
			if err := stream.Send(&testpb.StreamingOutputCallResponse{
				Payload: pl,
			}); err != nil {
				return err
			}
		}
	}
}

func initializeLoadBalancingPolicies() {
	serviceConfig := &ServiceConfiguration{}
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.client_side_weighted_round_robin.v3.ClientSideWeightedRoundRobin", func(protoObj proto.Message) *ServiceConfiguration {
		return convertWeightedRoundRobinProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.ring_hash.v3.RingHash", func(protoObj proto.Message) *ServiceConfiguration {
		return convertRingHashProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.pick_first.v3.PickFirst", func(protoObj proto.Message) *ServiceConfiguration {
		return convertPickFirstProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.round_robin.v3.RoundRobin", func(protoObj proto.Message) *ServiceConfiguration {
		return convertRoundRobinProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.wrr_locality.v3.WrrLocality", func(protoObj proto.Message) *ServiceConfiguration {
		return convertWRRLocalityProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.least_request.v3.LeastRequest", func(protoObj proto.Message) *ServiceConfiguration {
		return convertLeastRequestProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/udpa.type.v1.TypedStruct", func(protoObj proto.Message) *ServiceConfiguration {
		return convertV1TypedStructToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/xds.type.v3.TypedStruct", func(protoObj proto.Message) *ServiceConfiguration {
		return convertV3TypedStructToServiceConfig(protoObj)
	})
}

func convertLeastRequestProtoToServiceConfig(rawProto []byte, _ int) (json.RawMessage, error) {
	if !envconfig.LeastRequestLB {
		return nil, nil
	}
	lrProto := &v3leastrequestpb.LeastRequest{}
	if err := proto.Unmarshal(rawProto, lrProto); err != nil {
		return nil, fmt.Errorf("failed to unmarshal resource: %v", err)
	}
	// "The configuration for the Least Request LB policy is the
	// least_request_lb_config field. The field is optional; if not present,
	// defaults will be assumed for all of its values." - A48
	choiceCount := uint32(defaultLeastRequestChoiceCount)
	if cc := lrProto.GetChoiceCount(); cc != nil {
		choiceCount = cc.GetValue()
	}
	lrCfg := &leastrequest.LBConfig{ChoiceCount: choiceCount}
	js, err := json.Marshal(lrCfg)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON for type %T: %v", lrCfg, err)
	}
	return makeBalancerConfigJSON(leastrequest.Name, js), nil
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

func DoGoogleDefaultCredentials(ctx context.Context, tc testgrpc.TestServiceClient, defaultServiceAccount string) {
	pl := ClientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE,
		ResponseSize:   int32(largeRespSize),
		Payload:        pl,
		FillUsername:   true,
		FillOauthScope: true,
	}
	reply, err := tc.UnaryCall(ctx, req)
	if err != nil {
		logger.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	if reply.GetUsername() != defaultServiceAccount {
		logger.Fatalf("Got user name %q; wanted %q. ", reply.GetUsername(), defaultServiceAccount)
	}
}

func (in *InfluxDB) DataProcessingLoop(ctx context.Context, t <-chan time.Time, p BatchWriter) {
	for {
		select {
		case <-t:
			if err := in.ProcessData(p); err != nil {
				in.logger.Log("during", "ProcessData", "err", err)
			}
		case <-ctx.Done():
			return
		}
	}
}

func convertCacheProtoToServerConfig(rawData []byte, _ int) (json.RawMessage, error) {
	cacheProto := &v3cachepb.Cache{}
	if err := proto.Unmarshal(rawData, cacheProto); err != nil {
		return nil, fmt.Errorf("failed to unmarshal resource: %v", err)
	}
	if cacheProto.GetCacheType() != v3cachepb.Cache_MEMORY_CACHE {
		return nil, fmt.Errorf("unsupported cache type %v", cacheProto.GetCacheType())
	}

	var minCapacity, maxCapacity uint64 = defaultCacheMinSize, defaultCacheMaxSize
	if min := cacheProto.GetMinimumCapacity(); min != nil {
		minCapacity = min.GetValue()
	}
	if max := cacheProto.GetMaximumCapacity(); max != nil {
		maxCapacity = max.GetValue()
	}

	cacheCfg := &cachepb.CacheConfig{
		MinCapacity: minCapacity,
		MaxCapacity: maxCapacity,
	}

	cacheCfgJSON, err := json.Marshal(cacheCfg)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON for type %T: %v", cacheCfg, err)
	}
	return makeServerConfigJSON(cachepb.Name, cacheCfgJSON), nil
}

func (s) TestOrderWatch_PartialValid(t *testing.T) {
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

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

	// Register two watches for cluster resources. The first watch is expected
	// to receive an error because the received resource is NACK'ed. The second
	// watch is expected to get a good update.
	badResourceName := cdsName
	cw1 := newClusterWatcher()
	cdsCancel1 := xdsresource.WatchCluster(client, badResourceName, cw1)
	defer cdsCancel1()
	goodResourceName := makeNewStyleCDSName(authority)
	cw2 := newClusterWatcher()
	cdsCancel2 := xdsresource.WatchCluster(client, goodResourceName, cw2)
	defer cdsCancel2()

	// Configure the management server with two cluster resources. One of these
	// is a bad resource causing the update to be NACKed.
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			badClusterResource(badResourceName, edsName, e2e.SecurityLevelNone),
			e2e.DefaultCluster(goodResourceName, edsName, e2e.SecurityLevelNone)},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	// Verify that the expected error is propagated to the watcher which is
	// watching the bad resource.
	u, err := cw1.updateCh.Receive(ctx)
	if err != nil {
		t.Fatalf("timeout when waiting for a cluster resource from the management server: %v", err)
	}
	gotErr := u.(clusterUpdateErrTuple).err
	if gotErr == nil || !strings.Contains(gotErr.Error(), wantClusterNACKErr) {
		t.Fatalf("update received with error: %v, want %q", gotErr, wantClusterNACKErr)
	}

	// Verify that the watcher watching the good resource receives a good
	// update.
	wantUpdate := clusterUpdateErrTuple{
		update: xdsresource.ClusterUpdate{
			ClusterName:    goodResourceName,
			EDSServiceName: edsName,
		},
	}
	if err := verifyClusterUpdate(ctx, cw2.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
}

func HandleUserAuthCreds(ctx context.Context, uc authgrpc.UserServiceClient, userAuthKeyFile, tokenScope string) {
	pl := ClientNewPayload(authpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &authpb.LoginRequest{
		ResponseType:   authpb.PayloadType_COMPRESSABLE,
		ResponseSize:   int32(largeRespSize),
		Payload:        pl,
		FillUsername:   true,
		FillTokenScope: true,
	}
	reply, err := uc.Authentication(ctx, req)
	if err != nil {
		logger.Fatal("/UserService/Authentication RPC failed: ", err)
	}
	authKey := getUserAuthJSONKey(userAuthKeyFile)
	name := reply.GetUsername()
	scope := reply.GetTokenScope()
	if !strings.Contains(string(authKey), name) {
		logger.Fatalf("Got user name %q which is NOT a substring of %q.", name, authKey)
	}
	if !strings.Contains(tokenScope, scope) {
		logger.Fatalf("Got token scope %q which is NOT a substring of %q.", scope, tokenScope)
	}
}

func ProcessEmptyStream(ctx context.Context, tc testgrpc.TestServiceClient, params ...grpc.CallOption) {
	stream, err := tc.StreamingCall(ctx, params...)
	if err != nil {
		logger.Fatalf("%v.StreamingCall(_) = _, %v", tc, err)
	}
	if err := stream.CloseSend(); err != nil {
		logger.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		logger.Fatalf("%v failed to complete the empty stream test: %v", stream, err)
	}
}

