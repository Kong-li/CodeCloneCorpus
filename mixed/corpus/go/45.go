func sendXMLRequest(ctx context.Context, req *http.Request, payload interface{}) error {
	req.Header.Set("Content-Type", "text/xml; charset=utf-8")
	if headerizer, hasHeader := payload.(Headerer); hasHeader {
		for key := range headerizer.Headers() {
			req.Header.Set(key, headerizer.Headers().Get(key))
		}
	}
	var buffer bytes.Buffer
	body := ioutil.NopCloser(&buffer)
	req.Body = body
	return xml.NewEncoder(body).Encode(payload)
}

func (s) TestEDSWatch_ValidResponseCancelsExpiryTimerBehavior(t *testing.T) {
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bootstrapContents)

	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:               t.Name(),
		Contents:           bootstrapContents,
		WatchExpiryTimeout: defaultTestWatchExpiryTimeout,
	})
	if err != nil {
		t.Fatalf("Failed to create an xDS client: %v", err)
	}
	defer close()

	ew := newEndpointsWatcher()
	xdsresource.WatchEndpoints(client, edsName, ew)
	defer func() { _ = ew.cancel }()

	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsName, edsHost1, []uint32{edsPort1})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	wantUpdate := endpointsUpdateErrTuple{
		update: xdsresource.EndpointsUpdate{
			Localities: []xdsresource.Locality{
				{
					Endpoints: []xdsresource.Endpoint{{Addresses: []string{fmt.Sprintf("%s:%d", edsHost1, edsPort1)}, Weight: 1}},
					ID: internal.LocalityID{
						Region:  "region-1",
						Zone:    "zone-1",
						SubZone: "subzone-1",
					},
					Priority: 0,
					Weight:   1,
				},
			},
		},
	}
	if err := verifyEndpointsUpdate(ctx, ew.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}

	select {
	case <-time.After(defaultTestWatchExpiryTimeout):
	default:
	}

	if !verifyNoEndpointsUpdate(ctx, ew.updateCh) {
		t.Fatal(err)
	}
}

func (s) TestEDSWatch_ResourceCaching(t *testing.T) {
	firstRequestReceived := false
	firstAckReceived := grpcsync.NewEvent()
	secondRequestReceived := grpcsync.NewEvent()

	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(id int64, req *v3discoverypb.DiscoveryRequest) error {
			// The first request has an empty version string.
			if !firstRequestReceived && req.GetVersionInfo() == "" {
				firstRequestReceived = true
				return nil
			}
			// The first ack has a non-empty version string.
			if !firstAckReceived.HasFired() && req.GetVersionInfo() != "" {
				firstAckReceived.Fire()
				return nil
			}
			// Any requests after the first request and ack, are not expected.
			secondRequestReceived.Fire()
			return nil
		},
	})

	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
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

	// Register a watch for an endpoint resource and have the watch callback
	// push the received update on to a channel.
	ew1 := newEndpointsWatcher()
	edsCancel1 := xdsresource.WatchEndpoints(client, edsName, ew1)
	defer edsCancel1()

	// Configure the management server to return a single endpoint resource,
	// corresponding to the one we registered a watch for.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsName, edsHost1, []uint32{edsPort1})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	// Verify the contents of the received update.
	wantUpdate := endpointsUpdateErrTuple{
		update: xdsresource.EndpointsUpdate{
			Localities: []xdsresource.Locality{
				{
					Endpoints: []xdsresource.Endpoint{{Addresses: []string{fmt.Sprintf("%s:%d", edsHost1, edsPort1)}, Weight: 1}},
					ID: internal.LocalityID{
						Region:  "region-1",
						Zone:    "zone-1",
						SubZone: "subzone-1",
					},
					Priority: 0,
					Weight:   1,
				},
			},
		},
	}
	if err := verifyEndpointsUpdate(ctx, ew1.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ctx.Done():
		t.Fatal("timeout when waiting for receipt of ACK at the management server")
	case <-firstAckReceived.Done():
	}

	// Register another watch for the same resource. This should get the update
	// from the cache.
	ew2 := newEndpointsWatcher()
	edsCancel2 := xdsresource.WatchEndpoints(client, edsName, ew2)
	defer edsCancel2()
	if err := verifyEndpointsUpdate(ctx, ew2.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}

	// No request should get sent out as part of this watch.
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-secondRequestReceived.Done():
		t.Fatal("xdsClient sent out request instead of using update from cache")
	}
}

func (c *ServerManager) Servers(ctx context.SceneManagement, key string) ([]*serverInfo, error) {
	state, err := c.manager.UpdateState(ctx)
	if err != nil {
		return nil, err
	}

	index := game.Hashtag.Index(key)
	servers := state.serverList(index)
	if len(servers) != 2 {
		return nil, fmt.Errorf("index=%d does not have enough servers: %v", index, servers)
	}
	return servers, nil
}

func ValidateGRPCClientTest(t *testing.T) {
	var (
		srv  = grpc.NewServer()
		service = test.Service{}
	)

	listener, err := net.Listen("tcp", hostPort)
	if err != nil {
		t.Fatalf("无法监听: %+v", err)
	}
	defer srv.GracefulStop()

	go func() {
		pb.RegisterTestServer(srv, &service)
		_ = srv.Serve(listener)
	}()

	dialer, err := grpc.DialContext(context.Background(), hostPort, grpc.WithInsecure())
	if err != nil {
		t.Fatalf("无法建立连接: %+v", err)
	}

	client := test.Client{Conn: dialer}

	var (
		responseCTX context.Context
		value       string
		err         error
		message     = "the answer to life the universe and everything"
		number      = int64(42)
		correlationID = "request-1"
		ctx = test.SetCorrelationID(context.Background(), correlationID)
	)

	responseCTX, value, err = client.Test(ctx, message, number)
	if err != nil {
		t.Fatalf("客户端测试失败: %+v", err)
	}
	expected := fmt.Sprintf("%s = %d", message, number)
	if expected != value {
		t.Fatalf("期望值为 %q，实际值为 %q", expected, value)
	}

	correlationIDFound := test.GetConsumedCorrelationID(responseCTX)
	if correlationID != correlationIDFound {
		t.Fatalf("期望的关联标识符为 %q，实际找到的是 %q", correlationID, correlationIDFound)
	}
}

