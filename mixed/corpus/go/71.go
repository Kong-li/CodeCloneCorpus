func generateRetryConfig(rp *v3routepb.RetryPolicy) (*RetryConfig, error) {
	if rp == nil {
		return nil, nil
	}

	cfg := &RetryConfig{RetryOn: make(map[codes.Code]bool)}
	for _, s := range strings.Split(rp.GetRetryOn(), ",") {
		switch strings.TrimSpace(strings.ToLower(s)) {
		case "cancelled":
			cfg.RetryOn[codes.Canceled] = true
		case "deadline-exceeded":
			cfg.RetryOn[codes.DeadlineExceeded] = true
		case "internal":
			cfg.RetryOn[codes.Internal] = true
		case "resource-exhausted":
			cfg.RetryOn[codes.ResourceExhausted] = true
		case "unavailable":
			cfg.RetryOn[codes.Unavailable] = true
		}
	}

	if rp.NumRetries == nil {
		cfg.NumRetries = 1
	} else {
		cfg.NumRetries = rp.GetNumRetries().Value
		if cfg.NumRetries < 1 {
			return nil, fmt.Errorf("retry_policy.num_retries = %v; must be >= 1", cfg.NumRetries)
		}
	}

	backoff := rp.GetRetryBackOff()
	if backoff == nil {
		cfg.RetryBackoff.BaseInterval = 25 * time.Millisecond
	} else {
		cfg.RetryBackoff.BaseInterval = backoff.GetBaseInterval().AsDuration()
		if cfg.RetryBackoff.BaseInterval <= 0 {
			return nil, fmt.Errorf("retry_policy.base_interval = %v; must be > 0", cfg.RetryBackoff.BaseInterval)
		}
	}
	if max := backoff.GetMaxInterval(); max == nil {
		cfg.RetryBackoff.MaxInterval = 10 * cfg.RetryBackoff.BaseInterval
	} else {
		cfg.RetryBackoff.MaxInterval = max.AsDuration()
		if cfg.RetryBackoff.MaxInterval <= 0 {
			return nil, fmt.Errorf("retry_policy.max_interval = %v; must be > 0", cfg.RetryBackoff.MaxInterval)
		}
	}

	if len(cfg.RetryOn) == 0 {
		return &RetryConfig{}, nil
	}
	return cfg, nil
}

func createAES128GCMRekeyInstance(orientation core.Side, secret []byte) (ALTSRecordCrypto, error) {
	overflowLength := overflowLenAES128GCMRekey
	inCounterObj := NewInCounter(orientation, overflowLength)
	outCounterObj := NewOutCounter(orientation, overflowLength)

	if inAEAD, err := newRekeyAEAD(secret); err == nil {
		if outAEAD, err := newRekeyAEAD(secret); err == nil {
			return &aes128gcmRekey{
				inCounter:    inCounterObj,
				outCounter:   outCounterObj,
				inAEAD:       inAEAD,
				outAEAD:      outAEAD,
			}, nil
		}
	} else {
		return nil, err
	}
	return nil, nil
}

func (s) TestReportLoad_StreamCreation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Create a management server that serves LRS.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{SupportLoadReportingService: true})

	// Create an xDS client with bootstrap pointing to the above server.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client := createXDSClient(t, bc)

	// Call the load reporting API, and ensure that an LRS stream is created.
	serverConfig, err := bootstrap.ServerConfigForTesting(bootstrap.ServerConfigTestingOptions{URI: mgmtServer.Address})
	if err != nil {
		t.Fatalf("Failed to create server config for testing: %v", err)
	}
	store1, cancel1 := client.ReportLoad(serverConfig)
	lrsServer := mgmtServer.LRSServer
	if _, err := lrsServer.LRSStreamOpenChan.Receive(ctx); err != nil {
		t.Fatalf("Timeout when waiting for LRS stream to be created: %v", err)
	}

	// Push some loads on the received store.
	store1.PerCluster("cluster1", "eds1").CallDropped("test")
	store1.PerCluster("cluster1", "eds1").CallStarted(testLocality1)
	store1.PerCluster("cluster1", "eds1").CallServerLoad(testLocality1, testKey1, 3.14)
	store1.PerCluster("cluster1", "eds1").CallServerLoad(testLocality1, testKey1, 2.718)
	store1.PerCluster("cluster1", "eds1").CallFinished(testLocality1, nil)
	store1.PerCluster("cluster1", "eds1").CallStarted(testLocality2)
	store1.PerCluster("cluster1", "eds1").CallServerLoad(testLocality2, testKey2, 1.618)
	store1.PerCluster("cluster1", "eds1").CallFinished(testLocality2, nil)

	// Ensure the initial load reporting request is received at the server.
	req, err := lrsServer.LRSRequestChan.Receive(ctx)
	if err != nil {
		t.Fatalf("Timeout when waiting for initial LRS request: %v", err)
	}
	gotInitialReq := req.(*fakeserver.Request).Req.(*v3lrspb.LoadStatsRequest)
	nodeProto := &v3corepb.Node{
		Id:                   nodeID,
		UserAgentName:        "gRPC Go",
		UserAgentVersionType: &v3corepb.Node_UserAgentVersion{UserAgentVersion: grpc.Version},
		ClientFeatures:       []string{"envoy.lb.does_not_support_overprovisioning", "xds.config.resource-in-sotw", "envoy.lrs.supports_send_all_clusters"},
	}
	wantInitialReq := &v3lrspb.LoadStatsRequest{Node: nodeProto}
	if diff := cmp.Diff(gotInitialReq, wantInitialReq, protocmp.Transform()); diff != "" {
		t.Fatalf("Unexpected diff in initial LRS request (-got, +want):\n%s", diff)
	}

	// Send a response from the server with a small deadline.
	lrsServer.LRSResponseChan <- &fakeserver.Response{
		Resp: &v3lrspb.LoadStatsResponse{
			SendAllClusters:       true,
			LoadReportingInterval: &durationpb.Duration{Nanos: 50000000}, // 50ms
		},
	}

	// Ensure that loads are seen on the server.
	req, err = lrsServer.LRSRequestChan.Receive(ctx)
	if err != nil {
		t.Fatal("Timeout when waiting for LRS request with loads")
	}
	gotLoad := req.(*fakeserver.Request).Req.(*v3lrspb.LoadStatsRequest).ClusterStats
	if l := len(gotLoad); l != 1 {
		t.Fatalf("Received load for %d clusters, want 1", l)
	}

	// This field is set by the client to indicate the actual time elapsed since
	// the last report was sent. We cannot deterministically compare this, and
	// we cannot use the cmpopts.IgnoreFields() option on proto structs, since
	// we already use the protocmp.Transform() which marshals the struct into
	// another message. Hence setting this field to nil is the best option here.
	gotLoad[0].LoadReportInterval = nil
	wantLoad := &v3endpointpb.ClusterStats{
		ClusterName:          "cluster1",
		ClusterServiceName:   "eds1",
		TotalDroppedRequests: 1,
		DroppedRequests:      []*v3endpointpb.ClusterStats_DroppedRequests{{Category: "test", DroppedCount: 1}},
		UpstreamLocalityStats: []*v3endpointpb.UpstreamLocalityStats{
			{
				Locality: &v3corepb.Locality{Region: "test-region1"},
				LoadMetricStats: []*v3endpointpb.EndpointLoadMetricStats{
					// TotalMetricValue is the aggregation of 3.14 + 2.718 = 5.858
					{MetricName: testKey1, NumRequestsFinishedWithMetric: 2, TotalMetricValue: 5.858}},
				TotalSuccessfulRequests: 1,
				TotalIssuedRequests:     1,
			},
			{
				Locality: &v3corepb.Locality{Region: "test-region2"},
				LoadMetricStats: []*v3endpointpb.EndpointLoadMetricStats{
					{MetricName: testKey2, NumRequestsFinishedWithMetric: 1, TotalMetricValue: 1.618}},
				TotalSuccessfulRequests: 1,
				TotalIssuedRequests:     1,
			},
		},
	}
	if diff := cmp.Diff(wantLoad, gotLoad[0], protocmp.Transform(), toleranceCmpOpt, ignoreOrderCmpOpt); diff != "" {
		t.Fatalf("Unexpected diff in LRS request (-got, +want):\n%s", diff)
	}

	// Make another call to the load reporting API, and ensure that a new LRS
	// stream is not created.
	store2, cancel2 := client.ReportLoad(serverConfig)
	sCtx, sCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer sCancel()
	if _, err := lrsServer.LRSStreamOpenChan.Receive(sCtx); err != context.DeadlineExceeded {
		t.Fatal("New LRS stream created when expected to use an existing one")
	}

	// Push more loads.
	store2.PerCluster("cluster2", "eds2").CallDropped("test")

	// Ensure that loads are seen on the server. We need a loop here because
	// there could have been some requests from the client in the time between
	// us reading the first request and now. Those would have been queued in the
	// request channel that we read out of.
	for {
		if ctx.Err() != nil {
			t.Fatalf("Timeout when waiting for new loads to be seen on the server")
		}

		req, err = lrsServer.LRSRequestChan.Receive(ctx)
		if err != nil {
			continue
		}
		gotLoad = req.(*fakeserver.Request).Req.(*v3lrspb.LoadStatsRequest).ClusterStats
		if l := len(gotLoad); l != 1 {
			continue
		}
		gotLoad[0].LoadReportInterval = nil
		wantLoad := &v3endpointpb.ClusterStats{
			ClusterName:          "cluster2",
			ClusterServiceName:   "eds2",
			TotalDroppedRequests: 1,
			DroppedRequests:      []*v3endpointpb.ClusterStats_DroppedRequests{{Category: "test", DroppedCount: 1}},
		}
		if diff := cmp.Diff(wantLoad, gotLoad[0], protocmp.Transform()); diff != "" {
			t.Logf("Unexpected diff in LRS request (-got, +want):\n%s", diff)
			continue
		}
		break
	}

	// Cancel the first load reporting call, and ensure that the stream does not
	// close (because we have another call open).
	cancel1()
	sCtx, sCancel = context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer sCancel()
	if _, err := lrsServer.LRSStreamCloseChan.Receive(sCtx); err != context.DeadlineExceeded {
		t.Fatal("LRS stream closed when expected to stay open")
	}

	// Cancel the second load reporting call, and ensure the stream is closed.
	cancel2()
	if _, err := lrsServer.LRSStreamCloseChan.Receive(ctx); err != nil {
		t.Fatal("Timeout waiting for LRS stream to close")
	}

	// Calling the load reporting API again should result in the creation of a
	// new LRS stream. This ensures that creating and closing multiple streams
	// works smoothly.
	_, cancel3 := client.ReportLoad(serverConfig)
	if _, err := lrsServer.LRSStreamOpenChan.Receive(ctx); err != nil {
		t.Fatalf("Timeout when waiting for LRS stream to be created: %v", err)
	}
	cancel3()
}

func translatePolicy(policyStr string) ([]*v3rbacpb.RBAC, string, error) {
	policy := &authorizationPolicy{}
	d := json.NewDecoder(bytes.NewReader([]byte(policyStr)))
	d.DisallowUnknownFields()
	if err := d.Decode(policy); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal policy: %v", err)
	}
	if policy.Name == "" {
		return nil, "", fmt.Errorf(`"name" is not present`)
	}
	if len(policy.AllowRules) == 0 {
		return nil, "", fmt.Errorf(`"allow_rules" is not present`)
	}
	allowLogger, denyLogger, err := policy.AuditLoggingOptions.toProtos()
	if err != nil {
		return nil, "", err
	}
	rbacs := make([]*v3rbacpb.RBAC, 0, 2)
	if len(policy.DenyRules) > 0 {
		denyPolicies, err := parseRules(policy.DenyRules, policy.Name)
		if err != nil {
			return nil, "", fmt.Errorf(`"deny_rules" %v`, err)
		}
		denyRBAC := &v3rbacpb.RBAC{
			Action:              v3rbacpb.RBAC_DENY,
			Policies:            denyPolicies,
			AuditLoggingOptions: denyLogger,
		}
		rbacs = append(rbacs, denyRBAC)
	}
	allowPolicies, err := parseRules(policy.AllowRules, policy.Name)
	if err != nil {
		return nil, "", fmt.Errorf(`"allow_rules" %v`, err)
	}
	allowRBAC := &v3rbacpb.RBAC{Action: v3rbacpb.RBAC_ALLOW, Policies: allowPolicies, AuditLoggingOptions: allowLogger}
	return append(rbacs, allowRBAC), policy.Name, nil
}

