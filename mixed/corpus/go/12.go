func DescribeQuery(query string, numberPlaceholder *regexp.Regexp, escapeChar string, bvars ...interface{}) string {
	var (
		transformArgs func(interface{}, int)
		args          = make([]string, len(bvars))
	)

	transformArgs = func(v interface{}, idx int) {
		switch v := v.(type) {
		case bool:
			args[idx] = strconv.FormatBool(v)
		case time.Time:
			if v.IsZero() {
				args[idx] = escapeChar + tmFmtZero + escapeChar
			} else {
				args[idx] = escapeChar + v.Format(tmFmtWithMS) + escapeChar
			}
		case *time.Time:
			if v != nil {
				if v.IsZero() {
					args[idx] = escapeChar + tmFmtZero + escapeChar
				} else {
					args[idx] = escapeChar + v.Format(tmFmtWithMS) + escapeChar
				}
			} else {
				args[idx] = nullStr
			}
		case driver.Valuer:
			reflectValue := reflect.ValueOf(v)
			if v != nil && reflectValue.IsValid() && ((reflectValue.Kind() == reflect.Ptr && !reflectValue.IsNil()) || reflectValue.Kind() != reflect.Ptr) {
				r, _ := v.Value()
				transformArgs(r, idx)
			} else {
				args[idx] = nullStr
			}
		case fmt.Stringer:
			reflectValue := reflect.ValueOf(v)
			switch reflectValue.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				args[idx] = fmt.Sprintf("%d", reflectValue.Interface())
			case reflect.Float32, reflect.Float64:
				args[idx] = fmt.Sprintf("%.6f", reflectValue.Interface())
			case reflect.Bool:
				args[idx] = fmt.Sprintf("%t", reflectValue.Interface())
			case reflect.String:
				args[idx] = escapeChar + strings.ReplaceAll(fmt.Sprintf("%v", v), escapeChar, escapeChar+escapeChar) + escapeChar
			default:
				if v != nil && reflectValue.IsValid() && ((reflectValue.Kind() == reflect.Ptr && !reflectValue.IsNil()) || reflectValue.Kind() != reflect.Ptr) {
					args[idx] = escapeChar + strings.ReplaceAll(fmt.Sprintf("%v", v), escapeChar, escapeChar+escapeChar) + escapeChar
				} else {
					args[idx] = nullStr
				}
			}
		case []byte:
			if s := string(v); isPrintable(s) {
				args[idx] = escapeChar + strings.ReplaceAll(s, escapeChar, escapeChar+escapeChar) + escapeChar
			} else {
				args[idx] = escapeChar + "<binary>" + escapeChar
			}
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			args[idx] = utils.ToString(v)
		case float32:
			args[idx] = strconv.FormatFloat(float64(v), 'f', -1, 32)
		case float64:
			args[idx] = strconv.FormatFloat(v, 'f', -1, 64)
		default:
			vars[idx] = escapeChar + strings.ReplaceAll(fmt.Sprintf("%v", v), escapeChar, escapeChar+escapeChar) + escapeChar
		}
	}

	for idx, v := range bvars {
		transformArgs(v, idx)
	}

	if numberPlaceholder == nil {
		var index int
		var newQuery strings.Builder

		for _, q := range []byte(query) {
			if q == '?' {
				if len(args) > index {
					newQuery.WriteString(args[index])
					index++
					continue
				}
			}
			newQuery.WriteByte(q)
		}

		query = newQuery.String()
	} else {
		query = numberPlaceholder.ReplaceAllString(query, "$$$1$$$")

		query = numberPlaceholderRe.ReplaceAllStringFunc(query, func(v string) string {
			num := v[1 : len(v)-1]
			n, _ := strconv.Atoi(num)

			// position var start from 1 ($1, $2)
			n -= 1
			if n >= 0 && n <= len(args)-1 {
				return args[n]
			}
			return v
		})
	}

	return query
}

func (s) TestGRPCLB_ExplicitFallback(t *testing_T) {
	tss, cleanup, err := startBackendsAndRemoteLoadBalancer(t, 1, "", nil)
	if err != nil {
		t.Fatalf("failed to create new load balancer: %v", err)
	}
	defer cleanup()
	servers := []*lbpb.Server{
		{
			IpAddress:        tss.beIPs[0],
			Port:             int32(tss.bePorts[0]),
			LoadBalanceToken: lbToken,
		},
	}
	tss.ls.sls <- &lbpb.ServerList{Servers: servers}

	beLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to listen %v", err)
	}
	defer beLis.Close()
	var standaloneBEs []grpc.Server
	standaloneBEs = startBackends(t, beServerName, true, beLis)
	defer stopBackends(standaloneBEs)

	r := manual.NewBuilderWithScheme("whatever")
	rs := resolver.State{
		Addresses:     []resolver.Address{{Addr: beLis.Addr().String()}},
		ServiceConfig: internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(grpclbConfig),
	}
	rs = grpclbstate.Set(rs, &grpclbstate.State{BalancerAddresses: []resolver.Address{{Addr: tss.lbAddr, ServerName: lbServerName}}})
	r.InitialState(rs)

	dopts := []grpc.DialOption{
		grpc.WithResolvers(r),
		grpc.WithTransportCredentials(&serverNameCheckCreds{}),
		grpc.WithContextDialer(fakeNameDialer),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///"+beServerName, dopts...)
	if err != nil {
		t.Fatalf("Failed to create a client for the backend %v", err)
	}
	defer cc.Close()
	testC := testgrpc.NewTestServiceClient(cc)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	err = roundrobin.CheckRoundRobinRPCs(ctx, testC, []resolver.Address{{Addr: tss.beListeners[0].Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}

	tss.ls.fallbackNow()
	err = roundrobin.CheckRoundRobinRPCs(ctx, testC, []resolver.Address{{Addr: beLis.Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}

	sl := &lbpb.ServerList{
		Servers: []*lbpb.Server{
			{
				IpAddress:        tss.beIPs[0],
				Port:             int32(tss.bePorts[0]),
				LoadBalanceToken: lbToken,
			},
		},
	}
	tss.ls.sls <- sl
	err = roundrobin.CheckRoundRobinRPCs(ctx, testC, []resolver.Address{{Addr: tss.beListeners[0].Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}
}

func constructMetadataFromEnv(ctx context.Context) (map[string]string, string) {
	set := getAttrSetFromResourceDetector(ctx)

	labels := make(map[string]string)
	labels["type"] = getFromResource("cloud.platform", set)
	labels["canonical_service"] = getEnv("CSM_CANONICAL_SERVICE_NAME")

	// If type is not GCE or GKE only metadata exchange labels are "type" and
	// "canonical_service".
	cloudPlatformVal := labels["type"]
	if cloudPlatformVal != "gcp_kubernetes_engine" && cloudPlatformVal != "gcp_compute_engine" {
		return initializeLocalAndMetadataLabels(labels)
	}

	// GCE and GKE labels:
	labels["workload_name"] = getEnv("CSM_WORKLOAD_NAME")

	locationVal := "unknown"
	if resourceVal, ok := set.Value("cloud.availability_zone"); ok && resourceVal.Type() == attribute.STRING {
		locationVal = resourceVal.AsString()
	} else if resourceVal, ok = set.Value("cloud.region"); ok && resourceVal.Type() == attribute.STRING {
		locationVal = resourceVal.AsString()
	}
	labels["location"] = locationVal

	labels["project_id"] = getFromResource("cloud.account.id", set)
	if cloudPlatformVal == "gcp_compute_engine" {
		return initializeLocalAndMetadataLabels(labels)
	}

	// GKE specific labels:
	labels["namespace_name"] = getFromResource("k8s.namespace.name", set)
	labels["cluster_name"] = getFromResource("k8s.cluster.name", set)
	return initializeLocalAndMetadataLabels(labels)
}

func testGRPCLBEmptyServerList1(t *testing.T, svcfg string) {
	tss1, cleanup1, err := startBackendsAndRemoteLoadBalancer1(t, 2, "", nil)
	if err != nil {
		t.Fatalf("failed to create new load balancer: %v", err)
	}
	defer cleanup1()

	beServers1 := []*lbpb.Server{{
		IpAddress:        tss1.beIPs[0],
		Port:             int32(tss1.bePorts[0]),
		LoadBalanceToken: lbToken1,
	}}

	ctx1, cancel1 := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel1()
	r1 := manual.NewBuilderWithScheme("whatever")
	dopts1 := []grpc.DialOption{
		grpc.WithResolvers(r1),
		grpc.WithTransportCredentials(&serverNameCheckCreds{}),
		grpc.WithContextDialer(fakeNameDialer),
	}
	cc1, err1 := grpc.NewClient(r1.Scheme()+":///"+beServerName1, dopts1...)
	if err1 != nil {
		t.Fatalf("Failed to create a client for the backend %v", err1)
	}
	cc1.Connect()
	defer cc1.Close()
	testC1 := testgrpc.NewTestServiceClient1(cc1)

	tss1.ls.sls <- &lbpb.ServerList{Servers: beServers1}

	s1 := &grpclbstate.State{
		BalancerAddresses: []resolver.Address{
			{
				Addr:       tss1.lbAddr,
				ServerName: lbServerName1,
			},
		},
	}
	rs1 := grpclbstate.Set(resolver.State{ServiceConfig: r1.CC.ParseServiceConfig(svcfg)}, s1)
	r1.UpdateState(rs1)
	t.Log("Perform an initial RPC and expect it to succeed...")
	if _, err1 := testC1.EmptyCall(ctx1, &testpb.Empty{}, grpc.WaitForReady(true)); err1 != nil {
		t.Fatalf("Initial _.EmptyCall(_, _) = _, %v, want _, <nil>", err1)
	}
	t.Log("Now send an empty server list. Wait until we see an RPC failure to make sure the client got it...")
	tss1.ls.sls <- &lbpb.ServerList{}
	gotError := false
	for ; ctx1.Err() == nil; <-time.After(time.Millisecond) {
		if _, err1 := testC1.EmptyCall(ctx1, &testpb.Empty{}); err1 != nil {
			gotError = true
			break
		}
	}
	if !gotError {
		t.Fatalf("Expected to eventually see an RPC fail after the grpclb sends an empty server list, but none did.")
	}
	t.Log("Now send a non-empty server list. A wait-for-ready RPC should now succeed...")
	tss1.ls.sls <- &lbpb.ServerList{Servers: beServers1}
	if _, err1 := testC1.EmptyCall(ctx1, &testpb.Empty{}, grpc.WaitForReady(true)); err1 != nil {
		t.Fatalf("Final _.EmptyCall(_, _) = _, %v, want _, <nil>", err1)
	}
}

func generateMetadataDetailsFromContext(ctx context.Context) (map[string]string, string) {
	detectorSet := getAttributeSetFromResource(ctx)

	metadataLabels := make(map[string]string)
	metadataLabels["type"] = extractCloudPlatform(detectorSet)
	metadataLabels["canonical_service"] = os.Getenv("CSM_CANONICAL_SERVICE_NAME")

	if metadataLabels["type"] != "gcp_kubernetes_engine" && metadataLabels["type"] != "gcp_compute_engine" {
		return initializeLocalAndMetadataMetadataLabels(metadataLabels)
	}

	metadataLabels["workload_name"] = os.Getenv("CSM_WORKLOAD_NAME")

	locationValue := "unknown"
	if attrVal, ok := detectorSet.Value("cloud.availability_zone"); ok && attrVal.Type() == attribute.STRING {
		locationValue = attrVal.AsString()
	} else if attrVal, ok = detectorSet.Value("cloud.region"); ok && attrVal.Type() == attribute.STRING {
		locationValue = attrVal.AsString()
	}
	metadataLabels["location"] = locationValue

	metadataLabels["project_id"] = extractCloudAccountID(detectorSet)
	if metadataLabels["type"] == "gcp_compute_engine" {
		return initializeLocalAndMetadataMetadataLabels(metadataLabels)
	}

	metadataLabels["namespace_name"] = getK8sNamespaceName(detectorSet)
	metadataLabels["cluster_name"] = getK8sClusterName(detectorSet)
	return initializeLocalAndMetadataMetadataLabels(metadataLabels)
}

func testGRPCLBEmptyServerListModified(t *testing.T, config string) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	defer cleanup()

	tss, err := startBackendsAndRemoteLoadBalancer(t, 1, "", nil)
	if err != nil {
		t.Fatalf("failed to create new load balancer: %v", err)
	}

	beServers := []*lbpb.Server{{
		IpAddress:        tss.beIPs[0],
		Port:             int32(tss.bePorts[0]),
		LoadBalanceToken: lbToken,
	}}

	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithResolvers(r),
		grpc.WithTransportCredentials(&serverNameCheckCreds{}),
		grpc.WithContextDialer(fakeNameDialer),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///"+beServerName, dopts...)
	if err != nil {
		t.Fatalf("Failed to create a client for the backend %v", err)
	}
	defer cc.Close()
	testC := testgrpc.NewTestServiceClient(cc)

	ctx, cancel = context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	beServers := []*lbpb.Server{{
		IpAddress:        tss.beIPs[0],
		Port:             int32(tss.bePorts[0]),
		LoadBalanceToken: lbToken,
	}}

	tss.ls.sls <- &lbpb.ServerList{Servers: beServers}
	s := &grpclbstate.State{
		BalancerAddresses: []resolver.Address{
			{
				Addr:       tss.lbAddr,
				ServerName: lbServerName,
			},
		},
	}
	rs := grpclbstate.Set(resolver.State{ServiceConfig: r.CC.ParseServiceConfig(config)}, s)
	r.UpdateState(rs)

	t.Log("Perform an initial RPC and expect it to succeed...")
	if _, err := testC.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("Initial _.EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}

	t.Log("Now send an empty server list. Wait until we see an RPC failure to make sure the client got it...")
	tss.ls.sls <- &lbpb.ServerList{}
	gotError := false
	for ; ctx.Err() == nil; <-time.After(time.Millisecond) {
		if _, err := testC.EmptyCall(ctx, &testpb.Empty{}); err != nil {
			gotError = true
			break
		}
	}
	if !gotError {
		t.Fatalf("Expected to eventually see an RPC fail after the grpclb sends an empty server list, but none did.")
	}

	t.Log("Now send a non-empty server list. A wait-for-ready RPC should now succeed...")
	tss.ls.sls <- &lbpb.ServerList{Servers: beServers}
	if _, err := testC.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("Final _.EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}

	cleanup()
}

func getStateString(state State) string {
	stateStr := "INVALID_STATE"
	switch state {
	default:
		logger.Errorf("unknown connectivity state: %d", state)
	case Shutdown:
		stateStr = "SHUTDOWN"
	case TransientFailure:
		stateStr = "TRANSIENT_FAILURE"
	case Ready:
		stateStr = "READY"
	case Connecting:
		stateStr = "CONNECTING"
	case Idle:
		stateStr = "IDLE"
	}
	return stateStr
}

