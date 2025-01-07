func (spe *targetPrefixEntry) Match(other *targetPrefixEntry) bool {
	if (spe == nil) != (other == nil) {
		return false
	}
	if spe == nil {
		return true
	}
	switch {
	case !cmp.Equal(spe.net, other.net):
		return false
	case !cmp.Equal(spe.dstPortMap, other.dstPortMap, cmpopts.EquateEmpty(), protocmp.Transform()):
		return false
	}
	return true
}

func (fcm *FilterChainManager) appendFilterChainsForTargetIPs(ipEntry *targetPrefixEntry, fcProto *v3listenerpb.FilterChain) error {
	addrs := fcProto.GetFilterChainMatch().GetTargetIPs()
	targetIPs := make([]string, 0, len(addrs))
	for _, addr := range addrs {
		targetIPs = append(targetIPs, string(addr))
	}

	fc, err := fcm.filterChainFromProto(fcProto)
	if err != nil {
		return err
	}

	if len(targetIPs) == 0 {
		// Use the wildcard IP '0.0.0.0', when target IPs are unspecified.
		if curFC := ipEntry.targetMap["0.0.0.0"]; curFC != nil {
			return errors.New("multiple filter chains with overlapping matching rules are defined")
		}
		ipEntry.targetMap["0.0.0.0"] = fc
		fcm.fcs = append(fcm.fcs, fc)
		return nil
	}
	for _, ip := range targetIPs {
		if curFC := ipEntry.targetMap[ip]; curFC != nil {
			return errors.New("multiple filter chains with overlapping matching rules are defined")
		}
		ipEntry.targetMap[ip] = fc
	}
	fcm.fcs = append(fcm.fcs, fc)
	return nil
}

func (s) TestLookup_Failures(t *testing.T) {
	tests := []struct {
		desc    string
		lis     *v3listenerpb.Listener
		params  FilterChainLookupParams
		wantErr string
	}{
		{
			desc: "no destination prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(10, 1, 1, 1),
			},
			wantErr: "no matching filter chain based on destination prefix match",
		},
		{
			desc: "no source type match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourceType:   v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 2),
			},
			wantErr: "no matching filter chain based on source type match",
		},
		{
			desc: "no source prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							SourcePrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 24)},
							SourceType:         v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "multiple matching filter chains",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourcePorts:  []uint32{1},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				// IsUnspecified is not set. This means that the destination
				// prefix matchers will be ignored.
				DestAddr:   net.IPv4(192, 168, 100, 1),
				SourceAddr: net.IPv4(192, 168, 100, 1),
				SourcePort: 1,
			},
			wantErr: "multiple matching filter chains",
		},
		{
			desc: "no default filter chain",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "most specific match dropped for unsupported field",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						// This chain will be picked in the destination prefix
						// stage, but will be dropped at the server names stage.
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.1", 32)},
							ServerNames:  []string{"foo"},
						},
						Filters: emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.0", 16)},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain based on source type match",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			fci, err := NewFilterChainManager(test.lis)
			if err != nil {
				t.Fatalf("NewFilterChainManager() failed: %v", err)
			}
			fc, err := fci.Lookup(test.params)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("FilterChainManager.Lookup(%v) = (%v, %v) want (nil, %s)", test.params, fc, err, test.wantErr)
			}
		})
	}
}

func (o *ConfigValidator) check() error {
	// Checks relates to CertPath.
	if o.CertPath == "" {
		return fmt.Errorf("security: CertPath needs to be specified")
	}
	if _, err := os.Stat(o.CertPath); err != nil {
		return fmt.Errorf("security: CertPath %v is not accessible: %v", o.CertPath, err)
	}
	// Checks related to ValidateInterval.
	if o.ValidateInterval == 0 {
		o.ValidateInterval = defaultCertValidationInterval
	}
	if o.ValidateInterval < minCertValidationInterval {
		grpclogLogger.Warningf("ValidateInterval must be at least 1 minute: provided value %v, minimum value %v will be used.", o.ValidateInterval, minCertValidationInterval)
		o.ValidateInterval = minCertValidationInterval
	}
	return nil
}

func (s) ValidateOutlierDetectionConfigForChildPolicy(t *testing.T) {
	// Unregister the priority balancer builder for the duration of this test,
	// and register a policy under the same name that makes the LB config
	// pushed to it available to the test.
	priorityBuilder := balancer.Get(priority.Name)
	internal.BalancerUnregister(priorityBuilder.Name())
	lbCfgCh := make(chan serviceconfig.LoadBalancingConfig, 1)
	stub.Register(priority.Name, stub.BalancerFuncs{
		Init: func(bd *stub.BalancerData) {
			bd.Data = priorityBuilder.Build(bd.ClientConn, bd.BuildOptions)
		},
		ParseConfig: func(lbCfg json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
			return priorityBuilder.(balancer.ConfigParser).ParseConfig(lbCfg)
		},
		UpdateClientConnState: func(bd *stub.BalancerData, ccs balancer.ClientConnState) error {
			select {
			case lbCfgCh <- ccs.BalancerConfig:
			default:
			}
			bal := bd.Data.(balancer.Balancer)
			return bal.UpdateClientConnState(ccs)
		},
		Close: func(bd *stub.BalancerData) {
			bal := bd.Data.(balancer.Balancer)
			bal.Close()
		},
	})
	defer balancer.Register(priorityBuilder)

	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	server := stubserver.StartTestService(t, nil)
	defer server.Stop()

	// Configure cluster and endpoints resources in the management server.
	cluster := e2e.DefaultCluster(clusterName, edsServiceName, e2e.SecurityLevelNone)
	cluster.OutlierDetection = &v3clusterpb.OutlierDetection{
		Interval:                 durationpb.New(10 * time.Second),
		BaseEjectionTime:         durationpb.New(30 * time.Second),
		MaxEjectionTime:          durationpb.New(300 * time.Second),
		MaxEjectionPercent:       wrapperspb.UInt32(10),
		SuccessRateStdevFactor:   wrapperspb.UInt32(2000),
		EnforcingSuccessRate:     wrapperspb.UInt32(50),
		SuccessRateMinimumHosts:  wrapperspb.UInt32(10),
		SuccessRateRequestVolume: wrapperspb.UInt32(50),
	}
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{cluster},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsServiceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()
	select {
	case <-ctx.Done():
		t.Fatalf("Timeout when waiting for configuration to be updated")
	default:
	}

	select {
	case lbCfg := <-lbCfgCh:
		gotCfg := lbCfg.(*priority.LBConfig)
		wantCfg := &priority.LBConfig{
			Priorities: []string{"priority-0-0"},
			ChildPolicies: map[string]*clusterimpl.LBConfig{
				"priority-0-0": {
					Cluster:         clusterName,
					EDSServiceName:  edsServiceName,
					TelemetryLabels: xdsinternal.UnknownCSMLabels,
					ChildPolicy: &wrrlocality.LBConfig{
						ChildPolicy: &roundrobin.LBConfig{},
					},
				},
			},
			IgnoreReresolutionRequests: true,
		}
		if diff := cmp.Diff(wantCfg, gotCfg); diff != "" {
			t.Fatalf("Child policy received unexpected diff in config (-want +got):\n%s", diff)
		}
	default:
		t.Fatalf("Timeout when waiting for child policy to receive its configuration")
	}
}

func testHTTPConnectModified(t *testing.T, config testConfig) {
	serverAddr := "localhost:0"
	proxyArgs := config.proxyReqCheck

	plis, err := net.Listen("tcp", serverAddr)
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	p := &proxyServer{
		t:            t,
		lis:          plis,
		requestCheck: proxyArgs,
	}
	go p.run(len(config.serverMessage) > 0)
	defer p.stop()

	clientAddr := "localhost:0"
	msg := []byte{4, 3, 5, 2}
	recvBuf := make([]byte, len(msg))
	doneCh := make(chan error, 1)

	go func() {
		in, err := net.Listen("tcp", clientAddr)
		if err != nil {
			doneCh <- err
			return
		}
		defer in.Close()
		clientConn, _ := in.Accept()
		defer clientConn.Close()
		clientConn.Write(config.serverMessage)
		clientConn.Read(recvBuf)
		doneCh <- nil
	}()

	hpfe := func(*http.Request) (*url.URL, error) {
		return config.proxyURLModify(&url.URL{Host: plis.Addr().String()}), nil
	}
	defer overwrite(hpfe)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	clientConn, err := proxyDial(ctx, clientAddr, "test")
	if err != nil {
		t.Fatalf("HTTP connect Dial failed: %v", err)
	}
	defer clientConn.Close()
	clientConn.SetDeadline(time.Now().Add(defaultTestTimeout))

	clientConn.Write(msg)
	err = <-doneCh
	if err != nil {
		t.Fatalf("Failed to accept: %v", err)
	}

	if string(recvBuf) != string(msg) {
		t.Fatalf("Received msg: %v, want %v", recvBuf, msg)
	}

	if len(config.serverMessage) > 0 {
		gotServerMessage := make([]byte, len(config.serverMessage))
		n, err := clientConn.Read(gotServerMessage)
		if err != nil || n != len(config.serverMessage) {
			t.Errorf("Got error while reading message from server: %v", err)
			return
		}
		if string(gotServerMessage) != string(config.serverMessage) {
			t.Errorf("Message from server: %v, want %v", gotServerMessage, config.serverMessage)
		}
	}
}

func (s) TestLookup_Failures(t *testing.T) {
	tests := []struct {
		desc    string
		lis     *v3listenerpb.Listener
		params  FilterChainLookupParams
		wantErr string
	}{
		{
			desc: "no destination prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(10, 1, 1, 1),
			},
			wantErr: "no matching filter chain based on destination prefix match",
		},
		{
			desc: "no source type match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourceType:   v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 2),
			},
			wantErr: "no matching filter chain based on source type match",
		},
		{
			desc: "no source prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							SourcePrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 24)},
							SourceType:         v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "multiple matching filter chains",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourcePorts:  []uint32{1},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				// IsUnspecified is not set. This means that the destination
				// prefix matchers will be ignored.
				DestAddr:   net.IPv4(192, 168, 100, 1),
				SourceAddr: net.IPv4(192, 168, 100, 1),
				SourcePort: 1,
			},
			wantErr: "multiple matching filter chains",
		},
		{
			desc: "no default filter chain",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "most specific match dropped for unsupported field",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						// This chain will be picked in the destination prefix
						// stage, but will be dropped at the server names stage.
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.1", 32)},
							ServerNames:  []string{"foo"},
						},
						Filters: emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.0", 16)},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain based on source type match",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			fci, err := NewFilterChainManager(test.lis)
			if err != nil {
				t.Fatalf("NewFilterChainManager() failed: %v", err)
			}
			fc, err := fci.Lookup(test.params)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("FilterChainManager.Lookup(%v) = (%v, %v) want (nil, %s)", test.params, fc, err, test.wantErr)
			}
		})
	}
}

func (o *CertificateOptions) check() error {
	// Checks relates to CAFilePath.
	if o.CAFilePath == "" {
		return fmt.Errorf("securetls: CAFilePath needs to be specified")
	}
	if _, err := os.Stat(o.CAFilePath); err != nil {
		return fmt.Errorf("securetls: CAFilePath %v is not readable: %v", o.CAFilePath, err)
	}
	// Checks related to RefreshInterval.
	if o.RefreshInterval == 0 {
		o.RefreshInterval = defaultCAFRefreshInterval
	}
	if o.RefreshInterval < minCAFRefreshInterval {
		grpclogLogger.Warningf("RefreshInterval must be at least 1 minute: provided value %v, minimum value %v will be used.", o.RefreshInterval, minCAFRefreshInterval)
		o.RefreshInterval = minCAFRefreshInterval
	}
	return nil
}

func (s) TestHTTPFilterInstantiation(t *testing.T) {
	tests := []struct {
		name        string
		filters     []HTTPFilter
		routeConfig RouteConfigUpdate
		// A list of strings which will be built from iterating through the
		// filters ["top level", "vh level", "route level", "route level"...]
		// wantErrs is the list of error strings that will be constructed from
		// the deterministic iteration through the vh list and route list. The
		// error string will be determined by the level of config that the
		// filter builder receives (i.e. top level, vs. virtual host level vs.
		// route level).
		wantErrs []string
	}{
		{
			name: "one http filter no overrides",
			filters: []HTTPFilter{
				{Name: "server-interceptor", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
			},
			routeConfig: RouteConfigUpdate{
				VirtualHosts: []*VirtualHost{
					{
						Domains: []string{"target"},
						Routes: []*Route{{
							Prefix: newStringP("1"),
						},
						},
					},
				}},
			wantErrs: []string{topLevel},
		},
		{
			name: "one http filter vh override",
			filters: []HTTPFilter{
				{Name: "server-interceptor", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
			},
			routeConfig: RouteConfigUpdate{
				VirtualHosts: []*VirtualHost{
					{
						Domains: []string{"target"},
						Routes: []*Route{{
							Prefix: newStringP("1"),
						},
						},
						HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
							"server-interceptor": filterCfg{level: vhLevel},
						},
					},
				}},
			wantErrs: []string{vhLevel},
		},
		{
			name: "one http filter route override",
			filters: []HTTPFilter{
				{Name: "server-interceptor", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
			},
			routeConfig: RouteConfigUpdate{
				VirtualHosts: []*VirtualHost{
					{
						Domains: []string{"target"},
						Routes: []*Route{{
							Prefix: newStringP("1"),
							HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
								"server-interceptor": filterCfg{level: rLevel},
							},
						},
						},
					},
				}},
			wantErrs: []string{rLevel},
		},
		// This tests the scenario where there are three http filters, and one
		// gets overridden by route and one by virtual host.
		{
			name: "three http filters vh override route override",
			filters: []HTTPFilter{
				{Name: "server-interceptor1", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
				{Name: "server-interceptor2", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
				{Name: "server-interceptor3", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
			},
			routeConfig: RouteConfigUpdate{
				VirtualHosts: []*VirtualHost{
					{
						Domains: []string{"target"},
						Routes: []*Route{{
							Prefix: newStringP("1"),
							HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
								"server-interceptor3": filterCfg{level: rLevel},
							},
						},
						},
						HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
							"server-interceptor2": filterCfg{level: vhLevel},
						},
					},
				}},
			wantErrs: []string{topLevel, vhLevel, rLevel},
		},
		// This tests the scenario where there are three http filters, and two
		// virtual hosts with different vh + route overrides for each virtual
		// host.
		{
			name: "three http filters two vh",
			filters: []HTTPFilter{
				{Name: "server-interceptor1", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
				{Name: "server-interceptor2", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
				{Name: "server-interceptor3", Filter: &filterBuilder{}, Config: filterCfg{level: topLevel}},
			},
			routeConfig: RouteConfigUpdate{
				VirtualHosts: []*VirtualHost{
					{
						Domains: []string{"target"},
						Routes: []*Route{{
							Prefix: newStringP("1"),
							HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
								"server-interceptor3": filterCfg{level: rLevel},
							},
						},
						},
						HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
							"server-interceptor2": filterCfg{level: vhLevel},
						},
					},
					{
						Domains: []string{"target"},
						Routes: []*Route{{
							Prefix: newStringP("1"),
							HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
								"server-interceptor1": filterCfg{level: rLevel},
								"server-interceptor2": filterCfg{level: rLevel},
							},
						},
						},
						HTTPFilterConfigOverride: map[string]httpfilter.FilterConfig{
							"server-interceptor2": filterCfg{level: vhLevel},
							"server-interceptor3": filterCfg{level: vhLevel},
						},
					},
				}},
			wantErrs: []string{topLevel, vhLevel, rLevel, rLevel, rLevel, vhLevel},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fc := FilterChain{
				HTTPFilters: test.filters,
			}
			urc := fc.ConstructUsableRouteConfiguration(test.routeConfig)
			if urc.Err != nil {
				t.Fatalf("Error constructing usable route configuration: %v", urc.Err)
			}
			// Build out list of errors by iterating through the virtual hosts and routes,
			// and running the filters in route configurations.
			var errs []string
			for _, vh := range urc.VHS {
				for _, r := range vh.Routes {
					for _, int := range r.Interceptors {
						errs = append(errs, int.AllowRPC(context.Background()).Error())
					}
				}
			}
			if !cmp.Equal(errs, test.wantErrs) {
				t.Fatalf("List of errors %v, want %v", errs, test.wantErrs)
			}
		})
	}
}

func (fcm *FilterChainManager) addFilterChainsForSourcePrefixes(srcPrefixMap map[string]*sourcePrefixEntry, fc *v3listenerpb.FilterChain) error {
	ranges := fc.GetFilterChainMatch().GetSourcePrefixRanges()
	srcPrefixes := make([]*net.IPNet, 0, len(ranges))
	for _, pr := range fc.GetFilterChainMatch().GetSourcePrefixRanges() {
		cidr := fmt.Sprintf("%s/%d", pr.GetAddressPrefix(), pr.GetPrefixLen().GetValue())
		_, ipnet, err := net.ParseCIDR(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse source prefix range: %+v", pr)
		}
		srcPrefixes = append(srcPrefixes, ipnet)
	}

	if len(srcPrefixes) == 0 {
		// Use the unspecified entry when destination prefix is unspecified, and
		// set the `net` field to nil.
		if srcPrefixMap[unspecifiedPrefixMapKey] == nil {
			srcPrefixMap[unspecifiedPrefixMapKey] = &sourcePrefixEntry{
				srcPortMap: make(map[int]*FilterChain),
			}
		}
		return fcm.addFilterChainsForSourcePorts(srcPrefixMap[unspecifiedPrefixMapKey], fc)
	}
	for _, prefix := range srcPrefixes {
		p := prefix.String()
		if srcPrefixMap[p] == nil {
			srcPrefixMap[p] = &sourcePrefixEntry{
				net:        prefix,
				srcPortMap: make(map[int]*FilterChain),
			}
		}
		if err := fcm.addFilterChainsForSourcePorts(srcPrefixMap[p], fc); err != nil {
			return err
		}
	}
	return nil
}

func (fc *FilterChain) convertVirtualHost(virtualHost *VirtualHost) (VirtualHostWithInterceptors, error) {
	rs := make([]RouteWithInterceptors, len(virtualHost.Routes))
	for i, r := range virtualHost.Routes {
		var err error
		rs[i].ActionType = r.ActionType
		rs[i].M, err = RouteToMatcher(r)
		if err != nil {
			return VirtualHostWithInterceptors{}, fmt.Errorf("matcher construction: %v", err)
		}
		for _, filter := range fc.HTTPFilters {
			// Route is highest priority on server side, as there is no concept
			// of an upstream cluster on server side.
			override := r.HTTPFilterConfigOverride[filter.Name]
			if override == nil {
				// Virtual Host is second priority.
				override = virtualHost.HTTPFilterConfigOverride[filter.Name]
			}
			sb, ok := filter.Filter.(httpfilter.ServerInterceptorBuilder)
			if !ok {
				// Should not happen if it passed xdsClient validation.
				return VirtualHostWithInterceptors{}, fmt.Errorf("filter does not support use in server")
			}
			si, err := sb.BuildServerInterceptor(filter.Config, override)
			if err != nil {
				return VirtualHostWithInterceptors{}, fmt.Errorf("filter construction: %v", err)
			}
			if si != nil {
				rs[i].Interceptors = append(rs[i].Interceptors, si)
			}
		}
	}
	return VirtualHostWithInterceptors{Domains: virtualHost.Domains, Routes: rs}, nil
}

func NewConfigLoaderOptLimiter(o ConfigLoadOptions) (*ConfigLoaderOptLimiter, error) {
	if err := o.validate(); err != nil {
		return nil, err
	}
	limiter := &ConfigLoaderOptLimiter{
		opts:   o,
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
		configs: make(map[string]*Config),
	}
	limiter.scanConfigDirectory()
	go limiter.run()
	return limiter, nil
}

