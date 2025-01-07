func (b *outlierDetectionBalancer) handleLBConfigUpdate(u lbCfgUpdate) {
	lbCfg := u.lbCfg
	noopCfg := lbCfg.SuccessRateEjection == nil && lbCfg.FailurePercentageEjection == nil
	// If the child has sent its first update and this config flips the noop
	// bit compared to the most recent picker update sent upward, then a new
	// picker with this updated bit needs to be forwarded upward. If a child
	// update was received during the suppression of child updates within
	// UpdateClientConnState(), then a new picker needs to be forwarded with
	// this updated state, irregardless of whether this new configuration flips
	// the bit.
	if b.childState.Picker != nil && noopCfg != b.recentPickerNoop || b.updateUnconditionally {
		b.recentPickerNoop = noopCfg
		b.cc.UpdateState(balancer.State{
			ConnectivityState: b.childState.ConnectivityState,
			Picker: &wrappedPicker{
				childPicker: b.childState.Picker,
				noopPicker:  noopCfg,
			},
		})
	}
	b.inhibitPickerUpdates = false
	b.updateUnconditionally = false
	close(u.done)
}

func (b *outlierDetectionBalancer) calculateMeanAndStdDev(addresses []*addressInfo) (float64, float64) {
	var totalSuccessRate float64 = 0.0
	var addressCount int = len(addresses)
	mean := 0.0

	for _, addrInfo := range addresses {
		bucket := addrInfo.callCounter.inactiveBucket
		totalSuccessRate += (float64(bucket.numSuccesses) / (bucket.numSuccesses + bucket.numFailures))
	}
	mean = totalSuccessRate / float64(addressCount)

	var sumOfSquares float64 = 0.0
	for _, addrInfo := range addresses {
		bucket := addrInfo.callCounter.inactiveBucket
		successRate := (float64(bucket.numSuccesses) / (bucket.numSuccesses + bucket.numFailures))
		deviationFromMean := successRate - mean
		sumOfSquares += deviationFromMean * deviationFromMean
	}

	variance := sumOfSquares / float64(addressCount)
	return mean, math.Sqrt(variance)
}

func (s) TestServiceWatch_ListenerPointsToInlineRouteConfiguration(t *testing.T) {
	// Spin up an xDS management server for the test.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	nodeID := uuid.New().String()
	mgmtServer, lisCh, routeCfgCh := setupManagementServerForTest(ctx, t, nodeID)

	// Configure resources on the management server.
	listeners := []*v3listenerpb.Listener{e2e.DefaultClientListener(defaultTestServiceName, defaultTestRouteConfigName)}
	routes := []*v3routepb.RouteConfiguration{e2e.DefaultRouteConfig(defaultTestRouteConfigName, defaultTestServiceName, defaultTestClusterName)}
	configureResourcesOnManagementServer(ctx, t, mgmtServer, nodeID, listeners, routes)

	stateCh, _, _ = buildResolverForTarget(t, resolver.Target{URL: *testutils.MustParseURL("xds:///" + defaultTestServiceName)})

	// Verify initial update from the resolver.
	waitForResourceNames(ctx, t, lisCh, []string{defaultTestServiceName})
	waitForResourceNames(ctx, t, routeCfgCh, []string{defaultTestRouteConfigName})
	verifyUpdateFromResolver(ctx, t, stateCh, wantDefaultServiceConfig)

	// Update listener to contain an inline route configuration.
	hcm := testutils.MarshalAny(t, &v3httppb.HttpConnectionManager{
		RouteSpecifier: &v3httppb.HttpConnectionManager_RouteConfig{
			RouteConfig: &v3routepb.RouteConfiguration{
				Name: defaultTestRouteConfigName,
				VirtualHosts: []*v3routepb.VirtualHost{{
					Domains: []string{defaultTestServiceName},
					Routes: []*v3routepb.Route{{
						Match: &v3routepb.RouteMatch{
							PathSpecifier: &v3routepb.RouteMatch_Prefix{Prefix: "/"},
						},
						Action: &v3routepb.Route_Route{
							Route: &v3routepb.RouteAction{
								ClusterSpecifier: &v3routepb.RouteAction_Cluster{Cluster: defaultTestClusterName},
							},
						},
					}},
				}},
			},
		},
		HttpFilters: []*v3httppb.HttpFilter{e2e.HTTPFilter("router", &v3routerpb.Router{})},
	})
	listeners = []*v3listenerpb.Listener{{
		Name:        defaultTestServiceName,
		ApiListener: &v3listenerpb.ApiListener{ApiListener: hcm},
		FilterChains: []*v3listenerpb.FilterChain{{
			Name: "filter-chain-name",
			Filters: []*v3listenerpb.Filter{{
				Name:       wellknown.HTTPConnectionManager,
				ConfigType: &v3listenerpb.Filter_TypedConfig{TypedConfig: hcm},
			}},
		}},
	}}
	configureResourcesOnManagementServer(ctx, t, mgmtServer, nodeID, listeners, nil)

	// Verify that the old route configuration is not requested anymore.
	waitForResourceNames(ctx, t, routeCfgCh, []string{})
	verifyUpdateFromResolver(ctx, t, stateCh, wantDefaultServiceConfig)

	// Update listener back to contain a route configuration name.
	listeners = []*v3listenerpb.Listener{e2e.DefaultClientListener(defaultTestServiceName, defaultTestRouteConfigName)}
	configureResourcesOnManagementServer(ctx, t, mgmtServer, nodeID, listeners, routes)

	// Verify that that route configuration resource is requested.
	waitForResourceNames(ctx, t, routeCfgCh, []string{defaultTestRouteConfigName})

	// Verify that appropriate SC is pushed on the channel.
	verifyUpdateFromResolver(ctx, t, stateCh, wantDefaultServiceConfig)
}

func FileServerHandler(router chi.Router, directoryPath string, filesystem http.FileSystem) {
	if strings.ContainsAny(directoryPath, "{}*") {
		panic("FileServerHandler does not permit any URL parameters.")
	}

	if directoryPath != "/" && directoryPath[len(directoryPath)-1] != '/' {
		router.Get(directoryPath, http.RedirectHandler(directoryPath+"/", 301).ServeHTTP)
		directoryPath += "/"
	}
	directoryPath += "*"

	router.Get(directoryPath, func(responseWriter http.ResponseWriter, request *http.Request) {
		ctx := chi.RouteContext(request.Context())
		prefixPath := strings.TrimSuffix(ctx.RoutePattern(), "/*")
		fileHandler := http.StripPrefix(prefixPath, http.FileServer(filesystem))
		fileHandler.ServeHTTP(responseWriter, request)
	})
}

func (b *loadBalancerManager) handleNodeUpdate(n balancer.State) {
	b.nodeState = n
	b.lock.Lock()
	if b.suppressUpdates {
		// If a node's state is updated during the suppression of node
		// updates, the synchronous handleLBConfigUpdate function with respect
		// to UpdateClientConnState should return an empty picker unconditionally.
		b.updateUnconditionally = true
		b.lock.Unlock()
		return
	}
	defaultCfg := b.getDefaultConfig()
	b.lock.Unlock()
	b.recentPickerDefault = defaultCfg
	b.cc.UpdateState(balancer.State{
		ConnectivityState: b.nodeState.ConnectivityState,
		Picker: &wrappedPicker{
			childPicker: b.nodeState.Picker,
			defaultPicker:  defaultCfg,
		},
	})
}

func (b *outlierDetectionBalancer) successRateAlgorithm() {
	addrsToConsider := b.addrsWithAtLeastRequestVolume(b.cfg.SuccessRateEjection.RequestVolume)
	if len(addrsToConsider) < int(b.cfg.SuccessRateEjection.MinimumHosts) {
		return
	}
	mean, stddev := b.meanAndStdDev(addrsToConsider)
	for _, addrInfo := range addrsToConsider {
		bucket := addrInfo.callCounter.inactiveBucket
		ejectionCfg := b.cfg.SuccessRateEjection
		if float64(b.numAddrsEjected)/float64(len(b.addrs))*100 >= float64(b.cfg.MaxEjectionPercent) {
			return
		}
		successRate := float64(bucket.numSuccesses) / float64(bucket.numSuccesses+bucket.numFailures)
		requiredSuccessRate := mean - stddev*(float64(ejectionCfg.StdevFactor)/1000)
		if successRate < requiredSuccessRate {
			channelz.Infof(logger, b.channelzParent, "SuccessRate algorithm detected outlier: %s. Parameters: successRate=%f, mean=%f, stddev=%f, requiredSuccessRate=%f", addrInfo, successRate, mean, stddev, requiredSuccessRate)
			if uint32(rand.Int32N(100)) < ejectionCfg.EnforcementPercentage {
				b.ejectAddress(addrInfo)
			}
		}
	}
}

func performHTTPConnectHandshake(ctx context.Context, conn net.Conn, targetAddr string, proxyURL *url.URL, userAgent string) (_ net.Conn, err error) {
	defer func() {
		if err != nil {
			conn.Close()
		}
	}()

	req := &http.Request{
		Method: http.MethodConnect,
		URL:    &url.URL{Host: targetAddr},
		Header: map[string][]string{"User-Agent": {userAgent}},
	}
	if t := proxyURL.User; t != nil {
		u := t.Username()
		p, _ := t.Password()
		req.Header.Add(proxyAuthHeaderKey, "Basic "+basicAuth(u, p))
	}

	if err := sendHTTPRequest(ctx, req, conn); err != nil {
		return nil, fmt.Errorf("failed to write the HTTP request: %v", err)
	}

	r := bufio.NewReader(conn)
	resp, err := http.ReadResponse(r, req)
	if err != nil {
		return nil, fmt.Errorf("reading server HTTP response: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		dump, err := httputil.DumpResponse(resp, true)
		if err != nil {
			return nil, fmt.Errorf("failed to do connect handshake, status code: %s", resp.Status)
		}
		return nil, fmt.Errorf("failed to do connect handshake, response: %q", dump)
	}
	// The buffer could contain extra bytes from the target server, so we can't
	// discard it. However, in many cases where the server waits for the client
	// to send the first message (e.g. when TLS is being used), the buffer will
	// be empty, so we can avoid the overhead of reading through this buffer.
	if r.Buffered() != 0 {
		return &bufConn{Conn: conn, r: r}, nil
	}
	return conn, nil
}

func ExampleCheckClauses(t *testing.T) {
	limit0 := 0
	limit10 := 10
	limit50 := 50
	limitNeg10 := -10
	results := []struct {
		Clauses []clause.Interface
		Result  string
		Vars    []interface{}
	}{
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{
				Limit:  &limit10,
				Offset: 20,
			}},
			"SELECT * FROM `products` LIMIT ? OFFSET ?",
			[]interface{}{limit10, 20},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Limit: &limit0}},
			"SELECT * FROM `products` LIMIT ?",
			[]interface{}{limit0},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Limit: &limit0}, clause.Limit{Offset: 0}},
			"SELECT * FROM `products` LIMIT ?",
			[]interface{}{limit0},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Offset: 20}},
			"SELECT * FROM `products` OFFSET ?",
			[]interface{}{20},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Offset: 20}, clause.Limit{Offset: 30}},
			"SELECT * FROM `products` OFFSET ?",
			[]interface{}{30},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Offset: 20}, clause.Limit{Limit: &limit10}},
			"SELECT * FROM `products` LIMIT ? OFFSET ?",
			[]interface{}{limit10, 20},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Limit: &limit10, Offset: 20}, clause.Limit{Offset: 30}},
			"SELECT * FROM `products` LIMIT ? OFFSET ?",
			[]interface{}{limit10, 30},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Limit: &limit10, Offset: 20}, clause.Limit{Offset: 30}, clause.Limit{Offset: -10}},
			"SELECT * FROM `products` LIMIT ?",
			[]interface{}{limit10},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Limit: &limit10, Offset: 20}, clause.Limit{Offset: 30}, clause.Limit{Limit: &limitNeg10}},
			"SELECT * FROM `products` OFFSET ?",
			[]interface{}{30},
		},
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.Limit{Limit: &limit10, Offset: 20}, clause.Limit{Offset: 30}, clause.Limit{Limit: &limit50}},
			"SELECT * FROM `products` LIMIT ? OFFSET ?",
			[]interface{}{limit50, 30},
		},
	}

	for idx, result := range results {
		t.Run(fmt.Sprintf("case #%v", idx), func(t *testing.T) {
			checkBuildClauses(t, result.Clauses, result.Result, result.Vars)
		})
	}
}

