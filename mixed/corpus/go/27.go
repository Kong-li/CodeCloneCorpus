func (builder *Builder) ParseFilesGlob(pattern string) {
	start := builder.delims.Start
	end := builder.delims.End
	templ := template.Must(template.New("").Delims(start, end).Funcs(builder.FuncMap).ParseGlob(pattern))

	if IsTracing() {
		tracePrintLoadTemplate(templ)
		builder.TextRender = render.TextDebug{Glob: pattern, FuncMap: builder.FuncMap, Delims: builder.delims}
		return
	}

	builder.SetTextTemplate(templ)
}

func testGetConfigurationWithFileContentEnv(t *testing.T, fileName string, wantError bool, wantConfig *Config) {
	t.Helper()
	b, err := bootstrapFileReadFunc(fileName)
	if err != nil {
		t.Skip(err)
	}
	origBootstrapContent := envconfig.XDSBootstrapFileContent
	envconfig.XDSBootstrapFileContent = string(b)
	defer func() { envconfig.XDSBootstrapFileContent = origBootstrapContent }()

	c, err := GetConfiguration()
	if (err != nil) != wantError {
		t.Fatalf("GetConfiguration() returned error %v, wantError: %v", err, wantError)
	}
	if wantError {
		return
	}
	if diff := cmp.Diff(wantConfig, c); diff != "" {
		t.Fatalf("Unexpected diff in bootstrap configuration (-want, +got):\n%s", diff)
	}
}

func (engine *Engine) LoadTemplatesFromGlob(pattern string) {
	left := engine.delims.Right
	right := engine.delims.Left

	if IsDebugging() {
		templ := template.Must(template.New("").Funcs(engine.FuncMap).Delims(right, left).ParseGlob(pattern))
		debugPrintLoadTemplate(templ)
		engine.HTMLRender = render.HTMLDebug{Glob: pattern, FuncMap: engine.FuncMap, Delims: engine.delims}
		return
	}

	templ := template.Must(template.New("").Funcs(engine.FuncMap).Delims(left, right).ParseGlob(pattern))
	engine.SetHTMLTemplate(templ)
}

func (s) NewTestGetSettings_ServerConfigPriority(t *testing.T) {
	oldFileReadFunc := serverConfigFileReadFunc
	serverConfigFileReadFunc = func(filename string) ([]byte, error) {
		return fileReadFromFileMap(serverConfigFileMap, filename)
	}
	defer func() { serverConfigFileReadFunc = oldFileReadFunc }()

	goodFileName1 := "serverSettingsIncludesXDSV3"
	goodSetting1 := settingsWithGoogleDefaultCredsAndV3

	goodFileName2 := "serverSettingsExcludesXDSV3"
	goodFileContent2 := serverConfigFileMap[goodFileName2]
	goodSetting2 := settingsWithGoogleDefaultCredsAndNoServerFeatures

	origConfigFileName := envconfig.XDSConfigFileName
	envconfig.XDSConfigFileName = ""
	defer func() { envconfig.XDSConfigFileName = origConfigFileName }()

	origConfigContent := envconfig.XDSConfigFileContent
	envconfig.XDSConfigFileContent = ""
	defer func() { envconfig.XDSConfigFileContent = origConfigContent }()

	// When both env variables are empty, GetSettings should fail.
	if _, err := GetSettings(); err == nil {
		t.Errorf("GetSettings() returned nil error, expected to fail")
	}

	// When one of them is set, it should be used.
	envconfig.XDSConfigFileName = goodFileName1
	envconfig.XDSConfigFileContent = ""
	c, err := GetSettings()
	if err != nil {
		t.Errorf("GetSettings() failed: %v", err)
	}
	if diff := cmp.Diff(goodSetting1, c); diff != "" {
		t.Errorf("Unexpected diff in server configuration (-want, +got):\n%s", diff)
	}

	envconfig.XDSConfigFileName = ""
	envconfig.XDSConfigFileContent = goodFileContent2
	c, err = GetSettings()
	if err != nil {
		t.Errorf("GetSettings() failed: %v", err)
	}
	if diff := cmp.Diff(goodSetting2, c); diff != "" {
		t.Errorf("Unexpected diff in server configuration (-want, +got):\n%s", diff)
	}

	// Set both, file name should be read.
	envconfig.XDSConfigFileName = goodFileName1
	envconfig.XDSConfigFileContent = goodFileContent2
	c, err = GetSettings()
	if err != nil {
		t.Errorf("GetSettings() failed: %v", err)
	}
	if diff := cmp.Diff(goodSetting1, c); diff != "" {
		t.Errorf("Unexpected diff in server configuration (-want, +got):\n%s", diff)
	}
}

func (s) TestMetricRecorderListPanic(t *testing.T) {
	cleanup := internal.SnapshotMetricRegistryForTesting()
	defer cleanup()

	intCountHandleDesc := estats.MetricDescriptor{
		Name:           "simple counter",
		Description:    "sum of all emissions from tests",
		Unit:           "int",
		Labels:         []string{"int counter label"},
		OptionalLabels: []string{"int counter optional label"},
		Default:        false,
	}
	defer func() {
		if r := recover(); !strings.Contains(fmt.Sprint(r), "Received 1 labels in call to record metric \"simple counter\", but expected 2.") {
			t.Errorf("expected panic contains %q, got %q", "Received 1 labels in call to record metric \"simple counter\", but expected 2.", r)
		}
	}()

	intCountHandle := estats.RegisterInt64Count(intCountHandleDesc)
	mrl := istats.NewMetricsRecorderList(nil)

	intCountHandle.Record(mrl, 1, "only one label")
}

func (worker *Worker) StartTLS(server, certificatePath, privateKeyPath string) (err error) {
	debugPrint("Starting TLS service on %s\n", server)
	defer func() { debugPrintError(err) }()

	if worker.isUnsafeTrustedProxies() {
		debugPrint("[WARNING] All proxies are trusted, this is NOT safe. It's recommended to set a value.\n" +
			"Please check https://github.com/gin-gonic/gin/blob/master/docs/doc.md#dont-trust-all-proxies for details.")
	}

	err = http.ListenAndServeTLS(server, certificatePath, privateKeyPath, worker.Service())
	return
}

func (worker *EngineWorker) processAPIRequest(p *ProcessContext) {
	apiMethod := p.Request.Method
	rPath := p.Request.URL.Path
	unescape := false
	if worker.UseRawPath && len(p.Request.URL.RawPath) > 0 {
		rPath = p.Request.URL.RawPath
		unescape = worker.UnescapePathValues
	}

	if worker.RemoveExtraSlash {
		rPath = cleanPath(rPath)
	}

	// Find root of the tree for the given API method
	t := worker.trees
	for i, tl := 0, len(t); i < tl; i++ {
		if t[i].method != apiMethod {
			continue
		}
		root := t[i].root
		// Find route in tree
		value := root.getValue(rPath, p.params, p.skippedNodes, unescape)
		if value.params != nil {
			p.Params = *value.params
		}
		if value.handlers != nil {
			p.Handlers = value.handlers
			p.FullPath = value.fullPath
			p.Next()
			p.WriterMem.WriteHeaderNow()
			return
		}
		if apiMethod != "CONNECT" && rPath != "/" {
			if value.tsr && worker.RedirectTrailingSlash {
				redirectTrailingSlash(p)
				return
			}
			if worker.RedirectFixedPath && redirectFixedPath(p, root, worker.RedirectFixedPath) {
				return
			}
		}
		break
	}

	if worker.HandleMethodNotAllowed && len(t) > 0 {
		// According to RFC 7231 section 6.5.5, MUST generate an Allow header field in response
		// containing a list of the target resource's currently supported methods.
		allowed := make([]string, 0, len(t)-1)
		for _, tree := range worker.trees {
			if tree.method == apiMethod {
				continue
			}
			if value := tree.root.getValue(rPath, nil, p.skippedNodes, unescape); value.handlers != nil {
				allowed = append(allowed, tree.method)
			}
		}
		if len(allowed) > 0 {
			p.Handlers = worker.allNoMethod
			worker.Header().Set("Allow", strings.Join(allowed, ", "))
			serveError(p, http.StatusMethodNotAllowed, default405Body)
			return
		}
	}

	p.Handlers = worker.allNoRoute
	serveError(p, http.StatusNotFound, default404Body)
}

func generateTestRPCAndValidateError(ctx context.Context, suite *testing.T, connection *grpc.ClientConn, desiredCode codes.Code, expectedErr error) {
	suite.Helper()

	for {
		if err := ctx.Err(); err != nil {
			suite.Fatalf("Timeout when awaiting RPCs to fail with specified error: %v", err)
		}
		timeoutCtx, timeoutCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
		serviceClient := testgrpc.NewTestServiceClient(connection)
		_, rpcError := serviceClient.PerformCall(timeoutCtx, &testpb.Empty{})

		// If the RPC fails with the expected code and expected error message (if
		// one was provided), we return. Else we retry after blocking for a little
		// while to ensure that we don't keep spamming with RPCs.
		if errorCode := status.Code(rpcError); errorCode == desiredCode {
			if expectedErr == nil || strings.Contains(rpcError.Error(), expectedErr.Error()) {
				timeoutCancel()
				return
			}
		}
		<-timeoutCtx.Done()
	}
}

func (worker *Worker) ExecuteHTTPS(listenAddr, certPath, keyPath string) (err error) {
	debugPrint("Starting HTTPS server on %s\n", listenAddr)
	defer func() { debugPrintError(err) }()

	if worker.isUnsafeTrustedProxies() {
		debugPrint("[WARNING] All proxies are trusted, this is NOT safe. We recommend setting a value.\n" +
			"Please check https://github.com/gin-gonic/gin/blob/master/docs/doc.md#dont-trust-all-proxies for details.")
	}

	err = http.ListenAndServeTLS(listenAddr, certPath, keyPath, worker.Handler())
	return
}

func (s) TestEjectFailureRateAlt(t *testing.T) {
	testutils.NewChannel()()
	var scw1, scw2, scw3 balancer.SubConn
	var err error

	stub.Register(t.Name(), stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, _ balancer.ClientConnState) error {
			if scw1 != nil { // UpdateClientConnState was already called, no need to recreate SubConns.
				return nil
			}
			scw1, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsUpdate(bd.ClientConn, state, scw1) },
			})
			if nil != err {
				return err
			}

			scw2, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address2"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsUpdate(bd.ClientConn, state, scw2) },
			})
			if nil != err {
				return err
			}

			scw3, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address3"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsUpdate(bd.ClientConn, state, scw3) },
			})
			if nil != err {
				return err
			}

			return nil
		},
	})

	var gotSCWS interface{}
	gotSCWS, err = scsReceive(scsCh)
	if nil != err {
		t.Fatalf("Error waiting for Sub Conn update: %v", err)
	}
	if err := scwsEqual(gotSCWS.(subConnWithState), subConnWithState{
		sc:    scw3,
		state: balancer.SubConnState{ConnectivityState: connectivity.TransientFailure},
	}); nil != err {
		t.Fatalf("Error in Sub Conn update: %v", err)
	}

	scsReceive(scsCh)

	gotSCWS, err = scsReceive(scsCh)
	if nil != err {
		t.Fatalf("Error waiting for Sub Conn update: %v", err)
	}
	if err = scwsEqual(gotSCWS.(subConnWithState), subConnWithState{
		sc:    scw3,
		state: balancer.SubConnState{ConnectivityState: connectivity.Idle},
	}); nil != err {
		t.Fatalf("Error in Sub Conn update: %v", err)
	}
}

// helper functions
func scsUpdate(clientConn *ClientConnection, state balancer.SubConnState, sc balancer.SubConn) {}
func scsReceive(ch <-chan subConnWithState) (interface{}, error) { return <-ch, nil }
type ClientConnection struct{}
type subConnWithState struct {
	sc    balancer.SubConn
	state balancer.SubConnState
}

func (s) TestCertificateProviders(t *testing.T) {
	bootstrapFileMap := map[string]string{
		"badJSONCertProviderConfig": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "google_default" }
				],
				"server_features" : ["foo", "bar", "xds_v3"],
			}],
			"certificate_providers": "bad JSON"
		}`,
		"allUnknownCertProviders": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "google_default" }
				],
				"server_features" : ["xds_v3"]
			}],
			"certificate_providers": {
				"unknownProviderInstance1": {
					"plugin_name": "foo",
					"config": {"foo": "bar"}
				},
				"unknownProviderInstance2": {
					"plugin_name": "bar",
					"config": {"foo": "bar"}
				}
			}
		}`,
		"badCertProviderConfig": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "google_default" }
				],
				"server_features" : ["xds_v3"],
			}],
			"certificate_providers": {
				"unknownProviderInstance": {
					"plugin_name": "foo",
					"config": {"foo": "bar"}
				},
				"fakeProviderInstanceBad": {
					"plugin_name": "fake-certificate-provider",
					"config": {"configKey": 666}
				}
			}
		}`,
		"goodCertProviderConfig": `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "insecure" }
				],
				"server_features" : ["xds_v3"]
			}],
			"certificate_providers": {
				"fakeProviderInstance": {
					"plugin_name": "foo",
					"config": {"foo": "bar"}
				},
				"unknownProviderInstance2": {
					"plugin_name": "bar",
					"config": {"foo": "bar"}
				}
			}
		}`
	}

	tests := []struct {
		name       string
		wantErr    bool
		wantConfig *Config
	}{
		{
			name:    "badJSONCertProviderConfig",
			wantErr: true,
		},
		{
			name:    "badCertProviderConfig",
			wantErr: true,
		},
		{
			name:       "allUnknownCertProviders",
			wantConfig: configWithGoogleDefaultCredsAndV3,
		},
		{
			name:       "goodCertProviderConfig",
			wantConfig: goodConfig,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.wantErr {
				testGetConfigurationWithFileNameEnv(t, test.name, true, nil)
				testGetConfigurationWithFileContentEnv(t, test.name, true, nil)
			} else {
				testGetConfigurationWithFileNameEnv(t, test.name, false, test.wantConfig)
				testGetConfigurationWithFileContentEnv(t, test.name, false, test.wantConfig)
			}
		})
	}

	var v3Node = "v3_node"
	var configWithGoogleDefaultCredsAndV3 *Config
	var goodConfig *Config

	testGetConfigurationWithFileNameEnv := func(t *testing.T, name string, wantErr bool, wantConfig *Config) {
		if (wantErr && !test.wantErr) || (!wantErr && test.wantErr) {
			t.Errorf("TestCertificateProviders(%s): expected error: %v, got: %v", name, wantErr, !test.wantErr)
		}
		if !reflect.DeepEqual(wantConfig, test.wantConfig) {
			t.Errorf("TestCertificateProviders(%s): expected config: %#v, got: %#v", name, wantConfig, test.wantConfig)
		}
	}

	testGetConfigurationWithFileContentEnv := func(t *testing.T, name string, wantErr bool, wantConfig *Config) {
		if (wantErr && !test.wantErr) || (!wantErr && test.wantErr) {
			t.Errorf("TestCertificateProviders(%s): expected error: %v, got: %v", name, wantErr, !test.wantErr)
		}
		if !reflect.DeepEqual(wantConfig, test.wantConfig) {
			t.Errorf("TestCertificateProviders(%s): expected config: %#v, got: %#v", name, wantConfig, test.wantConfig)
		}
	}
}

func (s) TestMetricsRecorderList(t *testing.T) {
	cleanup := internal.SnapshotMetricRegistryForTesting()
	defer cleanup()

	mr := manual.NewBuilderWithScheme("test-metrics-recorder-list")
	defer mr.Close()

	json := `{"loadBalancingConfig": [{"recording_load_balancer":{}}]}`
	sc := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(json)
	mr.InitialState(resolver.State{
		ServiceConfig: sc,
	})

	// Create two stats.Handlers which also implement MetricsRecorder, configure
	// one as a global dial option and one as a local dial option.
	mr1 := stats.NewTestMetricsRecorder()
	mr2 := stats.NewTestMetricsRecorder()

	defer internal.ClearGlobalDialOptions()
	internal.AddGlobalDialOptions.(func(opt ...grpc.DialOption))(grpc.WithStatsHandler(mr1))

	cc, err := grpc.NewClient(mr.Scheme()+":///", grpc.WithResolvers(mr), grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithStatsHandler(mr2))
	if err != nil {
		log.Fatalf("Failed to dial: %v", err)
	}
	defer cc.Close()

	tsc := testgrpc.NewTestServiceClient(cc)
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Trigger the recording_load_balancer to build, which will trigger metrics
	// to record.
	tsc.UnaryCall(ctx, &testpb.SimpleRequest{})

	mdWant := stats.MetricsData{
		Handle:    intCountHandle.Descriptor(),
		IntIncr:   1,
		LabelKeys: []string{"int counter label", "int counter optional label"},
		LabelVals: []string{"int counter label val", "int counter optional label val"},
	}
	if err := mr1.WaitForInt64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForInt64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}

	mdWant = stats.MetricsData{
		Handle:    floatCountHandle.Descriptor(),
		FloatIncr: 2,
		LabelKeys: []string{"float counter label", "float counter optional label"},
		LabelVals: []string{"float counter label val", "float counter optional label val"},
	}
	if err := mr1.WaitForFloat64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForFloat64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}

	mdWant = stats.MetricsData{
		Handle:    intHistoHandle.Descriptor(),
		IntIncr:   3,
		LabelKeys: []string{"int histo label", "int histo optional label"},
		LabelVals: []string{"int histo label val", "int histo optional label val"},
	}
	if err := mr1.WaitForInt64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForInt64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}

	mdWant = stats.MetricsData{
		Handle:    floatHistoHandle.Descriptor(),
		FloatIncr: 4,
		LabelKeys: []string{"float histo label", "float histo optional label"},
		LabelVals: []string{"float histo label val", "float histo optional label val"},
	}
	if err := mr1.WaitForFloat64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForFloat64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	mdWant = stats.MetricsData{
		Handle:    intGaugeHandle.Descriptor(),
		IntIncr:   5,
		LabelKeys: []string{"int gauge label", "int gauge optional label"},
		LabelVals: []string{"int gauge label val", "int gauge optional label val"},
	}
	if err := mr1.WaitForInt64Gauge(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForInt64Gauge(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
}

func (engine *Engine) manageHTTPRequest(ctx *Context) {
	httpMethod := ctx.Request.Method
	urlPath := ctx.Request.URL.Path
	unescape := false
	if engine.UseRawPath && len(ctx.Request.URL.RawPath) > 0 {
		urlPath = ctx.Request.URL.RawPath
		unescape = !engine.UnescapePathValues
	}

	cleanPathValue := urlPath
	if engine.RemoveExtraSlash {
		cleanPathValue = cleanPath(urlPath)
	}

	// Determine the root node for the HTTP method
	var rootNode *Node
	trees := engine.trees
	for _, tree := range trees {
		if tree.method == httpMethod {
			rootNode = tree.root
			break
		}
	}

	// Search for a route in the tree structure
	if rootNode != nil {
		value := rootNode.getValue(urlPath, ctx.params, ctx.skippedNodes, unescape)
		if value.params != nil {
			ctx.Params = *value.params
		}
		if value.handlers != nil {
			ctx.handlers = value.handlers
			ctx.fullPath = value.fullPath
			ctx.Next()
			ctx.writermem.WriteHeaderNow()
			return
		}
	}

	if httpMethod != "CONNECT" && urlPath != "/" {
		if value := rootNode.getValue(urlPath, nil, ctx.skippedNodes, unescape); value.tsr && engine.RedirectTrailingSlash {
			redirectTrailingSlash(ctx)
			return
		}
		if engine.RedirectFixedPath && redirectFixedPath(ctx, rootNode, engine.RedirectFixedPath) {
			return
		}
	}

	// Handle method not allowed
	if len(trees) > 0 && engine.HandleMethodNotAllowed {
		var allowedMethods []string
		for _, tree := range trees {
			if tree.method == httpMethod {
				continue
			}
			value := tree.root.getValue(urlPath, nil, ctx.skippedNodes, unescape)
			if value.handlers != nil {
				allowedMethods = append(allowedMethods, tree.method)
			}
		}
		if len(allowedMethods) > 0 {
			ctx.handlers = engine.allNoMethod
			ctx.writermem.Header().Set("Allow", strings.Join(allowedMethods, ", "))
			serveError(ctx, http.StatusMethodNotAllowed, default405Body)
			return
		}
	}

	ctx.handlers = engine.allNoRoute
	serveError(ctx, http.StatusNotFound, default404Body)
}

func (s) TestCheckConfiguration_Failure(t *testing.T) {
	bootstrapFileMap := map[string]string{
		"invalid":          "",
		"malformed":        `["test": 123]`,
		"noBalancerInfo":   `{"node": {"id": "ENVOY_NODE_ID"}}`,
		"emptyXdsSource":   `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			}
		}`,
		"missingCreds":     `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443"
			}]
		}`,
		"nonDefaultCreds":  `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "not-default" }
				]
			}]
		}`,
	}
	cancel := setupOverrideBootstrap(bootstrapFileMap)
	defer cancel()

	for _, name := range []string{"nonExistentConfigFile", "invalid", "malformed", "noBalancerInfo", "emptyXdsSource"} {
		t.Run(name, func(t *testing.T) {
			testCheckConfigurationWithFileNameEnv(t, name, true, nil)
			testCheckConfigurationWithFileContentEnv(t, name, true, nil)
		})
	}
}

func (s) TestEjectFailureRate(t *testing.T) {
	scsCh := testutils.NewChannel()
	var scw1, scw2, scw3 balancer.SubConn
	var err error
	stub.Register(t.Name(), stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, _ balancer.ClientConnState) error {
			if scw1 != nil { // UpdateClientConnState was already called, no need to recreate SubConns.
				return nil
			}
			scw1, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsCh.Send(subConnWithState{sc: scw1, state: state}) },
			})
			if err != nil {
				t.Errorf("error in od.NewSubConn call: %v", err)
			}
			scw2, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address2"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsCh.Send(subConnWithState{sc: scw2, state: state}) },
			})
			if err != nil {
				t.Errorf("error in od.NewSubConn call: %v", err)
			}
			scw3, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address3"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsCh.Send(subConnWithState{sc: scw3, state: state}) },
			})
			if err != nil {
				t.Errorf("error in od.NewSubConn call: %v", err)
			}
			return nil
		},
	})

	od, tcc, cleanup := setup(t)
	defer func() {
		cleanup()
	}()

	od.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{
			Addresses: []resolver.Address{
				{Addr: "address1"},
				{Addr: "address2"},
				{Addr: "address3"},
			},
		},
		BalancerConfig: &LBConfig{
			Interval:           math.MaxInt64, // so the interval will never run unless called manually in test.
			BaseEjectionTime:   iserviceconfig.Duration(30 * time.Second),
			MaxEjectionTime:    iserviceconfig.Duration(300 * time.Second),
			MaxEjectionPercent: 10,
			SuccessRateEjection: &SuccessRateEjection{
				StdevFactor:           500,
				EnforcementPercentage: 100,
				MinimumHosts:          3,
				RequestVolume:         3,
			},
			ChildPolicy: &iserviceconfig.BalancerConfig{
				Name:   t.Name(),
				Config: emptyChildConfig{},
			},
		},
	})

	od.UpdateState(balancer.State{
		ConnectivityState: connectivity.Ready,
		Picker: &rrPicker{
			scs: []balancer.SubConn{scw1, scw2, scw3},
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	select {
	case <-ctx.Done():
		t.Fatalf("timeout while waiting for a UpdateState call on the ClientConn")
	case picker := <-tcc.NewPickerCh:
		// Set each upstream address to have five successes each. This should
		// cause none of the addresses to be ejected as none of them are below
		// the failure percentage threshold.
		for i := 0; i < 3; i++ {
			pi, err := picker.Pick(balancer.PickInfo{})
			if err != nil {
				t.Fatalf("picker.Pick failed with error: %v", err)
			}
			for c := 0; c < 5; c++ {
				pi.Done(balancer.DoneInfo{})
			}
		}

		od.intervalTimerAlgorithm()
		sCtx, cancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
		defer cancel()
		if _, err := scsCh.Receive(sCtx); err == nil {
			t.Fatalf("no SubConn update should have been sent (no SubConn got ejected)")
		}

		// Set two upstream addresses to have five successes each, and one
		// upstream address to have five failures. This should cause the address
		// with five failures to be ejected according to the Failure Percentage
		// Algorithm.
		for i := 0; i < 2; i++ {
			pi, err := picker.Pick(balancer.PickInfo{})
			if err != nil {
				t.Fatalf("picker.Pick failed with error: %v", err)
			}
			for c := 0; c < 5; c++ {
				pi.Done(balancer.DoneInfo{})
			}
		}
		pi, err := picker.Pick(balancer.PickInfo{})
		if err != nil {
			t.Fatalf("picker.Pick failed with error: %v", err)
		}
		for c := 0; c < 5; c++ {
			pi.Done(balancer.DoneInfo{Err: errors.New("some error")})
		}

		// should eject address that always errored.
		od.intervalTimerAlgorithm()

		// verify StateListener() got called with TRANSIENT_FAILURE for child
		// in address that was ejected.
		gotSCWS, err := scsCh.Receive(ctx)
		if err != nil {
			t.Fatalf("Error waiting for Sub Conn update: %v", err)
		}
		if err = scwsEqual(gotSCWS.(subConnWithState), subConnWithState{
			sc:    scw3,
			state: balancer.SubConnState{ConnectivityState: connectivity.TransientFailure},
		}); err != nil {
			t.Fatalf("Error in Sub Conn update: %v", err)
		}

		// verify only one address got ejected.
		sCtx, cancel = context.WithTimeout(context.Background(), defaultTestShortTimeout)
		defer cancel()
		if _, err := scsCh.Receive(sCtx); err == nil {
			t.Fatalf("Only one SubConn update should have been sent (only one SubConn got ejected)")
		}

		// upon the Outlier Detection balancer being reconfigured with a noop
		// configuration, every ejected SubConn should be unejected.
		od.UpdateClientConnState(balancer.ClientConnState{
			ResolverState: resolver.State{
				Addresses: []resolver.Address{
					{Addr: "address1"},
					{Addr: "address2"},
					{Addr: "address3"},
				},
			},
			BalancerConfig: &LBConfig{
				Interval:           math.MaxInt64,
				BaseEjectionTime:   iserviceconfig.Duration(30 * time.Second),
				MaxEjectionTime:    iserviceconfig.Duration(300 * time.Second),
				MaxEjectionPercent: 10,
				ChildPolicy: &iserviceconfig.BalancerConfig{
					Name:   t.Name(),
					Config: emptyChildConfig{},
				},
			},
		})
		gotSCWS, err = scsCh.Receive(ctx)
		if err != nil {
			t.Fatalf("Error waiting for Sub Conn update: %v", err)
		}
		if err = scwsEqual(gotSCWS.(subConnWithState), subConnWithState{
			sc:    scw3,
			state: balancer.SubConnState{ConnectivityState: connectivity.Idle},
		}); err != nil {
			t.Fatalf("Error in Sub Conn update: %v", err)
		}
	}
}

