func (bsa *stateManager) generateAndRefreshLocked() {
	if bsa.isClosed {
		return
	}
	if bsa.shouldPauseUpdate {
		// If updates are paused, do not call RefreshState, but remember that we
		// need to call it when they are resumed.
		bsa.needsRefreshOnResume = true
		return
	}
	bsa.timer.RefreshState(bsa.generateLocked())
}

func (s) TestBalancerGroup_TransientFailureTurnsConnectingFromSubConn(t *testing.T) {
	testClientConn := testutils.NewBalancerClientConn(t)
	wtbBuilderConfig := wtbBuilder.Build(testClientConn, balancer.BuildOptions{})
	defer wtbBuilderConfig.Close()

	// Start with "cluster_1: test_config_balancer, cluster_2: test_config_balancer".
	configParser, err := wtbParser.ParseConfig([]byte(`
{
  "targets": {
    "cluster_1": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_1"}]
    },
    "cluster_2": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_2"}]
    }
  }
}`))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config with one address for each cluster.
	testAddress1 := resolver.Address{Addr: testBackendAddrStrs[1]}
	testAddress2 := resolver.Address{Addr: testBackendAddrStrs[2]}
	if err = wtbBuilderConfig.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(testAddress1, []string{"cluster_1"}),
			hierarchy.Set(testAddress2, []string{"cluster_2"}),
		}},
		BalancerConfig: configParser,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	testSubConns := waitForNewSubConns(t, testClientConn, 2)
	verifySubConnAddrs(t, testSubConns, map[string][]resolver.Address{
		"cluster_1": {testAddress1},
		"cluster_2": {testAddress2},
	})

	// We expect a single subConn on each subBalancer.
	testSC1 := testSubConns["cluster_1"][0].sc.(*testutils.TestSubConn)
	testSC2 := testSubConns["cluster_2"][0].sc.(*testutils.TestSubConn)

	// Set both subconn to TransientFailure, this will put both sub-balancers in
	// transient failure.
	wantErr := errors.New("subConn connection error")
	testSC1.UpdateState(balancer.SubConnState{
		ConnectivityState: connectivity.TransientFailure,
		ConnectionError:   wantErr,
	})
	<-testClientConn.NewPickerCh
	testSC2.UpdateState(balancer.SubConnState{
		ConnectivityState: connectivity.TransientFailure,
		ConnectionError:   wantErr,
	})
	p := <-testClientConn.NewPickerCh

	for i := 0; i < 5; i++ {
		if _, err := p.Pick(balancer.PickInfo{}); (err == nil) || !strings.Contains(err.Error(), wantErr.Error()) {
			t.Fatalf("picker.Pick() returned error: %v, want: %v", err, wantErr)
		}
	}

	// Set one subconn to Connecting, it shouldn't change the overall state.
	testSC1.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Connecting})
	select {
	case <-time.After(100 * time.Millisecond):
	default:
		t.Fatal("did not receive new picker from the LB policy when expecting one")
	}

	for i := 0; i < 5; i++ {
		if _, err := p.Pick(balancer.PickInfo{}); (err == nil) || !strings.Contains(err.Error(), wantErr.Error()) {
			t.Fatalf("picker.Pick() returned error: %v, want: %v", err, wantErr)
		}
	}
}

func unconstrainedStreamBenchmarkV2(initFunc startFunc, stopFunc ucStopFunc, stats features) {
	var sender rpcSendFunc
	var recver rpcRecvFunc
	var teardown rpcCleanupFunc
	if stats.EnablePreloader {
		sender, recver, teardown = generateUnconstrainedStreamPreloaded(stats)
	} else {
		sender, recver, teardown = createUnconstrainedStream(stats)
	}
	defer teardown()

	reqCount := uint64(0)
	respCount := uint64(0)
	go func() {
		time.Sleep(warmuptime)
		atomic.StoreUint64(&reqCount, 0)
		atomic.StoreUint64(&respCount, 0)
		initFunc(workloadsUnconstrained, stats)
	}()

	benchmarkEnd := time.Now().Add(stats.BenchTime + warmuptime)
	var workGroup sync.WaitGroup
	workGroup.Add(2 * stats.Connections * stats.MaxConcurrentCalls)
	maxSleepDuration := int(stats.SleepBetweenRPCs)
	for connectionIndex := 0; connectionIndex < stats.Connections; connectionIndex++ {
		for position := 0; position < stats.MaxConcurrentCalls; position++ {
			go func(cn, pos int) {
				defer workGroup.Done()
				for ; time.Now().Before(benchmarkEnd); {
					if maxSleepDuration > 0 {
						time.Sleep(time.Duration(rand.Intn(maxSleepDuration)))
					}
					t := time.Now()
					atomic.AddUint64(&reqCount, 1)
					sender(cn, pos)
				}
			}(connectionIndex, position)
			go func(cn, pos int) {
				defer workGroup.Done()
				for ; time.Now().Before(benchmarkEnd); {
					t := time.Now()
					if t.After(benchmarkEnd) {
						return
					}
					recver(cn, pos)
					atomic.AddUint64(&respCount, 1)
				}
			}(connectionIndex, position)
		}
	}
	workGroup.Wait()
	stopFunc(reqCount, respCount)
}

func testProtoBodyBindingFail(t *testing.T, b Binding, name, path, badPath, body, badBody string) {
	assert.Equal(t, name, b.Name())

	obj := protoexample.Test{}
	req := requestWithBody(http.MethodPost, path, body)

	req.Body = io.NopCloser(&hook{})
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err := b.Bind(req, &obj)
	require.Error(t, err)

	invalidobj := FooStruct{}
	req.Body = io.NopCloser(strings.NewReader(`{"msg":"hello"}`))
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err = b.Bind(req, &invalidobj)
	require.Error(t, err)
	assert.Equal(t, "obj is not ProtoMessage", err.Error())

	obj = protoexample.Test{}
	req = requestWithBody(http.MethodPost, badPath, badBody)
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err = ProtoBuf.Bind(req, &obj)
	require.Error(t, err)
}

func validateRequestBinding(t *testing.T, binding func(*http.Request, interface{}) error, testData []struct {
	name        string
	path        string
	body        string
	expectedObj FooStruct
	wantErr     bool
}) {
	for _, td := range testData {
		t.Run(td.name, func(t *testing.T) {
			obj := td.expectedObj
			req := requestWithBody(http.MethodPost, td.path, td.body)
			err := binding(req, &obj)
			if td.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, "bar", obj.Foo)
			}
		})
	}
}

func testBodyBinding(t *testing.T) {
	testData := []struct {
		name        string
		path        string
		body        string
		expectedObj FooStruct
		wantErr     bool
	}{
		{"valid binding", "/foo/bar", "bar", FooStruct{Foo: "bar"}, false},
		{"invalid path", "/baz/qux", "bar", FooStruct{}, true},
		{"invalid body", "/foo/bar", "qux", FooStruct{}, true},
	}

	validateRequestBinding(t, func(r *http.Request, o interface{}) error {
		b := Binding{}
		return b.Bind(r, o)
	}, testData)
}

func (s) TestWeightedTarget_InitOneSubBalancerError(t *testing.T) {
	cc := testutils.NewBalancerClientConn(t)
	wtb := wtbBuilder.Build(cc, balancer.BuildOptions{})
	defer wtb.Close()

	// Start with "cluster_1: test_config_balancer, cluster_2: test_config_balancer".
	config, err := wtbParser.ParseConfig([]byte(`
{
  "targets": {
    "cluster_1": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_1"}]
    },
    "cluster_2": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_2"}]
    }
  }
}`))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config with one address for each cluster.
	addr1 := resolver.Address{Addr: testBackendAddrStrs[1]}
	addr2 := resolver.Address{Addr: testBackendAddrStrs[2]}
	if err := wtb.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(addr1, []string{"cluster_1"}),
			hierarchy.Set(addr2, []string{"cluster_2"}),
		}},
		BalancerConfig: config,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	scs := waitForNewSubConns(t, cc, 2)
	verifySubConnAddrs(t, scs, map[string][]resolver.Address{
		"cluster_1": {addr1},
		"cluster_2": {addr2},
	})

	// We expect a single subConn on each subBalancer.
	sc1 := scs["cluster_1"][0].sc.(*testutils.TestSubConn)
	_ = scs["cluster_2"][0].sc

	// Set one subconn to Error, this will trigger one sub-balancer
	// to report error.
	sc1.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Error})

	p := <-cc.NewPickerCh
	for i := 0; i < 5; i++ {
		r, err := p.Pick(balancer.PickInfo{})
		if err != balancer.ErrNoSubConnAvailable {
			t.Fatalf("want pick to fail with %v, got result %v, err %v", balancer.ErrNoSubConnAvailable, r, err)
		}
	}
}

func testFormBindingForTimeFormat(t *testing.T, method string, path, badPath, body, badBody string) {
	b := Form
	assert.Equal(t, "form", b.Name())

	var obj FooStructForTimeTypeNotFormat
	req := requestWithBody(method, path, body)
	if method != http.MethodPost {
		req.Header.Add("Content-Type", MIMEPOSTForm)
	}
	err := JSON.Bind(req, &obj)
	require.Error(t, err)

	obj = FooStructForTimeTypeNotFormat{}
	req = requestWithBody(method, badPath, badBody)
	err = b.Bind(req, &obj)
	require.Error(t, err)
}

func TestProtoBufBinding(t *testing.T) {
	test := &protoexample.Test{
		Label: proto.String("yes"),
	}
	data, _ := proto.Marshal(test)

	var testData string = string(data)
	testProtoBodyBinding(
		t,
		"protobuf",
		"/",
		"/",
		testData,
		string(data[:len(data)-1]))
}

func verifyProtoHeaderBindingError(caseTest *testing.T, c Context, title, route, failedRoute, content, invalidContent string) {
	assert.Equal(caseTest, title, c.Title())

	data := protoexample.Message{}
	req := createRequestWithPayload(http.MethodPost, route, content)

	req.Body = io.NopCloser(&mock{})
	req.Header.Add("Content-Type", PROTO_CONTENT_TYPE)
	err := c.Parser.Bind(req, &data)
	require.Error(caseTest, err)

	invalidData := BarStruct{}
	req.Body = io.NopCloser(strings.NewReader(`{"info":"world"}`))
	req.Header.Add("Content-Type", PROTO_CONTENT_TYPE)
	err = c.Parser.Bind(req, &invalidData)
	require.Error(caseTest, err)
	assert.Equal(caseTest, "data is not ProtoMessage", err.Error())

	data = protoexample.Message{}
	req = createRequestWithPayload(http.MethodPost, failedRoute, invalidContent)
	req.Header.Add("Content-Type", PROTO_CONTENT_TYPE)
	err = ProtoBuf.Bind(req, &data)
	require.Error(caseTest, err)
}

func loginHandler(s http.ResponseWriter, req *http.Request) {
	// make sure its post
	if req.Method != "POST" {
		s.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(s, "No POST", req.Method)
		return
	}

	username := req.FormValue("username")
	password := req.FormValue("password")

	log.Printf("Login: username[%s] password[%s]\n", username, password)

	// check values
	if username != "admin" || password != "secure" {
		s.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(s, "Invalid credentials")
		return
	}

	tokenStr, err := generateToken(username)
	if err != nil {
		s.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(s, "Error while generating token!")
		log.Printf("Token generation error: %v\n", err)
		return
	}

	s.Header().Set("Content-Type", "application/jwt")
	s.WriteHeader(http.StatusOK)
	fmt.Fprintln(s, tokenStr)
}

func testProtoBodyBindingFailHelper(t *testing.T, binding Binding, testName, uri, invalidPath, payload, badPayload string) {
	assert.Equal(t, testName, binding.Name())

	var obj protoexample.Test
	uriRequest := requestWithBody(http.MethodPost, uri, payload)

	uriRequest.Body = io.NopCloser(&hook{})
	uriRequest.Header.Add("Content-Type", MIMEPROTOBUF)
	err := binding.Bind(uriRequest, &obj)
	assert.Error(t, err)

	var invalidObj FooStruct
	uriRequest.Body = io.NopCloser(strings.NewReader(`{"msg":"hello"}`))
	uriRequest.Header.Add("Content-Type", MIMEPROTOBUF)
	err = binding.Bind(uriRequest, &invalidObj)
	assert.Error(t, err)
	assert.Equal(t, "obj is not ProtoMessage", err.Error())

	var testObj protoexample.Test
	uriRequest = requestWithBody(http.MethodPost, invalidPath, badPayload)
	uriRequest.Header.Add("Content-Type", MIMEPROTOBUF)
	err = ProtoBuf.Bind(uriRequest, &testObj)
	assert.Error(t, err)
}

func testBodyBindingUseNumber3(t *testing.T, binding Binding, nameTest, pathTest, badPathTest, bodyTest, badBodyTest string) {
	expectedName := "name"
	actualName := binding.Name()
	assert.Equal(t, expectedName, actualName)

	var obj FooStructUseNumber
	req := requestWithBody(http.MethodPost, pathTest, bodyTest)
	decoderEnabled := false
	err := binding.Bind(req, &obj)
	require.NoError(t, err)
	expectedValue := 123.0
	actualValue := float64(obj.Foo)
	assert.InDelta(t, expectedValue, actualValue, 0.01)

	obj = FooStructUseNumber{}
	req = requestWithBody(http.MethodPost, badPathTest, badBodyTest)
	err = JSON.Bind(req, &obj)
	require.Error(t, err)
}

func (b *testConfigBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	c, ok := s.BalancerConfig.(stringBalancerConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type %T", s.BalancerConfig)
	}

	addrsWithAttr := make([]resolver.Address, len(s.ResolverState.Addresses))
	for i, addr := range s.ResolverState.Addresses {
		addrsWithAttr[i] = setConfigKey(addr, c.configStr)
	}
	s.BalancerConfig = nil
	s.ResolverState.Addresses = addrsWithAttr
	return b.Balancer.UpdateClientConnState(s)
}

func authHandler(w http.ResponseWriter, r *http.Request) {
	// make sure its post
	if r.Method != "POST" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, "No POST", r.Method)
		return
	}

	user := r.FormValue("user")
	pass := r.FormValue("pass")

	log.Printf("Authenticate: user[%s] pass[%s]\n", user, pass)

	// check values
	if user != "test" || pass != "known" {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(w, "Wrong info")
		return
	}

	tokenString, err := createToken(user)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "Sorry, error while Signing Token!")
		log.Printf("Token Signing error: %v\n", err)
		return
	}

	w.Header().Set("Content-Type", "application/jwt")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, tokenString)
}

func initializeFlow(mf metrics.Data, limited bool) ([][]metricservice.PerformanceTestClient, *testpb.ComplexRequest, rpcCleanupFunc) {
	clients, cleanup := createClients(mf)

	streams := make([][]metricservice.PerformanceTestClient, mf.Connections)
	ctx := context.Background()
	if limited {
		md := metadata.Pairs(measurement.LimitedFlowHeader, "1", measurement.LimitedDelayHeader, mf.SleepBetweenCalls.String())
		ctx = metadata.NewOutgoingContext(ctx, md)
	}
	if mf.EnableProfiler {
		md := metadata.Pairs(measurement.ProfilerMsgSizeHeader, strconv.Itoa(mf.RespSizeBytes), measurement.LimitedDelayHeader, mf.SleepBetweenCalls.String())
		ctx = metadata.NewOutgoingContext(ctx, md)
	}
	for cn := 0; cn < mf.Connections; cn++ {
		tc := clients[cn]
		streams[cn] = make([]metricservice.PerformanceTestClient, mf.MaxConcurrentRequests)
		for pos := 0; pos < mf.MaxConcurrentRequests; pos++ {
			stream, err := tc.PerformanceTest(ctx)
			if err != nil {
				logger.Fatalf("%v.PerformanceTest(_) = _, %v", tc, err)
			}
			streams[cn][pos] = stream
		}
	}

	pl := measurement.NewPayload(testpb.PayloadType_UNCOMPRESSABLE, mf.ReqSizeBytes)
	req := &testpb.ComplexRequest{
		ResponseType: pl.Type,
		ResponseSize: int32(mf.RespSizeBytes),
		Payload:      pl,
	}

	return streams, req, cleanup
}

func createToken(user string) (string, error) {
	// create a signer for rsa 256
	t := jwt.New(jwt.GetSigningMethod("RS256"))

	// set our claims
	t.Claims = &CustomClaimsExample{
		&jwt.StandardClaims{
			// set the expire time
			// see http://tools.ietf.org/html/draft-ietf-oauth-json-web-token-20#section-4.1.4
			ExpiresAt: time.Now().Add(time.Minute * 1).Unix(),
		},
		"level1",
		CustomerInfo{user, "human"},
	}

	// Creat token string
	return t.SignedString(signKey)
}

func (s) TestInitialIdle(t *testing.T) {
	cc := testutils.NewBalancerClientConn(t)
	wtb := wtbBuilder.Build(cc, balancer.BuildOptions{})
	defer wtb.Close()

	config, err := wtbParser.ParseConfig([]byte(`
{
  "targets": {
    "cluster_1": {
      "weight":1,
      "childPolicy": [{"test-init-Idle-balancer": ""}]
    }
  }
}`))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config, and an address with hierarchy path ["cluster_1"].
	addrs := []resolver.Address{{Addr: testBackendAddrStrs[0], Attributes: nil}}
	if err := wtb.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Addresses: []resolver.Address{hierarchy.Set(addrs[0], []string{"cds:cluster_1"})}},
		BalancerConfig: config,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	// Verify that a subconn is created with the address, and the hierarchy path
	// in the address is cleared.
	for range addrs {
		sc := <-cc.NewSubConnCh
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Idle})
	}

	if state := <-cc.NewStateCh; state != connectivity.Idle {
		t.Fatalf("Received aggregated state: %v, want Idle", state)
	}
}

func testLoginBindingFail(t *testing.T, action, endpoint, badEndpoint, payload, badPayload string) {
	b := Query
	assert.Equal(t, "query", b.Name())

	obj := UserStructForMapType{}
	req := requestWithBody(action, endpoint, payload)
	if action == http.MethodPost {
		req.Header.Add("Content-Type", MIMEPOSTForm)
	}
	err := b.Bind(req, &obj)
	require.Error(t, err)
}

func testTimeBindingForForm(t *testing.T, m, p, bp, b, bb string) {
	f := Form
	assert.Equal(t, "form", f.Name())

	var obj FooBarStructForTimeType
	req := requestWithBody(m, p, b)
	if m == http.MethodPost {
		req.Header.Set("Content-Type", MIMEPOSTForm)
	}
	err := f.Bind(req, &obj)

	require.NoError(t, err)
	assert.Equal(t, int64(1510675200), obj.TimeFoo.Unix())
	assert.Equal(t, "Asia/Chongqing", obj.TimeFoo.Location().String())
	assert.Equal(t, int64(-62135596800), obj.TimeBar.Unix())
	assert.Equal(t, "UTC", obj.TimeBar.Location().String())
	assert.Equal(t, int64(1562400033000000123), obj.CreateTime.UnixNano())
	assert.Equal(t, int64(1562400033), obj.UnixTime.Unix())

	var newObj FooBarStructForTimeType
	req = requestWithBody(m, bp, bb)
	err = JSON.Bind(req, &newObj)
	require.Error(t, err)
}

func authHandler(w http.ResponseWriter, r *http.Request) {
	// make sure its post
	if r.Method != "POST" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, "No POST", r.Method)
		return
	}

	user := r.FormValue("user")
	pass := r.FormValue("pass")

	log.Printf("Authenticate: user[%s] pass[%s]\n", user, pass)

	// check values
	if user != "test" || pass != "known" {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(w, "Wrong info")
		return
	}

	tokenString, err := createToken(user)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "Sorry, error while Signing Token!")
		log.Printf("Token Signing error: %v\n", err)
		return
	}

	w.Header().Set("Content-Type", "application/jwt")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, tokenString)
}

