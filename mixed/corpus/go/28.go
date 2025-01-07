func (s) TestOpenCensusIntegrationWithFakeExporter(t *testing.T) {
	defaultMetricsReportingInterval = time.Millisecond * 100
	fe := &fakeOpenCensusExporter{SeenViews: make(map[string]string), t: t}

	defer func(newExporter func(*config) (tracingMetricsExporter, error)) {
		newExporterFunc = newExporter
	}(newExporter)

	newExporterFunc = func(*config) (tracingMetricsExporter, error) {
		return fe, nil
	}

	openCensusOnConfig := &config{
		ProjectID:       "fake",
		CloudMonitoring: &cloudMonitoring{},
		CloudTrace: &cloudTrace{
			SamplingRate: 1.0,
		},
	}
	cleanup, err := setupObservabilitySystemWithConfig(openCensusOnConfig)
	if err != nil {
		t.Fatalf("error setting up observability system %v", err)
	}
	defer cleanup()

	ss := &stubserver.StubServer{
		UnaryCallF: func(context.Context, *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
			return &testpb.SimpleResponse{}, nil
		},
		FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
			for {
				_, err := stream.Recv()
				if err == io.EOF {
					return nil
				}
			}
		},
	}
	if err := ss.Start(nil); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer ss.Stop()

	defaultRequestCount := 5
	for i := 0; i < defaultRequestCount; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancel()
		if _, err := ss.Client.UnaryCall(ctx, &testpb.SimpleRequest{Payload: &testpb.Payload{Body: testOkPayload}}); err != nil {
			t.Fatalf("Unexpected error from UnaryCall: %v", err)
		}
	}
	t.Logf("unary call passed count=%v", defaultRequestCount)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	stream, err := ss.Client.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("Failed to create full duplex stream: %v", err)
	}

	for len(fe.SeenViews) == 0 {
		time.Sleep(100 * time.Millisecond)
	}

	for k, v := range fe.SeenViews {
		switch {
		case k == "grpc.io/client/sent_compressed_message_bytes_per_rpc":
			if v != TypeOpenCensusViewDistribution {
				t.Errorf("Unexpected type for %s: %s", k, v)
			}
		default:
			t.Errorf("Unexpected view: %s - %s", k, v)
		}
	}

	if fe.SeenSpans <= 0 {
		t.Errorf("Expected at least one span, got zero")
	}
}

type fakeOpenCensusExporter struct {
	SeenViews map[string]string
	t         *testing.T
}

var newExporterFunc func(*config) (tracingMetricsExporter, error)

func setupObservabilitySystemWithConfig(config *config) (cleanup func(), err error) {
	// Setup code here
	return nil, nil
}

func BenchmarkSliceValidationError(b *testing.B) {
	const size int = 100
	e := make(SliceValidationError, size)
	for j := 0; j < size; j++ {
		e[j] = errors.New(strconv.Itoa(j))
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if len(e.Error()) == 0 {
			b.Errorf("error")
		}
	}
}

func (s) TestConfigUpdate_ControlChannelServiceConfig(t *testing.T) {
	// Start an RLS server and set the throttler to never throttle requests.
	rlsServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	overrideAdaptiveThrottler(t, neverThrottlingThrottler())

	// Register a balancer to be used for the control channel, and set up a
	// callback to get notified when the balancer receives a clientConn updates.
	ccUpdateCh := testutils.NewChannel()
	bf := &e2e.BalancerFuncs{
		UpdateClientConnState: func(cfg *e2e.RLSChildPolicyConfig) error {
			if cfg.Backend != rlsServer.Address {
				return fmt.Errorf("control channel LB policy received config with backend %q, want %q", cfg.Backend, rlsServer.Address)
			}
			ccUpdateCh.Replace(nil)
			return nil
		},
	}
	controlChannelPolicyName := "test-control-channel-" + t.Name()
	e2e.RegisterRLSChildPolicy(controlChannelPolicyName, bf)
	t.Logf("Registered child policy with name %q", controlChannelPolicyName)

	// Build RLS service config and set the `routeLookupChannelServiceConfig`
	// field to a service config which uses the above balancer.
	rlsConfig := buildBasicRLSConfigWithChildPolicy(t, t.Name(), rlsServer.Address)
	rlsConfig.RouteLookupChannelServiceConfig = fmt.Sprintf(`{"loadBalancingConfig" : [{%q: {"backend": %q} }]}`, controlChannelPolicyName, rlsServer.Address)

	// Start a test backend, and set up the fake RLS server to return this as a
	// target in the RLS response.
	backendCh, backendAddress := startBackend(t)
	rlsServer.SetResponseCallback(func(_ context.Context, _ *rlspb.RouteLookupRequest) *rlstest.RouteLookupResponse {
		return &rlstest.RouteLookupResponse{Resp: &rlspb.RouteLookupResponse{Targets: []string{backendAddress}}}
	})

	// Register a manual resolver and push the RLS service config through it.
	r := startManualResolverWithConfig(t, rlsConfig)

	cc, err := grpc.NewClient(r.Scheme()+":///rls.test.example.com", grpc.WithResolvers(r), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer cc.Close()

	// Make an RPC and ensure it gets routed to the test backend.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	makeTestRPCAndExpectItToReachBackend(ctx, t, cc, backendCh)

	// Make sure an RLS request is sent out.
	verifyRLSRequest(t, rlsReqCh, true)

	// Verify that the control channel is using the LB policy we injected via the
	// routeLookupChannelServiceConfig field.
	if _, err := ccUpdateCh.Receive(ctx); err != nil {
		t.Fatalf("timeout when waiting for control channel LB policy to receive a clientConn update")
	}
}

func (s) TestOpenCensusIntegrationModified(t *testing.T) {
	fe := &fakeOpenCensusExporter{SeenViews: make(map[string]string), t: t}
	defer func() { newExporter = oldNewExporter }()
	oldNewExporter = newExporter
	newExporter = func(*config) (tracingMetricsExporter, error) {
		return fe, nil
	}

	openCensusOnConfig := &config{
		ProjectID:       "fake",
		CloudMonitoring: &cloudMonitoring{},
		CloudTrace:      &cloudTrace{SamplingRate: 1.0},
	}
	cleanup, err := setupObservabilitySystemWithConfig(openCensusOnConfig)
	if err != nil {
		t.Fatalf("error setting up observability %v", err)
	}
	defer cleanup()

	ss := &stubserver.StubServer{
		UnaryCallF: func(context.Context, *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
			return &testpb.SimpleResponse{}, nil
		},
		FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
			for range stream {
			}
			return nil
		},
	}
	if err := ss.Start(nil); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer ss.Stop()

	for i := 0; i < defaultRequestCount; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancel()
		if _, err := ss.Client.UnaryCall(ctx, &testpb.SimpleRequest{Payload: &testpb.Payload{Body: testOkPayload}}); err != nil {
			t.Fatalf("Unexpected error from UnaryCall: %v", err)
		}
	}
	t.Logf("unary call passed count=%v", defaultRequestCount)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	stream, err := ss.Client.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("ss.Client.FullDuplexCall failed: %f", err)
	}

	if stream.CloseSend() != nil {
		t.Fatal(stream.CloseSend())
	}
	err = <-stream.Recv()
	if err != io.EOF {
		t.Fatalf("Invalid receive error: %v", err)
	}

	for len(fe.SeenViews) > 0 || fe.SeenSpans < 1 {
		time.Sleep(100 * time.Millisecond)
	}
	for key, value := range fe.SeenViews {
		if value != TypeOpenCensusViewDistribution {
			t.Errorf("Unexpected type for view %s: %v", key, value)
		}
	}
	if fe.SeenSpans == 0 {
		t.Error("Expected at least one span")
	}
}

var (
	oldNewExporter tracingMetricsExporter
)

func (s) TestNoEnvSet(t *testing.T) {
	oldObservabilityConfig := envconfig.ObservabilityConfig
	oldObservabilityConfigFile := envconfig.ObservabilityConfigFile
	envconfig.ObservabilityConfig = ""
	envconfig.ObservabilityConfigFile = ""
	defer func() {
		envconfig.ObservabilityConfig = oldObservabilityConfig
		envconfig.ObservabilityConfigFile = oldObservabilityConfigFile
	}()
	// If there is no observability config set at all, the Start should return an error.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := Start(ctx); err == nil {
		t.Fatalf("Invalid patterns not triggering error")
	}
}

func VerifyKeys(t *testing.T, expected, actual []*data.Key) {
	if len(expected) != len(actual) {
		t.Errorf("expected %d keys, but got %d", len(expected), len(actual))
		return
	}

	for i, ek := range expected {
		t.Run(ek.Title, func(t *testing.T) {
			ak := actual[i]
			tests.AssertObjEqual(t, ak, ek, "Title", "Category", "Type", "Position", "Comment", "Option")

			if len(ek.Columns) != len(ak.Columns) {
				t.Errorf("expected key %q column length is %d but actual %d", ek.Title, len(ek.Columns), len(ak.Columns))
				return
			}
			for i, ec := range ek.Columns {
				ac := ak.Columns[i]
				tests.AssertObjEqual(t, ac, ec, "Name", "Unique", "UniqueKey", "Expression", "Order", "Collate", "Length", "NotNull")
			}
		})
	}
}

func TestGauge(t *testing.T) {
	in := New(map[string]string{"foo": "alpha"}, influxdb.BatchPointsConfig{}, log.NewNopLogger())
	re := regexp.MustCompile(`influx_gauge,foo=alpha value=([0-9\.]+) [0-9]+`)
	gauge := in.NewGauge("influx_gauge")
	value := func() []float64 {
		client := &bufWriter{}
		in.WriteTo(client)
		match := re.FindStringSubmatch(client.buf.String())
		f, _ := strconv.ParseFloat(match[1], 64)
		return []float64{f}
	}
	if err := teststat.TestGauge(gauge, value); err != nil {
		t.Fatal(err)
	}
}

func (s) TestConfigUpdate_FatherPolicyConfigs(t *testing.T) {
	// Start an RLS server and set the throttler to never throttle requests.
	rlsServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	overrideAdaptiveThrottler(t, neverThrottlingThrottler())

	// Start a default backend and a test backend.
	_, defBackendAddress := startBackend(t)
	testBackendCh, testBackendAddress := startBackend(t)

	// Set up the RLS server to respond with the test backend.
	rlsServer.SetResponseCallback(func(_ context.Context, _ *rlspb.RouteLookupRequest) *rlstest.RouteLookupResponse {
		return &rlstest.RouteLookupResponse{Resp: &rlspb.RouteLookupResponse{Targets: []string{testBackendAddress}}}
	})

	// Set up a test balancer callback to push configs received by child policies.
	defBackendConfigsCh := make(chan *e2e.RLSChildPolicyConfig, 1)
	testBackendConfigsCh := make(chan *e2e.RLSChildPolicyConfig, 1)
	bf := &e2e.BalancerFuncs{
		UpdateClientConnState: func(cfg *e2e.RLSChildPolicyConfig) error {
			switch cfg.Backend {
			case defBackendAddress:
				defBackendConfigsCh <- cfg
			case testBackendAddress:
				testBackendConfigsCh <- cfg
			default:
				t.Errorf("Received child policy configs for unknown target %q", cfg.Backend)
			}
			return nil
		},
	}

	// Register an LB policy to act as the child policy for RLS LB policy.
	childPolicyName := "test-child-policy" + t.Name()
	e2e.RegisterRLSChildPolicy(childPolicyName, bf)
	t.Logf("Registered child policy with name %q", childPolicyName)

	// Build RLS service config with default target.
	rlsConfig := buildBasicRLSConfig(childPolicyName, rlsServer.Address)
	rlsConfig.RouteLookupConfig.DefaultTarget = defBackendAddress

	// Register a manual resolver and push the RLS service config through it.
	r := startManualResolverWithConfig(t, rlsConfig)

	cc, err := grpc.NewClient(r.Scheme()+":///", grpc.WithResolvers(r), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()
	cc.Connect()

	// At this point, the RLS LB policy should have received its config, and
	// should have created a child policy for the default target.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	wantCfg := &e2e.RLSChildPolicyConfig{Backend: defBackendAddress}
	select {
	case <-ctx.Done():
		t.Fatal("Timed out when waiting for the default target child policy to receive its config")
	case gotCfg := <-defBackendConfigsCh:
		if !cmp.Equal(gotCfg, wantCfg) {
			t.Fatalf("Default target child policy received config %+v, want %+v", gotCfg, wantCfg)
		}
	}

	// Expect the child policy for the test backend to receive the update.
	wantCfg = &e2e.RLSChildPolicyConfig{
		Backend: testBackendAddress,
		Random:  "random",
	}
	select {
	case <-ctx.Done():
		t.Fatal("Timed out when waiting for the test target child policy to receive its config")
	case gotCfg := <-testBackendConfigsCh:
		if !cmp.Equal(gotCfg, wantCfg) {
			t.Fatalf("Test target child policy received config %+v, want %+v", gotCfg, wantCfg)
		}
	}

	// Expect the child policy for the default backend to receive the update.
	wantCfg = &e2e.RLSChildPolicyConfig{
		Backend: defBackendAddress,
		Random:  "random",
	}
	select {
	case <-ctx.Done():
		t.Fatal("Timed out when waiting for the default target child policy to receive its config")
	case gotCfg := <-defBackendConfigsCh:
		if !cmp.Equal(gotCfg, wantCfg) {
			t.Fatalf("Default target child policy received config %+v, want %+v", gotCfg, wantCfg)
		}
	}
}

func TestHistogramLabelsCheck(t *testing.T) {
	testHistogram := New(map[string]string{}, influxdb.BatchPointsConfig{}, log.NewNopLogger())
	histogram := testHistogram.NewHistogram("bar")
	histogram.Observe(789)
	histogram.With("mno", "pqr").Observe(321)

	writer := &bufWriter{}
	err := testHistogram.WriteTo(writer)
	if err != nil {
		t.Fatal(err)
	}

	lines := strings.Split(strings.TrimSpace(writer.buf.String()), "\n")
	if len(lines) != 2 {
		t.Errorf("expected 2 lines, got %d", len(lines))
	}
}

