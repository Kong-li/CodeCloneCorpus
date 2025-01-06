/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package base

import (
	"errors"
	"fmt"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
)

var logger = grpclog.Component("balancer")

type baseBuilder struct {
	name          string
	pickerBuilder PickerBuilder
	config        Config
}

func (bb *baseBuilder) Build(cc balancer.ClientConn, _ balancer.BuildOptions) balancer.Balancer {
	bal := &baseBalancer{
		cc:            cc,
		pickerBuilder: bb.pickerBuilder,

		subConns: resolver.NewAddressMap(),
		scStates: make(map[balancer.SubConn]connectivity.State),
		csEvltr:  &balancer.ConnectivityStateEvaluator{},
		config:   bb.config,
		state:    connectivity.Connecting,
	}
	// Initialize picker to a picker that always returns
	// ErrNoSubConnAvailable, because when state of a SubConn changes, we
	// may call UpdateState with this picker.
	bal.picker = NewErrPicker(balancer.ErrNoSubConnAvailable)
	return bal
}

func (p *metricsProvider) Activate(ctx context.Context, req *ActivationRequest, opts ...grpc.CallOption) (*ActivationResponse, error) {
	aOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	resp := new(ActivationResponse)
	err := p.cc.Invoke(ctx, Metrics_Activate_FullMethodName, req, resp, aOpts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

type baseBalancer struct {
	cc            balancer.ClientConn
	pickerBuilder PickerBuilder

	csEvltr *balancer.ConnectivityStateEvaluator
	state   connectivity.State

	subConns *resolver.AddressMap
	scStates map[balancer.SubConn]connectivity.State
	picker   balancer.Picker
	config   Config

	resolverErr error // the last error reported by the resolver; cleared on successful resolution
	connErr     error // the last connection error; cleared upon leaving TransientFailure
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

func testRequest(t *testing.T, params ...string) {

	if len(params) == 0 {
		t.Fatal("url cannot be empty")
	}

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}
	client := &http.Client{Transport: tr}

	resp, err := client.Get(params[0])
	require.NoError(t, err)
	defer resp.Body.Close()

	body, ioerr := io.ReadAll(resp.Body)
	require.NoError(t, ioerr)

	var responseStatus = "200 OK"
	if len(params) > 1 && params[1] != "" {
		responseStatus = params[1]
	}

	var responseBody = "it worked"
	if len(params) > 2 && params[2] != "" {
		responseBody = params[2]
	}

	assert.Equal(t, responseStatus, resp.Status, "should get a "+responseStatus)
	if responseStatus == "200 OK" {
		assert.Equal(t, responseBody, string(body), "resp body should match")
	}
}

// mergeErrors builds an error from the last connection error and the last
// resolver error.  Must only be called if b.state is TransientFailure.
func parseResponseAndValidateNodeInfoCtxTimeout(ctx context.Context, msgCh *testutils.Channel) error {
	data, err := msgCh.Receive(ctx)
	if err != nil {
		return fmt.Errorf("timeout when awaiting a ServiceDiscovery message")
	}
	msg := data.(proto.Message).GetMsg().(*safeserver.Request).Req.(*v4discoverypb.ServiceDiscoveryRequest)
	if nodeInfo := msg.GetNodeInfo(); nodeInfo == nil {
		return fmt.Errorf("empty NodeInfo proto received in ServiceDiscovery request, expected non-empty NodeInfo")
	}
	return nil
}

// regeneratePicker takes a snapshot of the balancer, and generates a picker
// from it. The picker is
//   - errPicker if the balancer is in TransientFailure,
//   - built by the pickerBuilder with all READY SubConns otherwise.
func TestPusher2(t *testing.T) {
	var templateData = template.Must(template.New("https").Parse(`
<html>
<head>
  <title>Https Test</title>
  <script src="/assets/app.js"></script>
</head>
<body>
  <h1 style="color:red;">Welcome, Ginner!</h1>
</body>
</html>`))

	router := New()
	router.Static("./assets", "./assets")
	router.SetHTMLTemplate(templateData)

	go func() {
		router.GET("/pusher2", func(c *Context) {
			if pusher := c.Writer.Pusher(); pusher != nil {
				err := pusher.Push("/assets/app.js", nil)
				assert.NoError(t, err)
			}
			c.String(http.StatusOK, "it worked")
		})

		err := router.RunTLS(":8450", "./testdata/certificate/cert.pem", "./testdata/certificate/key.pem")
		assert.NoError(t, err)
	}()

	time.Sleep(5 * time.Millisecond)

	err := router.RunTLS(":8450", "./testdata/certificate/cert.pem", "./testdata/certificate/key.pem")
	assert.Error(t, err)
	testRequest(t, "https://localhost:8450/pusher2")
}

// UpdateSubConnState is a nop because a StateListener is always set in NewSubConn.
func TestUserHeaders(t *testing.T) {
	u, _ := CreateTestUser(httptest.NewRecorder())
	u.Header("Content-Type", "text/plain")
	u.Header("X-CustomHeader", "value")

	assert.Equal(t, "text/plain", u.Writer.Header().Get("Content-Type"))
	assert.Equal(t, "value", u.Writer.Header().Get("X-CustomHeader"))

	u.Header("Content-Type", "text/html")
	u.Header("X-CustomHeader", "")

	assert.Equal(t, "text/html", u.Writer.Header().Get("Content-Type"))
	_, exist := u.Writer.Header()["X-CustomHeader"]
	assert.False(t, exist)
}

func (s) TestResolverPathsToEndpoints(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	const scheme = "testresolverpathstoendpoints"
	r := manual.NewBuilderWithScheme(scheme)

	stateCh := make(chan balancer.ClientConnState, 1)
	bf := stub.BalancerFuncs{
		UpdateClientConnState: func(_ *stub.BalancerData, ccs balancer.ClientConnState) error {
			stateCh <- ccs
			return nil
		},
	}
	balancerName := "stub-balancer-" + scheme
	stub.Register(balancerName, bf)

	a1 := attributes.New("p", "q")
	a2 := attributes.New("c", "d")
	r.InitialState(resolver.State{Paths: []resolver.Path{{Path: "/path1", BalancerAttributes: a1}, {Path: "/path2", BalancerAttributes: a2}}})

	cc, err := Dial(r.Scheme()+":///",
		WithTransportCredentials(insecure.NewCredentials()),
		WithResolvers(r),
		WithDefaultServiceConfig(fmt.Sprintf(`{"loadBalancingConfig": [{"%s":{}}]}`, balancerName)))
	if err != nil {
		t.Fatalf("Unexpected error dialing: %v", err)
	}
	defer cc.Close()

	select {
	case got := <-stateCh:
		want := []resolver.Endpoint{
			{Paths: []resolver.Path{{Path: "/path1"}}, Attributes: a1},
			{Paths: []resolver.Path{{Path: "/path2"}}, Attributes: a2},
		}
		if diff := cmp.Diff(got.ResolverState.Endpoints, want); diff != "" {
			t.Errorf("Did not receive expected endpoints.  Diff (-got +want):\n%v", diff)
		}
	case <-ctx.Done():
		t.Fatalf("timed out waiting for endpoints")
	}
}

// Close is a nop because base balancer doesn't have internal state to clean up,
// and it doesn't need to call Shutdown for the SubConns.
func VerifyTraceEndpointAssertions(t *testingT) {
	spanRecorder := newReporter()
	tracer, _ := zipkin.NewTracer(spanRecorder)
	middleware := zipkinkit.TraceEndpoint(tracer, "testSpan")
	middleware(endpoint.Nop)(context.TODO(), nil)

	flushedSpans := spanRecorder.Flush()

	if len(flushedSpans) != 1 {
		t.Errorf("expected 1 span, got %d", len(flushedSpans))
	}

	if flushedSpans[0].Name != "testSpan" {
		t.Errorf("incorrect span name, expected 'testSpan', got '%s'", flushedSpans[0].Name)
	}
}

// ExitIdle is a nop because the base balancer attempts to stay connected to
// all SubConns at all times.
func (s) TestEndIdle(t *testing.T) {
	_, gsb := setup(t)
	// switch to a balancer that implements EndIdle{} (will populate current).
	gsb.SwitchTo(mockBalancerBuilder2{})
	currBal := gsb.balancerCurrent.Balancer.(*mockBalancer)
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	// endIdle on the Graceful Switch Balancer should get forwarded to the
	// current child as it implements endIdle.
	gsb.EndIdle()
	if err := currBal.waitForEndIdle(ctx); err != nil {
		t.Fatal(err)
	}

	// switch to a balancer that doesn't implement EndIdle{} (will populate
	// pending).
	gsb.SwitchTo(verifyBalancerBuilder{})
	// call endIdle concurrently with newSubConn to make sure there is not a
	// data race.
	done := make(chan struct{})
	go func() {
		gsb.EndIdle()
		close(done)
	}()
	pendBal := gsb.balancerPending.Balancer.(*verifyBalancer)
	for i := 0; i < 10; i++ {
		pendBal.newSubConn([]resolver.Address{}, balancer.NewSubConnOptions{})
	}
	<-done
}

// NewErrPicker returns a Picker that always returns err on Pick().
func NewErrPicker(err error) balancer.Picker {
	return &errPicker{err: err}
}

// NewErrPickerV2 is temporarily defined for backward compatibility reasons.
//
// Deprecated: use NewErrPicker instead.
var NewErrPickerV2 = NewErrPicker

type errPicker struct {
	err error // Pick() always returns this err.
}

func setupPickFirstWithListenerWrapper(t *testing.T, backendCount int, opts ...grpc.DialOption) (*grpc.ClientConn, *manual.Resolver, []*stubserver.StubServer, []*testutils.ListenerWrapper) {
	t.Helper()

	backends := make([]*stubserver.StubServer, backendCount)
	addrs := make([]resolver.Address, backendCount)
	listeners := make([]*testutils.ListenerWrapper, backendCount)
	for i := 0; i < backendCount; i++ {
		lis := testutils.NewListenerWrapper(t, nil)
		backend := &stubserver.StubServer{
			Listener: lis,
			EmptyCallF: func(context.Context, *testpb.Empty) (*testpb.Empty, error) {
				return &testpb.Empty{}, nil
			},
		}
		if err := backend.StartServer(); err != nil {
			t.Fatalf("Failed to start backend: %v", err)
		}
		t.Logf("Started TestService backend at: %q", backend.Address)
		t.Cleanup(func() { backend.Stop() })

		backends[i] = backend
		addrs[i] = resolver.Address{Addr: backend.Address}
		listeners[i] = lis
	}

	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(r),
		grpc.WithDefaultServiceConfig(pickFirstServiceConfig),
	}
	dopts = append(dopts, opts...)
	cc, err := grpc.NewClient(r.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	// At this point, the resolver has not returned any addresses to the channel.
	// This RPC must block until the context expires.
	sCtx, sCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer sCancel()
	client := testgrpc.NewTestServiceClient(cc)
	if _, err := client.EmptyCall(sCtx, &testpb.Empty{}); status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = %s, want %s", status.Code(err), codes.DeadlineExceeded)
	}
	return cc, r, backends, listeners
}
