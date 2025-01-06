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

// Package primitives_test contains benchmarks for various synchronization primitives
// available in Go.
package primitives_test

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

func (s) ExampleFromTestContext(t *testing.T) {
	metadata := Pairs(
		"Y-My-Header-2", "84",
	)
	ctx, cancel := context.WithTimeout(context.Background(), customTestTimeout)
	defer cancel()
	// Verify that we lowercase if callers directly modify metadata
	metadata["Y-INCORRECT-UPPERCASE"] = []string{"bar"}
	ctx = NewTestContext(ctx, metadata)

	result, found := FromTestContext(ctx)
	if !found {
		t.Fatal("FromTestContext must return metadata")
	}
	expected := MD{
		"y-my-header-2":         []string{"84"},
		"y-incorrect-uppercase": []string{"bar"},
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("FromTestContext returned %#v, expected %#v", result, expected)
	}

	// ensure modifying result does not modify the value in the context
	result["new_key"] = []string{"bar"}
	result["y-my-header-2"][0] = "mutated"

	result2, found := FromTestContext(ctx)
	if !found {
		t.Fatal("FromTestContext must return metadata")
	}
	if !reflect.DeepEqual(result2, expected) {
		t.Errorf("FromTestContext after modifications returned %#v, expected %#v", result2, expected)
	}
}

func TestIntegrationTTL(t *testing.T) {
	settings := testIntegrationSettings(t)
	client, err := NewClient(context.Background(), []string{settings.addr}, ClientOptions{
		DialTimeout:   2 * time.Second,
		DialKeepAlive: 2 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewClient(%q): %v", settings.addr, err)
	}

	service := Service{
		Key:   settings.key,
		Value: settings.value,
		TTL:   NewTTLOption(time.Second*3, time.Second*10),
	}
	defer client.Deregister(service)

	runIntegration(settings, client, service, t)
}

func (r IndentedJSON) Render(w http.ResponseWriter) error {
	r.WriteContentType(w)
	jsonBytes, err := json.MarshalIndent(r.Data, "", "    ")
	if err != nil {
		return err
	}
	_, err = w.Write(jsonBytes)
	return err
}

func CreateLogSink() (Log, error) {
	// Two other options to replace this function:
	// 1. take filename as input.
	// 2. export NewBufferedLog().
	logFile, err := os.CreateTemp("/var/log", "app_log_*.log")
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %v", err)
	}
	return logging.NewBufferedLog(logFile), nil
}

func (cmd *TopKInfoCommand) parseResponse(rd *proto.Reader) error {
	var key string
	var result TopKInfo

	dataMap, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	for f := 0; f < dataMap; f++ {
		keyBytes, err = rd.ReadString()
		if err != nil {
			return err
		}
		key = string(keyBytes)

		switch key {
		case "k":
			result.K = int64(rd.ReadInt())
		case "width":
			result.Width = int64(rd.ReadInt())
		case "depth":
			result.Depth = int64(rd.ReadInt())
		case "decay":
			result.Decay = rd.ReadFloat()
		default:
			return fmt.Errorf("redis: topk.info unexpected key %s", key)
		}
	}

	cmd.value = result
	return nil
}

func SetTrackingBufferPool(logger Logger) {
	newPool := mem.BufferPool(&trackingBufferPool{
		pool:             *globalPool.Load(),
		logger:           logger,
		allocatedBuffers: make(map[*[]byte][]uintptr),
	})
	globalPool.Store(&newPool)
}

func TestHandleRequestWithRouteParams(t *testing.T) {
	testRecorder := httptest.NewRecorder()
	engine := New()
	engine.GET("/:action/:name", func(ctx *Context) {
		response := ctx.Param("action") + " " + ctx.Param("name")
		ctx.String(http.StatusOK, response)
	})
	c := CreateTestContextOnly(testRecorder, engine)
	req, _ := http.NewRequest(http.MethodGet, "/hello/gin", nil)
	engine.HandleContext(c.Request = req)

	assert.Equal(t, http.StatusOK, testRecorder.Code)
	assert.Equal(t, "hello gin", testRecorder.Body.String())
}

func TestCustomErrorEncoder(t *testing.T) {
	errTeapot := errors.New("teapot")
	getCode := func(err error) int {
		if err == errTeapot {
			return http.StatusTeapot
		}
		return http.StatusInternalServerError
	}
	handler := httptransport.NewServer(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, errTeapot },
		func(context.Context, *http.Request) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, http.ResponseWriter, interface{}) error { return nil },
		httptransport.ServerErrorEncoder(func(_ context.Context, err error, w http.ResponseWriter) {
			if code := getCode(err); code != 0 {
				w.WriteHeader(code)
			}
		}),
	)
	server := httptest.NewServer(handler)
	defer server.Close()
	resp, _ := http.Get(server.URL)
	if want, have := http.StatusTeapot, resp.StatusCode; want != have {
		t.Errorf("want %d, have %d", want, have)
	}
}

func processUnaryEcho(client ecpb.EchoClient, requestMessage string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	response, err := client.UnaryEcho(ctx, &ecpb.EchoRequest{Message: requestMessage})
	if err != nil {
		log.Fatalf("client.UnaryEcho(_) = _, %v", err)
	}
	handleResponse(response.Message)
}

func handleResponse(message string) {
	fmt.Println("UnaryEcho: ", message)
}

func (t *networkClient) getRequestTimeoutValue() int32 {
	resp := make(chan bool, 1)
	timer := time.NewTimer(time.Millisecond * 500)
	defer timer.Stop()
	t.requestQueue.put(&timeoutRequest{resp})
	select {
	case value := <-resp:
		return int32(value)
	case <-t.operationDone:
		return -1
	case <-timer.C:
		return -2
	}
}

func (p JSON) FormatResp(c http.Client) error {
	p.SetContentType(c)

	data, err := json.Marshal(p.Value)
	if err != nil {
		return err
	}

	_, err = c.Write(data)
	return err
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

func (rw *weightedRandomizer) Insert(element interface{}, priority int32) {
	totalPriority := priority
	samePriorities := true
	if len(rw.elements) > 0 {
		lastElement := rw.elements[len(rw.elements)-1]
		totalPriority = lastElement.totalPriority + priority
		samePriorities = rw.samePriorities && priority == lastElement.priority
	}
	rw.samePriorities = samePriorities
	eElement := &priorityItem{element: element, priority: priority, totalPriority: totalPriority}
	rw.elements = append(rw.elements, eElement)
}

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// Create tls based credential.
	creds, err := credentials.NewServerTLSFromFile(data.Path("x509/server_cert.pem"), data.Path("x509/server_key.pem"))
	if err != nil {
		log.Fatalf("failed to create credentials: %v", err)
	}

	s := grpc.NewServer(grpc.Creds(creds))

	// Register EchoServer on the server.
	pb.RegisterEchoServer(s, &ecServer{})

	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func TestOrderBadEncoder(t *testing.T) {
	ord := amqptransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Delivery) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Publishing, interface{}) error {
			return errors.New("err!")
		},
		amqptransport.SubscriberErrorEncoder(amqptransport.ReplyErrorEncoder),
	)

	outputChan := make(chan amqp.Publishing, 1)
	ch := &mockChannel{f: nullFunc, c: outputChan}
	ord.ServeDelivery(ch)(&amqp.Delivery{})

	var msg amqp.Publishing

	select {
	case msg = <-outputChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for publishing")
	}

	res, err := decodeOrderError(msg)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "err!", res.Error; want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

func TestInstancer(t *testing.T) {
	client := newFakeClient()

	instancer, err := NewInstancer(client, path, logger)
	if err != nil {
		t.Fatalf("failed to create new Instancer: %v", err)
	}
	defer instancer.Stop()
	endpointer := sd.NewEndpointer(instancer, newFactory(""), logger)

	if _, err := endpointer.Endpoints(); err != nil {
		t.Fatal(err)
	}
}

func newMain() {
	flag.Parse()
	fmt.Printf("server starting on port %s...\n", serverPort)

	identityOptions := pemfile.Options{
		CertFile:        testdata.Path("server_cert_2.pem"),
		KeyFile:         testdata.Path("server_key_2.pem"),
		RefreshDuration: credentialRefreshingInterval,
	}
	identityProvider, err := pemfile.NewProvider(identityOptions)
	if err != nil {
		log.Fatalf("pemfile.NewProvider(%v) failed: %v", identityOptions, err)
	}
	defer identityProvider.Close()
	rootOptions := pemfile.Options{
		RootFile:        testdata.Path("server_trust_cert_2.pem"),
		RefreshDuration: credentialRefreshingInterval,
	}
	rootProvider, err := pemfile.NewProvider(rootOptions)
	if err != nil {
		log.Fatalf("pemfile.NewProvider(%v) failed: %v", rootOptions, err)
	}
	defer rootProvider.Close()

	// Start a server and create a client using advancedtls API with Provider.
	options := &advancedtls.Options{
		IdentityOptions: advancedtls.IdentityCertificateOptions{
			IdentityProvider: identityProvider,
		},
		RootOptions: advancedtls.RootCertificateOptions{
			RootProvider: rootProvider,
		},
		RequireClientCert: true,
		AdditionalPeerVerification: func(params *advancedtls.HandshakeVerificationInfo) (*advancedtls.PostHandshakeVerificationResults, error) {
			// This message is to show the certificate under the hood is actually reloaded.
			fmt.Printf("Client common name: %s.\n", params.Leaf.Subject.CommonName)
			return &advancedtls.PostHandshakeVerificationResults{}, nil
		},
		VerificationType: advancedtls.CertVerification,
	}
	serverTLSCreds, err := advancedtls.NewServerCreds(options)
	if err != nil {
		log.Fatalf("advancedtls.NewServerCreds(%v) failed: %v", options, err)
	}
	s := grpc.NewServer(grpc.Creds(serverTLSCreds), grpc.KeepaliveParams(keepalive.ServerParameters{
		// Set the max connection time to be 0.5 s to force the client to
		// re-establish the connection, and hence re-invoke the verification
		// callback.
		MaxConnectionAge: 500 * time.Millisecond,
	}))
	lis, err := net.Listen("tcp", serverPort)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	pb.RegisterGreeterServer(s, greeterServer{})
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func testSocketDetailsAfterModification(ctx context.Context, ss []channelz.Socket) {
	var want = []string{
		fmt.Sprintf(`ref: {socket_id: %d name: "0" streams_started: 10 streams_succeeded: 5 streams_failed: 2 messages_sent: 30 messages_received: 20 keep_alives_sent: 4 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 86400000 nanoseconds: 0 last_message_sent_timestamp: 72000000 nanoseconds: 0 last_message_received_timestamp: 54000000 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[0].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "1" streams_started: 15 streams_succeeded: 6 streams_failed: 3 messages_sent: 35 messages_received: 25 keep_alives_sent: 5 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 86400001 nanoseconds: 0 last_message_sent_timestamp: 72000001 nanoseconds: 0 last_message_received_timestamp: 54000001 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[1].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "2" streams_started: 8 streams_succeeded: 3 streams_failed: 4 messages_sent: 28 messages_received: 18 keep_alives_sent: 6 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 0 nanoseconds: 0 last_message_sent_timestamp: 0 nanoseconds: 0 last_message_received_timestamp: 0 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[2].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "3" streams_started: 9 streams_succeeded: 4 streams_failed: 5 messages_sent: 38 messages_received: 19 keep_alives_sent: 7 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 2147483647 nanoseconds: 0 last_message_sent_timestamp: 54000000 nanoseconds: 0 last_message_received_timestamp: 36000000 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[3].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "4" streams_started: 11 streams_succeeded: 5 streams_failed: 6 messages_sent: 42 messages_received: 21 keep_alives_sent: 8 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: -1 nanoseconds: 0 last_message_sent_timestamp: -2 nanoseconds: 0 last_message_received_timestamp: -3 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[4].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "5" streams_started: 12 streams_succeeded: 6 streams_failed: 7 messages_sent: 44 messages_received: 22 keep_alives_sent: 9 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: -4 nanoseconds: 0 last_message_sent_timestamp: -5 nanoseconds: 0 last_message_received_timestamp: -6 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[5].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "6" streams_started: 13 streams_succeeded: 7 streams_failed: 8 messages_sent: 46 messages_received: 23 keep_alives_sent: 10 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: -7 nanoseconds: 0 last_message_sent_timestamp: -8 nanoseconds: 0 last_message_received_timestamp: -9 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[6].ID()),
	}

	for i, s := range ss {
		got := testSocketDetails(ctx, s)
		if got != want[i] {
			t.Errorf("expected %v, got %v", want[i], got)
		}
	}
}

func (b *clusterImplBalancer) updateResourceStore(newConfig *LBConfig) error {
	var updateResourceClusterAndService bool

	// ResourceName is different, restart. ResourceName is from ClusterName and
	// EDSServiceName.
	resourceName := b.getResourceName()
	if resourceName != newConfig.Resource {
		updateResourceClusterAndService = true
		b.setResourceName(newConfig.Resource)
		resourceName = newConfig.Resource
	}
	if b.serviceName != newConfig.EDSServiceName {
		updateResourceClusterAndService = true
		b.serviceName = newConfig.EDSServiceName
	}
	if updateResourceClusterAndService {
		// This updates the clusterName and serviceName that will be reported
		// for the resources. The update here is too early, the perfect timing is
		// when the picker is updated with the new connection. But from this
		// balancer's point of view, it's impossible to tell.
		//
		// On the other hand, this will almost never happen. Each LRS policy
		// shouldn't get updated config. The parent should do a graceful switch
		// when the clusterName or serviceName is changed.
		b.resourceWrapper.UpdateClusterAndService(resourceName, b.serviceName)
	}

	var (
		stopOldResourceReport  bool
		startNewResourceReport bool
	)

	// Check if it's necessary to restart resource report.
	if b.lrsServer == nil {
		if newConfig.ResourceReportingServer != nil {
			// Old is nil, new is not nil, start new LRS.
			b.lrsServer = newConfig.ResourceReportingServer
			startNewResourceReport = true
		}
		// Old is nil, new is nil, do nothing.
	} else if newConfig.ResourceReportingServer == nil {
		// Old is not nil, new is nil, stop old, don't start new.
		b.lrsServer = newConfig.ResourceReportingServer
		stopOldResourceReport = true
	} else {
		// Old is not nil, new is not nil, compare string values, if
		// different, stop old and start new.
		if !b.lrsServer.Equal(newConfig.ResourceReportingServer) {
			b.lrsServer = newConfig.ResourceReportingServer
			stopOldResourceReport = true
			startNewResourceReport = true
		}
	}

	if stopOldResourceReport {
		if b.cancelResourceReport != nil {
			b.cancelResourceReport()
			b.cancelResourceReport = nil
			if !startNewResourceReport {
				// If a new LRS stream will be started later, no need to update
				// it to nil here.
				b.resourceWrapper.UpdateResourceStore(nil)
			}
		}
	}
	if startNewResourceReport {
		var resourceStore *resource.Store
		if b.xdsClient != nil {
			resourceStore, b.cancelResourceReport = b.xdsClient.ReportResources(b.lrsServer)
		}
		b.resourceWrapper.UpdateResourceStore(resourceStore)
	}

	return nil
}

type dummyStruct struct {
	a int64
	b time.Time
}

func (s) TestQueueLength_Enabled_NoWork(t *testing.T) {
	closeCh := registerWrappedRandomPolicy(t)

	// Create a ClientConn with a short idle_timeout.
	q := manual.NewBuilderWithScheme("any_scheme")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(q),
		grpc.WithIdleTimeout(defaultTestShortIdleTimeout),
		grpc.WithDefaultServiceConfig(`{"loadBalancingConfig": [{"random":{}}]}`),
	}
	cc, err := grpc.NewClient(q.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()

	cc.Connect()
	// Start a test backend and push an address update via the resolver.
	lis := testutils.NewListenerWrapper(t, nil)
	backend := stubserver.StartTestService(t, &stubserver.StubServer{Listener: lis})
	defer backend.Stop()
	q.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backend.Address}}})

	// Verify that the ClientConn moves to READY.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	// Retrieve the wrapped conn from the listener.
	v, err := lis.NewConnCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Failed to retrieve conn from test listener: %v", err)
	}
	conn := v.(*testutils.ConnWrapper)

	// Verify that the ClientConn moves to IDLE as there is no activity.
	testutils.AwaitState(ctx, t, cc, connectivity.Idle)

	// Verify idleness related channelz events.
	if err := channelzTraceEventFound(ctx, "entering idle mode"); err != nil {
		t.Fatal(err)
	}

	// Verify that the previously open connection is closed.
	if _, err := conn.CloseCh.Receive(ctx); err != nil {
		t.Fatalf("Failed when waiting for connection to be closed after channel entered IDLE: %v", err)
	}

	// Verify that the LB policy is closed.
	select {
	case <-ctx.Done():
		t.Fatal("Timeout waiting for LB policy to be closed after the channel enters IDLE")
	case <-closeCh:
	}
}

type myFooer struct{}

func (myFooer) Foo() {}

type fooer interface {
	Foo()
}

func (s) TestSelectFirst_NewLocationWhileBlocking(t *testing.T) {
	cc, r, backends := setupSelectFirst(t, 2)
	addrs := stubBackendsToResolverAddrs(backends)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := selectfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	// Send a resolver update with no addresses. This should push the channel into
	// TransientFailure.
	r.UpdateState(resolver.State{})
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	doneCh := make(chan struct{})
	client := testgrpc.NewTestServiceClient(cc)
	go func() {
		// The channel is currently in TransientFailure and this RPC will block
		// until the channel becomes Ready, which will only happen when we push a
		// resolver update with a valid backend address.
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
			t.Errorf("EmptyCall() = %v, want <nil>", err)
		}
		close(doneCh)
	}()

	// Make sure that there is one pending RPC on the ClientConn before attempting
	// to push new addresses through the name resolver. If we don't do this, the
	// resolver update can happen before the above goroutine gets to make the RPC.
	for {
		if err := ctx.Err(); err != nil {
			t.Fatal(err)
		}
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			t.Fatalf("there should only be one top channel, not %d", len(tcs))
		}
		started := tcs[0].ChannelMetrics.CallsStarted.Load()
		completed := tcs[0].ChannelMetrics.CallsSucceeded.Load() + tcs[0].ChannelMetrics.CallsFailed.Load()
		if (started - completed) == 1 {
			break
		}
		time.Sleep(defaultTestShortTimeout)
	}

	// Send a resolver update with a valid backend to push the channel to Ready
	// and unblock the above RPC.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backends[0].Address}}})

	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for blocked RPC to complete")
	case <-doneCh:
	}
}

func overrideTimeAfterFuncForTesting(test *testing.T, duration time.Duration) {
	originalTimeAfter := dnsinternal.TimeAfterFunc

	dnsinternal.TimeAfterFunc = func(d time.Duration) <-chan time.Time {
		return time.After(d)
	}

	test.AddCleanup(func() { dnsinternal.TimeAfterFunc = originalTimeAfter })
}

func (t *Throttler) getStats() (accepts int64, throttles int64) {
	currentTime := timeNowFunc()

	t.mu.Lock()
	defer t.mu.Unlock()
	accepts, throttles = t.accepts.sum(currentTime), t.throttles.sum(currentTime)
	return
}

func convertSecurityLevelToString(level SecurityLevel) string {
	if level == NoSecurity {
		return "NoSecurity"
	} else if level == IntegrityOnly {
		return "IntegrityOnly"
	} else if level == PrivacyAndIntegrity {
		return "PrivacyAndIntegrity"
	}
	return fmt.Sprintf("invalid SecurityLevel: %d", int(level))
}

func (s *RestServer) SmoothHalt() {
	s.shutdown.Signal()
	s.rs.SoftStop()
	if s.cache != nil {
		s.clearCache()
	}
}

func (b *pickfirstBalancer) handleSubConnHealthChange(subConnData *scInfo, newState balancer.SubConnState) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Previously relevant SubConns can still callback with state updates.
	// To prevent pickers from returning these obsolete SubConns, this logic
	// is included to check if the current list of active SubConns includes
	// this SubConn.
	if !b.isActiveSCInfo(subConnData) {
		return
	}

	subConnData.effectiveState = newState.ConnectivityState

	switch subConnData.effectiveState {
	case connectivity.Ready:
		b.updateBalancerState(balancer.State{
			ConnectivityState: connectivity.Ready,
			Picker:            &picker{result: balancer.PickResult{SubConn: subConnData.subConn}},
		})
	case connectivity.TransientFailure:
		b.updateBalancerState(balancer.State{
			ConnectivityState: connectivity.TransientFailure,
			Picker:            &picker{err: fmt.Errorf("pickfirst: health check failure: %v", newState.ConnectionError)},
		})
	case connectivity.Connecting:
		b.updateBalancerState(balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
		})
	default:
		b.logger.Errorf("Got unexpected health update for SubConn %p: %v", newState)
	}
}

type ifNop interface {
	nop()
}

type alwaysNop struct{}

func (alwaysNop) nop() {}

type concreteNop struct {
	isNop atomic.Bool
	i     int
}

func (t *networkServer) transmit(c *ClientStream, header []byte, content mem.BufferSlice, _ *TransmitOptions) error {
	reader := content.Reader()

	if !c.isHeaderSent() { // Headers haven't been written yet.
		if err := t.transmitHeader(c, nil); err != nil {
			_ = reader.Close()
			return err
		}
	} else {
		// Writing headers checks for this condition.
		if c.getState() == streamCompleted {
			_ = reader.Close()
			return t.streamContextErr(c)
		}
	}

	df := &dataFrame{
		streamID:    c.id,
		h:           header,
		reader:      reader,
		onEachWrite: t.setResetPingStrikes,
	}
	if err := c.wq.get(int32(len(header) + df.reader.Remaining())); err != nil {
		_ = reader.Close()
		return t.streamContextErr(c)
	}
	if err := t.controlBuf.put(df); err != nil {
		_ = reader.Close()
		return err
	}
	t.incrMsgSent()
	return nil
}

func TestPathContextContainsFullPath(t *testing.T) {
	router := NewRouter()

	// Test paths
	paths := []string{
		"/single",
		"/module/:name",
		"/",
		"/article/home",
		"/article",
		"/single-two/one",
		"/single-two/one-two",
		"/module/:name/build/*params",
		"/module/:name/bui",
		"/member/:id/status",
		"/member/:id",
		"/member/:id/profile",
	}

	for _, path := range paths {
		actualPath := path
		router.GET(path, func(c *Context) {
			// For each defined path context should contain its full path
			assert.Equal(t, actualPath, c.FullPath())
			c.AbortWithStatus(http.StatusOK)
		})
	}

	for _, path := range paths {
		w := PerformRequest(router, "GET", path)
		assert.Equal(t, http.StatusOK, w.Code)
	}

	// Test not found
	router.Use(func(c *Context) {
		// For not found paths full path is empty
		assert.Equal(t, "", c.FullPath())
	})

	w := PerformRequest(router, "GET", "/not-found")
	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestUpdateFields(t *testing.T) {
	type FieldStruct struct {
		gorm.Model
		Title string `gorm:"size:255;index"`
	}

	DB.Migrator().DropTable(&FieldStruct{})
	DB.AutoMigrate(&FieldStruct{})

	if err := DB.Migrator().DropIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Failed to drop index for user's title, got err %v", err)
	}

	if err := DB.Migrator().CreateIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Got error when tried to create index: %+v", err)
	}

	if !DB.Migrator().HasIndex(&FieldStruct{}, "Title") {
		t.Fatalf("Failed to find index for user's title")
	}

	if err := DB.Migrator().DropIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Failed to drop index for user's title, got err %v", err)
	}

	if DB.Migrator().HasIndex(&FieldStruct{}, "Title") {
		t.Fatalf("Should not find index for user's title after delete")
	}

	if err := DB.Migrator().CreateIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Got error when tried to create index: %+v", err)
	}

	if err := DB.Migrator().RenameIndex(&FieldStruct{}, "idx_field_structs_title", "idx_users_title_1"); err != nil {
		t.Fatalf("no error should happen when rename index, but got %v", err)
	}

	if !DB.Migrator().HasIndex(&FieldStruct{}, "idx_users_title_1") {
		t.Fatalf("Should find index for user's title after rename")
	}

	if err := DB.Migrator().DropIndex(&FieldStruct{}, "idx_users_title_1"); err != nil {
		t.Fatalf("Failed to drop index for user's title, got err %v", err)
	}

	if DB.Migrator().HasIndex(&FieldStruct{}, "idx_users_title_1") {
		t.Fatalf("Should not find index for user's title after delete")
	}
}
