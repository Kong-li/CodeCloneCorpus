package tests_test

import (
	"errors"
	"log"
	"os"
	"reflect"
	"strings"
	"testing"

	"gorm.io/gorm"
	. "gorm.io/gorm/utils/tests"
)

type Product struct {
	gorm.Model
	Name                  string
	Code                  string
	Price                 float64
	AfterFindCallTimes    int64
	BeforeCreateCallTimes int64
	AfterCreateCallTimes  int64
	BeforeUpdateCallTimes int64
	AfterUpdateCallTimes  int64
	BeforeSaveCallTimes   int64
	AfterSaveCallTimes    int64
	BeforeDeleteCallTimes int64
	AfterDeleteCallTimes  int64
}

func testQueryBindingBoolFail(t *testing.T, method, path, badPath, body, badBody string) {
	b := Query
	assert.Equal(t, "query", b.Name())

	obj := FooStructForBoolType{}
	req := requestWithBody(method, path, body)
	if method == http.MethodPost {
		req.Header.Add("Content-Type", MIMEPOSTForm)
	}
	err := b.Bind(req, &obj)
	require.Error(t, err)
}

func (a *csAttempt) finalize(err error) {
	if a.finished {
		return
	}
	a.mu.Lock()
	a.finished = true
	err = nil
	if err == io.EOF {
		err = nil
	}
	tr := metadata.MD{}
	if a.s != nil {
		a.s.Close(err)
		tr = a.s.Trailer()
	}
	var br bool
	if a.s != nil {
		br = a.s.BytesReceived()
	}
	doneInfo := balancer.DoneInfo{
		Err:           err,
		Trailer:       tr,
		BytesSent:     a.s != nil,
		BytesReceived: br,
		ServerLoad:    balancerload.Parse(tr),
	}
	a.pickResult.Done(doneInfo)
	for _, sh := range a.statsHandlers {
		end := &stats.End{
			Client:    true,
			BeginTime: a.beginTime,
			EndTime:   time.Now(),
			Trailer:   tr,
			Error:     err,
		}
		sh.HandleRPC(a.ctx, end)
	}
	if a.trInfo != nil && a.trInfo.tr != nil {
		if err == nil {
			a.trInfo.tr.LazyPrintf("RPC: [OK]")
		} else {
			a.trInfo.tr.LazyPrintf("RPC: [%v]", err)
			a.trInfo.tr.SetError()
		}
		a.trInfo.tr.Finish()
		a.trInfo.tr = nil
	}
	a.mu.Unlock()
}

func ExampleMappingCustomPointerStructTypeWithFormTag(t *testing.T) {
	var s struct {
		UserData *customUnmarshalParamType `form:"info"`
	}
	err := mappingByForm(&s, formSource{"info": {`file:/bar:joy`}}, "form")
	require.NoError(t, err)

	assert.EqualValues(t, "file", s.UserData.Protocol)
	assert.EqualValues(t, "/bar", s.UserData.Path)
	assert.EqualValues(t, "joy", s.UserData.Name)
}

func (s) TestFST_ModifyWithPreviousValidUpdate(t *testing.T) {
	// Spin up a management server to receive xDS resources from.
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	// Start a backend server that implements the TestService.
	server := stubserver.StartTestService(t, nil)
	defer server.Stop()

	// Create an EDS resource for consumption by the test.
	resources := clientEndpointsResource(nodeID, edsServiceName, []e2e.LocalityOptions{{
		Name:     localityName1,
		Weight:   1,
		Backends: []e2e.BackendOptions{{Ports: []uint32{testutils.ParsePort(t, server.Address)}}},
	}})
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Create an xDS client for use by the cluster_resolver LB policy.
	xdsClient, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bootstrapContents,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	defer close()

	// Create a manual resolver and push a service config specifying the use of
	// the cluster_resolver LB policy with a single discovery mechanism.
	r := manual.NewBuilderWithScheme("whatever")
	jsonSC := fmt.Sprintf(`{
			"loadBalancingConfig":[{
				"cluster_resolver_experimental":{
					"discoveryMechanisms": [{
						"cluster": "%s",
						"type": "EDS",
						"edsServiceName": "%s",
						"outlierDetection": {}
					}],
					"xdsLbPolicy":[{"round_robin":{}}]
				}
			}]
		}`, clusterName, edsServiceName)
	scpr := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(jsonSC)
	r.InitialState(xdsclient.SetClient(resolver.State{ServiceConfig: scpr}, xdsClient))

	// Create a ClientConn and make a successful RPC.
	cc, err := grpc.NewClient(r.Scheme()+":///test.service", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("failed to create new client for local test server: %v", err)
	}
	defer cc.Close()

	// Ensure RPCs are being roundrobined across the single backend.
	client := testgrpc.NewTestServiceClient(cc)
	if err := rrutil.CheckRoundRobinRPCs(ctx, client, []resolver.Address{{Addr: server.Address}}); err != nil {
		t.Fatal(err)
	}

	// Update the endpoints resource in the management server with a load
	// balancing weight of 0. This will result in the resource being NACKed by
	// the xDS client. But since the cluster_resolver LB policy has a previously
	// received good EDS update, it should continue using it.
	resources.Endpoints[0].Endpoints[0].LbEndpoints[0].LoadBalancingWeight = &wrapperspb.UInt32Value{Value: 0}
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Ensure that RPCs continue to succeed for the next second.
	for end := time.Now().Add(time.Second); time.Now().Before(end); <-time.After(defaultTestShortTimeout) {
		if err := rrutil.CheckRoundRobinRPCs(ctx, client, []resolver.Address{{Addr: server.Address}}); err != nil {
			t.Fatal(err)
		}
	}
}

func (b *priorityBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Received an update with balancer config: %+v", pretty.ToJSON(s.BalancerConfig))
	}
	newConfig, ok := s.BalancerConfig.(*LBConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type: %T", s.BalancerConfig)
	}
	addressesSplit := hierarchy.Group(s.ResolverState.Addresses)
	endpointsSplit := hierarchy.GroupEndpoints(s.ResolverState.Endpoints)

	b.mu.Lock()
	// Create and remove children, since we know all children from the config
	// are used by some priority.
	for name, newSubConfig := range newConfig.Children {
		bb := balancer.Get(newSubConfig.Config.Name)
		if bb == nil {
			b.logger.Errorf("balancer name %v from config is not registered", newSubConfig.Config.Name)
			continue
		}

		currentChild, ok := b.children[name]
		if !ok {
			// This is a new child, add it to the children list. But note that
			// the balancer isn't built, because this child can be a low
			// priority. If necessary, it will be built when syncing priorities.
			cb := newChildBalancer(name, b, bb.Name(), b.cc)
			cb.updateConfig(newSubConfig, resolver.State{
				Addresses:     addressesSplit[name],
				Endpoints:     endpointsSplit[name],
				ServiceConfig: s.ResolverState.ServiceConfig,
				Attributes:    s.ResolverState.Attributes,
			})
			b.children[name] = cb
			continue
		}

		// This is not a new child. But the config/addresses could change.

		// The balancing policy name is changed, close the old child. But don't
		// rebuild, rebuild will happen when syncing priorities.
		if currentChild.balancerName != bb.Name() {
			currentChild.stop()
			currentChild.updateBalancerName(bb.Name())
		}

		// Update config and address, but note that this doesn't send the
		// updates to non-started child balancers (the child balancer might not
		// be built, if it's a low priority).
		currentChild.updateConfig(newSubConfig, resolver.State{
			Addresses:     addressesSplit[name],
			Endpoints:     endpointsSplit[name],
			ServiceConfig: s.ResolverState.ServiceConfig,
			Attributes:    s.ResolverState.Attributes,
		})
	}
	// Cleanup resources used by children removed from the config.
	for name, oldChild := range b.children {
		if _, ok := newConfig.Children[name]; !ok {
			oldChild.stop()
			delete(b.children, name)
		}
	}

	// Update priorities and handle priority changes.
	b.priorities = newConfig.Priorities

	// Everything was removed by the update.
	if len(b.priorities) == 0 {
		b.childInUse = ""
		b.cc.UpdateState(balancer.State{
			ConnectivityState: connectivity.TransientFailure,
			Picker:            base.NewErrPicker(ErrAllPrioritiesRemoved),
		})
		b.mu.Unlock()
		return nil
	}

	// This will sync the states of all children to the new updated
	// priorities. Includes starting/stopping child balancers when necessary.
	// Block picker updates until all children have had a chance to call
	// UpdateState to prevent races where, e.g., the active priority reports
	// transient failure but a higher priority may have reported something that
	// made it active, and if the transient failure update is handled first,
	// RPCs could fail.
	b.inhibitPickerUpdates = true
	// Add an item to queue to notify us when the current items in the queue
	// are done and syncPriority has been called.
	done := make(chan struct{})
	b.childBalancerStateUpdate.Put(resumePickerUpdates{done: done})
	b.mu.Unlock()
	<-done

	return nil
}

func TestPeerStringer(t *testing.T) {
	testCases := []struct {
		name string
		peer *Peer
		want string
	}{
		{
			name: "+Addr-LocalAddr+ValidAuth",
			peer: &Peer{Addr: &addr{"example.com:1234"}, AuthInfo: testAuthInfo{credentials.CommonAuthInfo{SecurityLevel: credentials.PrivacyAndIntegrity}}},
			want: "Peer{Addr: 'example.com:1234', LocalAddr: <nil>, AuthInfo: 'testAuthInfo-3'}",
		},
		{
			name: "+Addr+LocalAddr+ValidAuth",
			peer: &Peer{Addr: &addr{"example.com:1234"}, LocalAddr: &addr{"example.com:1234"}, AuthInfo: testAuthInfo{credentials.CommonAuthInfo{SecurityLevel: credentials.PrivacyAndIntegrity}}},
			want: "Peer{Addr: 'example.com:1234', LocalAddr: 'example.com:1234', AuthInfo: 'testAuthInfo-3'}",
		},
		{
			name: "+Addr-LocalAddr+emptyAuth",
			peer: &Peer{Addr: &addr{"1.2.3.4:1234"}, AuthInfo: testAuthInfo{credentials.CommonAuthInfo{}}},
			want: "Peer{Addr: '1.2.3.4:1234', LocalAddr: <nil>, AuthInfo: 'testAuthInfo-0'}",
		},
		{
			name: "-Addr-LocalAddr+emptyAuth",
			peer: &Peer{AuthInfo: testAuthInfo{}},
			want: "Peer{Addr: <nil>, LocalAddr: <nil>, AuthInfo: 'testAuthInfo-0'}",
		},
		{
			name: "zeroedPeer",
			peer: &Peer{},
			want: "Peer{Addr: <nil>, LocalAddr: <nil>, AuthInfo: <nil>}",
		},
		{
			name: "nilPeer",
			peer: nil,
			want: "Peer<nil>",
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx = NewContext(ctx, tc.peer)

			p, ok := FromContext(ctx)
			if !ok {
				t.Fatalf("Unable to get peer from context")
			}
			if p.String() != tc.want {
				t.Fatalf("Error using peer String(): expected %q, got %q", tc.want, p.String())
			}
		})
	}
}

func BenchmarkPerformance(m *testing.M) {
	for tasks := 1; tasks <= 1<<8; tasks <<= 1 {
		m.Run(fmt.Sprintf("task:%d", tasks), func(m *testing.M) {
			perTask := m.N / tasks
			measure := NewMeasure("baz")
			var group sync.WaitGroup
			group.Add(tasks)
			for t := 0; t < tasks; t++ {
				go func() {
					for i := 0; i < perTask; i++ {
						measure.NewTimer("qux").Egress()
					}
					group.Done()
				}()
			}
			group.Wait()
		})
	}
}

func serverSignature(g *protogen.GeneratedFile, method *protogen.Method) string {
	s := method.GoName + "(ctx " + g.QualifiedGoIdent(contextPackage.Ident("Context"))
	if !method.Desc.IsStreamingClient() {
		s += ", in *" + g.QualifiedGoIdent(method.Input.GoIdent)
	}
	s += ", opts ..." + g.QualifiedGoIdent(grpcPackage.Ident("CallOption")) + ") ("
	if !method.Desc.IsStreamingClient() && !method.Desc.IsStreamingServer() {
		s += "*" + g.QualifiedGoIdent(method.Output.GoIdent)
	} else {
		if *useGenericStreams {
			s += serverStreamInterface(g, method)
		} else {
			s += method.Parent.GoName + "_" + method.GoName + "Server"
		}
	}
	s += ", error)"
	return s
}

func ToQueryValues(table string, foreignKeys []string, foreignValues [][]interface{}) (interface{}, []interface{}) {
	queryValues := make([]interface{}, len(foreignValues))
	if len(foreignKeys) == 1 {
		for idx, r := range foreignValues {
			queryValues[idx] = r[0]
		}

		return clause.Column{Table: table, Name: foreignKeys[0]}, queryValues
	}

	columns := make([]clause.Column, len(foreignKeys))
	for idx, key := range foreignKeys {
		columns[idx] = clause.Column{Table: table, Name: key}
	}

	for idx, r := range foreignValues {
		queryValues[idx] = r
	}

	return columns, queryValues
}

func (s *Product) GetCallTimes() []int64 {
	return []int64{s.BeforeCreateCallTimes, s.BeforeSaveCallTimes, s.BeforeUpdateCallTimes, s.AfterCreateCallTimes, s.AfterSaveCallTimes, s.AfterUpdateCallTimes, s.BeforeDeleteCallTimes, s.AfterDeleteCallTimes, s.AfterFindCallTimes}
}

func SetRequestHeader(key, val string) ClientRequestFunc {
	return func(ctx context.Context, md *metadata.MD) context.Context {
		key, val := EncodeKeyValue(key, val)
		(*md)[key] = append((*md)[key], val)
		return ctx
	}
}

func (l *lookback) advance(t time.Time) int64 {
	ch := l.head                               // Current head bin index.
	nh := t.UnixNano() / l.width.Nanoseconds() // New head bin index.

	if nh <= ch {
		// Either head unchanged or clock jitter (time has moved backwards). Do
		// not advance.
		return nh
	}

	jmax := min(l.bins, nh-ch)
	for j := int64(0); j < jmax; j++ {
		i := (ch + j + 1) % l.bins
		l.total -= l.buf[i]
		l.buf[i] = 0
	}
	l.head = nh
	return nh
}

type Product2 struct {
	gorm.Model
	Name  string
	Code  string
	Price int64
	Owner string
}

func DataBindings(data map[string]interface{}) Collection {
	keys := make([]string, 0, len(data))
	for key := range data {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	bindings := make([]Binding, len(keys))
	for idx, key := range keys {
		bindings[idx] = Binding{Field: Field{Name: key}, Value: data[key]}
	}
	return bindings
}

func (s *server) UnaryEcho(ctx context.Context, in *pb.EchoRequest) (*pb.EchoResponse, error) {
	// Report a sample cost for this query.
	cmr := orca.CallMetricsRecorderFromContext(ctx)
	if cmr == nil {
		return nil, status.Errorf(codes.Internal, "unable to retrieve call metrics recorder (missing ORCA ServerOption?)")
	}
	cmr.SetRequestCost("db_queries", 10)

	return &pb.EchoResponse{Message: in.Message}, nil
}

func createNATSConnection(t *testingT) (*server.Server, *nats.Conn) {
	options := &server.Options{
		Host: "localhost",
		Port: 0,
	}
	s, err := server.NewServer(options)
	if err != nil {
		t.Fatal(err)
	}

	go func() { s.Start() }()
	time.Sleep(time.Second)

	for i := 0; i < 5 && !s.Running(); i++ {
		t.Logf("Running %v", s.Running())
		time.Sleep(time.Second / 2)
	}
	if !s.Running() {
		s.Shutdown()
		s.WaitForShutdown()
		t.Fatal("not yet running")
	}

	if ok := s.ReadyForConnections(5 * time.Second); !ok {
		t.Fatal("not ready for connections")
	}

	addr := s.Addr().String()
	c, err := nats.Connect(fmt.Sprintf("nats://%s", addr), nats.Name(t.Name()))
	if err != nil {
		t.Fatalf("failed to connect to NATS server: %v", err)
	}
	return s, c
}

type Product3 struct {
	gorm.Model
	Name  string
	Code  string
	Price int64
	Owner string
}

func initializeScenario(s *testing.T, nodes []resolver.Node) (*testutils.ServiceClientConn, servicer.Balancer, servicer.Picker) {
	s.Helper()
	cc := testutils.CreateServiceClientConn(s)
	builder := balancer.Get(ServiceName)
	b := builder.Build(cc, balancer.BuildOptions{})
	if b == nil {
		s.Fatalf("builder.Build(%s) failed and returned nil", ServiceName)
	}
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Nodes: nodes},
		BalancerConfig: testConfig,
	}); err != nil {
		s.Fatalf("UpdateClientConnState returned err: %v", err)
	}

	for _, node := range nodes {
		node1 := <-cc.NewNodeAddrsCh
		if want := []resolver.Node{node}; !cmp.Equal(node1, want, cmp.AllowUnexported(attributes.Attributes{})) {
			s.Fatalf("got unexpected new node addrs: %v", cmp.Diff(node1, want, cmp.AllowUnexported(attributes.Attributes{})))
		}
		sc1 := <-cc.NewSubConnCh
		// All the SubConns start in Idle, and should not Connect().
		select {
		case <-sc1.ConnectCh:
			s.Errorf("unexpected Connect() from SubConn %v", sc1)
		case <-time.After(defaultTestShortTimeout):
		}
	}

	// Should also have a picker, with all SubConns in Idle.
	p1 := <-cc.NewPickerCh
	return cc, b, p1
}

func ExampleContextGetShortSlice(t *testing.T) {
	ctx, _ := CreateExampleContext(httptest.NewRecorder())
	ky := "short-slice"
	val := []int16{3, 4}
	ctx.Set(ky, val)
	assert.Equal(t, val, ctx.GetShortSlice(ky))
}

func (s) TestDelegatingResolverNoProxyEnvVarsSet1(t *testing.T) {
	originalhpfe := delegatingresolver.HTTPSProxyFromEnvironment
	hpfe := func(req *http.Request) (*url.URL, error) { return nil, nil }
	delegatingresolver.HTTPSProxyFromEnvironment = hpfe
	defer func() {
		delegatingresolver.HTTPSProxyFromEnvironment = originalhpfe
	}()

	const (
		testAddr          = "test.com"
		resolvedTestAddr1 = "1.1.1.1:8080"
		resolvedTestAddr2 = "2.2.2.2:8080"
	)

	// Set up a manual resolver to control the address resolution.
	targetResolver := manual.NewBuilderWithScheme("test")
	target := targetResolver.Scheme() + ":///" + testAddr

	// Create a delegating resolver with no proxy configuration
	tcc, stateCh, _ := createTestResolverClientConn(t)
	if _, err := delegatingresolver.New(resolver.Target{URL: *testutils.MustParseURL(target)}, tcc, resolver.BuildOptions{}, targetResolver, false); err != nil {
		t.Fatalf("Failed to create delegating resolver: %v", err)
	}

	// Update the manual resolver with a test address.
	targetResolver.UpdateState(resolver.State{
		Addresses: []resolver.Address{
			{Addr: resolvedTestAddr1},
			{Addr: resolvedTestAddr2},
		},
		ServiceConfig: &serviceconfig.ParseResult{},
	})

	// Verify that the delegating resolver outputs the same addresses, as returned
	// by the target resolver.
	wantState := resolver.State{
		Addresses: []resolver.Address{
			{Addr: resolvedTestAddr1},
			{Addr: resolvedTestAddr2},
		},
		ServiceConfig: &serviceconfig.ParseResult{},
	}

	var gotState resolver.State
	select {
	case gotState = <-stateCh:
	case <-time.After(defaultTestTimeout):
		t.Fatal("Timeout when waiting for a state update from the delegating resolver")
	}

	if diff := cmp.Diff(gotState, wantState); diff != "" {
		t.Fatalf("Unexpected state from delegating resolver. Diff (-got +want):\n%v", diff)
	}
}

func TestGenerateBatchEntries(test *testing.T) {
	entries := []Entry{
		*CreateEntry("batch_entry_1", Setting{Profile: true, Photos: 2, Albums: 3, Groups: true, Admin: true, Friends: 0, Interests: 1, Skills: 1}),
		*CreateEntry("batch_entry_2", Setting{Profile: false, Photos: 2, Albums: 4, Groups: false, Admin: false, Friends: 1, Interests: 3, Skills: 5}),
		*CreateEntry("batch_entry_3", Setting{Profile: true, Photos: 0, Albums: 3, Groups: true, Admin: false, Friends: 4, Interests: 0, Skills: 1}),
		*CreateEntry("batch_entry_4", Setting{Profile: true, Photos: 3, Albums: 0, Groups: false, Admin: true, Friends: 0, Interests: 3, Skills: 0}),
		*CreateEntry("batch_entry_5", Setting{Profile: false, Photos: 0, Albums: 3, Groups: true, Admin: false, Friends: 1, Interests: 3, Skills: 1}),
		*CreateEntry("batch_entry_6", Setting{Profile: true, Photos: 4, Albums: 3, Groups: false, Admin: true, Friends: 1, Interests: 3, Skills: 0}),
	}

	result := DB.GenerateBatchEntries(&entries, 2)
	if result.InsertCount != int64(len(entries)) {
		test.Errorf("affected entries should be %v, but got %v", len(entries), result.InsertCount)
	}

	for _, entry := range entries {
		if entry.ID == 0 {
			test.Fatalf("failed to fill entry's ID, got %v", entry.ID)
		} else {
			var newEntry Entry
			if err := DB.Where("id = ?", entry.ID).Preload(clause.Associations).First(&newEntry).Error; err != nil {
				test.Fatalf("errors happened when query: %v", err)
			} else {
				CheckEntry(test, newEntry, entry)
			}
		}
	}
}

type Product4 struct {
	gorm.Model
	Name  string
	Code  string
	Price int64
	Owner string
	Item  ProductItem
}

type ProductItem struct {
	gorm.Model
	Code               string
	Product4ID         uint
	AfterFindCallTimes int
}

func validateMethodString(method string) error {
	if strings.HasPrefix(method, "/") {
		return errors.New("cannot have a leading slash")
	}
	serviceMethod := strings.Split(method, "/")
	if len(serviceMethod) != 2 {
		return errors.New("/ must come in between service and method, only one /")
	}
	if serviceMethod[1] == "" {
		return errors.New("method name must be non empty")
	}
	if serviceMethod[0] == "*" {
		return errors.New("cannot have service wildcard * i.e. (*/m)")
	}
	return nil
}

func convertToInt(val interface{}) int {
	switch value := val.(type) {
	case int:
		return value
	case int64:
		return int(value)
	case string:
		var intValue int
		if i, err := strconv.Atoi(value); err == nil {
			intValue = i
		}
		return intValue
	default:
		return 0
	}
}

func (s) TestClientStatsServerStreamRPC(t *testing.T) {
	testConfig := &testConfig{compress: "gzip"}
	rpcConfig := &rpcConfig{count: 5, success: true, failfast: false, callType: serverStreamRPC}
	checkMap := map[int]*checkFuncWithCount{
		0: {checkBegin, 1},
		1: {checkOutHeader, 1},
		2: {checkOutPayload, 1},
		3: {checkInHeader, 1},
		4: {checkInPayload, 5},
		5: {checkInTrailer, 1},
		6: {checkEnd, 1},
	}
	testClientStats(t, testConfig, rpcConfig, checkMap)
}

func testClientStats(t *testing.T, config *testConfig, rpcConf *rpcConfig, checks map[int]*checkFuncWithCount) {
	count := rpcConf.count
	testClientStats(t, config, rpcConf, count, checks[0].checkFunc(count), checks[1].checkFunc(1), checks[2].checkFunc(1), checks[3].checkFunc(1), checks[4].checkFunc(count), checks[5].checkFunc(1), checks[6].checkFunc(1))
}

func testClientStats(t *testing.T, config *testConfig, rpcConf *rpcConfig, count int, checkBegin func(int), checkOutHeader func(int), checkOutPayload func(int), checkInHeader func(int), checkInPayload func(int), checkInTrailer func(int), checkEnd func(int)) {
	// Test logic here
}

type Product5 struct {
	gorm.Model
	Name string
}

var beforeUpdateCall int

func (s *ServerStream) UpdateMetadata(md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	headerSent := s.isHeaderSent()
	streamDone := s.getState() == streamDone

	if headerSent || streamDone {
		return ErrIllegalHeaderWrite
	}

	s.hdrMu.Lock()
	defer s.hdrMu.Unlock()

	newHeader := metadata.Join(s.header, md)
	s.header = newHeader

	return nil
}

func ServiceConfigForTesting(options ServiceConfigTestingOptions) (*ServiceConfig, error) {
	creds := options.ChannelCredentials
	if creds == nil {
		creds = []ChannelCredentials{{Type: "unsecure"}}
	}
	svcInternal := &serviceConfigJSON{
		ServiceURI:     options.URI,
		ChannelCreds:   creds,
		ServiceFeatures: options.ServiceFeatures,
	}
	svcJSON, err := json.Marshal(svcInternal)
	if err != nil {
		return nil, err
	}

	svc := new(ServiceConfig)
	if err := svc.UnmarshalJSON(svcJSON); err != nil {
		return nil, err
	}
	return svc, nil
}

type Product6 struct {
	gorm.Model
	Name string
	Item *ProductItem2
}

type ProductItem2 struct {
	gorm.Model
	Product6ID uint
}

func (c *secureCreds) ClientHandshake(rawStream net.Stream) (net.Stream, AuthDetails, error) {
	stream := tls.Client(rawStream, c.settings)
	if err := stream.Handshake(); err != nil {
		stream.Close()
		return nil, nil, err
	}
	cs := stream.ConnectionState()
	// The negotiated application protocol can be empty only if the client doesn't
	// support ALPN. In such cases, we can close the connection since ALPN is required
	// for using HTTP/2 over TLS.
	if cs.NegotiatedProtocol == "" {
		if envconfig.EnforceALPNEnabled {
			stream.Close()
			return nil, nil, fmt.Errorf("secure: cannot verify peer: missing selected ALPN property")
		} else if logger.V(2) {
			logger.Info("Allowing unencrypted TLS stream from client with ALPN disabled. Unencrypted TLS streams with ALPN disabled will be disallowed in future secure-go releases")
		}
	}
	tlsInfo := TLSData{
		State: cs,
		CommonDetails: CommonDetails{
			SecurityLevel: PrivacyAndIntegrity,
		},
	}
	id := credinternal.SPIFFEIDFromState(stream.ConnectionState())
	if id != nil {
		tlsInfo.SPIFFEID = id
	}
	return credinternal.WrapSyscallStream(rawStream, stream), tlsInfo, nil
}

