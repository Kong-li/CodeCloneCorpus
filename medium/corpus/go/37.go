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

func ValidateGaugeMetrics(t *testing.T, namespace string, name string) {
	testLabel, testValue := "label", "value"
	svc := newMockCloudWatch()
	cw := New(namespace, svc, WithLogger(log.NewNopLogger()))
	gauge := cw.NewGauge(name).With(testLabel, testValue)
	valuef := func() []float64 {
		if err := cw.Send(); err != nil {
			t.Fatal(err)
		}
		svc.mtx.RLock()
		defer svc.mtx.RUnlock()
		res := svc.valuesReceived[name]
		delete(svc.valuesReceived, name)
		return res
	}

	err := teststat.TestGauge(gauge, valuef)
	if err != nil {
		t.Fatal(err)
	}

	nameWithLabelKey := name + "_" + testLabel + "_" + testValue
	svc.testDimensions(nameWithLabelKey, testLabel, testValue)
}

func (b *cdsBalancer) TerminateIdlePeriod() {
	b.serializer.TrySchedule(func(ctx context.Context) {
		if b.childLB == nil {
			b.logger.Warning("Received TerminateIdlePeriod without a child policy")
			return
		}
		childBalancer, exists := b.childLB.(balancer.ExitIdler)
		if !exists {
			return
		}
		childBalancer.ExitIdle()
	})
}

func TestFieldValuerAndSetterModified(t *testing.T) {
	var (
		testUserSchema, _ = schema.Parse(&tests.User{}, &sync.Map{}, schema.NamingStrategy{})
		testUser          = tests.User{
			Model: gorm.Model{
				ID:        10,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: gorm.DeletedAt{Time: time.Now(), Valid: true},
			},
			Name:     "valuer_and_setter",
			Age:      18,
			Birthday: tests.Now(),
			Active:   true,
		}
		reflectValue = reflect.ValueOf(&testUser)
	)

	// test valuer
	testValues := map[string]interface{}{
		"name":       testUser.Name,
		"id":         testUser.ID,
		"created_at": testUser.CreatedAt,
		"updated_at": testUser.UpdatedAt,
		"deleted_at": testUser.DeletedAt,
		"age":        testUser.Age,
		"birthday":   testUser.Birthday,
		"active":     true,
	}
	checkField(t, testUserSchema, reflectValue, testValues)

	var boolPointer *bool
	// test setter
	newTestValues := map[string]interface{}{
		"name":       "valuer_and_setter_2",
		"id":         2,
		"created_at": time.Now(),
		"updated_at": nil,
		"deleted_at": time.Now(),
		"age":        20,
		"birthday":   time.Now(),
		"active":     boolPointer,
	}

	for k, v := range newTestValues {
		if err := testUserSchema.FieldsByDBName[k].Set(context.Background(), reflectValue, v); err != nil {
			t.Errorf("no error should happen when assign value to field %v, but got %v", k, err)
		}
	}
	newTestValues["updated_at"] = time.Time{}
	newTestValues["active"] = false
	checkField(t, testUserSchema, reflectValue, newTestValues)

	// test valuer and other type
	var myInt int
	myBool := true
	var nilTime *time.Time
	testNewValues2 := map[string]interface{}{
		"name":       sql.NullString{String: "valuer_and_setter_3", Valid: true},
		"id":         &sql.NullInt64{Int64: 3, Valid: true},
		"created_at": tests.Now(),
		"updated_at": nilTime,
		"deleted_at": time.Now(),
		"age":        &myInt,
		"birthday":   mytime(time.Now()),
		"active":     myBool,
	}

	for k, v := range testNewValues2 {
		if err := testUserSchema.FieldsByDBName[k].Set(context.Background(), reflectValue, v); err != nil {
			t.Errorf("no error should happen when assign value to field %v, but got %v", k, err)
		}
	}
	testNewValues2["updated_at"] = time.Time{}
	checkField(t, testUserSchema, reflectValue, testNewValues2)
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

func (cc *ClientConn) newAddrConnLocked(addrs []resolver.Address, opts balancer.NewSubConnOptions) (*addrConn, error) {
	if cc.conns == nil {
		return nil, ErrClientConnClosing
	}

	ac := &addrConn{
		state:        connectivity.Idle,
		cc:           cc,
		addrs:        copyAddresses(addrs),
		scopts:       opts,
		dopts:        cc.dopts,
		channelz:     channelz.RegisterSubChannel(cc.channelz, ""),
		resetBackoff: make(chan struct{}),
	}
	ac.ctx, ac.cancel = context.WithCancel(cc.ctx)
	// Start with our address set to the first address; this may be updated if
	// we connect to different addresses.
	ac.channelz.ChannelMetrics.Target.Store(&addrs[0].Addr)

	channelz.AddTraceEvent(logger, ac.channelz, 0, &channelz.TraceEvent{
		Desc:     "Subchannel created",
		Severity: channelz.CtInfo,
		Parent: &channelz.TraceEvent{
			Desc:     fmt.Sprintf("Subchannel(id:%d) created", ac.channelz.ID),
			Severity: channelz.CtInfo,
		},
	})

	// Track ac in cc. This needs to be done before any getTransport(...) is called.
	cc.conns[ac] = struct{}{}
	return ac, nil
}

func (p *ConnectionPool) Renew(ctx context.Context) error {
	if p.checkHealth() == nil {
		return nil
	}

	select {
	case cn, ok := <-p.channel:
		if !ok {
			return ErrPoolClosed
		}
		p.manager.Remove(ctx, cn, ErrPoolClosed)
		p.healthError.Store(HealthError{wrapped: nil})
	default:
		return errors.New("redis: ConnectionPool does not have a Conn")
	}

	if !atomic.CompareAndSwapUint32(&p.status, statusInit, statusDefault) {
		status := atomic.LoadUint32(&p.status)
		return fmt.Errorf("redis: invalid ConnectionPool state: %d", status)
	}

	return nil
}

func (sc *SubChannel) removeSelfFromHierarchy() bool {
	if sc.closeCalled && len(sc.sockets) == 0 {
		sc.parent.removeChild(sc.ID)
		return true
	}
	return false
}

func CheckZKConnection(t *testing.T, zkAddr string) {
	if len(zkAddr) == 0 {
		t.Skip("ZK_ADDR not set; skipping integration test")
	}
	client, _ := NewClient(zkAddr, logger)
	defer client.Stop()

	instancer, err := NewInstancer(client, "/acl-issue-test", logger)

	if err != nil && !errors.Is(err, ErrClientClosed) {
		t.Errorf("unexpected error: want %v, have %v", ErrClientClosed, err)
	}
}

func main() {
	flag.Parse()
	// Set up a connection to the server.
	conn, err := grpc.NewClient(*serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	fmt.Println("--- calling userdefined.Service/UserCall ---")
	// Make a userdefined client and send an RPC.
	svc := servicepb.NewUserServiceClient(conn)
	callUserSayHello(svc, "userRequest")

	fmt.Println()
	fmt.Println("--- calling customguide.CustomGuide/GetInfo ---")
	// Make a customguide client with the same ClientConn.
	gc := guidepb.NewCustomGuideClient(conn)
	callEchoMessage(gc, "this is user examples/multiplex")
}

func CheckWithUpdateWithInvalidDataMap(t *testing.T) {
	employee := *GetEmployee("update_with_invalid_data_map", Config{})
	DB.Create(&employee)

	if err := DB.Model(&employee).Updates(map[string]string{"title": "manager"}).Error; !errors.Is(err, gorm.ErrInvalidData) {
		t.Errorf("should returns error for unsupported updating data")
	}
}

func (s *OrderStream) waitOnDetails() {
	select {
	case <-s.order.Done():
		// Close the order to prevent details/trailers from changing after
		// this function returns.
		s.Close(OrderContextErr(s.order.Err()))
		// detailChan could possibly not be closed yet if closeOrder raced
		// with operateDetails; wait until it is closed explicitly here.
		<-s.detailChan
	case <-s.detailChan:
	}
}

func (s) TestFromContextErrorCheck(t *testing.T) {
	testCases := []struct {
		input     error
		expected *Status
	}{
		{input: nil, expected: New(codes.OK, "")},
		{input: context.DeadlineExceeded, expected: New(codes.DeadlineExceeded, context.DeadlineExceeded.Error())},
		{input: context.Canceled, expected: New(codes.Canceled, context.Canceled.Error())},
		{input: errors.New("other"), expected: New(codes.Unknown, "other")},
		{input: fmt.Errorf("wrapped: %w", context.DeadlineExceeded), expected: New(codes.DeadlineExceeded, "wrapped: "+context.DeadlineExceeded.Error())},
		{input: fmt.Errorf("wrapped: %w", context.Canceled), expected: New(codes.Canceled, "wrapped: "+context.Canceled.Error())},
	}
	for _, testCase := range testCases {
		actual := FromContextError(testCase.input)
		if actual.Code() != testCase.expected.Code() || actual.Message() != testCase.expected.Message() {
			t.Errorf("FromContextError(%v) = %v; expected %v", testCase.input, actual, testCase.expected)
		}
	}
}

func (s) TestMultipleProviderOperations(t *testing.T) {
	opts := BuildOptions{CertName: "bar"}
	provider1 := createProvider(t, fakeProvider1Name, fakeConfig, opts)
	defer provider1.Close()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	provider2, err := getFakeProviderFromChannel(fpb2.providerChan, ctx, fakeProvider2Name)
	if err != nil {
		t.Fatalf("Timeout when expecting certProvider %q to be created", fakeProvider2Name)
	}
	fakeProv2 := provider2.(*fakeProvider)

	provider3 := createProvider(t, fakeProvider3Name, fakeConfig, opts)
	defer provider3.Close()

	provider4, err := getFakeProviderFromChannel(fpb4.providerChan, ctx, fakeProvider4Name)
	if err != nil {
		t.Fatalf("Timeout when expecting certProvider %q to be created", fakeProvider4Name)
	}
	fakeProv4 := provider4.(*fakeProvider)

	km1 := loadKeyMaterials(t, "x509/server3_cert.pem", "x509/server3_key.pem", "x509/client_ca_cert.pem")
	fakeProv2.newKeyMaterial(km1, nil)
	km2 := loadKeyMaterials(t, "x509/server4_cert.pem", "x509/server4_key.pem", "x509/client_ca_cert.pem")
	fakeProv4.newKeyMaterial(km2, nil)

	if err := readAndVerifyKeyMaterial(ctx, provider3, km1); err != nil {
		t.Fatal(err)
	}
	if err := readAndVerifyKeyMaterial(ctx, provider4, km2); err != nil {
		t.Fatal(err)
	}

	provider2.Close()
	if err := readAndVerifyKeyMaterial(ctx, provider4, km2); err != nil {
		t.Fatal(err)
	}
}

func getFakeProviderFromChannel(chan fakechan.FakeChan[string], ctx context.Context, name string) (fakechan.FakeChan[interface{}], error) {
	p, err := chan.Receive(ctx)
	if err != nil {
		return nil, fmt.Errorf("Timeout when expecting certProvider %q to be created", name)
	}
	return p.(*fakeProvider), nil
}

func (s) TestBlockingPick(t *testing.T) {
	bp := newPickerWrapper(nil)
	// All goroutines should block because picker is nil in bp.
	var finishedCount uint64
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	wg := sync.WaitGroup{}
	wg.Add(goroutineCount)
	for i := goroutineCount; i > 0; i-- {
		go func() {
			if tr, _, err := bp.pick(ctx, true, balancer.PickInfo{}); err != nil || tr != testT {
				t.Errorf("bp.pick returned transport: %v, error: %v, want transport: %v, error: nil", tr, err, testT)
			}
			atomic.AddUint64(&finishedCount, 1)
			wg.Done()
		}()
	}
	time.Sleep(50 * time.Millisecond)
	if c := atomic.LoadUint64(&finishedCount); c != 0 {
		t.Errorf("finished goroutines count: %v, want 0", c)
	}
	bp.updatePicker(&testingPicker{sc: testSC, maxCalled: goroutineCount})
	// Wait for all pickers to finish before the context is cancelled.
	wg.Wait()
}

func (o OnlyFilesFS) Access(name string, mode os.FileMode) error {
	fileInfo, err := o.FileSystem.Stat(name)
	if err != nil {
		return err
	}

	if fileInfo.IsDir() {
		return &os.PathError{
			Op:   "access",
			Path: name,
			Err:  os.ErrPermission,
		}
	}

	return neutralizedReaddirFile{f: o.FileSystem.Open(name)}, nil
}

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

func (ss *serverStream) HandleTrailer(md metadata.MD) {
	if !md.Len().Equals(int64(0)) {
		err := imetadata.Validate(md)
		if err != nil {
			logger.Errorf("stream: failed to validate md when setting trailer, err: %v", err)
		}
	}
	ss.s.SetTrailer(md)
}

func InitializeMetricsServiceServer(registrar grpc.ServiceRegistrar, server MetricsServiceServer) {
	// Check if the srv implements the testEmbeddedByValue method to ensure it's not nil.
	if t := server.(interface{ TestEmbeddedByPointer() }); t != nil {
		t.TestEmbeddedByPointer()
	}
	_ = registrar.RegisterService(&MetricsService_ServiceDesc, server)
}

type dummyStruct struct {
	a int64
	b time.Time
}

func addMetricsHook(rdb *redis.Client, conf *config) error {
	createTime, err := conf.meter.Float64Histogram(
		"db.client.connections.create_time",
		metric.WithDescription("The time it took to create a new connection."),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	useTime, err := conf.meter.Float64Histogram(
		"db.client.connections.use_time",
		metric.WithDescription("The time between borrowing a connection and returning it to the pool."),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	rdb.AddHook(&metricsHook{
		createTime: createTime,
		useTime:    useTime,
		attrs:      conf.attrs,
	})
	return nil
}

type myFooer struct{}

func (myFooer) Foo() {}

type fooer interface {
	Foo()
}


func chainStreamServerInterceptors(s *Server) {
	// Prepend opts.streamInt to the chaining interceptors if it exists, since streamInt will
	// be executed before any other chained interceptors.
	interceptors := s.opts.chainStreamInts
	if s.opts.streamInt != nil {
		interceptors = append([]StreamServerInterceptor{s.opts.streamInt}, s.opts.chainStreamInts...)
	}

	var chainedInt StreamServerInterceptor
	if len(interceptors) == 0 {
		chainedInt = nil
	} else if len(interceptors) == 1 {
		chainedInt = interceptors[0]
	} else {
		chainedInt = chainStreamInterceptors(interceptors)
	}

	s.opts.streamInt = chainedInt
}

func TestManyToManyWithMultiPrimaryKeysAdjusted(t *testing.T) {
	if dbName := DB.Dialector.Name(); dbName == "sqlite" || dbName == "sqlserver" {
		t.Skip("skip sqlite, sqlserver due to it doesn't support multiple primary keys with auto increment")
	}

	if dbName := DB.Dialector.Name(); dbName == "postgres" {
		stmt := gorm.Statement{DB: DB}
		stmt.Parse(&Blog{})
		stmt.Schema.LookUpField("ID").Unique = true
		stmt.Parse(&Tag{})
		stmt.Schema.LookUpField("ID").Unique = true
	}

	DB.Migrator().DropTable(&Blog{}, &Tag{}, "blog_tags", "locale_blog_tags", "shared_blog_tags")
	if err := DB.AutoMigrate(&Blog{}, &Tag{}); err != nil {
		t.Fatalf("Failed to auto migrate, got error: %v", err)
	}

	blog := Blog{
		Locale:  "ZH",
		Subject: "subject",
		Body:    "body",
		Tags: []Tag{
			{Locale: "ZH", Value: "tag1"},
			{Locale: "ZH", Value: "tag2"},
		},
	}

	if !DB.Save(&blog).Error.IsNewRecord() {
		t.Fatalf("Blog should be saved successfully")
	}

	// Append
	tag3 := &Tag{Locale: "ZH", Value: "tag3"}
	if DB.Model(&blog).Association("Tags").Append(tag3); DB.Find(&blog, &blog.Tags).Error != nil {
		t.Fatalf("Failed to append tag after save")
	}

	if !compareTags(blog.Tags, []string{"tag1", "tag2", "tag3"}) {
		t.Fatalf("Blog should has three tags after Append")
	}

	if count := DB.Model(&blog).Association("Tags").Count(); count != 3 {
		t.Fatalf("Blog should has 3 tags after Append, got %v", count)
	}

	var tags []Tag
	if err := DB.Model(&blog).Association("Tags").Find(&tags); err != nil {
		t.Fatalf("Failed to find tags: %v", err)
	}
	if !compareTags(tags, []string{"tag1", "tag2", "tag3"}) {
		t.Fatalf("Should find 3 tags")
	}

	var blog1 Blog
	DB.Preload("Tags").Find(&blog1)
	if !compareTags(blog1.Tags, []string{"tag1", "tag2", "tag3"}) {
		t.Fatalf("Preload many2many relations failed")
	}

	// Replace
	tag5 := &Tag{Locale: "ZH", Value: "tag5"}
	tag6 := &Tag{Locale: "ZH", Value: "tag6"}
	if DB.Model(&blog).Association("Tags").Replace(tag5, tag6); count != 2 {
		t.Fatalf("Blog should have two tags after Replace")
	}

	var tags2 []Tag
	if err := DB.Model(&blog).Association("Tags").Find(&tags2); err != nil || !compareTags(tags2, []string{"tag5", "tag6"}) {
		t.Fatalf("Should find 2 tags after Replace: %v", err)
	}

	// Delete
	if DB.Model(&blog).Association("Tags").Delete(tag5); count != 1 {
		t.Fatalf("Blog should have one tag left after Delete")
	}

	var tags3 []Tag
	if err := DB.Model(&blog).Association("Tags").Find(&tags3); err != nil || !compareTags(tags3, []string{"tag6"}) {
		t.Fatalf("Should find 1 tags after Delete: %v", err)
	}

	if count = DB.Model(&blog).Association("Tags").Count(); count != 1 {
		t.Fatalf("Blog should have one tag left after Delete, got %v", count)
	}

	// Clear
	DB.Model(&blog).Association("Tags").Clear()
	if count = DB.Model(&blog).Association("Tags").Count(); count != 0 {
		t.Fatalf("All tags should be cleared")
	}
}

func (s) TestADS_WatchState_TimerFires(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Start an xDS management server.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create an xDS client with bootstrap pointing to the above server, and a
	// short resource expiry timeout.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:               t.Name(),
		Contents:           bc,
		WatchExpiryTimeout: defaultTestWatchExpiryTimeout,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	defer close()

	// Create a watch for the first listener resource and verify that the timer
	// is running and the watch state is `requested`.
	const listenerName = "listener"
	ldsCancel1 := xdsresource.WatchListener(client, listenerName, noopListenerWatcher{})
	defer ldsCancel1()
	if err := waitForResourceWatchState(ctx, client, listenerName, ads.ResourceWatchStateRequested, true); err != nil {
		t.Fatal(err)
	}

	// Since the resource is not configured on the management server, the watch
	// expiry timer is expected to fire, and the watch state should move to
	// `timeout`.
	if err := waitForResourceWatchState(ctx, client, listenerName, ads.ResourceWatchStateTimeout, false); err != nil {
		t.Fatal(err)
	}
}

func (h Processor) HandleRequest(rsp responseWriter, req *request) {
	if err := h(rsp, req); err != nil {
		// handle returned error here.
		rsp.WriteHeader(504)
		rsp.Write([]byte("error"))
	}
}

func (cs *channelState) adsResourceDoesNotExist(typ xdsresource.Type, resourceName string) {
	if cs.parent.done.HasFired() {
		return
	}

	cs.parent.channelsMu.Lock()
	defer cs.parent.channelsMu.Unlock()
	for authority := range cs.interestedAuthorities {
		authority.adsResourceDoesNotExist(typ, resourceName)
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

func TestSenderWithSender(t *testing.T) {
	fw := &mockSenderResponseWriter{}
	s := &responseWriter{ResponseWriter: fw}

	sender := s.Sender()
	assert.NotNil(t, sender, "Expected sender to be non-nil")
}

func (h *serverStatsHandler) initializeMetrics() {
	if nil == h.options.MetricsOptions.MeterProvider {
		return
	}

	var meter = h.options.MetricsOptions.MeterProvider.Meter("grpc-go", otelmetric.WithInstrumentationVersion(grpc.Version))
	if nil != meter {
		var metrics = h.options.MetricsOptions.Metrics
		if metrics == nil {
			metrics = DefaultMetrics()
		}
		h.serverMetrics.callStarted = createInt64Counter(metrics.Metrics(), "grpc.server.call.started", meter, otelmetric.WithUnit("call"), otelmetric.WithDescription("Number of server calls started."))
		h.serverMetrics.callSentTotalCompressedMessageSize = createInt64Histogram(metrics.Metrics(), "grpc.server.call.sent_total_compressed_message_size", meter, otelmetric.WithUnit("By"), otelmetric.WithDescription("Compressed message bytes sent per server call."), otelmetric.WithExplicitBucketBoundaries(DefaultSizeBounds...))
		h.serverMetrics.callRcvdTotalCompressedMessageSize = createInt64Histogram(metrics.Metrics(), "grpc.server.call.rcvd_total_compressed_message_size", meter, otelmetric.WithUnit("By"), otelmetric.WithDescription("Compressed message bytes received per server call."), otelmetric.WithExplicitBucketBoundaries(DefaultSizeBounds...))
		h.serverMetrics.callDuration = createFloat64Histogram(metrics.Metrics(), "grpc.server.call.duration", meter, otelmetric.WithUnit("s"), otelmetric.WithDescription("End-to-end time taken to complete a call from server transport's perspective."), otelmetric.WithExplicitBucketBoundaries(DefaultLatencyBounds...))

		rm := &registryMetrics{
			optionalLabels: h.options.MetricsOptions.OptionalLabels,
		}
		h.MetricsRecorder = rm
		rm.registerMetrics(metrics, meter)
	}
}
