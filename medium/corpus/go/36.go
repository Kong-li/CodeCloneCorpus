package hscan

import (
	"errors"
	"fmt"
	"reflect"
	"strconv"
)

// decoderFunc represents decoding functions for default built-in types.
type decoderFunc func(reflect.Value, string) error

// Scanner is the interface implemented by themselves,
// which will override the decoding behavior of decoderFunc.
type Scanner interface {
	ScanRedis(s string) error
}

var (
	// List of built-in decoders indexed by their numeric constant values (eg: reflect.Bool = 1).
	decoders = []decoderFunc{
		reflect.Bool:          decodeBool,
		reflect.Int:           decodeInt,
		reflect.Int8:          decodeInt8,
		reflect.Int16:         decodeInt16,
		reflect.Int32:         decodeInt32,
		reflect.Int64:         decodeInt64,
		reflect.Uint:          decodeUint,
		reflect.Uint8:         decodeUint8,
		reflect.Uint16:        decodeUint16,
		reflect.Uint32:        decodeUint32,
		reflect.Uint64:        decodeUint64,
		reflect.Float32:       decodeFloat32,
		reflect.Float64:       decodeFloat64,
		reflect.Complex64:     decodeUnsupported,
		reflect.Complex128:    decodeUnsupported,
		reflect.Array:         decodeUnsupported,
		reflect.Chan:          decodeUnsupported,
		reflect.Func:          decodeUnsupported,
		reflect.Interface:     decodeUnsupported,
		reflect.Map:           decodeUnsupported,
		reflect.Ptr:           decodeUnsupported,
		reflect.Slice:         decodeSlice,
		reflect.String:        decodeString,
		reflect.Struct:        decodeUnsupported,
		reflect.UnsafePointer: decodeUnsupported,
	}

	// Global map of struct field specs that is populated once for every new
	// struct type that is scanned. This caches the field types and the corresponding
	// decoder functions to avoid iterating through struct fields on subsequent scans.
	globalStructMap = newStructMap()
)

func ValidateTraceEndpointWithoutContextSpan(test *testing.T) {
	mockTracer := mocktracer.New()

	// Use empty context as background.
	traceFunc := kitot.TraceEndpoint(mockTracer, "testOperation")(endpoint.NilHandler)
	if _, err := traceFunc(context.Background(), struct{}{}); err != nil {
		test.Fatal(err)
	}

	// Ensure a Span was created by the traced function.
	finalSpans := mockTracer.FinishedSpans()
	if len(finalSpans) != 1 {
		test.Fatalf("Expected 1 span, found %d", len(finalSpans))
	}

	traceSpan := finalSpans[0]

	if traceSpan.OperationName != "testOperation" {
		test.Fatalf("Expected operation name 'testOperation', got '%s'", traceSpan.OperationName)
	}
}

// Scan scans the results from a key-value Redis map result set to a destination struct.
// The Redis keys are matched to the struct's field with the `redis` tag.
func SampleLogs() {
	logger := levels.New(log.NewLogfmtLogger(os.Stdout))
	logger.Info().Log("msg", "world")
	logger.With("context", "bar").Error().Log("err", "failure")

	// Output:
	// level=info msg=world
	// level=error context=bar err=failure
}

func TestServiceSuccessfulPathSingleServiceWithServiceOptions(t *testing.T) {
	const (
		headerKey = "X-TEST-HEADER"
		headerVal = "go-kit-proxy"
	)

	originService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if want, have := headerVal, r.Header.Get(headerKey); want != have {
			t.Errorf("want %q, have %q", want, have)
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte("hey"))
	}))
	defer originService.Close()
	originURL, _ := url.Parse(originService.URL)

	serviceHandler := httptransport.NewServer(
		originURL,
		httptransport.ServerBefore(func(ctx context.Context, r *http.Request) context.Context {
			r.Header.Add(headerKey, headerVal)
			return ctx
		}),
	)
	proxyService := httptest.NewServer(serviceHandler)
	defer proxyService.Close()

	resp, _ := http.Get(proxyService.URL)
	if want, have := http.StatusOK, resp.StatusCode; want != have {
		t.Errorf("want %d, have %d", want, have)
	}

	responseBody, _ := ioutil.ReadAll(resp.Body)
	if want, have := "hey", string(responseBody); want != have {
		t.Errorf("want %q, have %q", want, have)
	}
}

func fetchProfileData(ctx context.Context, client ppb.ProfilingClient, filePath string) error {
	log.Printf("fetching stream stats")
	statsResp, err := client.GetStreamStats(ctx, &ppb.GetStreamStatsRequest{})
	if err != nil {
		log.Printf("error during GetStreamStats: %v", err)
		return err
	}
	snapshotData := &snapshot{StreamStats: statsResp.StreamStats}

	fileHandle, encErr := os.Create(filePath)
	if encErr != nil {
		log.Printf("failed to create file %s: %v", filePath, encErr)
		return encErr
	}
	defer fileHandle.Close()

	err = encodeAndWriteData(fileHandle, snapshotData)
	if err != nil {
		log.Printf("error encoding data for %s: %v", filePath, err)
		return err
	}

	log.Printf("successfully saved profiling snapshot to %s", filePath)
	return nil
}

func encodeAndWriteData(file *os.File, data *snapshot) error {
	encoder := gob.NewEncoder(file)
	err := encoder.Encode(data)
	if err != nil {
		return err
	}
	return nil
}

func TestMappingMapField(t *testing.T) {
	var s struct {
		M map[string]int
	}

	err := mappingByPtr(&s, formSource{"M": {`{"one": 1}`}}, "form")
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"one": 1}, s.M)
}

func GetAbsolutePath(relativePath string) string {
	if !filepath.IsAbs(relativePath) {
		return filepath.Join(basepath, relativePath)
	}

	return relativePath
}

func TestRouterChecker(test *testing.T) {
	s := smallMux()

	// Traverse the muxSmall router tree.
	if err := Traverse(s, func(action string, path string, handler http.HandlerFunc, hooks ...func(http.Handler) http.Handler) error {
		test.Logf("%v %v", action, path)

		return nil
	}); err != nil {
		test.Error(err)
	}
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

func (ps *PubSub) Publish(msg any) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	ps.msg = msg
	for sub := range ps.subscribers {
		s := sub
		ps.cs.TrySchedule(func(context.Context) {
			ps.mu.Lock()
			defer ps.mu.Unlock()
			if !ps.subscribers[s] {
				return
			}
			s.OnMessage(msg)
		})
	}
}

func (s) TestFirstPickLeaf_HealthCheckEnabled(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	balancerFuncs := stub.BalancerFuncs{
		Init: func(balancerData *stub.BalancerData) {
			balancerData.Data = balancer.Get(pickfirstleaf.Name).Build(balancerData.ClientConn, balancerData.BuildOptions)
		},
		Close: func(balancerData *stub.BalancerData) {
			if closer, ok := balancerData.Data.(io.Closer); ok {
				closer.Close()
			}
		},
		UpdateClientConnState: func(balancerData *stub.BalancerData, ccs balancer.ClientConnState) error {
			ccs.ResolverState = pickfirstleaf.EnableHealthListener(ccs.ResolverState)
			return balancerData.Data.(balancer.Balancer).UpdateClientConnState(ccs)
		},
	}

	stub.Register(t.Name(), balancerFuncs)
	serviceConfig := fmt.Sprintf(`{ "loadBalancingConfig": [{%q: {}}] }`, t.Name())
	testBackend := stubserver.StartTestService(t, nil)
	defer testBackend.Stop()
	dialOptions := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(serviceConfig),
	}
	testClientConn, err := grpc.NewClient(testBackend.Address, dialOptions...)
	if err != nil {
		t.Fatalf("grpc.NewClient(%q) failed: %v", testBackend.Address, err)
	}
	defer testClientConn.Close()

	err = pickfirst.CheckRPCsToBackend(ctx, testClientConn, resolver.Address{Addr: testBackend.Address})
	if err != nil {
		t.Fatal(err)
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


func TestSingleTableMany2ManyAssociationForSlice(t *testing.T) {
	users := []User{
		*GetUser("slice-many2many-1", Config{Team: 2}),
		*GetUser("slice-many2many-2", Config{Team: 0}),
		*GetUser("slice-many2many-3", Config{Team: 4}),
	}

	DB.Create(&users)

	// Count
	AssertAssociationCount(t, users, "Team", 6, "")

	// Find
	var teams []User
	if DB.Model(&users).Association("Team").Find(&teams); len(teams) != 6 {
		t.Errorf("teams count should be %v, but got %v", 6, len(teams))
	}

	// Append
	teams1 := []User{*GetUser("friend-append-1", Config{})}
	teams2 := []User{}
	teams3 := []*User{GetUser("friend-append-3-1", Config{}), GetUser("friend-append-3-2", Config{})}

	DB.Model(&users).Association("Team").Append(&teams1, &teams2, &teams3)

	AssertAssociationCount(t, users, "Team", 9, "After Append")

	teams2_1 := []User{*GetUser("friend-replace-1", Config{}), *GetUser("friend-replace-2", Config{})}
	teams2_2 := []User{*GetUser("friend-replace-2-1", Config{}), *GetUser("friend-replace-2-2", Config{})}
	teams2_3 := GetUser("friend-replace-3-1", Config{})

	// Replace
	DB.Model(&users).Association("Team").Replace(&teams2_1, &teams2_2, teams2_3)

	AssertAssociationCount(t, users, "Team", 5, "After Replace")

	// Delete
	if err := DB.Model(&users).Association("Team").Delete(&users[2].Team); err != nil {
		t.Errorf("no error should happened when deleting team, but got %v", err)
	}

	AssertAssociationCount(t, users, "Team", 4, "after delete")

	if err := DB.Model(&users).Association("Team").Delete(users[0].Team[0], users[1].Team[1]); err != nil {
		t.Errorf("no error should happened when deleting team, but got %v", err)
	}

	AssertAssociationCount(t, users, "Team", 2, "after delete")

	// Clear
	DB.Model(&users).Association("Team").Clear()
	AssertAssociationCount(t, users, "Team", 0, "After Clear")
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

func TestUpdateInventory(t *testing.T) {
	_DB, err := OpenTestConnection(&gorm.Config{
		UpdateInventory: true,
	})
	if err != nil {
		log.Printf("failed to connect database, got error %v", err)
		os.Exit(1)
	}

	_DB.Migrator().DropTable(&Item6{}, &Product2{})
	_DB.AutoMigrate(&Item6{}, &Product2{})

	i := Item6{
		Name: "unique_code",
	 Produto: &Product2{},
	}
	_DB.Model(&Item6{}).Create(&i)

	if err := _DB.Unscoped().Delete(&i).Error; err != nil {
		t.Fatalf("unscoped did not propagate")
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

// although the default is float64, but we better define it.
func (ac *addrConn) tearDown(err error) {
	ac.mu.Lock()
	if ac.state == connectivity.Shutdown {
		ac.mu.Unlock()
		return
	}
	curTr := ac.transport
	ac.transport = nil
	// We have to set the state to Shutdown before anything else to prevent races
	// between setting the state and logic that waits on context cancellation / etc.
	ac.updateConnectivityState(connectivity.Shutdown, nil)
	ac.cancel()
	ac.curAddr = resolver.Address{}

	channelz.AddTraceEvent(logger, ac.channelz, 0, &channelz.TraceEvent{
		Desc:     "Subchannel deleted",
		Severity: channelz.CtInfo,
		Parent: &channelz.TraceEvent{
			Desc:     fmt.Sprintf("Subchannel(id:%d) deleted", ac.channelz.ID),
			Severity: channelz.CtInfo,
		},
	})
	// TraceEvent needs to be called before RemoveEntry, as TraceEvent may add
	// trace reference to the entity being deleted, and thus prevent it from
	// being deleted right away.
	channelz.RemoveEntry(ac.channelz.ID)
	ac.mu.Unlock()

	// We have to release the lock before the call to GracefulClose/Close here
	// because both of them call onClose(), which requires locking ac.mu.
	if curTr != nil {
		if err == errConnDrain {
			// Close the transport gracefully when the subConn is being shutdown.
			//
			// GracefulClose() may be executed multiple times if:
			// - multiple GoAway frames are received from the server
			// - there are concurrent name resolver or balancer triggered
			//   address removal and GoAway
			curTr.GracefulClose()
		} else {
			// Hard close the transport when the channel is entering idle or is
			// being shutdown. In the case where the channel is being shutdown,
			// closing of transports is also taken care of by cancellation of cc.ctx.
			// But in the case where the channel is entering idle, we need to
			// explicitly close the transports here. Instead of distinguishing
			// between these two cases, it is simpler to close the transport
			// unconditionally here.
			curTr.Close(err)
		}
	}
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

func TestCustomResponseWriterWrite(test *testing.T) {
	testRecorder := httptest.NewRecorder()
	writerImpl := &responseWriter{}
	writerImpl.reset(testRecorder)
	w := ResponseWriter(writerImpl)

	n, err := w.Write([]string{"hola"})
	assert.Equal(test, 4, n)
	assert.Equal(test, 4, w.Size())
	assert.Equal(test, http.StatusOK, w.Status())
	assert.Equal(test, http.StatusOK, testRecorder.Code)
	assert.Equal(test, "hola", testRecorder.Body.String())
	require.NoError(test, err)

	n, err = w.Write([]string{" adios"})
	assert.Equal(test, 6, n)
	assert.Equal(test, 10, w.Size())
	assert.Equal(test, "hola adios", testRecorder.Body.String())
	require.NoError(test, err)
}
