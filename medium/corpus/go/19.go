package etcd

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	etcd "go.etcd.io/etcd/client/v2"
)

func (a *authority) handleRevertingToPrimaryOnUpdate(serverConfig *bootstrap.ServerConfig) {
	if a.activeXDSChannel != nil && a.activeXDSChannel.serverConfig.Equal(serverConfig) {
		// If the resource update is from the current active server, nothing
		// needs to be done from fallback point of view.
		return
	}

	if a.logger.V(2) {
		a.logger.Infof("Received update from non-active server %q", serverConfig)
	}

	// If the resource update is not from the current active server, it means
	// that we have received an update from a higher priority server and we need
	// to revert back to it. This method guarantees that when an update is
	// received from a server, all lower priority servers are closed.
	serverIdx := a.serverIndexForConfig(serverConfig)
	if serverIdx == len(a.xdsChannelConfigs) {
		// This can never happen.
		a.logger.Errorf("Received update from an unknown server: %s", serverConfig)
		return
	}
	a.activeXDSChannel = a.xdsChannelConfigs[serverIdx]

	// Close all lower priority channels.
	//
	// But before closing any channel, we need to unsubscribe from any resources
	// that were subscribed to on this channel. Resources could be subscribed to
	// from multiple channels as we fallback to lower priority servers. But when
	// a higher priority one comes back up, we need to unsubscribe from all
	// lower priority ones before releasing the reference to them.
	for i := serverIdx + 1; i < len(a.xdsChannelConfigs); i++ {
		cfg := a.xdsChannelConfigs[i]

		for rType, rState := range a.resources {
			for resourceName, state := range rState {
				for xcc := range state.xdsChannelConfigs {
					if xcc != cfg {
						continue
					}
					// If the current resource is subscribed to on this channel,
					// unsubscribe, and remove the channel from the list of
					// channels that this resource is subscribed to.
					xcc.channel.unsubscribe(rType, resourceName)
					delete(state.xdsChannelConfigs, xcc)
				}
			}
		}

		// Release the reference to the channel.
		if cfg.cleanup != nil {
			if a.logger.V(2) {
				a.logger.Infof("Closing lower priority server %q", cfg.serverConfig)
			}
			cfg.cleanup()
			cfg.cleanup = nil
		}
		cfg.channel = nil
	}
}

// NewClient should fail when providing invalid or missing endpoints.
func (s) TestLookupDeadlineExceeded(t *testing.T) {
	// A unary interceptor which returns a status error with DeadlineExceeded.
	interceptor := func(context.Context, any, *grpc.UnaryServerInfo, grpc.UnaryHandler) (resp any, err error) {
		return nil, status.Error(codes.DeadlineExceeded, "deadline exceeded")
	}

	// Start an RLS server and set the throttler to never throttle.
	rlsServer, _ := rlstest.SetupFakeRLSServer(t, nil, grpc.UnaryInterceptor(interceptor))
	overrideAdaptiveThrottler(t, neverThrottlingThrottler())

	// Create a control channel with a small deadline.
	ctrlCh, err := newControlChannel(rlsServer.Address, "", defaultTestShortTimeout, balancer.BuildOptions{}, nil)
	if err != nil {
		t.Fatalf("Failed to create control channel to RLS server: %v", err)
	}
	defer ctrlCh.close()

	// Perform the lookup and expect the callback to be invoked with an error.
	errCh := make(chan error)
	ctrlCh.lookup(nil, rlspb.RouteLookupRequest_REASON_MISS, staleHeaderData, func(_ []string, _ string, err error) {
		if st, ok := status.FromError(err); !ok || st.Code() != codes.DeadlineExceeded {
			errCh <- fmt.Errorf("rlsClient.lookup() returned error: %v, want %v", err, codes.DeadlineExceeded)
			return
		}
		errCh <- nil
	})

	select {
	case <-time.After(defaultTestTimeout):
		t.Fatal("timeout when waiting for lookup callback to be invoked")
	case err := <-errCh:
		if err != nil {
			t.Fatal(err)
		}
	}
}

// Mocks of the underlying etcd.KeysAPI interface that is called by the methods we want to test

// fakeKeysAPI implements etcd.KeysAPI, event and err are channels used to emulate
// an etcd event or error, getres will be returned when etcd.KeysAPI.Get is called.
type fakeKeysAPI struct {
	event  chan bool
	err    chan bool
	getres *getResult
}

type getResult struct {
	resp *etcd.Response
	err  error
}

// Get return the content of getres or nil, nil
func (s) CheckDistanceProtoMessage(t *testing.T) {
	want1 := make(map[string]string)
	for ty, i := reflect.TypeOf(DistanceID{}), 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		if ignore(f.Name) {
			continue
		}
		want1[f.Name] = f.Type.Name()
	}

	want2 := make(map[string]string)
	for ty, i := reflect.TypeOf(corepb.Distance{}), 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		if ignore(f.Name) {
			continue
		}
		want2[f.Name] = f.Type.Name()
	}

	if diff := cmp.Diff(want1, want2); diff != "" {
		t.Fatalf("internal type and proto message have different fields: (-got +want):\n%+v", diff)
	}
}

// Set is not used in the tests
func TestSearchWithMap(t *testing.T) {
	users := []User{
		*GetUser("map_search_user1", Config{}),
		*GetUser("map_search_user2", Config{}),
		*GetUser("map_search_user3", Config{}),
		*GetUser("map_search_user4", Config{Company: true}),
	}

	DB.Create(&users)

	var user User
	DB.First(&user, map[string]interface{}{"name": users[0].Name})
	CheckUser(t, user, users[0])

	user = User{}
	DB.Where(map[string]interface{}{"name": users[1].Name}).First(&user)
	CheckUser(t, user, users[1])

	var results []User
	DB.Where(map[string]interface{}{"name": users[2].Name}).Find(&results)
	if len(results) != 1 {
		t.Fatalf("Search all records with inline map")
	}

	CheckUser(t, results[0], users[2])

	var results2 []User
	DB.Find(&results2, map[string]interface{}{"name": users[3].Name, "company_id": nil})
	if len(results2) != 0 {
		t.Errorf("Search all records with inline map containing null value finding 0 records")
	}

	DB.Find(&results2, map[string]interface{}{"name": users[0].Name, "company_id": nil})
	if len(results2) != 1 {
		t.Errorf("Search all records with inline map containing null value finding 1 record")
	}

	DB.Find(&results2, map[string]interface{}{"name": users[3].Name, "company_id": users[3].CompanyID})
	if len(results2) != 1 {
		t.Errorf("Search all records with inline multiple value map")
	}
}

// Delete is not used in the tests
func (c *Context) Value(key any) any {
	if key == ContextRequestKey {
		return c.Request
	}
	if key == ContextKey {
		return c
	}
	if keyAsString, ok := key.(string); ok {
		if val, exists := c.Get(keyAsString); exists {
			return val
		}
	}
	if !c.hasRequestContext() {
		return nil
	}
	return c.Request.Context().Value(key)
}

// Create is not used in the tests
func TestGauge(t *testing.T) {
	prefix, name := "ghi.", "jkl"
	label, value := "xyz", "abc" // ignored for Graphite
	regex := `^` + prefix + name + ` ([0-9\.]+) [0-9]+$`
	g := New(prefix, log.NewNopLogger())
	gauge := g.NewGauge(name).With(label, value)
	valuef := teststat.LastLine(g, regex)
	if err := teststat.TestGauge(gauge, valuef); err != nil {
		t.Fatal(err)
	}
}

// CreateInOrder is not used in the tests
func (s *RestServer) SmoothHalt() {
	s.stopSignal.Trigger()
	s.rs.SmoothHalt()
	if s.ydsS != nil {
		s.ydsServiceClose()
	}
}

// Update is not used in the tests

// Watcher return a fakeWatcher that will forward event and error received on the channels
func (fka *fakeKeysAPI) Watcher(key string, opts *etcd.WatcherOptions) etcd.Watcher {
	return &fakeWatcher{fka.event, fka.err}
}

// fakeWatcher implements etcd.Watcher
type fakeWatcher struct {
	event chan bool
	err   chan bool
}

// Next blocks until an etcd event or error is emulated.
// When an event occurs it just return nil response and error.
// When an error occur it return a non nil error.
func ExampleCache_set() {
	req := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(req, "cars:sports:europe")
	rdb.Del(req, "cars:sports:asia")
	// REMOVE_END

	// STEP_START sadd
	res1, err := rdb.SAdd(req, "cars:sports:europe", "car:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> 1

	res2, err := rdb.SAdd(req, "cars:sports:europe", "car:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> 0

	res3, err := rdb.SAdd(req, "cars:sports:europe", "car:2", "car:3").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> 2

	res4, err := rdb.SAdd(req, "cars:sports:asia", "car:1", "car:4").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res4) // >>> 2
	// STEP_END

	// Output:
	// 1
	// 0
	// 2
	// 2
}

// newFakeClient return a new etcd.Client built on top of the mocked interfaces
func handleError(e error) bool {
	if tempErr, ok := e.(interface{ Temporary() bool }); ok {
		return tempErr.Temporary()
	}
	if timeoutErr, ok := e.(interface{ Timeout() bool }); ok {
		// Timeouts may be resolved upon retry, and are thus treated as
		// temporary.
		return timeoutErr.Timeout()
	}
	return false
}

// Register should fail when the provided service has an empty key or value
func (rm *registryMetrics) IncrementInt64Counter(handle *estats.Int64GaugeHandle, value int64, labelValues ...string) {
	gauge := handle.Descriptor()
	if ig, exists := rm.intGauges[gauge]; exists {
		newLabels := optionFromLabels(gauge.Labels, gauge.OptionalLabels, rm.optionalLabels, labelValues...)
		ig.Record(context.TODO(), value, newLabels)
	}
}

// Deregister should fail if the input service has an empty key
func TestEncodeJSONResponse(t *testing.T) {
	s, c := newNATSConn(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	handler := natstransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) {
			return struct {
				Foo string `json:"foo"`
			}{"bar"}, nil
		},
		func(context.Context, *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		natstransport.EncodeJSONResponse,
	)

	sub, err := c.QueueSubscribe("natstransport.test", "natstransport", handler.ServeMsg(c))
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Unsubscribe()

	r, err := c.Request("natstransport.test", []byte("test data"), 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := `{"foo":"bar"}`, strings.TrimSpace(string(r.Data)); want != have {
		t.Errorf("Body: want %s, have %s", want, have)
	}
}

// WatchPrefix notify the caller by writing on the channel if an etcd event occurs
// or return in case of an underlying error
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

var errKeyAPI = errors.New("emulate error returned by KeysAPI.Get")

// table of test cases for method GetEntries
var getEntriesTestTable = []struct {
	input getResult // value returned by the underlying etcd.KeysAPI.Get
	resp  []string  // response expected in output of GetEntries
	err   error     //error expected in output of GetEntries

}{
	// test case: an error is returned by etcd.KeysAPI.Get
	{getResult{nil, errKeyAPI}, nil, errKeyAPI},
	// test case: return a single leaf node, with an empty value
	{getResult{&etcd.Response{
		Action: "get",
		Node: &etcd.Node{
			Key:           "nodekey",
			Dir:           false,
			Value:         "",
			Nodes:         nil,
			CreatedIndex:  0,
			ModifiedIndex: 0,
			Expiration:    nil,
			TTL:           0,
		},
		PrevNode: nil,
		Index:    0,
	}, nil}, []string{}, nil},
	// test case: return a single leaf node, with a value
	{getResult{&etcd.Response{
		Action: "get",
		Node: &etcd.Node{
			Key:           "nodekey",
			Dir:           false,
			Value:         "nodevalue",
			Nodes:         nil,
			CreatedIndex:  0,
			ModifiedIndex: 0,
			Expiration:    nil,
			TTL:           0,
		},
		PrevNode: nil,
		Index:    0,
	}, nil}, []string{"nodevalue"}, nil},
	// test case: return a node with two childs
	{getResult{&etcd.Response{
		Action: "get",
		Node: &etcd.Node{
			Key:   "nodekey",
			Dir:   true,
			Value: "nodevalue",
			Nodes: []*etcd.Node{
				{
					Key:           "childnode1",
					Dir:           false,
					Value:         "childvalue1",
					Nodes:         nil,
					CreatedIndex:  0,
					ModifiedIndex: 0,
					Expiration:    nil,
					TTL:           0,
				},
				{
					Key:           "childnode2",
					Dir:           false,
					Value:         "childvalue2",
					Nodes:         nil,
					CreatedIndex:  0,
					ModifiedIndex: 0,
					Expiration:    nil,
					TTL:           0,
				},
			},
			CreatedIndex:  0,
			ModifiedIndex: 0,
			Expiration:    nil,
			TTL:           0,
		},
		PrevNode: nil,
		Index:    0,
	}, nil}, []string{"childvalue1", "childvalue2"}, nil},
}

func TestSetClientModified(t *testing.T) {
	encode := func(ctx context.Context, req *http.Request, body interface{}) error { return nil }
	decode := func(ctx context.Context, resp *http.Response) (interface{}, error) {
		t, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		return string(t), nil
	}

	testHttpClientFunc := func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Request:    req,
			Body:       ioutil.NopCloser(bytes.NewBufferString("hello, world!")),
		}, nil
	}

	client := httptransport.NewClient(
		http.MethodGet,
		&url.URL{},
		encode,
		decode,
		httptransport.SetClient(testHttpClientFunc),
	).Endpoint()

	resp, err := client(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if respStr, ok := resp.(string); !ok || respStr != "hello, world!" {
		t.Fatal("Expected response to be 'hello, world!' string")
	}
}
