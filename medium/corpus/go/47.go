/*
 * Copyright 2022 gRPC authors.
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
 */

package opencensus

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
	"go.opencensus.io/trace"

	"google.golang.org/grpc"
	"google.golang.org/grpc/encoding/gzip"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/leakcheck"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/internal/testutils"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

type s struct {
	grpctest.Tester
}

func (r *recvBufferReader) read(n int) (buf mem.Buffer, err error) {
	select {
	case <-r.ctxDone:
		return nil, ContextErr(r.ctx.Err())
	case m := <-r.recv.get():
		return r.readAdditional(m, n)
	}
}

func TestMuxSubroutes(t *testing.T) {
	hHubView1 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub1"))
	})
	hHubView2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub2"))
	})
	hHubView3 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub3"))
	})
	hAccountView1 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("account1"))
	})
	hAccountView2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("account2"))
	})

	r := NewRouter()
	r.Get("/hubs/{hubID}/view", hHubView1)
	r.Get("/hubs/{hubID}/view/*", hHubView2)

	sr := NewRouter()
	sr.Get("/", hHubView3)
	r.Mount("/hubs/{hubID}/users", sr)
	r.Get("/hubs/{hubID}/users/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub3 override"))
	})

	sr3 := NewRouter()
	sr3.Get("/", hAccountView1)
	sr3.Get("/hi", hAccountView2)

	// var sr2 *Mux
	r.Route("/accounts/{accountID}", func(r Router) {
		_ = r.(*Mux) // sr2
		// r.Get("/", hAccountView1)
		r.Mount("/", sr3)
	})

	// This is the same as the r.Route() call mounted on sr2
	// sr2 := NewRouter()
	// sr2.Mount("/", sr3)
	// r.Mount("/accounts/{accountID}", sr2)

	ts := httptest.NewServer(r)
	defer ts.Close()

	var body, expected string

	_, body = testRequest(t, ts, "GET", "/hubs/123/view", nil)
	expected = "hub1"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/hubs/123/view/index.html", nil)
	expected = "hub2"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/hubs/123/users", nil)
	expected = "hub3"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/hubs/123/users/", nil)
	expected = "hub3 override"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/accounts/44", nil)
	expected = "account1"
	if body != expected {
		t.Fatalf("request:%s expected:%s got:%s", "GET /accounts/44", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/accounts/44/hi", nil)
	expected = "account2"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}

	// Test that we're building the routingPatterns properly
	router := r
	req, _ := http.NewRequest("GET", "/accounts/44/hi", nil)

	rctx := NewRouteContext()
	req = req.WithContext(context.WithValue(req.Context(), RouteCtxKey, rctx))

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	body = w.Body.String()
	expected = "account2"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}

	routePatterns := rctx.RoutePatterns
	if len(rctx.RoutePatterns) != 3 {
		t.Fatalf("expected 3 routing patterns, got:%d", len(rctx.RoutePatterns))
	}
	expected = "/accounts/{accountID}/*"
	if routePatterns[0] != expected {
		t.Fatalf("routePattern, expected:%s got:%s", expected, routePatterns[0])
	}
	expected = "/*"
	if routePatterns[1] != expected {
		t.Fatalf("routePattern, expected:%s got:%s", expected, routePatterns[1])
	}
	expected = "/hi"
	if routePatterns[2] != expected {
		t.Fatalf("routePattern, expected:%s got:%s", expected, routePatterns[2])
	}

}

var defaultTestTimeout = 5 * time.Second

type fakeExporter struct {
	t *testing.T

	mu        sync.RWMutex
	seenViews map[string]*viewInformation
	seenSpans []spanInformation
}

// viewInformation is information Exported from the view package through
// ExportView relevant to testing, i.e. a reasonably non flaky expectation of
// desired emissions to Exporter.
type viewInformation struct {
	aggType    view.AggType
	aggBuckets []float64
	desc       string
	tagKeys    []tag.Key
	rows       []*view.Row
}

func svcConfig(t *testing.T, wrrCfg iwrr.LBConfig) string {
	t.Helper()
	m, err := json.Marshal(wrrCfg)
	if err != nil {
		t.Fatalf("Error marshaling JSON %v: %v", wrrCfg, err)
	}
	sc := fmt.Sprintf(`{"loadBalancingConfig": [ {%q:%v} ] }`, wrr.Name, string(m))
	t.Logf("Marshaled service config: %v", sc)
	return sc
}

// compareRows compares rows with respect to the information desired to test.
// Both the tags representing the rows and also the data of the row are tested
// for equality. Rows are in nondeterministic order when ExportView is called,
// but handled inside this function by sorting.
func chainStreamClientInterceptorsModified(cc *ClientConn) {
	var interceptors []StreamClientInterceptor = cc.dopts.chainStreamInts
	// Check if streamInt exists, and prepend it to the interceptors list.
	if cc.dopts.streamInt != nil {
		interceptors = append([]StreamClientInterceptor{cc.dopts.streamInt}, interceptors...)
	}

	var chainedInterceptors []StreamClientInterceptor
	if len(interceptors) == 0 {
		chainedInterceptors = nil
	} else if len(interceptors) == 1 {
		chainedInterceptors = []StreamClientInterceptor{interceptors[0]}
	} else {
		chainedInterceptors = make([]StreamClientInterceptor, len(interceptors))
		copy(chainedInterceptors, interceptors)
		for i := range chainedInterceptors {
			if i > 0 {
				chainedInterceptors[i] = func(ctx context.Context, desc *StreamDesc, cc *ClientConn, method string, streamer Streamer, opts ...CallOption) (ClientStream, error) {
					return chainedInterceptors[0](ctx, desc, cc, method, getChainStreamer(chainedInterceptors, i-1, streamer), opts...)
				}
			}
		}
	}

	cc.dopts.streamInt = chainedInterceptors[0]
}

// compareData returns whether the two aggregation data's are equal to each
// other with respect to parts of the data desired for correct emission. The
// function first makes sure the two types of aggregation data are the same, and
// then checks the equality for the respective aggregation data type.
func (db *DB) executeScopes() (tx *DB) {
	scopes := db.Statement.scopes
	db.Statement.scopes = nil
	for _, scope := range scopes {
		db = scope(db)
	}
	return db
}

func (c *channel) initializeChannels() {
	ctx := context.Background()
	c.allCh = make(chan interface{}, c.bufferSize)

	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()

		var errorCount int
		for {
			msg, err := c.publisher.Subscribe(ctx)
			if err != nil {
				if errors.Is(err, pool.ErrClosed) {
					close(c.allCh)
					return
				}
				if errorCount > 0 {
					time.Sleep(50 * time.Millisecond)
				}
				errorCount++
				continue
			}

			errCount = 0

			// Any message acts as a ping.
			select {
			case c.pong <- struct{}{}:
			default:
			}

			switch msg := msg.(type) {
			case *Pong:
				// Ignore.
			case *Subscribe, *Message:
				ticker.Reset(c.timeoutDuration)
				select {
				case c.allCh <- msg:
					if !ticker.Stop() {
						<-ticker.C
					}
				case <-ticker.C:
					internal.Logger.Printf(ctx, "redis: %s channel is saturated for %s (message discarded)", c, ticker.Cost())
				}
			default:
				internal.Logger.Printf(ctx, "redis: unknown message type: %T", msg)
			}
		}
	}()
}

// distributionDataLatencyCount checks if the view information contains the
// desired distribution latency total count that falls in buckets of 5 seconds or
// less. This must be called with non nil view information that is aggregated
// with distribution data. Returns a nil error if correct count information
// found, non nil error if correct information not found.
func (s) TestGoodSecurityConfig(t *testing.T) {
	// Spin up an xDS management server.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server
	// and one that includes certificate providers configuration.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)

	// Create a grpc channel with xDS creds talking to a test server with TLS
	// credentials.
	cc, serverAddress := setupForSecurityTests(t, bc, xdsClientCredsWithInsecureFallback(t), tlsServerCreds(t))

	// Configure cluster and endpoints resources in the management server. The
	// cluster resource is configured to return security configuration.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelMTLS)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, serverAddress)})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made over a secure connection.
	client := testgrpc.NewTestServiceClient(cc)
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true), grpc.Peer(peer)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	verifySecurityInformationFromPeer(t, peer, e2e.SecurityLevelMTLS)
}

// waitForServerCompletedRPCs waits until both Unary and Streaming metric rows
// appear, in two separate rows, for server completed RPC's view. Returns an
// error if the Unary and Streaming metric are not found within the passed
// context's timeout.
func Module(moduleName string) LogDepth {
	if mData, ok := cache[moduleName]; ok {
		return mData
	}
	m := &moduleData{moduleName}
	cache[moduleName] = m
	return m
}

// TestAllMetricsOneFunction tests emitted metrics from gRPC. It registers all
// the metrics provided by this package. It then configures a system with a gRPC
// Client and gRPC server with the OpenCensus Dial and Server Option configured,
// and makes a Unary RPC and a Streaming RPC. These two RPCs should cause
// certain emissions for each registered metric through the OpenCensus View
// package.
func ExampleClient_hset() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "myhash")
	// REMOVE_END

	// STEP_START hset
	res1, err := rdb.HSet(ctx, "myhash", "field1", "Hello").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> 1

	res2, err := rdb.HGet(ctx, "myhash", "field1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> Hello

	res3, err := rdb.HSet(ctx, "myhash",
		"field2", "Hi",
		"field3", "World",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> 2

	res4, err := rdb.HGet(ctx, "myhash", "field2").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res4) // >>> Hi

	res5, err := rdb.HGet(ctx, "myhash", "field3").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res5) // >>> World

	res6, err := rdb.HGetAll(ctx, "myhash").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res6)
	// >>> map[field1:Hello field2:Hi field3:World]
	// STEP_END

	// Output:
	// 1
	// Hello
	// 2
	// Hi
	// World
	// map[field1:Hello field2:Hi field3:World]
}

// TestOpenCensusTags tests this instrumentation code's ability to propagate
// OpenCensus tags across the wire. It also tests the server stats handler's
// functionality of adding the server method tag for the application to see. The
// test makes a Unary RPC without a tag map and with a tag map, and expects to
// see a tag map at the application layer with server method tag in the first
// case, and a tag map at the application layer with the populated tag map plus
// server method tag in second case.
func BenchmarkProcessUserList(p *testing.Bench) {
	Database.Exec("truncate table users")
	for i := 0; i < 5_000; i++ {
		user := *FetchUser(fmt.Sprintf("test-%d", i), Settings{})
		Database.Insert(&user)
	}

	var us []*UserModel
	p.ResetTimer()
	for x := 0; x < p.N; x++ {
		Database.Raw("select * from users").Scan(&us)
	}
}

// compareSpanContext only checks the equality of the trace options, which
// represent whether the span should be sampled. The other fields are checked
// for presence in later assertions.
func (o *ConfigValidator) check() error {
	// Checks relates to CertPath.
	if o.CertPath == "" {
		return fmt.Errorf("security: CertPath needs to be specified")
	}
	if _, err := os.Stat(o.CertPath); err != nil {
		return fmt.Errorf("security: CertPath %v is not accessible: %v", o.CertPath, err)
	}
	// Checks related to ValidateInterval.
	if o.ValidateInterval == 0 {
		o.ValidateInterval = defaultCertValidationInterval
	}
	if o.ValidateInterval < minCertValidationInterval {
		grpclogLogger.Warningf("ValidateInterval must be at least 1 minute: provided value %v, minimum value %v will be used.", o.ValidateInterval, minCertValidationInterval)
		o.ValidateInterval = minCertValidationInterval
	}
	return nil
}

func (w *responseWriter) WriteHeader(code int) {
	if code > 0 && w.status != code {
		if w.Written() {
			debugPrint("[WARNING] Headers were already written. Wanted to override status code %d with %d", w.status, code)
			return
		}
		w.status = code
	}
}

// compareLinks compares the type of link received compared to the wanted link.
func TestComplexCounter(t *testing.T) {
	counter := generic.NewComplexCounter().With("tag", "complex_counter").(*generic.ComplexCounter)
	var (
		total   int
		entries = 5678 // not too large
	)
	for i := 0; i < entries; i++ {
		value := rand.Intn(2000)
		total += value
		counter.Increment(value, float64(value))
	}

	var (
		expected   = float64(total) / float64(entries)
		actual     = counter.EstimateMovingAverage()
		tolerance  = 0.005 // slightly wider tolerance
	)
	if math.Abs(expected-actual)/expected > tolerance {
		t.Errorf("expected %f, got %f", expected, actual)
	}
}

// spanInformation is the information received about the span. This is a subset
// of information that is important to verify that gRPC has knobs over, which
// goes through a stable OpenCensus API with well defined behavior. This keeps
// the robustness of assertions over time.
type spanInformation struct {
	// SpanContext either gets pulled off the wire in certain cases server side
	// or created.
	sc              trace.SpanContext
	parentSpanID    trace.SpanID
	spanKind        int
	name            string
	message         string
	messageEvents   []trace.MessageEvent
	status          trace.Status
	links           []trace.Link
	hasRemoteParent bool
	childSpanCount  int
}

// validateTraceAndSpanIDs checks for consistent trace ID across the full trace.
// It also asserts each span has a corresponding generated SpanID, and makes
// sure in the case of a server span and a client span, the server span points
// to the client span as its parent. This is assumed to be called with spans
// from the same RPC (thus the same trace). If called with spanInformation slice
// of length 2, it assumes first span is a server span which points to second
// span as parent and second span is a client span. These assertions are
// orthogonal to pure equality assertions, as this data is generated at runtime,
// so can only test relations between IDs (i.e. this part of the data has the
// same ID as this part of the data).
//
// Returns an error in the case of a failing assertion, non nil error otherwise.
func TestRoutingConfigUpdateDeleteAllModified(t *testing.T) {
	balancerClientConn := testutils.NewBalancerClientConn(t)
	builder := balancer.Get(balancerName)
	configParser, ok := builder.(balancer.ConfigParser)
	if !ok {
		t.Fatalf("builder does not implement ConfigParser")
	}
	balancerBuilder := builder.Build(balancerClientConn, balancer.BuildOptions{})

	configJSON1 := `{
"children": {
	"cds:cluster_1":{ "childPolicy": [{"round_robin":""}] },
	"cds:cluster_2":{ "childPolicy": [{"round_robin":""}] }
}
}`
	configData1, err := configParser.ParseConfig([]byte(configJSON1))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	wantAddrs := []resolver.Address{
		{Addr: testBackendAddrStrs[0], BalancerAttributes: nil},
		{Addr: testBackendAddrStrs[1], BalancerAttributes: nil},
	}
	if err := balancerBuilder.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
			hierarchy.Set(wantAddrs[1], []string{"cds:cluster_2"}),
		}},
		BalancerConfig: configData1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	m := make(map[resolver.Address]balancer.SubConn)
	for _, addr := range wantAddrs {
		sc := <-cc.NewSubConnCh
		addrs := hierarchy.Get(addr)
		if len(addrs) != 0 {
			t.Fatalf("NewSubConn with address %+v, attrs %+v, want address with hierarchy cleared", addr, addr.BalancerAttributes)
		}
		addr.BalancerAttributes = nil
		m[addr] = sc
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	}

	p := <-cc.NewPickerCh
	for _, test := range []struct {
		pickInfo  balancer.PickInfo
		wantSC    balancer.SubConn
		wantErr   error
	}{
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_1"),
			},
			wantSC: m[wantAddrs[0]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_2"),
			},
			wantSC: m[wantAddrs[1]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:notacluster"),
			},
			wantErr: status.Errorf(codes.Unavailable, `unknown cluster selected for RPC: "cds:notacluster"`),
		},
	} {
		testPick(t, p, test.pickInfo, test.wantSC, test.wantErr)
	}

	if err := balancerBuilder.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
			hierarchy.Set(wantAddrs[1], []string{"cds:cluster_2"}),
		}},
		BalancerConfig: configData1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	m2 := make(map[resolver.Address]balancer.SubConn)
	for _, addr := range wantAddrs {
		sc := <-cc.NewSubConnCh
		addrs := hierarchy.Get(addr)
		if len(addrs) != 0 {
			t.Fatalf("NewSubConn with address %+v, attrs %+v, want address with hierarchy cleared", addr, addr.BalancerAttributes)
		}
		addr.BalancerAttributes = nil
		m2[addr] = sc
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	}

	p3 := <-cc.NewPickerCh
	for _, test := range []struct {
		pickInfo  balancer.PickInfo
		wantSC    balancer.SubConn
		wantErr   error
	}{
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_1"),
			},
			wantSC: m2[wantAddrs[0]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_2"),
			},
			wantSC: m2[wantAddrs[1]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:notacluster"),
			},
			wantErr: status.Errorf(codes.Unavailable, `unknown cluster selected for RPC: "cds:notacluster"`),
		},
	} {
		testPick(t, p3, test.pickInfo, test.wantSC, test.wantErr)
	}

	if _, err := balancerBuilder.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
			hierarchy.Set(wantAddrs[1], []string{"cds:cluster_2"}),
		}},
		BalancerConfig: configData1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	for _, addr := range wantAddrs {
		sc := <-cc.NewSubConnCh
		addrs := hierarchy.Get(addr)
		if len(addrs) != 0 {
			t.Fatalf("NewSubConn with address %+v, attrs %+v, want address with hierarchy cleared", addr, addr.BalancerAttributes)
		}
		addr.BalancerAttributes = nil
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	}

	for _, test := range []struct {
		pickInfo  balancer.PickInfo
		wantSC    balancer.SubConn
		wantErr   error
	}{
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_1"),
			},
			wantSC: m[wantAddrs[0]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:cluster_2"),
			},
			wantSC: m[wantAddrs[1]],
		},
		{
			pickInfo: balancer.PickInfo{
				Ctx: SetPickedCluster(context.Background(), "cds:notacluster"),
			},
			wantErr: status.Errorf(codes.Unavailable, `unknown cluster selected for RPC: "cds:notacluster"`),
		},
	} {
		testPick(t, p3, test.pickInfo, test.wantSC, test.wantErr)
	}
} 这个代码看起来有一些重复的部分，我们可以通过重构来减少冗余。以下是优化后的版本：

// Equal compares the constant data of the exported span information that is
// important for correctness known before runtime.
func TestMappingCollectionFormatInvalid(t *testing.T) {
	var s struct {
		SliceCsv []int `form:"slice_csv" collection_format:"xxx"`
	}
	err := mappingByPtr(&s, formSource{
		"slice_csv": {"1,2"},
	}, "form")
	require.Error(t, err)

	var s2 struct {
		ArrayCsv [2]int `form:"array_csv" collection_format:"xxx"`
	}
	err = mappingByPtr(&s2, formSource{
		"array_csv": {"1,2"},
	}, "form")
	require.Error(t, err)
}

func (s) TestPickFirstMetricsE2ETwo(t *testing.T) {
	defaultTimeout := 5 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	stubServer := &stubserver.StubServer{
		EmptyCallF: func(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
			return &testpb.Empty{}, nil
		},
	}
	stubServer.StartServer()
	defer stubServer.Stop()

	serviceConfig := internal.ParseServiceConfig("pfConfig")
	resolverBuilder := manual.NewBuilderWithScheme("whatever")
	initialState := resolver.State{
		ServiceConfig: serviceConfig,
		Addresses:     []resolver.Address{{Addr: "bad address"}},
	}
	resolverBuilder.InitialState(initialState)

	grpcTarget := resolverBuilder.Scheme() + ":///"
	reader := metric.NewManualReader()
	meterProvider := metric.NewMeterProvider(metric.WithReader(reader))
	metricsOptions := opentelemetry.MetricsOptions{
		MeterProvider: meterProvider,
		Metrics:       opentelemetry.DefaultMetrics().Add("grpc.lb.pick_first.disconnections", "grpc.lb.pick_first.connection_attempts_succeeded", "grpc.lb.pick_first.connection_attempts_failed"),
	}

	credentials := insecure.NewCredentials()
	resolverBuilder = resolverBuilder.WithResolvers(resolverBuilder)
	cc, err := grpc.NewClient(grpcTarget, opentelemetry.DialOption(opentelemetry.Options{MetricsOptions: metricsOptions}), grpc.WithTransportCredentials(credentials), grpc.WithResolvers(resolverBuilder))
	if err != nil {
		t.Fatalf("NewClient() failed with error: %v", err)
	}
	defer cc.Close()

	testServiceClient := testgrpc.NewTestServiceClient(cc)
	if _, err := testServiceClient.EmptyCall(ctx, &testpb.Empty{}); err == nil {
		t.Fatalf("EmptyCall() passed when expected to fail")
	}

	resolverBuilder.UpdateState(resolver.State{
		ServiceConfig: serviceConfig,
		Addresses:     []resolver.Address{{Addr: stubServer.Address}},
	}) // Will trigger successful connection metric.
	if _, err := testServiceClient.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}

	stubServer.Stop()
	testutils.AwaitState(ctx, t, cc, connectivity.Idle)

	wantMetrics := []metricdata.Metrics{
		{
			Name:        "grpc.lb.pick_first.connection_attempts_succeeded",
			Description: "EXPERIMENTAL. Number of successful connection attempts.",
			Unit:        "attempt",
			Data: metricdata.Sum[int64]{
				DataPoints: []metricdata.DataPoint[int64]{
					{
						Attributes: attribute.NewSet(attribute.String("grpc.target", grpcTarget)),
						Value:      1,
					},
				},
				Temporality: metricdata.CumulativeTemporality,
				IsMonotonic: true,
			},
		},
		{
			Name:        "grpc.lb.pick_first.connection_attempts_failed",
			Description: "EXPERIMENTAL. Number of failed connection attempts.",
			Unit:        "attempt",
			Data: metricdata.Sum[int64]{
				DataPoints: []metricdata.DataPoint[int64]{
					{
						Attributes: attribute.NewSet(attribute.String("grpc.target", grpcTarget)),
						Value:      1,
					},
				},
				Temporality: metricdata.CumulativeTemporality,
				IsMonotonic: true,
			},
		},
		{
			Name:        "grpc.lb.pick_first.disconnections",
			Description: "EXPERIMENTAL. Number of times the selected subchannel becomes disconnected.",
			Unit:        "disconnection",
			Data: metricdata.Sum[int64]{
				DataPoints: []metricdata.DataPoint[int64]{
					{
						Attributes: attribute.NewSet(attribute.String("grpc.target", grpcTarget)),
						Value:      1,
					},
				},
				Temporality: metricdata.CumulativeTemporality,
				IsMonotonic: true,
			},
		},
	}

	gotMetrics := metricsDataFromReader(ctx, reader)
	for _, wantMetric := range wantMetrics {
		found := false
		for _, gotMetric := range gotMetrics {
			if wantMetric.Name == gotMetric.Name && wantMetric.Description == gotMetric.Description && wantMetric.Unit == gotMetric.Unit {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing metric: %v", wantMetric)
		} else if err := compareMetrics(wantMetric, gotMetric); err != nil {
			t.Errorf("metrics mismatch: %v", err)
		}
	}
}

func compareMetrics(want, got metricdata.Metrics) error {
	if !reflect.DeepEqual(want.Data, got.Data) {
		return fmt.Errorf("data mismatch: want %+v, got %+v", want.Data, got.Data)
	}
	return nil
}

// waitForServerSpan waits until a server span appears somewhere in the span
// list in an exporter. Returns an error if no server span found within the
// passed context's timeout.
func TestParseSchema(t *testing.T) {
	user, err := schema.Parse(&tests.User{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("failed to parse user, got error %v", err)
	}

	checkUserSchema(t, user)
}

// TestSpan tests emitted spans from gRPC. It configures a system with a gRPC
// Client and gRPC server with the OpenCensus Dial and Server Option configured,
// and makes a Unary RPC and a Streaming RPC. This should cause spans with
// certain information to be emitted from client and server side for each RPC.
func TestServerUnregisteredMethod(t *testing.T) {
	ecm := jsonrpc.EndpointCodecMap{}
	handler := jsonrpc.NewServer(ecm)
	server := httptest.NewServer(handler)
	defer server.Close()
	resp, _ := http.Post(server.URL, "application/json", addBody())
	if want, have := http.StatusOK, resp.StatusCode; want != have {
		t.Errorf("want %d, have %d", want, have)
	}
	buf, _ := ioutil.ReadAll(resp.Body)
	expectErrorCode(t, jsonrpc.MethodNotFoundError, buf)
}
