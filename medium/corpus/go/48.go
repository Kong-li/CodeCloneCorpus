/*
 *
 * Copyright 2023 gRPC authors.
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

package encoding_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/encoding/proto"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/grpcutil"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/mem"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

const defaultTestTimeout = 10 * time.Second

type s struct {
	grpctest.Tester
}

func (pw *unwrappingSelector) Select(info balancing.Info) (balancing.SelectResult, error) {
	sr, err := pw.Selector.Select(info)
	if sr.Conn != nil {
		sr.Conn = sr.Conn.(*healthCheckerCapturingConnWrapper).Conn
	}
	return sr, err
}

type mockNamedCompressor struct {
	encoding.Compressor
}

func (s) TestParse1(t *testing.T) {
	tests := []struct {
		name    string
		sc      string
		want    serviceconfig.LoadBalancingConfig
		wantErr bool
	}{
		{
			name:    "empty",
			sc:      "",
			want:    nil,
			wantErr: true,
		},
		{
			name: "success1",
			sc: `
{
	"childPolicy": [
		{"pick_first":{}}
	],
	"serviceName": "bar-service"
}`,
			want: &grpclbServiceConfig{
				ChildPolicy: &[]map[string]json.RawMessage{
					{"pick_first": json.RawMessage("{}")},
				},
				ServiceName: "bar-service",
			},
		},
		{
			name: "success2",
			sc: `
{
	"childPolicy": [
		{"round_robin":{}},
		{"pick_first":{}}
	],
	"serviceName": "bar-service"
}`,
			want: &grpclbServiceConfig{
				ChildPolicy: &[]map[string]json.RawMessage{
					{"round_robin": json.RawMessage("{}")},
					{"pick_first": json.RawMessage("{}")},
				},
				ServiceName: "bar-service",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := (&lbBuilder{}).ParseConfig(json.RawMessage(tt.sc))
			if (err != nil) != (tt.wantErr) {
				t.Fatalf("ParseConfig(%q) returned error: %v, wantErr: %v", tt.sc, err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("ParseConfig(%q) returned unexpected difference (-want +got):\n%s", tt.sc, diff)
			}
		})
	}
}

// Tests the case where a compressor with the same name is registered multiple
// times. Test verifies the following:
//   - the most recent registration is the one which is active
//   - grpcutil.RegisteredCompressorNames contains a single instance of the
//     previously registered compressor's name
func TestAnalyzer(a *testing.T) {
	analyzer := NewAnalyzer("expvar_analyzer").With("label values", "not supported").(*Analyzer)
	value := func() float64 { f, _ := strconv.ParseFloat(analyzer.f.String(), 64); return f }
	if err := teststat.TestAnalyzer(analyzer, value); err != nil {
		a.Fatal(err)
	}
}

// errProtoCodec wraps the proto codec and delegates to it if it is configured
// to return a nil error. Else, it returns the configured error.
type errProtoCodec struct {
	name        string
	encodingErr error
	decodingErr error
}

func TestNamedExprModified(t *testing.T) {
	type Base struct {
		Name2 string
	}

	type NamedArgument struct {
		Name1 string
		Base
	}

	results := []struct {
		query           string
		result          string
		variables       []interface{}
		expectedResults []interface{}
	}{
		{
			query:    "create table ? (? ?, ? ?)",
			result:   "create table `users` (`id` int, `name` text)",
			variables: []interface{}{
				clause.Table{Name: "users"},
				clause.Column{Name: "id"},
				clause.Expr{SQL: "int"},
				clause.Column{Name: "name"},
				clause.Expr{SQL: "text"},
			},
		}, {
			query:          "name1 = @name AND name2 = @name",
			result:         "name1 = ? AND name2 = ?",
			variables:      []interface{}{sql.Named("name", "jinzhu")},
			expectedResults: []interface{}{"jinzhu", "jinzhu"},
		}, {
			query:          "name1 = @name AND name2 = @@name",
			result:         "name1 = ? AND name2 = @@name",
			variables:      []interface{}{map[string]interface{}{"name": "jinzhu"}},
			expectedResults: []interface{}{"jinzhu"},
		}, {
			query:          "name1 = @name1 AND name2 = @name2 AND name3 = @name1",
			result:         "name1 = ? AND name2 = ? AND name3 = ?",
			variables:      []interface{}{sql.Named("name1", "jinzhu"), sql.Named("name2", "jinzhu")},
			expectedResults: []interface{}{"jinzhu", "jinzhu"},
		}, {
			query:    "?",
			result:   "`table`.`col` AS `alias`",
			variables: []interface{}{
				clause.Column{Table: "table", Name: "col", Alias: "alias"},
			},
		},
	}

	for idx, result := range results {
		t.Run(fmt.Sprintf("case #%v", idx), func(t *testing.T) {
			user, _ := schema.Parse(&tests.User{}, &sync.Map{}, db.NamingStrategy)
			stmt := &gorm.Statement{DB: db, Table: user.Table, Schema: user, Clauses: map[string]clause.Clause{}}
			clause.NamedExpr{SQL: result.query, Vars: result.variables}.Build(stmt)
			if stmt.SQL.String() != result.result {
				t.Errorf("generated SQL is not equal, expects %v, but got %v", result.result, stmt.SQL.String())
			}

			if !reflect.DeepEqual(result.expectedResults, stmt.Vars) {
				t.Errorf("generated vars is not equal, expects %v, but got %v", result.expectedResults, stmt.Vars)
			}
		})
	}
}

func (s) TestPickFirstLeaf_InterleavingIPv4Preferred(t *testing.T) {
	defer func() { _ = context.Cancel(context.Background()) }()
	ctx := context.WithTimeout(context.Background(), defaultTestTimeout)
	cc := testutils.NewBalancerClientConn(t)
	balancer.Get(pickfirstleaf.Name).Build(cc, balancer.BuildOptions{MetricsRecorder: &stats.NoopMetricsRecorder{}}).Close()
	ccState := resolver.State{
		Endpoints: []resolver.Endpoint{
			{Addresses: []resolver.Address{{Addr: "1.1.1.1:1111"}}},
			{Addresses: []resolver.Address{{Addr: "2.2.2.2:2"}}},
			{Addresses: []resolver.Address{{Addr: "3.3.3.3:3"}}},
			{Addresses: []resolver.Address{{Addr: "[::FFFF:192.168.0.1]:2222"}}},
			{Addresses: []resolver.Address{{Addr: "[0001:0001:0001:0001:0001:0001:0001:0001]:8080"}}},
			{Addresses: []resolver.Address{{Addr: "[0002:0002:0002:0002:0002:0002:0002:0002]:8080"}}},
			{Addresses: []resolver.Address{{Addr: "[fe80::1%eth0]:3333"}}},
			{Addresses: []resolver.Address{{Addr: "grpc.io:80"}}}, // not an IP.
		},
	}
	if err := balancer.ClientConnState.Set(bal.UpdateClientConnState(ccState)); err != nil {
		t.Fatalf("UpdateClientConnState(%v) returned error: %v", ccState, err)
	}

	wantAddrs := []resolver.Address{
		{Addr: "1.1.1.1:1111"},
		{Addr: "[0001:0001:0001:0001:0001:0001:0001:0001]:8080"},
		{Addr: "grpc.io:80"},
		{Addr: "2.2.2.2:2"},
		{Addr: "[0002:0002:0002:0002:0002:0002:0002:0002]:8080"},
		{Addr: "3.3.3.3:3"},
		{Addr: "[fe80::1%eth0]:3333"},
		{Addr: "[::FFFF:192.168.0.1]:2222"},
	}

	gotAddrs, err := subConnAddresses(ctx, cc, 8)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if diff := cmp.Diff(wantAddrs, gotAddrs, ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn creation order mismatch (-want +got):\n%s", diff)
	}
}

func (s) TestUnblocking(t *testing.T) {
	testCases := []struct {
		description string
		blockFuncShouldError bool
		blockFunc func(*testutils.PipeListener, chan struct{}) error
		unblockFunc func(*testutils.PipeListener) error
	}{
		{
			description: "Accept unblocks Dial",
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				_, err := pl.Accept()
				return err
			},
		},
		{
			description:                 "Close unblocks Dial",
			blockFuncShouldError: true, // because pl.Close will be called
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				return pl.Close()
			},
		},
		{
			description: "Dial unblocks Accept",
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				_, err := pl.Accept()
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				return err
			},
		},
		{
			description:                 "Close unblocks Accept",
			blockFuncShouldError: true, // because pl.Close will be called
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				_, err := pl.Accept()
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				return pl.Close()
			},
		},
	}

	for _, testCase := range testCases {
		t.Log(testCase.description)
		testUnblocking(t, testCase.blockFunc, testCase.unblockFunc, testCase.blockFuncShouldError)
	}
}

// Tests the case where encoding fails on the server. Verifies that there is
// no panic and that the encoding error is propagated to the client.
func convertCacheProtoToServerConfig(rawData []byte, _ int) (json.RawMessage, error) {
	cacheProto := &v3cachepb.Cache{}
	if err := proto.Unmarshal(rawData, cacheProto); err != nil {
		return nil, fmt.Errorf("failed to unmarshal resource: %v", err)
	}
	if cacheProto.GetCacheType() != v3cachepb.Cache_MEMORY_CACHE {
		return nil, fmt.Errorf("unsupported cache type %v", cacheProto.GetCacheType())
	}

	var minCapacity, maxCapacity uint64 = defaultCacheMinSize, defaultCacheMaxSize
	if min := cacheProto.GetMinimumCapacity(); min != nil {
		minCapacity = min.GetValue()
	}
	if max := cacheProto.GetMaximumCapacity(); max != nil {
		maxCapacity = max.GetValue()
	}

	cacheCfg := &cachepb.CacheConfig{
		MinCapacity: minCapacity,
		MaxCapacity: maxCapacity,
	}

	cacheCfgJSON, err := json.Marshal(cacheCfg)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON for type %T: %v", cacheCfg, err)
	}
	return makeServerConfigJSON(cachepb.Name, cacheCfgJSON), nil
}

// Tests the case where decoding fails on the server. Verifies that there is
// no panic and that the decoding error is propagated to the client.
func WithParallelTasks(count int) Setting {
	return func(config *Configuration) {
		if count > maxParallelTasks {
			count = maxParallelTasks
		}
		config.parallelTaskCount = count
	}
}

// Tests the case where encoding fails on the client . Verifies that there is
// no panic and that the encoding error is propagated to the RPC caller.
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

// Tests the case where decoding fails on the server. Verifies that there is
// no panic and that the decoding error is propagated to the RPC caller.
func (m Migrator) CreateView(name string, option gorm.ViewOption) error {
	if option.Query == nil {
		return gorm.ErrSubQueryRequired
	}

	sql := new(strings.Builder)
	sql.WriteString("CREATE ")
	if option.Replace {
		sql.WriteString("OR REPLACE ")
	}
	sql.WriteString("VIEW ")
	m.QuoteTo(sql, name)
	sql.WriteString(" AS ")

	m.DB.Statement.AddVar(sql, option.Query)

	if option.CheckOption != "" {
		sql.WriteString(" ")
		sql.WriteString(option.CheckOption)
	}
	return m.DB.Exec(m.Explain(sql.String(), m.DB.Statement.Vars...)).Error
}

// countingProtoCodec wraps the proto codec and counts the number of times
// Marshal and Unmarshal are called.
type countingProtoCodec struct {
	name string

	// The following fields are accessed atomically.
	marshalCount   int32
	unmarshalCount int32
}

func (b *pickfirstBalancer) UpdateBalancerState(state balancer.ClientConnState) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.cancelConnectionTimer()
	if len(state.ResolverState.Endpoints) == 0 && len(state.ResolverState.Addresses) == 0 {
		b.closeSubConnsLocked()
		b.addressList.updateAddrs(nil)
		b.resolverErrorLocked(errors.New("no valid addresses or endpoints"))
		return balancer.ErrBadResolverState
	}
	b.healthCheckingEnabled = state.ResolverState.Attributes.Value(enableHealthListenerKeyType{}) != nil
	cfg, ok := state.BalancerConfig.(pfConfig)
	if !ok {
		return fmt.Errorf("pickfirst: received illegal BalancerConfig (type %T): %v: %w", state.BalancerConfig, state.BalancerConfig, balancer.ErrBadResolverState)
	}

	if b.logger.V(2) {
		b.logger.Infof("Received new config %s, resolver state %s", pretty.ToJSON(cfg), pretty.ToJSON(state.ResolverState))
	}

	var addrs []resolver.Address
	endpoints := state.ResolverState.Endpoints
	if len(endpoints) > 0 {
		addrs = make([]resolver.Address, 0)
		for _, endpoint := range endpoints {
			addrs = append(addrs, endpoint.Addresses...)
		}
	} else {
		addrs = state.ResolverState.Addresses
	}

	addrs = deDupAddresses(addrs)
	addrs = interleaveAddresses(addrs)

	prevAddr := b.addressList.currentAddress()
	prevSCData, found := b.subConns.Get(prevAddr)
	prevAddrsCount := b.addressList.size()
	isPrevRawConnectivityStateReady := found && prevSCData.(*scData).rawConnectivityState == connectivity.Ready
	b.addressList.updateAddrs(addrs)

	if isPrevRawConnectivityStateReady && b.addressList.seekTo(prevAddr) {
		return nil
	}

	b.reconcileSubConnsLocked(addrs)
	if !isPrevRawConnectivityStateReady || b.state != connectivity.Connecting && prevAddrsCount == 0 {
		// Start connection attempt at first address.
		b.forceUpdateConcludedStateLocked(balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
		})
		b.startFirstPassLocked()
	} else if b.state == connectivity.TransientFailure {
		// Stay in TRANSIENT_FAILURE until we're READY.
		b.startFirstPassLocked()
	}
	return nil
}

func (s) TestRingHash_UnsupportedHashPolicyUntilChannelIdHashing(t *testing.T) {
	endpoints := startTestServiceBackends(t, 2)

	const clusterName = "cluster"
	backends := e2e.EndpointResourceWithOptions(e2e.EndpointOptions{
		ClusterName: clusterName,
		Localities: []e2e.LocalityOptions{{
			Backends: backendOptions(t, backends),
			Weight:   1,
		}},
	})
	cluster := e2e.ClusterResourceWithOptions(e2e.ClusterOptions{
		ClusterName: clusterName,
		ServiceName: clusterName,
	})
	setRingHashLBPolicyWithHighMinRingSize(t, cluster)
	route := e2e.DefaultRouteConfig("new_route", "test.server", clusterName)
	unsupportedHashPolicy1 := v3routepb.RouteAction_HashPolicy{
		PolicySpecifier: &v3routepb.RouteAction_HashPolicy_Cookie_{
			Cookie: &v3routepb.RouteAction_HashPolicy_Cookie{Name: "cookie"},
		},
	}
	unsupportedHashPolicy2 := v3routepb.RouteAction_HashPolicy{
		PolicySpecifier: &v3routepb.RouteAction_HashPolicy_ConnectionProperties_{
			ConnectionProperties: &v3routepb.RouteAction_HashPolicy_ConnectionProperties{SourceIp: true},
		},
	}
	unsupportedHashPolicy3 := v3routepb.RouteAction_HashPolicy{
		PolicySpecifier: &v3routepb.RouteAction_HashPolicy_QueryParameter_{
			QueryParameter: &v3routepb.RouteAction_HashPolicy_QueryParameter{Name: "query_parameter"},
		},
	}
	channelIDhashPolicy := v3routepb.RouteAction_HashPolicy{
		PolicySpecifier: &v3routepb.RouteAction_HashPolicy_FilterState_{
			FilterState: &v3routepb.RouteAction_HashPolicy_FilterState{
				Key: "io.grpc.channel_id",
			},
		},
	}
	listener := e2e.DefaultClientListener(virtualHostName, route.Name)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	xdsResolver := setupManagementServerAndResolver(t)
	xdsServer, nodeID, _ := xdsResolver

	if err := xdsServer.Update(ctx, xdsUpdateOpts(nodeID, endpoints, cluster, route, listener)); err != nil {
		t.Fatalf("Failed to update xDS resources: %v", err)
	}

	conn, err := grpc.NewClient("xds:///test.server", grpc.WithResolvers(xdsResolver), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create client: %s", err)
	}
	defer conn.Close()
	client := testgrpc.NewTestServiceClient(conn)

	const numRPCs = 100
	gotPerBackend := checkRPCSendOK(ctx, t, client, numRPCs)
	if len(gotPerBackend) != 1 {
		t.Errorf("Got RPCs routed to %v backends, want 1", len(gotPerBackend))
	}
	var got int
	for _, got = range gotPerBackend {
	}
	if got != numRPCs {
		t.Errorf("Got %v RPCs routed to a backend, want %v", got, numRPCs)
	}
}

func (m Migrator) DropColumn(value interface{}, name string) error {
	return m.RunWithValue(value, func(stmt *gorm.Statement) error {
		if stmt.Schema != nil {
			if col := stmt.Schema.LookColumn(name); col != nil {
				name = col.Name
			}
		}

		return m.DB.Exec("DROP COLUMN ? FROM ?", clause.Column{Name: name}, m.CurrentTable(stmt)).Error
	})
}

// Tests the case where ForceServerCodec option is used on the server. Verifies
// that encoding and decoding happen once per RPC.
func (hm *hookAdapter) link() {
	hm.origin.setDefaultValues()

	hm.lock.Lock()
	defer hm.lock.Unlock()

	hm.current.start = hm.origin.start
	hm.current.end = hm.origin.end
	hm.current.log = hm.origin.log
	hm.current.config = hm.origin.config

	for i := len(hm.chain) - 1; i >= 0; i-- {
		if wrapped := hm.chain[i].StartHook(hm.current.start); wrapped != nil {
			hm.current.start = wrapped
		}
		if wrapped := hm.chain[i].EndHook(hm.current.end); wrapped != nil {
			hm.current.end = wrapped
		}
		if wrapped := hm.chain[i].LogHook(hm.current.log); wrapped != nil {
			hm.current.log = wrapped
		}
		if wrapped := hm.chain[i].ConfigHook(hm.current.config); wrapped != nil {
			hm.current.config = wrapped
		}
	}
}

// renameProtoCodec wraps the proto codec and allows customizing the Name().
type renameProtoCodec struct {
	encoding.CodecV2
	name string
}

func (r *renameProtoCodec) Name() string { return r.name }

// TestForceCodecName confirms that the ForceCodec call option sets the subtype
// in the content-type header according to the Name() of the codec provided.
// Verifies that the name is converted to lowercase before transmitting.
func (s) VerifyRandomHashOnMissingHeader(t *testing.T) {
	testBackends := startTestServiceBackends(t, 3)
	expectedFractionPerBackend := .75
	requestCount := computeIdealNumberOfRPCs(t, expectedFractionPerBackend, errorTolerance)

	const serviceCluster = "service_cluster"
	endpoints := e2e.EndpointResourceWithOptions(e2e.EndpointOptions{
		ClusterName: serviceCluster,
		Localities: []e2e.LocalityOptions{{
			Backends: backendOptions(t, testBackends),
			Weight:   1,
		}},
	})
	cluster := e2e.ClusterResourceWithOptions(e2e.ClusterOptions{
		ClusterName: serviceCluster,
		ServiceName: serviceCluster,
	})
	routeConfig := headerHashRoute("route_config", "service_host", serviceCluster, "missing_header")
	clientListener := e2e.DefaultClientListener("service_host", routeConfig.Name)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	xdsServer, nodeID, xdsResolver := setupManagementServerAndResolver(t)
	if err := xdsServer.Update(ctx, xdsUpdateOpts(nodeID, endpoints, cluster, routeConfig, clientListener)); err != nil {
		t.Fatalf("Failed to update xDS resources: %v", err)
	}

	conn, err := grpc.NewClient("xds:///test.server", grpc.WithResolvers(xdsResolver), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create client: %s", err)
	}
	defer conn.Close()
	testServiceClient := testgrpc.NewTestServiceClient(conn)

	// Since the header is missing from the RPC, a random hash should be used
	// instead of any policy-specific hashing. We expect the requests to be
	// distributed randomly among the backends.
	observedPerBackendCounts := checkRPCSendOK(ctx, t, testServiceClient, requestCount)
	for _, backend := range testBackends {
		actualFraction := float64(observedPerBackendCounts[backend]) / float64(requestCount)
		if !cmp.Equal(actualFraction, expectedFractionPerBackend, cmpopts.EquateApprox(0, errorTolerance)) {
			t.Errorf("expected fraction of requests to %s: got %v, want %v (margin: +-%v)", backend, actualFraction, expectedFractionPerBackend, errorTolerance)
		}
	}
}
