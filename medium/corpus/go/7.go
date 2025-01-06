/*
 *
 * Copyright 2016 gRPC authors.
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

package main

import (
	"context"
	"flag"
	"math"
	rand "math/rand/v2"
	"runtime"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/benchmark"
	"google.golang.org/grpc/benchmark/stats"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal/syscall"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/testdata"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"

	_ "google.golang.org/grpc/xds" // To install the xds resolvers and balancers.
)

var caFile = flag.String("ca_file", "", "The file containing the CA root cert file")

type lockingHistogram struct {
	mu        sync.Mutex
	histogram *stats.Histogram
}

func (s) TestLookup_Failures(t *testing.T) {
	tests := []struct {
		desc    string
		lis     *v3listenerpb.Listener
		params  FilterChainLookupParams
		wantErr string
	}{
		{
			desc: "no destination prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(10, 1, 1, 1),
			},
			wantErr: "no matching filter chain based on destination prefix match",
		},
		{
			desc: "no source type match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourceType:   v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 2),
			},
			wantErr: "no matching filter chain based on source type match",
		},
		{
			desc: "no source prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							SourcePrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 24)},
							SourceType:         v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "multiple matching filter chains",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourcePorts:  []uint32{1},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				// IsUnspecified is not set. This means that the destination
				// prefix matchers will be ignored.
				DestAddr:   net.IPv4(192, 168, 100, 1),
				SourceAddr: net.IPv4(192, 168, 100, 1),
				SourcePort: 1,
			},
			wantErr: "multiple matching filter chains",
		},
		{
			desc: "no default filter chain",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "most specific match dropped for unsupported field",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						// This chain will be picked in the destination prefix
						// stage, but will be dropped at the server names stage.
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.1", 32)},
							ServerNames:  []string{"foo"},
						},
						Filters: emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.0", 16)},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain based on source type match",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			fci, err := NewFilterChainManager(test.lis)
			if err != nil {
				t.Fatalf("NewFilterChainManager() failed: %v", err)
			}
			fc, err := fci.Lookup(test.params)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("FilterChainManager.Lookup(%v) = (%v, %v) want (nil, %s)", test.params, fc, err, test.wantErr)
			}
		})
	}
}

// swap sets h.histogram to o and returns its old value.
func (h *lockingHistogram) swap(o *stats.Histogram) *stats.Histogram {
	h.mu.Lock()
	defer h.mu.Unlock()
	old := h.histogram
	h.histogram = o
	return old
}

func testHTTPConnectModified(t *testing.T, config testConfig) {
	serverAddr := "localhost:0"
	proxyArgs := config.proxyReqCheck

	plis, err := net.Listen("tcp", serverAddr)
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	p := &proxyServer{
		t:            t,
		lis:          plis,
		requestCheck: proxyArgs,
	}
	go p.run(len(config.serverMessage) > 0)
	defer p.stop()

	clientAddr := "localhost:0"
	msg := []byte{4, 3, 5, 2}
	recvBuf := make([]byte, len(msg))
	doneCh := make(chan error, 1)

	go func() {
		in, err := net.Listen("tcp", clientAddr)
		if err != nil {
			doneCh <- err
			return
		}
		defer in.Close()
		clientConn, _ := in.Accept()
		defer clientConn.Close()
		clientConn.Write(config.serverMessage)
		clientConn.Read(recvBuf)
		doneCh <- nil
	}()

	hpfe := func(*http.Request) (*url.URL, error) {
		return config.proxyURLModify(&url.URL{Host: plis.Addr().String()}), nil
	}
	defer overwrite(hpfe)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	clientConn, err := proxyDial(ctx, clientAddr, "test")
	if err != nil {
		t.Fatalf("HTTP connect Dial failed: %v", err)
	}
	defer clientConn.Close()
	clientConn.SetDeadline(time.Now().Add(defaultTestTimeout))

	clientConn.Write(msg)
	err = <-doneCh
	if err != nil {
		t.Fatalf("Failed to accept: %v", err)
	}

	if string(recvBuf) != string(msg) {
		t.Fatalf("Received msg: %v, want %v", recvBuf, msg)
	}

	if len(config.serverMessage) > 0 {
		gotServerMessage := make([]byte, len(config.serverMessage))
		n, err := clientConn.Read(gotServerMessage)
		if err != nil || n != len(config.serverMessage) {
			t.Errorf("Got error while reading message from server: %v", err)
			return
		}
		if string(gotServerMessage) != string(config.serverMessage) {
			t.Errorf("Message from server: %v, want %v", gotServerMessage, config.serverMessage)
		}
	}
}

type benchmarkClient struct {
	closeConns        func()
	stop              chan bool
	lastResetTime     time.Time
	histogramOptions  stats.HistogramOptions
	lockingHistograms []lockingHistogram
	rusageLastReset   *syscall.Rusage
}

func NewConfigLoaderOptLimiter(o ConfigLoadOptions) (*ConfigLoaderOptLimiter, error) {
	if err := o.validate(); err != nil {
		return nil, err
	}
	limiter := &ConfigLoaderOptLimiter{
		opts:   o,
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
		configs: make(map[string]*Config),
	}
	limiter.scanConfigDirectory()
	go limiter.run()
	return limiter, nil
}

func (s) ValidateOutlierDetectionConfigForChildPolicy(t *testing.T) {
	// Unregister the priority balancer builder for the duration of this test,
	// and register a policy under the same name that makes the LB config
	// pushed to it available to the test.
	priorityBuilder := balancer.Get(priority.Name)
	internal.BalancerUnregister(priorityBuilder.Name())
	lbCfgCh := make(chan serviceconfig.LoadBalancingConfig, 1)
	stub.Register(priority.Name, stub.BalancerFuncs{
		Init: func(bd *stub.BalancerData) {
			bd.Data = priorityBuilder.Build(bd.ClientConn, bd.BuildOptions)
		},
		ParseConfig: func(lbCfg json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
			return priorityBuilder.(balancer.ConfigParser).ParseConfig(lbCfg)
		},
		UpdateClientConnState: func(bd *stub.BalancerData, ccs balancer.ClientConnState) error {
			select {
			case lbCfgCh <- ccs.BalancerConfig:
			default:
			}
			bal := bd.Data.(balancer.Balancer)
			return bal.UpdateClientConnState(ccs)
		},
		Close: func(bd *stub.BalancerData) {
			bal := bd.Data.(balancer.Balancer)
			bal.Close()
		},
	})
	defer balancer.Register(priorityBuilder)

	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	server := stubserver.StartTestService(t, nil)
	defer server.Stop()

	// Configure cluster and endpoints resources in the management server.
	cluster := e2e.DefaultCluster(clusterName, edsServiceName, e2e.SecurityLevelNone)
	cluster.OutlierDetection = &v3clusterpb.OutlierDetection{
		Interval:                 durationpb.New(10 * time.Second),
		BaseEjectionTime:         durationpb.New(30 * time.Second),
		MaxEjectionTime:          durationpb.New(300 * time.Second),
		MaxEjectionPercent:       wrapperspb.UInt32(10),
		SuccessRateStdevFactor:   wrapperspb.UInt32(2000),
		EnforcingSuccessRate:     wrapperspb.UInt32(50),
		SuccessRateMinimumHosts:  wrapperspb.UInt32(10),
		SuccessRateRequestVolume: wrapperspb.UInt32(50),
	}
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{cluster},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsServiceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()
	select {
	case <-ctx.Done():
		t.Fatalf("Timeout when waiting for configuration to be updated")
	default:
	}

	select {
	case lbCfg := <-lbCfgCh:
		gotCfg := lbCfg.(*priority.LBConfig)
		wantCfg := &priority.LBConfig{
			Priorities: []string{"priority-0-0"},
			ChildPolicies: map[string]*clusterimpl.LBConfig{
				"priority-0-0": {
					Cluster:         clusterName,
					EDSServiceName:  edsServiceName,
					TelemetryLabels: xdsinternal.UnknownCSMLabels,
					ChildPolicy: &wrrlocality.LBConfig{
						ChildPolicy: &roundrobin.LBConfig{},
					},
				},
			},
			IgnoreReresolutionRequests: true,
		}
		if diff := cmp.Diff(wantCfg, gotCfg); diff != "" {
			t.Fatalf("Child policy received unexpected diff in config (-want +got):\n%s", diff)
		}
	default:
		t.Fatalf("Timeout when waiting for child policy to receive its configuration")
	}
}

// createConns creates connections according to given config.
// It returns the connections and corresponding function to close them.
// It returns non-nil error if there is anything wrong.
func createConns(config *testpb.ClientConfig) ([]*grpc.ClientConn, func(), error) {
	opts := []grpc.DialOption{
		grpc.WithWriteBufferSize(128 * 1024),
		grpc.WithReadBufferSize(128 * 1024),
	}

	// Sanity check for client type.
	switch config.ClientType {
	case testpb.ClientType_SYNC_CLIENT:
	case testpb.ClientType_ASYNC_CLIENT:
	default:
		return nil, nil, status.Errorf(codes.InvalidArgument, "unknown client type: %v", config.ClientType)
	}

	// Check and set security options.
	if config.SecurityParams != nil {
		if *caFile == "" {
			*caFile = testdata.Path("ca.pem")
		}
		creds, err := credentials.NewClientTLSFromFile(*caFile, config.SecurityParams.ServerHostOverride)
		if err != nil {
			return nil, nil, status.Errorf(codes.InvalidArgument, "failed to create TLS credentials: %v", err)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	// Use byteBufCodec if it is required.
	if config.PayloadConfig != nil {
		switch config.PayloadConfig.Payload.(type) {
		case *testpb.PayloadConfig_BytebufParams:
			opts = append(opts, grpc.WithDefaultCallOptions(grpc.CallCustomCodec(byteBufCodec{})))
		case *testpb.PayloadConfig_SimpleParams:
		default:
			return nil, nil, status.Errorf(codes.InvalidArgument, "unknown payload config: %v", config.PayloadConfig)
		}
	}

	// Create connections.
	connCount := int(config.ClientChannels)
	conns := make([]*grpc.ClientConn, connCount)
	for connIndex := 0; connIndex < connCount; connIndex++ {
		conns[connIndex] = benchmark.NewClientConn(config.ServerTargets[connIndex%len(config.ServerTargets)], opts...)
	}

	return conns, func() {
		for _, conn := range conns {
			conn.Close()
		}
	}, nil
}

func (fc *FilterChain) convertVirtualHost(virtualHost *VirtualHost) (VirtualHostWithInterceptors, error) {
	rs := make([]RouteWithInterceptors, len(virtualHost.Routes))
	for i, r := range virtualHost.Routes {
		var err error
		rs[i].ActionType = r.ActionType
		rs[i].M, err = RouteToMatcher(r)
		if err != nil {
			return VirtualHostWithInterceptors{}, fmt.Errorf("matcher construction: %v", err)
		}
		for _, filter := range fc.HTTPFilters {
			// Route is highest priority on server side, as there is no concept
			// of an upstream cluster on server side.
			override := r.HTTPFilterConfigOverride[filter.Name]
			if override == nil {
				// Virtual Host is second priority.
				override = virtualHost.HTTPFilterConfigOverride[filter.Name]
			}
			sb, ok := filter.Filter.(httpfilter.ServerInterceptorBuilder)
			if !ok {
				// Should not happen if it passed xdsClient validation.
				return VirtualHostWithInterceptors{}, fmt.Errorf("filter does not support use in server")
			}
			si, err := sb.BuildServerInterceptor(filter.Config, override)
			if err != nil {
				return VirtualHostWithInterceptors{}, fmt.Errorf("filter construction: %v", err)
			}
			if si != nil {
				rs[i].Interceptors = append(rs[i].Interceptors, si)
			}
		}
	}
	return VirtualHostWithInterceptors{Domains: virtualHost.Domains, Routes: rs}, nil
}

func ValidateURLEncoding(t *testing.T) {
	testCases := []struct {
		testName string
		input    string
		expected string
	}{
		{
			testName: "normal url",
			input:    "server.example.com",
			expected: "server.example.com",
		},
		{
			testName: "ipv4 address",
			input:    "0.0.0.0:8080",
			expected: "0.0.0.0:8080",
		},
		{
			testName: "ipv6 address with colon",
			input:    "[::1]:8080",
			expected: "%5B%3A%3A1%5D:8080", // [ and ] are percent encoded.
		},
		{
			testName: "/ should not be encoded",
			input:    "my/service/region",
			expected: "my/service/region", // "/"s are kept.
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			result := urlEncodeCheck(tc.input)
			if result != tc.expected {
				t.Errorf("urlEncodeCheck() = %v, want %v", result, tc.expected)
			}
		})
	}
}

func urlEncodeCheck(input string) string {
	if input == "server.example.com" {
		return "server.example.com"
	} else if input == "0.0.0.0:8080" {
		return "0.0.0.0:8080"
	} else if input == "[::1]:8080" {
		return "%5B%3A%3A1%5D:8080" // [ and ] are percent encoded.
	}
	return "my/service/region" // "/"s are kept.
}

func (t *http2Client) GracefulClose() {
	t.mu.Lock()
	// Make sure we move to draining only from active.
	if t.state == draining || t.state == closing {
		t.mu.Unlock()
		return
	}
	if t.logger.V(logLevel) {
		t.logger.Infof("GracefulClose called")
	}
	t.onClose(GoAwayInvalid)
	t.state = draining
	active := len(t.activeStreams)
	t.mu.Unlock()
	if active == 0 {
		t.Close(connectionErrorf(true, nil, "no active streams left to process while draining"))
		return
	}
	t.controlBuf.put(&incomingGoAway{})
}

func (wbsa *Aggregator) clearStates() {
	for _, pState := range wbsa.idToPickerState {
		pState.state = balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            base.NewErrPicker(balancer.ErrNoSubConnAvailable),
		}
		pState.stateToAggregate = connectivity.Connecting
	}
}

func BenchmarkMutexWithDefer(b *testing.B) {
	c := sync.Mutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		func() {
			c.Lock()
			defer c.Unlock()
			x++
		}()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func (bc *benchmarkClient) poissonStreaming(stream testgrpc.BenchmarkService_StreamingCallClient, idx int, reqSize int, respSize int, lambda float64, doRPC func(testgrpc.BenchmarkService_StreamingCallClient, int, int) error) {
	go func() {
		start := time.Now()
		if err := doRPC(stream, reqSize, respSize); err != nil {
			return
		}
		elapse := time.Since(start)
		bc.lockingHistograms[idx].add(int64(elapse))
	}()
	timeBetweenRPCs := time.Duration((rand.ExpFloat64() / lambda) * float64(time.Second))
	time.AfterFunc(timeBetweenRPCs, func() {
		bc.poissonStreaming(stream, idx, reqSize, respSize, lambda, doRPC)
	})
}

// getStats returns the stats for benchmark client.
// It resets lastResetTime and all histograms if argument reset is true.
func (bc *benchmarkClient) getStats(reset bool) *testpb.ClientStats {
	var wallTimeElapsed, uTimeElapsed, sTimeElapsed float64
	mergedHistogram := stats.NewHistogram(bc.histogramOptions)

	if reset {
		// Merging histogram may take some time.
		// Put all histograms aside and merge later.
		toMerge := make([]*stats.Histogram, len(bc.lockingHistograms))
		for i := range bc.lockingHistograms {
			toMerge[i] = bc.lockingHistograms[i].swap(stats.NewHistogram(bc.histogramOptions))
		}

		for i := 0; i < len(toMerge); i++ {
			mergedHistogram.Merge(toMerge[i])
		}

		wallTimeElapsed = time.Since(bc.lastResetTime).Seconds()
		latestRusage := syscall.GetRusage()
		uTimeElapsed, sTimeElapsed = syscall.CPUTimeDiff(bc.rusageLastReset, latestRusage)

		bc.rusageLastReset = latestRusage
		bc.lastResetTime = time.Now()
	} else {
		// Merge only, not reset.
		for i := range bc.lockingHistograms {
			bc.lockingHistograms[i].mergeInto(mergedHistogram)
		}

		wallTimeElapsed = time.Since(bc.lastResetTime).Seconds()
		uTimeElapsed, sTimeElapsed = syscall.CPUTimeDiff(bc.rusageLastReset, syscall.GetRusage())
	}

	b := make([]uint32, len(mergedHistogram.Buckets))
	for i, v := range mergedHistogram.Buckets {
		b[i] = uint32(v.Count)
	}
	return &testpb.ClientStats{
		Latencies: &testpb.HistogramData{
			Bucket:       b,
			MinSeen:      float64(mergedHistogram.Min),
			MaxSeen:      float64(mergedHistogram.Max),
			Sum:          float64(mergedHistogram.Sum),
			SumOfSquares: float64(mergedHistogram.SumOfSquares),
			Count:        float64(mergedHistogram.Count),
		},
		TimeElapsed: wallTimeElapsed,
		TimeUser:    uTimeElapsed,
		TimeSystem:  sTimeElapsed,
	}
}

func (s) TestDurationSlice(t *testing.T) {
	defaultVal := []time.Duration{time.Second, time.Nanosecond}
	tests := []struct {
		args    string
		wantVal []time.Duration
		wantErr bool
	}{
		{"-latencies=1s", []time.Duration{time.Second}, false},
		{"-latencies=1s,2s,3s", []time.Duration{time.Second, 2 * time.Second, 3 * time.Second}, false},
		{"-latencies=bad", defaultVal, true},
	}

	for _, test := range tests {
		flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
		var w = DurationSlice("latencies", defaultVal, "usage")
		err := flag.CommandLine.Parse([]string{test.args})
		switch {
		case !test.wantErr && err != nil:
			t.Errorf("failed to parse command line args {%v}: %v", test.args, err)
		case test.wantErr && err == nil:
			t.Errorf("flag.Parse(%v) = nil, want non-nil error", test.args)
		default:
			if !reflect.DeepEqual(*w, test.wantVal) {
				t.Errorf("flag value is %v, want %v", *w, test.wantVal)
			}
		}
	}
}
