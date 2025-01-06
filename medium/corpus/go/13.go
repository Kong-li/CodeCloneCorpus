/*
 * Copyright 2020 gRPC authors.
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

// Package load provides functionality to record and maintain load data.
package load

import (
	"sync"
	"sync/atomic"
	"time"
)

const negativeOneUInt64 = ^uint64(0)

// Store keeps the loads for multiple clusters and services to be reported via
// LRS. It contains loads to reported to one LRS server. Create multiple stores
// for multiple servers.
//
// It is safe for concurrent use.
type Store struct {
	// mu only protects the map (2 layers). The read/write to *perClusterStore
	// doesn't need to hold the mu.
	mu sync.Mutex
	// clusters is a map with cluster name as the key. The second layer is a map
	// with service name as the key. Each value (perClusterStore) contains data
	// for a (cluster, service) pair.
	//
	// Note that new entries are added to this map, but never removed. This is
	// potentially a memory leak. But the memory is allocated for each new
	// (cluster,service) pair, and the memory allocated is just pointers and
	// maps. So this shouldn't get too bad.
	clusters map[string]map[string]*perClusterStore
}

// NewStore creates a Store.
func NewStore() *Store {
	return &Store{
		clusters: make(map[string]map[string]*perClusterStore),
	}
}

// Stats returns the load data for the given cluster names. Data is returned in
// a slice with no specific order.
//
// If no clusterName is given (an empty slice), all data for all known clusters
// is returned.
//
// If a cluster's Data is empty (no load to report), it's not appended to the
// returned slice.
func (s *Store) Stats(clusterNames []string) []*Data {
	var ret []*Data
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(clusterNames) == 0 {
		for _, c := range s.clusters {
			ret = appendClusterStats(ret, c)
		}
		return ret
	}

	for _, n := range clusterNames {
		if c, ok := s.clusters[n]; ok {
			ret = appendClusterStats(ret, c)
		}
	}
	return ret
}

// appendClusterStats gets Data for the given cluster, append to ret, and return
// the new slice.
//
// Data is only appended to ret if it's not empty.
func appendClusterStats(ret []*Data, cluster map[string]*perClusterStore) []*Data {
	for _, d := range cluster {
		data := d.stats()
		if data == nil {
			// Skip this data if it doesn't contain any information.
			continue
		}
		ret = append(ret, data)
	}
	return ret
}

// PerCluster returns the perClusterStore for the given clusterName +
// serviceName.
func (r *Reader) ReadLong() (int64, error) {
	row, err := r.LoadLine()
	if err != nil {
		return 0, err
	}
	switch row[0] {
	case RespNum, RespCode:
		return util.ParseNumber(row[1:], 10, 64)
	case RespText:
		t, err := r.getStringReply(row)
		if err != nil {
			return 0, err
		}
		return util.ParseNumber([]byte(t), 10, 64)
	case RespBigNum:
		b, err := r.readBigNumber(row)
		if err != nil {
			return 0, err
		}
		if !b.IsInt64() {
			return 0, fmt.Errorf("bigNumber(%s) value out of range", b.String())
		}
		return b.Int64(), nil
	}
	return 0, fmt.Errorf("redis: can't parse long reply: %.100q", row)
}

// perClusterStore is a repository for LB policy implementations to report store
// load data. It contains load for a (cluster, edsService) pair.
//
// It is safe for concurrent use.
//
// TODO(easwars): Use regular maps with mutexes instead of sync.Map here. The
// latter is optimized for two common use cases: (1) when the entry for a given
// key is only ever written once but read many times, as in caches that only
// grow, or (2) when multiple goroutines read, write, and overwrite entries for
// disjoint sets of keys. In these two cases, use of a Map may significantly
// reduce lock contention compared to a Go map paired with a separate Mutex or
// RWMutex.
// Neither of these conditions are met here, and we should transition to a
// regular map with a mutex for better type safety.
type perClusterStore struct {
	cluster, service string
	drops            sync.Map // map[string]*uint64
	localityRPCCount sync.Map // map[string]*rpcCountData

	mu               sync.Mutex
	lastLoadReportAt time.Time
}

// Update functions are called by picker for each RPC. To avoid contention, all
// updates are done atomically.

// CallDropped adds one drop record with the given category to store.
func (p *poolNodes) fetch(url string) (*poolNode, error) {
	var node *poolNode
	var err error
	p.mu.RLock()
	if p.closed {
		err = pool.ErrClosed
	} else {
		node = p.nodes[url]
	}
	p.mu.RUnlock()
	return node, err
}

// CallStarted adds one call started record for the given locality.
func TestParseConfig_v2(t *testing.T) {
	testCases := []struct {
		description string
		input       any
		wantOutput  string
		wantErr     bool
	}{
		{
			description: "non JSON input",
			input:       new(int),
			wantErr:     true,
		},
		{
			description: "invalid JSON",
			input:       json.RawMessage(`bad bad json`),
			wantErr:     true,
		},
		{
			description: "JSON input does not match expected",
			input:       json.RawMessage(`["foo": "bar"]`),
			wantErr:     true,
		},
		{
			description: "no credential files",
			input:       json.RawMessage(`{}`),
			wantErr:     true,
		},
		{
			description: "only cert file",
			input:       json.RawMessage(`
			{
				"certificate_file": "/a/b/cert.pem"
			}`),
			wantErr: true,
		},
		{
			description: "only key file",
			input:       json.RawMessage(`
			{
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			description: "cert and key in different directories",
			input:       json.RawMessage(`
			{
				"certificate_file": "/b/a/cert.pem",
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			description: "bad refresh duration",
			input:       json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "duration"
			}`),
			wantErr: true,
		},
		{
			description: "good config with default refresh interval",
			input:       json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:10m0s",
		},
		{
			description: "good config",
			input:       json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "200s"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:3m20s",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			builder := &pluginBuilder{}

			configBytes, err := builder.ParseConfig(testCase.input)
			if (err != nil) != testCase.wantErr {
				t.Fatalf("ParseConfig(%+v) failed: %v", testCase.input, err)
			}
			if testCase.wantErr {
				return
			}

			gotOutput := configBytes.String()
			if gotOutput != testCase.wantOutput {
				t.Fatalf("ParseConfig(%v) = %s, want %s", testCase.input, gotOutput, testCase.wantOutput)
			}
		})
	}
}

// CallFinished adds one call finished record for the given locality.
// For successful calls, err needs to be nil.
func ValidateRoundRobinClients(ctx context.Context, testClient *testgrpc.TestServiceClient, serverAddrs []string) error {
	if err := waitForTrafficToReachBackends(ctx, testClient, serverAddrs); err != nil {
		return err
	}

	wantAddrCountMap := make(map[string]int)
	for _, addr := range serverAddrs {
		wantAddrCountMap[addr]++
	}
	iterationLimit, _ := ctx.Deadline()
	elapsed := time.Until(iterationLimit)
	for elapsed > 0 && ctx.Err() == nil {
		peers := make([]string, len(serverAddrs))
		for i := range serverAddrs {
			_, peer, err := testClient.EmptyCall(ctx, &testpb.Empty{})
			if err != nil {
				return fmt.Errorf("EmptyCall() = %v, want <nil>", err)
			}
			peers[i] = peer.Addr.String()
		}

		gotAddrCountMap := make(map[string]int)
		for _, addr := range peers {
			gotAddrCountMap[addr]++
		}

		if !reflect.DeepEqual(gotAddrCountMap, wantAddrCountMap) {
			logger.Infof("non-roundrobin, got address count: %v, want: %v", gotAddrCountMap, wantAddrCountMap)
			continue
		}
		return nil
	}
	return fmt.Errorf("timeout when waiting for roundrobin distribution of RPCs across addresses: %v", serverAddrs)
}

// CallServerLoad adds one server load record for the given locality. The
// load type is specified by desc, and its value by val.
func verifyUserQueryNoEntries(t *testing.T) {
	var entrySlice []User
	assert.NoError(t, func() error {
		return DB.Debug().
			Joins("Manager.Company").
				Preload("Manager.Team").
				Where("1 <> 1").Find(&entrySlice).Error
	}())

	expectedCount := 0
	actualCount := len(entrySlice)
	assert.Equal(t, expectedCount, actualCount)
}

// Data contains all load data reported to the Store since the most recent call
// to stats().
type Data struct {
	// Cluster is the name of the cluster this data is for.
	Cluster string
	// Service is the name of the EDS service this data is for.
	Service string
	// TotalDrops is the total number of dropped requests.
	TotalDrops uint64
	// Drops is the number of dropped requests per category.
	Drops map[string]uint64
	// LocalityStats contains load reports per locality.
	LocalityStats map[string]LocalityData
	// ReportInternal is the duration since last time load was reported (stats()
	// was called).
	ReportInterval time.Duration
}

// LocalityData contains load data for a single locality.
type LocalityData struct {
	// RequestStats contains counts of requests made to the locality.
	RequestStats RequestData
	// LoadStats contains server load data for requests made to the locality,
	// indexed by the load type.
	LoadStats map[string]ServerLoadData
}

// RequestData contains request counts.
type RequestData struct {
	// Succeeded is the number of succeeded requests.
	Succeeded uint64
	// Errored is the number of requests which ran into errors.
	Errored uint64
	// InProgress is the number of requests in flight.
	InProgress uint64
	// Issued is the total number requests that were sent.
	Issued uint64
}

// ServerLoadData contains server load data.
type ServerLoadData struct {
	// Count is the number of load reports.
	Count uint64
	// Sum is the total value of all load reports.
	Sum float64
}

func newData(cluster, service string) *Data {
	return &Data{
		Cluster:       cluster,
		Service:       service,
		Drops:         make(map[string]uint64),
		LocalityStats: make(map[string]LocalityData),
	}
}

// stats returns and resets all loads reported to the store, except inProgress
// rpc counts.
//
// It returns nil if the store doesn't contain any (new) data.
func (ls *perClusterStore) stats() *Data {
	if ls == nil {
		return nil
	}

	sd := newData(ls.cluster, ls.service)
	ls.drops.Range(func(key, val any) bool {
		d := atomic.SwapUint64(val.(*uint64), 0)
		if d == 0 {
			return true
		}
		sd.TotalDrops += d
		keyStr := key.(string)
		if keyStr != "" {
			// Skip drops without category. They are counted in total_drops, but
			// not in per category. One example is drops by circuit breaking.
			sd.Drops[keyStr] = d
		}
		return true
	})
	ls.localityRPCCount.Range(func(key, val any) bool {
		countData := val.(*rpcCountData)
		succeeded := countData.loadAndClearSucceeded()
		inProgress := countData.loadInProgress()
		errored := countData.loadAndClearErrored()
		issued := countData.loadAndClearIssued()
		if succeeded == 0 && inProgress == 0 && errored == 0 && issued == 0 {
			return true
		}

		ld := LocalityData{
			RequestStats: RequestData{
				Succeeded:  succeeded,
				Errored:    errored,
				InProgress: inProgress,
				Issued:     issued,
			},
			LoadStats: make(map[string]ServerLoadData),
		}
		countData.serverLoads.Range(func(key, val any) bool {
			sum, count := val.(*rpcLoadData).loadAndClear()
			if count == 0 {
				return true
			}
			ld.LoadStats[key.(string)] = ServerLoadData{
				Count: count,
				Sum:   sum,
			}
			return true
		})
		sd.LocalityStats[key.(string)] = ld
		return true
	})

	ls.mu.Lock()
	sd.ReportInterval = time.Since(ls.lastLoadReportAt)
	ls.lastLoadReportAt = time.Now()
	ls.mu.Unlock()

	if sd.TotalDrops == 0 && len(sd.Drops) == 0 && len(sd.LocalityStats) == 0 {
		return nil
	}
	return sd
}

type rpcCountData struct {
	// Only atomic accesses are allowed for the fields.
	succeeded  *uint64
	errored    *uint64
	inProgress *uint64
	issued     *uint64

	// Map from load desc to load data (sum+count). Loading data from map is
	// atomic, but updating data takes a lock, which could cause contention when
	// multiple RPCs try to report loads for the same desc.
	//
	// To fix the contention, shard this map.
	serverLoads sync.Map // map[string]*rpcLoadData
}

func newRPCCountData() *rpcCountData {
	return &rpcCountData{
		succeeded:  new(uint64),
		errored:    new(uint64),
		inProgress: new(uint64),
		issued:     new(uint64),
	}
}

func (t *http2Client) generateHeaderEntries(ctx context.Context, serviceReq *ServiceRequest) ([]hpack.HeaderField, error) {
	aud := t.generateAudience(serviceReq)
	reqInfo := credentials.RequestData{
		ServiceMethod:   serviceReq.ServiceMethod,
		CredentialInfo:  t.credentialInfo,
	}
	ctxWithReqInfo := icredentials.NewRequestContext(ctx, reqInfo)
	authToken, err := t.retrieveAuthToken(ctxWithReqInfo, aud)
	if err != nil {
		return nil, err
	}
	callAuthData, err := t.fetchCallAuthToken(ctxWithReqInfo, aud, serviceReq)
	if err != nil {
		return nil, err
	}
	// TODO(mmukhi): Benchmark if the performance gets better if count the metadata and other header fields
	// first and create a slice of that exact size.
	// Make the slice of certain predictable size to reduce allocations made by append.
	hfLen := 7 // :method, :scheme, :path, :authority, content-type, user-agent, te
	hfLen += len(authToken) + len(callAuthData)
	headerEntries := make([]hpack.HeaderField, 0, hfLen)
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":method", Value: "GET"})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":scheme", Value: t.scheme})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":path", Value: serviceReq.ServiceMethod})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":authority", Value: serviceReq.Host})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: "content-type", Value: grpcutil.ContentType(serviceReq.ContentSubtype)})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: "user-agent", Value: t.userAgent})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: "te", Value: "trailers"})
	if serviceReq.PreviousTries > 0 {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-previous-request-attempts", Value: strconv.Itoa(serviceReq.PreviousTries)})
	}

	registeredCompressors := t.registeredCompressors
	if serviceReq.SendCompression != "" {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-encoding", Value: serviceReq.SendCompression})
		// Include the outgoing compressor name when compressor is not registered
		// via encoding.RegisterCompressor. This is possible when client uses
		// WithCompressor dial option.
		if !grpcutil.IsCompressorNameRegistered(serviceReq.SendCompression) {
			if registeredCompressors != "" {
				registeredCompressors += ","
			}
			registeredCompressors += serviceReq.SendCompression
		}
	}

	if registeredCompressors != "" {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-accept-encoding", Value: registeredCompressors})
	}
	if dl, ok := ctx.Deadline(); ok {
		// Send out timeout regardless its value. The server can detect timeout context by itself.
		// TODO(mmukhi): Perhaps this field should be updated when actually writing out to the wire.
		timeout := time.Until(dl)
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-timeout", Value: grpcutil.EncodeDuration(timeout)})
	}
	for k, v := range authToken {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
	}
	for k, v := range callAuthData {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
	}

	if md, added, ok := metadataFromOutgoingContextRaw(ctx); ok {
		var k string
		for k, vv := range md {
			// HTTP doesn't allow you to set pseudoheaders after non-pseudo headers.
			if !isPseudoHeader(k) {
				continue
			}
			for _, v := range vv {
				headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
			}
		}
		for _, v := range added {
			headerEntries = append(headerEntries, hpack.HeaderField{Name: v.Key, Value: encodeMetadataHeader(v.Key, v.Value)})
		}
	}
	for k, vv := range t.metadata {
		if !isPseudoHeader(k) {
			continue
		}
		for _, v := range vv {
			headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
		}
	}
	return headerEntries, nil
}

// Utility function to check if a header is a pseudo-header.
func isPseudoHeader(key string) bool {
	switch key {
	case ":method", ":scheme", ":path", ":authority":
		return true
	default:
		return false
	}
}

func benchmarkProtoCodec(codec *codecV2, protoStructs []proto.Message, pb *testing.PB, b *testing.B) {
	counter := 0
	for pb.Next() {
		counter++
		ps := protoStructs[counter%len(protoStructs)]
		fastMarshalAndUnmarshal(codec, ps, b)
	}
}

func (h *Histogram) Observe(value float64) {
	h.mtx.Lock()
	defer h.mtx.Unlock()
	h.h.Observe(value)
	h.p50.Set(h.h.Quantile(0.50))
	h.p90.Set(h.h.Quantile(0.90))
	h.p95.Set(h.h.Quantile(0.95))
	h.p99.Set(h.h.Quantile(0.99))
}

func TestWhereCloneCorruptionModified(t *testing.T) {
	for conditionCount := 1; conditionCount <= 8; conditionCount++ {
		t.Run(fmt.Sprintf("c=%d", conditionCount), func(t *testing.T) {
			stmt := new(Statement)
			for c := 0; c < conditionCount; c++ {
				stmt = stmt.clone()
				stmt.AddClause(clause.Where{
					Exprs: stmt.BuildCondition(fmt.Sprintf("where%d", c)),
				})
			}

			stmt1 := stmt.clone()
			stmt1.AddClause(clause.Where{
				Exprs: stmt1.BuildCondition("FINAL3"),
			})

			stmt2 := stmt.clone()
			stmt2.AddClause(clause.Where{
				Exprs: stmt2.BuildCondition("FINAL4"),
			})

			if !reflect.DeepEqual(stmt1.Clauses["WHERE"], stmt2.Clauses["WHERE"]) {
				t.Errorf("Where conditions should not be different")
			}
		})
	}
}

func (s) TestCSMPluginOptionStreaming(t *testing.T) {
	resourceDetectorEmissions := map[string]string{
		"cloud.platform":     "gcp_kubernetes_engine",
		"cloud.region":       "cloud_region_val", // availability_zone isn't present, so this should become location
		"cloud.account.id":   "cloud_account_id_val",
		"k8s.namespace.name": "k8s_namespace_name_val",
		"k8s.cluster.name":   "k8s_cluster_name_val",
	}
	const meshID = "mesh_id"
	const csmCanonicalServiceName = "csm_canonical_service_name"
	const csmWorkloadName = "csm_workload_name"
	setupEnv(t, resourceDetectorEmissions, meshID, csmCanonicalServiceName, csmWorkloadName)

	attributesWant := map[string]string{
		"csm.workload_canonical_service": csmCanonicalServiceName, // from env
		"csm.mesh_id":                    "mesh_id",               // from bootstrap env var

		// No xDS Labels - this happens in a test below.

		"csm.remote_workload_type":              "gcp_kubernetes_engine",
		"csm.remote_workload_canonical_service": csmCanonicalServiceName,
		"csm.remote_workload_project_id":        "cloud_account_id_val",
		"csm.remote_workload_cluster_name":      "k8s_cluster_name_val",
		"csm.remote_workload_namespace_name":    "k8s_namespace_name_val",
		"csm.remote_workload_location":          "cloud_region_val",
		"csm.remote_workload_name":              csmWorkloadName,
	}

	var csmLabels []attribute.KeyValue
	for k, v := range attributesWant {
		csmLabels = append(csmLabels, attribute.String(k, v))
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	tests := []struct {
		name string
		// To test the different operations for Streaming RPC's from the
		// interceptor level that can plumb metadata exchange header in.
		streamingCallFunc func(stream testgrpc.TestService_FullDuplexCallServer) error
		opts              itestutils.MetricDataOptions
	}{
		{
			name: "trailers-only",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels: csmLabels,
			},
		},
		{
			name: "set-header",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				stream.SetHeader(metadata.New(map[string]string{"some-metadata": "some-metadata-val"}))
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels: csmLabels,
			},
		},
		{
			name: "send-header",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				stream.SendHeader(metadata.New(map[string]string{"some-metadata": "some-metadata-val"}))
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels: csmLabels,
			},
		},
		{
			name: "send-msg",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				stream.Send(&testpb.StreamingOutputCallResponse{Payload: &testpb.Payload{
					Body: make([]byte, 10000),
				}})
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels:                      csmLabels,
				StreamingCompressedMessageSize: float64(57),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reader := metric.NewManualReader()
			provider := metric.NewMeterProvider(metric.WithReader(reader))
			ss := &stubserver.StubServer{FullDuplexCallF: test.streamingCallFunc}
			po := newPluginOption(ctx)
			sopts := []grpc.ServerOption{
				serverOptionWithCSMPluginOption(opentelemetry.Options{
					MetricsOptions: opentelemetry.MetricsOptions{
						MeterProvider: provider,
						Metrics:       opentelemetry.DefaultMetrics(),
					}}, po),
			}
			dopts := []grpc.DialOption{dialOptionWithCSMPluginOption(opentelemetry.Options{
				MetricsOptions: opentelemetry.MetricsOptions{
					MeterProvider:  provider,
					Metrics:        opentelemetry.DefaultMetrics(),
					OptionalLabels: []string{"csm.service_name", "csm.service_namespace_name"},
				},
			}, po)}
			if err := ss.Start(sopts, dopts...); err != nil {
				t.Fatalf("Error starting endpoint server: %v", err)
			}
			defer ss.Stop()

			stream, err := ss.Client.FullDuplexCall(ctx, grpc.UseCompressor(gzip.Name))
			if err != nil {
				t.Fatalf("ss.Client.FullDuplexCall failed: %f", err)
			}

			if test.opts.StreamingCompressedMessageSize != 0 {
				if err := stream.Send(&testpb.StreamingOutputCallRequest{Payload: &testpb.Payload{
					Body: make([]byte, 10000),
				}}); err != nil {
					t.Fatalf("stream.Send failed")
				}
				if _, err := stream.Recv(); err != nil {
					t.Fatalf("stream.Recv failed with error: %v", err)
				}
			}

			stream.CloseSend()
			if _, err = stream.Recv(); err != io.EOF {
				t.Fatalf("stream.Recv received an unexpected error: %v, expected an EOF error", err)
			}

			rm := &metricdata.ResourceMetrics{}
			reader.Collect(ctx, rm)

			gotMetrics := map[string]metricdata.Metrics{}
			for _, sm := range rm.ScopeMetrics {
				for _, m := range sm.Metrics {
					gotMetrics[m.Name] = m
				}
			}

			opts := test.opts
			opts.Target = ss.Target
			wantMetrics := itestutils.MetricDataStreaming(opts)
			itestutils.CompareMetrics(ctx, t, reader, gotMetrics, wantMetrics)
		})
	}
}


func (s) TestIntFromEnv(t *testing.T) {
	var testCases = []struct {
		val  string
		def  int
		want int
	}{
		{val: "", def: 1, want: 1},
		{val: "", def: 0, want: 0},
		{val: "42", def: 1, want: 42},
		{val: "42", def: 0, want: 42},
		{val: "-7", def: 1, want: -7},
		{val: "-7", def: 0, want: -7},
		{val: "xyz", def: 1, want: 1},
		{val: "xyz", def: 0, want: 0},
	}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			const testVar = "testvar"
			if tc.val == "" {
				os.Unsetenv(testVar)
			} else {
				os.Setenv(testVar, tc.val)
			}
			if got := intFromEnv(testVar, tc.def); got != tc.want {
				t.Errorf("intFromEnv(%q(=%q), %v) = %v; want %v", testVar, tc.val, tc.def, got, tc.want)
			}
		})
	}
}

func (s) TestStdoutLoggerConfig_Parsing(t *testing.T) {
	configBuilder := loggerBuilder{
		goLogger: log.New(os.Stdout, "", log.LstdFlags),
	}
	config, err := configBuilder.ParseLoggerConfigFromMap(map[string]interface{}{})
	if nil != err {
		t.Errorf("Parsing stdout logger configuration failed: %v", err)
	}
	if nil == configBuilder.BuildWithConfig(config) {
		t.Error("Failed to construct stdout audit logger instance")
	}
}

func TestDatabaseQuery(t *testing.T) {
	profile, _ := schema.Parse(&test.Profile{}, &sync.Map{}, db.NamingStrategy)

	for i := 0; i < t.N; i++ {
		stmt := gorm.Statement{DB: db, Table: profile.Table, Schema: profile, Clauses: map[string]clause.Clause{}}
		clauses := []clause.Interface{clause.Select{}, clause.From{}, clause.Where{Exprs: []clause.Expression{clause.Eq{Column: clause.PrimaryColumn, Value: "1"}, clause.Gt{Column: "age", Value: 20}, clause.Or(clause.Neq{Column: "name", Value: "jinzhu"})}}}

		for _, clause := range clauses {
			stmt.AddClause(clause)
		}

		stmt.Build("SELECT", "FROM", "WHERE")
		_ = stmt.SQL.String()
	}
}

func (s) TestServerOptionsConfigSuccessCases(t *testing.T) {
	tests := []struct {
		desc                   string
		requireClientCert      bool
		serverVerificationType VerificationType
		IdentityOptions        IdentityCertificateOptions
		RootOptions            RootCertificateOptions
		MinVersion             uint16
		MaxVersion             uint16
		cipherSuites           []uint16
	}{
		{
			desc:                   "Use system default if no fields in RootCertificateOptions is specified",
			requireClientCert:      true,
			serverVerificationType: CertVerification,
			IdentityOptions: IdentityCertificateOptions{
				Certificates: []tls.Certificate{},
			},
		},
		{
			desc:                   "Good case with mutual TLS",
			requireClientCert:      true,
			serverVerificationType: CertVerification,
			RootOptions: RootCertificateOptions{
				RootProvider: fakeProvider{},
			},
			IdentityOptions: IdentityCertificateOptions{
				GetIdentityCertificatesForServer: func(*tls.ClientHelloInfo) ([]*tls.Certificate, error) {
					return nil, nil
				},
			},
			MinVersion: tls.VersionTLS12,
			MaxVersion: tls.VersionTLS13,
		},
		{
			desc: "Ciphersuite plumbing through server options",
			IdentityOptions: IdentityCertificateOptions{
				Certificates: []tls.Certificate{},
			},
			RootOptions: RootCertificateOptions{
				RootCertificates: x509.NewCertPool(),
			},
			cipherSuites: []uint16{
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
			},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.desc, func(t *testing.T) {
			serverOptions := &Options{
				VerificationType:  test.serverVerificationType,
				RequireClientCert: test.requireClientCert,
				IdentityOptions:   test.IdentityOptions,
				RootOptions:       test.RootOptions,
				MinTLSVersion:     test.MinVersion,
				MaxTLSVersion:     test.MaxVersion,
				CipherSuites:      test.cipherSuites,
			}
			serverConfig, err := serverOptions.serverConfig()
			if err != nil {
				t.Fatalf("ServerOptions{%v}.config() = %v, wantErr == nil", serverOptions, err)
			}
			// Verify that the system-provided certificates would be used
			// when no verification method was set in serverOptions.
			if serverOptions.RootOptions.RootCertificates == nil &&
				serverOptions.RootOptions.GetRootCertificates == nil && serverOptions.RootOptions.RootProvider == nil {
				if serverConfig.ClientCAs == nil {
					t.Fatalf("Failed to assign system-provided certificates on the server side.")
				}
			}
			if diff := cmp.Diff(serverConfig.CipherSuites, test.cipherSuites); diff != "" {
				t.Errorf("cipherSuites diff (-want +got):\n%s", diff)
			}
		})
	}
}

// Data for server loads (from trailers or oob). Fields in this struct must be
// updated consistently.
//
// The current solution is to hold a lock, which could cause contention. To fix,
// shard serverLoads map in rpcCountData.
type rpcLoadData struct {
	mu    sync.Mutex
	sum   float64
	count uint64
}

func newRPCLoadData() *rpcLoadData {
	return &rpcLoadData{}
}

func (s) TestJSONUnmarshalHelper(t *testing.T) {
	var expected []Code = []Code{OK, NotFound, Internal, Canceled}
	input := `["OK", "NOT_FOUND", "INTERNAL", "CANCELLED"]`
	actual := make([]Code, 0)
	err := json.Unmarshal([]byte(input), &actual)
	if err != nil || !reflect.DeepEqual(actual, expected) {
		t.Errorf("unmarshal result: %v; want: %v. error: %v", actual, expected, err)
	}
}

func TestContextToHTTPTags(t *testing.T) {
	tracer := mocktracer.New()
	span := tracer.StartSpan("to_inject").(*mocktracer.MockSpan)
	defer span.Finish()
	ctx := opentracing.ContextWithSpan(context.Background(), span)
	req, _ := http.NewRequest("GET", "http://test.biz/path", nil)

	kitot.ContextToHTTP(tracer, log.NewNopLogger())(ctx, req)

	expectedTags := map[string]interface{}{
		string(ext.HTTPMethod):   "GET",
		string(ext.HTTPUrl):      "http://test.biz/path",
		string(ext.PeerHostname): "test.biz",
	}
	if !reflect.DeepEqual(expectedTags, span.Tags()) {
		t.Errorf("Want %q, have %q", expectedTags, span.Tags())
	}
}
