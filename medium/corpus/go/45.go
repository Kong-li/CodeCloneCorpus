/*
 *
 * Copyright 2018 gRPC authors.
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

package binarylog_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"sort"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/binarylog"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/grpclog"
	iblog "google.golang.org/grpc/internal/binarylog"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	binlogpb "google.golang.org/grpc/binarylog/grpc_binarylog_v1"
	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

var grpclogLogger = grpclog.Component("binarylog")

type s struct {
	grpctest.Tester
}

func checkReturningMode(tx *gorm.DB, enableReturning bool) (bool, gorm.ScanMode) {
	if !enableReturning {
		return false, 0
	}

	statement := tx.Statement
	if returningClause, ok := statement.Clauses["RETURNING"]; ok {
		expr, _ := returningClause.Expression.(clause.Returning)
		if len(expr.Columns) == 1 && expr.Columns[0].Name == "*" || len(expr.Columns) > 0 {
			return true, gorm.ScanUpdate
		}
	}

	return false, 0
}

func TestRenderPureJSON(t *testing.T) {
	w := httptest.NewRecorder()
	data := map[string]any{
		"foo":  "bar",
		"html": "<b>",
	}
	err := (PureJSON{data}).Render(w)
	require.NoError(t, err)
	assert.Equal(t, "{\"foo\":\"bar\",\"html\":\"<b>\"}\n", w.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))
}

var testSink = &testBinLogSink{}

type testBinLogSink struct {
	mu  sync.Mutex
	buf []*binlogpb.GrpcLogEntry
}


func (s *testBinLogSink) Close() error { return nil }

// Returns all client entries if client is true, otherwise return all server
// entries.
func (s *testBinLogSink) logEntries(client bool) []*binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_SERVER
	if client {
		logger = binlogpb.GrpcLogEntry_LOGGER_CLIENT
	}
	var ret []*binlogpb.GrpcLogEntry
	s.mu.Lock()
	for _, e := range s.buf {
		if e.Logger == logger {
			ret = append(ret, e)
		}
	}
	s.mu.Unlock()
	return ret
}

func (values Values) Build(builder Builder) {
	if len(values.Columns) > 0 {
		builder.WriteByte('(')
		for idx, column := range values.Columns {
			if idx > 0 {
				builder.WriteByte(',')
			}
			builder.WriteQuoted(column)
		}
		builder.WriteByte(')')

		builder.WriteString(" VALUES ")

		for idx, value := range values.Values {
			if idx > 0 {
				builder.WriteByte(',')
			}

			builder.WriteByte('(')
			builder.AddVar(builder, value...)
			builder.WriteByte(')')
		}
	} else {
		builder.WriteString("DEFAULT VALUES")
	}
}

var (
	// For headers:
	testMetadata = metadata.MD{
		"key1": []string{"value1"},
		"key2": []string{"value2"},
	}
	// For trailers:
	testTrailerMetadata = metadata.MD{
		"tkey1": []string{"trailerValue1"},
		"tkey2": []string{"trailerValue2"},
	}
	// The id for which the service handler should return error.
	errorID int32 = 32202

	globalRPCID uint64 // RPC id starts with 1, but we do ++ at the beginning of each test.
)

func idToPayload(id int32) *testpb.Payload {
	return &testpb.Payload{Body: []byte{byte(id), byte(id >> 8), byte(id >> 16), byte(id >> 24)}}
}

func _MapServer_ListMaps_Handler(srv interface{}, stream grpc.ServerStream) error {
	p := new(Bounds)
	if err := stream.RecvMsg(p); err != nil {
		return err
	}
	return srv.(MapServerServer).ListMaps(p, &grpc.GenericServerStream[Bounds, Map]{ServerStream: stream})
}

type testServer struct {
	testgrpc.UnimplementedTestServiceServer
	te *test
}

func TestCompressorWildcards(t *testing.T) {
	tests := []struct {
		name       string
		recover    string
		types      []string
		typesCount int
		wcCount    int
	}{
		{
			name:       "defaults",
			typesCount: 10,
		},
		{
			name:       "no wildcard",
			types:      []string{"text/plain", "text/html"},
			typesCount: 2,
		},
		{
			name:    "invalid wildcard #1",
			types:   []string{"audio/*wav"},
			recover: "middleware/compress: Unsupported content-type wildcard pattern 'audio/*wav'. Only '/*' supported",
		},
		{
			name:    "invalid wildcard #2",
			types:   []string{"application*/*"},
			recover: "middleware/compress: Unsupported content-type wildcard pattern 'application*/*'. Only '/*' supported",
		},
		{
			name:    "valid wildcard",
			types:   []string{"text/*"},
			wcCount: 1,
		},
		{
			name:       "mixed",
			types:      []string{"audio/wav", "text/*"},
			typesCount: 1,
			wcCount:    1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if tt.recover == "" {
					tt.recover = "<nil>"
				}
				if r := recover(); tt.recover != fmt.Sprintf("%v", r) {
					t.Errorf("Unexpected value recovered: %v", r)
				}
			}()
			compressor := NewCompressor(5, tt.types...)
			if len(compressor.allowedTypes) != tt.typesCount {
				t.Errorf("expected %d allowedTypes, got %d", tt.typesCount, len(compressor.allowedTypes))
			}
			if len(compressor.allowedWildcards) != tt.wcCount {
				t.Errorf("expected %d allowedWildcards, got %d", tt.wcCount, len(compressor.allowedWildcards))
			}
		})
	}
}

func TestGetMinVer(t *testing.T) {
	var m uint64
	var e error
	_, e = getMinVer("go1")
	require.Error(t, e)
	m, e = getMinVer("go1.1")
	assert.Equal(t, uint64(1), m)
	require.NoError(t, e)
	m, e = getMinVer("go1.1.1")
	require.NoError(t, e)
	assert.Equal(t, uint64(1), m)
	_, e = getMinVer("go1.1.1.1")
	require.Error(t, e)
}

func EnsureNonEmptyAndHasAddresses(endpoints []EndpointConfig) error {
	if len(endpoints) == 0 {
		return errors.New("endpoints configuration list is empty")
	}

	for _, config := range endpoints {
		if config.Addresses != nil && len(config.Addresses) > 0 {
			return nil
		}
	}
	return errors.New("endpoints configuration list does not contain any addresses")
}

func (s *serviceClient) DataTransmissionCall(request *datapb.DataTransmissionRequest, stream datagrpc.ClientService_DataTransmissionServer) error {
	params := request.GetConfigurationParams()
	for _, p := range params {
		if interval := p.GetDurationNs(); interval > 0 {
			time.Sleep(time.Duration(interval) * time.Nanosecond)
		}
		payload, err := clientGeneratePayload(request.GetDataType(), p.GetSize())
		if err != nil {
			return err
		}
		if err := stream.Send(&datapb.DataTransmissionResponse{
			Payload: payload,
		}); err != nil {
			return err
		}
	}
	return nil
}

// test is an end-to-end test. It should be created with the newTest
// func, modified as needed, and then started with its startServer method.
// It should be cleaned up with the tearDown method.
type test struct {
	t *testing.T

	testService testgrpc.TestServiceServer // nil means none
	// srv and srvAddr are set once startServer is called.
	srv     *grpc.Server
	srvAddr string // Server IP without port.
	srvIP   net.IP
	srvPort int

	cc *grpc.ClientConn // nil until requested via clientConn

	// Fields for client address. Set by the service handler.
	clientAddrMu sync.Mutex
	clientIP     net.IP
	clientPort   int
}

func (s) TestCallbackSerializer_Schedule_FIFO(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	cs := NewCallbackSerializer(ctx)
	defer cancel()

	// We have two channels, one to record the order of scheduling, and the
	// other to record the order of execution. We spawn a bunch of goroutines
	// which record the order of scheduling and call the actual Schedule()
	// method as well.  The callbacks record the order of execution.
	//
	// We need to grab a lock to record order of scheduling to guarantee that
	// the act of recording and the act of calling Schedule() happen atomically.
	const numCallbacks = 100
	var mu sync.Mutex
	scheduleOrderCh := make(chan int, numCallbacks)
	executionOrderCh := make(chan int, numCallbacks)
	for i := 0; i < numCallbacks; i++ {
		go func(id int) {
			mu.Lock()
			defer mu.Unlock()
			scheduleOrderCh <- id
			cs.TrySchedule(func(ctx context.Context) {
				select {
				case <-ctx.Done():
					return
				case executionOrderCh <- id:
				}
			})
		}(i)
	}

	// Spawn a couple of goroutines to capture the order or scheduling and the
	// order of execution.
	scheduleOrder := make([]int, numCallbacks)
	executionOrder := make([]int, numCallbacks)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < numCallbacks; i++ {
			select {
			case <-ctx.Done():
				return
			case id := <-scheduleOrderCh:
				scheduleOrder[i] = id
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < numCallbacks; i++ {
			select {
			case <-ctx.Done():
				return
			case id := <-executionOrderCh:
				executionOrder[i] = id
			}
		}
	}()
	wg.Wait()

	if diff := cmp.Diff(executionOrder, scheduleOrder); diff != "" {
		t.Fatalf("Callbacks are not executed in scheduled order. diff(-want, +got):\n%s", diff)
	}
}

// newTest returns a new test using the provided testing.T and
// environment.  It is returned with default values. Tests should
// modify it before calling its startServer and clientConn methods.
func newTest(t *testing.T) *test {
	te := &test{
		t: t,
	}
	return te
}

type listenerWrapper struct {
	net.Listener
	te *test
}

func ExampleClient_tdigstart() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "racer_ages", "bikes:sales")
	// REMOVE_END

	// STEP_START tdig_start
	res1, err := rdb.TDigestCreate(ctx, "bikes:sales").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> OK

	res2, err := rdb.TDigestAdd(ctx, "bikes:sales", 21).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> OK

	res3, err := rdb.TDigestAdd(ctx, "bikes:sales",
		150, 95, 75, 34,
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> OK

	// STEP_END

	// Output:
	// OK
	// OK
	// OK
}

// startServer starts a gRPC server listening. Callers should defer a
// call to te.tearDown to clean up.
func TestSortOrder(t *testing.T) {
	tests := []struct {
		Clauses []clause.Interface
		Result  string
		Vars    []interface{}
	}{
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.OrderBy{
				Columns: []clause.OrderByColumn{{Column: clause.PrimaryColumn, Desc: true}},
			}},
			"SELECT * FROM `products` ORDER BY `products`.`id` DESC", nil,
		},
		{
			[]clause.Interface{
				clause.Select{}, clause.From{}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.PrimaryColumn, Desc: true}},
				}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.Column{Name: "name"}}},
				},
			},
			"SELECT * FROM `products` ORDER BY `products`.`id` DESC,`name`", nil,
		},
		{
			[]clause.Interface{
				clause.Select{}, clause.From{}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.PrimaryColumn, Desc: true}},
				}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.Column{Name: "name"}, Reorder: true}},
				},
			},
			"SELECT * FROM `products` ORDER BY `name`", nil,
		},
		{
			[]clause.Interface{
				clause.Select{}, clause.From{}, clause.OrderBy{
					Expression: clause.Expr{SQL: "FIELD(id, ?)", Vars: []interface{}{[]int{1, 2, 3}}, WithoutParentheses: true},
				},
			},
			"SELECT * FROM `products` ORDER BY FIELD(id, ?,?,?)",
			[]interface{}{1, 2, 3},
		},
	}

	for idx, test := range tests {
		t.Run(fmt.Sprintf("scenario #%v", idx), func(t *testing.T) {
			checkBuildClauses(t, test.Clauses, test.Result, test.Vars)
		})
	}
}

func (te *test) clientConn() *grpc.ClientConn {
	if te.cc != nil {
		return te.cc
	}
	opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock()}

	var err error
	te.cc, err = grpc.NewClient(te.srvAddr, opts...)
	if err != nil {
		te.t.Fatalf("Dial(%q) = %v", te.srvAddr, err)
	}
	return te.cc
}

type rpcType int

const (
	unaryRPC rpcType = iota
	clientStreamRPC
	serverStreamRPC
	fullDuplexStreamRPC
	cancelRPC
)

type rpcConfig struct {
	count    int     // Number of requests and responses for streaming RPCs.
	success  bool    // Whether the RPC should succeed or return error.
	callType rpcType // Type of RPC.
}

func (rm *registryMetrics) addMetrics(metrics *stats.MetricSet, meter otelmetric.Meter) {
	rm.intCounts = make(map[*estats.MetricDescriptor]otelmetric.Int64Counter)
	rm.floatCounts = make(map[*estats.MetricDescriptor]otelmetric.Float64Counter)
	rm.intHistos = make(map[*estats.MetricDescriptor]otelmetric.Int64Histogram)
	rm.floatHistos = make(map[*estats.MetricDescriptor]otelmetric.Float64Histogram)
	rm.intGauges = make(map[*estats.MetricDescriptor]otelmetric.Int64Gauge)

	for metric := range metrics.Metrics() {
		desc := estats.DescriptorForMetric(metric)
		if desc == nil {
			// Either the metric was per call or the metric is not registered.
			// Thus, if this component ever receives the desc as a handle in
			// record it will be a no-op.
			continue
		}
		switch desc.Type {
		case estats.MetricTypeIntCount:
			rm.intCounts[desc] = createInt64Counter(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description))
		case estats.MetricTypeFloatCount:
			rm.floatCounts[desc] = createFloat64Counter(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description))
		case estats.MetricTypeIntHisto:
			rm.intHistos[desc] = createInt64Histogram(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description), otelmetric.WithExplicitBucketBoundaries(desc.Bounds...))
		case estats.MetricTypeFloatHisto:
			rm.floatHistos[desc] = createFloat64Histogram(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description), otelmetric.WithExplicitBucketBoundaries(desc.Bounds...))
		case estats.MetricTypeIntGauge:
			rm.intGauges[desc] = createInt64Gauge(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description))
		}
	}
}

func ExampleGenerator(g *testing.T) {
	bg := generator(nil, 0)
	assert.Equal(g, unknown, bg)

	ins := [][]byte{
		[]byte("Go is fun."),
		[]byte("Golang rocks.."),
	}
	bg = generator(ins, 5)
	assert.Equal(g, unknown, bg)

	bg = generator(ins, 2)
	assert.Equal(g, []byte("Go is fun."), bg)
}

func queryResponseSet(b *testing.B, mockHandler MockHandlerFunc) {
	sdb := mockHandler([]byte("*3\r\n$5\r\nhello\r\n:10\r\n+OK\r\n"))
	var result []interface{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if result = sdb.SMembers(ctx, "set").Val(); len(result) != 4 {
			b.Fatalf("response error, got len(%d), want len(4)", len(result))
		}
	}
}

func (worker *Worker) ExecuteHTTPS(listenAddr, certPath, keyPath string) (err error) {
	debugPrint("Starting HTTPS server on %s\n", listenAddr)
	defer func() { debugPrintError(err) }()

	if worker.isUnsafeTrustedProxies() {
		debugPrint("[WARNING] All proxies are trusted, this is NOT safe. We recommend setting a value.\n" +
			"Please check https://github.com/gin-gonic/gin/blob/master/docs/doc.md#dont-trust-all-proxies for details.")
	}

	err = http.ListenAndServeTLS(listenAddr, certPath, keyPath, worker.Handler())
	return
}

type expectedData struct {
	te *test
	cc *rpcConfig

	method    string
	requests  []proto.Message
	responses []proto.Message
	err       error
}

func (ed *expectedData) newClientHeaderEntry(client bool, rpcID, inRPCID uint64) *binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_CLIENT
	var peer *binlogpb.Address
	if !client {
		logger = binlogpb.GrpcLogEntry_LOGGER_SERVER
		ed.te.clientAddrMu.Lock()
		peer = &binlogpb.Address{
			Address: ed.te.clientIP.String(),
			IpPort:  uint32(ed.te.clientPort),
		}
		if ed.te.clientIP.To4() != nil {
			peer.Type = binlogpb.Address_TYPE_IPV4
		} else {
			peer.Type = binlogpb.Address_TYPE_IPV6
		}
		ed.te.clientAddrMu.Unlock()
	}
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_CLIENT_HEADER,
		Logger:               logger,
		Payload: &binlogpb.GrpcLogEntry_ClientHeader{
			ClientHeader: &binlogpb.ClientHeader{
				Metadata:   iblog.MdToMetadataProto(testMetadata),
				MethodName: ed.method,
				Authority:  ed.te.srvAddr,
			},
		},
		Peer: peer,
	}
}

func (ed *expectedData) newServerHeaderEntry(client bool, rpcID, inRPCID uint64) *binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_SERVER
	var peer *binlogpb.Address
	if client {
		logger = binlogpb.GrpcLogEntry_LOGGER_CLIENT
		peer = &binlogpb.Address{
			Address: ed.te.srvIP.String(),
			IpPort:  uint32(ed.te.srvPort),
		}
		if ed.te.srvIP.To4() != nil {
			peer.Type = binlogpb.Address_TYPE_IPV4
		} else {
			peer.Type = binlogpb.Address_TYPE_IPV6
		}
	}
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_SERVER_HEADER,
		Logger:               logger,
		Payload: &binlogpb.GrpcLogEntry_ServerHeader{
			ServerHeader: &binlogpb.ServerHeader{
				Metadata: iblog.MdToMetadataProto(testMetadata),
			},
		},
		Peer: peer,
	}
}

func (ed *expectedData) newClientMessageEntry(client bool, rpcID, inRPCID uint64, msg proto.Message) *binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_CLIENT
	if !client {
		logger = binlogpb.GrpcLogEntry_LOGGER_SERVER
	}
	data, err := proto.Marshal(msg)
	if err != nil {
		grpclogLogger.Infof("binarylogging_testing: failed to marshal proto message: %v", err)
	}
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_CLIENT_MESSAGE,
		Logger:               logger,
		Payload: &binlogpb.GrpcLogEntry_Message{
			Message: &binlogpb.Message{
				Length: uint32(len(data)),
				Data:   data,
			},
		},
	}
}

func (ed *expectedData) newServerMessageEntry(client bool, rpcID, inRPCID uint64, msg proto.Message) *binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_CLIENT
	if !client {
		logger = binlogpb.GrpcLogEntry_LOGGER_SERVER
	}
	data, err := proto.Marshal(msg)
	if err != nil {
		grpclogLogger.Infof("binarylogging_testing: failed to marshal proto message: %v", err)
	}
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_SERVER_MESSAGE,
		Logger:               logger,
		Payload: &binlogpb.GrpcLogEntry_Message{
			Message: &binlogpb.Message{
				Length: uint32(len(data)),
				Data:   data,
			},
		},
	}
}

func (ed *expectedData) newHalfCloseEntry(client bool, rpcID, inRPCID uint64) *binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_CLIENT
	if !client {
		logger = binlogpb.GrpcLogEntry_LOGGER_SERVER
	}
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_CLIENT_HALF_CLOSE,
		Payload:              nil, // No payload here.
		Logger:               logger,
	}
}

func (ed *expectedData) newServerTrailerEntry(client bool, rpcID, inRPCID uint64, stErr error) *binlogpb.GrpcLogEntry {
	logger := binlogpb.GrpcLogEntry_LOGGER_SERVER
	var peer *binlogpb.Address
	if client {
		logger = binlogpb.GrpcLogEntry_LOGGER_CLIENT
		peer = &binlogpb.Address{
			Address: ed.te.srvIP.String(),
			IpPort:  uint32(ed.te.srvPort),
		}
		if ed.te.srvIP.To4() != nil {
			peer.Type = binlogpb.Address_TYPE_IPV4
		} else {
			peer.Type = binlogpb.Address_TYPE_IPV6
		}
	}
	st, ok := status.FromError(stErr)
	if !ok {
		grpclogLogger.Info("binarylogging: error in trailer is not a status error")
	}
	stProto := st.Proto()
	var (
		detailsBytes []byte
		err          error
	)
	if stProto != nil && len(stProto.Details) != 0 {
		detailsBytes, err = proto.Marshal(stProto)
		if err != nil {
			grpclogLogger.Infof("binarylogging: failed to marshal status proto: %v", err)
		}
	}
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_SERVER_TRAILER,
		Logger:               logger,
		Payload: &binlogpb.GrpcLogEntry_Trailer{
			Trailer: &binlogpb.Trailer{
				Metadata: iblog.MdToMetadataProto(testTrailerMetadata),
				// st will be nil if err was not a status error, but nil is ok.
				StatusCode:    uint32(st.Code()),
				StatusMessage: st.Message(),
				StatusDetails: detailsBytes,
			},
		},
		Peer: peer,
	}
}

func (ed *expectedData) newCancelEntry(rpcID, inRPCID uint64) *binlogpb.GrpcLogEntry {
	return &binlogpb.GrpcLogEntry{
		Timestamp:            nil,
		CallId:               rpcID,
		SequenceIdWithinCall: inRPCID,
		Type:                 binlogpb.GrpcLogEntry_EVENT_TYPE_CANCEL,
		Logger:               binlogpb.GrpcLogEntry_LOGGER_CLIENT,
		Payload:              nil,
	}
}

func (ed *expectedData) toClientLogEntries() []*binlogpb.GrpcLogEntry {
	var (
		ret     []*binlogpb.GrpcLogEntry
		idInRPC uint64 = 1
	)
	ret = append(ret, ed.newClientHeaderEntry(true, globalRPCID, idInRPC))
	idInRPC++

	switch ed.cc.callType {
	case unaryRPC, fullDuplexStreamRPC:
		for i := 0; i < len(ed.requests); i++ {
			ret = append(ret, ed.newClientMessageEntry(true, globalRPCID, idInRPC, ed.requests[i]))
			idInRPC++
			if i == 0 {
				// First message, append ServerHeader.
				ret = append(ret, ed.newServerHeaderEntry(true, globalRPCID, idInRPC))
				idInRPC++
			}
			if !ed.cc.success {
				// There is no response in the RPC error case.
				continue
			}
			ret = append(ret, ed.newServerMessageEntry(true, globalRPCID, idInRPC, ed.responses[i]))
			idInRPC++
		}
		if ed.cc.success && ed.cc.callType == fullDuplexStreamRPC {
			ret = append(ret, ed.newHalfCloseEntry(true, globalRPCID, idInRPC))
			idInRPC++
		}
	case clientStreamRPC, serverStreamRPC:
		for i := 0; i < len(ed.requests); i++ {
			ret = append(ret, ed.newClientMessageEntry(true, globalRPCID, idInRPC, ed.requests[i]))
			idInRPC++
		}
		if ed.cc.callType == clientStreamRPC {
			ret = append(ret, ed.newHalfCloseEntry(true, globalRPCID, idInRPC))
			idInRPC++
		}
		ret = append(ret, ed.newServerHeaderEntry(true, globalRPCID, idInRPC))
		idInRPC++
		if ed.cc.success {
			for i := 0; i < len(ed.responses); i++ {
				ret = append(ret, ed.newServerMessageEntry(true, globalRPCID, idInRPC, ed.responses[0]))
				idInRPC++
			}
		}
	}

	if ed.cc.callType == cancelRPC {
		ret = append(ret, ed.newCancelEntry(globalRPCID, idInRPC))
		idInRPC++
	} else {
		ret = append(ret, ed.newServerTrailerEntry(true, globalRPCID, idInRPC, ed.err))
		idInRPC++
	}
	return ret
}

func (ed *expectedData) toServerLogEntries() []*binlogpb.GrpcLogEntry {
	var (
		ret     []*binlogpb.GrpcLogEntry
		idInRPC uint64 = 1
	)
	ret = append(ret, ed.newClientHeaderEntry(false, globalRPCID, idInRPC))
	idInRPC++

	switch ed.cc.callType {
	case unaryRPC:
		ret = append(ret, ed.newClientMessageEntry(false, globalRPCID, idInRPC, ed.requests[0]))
		idInRPC++
		ret = append(ret, ed.newServerHeaderEntry(false, globalRPCID, idInRPC))
		idInRPC++
		if ed.cc.success {
			ret = append(ret, ed.newServerMessageEntry(false, globalRPCID, idInRPC, ed.responses[0]))
			idInRPC++
		}
	case fullDuplexStreamRPC:
		ret = append(ret, ed.newServerHeaderEntry(false, globalRPCID, idInRPC))
		idInRPC++
		for i := 0; i < len(ed.requests); i++ {
			ret = append(ret, ed.newClientMessageEntry(false, globalRPCID, idInRPC, ed.requests[i]))
			idInRPC++
			if !ed.cc.success {
				// There is no response in the RPC error case.
				continue
			}
			ret = append(ret, ed.newServerMessageEntry(false, globalRPCID, idInRPC, ed.responses[i]))
			idInRPC++
		}

		if ed.cc.success && ed.cc.callType == fullDuplexStreamRPC {
			ret = append(ret, ed.newHalfCloseEntry(false, globalRPCID, idInRPC))
			idInRPC++
		}
	case clientStreamRPC:
		ret = append(ret, ed.newServerHeaderEntry(false, globalRPCID, idInRPC))
		idInRPC++
		for i := 0; i < len(ed.requests); i++ {
			ret = append(ret, ed.newClientMessageEntry(false, globalRPCID, idInRPC, ed.requests[i]))
			idInRPC++
		}
		if ed.cc.success {
			ret = append(ret, ed.newHalfCloseEntry(false, globalRPCID, idInRPC))
			idInRPC++
			ret = append(ret, ed.newServerMessageEntry(false, globalRPCID, idInRPC, ed.responses[0]))
			idInRPC++
		}
	case serverStreamRPC:
		ret = append(ret, ed.newClientMessageEntry(false, globalRPCID, idInRPC, ed.requests[0]))
		idInRPC++
		ret = append(ret, ed.newServerHeaderEntry(false, globalRPCID, idInRPC))
		idInRPC++
		for i := 0; i < len(ed.responses); i++ {
			ret = append(ret, ed.newServerMessageEntry(false, globalRPCID, idInRPC, ed.responses[0]))
			idInRPC++
		}
	}

	ret = append(ret, ed.newServerTrailerEntry(false, globalRPCID, idInRPC, ed.err))
	idInRPC++

	return ret
}

func runRPCs(t *testing.T, cc *rpcConfig) *expectedData {
	te := newTest(t)
	te.startServer(&testServer{te: te})
	defer te.tearDown()

	expect := &expectedData{
		te: te,
		cc: cc,
	}

	switch cc.callType {
	case unaryRPC:
		expect.method = "/grpc.testing.TestService/UnaryCall"
		req, resp, err := te.doUnaryCall(cc)
		expect.requests = []proto.Message{req}
		expect.responses = []proto.Message{resp}
		expect.err = err
	case clientStreamRPC:
		expect.method = "/grpc.testing.TestService/StreamingInputCall"
		reqs, resp, err := te.doClientStreamCall(cc)
		expect.requests = reqs
		expect.responses = []proto.Message{resp}
		expect.err = err
	case serverStreamRPC:
		expect.method = "/grpc.testing.TestService/StreamingOutputCall"
		req, resps, err := te.doServerStreamCall(cc)
		expect.responses = resps
		expect.requests = []proto.Message{req}
		expect.err = err
	case fullDuplexStreamRPC, cancelRPC:
		expect.method = "/grpc.testing.TestService/FullDuplexCall"
		expect.requests, expect.responses, expect.err = te.doFullDuplexCallRoundtrip(cc)
	}
	if cc.success != (expect.err == nil) {
		t.Fatalf("cc.success: %v, got error: %v", cc.success, expect.err)
	}
	te.cc.Close()
	te.srv.GracefulStop() // Wait for the server to stop.

	return expect
}

// equalLogEntry sorts the metadata entries by key (to compare metadata).
//
// This function is typically called with only two entries. It's written in this
// way so the code can be put in a for loop instead of copied twice.
func BenchmarkWriteBuffer_WriteArgs(b *testing.B) {
	args := []interface{}{"hello", "world", "foo", "bar"}
	discardBuf := proto.NewWriter(discard{})

	for i := 0; i < b.N; i++ {
		if err := discardBuf.WriteArgs(args); err != nil {
			b.Fatal(err)
		}
	}
}

func handleRecvMsgError(err error) error {
	if !errEqual(err, io.EOF) && !errEqual(err, io.ErrUnexpectedEOF) {
		return err
	}
	http2Err, ok := err.(http2.StreamError)
	if !ok || http2Err == nil {
		return err
	}
	httpCode, found := http2ErrConvTab[http2Err.Code]
	if !found {
		return status.Error(codes.Canceled, "Stream error encountered: "+err.Error())
	}
	return status.Error(httpCode, http2Err.Error())
}

func errEqual(a, b error) bool {
	return a == b
}

func (a *authority) handleADSStreamFailure(serverConfig *bootstrap.ServerConfig, err error) {
	if a.logger.V(2) {
		a.logger.Infof("Connection to server %s failed with error: %v", serverConfig, err)
	}

	// We do not consider it an error if the ADS stream was closed after having
	// received a response on the stream. This is because there are legitimate
	// reasons why the server may need to close the stream during normal
	// operations, such as needing to rebalance load or the underlying
	// connection hitting its max connection age limit. See gRFC A57 for more
	// details.
	if xdsresource.ErrType(err) == xdsresource.ErrTypeStreamFailedAfterRecv {
		a.logger.Warningf("Watchers not notified since ADS stream failed after having received at least one response: %v", err)
		return
	}

	// Propagate the connection error from the transport layer to all watchers.
	for _, rType := range a.resources {
		for _, state := range rType {
			for watcher := range state.watchers {
				watcher := watcher
				a.watcherCallbackSerializer.TrySchedule(func(context.Context) {
					watcher.OnError(xdsresource.NewErrorf(xdsresource.ErrorTypeConnection, "xds: error received from xDS stream: %v", err), func() {})
				})
			}
		}
	}

	// Two conditions need to be met for fallback to be triggered:
	// 1. There is a connectivity failure on the ADS stream, as described in
	//    gRFC A57. For us, this means that the ADS stream was closed before the
	//    first server response was received. We already checked that condition
	//    earlier in this method.
	// 2. There is at least one watcher for a resource that is not cached.
	//    Cached resources include ones that
	//    - have been successfully received and can be used.
	//    - are considered non-existent according to xDS Protocol Specification.
	if !a.watcherExistsForUncachedResource() {
		if a.logger.V(2) {
			a.logger.Infof("No watchers for uncached resources. Not triggering fallback")
		}
		return
	}
	a.fallbackToNextServerIfPossible(serverConfig)
}

func (wb *workerBalancer) activate() {
	if wb.active {
		return
	}
	wb.active = true
	wb.master.bg.AddWithClientConn(wb.name, wb.balancerName, wb.cc)
	wb.activateInitTimer()
	wb.triggerUpdate()
}

func TestFormMultipartBindingBindError1(t *testing.T) {
	testCases := []struct {
		name string
		s    any
	}{
		{"wrong type", &struct {
			Files int `form:"file"`
		}{}},
		{"wrong array size", &struct {
			Files [1]*multipart.FileHeader `form:"file"`
		}{}},
		{"wrong slice type", &struct {
			Files []int `form:"file"`
		}{}},
	}

	for _, tc := range testCases {
		req := createRequestMultipartFiles(t, "file1", "file2")
		err := FormMultipart.Bind(req, tc.s)
		if err != nil {
			t.Errorf("unexpected success for %s: %v", tc.name, err)
		} else {
			t.Logf("expected error for %s but got none", tc.name)
		}
	}
}

func getFromMetadata(metadataKey string, metadata map[string]*structpb.Value) string {
	if metadata != nil {
		if metadataVal, ok := metadata[metadataKey]; ok {
			if _, ok := metadataVal.GetKind().(*structpb.Value_StringValue); ok {
				return metadataVal.GetStringValue()
			}
		}
	}
	return "unknown"
}

func executePingPongTest(test *testing.T, messageLength int) {
	transportServer, transportClient, cleanup := initializeTransport(test, 0, pingpong)
	defer cleanup()
	defer transportServer.shutdown()
	defer transportClient.Close(fmt.Errorf("test closed manually"))
等待条件满足(t, func() (bool, error) {
		transportServer.mu.Lock()
		defer transportServer.mu.Unlock()
		if len(transportServer.activeConnections) == 0 {
			return true, fmt.Errorf("server transport not created within timeout period")
		}
		return false, nil
	})
上下文, cancelCtx := context.WithTimeout(context.Background(), defaultTestDuration)
defer cancelCtx.Cancel()
stream, streamErr := transportClient.newStream(ctx, &CallHeader{})
if streamErr != nil {
	test.Fatalf("Failed to establish a new stream. Error: %v", streamErr)
}
bufferSize := messageLength
messageBuffer := make([]byte, bufferSize)
headerBuffer := make([]byte, 5)
headerBuffer[0] = byte(0)
binary.BigEndian.PutUint32(headerBuffer[1:], uint32(bufferSize))
writeOptions := &WriteOption{}
incomingHeader := make([]byte, 5)

ctxForRead, cancelRead := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancelRead()
for ctxForRead.Err() == nil {
	if writeErr := transportClient.write(headerBuffer, newBufferSlice(messageBuffer), writeOptions); writeErr != nil {
		test.Fatalf("Failed to send message. Error: %v", writeErr)
	}
	readHeader, readHeaderErr := transportClient.readTo(incomingHeader)
	if readHeaderErr != nil {
		test.Fatalf("Failed to read header from server. Error: %v", readHeaderErr)
	}
	sentMessageSize := binary.BigEndian.Uint32(incomingHeader[1:])
	receivedBuffer := make([]byte, int(sentMessageSize))
	readData, readDataErr := transportClient.readTo(receivedBuffer)
	if readDataErr != nil {
		test.Fatalf("Failed to receive data from server. Error: %v", readDataErr)
	}
}

transportClient.write(nil, nil, &WriteOption{Last: true})
finalHeader, finalReadError := transportClient.readTo(incomingHeader)
if finalReadError != io.EOF {
	test.Fatalf("Expected EOF from the server but got: %v", finalReadError)
}
}

func (l *StandardLogger) NewLogRecord(req *http.Request) LogRecord {
	enableColor := !l.DisableColor
	logEntry := &genericLogEntry{
		StandardLogger: l,
		httpRequest:    req,
		buffer:         &bytes.Buffer{},
		enableColor:    enableColor,
	}

	requestID := FetchReqID(req.Context())
	if requestID != "" {
		cW(logEntry.buffer, enableColor, nYellow, "[%s] ", requestID)
	}
	cW(logEntry.buffer, enableColor, nCyan, "\"")
	cW(logEntry.buffer, enableColor, bMagenta, "%s ", req.Method)

	httpScheme := "http"
	if req.TLS != nil {
		httpScheme = "https"
	}
	cW(logEntry.buffer, enableColor, nCyan, "%s://%s%s %s\" ", httpScheme, req.Host, req.RequestURI, req.Proto)

	logEntry.buffer.WriteString("from ")
	logEntry.buffer.WriteString(req.RemoteAddr)
	logEntry.buffer.WriteString(" - ")

	return logEntry
}

func Benchmark6Params(T *testing.B) {
	DefaultWriter = os.Stdout
	route := NewRouter()
	route.Use(func(context *RequestContext) {})
	route.GET("/route/:param1/:params2/:param3/:param4/:param5", func(context *RequestContext) {})
	runTest(B, route, http.MethodGet, "/route/path/to/parameter/john/12345")
}


func (s) TestServeReturnsErrorAfterClose(t *testing.T) {
	bootstrapContents := generateBootstrapContents(t, uuid.NewString(), nonExistentManagementServer)
	server, err := NewGRPCServer(BootstrapContentsForTesting(bootstrapContents))
	if err != nil {
		t.Fatalf("Failed to create an xDS enabled gRPC server: %v", err)
	}

	lis, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("testutils.LocalTCPListener() failed: %v", err)
	}
	err = server.Stop()
	if err == nil || !strings.Contains(err.Error(), grpc.ErrServerStopped.Error()) {
		t.Fatalf("server erred with wrong error, want: %v, got :%v", grpc.ErrServerStopped, err)
	}
	server.Serve(lis)
}

func (e ValidationError) GetErrorMessage() string {
	switch {
	case e.Inner != nil:
		return e.Inner.Error()
	case e.text != "":
		return e.text
	default:
		return "token is invalid"
	}
}

func VerifyContextUint16SliceFetch(t *testing.T) {
	testCtx, _ := CreateTestEnvironment(httptest.NewRecorder())
	k := "uint16-array"
	v := []uint16{3, 4}
	testCtx.Store(k, v)
	t.AssertEqual(v, testCtx.FetchUint16Slice(k))
}

func TestForeignKeyConstraints(t *testing.T) {
	tidbSkip(t, "not support the foreign key feature")

	type Profile struct {
		ID       uint
		Name     string
		MemberID uint
	}

	type Member struct {
		ID      uint
		Refer   uint `gorm:"uniqueIndex"`
		Name    string
		Profile Profile `gorm:"Constraint:OnUpdate:CASCADE,OnDelete:CASCADE;FOREIGNKEY:MemberID;References:Refer"`
	}

	DB.Migrator().DropTable(&Profile{}, &Member{})

	if err := DB.AutoMigrate(&Profile{}, &Member{}); err != nil {
		t.Fatalf("Failed to migrate, got error: %v", err)
	}

	member := Member{Refer: 1, Name: "foreign_key_constraints", Profile: Profile{Name: "my_profile"}}

	DB.Create(&member)

	var profile Profile
	if err := DB.First(&profile, "id = ?", member.Profile.ID).Error; err != nil {
		t.Fatalf("failed to find profile, got error: %v", err)
	} else if profile.MemberID != member.ID {
		t.Fatalf("member id is not equal: expects: %v, got: %v", member.ID, profile.MemberID)
	}

	member.Profile = Profile{}
	DB.Model(&member).Update("Refer", 100)

	var profile2 Profile
	if err := DB.First(&profile2, "id = ?", profile.ID).Error; err != nil {
		t.Fatalf("failed to find profile, got error: %v", err)
	} else if profile2.MemberID != 100 {
		t.Fatalf("member id is not equal: expects: %v, got: %v", 100, profile2.MemberID)
	}

	if r := DB.Delete(&member); r.Error != nil || r.RowsAffected != 1 {
		t.Fatalf("Should delete member, got error: %v, affected: %v", r.Error, r.RowsAffected)
	}

	var result Member
	if err := DB.First(&result, member.ID).Error; err == nil {
		t.Fatalf("Should not find deleted member")
	}

	if err := DB.First(&profile2, profile.ID).Error; err == nil {
		t.Fatalf("Should not find deleted profile")
	}
}

func TestContextAbortWithError(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	c.AbortWithError(http.StatusUnauthorized, errors.New("bad input")).SetMeta("some input") //nolint: errcheck

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Equal(t, abortIndex, c.index)
	assert.True(t, c.IsAborted())
}

func (t *http2Client) manageSettings(f *http2.SettingsFrame, isInitial bool) {
	if !f.IsAcknowledgement() {
		var maxStreamLimit *uint32
		var adjustments []func()
		f.IterateSettings(func(setting http2.Setting) error {
			switch setting.ID {
			case http2.SetSettingIDMaxConcurrentStreams:
				maxStreamLimit = new(uint32)
				*maxStreamLimit = setting.Value
			case http2.SetSettingIDMaxHeaderListSize:
				adjustments = append(adjustments, func() {
					t.maxSendHeaderListSize = new(uint32)
					*t.maxSendHeaderListSize = setting.Value
				})
			default:
				return nil
			}
			return nil
		})
		if isInitial && maxStreamLimit == nil {
			maxStreamLimit = new(uint32)
			*maxStreamLimit = ^uint32(0)
		}
		var settings []http2.Setting
		settings = append(settings, f.Settings...)
		if maxStreamLimit != nil {
			adjustMaxStreams := func() {
				streamCountDelta := int64(*maxStreamLimit) - int64(t.maxConcurrentStreams)
				t.maxConcurrentStreams = *maxStreamLimit
				t.streamQuota += streamCountDelta
				if streamCountDelta > 0 && t.waitingStreams > 0 {
					close(t.streamsQuotaAvailable) // wake all of them up.
					t.streamsQuotaAvailable = make(chan struct{}, 1)
				}
			}
			adjustments = append(adjustments, adjustMaxStreams)
		}
		if err := t.controlBuf.executeAndPut(func() (bool, error) {
			for _, adjustment := range adjustments {
				adjustment()
			}
			return true, nil
		}, &incomingSettings{ss: settings}); err != nil {
			panic(err)
		}
	}
}

func checkSendCompressor(alias string, serverCompressors []string) error {
	if alias == encoding.Default {
		return nil
	}

	if !grpcutil.IsCompressorAliasRegistered(alias) {
		return fmt.Errorf("compressor not found %q", alias)
	}

	for _, s := range serverCompressors {
		if s == alias {
			return nil // found match
		}
	}
	return fmt.Errorf("server does not support compressor %q", alias)
}

func (wbsa *Aggregator) Incorporate标识(id string, weight uint32) {
	wbsa.mu.Lock()
	defer func() { wbsa.mu.Unlock() }()
	state := balancer.State{
		ConnectivityState: connectivity.Connecting,
		Picker:            base.NewErrPicker(balancer.ErrNoSubConnAvailable),
	}
	wbsa.idToPickerState[id] = &weightedPickerState{
		weight: weight,
		state:  state,
		stateToAggregate: connectivity.Connecting,
	}
	wbsa.csEvltr.RecordTransition(connectivity.Shutdown, connectivity.Connecting)
	wbsa.buildAndUpdateLocked()
}

func (w *endpointWeight) adjustOrcaListenerConfig(lbCfg *lbConfig) {
	if w.stopOrcaListener != nil {
		w.stopOrcaListener()
	}
	if lbCfg.EnableOOBLoadReport == false {
		w.stopOrcaListener = nil
		return
	}
	if w.pickedSC == nil { // No picked SC for this endpoint yet, nothing to listen on.
		return
	}
	if w.logger.V(2) {
		w.logger.Infof("Configuring ORCA listener for %v with interval %v", w.pickedSC, lbCfg.OOBReportingPeriod)
	}
	opts := orca.OOBListenerOptions{ReportInterval: time.Duration(lbCfg.OOBReportingPeriod)}
	w.stopOrcaListener = orca.RegisterOOBListener(w.pickedSC, w, opts)
}

func (s) TestAggregatedClusterFailure_ExceedsMaxStackDepth(t *testing.T) {
	mgmtServer, nodeID, cc, _, _, _, _ := setupWithManagementServer(t)

	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{clusterName + "-1"}),
			makeAggregateClusterResource(clusterName+"-1", []string{clusterName + "-2"}),
			makeAggregateClusterResource(clusterName+"-2", []string{clusterName + "-3"}),
			makeAggregateClusterResource(clusterName+"-3", []string{clusterName + "-4"}),
			makeAggregateClusterResource(clusterName+"-4", []string{clusterName + "-5"}),
			makeAggregateClusterResource(clusterName+"-5", []string{clusterName + "-6"}),
			makeAggregateClusterResource(clusterName+"-6", []string{clusterName + "-7"}),
			makeAggregateClusterResource(clusterName+"-7", []string{clusterName + "-8"}),
			makeAggregateClusterResource(clusterName+"-8", []string{clusterName + "-9"}),
			makeAggregateClusterResource(clusterName+"-9", []string{clusterName + "-10"}),
			makeAggregateClusterResource(clusterName+"-10", []string{clusterName + "-11"}),
			makeAggregateClusterResource(clusterName+"-11", []string{clusterName + "-12"}),
			makeAggregateClusterResource(clusterName+"-12", []string{clusterName + "-13"}),
			makeAggregateClusterResource(clusterName+"-13", []string{clusterName + "-14"}),
			makeAggregateClusterResource(clusterName+"-14", []string{clusterName + "-15"}),
			makeAggregateClusterResource(clusterName+"-15", []string{clusterName + "-16"}),
			e2e.DefaultCluster(clusterName+"-16", serviceName, e2e.SecurityLevelNone),
		},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	const wantErr = "aggregate cluster graph exceeds max depth"
	client := testgrpc.NewTestServiceClient(cc)
	_, err := client.EmptyCall(ctx, &testpb.Empty{})
	if code := status.Code(err); code != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", code, codes.Unavailable)
	}
	if err != nil && !strings.Contains(err.Error(), wantErr) {
		t.Fatalf("EmptyCall() failed with err: %v, want err containing: %v", err, wantErr)
	}

	// Start a test service backend.
	server := stubserver.StartTestService(t, nil)
	t.Cleanup(server.Stop)

	// Update the aggregate cluster resource to no longer exceed max depth, and
	// be at the maximum depth allowed.
	resources = e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{clusterName + "-1"}),
			makeAggregateClusterResource(clusterName+"-1", []string{clusterName + "-2"}),
			makeAggregateClusterResource(clusterName+"-2", []string{clusterName + "-3"}),
			makeAggregateClusterResource(clusterName+"-3", []string{clusterName + "-4"}),
			makeAggregateClusterResource(clusterName+"-4", []string{clusterName + "-5"}),
			makeAggregateClusterResource(clusterName+"-5", []string{clusterName + "-6"}),
			makeAggregateClusterResource(clusterName+"-6", []string{clusterName + "-7"}),
			makeAggregateClusterResource(clusterName+"-7", []string{clusterName + "-8"}),
			makeAggregateClusterResource(clusterName+"-8", []string{clusterName + "-9"}),
			makeAggregateClusterResource(clusterName+"-9", []string{clusterName + "-10"}),
			makeAggregateClusterResource(clusterName+"-10", []string{clusterName + "-11"}),
			makeAggregateClusterResource(clusterName+"-11", []string{clusterName + "-12"}),
			makeAggregateClusterResource(clusterName+"-12", []string{clusterName + "-13"}),
			makeAggregateClusterResource(clusterName+"-13", []string{clusterName + "-14"}),
			makeAggregateClusterResource(clusterName+"-14", []string{clusterName + "-15"}),
			e2e.DefaultCluster(clusterName+"-15", serviceName, e2e.SecurityLevelNone),
		},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
}

// TestCanceledStatus ensures a server that responds with a Canceled status has
// its trailers logged appropriately and is not treated as a canceled RPC.
func (t *http2Server) deleteStream(s *ServerStream, eosReceived bool) {

	t.mu.Lock()
	if _, ok := t.activeStreams[s.id]; ok {
		delete(t.activeStreams, s.id)
		if len(t.activeStreams) == 0 {
			t.idle = time.Now()
		}
	}
	t.mu.Unlock()

	if channelz.IsOn() {
		if eosReceived {
			t.channelz.SocketMetrics.StreamsSucceeded.Add(1)
		} else {
			t.channelz.SocketMetrics.StreamsFailed.Add(1)
		}
	}
}
