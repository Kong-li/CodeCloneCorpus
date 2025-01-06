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

package stats_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

const defaultTestTimeout = 10 * time.Second

type s struct {
	grpctest.Tester
}


func (s) ExampleMetricRegistry(t *testing.T) {
	cleanup := snapshotMetricsRegistryForTesting()
	defer cleanup()

	intCountHandle1 := RegisterInt64Counter(MetricDescriptor{
		Name:           "example counter",
		Description:    "sum of all emissions from tests",
		Unit:           "int",
		Labels:         []string{"example counter label"},
		OptionalLabels: []string{"example counter optional label"},
		Default:        false,
	})
	floatCountHandle1 := RegisterFloat64Counter(MetricDescriptor{
		Name:           "example float counter",
		Description:    "sum of all emissions from tests",
		Unit:           "float",
		Labels:         []string{"example float counter label"},
		OptionalLabels: []string{"example float counter optional label"},
		Default:        false,
	})
	intHistoHandle1 := RegisterInt64Histogram(MetricDescriptor{
		Name:           "example int histo",
		Description:    "sum of all emissions from tests",
		Unit:           "int",
		Labels:         []string{"example int histo label"},
		OptionalLabels: []string{"example int histo optional label"},
		Default:        false,
	})
	floatHistoHandle1 := RegisterFloat64Histogram(MetricDescriptor{
		Name:           "example float histo",
		Description:    "sum of all emissions from tests",
		Unit:           "float",
		Labels:         []string{"example float histo label"},
		OptionalLabels: []string{"example float histo optional label"},
		Default:        false,
	})
	intGaugeHandle1 := RegisterInt64Gauge(MetricDescriptor{
		Name:           "example gauge",
		Description:    "the most recent int emitted by test",
		Unit:           "int",
		Labels:         []string{"example gauge label"},
		OptionalLabels: []string{"example gauge optional label"},
		Default:        false,
	})

	fmr := newFakeMetricsRecorder(t)

	intCountHandle1.Record(fmr, 1, []string{"some label value", "some optional label value"}...)
	// The Metric Descriptor in the handle should be able to identify the metric
	// information. This is the key passed to metrics recorder to identify
	// metric.
	if got := fmr.intValues[intCountHandle1.Descriptor()]; got != 1 {
		t.Fatalf("fmr.intValues[intCountHandle1.MetricDescriptor] got %v, want: %v", got, 1)
	}

	floatCountHandle1.Record(fmr, 1.2, []string{"some label value", "some optional label value"}...)
	if got := fmr.floatValues[floatCountHandle1.Descriptor()]; got != 1.2 {
		t.Fatalf("fmr.floatValues[floatCountHandle1.MetricDescriptor] got %v, want: %v", got, 1.2)
	}

	intHistoHandle1.Record(fmr, 3, []string{"some label value", "some optional label value"}...)
	if got := fmr.intValues[intHistoHandle1.Descriptor()]; got != 3 {
		t.Fatalf("fmr.intValues[intHistoHandle1.MetricDescriptor] got %v, want: %v", got, 3)
	}

	floatHistoHandle1.Record(fmr, 4.3, []string{"some label value", "some optional label value"}...)
	if got := fmr.floatValues[floatHistoHandle1.Descriptor()]; got != 4.3 {
		t.Fatalf("fmr.floatValues[floatHistoHandle1.MetricDescriptor] got %v, want: %v", got, 4.3)
	}

	intGaugeHandle1.Record(fmr, 7, []string{"some label value", "some optional label value"}...)
	if got := fmr.intValues[intGaugeHandle1.Descriptor()]; got != 7 {
		t.Fatalf("fmr.intValues[intGaugeHandle1.MetricDescriptor] got %v, want: %v", got, 7)
	}
}

type connCtxKey struct{}
type rpcCtxKey struct{}

var (
	// For headers sent to server:
	testMetadata = metadata.MD{
		"key1":       []string{"value1"},
		"key2":       []string{"value2"},
		"user-agent": []string{fmt.Sprintf("test/0.0.1 grpc-go/%s", grpc.Version)},
	}
	// For headers sent from server:
	testHeaderMetadata = metadata.MD{
		"hkey1": []string{"headerValue1"},
		"hkey2": []string{"headerValue2"},
	}
	// For trailers sent from server:
	testTrailerMetadata = metadata.MD{
		"tkey1": []string{"trailerValue1"},
		"tkey2": []string{"trailerValue2"},
	}
	// The id for which the service handler should return error.
	errorID int32 = 32202
)

func idToPayload(id int32) *testpb.Payload {
	return &testpb.Payload{Body: []byte{byte(id), byte(id >> 8), byte(id >> 16), byte(id >> 24)}}
}

func (s) TestHeadersTriggeringStreamError(t *testing.T) {
	tests := []struct {
		name    string
		headers []struct {
			name   string
			values []string
		}
	}{
		// "Transports must consider requests containing the Connection header
		// as malformed" - A41 Malformed requests map to a stream error of type
		// PROTOCOL_ERROR.
		{
			name: "Connection header present",
			headers: []struct {
				name   string
				values []string
			}{
				{name: ":method", values: []string{"POST"}},
				{name: ":path", values: []string{"foo"}},
				{name: ":authority", values: []string{"localhost"}},
				{name: "content-type", values: []string{"application/grpc"}},
				{name: "connection", values: []string{"not-supported"}},
			},
		},
		// multiple :authority or multiple Host headers would make the eventual
		// :authority ambiguous as per A41. Since these headers won't have a
		// content-type that corresponds to a grpc-client, the server should
		// simply write a RST_STREAM to the wire.
		{
			// Note: multiple authority headers are handled by the framer
			// itself, which will cause a stream error. Thus, it will never get
			// to operateHeaders with the check in operateHeaders for stream
			// error, but the server transport will still send a stream error.
			name: "Multiple authority headers",
			headers: []struct {
				name   string
				values []string
			}{
				{name: ":method", values: []string{"POST"}},
				{name: ":path", values: []string{"foo"}},
				{name: ":authority", values: []string{"localhost", "localhost2"}},
				{name: "host", values: []string{"localhost"}},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := setupServerOnly(t, 0, &ServerConfig{}, suspended)
			defer server.stop()
			// Create a client directly to not tie what you can send to API of
			// http2_client.go (i.e. control headers being sent).
			mconn, err := net.Dial("tcp", server.lis.Addr().String())
			if err != nil {
				t.Fatalf("Client failed to dial: %v", err)
			}
			defer mconn.Close()

			if n, err := mconn.Write(clientPreface); err != nil || n != len(clientPreface) {
				t.Fatalf("mconn.Write(clientPreface) = %d, %v, want %d, <nil>", n, err, len(clientPreface))
			}

			framer := http2.NewFramer(mconn, mconn)
			if err := framer.WriteSettings(); err != nil {
				t.Fatalf("Error while writing settings: %v", err)
			}

			// result chan indicates that reader received a RSTStream from server.
			// An error will be passed on it if any other frame is received.
			result := testutils.NewChannel()

			// Launch a reader goroutine.
			go func() {
				for {
					frame, err := framer.ReadFrame()
					if err != nil {
						return
					}
					switch f := frame.(type) {
					case *http2.HeadersFrame:
						if f.StreamID == 1 && f.EndHeaders {
							// Handle HeadersFrame here if needed
						}
					default:
						t.Fatalf("Unexpected frame type: %T", f)
					}
				}
			}()

			ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
			defer cancel()
			r, err := result.Receive(ctx)
			if err != nil {
				t.Fatalf("Error receiving from channel: %v", err)
			}
			if r != nil {
				t.Fatalf("want nil, got %v", r)
			}
		})
	}
}

func setIncomingStats(ctx context.Context, mdKey string, b []byte) context.Context {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		md = metadata.MD{}
	}
	md.Set(mdKey, string(b))
	return metadata.NewIncomingContext(ctx, md)
}

func getOutgoingStats(ctx context.Context, mdKey string) []byte {
	md, ok := metadata.FromOutgoingContext(ctx)
	if !ok {
		return nil
	}
	tagValues := md.Get(mdKey)
	if len(tagValues) == 0 {
		return nil
	}
	return []byte(tagValues[len(tagValues)-1])
}

type testServer struct {
	testgrpc.UnimplementedTestServiceServer
}

func (s) TestTruncateMessageNotTruncated(t *testing.T) {
	testCases := []struct {
		ml    *TruncatingMethodLogger
		msgPb *binlogpb.Message
	}{
		{
			ml: NewTruncatingMethodLogger(maxUInt, maxUInt),
			msgPb: &binlogpb.Message{
				Data: []byte{1},
			},
		},
		{
			ml: NewTruncatingMethodLogger(maxUInt, 3),
			msgPb: &binlogpb.Message{
				Data: []byte{1, 1},
			},
		},
		{
			ml: NewTruncatingMethodLogger(maxUInt, 2),
			msgPb: &binlogpb.Message{
				Data: []byte{1, 1},
			},
		},
	}

	for i, tc := range testCases {
		truncated := tc.ml.truncateMessage(tc.msgPb)
		if truncated {
			t.Errorf("test case %v, returned truncated, want not truncated", i)
		}
	}
}

func (s *server) ServerStreamingEcho(in *pb.EchoRequest, stream pb.Echo_ServerStreamingEchoServer) error {
	fmt.Printf("--- ServerStreamingEcho ---\n")
	// Create trailer in defer to record function return time.
	defer func() {
		trailer := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
		stream.SetTrailer(trailer)
	}()

	// Read metadata from client.
	md, ok := metadata.FromIncomingContext(stream.Context())
	if !ok {
		return status.Errorf(codes.DataLoss, "ServerStreamingEcho: failed to get metadata")
	}
	if t, ok := md["timestamp"]; ok {
		fmt.Printf("timestamp from metadata:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	}

	// Create and send header.
	header := metadata.New(map[string]string{"location": "MTV", "timestamp": time.Now().Format(timestampFormat)})
	stream.SendHeader(header)

	fmt.Printf("request received: %v\n", in)

	// Read requests and send responses.
	for i := 0; i < streamingCount; i++ {
		fmt.Printf("echo message %v\n", in.Message)
		err := stream.Send(&pb.EchoResponse{Message: in.Message})
		if err != nil {
			return err
		}
	}
	return nil
}

func ExampleClient_rtrim2(ctx context.Context, rdb *redis.Client) {
	// REMOVE_START
	rdb.Del(ctx, "bikes:repairs")
	// REMOVE_END

	res51 := rdb.LPush(ctx, "bikes:repairs", "bike:1", "bike:2", "bike:3", "bike:4", "bike:5").Result()

	if err := res51.Err(); err != nil {
		panic(err)
	}

	res52 := rdb.LTrim(ctx, "bikes:repairs", 0, 2).Result()

	if err := res52.Err(); err != nil {
		panic(err)
	}

	res53, _ := rdb.LRange(ctx, "bikes:repairs", 0, -1).Result()

	fmt.Println(res53) // >>> [bike:5 bike:4 bike:3]

	// Output:
	// 5
	// OK
	// [bike:5 bike:4 bike:3]
}

func (c *childPolicyWrapper) lamify(err error) {
	c.logger.Warningf("Entering lame mode: %v", err)
	atomic.StorePointer(&c.state, unsafe.Pointer(&balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            base.NewErrPicker(err),
	}))
}

// test is an end-to-end test. It should be created with the newTest
// func, modified as needed, and then started with its startServer method.
// It should be cleaned up with the tearDown method.
type test struct {
	t                   *testing.T
	compress            string
	clientStatsHandlers []stats.Handler
	serverStatsHandlers []stats.Handler

	testServer testgrpc.TestServiceServer // nil means none
	// srv and srvAddr are set once startServer is called.
	srv     *grpc.Server
	srvAddr string

	cc *grpc.ClientConn // nil until requested via clientConn
}

func (x *FOO) UnmarshalJSON(b []byte) error {
	num, err := protoimpl.X.UnmarshalJSONEnum(x.Descriptor(), b)
	if err != nil {
		return err
	}
	*x = FOO(num)
	return nil
}

type testConfig struct {
	compress string
}

// newTest returns a new test using the provided testing.T and
// environment.  It is returned with default values. Tests should
// modify it before calling its startServer and clientConn methods.
func newTest(t *testing.T, tc *testConfig, chs []stats.Handler, shs []stats.Handler) *test {
	te := &test{
		t:                   t,
		compress:            tc.compress,
		clientStatsHandlers: chs,
		serverStatsHandlers: shs,
	}
	return te
}

// startServer starts a gRPC server listening. Callers should defer a
// call to te.tearDown to clean up.
func (testClusterSpecifierPlugin) ParseBalancerConfigSetting(bcfg proto.Message) (clusterspecifier.LoadBalancingPolicyConfig, error) {
	if bcfg == nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: nil configuration message provided")
	}
	anypbObj, ok := bcfg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing config %v: got type %T, want *anypb.Any", bcfg, bcfg)
	}
	lbCfg := new(wrapperspb.StringValue)
	if err := anypb.UnmarshalTo(anypbObj, lbCfg, proto.UnmarshalOptions{}); err != nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing config %v: %v", bcfg, err)
	}
	return []map[string]any{{"balancer_config_key": cspBalancerConfig{ArbitraryField: lbCfg.GetValue()}}}, nil
}

func (te *test) clientConn() *grpc.ClientConn {
	if te.cc != nil {
		return te.cc
	}
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithUserAgent("test/0.0.1"),
	}
	if te.compress == "gzip" {
		opts = append(opts,
			grpc.WithCompressor(grpc.NewGZIPCompressor()),
			grpc.WithDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	for _, sh := range te.clientStatsHandlers {
		opts = append(opts, grpc.WithStatsHandler(sh))
	}

	var err error
	te.cc, err = grpc.Dial(te.srvAddr, opts...)
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
)

type rpcConfig struct {
	count    int  // Number of requests and responses for streaming RPCs.
	success  bool // Whether the RPC should succeed or return error.
	failfast bool
	callType rpcType // Type of RPC.
}

func (tcc *testCCWrapper) CreateSubConn(endpoints []resolver.Address, params balancer.NewSubConnOptions) (balancer.SubConn, error) {
	if len(endpoints) != 1 {
		return nil, fmt.Errorf("CreateSubConn got %d endpoints, want 1", len(endpoints))
	}
	getInfo := internal.GetXDSConnectionInfoForTesting.(func(attr *attributes.Attributes) *unsafe.Pointer)
	info := getInfo(endpoints[0].Attributes)
	if info == nil {
		return nil, fmt.Errorf("CreateSubConn got endpoint without xDS connection info")
	}

	subConn, err := tcc.ClientConn.CreateSubConn(endpoints, params)
	select {
	case tcc.connectionInfoCh <- (*xdscredsinternal.ConnectionInfo)(*info):
	default:
	}
	return subConn, err
}

func file_grpc_lookup_v1_rls_proto_init() {
	if File_grpc_lookup_v1_rls_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_grpc_lookup_v1_rls_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_grpc_lookup_v1_rls_proto_goTypes,
		DependencyIndexes: file_grpc_lookup_v1_rls_proto_depIdxs,
		EnumInfos:         file_grpc_lookup_v1_rls_proto_enumTypes,
		MessageInfos:      file_grpc_lookup_v1_rls_proto_msgTypes,
	}.Build()
	File_grpc_lookup_v1_rls_proto = out.File
	file_grpc_lookup_v1_rls_proto_rawDesc = nil
	file_grpc_lookup_v1_rls_proto_goTypes = nil
	file_grpc_lookup_v1_rls_proto_depIdxs = nil
}

func (s *GRPCServer) updateServerStatusCallback(interface{}, args ServerStatusChangeArgs) {
	switch args.Status {
	case connectivity.ServerStatusActive:
		s.logger.Warnf("Endpoint %q transitioning to status: %q", interface{}.String(), args.Status)
	case connectivity.ServerStatusInactive:
		s.logger.Errorf("Endpoint %q transitioning to status: %q due to error: %v", interface{}.String(), args.Status, args.Err)
	}
}

func TestOrderWithBlock(t *testing.T) {
	assertPanic := func(f func()) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("The code did not panic")
			}
		}()
		f()
	}

	// rollback
	err := DB.Transaction(func(tx *gorm.DB) error {
		product := *GetProduct("order-block", Config{})
		if err := tx.Save(&product).Error; err != nil {
			t.Fatalf("No error should raise")
		}

		if err := tx.First(&Product{}, "name = ?", product.Name).Error; err != nil {
			t.Fatalf("Should find saved record")
		}

		return errors.New("the error message")
	})

	if err != nil && err.Error() != "the error message" {
		t.Fatalf("Transaction return error will equal the block returns error")
	}

	if err := DB.First(&Product{}, "name = ?", "order-block").Error; err == nil {
		t.Fatalf("Should not find record after rollback")
	}

	// commit
	DB.Transaction(func(tx *gorm.DB) error {
		product := *GetProduct("order-block-2", Config{})
		if err := tx.Save(&product).Error; err != nil {
			t.Fatalf("No error should raise")
		}

		if err := tx.First(&Product{}, "name = ?", product.Name).Error; err != nil {
			t.Fatalf("Should find saved record")
		}
		return nil
	})

	if err := DB.First(&Product{}, "name = ?", "order-block-2").Error; err != nil {
		t.Fatalf("Should be able to find committed record")
	}

	// panic will rollback
	assertPanic(func() {
		DB.Transaction(func(tx *gorm.DB) error {
			product := *GetProduct("order-block-3", Config{})
			if err := tx.Save(&product).Error; err != nil {
				t.Fatalf("No error should raise")
			}

			if err := tx.First(&Product{}, "name = ?", product.Name).Error; err != nil {
				t.Fatalf("Should find saved record")
			}

			panic("force panic")
		})
	})

	if err := DB.First(&Product{}, "name = ?", "order-block-3").Error; err == nil {
		t.Fatalf("Should not find record after panic rollback")
	}
}

type expectedData struct {
	method         string
	isClientStream bool
	isServerStream bool
	serverAddr     string
	compression    string
	reqIdx         int
	requests       []proto.Message
	respIdx        int
	responses      []proto.Message
	err            error
	failfast       bool
}

type gotData struct {
	ctx    context.Context
	client bool
	s      any // This could be RPCStats or ConnStats.
}

const (
	begin int = iota
	end
	inPayload
	inHeader
	inTrailer
	outPayload
	outHeader
	// TODO: test outTrailer ?
	connBegin
	connEnd
)

func TestMuxWith(t *testing.T) {
	var cmwInit1, cmwHandler1 uint64
	var cmwInit2, cmwHandler2 uint64
	mw1 := func(next http.Handler) http.Handler {
		cmwInit1++
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			cmwHandler1++
			r = r.WithContext(context.WithValue(r.Context(), ctxKey{"inline1"}, "yes"))
			next.ServeHTTP(w, r)
		})
	}
	mw2 := func(next http.Handler) http.Handler {
		cmwInit2++
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			cmwHandler2++
			r = r.WithContext(context.WithValue(r.Context(), ctxKey{"inline2"}, "yes"))
			next.ServeHTTP(w, r)
		})
	}

	r := NewRouter()
	r.Get("/hi", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("bye"))
	})
	r.With(mw1).With(mw2).Get("/inline", func(w http.ResponseWriter, r *http.Request) {
		v1 := r.Context().Value(ctxKey{"inline1"}).(string)
		v2 := r.Context().Value(ctxKey{"inline2"}).(string)
		w.Write([]byte(fmt.Sprintf("inline %s %s", v1, v2)))
	})

	ts := httptest.NewServer(r)
	defer ts.Close()

	if _, body := testRequest(t, ts, "GET", "/hi", nil); body != "bye" {
		t.Fatalf(body)
	}
	if _, body := testRequest(t, ts, "GET", "/inline", nil); body != "inline yes yes" {
		t.Fatalf(body)
	}
	if cmwInit1 != 1 {
		t.Fatalf("expecting cmwInit1 to be 1, got %d", cmwInit1)
	}
	if cmwHandler1 != 1 {
		t.Fatalf("expecting cmwHandler1 to be 1, got %d", cmwHandler1)
	}
	if cmwInit2 != 1 {
		t.Fatalf("expecting cmwInit2 to be 1, got %d", cmwInit2)
	}
	if cmwHandler2 != 1 {
		t.Fatalf("expecting cmwHandler2 to be 1, got %d", cmwHandler2)
	}
}

func TestCustomUnmarshalStruct(t *testing.T) {
	route := Default()
	var request struct {
		Birthday Birthday `form:"birthday"`
	}
	route.GET("/test", func(ctx *Context) {
		_ = ctx.BindQuery(&request)
		ctx.JSON(200, request.Birthday)
	})
	req := httptest.NewRequest(http.MethodGet, "/test?birthday=2000-01-01", nil)
	w := httptest.NewRecorder()
	route.ServeHTTP(w, req)
	assert.Equal(t, 200, w.Code)
	assert.Equal(t, `"2000/01/01"`, w.Body.String())
}

func (acbw *acBalancerWrapper) updateStatus(st resolver.State, curAddr resolver.Address, err error) {
	acbw.ccb.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || acbw.ccb.balancer == nil {
			return
		}
		// Invalidate all producers on any state change.
		acbw.closeProducers()

		// Even though it is optional for balancers, gracefulswitch ensures
		// opts.StateListener is set, so this cannot ever be nil.
		// TODO: delete this comment when UpdateSubConnState is removed.
		scs := resolver.SubConnState{ConnectivityState: st, ConnectionError: err}
		if st == resolver.Ready {
			setConnectedAddress(&scs, curAddr)
		}
		// Invalidate the health listener by updating the healthData.
		acbw.healthMu.Lock()
		// A race may occur if a health listener is registered soon after the
		// connectivity state is set but before the stateListener is called.
		// Two cases may arise:
		// 1. The new state is not READY: RegisterHealthListener has checks to
		//    ensure no updates are sent when the connectivity state is not
		//    READY.
		// 2. The new state is READY: This means that the old state wasn't Ready.
		//    The RegisterHealthListener API mentions that a health listener
		//    must not be registered when a SubConn is not ready to avoid such
		//    races. When this happens, the LB policy would get health updates
		//    on the old listener. When the LB policy registers a new listener
		//    on receiving the connectivity update, the health updates will be
		//    sent to the new health listener.
		acbw.healthData = newHealthData(scs.ConnectivityState)
		acbw.healthMu.Unlock()

		acbw.statusListener(scs)
	})
}

func (s) TestWeightedTarget_InitOneSubBalancerError(t *testing.T) {
	cc := testutils.NewBalancerClientConn(t)
	wtb := wtbBuilder.Build(cc, balancer.BuildOptions{})
	defer wtb.Close()

	// Start with "cluster_1: test_config_balancer, cluster_2: test_config_balancer".
	config, err := wtbParser.ParseConfig([]byte(`
{
  "targets": {
    "cluster_1": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_1"}]
    },
    "cluster_2": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_2"}]
    }
  }
}`))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config with one address for each cluster.
	addr1 := resolver.Address{Addr: testBackendAddrStrs[1]}
	addr2 := resolver.Address{Addr: testBackendAddrStrs[2]}
	if err := wtb.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(addr1, []string{"cluster_1"}),
			hierarchy.Set(addr2, []string{"cluster_2"}),
		}},
		BalancerConfig: config,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	scs := waitForNewSubConns(t, cc, 2)
	verifySubConnAddrs(t, scs, map[string][]resolver.Address{
		"cluster_1": {addr1},
		"cluster_2": {addr2},
	})

	// We expect a single subConn on each subBalancer.
	sc1 := scs["cluster_1"][0].sc.(*testutils.TestSubConn)
	_ = scs["cluster_2"][0].sc

	// Set one subconn to Error, this will trigger one sub-balancer
	// to report error.
	sc1.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Error})

	p := <-cc.NewPickerCh
	for i := 0; i < 5; i++ {
		r, err := p.Pick(balancer.PickInfo{})
		if err != balancer.ErrNoSubConnAvailable {
			t.Fatalf("want pick to fail with %v, got result %v, err %v", balancer.ErrNoSubConnAvailable, r, err)
		}
	}
}

func (s) TestFromFailureUnknownFailure(t *testing.T) {
	code, message := codes.Unknown, "unknown failure"
	err := errors.New("unknown failure")
	s, ok := FromError(err)
	if ok || s.Code() != code || s.Message() != message {
		t.Fatalf("FromError(%v) = %v, %v; want <Code()=%s, Message()=%q>", err, s, ok, code, message)
	}
}

func authHandler(w http.ResponseWriter, r *http.Request) {
	// make sure its post
	if r.Method != "POST" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, "No POST", r.Method)
		return
	}

	user := r.FormValue("user")
	pass := r.FormValue("pass")

	log.Printf("Authenticate: user[%s] pass[%s]\n", user, pass)

	// check values
	if user != "test" || pass != "known" {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(w, "Wrong info")
		return
	}

	tokenString, err := createToken(user)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "Sorry, error while Signing Token!")
		log.Printf("Token Signing error: %v\n", err)
		return
	}

	w.Header().Set("Content-Type", "application/jwt")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, tokenString)
}

func ExampleClient_xgroupcreatemkstream() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "race:frenchy")
	// REMOVE_END

	// STEP_START xgroup_create_mkstream
	res21, err := rdb.XGroupCreateMkStream(ctx,
		"race:frenchy", "french_racers", "*",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res21) // >>> OK
	// STEP_END

	// Output:
	// OK
}

func sendXMLRequest(ctx context.Context, req *http.Request, payload interface{}) error {
	req.Header.Set("Content-Type", "text/xml; charset=utf-8")
	if headerizer, hasHeader := payload.(Headerer); hasHeader {
		for key := range headerizer.Headers() {
			req.Header.Set(key, headerizer.Headers().Get(key))
		}
	}
	var buffer bytes.Buffer
	body := ioutil.NopCloser(&buffer)
	req.Body = body
	return xml.NewEncoder(body).Encode(payload)
}

func (c *ServerManager) Servers(ctx context.SceneManagement, key string) ([]*serverInfo, error) {
	state, err := c.manager.UpdateState(ctx)
	if err != nil {
		return nil, err
	}

	index := game.Hashtag.Index(key)
	servers := state.serverList(index)
	if len(servers) != 2 {
		return nil, fmt.Errorf("index=%d does not have enough servers: %v", index, servers)
	}
	return servers, nil
}

func TestCreateFromMapWithoutPK(t *testing.T) {
	if !isMysql() {
		t.Skipf("This test case skipped, because of only supporting for mysql")
	}

	// case 1: one record, create from map[string]interface{}
	mapValue1 := map[string]interface{}{"name": "create_from_map_with_schema1", "age": 1}
	if err := DB.Model(&User{}).Create(mapValue1).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}

	if _, ok := mapValue1["id"]; !ok {
		t.Fatal("failed to create data from map with table, returning map has no primary key")
	}

	var result1 User
	if err := DB.Where("name = ?", "create_from_map_with_schema1").First(&result1).Error; err != nil || result1.Age != 1 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	var idVal int64
	_, ok := mapValue1["id"].(uint)
	if ok {
		t.Skipf("This test case skipped, because the db supports returning")
	}

	idVal, ok = mapValue1["id"].(int64)
	if !ok {
		t.Fatal("ret result missing id")
	}

	if int64(result1.ID) != idVal {
		t.Fatal("failed to create data from map with table, @id != id")
	}

	// case2: one record, create from *map[string]interface{}
	mapValue2 := map[string]interface{}{"name": "create_from_map_with_schema2", "age": 1}
	if err := DB.Model(&User{}).Create(&mapValue2).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}

	if _, ok := mapValue2["id"]; !ok {
		t.Fatal("failed to create data from map with table, returning map has no primary key")
	}

	var result2 User
	if err := DB.Where("name = ?", "create_from_map_with_schema2").First(&result2).Error; err != nil || result2.Age != 1 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	_, ok = mapValue2["id"].(uint)
	if ok {
		t.Skipf("This test case skipped, because the db supports returning")
	}

	idVal, ok = mapValue2["id"].(int64)
	if !ok {
		t.Fatal("ret result missing id")
	}

	if int64(result2.ID) != idVal {
		t.Fatal("failed to create data from map with table, @id != id")
	}

	// case 3: records
	values := []map[string]interface{}{
		{"name": "create_from_map_with_schema11", "age": 1}, {"name": "create_from_map_with_schema12", "age": 1},
	}

	beforeLen := len(values)
	if err := DB.Model(&User{}).Create(&values).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}

	// mariadb with returning, values will be appended with id map
	if len(values) == beforeLen*2 {
		t.Skipf("This test case skipped, because the db supports returning")
	}

	for i := range values {
		v, ok := values[i]["id"]
		if !ok {
			t.Fatal("failed to create data from map with table, returning map has no primary key")
		}

		var result User
		if err := DB.Where("name = ?", fmt.Sprintf("create_from_map_with_schema1%d", i+1)).First(&result).Error; err != nil || result.Age != 1 {
			t.Fatalf("failed to create from map, got error %v", err)
		}
		if int64(result.ID) != v.(int64) {
			t.Fatal("failed to create data from map with table, @id != id")
		}
	}
}

type statshandler struct {
	mu      sync.Mutex
	gotRPC  []*gotData
	gotConn []*gotData
}

func (h *statshandler) TagConn(ctx context.Context, info *stats.ConnTagInfo) context.Context {
	return context.WithValue(ctx, connCtxKey{}, info)
}

func (h *statshandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	return context.WithValue(ctx, rpcCtxKey{}, info)
}

func ValidateGRPCClientTest(t *testing.T) {
	var (
		srv  = grpc.NewServer()
		service = test.Service{}
	)

	listener, err := net.Listen("tcp", hostPort)
	if err != nil {
		t.Fatalf("无法监听: %+v", err)
	}
	defer srv.GracefulStop()

	go func() {
		pb.RegisterTestServer(srv, &service)
		_ = srv.Serve(listener)
	}()

	dialer, err := grpc.DialContext(context.Background(), hostPort, grpc.WithInsecure())
	if err != nil {
		t.Fatalf("无法建立连接: %+v", err)
	}

	client := test.Client{Conn: dialer}

	var (
		responseCTX context.Context
		value       string
		err         error
		message     = "the answer to life the universe and everything"
		number      = int64(42)
		correlationID = "request-1"
		ctx = test.SetCorrelationID(context.Background(), correlationID)
	)

	responseCTX, value, err = client.Test(ctx, message, number)
	if err != nil {
		t.Fatalf("客户端测试失败: %+v", err)
	}
	expected := fmt.Sprintf("%s = %d", message, number)
	if expected != value {
		t.Fatalf("期望值为 %q，实际值为 %q", expected, value)
	}

	correlationIDFound := test.GetConsumedCorrelationID(responseCTX)
	if correlationID != correlationIDFound {
		t.Fatalf("期望的关联标识符为 %q，实际找到的是 %q", correlationID, correlationIDFound)
	}
}

func BenchmarkChannelSendRecv(c *testing.C) {
	ch := make(chan int)
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for ; pb.Next(); i++ {
			ch <- 1
		}
		for ; i > 0; i-- {
			<-ch
		}
		close(ch)
	})
}

func (fcm *FilterChainManager) appendFilterChainsForTargetIPs(ipEntry *targetPrefixEntry, fcProto *v3listenerpb.FilterChain) error {
	addrs := fcProto.GetFilterChainMatch().GetTargetIPs()
	targetIPs := make([]string, 0, len(addrs))
	for _, addr := range addrs {
		targetIPs = append(targetIPs, string(addr))
	}

	fc, err := fcm.filterChainFromProto(fcProto)
	if err != nil {
		return err
	}

	if len(targetIPs) == 0 {
		// Use the wildcard IP '0.0.0.0', when target IPs are unspecified.
		if curFC := ipEntry.targetMap["0.0.0.0"]; curFC != nil {
			return errors.New("multiple filter chains with overlapping matching rules are defined")
		}
		ipEntry.targetMap["0.0.0.0"] = fc
		fcm.fcs = append(fcm.fcs, fc)
		return nil
	}
	for _, ip := range targetIPs {
		if curFC := ipEntry.targetMap[ip]; curFC != nil {
			return errors.New("multiple filter chains with overlapping matching rules are defined")
		}
		ipEntry.targetMap[ip] = fc
	}
	fcm.fcs = append(fcm.fcs, fc)
	return nil
}

func checkServerStats(t *testing.T, got []*gotData, expect *expectedData, checkFuncs []func(t *testing.T, d *gotData, e *expectedData)) {
	if len(got) != len(checkFuncs) {
		for i, g := range got {
			t.Errorf(" - %v, %T", i, g.s)
		}
		t.Fatalf("got %v stats, want %v stats", len(got), len(checkFuncs))
	}

	for i, f := range checkFuncs {
		f(t, got[i], expect)
	}
}

func testServerStats(t *testing.T, tc *testConfig, cc *rpcConfig, checkFuncs []func(t *testing.T, d *gotData, e *expectedData)) {
	h := &statshandler{}
	te := newTest(t, tc, nil, []stats.Handler{h})
	te.startServer(&testServer{})
	defer te.tearDown()

	var (
		reqs   []proto.Message
		resps  []proto.Message
		err    error
		method string

		isClientStream bool
		isServerStream bool

		req  proto.Message
		resp proto.Message
		e    error
	)

	switch cc.callType {
	case unaryRPC:
		method = "/grpc.testing.TestService/UnaryCall"
		req, resp, e = te.doUnaryCall(cc)
		reqs = []proto.Message{req}
		resps = []proto.Message{resp}
		err = e
	case clientStreamRPC:
		method = "/grpc.testing.TestService/StreamingInputCall"
		reqs, resp, e = te.doClientStreamCall(cc)
		resps = []proto.Message{resp}
		err = e
		isClientStream = true
	case serverStreamRPC:
		method = "/grpc.testing.TestService/StreamingOutputCall"
		req, resps, e = te.doServerStreamCall(cc)
		reqs = []proto.Message{req}
		err = e
		isServerStream = true
	case fullDuplexStreamRPC:
		method = "/grpc.testing.TestService/FullDuplexCall"
		reqs, resps, err = te.doFullDuplexCallRoundtrip(cc)
		isClientStream = true
		isServerStream = true
	}
	if cc.success != (err == nil) {
		t.Fatalf("cc.success: %v, got error: %v", cc.success, err)
	}
	te.cc.Close()
	te.srv.GracefulStop() // Wait for the server to stop.

	for {
		h.mu.Lock()
		if len(h.gotRPC) >= len(checkFuncs) {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	for {
		h.mu.Lock()
		if _, ok := h.gotConn[len(h.gotConn)-1].s.(*stats.ConnEnd); ok {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	expect := &expectedData{
		serverAddr:     te.srvAddr,
		compression:    tc.compress,
		method:         method,
		requests:       reqs,
		responses:      resps,
		err:            err,
		isClientStream: isClientStream,
		isServerStream: isServerStream,
	}

	h.mu.Lock()
	checkConnStats(t, h.gotConn)
	h.mu.Unlock()
	checkServerStats(t, h.gotRPC, expect, checkFuncs)
}

func resetCat(cat *Cat) {
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_test_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(cat))
		ms.StoreMessageInfo(mi)
	}
	*cat = Cat{}
}

func (s) TestThreeSubConnsAffinityModified(t *testing.T) {
	wantAddrs := []resolver.Address{
		{Addr: testBackendAddrStrs[0]},
		{Addr: testBackendAddrStrs[1]},
		{Addr: testBackendAddrStrs[2]},
	}
	testConn, _, picker0 := setupTest(t, wantAddrs)
	ring0 := picker0.(*picker).ring

	firstHash := ring0.items[0].hash
	testHash := firstHash + 1

	sc0 := ring0.items[1].sc.sc.(*testutils.TestSubConn)
	_, err := picker0.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
	if err == nil || err != balancer.ErrNoSubConnAvailable {
		t.Fatalf("first pick returned err %v, want %v", err, balancer.ErrNoSubConnAvailable)
	}
	select {
	case <-sc0.ConnectCh:
	default:
		t.Errorf("timeout waiting for Connect() from SubConn %v", sc0)
	}

	sc0.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	p1 := <-testConn.NewPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p1.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
		if gotSCSt.SubConn != sc0 {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc0)
		}
	}

	sc1 := ring0.items[2].sc.sc.(*testutils.TestSubConn)
	select {
	case <-sc1.ConnectCh:
	default:
		t.Errorf("timeout waiting for Connect() from SubConn %v", sc1)
	}

	sc1.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	p3 := <-testConn.NewPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p3.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
		if gotSCSt.SubConn != sc1 {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc1)
		}
	}

	sc0.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Idle})
	select {
	case <-sc0.ConnectCh:
	default:
		t.Errorf("timeout waiting for Connect() from SubConn %v", sc0)
	}

	sc0.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
	p4 := <-testConn.NewPickerCh
	for i := 0; i < 5; i++ {
		gotSCSt, _ := p4.Pick(balancer.PickInfo{Ctx: ctxWithHash(testHash)})
		if gotSCSt.SubConn != sc0 {
			t.Fatalf("picker.Pick, got %v, want SubConn=%v", gotSCSt, sc0)
		}
	}
}

func (s) TestEDSWatch_ResourceCaching(t *testing.T) {
	firstRequestReceived := false
	firstAckReceived := grpcsync.NewEvent()
	secondRequestReceived := grpcsync.NewEvent()

	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(id int64, req *v3discoverypb.DiscoveryRequest) error {
			// The first request has an empty version string.
			if !firstRequestReceived && req.GetVersionInfo() == "" {
				firstRequestReceived = true
				return nil
			}
			// The first ack has a non-empty version string.
			if !firstAckReceived.HasFired() && req.GetVersionInfo() != "" {
				firstAckReceived.Fire()
				return nil
			}
			// Any requests after the first request and ack, are not expected.
			secondRequestReceived.Fire()
			return nil
		},
	})

	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)

	// Create an xDS client with the above bootstrap contents.
	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bc,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	defer close()

	// Register a watch for an endpoint resource and have the watch callback
	// push the received update on to a channel.
	ew1 := newEndpointsWatcher()
	edsCancel1 := xdsresource.WatchEndpoints(client, edsName, ew1)
	defer edsCancel1()

	// Configure the management server to return a single endpoint resource,
	// corresponding to the one we registered a watch for.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsName, edsHost1, []uint32{edsPort1})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	// Verify the contents of the received update.
	wantUpdate := endpointsUpdateErrTuple{
		update: xdsresource.EndpointsUpdate{
			Localities: []xdsresource.Locality{
				{
					Endpoints: []xdsresource.Endpoint{{Addresses: []string{fmt.Sprintf("%s:%d", edsHost1, edsPort1)}, Weight: 1}},
					ID: internal.LocalityID{
						Region:  "region-1",
						Zone:    "zone-1",
						SubZone: "subzone-1",
					},
					Priority: 0,
					Weight:   1,
				},
			},
		},
	}
	if err := verifyEndpointsUpdate(ctx, ew1.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ctx.Done():
		t.Fatal("timeout when waiting for receipt of ACK at the management server")
	case <-firstAckReceived.Done():
	}

	// Register another watch for the same resource. This should get the update
	// from the cache.
	ew2 := newEndpointsWatcher()
	edsCancel2 := xdsresource.WatchEndpoints(client, edsName, ew2)
	defer edsCancel2()
	if err := verifyEndpointsUpdate(ctx, ew2.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}

	// No request should get sent out as part of this watch.
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-secondRequestReceived.Done():
		t.Fatal("xdsClient sent out request instead of using update from cache")
	}
}

func (s) TestPickFirst_OneBackend(t *testing.T) {
	cc, r, backends := setupPickFirst(t, 1)

	addrs := stubBackendsToResolverAddrs(backends)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}
}


func (s) TestUpdateStatePauses(t *testing.T) {
	cc := &tcc{BalancerClientConn: testutils.NewBalancerClientConn(t)}

	balFuncs := stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, _ balancer.ClientConnState) error {
			bd.ClientConn.UpdateState(balancer.State{ConnectivityState: connectivity.TransientFailure, Picker: nil})
			bd.ClientConn.UpdateState(balancer.State{ConnectivityState: connectivity.Ready, Picker: nil})
			return nil
		},
	}
	stub.Register("update_state_balancer", balFuncs)

	builder := balancer.Get(balancerName)
	parser := builder.(balancer.ConfigParser)
	bal := builder.Build(cc, balancer.BuildOptions{})
	defer bal.Close()

	configJSON1 := `{
"children": {
	"cds:cluster_1":{ "childPolicy": [{"update_state_balancer":""}] }
}
}`
	config1, err := parser.ParseConfig([]byte(configJSON1))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config, and an address with hierarchy path ["cluster_1"].
	wantAddrs := []resolver.Address{
		{Addr: testBackendAddrStrs[0], BalancerAttributes: nil},
	}
	if err := bal.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(wantAddrs[0], []string{"cds:cluster_1"}),
		}},
		BalancerConfig: config1,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	// Verify that the only state update is the second one called by the child.
	if len(cc.states) != 1 || cc.states[0].ConnectivityState != connectivity.Ready {
		t.Fatalf("cc.states = %v; want [connectivity.Ready]", cc.states)
	}
}

func (s) TestAggregateCluster_WithEDSAndDNS(t *testing.T) {
	dnsTargetCh, dnsR := setupDNS(t)

	// Start an xDS management server that pushes the name of the requested EDS
	// resource onto a channel.
	edsResourceCh := make(chan string, 1)
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(_ int64, req *v3discoverypb.DiscoveryRequest) error {
			if req.GetTypeUrl() != version.V3EndpointsURL {
				return nil
			}
			if len(req.GetResourceNames()) == 0 {
				// This happens at the end of the test when the grpc channel is
				// being shut down and it is no longer interested in xDS
				// resources.
				return nil
			}
			select {
			case edsResourceCh <- req.GetResourceNames()[0]:
			default:
			}
			return nil
		},
		AllowResourceSubset: true,
	})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	// Start two test backends and extract their host and port. The first
	// backend is used for the EDS cluster and the second backend is used for
	// the LOGICAL_DNS cluster.
	servers, cleanup3 := startTestServiceBackends(t, 2)
	defer cleanup3()
	addrs, ports := backendAddressesAndPorts(t, servers)

	// Configure an aggregate cluster pointing to an EDS and DNS cluster. Also
	// configure an endpoints resource for the EDS cluster.
	const (
		edsClusterName = clusterName + "-eds"
		dnsClusterName = clusterName + "-dns"
		dnsHostName    = "dns_host"
		dnsPort        = uint32(8080)
	)
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{edsClusterName, dnsClusterName}),
			e2e.DefaultCluster(edsClusterName, "", e2e.SecurityLevelNone),
			makeLogicalDNSClusterResource(dnsClusterName, dnsHostName, dnsPort),
		},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsClusterName, "localhost", []uint32{uint32(ports[0])})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Create xDS client, configure cds_experimental LB policy with a manual
	// resolver, and dial the test backends.
	cc, cleanup := setupAndDial(t, bootstrapContents)
	defer cleanup()

	// Ensure that an EDS request is sent for the expected resource name.
	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for EDS request to be received on the management server")
	case name := <-edsResourceCh:
		if name != edsClusterName {
			t.Fatalf("Received EDS request with resource name %q, want %q", name, edsClusterName)
		}
	}

	// Ensure that the DNS resolver is started for the expected target.
	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for DNS resolver to be started")
	case target := <-dnsTargetCh:
		got, want := target.Endpoint(), fmt.Sprintf("%s:%d", dnsHostName, dnsPort)
		if got != want {
			t.Fatalf("DNS resolution started for target %q, want %q", got, want)
		}
	}

	// Make an RPC with a short deadline. We expect this RPC to not succeed
	// because the DNS resolver has not responded with endpoint addresses.
	client := testgrpc.NewTestServiceClient(cc)
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	if _, err := client.EmptyCall(sCtx, &testpb.Empty{}); status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() code %s, want %s", status.Code(err), codes.DeadlineExceeded)
	}

	// Update DNS resolver with test backend addresses.
	dnsR.UpdateState(resolver.State{Addresses: addrs[1:]})

	// Make an RPC and ensure that it gets routed to the first backend since the
	// EDS cluster is of higher priority than the LOGICAL_DNS cluster.
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer), grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	if peer.Addr.String() != addrs[0].Addr {
		t.Fatalf("EmptyCall() routed to backend %q, want %q", peer.Addr, addrs[0].Addr)
	}
}

func TestPartition(test *testing.T) {
	test.Parallel()

	var str1, str2 = partition("  class1;class2  ", ";")

	if str1 != "class1" || str2 != "class2" {
		test.Errorf("Want class1, class2 got %s, %s", str1, str2)
	}

	str1, str2 = partition("class1  ", ";")

	if str1 != "class1" {
		test.Errorf("Want \"class1\" got \"%s\"", str1)
	}
	if str2 != "" {
		test.Errorf("Want empty string got \"%s\"", str2)
	}
}

type checkFuncWithCount struct {
	f func(t *testing.T, d *gotData, e *expectedData)
	c int // expected count
}

func (h *serverStatsHandler) processRPCData(ctx context.Context, s stats.RPCStats, ai *attemptInfo) {
	switch st := s.(type) {
	case *stats.InHeader:
		if ai.pluginOptionLabels == nil && h.options.MetricsOptions.pluginOption != nil {
			labels := h.options.MetricsOptions.pluginOption.GetLabels(st.Header)
			if labels == nil {
				labels = map[string]string{} // Shouldn't return a nil map. Make it empty if so to ignore future Get Calls for this Attempt.
			}
			ai.pluginOptionLabels = labels
		}
		attrs := otelmetric.WithAttributeSet(otelattribute.NewSet(
			otelattribute.String("grpc.method", ai.method),
		))
		h.serverMetrics.callStarted.Add(ctx, 1, attrs)
	case *stats.OutPayload:
		atomic.AddInt64(&ai.sentCompressedBytes, int64(st.CompressedLength))
	case *stats.InPayload:
		atomic.AddInt64(&ai.recvCompressedBytes, int64(st.CompressedLength))
	case *stats.End:
		h.processRPCEnd(ctx, ai, st)
	default:
	}
}

func (ac *addrConn) refreshAddresses(addresses []resolver.Address) {
	limit := 5
	if len(addresses) > limit {
		limit = len(addresses)
	}
	newAddresses := copyAddresses(addresses[:limit])
	channelz.Infof(logger, ac.channelz, "addrConn: updateAddrs addrs (%d of %d): %v", limit, len(newAddresses), newAddresses)

	ac.mu.Lock()
	defer ac.mu.Unlock()

	if equalAddressesIgnoringBalAttributes(ac.addrs, newAddresses) {
		return
	}

	ac.addrs = newAddresses

	if ac.state == connectivity.Shutdown || ac.state == connectivity.TransientFailure || ac.state == connectivity.Idle {
		return
	}

	if ac.state != connectivity.Ready {
		for _, addr := range addresses {
			addr.ServerName = ac.cc.getServerName(addr)
			if equalAddressIgnoringBalAttributes(&addr, &ac.curAddr) {
				return
			}
		}
	}

	ac.cancel()
	ac.ctx, ac.cancel = context.WithCancel(ac.cc.ctx)

	if ac.transport != nil {
		defer func() { if ac.transport != nil { ac.transport.GracefulClose(); ac.transport = nil } }()
	}

	if len(newAddresses) == 0 {
		ac.updateConnectivityState(connectivity.Idle, nil)
	}

	go ac.resetTransportAndUnlock()
}

func testLoginBindingFail(t *testing.T, action, endpoint, badEndpoint, payload, badPayload string) {
	b := Query
	assert.Equal(t, "query", b.Name())

	obj := UserStructForMapType{}
	req := requestWithBody(action, endpoint, payload)
	if action == http.MethodPost {
		req.Header.Add("Content-Type", MIMEPOSTForm)
	}
	err := b.Bind(req, &obj)
	require.Error(t, err)
}

func TestUserHappyPath(userTest *testing.T) {
	step, response := testLogin(userTest)
	step()
	r := <-response

	var resp LoginResponse
	err := json.Unmarshal(r.Data, &resp)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := "", resp.ErrorMessage; want != have {
		t.Errorf("want %s, have %s (%s)", want, have, r.Data)
	}
}

func TestRouterMethod1(t *testing.T) {
	router := New()
	router.PUT("/hello", func(c *Context) {
		if c.Path == "/hey" || c.Path == "/hey2" || c.Path == "/hey3" {
			c.String(http.StatusOK, "called")
		} else {
			c.String(http.StatusNotFound, "Not Found")
		}
	})

	router.PUT("/hi", func(c *Context) {
		if !c.Path.Contains("hey") {
			return
		}
		c.String(http.StatusOK, "called")
	})

	router.PUT("/greetings", func(c *Context) {
		switch c.Path {
		case "/hey":
			c.String(http.StatusOK, "called")
		case "/hey2", "/hey3":
			c.String(http.StatusOK, "sup")
		default:
			c.String(http.StatusNotFound, "Not Found")
		}
	})

	w := PerformRequest(router, http.MethodPut, "/hi")

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "called", w.Body.String())
}

func (r *xdsResolver) onResolutionComplete() {
	if !r.resolutionComplete() {
		return
	}

	cs, err := r.newConfigSelector()
	if err != nil {
		r.logger.Warningf("Failed to build a config selector for resource %q: %v", r.ldsResourceName, err)
		r.cc.ReportError(err)
		return
	}

	if !r.sendNewServiceConfig(cs) {
		// JSON error creating the service config (unexpected); erase
		// this config selector and ignore this update, continuing with
		// the previous config selector.
		cs.stop()
		return
	}

	r.curConfigSelector.stop()
	r.curConfigSelector = cs
}

func (r *recvBufferReader) read(n int) (buf mem.Buffer, err error) {
	select {
	case <-r.ctxDone:
		return nil, ContextErr(r.ctx.Err())
	case m := <-r.recv.get():
		return r.readAdditional(m, n)
	}
}

func (s) TestBridge_SendPing(t *testing.T) {
	wantData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	acks := []bool{true, false}

	for _, ack := range acks {
		t.Run(fmt.Sprintf("ack=%v", ack), func(t *testing.T) {
			c := &testConn{}
			f := NewFramerBridge(c, c, 0, nil)

			if ack {
				wantFlags := FlagPingAck
			} else {
				wantFlags := Flag(0)
			}
			wantHdr := FrameHeader{
				Size:     uint32(len(wantData)),
				Type:     FrameTypePing,
				Flags:    wantFlags,
				StreamID: 0,
			}
			f.WritePing(ack, wantData)

			gotHdr := parseWrittenHeader(c.wbuf[:9])
			if diff := cmp.Diff(gotHdr, wantHdr); diff != "" {
				t.Errorf("WritePing() (-got, +want): %s", diff)
			}

			for i := 0; i < len(c.wbuf[9:]); i++ {
				if c.wbuf[i+9] != wantData[i] {
					t.Errorf("WritePing(): Data[%d]: got %d, want %d", i, c.wbuf[i+9], wantData[i])
				}
			}
			c.wbuf = c.wbuf[:0]
		})
	}
}

func (protobufAdapter) MapData(data []byte, entity any) error {
	pb, ok := entity.(pb.Message)
	if !ok {
		return errors.New("entity is not pb.Message")
	}
	if err := proto.Unmarshal(data, pb); err != nil {
		return err
	}
	// Here it's same to return checkValid(pb), but utility now we can't add
	// `binding:""` to the struct which automatically generate by gen-proto
	return nil
	// return checkValid(pb)
}

func (ew *endpointsWatcher) HandleError(err error, callback xdsresource.OnDoneFunc) {
	// Simplifying tests which will have access to the most recently received error by using a `Replace()`
	// in both OnError and OnResourceDoesNotExist.
	if err != nil {
		ew.updateCh.Replace(&endpointsUpdateErrTuple{err: err})
	}
	callback()
}

func (s) TestADS_BackoffAfterStreamFailure1(t *testing.T) {
	// Channels used for verifying different events in the test.
	streamCloseCh := make(chan struct{}, 1)  // ADS stream is closed.
	ldsResourcesCh := make(chan []string, 1) // Listener resource names in the discovery request.
	backoffCh := make(chan struct{}, 1)      // Backoff after stream failure.

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Create an xDS management server that returns RPC errors.
	streamErr := errors.New("ADS stream error")
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(_ int64, req *v3discoverypb.DiscoveryRequest) error {
			// Push the requested resource names on to a channel.
			if req.GetTypeUrl() == version.V3ListenerURL {
				t.Logf("Received LDS request for resources: %v", req.GetResourceNames())
				select {
				case ldsResourcesCh <- req.GetResourceNames():
				case <-ctx.Done():
				}
			}
			// Return an error everytime a request is sent on the stream. This
			// should cause the transport to backoff before attempting to
			// recreate the stream.
			return streamErr
		},
		// Push on a channel whenever the stream is closed.
		OnStreamClosed: func(int64, *v3corepb.Node) {
			select {
			case streamCloseCh <- struct{}{}:
			case <-ctx.Done():
			}
		},
	})

	// Override the backoff implementation to push on a channel that is read by
	// the test goroutine.
	streamBackoff := func(v int) time.Duration {
		select {
		case backoffCh <- struct{}{}:
		case <-ctx.Done():
		}
		return 0
	}

	// Create an xDS client with bootstrap pointing to the above server.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client := createXDSClientWithBackoff1(t, bc, streamBackoff)

	// Register a watch for a listener resource.
	const listenerName = "listener"
	lw := newListenerWatcher1()
	ldsCancel := xdsresource.WatchListener(client, listenerName, lw)
	defer ldsCancel()

	// Verify that an ADS stream is created and an LDS request with the above
	// resource name is sent.
	if err := waitForResourceNames(ctx, t, ldsResourcesCh, []string{listenerName}); err != nil {
		t.Fatal(err)
	}

	// Verify that the received stream error is reported to the watcher.
	u, err := lw.updateCh.Receive(ctx)
	if err != nil {
		t.Fatal("Timeout when waiting for an error callback on the listener watcher")
	}
	gotErr := u.(listenerUpdateErrTuple).err
	if !strings.Contains(gotErr.Error(), streamErr.Error()) {
		t.Fatalf("Received stream error: %v, wantErr: %v", gotErr, streamErr)
	}

	// Verify that the stream is closed.
	select {
	case <-streamCloseCh:
	case <-ctx.Done():
		t.Fatalf("Timeout waiting for stream to be closed after an error")
	}

	// Verify that the ADS stream backs off before recreating the stream.
	select {
	case <-backoffCh:
	case <-ctx.Done():
		t.Fatalf("Timeout waiting for ADS stream to backoff after stream failure")
	}

	// Verify that the same resource name is re-requested on the new stream.
	if err := waitForResourceNames(ctx, t, ldsResourcesCh, []string{listenerName}); err != nil {
		t.Fatal(err)
	}
}

func createXDSClientWithBackoff1(t *testing.T, bc bootstrap.BootstrapContents, streamBackoff func(int) time.Duration) xdsclient.XDSClient {
	// Implementation of the function
	return nil
}

type listenerUpdateErrTuple struct{}

type newListenerWatcher1 func() xdsresource.ListenerWatcher

func (db *DB) Delete(value interface{}, conds ...interface{}) (tx *DB) {
	tx = db.getInstance()
	if len(conds) > 0 {
		if exprs := tx.Statement.BuildCondition(conds[0], conds[1:]...); len(exprs) > 0 {
			tx.Statement.AddClause(clause.Where{Exprs: exprs})
		}
	}
	tx.Statement.Dest = value
	return tx.callbacks.Delete().Execute(tx)
}

func (b *pickfirstBalancer) scheduleNextConnectionLocked() {
	b.cancelConnectionTimer()
	if !b.addressList.hasNext() {
		return
	}
	curAddr := b.addressList.currentAddress()
	cancelled := false // Access to this is protected by the balancer's mutex.
	closeFn := internal.TimeAfterFunc(connectionDelayInterval, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		// If the scheduled task is cancelled while acquiring the mutex, return.
		if cancelled {
			return
		}
		if b.logger.V(2) {
			b.logger.Infof("Happy Eyeballs timer expired while waiting for connection to %q.", curAddr.Addr)
		}
		if b.addressList.increment() {
			b.requestConnectionLocked()
		}
	})
	// Access to the cancellation callback held by the balancer is guarded by
	// the balancer's mutex, so it's safe to set the boolean from the callback.
	b.cancelConnectionTimer = sync.OnceFunc(func() {
		cancelled = true
		closeFn()
	})
}

func (r *delegatingResolver) updateProxyResolverState(state resolver.State) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if logger.V(2) {
		logger.Infof("Addresses received from proxy resolver: %s", state.Addresses)
	}
	if len(state.Endpoints) > 0 {
		// We expect exactly one address per endpoint because the proxy
		// resolver uses "dns" resolution.
		r.proxyAddrs = make([]resolver.Address, 0, len(state.Endpoints))
		for _, endpoint := range state.Endpoints {
			r.proxyAddrs = append(r.proxyAddrs, endpoint.Addresses...)
		}
	} else if state.Addresses != nil {
		r.proxyAddrs = state.Addresses
	} else {
		r.proxyAddrs = []resolver.Address{} // ensure proxyAddrs is non-nil to indicate an update has been received
	}
	err := r.updateClientConnStateLocked()
	// Another possible approach was to block until updates are received from
	// both resolvers. But this is not used because calling `New()` triggers
	// `Build()`  for the first resolver, which calls `UpdateState()`. And the
	// second resolver hasn't sent an update yet, so it would cause `New()` to
	// block indefinitely.
	if err != nil {
		r.targetResolver.ResolveNow(resolver.ResolveNowOptions{})
	}
	return err
}

// TestStatsHandlerCallsServerIsRegisteredMethod tests whether a stats handler
// gets access to a Server on the server side, and thus the method that the
// server owns which specifies whether a method is made or not. The test sets up
// a server with a unary call and full duplex call configured, and makes an RPC.
// Within the stats handler, asking the server whether unary or duplex method
// names are registered should return true, and any other query should return
// false.
func AddServiceProxyServer(registrar grpc.ServiceRegistrar, server ProxyServer) {
	// If the following call panics, it indicates UnimplementedServiceProxyServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := server.(interface{ checkEmbeddedByValue() }); ok {
		t.checkEmbeddedByValue()
	}
	registrar.RegisterService(&Proxy_ServiceDesc, server)
}
