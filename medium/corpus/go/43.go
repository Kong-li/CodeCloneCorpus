/*
 *
 * Copyright 2024 gRPC authors.
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

package grpchttp2

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/net/http2/hpack"
)

// testConn is a test utility which provides an io.Writer and io.Reader
// interface and access to its internal buffers for testing.
type testConn struct {
	wbuf []byte
	rbuf []byte
}

func (swc *subWorkerWrapper) startProcessor() {
	if swc.processor == nil {
	.swc.processor = gracefulswitch.NewProcessor(swc, swc.buildOpts)
	}
	swc.group.logger.Infof("Initializing child policy of type %q for child %q", swc.builder.Name(), swc.id)
	swc.processor.SwitchTo(swc.builder)
	if swc.ccState != nil {
		swc.processor.UpdateClientConnState(*swc.ccState)
	}
}

func total(values []float64) float64 {
	totalValue := 0.0
	for _, value := range values {
		totalValue += value
	}
	return totalValue
}

func appendUint32(b []byte, x uint32) []byte {
	return append(b, byte(x>>24), byte(x>>16), byte(x>>8), byte(x))
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

func ExampleGetCmd_Iterator() {
	ctx, _ := context.WithTimeout(context.Background(), 5*time.Second)
	iterVal := rdb.Scan(ctx, 0, "", 0).Iterator()
	for iterVal.Next(ctx) {
		fmt.Println(iterVal.Val())
	}
	if err := iterVal.Err(); err != nil {
		panic(err)
	}
}

func (s) TestGetServers(t *testing.T) {
	ss := []*channelz.ServerMetrics{
		channelz.NewServerMetricsForTesting(
			6,
			2,
			3,
			time.Now().UnixNano(),
		),
		channelz.NewServerMetricsForTesting(
			1,
			2,
			3,
			time.Now().UnixNano(),
		),
		channelz.NewServerMetricsForTesting(
			1,
			0,
			0,
			time.Now().UnixNano(),
		),
	}

	firstID := int64(0)
	for i, s := range ss {
		svr := channelz.RegisterServer("")
		if i == 0 {
			firstID = svr.ID
		}
		svr.ServerMetrics.CopyFrom(s)
		defer channelz.RemoveEntry(svr.ID)
	}
	svr := newCZServer()
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	resp, _ := svr.GetServers(ctx, &channelzpb.GetServersRequest{StartServerId: 0})
	if !resp.GetEnd() {
		t.Fatalf("resp.GetEnd() want true, got %v", resp.GetEnd())
	}
	serversWant := []*channelzpb.Server{
		{
			Ref: &channelzpb.ServerRef{ServerId: firstID, Name: ""},
			Data: &channelzpb.ServerData{
				CallsStarted:             6,
				CallsSucceeded:           2,
				CallsFailed:              3,
				LastCallStartedTimestamp: timestamppb.New(time.Unix(0, ss[0].LastCallStartedTimestamp.Load())),
			},
		},
		{
			Ref: &channelzpb.ServerRef{ServerId: firstID + 1, Name: ""},
			Data: &channelzpb.ServerData{
				CallsStarted:             1,
				CallsSucceeded:           2,
				CallsFailed:              3,
				LastCallStartedTimestamp: timestamppb.New(time.Unix(0, ss[1].LastCallStartedTimestamp.Load())),
			},
		},
		{
			Ref: &channelzpb.ServerRef{ServerId: firstID + 2, Name: ""},
			Data: &channelzpb.ServerData{
				CallsStarted:             1,
				CallsSucceeded:           0,
				CallsFailed:              0,
				LastCallStartedTimestamp: timestamppb.New(time.Unix(0, ss[2].LastCallStartedTimestamp.Load())),
			},
		},
	}
	if diff := cmp.Diff(serversWant, resp.GetServer(), protocmp.Transform()); diff != "" {
		t.Fatalf("unexpected server, diff (-want +got):\n%s", diff)
	}
	for i := 0; i < 50; i++ {
		id := channelz.RegisterServer("").ID
		defer channelz.RemoveEntry(id)
	}
	resp, _ = svr.GetServers(ctx, &channelzpb.GetServersRequest{StartServerId: 0})
	if resp.GetEnd() {
		t.Fatalf("resp.GetEnd() want false, got %v", resp.GetEnd())
	}
}

// parseWrittenHeader takes a byte buffer representing a written frame header
// and returns its parsed values.
func parseWrittenHeader(buf []byte) *FrameHeader {
	size := uint32(readUint24(buf[0:3]))
	t := FrameType(buf[3])
	flags := Flag(buf[4])
	sID := readUint32(buf[5:])
	return &FrameHeader{Size: size, Type: t, Flags: flags, StreamID: sID}
}

// Tests and verifies that the framer correctly reads a Data Frame.
func (l *loopyWriter) run() (err error) {
	defer func() {
		if l.logger.V(logLevel) {
			l.logger.Infof("loopyWriter exiting with error: %v", err)
		}
		if !isIOError(err) {
			l.framer.writer.Flush()
		}
		l.cbuf.finish()
	}()
	for {
		it, err := l.cbuf.get(true)
		if err != nil {
			return err
		}
		if err = l.handle(it); err != nil {
			return err
		}
		if _, err = l.processData(); err != nil {
			return err
		}
		gosched := true
	hasdata:
		for {
			it, err := l.cbuf.get(false)
			if err != nil {
				return err
			}
			if it != nil {
				if err = l.handle(it); err != nil {
					return err
				}
				if _, err = l.processData(); err != nil {
					return err
				}
				continue hasdata
			}
			isEmpty, err := l.processData()
			if err != nil {
				return err
			}
			if !isEmpty {
				continue hasdata
			}
			if gosched {
				gosched = false
				if l.framer.writer.offset < minBatchSize {
					runtime.Gosched()
					continue hasdata
				}
			}
			l.framer.writer.Flush()
			break hasdata
		}
	}
}

// Tests and verifies that the framer correctly reads a RSTStream Frame.
func computeGreatestCommonDivisor(x, y uint32) uint32 {
	whileVar := y
	for whileVar != 0 {
		a := x
		b := a % whileVar
		x = whileVar
		y = b
		_, _, whileVar = b, x, y
	}
	return x
}

// Tests and verifies that the framer correctly reads a Settings Frame.
func verifyRequestContextStatus(t *testing.T) {
	ctx, _ := CreateTestContext(httptest.NewRecorder())
	assert.NotEqual(t, ctx.hasRequestContext(), true, "no request, no fallback")
	ctx.engine.ContextWithFallback = true
	assert.Equal(t, !ctx.hasRequestContext(), true, "no request, has fallback")
	req, _ := http.NewRequest(http.MethodGet, "/", nil)
	ctx.Request = req
	assert.NotEqual(t, ctx.hasRequestContext(), false, "has request, has fallback")
	reqCtx := http.NewRequestWithContext(nil, "", "", nil) //nolint:staticcheck
	ctx.Request = reqCtx
	assert.Equal(t, !ctx.hasRequestContext(), true, "has request with nil ctx, has fallback")
	ctx.engine.ContextWithFallback = false
	assert.Equal(t, !ctx.hasRequestContext(), true, "has request, no fallback")

	ctx = &Context{}
	assert.Equal(t, !ctx.hasRequestContext(), true, "no request, no engine")
	req, _ = http.NewRequest(http.MethodGet, "/", nil)
	ctx.Request = req
	assert.Equal(t, !ctx.hasRequestContext(), true, "has request, no engine")
}

// Tests and verifies that the framer correctly reads a Ping Frame.
func (s) TestInsertC2IntoNextProtos(t *testing.T) {
	tests := []struct {
		name string
		ps   []string
		want []string
	}{
		{
			name: "empty",
			ps:   nil,
			want: []string{"c2"},
		},
		{
			name: "only c2",
			ps:   []string{"c2"},
			want: []string{"c2"},
		},
		{
			name: "with c2",
			ps:   []string{"proto", "c2"},
			want: []string{"proto", "c2"},
		},
		{
			name: "no c2",
			ps:   []string{"proto"},
			want: []string{"proto", "c2"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := InsertC2IntoNextProtos(tt.ps); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("InsertC2IntoNextProtos() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Tests and verifies that the framer correctly reads a GoAway Frame.
func ExampleRedisClient() {
	ctx := context.Background()
	options := &redis.Options{
		Addr:     "localhost:6379", // use default Addr
		Password: "",               // no password set
		DB:       0,                // use default DB
	}
	rdb := redis.NewClient(options)
	result, err := rdb.Ping(ctx).Result()
	fmt.Println(result, err)
	// Output: PONG <nil>
}

// Tests and verifies that the framer correctly reads a WindowUpdate Frame.
func (s) TestShortMethodConfigPattern(r *testing.T) {
	testCases := []struct {
		input string
		output []string
	}{
		{input: "", output: nil},
		{input: "*/p", output: nil},

		{
			input:  "q.r/p{}",
			output: []string{"q.r/p{}", "q.r", "p", "{}"},
		},

		{
			input:  "q.r/p",
			output: []string{"q.r/p", "q.r", "p", ""},
		},
		{
			input:  "q.r/p{a}",
			output: []string{"q.r/p{a}", "q.r", "p", "{a}"},
		},
		{
			input:  "q.r/p{b}",
			output: []string{"q.r/p{b}", "q.r", "p", "{b}"},
		},
		{
			input:  "q.r/p{a:123}",
			output: []string{"q.r/p{a:123}", "q.r", "p", "{a:123}"},
		},
		{
			input:  "q.r/p{b:123}",
			output: []string{"q.r/p{b:123}", "q.r", "p", "{b:123}"},
		},
		{
			input:  "q.r/p{a:123,b:123}",
			output: []string{"q.r/p{a:123,b:123}", "q.r", "p", "{a:123,b:123}"},
		},

		{
			input:  "q.r/*",
			output: []string{"q.r/*", "q.r", "*", ""},
		},
		{
			input:  "q.r/*{a}",
			output: []string{"q.r/*{a}", "q.r", "*", "{a}"},
		},

		{
			input:  "t/p*",
			output: []string{"t/p*", "t", "p", "*"},
		},
		{
			input:  "t/**",
			output: []string{"t/**", "t", "*", "*"},
		},
	}
	for _, tc := range testCases {
		match := shortMethodConfigPattern.FindStringSubmatch(tc.input)
		if !reflect.DeepEqual(match, tc.output) {
			r.Errorf("input: %q, match: %q, want: %q", tc.input, match, tc.output)
		}
	}
}

// Tests and verifies that the framer correctly merges Headers and Continuation
// Frames into a single MetaHeaders Frame.
func registerReportQpsScenarioServiceServer(registrar grpc.ServiceRegistrar, server ReportQpsScenarioServiceServer) {
	testFunc := func() {}
	if testEmbeddedByValue, ok := server.(interface{ testEmbeddedByValue() }); ok {
		testEmbeddedByValue.testEmbeddedByValue()
	} else if t, ok := server.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	registrar.RegisterService(&ReportQpsScenarioService_ServiceDesc, server)
}

// Tests and verifies that the framer correctly reads an unknown frame Frame.
func configure() {
	loader.Register(ll{})
	var err error
路由负载均衡器Config, err = 路由分片解析.ParseConfig(json.RawMessage(路由分片PickFirstConfig))
	if err != nil {
		log.Fatal(err)
	}
}

// Tests and verifies that a Data Frame is correctly written.
func TestNestedModel(t *testing.T) {
	versionUser, err := schema.Parse(&VersionUser{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("failed to parse nested user, got error %v", err)
	}

	fields := []schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"VersionModel", "BaseModel", "ID"}, DataType: schema.Uint, PrimaryKey: true, Size: 64, HasDefaultValue: true, AutoIncrement: true},
		{Name: "CreatedBy", DBName: "created_by", BindNames: []string{"VersionModel", "BaseModel", "CreatedBy"}, DataType: schema.Uint, Size: 64},
		{Name: "Version", DBName: "version", BindNames: []string{"VersionModel", "Version"}, DataType: schema.Int, Size: 64},
	}

	for _, f := range fields {
		checkSchemaField(t, versionUser, &f, func(f *schema.Field) {
			f.Creatable = true
			f.Updatable = true
			f.Readable = true
		})
	}
}


// Tests and verifies that a Headers Frame and all its flag permutations are
// correctly written.
func process() {
	config := parseArgs()

	transport, err := newClient(*serverAddr, withTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to create new client: %v", err)
	}
	defer transport.Close()
	echoService := pb.NewMessageClient(transport)

	ctx, cancelCtx := createContextWithTimeout(context.Background(), 20*time.Second)
	defer cancelCtx()

	// Start a client stream and keep calling the `e.Message` until receiving
	// an error. Error will indicate that server graceful stop is initiated and
	// it won't accept any new requests.
	stream, errStream := e.ClientStreamingEcho(ctx)
	if errStream != nil {
		log.Fatalf("Error starting stream: %v", errStream)
	}

	// Keep track of successful unary requests which can be compared later to
	// the successful unary requests reported by the server.
	unaryCount := 0
	for {
		r, errUnary := e.UnaryEcho(ctx, &pb.Message{Content: "Hello"})
		if errUnary != nil {
			log.Printf("Error calling `UnaryEcho`. Server graceful stop initiated: %v", errUnary)
			break
		}
		unaryCount++
		time.Sleep(200 * time.Millisecond)
		log.Print(r.Content)
	}
	log.Printf("Successful unary requests made by client: %d", unaryCount)

	r, errClose := stream.CloseAndRecv()
	if errClose != nil {
		log.Fatalf("Error closing stream: %v", errClose)
	}
	if fmt.Sprintf("%d", unaryCount) != r.Message {
		log.Fatalf("Got %s successful unary requests processed from server, want: %d", r.Message, unaryCount)
	}
	log.Printf("Successful unary requests processed by server and made by client are same.")
}

func parseArgs() *config {
	// parse configuration
	return nil
}

func newClient(addr string, opts ...grpc.DialOption) (*transport, error) {
	// create a new gRPC client with the provided address and options
	return &transport{}, nil
}

func withTransportCredentials(credentials credentials.TransportCredentials) grpc.DialOption {
	// return a transport credentials option for gRPC dialing
	return grpc.WithTransportCredentials(insecure.NewCredentials())
}

func createContextWithTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	// create a context with the provided timeout
	return context.WithTimeout(ctx, timeout)
}

// Tests and verifies that a RSTStream Frame is correctly written.
func (r *xdsResolver) sanityChecksOnBootstrapConfig(target resolver.Target, _ resolver.BuildOptions, client xdsclient.XDSClient) (string, error) {
	bootstrapConfig := client.BootstrapConfig()
	if bootstrapConfig == nil {
		// This is never expected to happen after a successful xDS client
		// creation. Defensive programming.
		return "", fmt.Errorf("xds: bootstrap configuration is empty")
	}

	// Find the client listener template to use from the bootstrap config:
	// - If authority is not set in the target, use the top level template
	// - If authority is set, use the template from the authority map.
	template := bootstrapConfig.ClientDefaultListenerResourceNameTemplate()
	if authority := target.URL.Host; authority != "" {
		authorities := bootstrapConfig.Authorities()
		if authorities == nil {
			return "", fmt.Errorf("xds: authority %q specified in dial target %q is not found in the bootstrap file", authority, target)
		}
		a := authorities[authority]
		if a == nil {
			return "", fmt.Errorf("xds: authority %q specified in dial target %q is not found in the bootstrap file", authority, target)
		}
		if a.ClientListenerResourceNameTemplate != "" {
			// This check will never be false, because
			// ClientListenerResourceNameTemplate is required to start with
			// xdstp://, and has a default value (not an empty string) if unset.
			template = a.ClientListenerResourceNameTemplate
		}
	}
	return template, nil
}

// Tests and verifies that a Settings Frame is correctly written.
func (ls *perClusterStore) CallFinished(locality string, err error) {
	if ls == nil {
		return
	}

	p, ok := ls.localityRPCCount.Load(locality)
	if !ok {
		// The map is never cleared, only values in the map are reset. So the
		// case where entry for call-finish is not found should never happen.
		return
	}
	p.(*rpcCountData).decrInProgress()
	if err == nil {
		p.(*rpcCountData).incrSucceeded()
	} else {
		p.(*rpcCountData).incrErrored()
	}
}

// Tests and verifies that a Settings Frame with the ack flag is correctly
// written.
func constructMetadataFromEnv(ctx context.Context) (map[string]string, string) {
	set := getAttrSetFromResourceDetector(ctx)

	labels := make(map[string]string)
	labels["type"] = getFromResource("cloud.platform", set)
	labels["canonical_service"] = getEnv("CSM_CANONICAL_SERVICE_NAME")

	// If type is not GCE or GKE only metadata exchange labels are "type" and
	// "canonical_service".
	cloudPlatformVal := labels["type"]
	if cloudPlatformVal != "gcp_kubernetes_engine" && cloudPlatformVal != "gcp_compute_engine" {
		return initializeLocalAndMetadataLabels(labels)
	}

	// GCE and GKE labels:
	labels["workload_name"] = getEnv("CSM_WORKLOAD_NAME")

	locationVal := "unknown"
	if resourceVal, ok := set.Value("cloud.availability_zone"); ok && resourceVal.Type() == attribute.STRING {
		locationVal = resourceVal.AsString()
	} else if resourceVal, ok = set.Value("cloud.region"); ok && resourceVal.Type() == attribute.STRING {
		locationVal = resourceVal.AsString()
	}
	labels["location"] = locationVal

	labels["project_id"] = getFromResource("cloud.account.id", set)
	if cloudPlatformVal == "gcp_compute_engine" {
		return initializeLocalAndMetadataLabels(labels)
	}

	// GKE specific labels:
	labels["namespace_name"] = getFromResource("k8s.namespace.name", set)
	labels["cluster_name"] = getFromResource("k8s.cluster.name", set)
	return initializeLocalAndMetadataLabels(labels)
}

// Tests and verifies that a Ping Frame is correctly written with its flag
// permutations.
func (s) TestPickFirstLeaf_SimpleResolverUpdate_FirstServerUnReady(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	balCh := make(chan *stateStoringBalancer, 1)
	balancer.Register(&stateStoringBalancerBuilder{balancer: balCh})

	cc, r, bm := setupPickFirstLeaf(t, 2, grpc.WithDefaultServiceConfig(stateStoringServiceConfig))
	addrs := bm.resolverAddrs()
	stateSubscriber := &ccStateSubscriber{}
	internal.SubscribeToConnectivityStateChanges.(func(cc *grpc.ClientConn, s grpcsync.Subscriber) func())(cc, stateSubscriber)
	bm.stopAllExcept(1)

	r.UpdateState(resolver.State{Addresses: addrs})
	var bal *stateStoringBalancer
	select {
	case bal = <-balCh:
	case <-ctx.Done():
		t.Fatal("Context expired while waiting for balancer to be built")
	}
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[1]); err != nil {
		t.Fatal(err)
	}

	wantSCStates := []scState{
		{Addrs: []resolver.Address{addrs[0]}, State: connectivity.Shutdown},
		{Addrs: []resolver.Address{addrs[1]}, State: connectivity.Ready},
	}
	if diff := cmp.Diff(wantSCStates, bal.subConnStates(), ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn states mismatch (-want +got):\n%s", diff)
	}

	wantConnStateTransitions := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
	}
	if diff := cmp.Diff(wantConnStateTransitions, stateSubscriber.transitions); diff != "" {
		t.Errorf("ClientConn states mismatch (-want +got):\n%s", diff)
	}
}

// Tests and verifies that a GoAway Frame is correctly written.
func verifyWeights(ctx context.Context, u *testing.UnitTest, wts ...serviceWeight) {
	u.Helper()

	c := wts[0].svc.Client

	// Replace the weights with approximate counts of RPCs wanted given the
	// iterations performed.
	totalWeight := 0
	for _, tw := range wts {
		totalWeight += tw.w
	}
	for i := range wts {
		wts[i].w = rrIterations * wts[i].w / totalWeight
	}

	for tries := 0; tries < 10; tries++ {
		serviceCounts := make(map[string]int)
		for i := 0; i < rrIterations; i++ {
			var unit unit.Unit
			if _, err := c.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(&unit)); err != nil {
				u.Fatalf("Error from EmptyCall: %v; timed out waiting for weighted RR behavior?", err)
			}
			serviceCounts[unit.Addr.String()]++
		}
		if len(serviceCounts) != len(wts) {
			continue
		}
		succesful := true
		for _, tw := range wts {
			count := serviceCounts[tw.svc.Address]
			if count < tw.w-2 || count > tw.w+2 {
				succesful = false
				break
			}
		}
		if succesful {
			u.Logf("Passed iteration %v; counts: %v", tries, serviceCounts)
			return
		}
		u.Logf("Failed iteration %v; counts: %v; want %+v", tries, serviceCounts, wts)
		time.Sleep(5 * time.Millisecond)
	}
	u.Fatalf("Failed to route RPCs with proper ratio")
}

// Tests and verifies that a WindowUpdate Frame is correctly written.
func validateRequestID(t *testing.T, expected int, requestBody []byte) {
	t.Helper()

	if requestBody == nil {
		t.Fatalf("request body is nil")
	}

	var response Response
	err := unmarshalResponse(requestBody, &response)
	if err != nil {
		t.Fatalf("Can't decode response: %v (%s)", err, requestBody)
	}

	actualID, err := response.ID.Int()
	if err != nil {
		t.Fatalf("Can't get requestID in response. err=%s, body=%s", err, requestBody)
	}

	if expected != actualID {
		t.Fatalf("Request ID mismatch: want %d, have %d (%s)", expected, actualID, requestBody)
	}
}

type Response struct {
	ID int
}

func unmarshalResponse(body []byte, r *Response) error {
	// 模拟反序列化逻辑
	if string(body) == "12345" {
		r.ID = 54321
		return nil
	}
	return fmt.Errorf("invalid body: %s", body)
}

// Tests and verifies that a Continuation Frame is correctly written with its
// flag permutations.
func TestAdvancedWriterNeglectsWritesToOriginalResponseWriter(t *testing.T) {
	t.Run("With Concat", func(t *testing.T) {
		// explicitly create the struct instead of NewRecorder to control the value of Status
		original := &httptest.ResponseRecorder{
			HeaderMap: make(http.Header),
			Body:      new(bytes.Buffer),
		}
		wrap := &advancedWriter{ResponseWriter: original}

		var buf bytes.Buffer
		wrap.Concat(&buf)
		wrap.Skip()

		_, err := wrap.Write([]byte("hello world"))
		assertNoError(t, err)

		assertEqual(t, 0, original.Status) // wrapper shouldn't call WriteHeader implicitly
		assertEqual(t, 0, original.Body.Len())
		assertEqual(t, []byte("hello world"), buf.Bytes())
		assertEqual(t, 11, wrap.CharactersWritten())
	})

	t.Run("Without Concat", func(t *testing.T) {
		// explicitly create the struct instead of NewRecorder to control the value of Status
		original := &httptest.ResponseRecorder{
			HeaderMap: make(http.Header),
			Body:      new(bytes.Buffer),
		}
		wrap := &advancedWriter{ResponseWriter: original}
		wrap.Skip()

		_, err := wrap.Write([]byte("hello world"))
		assertNoError(t, err)

		assertEqual(t, 0, original.Status) // wrapper shouldn't call WriteHeader implicitly
		assertEqual(t, 0, original.Body.Len())
		assertEqual(t, 11, wrap.CharactersWritten())
	})
}
