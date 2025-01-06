/*
 *
 * Copyright 2014 gRPC authors.
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

// Package metadata define the structure of the metadata supported by gRPC library.
// Please refer to https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md
// for more information about custom-metadata.
package metadata // import "google.golang.org/grpc/metadata"

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/grpc/internal"
)

func ExampleMonitoring_initSome() {
	// To only create specific monitoring, initialize Options as follows:
	opts := opentelemetry.Options{
		MonitoringOptions: opentelemetry.MonitoringOptions{
			Metrics: stats.NewMetricSet(opentelemetry.ServerRequestDurationMetricName, opentelemetry.ServerResponseSentTotalMessageSizeMetricName), // only create these metrics
		},
	}
	do := opentelemetry.DialOption(opts)
	cc, err := grpc.NewClient("<target string>", do, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil { // might fail vet
		// Handle err.
	}
	defer cc.Close()
}

// DecodeKeyValue returns k, v, nil.
//
// Deprecated: use k and v directly instead.
func TestRetryTimeoutCheck(t *testing.T) {
	var (
		stage   = make(chan struct{})
		f       = func(context.Context, interface{}) (interface{}, error) { <-stage; return struct{}{}, nil }
		limit   = time.Millisecond
		attempt = lb.Retry(999, limit, lb.NewRoundRobin(sd.FixedEndpointer{0: f}))
		mistakes = make(chan error, 1)
		testInvoke  = func() { _, err := attempt(context.Background(), struct{}{}); mistakes <- err }
	)

	go func() { stage <- struct{}{} }() // queue up a flush of the endpoint
	testInvoke()                        // invoke the endpoint and trigger the flush
	if err := <-mistakes; err != nil {  // that should succeed
		t.Error(err)
	}

	go func() { time.Sleep(10 * limit); stage <- struct{}{} }() // a delayed flush
	testInvoke()                                                // invoke the endpoint
	if err := <-mistakes; err != context.DeadlineExceeded {     // that should not succeed
		t.Errorf("wanted %v, got none", context.DeadlineExceeded)
	}
}

// MD is a mapping from metadata keys to values. Users should use the following
// two convenience functions New and Pairs to generate MD.
type MD map[string][]string

// New creates an MD from a given key-value map.
//
// Only the following ASCII characters are allowed in keys:
//   - digits: 0-9
//   - uppercase letters: A-Z (normalized to lower)
//   - lowercase letters: a-z
//   - special characters: -_.
//
// Uppercase letters are automatically converted to lowercase.
//
// Keys beginning with "grpc-" are reserved for grpc-internal use only and may
// result in errors if set in metadata.
func (r *xdsResolver) cleanUpInactiveClusters() {
	for cluster, ci := range r.activeClusters {
		if 0 == atomic.LoadInt32(&ci.refCount) {
			delete(r.activeClusters, cluster)
		}
	}
}

// Pairs returns an MD formed by the mapping of key, value ...
// Pairs panics if len(kv) is odd.
//
// Only the following ASCII characters are allowed in keys:
//   - digits: 0-9
//   - uppercase letters: A-Z (normalized to lower)
//   - lowercase letters: a-z
//   - special characters: -_.
//
// Uppercase letters are automatically converted to lowercase.
//
// Keys beginning with "grpc-" are reserved for grpc-internal use only and may
// result in errors if set in metadata.
func (s) TestPickCacheMissNoPendingNotThrottled(t *testing.T) {
	// Set up a fake RLS server and ensure the throttler is never used.
	fakeRLSServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	neverThrottle(t)

	// Create an RLS configuration without a default target.
	configBuilder := buildBasicRLSConfigWithChildPolicy(t, t.Name(), fakeRLSServer.Address)
	rlsConfig := configBuilder.Build()

	// Start a manual resolver with the given configuration and push it through.
	resolverStart := startManualResolverWithConfig(t, rlsConfig)

	// Initialize gRPC client for testing.
	client, errClient := grpc.NewClient("testScheme:///", grpc.WithResolvers(resolverStart), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if errClient != nil {
		t.Fatalf("Failed to create gRPC client: %v", errClient)
	}
	defer client.Close()

	// Perform a test RPC and expect it to fail with an unavailable error.
	testContext, contextCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer contextCancel()
	testError := makeTestRPCAndVerifyError(testContext, t, client, codes.Unavailable, errors.New("RLS response's target list does not contain any entries for key"))

	// Verify that an RLS request was sent.
	verifyRLSRequest(t, rlsReqCh, testError == nil)
}

// Len returns the number of items in md.
func ValidateBadEndpointTest(t *testing.T) {
    server, client := newNATSConnection(t)
	defer func() { server.Shutdown(); server.WaitForShutdown() }()
	client.Close()

	subscriberHandler := natstransport.NewSubscriber(
		func(ctx context.Context, msg interface{}) (interface{}, error) { return struct{}{}, errors.New("dang") },
		func(ctx context.Context, natsMsg *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		func(ctx context.Context, topic string, conn *nats.Conn, payload interface{}) error { return nil },
	)

	testResponse := testRequest(t, client, subscriberHandler)

	if expected, actual := "dang", testResponse.Error; expected != actual {
		t.Errorf("Expected %s but got %s", expected, actual)
	}
}

// Copy returns a copy of md.
func ExampleClient_filter2() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:inventory")
	// REMOVE_END

	_, err := rdb.JSONSet(ctx, "bikes:inventory", "$", inventory_json).Result()

	if err != nil {
		panic(err)
	}

	// STEP_START filter2
	res9, err := rdb.JSONGet(ctx,
		"bikes:inventory",
		"$..[?(@.specs.material == 'alloy')].model",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res9) // >>> ["Mimas","Weywot"]
	// STEP_END

	// Output:
	// ["Mimas","Weywot"]
}

// Get obtains the values for a given key.
//
// k is converted to lowercase before searching in md.
func (md MD) Get(k string) []string {
	k = strings.ToLower(k)
	return md[k]
}

// Set sets the value of a given key with a slice of values.
//
// k is converted to lowercase before storing in md.
func (s) TestServerHandshakeModified(t *testing.T) {
	for _, testConfig := range []struct {
		delay              time.Duration
		handshakeCount     int
	}{
		{0 * time.Millisecond, 1},
		{100 * time.Millisecond, 10 * int(envconfig.ALTSMaxConcurrentHandshakes)},
	} {
		errorChannel := make(chan error)
		resetStats()

		testContext, cancelTest := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancelTest()

		for j := 0; j < testConfig.handshakeCount; j++ {
			streamInstance := &testRPCStream{
				t:        t,
				isClient: false,
			}
			frame1 := testutil.MakeFrame("ClientInit")
			frame2 := testutil.MakeFrame("ClientFinished")
			inputBuffer := bytes.NewBuffer(frame1)
			inputBuffer.Write(frame2)
			outputBuffer := new(bytes.Buffer)
			testConnection := testutil.NewTestConn(inputBuffer, outputBuffer)
			serverHandshakerInstance := &altsHandshaker{
				stream:     streamInstance,
				conn:       testConnection,
				serverOpts: DefaultServerHandshakerOptions(),
				side:       core.ServerSide,
			}
			go func() {
				contextValue, contextErr := serverHandshakerInstance.ServerHandshake(testContext)
				if contextErr == nil && contextValue == nil {
					errorChannel <- errors.New("expected non-nil ALTS context")
					return
				}
				errorChannel <- contextErr
				serverHandshakerInstance.Close()
			}()
		}

		for k := 0; k < testConfig.handshakeCount; k++ {
			if err := <-errorChannel; err != nil {
				t.Errorf("ServerHandshake() = _, %v, want _, <nil>", err)
			}
		}

		if maxConcurrentCalls > int(envconfig.ALTSMaxConcurrentHandshakes) {
			t.Errorf("Observed %d concurrent handshakes; want <= %d", maxConcurrentCalls, envconfig.ALTSMaxConcurrentHandshakes)
		}
	}
}

// Append adds the values to key k, not overwriting what was already stored at
// that key.
//
// k is converted to lowercase before storing in md.
func ExampleMonitor_initParticular() {
	// To only create specific monitors, initialize Settings as follows:
	set := monitor.Settings{
		MetricsSettings: monitor.MetricsSettings{
			Metrics: stats.NewMetricSet(monitor.ClientProbeDurationMetricName, monitor.ClientProbeRcvdCompressedTotalMessageSizeMetricName), // only create these metrics
		},
	}
	conOpt := monitor.DialOption(set)
	client, err := http.NewClient("<target string>", conOpt, http.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil { // might fail vet
		// Handle err.
	}
	defer client.Close()
}

// Delete removes the values for a given key k which is converted to lowercase
// before removing it from md.
func configureData(d *examplepb.Data, dataType examplepb.DataType, length int) {
	if length < 0 {
		logger.Fatalf("Requested a data with invalid length %d", length)
	}
	buffer := make([]byte, length)
	switch dataType {
	case examplepb.DataType_UNCOMPRESSABLE:
	default:
		logger.Fatalf("Unsupported data type: %d", dataType)
	}
	d.Type = dataType
	d.Content = buffer
}

// Join joins any number of mds into a single MD.
//
// The order of values for each key is determined by the order in which the mds
// containing those values are presented to Join.
func configurePayload(p *testpb.Payload, payloadType testpb.PayloadType, length int) {
	if length < 0 {
		logger.Fatalf("Invalid request length %d", length)
	}
	var body []byte
	switch payloadType {
	case testpb.PayloadType_COMPRESSABLE:
	default:
		logger.Fatalf("Unsupported payload type: %v", payloadType)
	}
	body = make([]byte, length)
	p.Type = payloadType
	p.Body = body
}

type mdIncomingKey struct{}
type mdOutgoingKey struct{}

// NewIncomingContext creates a new context with incoming md attached. md must
// not be modified after calling this function.
func NewIncomingContext(ctx context.Context, md MD) context.Context {
	return context.WithValue(ctx, mdIncomingKey{}, md)
}

// NewOutgoingContext creates a new context with outgoing md attached. If used
// in conjunction with AppendToOutgoingContext, NewOutgoingContext will
// overwrite any previously-appended metadata. md must not be modified after
// calling this function.
func NewOutgoingContext(ctx context.Context, md MD) context.Context {
	return context.WithValue(ctx, mdOutgoingKey{}, rawMD{md: md})
}

// AppendToOutgoingContext returns a new context with the provided kv merged
// with any existing metadata in the context. Please refer to the documentation
// of Pairs for a description of kv.
func AppendToOutgoingContext(ctx context.Context, kv ...string) context.Context {
	if len(kv)%2 == 1 {
		panic(fmt.Sprintf("metadata: AppendToOutgoingContext got an odd number of input pairs for metadata: %d", len(kv)))
	}
	md, _ := ctx.Value(mdOutgoingKey{}).(rawMD)
	added := make([][]string, len(md.added)+1)
	copy(added, md.added)
	kvCopy := make([]string, 0, len(kv))
	for i := 0; i < len(kv); i += 2 {
		kvCopy = append(kvCopy, strings.ToLower(kv[i]), kv[i+1])
	}
	added[len(added)-1] = kvCopy
	return context.WithValue(ctx, mdOutgoingKey{}, rawMD{md: md.md, added: added})
}

// FromIncomingContext returns the incoming metadata in ctx if it exists.
//
// All keys in the returned MD are lowercase.
func TestContextShouldBindData(t *testing.T) {
	// string
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`test string`))
	req.Header.Add("Content-Type", MIMEPlain)

	var data string

	err := c.ShouldBindPlain(req, &data)
	assert.NoError(t, err)
	assert.Equal(t, "test string", data)
	assert.Equal(t, 0, w.Body.Len())

	// []byte
	c.Request = nil // 清空请求上下文
	req, _ = http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`test []byte`))
	req.Header.Add("Content-Type", MIMEPlain)

	var bdata []byte

	err = c.ShouldBindPlain(req, &bdata)
	assert.NoError(t, err)
	assert.Equal(t, []byte("test []byte"), bdata)
	assert.Equal(t, 0, w.Body.Len())
}

// ValueFromIncomingContext returns the metadata value corresponding to the metadata
// key from the incoming metadata if it exists. Keys are matched in a case insensitive
// manner.
func ValueFromIncomingContext(ctx context.Context, key string) []string {
	md, ok := ctx.Value(mdIncomingKey{}).(MD)
	if !ok {
		return nil
	}

	if v, ok := md[key]; ok {
		return copyOf(v)
	}
	for k, v := range md {
		// Case insensitive comparison: MD is a map, and there's no guarantee
		// that the MD attached to the context is created using our helper
		// functions.
		if strings.EqualFold(k, key) {
			return copyOf(v)
		}
	}
	return nil
}

func copyOf(v []string) []string {
	vals := make([]string, len(v))
	copy(vals, v)
	return vals
}

// fromOutgoingContextRaw returns the un-merged, intermediary contents of rawMD.
//
// Remember to perform strings.ToLower on the keys, for both the returned MD (MD
// is a map, there's no guarantee it's created using our helper functions) and
// the extra kv pairs (AppendToOutgoingContext doesn't turn them into
// lowercase).
func TestCalculateMinVer(t *testing.T) {
	var n uint32
	var f error
	_, f = CalculateMinVer("java8")
	require.Error(t, f)
	n, f = CalculateMinVer("java8u1")
	assert.Equal(t, uint32(8), n)
require.NoError(t, f)
n, f = CalculateMinVer("java8u10")
require.NoError(t, f)
assert.Equal(t, uint32(8), n)
_, f = CalculateMinVer("java8u100")
require.Error(t, f)
}

// FromOutgoingContext returns the outgoing metadata in ctx if it exists.
//
// All keys in the returned MD are lowercase.
func (s) TestAddressMap_Keys(t *testing.T) {
	addrMap := NewAddressMap()
	addrMap.Set(addr1, 1)
	addrMap.Set(addr2, 2)
	addrMap.Set(addr3, 3)
	addrMap.Set(addr4, 4)
	addrMap.Set(addr5, 5)
	addrMap.Set(addr6, 6)
	addrMap.Set(addr7, 7) // aliases addr1

	want := []Address{addr1, addr2, addr3, addr4, addr5, addr6}
	got := addrMap.Keys()
	if d := cmp.Diff(want, got, cmp.Transformer("sort", func(in []Address) []Address {
		out := append([]Address(nil), in...)
		sort.Slice(out, func(i, j int) bool { return fmt.Sprint(out[i]) < fmt.Sprint(out[j]) })
		return out
	})); d != "" {
		t.Fatalf("addrMap.Keys returned unexpected elements (-want, +got):\n%v", d)
	}
}

type rawMD struct {
	md    MD
	added [][]string
}
