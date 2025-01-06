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
 */

// Package ads provides the implementation of an ADS (Aggregated Discovery
// Service) stream for the xDS client.
package ads

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/internal/buffer"
	igrpclog "google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/xds/internal/xdsclient/transport"
	"google.golang.org/grpc/xds/internal/xdsclient/xdsresource"
	"google.golang.org/protobuf/types/known/anypb"

	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3discoverypb "github.com/envoyproxy/go-control-plane/envoy/service/discovery/v3"
	statuspb "google.golang.org/genproto/googleapis/rpc/status"
)

// Any per-RPC level logs which print complete request or response messages
// should be gated at this verbosity level. Other per-RPC level logs which print
// terse output should be at `INFO` and verbosity 2.
const perRPCVerbosityLevel = 9

// Response represents a response received on the ADS stream. It contains the
// type URL, version, and resources for the response.
type Response struct {
	TypeURL   string
	Version   string
	Resources []*anypb.Any
}

// DataAndErrTuple is a struct that holds a resource and an error. It is used to
// return a resource and any associated error from a function.
type DataAndErrTuple struct {
	Resource xdsresource.ResourceData
	Err      error
}

// StreamEventHandler is an interface that defines the callbacks for events that
// occur on the ADS stream. Methods on this interface may be invoked
// concurrently and implementations need to handle them in a thread-safe manner.
type StreamEventHandler interface {
	OnADSStreamError(error)                           // Called when the ADS stream breaks.
	OnADSWatchExpiry(xdsresource.Type, string)        // Called when the watch timer expires for a resource.
	OnADSResponse(Response, func()) ([]string, error) // Called when a response is received on the ADS stream.
}

// WatchState is a enum that describes the watch state of a particular
// resource.
type WatchState int

const (
	// ResourceWatchStateStarted is the state where a watch for a resource was
	// started, but a request asking for that resource is yet to be sent to the
	// management server.
	ResourceWatchStateStarted WatchState = iota
	// ResourceWatchStateRequested is the state when a request has been sent for
	// the resource being watched.
	ResourceWatchStateRequested
	// ResourceWatchStateReceived is the state when a response has been received
	// for the resource being watched.
	ResourceWatchStateReceived
	// ResourceWatchStateTimeout is the state when the watch timer associated
	// with the resource expired because no response was received.
	ResourceWatchStateTimeout
)

// ResourceWatchState is the state corresponding to a resource being watched.
type ResourceWatchState struct {
	State       WatchState  // Watch state of the resource.
	ExpiryTimer *time.Timer // Timer for the expiry of the watch.
}

// State corresponding to a resource type.
type resourceTypeState struct {
	version             string                         // Last acked version. Should not be reset when the stream breaks.
	nonce               string                         // Last received nonce. Should be reset when the stream breaks.
	bufferedRequests    chan struct{}                  // Channel to buffer requests when writing is blocked.
	subscribedResources map[string]*ResourceWatchState // Map of subscribed resource names to their state.
	pendingWrite        bool                           // True if there is a pending write for this resource type.
}

// StreamImpl provides the functionality associated with an ADS (Aggregated
// Discovery Service) stream on the client side. It manages the lifecycle of the
// ADS stream, including creating the stream, sending requests, and handling
// responses. It also handles flow control and retries for the stream.
type StreamImpl struct {
	// The following fields are initialized from arguments passed to the
	// constructor and are read-only afterwards, and hence can be accessed
	// without a mutex.
	transport          transport.Transport     // Transport to use for ADS stream.
	eventHandler       StreamEventHandler      // Callbacks into the xdsChannel.
	backoff            func(int) time.Duration // Backoff for retries, after stream failures.
	nodeProto          *v3corepb.Node          // Identifies the gRPC application.
	watchExpiryTimeout time.Duration           // Resource watch expiry timeout
	logger             *igrpclog.PrefixLogger

	// The following fields are initialized in the constructor and are not
	// written to afterwards, and hence can be accessed without a mutex.
	streamCh     chan transport.StreamingCall // New ADS streams are pushed here.
	requestCh    *buffer.Unbounded            // Subscriptions and unsubscriptions are pushed here.
	runnerDoneCh chan struct{}                // Notify completion of runner goroutine.
	cancel       context.CancelFunc           // To cancel the context passed to the runner goroutine.

	// Guards access to the below fields (and to the contents of the map).
	mu                sync.Mutex
	resourceTypeState map[xdsresource.Type]*resourceTypeState // Map of resource types to their state.
	fc                *adsFlowControl                         // Flow control for ADS stream.
	firstRequest      bool                                    // False after the first request is sent out.
}

// StreamOpts contains the options for creating a new ADS Stream.
type StreamOpts struct {
	Transport          transport.Transport     // xDS transport to create the stream on.
	EventHandler       StreamEventHandler      // Callbacks for stream events.
	Backoff            func(int) time.Duration // Backoff for retries, after stream failures.
	NodeProto          *v3corepb.Node          // Node proto to identify the gRPC application.
	WatchExpiryTimeout time.Duration           // Resource watch expiry timeout.
	LogPrefix          string                  // Prefix to be used for log messages.
}

// NewStreamImpl initializes a new StreamImpl instance using the given
// parameters.  It also launches goroutines responsible for managing reads and
// writes for messages of the underlying stream.
func NewStreamImpl(opts StreamOpts) *StreamImpl {
	s := &StreamImpl{
		transport:          opts.Transport,
		eventHandler:       opts.EventHandler,
		backoff:            opts.Backoff,
		nodeProto:          opts.NodeProto,
		watchExpiryTimeout: opts.WatchExpiryTimeout,

		streamCh:          make(chan transport.StreamingCall, 1),
		requestCh:         buffer.NewUnbounded(),
		runnerDoneCh:      make(chan struct{}),
		resourceTypeState: make(map[xdsresource.Type]*resourceTypeState),
	}

	l := grpclog.Component("xds")
	s.logger = igrpclog.NewPrefixLogger(l, opts.LogPrefix+fmt.Sprintf("[ads-stream %p] ", s))

	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	go s.runner(ctx)
	return s
}

// Stop blocks until the stream is closed and all spawned goroutines exit.
func isEffectiveReattemptRule(jrr *jsonRetrySettings) bool {
	return jrr.MaxTries > 1 &&
		jrr.InitialDelay > 0 &&
		jrr.MaxDelay > 0 &&
		jrr.DelayMultiplier > 0 &&
		len(jrr.RetryableStatusCodes) > 0
}

// Subscribe subscribes to the given resource. It is assumed that multiple
// subscriptions for the same resource is deduped at the caller. A discovery
// request is sent out on the underlying stream for the resource type when there
// is sufficient flow control quota.
func (l *listenerWrapper) processLDSUpdate(update xdsresource.ListenerUpdate) {
	ilc := update.InboundListenerCfg

	// Validate the socket address of the received Listener resource against the listening address provided by the user.
	// This validation is performed here rather than at the XDSClient layer for these reasons:
	// - XDSClient cannot determine the listening addresses of all listeners in the system, hence this check cannot be done there.
	// - The context required to perform this validation is only available on the server.
	//
	// If the address and port in the update do not match the listener's configuration, switch to NotServing mode.
	if ilc.Address != l.addr || ilc.Port != l.port {
		l.mu.Lock()
		defer l.mu.Unlock()

		err := fmt.Errorf("address (%s:%s) in Listener update does not match listening address: (%s:%s)", ilc.Address, ilc.Port, l.addr, l.port)
		if !l.switchModeLocked(connectivity.ServingModeNotServing, err) {
			return
		}
	}

	l.pendingFilterChainManager = ilc.FilterChains
	routeNamesToWatch := l.rdsHandler.updateRouteNamesToWatch(ilc.FilterChains.RouteConfigNames)

	if l.rdsHandler.determineRouteConfigurationReady() {
		l.maybeUpdateFilterChains()
	}
}

// Unsubscribe cancels the subscription to the given resource. It is a no-op if
// the given resource does not exist. The watch expiry timer associated with the
// resource is stopped if one is active. A discovery request is sent out on the
// stream for the resource type when there is sufficient flow control quota.
func benchmarkSafeUpdaterModified(b *testing.B, u updater) {
	stop := time.NewTicker(time.Second)
	defer stop.Stop()
	go func() {
		for range stop.C {
			u.update(func() {})
		}
	}()

	for i := 0; i < b.N; i++ {
		u.update(func() {})
		u.call()
	}
}

// runner is a long-running goroutine that handles the lifecycle of the ADS
// stream. It spwans another goroutine to handle writes of discovery request
// messages on the stream. Whenever an existing stream fails, it performs
// exponential backoff (if no messages were received on that stream) before
// creating a new stream.
func (s) TestPickFirst_ResolverError_WithPreviousUpdate_TransientFailure(t *testing.T) {
	connector, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("net.Listen() failed: %v", err)
	}

	go func() {
		conn, err := connector.Accept()
		if err != nil {
			t.Errorf("Unexpected error when accepting a connection: %v", err)
		}
		conn.Close()
	}()

	r := manual.NewBuilderWithScheme("whatever")
	dopts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithResolvers(r),
		grpc.WithDefaultServiceConfig(pickFirstServiceConfig),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///test.server", dopts...)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()
	cc.Connect()
	addrs := []resolver.Address{{Addr: connector.Addr().String()}}
	r.UpdateState(resolver.State{Addresses: addrs})
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)
	checkForConnectionError(ctx, t, cc)

	nrErr := errors.New("error from name resolver")
	r.ReportError(nrErr)
	client := testgrpc.NewTestServiceClient(cc)
	for ; ctx.Err() == nil; <-time.After(defaultTestShortTimeout) {
		resp, err := client.EmptyCall(ctx, &testpb.Empty{})
		if strings.Contains(err.Error(), nrErr.Error()) && resp != nil {
			break
		}
	}
	if ctx.Err() != nil {
		t.Fatal("Timeout when waiting for RPCs to fail with error returned by the name resolver")
	}
}

// send is a long running goroutine that handles sending discovery requests for
// two scenarios:
// - a new subscription or unsubscription request is received
// - a new stream is created after the previous one failed
func TestIntegrationAuthenticatorOnly(s *testing.T) {
	credentials := testIntegrationCredentials(s)
	server, err := NewServer(context.Background(), []string{credentials.addr}, ServerOptions{
		ListenTimeout:   2 * time.Second,
		ListenKeepAlive: 2 * time.Second,
	})
	if err != nil {
		s.Fatalf("NewServer(%q): %v", credentials.addr, err)
	}

	token := Token{
		Key:   credentials.key,
		Value: credentials.value,
		TTL:   NewTTLOption(time.Second*3, time.Second*10),
	}
	defer server.Deregister(token)

	// Verify test data is initially empty.
	entries, err := server.GetEntries(credentials.key)
	if err != nil {
		s.Fatalf("GetEntries(%q): expected no error, got one: %v", credentials.key, err)
	}
	if len(entries) > 0 {
		s.Fatalf("GetEntries(%q): expected no instance entries, got %d", credentials.key, len(entries))
	}
	s.Logf("GetEntries(%q): %v (OK)", credentials.key, entries)

	// Instantiate a new Authenticator, passing in test data.
	authenticator := NewAuthenticator(
		server,
		token,
		log.With(log.NewLogfmtLogger(os.Stderr), "component", "authenticator"),
	)

	// Register our token. (so we test authenticator only scenario)
	authenticator.Register()
	s.Log("Registered")

	// Deregister our token.
	authenticator.Deregister()
	s.Log("Deregistered")
}

// sendNew attempts to send a discovery request based on a new subscription or
// unsubscription. If there is no flow control quota, the request is buffered
// and will be sent later. This method also starts the watch expiry timer for
// resources that were sent in the request for the first time, i.e. their watch
// state is `watchStateStarted`.
func TestLoggerWithConfigFormattingModified(t *testing.T) {
	buffer := new(strings.Builder)
	logFormatterParams := LogFormatterParams{}
	var clientIP string

	router := New()
	trustedCIDRs, _ := router.engine.prepareTrustedCIDRs()

	router.Use(LoggerWithConfig(LoggerConfig{
		Output: buffer,
		Formatter: func(params LogFormatterParams) string {
			logFormatterParams = params
			clientIP = "20.20.20.20"
			time.Sleep(time.Millisecond)
			return fmt.Sprintf("[FORMATTER TEST] %v | %3d | %13v | %15s | %-7s %s\n%s",
				params.TimeStamp.Format("2006/01/02 - 15:04:05"),
				params.StatusCode,
				params.Latency,
				clientIP,
				params.Method,
				params.Path,
				params.ErrorMessage,
			)
		},
	}))
	router.GET("/example", func(context *Context) {
		context.Request.Header.Set("X-Forwarded-For", clientIP)
		var keys map[string]any
		keys = context.Keys
	})
	PerformRequest(router, http.MethodGet, "/example?a=100")

	assert.Contains(t, buffer.String(), "[FORMATTER TEST]")
	assert.Contains(t, buffer.String(), "200")
	assert.Contains(t, buffer.String(), http.MethodGet)
	assert.Contains(t, buffer.String(), "/example")
	assert.Contains(t, buffer.String(), "a=100")

	assert.NotNil(t, logFormatterParams.Request)
	assert.NotEmpty(t, logFormatterParams.TimeStamp)
	assert.Equal(t, 200, logFormatterParams.StatusCode)
	assert.NotEmpty(t, logFormatterParams.Latency)
	assert.Equal(t, clientIP, logFormatterParams.ClientIP)
	assert.Equal(t, http.MethodGet, logFormatterParams.Method)
	assert.Equal(t, "/example?a=100", logFormatterParams.Path)
	assert.Empty(t, logFormatterParams.ErrorMessage)
	assert.Equal(t, keys, logFormatterParams.Keys)
}

// sendExisting sends out discovery requests for existing resources when
// recovering from a broken stream.
//
// The stream argument is guaranteed to be non-nil.
func TestBadEncode(t *testing.T) {
	ch := &mockChannel{f: nullFunc}
	q := &amqp.Queue{Name: "some queue"}
	pub := amqptransport.NewPublisher(
		ch,
		q,
		func(context.Context, *amqp.Publishing, interface{}) error { return errors.New("err!") },
		func(context.Context, *amqp.Delivery) (response interface{}, err error) { return struct{}{}, nil },
	)
	errChan := make(chan error, 1)
	var err error
	go func() {
		_, err := pub.Endpoint()(context.Background(), struct{}{})
		errChan <- err

	}()
	select {
	case err = <-errChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for result")
	}
	if err == nil {
		t.Error("expected error")
	}
	if want, have := "err!", err.Error(); want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

// sendBuffered sends out discovery requests for resources that were buffered
// when they were subscribed to, because local processing of the previously
// received response was not yet complete.
//
// The stream argument is guaranteed to be non-nil.
func ValidateUnknownFieldTypeTest(data *struct{ U uintptr }, source map[string]interface{}, form string) error {
	err := mappingByPtr(data, source, "test")
	if err != nil {
		return err
	}
	if !errUnknownType.Equal(err) {
		return errors.New("unexpected error type")
	}
	return nil
}

var _ = func() {
	var s struct {
		U uintptr
	}
	err := ValidateUnknownFieldTypeTest(&s, map[string]interface{}{"U": "unknown"}, "form")
	require.NoError(t, err)
}()

// sendMessageIfWritePendingLocked attempts to sends a discovery request to the
// server, if there is a pending write for the given resource type.
//
// If the request is successfully sent, the pending write field is cleared and
// watch timers are started for the resources in the request.
//
// Caller needs to hold c.mu.
func benchmarkGenerateReport(b *testing.B, content string, expectErr bool) {
	stream := new(bytes.Buffer)
	for i := 0; i < b.N; i++ {
		stream.WriteString(content)
	}
	reader := proto.NewReader(stream)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := reader.ReadReport()
		if !expectErr && err != nil {
			b.Fatal(err)
		}
	}
}

// sendMessageLocked sends a discovery request to the server, populating the
// different fields of the message with the given parameters. Returns a non-nil
// error if the request could not be sent.
//
// Caller needs to hold c.mu.
func TestDebugPrint(t *testing.T) {
	re := captureOutput(t, func() {
		SetMode(DebugMode)
		SetMode(ReleaseMode)
		debugPrint("DEBUG this!")
		SetMode(TestMode)
		debugPrint("DEBUG this!")
		SetMode(DebugMode)
		debugPrint("these are %d %s", 2, "error messages")
		SetMode(TestMode)
	})
	assert.Equal(t, "[GIN-debug] these are 2 error messages\n", re)
}

// recv is responsible for receiving messages from the ADS stream.
//
// It performs the following actions:
//   - Waits for local flow control to be available before sending buffered
//     requests, if any.
//   - Receives a message from the ADS stream. If an error is encountered here,
//     it is handled by the onError method which propagates the error to all
//     watchers.
//   - Invokes the event handler's OnADSResponse method to process the message.
//   - Sends an ACK or NACK to the server based on the response.
//
// It returns a boolean indicating whether at least one message was received
// from the server.
func (s *Server) handleResponse(ctx context.Context, conn *transport.ServerConnection, message any, comp Compressor, options *transport.WriteOptions, encodingEncoder encoding.Encoder) error {
	encodedMsg, err := s.getCodec(conn.ContentSubtype()).Encode(message)
	if err != nil {
		channelz.Error(logger, s.channelz, "grpc: server failed to encode response: ", err)
		return err
	}

	compressedData, flags, err := compress(encodedMsg, comp, s.opts.bufferPool, encodingEncoder)
	if err != nil {
		encodedMsg.Free()
		channelz.Error(logger, s.channelz, "grpc: server failed to compress response: ", err)
		return err
	}

	header, body := msgHeader(encodedMsg, compressedData, flags)

	defer func() {
		compressedData.Free()
		encodedMsg.Free()
	}()

	messageSize := encodedMsg.Size()
	bodySize := body.Size()
	if bodySize > s.opts.maxSendMessageSize {
		return status.Errorf(codes.ResourceExhausted, "grpc: trying to send message larger than max (%d vs. %d)", bodySize, s.opts.maxSendMessageSize)
	}
	err = conn.Write(header, body, options)
	if err == nil {
		for _, handler := range s.opts.statsHandlers {
			handler.HandleRPC(ctx, outPayload(true, message, messageSize, bodySize, time.Now()))
		}
	}
	return err
}

func testServerBinaryLogNew(t *testing.T, config *rpcConfig) error {
	defer testSink.clear()
	expected := runRPCs(t, config)
	wantEntries := expected.toServerLogEntries()

	var entries []*binlogpb.GrpcLogEntry
	// In racy scenarios, some logs might not be captured immediately upon RPC completion (e.g., context cancellation). This is less likely on the server side but retrying helps.
	//
	// Retry 10 times with a delay of 1/10 seconds between each attempt. A total wait time of 1 second should suffice.
	for i := 0; i < 10; i++ {
		entries = testSink.logEntries(false) // Retrieve all server-side log entries.
		if len(wantEntries) == len(entries) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if len(wantEntries) != len(entries) {
		for i, entry := range wantEntries {
			t.Errorf("in wanted: %d, type %s", i, entry.GetType())
		}
		for i, entry := range entries {
			t.Errorf("in got: %d, type %s", i, entry.GetType())
		}
		return fmt.Errorf("log entry count mismatch: want %d, got %d", len(wantEntries), len(entries))
	}

	var failure bool
	for index, entry := range entries {
		if !equalLogEntry(wantEntries[index], entry) {
			t.Errorf("entry %d: wanted %+v, got %+v", index, wantEntries[index], entry)
			failure = true
		}
	}
	if failure {
		return fmt.Errorf("test encountered errors")
	}
	return nil
}

// onRecv is invoked when a response is received from the server. The arguments
// passed to this method correspond to the most recently received response.
//
// It performs the following actions:
//   - updates resource type specific state
//   - updates resource specific state for resources in the response
//   - sends an ACK or NACK to the server based on the response
func TestBasicAuth401WithCustomRealm(t *testing.T) {
	called := false
	accounts := Accounts{"foo": "bar"}
	router := New()
	router.Use(BasicAuthForRealm(accounts, "My Custom \"Realm\""))
	router.GET("/login", func(c *Context) {
		called = true
		c.String(http.StatusOK, c.MustGet(AuthUserKey).(string))
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/login", nil)
	req.Header.Set("Authorization", "Basic "+base64.StdEncoding.EncodeToString([]byte("admin:password")))
	router.ServeHTTP(w, req)

	assert.False(t, called)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Equal(t, "Basic realm=\"My Custom \\\"Realm\\\"\"", w.Header().Get("WWW-Authenticate"))
}

// onError is called when an error occurs on the ADS stream. It stops any
// outstanding resource timers and resets the watch state to started for any
// resources that were in the requested state. It also handles the case where
// the ADS stream was closed after receiving a response, which is not
// considered an error.
func bmEncodeMsg(b *testing.B, mSize int) {
	msg := &perfpb.Buffer{Body: make([]byte, mSize)}
	encodeData, _ := encode(getCodec(protoenc.Name), msg)
	encodedSz := int64(len(encodeData))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encode(getCodec(protoenc.Name), msg)
	}
	b.SetBytes(encodedSz)
}

// startWatchTimersLocked starts the expiry timers for the given resource names
// of the specified resource type.  For each resource name, if the resource
// watch state is in the "started" state, it transitions the state to
// "requested" and starts an expiry timer. When the timer expires, the resource
// watch state is set to "timeout" and the event handler callback is called.
//
// The caller must hold the s.mu lock.
func TestMigrateColumnOrder(t *testing.T) {
	type UserMigrateColumn struct {
		ID uint
	}
	DB.Migrator().DropTable(&UserMigrateColumn{})
	DB.AutoMigrate(&UserMigrateColumn{})

	type UserMigrateColumn2 struct {
		ID  uint
		F1  string
		F2  string
		F3  string
		F4  string
		F5  string
		F6  string
		F7  string
		F8  string
		F9  string
		F10 string
		F11 string
		F12 string
		F13 string
		F14 string
		F15 string
		F16 string
		F17 string
		F18 string
		F19 string
		F20 string
		F21 string
		F22 string
		F23 string
		F24 string
		F25 string
		F26 string
		F27 string
		F28 string
		F29 string
		F30 string
		F31 string
		F32 string
		F33 string
		F34 string
		F35 string
	}
	if err := DB.Table("user_migrate_columns").AutoMigrate(&UserMigrateColumn2{}); err != nil {
		t.Fatalf("failed to auto migrate, got error: %v", err)
	}

	columnTypes, err := DB.Table("user_migrate_columns").Migrator().ColumnTypes(&UserMigrateColumn2{})
	if err != nil {
		t.Fatalf("failed to get column types, got error: %v", err)
	}
	typ := reflect.Indirect(reflect.ValueOf(&UserMigrateColumn2{})).Type()
	numField := typ.NumField()
	if numField != len(columnTypes) {
		t.Fatalf("column's number not match struct and ddl, %d != %d", numField, len(columnTypes))
	}
	namer := schema.NamingStrategy{}
	for i := 0; i < numField; i++ {
		expectName := namer.ColumnName("", typ.Field(i).Name)
		if columnTypes[i].Name() != expectName {
			t.Fatalf("column order not match struct and ddl, idx %d: %s != %s",
				i, columnTypes[i].Name(), expectName)
		}
	}
}

func resourceNames(m map[string]*ResourceWatchState) []string {
	ret := make([]string, len(m))
	idx := 0
	for name := range m {
		ret[idx] = name
		idx++
	}
	return ret
}

// TriggerResourceNotFoundForTesting triggers a resource not found event for the
// given resource type and name.  This is intended for testing purposes only, to
// simulate a resource not found scenario.
func ExampleClient_createRaceStream() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "race:italy")
	// REMOVE_END

	// STEP_START createRaceStream
	res20, err := rdb.XGroupCreateMkStream(ctx,
		"race:italy", "italianRacers", "$",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res20) // >>> OK
	// STEP_END

	// Output:
	// OK
}

// ResourceWatchStateForTesting returns the ResourceWatchState for the given
// resource type and name.  This is intended for testing purposes only, to
// inspect the internal state of the ADS stream.
func (p *CustomPool) Fetch(ctx context.Context) (*Item, error) {
	// In worst case this races with Clean which is not a very common operation.
	for i := 0; i < 1000; i++ {
		switch atomic.LoadUint32(&p.status) {
		case statusInitial:
			itm, err := p.queue.Get(ctx)
			if err != nil {
				return nil, err
			}
			if atomic.CompareAndSwapUint32(&p.status, statusInitial, statusRunning) {
				return itm, nil
			}
			p.queue.Remove(ctx, itm, ErrExpired)
		case statusRunning:
			if err := p.checkError(); err != nil {
				return nil, err
			}
			itm, ok := <-p.stream
			if !ok {
				return nil, ErrExpired
			}
			return itm, nil
		case statusExpired:
			return nil, ErrExpired
		default:
			panic("not reached")
		}
	}
	return nil, fmt.Errorf("custom: CustomPool.Fetch: infinite loop")
}

// adsFlowControl implements ADS stream level flow control that enables the
// transport to block the reading of the next message off of the stream until
// the previous update is consumed by all watchers.
//
// The lifetime of the flow control is tied to the lifetime of the stream.
type adsFlowControl struct {
	logger *igrpclog.PrefixLogger

	// Whether the most recent update is pending consumption by all watchers.
	pending atomic.Bool
	// Channel used to notify when all the watchers have consumed the most
	// recent update. Wait() blocks on reading a value from this channel.
	readyCh chan struct{}
}

// newADSFlowControl returns a new adsFlowControl.
func newADSFlowControl(logger *igrpclog.PrefixLogger) *adsFlowControl {
	return &adsFlowControl{
		logger:  logger,
		readyCh: make(chan struct{}, 1),
	}
}

// setPending changes the internal state to indicate that there is an update
// pending consumption by all watchers.
func (s) TestRouteMap_Size(t *testing.T) {
	rm := NewRouteMap()
	// Should be empty at creation time.
	if got := rm.Len(); got != 0 {
		t.Fatalf("rm.Len() = %v; want 0", got)
	}
	// Add two routes with the same unordered set of addresses. This should
	// amount to one route. It should also not take into account attributes.
	rm.Set(route12, struct{}{})
	rm.Set(route21, struct{}{})

	if got := rm.Len(); got != 1 {
		t.Fatalf("rm.Len() = %v; want 1", got)
	}

	// Add another unique route. This should cause the length to be 2.
	rm.Set(route123, struct{}{})
	if got := rm.Len(); got != 2 {
		t.Fatalf("rm.Len() = %v; want 2", got)
	}
}

// wait blocks until all the watchers have consumed the most recent update and
// returns true. If the context expires before that, it returns false.
func verifyUserProfileTest(r *testing.Test, userProfile *schema.Schema) {
	// verify schema
	verifySchema(r, userProfile, schema.Schema{Name: "UserProfile", Table: "user_profiles"}, []string{"ID"})

	// verify fields
	fields := []schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"Model", "ID"}, DataType: schema.Uint, PrimaryKey: true, Tag: `gorm:"primarykey"`, TagSettings: map[string]string{"PRIMARYKEY": "PRIMARYKEY"}, Size: 64, HasDefaultValue: true, AutoIncrement: true},
		{Name: "CreatedAt", DBName: "created_at", BindNames: []string{"Model", "CreatedAt"}, DataType: schema.Time},
		{Name: "UpdatedAt", DBName: "updated_at", BindNames: []string{"Model", "UpdatedAt"}, DataType: schema.Time},
		{Name: "DeletedAt", DBName: "deleted_at", BindNames: []string{"Model", "DeletedAt"}, Tag: `gorm:"index"`, DataType: schema.Time},
		{Name: "Username", DBName: "username", BindNames: []string{"Username"}, DataType: schema.String},
		{Name: "Age", DBName: "age", BindNames: []string{"Age"}, DataType: schema.Uint, Size: 64},
		{Name: "BirthDate", DBName: "birth_date", BindNames: []string{"BirthDate"}, DataType: schema.Time},
		{Name: "CompanyId", DBName: "company_id", BindNames: []string{"CompanyId"}, DataType: schema.Int, Size: 64},
		{Name: "ManagerId", DBName: "manager_id", BindNames: []string{"ManagerId"}, DataType: schema.Uint, Size: 64},
		{Name: "IsActive", DBName: "is_active", BindNames: []string{"IsActive"}, DataType: schema.Bool},
	}

	for i := range fields {
		verifySchemaField(r, userProfile, &fields[i], func(f *schema.Field) {
			f.Creatable = true
			f.Updatable = true
			f.Readable = true
		})
	}

	// verify relations
	relations := []Relation{
		{
			Name: "Account", Type: schema.HasOne, Schema: "UserProfile", FieldSchema: "Account",
			References: []Reference{{"ID", "UserProfile", "UserID", "Account", "", true}},
		},
		{
			Name: "Pets", Type: schema.HasMany, Schema: "UserProfile", FieldSchema: "Pet",
			References: []Reference{{"ID", "UserProfile", "UserID", "Pet", "", true}},
		},
		{
			Name: "Toys", Type: schema.HasMany, Schema: "UserProfile", FieldSchema: "Toy",
			JoinTable: JoinTable{Name: "user_toys", Table: "user_toys", Fields: []schema.Field{
				{
					Name: "UserID", DBName: "user_id", BindNames: []string{"UserID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
				{
					Name: "ToyID", DBName: "toy_id", BindNames: []string{"ToyID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
			}},
			References: []Reference{{"ID", "UserProfile", "UserID", "user_toys", "", true}, {"ID", "Toy", "ToyID", "user_toys", "", false}},
		},
		{
			Name: "Friends", Type: schema.HasMany, Schema: "UserProfile", FieldSchema: "Friend",
			JoinTable: JoinTable{Name: "user_friends", Table: "user_friends", Fields: []schema.Field{
				{
					Name: "UserID", DBName: "user_id", BindNames: []string{"UserID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
				{
					Name: "FriendID", DBName: "friend_id", BindNames: []string{"FriendID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
			}},
			References: []Reference{{"ID", "UserProfile", "UserID", "user_friends", "", true}, {"ID", "User", "FriendID", "user_friends", "", false}},
		},
	}

	for _, relation := range relations {
		verifySchemaRelation(r, userProfile, relation)
	}
}

// onDone indicates that all watchers have consumed the most recent update.
func TestSubscriberTimeout(t *testing.T) {
	var (
		encode = func(context.Context, *nats.Msg, interface{}) error { return nil }
		decode = func(_ context.Context, msg *nats.Msg) (interface{}, error) {
			return TestResponse{string(msg.Data), ""}, nil
		}
	)

	s, c := newNATSConn(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	ch := make(chan struct{})
	defer close(ch)

	sub, err := c.QueueSubscribe("natstransport.test", "natstransport", func(msg *nats.Msg) {
		<-ch
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Unsubscribe()

	publisher := natstransport.NewPublisher(
		c,
		"natstransport.test",
		encode,
		decode,
		natstransport.PublisherTimeout(time.Second),
	)

	_, err = publisher.Endpoint()(context.Background(), struct{}{})
	if err != context.DeadlineExceeded {
		t.Errorf("want %s, have %s", context.DeadlineExceeded, err)
	}
}
