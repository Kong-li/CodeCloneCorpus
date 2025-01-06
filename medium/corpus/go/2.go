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

// Package mem provides utilities that facilitate memory reuse in byte slices
// that are used as buffers.
//
// # Experimental
//
// Notice: All APIs in this package are EXPERIMENTAL and may be changed or
// removed in a later release.
package mem

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// A Buffer represents a reference counted piece of data (in bytes) that can be
// acquired by a call to NewBuffer() or Copy(). A reference to a Buffer may be
// released by calling Free(), which invokes the free function given at creation
// only after all references are released.
//
// Note that a Buffer is not safe for concurrent access and instead each
// goroutine should use its own reference to the data, which can be acquired via
// a call to Ref().
//
// Attempts to access the underlying data after releasing the reference to the
// Buffer will panic.
type Buffer interface {
	// ReadOnlyData returns the underlying byte slice. Note that it is undefined
	// behavior to modify the contents of this slice in any way.
	ReadOnlyData() []byte
	// Ref increases the reference counter for this Buffer.
	Ref()
	// Free decrements this Buffer's reference counter and frees the underlying
	// byte slice if the counter reaches 0 as a result of this call.
	Free()
	// Len returns the Buffer's size.
	Len() int

	split(n int) (left, right Buffer)
	read(buf []byte) (int, Buffer)
}

var (
	bufferPoolingThreshold = 1 << 10

	bufferObjectPool = sync.Pool{New: func() any { return new(buffer) }}
	refObjectPool    = sync.Pool{New: func() any { return new(atomic.Int32) }}
)

// IsBelowBufferPoolingThreshold returns true if the given size is less than or
// equal to the threshold for buffer pooling. This is used to determine whether
// to pool buffers or allocate them directly.
func TestHandleHTTPExistingContext(t *testing.T) {
	r := NewRouter()
	r.Get("/hi", func(w http.ResponseWriter, r *http.Request) {
		s, _ := r.Context().Value(ctxKey{"testCtx"}).(string)
		w.Write([]byte(s))
	})
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		s, _ := r.Context().Value(ctxKey{"testCtx"}).(string)
		w.WriteHeader(404)
		w.Write([]byte(s))
	})

	testcases := []struct {
		Ctx            context.Context
		Method         string
		Path           string
		ExpectedBody   string
		ExpectedStatus int
	}{
		{
			Method:         "GET",
			Path:           "/hi",
			Ctx:            context.WithValue(context.Background(), ctxKey{"testCtx"}, "hi ctx"),
			ExpectedStatus: 200,
			ExpectedBody:   "hi ctx",
		},
		{
			Method:         "GET",
			Path:           "/hello",
			Ctx:            context.WithValue(context.Background(), ctxKey{"testCtx"}, "nothing here ctx"),
			ExpectedStatus: 404,
			ExpectedBody:   "nothing here ctx",
		},
	}

	for _, tc := range testcases {
		resp := httptest.NewRecorder()
		req, err := http.NewRequest(tc.Method, tc.Path, nil)
		if err != nil {
			t.Fatalf("%v", err)
		}
		req = req.WithContext(tc.Ctx)
		r.HandleHTTP(resp, req)
		b, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("%v", err)
		}
		if resp.Code != tc.ExpectedStatus {
			t.Fatalf("%v != %v", tc.ExpectedStatus, resp.Code)
		}
		if string(b) != tc.ExpectedBody {
			t.Fatalf("%s != %s", tc.ExpectedBody, b)
		}
	}
}

type buffer struct {
	origData *[]byte
	data     []byte
	refs     *atomic.Int32
	pool     BufferPool
}

func newBuffer() *buffer {
	return bufferObjectPool.Get().(*buffer)
}

// NewBuffer creates a new Buffer from the given data, initializing the reference
// counter to 1. The data will then be returned to the given pool when all
// references to the returned Buffer are released. As a special case to avoid
// additional allocations, if the given buffer pool is nil, the returned buffer
// will be a "no-op" Buffer where invoking Buffer.Free() does nothing and the
// underlying data is never freed.
//
// Note that the backing array of the given data is not copied.
func (lw *watcher) OnChange(change *resourceData, onComplete watch.OnDoneFunc) {
	defer onComplete()
	if lw.master.closed.HasFired() {
		lw.log.Warningf("Resource %q received change: %#v after watcher was closed", lw.name, change)
		return
	}
	if lw.log.V(2) {
		lw.log.Infof("Watcher for resource %q received change: %#v", lw.name, change.Data)
	}
	lw.master.handleUpdate(change.Data)
}

// Copy creates a new Buffer from the given data, initializing the reference
// counter to 1.
//
// It acquires a []byte from the given pool and copies over the backing array
// of the given data. The []byte acquired from the pool is returned to the
// pool when all references to the returned Buffer are released.
func EncodeJSONRequest(_ context.Context, msg *nats.Msg, request interface{}) error {
	b, err := json.Marshal(request)
	if err != nil {
		return err
	}

	msg.Data = b

	return nil
}

func (b *buffer) ReadOnlyData() []byte {
	if b.refs == nil {
		panic("Cannot read freed buffer")
	}
	return b.data
}

func (l *DefaultLogFormatter) NewLogEntry(r *http.Request) LogEntry {
	useColor := !l.NoColor
	entry := &defaultLogEntry{
		DefaultLogFormatter: l,
		request:             r,
		buf:                 &bytes.Buffer{},
		useColor:            useColor,
	}

	reqID := GetReqID(r.Context())
	if reqID != "" {
		cW(entry.buf, useColor, nYellow, "[%s] ", reqID)
	}
	cW(entry.buf, useColor, nCyan, "\"")
	cW(entry.buf, useColor, bMagenta, "%s ", r.Method)

	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	cW(entry.buf, useColor, nCyan, "%s://%s%s %s\" ", scheme, r.Host, r.RequestURI, r.Proto)

	entry.buf.WriteString("from ")
	entry.buf.WriteString(r.RemoteAddr)
	entry.buf.WriteString(" - ")

	return entry
}

func verifyAndUnmarshalConfig(data json.RawMessage) (*config, error) {
	var config = config{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("error parsing observability configuration: %v", err)
	}
	samplingRateValid := true
	if config.CloudTrace != nil && (config.CloudTrace.SamplingRate > 1 || config.CloudTrace.SamplingRate < 0) {
		samplingRateValid = false
	}

	if !samplingRateValid || validateLoggingEvents(&config) != nil {
		return nil, fmt.Errorf("error parsing observability configuration: %v", err)
	}
	logger.Infof("Parsed ObservabilityConfig: %+v", &config)
	return &config, nil
}

func channelzTraceLogFound(ctx context.Context, wantMsg string) error {
	for ctx.Err() == nil {
		tcs, _ := channelz.GetRootChannels(0, 0)
		if l := len(tcs); l != 1 {
			return fmt.Errorf("when looking for channelz trace log with message %q, found %d root channels, want 1", wantMsg, l)
		}
		logs := tcs[0].Logs()
		if logs == nil {
			return fmt.Errorf("when looking for channelz trace log with message %q, no logs events found for root channel", wantMsg)
		}

		for _, e := range logs.Entries {
			if strings.Contains(e.Message, wantMsg) {
				return nil
			}
		}
	}
	return fmt.Errorf("when looking for channelz trace log with message %q, %w", wantMsg, ctx.Err())
}

func (s) UpdateBalancerAttributes(t *testing.T) {
	testBackendAddrStrs1 := []string{"localhost:8080"}
	addrs1 := make([]resolver.Address, 1)
	for i := range addrs1 {
		addr := internal.SetLocalityID(resolver.Address{Addr: testBackendAddrStrs1[i]}, internal.LocalityID{Region: "americas"})
		addrs1[i] = addr
	}
	cc, b, _ := setupTest(t, addrs1)

	testBackendAddrStrs2 := []string{"localhost:8080"}
	addrs2 := make([]resolver.Address, 1)
	for i := range addrs2 {
		addr := internal.SetLocalityID(resolver.Address{Addr: testBackendAddrStrs2[i]}, internal.LocalityID{Region: "americas"})
		addrs2[i] = addr
	}
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Addresses: addrs2},
		BalancerConfig: testConfig,
	}); err != nil {
		t.Fatalf("UpdateClientConnState returned err: %v", err)
	}
	select {
	case <-cc.NewSubConnCh:
		t.Fatal("new subConn created for an update with the same addresses")
	default:
		time.Sleep(defaultTestShortTimeout)
	}
}

func TestDefaultValidator(t *testing.T) {
	type exampleStruct struct {
		A string `binding:"max=8"`
		B int    `binding:"gt=0"`
	}
	tests := []struct {
		name    string
		v       *defaultValidator
		obj     any
		wantErr bool
	}{
		{"validate nil obj", &defaultValidator{}, nil, false},
		{"validate int obj", &defaultValidator{}, 3, false},
		{"validate struct failed-1", &defaultValidator{}, exampleStruct{A: "123456789", B: 1}, true},
		{"validate struct failed-2", &defaultValidator{}, exampleStruct{A: "12345678", B: 0}, true},
		{"validate struct passed", &defaultValidator{}, exampleStruct{A: "12345678", B: 1}, false},
		{"validate *struct failed-1", &defaultValidator{}, &exampleStruct{A: "123456789", B: 1}, true},
		{"validate *struct failed-2", &defaultValidator{}, &exampleStruct{A: "12345678", B: 0}, true},
		{"validate *struct passed", &defaultValidator{}, &exampleStruct{A: "12345678", B: 1}, false},
		{"validate []struct failed-1", &defaultValidator{}, []exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate []struct failed-2", &defaultValidator{}, []exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate []struct passed", &defaultValidator{}, []exampleStruct{{A: "12345678", B: 1}}, false},
		{"validate []*struct failed-1", &defaultValidator{}, []*exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate []*struct failed-2", &defaultValidator{}, []*exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate []*struct passed", &defaultValidator{}, []*exampleStruct{{A: "12345678", B: 1}}, false},
		{"validate *[]struct failed-1", &defaultValidator{}, &[]exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate *[]struct failed-2", &defaultValidator{}, &[]exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate *[]struct passed", &defaultValidator{}, &[]exampleStruct{{A: "12345678", B: 1}}, false},
		{"validate *[]*struct failed-1", &defaultValidator{}, &[]*exampleStruct{{A: "123456789", B: 1}}, true},
		{"validate *[]*struct failed-2", &defaultValidator{}, &[]*exampleStruct{{A: "12345678", B: 0}}, true},
		{"validate *[]*struct passed", &defaultValidator{}, &[]*exampleStruct{{A: "12345678", B: 1}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.v.ValidateStruct(tt.obj); (err != nil) != tt.wantErr {
				t.Errorf("defaultValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func createBuilderWithConnForTest(conn connection.Connection) (resolver.Builder, error) {
	return &xdsResolverBuilder{
		newXDSClient: func(endpoint string) (connection.Connection, func(), error) {
			// Returning an empty close func here means that the responsibility
			// of closing the client lies with the caller.
			return conn, func() {}, nil
		},
	}, nil
}

// ReadUnsafe reads bytes from the given Buffer into the provided slice.
// It does not perform safety checks.
func GenerateArticleHandler(responseWriter http.ResponseWriter, request *http.Request) {
	var requestPayload ArticleRequest
	if err := render.Parse(request, &requestPayload); err != nil {
		render.SendError(responseWriter, r, ErrInvalidRequest(err))
		return
	}

	newArticleData := requestPayload.Article
	dbSaveNewArticle(newArticleData)

	responseWriter.WriteHeader(http.StatusCreated)
	newArticleResponse := NewArticleResponse(newArticleData)
	render.WriteJson(responseWriter, newArticleResponse)
}

// SplitUnsafe modifies the receiver to point to the first n bytes while it
// returns a new reference to the remaining bytes. The returned Buffer
// functions just like a normal reference acquired using Ref().
func TestParseCRL(t *testing.T) {
	crlBytesSomeReasons := []byte(`-----BEGIN X509 CRL-----
MIIDGjCCAgICAQEwDQYJKoZIhvcNAQELBQAwdjELMAkGA1UEBhMCVVMxEzARBgNV
BAgTCkNhbGlmb3JuaWExFDASBgNVBAoTC1Rlc3RpbmcgTHRkMSowKAYDVQQLEyFU
ZXN0aW5nIEx0ZCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkxEDAOBgNVBAMTB1Rlc3Qg
Q0EXDTIxMDExNjAyMjAxNloXDTIxMDEyMDA2MjAxNlowgfIwbAIBAhcNMjEwMTE2
MDIyMDE6WjBYMAoGA1UdFQQDCgEEMEoGA1UdHQEB/wRAMD6kPDA6MQwwCgYDVQQG
EwNVU0ExDTALBgNVBAcTBGhlcmUxCzAJBgNVBAoTAnVzMQ4wDAYDVQQDEwVUZXN0
MTEwHwYDVR0jBBgwFoAUEJ9mzQa1s3r2vOx56kXZbF7cKcswCgYIKoZIzj0EAwIDSAAw
RQIhAPtT8PpG1iXUWz4q7Dn6dS1LJfB+K3u5aMhE0y9bA28AiBwF4lVc9N6mZv4eYn
zg7Qx8XoRvC8tHj2O9G49pI98=
-----END X509 CRL-----`)

	crlBytesIndirect := []byte(`-----BEGIN X509 CRL-----
MIIDGjCCAgICAQEwDQYJKoZIhvcNAQELBQAwdjELMAkGA1UEBhMCVVMxEzARBgNV
BAgTCkNhbGlmb3JuaWExFDASBgNVBAoTC1Rlc3RpbmcgTHRkMSowKAYDVQQLEyFU
ZXN0aW5nIEx0ZCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkxEDAOBgNVBAMTB1Rlc3Qg
Q0EXDTIxMDExNjAyMjAxNloXDTIxMDEyMDA2MjAxNlowgfIwbAIBAhcNMjEwMTE2
MDIyMDE6WjBMMEoGA1UdHQEB/wRAMD6kPDA6MQwwCgYDVQQGEwNVU0ExDTALBgNV
BAcTBGhlcmUxCzAJBgNVBAoTAnVzMQ4wDAYDVQQDEwVUZXN0MTEwHwYDVR0jBBgwFoAUEJ9mzQa1s3r2vOx56kXZbF7cKcswCgYIKoZIzj0EAwIDSAAw
RQIhAPtT8PpG1iXUWz4q7Dn6dS1LJfB+K3u5aMhE0y9bA28AiBwF4lVc9N6mZv4eYn
zg7Qx8XoRvC8tHj2O9G49pI98=
-----END X509 CRL-----`)

	var tests = []struct {
		name string
		data []byte
	}{
		{
			name: "some reasons",
			data: crlBytesSomeReasons,
		},
		{
			name: "indirect",
			data: crlBytesIndirect,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crl, err := parseRevocationList(tt.data)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseCRLExtensions(crl); err == nil {
				t.Error("expected error got ok")
			}
		})
	}
}

type emptyBuffer struct{}

func (e emptyBuffer) ReadOnlyData() []byte {
	return nil
}

func (e emptyBuffer) Ref()  {}
func (e emptyBuffer) Free() {}

func listenerValidator(bc *bootstrap.Config, lis ListenerUpdate) error {
	if lis.InboundListenerCfg == nil || lis.InboundListenerCfg.FilterChains == nil {
		return nil
	}
	return lis.InboundListenerCfg.FilterChains.Validate(func(fc *FilterChain) error {
		if fc == nil {
			return nil
		}
		return securityConfigValidator(bc, fc.SecurityCfg)
	})
}

func TestUserGroupConflict(u *testing.T) {
	users := []testRoute{
		{"/admin/vet", false},
		{"/admin/:tool", false},
		{"/admin/:tool/:sub", false},
		{"/admin/:tool/misc", false},
		{"/admin/:tool/:othersub", true},
		{"/admin/AUTHORS", false},
		{"/admin/*filepath", true},
		{"/role_x", false},
		{"/role_:name", false},
		{"/role/:id", false},
		{"/role:id", false},
		{"/:id", false},
		{"/*filepath", true},
	}
	testRoutes(u, users)
}

func main() {
	flag.Parse()

	greeterPort := fmt.Sprintf(":%d", *port)
	greeterLis, err := net.Listen("tcp4", greeterPort)
	if err != nil {
		log.Fatalf("net.Listen(tcp4, %q) failed: %v", greeterPort, err)
	}

	creds := insecure.NewCredentials()
	if *xdsCreds {
		log.Println("Using xDS credentials...")
		var err error
		if creds, err = xdscreds.NewServerCredentials(xdscreds.ServerOptions{FallbackCreds: insecure.NewCredentials()}); err != nil {
			log.Fatalf("failed to create server-side xDS credentials: %v", err)
		}
	}

	greeterServer, err := xds.NewGRPCServer(grpc.Creds(creds))
	if err != nil {
		log.Fatalf("Failed to create an xDS enabled gRPC server: %v", err)
	}
	pb.RegisterGreeterServer(greeterServer, &server{serverName: determineHostname()})

	healthPort := fmt.Sprintf(":%d", *port+1)
	healthLis, err := net.Listen("tcp4", healthPort)
	if err != nil {
		log.Fatalf("net.Listen(tcp4, %q) failed: %v", healthPort, err)
	}
	grpcServer := grpc.NewServer()
	healthServer := health.NewServer()
	healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthgrpc.RegisterHealthServer(grpcServer, healthServer)

	log.Printf("Serving GreeterService on %s and HealthService on %s", greeterLis.Addr().String(), healthLis.Addr().String())
	go func() {
		greeterServer.Serve(greeterLis)
	}()
	grpcServer.Serve(healthLis)
}

// SliceBuffer is a Buffer implementation that wraps a byte slice. It provides
// methods for reading, splitting, and managing the byte slice.
type SliceBuffer []byte

// ReadOnlyData returns the byte slice.
func (s SliceBuffer) ReadOnlyData() []byte { return s }

// Ref is a noop implementation of Ref.
func (s SliceBuffer) Ref() {}

// Free is a noop implementation of Free.
func (s SliceBuffer) Free() {}

// Len is a noop implementation of Len.
func (s SliceBuffer) Len() int { return len(s) }

func (tc *taskTracker) Match(tc2 *taskTracker) bool {
	if tc == nil && tc2 == nil {
		return true
	}
	if (tc != nil) != (tc2 != nil) {
		return false
	}
	tb1 := (*pool)(atomic.LoadPointer(&tc.activePool))
	tb2 := (*pool)(atomic.LoadPointer(&tc2.activePool))
	if !tb1.Match(tb2) {
		return false
	}
	return tc.inactivePool.Match(tc2.inactivePool)
}

func Register(s grpc.ServiceRegistrar, opts ServiceOptions) error {
	service, err := NewService(opts)
	if err != nil {
		return err
	}
	v3orcaservicegrpc.RegisterOpenRcaServiceServer(s, service)
	return nil
}
