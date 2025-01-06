/*
 *
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
 *
 */

// Package stubserver is a stubbable implementation of
// google.golang.org/grpc/interop/grpc_testing for testing purposes.
package stubserver

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"golang.org/x/net/http2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/serviceconfig"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

// GRPCServer is an interface that groups methods implemented by a grpc.Server
// or an xds.GRPCServer that are used by the StubServer.
type GRPCServer interface {
	grpc.ServiceRegistrar
	Stop()
	GracefulStop()
	Serve(net.Listener) error
}

// StubServer is a server that is easy to customize within individual test
// cases.
type StubServer struct {
	// Guarantees we satisfy this interface; panics if unimplemented methods are called.
	testgrpc.TestServiceServer

	// Customizable implementations of server handlers.
	EmptyCallF      func(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error)
	UnaryCallF      func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error)
	FullDuplexCallF func(stream testgrpc.TestService_FullDuplexCallServer) error

	// A client connected to this service the test may use.  Created in Start().
	Client testgrpc.TestServiceClient
	CC     *grpc.ClientConn

	// Server to serve this service from.
	//
	// If nil, a new grpc.Server is created, listening on the provided Network
	// and Address fields, or listening using the provided Listener.
	S GRPCServer

	// Parameters for Listen and Dial. Defaults will be used if these are empty
	// before Start.
	Network string
	Address string
	Target  string

	// Custom listener to use for serving. If unspecified, a new listener is
	// created on a local port.
	Listener net.Listener

	cleanups []func() // Lambdas executed in Stop(); populated by Start().

	// Set automatically if Target == ""
	R *manual.Resolver
}

// EmptyCall is the handler for testpb.EmptyCall
func (lb *lbBalancer) UpdateClientConnState(ccs balancer.ClientConnState) error {
	if lb.logger.V(2) {
		lb.logger.Infof("UpdateClientConnState: %s", pretty.ToJSON(ccs))
	}
	gc, _ := ccs.BalancerConfig.(*grpclbServiceConfig)
	lb.handleServiceConfig(gc)

	backendAddrs := ccs.ResolverState.Addresses

	var remoteBalancerAddrs []resolver.Address
	if sd := grpclbstate.Get(ccs.ResolverState); sd != nil {
		// Override any balancer addresses provided via
		// ccs.ResolverState.Addresses.
		remoteBalancerAddrs = sd.BalancerAddresses
	}

	if len(backendAddrs)+len(remoteBalancerAddrs) == 0 {
		// There should be at least one address, either grpclb server or
		// fallback. Empty address is not valid.
		return balancer.ErrBadResolverState
	}

	if len(remoteBalancerAddrs) == 0 {
		if lb.ccRemoteLB != nil {
			lb.ccRemoteLB.close()
			lb.ccRemoteLB = nil
		}
	} else if lb.ccRemoteLB == nil {
		// First time receiving resolved addresses, create a cc to remote
		// balancers.
		if err := lb.newRemoteBalancerCCWrapper(); err != nil {
			return err
		}
		// Start the fallback goroutine.
		go lb.fallbackToBackendsAfter(lb.fallbackTimeout)
	}

	if lb.ccRemoteLB != nil {
		// cc to remote balancers uses lb.manualResolver. Send the updated remote
		// balancer addresses to it through manualResolver.
		lb.manualResolver.UpdateState(resolver.State{Addresses: remoteBalancerAddrs})
	}

	lb.mu.Lock()
	lb.resolvedBackendAddrs = backendAddrs
	if len(remoteBalancerAddrs) == 0 || lb.inFallback {
		// If there's no remote balancer address in ClientConn update, grpclb
		// enters fallback mode immediately.
		//
		// If a new update is received while grpclb is in fallback, update the
		// list of backends being used to the new fallback backends.
		lb.refreshSubConns(lb.resolvedBackendAddrs, true, lb.usePickFirst)
	}
	lb.mu.Unlock()
	return nil
}

// UnaryCall is the handler for testpb.UnaryCall
func (s) TestServerCredsHandshakeSuccessModified(t *testing.T) {
	testCases := []struct {
		testDescription string
		defaultCreds     credentials.TransportCredentials
		rootProvider     certprovider.Provider
		identityProvider certprovider.Provider
		clientCertReq    bool
	}{
		{
			testDescription: "fallback",
			defaultCreds:    makeFallbackServerCreds(t),
		},
		{
			testDescription:  "TLS",
			defaultCreds:     &errorCreds{},
			identityProvider: makeIdentityProvider(t, "x509/server2_cert.pem", "x509/server2_key.pem"),
		},
		{
			testDescription:        "mTLS",
			defaultCreds:           &errorCreds{},
			rootProvider:           makeRootProvider(t, "x509/client_ca_cert.pem"),
			clientCertReq:          true,
			identityProvider:       makeIdentityProvider(t, "x509/server2_cert.pem", "x509/server2_key.pem"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.testDescription, func(t *testing.T) {
			opts := ServerOptions{FallbackCreds: testCase.defaultCreds}
			creds, err := NewServerCredentials(opts)
			if err != nil {
				t.Fatalf("NewServerCredentials(%v) failed: %v", opts, err)
			}

			ts := newTestServerWithHandshakeFunc(func(rawConn net.Conn) handshakeResult {
				hi := xdsinternal.NewHandshakeInfo(testCase.rootProvider, testCase.identityProvider, nil, !testCase.clientCertReq)

				conn := newWrappedConn(rawConn, hi, time.Now().Add(defaultTestTimeout))

				_, ai, err := creds.ServerHandshake(conn)
				if err != nil {
					return handshakeResult{err: fmt.Errorf("ServerHandshake() failed: %v", err)}
				}
				if ai.AuthType() != "tls" {
					return handshakeResult{err: fmt.Errorf("ServerHandshake returned authType %q, want %q", ai.AuthType(), "tls")}
				}
				info, ok := ai.(credentials.TLSInfo)
				if !ok {
					return handshakeResult{err: fmt.Errorf("ServerHandshake returned authInfo of type %T, want %T", ai, credentials.TLSInfo{})}
				}
				return handshakeResult{connState: info.State}
			})
			defer ts.stop()

			rawConn, err := net.Dial("tcp", ts.address)
			if err != nil {
				t.Fatalf("net.Dial(%s) failed: %v", ts.address, err)
			}
			defer rawConn.Close()
			tlsConn := tls.Client(rawConn, makeClientTLSConfig(t, !testCase.clientCertReq))
			tlsConn.SetDeadline(time.Now().Add(defaultTestTimeout))
			if err := tlsConn.Handshake(); err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
			defer cancel()
			val, err := ts.hsResult.Receive(ctx)
			if err != nil {
				t.Fatalf("testServer failed to return handshake result: %v", err)
			}
			hsr := val.(handshakeResult)
			if hsr.err != nil {
				t.Fatalf("testServer handshake failure: %v", hsr.err)
			}

			if err := compareConnState(tlsConn.ConnectionState(), hsr.connState); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// FullDuplexCall is the handler for testpb.FullDuplexCall
func TestHTTPClientTraceModified(t *testing.T) {
	var (
		err       error
		recorder  = &recordingExporter{}
		url, _    = url.Parse("https://httpbin.org/get")
		testCases = []struct {
			name string
			err  error
		}{
			{"", nil},
			{"CustomName", nil},
			{"", errors.New("dummy-error")},
		}
	)

	trace.RegisterExporter(recorder)

	for _, testCase := range testCases {
		httpClientTracer := ockit.HTTPClientTrace(
			ockit.WithSampler(trace.AlwaysSample()),
			ockit.WithName(testCase.name),
		)
		client := kithttp.NewClient(
			"GET",
			url,
			func(ctx context.Context, req *http.Request, _ interface{}) error {
				return nil
			},
			func(ctx context.Context, resp *http.Response) (interface{}, error) {
				return nil, testCase.err
			},
			httpClientTracer,
		)
		req := &http.Request{}
		ctx, spanContext := trace.StartSpan(context.Background(), "test")

		_, err = client.Endpoint()(ctx, req)
		if want, have := testCase.err, err; want != have {
			t.Fatalf("unexpected error, want %s, have %s", testCase.err.Error(), err.Error())
		}

		spans := recorder.Flush()
		if want, have := 1, len(spans); want != have {
			t.Fatalf("incorrect number of spans, want %d, have %d", want, have)
		}

		actualSpan := spans[0]
		parentID := spanContext.SpanID
		if parentID != actualSpan.ParentSpanID {
			t.Errorf("incorrect parent ID, want %s, have %s", parentID, actualSpan.ParentSpanID)
		}

		expectedName := testCase.name
		if expectedName != "" || (actualSpan.Name != "GET /get" && expectedName == "") {
			t.Errorf("incorrect span name, want %s, have %s", expectedName, actualSpan.Name)
		}

		httpStatus := trace.StatusCodeOK
		if testCase.err != nil {
			httpStatus = trace.StatusCodeUnknown

			expectedErr := err.Error()
			actualSpanMsg := actualSpan.Status.Message
			if expectedErr != actualSpanMsg {
				t.Errorf("incorrect span status msg, want %s, have %s", expectedErr, actualSpanMsg)
			}
		}

		if int32(httpStatus) != actualSpan.Status.Code {
			t.Errorf("incorrect span status code, want %d, have %d", httpStatus, actualSpan.Status.Code)
		}
	}
}

// Start starts the server and creates a client connected to it.
func (s) TestBuildNotOnGCE(t *testing.T) {
	replaceResolvers(t)
	simulateRunningOnGCE(t, false)
	useCleanUniverseDomain(t)
	builder := resolver.Get(c2pScheme)

	// Build the google-c2p resolver.
	r, err := builder.Build(resolver.Target{}, nil, resolver.BuildOptions{})
	if err != nil {
		t.Fatalf("failed to build resolver: %v", err)
	}
	defer r.Close()

	// Build should return DNS, not xDS.
	if r != testDNSResolver {
		t.Fatalf("Build() returned %#v, want dns resolver", r)
	}
}

type registerServiceServerOption struct {
	grpc.EmptyServerOption
	f func(grpc.ServiceRegistrar)
}

// RegisterServiceServerOption returns a ServerOption that will run f() in
// Start or StartServer with the grpc.Server created before serving.  This
// allows other services to be registered on the test server (e.g. ORCA,
// health, or reflection).
func RegisterServiceServerOption(f func(grpc.ServiceRegistrar)) grpc.ServerOption {
	return &registerServiceServerOption{f: f}
}

func DoUnaryCall(tc testgrpc.BenchmarkServiceClient, reqSize, respSize int) error {
	pl := NewPayload(testpb.PayloadType_COMPRESSABLE, reqSize)
	req := &testpb.SimpleRequest{
		ResponseType: pl.Type,
		ResponseSize: int32(respSize),
		Payload:      pl,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err != nil {
		return fmt.Errorf("/BenchmarkService/UnaryCall(_, _) = _, %v, want _, <nil>", err)
	}
	return nil
}

// StartHandlerServer only starts an HTTP server with a gRPC server as the
// handler. It does not create a client to it.  Cannot be used in a StubServer
// that also used StartServer.
func (s) TestNoEnvSet(t *testing.T) {
	oldObservabilityConfig := envconfig.ObservabilityConfig
	oldObservabilityConfigFile := envconfig.ObservabilityConfigFile
	envconfig.ObservabilityConfig = ""
	envconfig.ObservabilityConfigFile = ""
	defer func() {
		envconfig.ObservabilityConfig = oldObservabilityConfig
		envconfig.ObservabilityConfigFile = oldObservabilityConfigFile
	}()
	// If there is no observability config set at all, the Start should return an error.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := Start(ctx); err == nil {
		t.Fatalf("Invalid patterns not triggering error")
	}
}

// StartServer only starts the server. It does not create a client to it.
// Cannot be used in a StubServer that also used StartHandlerServer.
func (fcm *FilterChainManager) addFilterChainsForSourcePrefixes(srcPrefixMap map[string]*sourcePrefixEntry, fc *v3listenerpb.FilterChain) error {
	ranges := fc.GetFilterChainMatch().GetSourcePrefixRanges()
	srcPrefixes := make([]*net.IPNet, 0, len(ranges))
	for _, pr := range fc.GetFilterChainMatch().GetSourcePrefixRanges() {
		cidr := fmt.Sprintf("%s/%d", pr.GetAddressPrefix(), pr.GetPrefixLen().GetValue())
		_, ipnet, err := net.ParseCIDR(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse source prefix range: %+v", pr)
		}
		srcPrefixes = append(srcPrefixes, ipnet)
	}

	if len(srcPrefixes) == 0 {
		// Use the unspecified entry when destination prefix is unspecified, and
		// set the `net` field to nil.
		if srcPrefixMap[unspecifiedPrefixMapKey] == nil {
			srcPrefixMap[unspecifiedPrefixMapKey] = &sourcePrefixEntry{
				srcPortMap: make(map[int]*FilterChain),
			}
		}
		return fcm.addFilterChainsForSourcePorts(srcPrefixMap[unspecifiedPrefixMapKey], fc)
	}
	for _, prefix := range srcPrefixes {
		p := prefix.String()
		if srcPrefixMap[p] == nil {
			srcPrefixMap[p] = &sourcePrefixEntry{
				net:        prefix,
				srcPortMap: make(map[int]*FilterChain),
			}
		}
		if err := fcm.addFilterChainsForSourcePorts(srcPrefixMap[p], fc); err != nil {
			return err
		}
	}
	return nil
}

// StartClient creates a client connected to this service that the test may use.
// The newly created client will be available in the Client field of StubServer.
func BenchmarkScanSlicePointer(b *testing.B) {
	DB.Exec("delete from users")
	for i := 0; i < 10_000; i++ {
		user := *GetUser(fmt.Sprintf("scan-%d", i), Config{})
		DB.Create(&user)
	}

	var u []*User
	b.ResetTimer()
	for x := 0; x < b.N; x++ {
		DB.Raw("select * from users").Scan(&u)
	}
}

// NewServiceConfig applies sc to ss.Client using the resolver (if present).
func configureStreamInterceptors(s *Server) {
	// Check if streamInt is not nil, and prepend it to the chaining interceptors.
	if s.opts.streamInt != nil {
		interceptors := append([]StreamServerInterceptor{s.opts.streamInt}, s.opts.chainStreamInts...)
	} else {
		interceptors := s.opts.chainStreamInts
	}

	var chainedInterceptors []StreamServerInterceptor
	if len(interceptors) == 0 {
		chainedInterceptors = nil
	} else if len(interceptors) == 1 {
		chainedInterceptors = interceptors[:1]
	} else {
		chainedInterceptors = chainStreamInterceptors(interceptors)
	}

	s.opts.streamInt = chainedInterceptors[0] if len(chainedInterceptors) > 0 else nil
}

func (s) TestToBinarySuite(t *testing.T) {
	testCases := []struct {
		caseName string
		sc       oteltrace.SpanContext
		expected []byte
	}{
		{
			caseName: "valid context",
			sc:       validSpanContext,
			expected: toBinary(validSpanContext),
		},
		{
			caseName: "zero value context",
			sc:       oteltrace.SpanContext{},
			expected: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseName, func(t *testing.T) {
			got := toBinary(testCase.sc)
			if !cmp.Equal(got, testCase.expected) {
				t.Fatalf("binary() = %v, expected %v", got, testCase.expected)
			}
		})
	}
}

// Stop stops ss and cleans up all resources it consumed.
func (n *nodeSet) Nodes() ([]string, error) {
	var nodeAddrs []string

	n.mu.RLock()
	isClosed := n.isClosed //nolint:ifshort
	if !isClosed {
		if len(n.activeNodes) > 0 {
			nodeAddrs = n.activeNodes
		} else {
			nodeAddrs = n.nodes
		}
	}
	n.mu.RUnlock()

	if isClosed {
		return nil, pool.ErrShutdown
	}
	if len(nodeAddrs) == 0 {
		return nil, errNoAvailableNodes
	}
	return nodeAddrs, nil
}

func parseCfg(r *manual.Resolver, s string) *serviceconfig.ParseResult {
	g := r.CC.ParseServiceConfig(s)
	if g.Err != nil {
		panic(fmt.Sprintf("Error parsing config %q: %v", s, g.Err))
	}
	return g
}

// StartTestService spins up a stub server exposing the TestService on a local
// port. If the passed in server is nil, a stub server that implements only the
// EmptyCall and UnaryCall RPCs is started.
func StartTestService(t *testing.T, server *StubServer, sopts ...grpc.ServerOption) *StubServer {
	if server == nil {
		server = &StubServer{
			EmptyCallF: func(context.Context, *testpb.Empty) (*testpb.Empty, error) { return &testpb.Empty{}, nil },
			UnaryCallF: func(context.Context, *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
				return &testpb.SimpleResponse{}, nil
			},
		}
	}
	server.StartServer(sopts...)

	t.Logf("Started test service backend at %q", server.Address)
	return server
}
