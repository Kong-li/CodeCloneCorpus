func ProcessBinaryQuery(mc servicecache.CacheClient, qSize, rSize int) error {
	bp := NewBody(testpb.BodyType_NON_COMPRESSIBLE, qSize)
	query := &testpb.ComplexQuery{
	响应类型: bp.Type,
	响应大小: int32(rSize),
	负载:      bp,
	}
	if _, err := mc.BinaryQuery(context.Background(), query); err != nil {
		return fmt.Errorf("/CacheService/BinaryQuery(_, _) = _, %v, want _, <nil>", err)
	}
	return nil
}

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

func (s) TestServerCredsProviderSwitch(t *testing.T) {
	opts := ServerOptions{FallbackCreds: &errorCreds{}}
	creds, err := NewServerCredentials(opts)
	if err != nil {
		t.Fatalf("NewServerCredentials(%v) failed: %v", opts, err)
	}

	// The first time the handshake function is invoked, it returns a
	// HandshakeInfo which is expected to fail. Further invocations return a
	// HandshakeInfo which is expected to succeed.
	cnt := 0
	// Create a test server which uses the xDS server credentials created above
	// to perform TLS handshake on incoming connections.
	ts := newTestServerWithHandshakeFunc(func(rawConn net.Conn) handshakeResult {
		cnt++
		var hi *xdsinternal.HandshakeInfo
		if cnt == 1 {
			// Create a HandshakeInfo which has a root provider which does not match
			// the certificate sent by the client.
			hi = xdsinternal.NewHandshakeInfo(makeRootProvider(t, "x509/server_ca_cert.pem"), makeIdentityProvider(t, "x509/client2_cert.pem", "x509/client2_key.pem"), nil, true)

			// Create a wrapped conn which can return the HandshakeInfo and
			// configured deadline to the xDS credentials' ServerHandshake()
			// method.
			conn := newWrappedConn(rawConn, hi, time.Now().Add(defaultTestTimeout))

			// ServerHandshake() on the xDS credentials is expected to fail.
			if _, _, err := creds.ServerHandshake(conn); err == nil {
				return handshakeResult{err: errors.New("ServerHandshake() succeeded when expected to fail")}
			}
			return handshakeResult{}
		}

		hi = xdsinternal.NewHandshakeInfo(makeRootProvider(t, "x509/client_ca_cert.pem"), makeIdentityProvider(t, "x509/server1_cert.pem", "x509/server1_key.pem"), nil, true)

		// Create a wrapped conn which can return the HandshakeInfo and
		// configured deadline to the xDS credentials' ServerHandshake()
		// method.
		conn := newWrappedConn(rawConn, hi, time.Now().Add(defaultTestTimeout))

		// Invoke the ServerHandshake() method on the xDS credentials
		// and make some sanity checks before pushing the result for
		// inspection by the main test body.
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

	for i := 0; i < 5; i++ {
		// Dial the test server, and trigger the TLS handshake.
		rawConn, err := net.Dial("tcp", ts.address)
		if err != nil {
			t.Fatalf("net.Dial(%s) failed: %v", ts.address, err)
		}
		defer rawConn.Close()
		tlsConn := tls.Client(rawConn, makeClientTLSConfig(t, true))
		tlsConn.SetDeadline(time.Now().Add(defaultTestTimeout))
		if err := tlsConn.Handshake(); err != nil {
			t.Fatal(err)
		}

		// Read the handshake result from the testServer which contains the
		// TLS connection state on the server-side and compare it with the
		// one received on the client-side.
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
		if i == 0 {
			// We expect the first handshake to fail. So, we skip checks which
			// compare connection state.
			continue
		}
		// AuthInfo contains a variety of information. We only verify a
		// subset here. This is the same subset which is verified in TLS
		// credentials tests.
		if err := compareConnState(tlsConn.ConnectionState(), hsr.connState); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRingShardingRebalanceLocked(t *testing.B) {
	var opts RingOptions = RingOptions{
		Addrs: make(map[string]string),
		// Disable heartbeat
		HeartbeatFrequency: time.Hour,
	}
	for i := 0; i < 100; i++ {
		k := fmt.Sprintf("shard%d", i)
		v := fmt.Sprintf(":63%02d", i)
		opts.Addrs[k] = v
	}

	ring, _ := NewRing(opts)
	defer ring.Close()

	for i := 0; i < b.N; i++ {
		ring.sharding.rebalanceLocked()
	}
	b.ResetTimer()
}

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

