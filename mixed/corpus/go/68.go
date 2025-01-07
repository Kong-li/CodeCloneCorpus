func TestServerUnregisteredMethod(t *testing.T) {
	ecm := jsonrpc.EndpointCodecMap{}
	handler := jsonrpc.NewServer(ecm)
	server := httptest.NewServer(handler)
	defer server.Close()
	resp, _ := http.Post(server.URL, "application/json", addBody())
	if want, have := http.StatusOK, resp.StatusCode; want != have {
		t.Errorf("want %d, have %d", want, have)
	}
	buf, _ := ioutil.ReadAll(resp.Body)
	expectErrorCode(t, jsonrpc.MethodNotFoundError, buf)
}

func processServiceEndpoints(endpoints []*v3endpointpb.LbEndpoint, addrMap map[string]bool) ([]Endpoint, error) {
	var processedEndpoints []Endpoint
	for _, endpoint := range endpoints {
		weight := uint32(1)
		if lbWeight := endpoint.GetLoadBalancingWeight(); lbWeight != nil && lbWeight.GetValue() > 0 {
			weight = lbWeight.GetValue()
		}
		addrs := append([]string{parseAddress(endpoint.GetEndpoint().GetAddress().GetSocketAddress())}, parseAdditionalAddresses(endpoint)...)

		for _, addr := range addrs {
			if addrMap[addr] {
				return nil, fmt.Errorf("duplicate endpoint with the same address %s", addr)
			}
			addrMap[addr] = true
		}

		processedEndpoints = append(processedEndpoints, Endpoint{
			HealthStatus: EndpointHealthStatus(endpoint.GetHealthStatus()),
			Addresses:    addrs,
			Weight:       weight,
		})
	}
	return processedEndpoints, nil
}

func parseAdditionalAddresses(lbEndpoint *v3endpointpb.LbEndpoint) []string {
	var addresses []string
	if envconfig.XDSDualstackEndpointsEnabled {
		for _, sa := range lbEndpoint.GetEndpoint().GetAdditionalAddresses() {
			addresses = append(addresses, parseAddress(sa.GetAddress().GetSocketAddress()))
		}
	}
	return addresses
}

func TestSingleServerAfter(t *testing.T) {
	var completion = make(chan struct{})
	ecm := jsonrpc.EndpointCodecMap{
		"multiply": jsonrpc.EndpointCodec{
			Endpoint: endpoint.Nop,
			Decode:   nopDecoder,
			Encode:   nopEncoder,
		},
	}
	handler := jsonrpc.NewServer(
		ecm,
		jsonrpc.ServerAfter(func(ctx context.Context, w http.ResponseWriter) context.Context {
			ctx = context.WithValue(ctx, "two", 2)

			return ctx
		}),
		jsonrpc.ServerAfter(func(ctx context.Context, w http.ResponseWriter) context.Context {
			if _, ok := ctx.Value("two").(int); !ok {
				t.Error("Value was not set properly when multiple ServerAfters are used")
			}

			close(completion)
			return ctx
		}),
	)
	server := httptest.NewServer(handler)
	defer server.Close()
	http.Post(server.URL, "application/json", multiplyBody()) // nolint

	select {
	case <-completion:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for finalizer")
	}
}

func TestCanFinalize(t *testing.T) {
	var done = make(chan struct{})
	var finalizerCalled bool
	ecm := jsonrpc.EndpointCodecMap{
		"add": jsonrpc.EndpointCodec{
			Endpoint: endpoint.Nop,
			Decode:   nopDecoder,
			Encode:   nopEncoder,
		},
	}
	handler := jsonrpc.NewServer(
		ecm,
		jsonrpc.ServerFinalizer(func(ctx context.Context, code int, req *http.Request) {
			finalizerCalled = true
			close(done)
		}),
	)
	server := httptest.NewServer(handler)
	defer server.Close()
	http.Post(server.URL, "application/json", addBody()) // nolint

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for finalizer")
	}

	if !finalizerCalled {
		t.Fatal("Finalizer was not called.")
	}
}

func (s) TestParsedTarget_WithCustomDialer(t *testing.T) {
	resetInitialResolverState()
	defScheme := resolver.GetDefaultScheme()
	tests := []struct {
		target            string
		wantParsed        resolver.Target
		wantDialerAddress string
	}{
		// unix:[local_path], unix:[/absolute], and unix://[/absolute] have
		// different behaviors with a custom dialer.
		{
			target:            "unix:a/b/c",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("unix:a/b/c")},
			wantDialerAddress: "unix:a/b/c",
		},
		{
			target:            "unix:/a/b/c",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("unix:/a/b/c")},
			wantDialerAddress: "unix:///a/b/c",
		},
		{
			target:            "unix:///a/b/c",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("unix:///a/b/c")},
			wantDialerAddress: "unix:///a/b/c",
		},
		{
			target:            "dns:///127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("dns:///127.0.0.1:50051")},
			wantDialerAddress: "127.0.0.1:50051",
		},
		{
			target:            ":///127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, ":///127.0.0.1:50051"))},
			wantDialerAddress: ":///127.0.0.1:50051",
		},
		{
			target:            "dns://authority/127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("dns://authority/127.0.0.1:50051")},
			wantDialerAddress: "127.0.0.1:50051",
		},
		{
			target:            "://authority/127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, "://authority/127.0.0.1:50051"))},
			wantDialerAddress: "://authority/127.0.0.1:50051",
		},
		{
			target:            "/unix/socket/address",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, "/unix/socket/address"))},
			wantDialerAddress: "/unix/socket/address",
		},
		{
			target:            "",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, ""))},
			wantDialerAddress: "",
		},
		{
			target:            "passthrough://a.server.com/google.com",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("passthrough://a.server.com/google.com")},
			wantDialerAddress: "google.com",
		},
	}

	for _, test := range tests {
		t.Run(test.target, func(t *testing.T) {
			addrCh := make(chan string, 1)
			dialer := func(_ context.Context, address string) (net.Conn, error) {
				addrCh <- address
				return nil, errors.New("dialer error")
			}

			cc, err := Dial(test.target, WithTransportCredentials(insecure.NewCredentials()), WithContextDialer(dialer))
			if err != nil {
				t.Fatalf("Dial(%q) failed: %v", test.target, err)
			}
			defer cc.Close()

			select {
			case addr := <-addrCh:
				if addr != test.wantDialerAddress {
					t.Fatalf("address in custom dialer is %q, want %q", addr, test.wantDialerAddress)
				}
			case <-time.After(time.Second):
				t.Fatal("timeout when waiting for custom dialer to be invoked")
			}
			if !cmp.Equal(cc.parsedTarget, test.wantParsed) {
				t.Errorf("cc.parsedTarget for dial target %q = %+v, want %+v", test.target, cc.parsedTarget, test.wantParsed)
			}
		})
	}
}

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

func (l *loopyWriter) cleanupStreamHandler(c *cleanupStream) error {
	c.onWrite()
	if str, ok := l.estdStreams[c.streamID]; ok {
		// On the server side it could be a trailers-only response or
		// a RST_STREAM before stream initialization thus the stream might
		// not be established yet.
		delete(l.estdStreams, c.streamID)
		str.deleteSelf()
		for head := str.itl.dequeueAll(); head != nil; head = head.next {
			if df, ok := head.it.(*dataFrame); ok {
				_ = df.reader.Close()
			}
		}
	}
	if c.rst { // If RST_STREAM needs to be sent.
		if err := l.framer.fr.WriteRSTStream(c.streamID, c.rstCode); err != nil {
			return err
		}
	}
	if l.draining && len(l.estdStreams) == 0 {
		// Flush and close the connection; we are done with it.
		return errors.New("finished processing active streams while in draining mode")
	}
	return nil
}

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

func convertDropPolicyToConfig(dropPolicy *v3endpointpb.ClusterLoadAssignment_Policy_DropOverload) OverloadDropConfig {
	dropPercentage := dropPolicy.GetDropPercentage()
	overloadDropConfig := OverloadDropConfig{}

	switch dropPercentage.GetDenominator() {
	case v3typepb.FractionalPercent_HUNDRED:
		overloadDropConfig.Denominator = 100
	case v3typepb.FractionalPercent_TEN_THOUSAND:
		overloadDropConfig.Denominator = 10000
	case v3typepb.FractionalPercent_MILLION:
		overloadDropConfig.Denominator = 1000000
	default:
		overloadDropConfig.Denominator = 100 // 默认值
	}

	overloadDropConfig.Numerator = dropPercentage.GetNumerator()
	overloadDropConfig.Category = dropPolicy.GetCategory()

	return overloadDropConfig
}

