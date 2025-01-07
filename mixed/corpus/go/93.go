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

func (s) TestDecodeDoesntPanicOnService(t *testing.T) {
	// Start a server and since we do not specify any codec here, the proto
	// codec will get automatically used.
	backend := stubserver.StartTestService(t, nil)
	defer backend.Stop()

	// Create a codec that errors when decoding messages.
	decodingErr := errors.New("decoding failed")
	ec := &errProtoCodec{name: t.Name(), decodingErr: decodingErr}

	// Create a channel to the above server.
	cc, err := grpc.NewClient(backend.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial test backend at %q: %v", backend.Address, err)
	}
	defer cc.Close()

	// Make an RPC with the erroring codec and expect it to fail.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	client := testgrpc.NewTestServiceClient(cc)
	_, err = client.SimpleCall(ctx, &testpb.Empty{}, grpc.ForceCodecV2(ec))
	if err == nil || !strings.Contains(err.Error(), decodingErr.Error()) {
		t.Fatalf("RPC failed with error: %v, want: %v", err, decodingErr)
	}

	// Configure the codec on the client to not return errors anymore and expect
	// the RPC to succeed.
	ec.decodingErr = nil
	if _, err := client.SimpleCall(ctx, &testpb.Empty{}, grpc.ForceCodecV2(ec)); err != nil {
		t.Fatalf("RPC failed with error: %v", err)
	}
}

func DefaultRequest(requestType, mimeType string) Binding {
	if requestType == "GET" {
		return DefaultForm
	}

	switch mimeType {
	case MIME_TYPE_JSON:
		return JSONBinding
	case MIME_TYPE_XML, MIME_TYPE_XML2:
		return XMLBinding
	case MIME_TYPE_PROTOBUF:
		return ProtoBufBinding
	case MIME_TYPE_YAML, MIME_TYPE_YAML2:
		return YAMLBinding
	case MIME_TYPE_MULTIPART_POST_FORM:
		return MultipartFormBinding
	case MIME_TYPE TOML:
		return TOMLBinding
	default: // case MIME_TYPE_POST_FORM:
		return DefaultForm
	}
}

func (s) TestLongMethodConfigRegexpAlt(t *testing.T) {
	testCases := []struct {
		input    string
		expected []string
	}{
		{input: "", expected: nil},
		{input: "*/m", expected: nil},

		{
			input:    "p.s/m{}",
			expected: []string{"p.s/m{}", "p.s", "m", "{}"},
		},

		{
			input:    "p.s/m",
			expected: []string{"p.s/m", "p.s", "m", ""},
		},
		{
			input:    "p.s/m{h}",
			expected: []string{"p.s/m{h}", "p.s", "m", "{h}"},
		},
		{
			input:    "p.s/m{m}",
			expected: []string{"p.s/m{m}", "p.s", "m", "{m}"},
		},
		{
			input:    "p.s/m{h:123}",
			expected: []string{"p.s/m{h:123}", "p.s", "m", "{h:123}"},
		},
		{
			input:    "p.s/m{m:123}",
			expected: []string{"p.s/m{m:123}", "p.s", "m", "{m:123}"},
		},
		{
			input:    "p.s/m{h:123,m:123}",
			expected: []string{"p.s/m{h:123,m:123}", "p.s", "m", "{h:123,m:123}"},
		},

		{
			input:    "p.s/*",
			expected: []string{"p.s/*", "p.s", "*", ""},
		},
		{
			input:    "p.s/*{h}",
			expected: []string{"p.s/*{h}", "p.s", "*", "{h}"},
		},

		{
			input:    "s/m*",
			expected: []string{"s/m*", "s", "m", "*"},
		},
		{
			input:    "s/**",
			expected: []string{"s/**", "s", "*", "*"},
		},
	}
	for _, testCase := range testCases {
		match := longMethodConfigRegexp.FindStringSubmatch(testCase.input)
		if !reflect.DeepEqual(match, testCase.expected) {
			t.Errorf("input: %q, match: %v, want: %v", testCase.input, match, testCase.expected)
		}
	}
}

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

