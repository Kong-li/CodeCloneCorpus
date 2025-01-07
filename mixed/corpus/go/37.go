func (s *userAccountClient) AsyncLoginCall(ctx context.Context, req *AsyncLoginRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[AsyncLoginResponse], error) {
	sOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := s.cc.NewStream(ctx, &UserAccount_ServiceDesc.Streams[0], UserAccount_AsyncLoginCall_FullMethodName, sOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[AsyncLoginRequest, AsyncLoginResponse]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(req); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

func TestQueueSubscriptionHandler(t *testing.T) {
	natsConn, consumer := createNATSConnectionAndConsumer(t)
	defer natsConn.Shutdown()
	defer consumer.Close()

	var (
		replyMsg = []byte(`{"Body": "go eat a fly ugly\n"}`)
		wg       sync.WaitGroup
		done     chan struct{}
	)

	subscriber := natstransport.NewSubscriber(
		endpoint.Nop,
		func(ctx context.Context, msg *nats.Msg) (interface{}, error) {
			return nil, nil
		},
		func(ctx context.Context, reply string, nc *nats.Conn, _ interface{}) error {
			err := json.Unmarshal(replyMsg, &response)
			if err != nil {
				return err
			}
			return consumer.Publish(reply, []byte(response.Body))
		},
		natstransport.SubscriberAfter(func(ctx context.Context, nc *nats.Conn) context.Context {
			ctx = context.WithValue(ctx, "one", 1)
			return ctx
		}),
		natstransport.SubscriberAfter(func(ctx context.Context, nc *nats.Conn) context.Context {
			if val, ok := ctx.Value("one").(int); !ok || val != 1 {
				t.Error("Value was not set properly when multiple ServerAfters are used")
			}
			close(done)
			return ctx
		}),
	)

	subscription, err := consumer.QueueSubscribe("natstransport.test", "subscriber", subscriber.ServeMsg(consumer))
	if err != nil {
		t.Fatal(err)
	}
	defer subscription.Unsubscribe()

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err = consumer.Request("natstransport.test", []byte("test data"), 2*time.Second)
		if err != nil {
			t.Fatal(err)
		}
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for finalizer")
	}

	wg.Wait()
}

func ValidateJSONResponse(t *testing.T) {
	s, c := initializeNATSConnection(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	responseHandler := natstransport.NewSubscriber(
		func(_ context.Context, _ interface{}) (interface{}, error) {
			return struct {
				Foo string `json:"foo"`
			}{"bar"}, nil
		},
		func(context.Context, *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		natstransport.EncodeJSONResponse,
	)

	subscription, err := c.QueueSubscribe("natstrtrans.test", "test.sub", responseHandler.ServeMsg(c))
	if err != nil {
		t.Fatal(err)
	}
	defer subscription.Unsubscribe()

	requestMessage, err := c.Request("natstrtrans.test", []byte("test data"), 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	expectedResponse := `{"foo":"bar"}`
	actualResponse := strings.TrimSpace(string(requestMessage.Data))

	if expectedResponse != actualResponse {
		t.Errorf("Response: want %s, have %s", expectedResponse, actualResponse)
	}
}

func initializeNATSConnection(t *testing.T) (*server, *nats.Conn) {
	s := server{}
	c := nats.Conn{}
	return &s, &c
}

func TestEncodeJSONResponse(t *testing.T) {
	s, c := newNATSConn(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	handler := natstransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) {
			return struct {
				Foo string `json:"foo"`
			}{"bar"}, nil
		},
		func(context.Context, *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		natstransport.EncodeJSONResponse,
	)

	sub, err := c.QueueSubscribe("natstransport.test", "natstransport", handler.ServeMsg(c))
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Unsubscribe()

	r, err := c.Request("natstransport.test", []byte("test data"), 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := `{"foo":"bar"}`, strings.TrimSpace(string(r.Data)); want != have {
		t.Errorf("Body: want %s, have %s", want, have)
	}
}

func (fs *bufferedSink) ProcessLogEntry(entry *logpb.Entry) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if !fs.flusherInitialized {
		// Initiate the write process when LogEntry is processed.
		fs.beginFlushRoutine()
		fs.flusherInitialized = true
	}
	if err := fs.outStream.Send(entry); err != nil {
		return err
	}
	return nil
}

func evaluateServiceVersions(s1, s2 *newspb.ServiceVersion) int {
	switch {
	case s1.GetMajor() > s2.GetMajor(),
		s1.GetMajor() == s2.GetMajor() && s1.GetMinor() > s2.GetMinor():
		return 1
	case s1.GetMajor() < s2.GetMajor(),
		s1.GetMajor() == s2.GetMajor() && s1.GetMinor() < s2.GetMinor():
		return -1
	}
	return 0
}

func parseDialTarget(target string) (string, string) {
	net := "tcp"
	m1 := strings.Index(target, ":")
	m2 := strings.Index(target, ":/")
	// handle unix:addr which will fail with url.Parse
	if m1 >= 0 && m2 < 0 {
		if n := target[0:m1]; n == "unix" {
			return n, target[m1+1:]
		}
	}
	if m2 >= 0 {
		t, err := url.Parse(target)
		if err != nil {
			return net, target
		}
		scheme := t.Scheme
		addr := t.Path
		if scheme == "unix" {
			if addr == "" {
				addr = t.Host
			}
			return scheme, addr
		}
	}
	return net, target
}

func convertGrpcMessageUnchecked(content string) string {
	var result strings.Builder
	for ; len(content) > 0; content = content[utf8.RuneLen(r)-1:] {
		r := utf8.RuneAt([]byte(content), 0)
		if _, size := utf8.DecodeRuneInString(string(r)); size > 1 {
			result.WriteString(fmt.Sprintf("%%%02X", []byte(string(r))[0]))
			continue
		}
		for _, b := range []byte(string(r)) {
			if b >= ' ' && b <= '~' && b != '%' {
				result.WriteByte(b)
			} else {
				result.WriteString(fmt.Sprintf("%%%02X", b))
			}
		}
	}
	return result.String()
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

func TestSubscriberBadEndpoint(t *testing.T) {
	s, c := newNATSConn(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	handler := natstransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, errors.New("dang") },
		func(context.Context, *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, string, *nats.Conn, interface{}) error { return nil },
	)

	resp := testRequest(t, c, handler)

	if want, have := "dang", resp.Error; want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

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

