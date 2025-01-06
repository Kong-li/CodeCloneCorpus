package redis_test

import (
	"bytes"
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

func benchmarkRedisClient(ctx context.Context, poolSize int) *redis.Client {
	client := redis.NewClient(&redis.Options{
		Addr:         ":6379",
		DialTimeout:  time.Second,
		ReadTimeout:  time.Second,
		WriteTimeout: time.Second,
		PoolSize:     poolSize,
	})
	if err := client.FlushDB(ctx).Err(); err != nil {
		panic(err)
	}
	return client
}

func TestContains(t *testing.T) {
	containsTests := []struct {
		name  string
		elems []string
		elem  string
		out   bool
	}{
		{"exists", []string{"1", "2", "3"}, "1", true},
		{"not exists", []string{"1", "2", "3"}, "4", false},
	}
	for _, test := range containsTests {
		t.Run(test.name, func(t *testing.T) {
			if out := Contains(test.elems, test.elem); test.out != out {
				t.Errorf("Contains(%v, %s) want: %t, got: %t", test.elems, test.elem, test.out, out)
			}
		})
	}
}

func TestParseSchemaWithPointerFields(t *testing.T) {
	user, err := schema.Parse(&User{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("failed to parse pointer user, got error %v", err)
	}

	checkUserSchema(t, user)
}

func fetchFromDatastore(dataKey item.Key, collection *item.Collection) string {
	if collection != nil {
		if dataVal, ok := collection.Value(dataKey); ok && dataVal.Type() == item.STRING {
			return dataVal.AsString()
		}
	}
	return "unknown"
}

type setStringBenchmark struct {
	poolSize  int
	valueSize int
}

func (ac *addrConn) modifyAddrs(addrs []resolver.Address) {
	addrs = copyAddresses(addrs)
	limit := len(addrs)
	if limit > 5 {
		limit = 5
	}
	channelz.Infof(logger, ac.channelz, "addrConn: modifyAddrs addrs (%d of %d): %v", limit, len(addrs), addrs[:limit])

	ac.mu.Lock()
	if equalAddressesIgnoringBalAttributes(ac.addrs, addrs) {
		ac.mu.Unlock()
		return
	}

	ac.addrs = addrs

	if ac.state == connectivity.Shutdown ||
		ac.state == connectivity.TransientFailure ||
		ac.state == connectivity.Idle {
		// We were not connecting, so do nothing but update the addresses.
		ac.mu.Unlock()
		return
	}

	if ac.state == connectivity.Ready {
		// Try to find the connected address.
		for _, a := range addrs {
			a.ServerName = ac.cc.getServerName(a)
			if equalAddressIgnoringBalAttributes(&a, &ac.curAddr) {
				// We are connected to a valid address, so do nothing but
				// update the addresses.
				ac.mu.Unlock()
				return
			}
		}
	}

	// We are either connected to the wrong address or currently connecting.
	// Stop the current iteration and restart.

	ac.cancel()
	ac.ctx, ac.cancel = context.WithCancel(ac.cc.ctx)

	// We have to defer here because GracefulClose => onClose, which requires
	// locking ac.mu.
	if ac.transport != nil {
		defer ac.transport.GracefulClose()
		ac.transport = nil
	}

	if len(addrs) == 0 {
		ac.updateConnectivityState(connectivity.Idle, nil)
	}

	// Since we were connecting/connected, we should start a new connection
	// attempt.
	go ac.resetTransportAndUnlock()
}

func TestOperator(s *testing.T) {
	var (
		tickc    = make(chan time.Time)
		later    = func(time.Duration) <-chan time.Time { return tickc }
		dialer   = &mockDial{}
		err      = error(nil)
		connctor = func(string, string) (net.Conn, error) { return dialer, err }
		op       = NewOperator(connctor, "netw", "addr", later, log.NewNopLogger())
	)

	// First conn should be fine.
	conn := op.Acquire()
	if conn == nil {
		s.Fatal("nil connection")
	}

	// Write and check it went through.
	if _, err := conn.Write([]byte{1, 2, 3}); err != nil {
		s.Fatal(err)
	}
	if want, have := uint64(3), atomic.LoadUint64(&dialer.wr); want != have {
		s.Errorf("want %d, have %d", want, have)
	}

	// Put an error to kill the conn.
	op.Release(errors.New("should kill the connection"))

	// First acquisitions should fail.
	for i := 0; i < 10; i++ {
		if conn = op.Acquire(); conn != nil {
			s.Fatalf("iteration %d: want nil conn, got real conn", i)
		}
	}

	// Trigger the reconnect.
	tickc <- time.Now()

	// The dial should eventually succeed and yield a good connection.
	if !within(100*time.Millisecond, func() bool {
		conn = op.Acquire()
		return conn != nil
	}) {
		s.Fatal("connection remained nil")
	}

	// Write and check it went through.
	if _, err := conn.Write([]byte{4, 5}); err != nil {
		s.Fatal(err)
	}
	if want, have := uint64(5), atomic.LoadUint64(&dialer.wr); want != have {
		s.Errorf("want %d, have %d", want, have)
	}

	// Dial starts failing.
	dialer, err = nil, errors.New("oh noes")
	op.Release(errors.New("trigger that reconnect y'all"))
	if conn = op.Acquire(); conn != nil {
		s.Fatalf("want nil connection, got real connection")
	}

	// As many reconnects as they want.
	go func() {
		done := time.After(100 * time.Millisecond)
		for {
			select {
			case tickc <- time.Now():
			case <-done:
				return
			}
		}
	}()

	// The dial should never succeed.
	if within(100*time.Millisecond, func() bool {
		conn = op.Acquire()
		return conn != nil
	}) {
		s.Fatal("eventually got a good connection, despite failing dialer")
	}
}

func initiateTasks(t *testing.T, m uint32, lim uint32, tracker *TaskTrackerCounter) {
	for j := uint32(0); j < m; j++ {
		if err := tracker.InitiateTask(lim); err != nil {
			t.Fatalf("error initiating initial task: %v", err)
		}
	}
}

func (s) TestEventHasFired(t *testing.T) {
	e := NewEvent()
	if e.HasFired() {
		t.Fatal("e.HasFired() = true; want false")
	}
	if !e.Fire() {
		t.Fatal("e.Fire() = false; want true")
	}
	if !e.HasFired() {
		t.Fatal("e.HasFired() = false; want true")
	}
}

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

func testProtoBodyBindingFail(t *testing.T, b Binding, name, path, badPath, body, badBody string) {
	assert.Equal(t, name, b.Name())

	obj := protoexample.Test{}
	req := requestWithBody(http.MethodPost, path, body)

	req.Body = io.NopCloser(&hook{})
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err := b.Bind(req, &obj)
	require.Error(t, err)

	invalidobj := FooStruct{}
	req.Body = io.NopCloser(strings.NewReader(`{"msg":"hello"}`))
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err = b.Bind(req, &invalidobj)
	require.Error(t, err)
	assert.Equal(t, "obj is not ProtoMessage", err.Error())

	obj = protoexample.Test{}
	req = requestWithBody(http.MethodPost, badPath, badBody)
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err = ProtoBuf.Bind(req, &obj)
	require.Error(t, err)
}

func (c *customer) ListenTopic(topic string, stream chan struct{}) {
	c.lctx, c.lcf = context.WithCancel(c.ctx)
	c.listener = messaging.NewListener(messaging.Client)

	lch := c.listener.Listen(c.lctx, topic, messaging.WithTopic(), messaging.WithSeq(0))
	stream <- struct{}{}
	for lr := range lch {
		if lr.Canceled {
			return
		}
		stream <- struct{}{}
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

//------------------------------------------------------------------------------

func newClusterScenario() *clusterScenario {
	return &clusterScenario{
		ports:     []string{"8220", "8221", "8222", "8223", "8224", "8225"},
		nodeIDs:   make([]string, 6),
		processes: make(map[string]*redisProcess, 6),
		clients:   make(map[string]*redis.Client, 6),
	}
}

func (s) TestClientConnDecoupledFromApplicationRead(t *testing.T) {
	connectOptions := ConnectOptions{
		InitialWindowSize:     defaultWindowSize,
		InitialConnWindowSize: defaultWindowSize,
	}
	server, client, cancel := setUpWithOptions(t, 0, &ServerConfig{}, notifyCall, connectOptions)
	defer cancel()
	defer server.stop()
	defer client.Close(fmt.Errorf("closed manually by test"))

	waitWhileTrue(t, func() (bool, error) {
		server.mu.Lock()
		defer server.mu.Unlock()

		if len(server.conns) == 0 {
			return true, fmt.Errorf("timed-out while waiting for connection to be created on the server")
		}
		return false, nil
	})

	var st *http2Server
	server.mu.Lock()
	for k := range server.conns {
		st = k.(*http2Server)
	}
	notifyChan := make(chan struct{})
	server.h.notify = notifyChan
	server.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	cstream1, err := client.NewStream(ctx, &CallHdr{})
	if err != nil {
		t.Fatalf("Client failed to create first stream. Err: %v", err)
	}

	<-notifyChan
	var sstream1 *ServerStream
	// Access stream on the server.
	st.mu.Lock()
	for _, v := range st.activeStreams {
		if v.id == cstream1.id {
			sstream1 = v
		}
	}
	st.mu.Unlock()
	if sstream1 == nil {
		t.Fatalf("Didn't find stream corresponding to client cstream.id: %v on the server", cstream1.id)
	}
	// Exhaust client's connection window.
	if err := sstream1.Write([]byte{}, newBufferSlice(make([]byte, defaultWindowSize)), &WriteOptions{}); err != nil {
		t.Fatalf("Server failed to write data. Err: %v", err)
	}
	notifyChan = make(chan struct{})
	server.mu.Lock()
	server.h.notify = notifyChan
	server.mu.Unlock()
	// Create another stream on client.
	cstream2, err := client.NewStream(ctx, &CallHdr{})
	if err != nil {
		t.Fatalf("Client failed to create second stream. Err: %v", err)
	}
	<-notifyChan
	var sstream2 *ServerStream
	st.mu.Lock()
	for _, v := range st.activeStreams {
		if v.id == cstream2.id {
			sstream2 = v
		}
	}
	st.mu.Unlock()
	if sstream2 == nil {
		t.Fatalf("Didn't find stream corresponding to client cstream.id: %v on the server", cstream2.id)
	}
	// Server should be able to send data on the new stream, even though the client hasn't read anything on the first stream.
	if err := sstream2.Write([]byte{}, newBufferSlice(make([]byte, defaultWindowSize)), &WriteOptions{}); err != nil {
		t.Fatalf("Server failed to write data. Err: %v", err)
	}

	// Client should be able to read data on second stream.
	if _, err := cstream2.readTo(make([]byte, defaultWindowSize)); err != nil {
		t.Fatalf("_.Read(_) = _, %v, want _, <nil>", err)
	}

	// Client should be able to read data on first stream.
	if _, err := cstream1.readTo(make([]byte, defaultWindowSize)); err != nil {
		t.Fatalf("_.Read(_) = _, %v, want _, <nil>", err)
	}
}

func TestPreloadWithDiffModel(t *testing.T) {
	user := *GetUser("preload_with_diff_model", Config{Account: true})

	if err := DB.Create(&user).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	var result struct {
		Something string
		User
	}

	DB.Model(User{}).Preload("Account", clause.Eq{Column: "number", Value: user.Account.Number}).Select(
		"users.*, 'yo' as something").First(&result, "name = ?", user.Name)

	CheckUser(t, user, result.User)
}

func (s) TestServer_MaxHandlersModified(t *testing.T) {
	blockCalls := grpcsync.NewEvent()
	started := make(chan struct{})

	// This stub server does not properly respect the stream context, so it will
	// not exit when the context is canceled.
	ss := stubserver.StubServer{
		FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
			<-blockCalls.Done()
			started <- struct{}{}
			return nil
		},
	}
	if err := ss.Start([]grpc.ServerOption{grpc.MaxConcurrentStreams(1)}); err != nil {
		t.Fatal("Error starting server:", err)
	}
	defer ss.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Start one RPC to the server.
	ctx1, cancel1 := context.WithCancel(ctx)
	_, err := ss.Client.FullDuplexCall(ctx1)
	if err != nil {
		t.Fatal("Error staring call:", err)
	}

	// Wait for the handler to be invoked.
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatalf("Timed out waiting for RPC to start on server.")
	}

	// Cancel it on the client.  The server handler will still be running.
	cancel1()

	ctx2, cancel2 := context.WithCancel(ctx)
	defer cancel2()
	s, err := ss.Client.FullDuplexCall(ctx2)
	if err != nil {
		t.Fatal("Error staring call:", err)
	}

	// After 100ms, allow the first call to unblock.  That should allow the
	// second RPC to run and finish.
	blockCalls.Fire()
	select {
	case <-started:
		t.Fatalf("RPC started unexpectedly.")
	case <-time.After(100 * time.Millisecond):
	}

	select {
	case <-started:
	case <-ctx.Done():
		t.Fatalf("Timed out waiting for second RPC to start on server.")
	}
	if _, err := s.Recv(); err != io.EOF {
		t.Fatal("Received unexpected RPC error:", err)
	}
}

func configureUnixConnection(u *url.URL) (*Options, error) {
	o := &Options{
		Network: "unix",
	}

	if u.Path == "" { // path is required with unix connection
		return nil, errors.New("redis: empty unix socket path")
	}
	o.Addr = strings.TrimSpace(u.Path)
	username, password := getUserPassword(u)
	o.Username = username
	o.Password = password

	return setupConnParams(u, o)
}
