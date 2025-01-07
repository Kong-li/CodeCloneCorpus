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

func TestIssue293(connManager *testing.T) {
	// The util/conn.Manager won't attempt to reconnect to the provided endpoint
	// if the endpoint is initially unavailable (e.g. dial tcp :8080:
	// getsockopt: connection refused). If the endpoint is up when
	// conn.NewManager is called and then goes down/up, it reconnects just fine.

	var (
		tickc  = make(chan time.Time)
		after  = func(d time.Duration) <-chan time.Time { return tickc }
		dialer = func(netw string, addr string) (net.Conn, error) {
			return nil, errors.New("fail")
		}
		mgr    = NewManager(dialer, "netw", "addr", after, log.NewNopLogger())
	)

	if conn := mgr.Take(); conn != nil {
		connManager.Fatal("first Take should have yielded nil conn, but didn't")
	}

	dialconn := &mockConn{}
	dialerr   = nil
	select {
	case tickc <- time.Now():
	default:
		connManager.Fatal("manager isn't listening for a tick, despite a failed dial")
	}

	if !within(time.Second, func() bool {
		return mgr.Take() != nil
	}) {
		connManager.Fatal("second Take should have yielded good conn, but didn't")
	}
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

