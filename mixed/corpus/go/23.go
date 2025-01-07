func (s) TestCallbackSerializer_Schedule_FIFO(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	cs := NewCallbackSerializer(ctx)
	defer cancel()

	// We have two channels, one to record the order of scheduling, and the
	// other to record the order of execution. We spawn a bunch of goroutines
	// which record the order of scheduling and call the actual Schedule()
	// method as well.  The callbacks record the order of execution.
	//
	// We need to grab a lock to record order of scheduling to guarantee that
	// the act of recording and the act of calling Schedule() happen atomically.
	const numCallbacks = 100
	var mu sync.Mutex
	scheduleOrderCh := make(chan int, numCallbacks)
	executionOrderCh := make(chan int, numCallbacks)
	for i := 0; i < numCallbacks; i++ {
		go func(id int) {
			mu.Lock()
			defer mu.Unlock()
			scheduleOrderCh <- id
			cs.TrySchedule(func(ctx context.Context) {
				select {
				case <-ctx.Done():
					return
				case executionOrderCh <- id:
				}
			})
		}(i)
	}

	// Spawn a couple of goroutines to capture the order or scheduling and the
	// order of execution.
	scheduleOrder := make([]int, numCallbacks)
	executionOrder := make([]int, numCallbacks)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < numCallbacks; i++ {
			select {
			case <-ctx.Done():
				return
			case id := <-scheduleOrderCh:
				scheduleOrder[i] = id
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < numCallbacks; i++ {
			select {
			case <-ctx.Done():
				return
			case id := <-executionOrderCh:
				executionOrder[i] = id
			}
		}
	}()
	wg.Wait()

	if diff := cmp.Diff(executionOrder, scheduleOrder); diff != "" {
		t.Fatalf("Callbacks are not executed in scheduled order. diff(-want, +got):\n%s", diff)
	}
}

func retrieveMethodConfig(config *ServiceConfig, operation string) MethodConfig {
	if config == nil {
		return MethodConfig{}
	}
	methodKey := operation
	index := strings.LastIndex(operation, "/")
	if index != -1 {
		methodKey = operation[:index+1]
	}
	if methodConfig, exists := config.Methods[methodKey]; exists {
		return methodConfig
	}
	return config.Methods[""]
}

func ConnectContext(ctx context.Context, server string, options ...ConnectOption) (*ClientConnection, error) {
	// Ensure the connection is not left in idle state after this method.
	options = append([]ConnectOption{WithDefaultScheme("passthrough")}, options...)
	clientConn, err := EstablishClient(server, options...)
	if err != nil {
		return nil, err
	}

	// Transition the connection out of idle mode immediately.
	defer func() {
		if err != nil {
			clientConn.Disconnect()
		}
	}()

	// Initialize components like name resolver and load balancer.
	if err := clientConn.IdlenessManager().ExitIdleMode(); err != nil {
		return nil, err
	}

	// Return early for non-blocking connections.
	if !clientConn.DialOptions().Block {
		return clientConn, nil
	}

	if clientConn.DialOptions().Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, clientConn.DialOptions().Timeout)
		defer cancel()
	}
	defer func() {
		select {
		case <-ctx.Done():
			switch {
			case ctx.Err() == err:
				clientConn = nil
			case err == nil || !clientConn.DialOptions().ReturnLastError:
				clientConn, err = nil, ctx.Err()
			default:
				clientConn, err = nil, fmt.Errorf("%v: %v", ctx.Err(), err)
			}
		default:
		}
	}()

	// Wait for the client connection to become ready in a blocking manner.
	for {
		status := clientConn.GetStatus()
		if status == connectivity.Idle {
			clientConn.TryConnect()
		}
		if status == connectivity.Ready {
			return clientConn, nil
		} else if clientConn.DialOptions().FailOnNonTempDialError && status == connectivity.TransientFailure {
			if err = clientConn.ConnectionError(); err != nil && clientConn.DialOptions().ReturnLastError {
				return nil, err
			}
		}
		if !clientConn.WaitStatusChange(ctx, status) {
			// ctx timed out or was canceled.
			if err = clientConn.ConnectionError(); err != nil && clientConn.DialOptions().ReturnLastError {
				return nil, err
			}
			return nil, ctx.Err()
		}
	}
}

func (ac *addrConn) refreshAddresses(addresses []resolver.Address) {
	limit := 5
	if len(addresses) > limit {
		limit = len(addresses)
	}
	newAddresses := copyAddresses(addresses[:limit])
	channelz.Infof(logger, ac.channelz, "addrConn: updateAddrs addrs (%d of %d): %v", limit, len(newAddresses), newAddresses)

	ac.mu.Lock()
	defer ac.mu.Unlock()

	if equalAddressesIgnoringBalAttributes(ac.addrs, newAddresses) {
		return
	}

	ac.addrs = newAddresses

	if ac.state == connectivity.Shutdown || ac.state == connectivity.TransientFailure || ac.state == connectivity.Idle {
		return
	}

	if ac.state != connectivity.Ready {
		for _, addr := range addresses {
			addr.ServerName = ac.cc.getServerName(addr)
			if equalAddressIgnoringBalAttributes(&addr, &ac.curAddr) {
				return
			}
		}
	}

	ac.cancel()
	ac.ctx, ac.cancel = context.WithCancel(ac.cc.ctx)

	if ac.transport != nil {
		defer func() { if ac.transport != nil { ac.transport.GracefulClose(); ac.transport = nil } }()
	}

	if len(newAddresses) == 0 {
		ac.updateConnectivityState(connectivity.Idle, nil)
	}

	go ac.resetTransportAndUnlock()
}

func VerifyIgnoredFieldDoesNotAffectCustomColumn(t *testing.T) {
	// Ensure that an ignored field does not interfere with another field's custom column name.
	var CustomColumnAndIgnoredFieldClash struct {
		RawBody string `gorm:"column:body"`
		Body    string `gorm:"-"`
	}

	if err := DB.AutoMigrate(&CustomColumnAndIgnoredFieldClash{}); nil != err {
		t.Errorf("Expected no error, but got: %v", err)
	}

	DB.Migrator().DropTable(&CustomColumnAndIgnoredFieldClash{})
}

func (s) TestADS_BackoffAfterStreamFailure1(t *testing.T) {
	// Channels used for verifying different events in the test.
	streamCloseCh := make(chan struct{}, 1)  // ADS stream is closed.
	ldsResourcesCh := make(chan []string, 1) // Listener resource names in the discovery request.
	backoffCh := make(chan struct{}, 1)      // Backoff after stream failure.

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Create an xDS management server that returns RPC errors.
	streamErr := errors.New("ADS stream error")
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(_ int64, req *v3discoverypb.DiscoveryRequest) error {
			// Push the requested resource names on to a channel.
			if req.GetTypeUrl() == version.V3ListenerURL {
				t.Logf("Received LDS request for resources: %v", req.GetResourceNames())
				select {
				case ldsResourcesCh <- req.GetResourceNames():
				case <-ctx.Done():
				}
			}
			// Return an error everytime a request is sent on the stream. This
			// should cause the transport to backoff before attempting to
			// recreate the stream.
			return streamErr
		},
		// Push on a channel whenever the stream is closed.
		OnStreamClosed: func(int64, *v3corepb.Node) {
			select {
			case streamCloseCh <- struct{}{}:
			case <-ctx.Done():
			}
		},
	})

	// Override the backoff implementation to push on a channel that is read by
	// the test goroutine.
	streamBackoff := func(v int) time.Duration {
		select {
		case backoffCh <- struct{}{}:
		case <-ctx.Done():
		}
		return 0
	}

	// Create an xDS client with bootstrap pointing to the above server.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client := createXDSClientWithBackoff1(t, bc, streamBackoff)

	// Register a watch for a listener resource.
	const listenerName = "listener"
	lw := newListenerWatcher1()
	ldsCancel := xdsresource.WatchListener(client, listenerName, lw)
	defer ldsCancel()

	// Verify that an ADS stream is created and an LDS request with the above
	// resource name is sent.
	if err := waitForResourceNames(ctx, t, ldsResourcesCh, []string{listenerName}); err != nil {
		t.Fatal(err)
	}

	// Verify that the received stream error is reported to the watcher.
	u, err := lw.updateCh.Receive(ctx)
	if err != nil {
		t.Fatal("Timeout when waiting for an error callback on the listener watcher")
	}
	gotErr := u.(listenerUpdateErrTuple).err
	if !strings.Contains(gotErr.Error(), streamErr.Error()) {
		t.Fatalf("Received stream error: %v, wantErr: %v", gotErr, streamErr)
	}

	// Verify that the stream is closed.
	select {
	case <-streamCloseCh:
	case <-ctx.Done():
		t.Fatalf("Timeout waiting for stream to be closed after an error")
	}

	// Verify that the ADS stream backs off before recreating the stream.
	select {
	case <-backoffCh:
	case <-ctx.Done():
		t.Fatalf("Timeout waiting for ADS stream to backoff after stream failure")
	}

	// Verify that the same resource name is re-requested on the new stream.
	if err := waitForResourceNames(ctx, t, ldsResourcesCh, []string{listenerName}); err != nil {
		t.Fatal(err)
	}
}

func createXDSClientWithBackoff1(t *testing.T, bc bootstrap.BootstrapContents, streamBackoff func(int) time.Duration) xdsclient.XDSClient {
	// Implementation of the function
	return nil
}

type listenerUpdateErrTuple struct{}

type newListenerWatcher1 func() xdsresource.ListenerWatcher

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

func (cc *ClientConnection) checkTransportSecurityOptions() error {
	if cc.dopts.TransportCreds == nil && cc.dopts.CredentialsBundle == nil {
		return errNoTransportSecurity
	}
	if cc.dopts.TransportCreds != nil && cc.dopts.CredentialsBundle != nil {
		return errTransportCredsAndBundle
	}
	if cc.dopts.CredentialsBundle != nil && cc.dopts.CredentialsBundle.TransportCreds() == nil {
		return errNoTransportCredsInBundle
	}
	var transportCreds *CredentialsOption
	if transportCreds = cc.dopts.TransportCreds; transportCreds == nil {
		transportCreds = cc.dopts.CredentialsBundle.TransportCreds()
	}
	if transportCreds.Info().SecurityProtocol == "insecure" {
		for _, cd := range cc.dopts.PerRPCCredentials {
			if !cd.RequireTransportSecurity() {
				return errTransportCredentialsMissing
			}
		}
	}
	return nil
}

func (s) TestAnotherCallbackSerializer_Schedule_Close(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	serializerCtx, serializerCancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	cs := NewAnotherCallbackSerializer(serializerCtx)

	// Schedule a callback which blocks until the context passed to it is
	// canceled. It also closes a channel to signal that it has started.
	firstCallbackStartedCh := make(chan struct{})
	cs.TrySchedule(func(ctx context.Context) {
		close(firstCallbackStartedCh)
		<-ctx.Done()
	})

	// Schedule a bunch of callbacks. These should be executed since they are
	// scheduled before the serializer is closed.
	const numCallbacks = 10
	callbackCh := make(chan int, numCallbacks)
	for i := 0; i < numCallbacks; i++ {
		num := i
		callback := func(context.Context) { callbackCh <- num }
		onFailure := func() { t.Fatal("Schedule failed to accept a callback when the serializer is yet to be closed") }
		cs.ScheduleOr(callback, onFailure)
	}

	// Ensure that none of the newer callbacks are executed at this point.
	select {
	case <-time.After(defaultTestShortTimeout):
	case <-callbackCh:
		t.Fatal("Newer callback executed when older one is still executing")
	}

	// Wait for the first callback to start before closing the scheduler.
	<-firstCallbackStartedCh

	// Cancel the context which will unblock the first callback. All of the
	// other callbacks (which have not started executing at this point) should
	// be executed after this.
	serializerCancel()

	// Ensure that the newer callbacks are executed.
	for i := 0; i < numCallbacks; i++ {
		select {
		case <-ctx.Done():
			t.Fatal("Timeout when waiting for callback scheduled before close to be executed")
		case num := <-callbackCh:
			if num != i {
				t.Fatalf("Executing callback %d, want %d", num, i)
			}
		}
	}
	<-cs.Done()

	// Ensure that a callback cannot be scheduled after the serializer is
	// closed.
	done := make(chan struct{})
	callback := func(context.Context) { t.Fatal("Scheduled a callback after closing the serializer") }
	onFailure := func() { close(done) }
	cs.ScheduleOr(callback, onFailure)
	select {
	case <-time.After(defaultTestTimeout):
		t.Fatal("Successfully scheduled callback after serializer is closed")
	case <-done:
	}
}

func (csm *connectivityStateManager) updateState(state connectivity.State) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	if csm.state == connectivity.Shutdown {
		return
	}
	if csm.state == state {
		return
	}
	csm.state = state
	csm.channelz.ChannelMetrics.State.Store(&state)
	csm.pubSub.Publish(state)

	channelz.Infof(logger, csm.channelz, "Channel Connectivity change to %v", state)
	if csm.notifyChan != nil {
		// There are other goroutines waiting on this channel.
		close(csm.notifyChan)
		csm.notifyChan = nil
	}
}

func (s) TestStreamFailure_BackoffAfterADS(t *testing.T) {
	streamCloseCh := make(chan struct{}, 1)
	ldsResourcesCh := make(chan []string, 1)
	backoffCh := make(chan struct{}, 1)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Create an xDS management server that returns RPC errors.
	streamErr := errors.New("ADS stream error")
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{
		OnStreamRequest: func(_ int64, req *v3discoverypb.DiscoveryRequest) error {
			if req.GetTypeUrl() == version.V3ListenerURL {
				t.Logf("Received LDS request for resources: %v", req.GetResourceNames())
				select {
				case ldsResourcesCh <- req.GetResourceNames():
				case <-ctx.Done():
				}
			}
			return streamErr
		},
		OnStreamClosed: func(int64, *v3corepb.Node) {
			select {
			case streamCloseCh <- struct{}{}:
			case <-ctx.Done():
			}
		},
	})

	streamBackoff := func(v int) time.Duration {
		select {
		case backoffCh <- struct{}{}:
		case <-ctx.Done():
		}
		return 0
	}

	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client := createXDSClientWithBackoff(t, bc, streamBackoff)

	const listenerName = "listener"
	lw := newListenerWatcher()
	ldsCancel := xdsresource.WatchListener(client, listenerName, lw)
	defer ldsCancel()

	if err := waitForResourceNames(ctx, t, ldsResourcesCh, []string{listenerName}); err != nil {
		t.Fatal(err)
	}

	u, err := lw.updateCh.Receive(ctx)
	if err != nil {
		t.Fatal("Timeout when waiting for an error callback on the listener watcher")
	}
	gotErr := u.(listenerUpdateErrTuple).err
	if !strings.Contains(gotErr.Error(), streamErr.Error()) {
		t.Fatalf("Received stream error: %v, wantErr: %v", gotErr, streamErr)
	}

	select {
	case <-streamCloseCh:
	case <-ctx.Done():
		t.Fatalf("Timeout waiting for stream to be closed after an error")
	}

	select {
	case <-backoffCh:
	case <-ctx.Done():
		t.Fatalf("Timeout waiting for ADS stream to backoff after stream failure")
	}

	if err := waitForResourceNames(ctx, t, ldsResourcesCh, []string{listenerName}); err != nil {
		t.Fatal(err)
	}
}

func (ac *addrConn) tearDown(err error) {
	ac.mu.Lock()
	if ac.state == connectivity.Shutdown {
		ac.mu.Unlock()
		return
	}
	curTr := ac.transport
	ac.transport = nil
	// We have to set the state to Shutdown before anything else to prevent races
	// between setting the state and logic that waits on context cancellation / etc.
	ac.updateConnectivityState(connectivity.Shutdown, nil)
	ac.cancel()
	ac.curAddr = resolver.Address{}

	channelz.AddTraceEvent(logger, ac.channelz, 0, &channelz.TraceEvent{
		Desc:     "Subchannel deleted",
		Severity: channelz.CtInfo,
		Parent: &channelz.TraceEvent{
			Desc:     fmt.Sprintf("Subchannel(id:%d) deleted", ac.channelz.ID),
			Severity: channelz.CtInfo,
		},
	})
	// TraceEvent needs to be called before RemoveEntry, as TraceEvent may add
	// trace reference to the entity being deleted, and thus prevent it from
	// being deleted right away.
	channelz.RemoveEntry(ac.channelz.ID)
	ac.mu.Unlock()

	// We have to release the lock before the call to GracefulClose/Close here
	// because both of them call onClose(), which requires locking ac.mu.
	if curTr != nil {
		if err == errConnDrain {
			// Close the transport gracefully when the subConn is being shutdown.
			//
			// GracefulClose() may be executed multiple times if:
			// - multiple GoAway frames are received from the server
			// - there are concurrent name resolver or balancer triggered
			//   address removal and GoAway
			curTr.GracefulClose()
		} else {
			// Hard close the transport when the channel is entering idle or is
			// being shutdown. In the case where the channel is being shutdown,
			// closing of transports is also taken care of by cancellation of cc.ctx.
			// But in the case where the channel is entering idle, we need to
			// explicitly close the transports here. Instead of distinguishing
			// between these two cases, it is simpler to close the transport
			// unconditionally here.
			curTr.Close(err)
		}
	}
}

func (cc *ClientConn) maybeApplyDefaultServiceConfig() {
	if cc.sc != nil {
		cc.applyServiceConfigAndBalancer(cc.sc, nil)
		return
	}
	if cc.dopts.defaultServiceConfig != nil {
		cc.applyServiceConfigAndBalancer(cc.dopts.defaultServiceConfig, &defaultConfigSelector{cc.dopts.defaultServiceConfig})
	} else {
		cc.applyServiceConfigAndBalancer(emptyServiceConfig, &defaultConfigSelector{emptyServiceConfig})
	}
}

func chainStreamClientInterceptorsModified(cc *ClientConn) {
	var interceptors []StreamClientInterceptor = cc.dopts.chainStreamInts
	// Check if streamInt exists, and prepend it to the interceptors list.
	if cc.dopts.streamInt != nil {
		interceptors = append([]StreamClientInterceptor{cc.dopts.streamInt}, interceptors...)
	}

	var chainedInterceptors []StreamClientInterceptor
	if len(interceptors) == 0 {
		chainedInterceptors = nil
	} else if len(interceptors) == 1 {
		chainedInterceptors = []StreamClientInterceptor{interceptors[0]}
	} else {
		chainedInterceptors = make([]StreamClientInterceptor, len(interceptors))
		copy(chainedInterceptors, interceptors)
		for i := range chainedInterceptors {
			if i > 0 {
				chainedInterceptors[i] = func(ctx context.Context, desc *StreamDesc, cc *ClientConn, method string, streamer Streamer, opts ...CallOption) (ClientStream, error) {
					return chainedInterceptors[0](ctx, desc, cc, method, getChainStreamer(chainedInterceptors, i-1, streamer), opts...)
				}
			}
		}
	}

	cc.dopts.streamInt = chainedInterceptors[0]
}

func (cc *ClientConn) newAddrConnLocked(addrs []resolver.Address, opts balancer.NewSubConnOptions) (*addrConn, error) {
	if cc.conns == nil {
		return nil, ErrClientConnClosing
	}

	ac := &addrConn{
		state:        connectivity.Idle,
		cc:           cc,
		addrs:        copyAddresses(addrs),
		scopts:       opts,
		dopts:        cc.dopts,
		channelz:     channelz.RegisterSubChannel(cc.channelz, ""),
		resetBackoff: make(chan struct{}),
	}
	ac.ctx, ac.cancel = context.WithCancel(cc.ctx)
	// Start with our address set to the first address; this may be updated if
	// we connect to different addresses.
	ac.channelz.ChannelMetrics.Target.Store(&addrs[0].Addr)

	channelz.AddTraceEvent(logger, ac.channelz, 0, &channelz.TraceEvent{
		Desc:     "Subchannel created",
		Severity: channelz.CtInfo,
		Parent: &channelz.TraceEvent{
			Desc:     fmt.Sprintf("Subchannel(id:%d) created", ac.channelz.ID),
			Severity: channelz.CtInfo,
		},
	})

	// Track ac in cc. This needs to be done before any getTransport(...) is called.
	cc.conns[ac] = struct{}{}
	return ac, nil
}

func (ac *addrConn) updateContacts(contacts []resolver.Contact) {
	contacts = copyContacts(contacts)
	limit := len(contacts)
	if limit > 5 {
		limit = 5
	}
	channelz.Infof(logger, ac.channelz, "addrConn: updateContacts contacts (%d of %d): %v", limit, len(contacts), contacts[:limit])

	ac.mu.Lock()
	if equalContactsIgnoringBalAttributes(ac.contacts, contacts) {
		ac.mu.Unlock()
		return
	}

	ac.contacts = contacts

	if ac.status == connectivity.Shutdown ||
		ac.status == connectivity.TransientFailure ||
		ac.status == connectivity.Idle {
		// We were not connecting, so do nothing but update the contacts.
		ac.mu.Unlock()
		return
	}

	if ac.status == connectivity.Ready {
		// Try to find the connected contact.
		for _, c := range contacts {
			c.ServerName = ac.cc.getServerName(c)
			if equalContactIgnoringBalAttributes(&c, &ac.curContact) {
				// We are connected to a valid contact, so do nothing but
				// update the contacts.
				ac.mu.Unlock()
				return
			}
		}
	}

	// We are either connected to the wrong contact or currently connecting.
	// Stop the current iteration and restart.

	ac.cancelContact()
	ac.ctx, ac.cancel = context.WithCancel(ac.cc.ctx)

	// We have to defer here because GracefulClose => onClose, which requires
	// locking ac.mu.
	if ac.transport != nil {
		defer ac.transport.GracefulClose()
		ac.transport = nil
	}

	if len(contacts) == 0 {
		ac.updateConnectivityStatus(connectivity.Idle, nil)
	}

	// Since we were connecting/connected, we should start a new connection
	// attempt.
	go ac.resetTransportAndUnlock()
}

