func (hi *HandshakeInfo) validateSAN(san string, isDNS bool) bool {
	for _, matcher := range hi.sanMatchers {
		if matcher.ExactMatch() != "" && !isDNS { // 变量位置和布尔值取反
			continue
		}
		if dnsMatch(matcher.ExactMatch(), san) { // 内联并修改变量名
			return true
		}
		if matcher.Match(san) {
			return true
		}
	}
	return false
}

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

func (s) TestTLS_DisabledALPNServer(t *testing.T) {
	defaultVal := envconfig.EnforceALPNEnabled
	defer func() {
		envconfig.EnforceALPNEnabled = defaultVal
	}()

	testCases := []struct {
		caseName     string
		isALPENforced bool
		expectedErr  bool
	}{
		{
			caseName:     "enforced",
			isALPENforced: true,
			expectedErr:  true,
		},
		{
			caseName: "not_enforced",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseName, func(t *testing.T) {
			envconfig.EnforceALPNEnabled = testCase.isALPENforced

			listener, err := net.Listen("tcp", "localhost:0")
			if err != nil {
				t.Fatalf("Error starting server: %v", err)
			}
			defer listener.Close()

			errCh := make(chan error, 1)
			go func() {
				conn, err := listener.Accept()
				if err != nil {
					errCh <- fmt.Errorf("listener.Accept returned error: %v", err)
					return
				}
				defer conn.Close()
				serverConfig := tls.Config{
					Certificates: []tls.Certificate{serverCert},
					NextProtos:   []string{"h2"},
				}
				_, _, err = credentials.NewTLS(&serverConfig).ServerHandshake(conn)
				if gotErr := (err != nil); gotErr != testCase.expectedErr {
					t.Errorf("ServerHandshake returned unexpected error: got=%v, want=%t", err, testCase.expectedErr)
				}
				close(errCh)
			}()

			serverAddr := listener.Addr().String()
			clientConfig := &tls.Config{
				Certificates: []tls.Certificate{serverCert},
				NextProtos:   []string{}, // Empty list indicates ALPN is disabled.
				RootCAs:      certPool,
				ServerName:   serverName,
			}
			conn, err = tls.Dial("tcp", serverAddr, clientConfig)
			if err != nil {
				t.Fatalf("tls.Dial(%s) failed: %v", serverAddr, err)
			}
			defer conn.Close()

			select {
			case <-time.After(defaultTestTimeout):
				t.Fatal("Timed out waiting for completion")
			case err := <-errCh:
				if err != nil {
					t.Fatalf("Unexpected server error: %v", err)
				}
			}
		})
	}
}

func (c *ClusterClient) cmdsInfo(ctx context.Context) (map[string]*CommandInfo, error) {
	// Try 3 random nodes.
	const nodeLimit = 3

	addrs, err := c.nodes.Addrs()
	if err != nil {
		return nil, err
	}

	var firstErr error

	perm := rand.Perm(len(addrs))
	if len(perm) > nodeLimit {
		perm = perm[:nodeLimit]
	}

	for _, idx := range perm {
		addr := addrs[idx]

		node, err := c.nodes.GetOrCreate(addr)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}

		info, err := node.Client.Command(ctx).Result()
		if err == nil {
			return info, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}

	if firstErr == nil {
		panic("not reached")
	}
	return nil, firstErr
}

func (c *ClusterClient) handleTxPipeline(ctx context.Context, commands []Commander) error {
	// Trim multi .. exec.
	commands = commands[1 : len(commands)-1]

	state, err := c.state.Fetch(ctx)
	if err != nil {
		setCmdsErr(commands, err)
		return err
	}

	cmdMap := c.mapCommandsBySlot(ctx, commands)
	for slot, cmds := range cmdMap {
		node, err := state.slotMasterNode(slot)
		if err != nil {
			setCmdsErr(cmds, err)
			continue
		}

		cmdMap := map[*clusterNode][]Commander{node: cmds}
		for attempt := 0; attempt <= c.opt.MaxRetries; attempt++ {
			if attempt > 0 {
				if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
					setCmdsErr(commands, err)
					return err
				}
			}

			failedCmds := newCmdMap()
			var wg sync.WaitGroup

			for node, cmds := range cmdMap {
				wg.Add(1)
				go func(node *clusterNode, cmds []Commander) {
					defer wg.Done()
					c.handleTxPipelineNode(ctx, node, cmds, failedCmds)
				}(node, cmds)
			}

			wg.Wait()
			if len(failedCmds.m) == 0 {
				break
			}
			cmdMap = failedCmds.m
		}
	}

	return commandsFirstErr(commands)
}

func (s) TestPickFirstLeaf_HealthListenerEnabled(t *testing.T) {
	defer func() { _ = t.Cleanup() }()
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	bf := stub.BalancerFuncs{
		Close:   func(bd *stub.BalancerData) { bd.Data.(balancer.Balancer).Close(); closeClientConnStateUpdate(cc, bd) },
		Init:    func(bd *stub.BalancerData) { bd.Data = balancer.Get(pickfirstleaf.Name).Build(bd.ClientConn, bd.BuildOptions); initBalancerData(bd) },
		UpdateClientConnState: func(bd *stub.BalancerData, ccs balancer.ClientConnState) error {
			ccs.ResolverState = pickfirstleaf.EnableHealthListener(ccs.ResolverState)
			return bd.Data.(balancer.Balancer).UpdateClientConnState(ccs)
		},
	}

	stub.Register(t.Name(), bf)
	svcCfg := fmt.Sprintf(`{ "loadBalancingConfig": [{%q: {}}] }`, t.Name())
	backend := stubserver.StartTestService(t, nil)
	defer backend.Stop()
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(svcCfg),
	}
	cc, err := grpc.NewClient(backend.Address, opts...)
	if err != nil {
		t.Fatalf("grpc.NewClient(%q) failed: %v", backend.Address, err)
		defer cc.Close()
	}

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, resolver.Address{Addr: backend.Address}); err != nil {
		t.Fatal(err)
	}
}

// 新增函数
func initBalancerData(bd *stub.BalancerData) {
	bd.ClientConn = nil
}

func closeClientConnStateUpdate(cc *clientConn, bd *stub.BalancerData) {
	cc.ClientConnState = nil
}

func (s) TestPickFirstLeaf_InterleavingIPv4Preferred(t *testing.T) {
	defer func() { _ = context.Cancel(context.Background()) }()
	ctx := context.WithTimeout(context.Background(), defaultTestTimeout)
	cc := testutils.NewBalancerClientConn(t)
	balancer.Get(pickfirstleaf.Name).Build(cc, balancer.BuildOptions{MetricsRecorder: &stats.NoopMetricsRecorder{}}).Close()
	ccState := resolver.State{
		Endpoints: []resolver.Endpoint{
			{Addresses: []resolver.Address{{Addr: "1.1.1.1:1111"}}},
			{Addresses: []resolver.Address{{Addr: "2.2.2.2:2"}}},
			{Addresses: []resolver.Address{{Addr: "3.3.3.3:3"}}},
			{Addresses: []resolver.Address{{Addr: "[::FFFF:192.168.0.1]:2222"}}},
			{Addresses: []resolver.Address{{Addr: "[0001:0001:0001:0001:0001:0001:0001:0001]:8080"}}},
			{Addresses: []resolver.Address{{Addr: "[0002:0002:0002:0002:0002:0002:0002:0002]:8080"}}},
			{Addresses: []resolver.Address{{Addr: "[fe80::1%eth0]:3333"}}},
			{Addresses: []resolver.Address{{Addr: "grpc.io:80"}}}, // not an IP.
		},
	}
	if err := balancer.ClientConnState.Set(bal.UpdateClientConnState(ccState)); err != nil {
		t.Fatalf("UpdateClientConnState(%v) returned error: %v", ccState, err)
	}

	wantAddrs := []resolver.Address{
		{Addr: "1.1.1.1:1111"},
		{Addr: "[0001:0001:0001:0001:0001:0001:0001:0001]:8080"},
		{Addr: "grpc.io:80"},
		{Addr: "2.2.2.2:2"},
		{Addr: "[0002:0002:0002:0002:0002:0002:0002:0002]:8080"},
		{Addr: "3.3.3.3:3"},
		{Addr: "[fe80::1%eth0]:3333"},
		{Addr: "[::FFFF:192.168.0.1]:2222"},
	}

	gotAddrs, err := subConnAddresses(ctx, cc, 8)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if diff := cmp.Diff(wantAddrs, gotAddrs, ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn creation order mismatch (-want +got):\n%s", diff)
	}
}

func (s) TestFirstPickLeaf_HealthCheckEnabled(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	balancerFuncs := stub.BalancerFuncs{
		Init: func(balancerData *stub.BalancerData) {
			balancerData.Data = balancer.Get(pickfirstleaf.Name).Build(balancerData.ClientConn, balancerData.BuildOptions)
		},
		Close: func(balancerData *stub.BalancerData) {
			if closer, ok := balancerData.Data.(io.Closer); ok {
				closer.Close()
			}
		},
		UpdateClientConnState: func(balancerData *stub.BalancerData, ccs balancer.ClientConnState) error {
			ccs.ResolverState = pickfirstleaf.EnableHealthListener(ccs.ResolverState)
			return balancerData.Data.(balancer.Balancer).UpdateClientConnState(ccs)
		},
	}

	stub.Register(t.Name(), balancerFuncs)
	serviceConfig := fmt.Sprintf(`{ "loadBalancingConfig": [{%q: {}}] }`, t.Name())
	testBackend := stubserver.StartTestService(t, nil)
	defer testBackend.Stop()
	dialOptions := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(serviceConfig),
	}
	testClientConn, err := grpc.NewClient(testBackend.Address, dialOptions...)
	if err != nil {
		t.Fatalf("grpc.NewClient(%q) failed: %v", testBackend.Address, err)
	}
	defer testClientConn.Close()

	err = pickfirst.CheckRPCsToBackend(ctx, testClientConn, resolver.Address{Addr: testBackend.Address})
	if err != nil {
		t.Fatal(err)
	}
}

func (s) TestPickFirstLeaf_SimpleResolverUpdate_FirstServerUnReady(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	balCh := make(chan *stateStoringBalancer, 1)
	balancer.Register(&stateStoringBalancerBuilder{balancer: balCh})

	cc, r, bm := setupPickFirstLeaf(t, 2, grpc.WithDefaultServiceConfig(stateStoringServiceConfig))
	addrs := bm.resolverAddrs()
	stateSubscriber := &ccStateSubscriber{}
	internal.SubscribeToConnectivityStateChanges.(func(cc *grpc.ClientConn, s grpcsync.Subscriber) func())(cc, stateSubscriber)
	bm.stopAllExcept(1)

	r.UpdateState(resolver.State{Addresses: addrs})
	var bal *stateStoringBalancer
	select {
	case bal = <-balCh:
	case <-ctx.Done():
		t.Fatal("Context expired while waiting for balancer to be built")
	}
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[1]); err != nil {
		t.Fatal(err)
	}

	wantSCStates := []scState{
		{Addrs: []resolver.Address{addrs[0]}, State: connectivity.Shutdown},
		{Addrs: []resolver.Address{addrs[1]}, State: connectivity.Ready},
	}
	if diff := cmp.Diff(wantSCStates, bal.subConnStates(), ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn states mismatch (-want +got):\n%s", diff)
	}

	wantConnStateTransitions := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
	}
	if diff := cmp.Diff(wantConnStateTransitions, stateSubscriber.transitions); diff != "" {
		t.Errorf("ClientConn states mismatch (-want +got):\n%s", diff)
	}
}

func (s) TestPickFirstLeaf_HaltConnectedServer_TargetServerRestart(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	balCh := make(chan *stateStoringBalancer, 1)
	balancer.Register(&stateStoringBalancerBuilder{balancer: balCh})
	cc, r, bm := setupPickFirstLeaf(t, 2, grpc.WithDefaultServiceConfig(stateStoringServiceConfig))
	addrs := bm.resolverAddrs()
	stateSubscriber := &ccStateSubscriber{}
	internal.SubscribeToConnectivityStateChanges.(func(cc *grpc.ClientConn, s grpcsync.Subscriber) func())(cc, stateSubscriber)

	// shutdown all active backends except the target.
	bm.stopAllExcept(0)

	r.UpdateState(resolver.State{Addresses: addrs})
	var bal *stateStoringBalancer
	select {
	case bal = <-balCh:
	case <-ctx.Done():
		t.Fatal("Context expired while waiting for balancer to be built")
	}
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	wantSCStates := []scState{
		{Addrs: []resolver.Address{addrs[0]}, State: connectivity.Ready},
	}

	if diff := cmp.Diff(wantSCStates, bal.subConnStates(), ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn states mismatch (-want +got):\n%s", diff)
	}

	// Shut down the connected server.
	bm.backends[0].halt()
	testutils.AwaitState(ctx, t, cc, connectivity.Idle)

	// Start the new target server.
	bm.backends[0].resume()

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff(wantSCStates, bal.subConnStates(), ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn states mismatch (-want +got):\n%s", diff)
	}

	wantConnStateTransitions := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
		connectivity.Idle,
		connectivity.Connecting,
		connectivity.Ready,
	}
	if diff := cmp.Diff(wantConnStateTransitions, stateSubscriber.transitions); diff != "" {
		t.Errorf("ClientConn states mismatch (-want +got):\n%s", diff)
	}
}

func (c *ClusterClient) processTxPipeline(ctx context.Context, cmds []Cmder) error {
	// Trim multi .. exec.
	cmds = cmds[1 : len(cmds)-1]

	state, err := c.state.Get(ctx)
	if err != nil {
		setCmdsErr(cmds, err)
		return err
	}

	cmdsMap := c.mapCmdsBySlot(ctx, cmds)
	for slot, cmds := range cmdsMap {
		node, err := state.slotMasterNode(slot)
		if err != nil {
			setCmdsErr(cmds, err)
			continue
		}

		cmdsMap := map[*clusterNode][]Cmder{node: cmds}
		for attempt := 0; attempt <= c.opt.MaxRedirects; attempt++ {
			if attempt > 0 {
				if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
					setCmdsErr(cmds, err)
					return err
				}
			}

			failedCmds := newCmdsMap()
			var wg sync.WaitGroup

			for node, cmds := range cmdsMap {
				wg.Add(1)
				go func(node *clusterNode, cmds []Cmder) {
					defer wg.Done()
					c.processTxPipelineNode(ctx, node, cmds, failedCmds)
				}(node, cmds)
			}

			wg.Wait()
			if len(failedCmds.m) == 0 {
				break
			}
			cmdsMap = failedCmds.m
		}
	}

	return cmdsFirstErr(cmds)
}

func (s) TestPickFirstLeaf_HappyEyeballs_TF_AfterEndOfList(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	originalTimer := pfinternal.TimeAfterFunc
	defer func() {
		pfinternal.TimeAfterFunc = originalTimer
	}()
	triggerTimer, timeAfter := mockTimer()
	pfinternal.TimeAfterFunc = timeAfter

	tmr := stats.NewTestMetricsRecorder()
	dialer := testutils.NewBlockingDialer()
	opts := []grpc.DialOption{
		grpc.WithDefaultServiceConfig(fmt.Sprintf(`{"loadBalancingConfig": [{"%s":{}}]}`, pickfirstleaf.Name)),
		grpc.WithContextDialer(dialer.DialContext),
		grpc.WithStatsHandler(tmr),
	}
	cc, rb, bm := setupPickFirstLeaf(t, 3, opts...)
	addrs := bm.resolverAddrs()
	holds := bm.holds(dialer)
	rb.UpdateState(resolver.State{Addresses: addrs})
	cc.Connect()

	testutils.AwaitState(ctx, t, cc, connectivity.Connecting)

	// Verify that only the first server is contacted.
	if holds[0].Wait(ctx) != true {
		t.Fatalf("Timeout waiting for server %d with address %q to be contacted", 0, addrs[0])
	}
	if holds[1].IsStarted() != false {
		t.Fatalf("Server %d with address %q contacted unexpectedly", 1, addrs[1])
	}
	if holds[2].IsStarted() != false {
		t.Fatalf("Server %d with address %q contacted unexpectedly", 2, addrs[2])
	}

	// Make the happy eyeballs timer fire once and verify that the
	// second server is contacted, but the third isn't.
	triggerTimer()
	if holds[1].Wait(ctx) != true {
		t.Fatalf("Timeout waiting for server %d with address %q to be contacted", 1, addrs[1])
	}
	if holds[2].IsStarted() != false {
		t.Fatalf("Server %d with address %q contacted unexpectedly", 2, addrs[2])
	}

	// Make the happy eyeballs timer fire once more and verify that the
	// third server is contacted.
	triggerTimer()
	if holds[2].Wait(ctx) != true {
		t.Fatalf("Timeout waiting for server %d with address %q to be contacted", 2, addrs[2])
	}

	// First SubConn Fails.
	holds[0].Fail(fmt.Errorf("test error"))
	tmr.WaitForInt64CountIncr(ctx, 1)

	// No TF should be reported until the first pass is complete.
	shortCtx, shortCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer shortCancel()
	testutils.AwaitNotState(shortCtx, t, cc, connectivity.TransientFailure)

	// Third SubConn fails.
	shortCtx, shortCancel = context.WithTimeout(ctx, defaultTestShortTimeout)
	defer shortCancel()
	holds[2].Fail(fmt.Errorf("test error"))
	tmr.WaitForInt64CountIncr(ctx, 1)
	testutils.AwaitNotState(shortCtx, t, cc, connectivity.TransientFailure)

	// Last SubConn fails, this should result in a TF update.
	holds[1].Fail(fmt.Errorf("test error"))
	tmr.WaitForInt64CountIncr(ctx, 1)
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Only connection attempt fails in this test.
	if got, _ := tmr.Metric("grpc.lb.pick_first.connection_attempts_succeeded"); got != 0 {
		t.Errorf("Unexpected data for metric %v, got: %v, want: %v", "grpc.lb.pick_first.connection_attempts_succeeded", got, 0)
	}
	if got, _ := tmr.Metric("grpc.lb.pick_first.connection_attempts_failed"); got != 1 {
		t.Errorf("Unexpected data for metric %v, got: %v, want: %v", "grpc.lb.pick_first.connection_attempts_failed", got, 1)
	}
	if got, _ := tmr.Metric("grpc.lb.pick_first.disconnections"); got != 0 {
		t.Errorf("Unexpected data for metric %v, got: %v, want: %v", "grpc.lb.pick_first.disconnections", got, 0)
	}
}

func (b *bdpEstimator) calculate(d [8]byte) {
	// Check if the ping acked for was the bdp ping.
	if bdpPing.data != d {
		return
	}
	b.mu.Lock()
	rttSample := time.Since(b.sentAt).Seconds()
	if b.sampleCount < 10 {
		// Bootstrap rtt with an average of first 10 rtt samples.
		b.rtt += (rttSample - b.rtt) / float64(b.sampleCount)
	} else {
		// Heed to the recent past more.
		b.rtt += (rttSample - b.rtt) * float64(alpha)
	}
	b.isSent = false
	// The number of bytes accumulated so far in the sample is smaller
	// than or equal to 1.5 times the real BDP on a saturated connection.
	bwCurrent := float64(b.sample) / (b.rtt * float64(1.5))
	if bwCurrent > b.bwMax {
		b.bwMax = bwCurrent
	}
	// If the current sample (which is smaller than or equal to the 1.5 times the real BDP) is
	// greater than or equal to 2/3rd our perceived bdp AND this is the maximum bandwidth seen so far, we
	// should update our perception of the network BDP.
	if float64(b.sample) >= beta*float64(b.bdp) && bwCurrent == b.bwMax && b.bdp != bdpLimit {
		sampleFloat := float64(b.sample)
		b.bdp = uint32(gamma * sampleFloat)
		if b.bdp > bdpLimit {
			b.bdp = bdpLimit
		}
		bdp := b.bdp
		b.mu.Unlock()
		b.updateFlowControl(bdp)
		return
	}
	b.mu.Unlock()
}

func (c *ClusterClient) cmdNode(
	ctx context.Context,
	cmdName string,
	slot int,
) (*clusterNode, error) {
	state, err := c.state.Get(ctx)
	if err != nil {
		return nil, err
	}

	if c.opt.ReadOnly {
		cmdInfo := c.cmdInfo(ctx, cmdName)
		if cmdInfo != nil && cmdInfo.ReadOnly {
			return c.slotReadOnlyNode(state, slot)
		}
	}
	return state.slotMasterNode(slot)
}

func (p *poolNodes) fetch(url string) (*poolNode, error) {
	var node *poolNode
	var err error
	p.mu.RLock()
	if p.closed {
		err = pool.ErrClosed
	} else {
		node = p.nodes[url]
	}
	p.mu.RUnlock()
	return node, err
}

func (s *serverState) slotBackupNode(slot int) (*serverNode, error) {
	hosts := s.slotServers(slot)
	switch len(hosts) {
	case 0:
		return s.servers.Random()
	case 1:
		return hosts[0], nil
	case 2:
		if backup := hosts[1]; !backup.Occupied() {
			return backup, nil
		}
		return hosts[0], nil
	default:
		var backup *serverNode
		for i := 0; i < 15; i++ {
			h := rand.Intn(len(hosts)-1) + 1
			backup = hosts[h]
			if !backup.Occupied() {
				return backup, nil
			}
		}

		// All backups are busy - use primary.
		return hosts[0], nil
	}
}

