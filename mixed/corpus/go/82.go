func (l *logger) setBlacklist(method string) error {
	if _, ok := l.config.Blacklist[method]; ok {
		return fmt.Errorf("conflicting blacklist rules for method %v found", method)
	}
	if _, ok := l.config.Methods[method]; ok {
		return fmt.Errorf("conflicting method rules for method %v found", method)
	}
	if l.config.Blacklist == nil {
		l.config.Blacklist = make(map[string]struct{})
	}
	l.config.Blacklist[method] = struct{}{}
	return nil
}

func (s) ExampleAsyncHandling(c *testing.C) {
	for _, test := range []struct {
		desc                string
		asyncFuncShouldFail bool
		asyncFunc           func(*utils.QueueWatcher, chan struct{}) error
		handleFunc          func(*utils.QueueWatcher) error
	}{
		{
			desc: "Watch unblocks Poll",
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				pw := qw.Poller()
				_, err := pw("", time.Duration(0))
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				_, err := qw.Watch()
				return err
			},
		},
		{
			desc:                 "Cancel unblocks Poll",
			asyncFuncShouldFail:  true, // because qw.Cancel will be called
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				pw := qw.Poller()
				_, err := pw("", time.Duration(0))
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				return qw.Cancel()
			},
		},
		{
			desc: "Poll unblocks Watch",
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				_, err := qw.Watch()
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				pw := qw.Poller()
				_, err := pw("", time.Duration(0))
				return err
			},
		},
		{
			desc:                 "Cancel unblocks Watch",
			asyncFuncShouldFail:  true, // because qw.Cancel will be called
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				_, err := qw.Watch()
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				return qw.Cancel()
			},
		},
	} {
		c.Log(test.desc)
		exampleAsyncHandling(c, test.asyncFunc, test.handleFunc, test.asyncFuncShouldFail)
	}
}

func (tr *testNetResolver) LookupHost(ctx context.Context, host string) ([]string, error) {
	if tr.lookupHostCh != nil {
		if err := tr.lookupHostCh.SendContext(ctx, nil); err != nil {
			return nil, err
		}
	}

	tr.mu.Lock()
	defer tr.mu.Unlock()

	if addrs, ok := tr.hostLookupTable[host]; ok {
		return addrs, nil
	}

	return nil, &net.DNSError{
		Err:         "hostLookup error",
		Name:        host,
		Server:      "fake",
		IsTemporary: true,
	}
}

func (s) TestUnblocking(t *testing.T) {
	testCases := []struct {
		description string
		blockFuncShouldError bool
		blockFunc func(*testutils.PipeListener, chan struct{}) error
		unblockFunc func(*testutils.PipeListener) error
	}{
		{
			description: "Accept unblocks Dial",
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				_, err := pl.Accept()
				return err
			},
		},
		{
			description:                 "Close unblocks Dial",
			blockFuncShouldError: true, // because pl.Close will be called
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				return pl.Close()
			},
		},
		{
			description: "Dial unblocks Accept",
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				_, err := pl.Accept()
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				return err
			},
		},
		{
			description:                 "Close unblocks Accept",
			blockFuncShouldError: true, // because pl.Close will be called
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				_, err := pl.Accept()
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				return pl.Close()
			},
		},
	}

	for _, testCase := range testCases {
		t.Log(testCase.description)
		testUnblocking(t, testCase.blockFunc, testCase.unblockFunc, testCase.blockFuncShouldError)
	}
}

func waitForServiceReady(clientConn *grpc.ClientConn) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for currentState := clientConn.GetState(); ; {
		if currentState == connectivity.Ready {
			return nil
		}
		if !clientConn.WaitForStateChange(ctx, currentState) {
			return ctx.Err()
		}
		currentState = clientConn.GetState()
	}
}

