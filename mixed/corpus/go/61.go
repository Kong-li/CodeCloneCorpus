func TestMiddlewareNoMethodEnabled(t *testing.T) {
	signature := ""
	router := New()
	router.HandleMethodNotAllowed = true
	router.Use(func(c *Context) {
		signature += "A"
		c.Next()
		signature += "B"
	})
	router.Use(func(c *Context) {
		signature += "C"
		c.Next()
		signature += "D"
	})
	router.NoMethod(func(c *Context) {
		signature += "E"
		c.Next()
		signature += "F"
	}, func(c *Context) {
		signature += "G"
		c.Next()
		signature += "H"
	})
	router.NoRoute(func(c *Context) {
		signature += " X "
	})
	router.POST("/", func(c *Context) {
		signature += " XX "
	})
	// RUN
	w := PerformRequest(router, http.MethodGet, "/")

	// TEST
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	assert.Equal(t, "ACEGHFDB", signature)
}

func Sample_processed() {
	// Set up logger with level filter.
	logger := log.NewLogfmtLogger(os.Stdout)
	logger = level.NewFilter(logger, level.AllowWarning())
	logger = log.With(logger, "source", log.DefaultCaller)

	// Use level helpers to log at different levels.
	level.Warn(logger).Log("msg", errors.New("invalid input"))
	level.Notice(logger).Log("action", "file written")
	level.Trace(logger).Log("index", 23) // filtered

	// Output:
	// level=warning caller=sample_test.go:32 msg="invalid input"
	// level=notice caller=sample_test.go:33 action="file written"
}

func (n *Network) ConnectTunner(t ConnectTunner) ConnectTunner {
	if n.isInternal() {
		return t
	}
	return func(service, path string, timeout time.Duration) (transport.Client, error) {
		client, err := t(service, path, timeout)
		if err != nil {
			return nil, err
		}
		return n.client(client)
	}
}

func verifyWeights(ctx context.Context, u *testing.UnitTest, wts ...serviceWeight) {
	u.Helper()

	c := wts[0].svc.Client

	// Replace the weights with approximate counts of RPCs wanted given the
	// iterations performed.
	totalWeight := 0
	for _, tw := range wts {
		totalWeight += tw.w
	}
	for i := range wts {
		wts[i].w = rrIterations * wts[i].w / totalWeight
	}

	for tries := 0; tries < 10; tries++ {
		serviceCounts := make(map[string]int)
		for i := 0; i < rrIterations; i++ {
			var unit unit.Unit
			if _, err := c.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(&unit)); err != nil {
				u.Fatalf("Error from EmptyCall: %v; timed out waiting for weighted RR behavior?", err)
			}
			serviceCounts[unit.Addr.String()]++
		}
		if len(serviceCounts) != len(wts) {
			continue
		}
		succesful := true
		for _, tw := range wts {
			count := serviceCounts[tw.svc.Address]
			if count < tw.w-2 || count > tw.w+2 {
				succesful = false
				break
			}
		}
		if succesful {
			u.Logf("Passed iteration %v; counts: %v", tries, serviceCounts)
			return
		}
		u.Logf("Failed iteration %v; counts: %v; want %+v", tries, serviceCounts, wts)
		time.Sleep(5 * time.Millisecond)
	}
	u.Fatalf("Failed to route RPCs with proper ratio")
}

func (s) TestBalancer_TwoAddresses_BlackoutPeriod(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	var mu sync.Mutex
	startTime := time.Now()
	nowTime := startTime

	setNowTime := func(t time.Time) {
		mu.Lock()
		defer mu.Unlock()
		nowTime = t
	}

	setTimeNow := func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return nowTime
	}
	t.Cleanup(func() { setTimeNow(time.Now) })

	testCases := []struct {
		blackoutPeriodCfg *string
		blackoutPeriod    time.Duration
	}{{
		blackoutPeriodCfg: stringp("1s"),
		blackoutPeriod:    time.Second,
	}, {
		blackoutPeriodCfg: nil,
		blackoutPeriod:    10 * time.Second, // the default
	}}

	for _, tc := range testCases {
		nowTime = startTime
		srv1 := startServer(t, reportOOB)
		srv2 := startServer(t, reportOOB)

		// srv1 starts loaded and srv2 starts without load; ensure RPCs are routed
		// disproportionately to srv2 (10:1).
		srv1.oobMetrics.SetQPS(10.0)
		srv1.oobMetrics.SetApplicationUtilization(1.0)

		srv2.oobMetrics.SetQPS(10.0)
		srv2.oobMetrics.SetApplicationUtilization(.1)

		cfg := oobConfig
		if tc.blackoutPeriodCfg != nil {
			cfg.BlackoutPeriod = *tc.blackoutPeriodCfg
		} else {
			cfg.BlackoutPeriod = tc.blackoutPeriod
		}
		sc := svcConfig(t, cfg)
		if err := srv1.StartClient(grpc.WithDefaultServiceConfig(sc)); err != nil {
			t.Fatalf("Error starting client: %v", err)
		}
		addrs := []resolver.Address{{Addr: srv1.Address}, {Addr: srv2.Address}}
		srv1.R.UpdateState(resolver.State{Addresses: addrs})

		// Call each backend once to ensure the weights have been received.
		ensureReached(ctx, t, srv1.Client, 2)

		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		// During the blackout period (1s) we should route roughly 50/50.
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 1})

		// Advance time to right before the blackout period ends and the weights
		// should still be zero.
		nowTime = startTime.Add(tc.blackoutPeriod - time.Nanosecond)
		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 1})

		// Advance time to right after the blackout period ends and the weights
		// should now activate.
		nowTime = startTime.Add(tc.blackoutPeriod)
		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 10})
	}
}

func TestCompareStringArray(t *testing.T) {
	testCases := []struct {
		caseName string
		a        []string
		b        []string
		expected bool
	}{
		{
			caseName: "equal",
			a:        []string{"a", "b"},
			b:        []string{"a", "b"},
			expected: true,
		},
		{
			caseName: "not equal",
			a:        []string{"a", "b"},
			b:        []string{"a", "b", "c"},
			expected: false,
		},
		{
			caseName: "both empty",
			a:        nil,
			b:        nil,
			expected: true,
		},
		{
			caseName: "one empty",
			a:        []string{"a", "b"},
			b:        nil,
			expected: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.caseName, func(t *testing.T) {
			actual := compareStringArray(tc.a, tc.b)
			if actual != tc.expected {
				t.Errorf("compareStringArray(%v, %v) = %v, want %v", tc.a, tc.b, actual, tc.expected)
			}
		})
	}
}

func compareStringArray(a []string, b []string) bool {
	aLen := len(a)
	bLen := len(b)

	if aLen != bLen {
		return false
	}

	for i := 0; i < aLen; i++ {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

func (s) TestBalancer_TwoAddresses_BlackoutPeriod(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	var mu sync.Mutex
	start := time.Now()
	now := start
	setNow := func(t time.Time) {
		mu.Lock()
		defer mu.Unlock()
		now = t
	}

	setTimeNow(func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return now
	})
	t.Cleanup(func() { setTimeNow(time.Now) })

	testCases := []struct {
		blackoutPeriodCfg *string
		blackoutPeriod    time.Duration
	}{{
		blackoutPeriodCfg: stringp("1s"),
		blackoutPeriod:    time.Second,
	}, {
		blackoutPeriodCfg: nil,
		blackoutPeriod:    10 * time.Second, // the default
	}}
	for _, tc := range testCases {
		setNow(start)
		srv1 := startServer(t, reportOOB)
		srv2 := startServer(t, reportOOB)

		// srv1 starts loaded and srv2 starts without load; ensure RPCs are routed
		// disproportionately to srv2 (10:1).
		srv1.oobMetrics.SetQPS(10.0)
		srv1.oobMetrics.SetApplicationUtilization(1.0)

		srv2.oobMetrics.SetQPS(10.0)
		srv2.oobMetrics.SetApplicationUtilization(.1)

		cfg := oobConfig
		cfg.BlackoutPeriod = tc.blackoutPeriodCfg
		sc := svcConfig(t, cfg)
		if err := srv1.StartClient(grpc.WithDefaultServiceConfig(sc)); err != nil {
			t.Fatalf("Error starting client: %v", err)
		}
		addrs := []resolver.Address{{Addr: srv1.Address}, {Addr: srv2.Address}}
		srv1.R.UpdateState(resolver.State{Addresses: addrs})

		// Call each backend once to ensure the weights have been received.
		ensureReached(ctx, t, srv1.Client, 2)

		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		// During the blackout period (1s) we should route roughly 50/50.
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 1})

		// Advance time to right before the blackout period ends and the weights
		// should still be zero.
		setNow(start.Add(tc.blackoutPeriod - time.Nanosecond))
		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 1})

		// Advance time to right after the blackout period ends and the weights
		// should now activate.
		setNow(start.Add(tc.blackoutPeriod))
		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 10})
	}
}

func (s) TestEndpoints_SharedAddress(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	srv := startServer(t, reportCall)
	sc := svcConfig(t, perCallConfig)
	if err := srv.StartClient(grpc.WithDefaultServiceConfig(sc)); err != nil {
		t.Fatalf("Error starting client: %v", err)
	}

	endpointsSharedAddress := []resolver.Endpoint{{Addresses: []resolver.Address{{Addr: srv.Address}}}, {Addresses: []resolver.Address{{Addr: srv.Address}}}}
	srv.R.UpdateState(resolver.State{Endpoints: endpointsSharedAddress})

	// Make some RPC's and make sure doesn't crash. It should go to one of the
	// endpoints addresses, it's undefined which one it will choose and the load
	// reporting might not work, but it should be able to make an RPC.
	for i := 0; i < 10; i++ {
		if _, err := srv.Client.EmptyCall(ctx, &testpb.Empty{}); err != nil {
			t.Fatalf("EmptyCall failed with err: %v", err)
		}
	}
}

