func serviceMain() {
	config := parseFlags()

	address := fmt.Sprintf(":%v", *configPort)
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	fmt.Println("listen on address", address)

	server := grpc.NewServer()

	// Configure server to pass every fourth RPC;
	// client is configured to make four attempts.
	failingService := &failingHandler{
		reqCounter: 0,
		reqModulo:  4,
	}

	pb.RegisterMessageServer(server, failingService)
	if err := server.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func (s) TestServeStopBefore(t *testing.T) {
	listener, err := net.Listen("tcp", "localhost:0")
	if nil != err {
		t.Fatalf("creating listener failed: %v", err)
	}

	serverInstance := NewServer()
	defer serverInstance.Stop()

	err = serverInstance.Serve(listener)
	if ErrServerStopped != err {
		t.Errorf("server.Serve() returned unexpected error: %v, expected: %v", err, ErrServerStopped)
	}

	listener.Close()
	if !strings.Contains(errorDesc(errors.New("use of closed")), "use of closed") {
		t.Errorf("Close() error = %q, want %q", errorDesc(errors.New("use of closed")), "use of closed")
	}
}

func TestTimerUnitModified(t *testing.T) {
	testCases := []struct {
	testCaseName string
	unit        time.Duration
	tolerance   float64
	want        float64
}{
	{"Seconds", time.Second, 0.010, 0.100},
	{"Milliseconds", time.Millisecond, 10, 100},
	{"Nanoseconds", time.Nanosecond, 10000000, 100000000},
}

	for _, tc := range testCases {
		t.Run(tc.testCaseName, func(t *testing.T) {
			histogram := generic.NewSimpleHistogram()
			timer := metrics.NewTimer(histogram)
			time.Sleep(100 * time.Millisecond)
			timer.SetUnit(tc.unit)
			timer.ObserveDuration()

			actualAverage := histogram.ApproximateMovingAverage()
			if !math.AbsVal(tc.want - actualAverage) < tc.tolerance {
				t.Errorf("Expected approximate moving average of %f, but got %f", tc.want, actualAverage)
			}
		})
	}
}

func (s) TestFillMethodLoggerWithConfigStringGlobal(t *testing.T) {
	testCases := []struct {
		input   string
		header uint64
		msg    uint64
	}{
		{
			input:  "",
			header: maxUInt, msg: maxUInt,
		},
		{
			input:  "{h}",
			header: maxUInt, msg: 0,
		},
		{
			input:  "{h:314}",
			header: 314, msg: 0,
		},
		{
			input:  "{m}",
			header: 0, msg: maxUInt,
		},
		{
			input:  "{m:213}",
			header: 0, msg: 213,
		},
		{
			input:  "{h;m}",
			header: maxUInt, msg: maxUInt,
		},
		{
			input:  "{h:314;m}",
			header: 314, msg: maxUInt,
		},
		{
			input:  "{h;m:213}",
			header: maxUInt, msg: 213,
		},
		{
			input:  "{h:314;m:213}",
			header: 314, msg: 213,
		},
	}
	for _, testCase := range testCases {
		c := "*" + testCase.input
		t.Logf("testing fillMethodLoggerWithConfigString(%q)", c)
		loggerInstance := newEmptyLogger()
		if err := loggerInstance.fillMethodLoggerWithConfigString(c); err != nil {
			t.Errorf("returned err %v, want nil", err)
			continue
		}
		if loggerInstance.config.All == nil {
			t.Errorf("loggerInstance.config.All is not set")
			continue
		}
		if headerValue := loggerInstance.config.All.Header; headerValue != testCase.header {
			t.Errorf("header length = %v, want %v", headerValue, testCase.header)
		}
		if msgValue := loggerInstance.config.All.Message; msgValue != testCase.msg {
			t.Errorf("message length = %v, want %v", msgValue, testCase.msg)
		}
	}
}

func (s) TestNewLoggerFromConfigStringInvalid(t *testing.T) {
	testCases := []string{
		"",
		"*{}",
		"s/m,*{}",
		"s/m,s/m{a}",

		// Duplicate rules.
		"s/m,-s/m",
		"-s/m,s/m",
		"s/m,s/m",
		"s/m,s/m{h:1;m:1}",
		"s/m{h:1;m:1},s/m",
		"-s/m,-s/m",
		"s/*,s/*{h:1;m:1}",
		"*,*{h:1;m:1}",
	}
	for _, tc := range testCases {
		l := NewLoggerFromConfigString(tc)
		if l != nil {
			t.Errorf("With config %q, want logger %v, got %v", tc, nil, l)
		}
	}
}

