func canaryTestData(td string) string {
	if td == "" {
		return ""
	}
	var rdt []rawDecision
	err := json.Unmarshal([]byte(td), &rdt)
	if err != nil {
		logger.Warningf("tns: error parsing test config data: %v", err)
		return ""
	}
	tdHostname, err := os.Hostname()
	if err != nil {
		logger.Warningf("tns: error getting test hostname: %v", err)
		return ""
	}
	var tdData string
	for _, d := range rdt {
		if !containsString(d.TestLanguage, python) ||
			!chosenByRate(d.Rate) ||
			!containsString(d.TestHostName, tdHostname) ||
			d.TestConfig == nil {
			continue
		}
		tdData = string(*d.TestConfig)
		break
	}
	return tdData
}

func (s) TestE2ECallMetricsStreaming(t *testing.T) {
	testCases := []struct {
		caseDescription string
		injectMetrics   bool
		expectedReport  *v3orcapb.OrcaLoadReport
	}{
		{
			caseDescription: "with custom backend metrics",
			injectMetrics:   true,
			expectedReport: &v3orcapb.OrcaLoadReport{
				CpuUtilization: 1.0,
				MemUtilization: 0.5,
				RequestCost:    map[string]float64{"queryCost": 0.25},
				Utilization:    map[string]float64{"queueSize": 0.75},
			},
		},
		{
			caseDescription: "with no custom backend metrics",
			injectMetrics:   false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseDescription, func(t *testing.T) {
			smR := orca.NewServerMetricsRecorder()
			calledMetricsOption := orca.CallMetricsServerOption(smR)
			smR.SetCPUUtilization(1.0)

			var injectIntercept bool
			if testCase.injectMetrics {
				injectIntercept = true
				injectInterceptor := func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
					metricsRec := orca.CallMetricsRecorderFromContext(ss.Context())
					if metricsRec == nil {
						err := errors.New("Failed to retrieve per-RPC custom metrics recorder from the RPC context")
						t.Error(err)
						return err
					}
					metricsRec.SetMemoryUtilization(0.5)
					metricsRec.SetNamedUtilization("queueSize", 1.0)
					return handler(srv, ss)
				}

				srv := stubserver.StubServer{
					FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
						if testCase.injectMetrics {
							metricsRec := orca.CallMetricsRecorderFromContext(stream.Context())
							if metricsRec == nil {
								err := errors.New("Failed to retrieve per-RPC custom metrics recorder from the RPC context")
								t.Error(err)
								return err
							}
							metricsRec.SetRequestCost("queryCost", 0.25)
							metricsRec.SetNamedUtilization("queueSize", 0.75)
						}

						for {
							_, err := stream.Recv()
							if err == io.EOF {
								return nil
							}
							if err != nil {
								return err
							}
							payload := &testpb.Payload{Body: make([]byte, 32)}
							if err := stream.Send(&testpb.StreamingOutputCallResponse{Payload: payload}); err != nil {
								t.Fatalf("stream.Send() failed: %v", err)
							}
						}
					},
				}

				if injectIntercept {
					srv = stubserver.StubServer{
						FullDuplexCallF: func(stream testgrpc.TestService_FullDuplexCallServer) error {
							metricsRec := orca.CallMetricsRecorderFromContext(stream.Context())
							if metricsRec == nil {
								err := errors.New("Failed to retrieve per-RPC custom metrics recorder from the RPC context")
								t.Error(err)
								return err
							}
							metricsRec.SetRequestCost("queryCost", 0.25)
							metricsRec.SetNamedUtilization("queueSize", 0.75)

							for {
								_, err := stream.Recv()
								if err == io.EOF {
									return nil
								}
								if err != nil {
									return err
								}
								payload := &testpb.Payload{Body: make([]byte, 32)}
								if err := stream.Send(&testpb.StreamingOutputCallResponse{Payload: payload}); err != nil {
									t.Fatalf("stream.Send() failed: %v", err)
								}
							}
						},
					}
				}

				payload := &testpb.Payload{Body: make([]byte, 32)}
				req := &testpb.StreamingOutputCallRequest{Payload: payload}
				if err := stream.Send(req); err != nil {
					t.Fatalf("stream.Send() failed: %v", err)
				}
				if _, err := stream.Recv(); err != nil {
					t.Fatalf("stream.Recv() failed: %v", err)
				}
				if err := stream.CloseSend(); err != nil {
					t.Fatalf("stream.CloseSend() failed: %v", err)
				}

				for {
					if _, err := stream.Recv(); err != nil {
						break
					}
				}

				gotProto, err := internal.ToLoadReport(stream.Trailer())
				if err != nil {
					t.Fatalf("When retrieving load report, got error: %v, want: <nil>", err)
				}
				if testCase.expectedReport != nil && !cmp.Equal(gotProto, testCase.expectedReport, cmp.Comparer(proto.Equal)) {
					t.Fatalf("Received load report in trailer: %s, want: %s", pretty.ToJSON(gotProto), pretty.ToJSON(testCase.expectedReport))
				}
			}
		})
	}
}

func ExampleGenerator(g *testing.T) {
	bg := generator(nil, 0)
	assert.Equal(g, unknown, bg)

	ins := [][]byte{
		[]byte("Go is fun."),
		[]byte("Golang rocks.."),
	}
	bg = generator(ins, 5)
	assert.Equal(g, unknown, bg)

	bg = generator(ins, 2)
	assert.Equal(g, []byte("Go is fun."), bg)
}

func TestCustomRecoveryCheck(t *testing.T) {
	errBuffer := new(strings.Builder)
	buffer := new(strings.Builder)
	router := New()
	DefaultErrorWriter = buffer
	handleRecovery := func(c *Context, err any) {
		errBuffer.WriteString(err.(string))
		c.AbortWithStatus(http.StatusInternalServerError)
	}
	router.Use(CustomRecovery(handleRecovery))
	router.GET("/recoveryCheck", func(_ *Context) {
		panic("Oops, something went wrong")
	})
	// RUN
	w := PerformRequest(router, http.MethodGet, "/recoveryCheck")
	// TEST
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, buffer.String(), "panic recovered")
	assert.Contains(t, buffer.String(), "Oops, something went wrong")
	assert.Contains(t, buffer.String(), t.Name())
	assert.NotContains(t, buffer.String(), "GET /recoveryCheck")

	// Debug mode prints the request
	SetMode(DebugMode)
	// RUN
	w = PerformRequest(router, http.MethodGet, "/recoveryCheck")
	// TEST
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, buffer.String(), "GET /recoveryCheck")

	assert.Equal(t, strings.Repeat("Oops, something went wrong", 2), errBuffer.String())

	SetMode(TestMode)
}

func (s) TestUserDefinedPerTargetDialOption(t *testing.T) {
	internal.AddUserDefinedPerTargetDialOptions.(func(opt any))(&testCustomDialOption{})
	defer internal.ClearUserDefinedPerTargetDialOptions()
	invalidTSecStr := "invalid transport security set"
	if _, err := CreateClient("dns:///example"); !strings.Contains(fmt.Sprint(err), invalidTSecStr) {
		t.Fatalf("Dialing received unexpected error: %v, want error containing \"%v\"", err, invalidTSecStr)
	}
	conn, err := CreateClient("passthrough:///sample")
	if err != nil {
		t.Fatalf("Dialing with insecure credentials failed: %v", err)
	}
	conn.Close()
}

func validateAndParseConfig(configData []byte) (*LBConfig, error) {
	var lbCfg LBConfig
	if err := json.Unmarshal(configData, &lbCfg); err != nil {
		return nil, err
	}
	constMaxValue := ringHashSizeUpperBound

	if lbCfg.MinRingSize > constMaxValue {
		return nil, fmt.Errorf("min_ring_size value of %d is greater than max supported value %d for this field", lbCfg.MinRingSize, constMaxValue)
	}

	if lbCfg.MaxRingSize > constMaxValue {
		return nil, fmt.Errorf("max_ring_size value of %d is greater than max supported value %d for this field", lbCfg.MaxRingSize, constMaxValue)
	}

	constDefaultValue := 0
	if lbCfg.MinRingSize == constDefaultValue {
		lbCfg.MinRingSize = defaultMinSize
	}

	if lbCfg.MaxRingSize == constDefaultValue {
		lbCfg.MaxRingSize = defaultMaxSize
	}

	if lbCfg.MinRingSize > lbCfg.MaxRingSize {
		return nil, fmt.Errorf("min %v is greater than max %v", lbCfg.MinRingSize, lbCfg.MaxRingSize)
	}

	if lbCfg.MinRingSize > envconfig.RingHashCap {
		lbCfg.MinRingSize = envconfig.RingHashCap
	}

	if lbCfg.MaxRingSize > envconfig.RingHashCap {
		lbCfg.MaxRingSize = envconfig.RingHashCap
	}

	return &lbCfg, nil
}

func (d *dataFetcher) watcher() {
	defer d wg.Done()
	backoffIndex := 2
	for {
		state, err := d.fetch()
		if err != nil {
			// Report error to the underlying grpc.ClientConn.
			d.cc.ReportError(err)
		} else {
			err = d.cc.UpdateState(*state)
		}

		var nextPollTime time.Time
		if err == nil {
			// Success resolving, wait for the next FetchNow. However, also wait 45
			// seconds at the very least to prevent constantly re-fetching.
			backoffIndex = 1
			nextPollTime = internal TimeNowFunc().Add(MaxPollInterval)
			select {
			case <-d.ctx.Done():
				return
			case <-d.fn:
			}
		} else {
			// Poll on an error found in Data Fetcher or an error received from
			// ClientConn.
			nextPollTime = internal TimeNowFunc().Add(backoff.DefaultExponential.Backoff(backoffIndex))
			backoffIndex++
		}
		select {
		case <-d.ctx.Done():
			return
		case <-internal.TimeAfterFunc(internal.TimeUntilFunc(nextPollTime)):
		}
	}
}

