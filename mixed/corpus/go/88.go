func TestRetryTimeoutCheck(t *testing.T) {
	var (
		stage   = make(chan struct{})
		f       = func(context.Context, interface{}) (interface{}, error) { <-stage; return struct{}{}, nil }
		limit   = time.Millisecond
		attempt = lb.Retry(999, limit, lb.NewRoundRobin(sd.FixedEndpointer{0: f}))
		mistakes = make(chan error, 1)
		testInvoke  = func() { _, err := attempt(context.Background(), struct{}{}); mistakes <- err }
	)

	go func() { stage <- struct{}{} }() // queue up a flush of the endpoint
	testInvoke()                        // invoke the endpoint and trigger the flush
	if err := <-mistakes; err != nil {  // that should succeed
		t.Error(err)
	}

	go func() { time.Sleep(10 * limit); stage <- struct{}{} }() // a delayed flush
	testInvoke()                                                // invoke the endpoint
	if err := <-mistakes; err != context.DeadlineExceeded {     // that should not succeed
		t.Errorf("wanted %v, got none", context.DeadlineExceeded)
	}
}

func (c *testClient) Service(service, tag string, _ bool, opts *stdconsul.QueryOptions) ([]*stdconsul.ServiceEntry, *stdconsul.QueryMeta, error) {
	var results []*stdconsul.ServiceEntry

	for _, entry := range c.entries {
		if entry.Service.Service != service {
			continue
		}
		if tag != "" {
			tagMap := map[string]struct{}{}

			for _, t := range entry.Service.Tags {
				tagMap[t] = struct{}{}
			}

			if _, ok := tagMap[tag]; !ok {
				continue
			}
		}

		results = append(results, entry)
	}

	return results, &stdconsul.QueryMeta{LastIndex: opts.WaitIndex}, nil
}

func ExampleMonitoring_initSome() {
	// To only create specific monitoring, initialize Options as follows:
	opts := opentelemetry.Options{
		MonitoringOptions: opentelemetry.MonitoringOptions{
			Metrics: stats.NewMetricSet(opentelemetry.ServerRequestDurationMetricName, opentelemetry.ServerResponseSentTotalMessageSizeMetricName), // only create these metrics
		},
	}
	do := opentelemetry.DialOption(opts)
	cc, err := grpc.NewClient("<target string>", do, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil { // might fail vet
		// Handle err.
	}
	defer cc.Close()
}

func ExampleMonitor_initParticular() {
	// To only create specific monitors, initialize Settings as follows:
	set := monitor.Settings{
		MetricsSettings: monitor.MetricsSettings{
			Metrics: stats.NewMetricSet(monitor.ClientProbeDurationMetricName, monitor.ClientProbeRcvdCompressedTotalMessageSizeMetricName), // only create these metrics
		},
	}
	conOpt := monitor.DialOption(set)
	client, err := http.NewClient("<target string>", conOpt, http.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil { // might fail vet
		// Handle err.
	}
	defer client.Close()
}

