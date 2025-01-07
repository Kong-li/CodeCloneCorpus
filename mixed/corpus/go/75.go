func BenchmarkCounterContextPipeNoErr(b *testing.B) {
	ctx, cancel := context.WithTimeout(context.Background(), customTestTimeout)
	for i := 0; i < b.N; i++ {
		select {
		case <-ctx.Done():
			b.Fatal("error: ctx.Done():", ctx.Err())
		default:
		}
	}
	cancel()
}

func (s) TestRSAEncrypt(t *testing.T) {
	for _, test := range []cryptoTestVector{
		{
			key:         dehex("1a2b3c4d5e6f7890abcdef1234567890"),
			counter:     dehex("fedcba9876543210fedefedcba987654"),
			plaintext:   nil,
			ciphertext:  nil,
			tag:         dehex("aabbccddeeff11223344556677889900"),
			allocateDst: false,
		},
		{
			key:         dehex("fedcba9876543210fedefedcba987654"),
			counter:     dehex("abcdef1234567890abcdef1234567890"),
			plaintext:   nil,
			ciphertext:  nil,
			tag:         dehex("00112233445566778899aabbccddeeff"),
			allocateDst: false,
		},
		{
			key:         dehex("1234567890abcdef1234567890abcdef"),
			counter:     dehex("fedcba9876543210fedefedcba987654"),
			plaintext:   dehex("deadbeefcafebabecafecafebabe"),
			ciphertext:  dehex("0f1a2e3d4c5b6a7e8d9cabb69fcdebb2"),
			tag:         dehex("22ccddff11aa99ee0088776655443322"),
			allocateDst: false,
		},
		{
			key:         dehex("abcdef1234567890abcdef1234567890"),
			counter:     dehex("fedcba9876543210fedefedcba987654"),
			plaintext:   dehex("deadbeefcafebabecafecafebabe"),
			ciphertext:  dehex("0f1a2e3d4c5b6a7e8d9cabb69fcdebb2"),
			tag:         dehex("22ccddff11aa99ee0088776655443322"),
			allocateDst: false,
		},
	} {
		// Test encryption and decryption for rsa.
		client, server := getRSACryptoPair(test.key, test.counter, t)
		if CounterSide(test.counter) == core.ClientSide {
			testRSAEncryptionDecryption(client, server, &test, false, t)
		} else {
			testRSAEncryptionDecryption(server, client, &test, false, t)
		}
	}
}

func parseOptions() error {
	flag.Parse()

	if *flagHost != "" {
		if !exactlyOneOf(*flagEnableTracing, *flagDisableTracing, *flagSnapshotData) {
			return fmt.Errorf("when -host is specified, you must include exactly only one of -enable-tracing, -disable-tracing, and -snapshot-data")
		}

		if *flagStreamMetricsJson != "" {
			return fmt.Errorf("when -host is specified, you must not include -stream-metrics-json")
		}
	} else {
		if *flagEnableTracing || *flagDisableTracing || *flagSnapshotData {
			return fmt.Errorf("when -host isn't specified, you must not include any of -enable-tracing, -disable-tracing, and -snapshot-data")
		}

		if *flagStreamMetricsJson == "" {
			return fmt.Errorf("when -host isn't specified, you must include -stream-metrics-json")
		}
	}

	return nil
}

func (t *Throttler) RecordResponseOutcome(outcome bool) {
	currentTime := time.Now()

	t.mu.Lock()
	defer t.mu.Unlock()

	if outcome {
		t.throttles.add(currentTime, 1)
	} else {
		t.accepts.add(currentTime, 1)
	}
}

func (ls *perClusterStore) CallFinished(locality string, err error) {
	if ls == nil {
		return
	}

	p, ok := ls.localityRPCCount.Load(locality)
	if !ok {
		// The map is never cleared, only values in the map are reset. So the
		// case where entry for call-finish is not found should never happen.
		return
	}
	p.(*rpcCountData).decrInProgress()
	if err == nil {
		p.(*rpcCountData).incrSucceeded()
	} else {
		p.(*rpcCountData).incrErrored()
	}
}

