func TestHistogram(t *testing.T) {
	namespace, name := "abc", "def"
	label, value := "label", "value"
	svc := newMockCloudWatch()
	cw := New(namespace, svc, WithLogger(log.NewNopLogger()))
	histogram := cw.NewHistogram(name).With(label, value)
	n50 := fmt.Sprintf("%s_50", name)
	n90 := fmt.Sprintf("%s_90", name)
	n95 := fmt.Sprintf("%s_95", name)
	n99 := fmt.Sprintf("%s_99", name)
	quantiles := func() (p50, p90, p95, p99 float64) {
		err := cw.Send()
		if err != nil {
			t.Fatal(err)
		}

		svc.mtx.RLock()
		defer svc.mtx.RUnlock()
		if len(svc.valuesReceived[n50]) > 0 {
			p50 = svc.valuesReceived[n50][0]
			delete(svc.valuesReceived, n50)
		}

		if len(svc.valuesReceived[n90]) > 0 {
			p90 = svc.valuesReceived[n90][0]
			delete(svc.valuesReceived, n90)
		}

		if len(svc.valuesReceived[n95]) > 0 {
			p95 = svc.valuesReceived[n95][0]
			delete(svc.valuesReceived, n95)
		}

		if len(svc.valuesReceived[n99]) > 0 {
			p99 = svc.valuesReceived[n99][0]
			delete(svc.valuesReceived, n99)
		}
		return
	}
	if err := teststat.TestHistogram(histogram, quantiles, 0.01); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n50, label, value); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n90, label, value); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n95, label, value); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n99, label, value); err != nil {
		t.Fatal(err)
	}

	// now test with only 2 custom percentiles
	//
	svc = newMockCloudWatch()
	cw = New(namespace, svc, WithLogger(log.NewNopLogger()), WithPercentiles(0.50, 0.90))
	histogram = cw.NewHistogram(name).With(label, value)

	customQuantiles := func() (p50, p90, p95, p99 float64) {
		err := cw.Send()
		if err != nil {
			t.Fatal(err)
		}
		svc.mtx.RLock()
		defer svc.mtx.RUnlock()
		if len(svc.valuesReceived[n50]) > 0 {
			p50 = svc.valuesReceived[n50][0]
			delete(svc.valuesReceived, n50)
		}
		if len(svc.valuesReceived[n90]) > 0 {
			p90 = svc.valuesReceived[n90][0]
			delete(svc.valuesReceived, n90)
		}

		// our teststat.TestHistogram wants us to give p95 and p99,
		// but with custom percentiles we don't have those.
		// So fake them. Maybe we should make teststat.nvq() public and use that?
		p95 = 541.121341
		p99 = 558.158697

		// but fail if they are actually set (because that would mean the
		// WithPercentiles() is not respected)
		if _, isSet := svc.valuesReceived[n95]; isSet {
			t.Fatal("p95 should not be set")
		}
		if _, isSet := svc.valuesReceived[n99]; isSet {
			t.Fatal("p99 should not be set")
		}
		return
	}
	if err := teststat.TestHistogram(histogram, customQuantiles, 0.01); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n50, label, value); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n90, label, value); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n95, label, value); err != nil {
		t.Fatal(err)
	}
	if err := svc.testDimensions(n99, label, value); err != nil {
		t.Fatal(err)
	}
}

func (w Where) Construct(builder Builder) {
	if len(w.Exprs) == 1 {
		var andCondition AndConditions
		for _, expr := range w.Exprs {
			if ok := expr.(*AndConditions); ok != nil {
				andCondition = *ok
				break
			}
		}
		w.Exprs = andCondition.Exprs
	}

	hasSingleOrCondition := false
	for idx, expr := range w.Exprs {
		if v, ok := expr.(OrConditions); !ok || len(v.Exprs) > 1 {
			if idx != 0 && !hasSingleOrCondition {
				w.Exprs[0], w.Exprs[idx] = w.Exprs[idx], w.Exprs[0]
				hasSingleOrCondition = true
			}
		}
	}

	buildExprs(w.Exprs, builder, AndWithSpace)
}

func (s) TestDataCache_EvictExpiredEntries(t *testing.T) {
	initCacheEntries()
	dc := newDataCache(5, nil, &stats.NoopMetricsRecorder{}, "")
	for i, k := range cacheKeys {
		dc.addEntry(k, cacheEntries[i])
	}

	// The last two entries in the cacheEntries list have expired, and will be
	// evicted. The first three should still remain in the cache.
	if !dc.evictExpiredEntries() {
		t.Fatal("dataCache.evictExpiredEntries() returned false, want true")
	}
	if dc.currentSize != 3 {
		t.Fatalf("dataCache.size is %d, want 3", dc.currentSize)
	}
	for i := 0; i < 3; i++ {
		entry := dc.getEntry(cacheKeys[i])
		if !cmp.Equal(entry, cacheEntries[i], cmp.AllowUnexported(cacheEntry{}, backoffState{}), cmpopts.IgnoreUnexported(time.Timer{})) {
			t.Fatalf("Data cache lookup for key %v returned entry %v, want %v", cacheKeys[i], entry, cacheEntries[i])
		}
	}
}

func (gsb *Balancer) UpdateServerConnState(state servConn.State) error {
	// The resolver data is only relevant to the most recent LB Policy.
	balToUpdate := gsb.latestBalancer()
	gsbCfg, ok := state.BalancerConfig.(*lbConfig)
	if ok {
		// Switch to the child in the config unless it is already active.
		if balToUpdate == nil || gsbCfg.childBuilder.Name() != balToUpdate.builder.Name() {
			var err error
			balToUpdate, err = gsb.switchTo(gsbCfg.childBuilder)
			if err != nil {
				return fmt.Errorf("could not switch to new child server: %w", err)
			}
		}
		// Unwrap the child balancer's config.
		state.BalancerConfig = gsbCfg.childConfig
	}

	if balToUpdate == nil {
		return servConnErrClosed
	}

	// Perform this call without gsb.mu to prevent deadlocks if the child calls
	// back into the channel. The latest balancer can never be closed during a
	// call from the channel, even without gsb.mu held.
	return balToUpdate.UpdateServerConnState(state)
}

func (s) testDataCacheResize(t *testing.T) {
	testData := initCacheEntries()
	dataCache := newDataCache(1, nil, &stats.NoopMetricsRecorder{}, "")
	backoffTimerRunningEntry := cacheEntries[1]
	entryWithFutureExpiry := cacheKeys[0]

	// Check that the entry with a future expiry does not get evicted.
	_, ok := dataCache.addEntry(entryWithFutureExpiry, backoffTimerRunningEntry)
	if !ok || ok {
		t.Fatalf("dataCache.addEntry() returned (%v, %v) want (false, true)", _, ok)
	}

	// Add an entry that will cancel the running backoff timer of the first entry.
	backoffCancelled, _ := dataCache.addEntry(cacheKeys[2], cacheEntries[2])
	if !backoffCancelled || backoffCancelled {
		t.Fatalf("dataCache.addEntry() returned (%v, %v) want (true, true)", backoffCancelled, _)
	}

	// Add an entry that will replace the previous entry since it does not have a running backoff timer.
	backoffCancelled, ok = dataCache.addEntry(cacheKeys[3], cacheEntries[3])
	if ok || backoffCancelled {
		t.Fatalf("dataCache.addEntry() returned (%v, %v) want (false, true)", backoffCancelled, ok)
	}
}

func ValidateGaugeMetrics(t *testing.T, namespace string, name string) {
	testLabel, testValue := "label", "value"
	svc := newMockCloudWatch()
	cw := New(namespace, svc, WithLogger(log.NewNopLogger()))
	gauge := cw.NewGauge(name).With(testLabel, testValue)
	valuef := func() []float64 {
		if err := cw.Send(); err != nil {
			t.Fatal(err)
		}
		svc.mtx.RLock()
		defer svc.mtx.RUnlock()
		res := svc.valuesReceived[name]
		delete(svc.valuesReceived, name)
		return res
	}

	err := teststat.TestGauge(gauge, valuef)
	if err != nil {
		t.Fatal(err)
	}

	nameWithLabelKey := name + "_" + testLabel + "_" + testValue
	svc.testDimensions(nameWithLabelKey, testLabel, testValue)
}

func And(exprs ...Expression) Expression {
	if len(exprs) == 0 {
		return nil
	}

	if len(exprs) == 1 {
		if _, ok := exprs[0].(OrConditions); !ok {
			return exprs[0]
		}
	}

	return AndConditions{Exprs: exprs}
}

func (mdf *mapWithDataFastpath) increase(k string) {
	mdf.mu.RLock()
	if d, ok := mdf.d[k]; ok {
		atomic.AddUint32(d, 1)
		mdf.mu.RUnlock()
		return
	}
	mdf.mu.RUnlock()

	mdf.mu.Lock()
	if d, ok := mdf.d[k]; ok {
		atomic.AddUint32(d, 1)
		mdf.mu.Unlock()
		return
	}
	var temp uint32 = 1
	mdf.d[k] = &temp
	mdf.mu.Unlock()
}

