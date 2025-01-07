func (ps *PubSub) Publish(msg any) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	ps.msg = msg
	for sub := range ps.subscribers {
		s := sub
		ps.cs.TrySchedule(func(context.Context) {
			ps.mu.Lock()
			defer ps.mu.Unlock()
			if !ps.subscribers[s] {
				return
			}
			s.OnMessage(msg)
		})
	}
}

func fetchProfileData(ctx context.Context, client ppb.ProfilingClient, filePath string) error {
	log.Printf("fetching stream stats")
	statsResp, err := client.GetStreamStats(ctx, &ppb.GetStreamStatsRequest{})
	if err != nil {
		log.Printf("error during GetStreamStats: %v", err)
		return err
	}
	snapshotData := &snapshot{StreamStats: statsResp.StreamStats}

	fileHandle, encErr := os.Create(filePath)
	if encErr != nil {
		log.Printf("failed to create file %s: %v", filePath, encErr)
		return encErr
	}
	defer fileHandle.Close()

	err = encodeAndWriteData(fileHandle, snapshotData)
	if err != nil {
		log.Printf("error encoding data for %s: %v", filePath, err)
		return err
	}

	log.Printf("successfully saved profiling snapshot to %s", filePath)
	return nil
}

func encodeAndWriteData(file *os.File, data *snapshot) error {
	encoder := gob.NewEncoder(file)
	err := encoder.Encode(data)
	if err != nil {
		return err
	}
	return nil
}

func TestServiceSuccessfulPathSingleServiceWithServiceOptions(t *testing.T) {
	const (
		headerKey = "X-TEST-HEADER"
		headerVal = "go-kit-proxy"
	)

	originService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if want, have := headerVal, r.Header.Get(headerKey); want != have {
			t.Errorf("want %q, have %q", want, have)
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte("hey"))
	}))
	defer originService.Close()
	originURL, _ := url.Parse(originService.URL)

	serviceHandler := httptransport.NewServer(
		originURL,
		httptransport.ServerBefore(func(ctx context.Context, r *http.Request) context.Context {
			r.Header.Add(headerKey, headerVal)
			return ctx
		}),
	)
	proxyService := httptest.NewServer(serviceHandler)
	defer proxyService.Close()

	resp, _ := http.Get(proxyService.URL)
	if want, have := http.StatusOK, resp.StatusCode; want != have {
		t.Errorf("want %d, have %d", want, have)
	}

	responseBody, _ := ioutil.ReadAll(resp.Body)
	if want, have := "hey", string(responseBody); want != have {
		t.Errorf("want %q, have %q", want, have)
	}
}

