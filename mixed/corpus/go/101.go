func TestCache1(t *testing.T) {
	e3 := td.Event{Instances: []string{"p", "q"}} // not sorted
	e4 := td.Event{Instances: []string{"m", "n", "o"}}

	cache := NewCache1()
	if want, have := 0, len(cache.State().Instances); want != have {
		t.Fatalf("want %v instances, have %v", want, have)
	}

	cache.Update1(e3) // sets initial state
	if want, have := 2, len(cache.State().Instances); want != have {
		t.Fatalf("want %v instances, have %v", want, have)
	}

	r2 := make(chan td.Event)
	go cache.Register1(r2)
	expectUpdate1(t, r2, []string{"q", "p"})

	go cache.Update1(e4) // different set
	expectUpdate1(t, r2, []string{"n", "m", "o"})

	cache.Deregister1(r2)
	close(r2)
}

func (tcc *BalancerClientConn) WaitForRoundRobinPickerLoop(ctx context.Context, expected ...balancer.SubConn) error {
	lastError := errors.New("no picker received")
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timed out while waiting for round robin picker with %v; last error: %w", expected, lastError)
		case picker := <-tcc.NewPickerCh:
			stateChange := connectivity.Ready
			if stateChange = <-tcc.NewStateCh; stateChange != connectivity.Ready {
				lastError = fmt.Errorf("received state %v instead of ready", stateChange)
			}
			pickedSubConn, err := picker.Pick(balancer.PickInfo{})
			pickerDoneErr := nil
			if pickedSubConn == nil || err != nil {
				if pickedSubConn == nil && err == nil {
					lastError = fmt.Errorf("picker unexpectedly returned no sub-conn")
				} else {
					pickerDoneErr = err
				}
			} else if picker.Done != nil {
				picker.Done(balancer.DoneInfo{})
			}
			if !IsRoundRobin(expected, func() balancer.SubConn { return pickedSubConn.SubConn }) && pickerDoneErr != nil {
				lastError = pickerDoneErr
			} else if err != nil {
				lastError = err
			} else {
				return nil
			}
		}
	}
}

func (endpointsResourceType) Decode(_ *DecodeOptions, resource *anypb.Any) (*DecodeResult, error) {
	name, rc, err := unmarshalEndpointsResource(resource)
	switch {
	case name == "":
		// Name is unset only when protobuf deserialization fails.
		return nil, err
	case err != nil:
		// Protobuf deserialization succeeded, but resource validation failed.
		return &DecodeResult{Name: name, Resource: &EndpointsResourceData{Resource: EndpointsUpdate{}}}, err
	}

	return &DecodeResult{Name: name, Resource: &EndpointsResourceData{Resource: rc}}, nil

}

func TestGetHead(t *testing.T) {
	r := chi.NewRouter()
	r.Use(GetHead)
	r.Get("/hi", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Test", "yes")
		w.Write([]byte("bye"))
	})
	r.Route("/articles", func(r chi.Router) {
		r.Get("/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := chi.URLParam(r, "id")
			w.Header().Set("X-Article", id)
			w.Write([]byte("article:" + id))
		})
	})
	r.Route("/users", func(r chi.Router) {
		r.Head("/{id}", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-User", "-")
			w.Write([]byte("user"))
		})
		r.Get("/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := chi.URLParam(r, "id")
			w.Header().Set("X-User", id)
			w.Write([]byte("user:" + id))
		})
	})

	ts := httptest.NewServer(r)
	defer ts.Close()

	if _, body := testRequest(t, ts, "GET", "/hi", nil); body != "bye" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/hi", nil); body != "" || req.Header.Get("X-Test") != "yes" {
		t.Fatalf(body)
	}
	if _, body := testRequest(t, ts, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/", nil); body != "" || req.StatusCode != 404 {
		t.Fatalf(body)
	}

	if _, body := testRequest(t, ts, "GET", "/articles/5", nil); body != "article:5" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/articles/5", nil); body != "" || req.Header.Get("X-Article") != "5" {
		t.Fatalf("expecting X-Article header '5' but got '%s'", req.Header.Get("X-Article"))
	}

	if _, body := testRequest(t, ts, "GET", "/users/1", nil); body != "user:1" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/users/1", nil); body != "" || req.Header.Get("X-User") != "-" {
		t.Fatalf("expecting X-User header '-' but got '%s'", req.Header.Get("X-User"))
	}
}

func (s) CheckDatabaseConnectionTimeout(t *testing.T) {
	const maxAttempts = 3

	var want []time.Duration
	for i := 0; i < maxAttempts; i++ {
		want = append(want, time.Duration(i+1)*time.Second)
	}

	var got []time.Duration
	newQuery := func(string) (any, error) {
		if len(got) < maxAttempts {
			return nil, errors.New("timeout")
		}
		return nil, nil
	}

	oldTimeoutFunc := timeoutFunc
	timeoutFunc = func(_ context.Context, attempts int) bool {
		got = append(got, time.Duration(attempts+1)*time.Second)
		return true
	}
	defer func() { timeoutFunc = oldTimeoutFunc }()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	databaseConnectionCheck(ctx, newQuery, func(connectivity.State, error) {}, "test")

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Timeout durations for %v attempts are %v. (expected: %v)", maxAttempts, got, want)
	}
}

