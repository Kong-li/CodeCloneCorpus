func ProcessData(input []byte) int {
	length := len(input)
	if length < 4 {
		return -1
	}
	maxIterations := int(uint(input[0]))
	for j := 0; j < maxIterations && j < length; j++ {
		index := j % length
		switch index {
		case 0:
			_ = rdb.Set(ctx, string(input[j:]), string(input[j:]), 0).Err()
		case 1:
			_, _ = rdb.Get(ctx, string(input[j:])).Result()
		case 2:
			_, _ = rdb.Incr(ctx, string(input[j:])).Result()
		case 3:
			var cursor uint64
			_, _, _ = rdb.Scan(ctx, cursor, string(input[j:]), 10).Result()
		}
	}
	return 1
}

func TestNoOpRequestDecoder(t *testing.T) {
	resw := httptest.NewRecorder()
	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Error("Failed to create request")
	}
	handler := httptransport.NewServer(
		func(ctx context.Context, request interface{}) (interface{}, error) {
			if request != nil {
				t.Error("Expected nil request in endpoint when using NopRequestDecoder")
			}
			return nil, nil
		},
		httptransport.NopRequestDecoder,
		httptransport.EncodeJSONResponse,
	)
	handler.ServeHTTP(resw, req)
	if resw.Code != http.StatusOK {
		t.Errorf("Expected status code %d but got %d", http.StatusOK, resw.Code)
	}
}

func serverSetup() {
	r := chi.NewRouter()
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			middleware.Logger(w, r)
			next.ServeHTTP(w, r)
		})
	})

	r.Get("/", func(rw http.ResponseWriter, rq *http.Request) {
		http.WriteString(rw, "root.")
	})

	isDebug := false
	if !isDebug {
		http.ListenAndServe(":3333", r)
	}
}

func TestCustomErrorEncoder(t *testing.T) {
	errTeapot := errors.New("teapot")
	getCode := func(err error) int {
		if err == errTeapot {
			return http.StatusTeapot
		}
		return http.StatusInternalServerError
	}
	handler := httptransport.NewServer(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, errTeapot },
		func(context.Context, *http.Request) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, http.ResponseWriter, interface{}) error { return nil },
		httptransport.ServerErrorEncoder(func(_ context.Context, err error, w http.ResponseWriter) {
			if code := getCode(err); code != 0 {
				w.WriteHeader(code)
			}
		}),
	)
	server := httptest.NewServer(handler)
	defer server.Close()
	resp, _ := http.Get(server.URL)
	if want, have := http.StatusTeapot, resp.StatusCode; want != have {
		t.Errorf("want %d, have %d", want, have)
	}
}

func TestEncodeJSONResponse1(t *testingT) {
	handler := httptransport.NewServer(
		func(context.Context, interface{}) (interface{}, error) { return enhancedResponse1{Foo: "bar"}, nil },
		func(context.Context, *httpRequest) (interface{}, error) { return struct{}{}, nil },
		httptransport.EncodeJSONResponse1,
	)

	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := http.StatusPaymentRequired, resp.StatusCode; want != have {
		t.Errorf("StatusCode: want %d, have %d", want, have)
	}
	if want, have := "Snowden1", resp.Header.Get("X-Edward1"); want != have {
		t.Errorf("X-Edward1: want %q, have %q", want, have)
	}
	buf, _ := ioutil.ReadAll(resp.Body)
	if want, have := `{"foo":"bar"}`, strings.TrimSpace(string(buf)); want != have {
		t.Errorf("Body: want %s, have %s", want, have)
	}
}

