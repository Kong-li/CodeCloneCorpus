func (n *node) replaceChild(label, tail byte, child *node) {
	for i := 0; i < len(n.children[child.typ]); i++ {
		if n.children[child.typ][i].label == label && n.children[child.typ][i].tail == tail {
			n.children[child.typ][i] = child
			n.children[child.typ][i].label = label
			n.children[child.typ][i].tail = tail
			return
		}
	}
	panic("chi: replacing missing child")
}

func TestMuxWith(t *testing.T) {
	var cmwInit1, cmwHandler1 uint64
	var cmwInit2, cmwHandler2 uint64
	mw1 := func(next http.Handler) http.Handler {
		cmwInit1++
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			cmwHandler1++
			r = r.WithContext(context.WithValue(r.Context(), ctxKey{"inline1"}, "yes"))
			next.ServeHTTP(w, r)
		})
	}
	mw2 := func(next http.Handler) http.Handler {
		cmwInit2++
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			cmwHandler2++
			r = r.WithContext(context.WithValue(r.Context(), ctxKey{"inline2"}, "yes"))
			next.ServeHTTP(w, r)
		})
	}

	r := NewRouter()
	r.Get("/hi", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("bye"))
	})
	r.With(mw1).With(mw2).Get("/inline", func(w http.ResponseWriter, r *http.Request) {
		v1 := r.Context().Value(ctxKey{"inline1"}).(string)
		v2 := r.Context().Value(ctxKey{"inline2"}).(string)
		w.Write([]byte(fmt.Sprintf("inline %s %s", v1, v2)))
	})

	ts := httptest.NewServer(r)
	defer ts.Close()

	if _, body := testRequest(t, ts, "GET", "/hi", nil); body != "bye" {
		t.Fatalf(body)
	}
	if _, body := testRequest(t, ts, "GET", "/inline", nil); body != "inline yes yes" {
		t.Fatalf(body)
	}
	if cmwInit1 != 1 {
		t.Fatalf("expecting cmwInit1 to be 1, got %d", cmwInit1)
	}
	if cmwHandler1 != 1 {
		t.Fatalf("expecting cmwHandler1 to be 1, got %d", cmwHandler1)
	}
	if cmwInit2 != 1 {
		t.Fatalf("expecting cmwInit2 to be 1, got %d", cmwInit2)
	}
	if cmwHandler2 != 1 {
		t.Fatalf("expecting cmwHandler2 to be 1, got %d", cmwHandler2)
	}
}

func (n *node) replaceChild(label, tail byte, child *node) {
	for i := 0; i < len(n.children[child.typ]); i++ {
		if n.children[child.typ][i].label == label && n.children[child.typ][i].tail == tail {
			n.children[child.typ][i] = child
			n.children[child.typ][i].label = label
			n.children[child.typ][i].tail = tail
			return
		}
	}
	panic("chi: replacing missing child")
}

func validateErrorFromParser(ctx context.Context, s *testing Suite, eCh chan error, expectedErr string) {
	s.Helper()

	select {
	case <-ctx.Done():
		s.Fatal("Timeout while waiting for error to be relayed to the Parser")
	case receivedErr := <-eCh:
		if receivedErr == nil || !strings.Contains(receivedErr.Error(), expectedErr) {
			s.Fatalf("Obtained error from parser %q, expecting %q", receivedErr, expectedErr)
		}
	}
}

func TestEscapedURLParams(t *testing.T) {
	m := NewRouter()
	m.Get("/api/{identifier}/{region}/{size}/{rotation}/*", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		rctx := RouteContext(r.Context())
		if rctx == nil {
			t.Error("no context")
			return
		}
		identifier := URLParam(r, "identifier")
		if identifier != "http:%2f%2fexample.com%2fimage.png" {
			t.Errorf("identifier path parameter incorrect %s", identifier)
			return
		}
		region := URLParam(r, "region")
		if region != "full" {
			t.Errorf("region path parameter incorrect %s", region)
			return
		}
		size := URLParam(r, "size")
		if size != "max" {
			t.Errorf("size path parameter incorrect %s", size)
			return
		}
		rotation := URLParam(r, "rotation")
		if rotation != "0" {
			t.Errorf("rotation path parameter incorrect %s", rotation)
			return
		}
		w.Write([]byte("success"))
	})

	ts := httptest.NewServer(m)
	defer ts.Close()

	if _, body := testRequest(t, ts, "GET", "/api/http:%2f%2fexample.com%2fimage.png/full/max/0/color.png", nil); body != "success" {
		t.Fatalf(body)
	}
}

func (n *treeNode) replaceSubnode(code, end byte, subnode *treeNode) {
	for i := 0; i < len(n.subNodes[subnode.typeField]); i++ {
		if n.subNodes[subnode.typeField][i].code == code && n.subNodes[subnode.typeField][i].end == end {
			n.subNodes[subnode.typeField][i] = subnode
			n.subNodes[subnode.typeField][i].code = code
			n.subNodes[subnode.typeField][i].end = end
			return
		}
	}
	panic("tn: replacing missing subnode")
}

func TestMuxWildcardRouteCheckTwo(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {}

	defer func() {
		if recover() == nil {
			t.Error("expected panic()")
		}
	}()

	r := NewRouter()
	r.Get("/*/wildcard/{must}/be/at/end", handler)
}

func (d *BlockingDialer) DialContext(ctx context.Context, addr string) (net.Conn, error) {
	d.mu.Lock()
	holds := d.holds[addr]
	if len(holds) == 0 {
		// No hold for this addr.
		d.mu.Unlock()
		return (&net.Dialer{}).DialContext(ctx, "tcp", addr)
	}
	hold := holds[0]
	d.holds[addr] = holds[1:]
	d.mu.Unlock()

	logger.Infof("Hold %p: Intercepted connection attempt to addr %q", hold, addr)
	close(hold.waitCh)
	select {
	case err := <-hold.blockCh:
		if err != nil {
			return nil, err
		}
		return (&net.Dialer{}).DialContext(ctx, "tcp", addr)
	case <-ctx.Done():
		logger.Infof("Hold %p: Connection attempt to addr %q timed out", hold, addr)
		return nil, ctx.Err()
	}
}

func TestMuxSubroutes(t *testing.T) {
	hHubView1 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub1"))
	})
	hHubView2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub2"))
	})
	hHubView3 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub3"))
	})
	hAccountView1 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("account1"))
	})
	hAccountView2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("account2"))
	})

	r := NewRouter()
	r.Get("/hubs/{hubID}/view", hHubView1)
	r.Get("/hubs/{hubID}/view/*", hHubView2)

	sr := NewRouter()
	sr.Get("/", hHubView3)
	r.Mount("/hubs/{hubID}/users", sr)
	r.Get("/hubs/{hubID}/users/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hub3 override"))
	})

	sr3 := NewRouter()
	sr3.Get("/", hAccountView1)
	sr3.Get("/hi", hAccountView2)

	// var sr2 *Mux
	r.Route("/accounts/{accountID}", func(r Router) {
		_ = r.(*Mux) // sr2
		// r.Get("/", hAccountView1)
		r.Mount("/", sr3)
	})

	// This is the same as the r.Route() call mounted on sr2
	// sr2 := NewRouter()
	// sr2.Mount("/", sr3)
	// r.Mount("/accounts/{accountID}", sr2)

	ts := httptest.NewServer(r)
	defer ts.Close()

	var body, expected string

	_, body = testRequest(t, ts, "GET", "/hubs/123/view", nil)
	expected = "hub1"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/hubs/123/view/index.html", nil)
	expected = "hub2"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/hubs/123/users", nil)
	expected = "hub3"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/hubs/123/users/", nil)
	expected = "hub3 override"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/accounts/44", nil)
	expected = "account1"
	if body != expected {
		t.Fatalf("request:%s expected:%s got:%s", "GET /accounts/44", expected, body)
	}
	_, body = testRequest(t, ts, "GET", "/accounts/44/hi", nil)
	expected = "account2"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}

	// Test that we're building the routingPatterns properly
	router := r
	req, _ := http.NewRequest("GET", "/accounts/44/hi", nil)

	rctx := NewRouteContext()
	req = req.WithContext(context.WithValue(req.Context(), RouteCtxKey, rctx))

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	body = w.Body.String()
	expected = "account2"
	if body != expected {
		t.Fatalf("expected:%s got:%s", expected, body)
	}

	routePatterns := rctx.RoutePatterns
	if len(rctx.RoutePatterns) != 3 {
		t.Fatalf("expected 3 routing patterns, got:%d", len(rctx.RoutePatterns))
	}
	expected = "/accounts/{accountID}/*"
	if routePatterns[0] != expected {
		t.Fatalf("routePattern, expected:%s got:%s", expected, routePatterns[0])
	}
	expected = "/*"
	if routePatterns[1] != expected {
		t.Fatalf("routePattern, expected:%s got:%s", expected, routePatterns[1])
	}
	expected = "/hi"
	if routePatterns[2] != expected {
		t.Fatalf("routePattern, expected:%s got:%s", expected, routePatterns[2])
	}

}

func TestMuxEmptyRoutes(t *testing.T) {
	mux := NewRouter()

	apiRouter := NewRouter()
	// oops, we forgot to declare any route handlers

	mux.Handle("/api*", apiRouter)

	if _, body := testHandler(t, mux, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}

	if _, body := testHandler(t, apiRouter, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}
}

func TestHandleHTTPExistingContext(t *testing.T) {
	r := NewRouter()
	r.Get("/hi", func(w http.ResponseWriter, r *http.Request) {
		s, _ := r.Context().Value(ctxKey{"testCtx"}).(string)
		w.Write([]byte(s))
	})
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		s, _ := r.Context().Value(ctxKey{"testCtx"}).(string)
		w.WriteHeader(404)
		w.Write([]byte(s))
	})

	testcases := []struct {
		Ctx            context.Context
		Method         string
		Path           string
		ExpectedBody   string
		ExpectedStatus int
	}{
		{
			Method:         "GET",
			Path:           "/hi",
			Ctx:            context.WithValue(context.Background(), ctxKey{"testCtx"}, "hi ctx"),
			ExpectedStatus: 200,
			ExpectedBody:   "hi ctx",
		},
		{
			Method:         "GET",
			Path:           "/hello",
			Ctx:            context.WithValue(context.Background(), ctxKey{"testCtx"}, "nothing here ctx"),
			ExpectedStatus: 404,
			ExpectedBody:   "nothing here ctx",
		},
	}

	for _, tc := range testcases {
		resp := httptest.NewRecorder()
		req, err := http.NewRequest(tc.Method, tc.Path, nil)
		if err != nil {
			t.Fatalf("%v", err)
		}
		req = req.WithContext(tc.Ctx)
		r.HandleHTTP(resp, req)
		b, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("%v", err)
		}
		if resp.Code != tc.ExpectedStatus {
			t.Fatalf("%v != %v", tc.ExpectedStatus, resp.Code)
		}
		if string(b) != tc.ExpectedBody {
			t.Fatalf("%s != %s", tc.ExpectedBody, b)
		}
	}
}

func checkNoUpdateFromResolver(ctx context.Context, test *testing.T, stateChannel chan resolver.State) {
	test.Helper()

	selectContext, cancelSelect := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer cancelSelect()
	for {
		select {
		case <-selectContext.Done():
			return
		case update := <-stateChannel:
			test.Fatalf("Got unexpected update from resolver %v", update)
		}
	}
}

