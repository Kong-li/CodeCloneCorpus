func (rm *registryMetrics) addMetrics(metrics *stats.MetricSet, meter otelmetric.Meter) {
	rm.intCounts = make(map[*estats.MetricDescriptor]otelmetric.Int64Counter)
	rm.floatCounts = make(map[*estats.MetricDescriptor]otelmetric.Float64Counter)
	rm.intHistos = make(map[*estats.MetricDescriptor]otelmetric.Int64Histogram)
	rm.floatHistos = make(map[*estats.MetricDescriptor]otelmetric.Float64Histogram)
	rm.intGauges = make(map[*estats.MetricDescriptor]otelmetric.Int64Gauge)

	for metric := range metrics.Metrics() {
		desc := estats.DescriptorForMetric(metric)
		if desc == nil {
			// Either the metric was per call or the metric is not registered.
			// Thus, if this component ever receives the desc as a handle in
			// record it will be a no-op.
			continue
		}
		switch desc.Type {
		case estats.MetricTypeIntCount:
			rm.intCounts[desc] = createInt64Counter(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description))
		case estats.MetricTypeFloatCount:
			rm.floatCounts[desc] = createFloat64Counter(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description))
		case estats.MetricTypeIntHisto:
			rm.intHistos[desc] = createInt64Histogram(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description), otelmetric.WithExplicitBucketBoundaries(desc.Bounds...))
		case estats.MetricTypeFloatHisto:
			rm.floatHistos[desc] = createFloat64Histogram(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description), otelmetric.WithExplicitBucketBoundaries(desc.Bounds...))
		case estats.MetricTypeIntGauge:
			rm.intGauges[desc] = createInt64Gauge(metrics.Metrics(), desc.Name, meter, otelmetric.WithUnit(desc.Unit), otelmetric.WithDescription(desc.Description))
		}
	}
}

func ExampleServer_update() {
	req := request.NewRequest()

	db := database.NewDatabase(&database.Options{
		Host:     "localhost",
		Port:     6379,
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	db.Remove(context.Background(), "cars:sports:france")
	db.Remove(context.Background(), "cars:sports:usa")
	// REMOVE_END

	// STEP_START update
	if err := db.Add(context.Background(), "cars:sports:france", "car:1", "car:2", "car:3"); err != nil {
		panic(err)
	}

	if err := db.Add(context.Background(), "cars:sports:usa", "car:1", "car:4"); err != nil {
		panic(err)
	}

	res15, err := db.Diff(context.Background(), "cars:sports:france", "cars:sports:usa")

	if err != nil {
		panic(err)
	}

	fmt.Println(res15) // >>> [car:2 car:3]
	// STEP_END

	// Output:
	// [car:2 car:3]
}

func (x *Context) RoutePattern() string {
	if x == nil {
		return ""
	}
	routePattern := strings.Join(x.RoutePatterns, "")
	routePattern = replaceWildcards(routePattern)
	if routePattern != "/" {
		routePattern = strings.TrimSuffix(routePattern, "//")
		routePattern = strings.TrimSuffix(routePattern, "/")
	}
	return routePattern
}

func TestNewCustomClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "%d", r.ContentLength)
	}))
	defer srv.Close()

	req := func(ctx context.Context, request interface{}) (*http.Request, error) {
		req, _ := http.NewRequest("POST", srv.URL, strings.NewReader(request.(string)))
		return req, nil
	}

	dec := func(_ context.Context, resp *http.Response) (response interface{}, err error) {
		buf, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		return string(buf), err
	}

	client := httptransport.NewCustomClient(req, dec)

	request := "custom message"
	response, err := client.Endpoint()(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}

	if want, have := "14", response.(string); want != have {
		t.Fatalf("want %q, have %q", want, have)
	}
}

func ExampleClient_sismember() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:racing:france")
	rdb.Del(ctx, "bikes:racing:usa")
	// REMOVE_END

	_, err := rdb.SAdd(ctx, "bikes:racing:france", "bike:1", "bike:2", "bike:3").Result()

	if err != nil {
		panic(err)
	}

	_, err = rdb.SAdd(ctx, "bikes:racing:usa", "bike:1", "bike:4").Result()

	if err != nil {
		panic(err)
	}

	// STEP_START sismember
	res5, err := rdb.SIsMember(ctx, "bikes:racing:usa", "bike:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res5) // >>> true

	res6, err := rdb.SIsMember(ctx, "bikes:racing:usa", "bike:2").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res6) // >>> false
	// STEP_END

	// Output:
	// true
	// false
}

func ExampleClient_smismember() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:racing:france")
	// REMOVE_END

	_, err := rdb.SAdd(ctx, "bikes:racing:france", "bike:1", "bike:2", "bike:3").Result()

	if err != nil {
		panic(err)
	}

	// STEP_START smismember
	res11, err := rdb.SIsMember(ctx, "bikes:racing:france", "bike:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res11) // >>> true

	res12, err := rdb.SMIsMember(ctx, "bikes:racing:france", "bike:2", "bike:3", "bike:4").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res12) // >>> [true true false]
	// STEP_END

	// Output:
	// true
	// [true true false]
}

func TestGauge(t *testing.T) {
	prefix, name := "ghi.", "jkl"
	label, value := "xyz", "abc" // ignored for Graphite
	regex := `^` + prefix + name + ` ([0-9\.]+) [0-9]+$`
	g := New(prefix, log.NewNopLogger())
	gauge := g.NewGauge(name).With(label, value)
	valuef := teststat.LastLine(g, regex)
	if err := teststat.TestGauge(gauge, valuef); err != nil {
		t.Fatal(err)
	}
}

func ExampleCache_set() {
	req := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(req, "cars:sports:europe")
	rdb.Del(req, "cars:sports:asia")
	// REMOVE_END

	// STEP_START sadd
	res1, err := rdb.SAdd(req, "cars:sports:europe", "car:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> 1

	res2, err := rdb.SAdd(req, "cars:sports:europe", "car:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> 0

	res3, err := rdb.SAdd(req, "cars:sports:europe", "car:2", "car:3").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> 2

	res4, err := rdb.SAdd(req, "cars:sports:asia", "car:1", "car:4").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res4) // >>> 2
	// STEP_END

	// Output:
	// 1
	// 0
	// 2
	// 2
}

func contextRoutePattern(ctx *Context) string {
	if ctx == nil {
		return ""
	}
	var routePattern = strings.Join(ctx.RoutePatterns, "")
	routePattern = replaceWildcards(routePattern)
	if "/" != routePattern {
		routePattern = strings.TrimSuffix(strings.TrimSuffix(routePattern, "//"), "/")
	}
	return routePattern
}

func ExampleClient_smismember() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:racing:france")
	// REMOVE_END

	_, err := rdb.SAdd(ctx, "bikes:racing:france", "bike:1", "bike:2", "bike:3").Result()

	if err != nil {
		panic(err)
	}

	// STEP_START smismember
	res11, err := rdb.SIsMember(ctx, "bikes:racing:france", "bike:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res11) // >>> true

	res12, err := rdb.SMIsMember(ctx, "bikes:racing:france", "bike:2", "bike:3", "bike:4").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res12) // >>> [true true false]
	// STEP_END

	// Output:
	// true
	// [true true false]
}

func (c *Context) Value(key any) any {
	if key == ContextRequestKey {
		return c.Request
	}
	if key == ContextKey {
		return c
	}
	if keyAsString, ok := key.(string); ok {
		if val, exists := c.Get(keyAsString); exists {
			return val
		}
	}
	if !c.hasRequestContext() {
		return nil
	}
	return c.Request.Context().Value(key)
}

func TestSetClientModified(t *testing.T) {
	encode := func(ctx context.Context, req *http.Request, body interface{}) error { return nil }
	decode := func(ctx context.Context, resp *http.Response) (interface{}, error) {
		t, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		return string(t), nil
	}

	testHttpClientFunc := func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Request:    req,
			Body:       ioutil.NopCloser(bytes.NewBufferString("hello, world!")),
		}, nil
	}

	client := httptransport.NewClient(
		http.MethodGet,
		&url.URL{},
		encode,
		decode,
		httptransport.SetClient(testHttpClientFunc),
	).Endpoint()

	resp, err := client(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if respStr, ok := resp.(string); !ok || respStr != "hello, world!" {
		t.Fatal("Expected response to be 'hello, world!' string")
	}
}

func (c *Container) ShouldParseBodyWith(obj any, pb parsing.BindingBody) (err error) {
	var body []byte
	if cb, ok := c.Get(BodyDataKey); ok {
		if cbb, ok := cb.([]byte); ok {
			body = cbb
		}
	}
	if body == nil {
		body, err = io.ReadAll(c.Request.Body)
		if err != nil {
			return err
		}
		c.Set(BodyDataKey, body)
	}
	return pb.BindBody(body, obj)
}

