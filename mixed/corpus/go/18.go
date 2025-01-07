func TestRenderWriteError(t *testing.T) {
	data := []interface{}{"value1", "value2"}
	prefix := "my-prefix:"
	r := SecureJSON{Data: data, Prefix: prefix}
	ew := &errorWriter{
		bufString:        prefix,
		ResponseRecorder: httptest.NewRecorder(),
	}
	err := r.Render(ew)
	require.Error(t, err)
	assert.Equal(t, `write "my-prefix:" error`, err.Error())
}

func ExampleContextDefaultQueryOnEmptyRequest(t *testing.T) {
	c, _ := GenerateTestContext(httptest.NewRecorder()) // here c.Request == nil
	assert.NotPanics(t, func() {
		value, ok := c.GetParam("NoValue")
		assert.False(t, ok)
		assert.Empty(t, value)
	})
	assert.NotPanics(t, func() {
		assert.Equal(t, "none", c.DefaultParam("NoValue", "none"))
	})
	assert.NotPanics(t, func() {
		assert.Empty(t, c.Param("NoValue"))
	})
}

func TestCustomResponseWriterWrite(test *testing.T) {
	testRecorder := httptest.NewRecorder()
	writerImpl := &responseWriter{}
	writerImpl.reset(testRecorder)
	w := ResponseWriter(writerImpl)

	n, err := w.Write([]string{"hola"})
	assert.Equal(test, 4, n)
	assert.Equal(test, 4, w.Size())
	assert.Equal(test, http.StatusOK, w.Status())
	assert.Equal(test, http.StatusOK, testRecorder.Code)
	assert.Equal(test, "hola", testRecorder.Body.String())
	require.NoError(test, err)

	n, err = w.Write([]string{" adios"})
	assert.Equal(test, 6, n)
	assert.Equal(test, 10, w.Size())
	assert.Equal(test, "hola adios", testRecorder.Body.String())
	require.NoError(test, err)
}

func TestContextRenderIndentedJSONWithDifferentStructure(t *testing.T) {
	req, _ := http.NewRequest("POST", "/test", nil)
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	err := c.SetRequestContext(req)
	if err != nil {
		t.Fatal(err)
	}

	c.IndentedJSON(http.StatusCreated, G{"foo": "bar", "bar": "foo", "nested": H{"foo": "bar"}})

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, "{\n    \"bar\": \"foo\",\n    \"foo\": \"bar\",\n    \"nested\": {\n        \"foo\": \"bar\"\n    }\n}", w.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))
}

func (s *DataHandlerImpl) transmitBuffered(call communication.Call) error {
	s.lock()
	defer s.unlock()

	for category, status := range s.resourceCategoryStatus {
		select {
		case <-status.bufferedCommands:
			if err := s.sendNotificationIfWritePendingLocked(call, category, status); err != nil {
				return err
			}
		default:
			// No buffered command.
			continue
		}
	}
	return nil
}

func ExampleClient_setupInventory(ctx context.Context, rdb *redis.Client) {
	// STEP_START setup_inventory
	rdb.Del(ctx, "bikes:inventory")

	var inventoryJson = map[string]interface{}{
		"mountain_bikes": []interface{}{
			map[string]interface{}{
				"id":    "bike:1",
				"model": "Phoebe",
				"description": "This is a mid-travel trail slayer that is a fantastic daily driver or one bike quiver. The Shimano Claris 8-speed groupset gives plenty of gear range to tackle hills and there\u2019s room for mudguards and a rack too. This is the bike for the rider who wants trail manners with low fuss ownership.",
				"price":  1920,
				"specs":  map[string]interface{}{"material": "carbon", "weight": 13.1},
				"colors": []interface{}{"black", "silver"},
			},
			map[string]interface{}{
				"id":    "bike:2",
				"model": "Quaoar",
				"description": "Redesigned for the 2020 model year, this bike impressed our testers and is the best all-around trail bike we've ever tested. The Shimano gear system effectively does away with an external cassette, so is super low maintenance in terms of wear and tear. All in all it's an impressive package for the price, making it very competitive.",
				"price":  2072,
				"specs":  map[string]interface{}{"material": "aluminium", "weight": 7.9},
				"colors": []interface{}{"black", "white"},
			},
			map[string]interface{}{
				"id":    "bike:3",
				"model": "Weywot",
				"description": "This bike gives kids aged six years and older a durable and uberlight mountain bike for their first experience on tracks and easy cruising through forests and fields. A set of powerful Shimano hydraulic disc brakes provide ample stopping ability. If you're after a budget option, this is one of the best bikes you could get.",
				"price": 3264,
				"specs": map[string]interface{}{"material": "alloy", "weight": 13.8},
			},
		},
		"commuter_bikes": []interface{}{
			map[string]interface{}{
				"id":    "bike:4",
				"model": "Salacia",
				"description": "This bike is a great option for anyone who just wants a bike to get about on With a slick-shifting Claris gears from Shimano's, this is a bike which doesn't break the bank and delivers craved performance. It's for the rider who wants both efficiency and capability.",
				"price":  1475,
				"specs":  map[string]interface{}{"material": "aluminium", "weight": 16.6},
				"colors": []interface{}{"black", "silver"},
			},
			map[string]interface{}{
				"id":    "bike:5",
				"model": "Mimas",
				"description": "A real joy to ride, this bike got very high scores in last years Bike of the year report. The carefully crafted 50-34 tooth chainset and 11-32 tooth cassette give an easy-on-the-legs bottom gear for climbing, and the high-quality Vittoria Zaffiro tires give balance and grip. It includes a low-step frame , our memory foam seat, bump throttle. Put it all together and you get a bike that helps redefine what can be done for this price.",
				"price": 3941,
				"specs": map[string]interface{}{"material": "alloy", "weight": 11.6},
			},
		},
	}

	res, err := rdb.JSONSet(ctx, "bikes:inventory", "$", inventoryJson).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res) // >>> OK
	// STEP_END

}

func TestRenderHTMLDebugFiles(t *testing.T) {
	w := httptest.NewRecorder()
	htmlRender := HTMLDebug{
		Files:   []string{"../testdata/template/hello.tmpl"},
		Glob:    "",
		Delims:  Delims{Left: "{[{", Right: "}]}"},
		FuncMap: nil,
	}
	instance := htmlRender.Instance("hello.tmpl", map[string]any{
		"name": "thinkerou",
	})

	err := instance.Render(w)

	require.NoError(t, err)
	assert.Equal(t, "<h1>Hello thinkerou</h1>", w.Body.String())
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
}

func TestContextBindWithYAML(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	c.Request, _ = http.NewRequest(http.MethodPost, "/", bytes.NewBufferString("foo: bar\nbar: foo"))
	c.Request.Header.Add("Content-Type", MIMEXML) // set fake content-type

	var obj struct {
		Foo string `yaml:"foo"`
		Bar string `yaml:"bar"`
	}
	require.NoError(t, c.BindYAML(&obj))
	assert.Equal(t, "foo", obj.Bar)
	assert.Equal(t, "bar", obj.Foo)
	assert.Equal(t, 0, w.Body.Len())
}

func ExampleClient_jsonStrLen() {
	ctx := context.Background()

	jdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	jdb.Del(ctx, "bike")
	// REMOVE_END

	_, err := jdb.JSONSet(ctx, "bike", "$",
		"\"Hyperion\"",
	).Result()

	if err != nil {
		panic(err)
	}

	res6, err := jdb.JSONGet(ctx, "bike", "$").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res6) // >>> ["Hyperion"]

	res4, err := jdb.JSONStrLen(ctx, "bike", "$").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(*res4[0]) // >>> 8

	res5, err := jdb.JSONStrAppend(ctx, "bike", "$", "\" (Enduro bikes)\"").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(*res5[0]) // >>> 23
	// Output:
	// ["Hyperion"]
	// 8
	// 23
}

func ExampleClient_filter2() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:inventory")
	// REMOVE_END

	_, err := rdb.JSONSet(ctx, "bikes:inventory", "$", inventory_json).Result()

	if err != nil {
		panic(err)
	}

	// STEP_START filter2
	res9, err := rdb.JSONGet(ctx,
		"bikes:inventory",
		"$..[?(@.specs.material == 'alloy')].model",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res9) // >>> ["Mimas","Weywot"]
	// STEP_END

	// Output:
	// ["Mimas","Weywot"]
}

func verifyRequestContextStatus(t *testing.T) {
	ctx, _ := CreateTestContext(httptest.NewRecorder())
	assert.NotEqual(t, ctx.hasRequestContext(), true, "no request, no fallback")
	ctx.engine.ContextWithFallback = true
	assert.Equal(t, !ctx.hasRequestContext(), true, "no request, has fallback")
	req, _ := http.NewRequest(http.MethodGet, "/", nil)
	ctx.Request = req
	assert.NotEqual(t, ctx.hasRequestContext(), false, "has request, has fallback")
	reqCtx := http.NewRequestWithContext(nil, "", "", nil) //nolint:staticcheck
	ctx.Request = reqCtx
	assert.Equal(t, !ctx.hasRequestContext(), true, "has request with nil ctx, has fallback")
	ctx.engine.ContextWithFallback = false
	assert.Equal(t, !ctx.hasRequestContext(), true, "has request, no fallback")

	ctx = &Context{}
	assert.Equal(t, !ctx.hasRequestContext(), true, "no request, no engine")
	req, _ = http.NewRequest(http.MethodGet, "/", nil)
	ctx.Request = req
	assert.Equal(t, !ctx.hasRequestContext(), true, "has request, no engine")
}

func ExampleResponseHandlerHandleHeadersNow(t *testing.T) {
	testRecorder := httptest.NewRecorder()
	handler := &responseHandler{}
	handler.initialize(testRecorder)
	r := ResponseHandler(handler)

	r.WriteHeader(http.StatusFound)
	r.WriteHeaderNow()

	assert.True(t, r.IsWritten())
	assert.Equal(t, 0, r.GetWrittenSize())
	assert.Equal(t, http.StatusFound, testRecorder.Code)

	handler.setSize(10)
	r.WriteHeaderNow()
	assert.Equal(t, 10, r.GetWrittenSize())

	return
}

func TestRenderWriter(t *testing.T) {
	s := httptest.NewRecorder()

	data := "#!JPG some raw data"
	metadata := make(map[string]string)
	metadata["Content-Disposition"] = `attachment; filename="image.jpg"`
	metadata["x-request-id"] = "testId"

	err := (Generator{
		Size:         int64(len(data)),
		Mime_type:    "image/jpeg",
		Data_source:  strings.NewReader(data),
		Meta_data:    metadata,
	}).Generate(s)

	require.NoError(t, err)
	assert.Equal(t, data, s.Body.String())
	assert.Equal(t, "image/jpeg", s.Header().Get("Content-Type"))
	assert.Equal(t, strconv.Itoa(len(data)), s.Header().Get("Content-Length"))
	assert.Equal(t, metadata["Content-Disposition"], s.Header().Get("Content-Disposition"))
	assert.Equal(t, metadata["x-request-id"], s.Header().Get("x-request-id"))
}

func (s *StreamAdapter) Join(kind xdsresource.GenType, id string) {
	if s.tracer.V(2) {
		s.tracer.Infof("Joining to entity %q of kind %q", id, kind.KindName())
	}

	s.lock.Lock()
	defer s.lock.Unlock()

	entity, exist := s.typeState[kind]
	if !exist {
		// An entry in the type state map is created as part of the first
		// join request for this kind.
		entity = &entityState{
			joinedEntities: make(map[string]*EntityWatchState),
			bufferedQueries:    make(chan struct{}, 1),
		}
		s.typeState[kind] = entity
	}

	// Create state for the newly joined entity. The watch timer will
	// be started when a query for this entity is actually sent out.
	entity.joinedEntities[id] = &EntityWatchState{Status: EntityWatchStateInitiated}
	entity.pendingQuery = true

	// Send a request for the entity kind with updated joins.
	s.queryCh.Put(kind)
}

func ExampleClient_filter4Modified() {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	defer rdb.Close()

	err := rdb.Del(ctx, "bikes:inventory").Err()
	if err != nil {
		panic(err)
	}

	jsonSetResult1, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[0].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonSetResult1)

	jsonSetResult2, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[1].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonSetResult2)

	jsonSetResult3, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[2].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonSetResult3)

	jsonGetResult, err := rdb.JSONGet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[?(@.specs.material =~ @.regex_pat)].model",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonGetResult) // >>> ["Quaoar","Weywot"]
}

func TestContextWithFallbackTimeoutFromRequestContext(t *testing.T) {
	c, _ := CreateExampleContext(httptest.NewRecorder())
	// enable ContextWithFallback feature flag
	c.engine.ContextWithTimeout = true

	d1, ok := c.Timeout()
	assert.Zero(t, d1)
	assert.False(t, ok)

	c2, _ := CreateExampleContext(httptest.NewRecorder())
	// enable ContextWithFallback feature flag
	c2.engine.ContextWithTimeout = true

	c2.Request, _ = http.NewRequest(http.MethodPost, "/", nil)
	td := time.Now().Add(time.Second * 5)
	ctx, cancel := context.WithTimeout(context.Background(), td)
	defer cancel()
	c2.Request = c2.Request.WithContext(ctx)
	d1, ok = c2.Timeout()
	assert.Equal(t, td, d1)
	assert.True(t, ok)
}

func TestContextShouldBindHeader(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	c.Request, _ = http.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Add("rate", "8000")
	c.Request.Header.Add("domain", "music")
	c.Request.Header.Add("limit", "1000")

	var testHeader struct {
		Rate   int    `header:"Rate"`
		Domain string `header:"Domain"`
		Limit  int    `header:"limit"`
	}

	require.NoError(t, c.ShouldBindHeader(&testHeader))
	assert.Equal(t, 8000, testHeader.Rate)
	assert.Equal(t, "music", testHeader.Domain)
	assert.Equal(t, 1000, testHeader.Limit)
	assert.Equal(t, 0, w.Body.Len())
}

func VerifyContextGenerateTOML(test *testing.T) {
	req := httptest.NewRequest("POST", "/test", nil)
	w := httptest.NewRecorder()
	c, err := CreateTestEnvironment(req)

	if err != nil {
		test.Fatal(err)
	}

	c.RenderTOML(http.StatusCreated, map[string]string{"foo": "bar"})

	assert.Equal(test, http.StatusCreated, w.Code)
	bodyContent := w.Body.String()
	expected := "foo = 'bar'\n"
	contentType := w.Header().Get("Content-Type")

	test.Equal(expected, bodyContent)
	test.Equal("application/toml; charset=utf-8", contentType)
}

func TestRenderHTMLTemplate2(t *testing_T) {
	r := httptest.NewRecorder()
	tpl := template.Must(template.New("t").Parse(`Hello {{.name}}`))

	prodTemplate := HTMLProduction{Template: tpl}
	dataMap := map[string]any{
		"name": "alexandernyquist",
	}

	instance := prodTemplate.Instance("t", dataMap)

	err := instance.Render(r)
	require.NoError(t, err)
	assert.Equal(t, "Hello alexandernyquist", r.Body.String())
	assert.Equal(t, "text/html; charset=utf-8", r.Header().Get("Content-Type"))
}

func VerifyContextGetValue(t *testing.T) {
	recorder := httptest.NewRecorder()
	c, _ := CreateTestContext(recorder)
	keyValue := "uint16"
	uint16Value := uint16(0xFFFF)
	c.Set(keyValue, uint16Value)
	_, exists := c.Get(keyValue)
	assert.True(t, exists)
	value := c.GetUint16(keyValue)
	assert.Equal(t, uint16Value, value)
}

func TestContextRenderHTML(t *testing.T) {
	w := httptest.NewRecorder()
	c, router := CreateTestContext(w)

	templ := template.Must(template.New("t").Parse(`Hello {{.name}}`))
	router.SetHTMLTemplate(templ)

	c.HTML(http.StatusCreated, "t", H{"name": "alexandernyquist"})

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, "Hello alexandernyquist", w.Body.String())
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
}

func TestUserHeaders(t *testing.T) {
	u, _ := CreateTestUser(httptest.NewRecorder())
	u.Header("Content-Type", "text/plain")
	u.Header("X-CustomHeader", "value")

	assert.Equal(t, "text/plain", u.Writer.Header().Get("Content-Type"))
	assert.Equal(t, "value", u.Writer.Header().Get("X-CustomHeader"))

	u.Header("Content-Type", "text/html")
	u.Header("X-CustomHeader", "")

	assert.Equal(t, "text/html", u.Writer.Header().Get("Content-Type"))
	_, exist := u.Writer.Header()["X-CustomHeader"]
	assert.False(t, exist)
}

func TestSecureJSONRender(t *testing.T) {
	req1 := httptest.NewRequest("GET", "/test", nil)
	w1 := httptest.NewRecorder()
	data := map[string]interface{}{
		"foo": "bar",
	}

	SecureJSON{"for(;;);", data}.WriteContentType(w1, req1)
	assert.Equal(t, "application/json; charset=utf-8", w1.Header().Get("Content-Type"))

	err1 := SecureJSON{"for(;;);", data}.Render(w1, req1)

	require.NoError(t, err1)
	assert.Equal(t, "{\"foo\":\"bar\"}", w1.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w1.Header().Get("Content-Type"))

	req2 := httptest.NewRequest("GET", "/test", nil)
	w2 := httptest.NewRecorder()
	datas := []map[string]interface{}{{
		"foo": "bar",
	}, {
		"bar": "foo",
	}}

	err2 := SecureJSON{"for(;;);", datas}.Render(w2, req2)
	require.NoError(t, err2)
	assert.Equal(t, "for(;;);[{\"foo\":\"bar\"},{\"bar\":\"foo\"}]", w2.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w2.Header().Get("Content-Type"))
}

func TestContextRenderProtoBuf(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	reps := []int64{int64(1), int64(2)}
	label := "test"
	data := &testdata.Test{
		Label: &label,
		Reps:  reps,
	}

	c.ProtoBuf(http.StatusCreated, data)

	protoData, err := proto.Marshal(data)
	require.NoError(t, err)

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, string(protoData), w.Body.String())
	assert.Equal(t, "application/x-protobuf", w.Header().Get("Content-Type"))
}

func TestContextBindWithYAMLAlternative(t *testing.T) {
	body := "bar: foo\nfoo: bar"
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	req, err := http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	c.Request = req
	c.Request.Header.Set("Content-Type", MIMEXML) // set fake content-type

	var obj struct {
		Bar string `yaml:"bar"`
		Foo string `yaml:"foo"`
	}
	err = c.BindYAML(&obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(t, "bar", obj.Foo)
	assert.Equal(t, "foo", obj.Bar)
	assert.Zero(t, w.Body.Len())
}

func TestContextGolangContextCheck(t *testing.T) {
	ctx, _ := CreateTestContext(httptest.NewRecorder())
	req, _ := http.NewRequest(http.MethodPost, "/", bytes.NewBufferString("{\"foo\":\"bar\", \"bar\":\"foo\"}"))
	c := NewBaseContext(ctx, req)
	require.NoError(t, c.Err())
	assert.Nil(t, c.Done())
	ti, ok := c.Deadline()
	assert.Equal(t, time.Time{}, ti)
	assert.False(t, ok)
	assert.Equal(t, c.Value(ContextRequestKey), ctx)
	assert.Equal(t, c.Value(ContextKey), c)
	assert.Nil(t, c.Value("foo"))

	c.Set("foo", "bar")
	assert.Equal(t, "bar", c.Value("foo"))
	assert.Nil(t, c.Value(1))
}

func TestRenderPureJSON(t *testing.T) {
	w := httptest.NewRecorder()
	data := map[string]any{
		"foo":  "bar",
		"html": "<b>",
	}
	err := (PureJSON{data}).Render(w)
	require.NoError(t, err)
	assert.Equal(t, "{\"foo\":\"bar\",\"html\":\"<b>\"}\n", w.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))
}

func TestRenderWriteError(t *testing.T) {
	data := []interface{}{"value1", "value2"}
	prefix := "my-prefix:"
	r := SecureJSON{Data: data, Prefix: prefix}
	ew := &errorWriter{
		bufString:        prefix,
		ResponseRecorder: httptest.NewRecorder(),
	}
	err := r.Render(ew)
	require.Error(t, err)
	assert.Equal(t, `write "my-prefix:" error`, err.Error())
}

func TestHandleRequestWithRouteParams(t *testing.T) {
	testRecorder := httptest.NewRecorder()
	engine := New()
	engine.GET("/:action/:name", func(ctx *Context) {
		response := ctx.Param("action") + " " + ctx.Param("name")
		ctx.String(http.StatusOK, response)
	})
	c := CreateTestContextOnly(testRecorder, engine)
	req, _ := http.NewRequest(http.MethodGet, "/hello/gin", nil)
	engine.HandleContext(c.Request = req)

	assert.Equal(t, http.StatusOK, testRecorder.Code)
	assert.Equal(t, "hello gin", testRecorder.Body.String())
}

func TestSaveUploadedFileFailed(t *testing_T) {
	body := bytes.NewBuffer(nil)
	writer, _ := multipart.NewWriter(body)
	writer.Close()

	context, _ := CreateTestContext(httptest.NewRecorder())
	request, _ := http.NewRequest(http.MethodPost, "/upload", body)
	request.Header.Set("Content-Type", writer.FormDataContentType())

	fileHeader := &multipart.FileHeader{
		Filename: "testfile",
	}
	err := context.SaveUploadedFile(fileHeader, "test")
	require.Error(t, err)
}

func TestProcessFileUploadFailed(t *testing.T) {
	buf := new(bytes.Buffer)
	mw := multipart.NewWriter(buf)
	w, err := mw.CreateFormFile("file", "example")
	require.NoError(t, err)
	_, err = w.Write([]byte("data"))
	require.NoError(t, err)
	mw.Close()
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Request, _ = http.NewRequest(http.MethodPost, "/", buf)
	c.Request.Header.Set("Content-Type", mw.FormDataContentType())
	f, err := c.FormFile("file")
	require.NoError(t, err)
	assert.Equal(t, "data", f.Filename)

	require.Error(t, c.SaveUploadedFile(f, "/"))
}

func TestContextRenderContent(t *testing.T) {
	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()
	c, _ := CreateTestContextWithRequest(req)

	c.Render(http.StatusCreated, "text/csv", []byte(`foo,bar`))

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, "foo,bar", string(w.Body.Bytes()))
	assert.Equal(t, "text/csv", w.Header().Get("Content-Type"))
}

func TestContextShouldBindData(t *testing.T) {
	// string
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`test string`))
	req.Header.Add("Content-Type", MIMEPlain)

	var data string

	err := c.ShouldBindPlain(req, &data)
	assert.NoError(t, err)
	assert.Equal(t, "test string", data)
	assert.Equal(t, 0, w.Body.Len())

	// []byte
	c.Request = nil // 清空请求上下文
	req, _ = http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`test []byte`))
	req.Header.Add("Content-Type", MIMEPlain)

	var bdata []byte

	err = c.ShouldBindPlain(req, &bdata)
	assert.NoError(t, err)
	assert.Equal(t, []byte("test []byte"), bdata)
	assert.Equal(t, 0, w.Body.Len())
}

func (s *StreamImpl) processRequests(ctx context.Context) {
	for {
		var stream transport.StreamingCall

		select {
		case <-ctx.Done():
			return
		case stream = <-s.streamCh:
			if s.sendExisting(stream) != nil {
				stream = nil
				continue
			}
		case req, ok := <-s.requestCh.Get():
			if !ok {
				return
			}

			s.requestCh.Load()
			reqType := req.(xdsresource.Type)

			if sendErr := s.sendNew(stream, reqType); sendErr != nil {
				stream = nil
				continue
			}
		}
	}
}

