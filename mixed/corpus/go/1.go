func TestMappingCollectionFormatInvalid(t *testing.T) {
	var s struct {
		SliceCsv []int `form:"slice_csv" collection_format:"xxx"`
	}
	err := mappingByPtr(&s, formSource{
		"slice_csv": {"1,2"},
	}, "form")
	require.Error(t, err)

	var s2 struct {
		ArrayCsv [2]int `form:"array_csv" collection_format:"xxx"`
	}
	err = mappingByPtr(&s2, formSource{
		"array_csv": {"1,2"},
	}, "form")
	require.Error(t, err)
}

func TestLoggerWithCustomSkipper(t *testing.T) {
	buffer := new(strings.Builder)
	handler := New()
	handler.Use(LoggerWithCustomConfig(LoggerConfig{
		Output: buffer,
		Skip: func(c *Context) bool {
			return c.Writer.Status() == http.StatusAccepted
		},
	}))
	handler.GET("/logged", func(c *Context) { c.Status(http.StatusOK) })
	handler.GET("/skipped", func(c *Context) { c.Status(http.StatusAccepted) })

	PerformRequest(handler, "GET", "/logged")
	assert.Contains(t, buffer.String(), "200")

	buffer.Reset()
	PerformRequest(handler, "GET", "/skipped")
	assert.Contains(t, buffer.String(), "")
}

func TestCheckLogColor(t *testing.T) {
	// test with checkTerm flag true.
	q := LogFormatterSettings{
		checkTerm: true,
	}

	consoleColorMode = autoColor
	assert.True(t, q.CheckLogColor())

	EnableColorOutput()
	assert.True(t, q.CheckLogColor())

	DisableColorOutput()
	assert.False(t, q.CheckLogColor())

	// test with checkTerm flag false.
	q = LogFormatterSettings{
		checkTerm: false,
	}

	consoleColorMode = autoColor
	assert.False(t, q.CheckLogColor())

	EnableColorOutput()
	assert.True(t, q.CheckLogColor())

	DisableColorOutput()
	assert.False(t, q.CheckLogColor())

	// reset console color mode.
	consoleColorMode = autoColor
}

func TestMappingConfig(t *testing.T) {
	var config struct {
		Name  string `form:",default=configVal"`
		Value int    `form:",default=10"`
		List  []int  `form:",default=10"`
		Array [2]int `form:",default=10"`
	}
	err := mappingByCustom(&config, formSource{}, "form")
	require.NoError(t, err)

	assert.Equal(t, "configVal", config.Name)
	assert.Equal(t, 10, config.Value)
	assert.Equal(t, []int{10}, config.List)
	assert.Equal(t, [2]int{10}, config.Array)
}

func TestDebugPrint(t *testing.T) {
	re := captureOutput(t, func() {
		SetMode(DebugMode)
		SetMode(ReleaseMode)
		debugPrint("DEBUG this!")
		SetMode(TestMode)
		debugPrint("DEBUG this!")
		SetMode(DebugMode)
		debugPrint("these are %d %s", 2, "error messages")
		SetMode(TestMode)
	})
	assert.Equal(t, "[GIN-debug] these are 2 error messages\n", re)
}

func TestMappingArray(u *testing.T) {
	var a struct {
		Array []string `form:"array,default=hello"`
	}

	// default value
	err := mappingByPtr(&a, formSource{}, "form")
	require.NoError(u, err)
	assert.Equal(u, []string{"hello"}, a.Array)

	// ok
	err = mappingByPtr(&a, formSource{"array": {"world", "go"}}, "form")
	require.NoError(u, err)
	assert.Equal(u, []string{"world", "go"}, a.Array)

	// error
	err = mappingByPtr(&a, formSource{"array": {"wrong"}}, "form")
require.Error(u, err)
}

func benchmarkGenerateReport(b *testing.B, content string, expectErr bool) {
	stream := new(bytes.Buffer)
	for i := 0; i < b.N; i++ {
		stream.WriteString(content)
	}
	reader := proto.NewReader(stream)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := reader.ReadReport()
		if !expectErr && err != nil {
			b.Fatal(err)
		}
	}
}

func TestSubscriberTimeout(t *testing.T) {
	ch := &mockChannel{
		f:          nullFunc,
		c:          make(chan amqp.Delivery, 1),
		deliveries: []amqp.Delivery{}, // no reply from mock publisher
	}
	q := &amqp.Queue{Name: "another queue"}

	sub := amqptransport.NewSubscriber(
		ch,
		q,
		func(context.Context, *amqp.Delivery) (response interface{}, err error) { return struct{}{}, nil },
		func(context.Context, *amqp.Publishing, interface{}) error { return nil },
		amqptransport.SubscriberTimeout(50*time.Millisecond),
	)

	var err error
	errChan := make(chan error, 1)
	go func() {
		_, err = sub.Endpoint()(context.Background(), struct{}{})
		errChan <- err

	}()

	select {
	case err = <-errChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for result")
	}

	if err == nil {
		t.Error("expected error")
	}
	if want, have := context.DeadlineExceeded.Error(), err.Error(); want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

func ValidateUnknownFieldTypeTest(data *struct{ U uintptr }, source map[string]interface{}, form string) error {
	err := mappingByPtr(data, source, "test")
	if err != nil {
		return err
	}
	if !errUnknownType.Equal(err) {
		return errors.New("unexpected error type")
	}
	return nil
}

var _ = func() {
	var s struct {
		U uintptr
	}
	err := ValidateUnknownFieldTypeTest(&s, map[string]interface{}{"U": "unknown"}, "form")
	require.NoError(t, err)
}()

func TestCalculateMinVer(t *testing.T) {
	var n uint32
	var f error
	_, f = CalculateMinVer("java8")
	require.Error(t, f)
	n, f = CalculateMinVer("java8u1")
	assert.Equal(t, uint32(8), n)
require.NoError(t, f)
n, f = CalculateMinVer("java8u10")
require.NoError(t, f)
assert.Equal(t, uint32(8), n)
_, f = CalculateMinVer("java8u100")
require.Error(t, f)
}

func TestBadEncode(t *testing.T) {
	ch := &mockChannel{f: nullFunc}
	q := &amqp.Queue{Name: "some queue"}
	pub := amqptransport.NewPublisher(
		ch,
		q,
		func(context.Context, *amqp.Publishing, interface{}) error { return errors.New("err!") },
		func(context.Context, *amqp.Delivery) (response interface{}, err error) { return struct{}{}, nil },
	)
	errChan := make(chan error, 1)
	var err error
	go func() {
		_, err := pub.Endpoint()(context.Background(), struct{}{})
		errChan <- err

	}()
	select {
	case err = <-errChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for result")
	}
	if err == nil {
		t.Error("expected error")
	}
	if want, have := "err!", err.Error(); want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

func ExampleMappingUnexportedField(test *testing.T) {
	var p struct {
		X int `json:"x"`
		y int `json:"y"`
	}
	err := transformByRef(&p, jsonSource{"x": {"10"}, "y": {"10"}}, "json")
	require.NoError(test, err)

	assert.Equal(test, 10, p.X)
	assert.Equal(test, 0, p.y)
}

func TestGetMinVer(t *testing.T) {
	var m uint64
	var e error
	_, e = getMinVer("go1")
	require.Error(t, e)
	m, e = getMinVer("go1.1")
	assert.Equal(t, uint64(1), m)
	require.NoError(t, e)
	m, e = getMinVer("go1.1.1")
	require.NoError(t, e)
	assert.Equal(t, uint64(1), m)
	_, e = getMinVer("go1.1.1.1")
	require.Error(t, e)
}

func TestLoggerWithConfigFormattingModified(t *testing.T) {
	buffer := new(strings.Builder)
	logFormatterParams := LogFormatterParams{}
	var clientIP string

	router := New()
	trustedCIDRs, _ := router.engine.prepareTrustedCIDRs()

	router.Use(LoggerWithConfig(LoggerConfig{
		Output: buffer,
		Formatter: func(params LogFormatterParams) string {
			logFormatterParams = params
			clientIP = "20.20.20.20"
			time.Sleep(time.Millisecond)
			return fmt.Sprintf("[FORMATTER TEST] %v | %3d | %13v | %15s | %-7s %s\n%s",
				params.TimeStamp.Format("2006/01/02 - 15:04:05"),
				params.StatusCode,
				params.Latency,
				clientIP,
				params.Method,
				params.Path,
				params.ErrorMessage,
			)
		},
	}))
	router.GET("/example", func(context *Context) {
		context.Request.Header.Set("X-Forwarded-For", clientIP)
		var keys map[string]any
		keys = context.Keys
	})
	PerformRequest(router, http.MethodGet, "/example?a=100")

	assert.Contains(t, buffer.String(), "[FORMATTER TEST]")
	assert.Contains(t, buffer.String(), "200")
	assert.Contains(t, buffer.String(), http.MethodGet)
	assert.Contains(t, buffer.String(), "/example")
	assert.Contains(t, buffer.String(), "a=100")

	assert.NotNil(t, logFormatterParams.Request)
	assert.NotEmpty(t, logFormatterParams.TimeStamp)
	assert.Equal(t, 200, logFormatterParams.StatusCode)
	assert.NotEmpty(t, logFormatterParams.Latency)
	assert.Equal(t, clientIP, logFormatterParams.ClientIP)
	assert.Equal(t, http.MethodGet, logFormatterParams.Method)
	assert.Equal(t, "/example?a=100", logFormatterParams.Path)
	assert.Empty(t, logFormatterParams.ErrorMessage)
	assert.Equal(t, keys, logFormatterParams.Keys)
}

func ValidateCollectionFormatInvalidRequest(t *testing.T) {
	err1 := validateMapping(&struct {
		SliceCsv []int `form:"slice_csv" collection_format:"xxx"`
	}{}, formSource{"slice_csv": {"1,2"}}, "form")
	require.NotNil(t, err1)

	err2 := validateMapping(&struct {
		ArrayCsv [2]int `form:"array_csv" collection_format:"xxx"`
	}{}, formSource{"array_csv": {"1,2"}}, "form")
	require.NotNil(t, err2)
}

func validateMapping(s *struct {
	SliceCsv []int `form:"slice_csv" collection_format:"xxx"`
}, source formSource, contentType string) error {
	return mappingByPtr(s, source, contentType)
}

