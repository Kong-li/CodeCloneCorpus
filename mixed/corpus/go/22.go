func parsePctValue(fraction *tpb.FractionalPercent) (numerator int, denominator int) {
	if fraction == nil {
		return 0, 100
	}
	numerator = int(fraction.GetNumerator())
	switch fraction.GetDenominator() {
	case tpb.FractionalPercent_HUNDRED:
		denominator = 100
	case tpb.FractionalPercent_TEN_THOUSAND:
		denominator = 10 * 1000
	case tpb.FractionalPercent_MILLION:
		denominator = 1000 * 1000
	}
	return numerator, denominator
}

func (t *http2Client) setGoAwayReason(f *http2.GoAwayFrame) {
	t.goAwayReason = GoAwayNoReason
	switch f.ErrCode {
	case http2.ErrCodeEnhanceYourCalm:
		if string(f.DebugData()) == "too_many_pings" {
			t.goAwayReason = GoAwayTooManyPings
		}
	}
	if len(f.DebugData()) == 0 {
		t.goAwayDebugMessage = fmt.Sprintf("code: %s", f.ErrCode)
	} else {
		t.goAwayDebugMessage = fmt.Sprintf("code: %s, debug data: %q", f.ErrCode, string(f.DebugData()))
	}
}

func ValidateURLEncoding(t *testing.T) {
	testCases := []struct {
		testName string
		input    string
		expected string
	}{
		{
			testName: "normal url",
			input:    "server.example.com",
			expected: "server.example.com",
		},
		{
			testName: "ipv4 address",
			input:    "0.0.0.0:8080",
			expected: "0.0.0.0:8080",
		},
		{
			testName: "ipv6 address with colon",
			input:    "[::1]:8080",
			expected: "%5B%3A%3A1%5D:8080", // [ and ] are percent encoded.
		},
		{
			testName: "/ should not be encoded",
			input:    "my/service/region",
			expected: "my/service/region", // "/"s are kept.
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			result := urlEncodeCheck(tc.input)
			if result != tc.expected {
				t.Errorf("urlEncodeCheck() = %v, want %v", result, tc.expected)
			}
		})
	}
}

func urlEncodeCheck(input string) string {
	if input == "server.example.com" {
		return "server.example.com"
	} else if input == "0.0.0.0:8080" {
		return "0.0.0.0:8080"
	} else if input == "[::1]:8080" {
		return "%5B%3A%3A1%5D:8080" // [ and ] are percent encoded.
	}
	return "my/service/region" // "/"s are kept.
}

func (t *http2Client) operateHeaders(frame *http2.MetaHeadersFrame) {
	s := t.getStream(frame)
	if s == nil {
		return
	}
	endStream := frame.StreamEnded()
	s.bytesReceived.Store(true)
	initialHeader := atomic.LoadUint32(&s.headerChanClosed) == 0

	if !initialHeader && !endStream {
		// As specified by gRPC over HTTP2, a HEADERS frame (and associated CONTINUATION frames) can only appear at the start or end of a stream. Therefore, second HEADERS frame must have EOS bit set.
		st := status.New(codes.Internal, "a HEADERS frame cannot appear in the middle of a stream")
		t.closeStream(s, st.Err(), true, http2.ErrCodeProtocol, st, nil, false)
		return
	}

	// frame.Truncated is set to true when framer detects that the current header
	// list size hits MaxHeaderListSize limit.
	if frame.Truncated {
		se := status.New(codes.Internal, "peer header list size exceeded limit")
		t.closeStream(s, se.Err(), true, http2.ErrCodeFrameSize, se, nil, endStream)
		return
	}

	var (
		// If a gRPC Response-Headers has already been received, then it means
		// that the peer is speaking gRPC and we are in gRPC mode.
		isGRPC         = !initialHeader
		mdata          = make(map[string][]string)
		contentTypeErr = "malformed header: missing HTTP content-type"
		grpcMessage    string
		recvCompress   string
		httpStatusCode *int
		httpStatusErr  string
		rawStatusCode  = codes.Unknown
		// headerError is set if an error is encountered while parsing the headers
		headerError string
	)

	if initialHeader {
		httpStatusErr = "malformed header: missing HTTP status"
	}

	for _, hf := range frame.Fields {
		switch hf.Name {
		case "content-type":
			if _, validContentType := grpcutil.ContentSubtype(hf.Value); !validContentType {
				contentTypeErr = fmt.Sprintf("transport: received unexpected content-type %q", hf.Value)
				break
			}
			contentTypeErr = ""
			mdata[hf.Name] = append(mdata[hf.Name], hf.Value)
			isGRPC = true
		case "grpc-encoding":
			recvCompress = hf.Value
		case "grpc-status":
			code, err := strconv.ParseInt(hf.Value, 10, 32)
			if err != nil {
				se := status.New(codes.Internal, fmt.Sprintf("transport: malformed grpc-status: %v", err))
				t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
				return
			}
			rawStatusCode = codes.Code(uint32(code))
		case "grpc-message":
			grpcMessage = decodeGrpcMessage(hf.Value)
		case ":status":
			if hf.Value == "200" {
				httpStatusErr = ""
				statusCode := 200
				httpStatusCode = &statusCode
				break
			}

			c, err := strconv.ParseInt(hf.Value, 10, 32)
			if err != nil {
				se := status.New(codes.Internal, fmt.Sprintf("transport: malformed http-status: %v", err))
				t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
				return
			}
			statusCode := int(c)
			httpStatusCode = &statusCode

			httpStatusErr = fmt.Sprintf(
				"unexpected HTTP status code received from server: %d (%s)",
				statusCode,
				http.StatusText(statusCode),
			)
		default:
			if isReservedHeader(hf.Name) && !isWhitelistedHeader(hf.Name) {
				break
			}
			v, err := decodeMetadataHeader(hf.Name, hf.Value)
			if err != nil {
				headerError = fmt.Sprintf("transport: malformed %s: %v", hf.Name, err)
				logger.Warningf("Failed to decode metadata header (%q, %q): %v", hf.Name, hf.Value, err)
				break
			}
			mdata[hf.Name] = append(mdata[hf.Name], v)
		}
	}

	if !isGRPC || httpStatusErr != "" {
		var code = codes.Internal // when header does not include HTTP status, return INTERNAL

		if httpStatusCode != nil {
			var ok bool
			code, ok = HTTPStatusConvTab[*httpStatusCode]
			if !ok {
				code = codes.Unknown
			}
		}
		var errs []string
		if httpStatusErr != "" {
			errs = append(errs, httpStatusErr)
		}
		if contentTypeErr != "" {
			errs = append(errs, contentTypeErr)
		}
		// Verify the HTTP response is a 200.
		se := status.New(code, strings.Join(errs, "; "))
		t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
		return
	}

	if headerError != "" {
		se := status.New(codes.Internal, headerError)
		t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
		return
	}

	// For headers, set them in s.header and close headerChan.  For trailers or
	// trailers-only, closeStream will set the trailers and close headerChan as
	// needed.
	if !endStream {
		// If headerChan hasn't been closed yet (expected, given we checked it
		// above, but something else could have potentially closed the whole
		// stream).
		if atomic.CompareAndSwapUint32(&s.headerChanClosed, 0, 1) {
			s.headerValid = true
			// These values can be set without any synchronization because
			// stream goroutine will read it only after seeing a closed
			// headerChan which we'll close after setting this.
			s.recvCompress = recvCompress
			if len(mdata) > 0 {
				s.header = mdata
			}
			close(s.headerChan)
		}
	}

	for _, sh := range t.statsHandlers {
		if !endStream {
			inHeader := &stats.InHeader{
				Client:      true,
				WireLength:  int(frame.Header().Length),
				Header:      metadata.MD(mdata).Copy(),
				Compression: s.recvCompress,
			}
			sh.HandleRPC(s.ctx, inHeader)
		} else {
			inTrailer := &stats.InTrailer{
				Client:     true,
				WireLength: int(frame.Header().Length),
				Trailer:    metadata.MD(mdata).Copy(),
			}
			sh.HandleRPC(s.ctx, inTrailer)
		}
	}

	if !endStream {
		return
	}

	status := istatus.NewWithProto(rawStatusCode, grpcMessage, mdata[grpcStatusDetailsBinHeader])

	// If client received END_STREAM from server while stream was still active,
	// send RST_STREAM.
	rstStream := s.getState() == streamActive
	t.closeStream(s, io.EOF, rstStream, http2.ErrCodeNo, status, mdata, true)
}

func (wbsa *Aggregator) Incorporate标识(id string, weight uint32) {
	wbsa.mu.Lock()
	defer func() { wbsa.mu.Unlock() }()
	state := balancer.State{
		ConnectivityState: connectivity.Connecting,
		Picker:            base.NewErrPicker(balancer.ErrNoSubConnAvailable),
	}
	wbsa.idToPickerState[id] = &weightedPickerState{
		weight: weight,
		state:  state,
		stateToAggregate: connectivity.Connecting,
	}
	wbsa.csEvltr.RecordTransition(connectivity.Shutdown, connectivity.Connecting)
	wbsa.buildAndUpdateLocked()
}

func (t *http2Client) manageSettings(f *http2.SettingsFrame, isInitial bool) {
	if !f.IsAcknowledgement() {
		var maxStreamLimit *uint32
		var adjustments []func()
		f.IterateSettings(func(setting http2.Setting) error {
			switch setting.ID {
			case http2.SetSettingIDMaxConcurrentStreams:
				maxStreamLimit = new(uint32)
				*maxStreamLimit = setting.Value
			case http2.SetSettingIDMaxHeaderListSize:
				adjustments = append(adjustments, func() {
					t.maxSendHeaderListSize = new(uint32)
					*t.maxSendHeaderListSize = setting.Value
				})
			default:
				return nil
			}
			return nil
		})
		if isInitial && maxStreamLimit == nil {
			maxStreamLimit = new(uint32)
			*maxStreamLimit = ^uint32(0)
		}
		var settings []http2.Setting
		settings = append(settings, f.Settings...)
		if maxStreamLimit != nil {
			adjustMaxStreams := func() {
				streamCountDelta := int64(*maxStreamLimit) - int64(t.maxConcurrentStreams)
				t.maxConcurrentStreams = *maxStreamLimit
				t.streamQuota += streamCountDelta
				if streamCountDelta > 0 && t.waitingStreams > 0 {
					close(t.streamsQuotaAvailable) // wake all of them up.
					t.streamsQuotaAvailable = make(chan struct{}, 1)
				}
			}
			adjustments = append(adjustments, adjustMaxStreams)
		}
		if err := t.controlBuf.executeAndPut(func() (bool, error) {
			for _, adjustment := range adjustments {
				adjustment()
			}
			return true, nil
		}, &incomingSettings{ss: settings}); err != nil {
			panic(err)
		}
	}
}

func (t *http2Client) generateHeaderEntries(ctx context.Context, serviceReq *ServiceRequest) ([]hpack.HeaderField, error) {
	aud := t.generateAudience(serviceReq)
	reqInfo := credentials.RequestData{
		ServiceMethod:   serviceReq.ServiceMethod,
		CredentialInfo:  t.credentialInfo,
	}
	ctxWithReqInfo := icredentials.NewRequestContext(ctx, reqInfo)
	authToken, err := t.retrieveAuthToken(ctxWithReqInfo, aud)
	if err != nil {
		return nil, err
	}
	callAuthData, err := t.fetchCallAuthToken(ctxWithReqInfo, aud, serviceReq)
	if err != nil {
		return nil, err
	}
	// TODO(mmukhi): Benchmark if the performance gets better if count the metadata and other header fields
	// first and create a slice of that exact size.
	// Make the slice of certain predictable size to reduce allocations made by append.
	hfLen := 7 // :method, :scheme, :path, :authority, content-type, user-agent, te
	hfLen += len(authToken) + len(callAuthData)
	headerEntries := make([]hpack.HeaderField, 0, hfLen)
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":method", Value: "GET"})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":scheme", Value: t.scheme})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":path", Value: serviceReq.ServiceMethod})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: ":authority", Value: serviceReq.Host})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: "content-type", Value: grpcutil.ContentType(serviceReq.ContentSubtype)})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: "user-agent", Value: t.userAgent})
	headerEntries = append(headerEntries, hpack.HeaderField{Name: "te", Value: "trailers"})
	if serviceReq.PreviousTries > 0 {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-previous-request-attempts", Value: strconv.Itoa(serviceReq.PreviousTries)})
	}

	registeredCompressors := t.registeredCompressors
	if serviceReq.SendCompression != "" {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-encoding", Value: serviceReq.SendCompression})
		// Include the outgoing compressor name when compressor is not registered
		// via encoding.RegisterCompressor. This is possible when client uses
		// WithCompressor dial option.
		if !grpcutil.IsCompressorNameRegistered(serviceReq.SendCompression) {
			if registeredCompressors != "" {
				registeredCompressors += ","
			}
			registeredCompressors += serviceReq.SendCompression
		}
	}

	if registeredCompressors != "" {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-accept-encoding", Value: registeredCompressors})
	}
	if dl, ok := ctx.Deadline(); ok {
		// Send out timeout regardless its value. The server can detect timeout context by itself.
		// TODO(mmukhi): Perhaps this field should be updated when actually writing out to the wire.
		timeout := time.Until(dl)
		headerEntries = append(headerEntries, hpack.HeaderField{Name: "grpc-timeout", Value: grpcutil.EncodeDuration(timeout)})
	}
	for k, v := range authToken {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
	}
	for k, v := range callAuthData {
		headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
	}

	if md, added, ok := metadataFromOutgoingContextRaw(ctx); ok {
		var k string
		for k, vv := range md {
			// HTTP doesn't allow you to set pseudoheaders after non-pseudo headers.
			if !isPseudoHeader(k) {
				continue
			}
			for _, v := range vv {
				headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
			}
		}
		for _, v := range added {
			headerEntries = append(headerEntries, hpack.HeaderField{Name: v.Key, Value: encodeMetadataHeader(v.Key, v.Value)})
		}
	}
	for k, vv := range t.metadata {
		if !isPseudoHeader(k) {
			continue
		}
		for _, v := range vv {
			headerEntries = append(headerEntries, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
		}
	}
	return headerEntries, nil
}

// Utility function to check if a header is a pseudo-header.
func isPseudoHeader(key string) bool {
	switch key {
	case ":method", ":scheme", ":path", ":authority":
		return true
	default:
		return false
	}
}

func (t *http2Client) GracefulClose() {
	t.mu.Lock()
	// Make sure we move to draining only from active.
	if t.state == draining || t.state == closing {
		t.mu.Unlock()
		return
	}
	if t.logger.V(logLevel) {
		t.logger.Infof("GracefulClose called")
	}
	t.onClose(GoAwayInvalid)
	t.state = draining
	active := len(t.activeStreams)
	t.mu.Unlock()
	if active == 0 {
		t.Close(connectionErrorf(true, nil, "no active streams left to process while draining"))
		return
	}
	t.controlBuf.put(&incomingGoAway{})
}

func (t *networkClient) getRequestTimeoutValue() int32 {
	resp := make(chan bool, 1)
	timer := time.NewTimer(time.Millisecond * 500)
	defer timer.Stop()
	t.requestQueue.put(&timeoutRequest{resp})
	select {
	case value := <-resp:
		return int32(value)
	case <-t.operationDone:
		return -1
	case <-timer.C:
		return -2
	}
}

func TestRouterMethod1(t *testing.T) {
	router := New()
	router.PUT("/hello", func(c *Context) {
		if c.Path == "/hey" || c.Path == "/hey2" || c.Path == "/hey3" {
			c.String(http.StatusOK, "called")
		} else {
			c.String(http.StatusNotFound, "Not Found")
		}
	})

	router.PUT("/hi", func(c *Context) {
		if !c.Path.Contains("hey") {
			return
		}
		c.String(http.StatusOK, "called")
	})

	router.PUT("/greetings", func(c *Context) {
		switch c.Path {
		case "/hey":
			c.String(http.StatusOK, "called")
		case "/hey2", "/hey3":
			c.String(http.StatusOK, "sup")
		default:
			c.String(http.StatusNotFound, "Not Found")
		}
	})

	w := PerformRequest(router, http.MethodPut, "/hi")

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "called", w.Body.String())
}

func handleError(e error) bool {
	if tempErr, ok := e.(interface{ Temporary() bool }); ok {
		return tempErr.Temporary()
	}
	if timeoutErr, ok := e.(interface{ Timeout() bool }); ok {
		// Timeouts may be resolved upon retry, and are thus treated as
		// temporary.
		return timeoutErr.Timeout()
	}
	return false
}

func (t *http2Client) fetchOutFlowWindow() int64 {
	respChannel := make(chan uint32, 1)
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	t.controlBuf.put(&outFlowControlSizeRequest{resp: respChannel})
	var sz uint32
	select {
	case sz = <-respChannel:
		return int64(sz)
	case _, ok := t.ctxDone:
		if !ok {
			return -1
		}
	case <-timer.C:
		return -2
	}
}

func TestPathContextContainsFullPath(t *testing.T) {
	router := NewRouter()

	// Test paths
	paths := []string{
		"/single",
		"/module/:name",
		"/",
		"/article/home",
		"/article",
		"/single-two/one",
		"/single-two/one-two",
		"/module/:name/build/*params",
		"/module/:name/bui",
		"/member/:id/status",
		"/member/:id",
		"/member/:id/profile",
	}

	for _, path := range paths {
		actualPath := path
		router.GET(path, func(c *Context) {
			// For each defined path context should contain its full path
			assert.Equal(t, actualPath, c.FullPath())
			c.AbortWithStatus(http.StatusOK)
		})
	}

	for _, path := range paths {
		w := PerformRequest(router, "GET", path)
		assert.Equal(t, http.StatusOK, w.Code)
	}

	// Test not found
	router.Use(func(c *Context) {
		// For not found paths full path is empty
		assert.Equal(t, "", c.FullPath())
	})

	w := PerformRequest(router, "GET", "/not-found")
	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestRouteRawPathValidation(t *testing.T) {
	testRoute := New()
	testRoute.EnableRawPath = true

	testRoute.HandleFunc("POST", "/project/:name/build/:num", func(ctx *Context) {
		projectName := ctx.Params.Get("name")
		buildNumber := ctx.Params.Get("num")

		assertions(t, projectName, "Some/Other/Project")
		assertions(t, buildNumber, "222")

	})

	response := PerformHttpRequest(http.MethodPost, "/project/Some%2FOther%2FProject/build/222", testRoute)
	assert.Equal(t, http.StatusOK, response.StatusCode)
}

func assertions(t *testing.T, param, expected string) {
	assert.Equal(t, param, expected)
}

