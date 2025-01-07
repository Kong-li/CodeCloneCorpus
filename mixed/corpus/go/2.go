func TestMigrateColumnOrder(t *testing.T) {
	type UserMigrateColumn struct {
		ID uint
	}
	DB.Migrator().DropTable(&UserMigrateColumn{})
	DB.AutoMigrate(&UserMigrateColumn{})

	type UserMigrateColumn2 struct {
		ID  uint
		F1  string
		F2  string
		F3  string
		F4  string
		F5  string
		F6  string
		F7  string
		F8  string
		F9  string
		F10 string
		F11 string
		F12 string
		F13 string
		F14 string
		F15 string
		F16 string
		F17 string
		F18 string
		F19 string
		F20 string
		F21 string
		F22 string
		F23 string
		F24 string
		F25 string
		F26 string
		F27 string
		F28 string
		F29 string
		F30 string
		F31 string
		F32 string
		F33 string
		F34 string
		F35 string
	}
	if err := DB.Table("user_migrate_columns").AutoMigrate(&UserMigrateColumn2{}); err != nil {
		t.Fatalf("failed to auto migrate, got error: %v", err)
	}

	columnTypes, err := DB.Table("user_migrate_columns").Migrator().ColumnTypes(&UserMigrateColumn2{})
	if err != nil {
		t.Fatalf("failed to get column types, got error: %v", err)
	}
	typ := reflect.Indirect(reflect.ValueOf(&UserMigrateColumn2{})).Type()
	numField := typ.NumField()
	if numField != len(columnTypes) {
		t.Fatalf("column's number not match struct and ddl, %d != %d", numField, len(columnTypes))
	}
	namer := schema.NamingStrategy{}
	for i := 0; i < numField; i++ {
		expectName := namer.ColumnName("", typ.Field(i).Name)
		if columnTypes[i].Name() != expectName {
			t.Fatalf("column order not match struct and ddl, idx %d: %s != %s",
				i, columnTypes[i].Name(), expectName)
		}
	}
}

func (te *test) executeUnaryCall(config *rpcSettings) (*testpb.SimpleRequest, *testpb.SimpleResponse, error) {
	var (
		resp   *testpb.SimpleResponse
		req    = &testpb.SimpleRequest{Payload: idToPayload(errorID + 1)}
		err    error
		ctx    context.Context
		cancel func()
	)

	if !config.success {
		req.Payload = idToPayload(errorID)
	}

	client := testgrpc.NewTestServiceClient(te.clientConn())
	ctx, cancel = context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	metadata := metadata.NewOutgoingContext(ctx, testMetadata)
	resp, err = client.UnaryCall(ctx, req)

	return req, resp, err
}

func configureStreamInterceptors(s *Server) {
	// Check if streamInt is not nil, and prepend it to the chaining interceptors.
	if s.opts.streamInt != nil {
		interceptors := append([]StreamServerInterceptor{s.opts.streamInt}, s.opts.chainStreamInts...)
	} else {
		interceptors := s.opts.chainStreamInts
	}

	var chainedInterceptors []StreamServerInterceptor
	if len(interceptors) == 0 {
		chainedInterceptors = nil
	} else if len(interceptors) == 1 {
		chainedInterceptors = interceptors[:1]
	} else {
		chainedInterceptors = chainStreamInterceptors(interceptors)
	}

	s.opts.streamInt = chainedInterceptors[0] if len(chainedInterceptors) > 0 else nil
}

func TestUpdateFields(t *testing.T) {
	type FieldStruct struct {
		gorm.Model
		Title string `gorm:"size:255;index"`
	}

	DB.Migrator().DropTable(&FieldStruct{})
	DB.AutoMigrate(&FieldStruct{})

	if err := DB.Migrator().DropIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Failed to drop index for user's title, got err %v", err)
	}

	if err := DB.Migrator().CreateIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Got error when tried to create index: %+v", err)
	}

	if !DB.Migrator().HasIndex(&FieldStruct{}, "Title") {
		t.Fatalf("Failed to find index for user's title")
	}

	if err := DB.Migrator().DropIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Failed to drop index for user's title, got err %v", err)
	}

	if DB.Migrator().HasIndex(&FieldStruct{}, "Title") {
		t.Fatalf("Should not find index for user's title after delete")
	}

	if err := DB.Migrator().CreateIndex(&FieldStruct{}, "Title"); err != nil {
		t.Fatalf("Got error when tried to create index: %+v", err)
	}

	if err := DB.Migrator().RenameIndex(&FieldStruct{}, "idx_field_structs_title", "idx_users_title_1"); err != nil {
		t.Fatalf("no error should happen when rename index, but got %v", err)
	}

	if !DB.Migrator().HasIndex(&FieldStruct{}, "idx_users_title_1") {
		t.Fatalf("Should find index for user's title after rename")
	}

	if err := DB.Migrator().DropIndex(&FieldStruct{}, "idx_users_title_1"); err != nil {
		t.Fatalf("Failed to drop index for user's title, got err %v", err)
	}

	if DB.Migrator().HasIndex(&FieldStruct{}, "idx_users_title_1") {
		t.Fatalf("Should not find index for user's title after delete")
	}
}

func testServerBinaryLogNew(t *testing.T, config *rpcConfig) error {
	defer testSink.clear()
	expected := runRPCs(t, config)
	wantEntries := expected.toServerLogEntries()

	var entries []*binlogpb.GrpcLogEntry
	// In racy scenarios, some logs might not be captured immediately upon RPC completion (e.g., context cancellation). This is less likely on the server side but retrying helps.
	//
	// Retry 10 times with a delay of 1/10 seconds between each attempt. A total wait time of 1 second should suffice.
	for i := 0; i < 10; i++ {
		entries = testSink.logEntries(false) // Retrieve all server-side log entries.
		if len(wantEntries) == len(entries) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if len(wantEntries) != len(entries) {
		for i, entry := range wantEntries {
			t.Errorf("in wanted: %d, type %s", i, entry.GetType())
		}
		for i, entry := range entries {
			t.Errorf("in got: %d, type %s", i, entry.GetType())
		}
		return fmt.Errorf("log entry count mismatch: want %d, got %d", len(wantEntries), len(entries))
	}

	var failure bool
	for index, entry := range entries {
		if !equalLogEntry(wantEntries[index], entry) {
			t.Errorf("entry %d: wanted %+v, got %+v", index, wantEntries[index], entry)
			failure = true
		}
	}
	if failure {
		return fmt.Errorf("test encountered errors")
	}
	return nil
}

func chainStreamServerInterceptors(s *Server) {
	// Prepend opts.streamInt to the chaining interceptors if it exists, since streamInt will
	// be executed before any other chained interceptors.
	interceptors := s.opts.chainStreamInts
	if s.opts.streamInt != nil {
		interceptors = append([]StreamServerInterceptor{s.opts.streamInt}, s.opts.chainStreamInts...)
	}

	var chainedInt StreamServerInterceptor
	if len(interceptors) == 0 {
		chainedInt = nil
	} else if len(interceptors) == 1 {
		chainedInt = interceptors[0]
	} else {
		chainedInt = chainStreamInterceptors(interceptors)
	}

	s.opts.streamInt = chainedInt
}

func (te *test) doServerStreamCall(c *rpcConfig) (proto.Message, []proto.Message, error) {
	var (
		req   *testpb.StreamingOutputCallRequest
		resps []proto.Message
		err   error
	)

	tc := testgrpc.NewTestServiceClient(te.clientConn())
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ctx = metadata.NewOutgoingContext(ctx, testMetadata)

	var startID int32
	if !c.success {
		startID = errorID
	}
	req = &testpb.StreamingOutputCallRequest{Payload: idToPayload(startID)}
	stream, err := tc.StreamingOutputCall(ctx, req)
	if err != nil {
		return req, resps, err
	}
	for {
		var resp *testpb.StreamingOutputCallResponse
		resp, err := stream.Recv()
		if err == io.EOF {
			return req, resps, nil
		} else if err != nil {
			return req, resps, err
		}
		resps = append(resps, resp)
	}
}

func (s *Server) handleResponse(ctx context.Context, conn *transport.ServerConnection, message any, comp Compressor, options *transport.WriteOptions, encodingEncoder encoding.Encoder) error {
	encodedMsg, err := s.getCodec(conn.ContentSubtype()).Encode(message)
	if err != nil {
		channelz.Error(logger, s.channelz, "grpc: server failed to encode response: ", err)
		return err
	}

	compressedData, flags, err := compress(encodedMsg, comp, s.opts.bufferPool, encodingEncoder)
	if err != nil {
		encodedMsg.Free()
		channelz.Error(logger, s.channelz, "grpc: server failed to compress response: ", err)
		return err
	}

	header, body := msgHeader(encodedMsg, compressedData, flags)

	defer func() {
		compressedData.Free()
		encodedMsg.Free()
	}()

	messageSize := encodedMsg.Size()
	bodySize := body.Size()
	if bodySize > s.opts.maxSendMessageSize {
		return status.Errorf(codes.ResourceExhausted, "grpc: trying to send message larger than max (%d vs. %d)", bodySize, s.opts.maxSendMessageSize)
	}
	err = conn.Write(header, body, options)
	if err == nil {
		for _, handler := range s.opts.statsHandlers {
			handler.HandleRPC(ctx, outPayload(true, message, messageSize, bodySize, time.Now()))
		}
	}
	return err
}

func combineStreamHandlerServers(handler *Server) {
	// Prepend opts.streamHdl to the combining handlers if it exists, since streamHdl will
	// be executed before any other combined handlers.
	handlers := handler.opts.combineStreamHnds
	if handler.opts.streamHdl != nil {
		handlers = append([]StreamHandlerInterceptor{handler.opts.streamHdl}, handler.opts.combineStreamHnds...)
	}

	var combinedHndl StreamHandlerInterceptor
	if len(handlers) == 0 {
		combinedHndl = nil
	} else if len(handlers) == 1 {
		combinedHndl = handlers[0]
	} else {
		combinedHndl = combineStreamHandlers(handlers)
	}

	handler.opts.streamHdl = combinedHndl
}

func (s *Server) processStreamingRPC(ctx context.Context, stream *transport.ServerStream, info *serviceInfo, sd *StreamDesc, trInfo *traceInfo) (err error) {
	if channelz.IsOn() {
		s.incrCallsStarted()
	}
	shs := s.opts.statsHandlers
	var statsBegin *stats.Begin
	if len(shs) != 0 {
		beginTime := time.Now()
		statsBegin = &stats.Begin{
			BeginTime:      beginTime,
			IsClientStream: sd.ClientStreams,
			IsServerStream: sd.ServerStreams,
		}
		for _, sh := range shs {
			sh.HandleRPC(ctx, statsBegin)
		}
	}
	ctx = NewContextWithServerTransportStream(ctx, stream)
	ss := &serverStream{
		ctx:                   ctx,
		s:                     stream,
		p:                     &parser{r: stream, bufferPool: s.opts.bufferPool},
		codec:                 s.getCodec(stream.ContentSubtype()),
		maxReceiveMessageSize: s.opts.maxReceiveMessageSize,
		maxSendMessageSize:    s.opts.maxSendMessageSize,
		trInfo:                trInfo,
		statsHandler:          shs,
	}

	if len(shs) != 0 || trInfo != nil || channelz.IsOn() {
		// See comment in processUnaryRPC on defers.
		defer func() {
			if trInfo != nil {
				ss.mu.Lock()
				if err != nil && err != io.EOF {
					ss.trInfo.tr.LazyLog(&fmtStringer{"%v", []any{err}}, true)
					ss.trInfo.tr.SetError()
				}
				ss.trInfo.tr.Finish()
				ss.trInfo.tr = nil
				ss.mu.Unlock()
			}

			if len(shs) != 0 {
				end := &stats.End{
					BeginTime: statsBegin.BeginTime,
					EndTime:   time.Now(),
				}
				if err != nil && err != io.EOF {
					end.Error = toRPCErr(err)
				}
				for _, sh := range shs {
					sh.HandleRPC(ctx, end)
				}
			}

			if channelz.IsOn() {
				if err != nil && err != io.EOF {
					s.incrCallsFailed()
				} else {
					s.incrCallsSucceeded()
				}
			}
		}()
	}

	if ml := binarylog.GetMethodLogger(stream.Method()); ml != nil {
		ss.binlogs = append(ss.binlogs, ml)
	}
	if s.opts.binaryLogger != nil {
		if ml := s.opts.binaryLogger.GetMethodLogger(stream.Method()); ml != nil {
			ss.binlogs = append(ss.binlogs, ml)
		}
	}
	if len(ss.binlogs) != 0 {
		md, _ := metadata.FromIncomingContext(ctx)
		logEntry := &binarylog.ClientHeader{
			Header:     md,
			MethodName: stream.Method(),
			PeerAddr:   nil,
		}
		if deadline, ok := ctx.Deadline(); ok {
			logEntry.Timeout = time.Until(deadline)
			if logEntry.Timeout < 0 {
				logEntry.Timeout = 0
			}
		}
		if a := md[":authority"]; len(a) > 0 {
			logEntry.Authority = a[0]
		}
		if peer, ok := peer.FromContext(ss.Context()); ok {
			logEntry.PeerAddr = peer.Addr
		}
		for _, binlog := range ss.binlogs {
			binlog.Log(ctx, logEntry)
		}
	}

	// If dc is set and matches the stream's compression, use it.  Otherwise, try
	// to find a matching registered compressor for decomp.
	if rc := stream.RecvCompress(); s.opts.dc != nil && s.opts.dc.Type() == rc {
		ss.dc = s.opts.dc
	} else if rc != "" && rc != encoding.Identity {
		ss.decomp = encoding.GetCompressor(rc)
		if ss.decomp == nil {
			st := status.Newf(codes.Unimplemented, "grpc: Decompressor is not installed for grpc-encoding %q", rc)
			ss.s.WriteStatus(st)
			return st.Err()
		}
	}

	// If cp is set, use it.  Otherwise, attempt to compress the response using
	// the incoming message compression method.
	//
	// NOTE: this needs to be ahead of all handling, https://github.com/grpc/grpc-go/issues/686.
	if s.opts.cp != nil {
		ss.cp = s.opts.cp
		ss.sendCompressorName = s.opts.cp.Type()
	} else if rc := stream.RecvCompress(); rc != "" && rc != encoding.Identity {
		// Legacy compressor not specified; attempt to respond with same encoding.
		ss.comp = encoding.GetCompressor(rc)
		if ss.comp != nil {
			ss.sendCompressorName = rc
		}
	}

	if ss.sendCompressorName != "" {
		if err := stream.SetSendCompress(ss.sendCompressorName); err != nil {
			return status.Errorf(codes.Internal, "grpc: failed to set send compressor: %v", err)
		}
	}

	ss.ctx = newContextWithRPCInfo(ss.ctx, false, ss.codec, ss.cp, ss.comp)

	if trInfo != nil {
		trInfo.tr.LazyLog(&trInfo.firstLine, false)
	}
	var appErr error
	var server any
	if info != nil {
		server = info.serviceImpl
	}
	if s.opts.streamInt == nil {
		appErr = sd.Handler(server, ss)
	} else {
		info := &StreamServerInfo{
			FullMethod:     stream.Method(),
			IsClientStream: sd.ClientStreams,
			IsServerStream: sd.ServerStreams,
		}
		appErr = s.opts.streamInt(server, ss, info, sd.Handler)
	}
	if appErr != nil {
		appStatus, ok := status.FromError(appErr)
		if !ok {
			// Convert non-status application error to a status error with code
			// Unknown, but handle context errors specifically.
			appStatus = status.FromContextError(appErr)
			appErr = appStatus.Err()
		}
		if trInfo != nil {
			ss.mu.Lock()
			ss.trInfo.tr.LazyLog(stringer(appStatus.Message()), true)
			ss.trInfo.tr.SetError()
			ss.mu.Unlock()
		}
		if len(ss.binlogs) != 0 {
			st := &binarylog.ServerTrailer{
				Trailer: ss.s.Trailer(),
				Err:     appErr,
			}
			for _, binlog := range ss.binlogs {
				binlog.Log(ctx, st)
			}
		}
		ss.s.WriteStatus(appStatus)
		// TODO: Should we log an error from WriteStatus here and below?
		return appErr
	}
	if trInfo != nil {
		ss.mu.Lock()
		ss.trInfo.tr.LazyLog(stringer("OK"), false)
		ss.mu.Unlock()
	}
	if len(ss.binlogs) != 0 {
		st := &binarylog.ServerTrailer{
			Trailer: ss.s.Trailer(),
			Err:     appErr,
		}
		for _, binlog := range ss.binlogs {
			binlog.Log(ctx, st)
		}
	}
	return ss.s.WriteStatus(statusOK)
}

func (s *testServer) HandleFullDuplex(stream testgrpc.TestService_FullDuplexCallServer) error {
	ctx := stream.Context()
	md, ok := metadata.FromContext(ctx)
	if ok {
		headerSent := false
		trailerSet := false

		if err := stream.SendHeader(md); err != nil {
			return status.Errorf(status.Code(err), "stream.SendHeader(%v) = %v, want %v", md, err, nil)
		} else {
			headerSent = true
		}

		stream.SetTrailer(testTrailerMetadata)
		trailerSet = true

		for {
			in, recvErr := stream.Recv()
			if recvErr == io.EOF {
				break
			}
			if recvErr != nil {
				return recvErr
			}

			payloadID := payloadToID(in.Payload)
			if payloadID == errorID {
				return fmt.Errorf("got error id: %v", payloadID)
			}

			outputResp := &testpb.StreamingOutputCallResponse{Payload: in.Payload}
			sendErr := stream.Send(outputResp)
			if sendErr != nil {
				return sendErr
			}
		}

		if !headerSent {
			stream.SendHeader(md)
		}
		if !trailerSet {
			stream.SetTrailer(testTrailerMetadata)
		}
	} else {
		return fmt.Errorf("metadata not found in context")
	}

	return nil
}

func TestMigrateArrayTypeModel(t *testing.T) {
	if DB.Dialector.Name() != "postgres" {
		return
	}

	type ArrayTypeModel struct {
		ID              uint
		Number          string     `gorm:"type:varchar(51);NOT NULL"`
		TextArray       []string   `gorm:"type:text[];NOT NULL"`
		NestedTextArray [][]string `gorm:"type:text[][]"`
		NestedIntArray  [][]int64  `gorm:"type:integer[3][3]"`
	}

	var err error
	DB.Migrator().DropTable(&ArrayTypeModel{})

	err = DB.AutoMigrate(&ArrayTypeModel{})
	AssertEqual(t, nil, err)

	ct, err := findColumnType(&ArrayTypeModel{}, "number")
	AssertEqual(t, nil, err)
	AssertEqual(t, "varchar", ct.DatabaseTypeName())

	ct, err = findColumnType(&ArrayTypeModel{}, "text_array")
	AssertEqual(t, nil, err)
	AssertEqual(t, "text[]", ct.DatabaseTypeName())

	ct, err = findColumnType(&ArrayTypeModel{}, "nested_text_array")
	AssertEqual(t, nil, err)
	AssertEqual(t, "text[]", ct.DatabaseTypeName())

	ct, err = findColumnType(&ArrayTypeModel{}, "nested_int_array")
	AssertEqual(t, nil, err)
	AssertEqual(t, "integer[]", ct.DatabaseTypeName())
}

func chainUnaryServerInterceptors(s *Server) {
	// Prepend opts.unaryInt to the chaining interceptors if it exists, since unaryInt will
	// be executed before any other chained interceptors.
	interceptors := s.opts.chainUnaryInts
	if s.opts.unaryInt != nil {
		interceptors = append([]UnaryServerInterceptor{s.opts.unaryInt}, s.opts.chainUnaryInts...)
	}

	var chainedInt UnaryServerInterceptor
	if len(interceptors) == 0 {
		chainedInt = nil
	} else if len(interceptors) == 1 {
		chainedInt = interceptors[0]
	} else {
		chainedInt = chainUnaryInterceptors(interceptors)
	}

	s.opts.unaryInt = chainedInt
}

func TestAdminAuthPass(t *testing.T) {
	credentials := Credentials{"root": "pass123"}
	handler := NewRouter()
	handler.Use(AdminAuth(credentials))
	handler.GET("/admin", func(c *RequestContext) {
		c.String(http.StatusOK, c.MustGet(AuthUserKey).(string))
	})

	writer := httptest.NewRecorder()
	request, _ := http.NewRequest(http.MethodGet, "/admin", nil)
	request.Header.Set("Authorization", adminAuthorizationHeader("root", "pass123"))
	handler.ServeHTTP(writer, request)

	assert.Equal(t, http.StatusOK, writer.Code)
	assert.Equal(t, "root", writer.Body.String())
}

func (s *Server) isRegisteredMethod(serviceMethod string) bool {
	if serviceMethod != "" && serviceMethod[0] == '/' {
		serviceMethod = serviceMethod[1:]
	}
	pos := strings.LastIndex(serviceMethod, "/")
	if pos == -1 { // Invalid method name syntax.
		return false
	}
	service := serviceMethod[:pos]
	method := serviceMethod[pos+1:]
	srv, knownService := s.services[service]
	if knownService {
		if _, ok := srv.methods[method]; ok {
			return true
		}
		if _, ok := srv.streams[method]; ok {
			return true
		}
	}
	return false
}

func TestBasicAuth401WithCustomRealm(t *testing.T) {
	called := false
	accounts := Accounts{"foo": "bar"}
	router := New()
	router.Use(BasicAuthForRealm(accounts, "My Custom \"Realm\""))
	router.GET("/login", func(c *Context) {
		called = true
		c.String(http.StatusOK, c.MustGet(AuthUserKey).(string))
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/login", nil)
	req.Header.Set("Authorization", "Basic "+base64.StdEncoding.EncodeToString([]byte("admin:password")))
	router.ServeHTTP(w, req)

	assert.False(t, called)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Equal(t, "Basic realm=\"My Custom \\\"Realm\\\"\"", w.Header().Get("WWW-Authenticate"))
}

func checkSendCompressor(alias string, serverCompressors []string) error {
	if alias == encoding.Default {
		return nil
	}

	if !grpcutil.IsCompressorAliasRegistered(alias) {
		return fmt.Errorf("compressor not found %q", alias)
	}

	for _, s := range serverCompressors {
		if s == alias {
			return nil // found match
		}
	}
	return fmt.Errorf("server does not support compressor %q", alias)
}

func (s *Service) Handle(lw net.Listener) error {
	s.mu.Lock()
	s.printf("handling")
	s.handling = true
	if s.lw == nil {
		// Handle called after Stop or GracefulStop.
		s.mu.Unlock()
		lw.Close()
		return ErrServiceStopped
	}

	s.handleWG.Add(1)
	defer func() {
		s.handleWG.Done()
		if s.hq.HasFired() {
			// Stop or GracefulStop called; block until done and return nil.
			<-s.done.Done()
		}
	}()

	hs := &handleSocket{
		Listener: lw,
		channelz: channelz.RegisterSocket(&channelz.Socket{
			SocketType:    channelz.SocketTypeHandle,
			Parent:        s.channelz,
			RefName:       lw.Addr().String(),
			LocalAddr:     lw.Addr(),
			SocketOptions: channelz.GetSocketOption(lw)},
		),
	}
	s.lw[hs] = true

	defer func() {
		s.mu.Lock()
		if s.lw != nil && s.lw[hs] {
			hs.Close()
			delete(s.lw, hs)
		}
		s.mu.Unlock()
	}()

	s.mu.Unlock()
	channelz.Info(logger, hs.channelz, "HandleSocket created")

	var tempDelay time.Duration // how long to sleep on accept failure
	for {
		rawConn, err := lw.Accept()
		if err != nil {
			if ne, ok := err.(interface {
				Temporary() bool
			}); ok && ne.Temporary() {
				if tempDelay == 0 {
					tempDelay = 5 * time.Millisecond
				} else {
					tempDelay *= 2
				}
				if max := 1 * time.Second; tempDelay > max {
					tempDelay = max
				}
				s.mu.Lock()
				s.printf("Accept error: %v; retrying in %v", err, tempDelay)
				s.mu.Unlock()
				timer := time.NewTimer(tempDelay)
				select {
				case <-timer.C:
				case <-s.hq.Done():
					timer.Stop()
					return nil
				}
				continue
			}
			s.mu.Lock()
			s.printf("done handling; Accept = %v", err)
			s.mu.Unlock()

			if s.hq.HasFired() {
				return nil
			}
			return err
		}
		tempDelay = 0
		// Start a new goroutine to deal with rawConn so we don't stall this Accept
		// loop goroutine.
		//
		// Make sure we account for the goroutine so GracefulStop doesn't nil out
		// s.conns before this conn can be added.
		s.handleWG.Add(1)
		go func() {
			s.handleRawConn(lw.Addr().String(), rawConn)
			s.handleWG.Done()
		}()
	}
}

func bmEncodeMsg(b *testing.B, mSize int) {
	msg := &perfpb.Buffer{Body: make([]byte, mSize)}
	encodeData, _ := encode(getCodec(protoenc.Name), msg)
	encodedSz := int64(len(encodeData))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encode(getCodec(protoenc.Name), msg)
	}
	b.SetBytes(encodedSz)
}

