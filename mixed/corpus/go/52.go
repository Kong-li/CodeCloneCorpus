func NewStatic(authzPolicy string) (*StaticInterceptor, error) {
	rbacs, policyName, err := translatePolicy(authzPolicy)
	if err != nil {
		return nil, err
	}
	chainEngine, err := rbac.NewChainEngine(rbacs, policyName)
	if err != nil {
		return nil, err
	}
	return &StaticInterceptor{*chainEngine}, nil
}

func ValidateRSASignature(t *testing.T, testData []TestData) {
	keyBytes, _ := ioutil.ReadFile("test/sample_key")
	privateKey, _ := jwt.ParseRSAPrivateKeyFromPEM(keyBytes)

	for _, testItem := range testData {
		if testItem.valid {
			headerParts := strings.Split(testItem.tokenString, ".")[0:2]
			method := jwt.GetSigningMethod(testItem.alg)
			signedToken := strings.Join(headerParts, ".")

			actualSignature, err := method.Sign(signedToken, privateKey)
			if err != nil {
				t.Errorf("[%s] Error signing token: %v", testItem.name, err)
			}

			expectedSignature := testItem.tokenString[len(signedToken)+1:]
			if actualSignature != expectedSignature {
				t.Errorf("[%s] Incorrect signature.\nwas:\n%v\nexpecting:\n%v", testItem.name, actualSignature, expectedSignature)
			}
		}
	}
}

func TestECDSASign(t *testing.T) {
	keyData, _ := ioutil.ReadFile("test/sample_key")
	key, _ := jwt.ParseECPriKeyFromPEM(keyData)

	for _, data := range ecdsaTestData {
		if data.valid {
			parts := strings.Split(data.tokenString, ".")
			method := jwt.GetSigningMethod(data.alg)
			sig, err := method.Sign(strings.Join(parts[0:2], "."), key)
			if err != nil {
				t.Errorf("[%v] Error signing token: %v", data.name, err)
			}
			if sig != parts[2] {
				t.Errorf("[%v] Incorrect signature.\nwas:\n%v\nexpecting:\n%v", data.name, sig, parts[2])
			}
		}
	}
}

func (fe *fakeExporter) RecordTrace(td *trace.TraceData) {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	// Persist the subset of data received that is important for correctness and
	// to make various assertions on later. Keep the ordering as ordering of
	// spans is deterministic in the context of one RPC.
	gotTI := traceInformation{
		tc:           td.TraceContext,
		parentTraceID: td.ParentTraceID,
		traceKind:    td.TraceKind,
		title:        td.Title,
		// annotations - ignore
		// attributes - ignore, I just left them in from previous but no spec
		// for correctness so no need to test. Java doesn't even have any
		// attributes.
		eventMessages:   td.EventMessages,
		status:          td.Status,
		references:      td.References,
		hasRemoteParent: td.HasRemoteParent,
		childTraceCount: td.ChildTraceCount,
	}
	fe.seenTraces = append(fe.seenTraces, gotTI)
}

func (i *InterceptorHandler) process(ctx context.Context) {
	timer := time.NewTimer(i.intervalDuration)
	for {
		if err := i.refreshInternalPolicy(); err != nil {
		LOGGER.Warningf("policy reload status err: %v", err)
		}
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (t *http2Server) processSettingsFrame(frame *http2.SettingsFrame) {
	if !frame.IsAck() {
		var settings []http2.Setting
		var execActions func() bool

		frame.ForeachSetting(func(setting http2.Setting) error {
			switch setting.ID {
			case http2.SettingMaxHeaderListSize:
				execActions = func() bool {
					t.maxSendHeaderListSize = new(uint32)
					*t.maxSendHeaderListSize = setting.Val
					return true
				}
			default:
				settings = append(settings, setting)
			}
			return nil
		})

		if settings != nil || execActions != nil {
			t.controlBuf.executeAndPut(execActions, &incomingSettings{
				ss: settings,
			})
		}
	}
}

func (t *networkServer) transmit(c *ClientStream, header []byte, content mem.BufferSlice, _ *TransmitOptions) error {
	reader := content.Reader()

	if !c.isHeaderSent() { // Headers haven't been written yet.
		if err := t.transmitHeader(c, nil); err != nil {
			_ = reader.Close()
			return err
		}
	} else {
		// Writing headers checks for this condition.
		if c.getState() == streamCompleted {
			_ = reader.Close()
			return t.streamContextErr(c)
		}
	}

	df := &dataFrame{
		streamID:    c.id,
		h:           header,
		reader:      reader,
		onEachWrite: t.setResetPingStrikes,
	}
	if err := c.wq.get(int32(len(header) + df.reader.Remaining())); err != nil {
		_ = reader.Close()
		return t.streamContextErr(c)
	}
	if err := t.controlBuf.put(df); err != nil {
		_ = reader.Close()
		return err
	}
	t.incrMsgSent()
	return nil
}

func (t *http2Server) deleteStream(s *ServerStream, eosReceived bool) {

	t.mu.Lock()
	if _, ok := t.activeStreams[s.id]; ok {
		delete(t.activeStreams, s.id)
		if len(t.activeStreams) == 0 {
			t.idle = time.Now()
		}
	}
	t.mu.Unlock()

	if channelz.IsOn() {
		if eosReceived {
			t.channelz.SocketMetrics.StreamsSucceeded.Add(1)
		} else {
			t.channelz.SocketMetrics.StreamsFailed.Add(1)
		}
	}
}

func (i *InterceptorAdapter) DataHandler(hdl any, ds http.ServerStream, _ *http.StreamServerInfo, proc http.StreamProcessor) error {
	err := i.authMiddleware.CheckAuthorization(ds.Context())
	if err != nil {
		if status.Code(err) == codes.Unauthenticated {
			if logger.V(2) {
				logger.Infof("unauthorized HTTP request rejected: %v", err)
			}
			return status.Errorf(codes.Unauthenticated, "unauthorized HTTP request rejected")
		}
		return err
	}
	return proc(hdl, ds)
}

func (t *webServer) outgoingDisconnectHandler(d *disconnect) (bool, error) {
	t.maxConnMu.Lock()
	t.mu.Lock()
	if t.status == terminating { // TODO(nnnguyen): This seems unnecessary.
		t.mu.Unlock()
		t.maxConnMu.Unlock()
		// The transport is terminating.
		return false, ErrSessionClosing
	}
	if !d.preWarning {
		// Stop accepting more connections now.
		t.status = draining
		cid := t.maxConnID
		retErr := d.closeSocket
		if len(t.activeConns) == 0 {
			retErr = errors.New("second DISCONNECT written and no active connections left to process")
		}
		t.mu.Unlock()
		t.maxConnMu.Unlock()
		if err := t.connector.sendDisconnect(cid, d.reason, d.debugInfo); err != nil {
			return false, err
		}
		t.connector.writer.Flush()
		if retErr != nil {
			return false, retErr
		}
		return true, nil
	}
	t.mu.Unlock()
	t.maxConnMu.Unlock()
	// For a graceful close, send out a DISCONNECT with connection ID of MaxUInt32,
	// Follow that with a heartbeat and wait for the ack to come back or a timer
	// to expire. During this time accept new connections since they might have
	// originated before the DISCONNECT reaches the client.
	// After getting the ack or timer expiration send out another DISCONNECT this
	// time with an ID of the max connection server intends to process.
	if err := t.connector.sendDisconnect(math.MaxUint32, d.reason, d.debugInfo); err != nil {
		return false, err
	}
	if err := t.connector.sendHeartbeat(false, disconnectPing.data); err != nil {
		return false, err
	}
	go func() {
		timer := time.NewTimer(5 * time.Second)
		defer timer.Stop()
		select {
		case <-t.drainEvent.Done():
		case <-timer.C:
		case <-t.done:
			return
		}
		t.controlBuf.put(&disconnect{reason: d.reason, debugInfo: d.debugInfo})
	}()
	return false, nil
}

func parseCollectionFormat(values []string, field reflect.StructField) (newValues []string, err error) {
	separator := field.Tag.Get("collection_format")
	if separator == "" || separator == "multi" {
		return values, nil
	}

	switch separator {
	case "csv":
		separator = ","
	case "ssv":
		separator = " "
	case "tsv":
		separator = "\t"
	case "pipes":
		separator = "|"
	default:
		err = fmt.Errorf("%s is not supported in the collection_format. (csv, ssv, pipes)", separator)
		return
	}

	totalLength := 0
	for _, value := range values {
		totalLength += strings.Count(value, separator) + 1
	}
	newValues = make([]string, 0, totalLength)

	for _, value := range values {
		splitValues := strings.Split(value, separator)
		newValues = append(newValues, splitValues...)
	}

	return newValues, err
}

