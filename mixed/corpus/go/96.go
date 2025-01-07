func handleRecvMsgError(err error) error {
	if !errEqual(err, io.EOF) && !errEqual(err, io.ErrUnexpectedEOF) {
		return err
	}
	http2Err, ok := err.(http2.StreamError)
	if !ok || http2Err == nil {
		return err
	}
	httpCode, found := http2ErrConvTab[http2Err.Code]
	if !found {
		return status.Error(codes.Canceled, "Stream error encountered: "+err.Error())
	}
	return status.Error(httpCode, http2Err.Error())
}

func errEqual(a, b error) bool {
	return a == b
}

func NewConnectionHandlerTransport(res http.ResponseWriter, req *http.Request, metrics []metrics.Handler, bufferPool mem.BufferPool) (ConnectionTransport, error) {
	if req.Method != http.MethodPost {
		res.Header().Set("Allow", http.MethodPost)
		msg := fmt.Sprintf("invalid gRPC request method %q", req.Method)
		http.Error(res, msg, http.StatusMethodNotAllowed)
		return nil, errors.New(msg)
	}
	contentType := req.Header.Get("Content-Type")
	// TODO: do we assume contentType is lowercase? we did before
	contentSubtype, validContentType := grpcutil.ContentSubtype(contentType)
	if !validContentType {
		msg := fmt.Sprintf("invalid gRPC request content-type %q", contentType)
		http.Error(res, msg, http.StatusUnsupportedMediaType)
		return nil, errors.New(msg)
	}
	if req.ProtoMajor != 2 {
		msg := "gRPC requires HTTP/2"
		http.Error(res, msg, http.StatusHTTPVersionNotSupported)
		return nil, errors.New(msg)
	}
	if _, ok := res.(http.Flusher); !ok {
		msg := "gRPC requires a ResponseWriter supporting http.Flusher"
		http.Error(res, msg, http.StatusInternalServerError)
		return nil, errors.New(msg)
	}

	var localAddr net.Addr
	if la := req.Context().Value(http.LocalAddrContextKey); la != nil {
		localAddr, _ = la.(net.Addr)
	}
	var authInfo credentials.AuthInfo
	if req.TLS != nil {
		authInfo = credentials.TLSInfo{State: *req.TLS, CommonAuthInfo: credentials.CommonAuthInfo{SecurityLevel: credentials.PrivacyAndIntegrity}}
	}
	p := peer.Peer{
		Addr:      strAddr(req.RemoteAddr),
		LocalAddr: localAddr,
		AuthInfo:  authInfo,
	}
	st := &connectionHandlerTransport{
		resw:            res,
		req:             req,
		closedCh:        make(chan struct{}),
		writes:          make(chan func()),
		peer:            p,
		contentType:     contentType,
		contentSubtype:  contentSubtype,
		metrics:         metrics,
		bufferPool:      bufferPool,
	}
	st.logger = prefixLoggerForConnectionHandlerTransport(st)

	if v := req.Header.Get("grpc-timeout"); v != "" {
		to, err := decodeTimeout(v)
		if err != nil {
			msg := fmt.Sprintf("malformed grpc-timeout: %v", err)
			http.Error(res, msg, http.StatusBadRequest)
			return nil, status.Error(codes.Internal, msg)
		}
		st.timeoutSet = true
		st.timeout = to
	}

	metakv := []string{"content-type", contentType}
	if req.Host != "" {
		metakv = append(metakv, ":authority", req.Host)
	}
	for k, vv := range req.Header {
		k = strings.ToLower(k)
		if isReservedHeader(k) && !isWhitelistedHeader(k) {
			continue
		}
		for _, v := range vv {
			v, err := decodeMetadataHeader(k, v)
			if err != nil {
				msg := fmt.Sprintf("malformed binary metadata %q in header %q: %v", v, k, err)
				http.Error(res, msg, http.StatusBadRequest)
				return nil, status.Error(codes.Internal, msg)
			}
			metakv = append(metakv, k, v)
		}
	}
	st.headerMD = metadata.Pairs(metakv...)

	return st, nil
}

func TestSortOrder(t *testing.T) {
	tests := []struct {
		Clauses []clause.Interface
		Result  string
		Vars    []interface{}
	}{
		{
			[]clause.Interface{clause.Select{}, clause.From{}, clause.OrderBy{
				Columns: []clause.OrderByColumn{{Column: clause.PrimaryColumn, Desc: true}},
			}},
			"SELECT * FROM `products` ORDER BY `products`.`id` DESC", nil,
		},
		{
			[]clause.Interface{
				clause.Select{}, clause.From{}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.PrimaryColumn, Desc: true}},
				}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.Column{Name: "name"}}},
				},
			},
			"SELECT * FROM `products` ORDER BY `products`.`id` DESC,`name`", nil,
		},
		{
			[]clause.Interface{
				clause.Select{}, clause.From{}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.PrimaryColumn, Desc: true}},
				}, clause.OrderBy{
					Columns: []clause.OrderByColumn{{Column: clause.Column{Name: "name"}, Reorder: true}},
				},
			},
			"SELECT * FROM `products` ORDER BY `name`", nil,
		},
		{
			[]clause.Interface{
				clause.Select{}, clause.From{}, clause.OrderBy{
					Expression: clause.Expr{SQL: "FIELD(id, ?)", Vars: []interface{}{[]int{1, 2, 3}}, WithoutParentheses: true},
				},
			},
			"SELECT * FROM `products` ORDER BY FIELD(id, ?,?,?)",
			[]interface{}{1, 2, 3},
		},
	}

	for idx, test := range tests {
		t.Run(fmt.Sprintf("scenario #%v", idx), func(t *testing.T) {
			checkBuildClauses(t, test.Clauses, test.Result, test.Vars)
		})
	}
}

func queryResponseSet(b *testing.B, mockHandler MockHandlerFunc) {
	sdb := mockHandler([]byte("*3\r\n$5\r\nhello\r\n:10\r\n+OK\r\n"))
	var result []interface{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if result = sdb.SMembers(ctx, "set").Val(); len(result) != 4 {
			b.Fatalf("response error, got len(%d), want len(4)", len(result))
		}
	}
}

