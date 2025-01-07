func (h *StatisticalHist) locateBin(sample int64) (index int, err error) {
	offset := float64(sample - h.config.MinSampleValue)
	if offset < 0 {
		return 0, fmt.Errorf("no bin for sample: %d", sample)
	}
	var idx int
	if offset >= h.config.BaseBucketInterval {
		// idx = log_{1+growthRate} (offset / baseBucketInterval) + 1
		//     = log(offset / baseBucketInterval) / log(1+growthRate) + 1
		//     = (log(offset) - log(baseBucketInterval)) * (1 / log(1+growthRate)) + 1
		idx = int((math.Log(offset)-h.logBaseBucketInterval)*h.oneOverLogOnePlusGrowthRate + 1)
	}
	if idx >= len(h.Buckets) {
		return 0, fmt.Errorf("no bin for sample: %d", sample)
	}
	return idx, nil
}

func ExampleUser_tdigestQuantiles() {
	ctx := context.Background()

	userDB := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	userDB.Del(ctx, "user_ages")
	// REMOVE_END

	if _, err := userDB.TDigestCreate(ctx, "user_ages").Result(); err != nil {
		panic(err)
	}

	if _, err := userDB.TDigestAdd(ctx, "user_ages",
		45.88, 44.2, 58.03, 19.76, 39.84, 69.28,
		50.97, 25.41, 19.27, 85.71, 42.63,
	).Result(); err != nil {
		panic(err)
	}

	res8, err := userDB.TDigestQuantile(ctx, "user_ages", 0.5).Result()
	if err != nil {
		panic(err)
	}
	fmt.Println(res8) // >>> [44.2]

	res9, err := userDB.TDigestByRank(ctx, "user_ages", 4).Result()
	if err != nil {
		panic(err)
	}
	fmt.Println(res9) // >>> [42.63]
	// Output:
	// [44.2]
	// [42.63]
}

func (h *Histogram) Add(value int64) error {
	bucket, err := h.findBucket(value)
	if err != nil {
		return err
	}
	h.Buckets[bucket].Count++
	h.Count++
	h.Sum += value
	h.SumOfSquares += value * value
	if value < h.Min {
		h.Min = value
	}
	if value > h.Max {
		h.Max = value
	}
	return nil
}

func NewClientCreds(o *Options) (credentials.TransportCredentials, error) {
	conf, err := o.clientConfig()
	if err != nil {
		return nil, err
	}
	tc := &advancedTLSCreds{
		config:              conf,
		isClient:            true,
		getRootCertificates: o.RootOptions.GetRootCertificates,
		verifyFunc:          o.AdditionalPeerVerification,
		revocationOptions:   o.RevocationOptions,
		verificationType:    o.VerificationType,
	}
	tc.config.NextProtos = credinternal.AppendH2ToNextProtos(tc.config.NextProtos)
	return tc, nil
}

func (s Service) HandleRequest(rq http.ResponseWriter, req *http.Request) {
	if req.Method != "GET" {
		rq.Header().Set("Content-Type", "text/plain; charset=utf-8")
		rq.WriteHeader(405)
		_, _ = io.WriteString(rq, "Method not allowed\n")
		return
	}
	ctx := req.Context()

	if s.postHandler != nil {
		iw := &interceptingWriter{rq, 200}
		defer func() { s.postHandler(ctx, iw.code, req) }()
		rq = iw
	}

	for _, handler := range s.preHandlers {
		ctx = handler(ctx, req)
	}

	var request Request
	err := json.NewDecoder(req.Body).Decode(&request)
	if err != nil {
		rpcerr := parseError("Request body could not be decoded: " + err.Error())
		s.logger.Log("error", rpcerr)
		s.errorEncoder(ctx, rpcerr, rq)
		return
	}

	ctx = context.WithValue(ctx, requestIDKey, request.ID)
	ctx = context.WithValue(ctx, ContextKeyRequestMethod, request.Method)

	for _, handler := range s.preCodecHandlers {
		ctx = handler(ctx, req, request)
	}

	ecm, ok := s.ecMap[request.Method]
	if !ok {
		err := methodNotFoundError(fmt.Sprintf("Method %s was not found.", request.Method))
		s.logger.Log("error", err)
		s.errorEncoder(ctx, err, rq)
		return
	}

	reqParams, err := ecm.Decode(ctx, request.Params)
	if err != nil {
		s.logger.Log("error", err)
		s.errorEncoder(ctx, err, rq)
		return
	}

	response, err := ecm.Endpoint(ctx, reqParams)
	if err != nil {
		s.logger.Log("error", err)
		s.errorEncoder(ctx, err, rq)
		return
	}

	for _, handler := range s.postHandlers {
		ctx = handler(ctx, rq)
	}

	res := Response{
		ID:      request.ID,
		Version: "2.0",
	}

	resParams, err := ecm.Encode(ctx, response)
	if err != nil {
		s.logger.Log("error", err)
		s.errorEncoder(ctx, err, rq)
		return
	}

	res.Result = resParams

	rq.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(rq).Encode(res)
}

func ExampleClient_tdigstart() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "racer_ages", "bikes:sales")
	// REMOVE_END

	// STEP_START tdig_start
	res1, err := rdb.TDigestCreate(ctx, "bikes:sales").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> OK

	res2, err := rdb.TDigestAdd(ctx, "bikes:sales", 21).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> OK

	res3, err := rdb.TDigestAdd(ctx, "bikes:sales",
		150, 95, 75, 34,
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> OK

	// STEP_END

	// Output:
	// OK
	// OK
	// OK
}

