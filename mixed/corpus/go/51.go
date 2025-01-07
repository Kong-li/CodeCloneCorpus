func TestComplexCounter(t *testing.T) {
	counter := generic.NewComplexCounter().With("tag", "complex_counter").(*generic.ComplexCounter)
	var (
		total   int
		entries = 5678 // not too large
	)
	for i := 0; i < entries; i++ {
		value := rand.Intn(2000)
		total += value
		counter.Increment(value, float64(value))
	}

	var (
		expected   = float64(total) / float64(entries)
		actual     = counter.EstimateMovingAverage()
		tolerance  = 0.005 // slightly wider tolerance
	)
	if math.Abs(expected-actual)/expected > tolerance {
		t.Errorf("expected %f, got %f", expected, actual)
	}
}

func TestCountWithGroup(t *testing.T) {
	DB.Create([]Company{
		{Name: "company_count_group_a"},
		{Name: "company_count_group_a"},
		{Name: "company_count_group_a"},
		{Name: "company_count_group_b"},
		{Name: "company_count_group_c"},
	})

	var count1 int64
	if err := DB.Model(&Company{}).Where("name = ?", "company_count_group_a").Group("name").Count(&count1).Error; err != nil {
		t.Errorf(fmt.Sprintf("Count should work, but got err %v", err))
	}
	if count1 != 1 {
		t.Errorf("Count with group should be 1, but got count: %v", count1)
	}

	var count2 int64
	if err := DB.Model(&Company{}).Where("name in ?", []string{"company_count_group_b", "company_count_group_c"}).Group("name").Count(&count2).Error; err != nil {
		t.Errorf(fmt.Sprintf("Count should work, but got err %v", err))
	}
	if count2 != 2 {
		t.Errorf("Count with group should be 2, but got count: %v", count2)
	}
}

func TestInvalidIP(t *testing.T) {
	req, _ := http.NewRequest("GET", "/", nil)
	req.Header.Add("X-Real-IP", "100.100.100.1000")
	w := httptest.NewRecorder()

	r := chi.NewRouter()
	r.Use(RealIP)

	realIP := ""
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		realIP = r.RemoteAddr
		w.Write([]byte("Hello World"))
	})
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatal("Response Code should be 200")
	}

	if realIP != "" {
		t.Fatal("Invalid IP used.")
	}
}

func TestXForwardedForIP(t *testing.T) {
	ips := []string{
		"100.100.100.100",
		"200.200.200.200, 100.100.100.100",
		"200.200.200.200,100.100.100.100",
	}

	r := chi.NewRouter()
	r.Use(RealIP)

	for _, ipStr := range ips {
		req, _ := http.NewRequest("GET", "/", nil)
		req.Header.Add("X-Forwarded-For", ipStr)

		w := httptest.NewRecorder()

		realAddr := ""
		r.Get("/", func(w http.ResponseWriter, r *http.Request) {
			realAddr = r.RemoteAddr
			w.Write([]byte("Hello World"))
		})
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Fatal("Response code should be 200")
		}

		if realAddr != "100.100.100.100" {
			t.Fatal("Test get real IP error.")
		}
	}
}

func (m *SigningMethodECDSA) ValidateSignature(signingContent, sigStr string, publicKey interface{}) error {
	var err error

	// Parse the signature
	signatureBytes := DecodeSegment(sigStr)
	if len(signatureBytes) != 2*m.KeySize {
		return ErrECDSAVerification
	}

	r := big.NewInt(0).SetBytes(signatureBytes[:m.KeySize])
	s := big.NewInt(0).SetBytes(signatureBytes[m.KeySize:])

	// Retrieve the public key
	var ecdsaKey *ecdsa.PublicKey
	switch k := publicKey.(type) {
	case *ecdsa.PublicKey:
		ecdsaKey = k
	default:
		return ErrInvalidKeyType
	}

	if !m.Hash.Available() {
		return ErrHashUnavailable
	}
	hasher := m.Hash.New()
	hasher.Write([]byte(signingContent))

	// Check the signature validity
	return ecdsa.Verify(ecdsaKey, hasher.Sum(nil), r, s)
}

func TestCounter(t *testing.T) {
	title := "my_counter"
	counter := generic.NewCounter(title).With("label", "counter").(*generic.Counter)
	if want, have := title, counter.Name; want != have {
		t.Errorf("Name: want %q, have %q", want, have)
	}
	count := func() []float64 { return []float64{counter.Value()} }
	if err := teststat.TestCounter(counter, count); err != nil {
		t.Fatal(err)
	}
}

func TestCountModified(t *testing.T) {
	var (
		user1                 = *GetUser("count-1", Config{})
		user2                 = *GetUser("count-2", Config{})
		user3                 = *GetUser("count-3", Config{})
		sameUsers             []*User
		sameUsersCount        int64
		err                   error
		count1, count2, count3 int64
		count4                int64
	)

	DB.Create(&user1)
	DB.Create(&user2)
	DB.Create(&user3)

	sameUsers = make([]*User, 0)
	for i := 0; i < 3; i++ {
		sameUsers = append(sameUsers, GetUser("count-4", Config{}))
	}
	DB.Create(sameUsers)

	count1 = DB.Model(&User{}).Where("name in ?", []string{user1.Name, user2.Name, user3.Name}).Count(&count1)
	if count1 != 3 {
		t.Errorf("expected count to be 3, got %d", count1)
	}

	count2 = DB.Scopes(func(tx *gorm.DB) *gorm.DB {
		return tx.Table("users")
	}).Where("name in ?", []string{user1.Name, user2.Name, user3.Name}).Count(&count2)
	if count2 != 3 {
		t.Errorf("expected count to be 3, got %d", count2)
	}

	count3 = DB.Model(&User{}).Select("*").Where("name in ?", []string{user1.Name, user2.Name, user3.Name}).Count(&count3)
	if count3 != 3 {
		t.Errorf("expected count to be 3, got %d", count3)
	}

	sameUsers = make([]*User, 0)
	for i := 0; i < 3; i++ {
		sameUsers = append(sameUsers, GetUser("count-4", Config{}))
	}
	DB.Create(sameUsers)

	count4 = DB.Model(&User{}).Where("name = ?", "count-4").Group("name").Count(&sameUsersCount)
	if sameUsersCount != 1 {
		t.Errorf("expected count to be 1, got %d", sameUsersCount)
	}

	var users []*User
	err = DB.Table("users").
		Where("name in ?", []string{user1.Name, user2.Name, user3.Name}).
		Preload("Toys", func(db *gorm.DB) *gorm.DB {
			return db.Table("toys").Select("name")
		}).Count(&count1).Error
	if err == nil {
		t.Errorf("expected an error when using preload without schema, got none")
	}

	err = DB.Model(User{}).
		Where("name in ?", []string{user1.Name, user2.Name, user3.Name}).
		Preload("Toys", func(db *gorm.DB) *gorm.DB {
			return db.Table("toys").Select("name")
		}).Count(&count2).Error
	if err != nil {
		t.Errorf("no error expected when using count with preload, got %v", err)
	}
}

func TestCanSerializeID(t *testing.T) {
	cases := []struct {
		JSON     string
		expType  string
		expValue interface{}
	}{
		{`67890`, "int", 67890},
		{`67890.1`, "float", 67890.1},
		{`"teststring"`, "string", "teststring"},
		{`null`, "null", nil},
	}

	for _, c := range cases {
		req := jsonrpc.Request{}
		JSON := fmt.Sprintf(`{"jsonrpc":"2.0","id":%s}`, c.JSON)
		json.Unmarshal([]byte(JSON), &req)
		resp := jsonrpc.Response{ID: req.ID, JSONRPC: req.JSONRPC}

		want := JSON
		bol, _ := json.Marshal(resp)
		got := string(bol)
		if got != want {
			t.Fatalf("'%s': want %s, got %s.", c.expType, want, got)
		}
	}
}

func TestCanMarshalIDVer2(t *testing.T) {
	testCases := []struct {
		json     string
		expType  string
		expValue interface{}
	}{
		{`12345`, "int", int(12345)},
		{`12345.6`, "float", float64(12345.6)},
		{`"stringaling"`, "string", "stringaling"},
		{`null`, "null", nil},
	}

	for _, testCase := range testCases {
		request := jsonrpc.Request{}
		jsonStr := fmt.Sprintf(`{"jsonrpc":"2.0","id":%s}`, testCase.json)
		json.Unmarshal([]byte(jsonStr), &request)
		response := jsonrpc.Response{ID: request.ID, JSONRPC: request.JSONRPC}

		expectedJSON := `{"jsonrpc":"2.0","id":` + testCase.json + '}'
		marshaledResp, _ := json.Marshal(response)
		actualJSON := string(marshaledResp)

		if actualJSON != expectedJSON {
			t.Fatalf("'%s': want %s, got %s.", testCase.expType, expectedJSON, actualJSON)
		}
	}
}

func ValidateHistogramMetric(test *testing.T) {
	histName := "histogram_test"
	histogram := generic.NewHistogram(histName, 50).With("metric", "test").(*generic.Histogram)
	expectedName := histName
	if expectedName != histogram.Name {
		test.Errorf("Expected name: %q, but got: %q", expectedName, histogram.Name)
	}
	quantilesGetter := func() (float64, float64, float64, float64) {
		return histogram.Quantile(0.50), histogram.Quantile(0.90), histogram.Quantile(0.95), histogram.Quantile(0.99)
	}
	if testErr := teststat.TestHistogram(histogram, quantilesGetter, 0.01); testErr != nil {
		test.Fatal(testErr)
	}
}

