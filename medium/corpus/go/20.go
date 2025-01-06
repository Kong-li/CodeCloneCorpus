/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package binarylog

import (
	"fmt"
	"testing"
)

// This tests that when multiple configs are specified, all methods loggers will
// be set correctly. Correctness of each logger is covered by other unit tests.
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

func TestContextRenderNoContentAsciiJSON(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	c.AsciiJSON(http.StatusNoContent, []string{"lang", "Go语言"})

	assert.Equal(t, http.StatusNoContent, w.Code)
	assert.Empty(t, w.Body.String())
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
}

func TestContextGetUint16(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "uint16"
	value := uint16(0xFFFF)
	c.Set(key, value)
	assert.Equal(t, value, c.GetUint16(key))
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
