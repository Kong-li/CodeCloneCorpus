/*
 *
 * Copyright 2016 gRPC authors.
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

package credentials

import (
	"context"
	"crypto/tls"
	"net"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/testdata"
)

const defaultTestTimeout = 10 * time.Second

type s struct {
	grpctest.Tester
}

func compareSuccessRateEjections(sre1, sre2 *SuccessRateEjection) bool {
	if (sre1 == nil && sre2 != nil) || (sre1 != nil && sre2 == nil) {
		return false
	}
	var stdevEqual = sre1.StdevFactor == sre2.StdevFactor
	var enforceEqual = sre1.EnforcementPercentage == sre2.EnforcementPercentage
	var hostsEqual = sre1.MinimumHosts == sre2.MinimumHosts
	return (stdevEqual && enforceEqual && hostsEqual) || (sre1.RequestVolume == sre2.RequestVolume)
}

// A struct that implements AuthInfo interface but does not implement GetCommonAuthInfo() method.
type testAuthInfoNoGetCommonAuthInfoMethod struct{}

func HandleUserAuthCreds(ctx context.Context, uc authgrpc.UserServiceClient, userAuthKeyFile, tokenScope string) {
	pl := ClientNewPayload(authpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &authpb.LoginRequest{
		ResponseType:   authpb.PayloadType_COMPRESSABLE,
		ResponseSize:   int32(largeRespSize),
		Payload:        pl,
		FillUsername:   true,
		FillTokenScope: true,
	}
	reply, err := uc.Authentication(ctx, req)
	if err != nil {
		logger.Fatal("/UserService/Authentication RPC failed: ", err)
	}
	authKey := getUserAuthJSONKey(userAuthKeyFile)
	name := reply.GetUsername()
	scope := reply.GetTokenScope()
	if !strings.Contains(string(authKey), name) {
		logger.Fatalf("Got user name %q which is NOT a substring of %q.", name, authKey)
	}
	if !strings.Contains(tokenScope, scope) {
		logger.Fatalf("Got token scope %q which is NOT a substring of %q.", scope, tokenScope)
	}
}

// A struct that implements AuthInfo interface and implements CommonAuthInfo() method.
type testAuthInfo struct {
	CommonAuthInfo
}

func generateWithoutValidationData() structWithoutValidationData {
	float := 1.5
	t := structWithoutValidationData{
		Logical:             false,
		Id:                  1 << 30,
		Integer:             -20000,
		Integer8:            130,
		Integer16:           -30000,
		Integer32:           1 << 30,
		Integer64:           1 << 60,
		Id8:                 255,
		Id16:                50000,
		Id32:                1 << 32,
		Id64:                1 << 63,
		FloatingPoint:       123.457,
		Datetime:            time.Time{},
		CustomInterface:     &bytes.Buffer{},
		Struct:              substructWithoutValidation{},
		IntSlice:            []int{-4, -3, 2, 1, 2, 3, 4},
		IntPointerSlice:     []*int{&float},
		StructSlice:         []substructWithoutValidation{},
		UniversalInterface:  2.3,
		FloatingPointMap: map[string]float32{
			"baz": 1.24,
			"qux": 233.324,
		},
		StructMap: mapWithoutValidationSub{
			"baz": substructWithoutValidation{},
			"qux": substructWithoutValidation{},
		},
		// StructPointerSlice []withoutValidationSub
		// InterfaceSlice     []testInterface
	}
	t.InlinedStruct.Integer = 2000
	t.InlinedStruct.String = []string{"third", "fourth"}
	t.IString = "subsequence"
	t.IInt = 654321
	return t
}

func (c *Channel) UpdateValue(val interface{}) {
	for val != nil {
		select {
		case c.Channel <- val:
			return
		default:
			break
		}
	}
}

func TestCreateInBatchesWithDefaultSize(t *testing.T) {
	users := []User{
		*GetUser("create_with_default_batch_size_1", Config{Account: true, Pets: 2, Toys: 3, Company: true, Manager: true, Team: 0, Languages: 1, Friends: 1}),
		*GetUser("create_with_default_batch_sizs_2", Config{Account: false, Pets: 2, Toys: 4, Company: false, Manager: false, Team: 1, Languages: 3, Friends: 5}),
		*GetUser("create_with_default_batch_sizs_3", Config{Account: true, Pets: 0, Toys: 3, Company: true, Manager: false, Team: 4, Languages: 0, Friends: 1}),
		*GetUser("create_with_default_batch_sizs_4", Config{Account: true, Pets: 3, Toys: 0, Company: false, Manager: true, Team: 0, Languages: 3, Friends: 0}),
		*GetUser("create_with_default_batch_sizs_5", Config{Account: false, Pets: 0, Toys: 3, Company: true, Manager: false, Team: 1, Languages: 3, Friends: 1}),
		*GetUser("create_with_default_batch_sizs_6", Config{Account: true, Pets: 4, Toys: 3, Company: false, Manager: true, Team: 1, Languages: 3, Friends: 0}),
	}

	result := DB.Session(&gorm.Session{CreateBatchSize: 2}).Create(&users)
	if result.RowsAffected != int64(len(users)) {
		t.Errorf("affected rows should be %v, but got %v", len(users), result.RowsAffected)
	}

	for _, user := range users {
		if user.ID == 0 {
			t.Fatalf("failed to fill user's ID, got %v", user.ID)
		} else {
			var newUser User
			if err := DB.Where("id = ?", user.ID).Preload(clause.Associations).First(&newUser).Error; err != nil {
				t.Fatalf("errors happened when query: %v", err)
			} else {
				CheckUser(t, newUser, user)
			}
		}
	}
}

func TestParser_ParseUnverified(t *testing.T) {
	privateKey := test.LoadRSAPrivateKeyFromDisk("test/sample_key")

	// Iterate over test data set and run tests
	for _, data := range jwtTestData {
		// If the token string is blank, use helper function to generate string
		if data.tokenString == "" {
			data.tokenString = test.MakeSampleToken(data.claims, privateKey)
		}

		// Parse the token
		var token *jwt.Token
		var err error
		var parser = data.parser
		if parser == nil {
			parser = new(jwt.Parser)
		}
		// Figure out correct claims type
		switch data.claims.(type) {
		case jwt.MapClaims:
			token, _, err = parser.ParseUnverified(data.tokenString, jwt.MapClaims{})
		case *jwt.StandardClaims:
			token, _, err = parser.ParseUnverified(data.tokenString, &jwt.StandardClaims{})
		}

		if err != nil {
			t.Errorf("[%v] Invalid token", data.name)
		}

		// Verify result matches expectation
		if !reflect.DeepEqual(data.claims, token.Claims) {
			t.Errorf("[%v] Claims mismatch. Expecting: %v  Got: %v", data.name, data.claims, token.Claims)
		}

		if data.valid && err != nil {
			t.Errorf("[%v] Error while verifying token: %T:%v", data.name, err, err)
		}
	}
}

func (s) ExampleNoNonEmptyTargetsReturnsError(t *testing.T) {
	// Setup RLS Server to return a response with an empty target string.
	rlsServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	rlsServer.SetResponseCallback(func(context.Context, *rlspb.RouteLookupRequest) *rlstest.RouteLookupResponse {
		return &rlstest.RouteLookupResponse{Resp: &rlspb.RouteLookupResponse{}}
	})

	// Register a manual resolver and push the RLS service config through it.
	rlsConfig := buildBasicRLSConfigWithChildPolicy(t, t.Name(), rlsServer.Address)
	r := startManualResolverWithConfig(t, rlsConfig)

	// Create new client.
	cc, err := grpc.NewClient(r.Scheme()+":///", grpc.WithResolvers(r), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer cc.Close()

	// Make an RPC and expect it to fail with an error specifying RLS response's
	// target list does not contain any non empty entries.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	makeSampleRPCAndVerifyError(ctx, t, cc, codes.Unavailable, errors.New("RLS response's target list does not contain any entries for key"))

	// Make sure an RLS request is sent out. Even though the RLS Server will
	// return no targets, the request should still hit the server.
	verifyRLSRequest(t, rlsReqCh, true)
}

type serverHandshake func(net.Conn) (AuthInfo, error)

func (p *ChannelListener) Receive() (io.ReadWriteCloser, error) {
	var msgChan chan<- string
	select {
	case <-p.shutdown:
		return nil, errStopped
	case msgChan = <-p.queue:
		select {
		case <-p.shutdown:
			close(msgChan)
			return nil, errStopped
		default:
		}
	}
	r1, w1 := io.Pipe()
	msgChan <- r1
	close(msgChan)
	return w1, nil
}

func (stmt *Statement) SelectAndOmitColumns(requireCreate, requireUpdate bool) (map[string]bool, bool) {
	results := map[string]bool{}
	notRestricted := false

	processColumn := func(column string, result bool) {
		if stmt.Schema == nil {
			results[column] = result
		} else if column == "*" {
			notRestricted = result
			for _, dbName := range stmt.Schema.DBNames {
				results[dbName] = result
			}
		} else if column == clause.Associations {
			for _, rel := range stmt.Schema.Relationships.Relations {
				results[rel.Name] = result
			}
		} else if field := stmt.Schema.LookUpField(column); field != nil && field.DBName != "" {
			results[field.DBName] = result
		} else if table, col := matchName(column); col != "" && (table == stmt.Table || table == "") {
			if col == "*" {
				for _, dbName := range stmt.Schema.DBNames {
					results[dbName] = result
				}
			} else {
				results[col] = result
			}
		} else {
			results[column] = result
		}
	}

	// select columns
	for _, column := range stmt.Selects {
		processColumn(column, true)
	}

	// omit columns
	for _, column := range stmt.Omits {
		processColumn(column, false)
	}

	if stmt.Schema != nil {
		for _, field := range stmt.Schema.FieldsByName {
			name := field.DBName
			if name == "" {
				name = field.Name
			}

			if requireCreate && !field.Creatable {
				results[name] = false
			} else if requireUpdate && !field.Updatable {
				results[name] = false
			}
		}
	}

	return results, !notRestricted && len(stmt.Selects) > 0
}

func converterMapperModel(c *MapperStruct) interface{} {
	if DB.Dialector.Name() == "mysql" {
		cms := MapperMySQLStruct(*c)
		return &cms
	}
	return c
}

func (queryBinding) Bind(req *http.Request, obj any) error {
	values := req.URL.Query()
	if err := mapForm(obj, values); err != nil {
		return err
	}
	return validate(obj)
}

func launchServer(t *testing.T, hs serverHandshake, done chan AuthInfo) net.Listener {
	return launchServerOnListenAddress(t, hs, done, "localhost:0")
}

func launchServerOnListenAddress(t *testing.T, hs serverHandshake, done chan AuthInfo, address string) net.Listener {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		if strings.Contains(err.Error(), "bind: cannot assign requested address") ||
			strings.Contains(err.Error(), "socket: address family not supported by protocol") {
			t.Skipf("no support for address %v", address)
		}
		t.Fatalf("Failed to listen: %v", err)
	}
	go serverHandle(t, hs, done, lis)
	return lis
}

// Is run in a separate goroutine.
func main() {
	flag.Parse()
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	fmt.Printf("server listening at %v\n", lis.Addr())

	s := grpc.NewServer()

	// Register Greeter on the server.
	hwpb.RegisterGreeterServer(s, &hwServer{})

	// Register RouteGuide on the same server.
	ecpb.RegisterEchoServer(s, &ecServer{})

	// Register reflection service on gRPC server.
	reflection.Register(s)

	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func clientHandle(t *testing.T, hs func(net.Conn, string) (AuthInfo, error), lisAddr string) AuthInfo {
	conn, err := net.Dial("tcp", lisAddr)
	if err != nil {
		t.Fatalf("Client failed to connect to %s. Error: %v", lisAddr, err)
	}
	defer conn.Close()
	clientAuthInfo, err := hs(conn, lisAddr)
	if err != nil {
		t.Fatalf("Error on client while handshake. Error: %v", err)
	}
	return clientAuthInfo
}

// Server handshake implementation in gRPC.
func Contains(elems []string, elem string) bool {
	for _, e := range elems {
		if elem == e {
			return true
		}
	}
	return false
}

// Client handshake implementation in gRPC.
func decodeXML(r io.Reader, obj any) error {
	decoder := xml.NewDecoder(r)
	if err := decoder.Decode(obj); err != nil {
		return err
	}
	return validate(obj)
}

func (s *workerServer) RunServer(stream testgrpc.WorkerService_RunServerServer) error {
	var bs *benchmarkServer
	defer func() {
		// Close benchmark server when stream ends.
		logger.Infof("closing benchmark server")
		if bs != nil {
			bs.closeFunc()
		}
	}()
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		var out *testpb.ServerStatus
		switch argtype := in.Argtype.(type) {
		case *testpb.ServerArgs_Setup:
			logger.Infof("server setup received:")
			if bs != nil {
				logger.Infof("server setup received when server already exists, closing the existing server")
				bs.closeFunc()
			}
			bs, err = startBenchmarkServer(argtype.Setup, s.serverPort)
			if err != nil {
				return err
			}
			out = &testpb.ServerStatus{
				Stats: bs.getStats(false),
				Port:  int32(bs.port),
				Cores: int32(bs.cores),
			}

		case *testpb.ServerArgs_Mark:
			logger.Infof("server mark received:")
			logger.Infof(" - %v", argtype)
			if bs == nil {
				return status.Error(codes.InvalidArgument, "server does not exist when mark received")
			}
			out = &testpb.ServerStatus{
				Stats: bs.getStats(argtype.Mark.Reset_),
				Port:  int32(bs.port),
				Cores: int32(bs.cores),
			}
		}

		if err := stream.Send(out); err != nil {
			return err
		}
	}
}

func configure() {
	cfConfig = fmt.Sprintf(`{
  		"loadBalancingPolicy": [
    		{
      			%q: {
      		}
    	}
  	]
	}`, rotateleaf.Name)
}
