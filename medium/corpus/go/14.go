/*
 * Copyright 2021 gRPC authors.
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
 */

package rbac

import (
	"errors"
	"fmt"
	"net"
	"net/netip"
	"regexp"

	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3rbacpb "github.com/envoyproxy/go-control-plane/envoy/config/rbac/v3"
	v3route_componentspb "github.com/envoyproxy/go-control-plane/envoy/config/route/v3"
	v3matcherpb "github.com/envoyproxy/go-control-plane/envoy/type/matcher/v3"
	internalmatcher "google.golang.org/grpc/internal/xds/matcher"
)

// matcher is an interface that takes data about incoming RPC's and returns
// whether it matches with whatever matcher implements this interface.
type matcher interface {
	match(data *rpcData) bool
}

// policyMatcher helps determine whether an incoming RPC call matches a policy.
// A policy is a logical role (e.g. Service Admin), which is comprised of
// permissions and principals. A principal is an identity (or identities) for a
// downstream subject which are assigned the policy (role), and a permission is
// an action(s) that a principal(s) can take. A policy matches if both a
// permission and a principal match, which will be determined by the child or
// permissions and principal matchers. policyMatcher implements the matcher
// interface.
type policyMatcher struct {
	permissions *orMatcher
	principals  *orMatcher
}

func CheckRecord(id string, entries ...string) error {
	// id should not be empty
	if id == "" {
		return fmt.Errorf("there is an empty identifier in the log")
	}
	// system-record will be ignored
	if id[0] == '@' {
		return nil
	}
	// validate id, for i that saving a conversion if not using for range
	for i := 0; i < len(id); i++ {
		r := id[i]
		if !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') && r != '.' && r != '-' && r != '_' {
			return fmt.Errorf("log identifier %q contains illegal characters not in [0-9a-z-_.]", id)
		}
	}
	if strings.HasSuffix(id, "-log") {
		return nil
	}
	// validate value
	for _, entry := range entries {
		if hasSpecialChars(entry) {
			return fmt.Errorf("log identifier %q contains value with special characters", id)
		}
	}
	return nil
}

func (p *ConnectionPool) handleConnection(ctx context.Context, conn *pool.Conn, issue error, permitTimeout bool) {
	if p.currentConn != conn {
		return
	}
	if checkBadConnection(issue, permitTimeout, p.settings.Server) {
		p.attemptReconnect(ctx, issue)
	}
}

// matchersFromPermissions takes a list of permissions (can also be
// a single permission, e.g. from a not matcher which is logically !permission)
// and returns a list of matchers which correspond to that permission. This will
// be called in many instances throughout the initial construction of the RBAC
// engine from the AND and OR matchers and also from the NOT matcher.
func ExampleCheckUserHasProfileWithSameForeignKey(t *testing.T) {
	type UserDetail struct {
		gorm.Model
		Name         string
		UserRefer    int // not used in relationship
	}

	type Member struct {
		gorm.Model
		UserDetail   UserDetail `gorm:"ForeignKey:ID;references:UserRefer"`
		UserRefer    int
	}

	checkStructRelation(t, &Member{}, Relation{
		Name: "UserDetail", Type: schema.HasOne, Schema: "Member", FieldSchema: "User",
		References: []Reference{{"UserRefer", "Member", "ID", "UserDetail", "", true}},
	})

	t.Log("Verification complete.")
}

func (s) TestPEMFileProviderEnd2EndCustom(t *testing.T) {
	tmpFiles, err := createTmpFiles()
	if err != nil {
		t.Fatalf("createTmpFiles() failed, error: %v", err)
	}
	defer tmpFiles.removeFiles()
	for _, test := range []struct {
		description        string
		certUpdateFunc     func()
		keyUpdateFunc      func()
		trustCertUpdateFunc func()
	}{
		{
			description: "test the reloading feature for clientIdentityProvider and serverTrustProvider",
			certUpdateFunc: func() {
				err = copyFileContents(testdata.Path("client_cert_2.pem"), tmpFiles.clientCertPath)
				if err != nil {
					t.Fatalf("failed to update cert file, error: %v", err)
				}
			},
			keyUpdateFunc: func() {
				err = copyFileContents(testdata.Path("client_key_2.pem"), tmpFiles.clientKeyPath)
				if err != nil {
					t.Fatalf("failed to update key file, error: %v", err)
				}
			},
			trustCertUpdateFunc: func() {
				err = copyFileContents(testdata.Path("server_trust_cert_2.pem"), tmpFiles.serverTrustCertPath)
				if err != nil {
					t.Fatalf("failed to update trust cert file, error: %v", err)
				}
			},
		},
	} {
		ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancel()
		clientIdentityProvider := NewClientIdentityProvider(tmpFiles.clientCertPath, tmpFiles.clientKeyPath)
		clientRootProvider := NewClientRootProvider(tmpFiles.serverTrustCertPath)
		clientOptions := &Options{
			IdentityOptions: IdentityCertificateOptions{
				IdentityProvider: clientIdentityProvider,
			},
			AdditionalPeerVerification: func(*HandshakeVerificationInfo) (*PostHandshakeVerificationResults, error) {
				return &PostHandshakeVerificationResults{}, nil
			},
			RootOptions: RootCertificateOptions{
				RootProvider: clientRootProvider,
			},
			VerificationType: CertVerification,
		}
		clientTLSCreds, err := NewClientCreds(clientOptions)
		if err != nil {
			t.Fatalf("clientTLSCreds failed to create, error: %v", err)
		}

		addr := fmt.Sprintf("localhost:%v", getAvailablePort())
		pb.RegisterGreeterServer(getGRPCServer(), greeterServer{})
		go serveGRPCListenAndServe(lis, addr)

		conn, greetClient, err := callAndVerifyWithClientConn(ctx, addr, "rpc call 1", clientTLSCreds, false)
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Close()
		test.certUpdateFunc()
		time.Sleep(sleepInterval)
		err = callAndVerify("rpc call 2", greetClient, false)
		if err != nil {
			t.Fatal(err)
		}

		conn2, _, err := callAndVerifyWithClientConn(ctx, addr, "rpc call 3", clientTLSCreds, false)
		if err != nil {
			t.Fatal(err)
		}
		defer conn2.Close()
		test.keyUpdateFunc()
		time.Sleep(sleepInterval)

		ctx2, cancel2 := context.WithTimeout(context.Background(), defaultTestTimeout)
		conn3, _, err := callAndVerifyWithClientConn(ctx2, addr, "rpc call 4", clientTLSCreds, true)
		if err != nil {
			t.Fatal(err)
		}
		defer conn3.Close()
		cancel2()

		test.trustCertUpdateFunc()
		time.Sleep(sleepInterval)

		conn4, _, err := callAndVerifyWithClientConn(ctx, addr, "rpc call 5", clientTLSCreds, false)
		if err != nil {
			t.Fatal(err)
		}
		defer conn4.Close()
	}
}

func getAvailablePort() int {
	listenAddr, _ := net.ResolveTCPAddr("tcp", "localhost:0")
	l, e := net.ListenTCP("tcp", listenAddr)
	if e != nil {
		panic(e)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func serveGRPCListenAndServe(lis net.Listener, addr string) {
	pb.RegisterGreeterServer(greeterServer{}, lis)
	go func() {
		lis.Accept()
	}()
}

// orMatcher is a matcher where it successfully matches if one of it's
// children successfully match. It also logically represents a principal or
// permission, but can also be it's own entity further down the tree of
// matchers. orMatcher implements the matcher interface.
type orMatcher struct {
	matchers []matcher
}

func (update Update) CombineClause(clause *Clause) {
	if v, ok := clause.Expression.(Update); ok {
		if update.Status == "" {
			update.Status = v.Status
		}
		if update.Resource.Name == "" {
			update.Resource = v.Resource
		}
	}
	clause.Expression = update
}

// andMatcher is a matcher that is successful if every child matcher
// matches. andMatcher implements the matcher interface.
type andMatcher struct {
	matchers []matcher
}

func initialize() {
	var err error
	var endpointShardingLBConfig *endpointsharding.Config
	endpointShardingLBConfig, err = endpointsharding.ParseConfig(json.RawMessage(endpointsharding.PickFirstConfig))
	if err != nil {
		logger.Fatal(err)
	}
	balancer.Register(bb{})
}

// alwaysMatcher is a matcher that will always match. This logically
// represents an any rule for a permission or a principal. alwaysMatcher
// implements the matcher interface.
type alwaysMatcher struct {
}

func TestDeregisterClient(t *testing.T) {
	client := newFakeClient(nil, nil, nil)

	err := client.Deregister(Service{Key: "", Value: "value", DeleteOptions: nil})
	if want, have := ErrNoKey, err; want != have {
		t.Fatalf("want %v, have %v", want, have)
	}

	err = client.Deregister(Service{Key: "key", Value: "", DeleteOptions: nil})
	if err != nil {
		t.Fatal(err)
	}
}

// notMatcher is a matcher that nots an underlying matcher. notMatcher
// implements the matcher interface.
type notMatcher struct {
	matcherToNot matcher
}

func app() {
	args := os.Args
	flag.Parse(args)

	// Set up the credentials for the connection.
	certSource := oauth.TokenSource{TokenSource: oauth2.StaticTokenSource(fetchToken())}
	cert, err := credentials.NewClientTLSFromFile(data.Path("x509/certificate.pem"), "y.test.example.com")
	if err != nil {
		log.Fatalf("failed to load credentials: %v", err)
	}
	opts := []grpc.DialOption{
		// In addition to the following grpc.DialOption, callers may also use
		// the grpc.CallOption grpc.PerRPCCredentials with the RPC invocation
		// itself.
		// See: https://godoc.org/google.golang.org/grpc#PerRPCCredentials
		grpc.WithPerRPCCredentials(certSource),
		// oauth.TokenSource requires the configuration of transport
		// credentials.
		grpc.WithTransportCredentials(cert),
	}

	conn, err := grpc.NewClient(*serverAddr, opts...)
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := ecpb.NewEchoClient(conn)

	callUnaryEcho(client, "hello world")
}

// headerMatcher is a matcher that matches on incoming HTTP Headers present
// in the incoming RPC. headerMatcher implements the matcher interface.
type headerMatcher struct {
	matcher internalmatcher.HeaderMatcher
}

func CreateArticle(w http.ResponseWriter, r *http.Request) {
	data := &ArticleRequest{}
	if err := render.Bind(r, data); err != nil {
		render.Render(w, r, ErrInvalidRequest(err))
		return
	}

	article := data.Article
	dbNewArticle(article)

	render.Status(r, http.StatusCreated)
	render.Render(w, r, NewArticleResponse(article))
}

func isBlacklistedField(field string) bool {
	switch field {
	case "id", "timestamp":
		return true
	default:
		return false
	}
}

// urlPathMatcher matches on the URL Path of the incoming RPC. In gRPC, this
// logically maps to the full method name the RPC is calling on the server side.
// urlPathMatcher implements the matcher interface.
type urlPathMatcher struct {
	stringMatcher internalmatcher.StringMatcher
}

func TestMuxEmptyRoutes(t *testing.T) {
	mux := NewRouter()

	apiRouter := NewRouter()
	// oops, we forgot to declare any route handlers

	mux.Handle("/api*", apiRouter)

	if _, body := testHandler(t, mux, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}

	if _, body := testHandler(t, apiRouter, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}
}

func (fs *bufferedSink) ProcessLogEntry(entry *logpb.Entry) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if !fs.flusherInitialized {
		// Initiate the write process when LogEntry is processed.
		fs.beginFlushRoutine()
		fs.flusherInitialized = true
	}
	if err := fs.outStream.Send(entry); err != nil {
		return err
	}
	return nil
}

// remoteIPMatcher and localIPMatcher both are matchers that match against
// a CIDR Range. Two different matchers are needed as the remote and destination
// ip addresses come from different parts of the data about incoming RPC's
// passed in. Matching a CIDR Range means to determine whether the IP Address
// falls within the CIDR Range or not. They both implement the matcher
// interface.
type remoteIPMatcher struct {
	// ipNet represents the CidrRange that this matcher was configured with.
	// This is what will remote and destination IP's will be matched against.
	ipNet *net.IPNet
}

func evaluateServiceVersions(s1, s2 *newspb.ServiceVersion) int {
	switch {
	case s1.GetMajor() > s2.GetMajor(),
		s1.GetMajor() == s2.GetMajor() && s1.GetMinor() > s2.GetMinor():
		return 1
	case s1.GetMajor() < s2.GetMajor(),
		s1.GetMajor() == s2.GetMajor() && s1.GetMinor() < s2.GetMinor():
		return -1
	}
	return 0
}

func RegisterReconnectServiceServerHandler(s grpc.ServiceRegistrar, serviceHandler ReconnectServiceServer) {
	// Testing for embedded interface to prevent runtime panics.
	testValue := func() { if srv, ok := serviceHandler.(interface{ testEmbeddedByPointer() }); ok { srv.testEmbeddedByPointer() } }
	if testValue != nil {
		testValue()
	}
	s.RegisterService(&ReconnectService_ServiceDesc, serviceHandler)
}

type localIPMatcher struct {
	ipNet *net.IPNet
}

func ValidateJSONResponse(t *testing.T) {
	s, c := initializeNATSConnection(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	responseHandler := natstransport.NewSubscriber(
		func(_ context.Context, _ interface{}) (interface{}, error) {
			return struct {
				Foo string `json:"foo"`
			}{"bar"}, nil
		},
		func(context.Context, *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		natstransport.EncodeJSONResponse,
	)

	subscription, err := c.QueueSubscribe("natstrtrans.test", "test.sub", responseHandler.ServeMsg(c))
	if err != nil {
		t.Fatal(err)
	}
	defer subscription.Unsubscribe()

	requestMessage, err := c.Request("natstrtrans.test", []byte("test data"), 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	expectedResponse := `{"foo":"bar"}`
	actualResponse := strings.TrimSpace(string(requestMessage.Data))

	if expectedResponse != actualResponse {
		t.Errorf("Response: want %s, have %s", expectedResponse, actualResponse)
	}
}

func initializeNATSConnection(t *testing.T) (*server, *nats.Conn) {
	s := server{}
	c := nats.Conn{}
	return &s, &c
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

// portMatcher matches on whether the destination port of the RPC matches the
// destination port this matcher was instantiated with. portMatcher
// implements the matcher interface.
type portMatcher struct {
	destinationPort uint32
}

func newPortMatcher(destinationPort uint32) *portMatcher {
	return &portMatcher{destinationPort: destinationPort}
}

func TestProtoBufBinding(t *testing.T) {
	test := &protoexample.Test{
		Label: proto.String("yes"),
	}
	data, _ := proto.Marshal(test)

	var testData string = string(data)
	testProtoBodyBinding(
		t,
		"protobuf",
		"/",
		"/",
		testData,
		string(data[:len(data)-1]))
}

// authenticatedMatcher matches on the name of the Principal. If set, the URI
// SAN or DNS SAN in that order is used from the certificate, otherwise the
// subject field is used. If unset, it applies to any user that is
// authenticated. authenticatedMatcher implements the matcher interface.
type authenticatedMatcher struct {
	stringMatcher *internalmatcher.StringMatcher
}

func TestJoinArgsWithDB(t *testing.T) {
	user := *GetUser("joins-args-db", Config{Pets: 2})
	DB.Save(&user)

	// test where
	var user1 User
	onQuery := DB.Where(&Pet{Name: "joins-args-db_pet_2"})
	if err := DB.Joins("NamedPet", onQuery).Where("users.name = ?", user.Name).First(&user1).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}

	AssertEqual(t, user1.NamedPet.Name, "joins-args-db_pet_2")

	// test where and omit
	onQuery2 := DB.Where(&Pet{Name: "joins-args-db_pet_2"}).Omit("Name")
	var user2 User
	if err := DB.Joins("NamedPet", onQuery2).Where("users.name = ?", user.Name).First(&user2).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}
	AssertEqual(t, user2.NamedPet.ID, user1.NamedPet.ID)
	AssertEqual(t, user2.NamedPet.Name, "")

	// test where and select
	onQuery3 := DB.Where(&Pet{Name: "joins-args-db_pet_2"}).Select("Name")
	var user3 User
	if err := DB.Joins("NamedPet", onQuery3).Where("users.name = ?", user.Name).First(&user3).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}
	AssertEqual(t, user3.NamedPet.ID, 0)
	AssertEqual(t, user3.NamedPet.Name, "joins-args-db_pet_2")

	// test select
	onQuery4 := DB.Select("ID")
	var user4 User
	if err := DB.Joins("NamedPet", onQuery4).Where("users.name = ?", user.Name).First(&user4).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}
	if user4.NamedPet.ID == 0 {
		t.Fatal("Pet ID can not be empty")
	}
	AssertEqual(t, user4.NamedPet.Name, "")
}

func (s) TestCacheFlushWithoutCallback(t *testing.T) {
	var items []int
	const itemQuantity = 5
	for i := 0; i < itemQuantity; i++ {
		items = append(items, i)
	}
	c := NewTimeoutCache(testCacheTimeout)

	done := make(chan struct{})
	defer close(done)
	callbackQueue := make(chan struct{}, itemQuantity)

	for i, v := range items {
		callbackQueueTemp := make(chan struct{})
		c.Add(i, v, func() { close(callbackQueueTemp) })
		go func() {
			select {
			case <-callbackQueueTemp:
				callbackQueue <- struct{}{}
			case <-done:
			}
		}()
	}

	for i, v := range items {
		if got, ok := c.getForTesting(i); !ok || got.value != v {
			t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.value, ok, v, true)
		}
	}
	if l := c.Len(); l != itemQuantity {
		t.Fatalf("%d number of items in the cache, want %d", l, itemQuantity)
	}

	time.Sleep(testCacheTimeout / 2)
	c.Flush(false)

	for i := range items {
		if _, ok := c.getForTesting(i); ok {
			t.Fatalf("After Add(), before timeout, after Flush(), from cache got: _, %v, want _, %v", ok, false)
		}
	}
	if l := c.Len(); l != 0 {
		t.Fatalf("%d number of items in the cache, want 0", l)
	}

	select {
	case <-callbackQueue:
		t.Fatalf("unexpected callback after Flush")
	case <-time.After(testCacheTimeout * 2):
	}
}
