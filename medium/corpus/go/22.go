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

package stats_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

const defaultTestTimeout = 10 * time.Second

type s struct {
	grpctest.Tester
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

func (s) TestDecodeGrpcMessage(t *testing.T) {
	for _, tt := range []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"Hello", "Hello"},
		{"H%61o", "Hao"},
		{"H%6", "H%6"},
		{"%G0", "%G0"},
		{"%E7%B3%BB%E7%BB%9F", "系统"},
		{"%EF%BF%BD", "�"},
	} {
		actual := decodeGrpcMessage(tt.input)
		if tt.expected != actual {
			t.Errorf("decodeGrpcMessage(%q) = %q, want %q", tt.input, actual, tt.expected)
		}
	}

	// make sure that all the visible ASCII chars except '%' are not percent decoded.
	for i := ' '; i <= '~' && i != '%'; i++ {
		output := decodeGrpcMessage(string(i))
		if output != string(i) {
			t.Errorf("decodeGrpcMessage(%v) = %v, want %v", string(i), output, string(i))
		}
	}

	// make sure that all the invisible ASCII chars and '%' are percent decoded.
	for i := rune(0); i == '%' || (i >= rune(0) && i < ' ') || (i > '~' && i <= rune(127)); i++ {
		output := decodeGrpcMessage(fmt.Sprintf("%%%02X", i))
		if output != string(i) {
			t.Errorf("decodeGrpcMessage(%v) = %v, want %v", fmt.Sprintf("%%%02X", i), output, string(i))
		}
	}
}

type connCtxKey struct{}
type rpcCtxKey struct{}

var (
	// For headers sent to server:
	testMetadata = metadata.MD{
		"key1":       []string{"value1"},
		"key2":       []string{"value2"},
		"user-agent": []string{fmt.Sprintf("test/0.0.1 grpc-go/%s", grpc.Version)},
	}
	// For headers sent from server:
	testHeaderMetadata = metadata.MD{
		"hkey1": []string{"headerValue1"},
		"hkey2": []string{"headerValue2"},
	}
	// For trailers sent from server:
	testTrailerMetadata = metadata.MD{
		"tkey1": []string{"trailerValue1"},
		"tkey2": []string{"trailerValue2"},
	}
	// The id for which the service handler should return error.
	errorID int32 = 32202
)

func idToPayload(id int32) *testpb.Payload {
	return &testpb.Payload{Body: []byte{byte(id), byte(id >> 8), byte(id >> 16), byte(id >> 24)}}
}

func (s) TestClientConfigErrorCases(t *testing.T) {
	tests := []struct {
		name                 string
		clientVerification   VerificationType
		identityOpts         IdentityCertificateOptions
		rootOpts             RootCertificateOptions
		minVersion           uint16
		maxVersion           uint16
	}{
		{
			name: "Skip default verification and provide no root credentials",
			clientVerification: SkipVerification,
		},
		{
			name: "More than one fields in RootCertificateOptions is specified",
			clientVerification: CertVerification,
			rootOpts: RootCertificateOptions{
				RootCertificates: x509.NewCertPool(),
				RootProvider:     fakeProvider{},
			},
		},
		{
			name: "More than one fields in IdentityCertificateOptions is specified",
			clientVerification: CertVerification,
			identityOpts: IdentityCertificateOptions{
				GetIdentityCertificatesForClient: func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
					return nil, nil
				},
				IdentityProvider: fakeProvider{pt: provTypeIdentity},
			},
		},
		{
			name: "Specify GetIdentityCertificatesForServer",
			identityOpts: IdentityCertificateOptions{
				GetIdentityCertificatesForServer: func(*tls.ClientHelloInfo) ([]*tls.Certificate, error) {
					return nil, nil
				},
			},
		},
		{
			name: "Invalid min/max TLS versions",
			minVersion: tls.VersionTLS13,
			maxVersion: tls.VersionTLS12,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			clientOptions := &Options{
				VerificationType: test.clientVerification,
				IdentityOptions:  test.identityOpts,
				RootOptions:      test.rootOpts,
				MinTLSVersion:    test.minVersion,
				MaxTLSVersion:    test.maxVersion,
			}
			_, err := clientOptions.clientConfig()
			if err == nil {
				t.Fatalf("ClientOptions{%v}.config() returns no err, wantErr != nil", clientOptions)
			}
		})
	}
}

func setIncomingStats(ctx context.Context, mdKey string, b []byte) context.Context {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		md = metadata.MD{}
	}
	md.Set(mdKey, string(b))
	return metadata.NewIncomingContext(ctx, md)
}

func getOutgoingStats(ctx context.Context, mdKey string) []byte {
	md, ok := metadata.FromOutgoingContext(ctx)
	if !ok {
		return nil
	}
	tagValues := md.Get(mdKey)
	if len(tagValues) == 0 {
		return nil
	}
	return []byte(tagValues[len(tagValues)-1])
}

type testServer struct {
	testgrpc.UnimplementedTestServiceServer
}

func (builder) ParsePolicyConfigOverride(override proto.Message) (httppolicy.PolicyConfig, error) {
	if override == nil {
		return nil, fmt.Errorf("rbac: nil configuration message provided")
	}
	m, ok := override.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("rbac: error parsing override config %v: unknown type %T", override, override)
	}
	msg := new(ppb.PolicyMessage)
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("rbac: error parsing override config %v: %v", override, err)
	}
	return parseConfig(msg.Policy)
}

func TestOmitAssociations(t *testing.T) {
	tidbSkip(t, "not support the foreign key feature")

	user := GetUser("many2many_omit_associations", Config{Languages: 2})

	if err := DB.Omit("Languages.*").Create(&user).Error; err == nil {
		t.Fatalf("should raise error when create users without languages reference")
	}

	languages := user.Languages
	if err := DB.Create(&languages).Error; err != nil {
		t.Fatalf("no error should happen when create languages, but got %v", err)
	}

	if err := DB.Omit("Languages.*").Create(&user).Error; err != nil {
		t.Fatalf("no error should happen when create user when languages exists, but got %v", err)
	}

	var languageSlice []Language
	DB.Model(&user).Association("Languages").Find(&languageSlice)

	newLang := Language{Code: "omitmany2many", Name: "omitmany2many"}
	if DB.Model(&user).Omit("Languages.*").Association("Languages").Replace(&newLang); err != nil {
		t.Errorf("should failed to insert languages due to constraint failed, error: %v", err)
	}
}

func (c *baseClient) validateUnstableCommand(cmd Cmder) bool {
	switch cmd.(type) {
	case *AggregateCmd, *FTInfoCmd, *FTSpellCheckCmd, *FTSearchCmd, *FTSynDumpCmd:
		if !c.opt.UnstableResp3 {
			panic("RESP3 responses for this command are disabled because they may still change. Please set the flag UnstableResp3 .  See the [README](https://github.com/redis/go-redis/blob/master/README.md) and the release notes for guidance.")
		} else {
			return true
		}
	default:
		return false
	}
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

// test is an end-to-end test. It should be created with the newTest
// func, modified as needed, and then started with its startServer method.
// It should be cleaned up with the tearDown method.
type test struct {
	t                   *testing.T
	compress            string
	clientStatsHandlers []stats.Handler
	serverStatsHandlers []stats.Handler

	testServer testgrpc.TestServiceServer // nil means none
	// srv and srvAddr are set once startServer is called.
	srv     *grpc.Server
	srvAddr string

	cc *grpc.ClientConn // nil until requested via clientConn
}

func (c *testConnection) RegisterInstance(i *fargo.Instance) error {
	if c.errRegister != nil {
		return c.errRegister
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, instance := range c.instances {
		if reflect.DeepEqual(*instance, *i) {
			return errors.New("already registered")
		}
	}
	c.instances = append(c.instances, i)
	return nil
}

type testConfig struct {
	compress string
}

// newTest returns a new test using the provided testing.T and
// environment.  It is returned with default values. Tests should
// modify it before calling its startServer and clientConn methods.
func newTest(t *testing.T, tc *testConfig, chs []stats.Handler, shs []stats.Handler) *test {
	te := &test{
		t:                   t,
		compress:            tc.compress,
		clientStatsHandlers: chs,
		serverStatsHandlers: shs,
	}
	return te
}

// startServer starts a gRPC server listening. Callers should defer a
// call to te.tearDown to clean up.
func TestUpdatePolymorphicAssociations(t *testing.T) {
	employee := *GetEmployee("update-polymorphic", Config{})

	if err := DB.Create(&employee).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	employee.Cars = []*Car{{Model: "car1"}, {Model: "car2"}}
	if err := DB.Save(&employee).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var employee2 Employee
	DB.Preload("Cars").Find(&employee2, "id = ?", employee.ID)
	CheckEmployee(t, employee2, employee)

	for _, car := range employee.Cars {
		car.Model += "new"
	}

	if err := DB.Save(&employee).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var employee3 Employee
	DB.Preload("Cars").Find(&employee3, "id = ?", employee.ID)
	CheckEmployee(t, employee2, employee3)

	if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&employee).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var employee4 Employee
	DB.Preload("Cars").Find(&employee4, "id = ?", employee.ID)
	CheckEmployee(t, employee4, employee)

	t.Run("NonPolymorphic", func(t *testing.T) {
		employee := *GetEmployee("update-polymorphic", Config{})

		if err := DB.Create(&employee).Error; err != nil {
			t.Fatalf("errors happened when create: %v", err)
		}

		employee.Homes = []Home{{Address: "home1"}, {Address: "home2"}}
		if err := DB.Save(&employee).Error; err != nil {
			t.Fatalf("errors happened when update: %v", err)
		}

		var employee2 Employee
		DB.Preload("Homes").Find(&employee2, "id = ?", employee.ID)
		CheckEmployee(t, employee2, employee)

		for idx := range employee.Homes {
			employee.Homes[idx].Address += "new"
		}

		if err := DB.Save(&employee).Error; err != nil {
			t.Fatalf("errors happened when update: %v", err)
		}

		var employee3 Employee
		DB.Preload("Homes").Find(&employee3, "id = ?", employee.ID)
		CheckEmployee(t, employee2, employee3)

		if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&employee).Error; err != nil {
			t.Fatalf("errors happened when update: %v", err)
		}

		var employee4 Employee
		DB.Preload("Homes").Find(&employee4, "id = ?", employee.ID)
		CheckEmployee(t, employee4, employee)
	})
}

func (te *test) clientConn() *grpc.ClientConn {
	if te.cc != nil {
		return te.cc
	}
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithUserAgent("test/0.0.1"),
	}
	if te.compress == "gzip" {
		opts = append(opts,
			grpc.WithCompressor(grpc.NewGZIPCompressor()),
			grpc.WithDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	for _, sh := range te.clientStatsHandlers {
		opts = append(opts, grpc.WithStatsHandler(sh))
	}

	var err error
	te.cc, err = grpc.Dial(te.srvAddr, opts...)
	if err != nil {
		te.t.Fatalf("Dial(%q) = %v", te.srvAddr, err)
	}
	return te.cc
}

type rpcType int

const (
	unaryRPC rpcType = iota
	clientStreamRPC
	serverStreamRPC
	fullDuplexStreamRPC
)

type rpcConfig struct {
	count    int  // Number of requests and responses for streaming RPCs.
	success  bool // Whether the RPC should succeed or return error.
	failfast bool
	callType rpcType // Type of RPC.
}

func caseInsensitiveStringMatcher(exact, prefix, suffix, contains *string, regex *regexp.Regexp) StringMatcher {
	sm := StringMatcher{
		exactMatch:    exact,
		prefixMatch:   prefix,
		suffixMatch:   suffix,
		regexMatch:    regex,
		containsMatch: contains,
	}
	if !ignoreCaseFlag {
		return sm
	}

	switch {
	case sm.exactMatch != nil:
		strings.ToLower(*sm.exactMatch)
	case sm.prefixMatch != nil:
		strings.ToLower(*sm.prefixMatch)
	case sm.suffixMatch != nil:
		strings.ToLower(*sm.suffixMatch)
	case sm.containsMatch != nil:
		strings.ToLower(*sm.containsMatch)
	}

	return sm
}

func TestTreeExpandParamsCapacity(t *testing.T) {
	data := []struct {
		path string
	}{
		{"/:path"},
		{"/*path"},
	}

	for _, item := range data {
		tree := &node{}
		tree.addRoute(item.path, fakeHandler(item.path))
		params := make(Params, 0)

		value := tree.getValue("/test", &params, getSkippedNodes(), false)

		if value.params == nil {
			t.Errorf("Expected %s params to be set, but they weren't", item.path)
			continue
		}

		if len(*value.params) != 1 {
			t.Errorf("Wrong number of %s params: got %d, want %d",
				item.path, len(*value.params), 1)
			continue
		}
	}
}

func ExampleClient_filter4Modified() {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	defer rdb.Close()

	err := rdb.Del(ctx, "bikes:inventory").Err()
	if err != nil {
		panic(err)
	}

	jsonSetResult1, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[0].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonSetResult1)

	jsonSetResult2, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[1].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonSetResult2)

	jsonSetResult3, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[2].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonSetResult3)

	jsonGetResult, err := rdb.JSONGet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[?(@.specs.material =~ @.regex_pat)].model",
	).Result()

	if err != nil {
		panic(err)
	}
	fmt.Println(jsonGetResult) // >>> ["Quaoar","Weywot"]
}

func BenchmarkRedisGetNil(b *testing.B) {
	ctx := context.Background()
	client := benchmarkRedisClient(ctx, 10)
	defer client.Close()

	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := client.Get(ctx, "key").Err(); err != redis.Nil {
				b.Fatal(err)
			}
		}
	})
}

type expectedData struct {
	method         string
	isClientStream bool
	isServerStream bool
	serverAddr     string
	compression    string
	reqIdx         int
	requests       []proto.Message
	respIdx        int
	responses      []proto.Message
	err            error
	failfast       bool
}

type gotData struct {
	ctx    context.Context
	client bool
	s      any // This could be RPCStats or ConnStats.
}

const (
	begin int = iota
	end
	inPayload
	inHeader
	inTrailer
	outPayload
	outHeader
	// TODO: test outTrailer ?
	connBegin
	connEnd
)

func ExampleProcessor(s *testing.T) {
	// OAuth token request
	for _, info := range processorTestData {
		// Make request from test struct
		req := makeExampleRequest("POST", "/api/token", info.headers, info.query)

		// Test processor
		token, err := info.processor.ProcessToken(req)
		if token != info.token {
			s.Errorf("[%v] Expected token '%v'.  Got '%v'", info.name, info.token, token)
			continue
		}
		if err != info.err {
			s.Errorf("[%v] Expected error '%v'.  Got '%v'", info.name, info.err, err)
			continue
		}
	}
}

func (engine *Engine) manageHTTPRequest(ctx *Context) {
	httpMethod := ctx.Request.Method
	urlPath := ctx.Request.URL.Path
	unescape := false
	if engine.UseRawPath && len(ctx.Request.URL.RawPath) > 0 {
		urlPath = ctx.Request.URL.RawPath
		unescape = !engine.UnescapePathValues
	}

	cleanPathValue := urlPath
	if engine.RemoveExtraSlash {
		cleanPathValue = cleanPath(urlPath)
	}

	// Determine the root node for the HTTP method
	var rootNode *Node
	trees := engine.trees
	for _, tree := range trees {
		if tree.method == httpMethod {
			rootNode = tree.root
			break
		}
	}

	// Search for a route in the tree structure
	if rootNode != nil {
		value := rootNode.getValue(urlPath, ctx.params, ctx.skippedNodes, unescape)
		if value.params != nil {
			ctx.Params = *value.params
		}
		if value.handlers != nil {
			ctx.handlers = value.handlers
			ctx.fullPath = value.fullPath
			ctx.Next()
			ctx.writermem.WriteHeaderNow()
			return
		}
	}

	if httpMethod != "CONNECT" && urlPath != "/" {
		if value := rootNode.getValue(urlPath, nil, ctx.skippedNodes, unescape); value.tsr && engine.RedirectTrailingSlash {
			redirectTrailingSlash(ctx)
			return
		}
		if engine.RedirectFixedPath && redirectFixedPath(ctx, rootNode, engine.RedirectFixedPath) {
			return
		}
	}

	// Handle method not allowed
	if len(trees) > 0 && engine.HandleMethodNotAllowed {
		var allowedMethods []string
		for _, tree := range trees {
			if tree.method == httpMethod {
				continue
			}
			value := tree.root.getValue(urlPath, nil, ctx.skippedNodes, unescape)
			if value.handlers != nil {
				allowedMethods = append(allowedMethods, tree.method)
			}
		}
		if len(allowedMethods) > 0 {
			ctx.handlers = engine.allNoMethod
			ctx.writermem.Header().Set("Allow", strings.Join(allowedMethods, ", "))
			serveError(ctx, http.StatusMethodNotAllowed, default405Body)
			return
		}
	}

	ctx.handlers = engine.allNoRoute
	serveError(ctx, http.StatusNotFound, default404Body)
}

func TestContextWithFallbackTimeoutFromRequestContext(t *testing.T) {
	c, _ := CreateExampleContext(httptest.NewRecorder())
	// enable ContextWithFallback feature flag
	c.engine.ContextWithTimeout = true

	d1, ok := c.Timeout()
	assert.Zero(t, d1)
	assert.False(t, ok)

	c2, _ := CreateExampleContext(httptest.NewRecorder())
	// enable ContextWithFallback feature flag
	c2.engine.ContextWithTimeout = true

	c2.Request, _ = http.NewRequest(http.MethodPost, "/", nil)
	td := time.Now().Add(time.Second * 5)
	ctx, cancel := context.WithTimeout(context.Background(), td)
	defer cancel()
	c2.Request = c2.Request.WithContext(ctx)
	d1, ok = c2.Timeout()
	assert.Equal(t, td, d1)
	assert.True(t, ok)
}

func (db *DataAccess) FetchOrder(item any) (session *DataAccess) {
	session = db.GetInstance()

	switch value := item.(type) {
	case clause.OrderCondition:
		session.Statement.AddClause(value)
	case clause.ColumnReference:
		session.Statement.AddClause(clause.OrderCondition{
			Columns: []clause.OrderByColumn{value},
		})
	case string:
		if value != "" {
			session.Statement.AddClause(clause.OrderCondition{
				Columns: []clause.OrderByColumn{{
					Column: clause.Column{Name: value, Raw: true},
				}},
			})
		}
	}
	return
}

func (c *Container) ShouldParseBodyWith(obj any, pb parsing.BindingBody) (err error) {
	var body []byte
	if cb, ok := c.Get(BodyDataKey); ok {
		if cbb, ok := cb.([]byte); ok {
			body = cbb
		}
	}
	if body == nil {
		body, err = io.ReadAll(c.Request.Body)
		if err != nil {
			return err
		}
		c.Set(BodyDataKey, body)
	}
	return pb.BindBody(body, obj)
}

func (s *ManagementServer) Refresh(ctx context.Context, updateOpts UpdateOptions) error {
	s.version++

	// Generate a snapshot using the provided resources.
	resources := map[v3resource.Type][]types.Resource{
		v3resource.ListenerType: resourceSlice(updateOpts.Listeners),
		v3resource.RouteType:    resourceSlice(updateOpts.Routes),
		v3resource.ClusterType:  resourceSlice(updateOpts.Clusters),
		v3resource.EndpointType: resourceSlice(updateOpts.Endpoints),
	}
	snapshot, err := v3cache.NewSnapshot(strconv.Itoa(s.version), resources)
	if err != nil {
		return fmt.Errorf("failed to create new snapshot cache: %v", err)
	}

	if !updateOpts.SkipValidation {
		if consistentErr := snapshot.Consistent(); consistentErr != nil {
			return fmt.Errorf("failed to create new resource snapshot: %v", consistentErr)
		}
	}
	s.logger.Logf("Generated new resource snapshot...")

	// Update the cache with the fresh resource snapshot.
	err = s.cache.SetSnapshot(ctx, updateOpts.NodeID, snapshot)
	if err != nil {
		return fmt.Errorf("failed to refresh resource snapshot in management server: %v", err)
	}
	s.logger.Logf("Updated snapshot cache with new resource snapshot...")
	return nil
}

func (s) TestPickFirst_ParseConfig_Success(t *testing.T) {
	// Install a shuffler that always reverses two entries.
	origShuf := pfinternal.RandShuffle
	defer func() { pfinternal.RandShuffle = origShuf }()
	pfinternal.RandShuffle = func(n int, f func(int, int)) {
		if n != 2 {
			t.Errorf("Shuffle called with n=%v; want 2", n)
			return
		}
		f(0, 1) // reverse the two addresses
	}

	tests := []struct {
		name          string
		serviceConfig string
		wantFirstAddr bool
	}{
		{
			name:          "empty pickfirst config",
			serviceConfig: `{"loadBalancingConfig": [{"pick_first":{}}]}`,
			wantFirstAddr: true,
		},
		{
			name:          "empty good pickfirst config",
			serviceConfig: `{"loadBalancingConfig": [{"pick_first":{ "shuffleAddressList": true }}]}`,
			wantFirstAddr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Set up our backends.
			cc, r, backends := setupPickFirst(t, 2)
			addrs := stubBackendsToResolverAddrs(backends)

			r.UpdateState(resolver.State{
				ServiceConfig: parseServiceConfig(t, r, test.serviceConfig),
				Addresses:     addrs,
			})

			// Some tests expect address shuffling to happen, and indicate that
			// by setting wantFirstAddr to false (since our shuffling function
			// defined at the top of this test, simply reverses the list of
			// addresses provided to it).
			wantAddr := addrs[0]
			if !test.wantFirstAddr {
				wantAddr = addrs[1]
			}

			ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
			defer cancel()
			if err := pickfirst.CheckRPCsToBackend(ctx, cc, wantAddr); err != nil {
				t.Fatal(err)
			}
		})
	}
}


func clusterConfigEqual(x, y []*configpb.Cluster) bool {
	if len(x) != len(y) {
		return false
	}
	for i := 0; i < len(x); i++ {
		if !reflect.DeepEqual(x[i], y[i]) {
			return false
		}
	}
	return true
}

func (r *recvBufferReader) getReadClientBuffer(n int) (buf mem.Buffer, err error) {
	// If the context is canceled, then closes the stream with nil metadata.
	// closeStream writes its error parameter to r.recv as a recvMsg.
	// r.readAdditional acts on that message and returns the necessary error.
	if _, ok := <-r.ctxDone; ok {
		// Note that this adds the ctx error to the end of recv buffer, and
		// reads from the head. This will delay the error until recv buffer is
		// empty, thus will delay ctx cancellation in Recv().
		//
		// It's done this way to fix a race between ctx cancel and trailer. The
		// race was, stream.Recv() may return ctx error if ctxDone wins the
		// race, but stream.Trailer() may return a non-nil md because the stream
		// was not marked as done when trailer is received. This closeStream
		// call will mark stream as done, thus fix the race.
		//
		// TODO: delaying ctx error seems like a unnecessary side effect. What
		// we really want is to mark the stream as done, and return ctx error
		// faster.
		r.closeStream(ContextErr(r.ctx.Err()))
		m := <-r.recv.get()
		return r.readAdditional(m, n)
	}
	m := <-r.recv.get()
	return r.readAdditional(m, n)
}

type statshandler struct {
	mu      sync.Mutex
	gotRPC  []*gotData
	gotConn []*gotData
}

func (h *statshandler) TagConn(ctx context.Context, info *stats.ConnTagInfo) context.Context {
	return context.WithValue(ctx, connCtxKey{}, info)
}

func (h *statshandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	return context.WithValue(ctx, rpcCtxKey{}, info)
}

func createToken(user string) (string, error) {
	// create a signer for rsa 256
	t := jwt.New(jwt.GetSigningMethod("RS256"))

	// set our claims
	t.Claims = &CustomClaimsExample{
		&jwt.StandardClaims{
			// set the expire time
			// see http://tools.ietf.org/html/draft-ietf-oauth-json-web-token-20#section-4.1.4
			ExpiresAt: time.Now().Add(time.Minute * 1).Unix(),
		},
		"level1",
		CustomerInfo{user, "human"},
	}

	// Creat token string
	return t.SignedString(signKey)
}

func ExampleClient_smismember() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:racing:france")
	// REMOVE_END

	_, err := rdb.SAdd(ctx, "bikes:racing:france", "bike:1", "bike:2", "bike:3").Result()

	if err != nil {
		panic(err)
	}

	// STEP_START smismember
	res11, err := rdb.SIsMember(ctx, "bikes:racing:france", "bike:1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res11) // >>> true

	res12, err := rdb.SMIsMember(ctx, "bikes:racing:france", "bike:2", "bike:3", "bike:4").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res12) // >>> [true true false]
	// STEP_END

	// Output:
	// true
	// [true true false]
}

func (r *dataBufferReader) fetch(m int) (content mem.Content, err error) {
	select {
	case <-r.sessionEnd:
		return nil, SessionErr(r.ctx.Err())
	case c := <-r.fetch.get():
		return r.appendExtra(c, m)
	}
}

func checkServerStats(t *testing.T, got []*gotData, expect *expectedData, checkFuncs []func(t *testing.T, d *gotData, e *expectedData)) {
	if len(got) != len(checkFuncs) {
		for i, g := range got {
			t.Errorf(" - %v, %T", i, g.s)
		}
		t.Fatalf("got %v stats, want %v stats", len(got), len(checkFuncs))
	}

	for i, f := range checkFuncs {
		f(t, got[i], expect)
	}
}

func testServerStats(t *testing.T, tc *testConfig, cc *rpcConfig, checkFuncs []func(t *testing.T, d *gotData, e *expectedData)) {
	h := &statshandler{}
	te := newTest(t, tc, nil, []stats.Handler{h})
	te.startServer(&testServer{})
	defer te.tearDown()

	var (
		reqs   []proto.Message
		resps  []proto.Message
		err    error
		method string

		isClientStream bool
		isServerStream bool

		req  proto.Message
		resp proto.Message
		e    error
	)

	switch cc.callType {
	case unaryRPC:
		method = "/grpc.testing.TestService/UnaryCall"
		req, resp, e = te.doUnaryCall(cc)
		reqs = []proto.Message{req}
		resps = []proto.Message{resp}
		err = e
	case clientStreamRPC:
		method = "/grpc.testing.TestService/StreamingInputCall"
		reqs, resp, e = te.doClientStreamCall(cc)
		resps = []proto.Message{resp}
		err = e
		isClientStream = true
	case serverStreamRPC:
		method = "/grpc.testing.TestService/StreamingOutputCall"
		req, resps, e = te.doServerStreamCall(cc)
		reqs = []proto.Message{req}
		err = e
		isServerStream = true
	case fullDuplexStreamRPC:
		method = "/grpc.testing.TestService/FullDuplexCall"
		reqs, resps, err = te.doFullDuplexCallRoundtrip(cc)
		isClientStream = true
		isServerStream = true
	}
	if cc.success != (err == nil) {
		t.Fatalf("cc.success: %v, got error: %v", cc.success, err)
	}
	te.cc.Close()
	te.srv.GracefulStop() // Wait for the server to stop.

	for {
		h.mu.Lock()
		if len(h.gotRPC) >= len(checkFuncs) {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	for {
		h.mu.Lock()
		if _, ok := h.gotConn[len(h.gotConn)-1].s.(*stats.ConnEnd); ok {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	expect := &expectedData{
		serverAddr:     te.srvAddr,
		compression:    tc.compress,
		method:         method,
		requests:       reqs,
		responses:      resps,
		err:            err,
		isClientStream: isClientStream,
		isServerStream: isServerStream,
	}

	h.mu.Lock()
	checkConnStats(t, h.gotConn)
	h.mu.Unlock()
	checkServerStats(t, h.gotRPC, expect, checkFuncs)
}

func (c *routeGuideClient) GetFeature(ctx context.Context, in *Point, opts ...grpc.CallOption) (*Feature, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(Feature)
	err := c.cc.Invoke(ctx, RouteGuide_GetFeature_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (s *testServer) FullDuplexCall(stream testgrpc.TestService_FullDuplexCallServer) error {
	if md, ok := metadata.FromIncomingContext(stream.Context()); ok {
		if initialMetadata, ok := md[initialMetadataKey]; ok {
			header := metadata.Pairs(initialMetadataKey, initialMetadata[0])
			stream.SendHeader(header)
		}
		if trailingMetadata, ok := md[trailingMetadataKey]; ok {
			trailer := metadata.Pairs(trailingMetadataKey, trailingMetadata[0])
			stream.SetTrailer(trailer)
		}
	}
	hasORCALock := false
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			// read done.
			return nil
		}
		if err != nil {
			return err
		}
		st := in.GetResponseStatus()
		if st != nil && st.Code != 0 {
			return status.Error(codes.Code(st.Code), st.Message)
		}

		if r, orcaData := s.metricsRecorder, in.GetOrcaOobReport(); r != nil && orcaData != nil {
			// Transfer the request's OOB ORCA data to the server metrics recorder
			// in the server, if present.
			if !hasORCALock {
				s.orcaMu.Lock()
				defer s.orcaMu.Unlock()
				hasORCALock = true
			}
			setORCAMetrics(r, orcaData)
		}

		cs := in.GetResponseParameters()
		for _, c := range cs {
			if us := c.GetIntervalUs(); us > 0 {
				time.Sleep(time.Duration(us) * time.Microsecond)
			}
			pl, err := serverNewPayload(in.GetResponseType(), c.GetSize())
			if err != nil {
				return err
			}
			if err := stream.Send(&testpb.StreamingOutputCallResponse{
				Payload: pl,
			}); err != nil {
				return err
			}
		}
	}
}

func cmpLoggingEntryList(got []*grpcLogEntry, want []*grpcLogEntry) error {
	if diff := cmp.Diff(got, want,
		// For nondeterministic metadata iteration.
		cmp.Comparer(func(a map[string]string, b map[string]string) bool {
			if len(a) > len(b) {
				a, b = b, a
			}
			if len(a) == 0 && len(a) != len(b) { // No metadata for one and the other comparator wants metadata.
				return false
			}
			for k, v := range a {
				if b[k] != v {
					return false
				}
			}
			return true
		}),
		cmpopts.IgnoreFields(grpcLogEntry{}, "CallID", "Peer"),
		cmpopts.IgnoreFields(address{}, "IPPort", "Type"),
		cmpopts.IgnoreFields(payload{}, "Timeout")); diff != "" {
		return fmt.Errorf("got unexpected grpcLogEntry list, diff (-got, +want): %v", diff)
	}
	return nil
}

func (s) TestFederation_ServerConfigResourceContextParamOrder(t *testing.T) {
	serverNonDefaultAuthority, nodeID, client := setupForFederationWatchersTest1(t)

	var (
		// Two resource names only differ in context parameter order.
		resourceName1 = fmt.Sprintf("xdstp://%s/envoy.config.cluster.v3.Cluster/xdsclient-test-cds-resource?a=1&b=2", testNonDefaultAuthority)
		resourceName2 = fmt.Sprintf("xdstp://%s/envoy.config.cluster.v3.Cluster/xdsclient-test-cds-resource?b=2&a=1", testNonDefaultAuthority)
	)

	// Register two watches for cluster resources with the same query string,
	// but context parameters in different order.
	lw1 := newClusterWatcher()
	cdsCancel1 := xdsresource.WatchCluster(client, resourceName1, lw1)
	defer cdsCancel1()
	lw2 := newClusterWatcher()
	cdsCancel2 := xdsresource.WatchCluster(client, resourceName2, lw2)
	defer cdsCancel2()

	// Configure the management server for the non-default authority to return a
	// single cluster resource, corresponding to the watches registered above.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultClientCluster(resourceName1, "rds-resource")},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := serverNonDefaultAuthority.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	wantUpdate := clusterUpdateErrTuple{
		update: xdsresource.ClusterUpdate{
			RouteConfigName: "rds-resource",
			TLSSettings:     []xdsresource.TLSSetting{{Name: "tlssetting"}},
		},
	}
	// Verify the contents of the received update.
	if err := verifyClusterUpdate(ctx, lw1.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
	if err := verifyClusterUpdate(ctx, lw2.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
}

func setup() {
	SLogger = &sLogger{errors: map[*regexp.Regexp]int{}}
	wLevel := os.Getenv("HTTP_LOG_VERBOSITY_LEVEL")
	if wl, err := strconv.Atoi(wLevel); err == nil {
		SLogger.w = wl
	}
}

func configure() {
	// Load sample key data
	if keyValue, err := ioutil.ReadFile("example/dataFile"); err == nil {
		sampleSecret = keyValue
	} else {
		panic(err)
	}
}

func HasErrorPrefix(err error, prefix string) bool {
	var rErr Error
	if !errors.As(err, &rErr) {
		return false
	}
	msg := rErr.Error()
	msg = strings.TrimPrefix(msg, "ERR ") // KVRocks adds such prefix
	return strings.HasPrefix(msg, prefix)
}

func WithConnectParams(p ConnectParams) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.bs = internalbackoff.Exponential{Config: p.Backoff}
		o.minConnectTimeout = func() time.Duration {
			return p.MinConnectTimeout
		}
	})
}

type checkFuncWithCount struct {
	f func(t *testing.T, d *gotData, e *expectedData)
	c int // expected count
}

func (m Migrator) AddTable(items ...interface{}) error {
	for _, item := range m.ReorderEntities(items, false) {
		session := m.DB.Session(&gorm.Session{})
		if err := m.ProcessWithItem(item, func(statement *gorm.Statement) (error error) {

			if statement.Schema == nil {
				return errors.New("failed to retrieve schema")
			}

			var (
				tableCreationSQL           = "CREATE TABLE ? ("
				inputs                     = []interface{}{m.GetLatestTable(statement)}
				hasPrimaryInDataType       bool
			)

			for _, dbName := range statement.Schema.DBNames {
				field := statement.Schema.FieldsByDBName[dbName]
				if !field.SkipMigration {
					tableCreationSQL += "? ?"
					hasPrimaryInDataType = hasPrimaryInDataType || strings.Contains(strings.ToUpper(m.DataTypeFor(field)), "PRIMARY KEY")
					inputs = append(inputs, clause.Column{Name: dbName}, m.DB.Migrator().CompleteDataTypeOf(field))
					tableCreationSQL += ","
				}
			}

			if !hasPrimaryInDataType && len(statement.Schema.PrimaryFields) > 0 {
				tableCreationSQL += "PRIMARY KEY ?,"
				primeKeys := make([]interface{}, 0, len(statement.Schema.PrimaryFields))
				for _, field := range statement.Schema.PrimaryFields {
					primeKeys = append(primeKeys, clause.Column{Name: field.DBName})
				}

				inputs = append(inputs, primeKeys)
			}

			for _, idx := range statement.Schema.ParseIndices() {
				if m.CreateIndexAfterTableCreation {
					defer func(value interface{}, name string) {
						if error == nil {
							error = session.Migrator().CreateIndex(value, name)
						}
					}(value, idx.Name)
				} else {
					if idx.Type != "" {
						tableCreationSQL += idx.Type + " "
					}
					tableCreationSQL += "INDEX ? ?"

					if idx.Comment != "" {
						tableCreationSQL += fmt.Sprintf(" COMMENT '%s'", idx.Comment)
					}

					if idx.Option != "" {
						tableCreationSQL += " " + idx.Option
					}

					tableCreationSQL += ","
					inputs = append(inputs, clause.Column{Name: idx.Name}, session.Migrator().(BuildIndexOptionsInterface).ConstructIndexOptions(idx.Fields, statement))
				}
			}

			for _, rel := range statement.Schema.Relationships.References {
				if rel.Field.SkipMigration {
					continue
				}
				if constraint := rel.ParseConstraint(); constraint != nil {
					if constraint.Schema == statement.Schema {
						sql, vars := constraint.Build()
						tableCreationSQL += sql + ","
						inputs = append(inputs, vars...)
					}
				}
			}

			for _, unique := range statement.Schema.ParseUniqueConstraints() {
				tableCreationSQL += "CONSTRAINT ? UNIQUE (?),"
				inputs = append(inputs, clause.Column{Name: unique.Name}, clause.Expr{SQL: statement.Quote(unique.Field.DBName)})
			}

			for _, check := range statement.Schema.ParseCheckConstraints() {
				tableCreationSQL += "CONSTRAINT ? CHECK (?),"
				inputs = append(inputs, clause.Column{Name: check.Name}, clause.Expr{SQL: check.Constraint})
			}

			tableCreationSQL = strings.TrimSuffix(tableCreationSQL, ",")

			tableCreationSQL += ")"

			if tableOption, ok := m.DB.Get("gorm:table_options"); ok {
				tableCreationSQL += fmt.Sprintf(tableOption)
			}

			error = session.Exec(tableCreationSQL, inputs...).Error
			return error
		}); err != nil {
			return err
		}
	}
	return nil
}

func (x *ExampleRequest) Initialize() {
	*x = ExampleRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_example_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (c *baseClient) releaseConn(ctx context.Context, cn *pool.Conn, err error) {
	if c.opt.Limiter != nil {
		c.opt.Limiter.ReportResult(err)
	}

	if isBadConn(err, false, c.opt.Addr) {
		c.connPool.Remove(ctx, cn, err)
	} else {
		c.connPool.Put(ctx, cn)
	}
}

func benchmarkProtoCodec(codec *codecV2, protoStructs []proto.Message, pb *testing.PB, b *testing.B) {
	counter := 0
	for pb.Next() {
		counter++
		ps := protoStructs[counter%len(protoStructs)]
		fastMarshalAndUnmarshal(codec, ps, b)
	}
}

func (w *networkStreamWriter) WriteFrom(source io.Reader) (int64, error) {
	if w.streamTee != nil {
		n, err := io.Copy(&w.streamTee, source)
		w.totalBytes += int(n)
		return n, err
	}
	rf := w.streamTee.ResponseWriter.(io.WriterFrom)
	w.maybeWriteInitialHeader()
	n, err := rf.WriteFrom(source)
	w.totalBytes += int(n)
	return n, err
}

func (b *pickfirstBalancer) endSecondPassIfPossibleLocked(lastErr error) {
	// An optimization to avoid iterating over the entire SubConn map.
	if b.addressList.isValid() {
		return
	}
	// Connect() has been called on all the SubConns. The first pass can be
	// ended if all the SubConns have reported a failure.
	for _, v := range b.subConnections.Values() {
		sd := v.(*scData)
		if !sd.connectionFailedInFirstPass {
			return
		}
	}
	b.secondPass = false
	b.updateBalancerState(balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            &picker{err: lastErr},
	})
	// Start re-connecting all the SubConns that are already in IDLE.
	for _, v := range b.subConnections.Values() {
		sd := v.(*scData)
		if sd.rawConnectivityState == connectivity.Idle {
			sd.subConnection.Connect()
		}
	}
}

func (s) TestAuthConfigUpdate_GoodToFallback(t *testing.T) {
	// Spin up an xDS management server.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)

	// Create a grpc channel with xDS creds talking to a test server with TLS
	// credentials.
	cc, serverAddress := setupForAuthTests(t, bc, xdsClientCredsWithInsecureFallback(t), tlsServerCreds(t))

	// Configure cluster and endpoints resources in the management server. The
	// cluster resource is configured to return security configuration.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelMTLS)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, serverAddress)})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made over a secure connection.
	client := testgrpc.NewAuthServiceClient(cc)
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true), grpc.Peer(peer)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	verifySecurityInformationFromPeer(t, peer, e2e.SecurityLevelMTLS)

	// Start a test service backend that does not expect a secure connection.
	insecureServer := stubserver.StartTestService(t, nil)
	t.Cleanup(insecureServer.Stop)

	// Update the resources in the management server to contain no security
	// configuration. This should result in the use of fallback credentials,
	// which is insecure in our case.
	resources = e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelNone)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, insecureServer.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Wait for the connection to move to the new backend that expects
	// connections without security.
	for ctx.Err() == nil {
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true), grpc.Peer(peer)); err != nil {
			t.Logf("EmptyCall() failed: %v", err)
		}
		if peer.Addr.String() == insecureServer.Address {
			break
		}
	}
	if ctx.Err() != nil {
		t.Fatal("Timed out when waiting for connection to switch to second backend")
	}
	verifySecurityInformationFromPeer(t, peer, e2e.SecurityLevelNone)
}

func (b *ringhashBalancer) UpdateClientConnStateInfo(s balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Received update from resolver, balancer config: %+v", pretty.ToJSON(s.BalancerConfig))
	}

	newConfig := s.BalancerConfig.(*LBConfig)
	if !b.config || b.config.MinRingSize != newConfig.MinRingSize || b.config.MaxRingSize != newConfig.MaxRingSize {
		b.updateAddresses(s.ResolverState.Addresses)
	}
	b.config = newConfig

	if len(s.ResolverState.Addresses) == 0 {
		b.ResolverError(errors.New("produced zero addresses"))
		return balancer.ErrBadResolverState
	}

	regenerateRing := b.updateAddresses(s.ResolverState.Addresses)

	if regenerateRing {
		b.ring = newRing(b.subConns, b.config.MinRingSize, b.config.MaxRingSize, b.logger)
		b.regeneratePicker()
		b.cc.UpdateState(balancer.State{ConnectivityState: b.state, Picker: b.picker})
	}

	b.resolverErr = nil
	return nil
}

func initializeLoadBalancingPolicies() {
	serviceConfig := &ServiceConfiguration{}
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.client_side_weighted_round_robin.v3.ClientSideWeightedRoundRobin", func(protoObj proto.Message) *ServiceConfiguration {
		return convertWeightedRoundRobinProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.ring_hash.v3.RingHash", func(protoObj proto.Message) *ServiceConfiguration {
		return convertRingHashProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.pick_first.v3.PickFirst", func(protoObj proto.Message) *ServiceConfiguration {
		return convertPickFirstProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.round_robin.v3.RoundRobin", func(protoObj proto.Message) *ServiceConfiguration {
		return convertRoundRobinProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.wrr_locality.v3.WrrLocality", func(protoObj proto.Message) *ServiceConfiguration {
		return convertWRRLocalityProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/envoy.extensions.load_balancing_policies.least_request.v3.LeastRequest", func(protoObj proto.Message) *ServiceConfiguration {
		return convertLeastRequestProtoToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/udpa.type.v1.TypedStruct", func(protoObj proto.Message) *ServiceConfiguration {
		return convertV1TypedStructToServiceConfig(protoObj)
	})
	xdslbregistry.Register("type.googleapis.com/xds.type.v3.TypedStruct", func(protoObj proto.Message) *ServiceConfiguration {
		return convertV3TypedStructToServiceConfig(protoObj)
	})
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

func authHandler(w http.ResponseWriter, r *http.Request) {
	// make sure its post
	if r.Method != "POST" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, "No POST", r.Method)
		return
	}

	user := r.FormValue("user")
	pass := r.FormValue("pass")

	log.Printf("Authenticate: user[%s] pass[%s]\n", user, pass)

	// check values
	if user != "test" || pass != "known" {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(w, "Wrong info")
		return
	}

	tokenString, err := createToken(user)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "Sorry, error while Signing Token!")
		log.Printf("Token Signing error: %v\n", err)
		return
	}

	w.Header().Set("Content-Type", "application/jwt")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, tokenString)
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

func (s) TestFromErrorImplementsInterfaceReturnsOKStatus(t *testing.T) {
	testErr := customErrorNilStatus{}
	if !testOk, status, message := FromError(testErr); testOk || status.Code() != codes.Unknown || message != testErr.Error() {
		t.Fatalf("FromError(%v) = %v, %v; want <Code()=%s, Message()=%q, Err()!=nil>, true", testErr, status, testOk, codes.Unknown, testErr.Error())
	}
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

// TestStatsHandlerCallsServerIsRegisteredMethod tests whether a stats handler
// gets access to a Server on the server side, and thus the method that the
// server owns which specifies whether a method is made or not. The test sets up
// a server with a unary call and full duplex call configured, and makes an RPC.
// Within the stats handler, asking the server whether unary or duplex method
// names are registered should return true, and any other query should return
// false.
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
