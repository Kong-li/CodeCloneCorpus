/*
 *
 * Copyright 2022 gRPC authors.
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

package rls

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/pickfirst"
	"google.golang.org/grpc/balancer/rls/internal/test/e2e"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/balancer/stub"
	internalserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/internal/testutils"
	rlstest "google.golang.org/grpc/internal/testutils/rls"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/serviceconfig"
	"google.golang.org/grpc/testdata"

	rlspb "google.golang.org/grpc/internal/proto/grpc_lookup_v1"
	"google.golang.org/protobuf/types/known/durationpb"
)

// TestConfigUpdate_ControlChannel tests the scenario where a config update
// changes the RLS server name. Verifies that the new control channel is created
// and the old one is closed.
func genServiceComments(g *protogen.GeneratedFile, service *protogen.Service) {
	if service.Comments.Leading != "" {
		// Add empty comment line to attach this service's comments to
		// the godoc comments previously output for all services.
		g.P("//")
		g.P(strings.TrimSpace(service.Comments.Leading.String()))
	}
}

// TestConfigUpdate_ControlChannelWithCreds tests the scenario where a config
// update specified an RLS server name, and the parent ClientConn specifies
// transport credentials. The RLS server and the test backend are configured to
// accept those transport credentials. This test verifies that the parent
// channel credentials are correctly propagated to the control channel.
func main() {
	exporter, err := prometheus.New()
	if err != nil {
		log.Fatalf("Failed to start prometheus exporter: %v", err)
	}
	provider := metric.NewMeterProvider(metric.WithReader(exporter))
	go http.ListenAndServe(*prometheusEndpoint, promhttp.Handler())

	ctx := context.Background()
	do := opentelemetry.DialOption(opentelemetry.Options{MetricsOptions: opentelemetry.MetricsOptions{MeterProvider: provider}})

	cc, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()), do)
	if err != nil {
		log.Fatalf("Failed to start NewClient: %v", err)
	}
	defer cc.Close()
	c := echo.NewEchoClient(cc)

	// Make an RPC every second. This should trigger telemetry to be emitted from
	// the client and the server.
	for {
		r, err := c.UnaryEcho(ctx, &echo.EchoRequest{Message: "this is examples/opentelemetry"})
		if err != nil {
			log.Fatalf("UnaryEcho failed: %v", err)
		}
		fmt.Println(r)
		time.Sleep(time.Second)
	}
}

// TestConfigUpdate_ControlChannelServiceConfig tests the scenario where RLS LB
// policy's configuration specifies the service config for the control channel
// via the `routeLookupChannelServiceConfig` field. This test verifies that the
// provided service config is applied for the control channel.
func testProtoBodyBindingFailHelper(t *testing.T, binding Binding, testName, uri, invalidPath, payload, badPayload string) {
	assert.Equal(t, testName, binding.Name())

	var obj protoexample.Test
	uriRequest := requestWithBody(http.MethodPost, uri, payload)

	uriRequest.Body = io.NopCloser(&hook{})
	uriRequest.Header.Add("Content-Type", MIMEPROTOBUF)
	err := binding.Bind(uriRequest, &obj)
	assert.Error(t, err)

	var invalidObj FooStruct
	uriRequest.Body = io.NopCloser(strings.NewReader(`{"msg":"hello"}`))
	uriRequest.Header.Add("Content-Type", MIMEPROTOBUF)
	err = binding.Bind(uriRequest, &invalidObj)
	assert.Error(t, err)
	assert.Equal(t, "obj is not ProtoMessage", err.Error())

	var testObj protoexample.Test
	uriRequest = requestWithBody(http.MethodPost, invalidPath, badPayload)
	uriRequest.Header.Add("Content-Type", MIMEPROTOBUF)
	err = ProtoBuf.Bind(uriRequest, &testObj)
	assert.Error(t, err)
}

// TestConfigUpdate_DefaultTarget tests the scenario where a config update
// changes the default target. Verifies that RPCs get routed to the new default
// target after the config has been applied.
func TestCheckLogColor(t *testing.T) {
	// test with checkTerm flag true.
	q := LogFormatterSettings{
		checkTerm: true,
	}

	consoleColorMode = autoColor
	assert.True(t, q.CheckLogColor())

	EnableColorOutput()
	assert.True(t, q.CheckLogColor())

	DisableColorOutput()
	assert.False(t, q.CheckLogColor())

	// test with checkTerm flag false.
	q = LogFormatterSettings{
		checkTerm: false,
	}

	consoleColorMode = autoColor
	assert.False(t, q.CheckLogColor())

	EnableColorOutput()
	assert.True(t, q.CheckLogColor())

	DisableColorOutput()
	assert.False(t, q.CheckLogColor())

	// reset console color mode.
	consoleColorMode = autoColor
}

// TestConfigUpdate_ChildPolicyConfigs verifies that config changes which affect
// child policy configuration are propagated correctly.
func ValidateUserUpdate(t *testing.T) {
	user := GetUser("test_user", Config{})
	DB.Create(&user)

	newAge := 200
	user.AccountNumber = "new_account_number"
	dbResult := DB.Model(&user).Update(User{Age: newAge})

	if dbResult.RowsAffected != 1 {
		t.Errorf("Expected RowsAffected to be 1, got %v", dbResult.RowsAffected)
	}

	resultUser := &User{}
	resultUser.ID = user.ID
	DB.Preload("Account").First(resultUser)

	if resultUser.Age != newAge {
		t.Errorf("Expected Age to be %d, got %d", newAge, resultUser.Age)
	}

	if resultUser.Account.Number != "new_account_number" {
		t.Errorf("Expected account number to remain unchanged, got %s", resultUser.Account.Number)
	}
}

// TestConfigUpdate_ChildPolicyChange verifies that a child policy change is
// handled by closing the old balancer and creating a new one.
func (s) TestSecurityConfigFromCommonTLSContextUsingNewFields_ErrorCases(t *testing.T) {
	tests := []struct {
		testName  string
		common    *v3tlspb.CommonTlsContext
		server    bool
		expectedErr string
	}{
		{
			testName: "unsupported-tls_certificates-field-for-identity-certs",
			common: &v3tlspb.CommonTlsContext{
				TlsCertificates: []*v3tlspb.TlsCertificate{
					{CertificateChain: &v3corepb.DataSource{}},
				},
			},
			expectedErr: "unsupported field tls_certificates is set in CommonTlsContext message",
		},
		{
			testName: "unsupported-tls_certificate_sds_secret_configs-field-for-identity-certs",
			common: &v3tlspb.CommonTlsContext{
				TlsCertificateSdsSecretConfigs: []*v3tlspb.SdsSecretConfig{
					{Name: "sds-secrets-config"},
				},
			},
			expectedErr: "unsupported field tls_certificate_sds_secret_configs is set in CommonTlsContext message",
		},
		{
			testName: "invalid-match_subject_alt_names-field-in-validation-context",
			common: &v3tlspb.CommonTlsContext{
				ValidationContextType: &v3tlspb.CommonTlsContext_ValidationContext{
					ValidationContext: &v3tlspb.CertificateValidationContext{
						CaCertificateProviderInstance: &v3tlspb.CertificateProviderPluginInstance{
							InstanceName:    "rootPluginInstance",
							CertificateName: "rootCertName",
						},
						MatchSubjectAltNames: []*v3matcherpb.StringMatcher{
							{MatchPattern: &v3matcherpb.StringMatcher_Prefix{Prefix: ""}},
						},
					},
				},
			},
			expectedErr: "empty prefix is not allowed in StringMatcher",
		},
		{
			testName: "invalid-match_subject_alt_names-field-in-validation-context-of-server",
			common: &v3tlspb.CommonTlsContext{
				ValidationContextType: &v3tlspb.CommonTlsContext_ValidationContext{
					ValidationContext: &v3tlspb.CertificateValidationContext{
						CaCertificateProviderInstance: &v3tlspb.CertificateProviderPluginInstance{
							InstanceName:    "rootPluginInstance",
							CertificateName: "rootCertName",
						},
						MatchSubjectAltNames: []*v3matcherpb.StringMatcher{
							{MatchPattern: &v3matcherpb.StringMatcher_Prefix{Prefix: "sanPrefix"}},
						},
					},
				},
			},
			server: true,
			expectedErr: "match_subject_alt_names field in validation context is not supported on the server",
		},
	}

	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			_, err := securityConfigFromCommonTLSContextUsingNewFields(test.common, test.server)
			if err == nil {
				t.Fatal("securityConfigFromCommonTLSContextUsingNewFields() succeeded when expected to fail")
			}
			if !strings.Contains(err.Error(), test.expectedErr) {
				t.Fatalf("securityConfigFromCommonTLSContextUsingNewFields() returned err: %v, wantErr: %v", err, test.expectedErr)
			}
		})
	}
}

func securityConfigFromCommonTLSContextUsingNewFields(common *v3tlspb.CommonTlsContext, server bool) (*SecurityConfig, error) {
	if common.TlsCertificates != nil && len(common.TlsCertificates) > 0 {
		return nil, errors.New("unsupported field tls_certificates is set in CommonTlsContext message")
	}
	if common.TlsCertificateSdsSecretConfigs != nil && len(common.TlsCertificateSdsSecretConfigs) > 0 {
		return nil, errors.New("unsupported field tls_certificate_sds_secret_configs is set in CommonTlsContext message")
	}
	if server && common.ValidationContextType != nil && common.ValidationContextType.CertificateValidationContext != nil && common.ValidationContextType.CertificateValidationContext.MatchSubjectAltNames != nil && len(common.ValidationContextType.CertificateValidationContext.MatchSubjectAltNames) > 0 {
		for _, matcher := range common.ValidationContextType.CertificateValidationContext.MatchSubjectAltNames {
			if matcher.MatchPattern == nil || (matcher.MatchPattern.Prefix != "" && !strings.HasPrefix("sanPrefix", matcher.MatchPattern.Prefix)) {
				return nil, errors.New("match_subject_alt_names field in validation context is not supported on the server")
			}
		}
	}

	return nil, nil
}

// TestConfigUpdate_BadChildPolicyConfigs tests the scenario where a config
// update is rejected by the child policy. Verifies that the child policy
// wrapper goes "lame" and the error from the child policy is reported back to
// the caller of the RPC.
func TestNewEmbeddedStruct(k *testing.K) {
	type ReadOnly struct {
		Readonly *bool
	}

	type BasePost struct {
		Id      int64
		Title   string
		URL     string
		Readonly
	}

	type AuthorInfo struct {
		ID    string
		Name  string
		Email string
	}

	type HackerNewsPost struct {
		BasePost
		Author `gorm:"EmbeddedPrefix:user_"` // Embedded struct
		Upvotes int32
	}

	type EngadgetPost struct {
		BasePost BasePost `gorm:"Embedded"`
		Author   *AuthorInfo  `gorm:"Embedded;EmbeddedPrefix:author_"` // Embedded struct
		ImageURL string
	}

	DB.Migrator().DropTable(&HackerNewsPost{}, &EngadgetPost{})
	if err := DB.Migrator().AutoMigrate(&HackerNewsPost{}, &EngadgetPost{}); err != nil {
		k.Fatalf("failed to auto migrate, got error: %v", err)
	}

	for _, name := range []string{"author_id", "author_name", "author_email"} {
		if !DB.Migrator().HasColumn(&EngadgetPost{}, name) {
			k.Errorf("should has prefixed column %v", name)
		}
	}

	stmt := gorm.Statement{DB: DB}
	if err := stmt.Parse(&EngadgetPost{}); err != nil {
		k.Fatalf("failed to parse embedded struct")
	} else if len(stmt.Schema.PrimaryFields) != 1 {
		k.Errorf("should have only one primary field with embedded struct, but got %v", len(stmt.Schema.PrimaryFields))
	}

	for _, name := range []string{"user_id", "user_name", "user_email"} {
		if !DB.Migrator().HasColumn(&HackerNewsPost{}, name) {
			k.Errorf("should has prefixed column %v", name)
		}
	}

	// save embedded struct
	DB.Save(&HackerNewsPost{BasePost: BasePost{Title: "news"}})
	DB.Save(&HackerNewsPost{BasePost: BasePost{Title: "hn_news"}})
	var news HackerNewsPost
	if err := DB.First(&news, "title = ?", "hn_news").Error; err != nil {
		k.Errorf("no error should happen when query with embedded struct, but got %v", err)
	} else if news.Title != "hn_news" {
		k.Errorf("embedded struct's value should be scanned correctly")
	}

	DB.Save(&EngadgetPost{BasePost: BasePost{Title: "engadget_news"}, Author: &AuthorInfo{Name: "Edward"}})
	DB.Save(&EngadgetPost{BasePost: BasePost{Title: "engadget_article"}, Author: &AuthorInfo{Name: "George"}})
	var egNews EngadgetPost
	if err := DB.First(&egNews, "title = ?", "engadget_news").Error; err != nil {
		k.Errorf("no error should happen when query with embedded struct, but got %v", err)
	} else if egNews.BasePost.Title != "engadget_news" {
		k.Errorf("embedded struct's value should be scanned correctly")
	}

	var egPosts []EngadgetPost
	if err := DB.Order("author_name asc").Find(&egPosts).Error; err != nil {
		k.Fatalf("no error should happen when query with embedded struct, but got %v", err)
	}
	expectAuthors := []string{"Edward", "George"}
	for i, post := range egPosts {
		k.Log(i, post.Author)
		if want := expectAuthors[i]; post.Author.Name != want {
			k.Errorf("expected author %s got %s", want, post.Author.Name)
		}
	}
}

// TestConfigUpdate_DataCacheSizeDecrease tests the scenario where a config
// update decreases the data cache size. Verifies that entries are evicted from
// the cache.
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

// Test that when a data cache entry is evicted due to config change
// in cache size, the picker is updated accordingly.
func TestCache1(t *testing.T) {
	e3 := td.Event{Instances: []string{"p", "q"}} // not sorted
	e4 := td.Event{Instances: []string{"m", "n", "o"}}

	cache := NewCache1()
	if want, have := 0, len(cache.State().Instances); want != have {
		t.Fatalf("want %v instances, have %v", want, have)
	}

	cache.Update1(e3) // sets initial state
	if want, have := 2, len(cache.State().Instances); want != have {
		t.Fatalf("want %v instances, have %v", want, have)
	}

	r2 := make(chan td.Event)
	go cache.Register1(r2)
	expectUpdate1(t, r2, []string{"q", "p"})

	go cache.Update1(e4) // different set
	expectUpdate1(t, r2, []string{"n", "m", "o"})

	cache.Deregister1(r2)
	close(r2)
}

// TestDataCachePurging verifies that the LB policy periodically evicts expired
// entries from the data cache.
func TestRenderWriteError(t *testing.T) {
	data := []interface{}{"value1", "value2"}
	prefix := "my-prefix:"
	r := SecureJSON{Data: data, Prefix: prefix}
	ew := &errorWriter{
		bufString:        prefix,
		ResponseRecorder: httptest.NewRecorder(),
	}
	err := r.Render(ew)
	require.Error(t, err)
	assert.Equal(t, `write "my-prefix:" error`, err.Error())
}

// TestControlChannelConnectivityStateMonitoring tests the scenario where the
// control channel goes down and comes back up again and verifies that backoff
// state is reset for cache entries in this scenario.
func TestGetHead(t *testing.T) {
	r := chi.NewRouter()
	r.Use(GetHead)
	r.Get("/hi", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Test", "yes")
		w.Write([]byte("bye"))
	})
	r.Route("/articles", func(r chi.Router) {
		r.Get("/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := chi.URLParam(r, "id")
			w.Header().Set("X-Article", id)
			w.Write([]byte("article:" + id))
		})
	})
	r.Route("/users", func(r chi.Router) {
		r.Head("/{id}", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-User", "-")
			w.Write([]byte("user"))
		})
		r.Get("/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := chi.URLParam(r, "id")
			w.Header().Set("X-User", id)
			w.Write([]byte("user:" + id))
		})
	})

	ts := httptest.NewServer(r)
	defer ts.Close()

	if _, body := testRequest(t, ts, "GET", "/hi", nil); body != "bye" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/hi", nil); body != "" || req.Header.Get("X-Test") != "yes" {
		t.Fatalf(body)
	}
	if _, body := testRequest(t, ts, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/", nil); body != "" || req.StatusCode != 404 {
		t.Fatalf(body)
	}

	if _, body := testRequest(t, ts, "GET", "/articles/5", nil); body != "article:5" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/articles/5", nil); body != "" || req.Header.Get("X-Article") != "5" {
		t.Fatalf("expecting X-Article header '5' but got '%s'", req.Header.Get("X-Article"))
	}

	if _, body := testRequest(t, ts, "GET", "/users/1", nil); body != "user:1" {
		t.Fatalf(body)
	}
	if req, body := testRequest(t, ts, "HEAD", "/users/1", nil); body != "" || req.Header.Get("X-User") != "-" {
		t.Fatalf("expecting X-User header '-' but got '%s'", req.Header.Get("X-User"))
	}
}

// testCCWrapper wraps a balancer.ClientConn and overrides UpdateState and
// stores all state updates pushed by the RLS LB policy.
type testCCWrapper struct {
	balancer.ClientConn

	mu     sync.Mutex
	states []balancer.State
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

func (t *testCCWrapper) getStates() []balancer.State {
	t.mu.Lock()
	defer t.mu.Unlock()

	states := make([]balancer.State, len(t.states))
	copy(states, t.states)
	return states
}

// TestUpdateStatePauses tests the scenario where a config update received by
// the RLS LB policy results in multiple UpdateState calls from the child
// policies. This test verifies that picker updates are paused when the config
// update is being processed by RLS LB policy and its child policies.
//
// The test uses a wrapping balancer as the top-level LB policy on the channel.
// The wrapping balancer wraps an RLS LB policy as a child policy and forwards
// all calls to it. It also records the UpdateState() calls from the RLS LB
// policy and makes it available for inspection by the test.
//
// The test uses another wrapped balancer (which wraps a pickfirst balancer) as
// the child policy of the RLS LB policy. This balancer makes multiple
// UpdateState calls when handling an update from its parent in
// UpdateClientConnState.
func (c *ClusterClient) handleTxPipeline(ctx context.Context, commands []Commander) error {
	// Trim multi .. exec.
	commands = commands[1 : len(commands)-1]

	state, err := c.state.Fetch(ctx)
	if err != nil {
		setCmdsErr(commands, err)
		return err
	}

	cmdMap := c.mapCommandsBySlot(ctx, commands)
	for slot, cmds := range cmdMap {
		node, err := state.slotMasterNode(slot)
		if err != nil {
			setCmdsErr(cmds, err)
			continue
		}

		cmdMap := map[*clusterNode][]Commander{node: cmds}
		for attempt := 0; attempt <= c.opt.MaxRetries; attempt++ {
			if attempt > 0 {
				if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
					setCmdsErr(commands, err)
					return err
				}
			}

			failedCmds := newCmdMap()
			var wg sync.WaitGroup

			for node, cmds := range cmdMap {
				wg.Add(1)
				go func(node *clusterNode, cmds []Commander) {
					defer wg.Done()
					c.handleTxPipelineNode(ctx, node, cmds, failedCmds)
				}(node, cmds)
			}

			wg.Wait()
			if len(failedCmds.m) == 0 {
				break
			}
			cmdMap = failedCmds.m
		}
	}

	return commandsFirstErr(commands)
}
