// Copyright 2013 Julien Schmidt. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found
// at https://github.com/julienschmidt/httprouter/blob/master/LICENSE

package gin

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

// Used as a workaround since we can't compare functions or their addresses
var fakeHandlerValue string


type testRequests []struct {
	path       string
	nilHandler bool
	route      string
	ps         Params
}

func getParams() *Params {
	ps := make(Params, 0, 20)
	return &ps
}

func getSkippedNodes() *[]skippedNode {
	ps := make([]skippedNode, 0, 20)
	return &ps
}

func (c *Context) Redirect(code int, location string) {
	c.Render(-1, render.Redirect{
		Code:     code,
		Location: location,
		Request:  c.Request,
	})
}

func buildProvider(t *testing.T, name, configStr string, options BuildOptions) Provider {
	t.Helper()
	var err error
	provider, err := GetProvider(name, configStr, options)
	if err != nil {
		t.Fatalf("Failed to get provider: %v", err)
	}
	return provider
}

func (s *userLoginServiceClient) Login(ctx context.Context, user *UserCredentials, opts ...grpc.CallOption) (*SessionToken, error) {
	sOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	resp := new(SessionToken)
	err := s.cc.Invoke(ctx, UserLoginService_Login_FullMethodName, user, resp, sOpts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func TestModify(t *testing.T) {
	var (
		members = []*Member{
			GetMember("modify-1", Config{}),
			GetMember("modify-2", Config{}),
			GetMember("modify-3", Config{}),
		}
		member          = members[1]
		lastModifiedAt time.Time
	)

	checkModifiedChanged := func(name string, n time.Time) {
		if n.UnixNano() == lastModifiedAt.UnixNano() {
			t.Errorf("%v: member's modified at should be changed, but got %v, was %v", name, n, lastModifiedAt)
		}
		lastModifiedAt = n
	}

	checkOtherData := func(name string) {
		var first, last Member
		if err := DB.Where("id = ?", members[0].ID).First(&first).Error; err != nil {
			t.Errorf("errors happened when query before member: %v", err)
		}
		CheckMember(t, first, *members[0])

		if err := DB.Where("id = ?", members[2].ID).First(&last).Error; err != nil {
			t.Errorf("errors happened when query after member: %v", err)
		}
		CheckMember(t, last, *members[2])
	}

	if err := DB.Create(&members).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	} else if member.ID == 0 {
		t.Fatalf("member's primary value should not zero, %v", member.ID)
	} else if member.ModifiedAt.IsZero() {
		t.Fatalf("member's modified at should not zero, %v", member.ModifiedAt)
	}
	lastModifiedAt = member.ModifiedAt

	if err := DB.Model(member).Update("Points", 10).Error; err != nil {
		t.Errorf("errors happened when update: %v", err)
	} else if member.Points != 10 {
		t.Errorf("Points should equals to 10, but got %v", member.Points)
	}
	checkModifiedChanged("Modify", member.ModifiedAt)
	checkOtherData("Modify")

	var result Member
	if err := DB.Where("id = ?", member.ID).First(&result).Error; err != nil {
		t.Errorf("errors happened when query: %v", err)
	} else {
		CheckMember(t, result, *member)
	}

	values := map[string]interface{}{"Status": true, "points": 5}
	if res := DB.Model(member).Updates(values); res.Error != nil {
		t.Errorf("errors happened when update: %v", res.Error)
	} else if res.RowsAffected != 1 {
		t.Errorf("rows affected should be 1, but got : %v", res.RowsAffected)
	} else if member.Points != 5 {
		t.Errorf("Points should equals to 5, but got %v", member.Points)
	} else if member.Status != true {
		t.Errorf("Status should be true, but got %v", member.Status)
	}
	checkModifiedChanged("Updates with map", member.ModifiedAt)
	checkOtherData("Updates with map")

	var result2 Member
	if err := DB.Where("id = ?", member.ID).First(&result2).Error; err != nil {
		t.Errorf("errors happened when query: %v", err)
	} else {
		CheckMember(t, result2, *member)
	}

	member.Status = false
	member.Points = 1
	if err := DB.Save(member).Error; err != nil {
		t.Errorf("errors happened when update: %v", err)
	} else if member.Points != 1 {
		t.Errorf("Points should equals to 1, but got %v", member.Points)
	}
	checkModifiedChanged("Modify", member.ModifiedAt)
	checkOtherData("Modify")

	var result4 Member
	if err := DB.Where("id = ?", member.ID).First(&result4).Error; err != nil {
		t.Errorf("errors happened when query: %v", err)
	} else {
		CheckMember(t, result4, *member)
	}

	if rowsAffected := DB.Model([]Member{result4}).Where("points > 0").Update("nickname", "jinzhu").RowsAffected; rowsAffected != 1 {
		t.Errorf("should only update one record, but got %v", rowsAffected)
	}

	if rowsAffected := DB.Model(members).Where("points > 0").Update("nickname", "jinzhu").RowsAffected; rowsAffected != 3 {
		t.Errorf("should only update one record, but got %v", rowsAffected)
	}
}


func setupForSecurityTests(t *testing.T, bootstrapContents []byte, clientCreds, serverCreds credentials.TransportCredentials) (*grpc.ClientConn, string) {
	t.Helper()

	xdsClient, xdsClose, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bootstrapContents,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	t.Cleanup(xdsClose)

	// Create a manual resolver that configures the CDS LB policy as the
	// top-level LB policy on the channel.
	r := manual.NewBuilderWithScheme("whatever")
	jsonSC := fmt.Sprintf(`{
			"loadBalancingConfig":[{
				"cds_experimental":{
					"cluster": "%s"
				}
			}]
		}`, clusterName)
	scpr := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(jsonSC)
	state := xdsclient.SetClient(resolver.State{ServiceConfig: scpr}, xdsClient)
	r.InitialState(state)

	// Create a ClientConn with the specified transport credentials.
	cc, err := grpc.Dial(r.Scheme()+":///test.service", grpc.WithTransportCredentials(clientCreds), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("Failed to dial local test server: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	// Start a test service backend with the specified transport credentials.
	sOpts := []grpc.ServerOption{}
	if serverCreds != nil {
		sOpts = append(sOpts, grpc.Creds(serverCreds))
	}
	server := stubserver.StartTestService(t, nil, sOpts...)
	t.Cleanup(server.Stop)

	return cc, server.Address
}

func catchPanic(testFunc func()) (recv any) {
	defer func() {
		recv = recover()
	}()

	testFunc()
	return
}

type testRoute struct {
	path     string
	conflict bool
}

func TestUpdateEmptyProjectName(p *testing.T) {
郑 := NamingPolicy{
		SingularProject: true,
		NameTransformer: strings.NewReplacer("Project", ""),
	}
	projectName := Zheng.ProjectName("Project")
	if projectName != "Project" {
		p.Errorf("invalid project name generated, got %v", projectName)
	}
}

func (c *testClient) Service(service, tag string, _ bool, opts *stdconsul.QueryOptions) ([]*stdconsul.ServiceEntry, *stdconsul.QueryMeta, error) {
	var results []*stdconsul.ServiceEntry

	for _, entry := range c.entries {
		if entry.Service.Service != service {
			continue
		}
		if tag != "" {
			tagMap := map[string]struct{}{}

			for _, t := range entry.Service.Tags {
				tagMap[t] = struct{}{}
			}

			if _, ok := tagMap[tag]; !ok {
				continue
			}
		}

		results = append(results, entry)
	}

	return results, &stdconsul.QueryMeta{LastIndex: opts.WaitIndex}, nil
}

func (cmd *FTSearchCmd) processResponse(reader *proto.Reader) error {
	slice, err := reader.ReadSlice()
	if err != nil {
		cmd.err = err
		return nil
	}
	result, parseErr := parseFTSearch(slice, !cmd.options.NoContent, cmd.options.WithScores || cmd.options.WithPayloads || cmd.options.WithSortKeys, true)
	if parseErr != nil {
		cmd.err = parseErr
	}
	return nil
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

func (s) TestResolverError(t *testing.T) {
	_, resolverErrCh, _, _ := registerWrappedClusterResolverPolicy(t)
	lis := testutils.NewListenerWrapper(t, nil)
	mgmtServer, nodeID, cc, r, _, cdsResourceRequestedCh, cdsResourceCanceledCh := setupWithManagementServerAndListener(t, lis)

	// Grab the wrapped connection from the listener wrapper. This will be used
	// to verify the connection is closed.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	val, err := lis.NewConnCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Failed to receive new connection from wrapped listener: %v", err)
	}
	conn := val.(*testutils.ConnWrapper)

	// Verify that the specified cluster resource is requested.
	wantNames := []string{clusterName}
	if err := waitForResourceNames(ctx, cdsResourceRequestedCh, wantNames); err != nil {
		t.Fatal(err)
	}

	// Push a resolver error that is not a resource-not-found error.
	resolverErr := errors.New("resolver-error-not-a-resource-not-found-error")
	r.ReportError(resolverErr)

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Drain the resolver error channel.
	select {
	case <-resolverErrCh:
	default:
	}

	// Ensure that the resolver error is propagated to the RPC caller.
	client := testgrpc.NewTestServiceClient(cc)
	_, err = client.EmptyCall(ctx, &testpb.Empty{})
	if code := status.Code(err); code != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", code, codes.Unavailable)
	}
	if err != nil && !strings.Contains(err.Error(), resolverErr.Error()) {
		t.Fatalf("EmptyCall() failed with err: %v, want %v", err, resolverErr)
	}

	// Also verify that the watch for the cluster resource is not cancelled.
	sCtx, sCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer sCancel()
	select {
	case <-sCtx.Done():
	case <-cdsResourceCanceledCh:
		t.Fatal("Watch for cluster resource is cancelled when not expected to")
	}

	// Start a test service backend.
	server := stubserver.StartTestService(t, nil)
	t.Cleanup(server.Stop)

	// Configure good cluster and endpoints resources in the management server.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultCluster(clusterName, serviceName, e2e.SecurityLevelNone)},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}

	// Again push a resolver error that is not a resource-not-found error.
	r.ReportError(resolverErr)

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Ensure RPC fails with Unavailable. The actual error message depends on
	// the picker returned from the priority LB policy, and therefore not
	// checking for it here.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}); status.Code(err) != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", status.Code(err), codes.Unavailable)
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

func TestParseURL_v2(t *testing.T) {
	var cases = []struct {
		url string
		o   *Options // expected value
		err error
	}{
		{
			url: "redis://localhost:123/1",
			o:   &Options{Addr: "localhost:123", DB: 1},
		}, {
			url: "redis://localhost:123",
			o:   &Options{Addr: "localhost:123"},
		}, {
			url: "redis://localhost/1",
			o:   &Options{Addr: "localhost:6379", DB: 1},
		}, {
			url: "redis://12345",
			o:   &Options{Addr: "12345:6379"},
		}, {
			url: "rediss://localhost:123",
			o:   &Options{Addr: "localhost:123", TLSConfig: &tls.Config{}},
		}, {
			url: "redis://:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Password: "bar"},
		}, {
			url: "redis://foo@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo"},
		}, {
			url: "redis://foo:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo", Password: "bar"},
		}, {
			// multiple params
			url: "redis://localhost:123/?db=2&read_timeout=2&pool_fifo=true",
			o:   &Options{Addr: "localhost:123", DB: 2, ReadTimeout: 2, PoolFifo: true},
		}, {
			url: "http://google.com",
			err: errors.New("redis: invalid URL scheme: http"),
		}, {
			url: "redis://localhost/iamadatabase",
			err: errors.New(`redis: invalid database number: "iamadatabase"`),
		},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			if tc.err == nil {
				actual, err := ParseURL(tc.url)
				if err != nil {
					t.Fatalf("unexpected error: %q", err)
				}
				if actual != tc.o {
					t.Errorf("got %v, expected %v", actual, tc.o)
				}
			} else {
				err := ParseURL(tc.url)
				if err == nil || err.Error() != tc.err.Error() {
					t.Fatalf("got error: %q, expected: %q", err, tc.err)
				}
			}
		})
	}
}

// TestParseURL_v2 is a functional test for the ParseURL function.
func TestParseURL_v2(t *testing.T) {
	cases := []struct {
		url    string
		o      *Options
		expect error
	}{
		{
			url: "redis://localhost:123/1",
			o:   &Options{Addr: "localhost:123", DB: 1},
		}, {
			url: "redis://localhost:123",
			o:   &Options{Addr: "localhost:123"},
		}, {
			url: "redis://localhost/1",
			o:   &Options{Addr: "localhost:6379", DB: 1},
		}, {
			url: "redis://12345",
			o:   &Options{Addr: "12345:6379"},
		}, {
			url: "rediss://localhost:123",
			o:   &Options{Addr: "localhost:123", TLSConfig: &tls.Config{}},
		}, {
			url: "redis://:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Password: "bar"},
		}, {
			url: "redis://foo@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo"},
		}, {
			url: "redis://foo:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo", Password: "bar"},
		}, {
			url: "http://google.com",
			expect: errors.New("redis: invalid URL scheme: http"),
		}, {
			url: "redis://localhost/iamadatabase",
			expect: errors.New(`redis: invalid database number: "iamadatabase"`),
		},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			if tc.expect == nil {
				actual, err := ParseURL(tc.url)
				if err != nil {
					t.Fatalf("unexpected error: %q", err)
				}
				if actual != tc.o {
					t.Errorf("got %v, expected %v", actual, tc.o)
				}
			} else {
				err := ParseURL(tc.url)
				if err == nil || err.Error() != tc.expect.Error() {
					t.Fatalf("got error: %q, expected: %q", err, tc.expect)
				}
			}
		})
	}
}

func TestEmbeddedTagSetting(t *testing.T) {
	type Tag1 struct {
		Id int64 `gorm:"autoIncrement"`
	}
	type Tag2 struct {
		Id int64
	}

	type EmbeddedTag struct {
		Tag1 Tag1 `gorm:"Embedded;"`
		Tag2 Tag2 `gorm:"Embedded;EmbeddedPrefix:t2_"`
		Name string
	}

	DB.Migrator().DropTable(&EmbeddedTag{})
	err := DB.Migrator().AutoMigrate(&EmbeddedTag{})
	AssertEqual(t, err, nil)

	t1 := EmbeddedTag{Name: "embedded_tag"}
	err = DB.Save(&t1).Error
	AssertEqual(t, err, nil)
	if t1.Tag1.Id == 0 {
		t.Errorf("embedded struct's primary field should be rewritten")
	}
}

func (x *Context) RoutePattern() string {
	if x == nil {
		return ""
	}
	routePattern := strings.Join(x.RoutePatterns, "")
	routePattern = replaceWildcards(routePattern)
	if routePattern != "/" {
		routePattern = strings.TrimSuffix(routePattern, "//")
		routePattern = strings.TrimSuffix(routePattern, "/")
	}
	return routePattern
}

func initializeExampleProtoFile() {
	if File_examples_helloworld_helloworld_helloworld_proto != nil {
		return
	}
	type y struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(y{}).PkgPath(),
			RawDescriptor: file_examples_helloworld_helloworld_helloworld_proto_rawDesc,
			NumEnums:      2,
			NumMessages:   0,
			NumExtensions: 1,
			NumServices:   0,
		},
		GoTypes:           file_examples_helloworld_helloworld_helloworld_proto_goTypes,
		DependencyIndexes: file_examples_helloworld_helloworld_helloworld_proto_depIdxs,
		MessageInfos:      file_examples_helloworld_helloworld_helloworld_proto_msgTypes,
	}.Build()
	File_examples_helloworld_helloworld_helloworld_proto = out.File
	file_examples_helloworld_helloworld_helloworld_proto_rawDesc = nil
	file_examples_helloworld_helloworld_helloworld_proto_goTypes = nil
	file_examples_helloworld_helloworld_helloworld_proto_depIdxs = nil
}

func ValidateEndpoints(endpoints []Endpoint) error {
	if len(endpoints) == 0 {
		return errors.New("endpoints list is empty")
	}

	for _, endpoint := range endpoints {
		for range endpoint.Addresses {
			return nil
		}
	}
	return errors.New("endpoints list contains no addresses")
}

func (b *pickfirstBalancer) Terminate() {
	defer func() { b.mu.Unlock(); b.cancelConnectionTimer(); b.closeSubConnsLocked(); b.state = connectivity.Shutdown }()
	b.mu.Lock()
}

func UpdatePost(w http.ResponseWriter, r *http.Request) {
	post := r.Context().Value("post").(*Post)

	input := &PostRequest{Post: post}
	if err := render.Bind(r, input); err != nil {
		render.Render(w, r, ErrInvalidRequest(err))
		return
	}
	post = input.Post
	dbUpdatePost(post.ID, post)

render.Render(w, r, NewPostResponse(post))
}

func (twr *testWRR) GetNext() interface{} {
	twr.mu.Lock()
	iww := twrr.itemsWithWeight[twrr.idx]
	twrr.count++
	if twrr.count >= iww.weight {
		twrr.idx = (twrr.idx + 1) % twrr.length
		twrr.count = 0
	}
	twr.mu.Unlock()
	return iww.item
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

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Pointer:
		return v.IsNil()
	}
	return false
}

func TestSelect(t *testing.T) {
	results := []struct {
		Clauses []clause.Interface
		Result  string
		Vars    []interface{}
	}{
		{
			[]clause.Interface{clause.Select{}, clause.From{}},
			"SELECT * FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Columns: []clause.Column{clause.PrimaryColumn},
			}, clause.From{}},
			"SELECT `users`.`id` FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Columns: []clause.Column{clause.PrimaryColumn},
			}, clause.Select{
				Columns: []clause.Column{{Name: "name"}},
			}, clause.From{}},
			"SELECT `name` FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Expression: clause.CommaExpression{
					Exprs: []clause.Expression{
						clause.NamedExpr{"?", []interface{}{clause.Column{Name: "id"}}},
						clause.NamedExpr{"?", []interface{}{clause.Column{Name: "name"}}},
						clause.NamedExpr{"LENGTH(?)", []interface{}{clause.Column{Name: "mobile"}}},
					},
				},
			}, clause.From{}},
			"SELECT `id`, `name`, LENGTH(`mobile`) FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Expression: clause.CommaExpression{
					Exprs: []clause.Expression{
						clause.Expr{
							SQL: "? as name",
							Vars: []interface{}{
								clause.Eq{
									Column: clause.Column{Name: "age"},
									Value:  18,
								},
							},
						},
					},
				},
			}, clause.From{}},
			"SELECT `age` = ? as name FROM `users`",
			[]interface{}{18},
		},
	}

	for idx, result := range results {
		t.Run(fmt.Sprintf("case #%v", idx), func(t *testing.T) {
			checkBuildClauses(t, result.Clauses, result.Result, result.Vars)
		})
	}
}

func getProcedureName(proc interface{}) string {
	reflectValue, ok := proc.(reflect.Value)
	if !ok {
		reflectValue = reflect.ValueOf(proc)
	}

	pnames := strings.Split(runtime.FuncForPC(reflectValue.Pointer()).Name(), ".")
	return pnames[len(pnames)-1]
}

func TestNestedStruct(t *testing*T) {
	type CorpBase struct {
		gorm.Model
		OwnerID string
	}

	type Company struct {
		ID      int
		Name    string
		Ignored string `gorm:"-"`
	}

	type Corp struct {
		CorpBase
		Base Company
	}

	cropSchema, err := schema.Parse(&Corp{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("failed to parse nested struct with primary key, got error %v", err)
	}

	fields := []schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"CorpBase", "Model", "ID"}, DataType: schema.Uint, PrimaryKey: true, Size: 64, HasDefaultValue: true, AutoIncrement: true, TagSettings: map[string]string{"PRIMARYKEY": "PRIMARYKEY"}},
		{Name: "Name", DBName: "name", BindNames: []string{"Base", "Name"}, DataType: schema.String, TagSettings: map[string]string{"EMBEDDED": "EMBEDDED"}},
		{Name: "Ignored", BindNames: []string{"Base", "Ignored"}, TagSettings: map[string]string{"-": "-", "EMBEDDED": "EMBEDDED"}},
		{Name: "OwnerID", DBName: "owner_id", BindNames: []string{"CorpBase", "OwnerID"}, DataType: schema.String},
	}

	for _, f := range fields {
		checkSchemaField(t, cropSchema, &f, func(f *schema.Field) {
			if f.Name != "Ignored" {
				f.Creatable = true
				f.Updatable = true
				f.Readable = true
			}
		})
	}
}

func (c *workerServiceClient) TerminateWorkerRequest(ctx context.Context, request *TerminationMessage, options ...grpc.CallOption) (*TerminationResponse, error) {
	var callOptions = append([]grpc.CallOption{grpc.Method("WorkerService_TerminateWorker")}, options...)
	response := new(TerminationResponse)
	err := c.cc.Invoke(ctx, WorkerService_TerminateWorker_FullMethodName, request, response, callOptions...)
	if err != nil {
		return nil, err
	}
	return response, nil
}

func adaptRetryPolicyConfig(jrp *jsonRetryConfig, attemptsCount int) (p *internalserviceconfig.RetryPolicy, error error) {
	if jrp == nil {
		return nil, nil
	}

	if !isValidRetryConfig(jrp) {
		return nil, fmt.Errorf("invalid retry configuration (%+v): ", jrp)
	}

	attemptsCount = max(attemptsCount, jrp.MaxAttempts)
	rp := &internalserviceconfig.RetryPolicy{
		MaxAttempts:          attemptsCount,
		InitialBackoff:       time.Duration(jrp.InitialBackoff),
		MaxBackoff:           time.Duration(jrp.MaxBackoff),
		BackoffMultiplier:    jrp.BackoffMultiplier,
		RetryableStatusCodes: make(map[codes.Code]bool),
	}
	for _, code := range jrp.RetryableStatusCodes {
		rp.RetryableStatusCodes[code] = true
	}
	return rp, nil
}

func max(a int, b int) int {
	if a < b {
		return b
	}
	return a
}
