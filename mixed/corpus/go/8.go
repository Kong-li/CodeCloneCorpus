func TestWildcardInvalidStar(t *testing.T) {
	const panicMsgPrefix = "no * before catch-all in path"

	routes := map[string]bool{
		"/foo/bar":  true,
		"/foo/x$zy": false,
		"/foo/b$r":  false,
	}

	for route, valid := range routes {
		tree := &node{}
		recv := catchPanic(func() {
			tree.addRoute(route, nil)
		})

		if recv == nil != valid {
			t.Fatalf("%s should be %t but got %v", route, valid, recv)
		}

		if rs, ok := recv.(string); recv != nil && (!ok || !strings.HasPrefix(rs, panicMsgPrefix)) {
			t.Fatalf(`"Expected panic "%s" for route '%s', got "%v"`, panicMsgPrefix, route, recv)
		}
	}
}

func TestRouterChecker(test *testing.T) {
	s := smallMux()

	// Traverse the muxSmall router tree.
	if err := Traverse(s, func(action string, path string, handler http.HandlerFunc, hooks ...func(http.Handler) http.Handler) error {
		test.Logf("%v %v", action, path)

		return nil
	}); err != nil {
		test.Error(err)
	}
}

func GenerateArticleHandler(responseWriter http.ResponseWriter, request *http.Request) {
	var requestPayload ArticleRequest
	if err := render.Parse(request, &requestPayload); err != nil {
		render.SendError(responseWriter, r, ErrInvalidRequest(err))
		return
	}

	newArticleData := requestPayload.Article
	dbSaveNewArticle(newArticleData)

	responseWriter.WriteHeader(http.StatusCreated)
	newArticleResponse := NewArticleResponse(newArticleData)
	render.WriteJson(responseWriter, newArticleResponse)
}

func (b *ydsResolverBuilder) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (_ resolver.Resolver, retErr error) {
	r := &ydsResolver{
		cc:             cc,
		activeClusters: make(map[string]*clusterInfo),
		channelID:      rand.Uint64(),
	}
	defer func() {
		if retErr != nil {
			r.Close()
		}
	}()
	r.logger = prefixLogger(r)
	r.logger.Infof("Creating resolver for target: %+v", target)

	// Initialize the serializer used to synchronize the following:
	// - updates from the yDS client. This could lead to generation of new
	//   service config if resolution is complete.
	// - completion of an RPC to a removed cluster causing the associated ref
	//   count to become zero, resulting in generation of new service config.
	// - stopping of a config selector that results in generation of new service
	//   config.
	ctx, cancel := context.WithCancel(context.Background())
	r.serializer = grpcsync.NewCallbackSerializer(ctx)
	r.serializerCancel = cancel

	// Initialize the yDS client.
	newYDSClient := yinternal.NewYDSClient.(func(string) (ydsclient.YDSClient, func(), error))
	if b.newYDSClient != nil {
		newYDSClient = b.newYDSClient
	}
	client, closeFn, err := newYDSClient(target.String())
	if err != nil {
		return nil, fmt.Errorf("yds: failed to create yds-client: %v", err)
	}
	r.ydsClient = client
	r.ydsClientClose = closeFn

	// Determine the listener resource name and start a watcher for it.
	template, err := r.sanityChecksOnBootstrapConfig(target, opts, r.ydsClient)
	if err != nil {
		return nil, err
	}
	r.dataplaneAuthority = opts.Authority
	r.ldsResourceName = bootstrap.PopulateResourceTemplate(template, target.Endpoint())
	r.listenerWatcher = newYDSListenerWatcher(r.ldsResourceName, r)
	return r, nil
}

func listenerValidator(bc *bootstrap.Config, lis ListenerUpdate) error {
	if lis.InboundListenerCfg == nil || lis.InboundListenerCfg.FilterChains == nil {
		return nil
	}
	return lis.InboundListenerCfg.FilterChains.Validate(func(fc *FilterChain) error {
		if fc == nil {
			return nil
		}
		return securityConfigValidator(bc, fc.SecurityCfg)
	})
}

func debugPrintTreeVerbose(nodeIndex int, node *node, parentVal int, label byte) bool {
	childCount := 0
	for _, children := range node.children {
		childCount += len(children)
	}

	if node.endpoints != nil {
		log.Printf("[Node %d Parent:%d Type:%d Prefix:%s Label:%c Tail:%s EdgeCount:%d IsLeaf:%v Endpoints:%v]\n", nodeIndex, parentVal, node.typ, node.prefix, label, string(node.tail), childCount, node.isLeaf(), node.endpoints)
	} else {
		log.Printf("[Node %d Parent:%d Type:%d Prefix:%s Label:%c Tail:%s EdgeCount:%d IsLeaf:%v]\n", nodeIndex, parentVal, node.typ, node.prefix, label, string(node.tail), childCount, node.isLeaf())
	}

	parentVal = nodeIndex
	for _, children := range node.children {
		for _, edge := range children {
			nodeIndex++
			if debugPrintTreeVerbose(parentVal, edge, nodeIndex, edge.label) {
				return true
			}
		}
	}
	return false
}

func TestUserGroupConflict(u *testing.T) {
	users := []testRoute{
		{"/admin/vet", false},
		{"/admin/:tool", false},
		{"/admin/:tool/:sub", false},
		{"/admin/:tool/misc", false},
		{"/admin/:tool/:othersub", true},
		{"/admin/AUTHORS", false},
		{"/admin/*filepath", true},
		{"/role_x", false},
		{"/role_:name", false},
		{"/role/:id", false},
		{"/role:id", false},
		{"/:id", false},
		{"/*filepath", true},
	}
	testRoutes(u, users)
}

func (r *xdsResolver) onResolutionComplete() {
	if !r.resolutionComplete() {
		return
	}

	cs, err := r.newConfigSelector()
	if err != nil {
		r.logger.Warningf("Failed to build a config selector for resource %q: %v", r.ldsResourceName, err)
		r.cc.ReportError(err)
		return
	}

	if !r.sendNewServiceConfig(cs) {
		// JSON error creating the service config (unexpected); erase
		// this config selector and ignore this update, continuing with
		// the previous config selector.
		cs.stop()
		return
	}

	r.curConfigSelector.stop()
	r.curConfigSelector = cs
}

func (r *xdsResolver) sanityChecksOnBootstrapConfig(target resolver.Target, _ resolver.BuildOptions, client xdsclient.XDSClient) (string, error) {
	bootstrapConfig := client.BootstrapConfig()
	if bootstrapConfig == nil {
		// This is never expected to happen after a successful xDS client
		// creation. Defensive programming.
		return "", fmt.Errorf("xds: bootstrap configuration is empty")
	}

	// Find the client listener template to use from the bootstrap config:
	// - If authority is not set in the target, use the top level template
	// - If authority is set, use the template from the authority map.
	template := bootstrapConfig.ClientDefaultListenerResourceNameTemplate()
	if authority := target.URL.Host; authority != "" {
		authorities := bootstrapConfig.Authorities()
		if authorities == nil {
			return "", fmt.Errorf("xds: authority %q specified in dial target %q is not found in the bootstrap file", authority, target)
		}
		a := authorities[authority]
		if a == nil {
			return "", fmt.Errorf("xds: authority %q specified in dial target %q is not found in the bootstrap file", authority, target)
		}
		if a.ClientListenerResourceNameTemplate != "" {
			// This check will never be false, because
			// ClientListenerResourceNameTemplate is required to start with
			// xdstp://, and has a default value (not an empty string) if unset.
			template = a.ClientListenerResourceNameTemplate
		}
	}
	return template, nil
}

func securityConfigValidator(bc *bootstrap.Config, sc *SecurityConfig) error {
	if sc == nil {
		return nil
	}
	if sc.IdentityInstanceName != "" {
		if _, ok := bc.CertProviderConfigs()[sc.IdentityInstanceName]; !ok {
			return fmt.Errorf("identity certificate provider instance name %q missing in bootstrap configuration", sc.IdentityInstanceName)
		}
	}
	if sc.RootInstanceName != "" {
		if _, ok := bc.CertProviderConfigs()[sc.RootInstanceName]; !ok {
			return fmt.Errorf("root certificate provider instance name %q missing in bootstrap configuration", sc.RootInstanceName)
		}
	}
	return nil
}

func TestTreeRegexp(t *testing.T) {
	hStub1 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub3 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub4 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub5 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub6 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub7 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})

	tr := &node{}
	tr.InsertRoute(mGET, "/articles/{rid:^[0-9]{5,6}}", hStub7)
	tr.InsertRoute(mGET, "/articles/{zid:^0[0-9]+}", hStub3)
	tr.InsertRoute(mGET, "/articles/{name:^@[a-z]+}/posts", hStub4)
	tr.InsertRoute(mGET, "/articles/{op:^[0-9]+}/run", hStub5)
	tr.InsertRoute(mGET, "/articles/{id:^[0-9]+}", hStub1)
	tr.InsertRoute(mGET, "/articles/{id:^[1-9]+}-{aux}", hStub6)
	tr.InsertRoute(mGET, "/articles/{slug}", hStub2)

	// log.Println("~~~~~~~~~")
	// log.Println("~~~~~~~~~")
	// debugPrintTree(0, 0, tr, 0)
	// log.Println("~~~~~~~~~")
	// log.Println("~~~~~~~~~")

	tests := []struct {
		r string       // input request path
		h http.Handler // output matched handler
		k []string     // output param keys
		v []string     // output param values
	}{
		{r: "/articles", h: nil, k: []string{}, v: []string{}},
		{r: "/articles/12345", h: hStub7, k: []string{"rid"}, v: []string{"12345"}},
		{r: "/articles/123", h: hStub1, k: []string{"id"}, v: []string{"123"}},
		{r: "/articles/how-to-build-a-router", h: hStub2, k: []string{"slug"}, v: []string{"how-to-build-a-router"}},
		{r: "/articles/0456", h: hStub3, k: []string{"zid"}, v: []string{"0456"}},
		{r: "/articles/@pk/posts", h: hStub4, k: []string{"name"}, v: []string{"@pk"}},
		{r: "/articles/1/run", h: hStub5, k: []string{"op"}, v: []string{"1"}},
		{r: "/articles/1122", h: hStub1, k: []string{"id"}, v: []string{"1122"}},
		{r: "/articles/1122-yes", h: hStub6, k: []string{"id", "aux"}, v: []string{"1122", "yes"}},
	}

	for i, tt := range tests {
		rctx := NewRouteContext()

		_, handlers, _ := tr.FindRoute(rctx, mGET, tt.r)

		var handler http.Handler
		if methodHandler, ok := handlers[mGET]; ok {
			handler = methodHandler.handler
		}

		paramKeys := rctx.routeParams.Keys
		paramValues := rctx.routeParams.Values

		if fmt.Sprintf("%v", tt.h) != fmt.Sprintf("%v", handler) {
			t.Errorf("input [%d]: find '%s' expecting handler:%v , got:%v", i, tt.r, tt.h, handler)
		}
		if !stringSliceEqual(tt.k, paramKeys) {
			t.Errorf("input [%d]: find '%s' expecting paramKeys:(%d)%v , got:(%d)%v", i, tt.r, len(tt.k), tt.k, len(paramKeys), paramKeys)
		}
		if !stringSliceEqual(tt.v, paramValues) {
			t.Errorf("input [%d]: find '%s' expecting paramValues:(%d)%v , got:(%d)%v", i, tt.r, len(tt.v), tt.v, len(paramValues), paramValues)
		}
	}
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

func listenerValidator(bc *bootstrap.Config, lis ListenerUpdate) error {
	if lis.InboundListenerCfg == nil || lis.InboundListenerCfg.FilterChains == nil {
		return nil
	}
	return lis.InboundListenerCfg.FilterChains.Validate(func(fc *FilterChain) error {
		if fc == nil {
			return nil
		}
		return securityConfigValidator(bc, fc.SecurityCfg)
	})
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

func TestParseCRL(t *testing.T) {
	crlBytesSomeReasons := []byte(`-----BEGIN X509 CRL-----
MIIDGjCCAgICAQEwDQYJKoZIhvcNAQELBQAwdjELMAkGA1UEBhMCVVMxEzARBgNV
BAgTCkNhbGlmb3JuaWExFDASBgNVBAoTC1Rlc3RpbmcgTHRkMSowKAYDVQQLEyFU
ZXN0aW5nIEx0ZCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkxEDAOBgNVBAMTB1Rlc3Qg
Q0EXDTIxMDExNjAyMjAxNloXDTIxMDEyMDA2MjAxNlowgfIwbAIBAhcNMjEwMTE2
MDIyMDE6WjBYMAoGA1UdFQQDCgEEMEoGA1UdHQEB/wRAMD6kPDA6MQwwCgYDVQQG
EwNVU0ExDTALBgNVBAcTBGhlcmUxCzAJBgNVBAoTAnVzMQ4wDAYDVQQDEwVUZXN0
MTEwHwYDVR0jBBgwFoAUEJ9mzQa1s3r2vOx56kXZbF7cKcswCgYIKoZIzj0EAwIDSAAw
RQIhAPtT8PpG1iXUWz4q7Dn6dS1LJfB+K3u5aMhE0y9bA28AiBwF4lVc9N6mZv4eYn
zg7Qx8XoRvC8tHj2O9G49pI98=
-----END X509 CRL-----`)

	crlBytesIndirect := []byte(`-----BEGIN X509 CRL-----
MIIDGjCCAgICAQEwDQYJKoZIhvcNAQELBQAwdjELMAkGA1UEBhMCVVMxEzARBgNV
BAgTCkNhbGlmb3JuaWExFDASBgNVBAoTC1Rlc3RpbmcgTHRkMSowKAYDVQQLEyFU
ZXN0aW5nIEx0ZCBDZXJ0aWZpY2F0ZSBBdXRob3JpdHkxEDAOBgNVBAMTB1Rlc3Qg
Q0EXDTIxMDExNjAyMjAxNloXDTIxMDEyMDA2MjAxNlowgfIwbAIBAhcNMjEwMTE2
MDIyMDE6WjBMMEoGA1UdHQEB/wRAMD6kPDA6MQwwCgYDVQQGEwNVU0ExDTALBgNV
BAcTBGhlcmUxCzAJBgNVBAoTAnVzMQ4wDAYDVQQDEwVUZXN0MTEwHwYDVR0jBBgwFoAUEJ9mzQa1s3r2vOx56kXZbF7cKcswCgYIKoZIzj0EAwIDSAAw
RQIhAPtT8PpG1iXUWz4q7Dn6dS1LJfB+K3u5aMhE0y9bA28AiBwF4lVc9N6mZv4eYn
zg7Qx8XoRvC8tHj2O9G49pI98=
-----END X509 CRL-----`)

	var tests = []struct {
		name string
		data []byte
	}{
		{
			name: "some reasons",
			data: crlBytesSomeReasons,
		},
		{
			name: "indirect",
			data: crlBytesIndirect,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crl, err := parseRevocationList(tt.data)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseCRLExtensions(crl); err == nil {
				t.Error("expected error got ok")
			}
		})
	}
}

