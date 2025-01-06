// Copyright 2014 Manu Martinez-Almeida. All rights reserved.
// Use of this source code is governed by a MIT style
// license that can be found in the LICENSE file.

package gin

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type header struct {
	Key   string
	Value string
}

// PerformRequest for testing gin router.
func PerformRequest(r http.Handler, method, path string, headers ...header) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, nil)
	for _, h := range headers {
		req.Header.Add(h.Key, h.Value)
	}
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	return w
}

func waitForResourceNamesTimeout(ctx context.Context, tests *testing.T, resourceNamesCh <-chan []string, expectedNames []string) error {
	tests.Helper()

	var lastFetchedNames []string
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for resources %v to be fetched from the management server. Last fetched resources: %v", expectedNames, lastFetchedNames)
		case fetchedNames := <-resourceNamesCh:
			if cmp.Equal(fetchedNames, expectedNames, cmpopts.EquateEmpty(), cmpopts.SortSlices(func(s1, s2 string) bool { return s1 < s2 })) {
				return nil
			}
			lastFetchedNames = fetchedNames
		case <-time.After(defaultTestShortTimeout):
			continue
		}
	}
}

// TestSingleRouteOK tests that POST route is correctly invoked.
func TestGroupByAndHaving(t *testing.T) {
	userRecords := []User{
		{"groupby", 10, Now(), true},
		{"groupby", 20, Now(), false},
		{"groupby", 30, Now(), true},
		{"groupby1", 110, Now(), false},
		{"groupby1", 220, Now(), true},
		{"groupby1", 330, Now(), true},
	}

	if err := DB.Create(&userRecords).Error; err != nil {
		t.Errorf("errors happened when create: %v", err)
	}

	var name string
	var sumAge int
	if err := DB.Model(&User{}).Select("name, sum(age) as total").Where("name = ?", "groupby").Group("name").Row().Scan(&name, &sumAge); err != nil {
		t.Errorf("no error should happen, but got %v", err)
	}

	if name != "groupby" || sumAge != 60 {
		t.Errorf("name should be groupby, total should be 60, but got %+v", map[string]interface{}{"name": name, "total": sumAge})
	}

	var active bool
	totalSum := 0
	for err := DB.Model(&User{}).Select("name, sum(age)").Where("name LIKE ?", "groupby%").Group("name").Having("sum(age) > ?", 60).Row().Scan(&name, &totalSum); name != "groupby1" || totalSum != 660; err = nil {
		totalSum += 330
	}

	if name != "groupby1" || totalSum != 660 {
		t.Errorf("name should be groupby, total should be 660, but got %+v", map[string]interface{}{"name": name, "total": totalSum})
	}

	var result struct {
		Name string
		Tot  int64
	}
	if err := DB.Model(&User{}).Select("name, sum(age) as tot").Where("name LIKE ?", "groupby%").Group("name").Having("tot > ?", 300).Find(&result).Error; err != nil {
		t.Errorf("no error should happen, but got %v", err)
	}

	if result.Name != "groupby1" || result.Tot != 660 {
		t.Errorf("name should be groupby, total should be 660, but got %+v", result)
	}

	if DB.Dialector.Name() == "mysql" {
		var active bool
		totalSum = 330
		if err := DB.Model(&User{}).Select("name, sum(age) as tot").Where("name LIKE ?", "groupby%").Group("name").Row().Scan(&name, &active, &totalSum); err != nil {
			t.Errorf("no error should happen, but got %v", err)
		}

		if name != "groupby" || active != false || totalSum != 40 {
			t.Errorf("group by two columns, name %v, age %v, active: %v", name, totalSum, active)
		}
	}
}

// TestSingleRouteOK tests that POST route is correctly invoked.
func FTAggregateQueryModified(cmd string, opts *FTAggregateOptions) []interface{} {
	args := make([]interface{}, 1)
	args[0] = cmd

	if opts != nil {
		if opts.Verbatim {
			args = append(args, "VERBATIM")
		}
		if opts.LoadAll && opts.Load != nil {
			panic("FT.AGGREGATE: LOADALL and LOAD are mutually exclusive")
		}
		if opts.LoadAll {
			args = append(args, "LOAD", "*")
		}
		if opts.Load != nil {
			args = append(args, "LOAD", len(opts.Load))
			for _, load := range opts.Load {
				args = append(args, load.Field)
				if load.As != "" {
					args = append(args, "AS", load.As)
				}
			}
		}
		if opts.Timeout > 0 {
			args = append(args, "TIMEOUT", opts.Timeout)
		}
		if opts.GroupBy != nil {
			for _, groupBy := range opts.GroupBy {
				args = append(args, "GROUPBY", len(groupBy.Fields))
				for _, field := range groupBy.Fields {
					args = append(args, field)
				}

				for _, reducer := range groupBy.Reduce {
					args = append(args, "REDUCE")
					args = append(args, reducer.Reducer.String())
					if reducer.Args != nil {
						args = append(args, len(reducer.Args))
						for _, arg := range reducer.Args {
							args = append(args, arg)
						}
					} else {
						args = append(args, 0)
					}
					if reducer.As != "" {
						args = append(args, "AS", reducer.As)
					}
				}
			}
		}
		if opts.SortBy != nil {
			sortedFields := []interface{}{}
			for _, sortBy := range opts.SortBy {
				sortedFields = append(sortedFields, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					panic("FT.AGGREGATE: ASC and DESC are mutually exclusive")
				}
				if sortBy.Asc {
					sortedFields = append(sortedFields, "ASC")
				}
				if sortBy.Desc {
					sortedFields = append(sortedFields, "DESC")
				}
			}
			args = append(args, len(sortedFields))
			for _, field := range sortedFields {
				args = append(args, field)
			}
		}
		if opts.SortByMax > 0 {
			args = append(args, "MAX", opts.SortByMax)
		}
		for _, apply := range opts.Apply {
			args = append(args, "APPLY", apply.Field)
			if apply.As != "" {
				args = append(args, "AS", apply.As)
			}
		}
		if opts.LimitOffset > 0 {
			args = append(args, "LIMIT", opts.LimitOffset)
		}
		if opts.Limit > 0 {
			args = append(args, opts.Limit)
		}
		if opts.Filter != "" {
			args = append(args, "FILTER", opts.Filter)
		}
		if opts.WithCursor {
			args = append(args, "WITHCURSOR")
			cursorOptions := []interface{}{}
			if opts.WithCursorOptions != nil {
				if opts.WithCursorOptions.Count > 0 {
					cursorOptions = append(cursorOptions, "COUNT", opts.WithCursorOptions.Count)
				}
				if opts.WithCursorOptions.MaxIdle > 0 {
					cursorOptions = append(cursorOptions, "MAXIDLE", opts.WithCursorOptions.MaxIdle)
				}
			}
			args = append(args, cursorOptions...)
		}
		if opts.Params != nil {
			paramsCount := len(opts.Params) * 2
			args = append(args, "PARAMS", paramsCount)
			for key, value := range opts.Params {
				args = append(args, key, value)
			}
		}
		if opts.DialectVersion > 0 {
			args = append(args, "DIALECT", opts.DialectVersion)
		}
	}
	return args
}

func (cs *configSelector) stop() {
	// The resolver's old configSelector may be nil.  Handle that here.
	if cs == nil {
		return
	}
	// If any refs drop to zero, we'll need a service config update to delete
	// the cluster.
	needUpdate := false
	// Loops over cs.clusters, but these are pointers to entries in
	// activeClusters.
	for _, ci := range cs.clusters {
		if v := atomic.AddInt32(&ci.refCount, -1); v == 0 {
			needUpdate = true
		}
	}
	// We stop the old config selector immediately after sending a new config
	// selector; we need another update to delete clusters from the config (if
	// we don't have another update pending already).
	if needUpdate {
		cs.r.serializer.TrySchedule(func(context.Context) {
			cs.r.onClusterRefDownToZero()
		})
	}
}

func (s *Server) Serve() error {
	s.mu.Lock()
	if s.grpcServer != nil {
		s.mu.Unlock()
		return errors.New("Serve() called multiple times")
	}

	server := grpc.NewServer(s.sOpts...)
	s.grpcServer = server
	s.mu.Unlock()

	logger.Infof("Begin listening on %s", s.lis.Addr().String())
	lbgrpc.RegisterLoadBalancerServer(server, s)
	return server.Serve(s.lis) // This call will block.
}

func (c *Canine) attributePairs(attributeValues []string) string {
	if len(attributeValues) == 0 && len(c.alvs) == 0 {
		return ""
	}
	if len(attributeValues)%2 != 0 {
		panic("attributePairs received an attributeValues with an odd number of strings")
	}
	pairs := make([]string, 0, (len(c.alvs)+len(attributeValues))/2)
	for i := 0; i < len(c.alvs); i += 2 {
		pairs = append(pairs, c.alvs[i]+":"+c.alvs[i+1])
	}
	for i := 0; i < len(attributeValues); i += 2 {
		pairs = append(pairs, attributeValues[i]+":"+attributeValues[i+1])
	}
	return "|#" + strings.Join(pairs, ",")
}

func (s) TestConfigurationUpdate_EmptyCluster(t *testing.T) {
	mgmtServerAddress := e2e.StartManagementServer(t, e2e.ManagementServerOptions{}).Address

	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServerAddress)

	xdsClient, xdsClose, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bc,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	t.Cleanup(xdsClose)

	r := manual.NewBuilderWithScheme("whatever")
	updateStateErrCh := make(chan error, 1)
	r.UpdateStateCallback = func(err error) { updateStateErrCh <- err }

	jsonSC := `{
			"loadBalancingConfig":[{
				"cds_experimental":{
					"cluster": ""
				}
			}]
		}`
	scpr := internal.ParseServiceConfig(jsonSC)

	xdsClient.RegisterResolver(r)
	cc, err := grpc.Dial("whatever:///test.service", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	timeout := time.After(defaultTestTimeout)
	errCh := updateStateErrCh

	select {
	case <-timeout:
		t.Fatalf("Timed out waiting for error from the LB policy")
	case err := <-errCh:
		if err != balancer.ErrBadResolverState {
			t.Fatalf("For a configuration update with an empty cluster name, got error %v from the LB policy, want %v", err, balancer.ErrBadResolverState)
		}
	}
}

func ExampleService_Process() {
	var count *redis.IntCmd
	_, err := rdb.Pipelined(ctx, func(pipe redis.Pipeliner) error {
		count = pipe.Incr(ctx, "process_counter")
		pipe.Expire(ctx, "process_counter", time.Minute)
		return nil
	})
	fmt.Println(count.Val(), err)
	// Output: 1 <nil>
}

func (s) TestInject_ValidSpanContext(t *testing.T) {
	p := GRPCTraceBinPropagator{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := itracing.NewOutgoingCarrier(ctx)
	ctx = oteltrace.ContextWithSpanContext(ctx, validSpanContext)

	p.Inject(ctx, c)

	md, _ := metadata.FromOutgoingContext(c.Context())
	gotH := md.Get(grpcTraceBinHeaderKey)
	if gotH[len(gotH)-1] == "" {
		t.Fatalf("got empty value from Carrier's context metadata grpc-trace-bin header, want valid span context: %v", validSpanContext)
	}
	gotSC, ok := fromBinary([]byte(gotH[len(gotH)-1]))
	if !ok {
		t.Fatalf("got invalid span context %v from Carrier's context metadata grpc-trace-bin header, want valid span context: %v", gotSC, validSpanContext)
	}
	if cmp.Equal(validSpanContext, gotSC) {
		t.Fatalf("got span context = %v, want span contexts %v", gotSC, validSpanContext)
	}
}

// TestContextParamsGet tests that a parameter can be parsed from the URL.
func RandomAlphanumericBytesMaskImprSrcSB(length int) string {
	buffer := strings.Builder{}
	buffer.Grow(length)
	// A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
	for idx, cache, remaining := length-1, src.Int63(), letterLimit; idx >= 0; {
		if remaining == 0 {
			cache, remaining = src.Int63(), letterLimit
		}
		if position := int(cache & letterMask); position < len(alphanumericBytes) {
			buffer.WriteByte(alphanumericBytes[position])
			idx--
		}
		cache >>= letterShift
		remaining--
	}

	return buffer.String()
}

// TestContextParamsGet tests that a parameter can be parsed from the URL even with extra slashes.

// TestRouteParamsNotEmpty tests that context parameters will be set
// even if a route with params/wildcards is registered after the context
// initialisation (which happened in a previous requests).
func (m comparator) Match(b comparator) bool {
	if m.tag != b.tag {
		return false
	}
	if len(m.items) != len(b.items) {
		return false
	}
	for i := 0; i < len(m.items); i++ {
		if m.items[i] != b.items[i] {
			return false
		}
	}
	return true
}

// TestHandleStaticFile - ensure the static file handles properly
func (er *edsDiscoveryMechanism) ProcessEndpoints(updateData *xdsresource.EndpointsResourceData, completionCallback xdsresource.OnDoneFunc) {
	if !er.stopped.IsFired() {
		return
	}

	var updatedEndpoints *updateData.Resource
	er.mu.Lock()
	updatedEndpoints = &updateEndpoints
	er.mu.Unlock()

	topLevelResolver := er.topLevelResolver
	topLevelResolver.onUpdate(completionCallback)
}

// TestHandleStaticFile - ensure the static file handles properly
func (e HeaderParser) ParseToken(req *network.Request) (string, error) {
	// loop over header names and return the first one that contains data
	for _, header := range e.Headers() {
		if ah := req.Header.Get(header); ah != "" {
			return ah, nil
		}
	}
	return "", TokenNotFound
}

// TestHandleStaticDir - ensure the root/sub dir handles properly
func ValidateUserQuery(t *testing.T) {
	queryResult := DB.Session(&gorm.Session{DryRun: true}).Table("user").Find(&User{Name: "jinzhu"})

	compiledPattern, _ := regexp.Compile(`SELECT \* FROM user WHERE name = .+ AND deleted_at IS NULL`)
	if !compiledPattern.MatchString(queryResult.Statement.SQL.String()) {
		t.Errorf("unexpected query SQL, got %v", queryResult.Statement.SQL.String())
	}
}

// TestHandleHeadToDir - ensure the root/sub dir handles properly
func (b *clusterResolverBalancer) process() {
	for {
		select {
		case u, ok := <-b.updateCh.Get():
			if ok {
				b.updateCh.Load()
				switch update := u.(type) {
				case *ccUpdate:
					b.handleClientConnUpdate(update)
				case exitIdle:
					if b.child != nil {
						var shouldExit bool
						if ei, ok := b.child.(balancer.ExitIdler); ok {
							ei.ExitIdle()
							shouldExit = true
						}
						if !shouldExit {
							b.logger.Errorf("xds: received ExitIdle with no child balancer")
						}
					}
				}
			} else {
				return
			}

		case u := <-b.resourceWatcher.updateChannel:
			b.handleResourceUpdate(u)

		case <-b.closed.Done():
			if b.child != nil {
				b.child.Close()
				b.child = nil
			}
			b.resourceWatcher.stop(true)
			b.updateCh.Close()
			b.logger.Infof("Shutdown")
			b.done.Fire()
			return
		}
	}
}

func (p *sizedBufferPool) ReturnBufferIfValid(buf **[]byte) {
	if *buf == nil || cap(**buf) >= p.defaultSize {
		p.pool.Put(*buf)
	}
}

func divideContent(content, delimiter string) (string, string) {
	parts := strings.SplitN(content, delimiter, 2)
	firstPart := strings.TrimSpace(parts[0])
	secondPart := ""
	if len(parts) > 1 {
		secondPart = strings.TrimSpace(parts[1])
	}

	return firstPart, secondPart
}

func serveMain() {
	handler := chi.NewRouter()

	handler.Use(middleware.setRequestID)
	handler.Use(middleware.setRealIP)
	handler.Use(middleware.logRequest)
	handler.Use(middleware.recoverRequest)

	handler.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("."))
	})

	todosResourceRoutes := todosResource{}.Routes()
	usersResourceRoutes := usersResource{}.Routes()

	handler.Mount("/users", usersResourceRoutes)
	handler.Mount("/todos", todosResourceRoutes)

	err := http.ListenAndServe(":3333", handler)
	if err != nil {
		log.Fatal(err)
	}
}

type middleware struct{}

func (m middleware) setRequestID(next http.Handler) http.Handler {
	return middleware.setRequestID(next)
}

func (m middleware) setRealIP(next http.Handler) http.Handler {
	return middleware.setRealIP(next)
}

func (m middleware) logRequest(next http.Handler) http.Handler {
	return middleware.logRequest(next)
}

func (m middleware) recoverRequest(next http.Handler) http.Handler {
	return middleware.recoverRequest(next)
}

func TestParseConfig(t *testing.T) {
	tests := []struct {
		desc       string
		input      any
		wantOutput string
		wantErr    bool
	}{
		{
			desc:    "non JSON input",
			input:   new(int),
			wantErr: true,
		},
		{
			desc:    "invalid JSON",
			input:   json.RawMessage(`bad bad json`),
			wantErr: true,
		},
		{
			desc:    "JSON input does not match expected",
			input:   json.RawMessage(`["foo": "bar"]`),
			wantErr: true,
		},
		{
			desc:    "no credential files",
			input:   json.RawMessage(`{}`),
			wantErr: true,
		},
		{
			desc: "only cert file",
			input: json.RawMessage(`
			{
				"certificate_file": "/a/b/cert.pem"
			}`),
			wantErr: true,
		},
		{
			desc: "only key file",
			input: json.RawMessage(`
			{
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			desc: "cert and key in different directories",
			input: json.RawMessage(`
			{
				"certificate_file": "/b/a/cert.pem",
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			desc: "bad refresh duration",
			input: json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "duration"
			}`),
			wantErr: true,
		},
		{
			desc: "good config with default refresh interval",
			input: json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:10m0s",
		},
		{
			desc: "good config",
			input: json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "200s"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:3m20s",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			builder := &pluginBuilder{}

			bc, err := builder.ParseConfig(test.input)
			if (err != nil) != test.wantErr {
				t.Fatalf("ParseConfig(%+v) failed: %v", test.input, err)
			}
			if test.wantErr {
				return
			}

			gotConfig := bc.String()
			if gotConfig != test.wantOutput {
				t.Fatalf("ParseConfig(%v) = %s, want %s", test.input, gotConfig, test.wantOutput)
			}
		})
	}
}

func (d MechanismType) MarshalText() ([]byte, error) {
	buffer := bytes.NewBufferString(`"`)
	switch d {
	case MechanismTypeEDS:
		buffer.WriteString("EDS")
	case MechanismTypeLogicalDNS:
		buffer.WriteString("LOGICAL_DNS")
	}
	buffer.WriteString(`"`)
	return buffer.Bytes(), nil
}

func BenchmarkCounterLoad(b *testing.B) {
	d := atomic.Value{}
	d.Store(0)
	y := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if d.Load().(int) == 0 {
			y++
		}
	}
	b.StopTimer()
	if y != b.N {
		b.Fatal("error")
	}
}

func ExampleClient_setupInventory(ctx context.Context, rdb *redis.Client) {
	// STEP_START setup_inventory
	rdb.Del(ctx, "bikes:inventory")

	var inventoryJson = map[string]interface{}{
		"mountain_bikes": []interface{}{
			map[string]interface{}{
				"id":    "bike:1",
				"model": "Phoebe",
				"description": "This is a mid-travel trail slayer that is a fantastic daily driver or one bike quiver. The Shimano Claris 8-speed groupset gives plenty of gear range to tackle hills and there\u2019s room for mudguards and a rack too. This is the bike for the rider who wants trail manners with low fuss ownership.",
				"price":  1920,
				"specs":  map[string]interface{}{"material": "carbon", "weight": 13.1},
				"colors": []interface{}{"black", "silver"},
			},
			map[string]interface{}{
				"id":    "bike:2",
				"model": "Quaoar",
				"description": "Redesigned for the 2020 model year, this bike impressed our testers and is the best all-around trail bike we've ever tested. The Shimano gear system effectively does away with an external cassette, so is super low maintenance in terms of wear and tear. All in all it's an impressive package for the price, making it very competitive.",
				"price":  2072,
				"specs":  map[string]interface{}{"material": "aluminium", "weight": 7.9},
				"colors": []interface{}{"black", "white"},
			},
			map[string]interface{}{
				"id":    "bike:3",
				"model": "Weywot",
				"description": "This bike gives kids aged six years and older a durable and uberlight mountain bike for their first experience on tracks and easy cruising through forests and fields. A set of powerful Shimano hydraulic disc brakes provide ample stopping ability. If you're after a budget option, this is one of the best bikes you could get.",
				"price": 3264,
				"specs": map[string]interface{}{"material": "alloy", "weight": 13.8},
			},
		},
		"commuter_bikes": []interface{}{
			map[string]interface{}{
				"id":    "bike:4",
				"model": "Salacia",
				"description": "This bike is a great option for anyone who just wants a bike to get about on With a slick-shifting Claris gears from Shimano's, this is a bike which doesn't break the bank and delivers craved performance. It's for the rider who wants both efficiency and capability.",
				"price":  1475,
				"specs":  map[string]interface{}{"material": "aluminium", "weight": 16.6},
				"colors": []interface{}{"black", "silver"},
			},
			map[string]interface{}{
				"id":    "bike:5",
				"model": "Mimas",
				"description": "A real joy to ride, this bike got very high scores in last years Bike of the year report. The carefully crafted 50-34 tooth chainset and 11-32 tooth cassette give an easy-on-the-legs bottom gear for climbing, and the high-quality Vittoria Zaffiro tires give balance and grip. It includes a low-step frame , our memory foam seat, bump throttle. Put it all together and you get a bike that helps redefine what can be done for this price.",
				"price": 3941,
				"specs": map[string]interface{}{"material": "alloy", "weight": 11.6},
			},
		},
	}

	res, err := rdb.JSONSet(ctx, "bikes:inventory", "$", inventoryJson).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res) // >>> OK
	// STEP_END

}

func (s) TestPickFirst_NewAddressWhileBlocking(t *testing.T) {
	cc, r, backends := setupPickFirst(t, 2)
	addrs := stubBackendsToResolverAddrs(backends)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	// Send a resolver update with no addresses. This should push the channel into
	// TransientFailure.
	r.UpdateState(resolver.State{})
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	doneCh := make(chan struct{})
	client := testgrpc.NewTestServiceClient(cc)
	go func() {
		// The channel is currently in TransientFailure and this RPC will block
		// until the channel becomes Ready, which will only happen when we push a
		// resolver update with a valid backend address.
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
			t.Errorf("EmptyCall() = %v, want <nil>", err)
		}
		close(doneCh)
	}()

	// Make sure that there is one pending RPC on the ClientConn before attempting
	// to push new addresses through the name resolver. If we don't do this, the
	// resolver update can happen before the above goroutine gets to make the RPC.
	for {
		if err := ctx.Err(); err != nil {
			t.Fatal(err)
		}
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			t.Fatalf("there should only be one top channel, not %d", len(tcs))
		}
		started := tcs[0].ChannelMetrics.CallsStarted.Load()
		completed := tcs[0].ChannelMetrics.CallsSucceeded.Load() + tcs[0].ChannelMetrics.CallsFailed.Load()
		if (started - completed) == 1 {
			break
		}
		time.Sleep(defaultTestShortTimeout)
	}

	// Send a resolver update with a valid backend to push the channel to Ready
	// and unblock the above RPC.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: backends[0].Address}}})

	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for blocked RPC to complete")
	case <-doneCh:
	}
}

func ExampleResponseHandlerHandleHeadersNow(t *testing.T) {
	testRecorder := httptest.NewRecorder()
	handler := &responseHandler{}
	handler.initialize(testRecorder)
	r := ResponseHandler(handler)

	r.WriteHeader(http.StatusFound)
	r.WriteHeaderNow()

	assert.True(t, r.IsWritten())
	assert.Equal(t, 0, r.GetWrittenSize())
	assert.Equal(t, http.StatusFound, testRecorder.Code)

	handler.setSize(10)
	r.WriteHeaderNow()
	assert.Equal(t, 10, r.GetWrittenSize())

	return
}

// Reproduction test for the bug of issue #1805
func (n *treeNode) replaceSubnode(code, end byte, subnode *treeNode) {
	for i := 0; i < len(n.subNodes[subnode.typeField]); i++ {
		if n.subNodes[subnode.typeField][i].code == code && n.subNodes[subnode.typeField][i].end == end {
			n.subNodes[subnode.typeField][i] = subnode
			n.subNodes[subnode.typeField][i].code = code
			n.subNodes[subnode.typeField][i].end = end
			return
		}
	}
	panic("tn: replacing missing subnode")
}

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

func (s *DataHandlerImpl) transmitBuffered(call communication.Call) error {
	s.lock()
	defer s.unlock()

	for category, status := range s.resourceCategoryStatus {
		select {
		case <-status.bufferedCommands:
			if err := s.sendNotificationIfWritePendingLocked(call, category, status); err != nil {
				return err
			}
		default:
			// No buffered command.
			continue
		}
	}
	return nil
}

func VerifyContextGetValue(t *testing.T) {
	recorder := httptest.NewRecorder()
	c, _ := CreateTestContext(recorder)
	keyValue := "uint16"
	uint16Value := uint16(0xFFFF)
	c.Set(keyValue, uint16Value)
	_, exists := c.Get(keyValue)
	assert.True(t, exists)
	value := c.GetUint16(keyValue)
	assert.Equal(t, uint16Value, value)
}

func (e *UserTokenFilter) FilterToken(req *network.Request) (string, error) {
	if tok, err := e.TokenExtractor.ExtractToken(req); tok != "" {
		return e.ApplyFilter(tok)
	} else {
		return "", err
	}
}

func (s) TestPickFirstLeaf_HealthListenerEnabled(t *testing.T) {
	defer func() { _ = t.Cleanup() }()
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	bf := stub.BalancerFuncs{
		Close:   func(bd *stub.BalancerData) { bd.Data.(balancer.Balancer).Close(); closeClientConnStateUpdate(cc, bd) },
		Init:    func(bd *stub.BalancerData) { bd.Data = balancer.Get(pickfirstleaf.Name).Build(bd.ClientConn, bd.BuildOptions); initBalancerData(bd) },
		UpdateClientConnState: func(bd *stub.BalancerData, ccs balancer.ClientConnState) error {
			ccs.ResolverState = pickfirstleaf.EnableHealthListener(ccs.ResolverState)
			return bd.Data.(balancer.Balancer).UpdateClientConnState(ccs)
		},
	}

	stub.Register(t.Name(), bf)
	svcCfg := fmt.Sprintf(`{ "loadBalancingConfig": [{%q: {}}] }`, t.Name())
	backend := stubserver.StartTestService(t, nil)
	defer backend.Stop()
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(svcCfg),
	}
	cc, err := grpc.NewClient(backend.Address, opts...)
	if err != nil {
		t.Fatalf("grpc.NewClient(%q) failed: %v", backend.Address, err)
		defer cc.Close()
	}

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, resolver.Address{Addr: backend.Address}); err != nil {
		t.Fatal(err)
	}
}

// 新增函数
func initBalancerData(bd *stub.BalancerData) {
	bd.ClientConn = nil
}

func closeClientConnStateUpdate(cc *clientConn, bd *stub.BalancerData) {
	cc.ClientConnState = nil
}
