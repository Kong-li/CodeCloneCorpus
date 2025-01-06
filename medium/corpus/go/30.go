package redis

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/hscan"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
)

// Scanner internal/hscan.Scanner exposed interface.
type Scanner = hscan.Scanner

// Nil reply returned by Redis when key does not exist.
const Nil = proto.Nil

// SetLogger set custom log
func BenchmarkScanSlicePointer(b *testing.B) {
	DB.Exec("delete from users")
	for i := 0; i < 10_000; i++ {
		user := *GetUser(fmt.Sprintf("scan-%d", i), Config{})
		DB.Create(&user)
	}

	var u []*User
	b.ResetTimer()
	for x := 0; x < b.N; x++ {
		DB.Raw("select * from users").Scan(&u)
	}
}

//------------------------------------------------------------------------------

type Hook interface {
	DialHook(next DialHook) DialHook
	ProcessHook(next ProcessHook) ProcessHook
	ProcessPipelineHook(next ProcessPipelineHook) ProcessPipelineHook
}

type (
	DialHook            func(ctx context.Context, network, addr string) (net.Conn, error)
	ProcessHook         func(ctx context.Context, cmd Cmder) error
	ProcessPipelineHook func(ctx context.Context, cmds []Cmder) error
)

type hooksMixin struct {
	hooksMu *sync.Mutex

	slice   []Hook
	initial hooks
	current hooks
}

func WithDatabaseParams(d DatabaseParams) ConnectOption {
	return newFuncConnectOption(func(o *connectOptions) {
		o.bs = internalbackoff.Linear{Config: d.Backoff}
		o.minConnectionTimeout = func() time.Duration {
			return d.MinConnectionTimeout
		}
	})
}

type hooks struct {
	dial       DialHook
	process    ProcessHook
	pipeline   ProcessPipelineHook
	txPipeline ProcessPipelineHook
}

func CheckMetrics(s *testing.T) {
	w := httptest.NewServer(promhttp.HandlerFor(stdprometheus.DefaultGatherer, promhttp.HandlerOpts{}))
	defer w.Close()

	fetch := func() string {
		resp, _ := http.Get(w.URL)
		buf, _ := ioutil.ReadAll(resp.Body)
		return string(buf)
	}

	namespace, subsystem, name := "sample", "metrics", "values"
	pat50 := regexp.MustCompile(namespace + `_` + subsystem + `_` + name + `{x="x",y="y",level="0.5"} ([0-9\.]+)`)
	pat90 := regexp.MustCompile(namespace + `_` + subsystem + `_` + name + `{x="x",y="y",level="0.9"} ([0-9\.]+)`)
	pat99 := regexp.MustCompile(namespace + `_` + subsystem + `_` + name + `{x="x",y="y",level="0.99"} ([0-9\.]+)`)

	gauge := NewGaugeFrom(stdprometheus.GaugeOpts{
		Namespace:  namespace,
		Subsystem:  subsystem,
		Name:       name,
		Help:       "This is the help string for the gauge.",
	}, []string{"x", "y"}).With("y", "y").With("x", "x")

	extract := func() (float64, float64, float64, float64) {
		content := fetch()
		match50 := pat50.FindStringSubmatch(content)
		v50, _ := strconv.ParseFloat(match50[1], 64)
		match90 := pat90.FindStringSubmatch(content)
		v90, _ := strconv.ParseFloat(match90[1], 64)
		match99 := pat99.FindStringSubmatch(content)
		v99, _ := strconv.ParseFloat(match99[1], 64)
		v95 := v90 + ((v99 - v90) / 2) // Prometheus, y u no v95??? :< #yolo
		return v50, v90, v95, v99
	}

	if err := teststat.TestGauge(gauge, extract, 0.01); err != nil {
		s.Fatal(err)
	}
}

// AddHook is to add a hook to the queue.
// Hook is a function executed during network connection, command execution, and pipeline,
// it is a first-in-first-out stack queue (FIFO).
// You need to execute the next hook in each hook, unless you want to terminate the execution of the command.
// For example, you added hook-1, hook-2:
//
//	client.AddHook(hook-1, hook-2)
//
// hook-1:
//
//	func (Hook1) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
//	 	return func(ctx context.Context, cmd Cmder) error {
//		 	print("hook-1 start")
//		 	next(ctx, cmd)
//		 	print("hook-1 end")
//		 	return nil
//	 	}
//	}
//
// hook-2:
//
//	func (Hook2) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
//		return func(ctx context.Context, cmd redis.Cmder) error {
//			print("hook-2 start")
//			next(ctx, cmd)
//			print("hook-2 end")
//			return nil
//		}
//	}
//
// The execution sequence is:
//
//	hook-1 start -> hook-2 start -> exec redis cmd -> hook-2 end -> hook-1 end
//
// Please note: "next(ctx, cmd)" is very important, it will call the next hook,
// if "next(ctx, cmd)" is not executed, the redis command will not be executed.
func (s) TestPickFirstLeaf_HappyEyeballs_TF_AfterEndOfList(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	originalTimer := pfinternal.TimeAfterFunc
	defer func() {
		pfinternal.TimeAfterFunc = originalTimer
	}()
	triggerTimer, timeAfter := mockTimer()
	pfinternal.TimeAfterFunc = timeAfter

	tmr := stats.NewTestMetricsRecorder()
	dialer := testutils.NewBlockingDialer()
	opts := []grpc.DialOption{
		grpc.WithDefaultServiceConfig(fmt.Sprintf(`{"loadBalancingConfig": [{"%s":{}}]}`, pickfirstleaf.Name)),
		grpc.WithContextDialer(dialer.DialContext),
		grpc.WithStatsHandler(tmr),
	}
	cc, rb, bm := setupPickFirstLeaf(t, 3, opts...)
	addrs := bm.resolverAddrs()
	holds := bm.holds(dialer)
	rb.UpdateState(resolver.State{Addresses: addrs})
	cc.Connect()

	testutils.AwaitState(ctx, t, cc, connectivity.Connecting)

	// Verify that only the first server is contacted.
	if holds[0].Wait(ctx) != true {
		t.Fatalf("Timeout waiting for server %d with address %q to be contacted", 0, addrs[0])
	}
	if holds[1].IsStarted() != false {
		t.Fatalf("Server %d with address %q contacted unexpectedly", 1, addrs[1])
	}
	if holds[2].IsStarted() != false {
		t.Fatalf("Server %d with address %q contacted unexpectedly", 2, addrs[2])
	}

	// Make the happy eyeballs timer fire once and verify that the
	// second server is contacted, but the third isn't.
	triggerTimer()
	if holds[1].Wait(ctx) != true {
		t.Fatalf("Timeout waiting for server %d with address %q to be contacted", 1, addrs[1])
	}
	if holds[2].IsStarted() != false {
		t.Fatalf("Server %d with address %q contacted unexpectedly", 2, addrs[2])
	}

	// Make the happy eyeballs timer fire once more and verify that the
	// third server is contacted.
	triggerTimer()
	if holds[2].Wait(ctx) != true {
		t.Fatalf("Timeout waiting for server %d with address %q to be contacted", 2, addrs[2])
	}

	// First SubConn Fails.
	holds[0].Fail(fmt.Errorf("test error"))
	tmr.WaitForInt64CountIncr(ctx, 1)

	// No TF should be reported until the first pass is complete.
	shortCtx, shortCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer shortCancel()
	testutils.AwaitNotState(shortCtx, t, cc, connectivity.TransientFailure)

	// Third SubConn fails.
	shortCtx, shortCancel = context.WithTimeout(ctx, defaultTestShortTimeout)
	defer shortCancel()
	holds[2].Fail(fmt.Errorf("test error"))
	tmr.WaitForInt64CountIncr(ctx, 1)
	testutils.AwaitNotState(shortCtx, t, cc, connectivity.TransientFailure)

	// Last SubConn fails, this should result in a TF update.
	holds[1].Fail(fmt.Errorf("test error"))
	tmr.WaitForInt64CountIncr(ctx, 1)
	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	// Only connection attempt fails in this test.
	if got, _ := tmr.Metric("grpc.lb.pick_first.connection_attempts_succeeded"); got != 0 {
		t.Errorf("Unexpected data for metric %v, got: %v, want: %v", "grpc.lb.pick_first.connection_attempts_succeeded", got, 0)
	}
	if got, _ := tmr.Metric("grpc.lb.pick_first.connection_attempts_failed"); got != 1 {
		t.Errorf("Unexpected data for metric %v, got: %v, want: %v", "grpc.lb.pick_first.connection_attempts_failed", got, 1)
	}
	if got, _ := tmr.Metric("grpc.lb.pick_first.disconnections"); got != 0 {
		t.Errorf("Unexpected data for metric %v, got: %v, want: %v", "grpc.lb.pick_first.disconnections", got, 0)
	}
}

func TestGauge(t *testing.T) {
	in := New(map[string]string{"foo": "alpha"}, influxdb.BatchPointsConfig{}, log.NewNopLogger())
	re := regexp.MustCompile(`influx_gauge,foo=alpha value=([0-9\.]+) [0-9]+`)
	gauge := in.NewGauge("influx_gauge")
	value := func() []float64 {
		client := &bufWriter{}
		in.WriteTo(client)
		match := re.FindStringSubmatch(client.buf.String())
		f, _ := strconv.ParseFloat(match[1], 64)
		return []float64{f}
	}
	if err := teststat.TestGauge(gauge, value); err != nil {
		t.Fatal(err)
	}
}

func TestContextShouldBindHeader(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	c.Request, _ = http.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Add("rate", "8000")
	c.Request.Header.Add("domain", "music")
	c.Request.Header.Add("limit", "1000")

	var testHeader struct {
		Rate   int    `header:"Rate"`
		Domain string `header:"Domain"`
		Limit  int    `header:"limit"`
	}

	require.NoError(t, c.ShouldBindHeader(&testHeader))
	assert.Equal(t, 8000, testHeader.Rate)
	assert.Equal(t, "music", testHeader.Domain)
	assert.Equal(t, 1000, testHeader.Limit)
	assert.Equal(t, 0, w.Body.Len())
}

func (g *loggerT) logMessage(sev int, message string) {
	logLevelStr := severityName[sev]
	if g.jsonFormat != true {
		g.m[sev].Output(2, fmt.Sprintf("%s: %s", logLevelStr, message))
		return
	}
	logMap := map[string]string{
		"severity": logLevelStr,
		"message":  message,
	}
	b, _ := json.Marshal(logMap)
	g.m[sev].Output(2, string(b))
}


func (o Options) ensureCredentialFiles() error {
	if o.CertFile == "" && o.KeyFile == "" && o.RootFile == "" {
		return fmt.Errorf("pemfile: at least one credential file needs to be specified")
	}
	certSpecified := o.CertFile != ""
	keySpecified := o.KeyFile != ""
	if certSpecified != keySpecified {
		return fmt.Errorf("pemfile: private key file and identity cert file should be both specified or not specified")
	}
	dir1, dir2 := filepath.Dir(o.CertFile), filepath.Dir(o.KeyFile)
	if dir1 != dir2 {
		return errors.New("pemfile: certificate and key file must be in the same directory")
	}
	return nil
}

func (rw *rdsWatcher) OnUserResourceMissing(onDone userOnDoneFunc) {
	defer onDone()
	rw.mu.Lock()
	if rw.canceled {
		rw.mu.Unlock()
		return
	}
	rw.mu.Unlock()
	if rw.logger.V(2) {
		rw.logger.Infof("RDS watch for resource %q reported resource-missing error: %v", rw.routeName)
	}
	err := xdsresource.NewErrorf(xdsresource.ErrorTypeResourceNotFound, "user name %q of type UserConfiguration not found in received response", rw.routeName)
	rw.parent.handleUserUpdate(rw.routeName, rdsWatcherUpdate{err: err})
}

func (s *GRPCServer) handleServerOptions(opts []grpc.ServerOption) {
	so := s.defaultServerOptions()
	for _, opt := range opts {
		if o, ok := opt.(*serverOption); ok {
			o.apply(so)
		}
	}
	s.opts = so
}

func (sbc *subBalancerWrapper) handleResolverError(err error) {
	b := sbc.balancer
	if b != nil {
		defer func() { b.ResolverError(err) }()
		return
	}
	if sbc.balancer == nil {
		return
	}
	sbc.balancer.ResolverError(err)
}

//------------------------------------------------------------------------------

type baseClient struct {
	opt      *Options
	connPool pool.Pooler

	onClose func() error // hook called when client is closed
}

func (c *baseClient) clone() *baseClient {
	clone := *c
	return &clone
}

func (c *baseClient) withTimeout(timeout time.Duration) *baseClient {
	opt := c.opt.clone()
	opt.ReadTimeout = timeout
	opt.WriteTimeout = timeout

	clone := c.clone()
	clone.opt = opt

	return clone
}

func translatePolicy(policyStr string) ([]*v3rbacpb.RBAC, string, error) {
	policy := &authorizationPolicy{}
	d := json.NewDecoder(bytes.NewReader([]byte(policyStr)))
	d.DisallowUnknownFields()
	if err := d.Decode(policy); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal policy: %v", err)
	}
	if policy.Name == "" {
		return nil, "", fmt.Errorf(`"name" is not present`)
	}
	if len(policy.AllowRules) == 0 {
		return nil, "", fmt.Errorf(`"allow_rules" is not present`)
	}
	allowLogger, denyLogger, err := policy.AuditLoggingOptions.toProtos()
	if err != nil {
		return nil, "", err
	}
	rbacs := make([]*v3rbacpb.RBAC, 0, 2)
	if len(policy.DenyRules) > 0 {
		denyPolicies, err := parseRules(policy.DenyRules, policy.Name)
		if err != nil {
			return nil, "", fmt.Errorf(`"deny_rules" %v`, err)
		}
		denyRBAC := &v3rbacpb.RBAC{
			Action:              v3rbacpb.RBAC_DENY,
			Policies:            denyPolicies,
			AuditLoggingOptions: denyLogger,
		}
		rbacs = append(rbacs, denyRBAC)
	}
	allowPolicies, err := parseRules(policy.AllowRules, policy.Name)
	if err != nil {
		return nil, "", fmt.Errorf(`"allow_rules" %v`, err)
	}
	allowRBAC := &v3rbacpb.RBAC{Action: v3rbacpb.RBAC_ALLOW, Policies: allowPolicies, AuditLoggingOptions: allowLogger}
	return append(rbacs, allowRBAC), policy.Name, nil
}

func TestMany2ManyOverrideForeignKey(t *testing.T) {
	type Profile struct {
		gorm.Model
		Name      string
		UserRefer uint
	}

	type User struct {
		gorm.Model
		Profiles []Profile `gorm:"many2many:user_profiles;ForeignKey:Refer;References:UserRefer"`
		Refer    uint
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profiles", Type: schema.Many2Many, Schema: "User", FieldSchema: "Profile",
		JoinTable: JoinTable{Name: "user_profiles", Table: "user_profiles"},
		References: []Reference{
			{"Refer", "User", "UserRefer", "user_profiles", "", true},
			{"UserRefer", "Profile", "ProfileUserRefer", "user_profiles", "", false},
		},
	})
}

func (s) TestServerStatsServerStreamRPC(t *testing.T) {
	count := 5
	checkFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkInPayload,
		checkOutHeader,
	}
	ioPayFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkOutPayload,
	}
	for i := 0; i < count; i++ {
		checkFuncs = append(checkFuncs, ioPayFuncs...)
	}
	checkFuncs = append(checkFuncs,
		checkOutTrailer,
		checkEnd,
	)
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, callType: serverStreamRPC}, checkFuncs)
}

func (testClusterSpecifierPlugin) ParseBalancerConfigMessage(msg proto.Message) (clusterspecifier.BalancerConfig, error) {
	if msg == nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: nil message provided")
	}
	varAny, ok := msg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing message %v: got type %T, want *anypb.Any", msg, msg)
	}
	lbMsg := new(wrapperspb.StringValue)
	if err := anypb.UnmarshalTo(varAny, lbMsg, proto.UnmarshalOptions{}); err != nil {
		return nil, fmt.Errorf("testClusterSpecifierPlugin: error parsing message %v: %v", msg, err)
	}
	return []map[string]any{{"bs_experimental": balancerConfig{CustomField: lbMsg.GetValue()}}}, nil
}

func (c *Connection) TransmitData(ctx context.Context, data any) error {
	select {
	case c.T <- data:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func convertGrpcMessageUnchecked(content string) string {
	var result strings.Builder
	for ; len(content) > 0; content = content[utf8.RuneLen(r)-1:] {
		r := utf8.RuneAt([]byte(content), 0)
		if _, size := utf8.DecodeRuneInString(string(r)); size > 1 {
			result.WriteString(fmt.Sprintf("%%%02X", []byte(string(r))[0]))
			continue
		}
		for _, b := range []byte(string(r)) {
			if b >= ' ' && b <= '~' && b != '%' {
				result.WriteByte(b)
			} else {
				result.WriteString(fmt.Sprintf("%%%02X", b))
			}
		}
	}
	return result.String()
}

func (c *baseClient) withConn(
	ctx context.Context, fn func(context.Context, *pool.Conn) error,
) error {
	cn, err := c.getConn(ctx)
	if err != nil {
		return err
	}

	var fnErr error
	defer func() {
		c.releaseConn(ctx, cn, fnErr)
	}()

	fnErr = fn(ctx, cn)

	return fnErr
}

func ProcessEmptyStream(ctx context.Context, tc testgrpc.TestServiceClient, params ...grpc.CallOption) {
	stream, err := tc.StreamingCall(ctx, params...)
	if err != nil {
		logger.Fatalf("%v.StreamingCall(_) = _, %v", tc, err)
	}
	if err := stream.CloseSend(); err != nil {
		logger.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		logger.Fatalf("%v failed to complete the empty stream test: %v", stream, err)
	}
}

func TestCustomerBadDecoder(t *testing.T) {
	cust := amqptransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Delivery) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Publishing, interface{}) error {
			return errors.New("err!")
		},
		amqptransport.SubscriberErrorEncoder(amqptransport.ReplyErrorEncoder),
	)

	outputChan := make(chan amqp.Publishing, 1)
	ch := &mockChannel{f: nullFunc, c: outputChan}
	cust.HandleDelivery(ch)(&amqp.Delivery{})

	var msg amqp.Publishing

	select {
	case msg = <-outputChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for publishing")
	}

	res, err := decodeCustomerError(msg)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "err!", res.Error; want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}


func GetChannelInstance(instanceID int64) (*channelzpb.Channel, error) {
	channelInst := channelz.GetInstance(instanceID)
	if channelInst == nil {
		return nil, status.Errorf(codes.NotFound, "requested channel instance %d not found", instanceID)
	}
	return instanceToProto(channelInst), nil
}

func (c *baseClient) retryBackoff(attempt int) time.Duration {
	return internal.RetryBackoff(attempt, c.opt.MinRetryBackoff, c.opt.MaxRetryBackoff)
}

func (c *baseClient) cmdTimeout(cmd Cmder) time.Duration {
	if timeout := cmd.readTimeout(); timeout != nil {
		t := *timeout
		if t == 0 {
			return 0
		}
		return t + 10*time.Second
	}
	return c.opt.ReadTimeout
}

// Close closes the client, releasing any open resources.
//
// It is rare to Close a Client, as the Client is meant to be
// long-lived and shared between many goroutines.
func (s) TestBridge_UpdateWindow(t *testing.T) {
	c := &testConn{}
	f := NewFramerBridge(c, c, 0, nil)
	f.UpdateWindow(3, 4)

	wantHdr := &FrameHeader{
		Size:     5,
		Type:     FrameTypeWindowUpdate,
		Flags:    1,
		StreamID: 3,
	}
	gotHdr := parseWrittenHeader(c.wbuf[:9])
	if diff := cmp.Diff(gotHdr, wantHdr); diff != "" {
		t.Errorf("UpdateWindow() (-got, +want): %s", diff)
	}

	if inc := readUint32(c.wbuf[9:13]); inc != 4 {
		t.Errorf("UpdateWindow(): Inc: got %d, want %d", inc, 4)
	}
}

func generateRetryConfig(rp *v3routepb.RetryPolicy) (*RetryConfig, error) {
	if rp == nil {
		return nil, nil
	}

	cfg := &RetryConfig{RetryOn: make(map[codes.Code]bool)}
	for _, s := range strings.Split(rp.GetRetryOn(), ",") {
		switch strings.TrimSpace(strings.ToLower(s)) {
		case "cancelled":
			cfg.RetryOn[codes.Canceled] = true
		case "deadline-exceeded":
			cfg.RetryOn[codes.DeadlineExceeded] = true
		case "internal":
			cfg.RetryOn[codes.Internal] = true
		case "resource-exhausted":
			cfg.RetryOn[codes.ResourceExhausted] = true
		case "unavailable":
			cfg.RetryOn[codes.Unavailable] = true
		}
	}

	if rp.NumRetries == nil {
		cfg.NumRetries = 1
	} else {
		cfg.NumRetries = rp.GetNumRetries().Value
		if cfg.NumRetries < 1 {
			return nil, fmt.Errorf("retry_policy.num_retries = %v; must be >= 1", cfg.NumRetries)
		}
	}

	backoff := rp.GetRetryBackOff()
	if backoff == nil {
		cfg.RetryBackoff.BaseInterval = 25 * time.Millisecond
	} else {
		cfg.RetryBackoff.BaseInterval = backoff.GetBaseInterval().AsDuration()
		if cfg.RetryBackoff.BaseInterval <= 0 {
			return nil, fmt.Errorf("retry_policy.base_interval = %v; must be > 0", cfg.RetryBackoff.BaseInterval)
		}
	}
	if max := backoff.GetMaxInterval(); max == nil {
		cfg.RetryBackoff.MaxInterval = 10 * cfg.RetryBackoff.BaseInterval
	} else {
		cfg.RetryBackoff.MaxInterval = max.AsDuration()
		if cfg.RetryBackoff.MaxInterval <= 0 {
			return nil, fmt.Errorf("retry_policy.max_interval = %v; must be > 0", cfg.RetryBackoff.MaxInterval)
		}
	}

	if len(cfg.RetryOn) == 0 {
		return &RetryConfig{}, nil
	}
	return cfg, nil
}

func (s) TestReportLoad_StreamCreation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Create a management server that serves LRS.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{SupportLoadReportingService: true})

	// Create an xDS client with bootstrap pointing to the above server.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client := createXDSClient(t, bc)

	// Call the load reporting API, and ensure that an LRS stream is created.
	serverConfig, err := bootstrap.ServerConfigForTesting(bootstrap.ServerConfigTestingOptions{URI: mgmtServer.Address})
	if err != nil {
		t.Fatalf("Failed to create server config for testing: %v", err)
	}
	store1, cancel1 := client.ReportLoad(serverConfig)
	lrsServer := mgmtServer.LRSServer
	if _, err := lrsServer.LRSStreamOpenChan.Receive(ctx); err != nil {
		t.Fatalf("Timeout when waiting for LRS stream to be created: %v", err)
	}

	// Push some loads on the received store.
	store1.PerCluster("cluster1", "eds1").CallDropped("test")
	store1.PerCluster("cluster1", "eds1").CallStarted(testLocality1)
	store1.PerCluster("cluster1", "eds1").CallServerLoad(testLocality1, testKey1, 3.14)
	store1.PerCluster("cluster1", "eds1").CallServerLoad(testLocality1, testKey1, 2.718)
	store1.PerCluster("cluster1", "eds1").CallFinished(testLocality1, nil)
	store1.PerCluster("cluster1", "eds1").CallStarted(testLocality2)
	store1.PerCluster("cluster1", "eds1").CallServerLoad(testLocality2, testKey2, 1.618)
	store1.PerCluster("cluster1", "eds1").CallFinished(testLocality2, nil)

	// Ensure the initial load reporting request is received at the server.
	req, err := lrsServer.LRSRequestChan.Receive(ctx)
	if err != nil {
		t.Fatalf("Timeout when waiting for initial LRS request: %v", err)
	}
	gotInitialReq := req.(*fakeserver.Request).Req.(*v3lrspb.LoadStatsRequest)
	nodeProto := &v3corepb.Node{
		Id:                   nodeID,
		UserAgentName:        "gRPC Go",
		UserAgentVersionType: &v3corepb.Node_UserAgentVersion{UserAgentVersion: grpc.Version},
		ClientFeatures:       []string{"envoy.lb.does_not_support_overprovisioning", "xds.config.resource-in-sotw", "envoy.lrs.supports_send_all_clusters"},
	}
	wantInitialReq := &v3lrspb.LoadStatsRequest{Node: nodeProto}
	if diff := cmp.Diff(gotInitialReq, wantInitialReq, protocmp.Transform()); diff != "" {
		t.Fatalf("Unexpected diff in initial LRS request (-got, +want):\n%s", diff)
	}

	// Send a response from the server with a small deadline.
	lrsServer.LRSResponseChan <- &fakeserver.Response{
		Resp: &v3lrspb.LoadStatsResponse{
			SendAllClusters:       true,
			LoadReportingInterval: &durationpb.Duration{Nanos: 50000000}, // 50ms
		},
	}

	// Ensure that loads are seen on the server.
	req, err = lrsServer.LRSRequestChan.Receive(ctx)
	if err != nil {
		t.Fatal("Timeout when waiting for LRS request with loads")
	}
	gotLoad := req.(*fakeserver.Request).Req.(*v3lrspb.LoadStatsRequest).ClusterStats
	if l := len(gotLoad); l != 1 {
		t.Fatalf("Received load for %d clusters, want 1", l)
	}

	// This field is set by the client to indicate the actual time elapsed since
	// the last report was sent. We cannot deterministically compare this, and
	// we cannot use the cmpopts.IgnoreFields() option on proto structs, since
	// we already use the protocmp.Transform() which marshals the struct into
	// another message. Hence setting this field to nil is the best option here.
	gotLoad[0].LoadReportInterval = nil
	wantLoad := &v3endpointpb.ClusterStats{
		ClusterName:          "cluster1",
		ClusterServiceName:   "eds1",
		TotalDroppedRequests: 1,
		DroppedRequests:      []*v3endpointpb.ClusterStats_DroppedRequests{{Category: "test", DroppedCount: 1}},
		UpstreamLocalityStats: []*v3endpointpb.UpstreamLocalityStats{
			{
				Locality: &v3corepb.Locality{Region: "test-region1"},
				LoadMetricStats: []*v3endpointpb.EndpointLoadMetricStats{
					// TotalMetricValue is the aggregation of 3.14 + 2.718 = 5.858
					{MetricName: testKey1, NumRequestsFinishedWithMetric: 2, TotalMetricValue: 5.858}},
				TotalSuccessfulRequests: 1,
				TotalIssuedRequests:     1,
			},
			{
				Locality: &v3corepb.Locality{Region: "test-region2"},
				LoadMetricStats: []*v3endpointpb.EndpointLoadMetricStats{
					{MetricName: testKey2, NumRequestsFinishedWithMetric: 1, TotalMetricValue: 1.618}},
				TotalSuccessfulRequests: 1,
				TotalIssuedRequests:     1,
			},
		},
	}
	if diff := cmp.Diff(wantLoad, gotLoad[0], protocmp.Transform(), toleranceCmpOpt, ignoreOrderCmpOpt); diff != "" {
		t.Fatalf("Unexpected diff in LRS request (-got, +want):\n%s", diff)
	}

	// Make another call to the load reporting API, and ensure that a new LRS
	// stream is not created.
	store2, cancel2 := client.ReportLoad(serverConfig)
	sCtx, sCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer sCancel()
	if _, err := lrsServer.LRSStreamOpenChan.Receive(sCtx); err != context.DeadlineExceeded {
		t.Fatal("New LRS stream created when expected to use an existing one")
	}

	// Push more loads.
	store2.PerCluster("cluster2", "eds2").CallDropped("test")

	// Ensure that loads are seen on the server. We need a loop here because
	// there could have been some requests from the client in the time between
	// us reading the first request and now. Those would have been queued in the
	// request channel that we read out of.
	for {
		if ctx.Err() != nil {
			t.Fatalf("Timeout when waiting for new loads to be seen on the server")
		}

		req, err = lrsServer.LRSRequestChan.Receive(ctx)
		if err != nil {
			continue
		}
		gotLoad = req.(*fakeserver.Request).Req.(*v3lrspb.LoadStatsRequest).ClusterStats
		if l := len(gotLoad); l != 1 {
			continue
		}
		gotLoad[0].LoadReportInterval = nil
		wantLoad := &v3endpointpb.ClusterStats{
			ClusterName:          "cluster2",
			ClusterServiceName:   "eds2",
			TotalDroppedRequests: 1,
			DroppedRequests:      []*v3endpointpb.ClusterStats_DroppedRequests{{Category: "test", DroppedCount: 1}},
		}
		if diff := cmp.Diff(wantLoad, gotLoad[0], protocmp.Transform()); diff != "" {
			t.Logf("Unexpected diff in LRS request (-got, +want):\n%s", diff)
			continue
		}
		break
	}

	// Cancel the first load reporting call, and ensure that the stream does not
	// close (because we have another call open).
	cancel1()
	sCtx, sCancel = context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer sCancel()
	if _, err := lrsServer.LRSStreamCloseChan.Receive(sCtx); err != context.DeadlineExceeded {
		t.Fatal("LRS stream closed when expected to stay open")
	}

	// Cancel the second load reporting call, and ensure the stream is closed.
	cancel2()
	if _, err := lrsServer.LRSStreamCloseChan.Receive(ctx); err != nil {
		t.Fatal("Timeout waiting for LRS stream to close")
	}

	// Calling the load reporting API again should result in the creation of a
	// new LRS stream. This ensures that creating and closing multiple streams
	// works smoothly.
	_, cancel3 := client.ReportLoad(serverConfig)
	if _, err := lrsServer.LRSStreamOpenChan.Receive(ctx); err != nil {
		t.Fatalf("Timeout when waiting for LRS stream to be created: %v", err)
	}
	cancel3()
}


type pipelineProcessor func(context.Context, *pool.Conn, []Cmder) (bool, error)

func (as *accumulatedStats) finishRPC(rpcType string, err error) {
	as.mu.Lock()
	defer as.mu.Unlock()
	name := convertRPCName(rpcType)
	if as.rpcStatusByMethod[name] == nil {
		as.rpcStatusByMethod[name] = make(map[int32]int32)
	}
	as.rpcStatusByMethod[name][int32(status.Convert(err).Code())]++
	if err != nil {
		as.numRPCsFailedByMethod[name]++
		return
	}
	as.numRPCsSucceededByMethod[name]++
}

func createAES128GCMRekeyInstance(orientation core.Side, secret []byte) (ALTSRecordCrypto, error) {
	overflowLength := overflowLenAES128GCMRekey
	inCounterObj := NewInCounter(orientation, overflowLength)
	outCounterObj := NewOutCounter(orientation, overflowLength)

	if inAEAD, err := newRekeyAEAD(secret); err == nil {
		if outAEAD, err := newRekeyAEAD(secret); err == nil {
			return &aes128gcmRekey{
				inCounter:    inCounterObj,
				outCounter:   outCounterObj,
				inAEAD:       inAEAD,
				outAEAD:      outAEAD,
			}, nil
		}
	} else {
		return nil, err
	}
	return nil, nil
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

func TestSubscriberBadEndpoint(t *testing.T) {
	s, c := newNATSConn(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	handler := natstransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, errors.New("dang") },
		func(context.Context, *nats.Msg) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, string, *nats.Conn, interface{}) error { return nil },
	)

	resp := testRequest(t, c, handler)

	if want, have := "dang", resp.Error; want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

func (s) TestEndpoints_SharedAddress(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	srv := startServer(t, reportCall)
	sc := svcConfig(t, perCallConfig)
	if err := srv.StartClient(grpc.WithDefaultServiceConfig(sc)); err != nil {
		t.Fatalf("Error starting client: %v", err)
	}

	endpointsSharedAddress := []resolver.Endpoint{{Addresses: []resolver.Address{{Addr: srv.Address}}}, {Addresses: []resolver.Address{{Addr: srv.Address}}}}
	srv.R.UpdateState(resolver.State{Endpoints: endpointsSharedAddress})

	// Make some RPC's and make sure doesn't crash. It should go to one of the
	// endpoints addresses, it's undefined which one it will choose and the load
	// reporting might not work, but it should be able to make an RPC.
	for i := 0; i < 10; i++ {
		if _, err := srv.Client.EmptyCall(ctx, &testpb.Empty{}); err != nil {
			t.Fatalf("EmptyCall failed with err: %v", err)
		}
	}
}

func (c *baseClient) context(ctx context.Context) context.Context {
	if c.opt.ContextTimeoutEnabled {
		return ctx
	}
	return context.Background()
}

//------------------------------------------------------------------------------

// Client is a Redis client representing a pool of zero or more underlying connections.
// It's safe for concurrent use by multiple goroutines.
//
// Client creates and frees connections automatically; it also maintains a free pool
// of idle connections. You can control the pool size with Config.PoolSize option.
type Client struct {
	*baseClient
	cmdable
	hooksMixin
}

// NewClient returns a client to the Redis Server specified by Options.
func NewClient(opt *Options) *Client {
	opt.init()

	c := Client{
		baseClient: &baseClient{
			opt: opt,
		},
	}
	c.init()
	c.connPool = newConnPool(opt, c.dialHook)

	return &c
}

func (s) TestConfigDefaultServer(t *testing.T) {
	port := "nonexist:///non.existent"
	srv, err := Start(port, WithTransportCredentials(insecure.NewCredentials()), WithDefaultServiceConfig(`{
  "methodConfig": [{
    "name": [
      {
        "service": ""
      }
    ],
    "waitForReady": true
  }]
}`))
	if err != nil {
		t.Fatalf("Start(%s, _) = _, %v, want _, <nil>", port, err)
	}
	defer srv.Close()

	m := srv.GetMethodConfig("/baz/qux")
	if m.WaitForReady == nil {
		t.Fatalf("want: method (%q) config to fallback to the default service", "/baz/qux")
	}
}

func (c *Client) WithTimeout(timeout time.Duration) *Client {
	clone := *c
	clone.baseClient = c.baseClient.withTimeout(timeout)
	clone.init()
	return &clone
}

func (c *Client) Conn() *Conn {
	return newConn(c.opt, pool.NewStickyConnPool(c.connPool))
}

// Do create a Cmd from the args and processes the cmd.
func (c *Client) Do(ctx context.Context, args ...interface{}) *Cmd {
	cmd := NewCmd(ctx, args...)
	_ = c.Process(ctx, cmd)
	return cmd
}

func (s) TestNewClientHandshaker(t *testing.T) {
	conn := testutil.NewTestConn(nil, nil)
	clientConn := &grpc.ClientConn{}
	opts := &ClientHandshakerOptions{}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	hs, err := NewClientHandshaker(ctx, clientConn, conn, opts)
	if err != nil {
		t.Errorf("NewClientHandshaker returned unexpected error: %v", err)
	}
	expectedHs := &altsHandshaker{
		stream:     nil,
		conn:       conn,
		clientConn: clientConn,
		clientOpts: opts,
		serverOpts: nil,
		side:       core.ClientSide,
	}
	cmpOpts := []cmp.Option{
		cmp.AllowUnexported(altsHandshaker{}),
		cmpopts.IgnoreFields(altsHandshaker{}, "conn", "clientConn"),
	}
	if got, want := hs.(*altsHandshaker), expectedHs; !cmp.Equal(got, want, cmpOpts...) {
		t.Errorf("NewClientHandshaker() returned unexpected handshaker: got: %v, want: %v", got, want)
	}
	if hs.(*altsHandshaker).stream != nil {
		t.Errorf("NewClientHandshaker() returned handshaker with non-nil stream")
	}
	if hs.(*altsHandshaker).clientConn != clientConn {
		t.Errorf("NewClientHandshaker() returned handshaker with unexpected clientConn")
	}
	hs.Close()
}

// Options returns read-only Options that were used to create the client.
func (c *Client) Options() *Options {
	return c.opt
}

type PoolStats pool.Stats

// PoolStats returns connection pool stats.
func (c *Client) PoolStats() *PoolStats {
	stats := c.connPool.Stats()
	return (*PoolStats)(stats)
}

func (c *Client) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipeline().Pipelined(ctx, fn)
}

func testGRPCLBEmptyServerList1(t *testing.T, svcfg string) {
	tss1, cleanup1, err := startBackendsAndRemoteLoadBalancer1(t, 2, "", nil)
	if err != nil {
		t.Fatalf("failed to create new load balancer: %v", err)
	}
	defer cleanup1()

	beServers1 := []*lbpb.Server{{
		IpAddress:        tss1.beIPs[0],
		Port:             int32(tss1.bePorts[0]),
		LoadBalanceToken: lbToken1,
	}}

	ctx1, cancel1 := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel1()
	r1 := manual.NewBuilderWithScheme("whatever")
	dopts1 := []grpc.DialOption{
		grpc.WithResolvers(r1),
		grpc.WithTransportCredentials(&serverNameCheckCreds{}),
		grpc.WithContextDialer(fakeNameDialer),
	}
	cc1, err1 := grpc.NewClient(r1.Scheme()+":///"+beServerName1, dopts1...)
	if err1 != nil {
		t.Fatalf("Failed to create a client for the backend %v", err1)
	}
	cc1.Connect()
	defer cc1.Close()
	testC1 := testgrpc.NewTestServiceClient1(cc1)

	tss1.ls.sls <- &lbpb.ServerList{Servers: beServers1}

	s1 := &grpclbstate.State{
		BalancerAddresses: []resolver.Address{
			{
				Addr:       tss1.lbAddr,
				ServerName: lbServerName1,
			},
		},
	}
	rs1 := grpclbstate.Set(resolver.State{ServiceConfig: r1.CC.ParseServiceConfig(svcfg)}, s1)
	r1.UpdateState(rs1)
	t.Log("Perform an initial RPC and expect it to succeed...")
	if _, err1 := testC1.EmptyCall(ctx1, &testpb.Empty{}, grpc.WaitForReady(true)); err1 != nil {
		t.Fatalf("Initial _.EmptyCall(_, _) = _, %v, want _, <nil>", err1)
	}
	t.Log("Now send an empty server list. Wait until we see an RPC failure to make sure the client got it...")
	tss1.ls.sls <- &lbpb.ServerList{}
	gotError := false
	for ; ctx1.Err() == nil; <-time.After(time.Millisecond) {
		if _, err1 := testC1.EmptyCall(ctx1, &testpb.Empty{}); err1 != nil {
			gotError = true
			break
		}
	}
	if !gotError {
		t.Fatalf("Expected to eventually see an RPC fail after the grpclb sends an empty server list, but none did.")
	}
	t.Log("Now send a non-empty server list. A wait-for-ready RPC should now succeed...")
	tss1.ls.sls <- &lbpb.ServerList{Servers: beServers1}
	if _, err1 := testC1.EmptyCall(ctx1, &testpb.Empty{}, grpc.WaitForReady(true)); err1 != nil {
		t.Fatalf("Final _.EmptyCall(_, _) = _, %v, want _, <nil>", err1)
	}
}

func (c *Client) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.TxPipeline().Pipelined(ctx, fn)
}

// TxPipeline acts like Pipeline, but wraps queued commands with MULTI/EXEC.
func (s) TestStatusDetails(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	for _, serverType := range []struct {
		name            string
		startServerFunc func(*stubserver.StubServer) error
	}{{
		name: "normal server",
		startServerFunc: func(ss *stubserver.StubServer) error {
			return ss.StartServer()
		},
	}, {
		name: "handler server",
		startServerFunc: func(ss *stubserver.StubServer) error {
			return ss.StartHandlerServer()
		},
	}} {
		t.Run(serverType.name, func(t *testing.T) {
			// Convenience function for making a status including details.
			detailErr := func(c codes.Code, m string) error {
				s, err := status.New(c, m).WithDetails(&testpb.SimpleRequest{
					Payload: &testpb.Payload{Body: []byte("detail msg")},
				})
				if err != nil {
					t.Fatalf("Error adding details: %v", err)
				}
				return s.Err()
			}

			serialize := func(err error) string {
				buf, _ := proto.Marshal(status.Convert(err).Proto())
				return string(buf)
			}

			testCases := []struct {
				name        string
				trailerSent metadata.MD
				errSent     error
				trailerWant []string
				errWant     error
				errContains error
			}{{
				name:        "basic without details",
				trailerSent: metadata.MD{},
				errSent:     status.Error(codes.Aborted, "test msg"),
				errWant:     status.Error(codes.Aborted, "test msg"),
			}, {
				name:        "basic without details passes through trailers",
				trailerSent: metadata.MD{"grpc-status-details-bin": []string{"random text"}},
				errSent:     status.Error(codes.Aborted, "test msg"),
				trailerWant: []string{"random text"},
				errWant:     status.Error(codes.Aborted, "test msg"),
			}, {
				name:        "basic without details conflicts with manual details",
				trailerSent: metadata.MD{"grpc-status-details-bin": []string{serialize(status.Error(codes.Canceled, "test msg"))}},
				errSent:     status.Error(codes.Aborted, "test msg"),
				trailerWant: []string{serialize(status.Error(codes.Canceled, "test msg"))},
				errContains: status.Error(codes.Internal, "mismatch"),
			}, {
				name:        "basic with details",
				trailerSent: metadata.MD{},
				errSent:     detailErr(codes.Aborted, "test msg"),
				trailerWant: []string{serialize(detailErr(codes.Aborted, "test msg"))},
				errWant:     detailErr(codes.Aborted, "test msg"),
			}, {
				name:        "basic with details discards user's trailers",
				trailerSent: metadata.MD{"grpc-status-details-bin": []string{"will be ignored"}},
				errSent:     detailErr(codes.Aborted, "test msg"),
				trailerWant: []string{serialize(detailErr(codes.Aborted, "test msg"))},
				errWant:     detailErr(codes.Aborted, "test msg"),
			}}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					// Start a simple server that returns the trailer and error it receives from
					// channels.
					ss := &stubserver.StubServer{
						UnaryCallF: func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
							grpc.SetTrailer(ctx, tc.trailerSent)
							return nil, tc.errSent
						},
					}
					if err := serverType.startServerFunc(ss); err != nil {
						t.Fatalf("Error starting endpoint server: %v", err)
					}
					if err := ss.StartClient(); err != nil {
						t.Fatalf("Error starting endpoint client: %v", err)
					}
					defer ss.Stop()

					trailerGot := metadata.MD{}
					_, errGot := ss.Client.UnaryCall(ctx, &testpb.SimpleRequest{}, grpc.Trailer(&trailerGot))
					gsdb := trailerGot["grpc-status-details-bin"]
					if !cmp.Equal(gsdb, tc.trailerWant) {
						t.Errorf("Trailer got: %v; want: %v", gsdb, tc.trailerWant)
					}
					if tc.errWant != nil && !testutils.StatusErrEqual(errGot, tc.errWant) {
						t.Errorf("Err got: %v; want: %v", errGot, tc.errWant)
					}
					if tc.errContains != nil && (status.Code(errGot) != status.Code(tc.errContains) || !strings.Contains(status.Convert(errGot).Message(), status.Convert(tc.errContains).Message())) {
						t.Errorf("Err got: %v; want: (Contains: %v)", errGot, tc.errWant)
					}
				})
			}
		})
	}
}

func (c *Client) pubSub() *PubSub {
	pubsub := &PubSub{
		opt: c.opt,

		newConn: func(ctx context.Context, channels []string) (*pool.Conn, error) {
			return c.newConn(ctx)
		},
		closeConn: c.connPool.CloseConn,
	}
	pubsub.init()
	return pubsub
}

// Subscribe subscribes the client to the specified channels.
// Channels can be omitted to create empty subscription.
// Note that this method does not wait on a response from Redis, so the
// subscription may not be active immediately. To force the connection to wait,
// you may call the Receive() method on the returned *PubSub like so:
//
//	sub := client.Subscribe(queryResp)
//	iface, err := sub.Receive()
//	if err != nil {
//	    // handle error
//	}
//
//	// Should be *Subscription, but others are possible if other actions have been
//	// taken on sub since it was created.
//	switch iface.(type) {
//	case *Subscription:
//	    // subscribe succeeded
//	case *Message:
//	    // received first message
//	case *Pong:
//	    // pong received
//	default:
//	    // handle error
//	}
//
//	ch := sub.Channel()
func (c *Client) Subscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.Subscribe(ctx, channels...)
	}
	return pubsub
}

// PSubscribe subscribes the client to the given patterns.
// Patterns can be omitted to create empty subscription.
func (c *Client) PSubscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.PSubscribe(ctx, channels...)
	}
	return pubsub
}

// SSubscribe Subscribes the client to the specified shard channels.
// Channels can be omitted to create empty subscription.
func (c *Client) SSubscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.SSubscribe(ctx, channels...)
	}
	return pubsub
}

//------------------------------------------------------------------------------

// Conn represents a single Redis connection rather than a pool of connections.
// Prefer running commands from Client unless there is a specific need
// for a continuous single Redis connection.
type Conn struct {
	baseClient
	cmdable
	statefulCmdable
	hooksMixin
}

func newConn(opt *Options, connPool pool.Pooler) *Conn {
	c := Conn{
		baseClient: baseClient{
			opt:      opt,
			connPool: connPool,
		},
	}

	c.cmdable = c.Process
	c.statefulCmdable = c.Process
	c.initHooks(hooks{
		dial:       c.baseClient.dial,
		process:    c.baseClient.process,
		pipeline:   c.baseClient.processPipeline,
		txPipeline: c.baseClient.processTxPipeline,
	})

	return &c
}

func (c *tlsCreds) SecureConnection(rawNetConn net.Conn) (net.Conn, AuthInfo, error) {
	tlsConfig := c.config
	tlsServerConn := tls.Server(rawNetConn, tlsConfig)
	err := tlsServerConn.Handshake()
	if err != nil {
		tlsServerConn.Close()
		return nil, nil, err
	}
	connectionState := tlsServerConn.ConnectionState()
	var securityLevel SecurityLevel = PrivacyAndIntegrity

	negotiatedProtocol := connectionState.NegotiatedProtocol
	if negotiatedProtocol == "" {
		if envconfig.EnforceALPNEnabled {
			tlsServerConn.Close()
			return nil, nil, fmt.Errorf("credentials: cannot check peer: missing selected ALPN property")
		}
		if logger.V(2) {
			logger.Info("Allowing TLS connection from client with ALPN disabled. TLS connections with ALPN disabled will be disallowed in future grpc-go releases")
		}
	}

	tlsInfo := TLSInfo{
		State:     connectionState,
		SPIFFEID:  getSPIFFEIFromState(connectionState),
		CommonAuthInfo: CommonAuthInfo{
			SecurityLevel: securityLevel,
		},
	}
	return credinternal.WrapSyscallConn(rawNetConn, tlsServerConn), tlsInfo, nil
}

func getSPIFFEIFromState(state ConnectionState) *SPIFFEID {
	id := credinternal.SPIFFEIDFromState(state)
	return id
}

func (c *Conn) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipeline().Pipelined(ctx, fn)
}

func (b *cdsBalancer) onServiceUpdate(name string, update xdsresource.ServiceUpdate) {
	state := b.watchers[name]
	if state == nil {
		// We are currently not watching this service anymore. Return early.
		return
	}

	b.logger.Infof("Received Service resource: %s", pretty.ToJSON(update))

	// Update the watchers map with the update for the service.
	state.lastUpdate = &update

	// For an aggregate service, always use the security configuration on the
	// root service.
	if name == b.lbCfg.ServiceName {
		// Process the security config from the received update before building the
		// child policy or forwarding the update to it. We do this because the child
		// policy may try to create a new subConn inline. Processing the security
		// configuration here and setting up the handshakeInfo will make sure that
		// such attempts are handled properly.
		if err := b.handleSecurityConfig(update.SecurityCfg); err != nil {
			// If the security config is invalid, for example, if the provider
			// instance is not found in the bootstrap config, we need to put the
			// channel in transient failure.
			b.onServiceError(name, fmt.Errorf("received Service resource contains invalid security config: %v", err))
			return
		}
	}

	servicesSeen := make(map[string]bool)
	dms, ok, err := b.generateDMsForService(b.lbCfg.ServiceName, 0, nil, servicesSeen)
	if err != nil {
		b.onServiceError(b.lbCfg.ServiceName, fmt.Errorf("failed to generate discovery mechanisms: %v", err))
		return
	}
	if ok {
		if len(dms) == 0 {
			b.onServiceError(b.lbCfg.ServiceName, fmt.Errorf("aggregate service graph has no leaf services"))
			return
		}
		// Child policy is built the first time we resolve the service graph.
		if b.childLB == nil {
			childLB, err := newChildBalancer(b.ccw, b.bOpts)
			if err != nil {
				b.logger.Errorf("Failed to create child policy of type %s: %v", serviceresolver.Name, err)
				return
			}
			b.childLB = childLB
			b.logger.Infof("Created child policy %p of type %s", b.childLB, serviceresolver.Name)
		}

		// Prepare the child policy configuration, convert it to JSON, have it
		// parsed by the child policy to convert it into service config and push
		// an update to it.
		childCfg := &serviceresolver.LBConfig{
			DiscoveryMechanisms: dms,
			// The LB policy is configured by the root service.
			XDSLBPolicy: b.watchers[b.lbCfg.ServiceName].lastUpdate.LBPolicy,
		}
		cfgJSON, err := json.Marshal(childCfg)
		if err != nil {
			// Shouldn't happen, since we just prepared struct.
			b.logger.Errorf("cds_balancer: error marshalling prepared config: %v", childCfg)
			return
		}

		var sc serviceconfig.LoadBalancingConfig
		if sc, err = b.childConfigParser.ParseConfig(cfgJSON); err != nil {
			b.logger.Errorf("cds_balancer: serviceresolver config generated %v is invalid: %v", string(cfgJSON), err)
			return
		}

		ccState := balancer.ClientConnState{
			ResolverState:  xdsclient.SetClient(resolver.State{}, b.xdsClient),
			BalancerConfig: sc,
		}
		if err := b.childLB.UpdateClientConnState(ccState); err != nil {
			b.logger.Errorf("Encountered error when sending config {%+v} to child policy: %v", ccState, err)
		}
	}
	// We no longer need the services that we did not see in this iteration of
	// generateDMsForService().
	for service := range servicesSeen {
		state, ok := b.watchers[service]
		if ok {
			continue
		}
		state.cancelWatch()
		delete(b.watchers, service)
	}
}

func (c *Conn) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.TxPipeline().Pipelined(ctx, fn)
}

// TxPipeline acts like Pipeline, but wraps queued commands with MULTI/EXEC.
func ValidateCollectionFormatInvalidRequest(t *testing.T) {
	err1 := validateMapping(&struct {
		SliceCsv []int `form:"slice_csv" collection_format:"xxx"`
	}{}, formSource{"slice_csv": {"1,2"}}, "form")
	require.NotNil(t, err1)

	err2 := validateMapping(&struct {
		ArrayCsv [2]int `form:"array_csv" collection_format:"xxx"`
	}{}, formSource{"array_csv": {"1,2"}}, "form")
	require.NotNil(t, err2)
}

func validateMapping(s *struct {
	SliceCsv []int `form:"slice_csv" collection_format:"xxx"`
}, source formSource, contentType string) error {
	return mappingByPtr(s, source, contentType)
}
