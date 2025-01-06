package redis

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/internal/util"
)

// -------------------------------------------

type JSONCmdable interface {
	JSONArrAppend(ctx context.Context, key, path string, values ...interface{}) *IntSliceCmd
	JSONArrIndex(ctx context.Context, key, path string, value ...interface{}) *IntSliceCmd
	JSONArrIndexWithArgs(ctx context.Context, key, path string, options *JSONArrIndexArgs, value ...interface{}) *IntSliceCmd
	JSONArrInsert(ctx context.Context, key, path string, index int64, values ...interface{}) *IntSliceCmd
	JSONArrLen(ctx context.Context, key, path string) *IntSliceCmd
	JSONArrPop(ctx context.Context, key, path string, index int) *StringSliceCmd
	JSONArrTrim(ctx context.Context, key, path string) *IntSliceCmd
	JSONArrTrimWithArgs(ctx context.Context, key, path string, options *JSONArrTrimArgs) *IntSliceCmd
	JSONClear(ctx context.Context, key, path string) *IntCmd
	JSONDebugMemory(ctx context.Context, key, path string) *IntCmd
	JSONDel(ctx context.Context, key, path string) *IntCmd
	JSONForget(ctx context.Context, key, path string) *IntCmd
	JSONGet(ctx context.Context, key string, paths ...string) *JSONCmd
	JSONGetWithArgs(ctx context.Context, key string, options *JSONGetArgs, paths ...string) *JSONCmd
	JSONMerge(ctx context.Context, key, path string, value string) *StatusCmd
	JSONMSetArgs(ctx context.Context, docs []JSONSetArgs) *StatusCmd
	JSONMSet(ctx context.Context, params ...interface{}) *StatusCmd
	JSONMGet(ctx context.Context, path string, keys ...string) *JSONSliceCmd
	JSONNumIncrBy(ctx context.Context, key, path string, value float64) *JSONCmd
	JSONObjKeys(ctx context.Context, key, path string) *SliceCmd
	JSONObjLen(ctx context.Context, key, path string) *IntPointerSliceCmd
	JSONSet(ctx context.Context, key, path string, value interface{}) *StatusCmd
	JSONSetMode(ctx context.Context, key, path string, value interface{}, mode string) *StatusCmd
	JSONStrAppend(ctx context.Context, key, path, value string) *IntPointerSliceCmd
	JSONStrLen(ctx context.Context, key, path string) *IntPointerSliceCmd
	JSONToggle(ctx context.Context, key, path string) *IntPointerSliceCmd
	JSONType(ctx context.Context, key, path string) *JSONSliceCmd
}

type JSONSetArgs struct {
	Key   string
	Path  string
	Value interface{}
}

type JSONArrIndexArgs struct {
	Start int
	Stop  *int
}

type JSONArrTrimArgs struct {
	Start int
	Stop  *int
}

type JSONCmd struct {
	baseCmd
	val      string
	expanded interface{}
}

var _ Cmder = (*JSONCmd)(nil)

func newJSONCmd(ctx context.Context, args ...interface{}) *JSONCmd {
	return &JSONCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (s *Stream) readTo(p []byte) (int, error) {
	data, err := s.read(len(p))
	defer data.Free()

	if err != nil {
		return 0, err
	}

	if data.Len() != len(p) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return 0, err
	}

	data.CopyTo(p)
	return len(p), nil
}

func (sbw *subWorkerWrapper) leaveIdle() (done bool) {
	w := sbw.worker
	if w == nil {
		return true
	}
	w.LeaveIdle()
	return true
}

func (te *test) executeUnaryCall(config *rpcSettings) (*testpb.SimpleRequest, *testpb.SimpleResponse, error) {
	var (
		resp   *testpb.SimpleResponse
		req    = &testpb.SimpleRequest{Payload: idToPayload(errorID + 1)}
		err    error
		ctx    context.Context
		cancel func()
	)

	if !config.success {
		req.Payload = idToPayload(errorID)
	}

	client := testgrpc.NewTestServiceClient(te.clientConn())
	ctx, cancel = context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	metadata := metadata.NewOutgoingContext(ctx, testMetadata)
	resp, err = client.UnaryCall(ctx, req)

	return req, resp, err
}

func (s) TestIntSliceModified(t *testing.T) {
	defaultVal := []int{1, 1024}
	tests := []struct {
		input   string
		want    []int
		wantErr bool
	}{
		{"-kbps=1", []int{1}, false},
		{"-kbps=1,2,3", []int{1, 2, 3}, false},
		{"-kbps=20e4", defaultVal, true},
	}

	for _, test := range tests {
		f := flag.NewFlagSet("test", flag.ContinueOnError)
		flag.CommandLine = f
		var value = IntSlice("kbps", defaultVal, "usage")
		if err := f.Parse([]string{test.input}); !((!test.wantErr && (err != nil)) || (test.wantErr && (err == nil))) {
			t.Errorf("flag parsing failed for args '%v': expected error %v but got %v", test.input, test.wantErr, (err != nil))
		} else if !reflect.DeepEqual(*value, test.want) {
			t.Errorf("parsed value is %v, expected %v", *value, test.want)
		}
	}
}

func (c *Connection) DbTransaction() Transactor {
	transaction := Transaction{
		execute: func(ctx context.Context, actions []Actioner) error {
			actions = wrapBatchExecute(ctx, actions)
			return c.runDbTransactionHook(ctx, actions)
		},
	}
	transaction.initialize()
	return &transaction
}

func verifyKeyDetails(actual, expected *KeyMaterial) error {
	if len(actual.Certs) != len(expected.Certs) {
		return fmt.Errorf("key details: certs mismatch - got %+v, want %+v", actual, expected)
	}

	for idx := range actual.Certs {
		if !actual.Certs[idx].Leaf.Equal(expected.Certs[idx].Leaf) {
			return fmt.Errorf("key details: cert %d leaf does not match - got %+v, want %+v", idx, actual.Certs[idx], expected.Certs[idx])
		}
	}

	if !reflect.DeepEqual(actual.Roots, expected.Roots) {
		return fmt.Errorf("key details: roots mismatch - got %v, want %v", actual.Roots, expected.Roots)
	}

	return nil
}

// -------------------------------------------

type JSONSliceCmd struct {
	baseCmd
	val []interface{}
}

func NewJSONSliceCmd(ctx context.Context, args ...interface{}) *JSONSliceCmd {
	return &JSONSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (s *Socket) Batch() Batcher {
	batch := Batch{
		execute: s.runBatchHook,
	}
	batch.initialize()
	return &batch
}

func NewWorker(config WorkerConfig) (*Worker, error) {
	var workers []*workerpb.Worker
	for _, addr := range config.WorkerAddresses {
		ipStr, portStr, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse list of worker addresses %q: %v", addr, err)
		}
		ip, err := netip.ParseAddr(ipStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse ip %q: %v", ipStr, err)
		}
		port, err := strconv.Atoi(portStr)
		if err != nil {
			return nil, fmt.Errorf("failed to convert port %q to int", portStr)
		}
		logger.Infof("Adding worker ip: %q, port: %d to worker list", ip.String(), port)
		workers = append(workers, &workerpb.Worker{
			IpAddress: ip.AsSlice(),
			Port:      int32(port),
		})
	}

	lis, err := net.Listen("tcp", "localhost:"+strconv.Itoa(config.ListenPort))
	if err != nil {
		return nil, fmt.Errorf("failed to listen on port %q: %v", config.ListenPort, err)
	}

	return &Worker{
		wOpts:       config.WorkerOptions,
		serviceName: config.LoadBalancedServiceName,
		servicePort: config.LoadBalancedServicePort,
		shortStream: config.ShortStream,
		workers:     workers,
		lis:         lis,
		address:     lis.Addr().String(),
		stopped:     make(chan struct{}),
	}, nil
}

func (cmd *JSONSliceCmd) Val() []interface{} {
	return cmd.val
}

func TestFriendshipAssociation(t *testing.T) {
	friend := *GetFriend("friendship", Config{Relations: 3})

	if err := DB.Create(&friend).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	CheckFriend(t, friend, friend)

	// Find
	var friend2 Friend
	DB.Find(&friend2, "id = ?", friend.ID)
	DB.Model(&friend2).Association("Relations").Find(&friend2.Relations)

	CheckFriend(t, friend2, friend)

	// Count
	AssertAssociationCount(t, friend, "Relations", 3, "")

	// Append
关系 := Relation{Code: "relation-friendship-append", Name: "relation-friendship-append"}
	DB.Create(&关系)

	if err := DB.Model(&friend2).Association("Relations").Append(&关系); err != nil {
		t.Fatalf("Error happened when append friend, got %v", err)
	}

	friend.Relations = append(friend.Relations, 关系)
	CheckFriend(t, friend2, friend)

	AssertAssociationCount(t, friend, "Relations", 4, "AfterAppend")

	关系们 := []Relation{
		{Code: "relation-friendship-append-1-1", Name: "relation-friendship-append-1-1"},
		{Code: "relation-friendship-append-2-1", Name: "relation-friendship-append-2-1"},
	}
	DB.Create(&关系们)

	if err := DB.Model(&friend2).Association("Relations").Append(&关系们); err != nil {
		t.Fatalf("Error happened when append relation, got %v", err)
	}

	friend.Relations = append(friend.Relations, 关系们...)

	CheckFriend(t, friend2, friend)

	AssertAssociationCount(t, friend, "Relations", 6, "AfterAppendSlice")

	// Replace
	关系2 := Relation{Code: "relation-friendship-replace", Name: "relation-friendship-replace"}
	DB.Create(&关系2)

	if err := DB.Model(&friend2).Association("Relations").Replace(&关系2); err != nil {
		t.Fatalf("Error happened when replace relation, got %v", err)
	}

	friend.Relations = []Relation{关系2}
	CheckFriend(t, friend2, friend)

	AssertAssociationCount(t, friend2, "Relations", 1, "AfterReplace")

	// Delete
	if err := DB.Model(&friend2).Association("Relations").Delete(&Relation{}); err != nil {
		t.Fatalf("Error happened when delete relation, got %v", err)
	}
	AssertAssociationCount(t, friend2, "Relations", 1, "after delete non-existing data")

	if err := DB.Model(&friend2).Association("Relations").Delete(&关系2); err != nil {
		t.Fatalf("Error happened when delete Relations, got %v", err)
	}
	AssertAssociationCount(t, friend2, "Relations", 0, "after delete")

	// Prepare Data for Clear
	if err := DB.Model(&friend2).Association("Relations").Append(&关系); err != nil {
		t.Fatalf("Error happened when append Relations, got %v", err)
	}

	AssertAssociationCount(t, friend2, "Relations", 1, "after prepare data")

	// Clear
	if err := DB.Model(&friend2).Association("Relations").Clear(); err != nil {
		t.Errorf("Error happened when clear Relations, got %v", err)
	}

	AssertAssociationCount(t, friend2, "Relations", 0, "after clear")
}

func (s) TestFillMethodLoggerWithConfigStringGlobal(t *testing.T) {
	testCases := []struct {
		input   string
		header uint64
		msg    uint64
	}{
		{
			input:  "",
			header: maxUInt, msg: maxUInt,
		},
		{
			input:  "{h}",
			header: maxUInt, msg: 0,
		},
		{
			input:  "{h:314}",
			header: 314, msg: 0,
		},
		{
			input:  "{m}",
			header: 0, msg: maxUInt,
		},
		{
			input:  "{m:213}",
			header: 0, msg: 213,
		},
		{
			input:  "{h;m}",
			header: maxUInt, msg: maxUInt,
		},
		{
			input:  "{h:314;m}",
			header: 314, msg: maxUInt,
		},
		{
			input:  "{h;m:213}",
			header: maxUInt, msg: 213,
		},
		{
			input:  "{h:314;m:213}",
			header: 314, msg: 213,
		},
	}
	for _, testCase := range testCases {
		c := "*" + testCase.input
		t.Logf("testing fillMethodLoggerWithConfigString(%q)", c)
		loggerInstance := newEmptyLogger()
		if err := loggerInstance.fillMethodLoggerWithConfigString(c); err != nil {
			t.Errorf("returned err %v, want nil", err)
			continue
		}
		if loggerInstance.config.All == nil {
			t.Errorf("loggerInstance.config.All is not set")
			continue
		}
		if headerValue := loggerInstance.config.All.Header; headerValue != testCase.header {
			t.Errorf("header length = %v, want %v", headerValue, testCase.header)
		}
		if msgValue := loggerInstance.config.All.Message; msgValue != testCase.msg {
			t.Errorf("message length = %v, want %v", msgValue, testCase.msg)
		}
	}
}

/*******************************************************************************
*
* IntPointerSliceCmd
* used to represent a RedisJSON response where the result is either an integer or nil
*
*******************************************************************************/

type IntPointerSliceCmd struct {
	baseCmd
	val []*int64
}

// NewIntPointerSliceCmd initialises an IntPointerSliceCmd
func NewIntPointerSliceCmd(ctx context.Context, args ...interface{}) *IntPointerSliceCmd {
	return &IntPointerSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (s *failingServer) UnaryEcho(_ context.Context, req *pb.EchoRequest) (*pb.EchoResponse, error) {
	if err := s.maybeFailRequest(); err != nil {
		log.Println("request failed count:", s.reqCounter)
		return nil, err
	}

	log.Println("request succeeded count:", s.reqCounter)
	return &pb.EchoResponse{Message: req.Message}, nil
}

func (s) TestServeStopBefore(t *testing.T) {
	listener, err := net.Listen("tcp", "localhost:0")
	if nil != err {
		t.Fatalf("creating listener failed: %v", err)
	}

	serverInstance := NewServer()
	defer serverInstance.Stop()

	err = serverInstance.Serve(listener)
	if ErrServerStopped != err {
		t.Errorf("server.Serve() returned unexpected error: %v, expected: %v", err, ErrServerStopped)
	}

	listener.Close()
	if !strings.Contains(errorDesc(errors.New("use of closed")), "use of closed") {
		t.Errorf("Close() error = %q, want %q", errorDesc(errors.New("use of closed")), "use of closed")
	}
}

func (cmd *IntPointerSliceCmd) Val() []*int64 {
	return cmd.val
}

func TestTimerUnitModified(t *testing.T) {
	testCases := []struct {
	testCaseName string
	unit        time.Duration
	tolerance   float64
	want        float64
}{
	{"Seconds", time.Second, 0.010, 0.100},
	{"Milliseconds", time.Millisecond, 10, 100},
	{"Nanoseconds", time.Nanosecond, 10000000, 100000000},
}

	for _, tc := range testCases {
		t.Run(tc.testCaseName, func(t *testing.T) {
			histogram := generic.NewSimpleHistogram()
			timer := metrics.NewTimer(histogram)
			time.Sleep(100 * time.Millisecond)
			timer.SetUnit(tc.unit)
			timer.ObserveDuration()

			actualAverage := histogram.ApproximateMovingAverage()
			if !math.AbsVal(tc.want - actualAverage) < tc.tolerance {
				t.Errorf("Expected approximate moving average of %f, but got %f", tc.want, actualAverage)
			}
		})
	}
}

func (s) TestWrapSyscallConn(t *testing.T) {
	sc := &syscallConn{}
	nsc := &nonSyscallConn{}

	wrapConn := WrapSyscallConn(sc, nsc)
	if _, ok := wrapConn.(syscall.Conn); !ok {
		t.Errorf("returned conn (type %T) doesn't implement syscall.Conn, want implement", wrapConn)
	}
}

//------------------------------------------------------------------------------

// JSONArrAppend adds the provided JSON values to the end of the array at the given path.
// For more information, see https://redis.io/commands/json.arrappend
func (c cmdable) JSONArrAppend(ctx context.Context, key, path string, values ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRAPPEND", key, path}
	args = append(args, values...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrIndex searches for the first occurrence of the provided JSON value in the array at the given path.
// For more information, see https://redis.io/commands/json.arrindex
func (c cmdable) JSONArrIndex(ctx context.Context, key, path string, value ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRINDEX", key, path}
	args = append(args, value...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrIndexWithArgs searches for the first occurrence of a JSON value in an array while allowing the start and
// stop options to be provided.
// For more information, see https://redis.io/commands/json.arrindex
func (c cmdable) JSONArrIndexWithArgs(ctx context.Context, key, path string, options *JSONArrIndexArgs, value ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRINDEX", key, path}
	args = append(args, value...)

	if options != nil {
		args = append(args, options.Start)
		if options.Stop != nil {
			args = append(args, *options.Stop)
		}
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrInsert inserts the JSON values into the array at the specified path before the index (shifts to the right).
// For more information, see https://redis.io/commands/json.arrinsert
func (c cmdable) JSONArrInsert(ctx context.Context, key, path string, index int64, values ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRINSERT", key, path, index}
	args = append(args, values...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrLen reports the length of the JSON array at the specified path in the given key.
// For more information, see https://redis.io/commands/json.arrlen
func (c cmdable) JSONArrLen(ctx context.Context, key, path string) *IntSliceCmd {
	args := []interface{}{"JSON.ARRLEN", key, path}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrPop removes and returns an element from the specified index in the array.
// For more information, see https://redis.io/commands/json.arrpop
func (c cmdable) JSONArrPop(ctx context.Context, key, path string, index int) *StringSliceCmd {
	args := []interface{}{"JSON.ARRPOP", key, path, index}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrTrim trims an array to contain only the specified inclusive range of elements.
// For more information, see https://redis.io/commands/json.arrtrim
func (c cmdable) JSONArrTrim(ctx context.Context, key, path string) *IntSliceCmd {
	args := []interface{}{"JSON.ARRTRIM", key, path}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrTrimWithArgs trims an array to contain only the specified inclusive range of elements.
// For more information, see https://redis.io/commands/json.arrtrim
func (c cmdable) JSONArrTrimWithArgs(ctx context.Context, key, path string, options *JSONArrTrimArgs) *IntSliceCmd {
	args := []interface{}{"JSON.ARRTRIM", key, path}

	if options != nil {
		args = append(args, options.Start)

		if options.Stop != nil {
			args = append(args, *options.Stop)
		}
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONClear clears container values (arrays/objects) and sets numeric values to 0.
// For more information, see https://redis.io/commands/json.clear
func (c cmdable) JSONClear(ctx context.Context, key, path string) *IntCmd {
	args := []interface{}{"JSON.CLEAR", key, path}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONDebugMemory reports a value's memory usage in bytes (unimplemented)
// For more information, see https://redis.io/commands/json.debug-memory
func (c cmdable) JSONDebugMemory(ctx context.Context, key, path string) *IntCmd {
	panic("not implemented")
}

// JSONDel deletes a value.
// For more information, see https://redis.io/commands/json.del
func (c cmdable) JSONDel(ctx context.Context, key, path string) *IntCmd {
	args := []interface{}{"JSON.DEL", key, path}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONForget deletes a value.
// For more information, see https://redis.io/commands/json.forget
func (c cmdable) JSONForget(ctx context.Context, key, path string) *IntCmd {
	args := []interface{}{"JSON.FORGET", key, path}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONGet returns the value at path in JSON serialized form. JSON.GET returns an
// array of strings. This function parses out the wrapping array but leaves the
// internal strings unprocessed by default (see Val())
// For more information - https://redis.io/commands/json.get/
func (c cmdable) JSONGet(ctx context.Context, key string, paths ...string) *JSONCmd {
	args := make([]interface{}, len(paths)+2)
	args[0] = "JSON.GET"
	args[1] = key
	for n, path := range paths {
		args[n+2] = path
	}
	cmd := newJSONCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type JSONGetArgs struct {
	Indent  string
	Newline string
	Space   string
}

// JSONGetWithArgs - Retrieves the value of a key from a JSON document.
// This function also allows for specifying additional options such as:
// Indention, NewLine and Space
// For more information - https://redis.io/commands/json.get/
func (c cmdable) JSONGetWithArgs(ctx context.Context, key string, options *JSONGetArgs, paths ...string) *JSONCmd {
	args := []interface{}{"JSON.GET", key}
	if options != nil {
		if options.Indent != "" {
			args = append(args, "INDENT", options.Indent)
		}
		if options.Newline != "" {
			args = append(args, "NEWLINE", options.Newline)
		}
		if options.Space != "" {
			args = append(args, "SPACE", options.Space)
		}
		for _, path := range paths {
			args = append(args, path)
		}
	}
	cmd := newJSONCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONMerge merges a given JSON value into matching paths.
// For more information, see https://redis.io/commands/json.merge
func (c cmdable) JSONMerge(ctx context.Context, key, path string, value string) *StatusCmd {
	args := []interface{}{"JSON.MERGE", key, path, value}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONMGet returns the values at the specified path from multiple key arguments.
// Note - the arguments are reversed when compared with `JSON.MGET` as we want
// to follow the pattern of having the last argument be variable.
// For more information, see https://redis.io/commands/json.mget
func (c cmdable) JSONMGet(ctx context.Context, path string, keys ...string) *JSONSliceCmd {
	args := make([]interface{}, len(keys)+1)
	args[0] = "JSON.MGET"
	for n, key := range keys {
		args[n+1] = key
	}
	args = append(args, path)
	cmd := NewJSONSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONMSetArgs sets or updates one or more JSON values according to the specified key-path-value triplets.
// For more information, see https://redis.io/commands/json.mset
func (c cmdable) JSONMSetArgs(ctx context.Context, docs []JSONSetArgs) *StatusCmd {
	args := []interface{}{"JSON.MSET"}
	for _, doc := range docs {
		args = append(args, doc.Key, doc.Path, doc.Value)
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) JSONMSet(ctx context.Context, params ...interface{}) *StatusCmd {
	args := []interface{}{"JSON.MSET"}
	args = append(args, params...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONNumIncrBy increments the number value stored at the specified path by the provided number.
// For more information, see https://redis.io/docs/latest/commands/json.numincrby/
func (c cmdable) JSONNumIncrBy(ctx context.Context, key, path string, value float64) *JSONCmd {
	args := []interface{}{"JSON.NUMINCRBY", key, path, value}
	cmd := newJSONCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONObjKeys returns the keys in the object that's referenced by the specified path.
// For more information, see https://redis.io/commands/json.objkeys
func (c cmdable) JSONObjKeys(ctx context.Context, key, path string) *SliceCmd {
	args := []interface{}{"JSON.OBJKEYS", key, path}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONObjLen reports the number of keys in the JSON object at the specified path in the given key.
// For more information, see https://redis.io/commands/json.objlen
func (c cmdable) JSONObjLen(ctx context.Context, key, path string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.OBJLEN", key, path}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONSet sets the JSON value at the given path in the given key. The value must be something that
// can be marshaled to JSON (using encoding/JSON) unless the argument is a string or a []byte when we assume that
// it can be passed directly as JSON.
// For more information, see https://redis.io/commands/json.set
func (c cmdable) JSONSet(ctx context.Context, key, path string, value interface{}) *StatusCmd {
	return c.JSONSetMode(ctx, key, path, value, "")
}

// JSONSetMode sets the JSON value at the given path in the given key and allows the mode to be set
// (the mode value must be "XX" or "NX"). The value must be something that can be marshaled to JSON (using encoding/JSON) unless
// the argument is a string or []byte when we assume that it can be passed directly as JSON.
// For more information, see https://redis.io/commands/json.set
func (c cmdable) JSONSetMode(ctx context.Context, key, path string, value interface{}, mode string) *StatusCmd {
	var bytes []byte
	var err error
	switch v := value.(type) {
	case string:
		bytes = []byte(v)
	case []byte:
		bytes = v
	default:
		bytes, err = json.Marshal(v)
	}
	args := []interface{}{"JSON.SET", key, path, util.BytesToString(bytes)}
	if mode != "" {
		switch strings.ToUpper(mode) {
		case "XX", "NX":
			args = append(args, strings.ToUpper(mode))

		default:
			panic("redis: JSON.SET mode must be NX or XX")
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	if err != nil {
		cmd.SetErr(err)
	} else {
		_ = c(ctx, cmd)
	}
	return cmd
}

// JSONStrAppend appends the JSON-string values to the string at the specified path.
// For more information, see https://redis.io/commands/json.strappend
func (c cmdable) JSONStrAppend(ctx context.Context, key, path, value string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.STRAPPEND", key, path, value}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONStrLen reports the length of the JSON String at the specified path in the given key.
// For more information, see https://redis.io/commands/json.strlen
func (c cmdable) JSONStrLen(ctx context.Context, key, path string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.STRLEN", key, path}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONToggle toggles a Boolean value stored at the specified path.
// For more information, see https://redis.io/commands/json.toggle
func (c cmdable) JSONToggle(ctx context.Context, key, path string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.TOGGLE", key, path}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONType reports the type of JSON value at the specified path.
// For more information, see https://redis.io/commands/json.type
func (c cmdable) JSONType(ctx context.Context, key, path string) *JSONSliceCmd {
	args := []interface{}{"JSON.TYPE", key, path}
	cmd := NewJSONSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
