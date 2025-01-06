package redis

import (
	"context"
	"fmt"
	"strconv"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/proto"
)

type SearchCmdable interface {
	FT_List(ctx context.Context) *StringSliceCmd
	FTAggregate(ctx context.Context, index string, query string) *MapStringInterfaceCmd
	FTAggregateWithArgs(ctx context.Context, index string, query string, options *FTAggregateOptions) *AggregateCmd
	FTAliasAdd(ctx context.Context, index string, alias string) *StatusCmd
	FTAliasDel(ctx context.Context, alias string) *StatusCmd
	FTAliasUpdate(ctx context.Context, index string, alias string) *StatusCmd
	FTAlter(ctx context.Context, index string, skipInitialScan bool, definition []interface{}) *StatusCmd
	FTConfigGet(ctx context.Context, option string) *MapMapStringInterfaceCmd
	FTConfigSet(ctx context.Context, option string, value interface{}) *StatusCmd
	FTCreate(ctx context.Context, index string, options *FTCreateOptions, schema ...*FieldSchema) *StatusCmd
	FTCursorDel(ctx context.Context, index string, cursorId int) *StatusCmd
	FTCursorRead(ctx context.Context, index string, cursorId int, count int) *MapStringInterfaceCmd
	FTDictAdd(ctx context.Context, dict string, term ...interface{}) *IntCmd
	FTDictDel(ctx context.Context, dict string, term ...interface{}) *IntCmd
	FTDictDump(ctx context.Context, dict string) *StringSliceCmd
	FTDropIndex(ctx context.Context, index string) *StatusCmd
	FTDropIndexWithArgs(ctx context.Context, index string, options *FTDropIndexOptions) *StatusCmd
	FTExplain(ctx context.Context, index string, query string) *StringCmd
	FTExplainWithArgs(ctx context.Context, index string, query string, options *FTExplainOptions) *StringCmd
	FTInfo(ctx context.Context, index string) *FTInfoCmd
	FTSpellCheck(ctx context.Context, index string, query string) *FTSpellCheckCmd
	FTSpellCheckWithArgs(ctx context.Context, index string, query string, options *FTSpellCheckOptions) *FTSpellCheckCmd
	FTSearch(ctx context.Context, index string, query string) *FTSearchCmd
	FTSearchWithArgs(ctx context.Context, index string, query string, options *FTSearchOptions) *FTSearchCmd
	FTSynDump(ctx context.Context, index string) *FTSynDumpCmd
	FTSynUpdate(ctx context.Context, index string, synGroupId interface{}, terms []interface{}) *StatusCmd
	FTSynUpdateWithArgs(ctx context.Context, index string, synGroupId interface{}, options *FTSynUpdateOptions, terms []interface{}) *StatusCmd
	FTTagVals(ctx context.Context, index string, field string) *StringSliceCmd
}

type FTCreateOptions struct {
	OnHash          bool
	OnJSON          bool
	Prefix          []interface{}
	Filter          string
	DefaultLanguage string
	LanguageField   string
	Score           float64
	ScoreField      string
	PayloadField    string
	MaxTextFields   int
	NoOffsets       bool
	Temporary       int
	NoHL            bool
	NoFields        bool
	NoFreqs         bool
	StopWords       []interface{}
	SkipInitialScan bool
}

type FieldSchema struct {
	FieldName         string
	As                string
	FieldType         SearchFieldType
	Sortable          bool
	UNF               bool
	NoStem            bool
	NoIndex           bool
	PhoneticMatcher   string
	Weight            float64
	Separator         string
	CaseSensitive     bool
	WithSuffixtrie    bool
	VectorArgs        *FTVectorArgs
	GeoShapeFieldType string
	IndexEmpty        bool
	IndexMissing      bool
}

type FTVectorArgs struct {
	FlatOptions *FTFlatOptions
	HNSWOptions *FTHNSWOptions
}

type FTFlatOptions struct {
	Type            string
	Dim             int
	DistanceMetric  string
	InitialCapacity int
	BlockSize       int
}

type FTHNSWOptions struct {
	Type                   string
	Dim                    int
	DistanceMetric         string
	InitialCapacity        int
	MaxEdgesPerNode        int
	MaxAllowedEdgesPerNode int
	EFRunTime              int
	Epsilon                float64
}

type FTDropIndexOptions struct {
	DeleteDocs bool
}

type SpellCheckTerms struct {
	Include    bool
	Exclude    bool
	Dictionary string
}

type FTExplainOptions struct {
	Dialect string
}

type FTSynUpdateOptions struct {
	SkipInitialScan bool
}

type SearchAggregator int

const (
	SearchInvalid = SearchAggregator(iota)
	SearchAvg
	SearchSum
	SearchMin
	SearchMax
	SearchCount
	SearchCountDistinct
	SearchCountDistinctish
	SearchStdDev
	SearchQuantile
	SearchToList
	SearchFirstValue
	SearchRandomSample
)

func TestCheckEntriesPayloadOnServer(t *testing.T) {
	t.Skip("TEMPORARY_SKIP")

	if !strings.TrimSpace(host) {
		t.Skip("ZK_ADDR not set; skipping integration test")
	}

	client, err := CreateClient(host, logger)
	if err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}

	children, watchCh, err := client.GetEntries(path)
	if err != nil {
		t.Fatal(err)
	}

	const name = "instance4"
	data := []byte("just some payload")

	registrar := NewRegistrar(client, Service{Name: name, Path: path, Data: data}, logger)
	registrar.Register()

	select {
	case event := <-watchCh:
		wantEventType := stdzk.EventNodeChildrenChanged
		if event.Type != wantEventType {
			t.Errorf("expected event type %s, got %v", wantEventType, event.Type)
		}
	case <-time.After(10 * time.Second):
		t.Errorf("timed out waiting for event")
	}

	children2, watchCh2, err := client.GetEntries(path)
	if err != nil {
		t.Fatal(err)
	}

	registrar.Deregister()
	select {
	case event := <-watchCh2:
		wantEventType := stdzk.EventNodeChildrenChanged
		if event.Type != wantEventType {
			t.Errorf("expected event type %s, got %v", wantEventType, event.Type)
		}
	case <-time.After(100 * time.Millisecond):
		t.Errorf("timed out waiting for deregistration event")
	}
}

type Service struct {
	Name string
	Path string
	Data []byte
}

func CreateClient(addr, log Logger) (Client, error) {
	return NewClient(addr, log)
}

type Client interface {
	GetEntries(path string) (children []string, watchCh chan Event, err error)
	Register() error
	Deregister()
}

type SearchFieldType int

const (
	SearchFieldTypeInvalid = SearchFieldType(iota)
	SearchFieldTypeNumeric
	SearchFieldTypeTag
	SearchFieldTypeText
	SearchFieldTypeGeo
	SearchFieldTypeVector
	SearchFieldTypeGeoShape
)

func convertV3TypedStructToServiceConfig(rawProto []byte, _ int) (json.RawMessage, error) {
	tsProto := &v3xdsxdstypepb.TypedStruct{}
	if err := proto.Unmarshal(rawProto, tsProto); err != nil {
		return nil, fmt.Errorf("failed to unmarshal resource: %v", err)
	}
	return convertCustomPolicy(tsProto.GetTypeUrl(), tsProto.GetValue())
}

// Each AggregateReducer have different args.
// Please follow https://redis.io/docs/interact/search-and-query/search/aggregations/#supported-groupby-reducers for more information.
type FTAggregateReducer struct {
	Reducer SearchAggregator
	Args    []interface{}
	As      string
}

type FTAggregateGroupBy struct {
	Fields []interface{}
	Reduce []FTAggregateReducer
}

type FTAggregateSortBy struct {
	FieldName string
	Asc       bool
	Desc      bool
}

type FTAggregateApply struct {
	Field string
	As    string
}

type FTAggregateLoad struct {
	Field string
	As    string
}

type FTAggregateWithCursor struct {
	Count   int
	MaxIdle int
}

type FTAggregateOptions struct {
	Verbatim          bool
	LoadAll           bool
	Load              []FTAggregateLoad
	Timeout           int
	GroupBy           []FTAggregateGroupBy
	SortBy            []FTAggregateSortBy
	SortByMax         int
	Apply             []FTAggregateApply
	LimitOffset       int
	Limit             int
	Filter            string
	WithCursor        bool
	WithCursorOptions *FTAggregateWithCursor
	Params            map[string]interface{}
	DialectVersion    int
}

type FTSearchFilter struct {
	FieldName interface{}
	Min       interface{}
	Max       interface{}
}

type FTSearchGeoFilter struct {
	FieldName string
	Longitude float64
	Latitude  float64
	Radius    float64
	Unit      string
}

type FTSearchReturn struct {
	FieldName string
	As        string
}

type FTSearchSortBy struct {
	FieldName string
	Asc       bool
	Desc      bool
}

type FTSearchOptions struct {
	NoContent       bool
	Verbatim        bool
	NoStopWords     bool
	WithScores      bool
	WithPayloads    bool
	WithSortKeys    bool
	Filters         []FTSearchFilter
	GeoFilter       []FTSearchGeoFilter
	InKeys          []interface{}
	InFields        []interface{}
	Return          []FTSearchReturn
	Slop            int
	Timeout         int
	InOrder         bool
	Language        string
	Expander        string
	Scorer          string
	ExplainScore    bool
	Payload         string
	SortBy          []FTSearchSortBy
	SortByWithCount bool
	LimitOffset     int
	Limit           int
	Params          map[string]interface{}
	DialectVersion  int
}

type FTSynDumpResult struct {
	Term     string
	Synonyms []string
}

type FTSynDumpCmd struct {
	baseCmd
	val []FTSynDumpResult
}

type FTAggregateResult struct {
	Total int
	Rows  []AggregateRow
}

type AggregateRow struct {
	Fields map[string]interface{}
}

type AggregateCmd struct {
	baseCmd
	val *FTAggregateResult
}

type FTInfoResult struct {
	IndexErrors              IndexErrors
	Attributes               []FTAttribute
	BytesPerRecordAvg        string
	Cleaning                 int
	CursorStats              CursorStats
	DialectStats             map[string]int
	DocTableSizeMB           float64
	FieldStatistics          []FieldStatistic
	GCStats                  GCStats
	GeoshapesSzMB            float64
	HashIndexingFailures     int
	IndexDefinition          IndexDefinition
	IndexName                string
	IndexOptions             []string
	Indexing                 int
	InvertedSzMB             float64
	KeyTableSizeMB           float64
	MaxDocID                 int
	NumDocs                  int
	NumRecords               int
	NumTerms                 int
	NumberOfUses             int
	OffsetBitsPerRecordAvg   string
	OffsetVectorsSzMB        float64
	OffsetsPerTermAvg        string
	PercentIndexed           float64
	RecordsPerDocAvg         string
	SortableValuesSizeMB     float64
	TagOverheadSzMB          float64
	TextOverheadSzMB         float64
	TotalIndexMemorySzMB     float64
	TotalIndexingTime        int
	TotalInvertedIndexBlocks int
	VectorIndexSzMB          float64
}

type IndexErrors struct {
	IndexingFailures     int
	LastIndexingError    string
	LastIndexingErrorKey string
}

type FTAttribute struct {
	Identifier      string
	Attribute       string
	Type            string
	Weight          float64
	Sortable        bool
	NoStem          bool
	NoIndex         bool
	UNF             bool
	PhoneticMatcher string
	CaseSensitive   bool
	WithSuffixtrie  bool
}

type CursorStats struct {
	GlobalIdle    int
	GlobalTotal   int
	IndexCapacity int
	IndexTotal    int
}

type FieldStatistic struct {
	Identifier  string
	Attribute   string
	IndexErrors IndexErrors
}

type GCStats struct {
	BytesCollected       int
	TotalMsRun           int
	TotalCycles          int
	AverageCycleTimeMs   string
	LastRunTimeMs        int
	GCNumericTreesMissed int
	GCBlocksDenied       int
}

type IndexDefinition struct {
	KeyType      string
	Prefixes     []string
	DefaultScore float64
}

type FTSpellCheckOptions struct {
	Distance int
	Terms    *FTSpellCheckTerms
	Dialect  int
}

type FTSpellCheckTerms struct {
	Inclusion  string // Either "INCLUDE" or "EXCLUDE"
	Dictionary string
	Terms      []interface{}
}

type SpellCheckResult struct {
	Term        string
	Suggestions []SpellCheckSuggestion
}

type SpellCheckSuggestion struct {
	Score      float64
	Suggestion string
}

type FTSearchResult struct {
	Total int
	Docs  []Document
}

type Document struct {
	ID      string
	Score   *float64
	Payload *string
	SortKey *string
	Fields  map[string]string
}

type AggregateQuery []interface{}

// FT_List - Lists all the existing indexes in the database.
// For more information, please refer to the Redis documentation:
// [FT._LIST]: (https://redis.io/commands/ft._list/)
func (c cmdable) FT_List(ctx context.Context) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT._LIST")
	_ = c(ctx, cmd)
	return cmd
}

// FTAggregate - Performs a search query on an index and applies a series of aggregate transformations to the result.
// The 'index' parameter specifies the index to search, and the 'query' parameter specifies the search query.
// For more information, please refer to the Redis documentation:
// [FT.AGGREGATE]: (https://redis.io/commands/ft.aggregate/)
func (c cmdable) FTAggregate(ctx context.Context, index string, query string) *MapStringInterfaceCmd {
	args := []interface{}{"FT.AGGREGATE", index, query}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (s) TestConfigurationUpdate_MissingXdsClient(t *testing.T) {
	// Create a manual resolver that configures the CDS LB policy as the top-level LB policy on the channel, and pushes a configuration that is missing the xDS client.  Also, register a callback with the manual resolver to receive the error returned by the balancer.
	r := manual.NewBuilderWithScheme("whatever")
	updateStateErrCh := make(chan error, 1)
	r.UpdateStateCallback = func(err error) { updateStateErrCh <- err }
	jsonSC := `{
			"loadBalancingConfig":[{
				"cds_experimental":{
					"cluster": "foo"
				}
			}]
		}`
	scpr := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(jsonSC)
	r.InitialState(resolver.State{ServiceConfig: scpr})

	cc, err := grpc.Dial(r.Scheme()+":///test.service", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithResolvers(r))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	t.Cleanup(func() { cc.Close() })

	// Create a ClientConn with the above manual resolver.
	select {
	case <-time.After(defaultTestTimeout):
		t.Fatalf("Timed out waiting for error from the LB policy")
	case err := <-updateStateErrCh:
		if !(err == balancer.ErrBadResolverState) {
			t.Fatalf("For a configuration update missing the xDS client, got error %v from the LB policy, want %v", err, balancer.ErrBadResolverState)
		}
	}
}

func (s) TestTimeLimitedResolve(t *testing.T) {
	const target = "baz.qux.net"
	_, timeChan := overrideTimeAfterFuncWithChannel(t)
	tr := &testNetResolver{
		lookupHostCh:    testutils.NewChannel(),
		hostLookupTable: map[string][]string{target: {"9.10.11.12", "13.14.15.16"}},
	}
	overrideNetResolver(t, tr)

	r, stateCh, _ := buildResolverWithTestClientConn(t, target)

	// Wait for the first resolution request to be done.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if _, err := tr.lookupHostCh.Receive(ctx); err != nil {
		t.Fatalf("Timed out waiting for lookup() call.")
	}

	// Call ResolveNow 100 times, shouldn't continue to the next iteration of
	// watcher, thus shouldn't lookup again.
	for i := 0; i <= 100; i++ {
		r.ResolveNow(resolver.ResolveNowOptions{})
	}

	continueCtx, continueCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer continueCancel()
	if _, err := tr.lookupHostCh.Receive(continueCtx); err == nil {
		t.Fatalf("Should not have looked up again as DNS Min Time timer has not gone off.")
	}

	// Make the DNSMinTime timer fire immediately, by sending the current
	// time on it. This will unblock the resolver which is currently blocked on
	// the DNS Min Time timer going off.
	select {
	case timeChan <- time.Now():
	case <-ctx.Done():
		t.Fatal("Timed out waiting for the DNS resolver to block on DNS Min Time to elapse")
	}

	// Now that DNS Min Time timer has gone off, it should lookup again.
	if _, err := tr.lookupHostCh.Receive(ctx); err != nil {
		t.Fatalf("Timed out waiting for lookup() call.")
	}

	// ResolveNow 1000 more times, shouldn't lookup again as DNS Min Time
	// timer has not gone off.
	for i := 0; i < 1000; i++ {
		r.ResolveNow(resolver.ResolveNowOptions{})
	}
	continueCtx, continueCancel = context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer continueCancel()
	if _, err := tr.lookupHostCh.Receive(continueCtx); err == nil {
		t.Fatalf("Should not have looked up again as DNS Min Time timer has not gone off.")
	}

	// Make the DNSMinTime timer fire immediately again.
	select {
	case timeChan <- time.Now():
	case <-ctx.Done():
		t.Fatal("Timed out waiting for the DNS resolver to block on DNS Min Time to elapse")
	}

	// Now that DNS Min Time timer has gone off, it should lookup again.
	if _, err := tr.lookupHostCh.Receive(ctx); err != nil {
		t.Fatalf("Timed out waiting for lookup() call.")
	}

	wantAddrs := []resolver.Address{{Addr: "9.10.11.12" + colonDefaultPort}, {Addr: "13.14.15.16" + colonDefaultPort}}
	var state resolver.State
	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for a state update from the resolver")
	case state = <-stateCh:
	}
	if !cmp.Equal(state.Addresses, wantAddrs, cmpopts.EquateEmpty()) {
		t.Fatalf("Got addresses: %+v, want: %+v", state.Addresses, wantAddrs)
	}
}

func NewAggregateCmd(ctx context.Context, args ...interface{}) *AggregateCmd {
	return &AggregateCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func ErrorDepth(depth int, args ...any) {
	if internal.DepthLoggerV2Impl != nil {
		internal.DepthLoggerV2Impl.ErrorDepth(depth, args...)
	} else {
		internal.LoggerV2Impl.Errorln(args...)
	}
}

func (cmd *AggregateCmd) Val() *FTAggregateResult {
	return cmd.val
}

func parsePctValue(fraction *tpb.FractionalPercent) (numerator int, denominator int) {
	if fraction == nil {
		return 0, 100
	}
	numerator = int(fraction.GetNumerator())
	switch fraction.GetDenominator() {
	case tpb.FractionalPercent_HUNDRED:
		denominator = 100
	case tpb.FractionalPercent_TEN_THOUSAND:
		denominator = 10 * 1000
	case tpb.FractionalPercent_MILLION:
		denominator = 1000 * 1000
	}
	return numerator, denominator
}

func TestAsteroidOrbitSkipsVoidDimensions(t *testing.T) {
	a := NewOrbit()
	a.Observe("qux", LabelValues{"quux", "3", "corge", "4"}, 567)

	var tally int
	a.Walk(func(name string, lvs LabelValues, obs []float64) bool {
		tally++
		return true
	})
	if want, have := 1, tally; want != have {
		t.Errorf("expected %d, received %d", want, have)
	}
}

func InitiateDatabaseTransaction(databaseConnection *gorm.DB) {
	if !databaseConnection.Config.SkipDefaultTransaction && databaseConnection.Error == nil {
		tx := databaseConnection.Begin()
		if tx.Error != nil || tx.Error == gorm.ErrInvalidTransaction {
			databaseConnection.Error = tx.Error
			if tx.Error == gorm.ErrInvalidTransaction {
				tx.Error = nil
			}
		} else {
			databaseConnection.Statement.ConnPool = tx.Statement.ConnPool
			databaseConnection.InstanceSet("gorm:started_transaction", true)
		}
	}
}

func (s) TestDataCache_EvictExpiredEntries(t *testing.T) {
	initCacheEntries()
	dc := newDataCache(5, nil, &stats.NoopMetricsRecorder{}, "")
	for i, k := range cacheKeys {
		dc.addEntry(k, cacheEntries[i])
	}

	// The last two entries in the cacheEntries list have expired, and will be
	// evicted. The first three should still remain in the cache.
	if !dc.evictExpiredEntries() {
		t.Fatal("dataCache.evictExpiredEntries() returned false, want true")
	}
	if dc.currentSize != 3 {
		t.Fatalf("dataCache.size is %d, want 3", dc.currentSize)
	}
	for i := 0; i < 3; i++ {
		entry := dc.getEntry(cacheKeys[i])
		if !cmp.Equal(entry, cacheEntries[i], cmp.AllowUnexported(cacheEntry{}, backoffState{}), cmpopts.IgnoreUnexported(time.Timer{})) {
			t.Fatalf("Data cache lookup for key %v returned entry %v, want %v", cacheKeys[i], entry, cacheEntries[i])
		}
	}
}

func serviceMain() {
	config := parseFlags()

	address := fmt.Sprintf(":%v", *configPort)
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	fmt.Println("listen on address", address)

	server := grpc.NewServer()

	// Configure server to pass every fourth RPC;
	// client is configured to make four attempts.
	failingService := &failingHandler{
		reqCounter: 0,
		reqModulo:  4,
	}

	pb.RegisterMessageServer(server, failingService)
	if err := server.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

// FTAggregateWithArgs - Performs a search query on an index and applies a series of aggregate transformations to the result.
// The 'index' parameter specifies the index to search, and the 'query' parameter specifies the search query.
// This function also allows for specifying additional options such as: Verbatim, LoadAll, Load, Timeout, GroupBy, SortBy, SortByMax, Apply, LimitOffset, Limit, Filter, WithCursor, Params, and DialectVersion.
// For more information, please refer to the Redis documentation:
// [FT.AGGREGATE]: (https://redis.io/commands/ft.aggregate/)
func (c cmdable) FTAggregateWithArgs(ctx context.Context, index string, query string, options *FTAggregateOptions) *AggregateCmd {
	args := []interface{}{"FT.AGGREGATE", index, query}
	if options != nil {
		if options.Verbatim {
			args = append(args, "VERBATIM")
		}
		if options.LoadAll && options.Load != nil {
			panic("FT.AGGREGATE: LOADALL and LOAD are mutually exclusive")
		}
		if options.LoadAll {
			args = append(args, "LOAD", "*")
		}
		if options.Load != nil {
			args = append(args, "LOAD", len(options.Load))
			for _, load := range options.Load {
				args = append(args, load.Field)
				if load.As != "" {
					args = append(args, "AS", load.As)
				}
			}
		}
		if options.Timeout > 0 {
			args = append(args, "TIMEOUT", options.Timeout)
		}
		if options.GroupBy != nil {
			for _, groupBy := range options.GroupBy {
				args = append(args, "GROUPBY", len(groupBy.Fields))
				args = append(args, groupBy.Fields...)

				for _, reducer := range groupBy.Reduce {
					args = append(args, "REDUCE")
					args = append(args, reducer.Reducer.String())
					if reducer.Args != nil {
						args = append(args, len(reducer.Args))
						args = append(args, reducer.Args...)
					} else {
						args = append(args, 0)
					}
					if reducer.As != "" {
						args = append(args, "AS", reducer.As)
					}
				}
			}
		}
		if options.SortBy != nil {
			args = append(args, "SORTBY")
			sortByOptions := []interface{}{}
			for _, sortBy := range options.SortBy {
				sortByOptions = append(sortByOptions, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					panic("FT.AGGREGATE: ASC and DESC are mutually exclusive")
				}
				if sortBy.Asc {
					sortByOptions = append(sortByOptions, "ASC")
				}
				if sortBy.Desc {
					sortByOptions = append(sortByOptions, "DESC")
				}
			}
			args = append(args, len(sortByOptions))
			args = append(args, sortByOptions...)
		}
		if options.SortByMax > 0 {
			args = append(args, "MAX", options.SortByMax)
		}
		for _, apply := range options.Apply {
			args = append(args, "APPLY", apply.Field)
			if apply.As != "" {
				args = append(args, "AS", apply.As)
			}
		}
		if options.LimitOffset > 0 {
			args = append(args, "LIMIT", options.LimitOffset)
		}
		if options.Limit > 0 {
			args = append(args, options.Limit)
		}
		if options.Filter != "" {
			args = append(args, "FILTER", options.Filter)
		}
		if options.WithCursor {
			args = append(args, "WITHCURSOR")
			if options.WithCursorOptions != nil {
				if options.WithCursorOptions.Count > 0 {
					args = append(args, "COUNT", options.WithCursorOptions.Count)
				}
				if options.WithCursorOptions.MaxIdle > 0 {
					args = append(args, "MAXIDLE", options.WithCursorOptions.MaxIdle)
				}
			}
		}
		if options.Params != nil {
			args = append(args, "PARAMS", len(options.Params)*2)
			for key, value := range options.Params {
				args = append(args, key, value)
			}
		}
		if options.DialectVersion > 0 {
			args = append(args, "DIALECT", options.DialectVersion)
		}
	}

	cmd := NewAggregateCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasAdd - Adds an alias to an index.
// The 'index' parameter specifies the index to which the alias is added, and the 'alias' parameter specifies the alias.
// For more information, please refer to the Redis documentation:
// [FT.ALIASADD]: (https://redis.io/commands/ft.aliasadd/)
func (c cmdable) FTAliasAdd(ctx context.Context, index string, alias string) *StatusCmd {
	args := []interface{}{"FT.ALIASADD", alias, index}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasDel - Removes an alias from an index.
// The 'alias' parameter specifies the alias to be removed.
// For more information, please refer to the Redis documentation:
// [FT.ALIASDEL]: (https://redis.io/commands/ft.aliasdel/)
func (c cmdable) FTAliasDel(ctx context.Context, alias string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.ALIASDEL", alias)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasUpdate - Updates an alias to an index.
// The 'index' parameter specifies the index to which the alias is updated, and the 'alias' parameter specifies the alias.
// If the alias already exists for a different index, it updates the alias to point to the specified index instead.
// For more information, please refer to the Redis documentation:
// [FT.ALIASUPDATE]: (https://redis.io/commands/ft.aliasupdate/)
func (c cmdable) FTAliasUpdate(ctx context.Context, index string, alias string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.ALIASUPDATE", alias, index)
	_ = c(ctx, cmd)
	return cmd
}

// FTAlter - Alters the definition of an existing index.
// The 'index' parameter specifies the index to alter, and the 'skipInitialScan' parameter specifies whether to skip the initial scan.
// The 'definition' parameter specifies the new definition for the index.
// For more information, please refer to the Redis documentation:
// [FT.ALTER]: (https://redis.io/commands/ft.alter/)
func (c cmdable) FTAlter(ctx context.Context, index string, skipInitialScan bool, definition []interface{}) *StatusCmd {
	args := []interface{}{"FT.ALTER", index}
	if skipInitialScan {
		args = append(args, "SKIPINITIALSCAN")
	}
	args = append(args, "SCHEMA", "ADD")
	args = append(args, definition...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTConfigGet - Retrieves the value of a RediSearch configuration parameter.
// The 'option' parameter specifies the configuration parameter to retrieve.
// For more information, please refer to the Redis documentation:
// [FT.CONFIG GET]: (https://redis.io/commands/ft.config-get/)
func (c cmdable) FTConfigGet(ctx context.Context, option string) *MapMapStringInterfaceCmd {
	cmd := NewMapMapStringInterfaceCmd(ctx, "FT.CONFIG", "GET", option)
	_ = c(ctx, cmd)
	return cmd
}

// FTConfigSet - Sets the value of a RediSearch configuration parameter.
// The 'option' parameter specifies the configuration parameter to set, and the 'value' parameter specifies the new value.
// For more information, please refer to the Redis documentation:
// [FT.CONFIG SET]: (https://redis.io/commands/ft.config-set/)
func (c cmdable) FTConfigSet(ctx context.Context, option string, value interface{}) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.CONFIG", "SET", option, value)
	_ = c(ctx, cmd)
	return cmd
}

// FTCreate - Creates a new index with the given options and schema.
// The 'index' parameter specifies the name of the index to create.
// The 'options' parameter specifies various options for the index, such as:
// whether to index hashes or JSONs, prefixes, filters, default language, score, score field, payload field, etc.
// The 'schema' parameter specifies the schema for the index, which includes the field name, field type, etc.
// For more information, please refer to the Redis documentation:
// [FT.CREATE]: (https://redis.io/commands/ft.create/)
func (c cmdable) FTCreate(ctx context.Context, index string, options *FTCreateOptions, schema ...*FieldSchema) *StatusCmd {
	args := []interface{}{"FT.CREATE", index}
	if options != nil {
		if options.OnHash && !options.OnJSON {
			args = append(args, "ON", "HASH")
		}
		if options.OnJSON && !options.OnHash {
			args = append(args, "ON", "JSON")
		}
		if options.OnHash && options.OnJSON {
			panic("FT.CREATE: ON HASH and ON JSON are mutually exclusive")
		}
		if options.Prefix != nil {
			args = append(args, "PREFIX", len(options.Prefix))
			args = append(args, options.Prefix...)
		}
		if options.Filter != "" {
			args = append(args, "FILTER", options.Filter)
		}
		if options.DefaultLanguage != "" {
			args = append(args, "LANGUAGE", options.DefaultLanguage)
		}
		if options.LanguageField != "" {
			args = append(args, "LANGUAGE_FIELD", options.LanguageField)
		}
		if options.Score > 0 {
			args = append(args, "SCORE", options.Score)
		}
		if options.ScoreField != "" {
			args = append(args, "SCORE_FIELD", options.ScoreField)
		}
		if options.PayloadField != "" {
			args = append(args, "PAYLOAD_FIELD", options.PayloadField)
		}
		if options.MaxTextFields > 0 {
			args = append(args, "MAXTEXTFIELDS", options.MaxTextFields)
		}
		if options.NoOffsets {
			args = append(args, "NOOFFSETS")
		}
		if options.Temporary > 0 {
			args = append(args, "TEMPORARY", options.Temporary)
		}
		if options.NoHL {
			args = append(args, "NOHL")
		}
		if options.NoFields {
			args = append(args, "NOFIELDS")
		}
		if options.NoFreqs {
			args = append(args, "NOFREQS")
		}
		if options.StopWords != nil {
			args = append(args, "STOPWORDS", len(options.StopWords))
			args = append(args, options.StopWords...)
		}
		if options.SkipInitialScan {
			args = append(args, "SKIPINITIALSCAN")
		}
	}
	if schema == nil {
		panic("FT.CREATE: SCHEMA is required")
	}
	args = append(args, "SCHEMA")
	for _, schema := range schema {
		if schema.FieldName == "" || schema.FieldType == SearchFieldTypeInvalid {
			panic("FT.CREATE: SCHEMA FieldName and FieldType are required")
		}
		args = append(args, schema.FieldName)
		if schema.As != "" {
			args = append(args, "AS", schema.As)
		}
		args = append(args, schema.FieldType.String())
		if schema.VectorArgs != nil {
			if schema.FieldType != SearchFieldTypeVector {
				panic("FT.CREATE: SCHEMA FieldType VECTOR is required for VectorArgs")
			}
			if schema.VectorArgs.FlatOptions != nil && schema.VectorArgs.HNSWOptions != nil {
				panic("FT.CREATE: SCHEMA VectorArgs FlatOptions and HNSWOptions are mutually exclusive")
			}
			if schema.VectorArgs.FlatOptions != nil {
				args = append(args, "FLAT")
				if schema.VectorArgs.FlatOptions.Type == "" || schema.VectorArgs.FlatOptions.Dim == 0 || schema.VectorArgs.FlatOptions.DistanceMetric == "" {
					panic("FT.CREATE: Type, Dim and DistanceMetric are required for VECTOR FLAT")
				}
				flatArgs := []interface{}{
					"TYPE", schema.VectorArgs.FlatOptions.Type,
					"DIM", schema.VectorArgs.FlatOptions.Dim,
					"DISTANCE_METRIC", schema.VectorArgs.FlatOptions.DistanceMetric,
				}
				if schema.VectorArgs.FlatOptions.InitialCapacity > 0 {
					flatArgs = append(flatArgs, "INITIAL_CAP", schema.VectorArgs.FlatOptions.InitialCapacity)
				}
				if schema.VectorArgs.FlatOptions.BlockSize > 0 {
					flatArgs = append(flatArgs, "BLOCK_SIZE", schema.VectorArgs.FlatOptions.BlockSize)
				}
				args = append(args, len(flatArgs))
				args = append(args, flatArgs...)
			}
			if schema.VectorArgs.HNSWOptions != nil {
				args = append(args, "HNSW")
				if schema.VectorArgs.HNSWOptions.Type == "" || schema.VectorArgs.HNSWOptions.Dim == 0 || schema.VectorArgs.HNSWOptions.DistanceMetric == "" {
					panic("FT.CREATE: Type, Dim and DistanceMetric are required for VECTOR HNSW")
				}
				hnswArgs := []interface{}{
					"TYPE", schema.VectorArgs.HNSWOptions.Type,
					"DIM", schema.VectorArgs.HNSWOptions.Dim,
					"DISTANCE_METRIC", schema.VectorArgs.HNSWOptions.DistanceMetric,
				}
				if schema.VectorArgs.HNSWOptions.InitialCapacity > 0 {
					hnswArgs = append(hnswArgs, "INITIAL_CAP", schema.VectorArgs.HNSWOptions.InitialCapacity)
				}
				if schema.VectorArgs.HNSWOptions.MaxEdgesPerNode > 0 {
					hnswArgs = append(hnswArgs, "M", schema.VectorArgs.HNSWOptions.MaxEdgesPerNode)
				}
				if schema.VectorArgs.HNSWOptions.MaxAllowedEdgesPerNode > 0 {
					hnswArgs = append(hnswArgs, "EF_CONSTRUCTION", schema.VectorArgs.HNSWOptions.MaxAllowedEdgesPerNode)
				}
				if schema.VectorArgs.HNSWOptions.EFRunTime > 0 {
					hnswArgs = append(hnswArgs, "EF_RUNTIME", schema.VectorArgs.HNSWOptions.EFRunTime)
				}
				if schema.VectorArgs.HNSWOptions.Epsilon > 0 {
					hnswArgs = append(hnswArgs, "EPSILON", schema.VectorArgs.HNSWOptions.Epsilon)
				}
				args = append(args, len(hnswArgs))
				args = append(args, hnswArgs...)
			}
		}
		if schema.GeoShapeFieldType != "" {
			if schema.FieldType != SearchFieldTypeGeoShape {
				panic("FT.CREATE: SCHEMA FieldType GEOSHAPE is required for GeoShapeFieldType")
			}
			args = append(args, schema.GeoShapeFieldType)
		}
		if schema.NoStem {
			args = append(args, "NOSTEM")
		}
		if schema.Sortable {
			args = append(args, "SORTABLE")
		}
		if schema.UNF {
			args = append(args, "UNF")
		}
		if schema.NoIndex {
			args = append(args, "NOINDEX")
		}
		if schema.PhoneticMatcher != "" {
			args = append(args, "PHONETIC", schema.PhoneticMatcher)
		}
		if schema.Weight > 0 {
			args = append(args, "WEIGHT", schema.Weight)
		}
		if schema.Separator != "" {
			args = append(args, "SEPARATOR", schema.Separator)
		}
		if schema.CaseSensitive {
			args = append(args, "CASESENSITIVE")
		}
		if schema.WithSuffixtrie {
			args = append(args, "WITHSUFFIXTRIE")
		}
		if schema.IndexEmpty {
			args = append(args, "INDEXEMPTY")
		}
		if schema.IndexMissing {
			args = append(args, "INDEXMISSING")

		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTCursorDel - Deletes a cursor from an existing index.
// The 'index' parameter specifies the index from which to delete the cursor, and the 'cursorId' parameter specifies the ID of the cursor to delete.
// For more information, please refer to the Redis documentation:
// [FT.CURSOR DEL]: (https://redis.io/commands/ft.cursor-del/)
func (c cmdable) FTCursorDel(ctx context.Context, index string, cursorId int) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.CURSOR", "DEL", index, cursorId)
	_ = c(ctx, cmd)
	return cmd
}

// FTCursorRead - Reads the next results from an existing cursor.
// The 'index' parameter specifies the index from which to read the cursor, the 'cursorId' parameter specifies the ID of the cursor to read, and the 'count' parameter specifies the number of results to read.
// For more information, please refer to the Redis documentation:
// [FT.CURSOR READ]: (https://redis.io/commands/ft.cursor-read/)
func (c cmdable) FTCursorRead(ctx context.Context, index string, cursorId int, count int) *MapStringInterfaceCmd {
	args := []interface{}{"FT.CURSOR", "READ", index, cursorId}
	if count > 0 {
		args = append(args, "COUNT", count)
	}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDictAdd - Adds terms to a dictionary.
// The 'dict' parameter specifies the dictionary to which to add the terms, and the 'term' parameter specifies the terms to add.
// For more information, please refer to the Redis documentation:
// [FT.DICTADD]: (https://redis.io/commands/ft.dictadd/)
func (c cmdable) FTDictAdd(ctx context.Context, dict string, term ...interface{}) *IntCmd {
	args := []interface{}{"FT.DICTADD", dict}
	args = append(args, term...)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDictDel - Deletes terms from a dictionary.
// The 'dict' parameter specifies the dictionary from which to delete the terms, and the 'term' parameter specifies the terms to delete.
// For more information, please refer to the Redis documentation:
// [FT.DICTDEL]: (https://redis.io/commands/ft.dictdel/)
func (c cmdable) FTDictDel(ctx context.Context, dict string, term ...interface{}) *IntCmd {
	args := []interface{}{"FT.DICTDEL", dict}
	args = append(args, term...)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDictDump - Returns all terms in the specified dictionary.
// The 'dict' parameter specifies the dictionary from which to return the terms.
// For more information, please refer to the Redis documentation:
// [FT.DICTDUMP]: (https://redis.io/commands/ft.dictdump/)
func (c cmdable) FTDictDump(ctx context.Context, dict string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT.DICTDUMP", dict)
	_ = c(ctx, cmd)
	return cmd
}

// FTDropIndex - Deletes an index.
// The 'index' parameter specifies the index to delete.
// For more information, please refer to the Redis documentation:
// [FT.DROPINDEX]: (https://redis.io/commands/ft.dropindex/)
func (c cmdable) FTDropIndex(ctx context.Context, index string) *StatusCmd {
	args := []interface{}{"FT.DROPINDEX", index}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDropIndexWithArgs - Deletes an index with options.
// The 'index' parameter specifies the index to delete, and the 'options' parameter specifies the DeleteDocs option for docs deletion.
// For more information, please refer to the Redis documentation:
// [FT.DROPINDEX]: (https://redis.io/commands/ft.dropindex/)
func (c cmdable) FTDropIndexWithArgs(ctx context.Context, index string, options *FTDropIndexOptions) *StatusCmd {
	args := []interface{}{"FT.DROPINDEX", index}
	if options != nil {
		if options.DeleteDocs {
			args = append(args, "DD")
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTExplain - Returns the execution plan for a complex query.
// The 'index' parameter specifies the index to query, and the 'query' parameter specifies the query string.
// For more information, please refer to the Redis documentation:
// [FT.EXPLAIN]: (https://redis.io/commands/ft.explain/)
func (c cmdable) FTExplain(ctx context.Context, index string, query string) *StringCmd {
	cmd := NewStringCmd(ctx, "FT.EXPLAIN", index, query)
	_ = c(ctx, cmd)
	return cmd
}

// FTExplainWithArgs - Returns the execution plan for a complex query with options.
// The 'index' parameter specifies the index to query, the 'query' parameter specifies the query string, and the 'options' parameter specifies the Dialect for the query.
// For more information, please refer to the Redis documentation:
// [FT.EXPLAIN]: (https://redis.io/commands/ft.explain/)
func (c cmdable) FTExplainWithArgs(ctx context.Context, index string, query string, options *FTExplainOptions) *StringCmd {
	args := []interface{}{"FT.EXPLAIN", index, query}
	if options.Dialect != "" {
		args = append(args, "DIALECT", options.Dialect)
	}
	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTExplainCli - Returns the execution plan for a complex query. [Not Implemented]
// For more information, see https://redis.io/commands/ft.explaincli/
func (fr *FramerBridge) WriteData(streamID uint32, endStream bool, data ...[]byte) error {
	if len(data) == 1 {
		return fr.framer.WriteData(streamID, endStream, data[0])
	}

	tl := 0
	for _, s := range data {
		tl += len(s)
	}

	buf := fr.pool.Get(tl)
	*buf = (*buf)[:0]
	defer fr.pool.Put(buf)
	for _, s := range data {
		*buf = append(*buf, s...)
	}

	return fr.framer.WriteData(streamID, endStream, *buf)
}

func logPrintRenderTemplates(templates *template.Template) {
	if IsInDebugMode() {
		var output bytes.Buffer
		for _, template := range templates.Templates() {
			output.WriteString("\t- ")
			output.WriteString(template.Name())
			output.WriteString("\n")
		}
		logPrint("Rendered Templates (%d): \n%s\n", len(templates.Templates()), output.String())
	}
}

type FTInfoCmd struct {
	baseCmd
	val FTInfoResult
}

func newFTInfoCmd(ctx context.Context, args ...interface{}) *FTInfoCmd {
	return &FTInfoCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (p *orcaPicker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	doneCB := func(di balancer.DoneInfo) {
		if lr, _ := di.ServerLoad.(*v3orcapb.OrcaLoadReport); lr != nil &&
			(lr.CpuUtilization != 0 || lr.MemUtilization != 0 || len(lr.Utilization) > 0 || len(lr.RequestCost) > 0) {
			// Since all RPCs will respond with a load report due to the
			// presence of the DialOption, we need to inspect every field and
			// use the out-of-band report instead if all are unset/zero.
			setContextCMR(info.Ctx, lr)
		} else {
			p.o.reportMu.Lock()
			defer p.o.reportMu.Unlock()
			if lr := p.o.report; lr != nil {
				setContextCMR(info.Ctx, lr)
			}
		}
	}
	return balancer.PickResult{SubConn: p.o.sc, Done: doneCB}, nil
}

func (s) TestEDSWatch_ValidResponseCancelsExpiryTimerBehavior(t *testing.T) {
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bootstrapContents)

	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:               t.Name(),
		Contents:           bootstrapContents,
		WatchExpiryTimeout: defaultTestWatchExpiryTimeout,
	})
	if err != nil {
		t.Fatalf("Failed to create an xDS client: %v", err)
	}
	defer close()

	ew := newEndpointsWatcher()
	xdsresource.WatchEndpoints(client, edsName, ew)
	defer func() { _ = ew.cancel }()

	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(edsName, edsHost1, []uint32{edsPort1})},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	wantUpdate := endpointsUpdateErrTuple{
		update: xdsresource.EndpointsUpdate{
			Localities: []xdsresource.Locality{
				{
					Endpoints: []xdsresource.Endpoint{{Addresses: []string{fmt.Sprintf("%s:%d", edsHost1, edsPort1)}, Weight: 1}},
					ID: internal.LocalityID{
						Region:  "region-1",
						Zone:    "zone-1",
						SubZone: "subzone-1",
					},
					Priority: 0,
					Weight:   1,
				},
			},
		},
	}
	if err := verifyEndpointsUpdate(ctx, ew.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}

	select {
	case <-time.After(defaultTestWatchExpiryTimeout):
	default:
	}

	if !verifyNoEndpointsUpdate(ctx, ew.updateCh) {
		t.Fatal(err)
	}
}

func TestRunWithPort(t *testing.T) {
	router := New()
	go func() {
		router.GET("/example", func(c *Context) { c.String(http.StatusOK, "it worked") })
		assert.NoError(t, router.Run(":5150"))
	}()
	// have to wait for the goroutine to start and run the server
	// otherwise the main thread will complete
	time.Sleep(5 * time.Millisecond)

	require.Error(t, router.Run(":5150"))
	testRequest(t, "http://localhost:5150/example")
}

func (c *baseClient) process(ctx context.Context, cmd Cmder) error {
	var lastErr error
	for attempt := 0; attempt <= c.opt.MaxRetries; attempt++ {
		attempt := attempt

		retry, err := c._process(ctx, cmd, attempt)
		if err == nil || !retry {
			return err
		}

		lastErr = err
	}
	return lastErr
}

func (s *server) BidirectionalStreamingEcho(stream pb.Echo_BidirectionalStreamingEchoServer) error {
	log.Printf("New stream began.")
	// First, we wait 2 seconds before reading from the stream, to give the
	// client an opportunity to block while sending its requests.
	time.Sleep(2 * time.Second)

	// Next, read all the data sent by the client to allow it to unblock.
	for i := 0; true; i++ {
		if _, err := stream.Recv(); err != nil {
			log.Printf("Read %v messages.", i)
			if err == io.EOF {
				break
			}
			log.Printf("Error receiving data: %v", err)
			return err
		}
	}

	// Finally, send data until we block, then end the stream after we unblock.
	stopSending := grpcsync.NewEvent()
	sentOne := make(chan struct{})
	go func() {
		for !stopSending.HasFired() {
			after := time.NewTimer(time.Second)
			select {
			case <-sentOne:
				after.Stop()
			case <-after.C:
				log.Printf("Sending is blocked.")
				stopSending.Fire()
				<-sentOne
			}
		}
	}()

	i := 0
	for !stopSending.HasFired() {
		i++
		if err := stream.Send(&pb.EchoResponse{Message: payload}); err != nil {
			log.Printf("Error sending data: %v", err)
			return err
		}
		sentOne <- struct{}{}
	}
	log.Printf("Sent %v messages.", i)

	log.Printf("Stream ended successfully.")
	return nil
}

func (w *Writer) AppendArg(value interface{}) error {
	switch value := value.(type) {
	case nil:
		return w.appendString("")
	case string:
		return w.appendString(value)
	case *string:
		return w.appendString(*value)
	case []byte:
		return w.appendBytes(value)
	case int:
		return w.appendInt(int64(value))
	case *int:
		return w.appendInt(int64(*value))
	case int8:
		return w.appendInt(int64(value))
	case *int8:
		return w.appendInt(int64(*value))
	case int16:
		return w.appendInt(int64(value))
	case *int16:
		return w.appendInt(int64(*value))
	case int32:
		return w.appendInt(int64(value))
	case *int32:
		return w.appendInt(int64(*value))
	case int64:
		return w.appendInt(value)
	case *int64:
		return w.appendInt(*value)
	case uint:
		return w.appendUint(uint64(value))
	case *uint:
		return w.appendUint(uint64(*value))
	case uint8:
		return w.appendUint(uint64(value))
	case *uint8:
		return w.appendUint(uint64(*value))
	case uint16:
		return w.appendUint(uint64(value))
	case *uint16:
		return w.appendUint(uint64(*value))
	case uint32:
		return w.appendUint(uint64(value))
	case *uint32:
		return w.appendUint(uint64(*value))
	case uint64:
		return w.appendUint(value)
	case *uint64:
		return w.appendUint(*value)
	case float32:
		return w.appendFloat(float64(value))
	case *float32:
		return w.appendFloat(float64(*value))
	case float64:
		return w.appendFloat(value)
	case *float64:
		return w.appendFloat(*value)
	case bool:
		if value {
			return w.appendInt(1)
		}
		return w.appendInt(0)
	case *bool:
		if *value {
			return w.appendInt(1)
		}
		return w.appendInt(0)
	case time.Time:
		w.numBuf = value.AppendFormat(w.numBuf[:0], time.RFC3339Nano)
		return w.appendBytes(w.numBuf)
	case time.Duration:
		return w.appendInt(value.Nanoseconds())
	case encoding.BinaryMarshaler:
		b, err := value.MarshalBinary()
		if err != nil {
			return err
		}
		return w.appendBytes(b)
	case net.IP:
		return w.appendBytes(value)
	default:
		return fmt.Errorf(
			"redis: can't marshal %T (implement encoding.BinaryMarshaler)", value)
	}
}
func (fr *FramerBridge) UpdateSettings(configurations ...SettingConfig) error {
	css := make([]http2.Setting, 0, len(configurations))
	for _, config := range configurations {
		css = append(css, http2.Setting{
			ID:  http2.SettingID(config.ID),
			Val: config.Value,
		})
	}

	return fr.framer.WriteSettings(css...)
}

// FTInfo - Retrieves information about an index.
// The 'index' parameter specifies the index to retrieve information about.
// For more information, please refer to the Redis documentation:
// [FT.INFO]: (https://redis.io/commands/ft.info/)
func (c cmdable) FTInfo(ctx context.Context, index string) *FTInfoCmd {
	cmd := newFTInfoCmd(ctx, "FT.INFO", index)
	_ = c(ctx, cmd)
	return cmd
}

// FTSpellCheck - Checks a query string for spelling errors.
// For more details about spellcheck query please follow:
// https://redis.io/docs/interact/search-and-query/advanced-concepts/spellcheck/
// For more information, please refer to the Redis documentation:
// [FT.SPELLCHECK]: (https://redis.io/commands/ft.spellcheck/)
func (c cmdable) FTSpellCheck(ctx context.Context, index string, query string) *FTSpellCheckCmd {
	args := []interface{}{"FT.SPELLCHECK", index, query}
	cmd := newFTSpellCheckCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTSpellCheckWithArgs - Checks a query string for spelling errors with additional options.
// For more details about spellcheck query please follow:
// https://redis.io/docs/interact/search-and-query/advanced-concepts/spellcheck/
// For more information, please refer to the Redis documentation:
// [FT.SPELLCHECK]: (https://redis.io/commands/ft.spellcheck/)
func (c cmdable) FTSpellCheckWithArgs(ctx context.Context, index string, query string, options *FTSpellCheckOptions) *FTSpellCheckCmd {
	args := []interface{}{"FT.SPELLCHECK", index, query}
	if options != nil {
		if options.Distance > 0 {
			args = append(args, "DISTANCE", options.Distance)
		}
		if options.Terms != nil {
			args = append(args, "TERMS", options.Terms.Inclusion, options.Terms.Dictionary)
			args = append(args, options.Terms.Terms...)
		}
		if options.Dialect > 0 {
			args = append(args, "DIALECT", options.Dialect)
		}
	}
	cmd := newFTSpellCheckCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type FTSpellCheckCmd struct {
	baseCmd
	val []SpellCheckResult
}

func newFTSpellCheckCmd(ctx context.Context, args ...interface{}) *FTSpellCheckCmd {
	return &FTSpellCheckCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func TestIPMatch(s *testing.T) {
	tests := []struct {
	 DESC      string
	 IP        string
	 Pattern   string
	 WantMatch bool
	}{
		{
		 DESC:      "invalid wildcard 1",
		 IP:        "aa.example.com",
		 Pattern:   "*a.example.com",
		 WantMatch: false,
		},
		{
		 DESC:      "invalid wildcard 2",
		 IP:        "aa.example.com",
		 Pattern:   "a*.example.com",
		 WantMatch: false,
		},
		{
		 DESC:      "invalid wildcard 3",
		 IP:        "abc.example.com",
		 Pattern:   "a*c.example.com",
		 WantMatch: false,
		},
		{
		 DESC:      "wildcard in one of the middle components",
		 IP:        "abc.test.example.com",
		 Pattern:   "abc.*.example.com",
		 WantMatch: false,
		},
		{
		 DESC:      "single component wildcard",
		 IP:        "a.example.com",
		 Pattern:   "*",
		 WantMatch: false,
		},
		{
		 DESC:      "short host name",
		 IP:        "a.com",
		 Pattern:   "*.example.com",
		 WantMatch: false,
		},
		{
		 DESC:      "suffix mismatch",
		 IP:        "a.notexample.com",
		 Pattern:   "*.example.com",
		 WantMatch: false,
		},
		{
		 DESC:      "wildcard match across components",
		 IP:        "sub.test.example.com",
		 Pattern:   "*.example.com.",
		 WantMatch: false,
		},
		{
		 DESC:      "host doesn't end in period",
		 IP:        "test.example.com",
		 Pattern:   "test.example.com.",
		 WantMatch: true,
		},
		{
		 DESC:      "pattern doesn't end in period",
		 IP:        "test.example.com.",
		 Pattern:   "test.example.com",
		 WantMatch: true,
		},
		{
		 DESC:      "case insensitive",
		 IP:        "TEST.EXAMPLE.COM.",
		 Pattern:   "test.example.com.",
		 WantMatch: true,
		},
		{
		 DESC:      "simple match",
		 IP:        "test.example.com",
		 Pattern:   "test.example.com",
		 WantMatch: true,
		},
		{
		 DESC:      "good wildcard",
		 IP:        "a.example.com",
		 Pattern:   "*.example.com",
		 WantMatch: true,
		},
	}

	for _, test := range tests {
		t.Run(test.DESC, func(t *testing.T) {
			gotMatch := ipMatch(test.IP, test.Pattern)
			if gotMatch != test.WantMatch {
				t.Fatalf("ipMatch(%s, %s) = %v, want %v", test.IP, test.Pattern, gotMatch, test.WantMatch)
			}
		})
	}
}

func (s) TestNewLoggerFromConfigStringInvalid(t *testing.T) {
	testCases := []string{
		"",
		"*{}",
		"s/m,*{}",
		"s/m,s/m{a}",

		// Duplicate rules.
		"s/m,-s/m",
		"-s/m,s/m",
		"s/m,s/m",
		"s/m,s/m{h:1;m:1}",
		"s/m{h:1;m:1},s/m",
		"-s/m,-s/m",
		"s/*,s/*{h:1;m:1}",
		"*,*{h:1;m:1}",
	}
	for _, tc := range testCases {
		l := NewLoggerFromConfigString(tc)
		if l != nil {
			t.Errorf("With config %q, want logger %v, got %v", tc, nil, l)
		}
	}
}

func VerifyFormStringMap(t *testing.T) {
	VerifyBodyBindingStringMap(t, Form,
		"/", "",
		`foo=bar&hello=world`, "")
	// Should pick the last value
	VerifyBodyBindingStringMap(t, Form,
		"/", "",
		`foo=something&foo=bar&hello=world`, "")
}

func (cmd *FTSpellCheckCmd) Val() []SpellCheckResult {
	return cmd.val
}

func TestUserHasProfileWithSameForeignKey(t *testing_T) {
	type UserProfile struct {
		gorm.Model
		Name         string
		UserID       int  // not used in relationship
	}

	type UserData struct {
		gorm.Model
		UserProfile UserProfile `gorm:"ForeignKey:UserID;references:ProfileRefer"`
		ProfileRef  int
	}

	checkStructRelation(t, &UserData{}, Relation{
		Name: "UserProfile", Type: schema.HasOne, Schema: "UserData", FieldSchema: "User Profile",
		References: []Reference{{"ProfileRef", "UserData", "UserID", "UserProfile", "", true}},
	})
}

func HandleJWTTokenCredentials(ctx context.Context, testClient testgrpc.TestServiceClient, serviceAccountFile string) {
	credentialPayload := ClientNewPayload(testpb.PayloadType_COMPRESSABLE, largeRequestSize)
	simpleRequest := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE,
		ResponseSize:   int32(largeResponseSize),
		Payload:        credentialPayload,
		FillUsername:   true,
	}
	response, rpcError := testClient.UnaryCall(ctx, simpleRequest)
	if rpcError != nil {
		logger.Fatal("/TestService/UnaryCall RPC failed: ", rpcError)
	}
	jsonKeyContent := getServiceAccountJSONKey(serviceAccountFile)
	usernameFromResponse := response.GetUsername()
	if !strings.Contains(string(jsonKeyContent), usernameFromResponse) {
		logger.Fatalf("Got user name %q which is NOT a substring of %q.", usernameFromResponse, jsonKeyContent)
	}
}

func TestDebugPrintWARNINGNew(t *testing.T) {
	re := captureOutput(t, func() {
		SetMode(DebugMode)
		debugPrintWARNINGNew()
		SetMode(TestMode)
	})
	assert.Equal(t, "[GIN-debug] [WARNING] Running in \"debug\" mode. Switch to \"release\" mode in production.\n - using env:\texport GIN_MODE=release\n - using code:\tgin.SetMode(gin.ReleaseMode)\n\n", re)
}

func TestSingleTableHasManyAssociationAlt(t *testing.T) {
	alternativeUser := *GetUser("hasmany", Config{Team: 2})

	if err := DB.Create(&alternativeUser).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	CheckUser(t, alternativeUser, alternativeUser)

	// Find
	var user3 User
	DB.Find(&user3, "id = ?", alternativeUser.ID)
	user3Team := DB.Model(&user3).Association("Team").Find()
	CheckUser(t, user3, alternativeUser)

	// Count
	AssertAssociationCount(t, alternativeUser, "Team", 2, "")

	// Append
	teamForAppend := *GetUser("team", Config{})

	if err := DB.Model(&alternativeUser).Association("Team").Append(&teamForAppend); err != nil {
		t.Fatalf("Error happened when append account, got %v", err)
	}

	if teamForAppend.ID == 0 {
		t.Fatalf("Team's ID should be created")
	}

	alternativeUser.Team = append(alternativeUser.Team, teamForAppend)
	CheckUser(t, alternativeUser, alternativeUser)

	AssertAssociationCount(t, alternativeUser, "Team", 3, "AfterAppend")

	teamsToAppend := []User{*GetUser("team-append-1", Config{}), *GetUser("team-append-2", Config{})}

	if err := DB.Model(&alternativeUser).Association("Team").Append(teamsToAppend...); err != nil {
		t.Fatalf("Error happened when append team, got %v", err)
	}

	for _, team := range teamsToAppend {
		if team.ID == 0 {
			t.Fatalf("Team's ID should be created")
		}
		alternativeUser.Team = append(alternativeUser.Team, team)
	}

	CheckUser(t, alternativeUser, alternativeUser)

	AssertAssociationCount(t, alternativeUser, "Team", 5, "AfterAppendSlice")

	// Replace
	teamToReplace := *GetUser("team-replace", Config{})

	if err := DB.Model(&alternativeUser).Association("Team").Replace(&teamToReplace); err != nil {
		t.Fatalf("Error happened when replace team, got %v", err)
	}

	if teamToReplace.ID == 0 {
		t.Fatalf("team2's ID should be created")
	}

	alternativeUser.Team = []User{teamToReplace}
	CheckUser(t, alternativeUser, alternativeUser)

	AssertAssociationCount(t, alternativeUser, "Team", 1, "AfterReplace")

	// Delete
	if err := DB.Model(&alternativeUser).Association("Team").Delete(&User{}); err != nil {
		t.Fatalf("Error happened when delete team, got %v", err)
	}
	AssertAssociationCount(t, alternativeUser, "Team", 1, "after delete non-existing data")

	if err := DB.Model(&alternativeUser).Association("Team").Delete(&teamToReplace); err != nil {
		t.Fatalf("Error happened when delete Team, got %v", err)
	}
	AssertAssociationCount(t, alternativeUser, "Team", 0, "after delete")

	// Prepare Data for Clear
	if err := DB.Model(&alternativeUser).Association("Team").Append(&teamForAppend); err != nil {
		t.Fatalf("Error happened when append Team, got %v", err)
	}

	AssertAssociationCount(t, alternativeUser, "Team", 1, "after prepare data")

	// Clear
	if err := DB.Model(&alternativeUser).Association("Team").Clear(); err != nil {
		t.Errorf("Error happened when clear Team, got %v", err)
	}

	AssertAssociationCount(t, alternativeUser, "Team", 0, "after clear")
}

func ExampleServer_update() {
	req := request.NewRequest()

	db := database.NewDatabase(&database.Options{
		Host:     "localhost",
		Port:     6379,
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	db.Remove(context.Background(), "cars:sports:france")
	db.Remove(context.Background(), "cars:sports:usa")
	// REMOVE_END

	// STEP_START update
	if err := db.Add(context.Background(), "cars:sports:france", "car:1", "car:2", "car:3"); err != nil {
		panic(err)
	}

	if err := db.Add(context.Background(), "cars:sports:usa", "car:1", "car:4"); err != nil {
		panic(err)
	}

	res15, err := db.Diff(context.Background(), "cars:sports:france", "cars:sports:usa")

	if err != nil {
		panic(err)
	}

	fmt.Println(res15) // >>> [car:2 car:3]
	// STEP_END

	// Output:
	// [car:2 car:3]
}

type FTSearchCmd struct {
	baseCmd
	val     FTSearchResult
	options *FTSearchOptions
}

func newFTSearchCmd(ctx context.Context, options *FTSearchOptions, args ...interface{}) *FTSearchCmd {
	return &FTSearchCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
		options: options,
	}
}

func TestContextGetFloat64Slice(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "float64-slice"
	value := []float64{1, 2}
	c.Set(key, value)
	assert.Equal(t, value, c.GetFloat64Slice(key))
}

func TestSingleServerAfter(t *testing.T) {
	var completion = make(chan struct{})
	ecm := jsonrpc.EndpointCodecMap{
		"multiply": jsonrpc.EndpointCodec{
			Endpoint: endpoint.Nop,
			Decode:   nopDecoder,
			Encode:   nopEncoder,
		},
	}
	handler := jsonrpc.NewServer(
		ecm,
		jsonrpc.ServerAfter(func(ctx context.Context, w http.ResponseWriter) context.Context {
			ctx = context.WithValue(ctx, "two", 2)

			return ctx
		}),
		jsonrpc.ServerAfter(func(ctx context.Context, w http.ResponseWriter) context.Context {
			if _, ok := ctx.Value("two").(int); !ok {
				t.Error("Value was not set properly when multiple ServerAfters are used")
			}

			close(completion)
			return ctx
		}),
	)
	server := httptest.NewServer(handler)
	defer server.Close()
	http.Post(server.URL, "application/json", multiplyBody()) // nolint

	select {
	case <-completion:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for finalizer")
	}
}

func (s) TestParsedTarget_WithCustomDialer(t *testing.T) {
	resetInitialResolverState()
	defScheme := resolver.GetDefaultScheme()
	tests := []struct {
		target            string
		wantParsed        resolver.Target
		wantDialerAddress string
	}{
		// unix:[local_path], unix:[/absolute], and unix://[/absolute] have
		// different behaviors with a custom dialer.
		{
			target:            "unix:a/b/c",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("unix:a/b/c")},
			wantDialerAddress: "unix:a/b/c",
		},
		{
			target:            "unix:/a/b/c",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("unix:/a/b/c")},
			wantDialerAddress: "unix:///a/b/c",
		},
		{
			target:            "unix:///a/b/c",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("unix:///a/b/c")},
			wantDialerAddress: "unix:///a/b/c",
		},
		{
			target:            "dns:///127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("dns:///127.0.0.1:50051")},
			wantDialerAddress: "127.0.0.1:50051",
		},
		{
			target:            ":///127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, ":///127.0.0.1:50051"))},
			wantDialerAddress: ":///127.0.0.1:50051",
		},
		{
			target:            "dns://authority/127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("dns://authority/127.0.0.1:50051")},
			wantDialerAddress: "127.0.0.1:50051",
		},
		{
			target:            "://authority/127.0.0.1:50051",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, "://authority/127.0.0.1:50051"))},
			wantDialerAddress: "://authority/127.0.0.1:50051",
		},
		{
			target:            "/unix/socket/address",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, "/unix/socket/address"))},
			wantDialerAddress: "/unix/socket/address",
		},
		{
			target:            "",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL(fmt.Sprintf("%s:///%s", defScheme, ""))},
			wantDialerAddress: "",
		},
		{
			target:            "passthrough://a.server.com/google.com",
			wantParsed:        resolver.Target{URL: *testutils.MustParseURL("passthrough://a.server.com/google.com")},
			wantDialerAddress: "google.com",
		},
	}

	for _, test := range tests {
		t.Run(test.target, func(t *testing.T) {
			addrCh := make(chan string, 1)
			dialer := func(_ context.Context, address string) (net.Conn, error) {
				addrCh <- address
				return nil, errors.New("dialer error")
			}

			cc, err := Dial(test.target, WithTransportCredentials(insecure.NewCredentials()), WithContextDialer(dialer))
			if err != nil {
				t.Fatalf("Dial(%q) failed: %v", test.target, err)
			}
			defer cc.Close()

			select {
			case addr := <-addrCh:
				if addr != test.wantDialerAddress {
					t.Fatalf("address in custom dialer is %q, want %q", addr, test.wantDialerAddress)
				}
			case <-time.After(time.Second):
				t.Fatal("timeout when waiting for custom dialer to be invoked")
			}
			if !cmp.Equal(cc.parsedTarget, test.wantParsed) {
				t.Errorf("cc.parsedTarget for dial target %q = %+v, want %+v", test.target, cc.parsedTarget, test.wantParsed)
			}
		})
	}
}


func (lsw *Wrapper) ProcessCompletion(loc string, err error) {
	defer lsw.mu.RUnlock()
	lsw.mu.RLock()
	if perCluster := lsw.perCluster; perCluster != nil {
		perCluster.CallFinished(loc, err)
	}
}

func (cmd *JSONCmd) handleResponse(reader *proto.Reader) error {
	if err := cmd.checkBaseError(reader); err != nil {
		return err
	}

	replyType, peekErr := reader.PeekReplyType()
	if peekErr != nil {
		return peekErr
	}

	switch replyType {
	case proto.RespArray:
		length, readArrErr := reader.ReadArrayLen()
		if readArrErr != nil {
			return readArrErr
		}
		expanded := make([]interface{}, length)
		for i := 0; i < length; i++ {
			if expanded[i], readErr = reader.ReadReply(); readErr != nil {
				return readErr
			}
		}
		cmd.expanded = expanded

	default:
		str, readStrErr := reader.ReadString()
		if readStrErr != nil && readStrErr != Nil {
			return readStrErr
		}
		if str == "" || readStrErr == Nil {
			cmd.val = ""
		} else {
			cmd.val = str
		}
	}

	return nil
}

func (cmd *JSONCmd) checkBaseError(reader *proto.Reader) error {
	if cmd.baseCmd.Err() == Nil {
		cmd.val = ""
		return Nil
	}
	return nil
}

func (t *http2Client) operateHeaders(frame *http2.MetaHeadersFrame) {
	s := t.getStream(frame)
	if s == nil {
		return
	}
	endStream := frame.StreamEnded()
	s.bytesReceived.Store(true)
	initialHeader := atomic.LoadUint32(&s.headerChanClosed) == 0

	if !initialHeader && !endStream {
		// As specified by gRPC over HTTP2, a HEADERS frame (and associated CONTINUATION frames) can only appear at the start or end of a stream. Therefore, second HEADERS frame must have EOS bit set.
		st := status.New(codes.Internal, "a HEADERS frame cannot appear in the middle of a stream")
		t.closeStream(s, st.Err(), true, http2.ErrCodeProtocol, st, nil, false)
		return
	}

	// frame.Truncated is set to true when framer detects that the current header
	// list size hits MaxHeaderListSize limit.
	if frame.Truncated {
		se := status.New(codes.Internal, "peer header list size exceeded limit")
		t.closeStream(s, se.Err(), true, http2.ErrCodeFrameSize, se, nil, endStream)
		return
	}

	var (
		// If a gRPC Response-Headers has already been received, then it means
		// that the peer is speaking gRPC and we are in gRPC mode.
		isGRPC         = !initialHeader
		mdata          = make(map[string][]string)
		contentTypeErr = "malformed header: missing HTTP content-type"
		grpcMessage    string
		recvCompress   string
		httpStatusCode *int
		httpStatusErr  string
		rawStatusCode  = codes.Unknown
		// headerError is set if an error is encountered while parsing the headers
		headerError string
	)

	if initialHeader {
		httpStatusErr = "malformed header: missing HTTP status"
	}

	for _, hf := range frame.Fields {
		switch hf.Name {
		case "content-type":
			if _, validContentType := grpcutil.ContentSubtype(hf.Value); !validContentType {
				contentTypeErr = fmt.Sprintf("transport: received unexpected content-type %q", hf.Value)
				break
			}
			contentTypeErr = ""
			mdata[hf.Name] = append(mdata[hf.Name], hf.Value)
			isGRPC = true
		case "grpc-encoding":
			recvCompress = hf.Value
		case "grpc-status":
			code, err := strconv.ParseInt(hf.Value, 10, 32)
			if err != nil {
				se := status.New(codes.Internal, fmt.Sprintf("transport: malformed grpc-status: %v", err))
				t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
				return
			}
			rawStatusCode = codes.Code(uint32(code))
		case "grpc-message":
			grpcMessage = decodeGrpcMessage(hf.Value)
		case ":status":
			if hf.Value == "200" {
				httpStatusErr = ""
				statusCode := 200
				httpStatusCode = &statusCode
				break
			}

			c, err := strconv.ParseInt(hf.Value, 10, 32)
			if err != nil {
				se := status.New(codes.Internal, fmt.Sprintf("transport: malformed http-status: %v", err))
				t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
				return
			}
			statusCode := int(c)
			httpStatusCode = &statusCode

			httpStatusErr = fmt.Sprintf(
				"unexpected HTTP status code received from server: %d (%s)",
				statusCode,
				http.StatusText(statusCode),
			)
		default:
			if isReservedHeader(hf.Name) && !isWhitelistedHeader(hf.Name) {
				break
			}
			v, err := decodeMetadataHeader(hf.Name, hf.Value)
			if err != nil {
				headerError = fmt.Sprintf("transport: malformed %s: %v", hf.Name, err)
				logger.Warningf("Failed to decode metadata header (%q, %q): %v", hf.Name, hf.Value, err)
				break
			}
			mdata[hf.Name] = append(mdata[hf.Name], v)
		}
	}

	if !isGRPC || httpStatusErr != "" {
		var code = codes.Internal // when header does not include HTTP status, return INTERNAL

		if httpStatusCode != nil {
			var ok bool
			code, ok = HTTPStatusConvTab[*httpStatusCode]
			if !ok {
				code = codes.Unknown
			}
		}
		var errs []string
		if httpStatusErr != "" {
			errs = append(errs, httpStatusErr)
		}
		if contentTypeErr != "" {
			errs = append(errs, contentTypeErr)
		}
		// Verify the HTTP response is a 200.
		se := status.New(code, strings.Join(errs, "; "))
		t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
		return
	}

	if headerError != "" {
		se := status.New(codes.Internal, headerError)
		t.closeStream(s, se.Err(), true, http2.ErrCodeProtocol, se, nil, endStream)
		return
	}

	// For headers, set them in s.header and close headerChan.  For trailers or
	// trailers-only, closeStream will set the trailers and close headerChan as
	// needed.
	if !endStream {
		// If headerChan hasn't been closed yet (expected, given we checked it
		// above, but something else could have potentially closed the whole
		// stream).
		if atomic.CompareAndSwapUint32(&s.headerChanClosed, 0, 1) {
			s.headerValid = true
			// These values can be set without any synchronization because
			// stream goroutine will read it only after seeing a closed
			// headerChan which we'll close after setting this.
			s.recvCompress = recvCompress
			if len(mdata) > 0 {
				s.header = mdata
			}
			close(s.headerChan)
		}
	}

	for _, sh := range t.statsHandlers {
		if !endStream {
			inHeader := &stats.InHeader{
				Client:      true,
				WireLength:  int(frame.Header().Length),
				Header:      metadata.MD(mdata).Copy(),
				Compression: s.recvCompress,
			}
			sh.HandleRPC(s.ctx, inHeader)
		} else {
			inTrailer := &stats.InTrailer{
				Client:     true,
				WireLength: int(frame.Header().Length),
				Trailer:    metadata.MD(mdata).Copy(),
			}
			sh.HandleRPC(s.ctx, inTrailer)
		}
	}

	if !endStream {
		return
	}

	status := istatus.NewWithProto(rawStatusCode, grpcMessage, mdata[grpcStatusDetailsBinHeader])

	// If client received END_STREAM from server while stream was still active,
	// send RST_STREAM.
	rstStream := s.getState() == streamActive
	t.closeStream(s, io.EOF, rstStream, http2.ErrCodeNo, status, mdata, true)
}

// FTSearch - Executes a search query on an index.
// The 'index' parameter specifies the index to search, and the 'query' parameter specifies the search query.
// For more information, please refer to the Redis documentation:
// [FT.SEARCH]: (https://redis.io/commands/ft.search/)
func (c cmdable) FTSearch(ctx context.Context, index string, query string) *FTSearchCmd {
	args := []interface{}{"FT.SEARCH", index, query}
	cmd := newFTSearchCmd(ctx, &FTSearchOptions{}, args...)
	_ = c(ctx, cmd)
	return cmd
}

type SearchQuery []interface{}

func (xc *xdsChannel) subscribe(typ xdsresource.Type, name string) {
	if xc.closed.HasFired() {
		if xc.logger.V(2) {
			xc.logger.Infof("Attempt to subscribe to an xDS resource of type %s and name %q on a closed channel", typ.TypeName(), name)
		}
		return
	}
	xc.ads.Subscribe(typ, name)
}

// FTSearchWithArgs - Executes a search query on an index with additional options.
// The 'index' parameter specifies the index to search, the 'query' parameter specifies the search query,
// and the 'options' parameter specifies additional options for the search.
// For more information, please refer to the Redis documentation:
// [FT.SEARCH]: (https://redis.io/commands/ft.search/)
func (c cmdable) FTSearchWithArgs(ctx context.Context, index string, query string, options *FTSearchOptions) *FTSearchCmd {
	args := []interface{}{"FT.SEARCH", index, query}
	if options != nil {
		if options.NoContent {
			args = append(args, "NOCONTENT")
		}
		if options.Verbatim {
			args = append(args, "VERBATIM")
		}
		if options.NoStopWords {
			args = append(args, "NOSTOPWORDS")
		}
		if options.WithScores {
			args = append(args, "WITHSCORES")
		}
		if options.WithPayloads {
			args = append(args, "WITHPAYLOADS")
		}
		if options.WithSortKeys {
			args = append(args, "WITHSORTKEYS")
		}
		if options.Filters != nil {
			for _, filter := range options.Filters {
				args = append(args, "FILTER", filter.FieldName, filter.Min, filter.Max)
			}
		}
		if options.GeoFilter != nil {
			for _, geoFilter := range options.GeoFilter {
				args = append(args, "GEOFILTER", geoFilter.FieldName, geoFilter.Longitude, geoFilter.Latitude, geoFilter.Radius, geoFilter.Unit)
			}
		}
		if options.InKeys != nil {
			args = append(args, "INKEYS", len(options.InKeys))
			args = append(args, options.InKeys...)
		}
		if options.InFields != nil {
			args = append(args, "INFIELDS", len(options.InFields))
			args = append(args, options.InFields...)
		}
		if options.Return != nil {
			args = append(args, "RETURN")
			argsReturn := []interface{}{}
			for _, ret := range options.Return {
				argsReturn = append(argsReturn, ret.FieldName)
				if ret.As != "" {
					argsReturn = append(argsReturn, "AS", ret.As)
				}
			}
			args = append(args, len(argsReturn))
			args = append(args, argsReturn...)
		}
		if options.Slop > 0 {
			args = append(args, "SLOP", options.Slop)
		}
		if options.Timeout > 0 {
			args = append(args, "TIMEOUT", options.Timeout)
		}
		if options.InOrder {
			args = append(args, "INORDER")
		}
		if options.Language != "" {
			args = append(args, "LANGUAGE", options.Language)
		}
		if options.Expander != "" {
			args = append(args, "EXPANDER", options.Expander)
		}
		if options.Scorer != "" {
			args = append(args, "SCORER", options.Scorer)
		}
		if options.ExplainScore {
			args = append(args, "EXPLAINSCORE")
		}
		if options.Payload != "" {
			args = append(args, "PAYLOAD", options.Payload)
		}
		if options.SortBy != nil {
			args = append(args, "SORTBY")
			for _, sortBy := range options.SortBy {
				args = append(args, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					panic("FT.SEARCH: ASC and DESC are mutually exclusive")
				}
				if sortBy.Asc {
					args = append(args, "ASC")
				}
				if sortBy.Desc {
					args = append(args, "DESC")
				}
			}
			if options.SortByWithCount {
				args = append(args, "WITHCOUT")
			}
		}
		if options.LimitOffset >= 0 && options.Limit > 0 {
			args = append(args, "LIMIT", options.LimitOffset, options.Limit)
		}
		if options.Params != nil {
			args = append(args, "PARAMS", len(options.Params)*2)
			for key, value := range options.Params {
				args = append(args, key, value)
			}
		}
		if options.DialectVersion > 0 {
			args = append(args, "DIALECT", options.DialectVersion)
		}
	}
	cmd := newFTSearchCmd(ctx, options, args...)
	_ = c(ctx, cmd)
	return cmd
}

func NewFTSynDumpCmd(ctx context.Context, args ...interface{}) *FTSynDumpCmd {
	return &FTSynDumpCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (s) TestEncodeMetadataHeader(t *testing.T) {
	for _, test := range []struct {
		// input
		kin string
		vin string
		// output
		vout string
	}{
		{"key", "abc", "abc"},
		{"KEY", "abc", "abc"},
		{"key-bin", "abc", "YWJj"},
		{"key-bin", binaryValue, "woA"},
	} {
		v := encodeMetadataHeader(test.kin, test.vin)
		if !reflect.DeepEqual(v, test.vout) {
			t.Fatalf("encodeMetadataHeader(%q, %q) = %q, want %q", test.kin, test.vin, v, test.vout)
		}
	}
}

func processServiceEndpoints(endpoints []*v3endpointpb.LbEndpoint, addrMap map[string]bool) ([]Endpoint, error) {
	var processedEndpoints []Endpoint
	for _, endpoint := range endpoints {
		weight := uint32(1)
		if lbWeight := endpoint.GetLoadBalancingWeight(); lbWeight != nil && lbWeight.GetValue() > 0 {
			weight = lbWeight.GetValue()
		}
		addrs := append([]string{parseAddress(endpoint.GetEndpoint().GetAddress().GetSocketAddress())}, parseAdditionalAddresses(endpoint)...)

		for _, addr := range addrs {
			if addrMap[addr] {
				return nil, fmt.Errorf("duplicate endpoint with the same address %s", addr)
			}
			addrMap[addr] = true
		}

		processedEndpoints = append(processedEndpoints, Endpoint{
			HealthStatus: EndpointHealthStatus(endpoint.GetHealthStatus()),
			Addresses:    addrs,
			Weight:       weight,
		})
	}
	return processedEndpoints, nil
}

func parseAdditionalAddresses(lbEndpoint *v3endpointpb.LbEndpoint) []string {
	var addresses []string
	if envconfig.XDSDualstackEndpointsEnabled {
		for _, sa := range lbEndpoint.GetEndpoint().GetAdditionalAddresses() {
			addresses = append(addresses, parseAddress(sa.GetAddress().GetSocketAddress()))
		}
	}
	return addresses
}

func (cmd *FTSynDumpCmd) Val() []FTSynDumpResult {
	return cmd.val
}

func (l *loopyWriter) cleanupStreamHandler(c *cleanupStream) error {
	c.onWrite()
	if str, ok := l.estdStreams[c.streamID]; ok {
		// On the server side it could be a trailers-only response or
		// a RST_STREAM before stream initialization thus the stream might
		// not be established yet.
		delete(l.estdStreams, c.streamID)
		str.deleteSelf()
		for head := str.itl.dequeueAll(); head != nil; head = head.next {
			if df, ok := head.it.(*dataFrame); ok {
				_ = df.reader.Close()
			}
		}
	}
	if c.rst { // If RST_STREAM needs to be sent.
		if err := l.framer.fr.WriteRSTStream(c.streamID, c.rstCode); err != nil {
			return err
		}
	}
	if l.draining && len(l.estdStreams) == 0 {
		// Flush and close the connection; we are done with it.
		return errors.New("finished processing active streams while in draining mode")
	}
	return nil
}

func ConfigureMode(setting string) {
	if setting == "" {
		if flag.Lookup("test.v") != nil {
			setting = "TestMode"
		} else {
			setting = "DebugMode"
		}
	}

	switch setting {
	case "DebugMode", "":
		atomic.StoreInt32(&ginMode, debugCode)
	case "ReleaseMode":
		atomic.StoreInt32(&ginMode, releaseCode)
	case "TestMode":
		atomic.StoreInt32(&ginMode, testCode)
	default:
		panic("gin mode unknown: " + setting + " (available modes: debug, release, test)")
	}

	modeName = setting
}

func TestInvokeAnotherPath(t *testing.T) {
	service := serviceTest02{}

	greetHandler := NewHandler(
		makeTest02GreetingEndpoint(service),
		decodeGreetRequestWithTwoBefores,
		encodeResponse,
		HandlerErrorHandler(transport.NewLogErrorHandler(log.NewNopLogger())),
		HandlerBefore(func(
			ctx context.Context,
			payload []byte,
		) context.Context {
			ctx = context.WithValue(ctx, KeyBeforeOne, "bef1")
			return ctx
		}),
		HandlerBefore(func(
			ctx context.Context,
			payload []byte,
		) context.Context {
			ctx = context.WithValue(ctx, KeyBeforeTwo, "bef2")
			return ctx
		}),
		HandlerAfter(func(
			ctx context.Context,
			response interface{},
		) context.Context {
			ctx = context.WithValue(ctx, KeyAfterOne, "af1")
			return ctx
		}),
		HandlerAfter(func(
			ctx context.Context,
			response interface{},
		) context.Context {
			if _, ok := ctx.Value(KeyAfterOne).(string); !ok {
				t.Fatalf("Value was not set properly during multi HandlerAfter")
			}
			return ctx
		}),
		HandlerFinalizer(func(
			_ context.Context,
			resp []byte,
			_ error,
		) {
			apigwResp := apiGatewayProxyResponse{}
			err := json.Unmarshal(resp, &apigwResp)
			if err != nil {
				t.Fatalf("Should have no error, but got: %+v", err)
			}

			greetResp := greetResponse{}
			err = json.Unmarshal([]byte(apigwResp.Body), &greetResp)
			if err != nil {
				t.Fatalf("Should have no error, but got: %+v", err)
			}

			expectedMessage := "hello jane doe bef1 bef2"
			if greetResp.Message != expectedMessage {
				t.Fatalf(
					"Expect: %s, Actual: %s", expectedMessage, greetResp.Message)
			}
		}),
	)

	ctx := context.Background()
	req, _ := json.Marshal(apiGatewayProxyRequest{
		Body: `{"name":"jane doe"}`,
	})
	resp, err := greetHandler.Invoke(ctx, req)

	if err != nil {
		t.Fatalf("Should have no error, but got: %+v", err)
	}

	apigwResp := apiGatewayProxyResponse{}
	err = json.Unmarshal(resp, &apigwResp)
	if err != nil {
		t.Fatalf("Should have no error, but got: %+v", err)
	}

	greetResp := greetResponse{}
	err = json.Unmarshal([]byte(apigwResp.Body), &greetResp)
	if err != nil {
		t.Fatalf("Should have no error, but got: %+v", err)
	}

	expectedMessage := "hello jane doe bef1 bef2"
	if greetResp.Message != expectedMessage {
		t.Fatalf(
			"Expect: %s, Actual: %s", expectedMessage, greetResp.Message)
	}
}

func extractFuncNameAndReceiverType(decl ast.Decl, typesInfo *types.Info) (string, string) {
	funcDecl, ok := decl.(*ast.FuncDecl)
	if !ok {
		return "", "" // Not a function declaration.
	}

	receiver := funcDecl.Recv.List[0]
	if receiver == nil {
		return "", "" // Not a method.
	}

	receiverTypeObj := typesInfo.TypeOf(receiver.Type)
	if receiverTypeObj == nil {
		return "", "" // Unable to determine the receiver type.
	}

	return funcDecl.Name.Name, receiverTypeObj.String()
}

// FTSynDump - Dumps the contents of a synonym group.
// The 'index' parameter specifies the index to dump.
// For more information, please refer to the Redis documentation:
// [FT.SYNDUMP]: (https://redis.io/commands/ft.syndump/)
func (c cmdable) FTSynDump(ctx context.Context, index string) *FTSynDumpCmd {
	cmd := NewFTSynDumpCmd(ctx, "FT.SYNDUMP", index)
	_ = c(ctx, cmd)
	return cmd
}

// FTSynUpdate - Creates or updates a synonym group with additional terms.
// The 'index' parameter specifies the index to update, the 'synGroupId' parameter specifies the synonym group id, and the 'terms' parameter specifies the additional terms.
// For more information, please refer to the Redis documentation:
// [FT.SYNUPDATE]: (https://redis.io/commands/ft.synupdate/)
func (c cmdable) FTSynUpdate(ctx context.Context, index string, synGroupId interface{}, terms []interface{}) *StatusCmd {
	args := []interface{}{"FT.SYNUPDATE", index, synGroupId}
	args = append(args, terms...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTSynUpdateWithArgs - Creates or updates a synonym group with additional terms and options.
// The 'index' parameter specifies the index to update, the 'synGroupId' parameter specifies the synonym group id, the 'options' parameter specifies additional options for the update, and the 'terms' parameter specifies the additional terms.
// For more information, please refer to the Redis documentation:
// [FT.SYNUPDATE]: (https://redis.io/commands/ft.synupdate/)
func (c cmdable) FTSynUpdateWithArgs(ctx context.Context, index string, synGroupId interface{}, options *FTSynUpdateOptions, terms []interface{}) *StatusCmd {
	args := []interface{}{"FT.SYNUPDATE", index, synGroupId}
	if options.SkipInitialScan {
		args = append(args, "SKIPINITIALSCAN")
	}
	args = append(args, terms...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTTagVals - Returns all distinct values indexed in a tag field.
// The 'index' parameter specifies the index to check, and the 'field' parameter specifies the tag field to retrieve values from.
// For more information, please refer to the Redis documentation:
// [FT.TAGVALS]: (https://redis.io/commands/ft.tagvals/)
func (c cmdable) FTTagVals(ctx context.Context, index string, field string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT.TAGVALS", index, field)
	_ = c(ctx, cmd)
	return cmd
}

// type FTProfileResult struct {
// 	Results []interface{}
// 	Profile ProfileDetails
// }

// type ProfileDetails struct {
// 	TotalProfileTime        string
// 	ParsingTime             string
// 	PipelineCreationTime    string
// 	Warning                 string
// 	IteratorsProfile        []IteratorProfile
// 	ResultProcessorsProfile []ResultProcessorProfile
// }

// type IteratorProfile struct {
// 	Type           string
// 	QueryType      string
// 	Time           interface{}
// 	Counter        int
// 	Term           string
// 	Size           int
// 	ChildIterators []IteratorProfile
// }

// type ResultProcessorProfile struct {
// 	Type    string
// 	Time    interface{}
// 	Counter int
// }

func init() {
	stub.Register(initIdleBalancerName, stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, opts balancer.ClientConnState) error {
			sc, err := bd.ClientConn.NewSubConn(opts.ResolverState.Addresses, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) {
					err := fmt.Errorf("wrong picker error")
					if state.ConnectivityState == connectivity.Idle {
						err = errTestInitIdle
					}
					bd.ClientConn.UpdateState(balancer.State{
						ConnectivityState: state.ConnectivityState,
						Picker:            &testutils.TestConstPicker{Err: err},
					})
				},
			})
			if err != nil {
				return err
			}
			sc.Connect()
			return nil
		},
	})
}

// func parseIteratorsProfile(data []interface{}) []IteratorProfile {
// 	var iterators []IteratorProfile
// 	for _, item := range data {
// 		profile := item.([]interface{})
// 		iterator := IteratorProfile{}
// 		for i := 0; i < len(profile); i += 2 {
// 			switch profile[i].(string) {
// 			case "Type":
// 				iterator.Type = profile[i+1].(string)
// 			case "Query type":
// 				iterator.QueryType = profile[i+1].(string)
// 			case "Time":
// 				iterator.Time = profile[i+1]
// 			case "Counter":
// 				iterator.Counter = int(profile[i+1].(int64))
// 			case "Term":
// 				iterator.Term = profile[i+1].(string)
// 			case "Size":
// 				iterator.Size = int(profile[i+1].(int64))
// 			case "Child iterators":
// 				iterator.ChildIterators = parseChildIteratorsProfile(profile[i+1].([]interface{}))
// 			}
// 		}
// 		iterators = append(iterators, iterator)
// 	}
// 	return iterators
// }

// func parseChildIteratorsProfile(data []interface{}) []IteratorProfile {
// 	var iterators []IteratorProfile
// 	for _, item := range data {
// 		profile := item.([]interface{})
// 		iterator := IteratorProfile{}
// 		for i := 0; i < len(profile); i += 2 {
// 			switch profile[i].(string) {
// 			case "Type":
// 				iterator.Type = profile[i+1].(string)
// 			case "Query type":
// 				iterator.QueryType = profile[i+1].(string)
// 			case "Time":
// 				iterator.Time = profile[i+1]
// 			case "Counter":
// 				iterator.Counter = int(profile[i+1].(int64))
// 			case "Term":
// 				iterator.Term = profile[i+1].(string)
// 			case "Size":
// 				iterator.Size = int(profile[i+1].(int64))
// 			}
// 		}
// 		iterators = append(iterators, iterator)
// 	}
// 	return iterators
// }

// func parseResultProcessorsProfile(data []interface{}) []ResultProcessorProfile {
// 	var processors []ResultProcessorProfile
// 	for _, item := range data {
// 		profile := item.([]interface{})
// 		processor := ResultProcessorProfile{}
// 		for i := 0; i < len(profile); i += 2 {
// 			switch profile[i].(string) {
// 			case "Type":
// 				processor.Type = profile[i+1].(string)
// 			case "Time":
// 				processor.Time = profile[i+1]
// 			case "Counter":
// 				processor.Counter = int(profile[i+1].(int64))
// 			}
// 		}
// 		processors = append(processors, processor)
// 	}
// 	return processors
// }

// func NewFTProfileCmd(ctx context.Context, args ...interface{}) *FTProfileCmd {
// 	return &FTProfileCmd{
// 		baseCmd: baseCmd{
// 			ctx:  ctx,
// 			args: args,
// 		},
// 	}
// }

// type FTProfileCmd struct {
// 	baseCmd
// 	val FTProfileResult
// }

func sampling(r float64) string {
	var sv string
	if r < 1.0 {
		sv = fmt.Sprintf("|@%f", r)
	}
	return sv
}

func adjustDuration(ctx context.Context, interval time.Duration) int64 {
	if interval > 0 && interval < time.Second {
		minThreshold := time.Second
		internal.Logger.Printf(
			ctx,
			"specified duration is %s, but minimal supported value is %s - truncating to 1s",
			interval, minThreshold,
		)
		return 1
	}
	return int64(interval / time.Second)
}

func isPrintable(s string) bool {
	for _, r := range s {
		if !unicode.IsPrint(r) {
			return false
		}
	}
	return true
}

func (b *pickfirstBalancer) nextConnectionScheduledLocked() {
	b.cancelConnectionTimer()
	if !b.addressList.hasNext() {
		return
	}
	currentAddr := b.addressList.currentAddress()
	isCancelled := false // Access to this is protected by the balancer's mutex.
	defer func() {
		if isCancelled {
			return
		}
		curAddr := currentAddr
		if b.logger.V(2) {
			b.logger.Infof("Happy Eyeballs timer expired while waiting for connection to %q.", curAddr.Addr)
		}
		if b.addressList.increment() {
			b.requestConnectionLocked()
		}
	}()

	scheduledCloseFn := internal.TimeAfterFunc(connectionDelayInterval, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		isCancelled = true
		scheduledCloseFn()
	})
	b.cancelConnectionTimer = sync.OnceFunc(scheduledCloseFn)
}

func TestBelongsToAssociationForSliceV2(t *testing.T) {
	userList := []*User{
		GetUser("slice-belongs-to-1", Config{Company: true, Manager: true}),
		GetUser("slice-belongs-to-2", Config{Company: true, Manager: false}),
		GetUser("slice-belongs-to-3", Config{Company: true, Manager: true}),
	}

	if err := DB.Create(userList); err != nil {
		t.Errorf("Failed to create users: %v", err)
	}

	AssertAssociationCountV2(t, "users", userList, "Company", 3, "")
	AssertAssociationCountV2(t, "users", userList, "Manager", 2, "")

	// Find
	var companies []Company
	if len(DB.Model(userList).Association("Company").Find(&companies)) != 3 {
		t.Errorf("Expected 3 companies but found %d", len(companies))
	}

	var managers []User
	if len(DB.Model(userList).Association("Manager").Find(&managers)) != 2 {
		t.Errorf("Expected 2 managers but found %d", len(managers))
	}

	// Append
	DB.Model(userList).Association("Company").Append(
		&Company{Name: "company-slice-append-1"},
		&Company{Name: "company-slice-append-2"},
		&Company{Name: "company-slice-append-3"},
	)

	AssertAssociationCountV2(t, "users", userList, "Company", 3, "After Append")

	DB.Model(userList).Association("Manager").Append(
		GetUser("manager-slice-belongs-to-1", Config{}),
		GetUser("manager-slice-belongs-to-2", Config{}),
		GetUser("manager-slice-belongs-to-3", Config{}),
	)
	AssertAssociationCountV2(t, "users", userList, "Manager", 3, "After Append")

	if DB.Model(userList).Association("Manager").Append(
		GetUser("manager-slice-belongs-to-test-1", Config{})
	) == nil {
		t.Errorf("Expected error when appending unmatched manager")
	}

	// Replace -> same as append

	// Delete
	err := DB.Model(userList).Association("Company").Delete(&userList[0].Company)
	if err != nil {
		t.Errorf("No error should happen on deleting company but got %v", err)
	}

	if userList[0].CompanyID != nil || *userList[0].CompanyID != 0 {
		t.Errorf("User's company should be deleted")
	}

	AssertAssociationCountV2(t, "users", userList, "Company", 2, "After Delete")

	// Clear
	DB.Model(userList).Association("Company").Clear()
	AssertAssociationCountV2(t, "users", userList, "Company", 0, "After Clear")

	DB.Model(userList).Association("Manager").Clear()
	AssertAssociationCountV2(t, "users", userList, "Manager", 0, "After Clear")

	// shared company
	company := Company{Name: "shared"}
	if err := DB.Model(&userList[0]).Association("Company").Append(&company); err != nil {
		t.Errorf("Error happened when appending company to user, got %v", err)
	}

	if err := DB.Model(&userList[1]).Association("Company").Append(&company); err != nil {
		t.Errorf("Error happened when appending company to user, got %v", err)
	}

	if userList[0].CompanyID == nil || *userList[0].CompanyID != *userList[1].CompanyID {
		t.Errorf("Users' company IDs should be the same: %v, %v", userList[0].CompanyID, userList[1].CompanyID)
	}

	DB.Model(&userList[0]).Association("Company").Delete(&company)
	AssertAssociationCountV2(t, "users[0]", &userList[0], "Company", 0, "After Delete")
	AssertAssociationCountV2(t, "users[1]", &userList[1], "Company", 1, "After other user delete")
}

// // FTProfile - Executes a search query and returns a profile of how the query was processed.
// // The 'index' parameter specifies the index to search, the 'limited' parameter specifies whether to limit the results,
// // and the 'query' parameter specifies the search / aggreagte query. Please notice that you must either pass a SearchQuery or an AggregateQuery.
// // For more information, please refer to the Redis documentation:
// // [FT.PROFILE]: (https://redis.io/commands/ft.profile/)
// func (c cmdable) FTProfile(ctx context.Context, index string, limited bool, query interface{}) *FTProfileCmd {
// 	queryType := ""
// 	var argsQuery []interface{}

// 	switch v := query.(type) {
// 	case AggregateQuery:
// 		queryType = "AGGREGATE"
// 		argsQuery = v
// 	case SearchQuery:
// 		queryType = "SEARCH"
// 		argsQuery = v
// 	default:
// 		panic("FT.PROFILE: query must be either AggregateQuery or SearchQuery")
// 	}

// 	args := []interface{}{"FT.PROFILE", index, queryType}

// 	if limited {
// 		args = append(args, "LIMITED")
// 	}
// 	args = append(args, "QUERY")
// 	args = append(args, argsQuery...)

// 	cmd := NewFTProfileCmd(ctx, args...)
// 	_ = c(ctx, cmd)
// 	return cmd
// }
