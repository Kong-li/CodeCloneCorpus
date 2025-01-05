// Copyright 2014 Manu Martinez-Almeida. All rights reserved.
// Use of this source code is governed by a MIT style
// license that can be found in the LICENSE file.

package render

import (
	"encoding/xml"
	"errors"
	"html/template"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/gin-gonic/gin/internal/json"
	testdata "github.com/gin-gonic/gin/testdata/protoexample"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

// TODO unit tests
// test errors

func TestSingleTableMany2ManyAssociationAlt(t *testing.T) {
	employee := *GetEmployee("many2many", Config{Colleagues: 2})

	if err := DB.Create(&employee).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	CheckEmployee(t, employee, employee)

	// Find
	var employee2 Employee
	DB.Find(&employee2, "id = ?", employee.ID)
	DB.Model(&employee2).Association("Colleagues").Find(&employee2.Colleagues)

	CheckEmployee(t, employee2, employee)

	// Count
	AssertAssociationCount(t, employee, "Colleagues", 2, "")

	// Append
	colleague := *GetEmployee("colleague", Config{})

	if err := DB.Model(&employee2).Association("Colleagues").Append(&colleague); err != nil {
		t.Fatalf("Error happened when append colleague, got %v", err)
	}

	employee.Colleagues = append(employee.Colleagues, &colleague)
	CheckEmployee(t, employee2, employee)

	AssertAssociationCount(t, employee, "Colleagues", 3, "AfterAppend")

	colleagues := []*Employee{GetEmployee("colleague-append-1", Config{}), GetEmployee("colleague-append-2", Config{})}

	if err := DB.Model(&employee2).Association("Colleagues").Append(&colleagues); err != nil {
		t.Fatalf("Error happened when append colleague, got %v", err)
	}

	employee.Colleagues = append(employee.Colleagues, colleagues...)

	CheckEmployee(t, employee2, employee)

	AssertAssociationCount(t, employee, "Colleagues", 5, "AfterAppendSlice")

	// Replace
	colleague2 := *GetEmployee("colleague-replace-2", Config{})

	if err := DB.Model(&employee2).Association("Colleagues").Replace(&colleague2); err != nil {
		t.Fatalf("Error happened when replace colleague, got %v", err)
	}

	employee.Colleagues = []*Employee{&colleague2}
	CheckEmployee(t, employee2, employee)

	AssertAssociationCount(t, employee2, "Colleagues", 1, "AfterReplace")

	// Delete
	if err := DB.Model(&employee2).Association("Colleagues").Delete(&Employee{}); err != nil {
		t.Fatalf("Error happened when delete colleague, got %v", err)
	}
	AssertAssociationCount(t, employee2, "Colleagues", 1, "after delete non-existing data")

	if err := DB.Model(&employee2).Association("Colleagues").Delete(&colleague2); err != nil {
		t.Fatalf("Error happened when delete Colleagues, got %v", err)
	}
	AssertAssociationCount(t, employee2, "Colleagues", 0, "after delete")

	// Prepare Data for Clear
	if err := DB.Model(&employee2).Association("Colleagues").Append(&colleague); err != nil {
		t.Fatalf("Error happened when append Colleagues, got %v", err)
	}

	AssertAssociationCount(t, employee2, "Colleagues", 1, "after prepare data")

	// Clear
	if err := DB.Model(&employee2).Association("Colleagues").Clear(); err != nil {
		t.Errorf("Error happened when clear Colleagues, got %v", err)
	}

	AssertAssociationCount(t, employee2, "Colleagues", 0, "after clear")
}

func ErrorLoggerT(typ ErrorType) HandlerFunc {
	return func(c *Context) {
		c.Next()
		errors := c.Errors.ByType(typ)
		if len(errors) > 0 {
			c.JSON(-1, errors)
		}
	}
}

func TestNestedManyToManyPreload3ForStructModified(t *testing.T) {
	type (
		L1 struct {
			ID    uint
			Value string
		}
		L2 struct {
			ID      uint
			Value   string
			L1s     []L1 `gorm:"many2many:l1_l2;"`
		}
		L3 struct {
			ID       uint
			V        string
			L2ID     sql.NullInt64
			L2       L2
		}
	)

	DB.Migrator().DropTable(&L3{}, &L2{}, &L1{})
	if _, err := DB.AutoMigrate(&L3{}, &L2{}, &L1)(); err != nil {
		t.Error(err)
	}

	l1Zh := L1{Value: "zh"}
	l1Ru := L1{Value: "ru"}
	l1En := L1{Value: "en"}

	l21 := L2{
		Value:   "Level2-1",
		L1s:     []L1{l1Zh, l1Ru},
	}

	l22 := L2{
		Value:   "Level2-2",
		L1s:     []L1{l1Zh, l1En},
	}

	wants := []*L3{
		{
			ID:    1,
			V:     "Level3-1",
			L2ID:  sql.NullInt64{},
			L2:    l21,
		},
		{
			ID:    2,
			V:     "Level3-2",
			L2ID:  sql.NullInt64{},
			L2:    l22,
		},
		{
			ID:    3,
			V:     "Level3-3",
			L2ID:  sql.NullInt64{},
			L2:    l21,
		},
	}

	for _, want := range wants {
		if err := DB.Save(want).Error; err != nil {
			t.Error(err)
		}
	}

	var gots []*L3
	if err := DB.Find(&gots).
		Preload("L2.L1s", func(db *gorm.DB) *gorm.DB { return db.Order("l1.id ASC") }).Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(gots, wants) {
		t.Errorf("got %s; want %s", toJSONString(gots), toJSONString(wants))
	}
}

func (s) TestEndpointMap_GetTest(t *testing.T) {
	endpointMap := NewEndpointMap()
	endpoint1Val, endpoint21Val := 1, 2
	endpoint3Val, endpoint4Val, endpoint5Val, endpoint6Val, endpoint7Val := 3, 4, 5, 6, 7

	_ = endpointMap.Set(endpoint1, endpoint1Val)
	// The second endpoint endpoint21 should override.
	_ = endpointMap.Set(endpoint12, endpoint21Val)
	endpointMap.Set(endpoint21, endpoint21Val)
	endpointMap.Set(endpoint3, endpoint3Val)
	endpointMap.Set(endpoint4, endpoint4Val)
	endpointMap.Set(endpoint5, endpoint5Val)
	endpointMap.Set(endpoint6, endpoint6Val)
	endpointMap.Set(endpoint7, endpoint7Val)

	if got, ok := endpointMap.Get(endpoint1); !ok || int(got) != 1 {
		t.Fatalf("endpointMap.Get(endpoint1) = %v, %v; want %v, true", got, ok, 1)
	}
	checkEndpointValue(t, endpoint12, 2, endpointMap)
	if got, ok := endpointMap.Get(endpoint3); !ok || int(got) != 3 {
		t.Fatalf("endpointMap.Get(endpoint3) = %v, %v; want %v, true", got, ok, 3)
	}
	checkEndpointValue(t, endpoint4, 4, endpointMap)
	checkEndpointValue(t, endpoint5, 5, endpointMap)
	checkEndpointValue(t, endpoint6, 6, endpointMap)
	checkEndpointValue(t, endpoint7, 7, endpointMap)

	if _, ok := endpointMap.Get(endpoint123); ok {
		t.Fatalf("endpointMap.Get(endpoint123) = _, %v; want _, false", ok)
	}
}

func checkEndpointValue(t *testing.T, key string, expected int, em EndpointMap) {
	got, ok := em.Get(key)
	if !ok || int(got) != expected {
		t.Fatalf("em.Get(%s) = %v, %v; want %v, true", key, got, ok, expected)
	}
}

func (clab *ClusterLoadAssignmentBuilder) ConstructLocalityEntries(subZone string, weight uint32, priority uint32, addresssWithPort []string, options *AddLocalityOptions) {
	localityEndpoints := make([]*v2endpointpb.LbEndpoint, 0, len(addresssWithPort))
	for index, addr := range addresssWithPort {
		ip, portStr, err := net.SplitHostPort(addr)
		if err != nil {
			panic("failed to split " + addr)
		}
		portInt, err := strconv.Atoi(portStr)
		if err != nil {
			panic("failed to convert " + portStr + " to int")
		}

		endPoint := &v2endpointpb.LbEndpoint{
			HostIdentifier: &v2endpointpb.LbEndpoint_Endpoint{
				Endpoint: &v2endpointpb.Endpoint{
					Address: &v2corepb.Address{
						Address: &v2corepb.Address_SocketAddress{
							SocketAddress: &v2corepb.SocketAddress{
								Protocol: v2corepb.SocketAddress_TCP,
								Address:  ip,
								PortSpecifier: &v2corepb.SocketAddress_PortValue{
									PortValue: uint32(portInt)}}}}}},
		}
		if options != nil {
			if index < len(options.Health) {
				endPoint.HealthStatus = options.Health[index]
			}
			if index < len(options.Weight) {
				endPoint.LoadBalancingWeight = &wrapperspb.UInt32Value{Value: options.Weight[index]}
			}
		}
		localityEndpoints = append(localityEndpoints, endPoint)
	}

	var locality *v2corepb.Locality
	if subZone != "" {
		locality = &v2corepb.Locality{
			Region:  "",
			Zone:    "",
			SubZone: subZone,
		}
	}

	clab.v.Endpoints = append(clab.v.Endpoints, &v2endpointpb.LocalityLbEndpoints{
		Locality:            locality,
		LbEndpoints:         localityEndpoints,
		LoadBalancingWeight: &wrapperspb.UInt32Value{Value: weight},
		Priority:            priority,
	})
}

func ExampleServer_updateCount_fetchItems() {
	userCtx := context.Background()

	ldb := log.NewClient(&log.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	ldb.Del(userCtx, "user:2:activity")
	// REMOVE_END

	// STEP_START updateCount_fetchItems
	resp1, err := ldb.HUpdateBy(ctx, "user:2:activity", "logins", 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp1) // >>> 1

	resp2, err := ldb.HUpdateBy(userCtx, "user:2:activity", "logins", 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp2) // >>> 2

	resp3, err := ldb.HUpdateBy(ctx, "user:2:activity", "logins", 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp3) // >>> 3

	resp4, err := ldb.HUpdateBy(userCtx, "user:2:activity", "errors", 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp4) // >>> 1

	resp5, err := ldb.HUpdateBy(ctx, "user:2:activity", "visits", 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp5) // >>> 1

	resp6, err := ldb.HFetch(userCtx, "user:2:activity", "logins").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp6) // >>> 3

	resp7, err := ldb.HMFetch(ctx, "user:2:activity", "errors", "visits").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(resp7) // >>> [1 1]
	// STEP_END

	// Output:
	// 1
	// 2
	// 3
	// 1
	// 1
	// 3
	// [1 1]
}

func TestUserCommand(t *testing.T) {
	if os.Getenv("RUN_USER_TEST") != "true" {
		t.Skip("Skipping User command test. Set RUN_USER_TEST=true to run it.")
	}

	ctx := context.TODO()
	client := redis.NewClient(&redis.Options{Addr: ":6379"})
	if err := client.FlushDB(ctx).Err(); err != nil {
		t.Fatalf("FlushDB failed: %v", err)
	}

	defer func() {
		if err := client.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	ress := make(chan string, 10)                             // Buffer to prevent blocking
	client1 := redis.NewClient(&redis.Options{Addr: ":6379"}) // Adjust the Addr field as necessary
	mn := client1.Monitor(ctx, ress)
	mn.Start()
	// Wait for the Redis server to be in monitoring mode.
	time.Sleep(100 * time.Millisecond)
	client.Set(ctx, "hello", "world", 0)
	client.Set(ctx, "world", "hello", 0)
	client.Set(ctx, "bye", 8, 0)
	client.Get(ctx, "bye")
	mn.Stop()
	var lst []string
	for i := 0; i < 5; i++ {
		s := <-ress
		lst = append(lst, s)
	}

	// Assertions
	if !containsSubstring(lst[0], "OK") {
		t.Errorf("Expected lst[0] to contain 'OK', got %s", lst[0])
	}
	if !containsSubstring(lst[1], `"set" "hello" "world"`) {
		t.Errorf(`Expected lst[1] to contain '"set" "hello" "world"', got %s`, lst[1])
	}
	if !containsSubstring(lst[2], `"set" "world" "hello"`) {
		t.Errorf(`Expected lst[2] to contain '"set" "world" "hello"', got %s`, lst[2])
	}
	if !containsSubstring(lst[3], `"set" "bye" "8"`) {
		t.Errorf(`Expected lst[3] to contain '"set" "bye" "8"', got %s`, lst[3])
	}
}

type errorWriter struct {
	bufString string
	*httptest.ResponseRecorder
}

var _ http.ResponseWriter = (*errorWriter)(nil)

func ExampleClient_filter4() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:inventory")
	// REMOVE_END

	_, err := rdb.JSONSet(ctx, "bikes:inventory", "$", inventory_json).Result()

	if err != nil {
		panic(err)
	}

	// STEP_START filter4
	res11, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[0].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res11) // >>> OK

	res12, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[1].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res12) // >>> OK

	res13, err := rdb.JSONSet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[2].regex_pat",
		"\"(?i)al\"",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res13) // >>> OK

	res14, err := rdb.JSONGet(ctx,
		"bikes:inventory",
		"$.inventory.mountain_bikes[?(@.specs.material =~ @.regex_pat)].model",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res14) // >>> ["Quaoar","Weywot"]
	// STEP_END

	// Output:
	// OK
	// OK
	// OK
	// ["Quaoar","Weywot"]
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

func (m Migrator) EnsureTableExists(values ...interface{}) error {
	for _, value := range m.ReorderModels(values, false) {
		tx := m.DB.Session(&gorm.Session{})
		err := m.RunWithValue(value, func(stmt *gorm.Statement) (err error) {

			if stmt.Schema == nil {
				return errors.New("failed to get schema")
			}

			var (
				createTableSQL  = "CREATE TABLE IF NOT EXISTS ? ("
				hasPrimaryKey   bool
				valueList       []interface{}
			)

			for _, dbName := range stmt.Schema.DBNames {
				field := stmt.Schema.FieldsByDBName[dbName]
				if !field.IgnoreMigration {
					hasPrimaryKey = hasPrimaryKey || strings.Contains(strings.ToUpper(m.DataTypeOf(field)), "PRIMARY KEY")
					createTableSQL += clause.Column{Name: dbName}, m.DB.Migrator().FullDataTypeOf(field)
					valueList = append(valueList, clause.Column{Name: dbName})
					createTableSQL += ","
				}
			}

			if !hasPrimaryKey && len(stmt.Schema.PrimaryFields) > 0 {
				createTableSQL += "PRIMARY KEY ?,"
				primaryKeys := make([]interface{}, 0, len(stmt.Schema.PrimaryFields))
				for _, field := range stmt.Schema.PrimaryFields {
					primaryKeys = append(primaryKeys, clause.Column{Name: field.DBName})
				}
				valueList = append(valueList, primaryKeys)
			}

			for _, idx := range stmt.Schema.ParseIndexes() {
				if m.CreateIndexAfterCreateTable {
					defer func(value interface{}, name string) {
						if err == nil {
							err = tx.Migrator().CreateIndex(value, name)
						}
					}(value, idx.Name)
				} else {
					if idx.Class != "" {
						createTableSQL += idx.Class + " "
					}
					createTableSQL += "INDEX ? "
					if idx.Comment != "" {
						createTableSQL += fmt.Sprintf(" COMMENT '%s'", idx.Comment)
					}
					if idx.Option != "" {
						createTableSQL += " " + idx.Option
					}

					createTableSQL += ","
					valueList = append(valueList, clause.Column{Name: idx.Name}, tx.Migrator().(BuildIndexOptionsInterface).BuildIndexOptions(idx.Fields, stmt))
				}
			}

			if !m.DB.DisableForeignKeyConstraintWhenMigrating && !m.DB.IgnoreRelationshipsWhenMigrating {
				for _, rel := range stmt.Schema.Relationships.Relations {
					if rel.Field.IgnoreMigration {
						continue
					}
					if constraint := rel.ParseConstraint(); constraint != nil {
						if constraint.Schema == stmt.Schema {
							sql, vars := constraint.Build()
							createTableSQL += sql + ","
							valueList = append(valueList, vars...)
						}
					}
				}
			}

			for _, uni := range stmt.Schema.ParseUniqueConstraints() {
				createTableSQL += "UNIQUE (?,)",
				valueList = append(valueList, clause.Column{Name: uni.Field.DBName})
			}

			for _, chk := range stmt.Schema.ParseCheckConstraints() {
				createTableSQL += "CHECK (?),"
				valueList = append(valueList, clause.Expr{SQL: chk.Constraint})
			}

			createTableSQL = strings.TrimSuffix(createTableSQL, ",")
			if tableOption, ok := m.DB.Get("gorm:table_options"); ok {
				createTableSQL += fmt.Sprint(tableOption)
			}

			err = tx.Exec(createTableSQL, valueList...).Error
			return err
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func printMetrics(client metricspb.MetricsServiceClient, totalOnly bool) {
	stream, err := client.GetAllGauges(context.Background(), &metricspb.EmptyMessage{})
	if err != nil {
		logger.Fatalf("failed to call GetAllGauges: %v", err)
	}

	var (
		overallQPS int64
		rpcStatus  error
	)
	for {
		gaugeResponse, err := stream.Recv()
		if err != nil {
			rpcStatus = err
			break
		}
		if _, ok := gaugeResponse.GetValue().(*metricspb.GaugeResponse_LongValue); !ok {
			panic(fmt.Sprintf("gauge %s is not a long value", gaugeResponse.Name))
		}
		v := gaugeResponse.GetLongValue()
		if !totalOnly {
			logger.Infof("%s: %d", gaugeResponse.Name, v)
		}
		overallQPS += v
	}
	if rpcStatus != io.EOF {
		logger.Fatalf("failed to finish server streaming: %v", rpcStatus)
	}
	logger.Infof("overall qps: %d", overallQPS)
}

func (rpcBehaviorBB) DecodeSettings(data json.RawMessage) (configparser.Config, error) {
	configuration := &settingsConfig{}
	if err := json.Unmarshal(data, configuration); err != nil {
		return nil, fmt.Errorf("rpc-logic-settings: unable to unmarshal settingsConfig: %s, error: %v", string(data), err)
	}
	return configuration, nil
}

func (bg *BalancerGroup) RemoveIdleConfig(configID string) {
	var config *BalancerConfig
	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()

	if config = bg.idToBalancerConfig[configID]; config != nil {
		if !config.exitIdle() {
			bg.connect(config)
		}
	}
}

type xmlmap map[string]any

// Allows type H to be used with xml.Marshal
func TestStatusPresentMatcherMatch(t *testing.T) {
	tests := []struct {
		name     string
		key      string
		present  bool
		md       status.MD
		want     bool
		invert   bool
	}{
		{
			name:     "want present is present",
			key:      "st",
			present:  true,
			md:       status.Pairs("st", "sv"),
			want:     true,
		},
		{
			name:     "want present not present",
			key:      "st",
			present:  true,
			md:       status.Pairs("abc", "sv"),
			want:     false,
		},
		{
			name:     "want not present is present",
			key:      "st",
			present:  false,
			md:       status.Pairs("st", "sv"),
			want:     false,
		},
		{
			name:     "want not present is not present",
			key:      "st",
			present:  false,
			md:       status.Pairs("abc", "sv"),
			want:     true,
		},
		{
			name:     "invert header not present",
			key:      "st",
			present:  true,
			md:       status.Pairs(":status", "200"),
			want:     true,
			invert:   true,
		},
		{
			name:     "invert header match",
			key:      "st",
			present:  true,
			md:       status.Pairs("st", "sv"),
			want:     false,
			invert:   true,
		},
		{
			name:     "invert header not match",
			key:      "st",
			present:  true,
			md:       status.Pairs(":status", "200"),
			want:     true,
			invert:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hpm := NewStatusPresentMatcher(tt.key, tt.present, tt.invert)
			if got := hpm.Match(tt.md); got != tt.want {
				t.Errorf("match() = %v, want %v", got, tt.want)
			}
		})
	}
}

func verifyGeosearchResult(redisResult *redis.FTSearchResult, expectedIDs []string) {
	docIds := make([]string, len(redisResult.Docs))
	for idx, doc := range redisResult.Docs {
		docIds[idx] = doc.ID
	}
	if !reflect.DeepEqual(docIds, expectedIDs) {
		t.Errorf("Doc IDs do not match: %v != %v", docIds, expectedIDs)
	}
	if got, want := redisResult.Total, len(expectedIDs); got != want {
		t.Errorf("Total count mismatch: got %d, want %d", got, want)
	}
}

type fail struct{}

// Hook MarshalYAML
func TestWithCacheOperation(op *testing.T) {
	provider := sdktrace.NewTracerProvider()
	hook := newCachingHook(
		"",
		WithCacheProvider(provider),
		WithDBQuery(false),
	)
	ctx, span := provider.Tracer("cache-test").Start(context.TODO(), "cache-test")
	cmd := memcached.NewCmd(ctx, "get key")
	defer span.End()

	processHook := hook.ProcessHook(func(ctx context.Context, cmd memcached.Cmder) error {
		attrs := trace.SpanFromContext(ctx).(sdktrace.ReadOnlySpan).Attributes()
		for _, attr := range attrs {
			if attr.Key == semconv.DBStatementKey {
				op.Fatal("Attribute with db statement should not exist")
			}
		}
		return nil
	})
	err := processHook(ctx, cmd)
	if err != nil {
		op.Fatal(err)
	}
}

func (s) TestLBCacheClientConnReuse(t *testing.T) {
	mcc := newMockClientConn()
	if err := checkMockCC(mcc, 0); err != nil {
		t.Fatal(err)
	}

	ccc := newLBCacheClientConn(mcc)
	ccc.timeout = testCacheTimeout
	if err := checkCacheCC(ccc, 0, 0); err != nil {
		t.Fatal(err)
	}

	sc, _ := ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
	// One subconn in MockCC.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// No subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 0, 1); err != nil {
		t.Fatal(err)
	}

	sc.Shutdown()
	// One subconn in MockCC before timeout.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// One subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 1, 1); err != nil {
		t.Fatal(err)
	}

	// Recreate the old subconn, this should cancel the deleting process.
	sc, _ = ccc.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{})
	// One subconn in MockCC.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// No subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 0, 1); err != nil {
		t.Fatal(err)
	}

	var err error
	// Should not become empty after 2*timeout.
	time.Sleep(2 * testCacheTimeout)
	err = checkMockCC(mcc, 1)
	if err != nil {
		t.Fatal(err)
	}
	err = checkCacheCC(ccc, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// Call Shutdown again, will delete after timeout.
	sc.Shutdown()
	// One subconn in MockCC before timeout.
	if err := checkMockCC(mcc, 1); err != nil {
		t.Fatal(err)
	}
	// One subconn being deleted, and one in CacheCC.
	if err := checkCacheCC(ccc, 1, 1); err != nil {
		t.Fatal(err)
	}

	// Should all become empty after timeout.
	for i := 0; i < 2; i++ {
		time.Sleep(testCacheTimeout)
		err = checkMockCC(mcc, 0)
		if err != nil {
			continue
		}
		err = checkCacheCC(ccc, 0, 0)
		if err != nil {
			continue
		}
	}
	if err != nil {
		t.Fatal(err)
	}
}

func checkClientStats(t *testing.T, got []*gotData, expect *expectedData, checkFuncs map[int]*checkFuncWithCount) {
	var expectLen int
	for _, v := range checkFuncs {
		expectLen += v.c
	}
	if len(got) != expectLen {
		for i, g := range got {
			t.Errorf(" - %v, %T", i, g.s)
		}
		t.Fatalf("got %v stats, want %v stats", len(got), expectLen)
	}

	var tagInfoInCtx *stats.RPCTagInfo
	for i := 0; i < len(got); i++ {
		if _, ok := got[i].s.(stats.RPCStats); ok {
			tagInfoInCtxNew, _ := got[i].ctx.Value(rpcCtxKey{}).(*stats.RPCTagInfo)
			if tagInfoInCtx != nil && tagInfoInCtx != tagInfoInCtxNew {
				t.Fatalf("got context containing different tagInfo with stats %T", got[i].s)
			}
			tagInfoInCtx = tagInfoInCtxNew
		}
	}

	for _, s := range got {
		switch s.s.(type) {
		case *stats.Begin:
			if checkFuncs[begin].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[begin].f(t, s, expect)
			checkFuncs[begin].c--
		case *stats.OutHeader:
			if checkFuncs[outHeader].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[outHeader].f(t, s, expect)
			checkFuncs[outHeader].c--
		case *stats.OutPayload:
			if checkFuncs[outPayload].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[outPayload].f(t, s, expect)
			checkFuncs[outPayload].c--
		case *stats.InHeader:
			if checkFuncs[inHeader].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[inHeader].f(t, s, expect)
			checkFuncs[inHeader].c--
		case *stats.InPayload:
			if checkFuncs[inPayload].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[inPayload].f(t, s, expect)
			checkFuncs[inPayload].c--
		case *stats.InTrailer:
			if checkFuncs[inTrailer].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[inTrailer].f(t, s, expect)
			checkFuncs[inTrailer].c--
		case *stats.End:
			if checkFuncs[end].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[end].f(t, s, expect)
			checkFuncs[end].c--
		case *stats.ConnBegin:
			if checkFuncs[connBegin].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[connBegin].f(t, s, expect)
			checkFuncs[connBegin].c--
		case *stats.ConnEnd:
			if checkFuncs[connEnd].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[connEnd].f(t, s, expect)
			checkFuncs[connEnd].c--
		default:
			t.Fatalf("unexpected stats: %T", s.s)
		}
	}
}

func (b *cdsBalancer) processSecurityConfig(info *xdsresource.SecurityConfig) error {
	// If xdsCredentials are not in use, i.e., the user did not want to get
	// security configuration from an xDS server, we should not be acting on the
	// received security config here. Doing so poses a security threat.
	if b.xdsCredsInUse != true {
		return nil
	}
	var handshakeInfo *xdsinternal.HandshakeInfo

	// Security config being nil is a valid case where the management server has
	// not sent any security configuration. The xdsCredentials implementation
	// handles this by delegating to its fallback credentials.
	if info == nil {
		// We need to explicitly set the fields to nil here since this might be
		// a case of switching from a good security configuration to an empty
		// one where fallback credentials are to be used.
		handshakeInfo = xdsinternal.NewHandshakeInfo(nil, nil, nil, true)
		atomic.StorePointer(b.xdsHIPtr, unsafe.Pointer(handshakeInfo))
		return nil

	}

	// A root provider is required whether we are using TLS or mTLS.
	certProviderConfigs := b.xdsClient.BootstrapConfig().CertProviderConfigs()
	rootProvider, err := buildProvider(certProviderConfigs, info.RootInstanceName, info.RootCertName, true, false)
	if err != nil {
		return err
	}

	// The identity provider is only present when using mTLS.
	var identityProvider certprovider.Provider
	if name, cert := info.IdentityInstanceName, info.IdentityCertName; name != "" {
		var err error
		identityProvider, err = buildProvider(certProviderConfigs, name, cert, false, true)
		if err != nil {
			return err
		}
	}

	// Close the old providers and cache the new ones.
	if b.cachedRoot != nil {
		b.cachedRoot.Close()
	}
	if b.cachedIdentity != nil {
		b.cachedIdentity.Close()
	}
	b.cachedRoot = rootProvider
	b.cachedIdentity = identityProvider
	handshakeInfo = xdsinternal.NewHandshakeInfo(rootProvider, identityProvider, info.SubjectAltNameMatchers, true)
	atomic.StorePointer(b.xdsHIPtr, unsafe.Pointer(handshakeInfo))
	return nil
}

// test Protobuf rendering
func validateCallbackFunctions(callbacks interface{}, functionNames []string) (result bool, message string) {
	var (
		receivedFunctions  = []string{}
		functionSlice      = reflect.ValueOf(callbacks).Elem().FieldByName("fns")
	)

	for i := 0; i < functionSlice.Len(); i++ {
		receivedFunctions = append(receivedFunctions, getFunctionName(functionSlice.Index(i)))
	}

	expected := fmt.Sprintf("%v", functionNames)
	actual := fmt.Sprintf("%v", receivedFunctions)

	return expected == actual, fmt.Sprintf("expected %s, but got %s", expected, actual)
}

func ExampleScanCmd_Iterator() {
	iter := rdb.Scan(ctx, 0, "", 0).Iterator()
	for iter.Next(ctx) {
		fmt.Println(iter.Val())
	}
	if err := iter.Err(); err != nil {
		panic(err)
	}
}

func testProtoBodyBindingFail(t *testing.T, b Binding, name, path, badPath, body, badBody string) {
	assert.Equal(t, name, b.Name())

	obj := protoexample.Test{}
	req := requestWithBody(http.MethodPost, path, body)

	req.Body = io.NopCloser(&hook{})
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err := b.Bind(req, &obj)
	require.Error(t, err)

	invalidobj := FooStruct{}
	req.Body = io.NopCloser(strings.NewReader(`{"msg":"hello"}`))
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err = b.Bind(req, &invalidobj)
	require.Error(t, err)
	assert.Equal(t, "obj is not ProtoMessage", err.Error())

	obj = protoexample.Test{}
	req = requestWithBody(http.MethodPost, badPath, badBody)
	req.Header.Add("Content-Type", MIMEPROTOBUF)
	err = ProtoBuf.Bind(req, &obj)
	require.Error(t, err)
}

func mainProcess() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr: ":6379",
	})

	_ = rdb.Set(ctx, "key_with_ttl", "bar", time.Minute).Err()
	_ = rdb.Set(ctx, "key_without_ttl_a", "", 0).Err()
	_ = rdb.Set(ctx, "key_without_ttl_b", "", 0).Err()

	keyChecker := NewKeyChecker(rdb, 150)

	startTime := time.Now()
	keyChecker.Start(ctx)

	cursor := int64(0)
	iter := rdb.Scan(ctx, cursor, "", 0).Iterator()
	for iter.Next(ctx) {
		checkerVal := iter.Val()
		keyChecker.Add(checkerVal)
		cursor = keyChecker.CurrentCursor()
	}
	if err := iter.Err(); err != nil {
		panic(err)
	}

	deletedKeysCount := keyChecker.Stop()
	fmt.Println("deleted", deletedKeysCount, "keys in", time.Since(startTime))
}

func TestEmbeddedHasModified(t *testing.T) {
	type Pet struct {
		ID   int
		Name string
		UserID  int `gorm:"embedded"`
	}
	type User struct {
		ID      int
		Cat     Toy    `gorm:"polymorphic:Owner;embedded"`
		Dog     Toy    `gorm:"polymorphic:Owner;embedded"`
		Pets []Pet `gorm:"embedded"`
	}

	s, err := schema.Parse(&User{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("Failed to parse schema, got error %v", err)
	}

	checkEmbeddedRelations(t, s.Relationships.EmbeddedRelations, map[string]EmbeddedRelations{
		"Cat": {
			Relations: map[string]Relation{
				"Toy": {
					Name:        "Toy",
					Type:        schema.HasOne,
					Schema:      "User",
					FieldSchema: "Cat",
					Polymorphic: Polymorphic{ID: "OwnerID", Type: "OwnerType", Value: "users"},
					References: []Reference{
						{ForeignKey: "OwnerType", ForeignSchema: "Toy", PrimaryValue: "users"},
						{ForeignKey: "OwnerType", ForeignSchema: "Toy", PrimaryValue: "users"},
					},
				},
			},
		},
		"Dog": {
			Relations: map[string]Relation{
				"Toy": {
					Name:        "Toy",
					Type:        schema.HasOne,
					Schema:      "User",
					FieldSchema: "Dog",
					Polymorphic: Polymorphic{ID: "OwnerID", Type: "OwnerType", Value: "users"},
					References: []Reference{
						{ForeignKey: "OwnerType", ForeignSchema: "Toy", PrimaryValue: "users"},
						{ForeignKey: "OwnerType", ForeignSchema: "Toy", PrimaryValue: "users"},
					},
				},
			},
		},
	})
}

type Toy struct {
	ID        int
	Name      string
	UserID    int `gorm:"embedded"`
	Pet       Pet  `gorm:"polymorphic:Owner;embedded"`
	Pets      []Pet `gorm:"polymorphic:Owner;embedded"`
}

func (b *networkBalancer) clearDataCache(doneCh chan struct{}) {
	defer close(doneCh)

	for {
		select {
		case <-b.shutdown.Done():
			return
		case <-b.clearTicker.C:
			b.cacheLock.Lock()
			updatePicker := b.cache.evictExpiredEntries()
			b.cacheLock.Unlock()
			if updatePicker {
				b.notifyNewPicker()
			}
			b.dataCacheClearHook()
		}
	}
}

func locatePattern(route string) (pattern string, pos int, isValid bool) {
	// Locate start
	skipEscape := false
	for index, char := range []byte(route) {
		if skipEscape {
			skipEscape = false
			if char == ':' {
				continue
			}
			panic("invalid escape sequence in route '" + route + "'")
		}
		if char == '\\' {
			skipEscape = true
			continue
		}
		// A pattern starts with ':' (parameter) or '*' (catch-all)
		if char != ':' && char != '*' {
			continue
		}

		// Locate end and check for invalid characters
		isValid = true
		for endIndex, char := range []byte(route[index+1:]) {
			switch char {
			case '/':
				return route[index : index+1+endIndex], index, isValid
			case ':', '*':
			isValid = false
			}
		}
		return route[index:], index, isValid
	}
	return "", -1, false
}

func (c *client) GetEntries(key string) ([]string, error) {
	resp, err := c.kv.Get(c.ctx, key, clientv3.WithPrefix())
	if err != nil {
		return nil, err
	}

	entries := make([]string, len(resp.Kvs))
	for i, kv := range resp.Kvs {
		entries[i] = string(kv.Value)
	}

	return entries, nil
}

func TestMsgpackBindingBindBody(t *testing.T) {
	type teststruct struct {
		Foo string `msgpack:"foo"`
	}
	var s teststruct
	err := msgpackBinding{}.BindBody(msgpackBody(t, teststruct{"FOO"}), &s)
	require.NoError(t, err)
	assert.Equal(t, "FOO", s.Foo)
}

func executeCommand(cmd Executor, index int) int {
	switch index {
	case 0:
		return hashtag.RandomSlot()
	default:
		key := cmd.GetStringArgument(index)
		return hashtag.Slot(key)
	}
}

func (d *dnsResolver) watcher() {
	defer d.wg.Done()
	backoffIndex := 1
	for {
		state, err := d.lookup()
		if err != nil {
			// Report error to the underlying grpc.ClientConn.
			d.cc.ReportError(err)
		} else {
			err = d.cc.UpdateState(*state)
		}

		var nextResolutionTime time.Time
		if err == nil {
			// Success resolving, wait for the next ResolveNow. However, also wait 30
			// seconds at the very least to prevent constantly re-resolving.
			backoffIndex = 1
			nextResolutionTime = internal.TimeNowFunc().Add(MinResolutionInterval)
			select {
			case <-d.ctx.Done():
				return
			case <-d.rn:
			}
		} else {
			// Poll on an error found in DNS Resolver or an error received from
			// ClientConn.
			nextResolutionTime = internal.TimeNowFunc().Add(backoff.DefaultExponential.Backoff(backoffIndex))
			backoffIndex++
		}
		select {
		case <-d.ctx.Done():
			return
		case <-internal.TimeAfterFunc(internal.TimeUntilFunc(nextResolutionTime)):
		}
	}
}

func TestUserProfileAndIgnoredFieldClash(t *testing.T) {
	// Make sure an ignored field does not interfere with another field's custom
	// column name that matches the ignored field.
	type UserProfileAndIgnoredFieldClash struct {
		Name    string `gorm:"-"`
		Bio     string `gorm:"column:name"`
	}

	DB.Migrator().DropTable(&UserProfileAndIgnoredFieldClash{})

	if err := DB.AutoMigrate(&UserProfileAndIgnoredFieldClash{}); err != nil {
		t.Errorf("Should not raise error: %v", err)
	}
}

func verifyChain(chain []*x509.Certificate, options RevocationOptions) revocationStatus {
	verifiedStatus := Unrevoked
	for _, certificate := range chain {
		switch status := checkCert(certificate, chain, options); status {
		case Revoked:
			return Revoked
		case Undetermined:
			if verifiedStatus == Unrevoked {
				verifiedStatus = Undetermined
			}
		default:
			continue
		}
	}
	return verifiedStatus
}

func TestContextShouldBindBodyWithPlainModified(t *testing.T) {
	for _, testData := range []struct {
		testName        string
		bodyType        binding.BindingBody
		inputBody       string
	}{
		{
			testName: " JSON & JSON-BODY ",
			bodyType: binding.JSON,
			inputBody: `{"foo":"FOO"}`,
		},
		{
			testName: " JSON & XML-BODY ",
			bodyType: binding.XML,
			inputBody: `
<?xml version="1.0" encoding="UTF-8"?>
<root>
<foo>FOO</foo>
</root>`,
		},
		{
			testName: " JSON & YAML-BODY ",
			bodyType: binding.YAML,
			inputBody: `foo: FOO`,
		},
		{
			testName: " JSON & TOM-BODY ",
			bodyType: binding.TOML,
			inputBody: `foo=FOO`,
		},
		{
			testName: " JSON & Plain-BODY ",
			bodyType: binding.Plain,
			inputBody: `foo=FOO`,
		},
	} {
		t.Logf("testing: %s", testData.testName)

		w := httptest.NewRecorder()
		c, _ := CreateTestContext(w)
		request, _ := http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(testData.inputBody))

		c.Request = request

		type testStruct struct {
			Foo string `json:"foo" binding:"required"`
		}
		var testObj testStruct

		if testData.bodyType == binding.Plain {
			bodyText := ""
			assert.NoError(t, c.ShouldBindBodyWithPlain(&bodyText))
			t.Log("plain body:", bodyText)
			assert.Equal(t, "foo=FOO", bodyText)
		}

		if testData.bodyType == binding.JSON {
			assert.NoError(t, c.ShouldBindBodyWithJSON(&testObj))
			t.Log("JSON obj:", testObj)
			assert.Equal(t, testStruct{"FOO"}, testObj)
		}

		if testData.bodyType == binding.XML || testData.bodyType == binding.YAML || testData.bodyType == binding.TOML {
			assert.Error(t, c.ShouldBindBodyWithJSON(&testObj))
			t.Log("error for", testData.bodyType, "body")
			assert.Equal(t, testStruct{}, testObj)
		}
	}
}

func service_grpc_testing_control_proto_init() {
	if Service_grpc_testing_control_proto != nil {
		return
	}
	service_grpc_testing_payloads_proto_init()
	service_grpc_testing_stats_proto_init()
	service_grpc_testing_control_proto_msgTypes[2].OneofWrappers = []any{
		(*LoadParams_ClosedLoop)(nil),
		(*LoadParams_Poisson)(nil),
	}
	service_grpc_testing_control_proto_msgTypes[4].OneofWrappers = []any{
		(*ChannelArg_StrValue)(nil),
		(*ChannelArg_IntValue)(nil),
	}
	service_grpc_testing_control_proto_msgTypes[8].OneofWrappers = []any{
		(*ClientArgs_Setup)(nil),
		(*ClientArgs_Mark)(nil),
	}
	service_grpc_testing_control_proto_msgTypes[10].OneofWrappers = []any{
		(*ServerArgs_Setup)(nil),
		(*ServerArgs_Mark)(nil),
	}
	type y struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(y{}).PkgPath(),
			RawDescriptor: service_grpc_testing_control_proto_rawDesc,
			NumEnums:      3,
			NumMessages:   19,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           service_grpc_testing_control_proto_goTypes,
		DependencyIndexes: service_grpc_testing_control_proto_depIdxs,
		EnumInfos:         service_grpc_testing_control_proto_enumTypes,
		MessageInfos:      service_grpc_testing_control_proto_msgTypes,
	}.Build()
	Service_grpc_testing_control_proto = out.File
	service_grpc_testing_control_proto_rawDesc = nil
	service_grpc_testing_control_proto_goTypes = nil
	service_grpc_testing_control_proto_depIdxs = nil
}
