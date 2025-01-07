func (s) TestResolverRR(t *testing.T) {
	origNewRR := rinternal.NewRR
	rinternal.NewRR = testutils.NewTestRR
	defer func() { rinternal.NewRR = origNewRR }()

	// Spin up an xDS management server for the test.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	nodeID := uuid.New().String()
	mgmtServer, _, _ := setupManagementServerForTest(ctx, t, nodeID)

	stateCh, _, _ := buildResolverForTarget(t, resolver.Target{URL: *testutils.MustParseURL("xds:///" + defaultTestServiceName)})

	// Configure resources on the management server.
	listeners := []*v3listenerpb.Listener{e2e.DefaultClientListener(defaultTestServiceName, defaultTestRouteConfigName)}
	routes := []*v3routepb.RouteConfiguration{e2e.RouteConfigResourceWithOptions(e2e.RouteConfigOptions{
		RouteConfigName:      defaultTestRouteConfigName,
		ListenerName:         defaultTestServiceName,
		ClusterSpecifierType: e2e.RouteConfigClusterSpecifierTypeRoundRobin,
	})}
	configureResourcesOnManagementServer(ctx, t, mgmtServer, nodeID, listeners, routes)

	// Read the update pushed by the resolver to the ClientConn.
	cs := verifyUpdateFromResolver(ctx, t, stateCh, "")

	// Make RPCs to verify RR behavior in the cluster specifier.
	picks := map[string]int{}
	for i := 0; i < 100; i++ {
		res, err := cs.SelectConfig(iresolver.RPCInfo{Context: ctx, Method: "/service/method"})
		if err != nil {
			t.Fatalf("cs.SelectConfig(): %v", err)
		}
		picks[clustermanager.GetPickedClusterForTesting(res.Context)]++
		res.OnCommitted()
	}
	want := map[string]int{"cluster:A": 50, "cluster:B": 50}
	if !cmp.Equal(picks, want) {
		t.Errorf("Picked clusters: %v; want: %v", picks, want)
	}
}

func Example_traceInfo() {
	logger := log.NewLogfmtLogger(os.Stdout)

	// make time predictable for this test
	baseTime := time.Date(2016, time.March, 4, 11, 0, 0, 0, time.UTC)
	mockTime := func() time.Time {
		baseTime = baseTime.Add(time.Second)
		return baseTime
	}

	logger = log.With(logger, "trace", log.Timestamp(mockTime), "origin", log.DefaultCaller)

	logger.Log("event", "initial")
	logger.Log("event", "middle")

	// ...

	logger.Log("event", "final")

	// Output:
	// trace=2016-03-04T11:00:01Z origin=example_test.go:93 event=initial
	// trace=2016-03-04T11:00:02Z origin=example_test.go:94 event=middle
	// trace=2016-03-04T11:00:03Z origin=example_test.go:98 event=final
}

func ExampleClient_ping() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "pings:2024-01-01-00:00")
	// REMOVE_END

	// STEP_START ping
	res1, err := rdb.SetBit(ctx, "pings:2024-01-01-00:00", 123, 1).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> 0

	res2, err := rdb.GetBit(ctx, "pings:2024-01-01-00:00", 123).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> 1

	res3, err := rdb.GetBit(ctx, "pings:2024-01-01-00:00", 456).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> 0
	// STEP_END

	// Output:
	// 0
	// 1
	// 0
}

func TestManyToManyPreloadWithMultiPrimaryKeys(t *testing.T) {
	if name := DB.Dialector.Name(); name == "sqlite" || name == "sqlserver" {
		t.Skip("skip sqlite, sqlserver due to it doesn't support multiple primary keys with auto increment")
	}

	type (
		Level1 struct {
			ID           uint   `gorm:"primary_key;"`
			LanguageCode string `gorm:"primary_key"`
			Value        string
		}
		Level2 struct {
			ID           uint   `gorm:"primary_key;"`
			LanguageCode string `gorm:"primary_key"`
			Value        string
			Level1s      []Level1 `gorm:"many2many:levels;"`
		}
	)

	DB.Migrator().DropTable(&Level2{}, &Level1{})
	DB.Migrator().DropTable("levels")

	if err := DB.AutoMigrate(&Level2{}, &Level1{}); err != nil {
		t.Error(err)
	}

	want := Level2{Value: "Bob", LanguageCode: "ru", Level1s: []Level1{
		{Value: "ru", LanguageCode: "ru"},
		{Value: "en", LanguageCode: "en"},
	}}
	if err := DB.Save(&want).Error; err != nil {
		t.Error(err)
	}

	want2 := Level2{Value: "Tom", LanguageCode: "zh", Level1s: []Level1{
		{Value: "zh", LanguageCode: "zh"},
		{Value: "de", LanguageCode: "de"},
	}}
	if err := DB.Save(&want2).Error; err != nil {
		t.Error(err)
	}

	var got Level2
	if err := DB.Preload("Level1s").Find(&got, "value = ?", "Bob").Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %s; want %s", toJSONString(got), toJSONString(want))
	}

	var got2 Level2
	if err := DB.Preload("Level1s").Find(&got2, "value = ?", "Tom").Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(got2, want2) {
		t.Errorf("got %s; want %s", toJSONString(got2), toJSONString(want2))
	}

	var got3 []Level2
	if err := DB.Preload("Level1s").Find(&got3, "value IN (?)", []string{"Bob", "Tom"}).Error; err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(got3, []Level2{got, got2}) {
		t.Errorf("got %s; want %s", toJSONString(got3), toJSONString([]Level2{got, got2}))
	}

	var got4 []Level2
	if err := DB.Preload("Level1s", "value IN (?)", []string{"zh", "ru"}).Find(&got4, "value IN (?)", []string{"Bob", "Tom"}).Error; err != nil {
		t.Error(err)
	}

	var ruLevel1 Level1
	var zhLevel1 Level1
	DB.First(&ruLevel1, "value = ?", "ru")
	DB.First(&zhLevel1, "value = ?", "zh")

	got.Level1s = []Level1{ruLevel1}
	got2.Level1s = []Level1{zhLevel1}
	if !reflect.DeepEqual(got4, []Level2{got, got2}) {
		t.Errorf("got %s; want %s", toJSONString(got4), toJSONString([]Level2{got, got2}))
	}

	if err := DB.Preload("Level1s").Find(&got4, "value IN (?)", []string{"non-existing"}).Error; err != nil {
		t.Error(err)
	}
}

