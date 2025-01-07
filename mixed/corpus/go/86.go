func TestManyToManyWithMultiPrimaryKeysAdjusted(t *testing.T) {
	if dbName := DB.Dialector.Name(); dbName == "sqlite" || dbName == "sqlserver" {
		t.Skip("skip sqlite, sqlserver due to it doesn't support multiple primary keys with auto increment")
	}

	if dbName := DB.Dialector.Name(); dbName == "postgres" {
		stmt := gorm.Statement{DB: DB}
		stmt.Parse(&Blog{})
		stmt.Schema.LookUpField("ID").Unique = true
		stmt.Parse(&Tag{})
		stmt.Schema.LookUpField("ID").Unique = true
	}

	DB.Migrator().DropTable(&Blog{}, &Tag{}, "blog_tags", "locale_blog_tags", "shared_blog_tags")
	if err := DB.AutoMigrate(&Blog{}, &Tag{}); err != nil {
		t.Fatalf("Failed to auto migrate, got error: %v", err)
	}

	blog := Blog{
		Locale:  "ZH",
		Subject: "subject",
		Body:    "body",
		Tags: []Tag{
			{Locale: "ZH", Value: "tag1"},
			{Locale: "ZH", Value: "tag2"},
		},
	}

	if !DB.Save(&blog).Error.IsNewRecord() {
		t.Fatalf("Blog should be saved successfully")
	}

	// Append
	tag3 := &Tag{Locale: "ZH", Value: "tag3"}
	if DB.Model(&blog).Association("Tags").Append(tag3); DB.Find(&blog, &blog.Tags).Error != nil {
		t.Fatalf("Failed to append tag after save")
	}

	if !compareTags(blog.Tags, []string{"tag1", "tag2", "tag3"}) {
		t.Fatalf("Blog should has three tags after Append")
	}

	if count := DB.Model(&blog).Association("Tags").Count(); count != 3 {
		t.Fatalf("Blog should has 3 tags after Append, got %v", count)
	}

	var tags []Tag
	if err := DB.Model(&blog).Association("Tags").Find(&tags); err != nil {
		t.Fatalf("Failed to find tags: %v", err)
	}
	if !compareTags(tags, []string{"tag1", "tag2", "tag3"}) {
		t.Fatalf("Should find 3 tags")
	}

	var blog1 Blog
	DB.Preload("Tags").Find(&blog1)
	if !compareTags(blog1.Tags, []string{"tag1", "tag2", "tag3"}) {
		t.Fatalf("Preload many2many relations failed")
	}

	// Replace
	tag5 := &Tag{Locale: "ZH", Value: "tag5"}
	tag6 := &Tag{Locale: "ZH", Value: "tag6"}
	if DB.Model(&blog).Association("Tags").Replace(tag5, tag6); count != 2 {
		t.Fatalf("Blog should have two tags after Replace")
	}

	var tags2 []Tag
	if err := DB.Model(&blog).Association("Tags").Find(&tags2); err != nil || !compareTags(tags2, []string{"tag5", "tag6"}) {
		t.Fatalf("Should find 2 tags after Replace: %v", err)
	}

	// Delete
	if DB.Model(&blog).Association("Tags").Delete(tag5); count != 1 {
		t.Fatalf("Blog should have one tag left after Delete")
	}

	var tags3 []Tag
	if err := DB.Model(&blog).Association("Tags").Find(&tags3); err != nil || !compareTags(tags3, []string{"tag6"}) {
		t.Fatalf("Should find 1 tags after Delete: %v", err)
	}

	if count = DB.Model(&blog).Association("Tags").Count(); count != 1 {
		t.Fatalf("Blog should have one tag left after Delete, got %v", count)
	}

	// Clear
	DB.Model(&blog).Association("Tags").Clear()
	if count = DB.Model(&blog).Association("Tags").Count(); count != 0 {
		t.Fatalf("All tags should be cleared")
	}
}

func addMetricsHook(rdb *redis.Client, conf *config) error {
	createTime, err := conf.meter.Float64Histogram(
		"db.client.connections.create_time",
		metric.WithDescription("The time it took to create a new connection."),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	useTime, err := conf.meter.Float64Histogram(
		"db.client.connections.use_time",
		metric.WithDescription("The time between borrowing a connection and returning it to the pool."),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	rdb.AddHook(&metricsHook{
		createTime: createTime,
		useTime:    useTime,
		attrs:      conf.attrs,
	})
	return nil
}

func (s) TestADS_WatchState_TimerFires(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Start an xDS management server.
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	// Create an xDS client with bootstrap pointing to the above server, and a
	// short resource expiry timeout.
	nodeID := uuid.New().String()
	bc := e2e.DefaultBootstrapContents(t, nodeID, mgmtServer.Address)
	testutils.CreateBootstrapFileForTesting(t, bc)
	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:               t.Name(),
		Contents:           bc,
		WatchExpiryTimeout: defaultTestWatchExpiryTimeout,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	defer close()

	// Create a watch for the first listener resource and verify that the timer
	// is running and the watch state is `requested`.
	const listenerName = "listener"
	ldsCancel1 := xdsresource.WatchListener(client, listenerName, noopListenerWatcher{})
	defer ldsCancel1()
	if err := waitForResourceWatchState(ctx, client, listenerName, ads.ResourceWatchStateRequested, true); err != nil {
		t.Fatal(err)
	}

	// Since the resource is not configured on the management server, the watch
	// expiry timer is expected to fire, and the watch state should move to
	// `timeout`.
	if err := waitForResourceWatchState(ctx, client, listenerName, ads.ResourceWatchStateTimeout, false); err != nil {
		t.Fatal(err)
	}
}

