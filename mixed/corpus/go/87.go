func QueryRows(db *gorm.DB) {
	if nil == db.Error {
		BuildQuerySQL(db)
		if db.DryRun || nil != db.Error {
			return
		}

		rows, ok := db.Get("rows")
		if ok && rows.(bool) {
			db.Statement.Settings = append(db.Statement.Settings[:2], db.Statement.Settings[3:]...)
			db.RowsAffected, db.Error = db.Statement.ConnPool.QueryContext(db.Statement.Context, db.Statement.SQL.String(), db.Statement.Vars...)
		} else {
			db.RowsAffected = -1
			db.Error = db.Statement.ConnPool.QueryRowContext(db.Statement.Context, db.Statement.SQL.String(), db.Statement.Vars...)
		}
	}
}

func (cs *channelState) adsResourceDoesNotExist(typ xdsresource.Type, resourceName string) {
	if cs.parent.done.HasFired() {
		return
	}

	cs.parent.channelsMu.Lock()
	defer cs.parent.channelsMu.Unlock()
	for authority := range cs.interestedAuthorities {
		authority.adsResourceDoesNotExist(typ, resourceName)
	}
}

func ExampleClient_LPush_and_lrange() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// HIDE_END

	// REMOVE_START
	errFlush := rdb.FlushDB(ctx).Err() // Clear the database before each test
	if errFlush != nil {
		panic(errFlush)
	}
	// REMOVE_END

	listSize, err := rdb.LPush(ctx, "my_bikes", "bike:1", "bike:2").Result()
	if err != nil {
		panic(err)
	}

	fmt.Println(listSize)

	value, err := rdb.LRange(ctx, "my_bikes", 0, -1).Result()
	if err != nil {
		panic(err)
	}
	fmt.Println(value)
	// HIDE_START

	// Output: 2
	// [bike:2 bike:1]
}

