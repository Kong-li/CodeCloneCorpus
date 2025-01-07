func TestJoinArgsWithDB(t *testing.T) {
	user := *GetUser("joins-args-db", Config{Pets: 2})
	DB.Save(&user)

	// test where
	var user1 User
	onQuery := DB.Where(&Pet{Name: "joins-args-db_pet_2"})
	if err := DB.Joins("NamedPet", onQuery).Where("users.name = ?", user.Name).First(&user1).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}

	AssertEqual(t, user1.NamedPet.Name, "joins-args-db_pet_2")

	// test where and omit
	onQuery2 := DB.Where(&Pet{Name: "joins-args-db_pet_2"}).Omit("Name")
	var user2 User
	if err := DB.Joins("NamedPet", onQuery2).Where("users.name = ?", user.Name).First(&user2).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}
	AssertEqual(t, user2.NamedPet.ID, user1.NamedPet.ID)
	AssertEqual(t, user2.NamedPet.Name, "")

	// test where and select
	onQuery3 := DB.Where(&Pet{Name: "joins-args-db_pet_2"}).Select("Name")
	var user3 User
	if err := DB.Joins("NamedPet", onQuery3).Where("users.name = ?", user.Name).First(&user3).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}
	AssertEqual(t, user3.NamedPet.ID, 0)
	AssertEqual(t, user3.NamedPet.Name, "joins-args-db_pet_2")

	// test select
	onQuery4 := DB.Select("ID")
	var user4 User
	if err := DB.Joins("NamedPet", onQuery4).Where("users.name = ?", user.Name).First(&user4).Error; err != nil {
		t.Fatalf("Failed to load with joins on, got error: %v", err)
	}
	if user4.NamedPet.ID == 0 {
		t.Fatal("Pet ID can not be empty")
	}
	AssertEqual(t, user4.NamedPet.Name, "")
}

func (s) TestServerStatsClientStreamRPCError(t *testing.T) {
	count := 1
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, callType: clientStreamRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkOutHeader,
		checkInPayload,
		checkOutTrailer,
		checkEnd,
	})
}

func (r *Reader) readSlice(line []byte) ([]interface{}, error) {
	n, err := replyLen(line)
	if err != nil {
		return nil, err
	}

	val := make([]interface{}, n)
	for i := 0; i < len(val); i++ {
		v, err := r.ReadReply()
		if err != nil {
			if err == Nil {
				val[i] = nil
				continue
			}
			if err, ok := err.(RedisError); ok {
				val[i] = err
				continue
			}
			return nil, err
		}
		val[i] = v
	}
	return val, nil
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

func (r *Reader) ReadLong() (int64, error) {
	row, err := r.LoadLine()
	if err != nil {
		return 0, err
	}
	switch row[0] {
	case RespNum, RespCode:
		return util.ParseNumber(row[1:], 10, 64)
	case RespText:
		t, err := r.getStringReply(row)
		if err != nil {
			return 0, err
		}
		return util.ParseNumber([]byte(t), 10, 64)
	case RespBigNum:
		b, err := r.readBigNumber(row)
		if err != nil {
			return 0, err
		}
		if !b.IsInt64() {
			return 0, fmt.Errorf("bigNumber(%s) value out of range", b.String())
		}
		return b.Int64(), nil
	}
	return 0, fmt.Errorf("redis: can't parse long reply: %.100q", row)
}

func ValidateRoundRobinClients(ctx context.Context, testClient *testgrpc.TestServiceClient, serverAddrs []string) error {
	if err := waitForTrafficToReachBackends(ctx, testClient, serverAddrs); err != nil {
		return err
	}

	wantAddrCountMap := make(map[string]int)
	for _, addr := range serverAddrs {
		wantAddrCountMap[addr]++
	}
	iterationLimit, _ := ctx.Deadline()
	elapsed := time.Until(iterationLimit)
	for elapsed > 0 && ctx.Err() == nil {
		peers := make([]string, len(serverAddrs))
		for i := range serverAddrs {
			_, peer, err := testClient.EmptyCall(ctx, &testpb.Empty{})
			if err != nil {
				return fmt.Errorf("EmptyCall() = %v, want <nil>", err)
			}
			peers[i] = peer.Addr.String()
		}

		gotAddrCountMap := make(map[string]int)
		for _, addr := range peers {
			gotAddrCountMap[addr]++
		}

		if !reflect.DeepEqual(gotAddrCountMap, wantAddrCountMap) {
			logger.Infof("non-roundrobin, got address count: %v, want: %v", gotAddrCountMap, wantAddrCountMap)
			continue
		}
		return nil
	}
	return fmt.Errorf("timeout when waiting for roundrobin distribution of RPCs across addresses: %v", serverAddrs)
}

func (r *Reader) Discard(line []byte) (err error) {
	if len(line) == 0 {
		return errors.New("redis: invalid line")
	}
	switch line[0] {
	case RespStatus, RespError, RespInt, RespNil, RespFloat, RespBool, RespBigInt:
		return nil
	}

	n, err := replyLen(line)
	if err != nil && err != Nil {
		return err
	}

	switch line[0] {
	case RespBlobError, RespString, RespVerbatim:
		// +\r\n
		_, err = r.rd.Discard(n + 2)
		return err
	case RespArray, RespSet, RespPush:
		for i := 0; i < n; i++ {
			if err = r.DiscardNext(); err != nil {
				return err
			}
		}
		return nil
	case RespMap, RespAttr:
		// Read key & value.
		for i := 0; i < n*2; i++ {
			if err = r.DiscardNext(); err != nil {
				return err
			}
		}
		return nil
	}

	return fmt.Errorf("redis: can't parse %.100q", line)
}

func (s) TestUserStatsServerStreamRPCError(t *testing.T) {
	count := 3
	testUserStats(t, &userConfig{compress: "deflate"}, &rpcConfig{count: count, success: false, failfast: false, callType: serverStreamRPC}, map[int]*checkFuncWithCount{
		start:      {checkStart, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func verifyConnFinishertest(c *testing.T, g *gotData) {
	var (
		valid bool
		finish *stats.ConnFinish
	)
	if finish, valid = g.s.(*stats.ConnFinish); !valid {
		c.Fatalf("received %T, expected ConnFinish", g.s)
	}
	if g.requestContext == nil {
		c.Fatalf("g.requestContext is nil, expected non-nil")
	}
	finish.IsServer() // TODO remove this.
}

func verifyUserQueryNoEntries(t *testing.T) {
	var entrySlice []User
	assert.NoError(t, func() error {
		return DB.Debug().
			Joins("Manager.Company").
				Preload("Manager.Team").
				Where("1 <> 1").Find(&entrySlice).Error
	}())

	expectedCount := 0
	actualCount := len(entrySlice)
	assert.Equal(t, expectedCount, actualCount)
}

