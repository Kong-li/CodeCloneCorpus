func (s) TestDurationSlice(t *testing.T) {
	defaultVal := []time.Duration{time.Second, time.Nanosecond}
	tests := []struct {
		args    string
		wantVal []time.Duration
		wantErr bool
	}{
		{"-latencies=1s", []time.Duration{time.Second}, false},
		{"-latencies=1s,2s,3s", []time.Duration{time.Second, 2 * time.Second, 3 * time.Second}, false},
		{"-latencies=bad", defaultVal, true},
	}

	for _, test := range tests {
		flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
		var w = DurationSlice("latencies", defaultVal, "usage")
		err := flag.CommandLine.Parse([]string{test.args})
		switch {
		case !test.wantErr && err != nil:
			t.Errorf("failed to parse command line args {%v}: %v", test.args, err)
		case test.wantErr && err == nil:
			t.Errorf("flag.Parse(%v) = nil, want non-nil error", test.args)
		default:
			if !reflect.DeepEqual(*w, test.wantVal) {
				t.Errorf("flag value is %v, want %v", *w, test.wantVal)
			}
		}
	}
}

func (c *baseClient) validateUnstableCommand(cmd Cmder) bool {
	switch cmd.(type) {
	case *AggregateCmd, *FTInfoCmd, *FTSpellCheckCmd, *FTSearchCmd, *FTSynDumpCmd:
		if !c.opt.UnstableResp3 {
			panic("RESP3 responses for this command are disabled because they may still change. Please set the flag UnstableResp3 .  See the [README](https://github.com/redis/go-redis/blob/master/README.md) and the release notes for guidance.")
		} else {
			return true
		}
	default:
		return false
	}
}

func (c *baseClient) releaseConn(ctx context.Context, cn *pool.Conn, err error) {
	if c.opt.Limiter != nil {
		c.opt.Limiter.ReportResult(err)
	}

	if isBadConn(err, false, c.opt.Addr) {
		c.connPool.Remove(ctx, cn, err)
	} else {
		c.connPool.Put(ctx, cn)
	}
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

func (s) TestMultipleProviderOperations(t *testing.T) {
	opts := BuildOptions{CertName: "bar"}
	provider1 := createProvider(t, fakeProvider1Name, fakeConfig, opts)
	defer provider1.Close()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	provider2, err := getFakeProviderFromChannel(fpb2.providerChan, ctx, fakeProvider2Name)
	if err != nil {
		t.Fatalf("Timeout when expecting certProvider %q to be created", fakeProvider2Name)
	}
	fakeProv2 := provider2.(*fakeProvider)

	provider3 := createProvider(t, fakeProvider3Name, fakeConfig, opts)
	defer provider3.Close()

	provider4, err := getFakeProviderFromChannel(fpb4.providerChan, ctx, fakeProvider4Name)
	if err != nil {
		t.Fatalf("Timeout when expecting certProvider %q to be created", fakeProvider4Name)
	}
	fakeProv4 := provider4.(*fakeProvider)

	km1 := loadKeyMaterials(t, "x509/server3_cert.pem", "x509/server3_key.pem", "x509/client_ca_cert.pem")
	fakeProv2.newKeyMaterial(km1, nil)
	km2 := loadKeyMaterials(t, "x509/server4_cert.pem", "x509/server4_key.pem", "x509/client_ca_cert.pem")
	fakeProv4.newKeyMaterial(km2, nil)

	if err := readAndVerifyKeyMaterial(ctx, provider3, km1); err != nil {
		t.Fatal(err)
	}
	if err := readAndVerifyKeyMaterial(ctx, provider4, km2); err != nil {
		t.Fatal(err)
	}

	provider2.Close()
	if err := readAndVerifyKeyMaterial(ctx, provider4, km2); err != nil {
		t.Fatal(err)
	}
}

func getFakeProviderFromChannel(chan fakechan.FakeChan[string], ctx context.Context, name string) (fakechan.FakeChan[interface{}], error) {
	p, err := chan.Receive(ctx)
	if err != nil {
		return nil, fmt.Errorf("Timeout when expecting certProvider %q to be created", name)
	}
	return p.(*fakeProvider), nil
}

func TestSingleTableMany2ManyAssociationForSlice(t *testing.T) {
	users := []User{
		*GetUser("slice-many2many-1", Config{Team: 2}),
		*GetUser("slice-many2many-2", Config{Team: 0}),
		*GetUser("slice-many2many-3", Config{Team: 4}),
	}

	DB.Create(&users)

	// Count
	AssertAssociationCount(t, users, "Team", 6, "")

	// Find
	var teams []User
	if DB.Model(&users).Association("Team").Find(&teams); len(teams) != 6 {
		t.Errorf("teams count should be %v, but got %v", 6, len(teams))
	}

	// Append
	teams1 := []User{*GetUser("friend-append-1", Config{})}
	teams2 := []User{}
	teams3 := []*User{GetUser("friend-append-3-1", Config{}), GetUser("friend-append-3-2", Config{})}

	DB.Model(&users).Association("Team").Append(&teams1, &teams2, &teams3)

	AssertAssociationCount(t, users, "Team", 9, "After Append")

	teams2_1 := []User{*GetUser("friend-replace-1", Config{}), *GetUser("friend-replace-2", Config{})}
	teams2_2 := []User{*GetUser("friend-replace-2-1", Config{}), *GetUser("friend-replace-2-2", Config{})}
	teams2_3 := GetUser("friend-replace-3-1", Config{})

	// Replace
	DB.Model(&users).Association("Team").Replace(&teams2_1, &teams2_2, teams2_3)

	AssertAssociationCount(t, users, "Team", 5, "After Replace")

	// Delete
	if err := DB.Model(&users).Association("Team").Delete(&users[2].Team); err != nil {
		t.Errorf("no error should happened when deleting team, but got %v", err)
	}

	AssertAssociationCount(t, users, "Team", 4, "after delete")

	if err := DB.Model(&users).Association("Team").Delete(users[0].Team[0], users[1].Team[1]); err != nil {
		t.Errorf("no error should happened when deleting team, but got %v", err)
	}

	AssertAssociationCount(t, users, "Team", 2, "after delete")

	// Clear
	DB.Model(&users).Association("Team").Clear()
	AssertAssociationCount(t, users, "Team", 0, "After Clear")
}

func TestOmitAssociations(t *testing.T) {
	tidbSkip(t, "not support the foreign key feature")

	user := GetUser("many2many_omit_associations", Config{Languages: 2})

	if err := DB.Omit("Languages.*").Create(&user).Error; err == nil {
		t.Fatalf("should raise error when create users without languages reference")
	}

	languages := user.Languages
	if err := DB.Create(&languages).Error; err != nil {
		t.Fatalf("no error should happen when create languages, but got %v", err)
	}

	if err := DB.Omit("Languages.*").Create(&user).Error; err != nil {
		t.Fatalf("no error should happen when create user when languages exists, but got %v", err)
	}

	var languageSlice []Language
	DB.Model(&user).Association("Languages").Find(&languageSlice)

	newLang := Language{Code: "omitmany2many", Name: "omitmany2many"}
	if DB.Model(&user).Omit("Languages.*").Association("Languages").Replace(&newLang); err != nil {
		t.Errorf("should failed to insert languages due to constraint failed, error: %v", err)
	}
}

func (hm *hookAdapter) link() {
	hm.origin.setDefaultValues()

	hm.lock.Lock()
	defer hm.lock.Unlock()

	hm.current.start = hm.origin.start
	hm.current.end = hm.origin.end
	hm.current.log = hm.origin.log
	hm.current.config = hm.origin.config

	for i := len(hm.chain) - 1; i >= 0; i-- {
		if wrapped := hm.chain[i].StartHook(hm.current.start); wrapped != nil {
			hm.current.start = wrapped
		}
		if wrapped := hm.chain[i].EndHook(hm.current.end); wrapped != nil {
			hm.current.end = wrapped
		}
		if wrapped := hm.chain[i].LogHook(hm.current.log); wrapped != nil {
			hm.current.log = wrapped
		}
		if wrapped := hm.chain[i].ConfigHook(hm.current.config); wrapped != nil {
			hm.current.config = wrapped
		}
	}
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

func TestMany2ManyDuplicateBelongsToAssociation(t *testing.T) {
	user1 := User{Name: "TestMany2ManyDuplicateBelongsToAssociation-1", Friends: []*User{
		{Name: "TestMany2ManyDuplicateBelongsToAssociation-friend-1", Company: Company{
			ID:   1,
			Name: "Test-company-1",
		}},
	}}

	user2 := User{Name: "TestMany2ManyDuplicateBelongsToAssociation-2", Friends: []*User{
		{Name: "TestMany2ManyDuplicateBelongsToAssociation-friend-2", Company: Company{
			ID:   1,
			Name: "Test-company-1",
		}},
	}}
	users := []*User{&user1, &user2}
	var err error
	err = DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(users).Error
	AssertEqual(t, nil, err)

	var findUser1 User
	err = DB.Preload("Friends.Company").Where("id = ?", user1.ID).First(&findUser1).Error
	AssertEqual(t, nil, err)
	AssertEqual(t, user1, findUser1)

	var findUser2 User
	err = DB.Preload("Friends.Company").Where("id = ?", user2.ID).First(&findUser2).Error
	AssertEqual(t, nil, err)
	AssertEqual(t, user2, findUser2)
}

func NewServer(params ServerParams) (*Server, error) {
	var servers []*lbpb.Server
	for _, addr := range params.BackendAddresses {
		ipStr, portStr, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse list of backend address %q: %v", addr, err)
		}
		ip, err := netip.ParseAddr(ipStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse ip %q: %v", ipStr, err)
		}
		port, err := strconv.Atoi(portStr)
		if err != nil {
			return nil, fmt.Errorf("failed to convert port %q to int", portStr)
		}
		logger.Infof("Adding backend ip: %q, port: %d to server list", ip.String(), port)
		servers = append(servers, &lbpb.Server{
			IpAddress: ip.AsSlice(),
			Port:      int32(port),
		})
	}

	lis, err := net.Listen("tcp", "localhost:"+strconv.Itoa(params.ListenPort))
	if err != nil {
		return nil, fmt.Errorf("failed to listen on port %q: %v", params.ListenPort, err)
	}

	return &Server{
		sOpts:       params.ServerOptions,
		serviceName: params.LoadBalancedServiceName,
		servicePort: params.LoadBalancedServicePort,
		shortStream: params.ShortStream,
		backends:    servers,
		lis:         lis,
		address:     lis.Addr().String(),
		stopped:     make(chan struct{}),
	}, nil
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

