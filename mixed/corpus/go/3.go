func ExampleClient_createRaceStream() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "race:italy")
	// REMOVE_END

	// STEP_START createRaceStream
	res20, err := rdb.XGroupCreateMkStream(ctx,
		"race:italy", "italianRacers", "$",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res20) // >>> OK
	// STEP_END

	// Output:
	// OK
}

func (s) TestRouteMap_Size(t *testing.T) {
	rm := NewRouteMap()
	// Should be empty at creation time.
	if got := rm.Len(); got != 0 {
		t.Fatalf("rm.Len() = %v; want 0", got)
	}
	// Add two routes with the same unordered set of addresses. This should
	// amount to one route. It should also not take into account attributes.
	rm.Set(route12, struct{}{})
	rm.Set(route21, struct{}{})

	if got := rm.Len(); got != 1 {
		t.Fatalf("rm.Len() = %v; want 1", got)
	}

	// Add another unique route. This should cause the length to be 2.
	rm.Set(route123, struct{}{})
	if got := rm.Len(); got != 2 {
		t.Fatalf("rm.Len() = %v; want 2", got)
	}
}

func TestNestedModel(t *testing.T) {
	versionUser, err := schema.Parse(&VersionUser{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("failed to parse nested user, got error %v", err)
	}

	fields := []schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"VersionModel", "BaseModel", "ID"}, DataType: schema.Uint, PrimaryKey: true, Size: 64, HasDefaultValue: true, AutoIncrement: true},
		{Name: "CreatedBy", DBName: "created_by", BindNames: []string{"VersionModel", "BaseModel", "CreatedBy"}, DataType: schema.Uint, Size: 64},
		{Name: "Version", DBName: "version", BindNames: []string{"VersionModel", "Version"}, DataType: schema.Int, Size: 64},
	}

	for _, f := range fields {
		checkSchemaField(t, versionUser, &f, func(f *schema.Field) {
			f.Creatable = true
			f.Updatable = true
			f.Readable = true
		})
	}
}

func verifyUserProfileTest(r *testing.Test, userProfile *schema.Schema) {
	// verify schema
	verifySchema(r, userProfile, schema.Schema{Name: "UserProfile", Table: "user_profiles"}, []string{"ID"})

	// verify fields
	fields := []schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"Model", "ID"}, DataType: schema.Uint, PrimaryKey: true, Tag: `gorm:"primarykey"`, TagSettings: map[string]string{"PRIMARYKEY": "PRIMARYKEY"}, Size: 64, HasDefaultValue: true, AutoIncrement: true},
		{Name: "CreatedAt", DBName: "created_at", BindNames: []string{"Model", "CreatedAt"}, DataType: schema.Time},
		{Name: "UpdatedAt", DBName: "updated_at", BindNames: []string{"Model", "UpdatedAt"}, DataType: schema.Time},
		{Name: "DeletedAt", DBName: "deleted_at", BindNames: []string{"Model", "DeletedAt"}, Tag: `gorm:"index"`, DataType: schema.Time},
		{Name: "Username", DBName: "username", BindNames: []string{"Username"}, DataType: schema.String},
		{Name: "Age", DBName: "age", BindNames: []string{"Age"}, DataType: schema.Uint, Size: 64},
		{Name: "BirthDate", DBName: "birth_date", BindNames: []string{"BirthDate"}, DataType: schema.Time},
		{Name: "CompanyId", DBName: "company_id", BindNames: []string{"CompanyId"}, DataType: schema.Int, Size: 64},
		{Name: "ManagerId", DBName: "manager_id", BindNames: []string{"ManagerId"}, DataType: schema.Uint, Size: 64},
		{Name: "IsActive", DBName: "is_active", BindNames: []string{"IsActive"}, DataType: schema.Bool},
	}

	for i := range fields {
		verifySchemaField(r, userProfile, &fields[i], func(f *schema.Field) {
			f.Creatable = true
			f.Updatable = true
			f.Readable = true
		})
	}

	// verify relations
	relations := []Relation{
		{
			Name: "Account", Type: schema.HasOne, Schema: "UserProfile", FieldSchema: "Account",
			References: []Reference{{"ID", "UserProfile", "UserID", "Account", "", true}},
		},
		{
			Name: "Pets", Type: schema.HasMany, Schema: "UserProfile", FieldSchema: "Pet",
			References: []Reference{{"ID", "UserProfile", "UserID", "Pet", "", true}},
		},
		{
			Name: "Toys", Type: schema.HasMany, Schema: "UserProfile", FieldSchema: "Toy",
			JoinTable: JoinTable{Name: "user_toys", Table: "user_toys", Fields: []schema.Field{
				{
					Name: "UserID", DBName: "user_id", BindNames: []string{"UserID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
				{
					Name: "ToyID", DBName: "toy_id", BindNames: []string{"ToyID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
			}},
			References: []Reference{{"ID", "UserProfile", "UserID", "user_toys", "", true}, {"ID", "Toy", "ToyID", "user_toys", "", false}},
		},
		{
			Name: "Friends", Type: schema.HasMany, Schema: "UserProfile", FieldSchema: "Friend",
			JoinTable: JoinTable{Name: "user_friends", Table: "user_friends", Fields: []schema.Field{
				{
					Name: "UserID", DBName: "user_id", BindNames: []string{"UserID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
				{
					Name: "FriendID", DBName: "friend_id", BindNames: []string{"FriendID"}, DataType: schema.Uint,
					Tag: `gorm:"primarykey"`, Creatable: true, Updatable: true, Readable: true, PrimaryKey: true, Size: 64,
				},
			}},
			References: []Reference{{"ID", "UserProfile", "UserID", "user_friends", "", true}, {"ID", "User", "FriendID", "user_friends", "", false}},
		},
	}

	for _, relation := range relations {
		verifySchemaRelation(r, userProfile, relation)
	}
}

func (p *CustomPool) Fetch(ctx context.Context) (*Item, error) {
	// In worst case this races with Clean which is not a very common operation.
	for i := 0; i < 1000; i++ {
		switch atomic.LoadUint32(&p.status) {
		case statusInitial:
			itm, err := p.queue.Get(ctx)
			if err != nil {
				return nil, err
			}
			if atomic.CompareAndSwapUint32(&p.status, statusInitial, statusRunning) {
				return itm, nil
			}
			p.queue.Remove(ctx, itm, ErrExpired)
		case statusRunning:
			if err := p.checkError(); err != nil {
				return nil, err
			}
			itm, ok := <-p.stream
			if !ok {
				return nil, ErrExpired
			}
			return itm, nil
		case statusExpired:
			return nil, ErrExpired
		default:
			panic("not reached")
		}
	}
	return nil, fmt.Errorf("custom: CustomPool.Fetch: infinite loop")
}

func (p *ConnectionPool) Renew(ctx context.Context) error {
	if p.checkHealth() == nil {
		return nil
	}

	select {
	case cn, ok := <-p.channel:
		if !ok {
			return ErrPoolClosed
		}
		p.manager.Remove(ctx, cn, ErrPoolClosed)
		p.healthError.Store(HealthError{wrapped: nil})
	default:
		return errors.New("redis: ConnectionPool does not have a Conn")
	}

	if !atomic.CompareAndSwapUint32(&p.status, statusInit, statusDefault) {
		status := atomic.LoadUint32(&p.status)
		return fmt.Errorf("redis: invalid ConnectionPool state: %d", status)
	}

	return nil
}

func ExampleClient_xgroupcreatemkstream() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "race:frenchy")
	// REMOVE_END

	// STEP_START xgroup_create_mkstream
	res21, err := rdb.XGroupCreateMkStream(ctx,
		"race:frenchy", "french_racers", "*",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res21) // >>> OK
	// STEP_END

	// Output:
	// OK
}

func (p *StickyConnPool) Get(ctx context.Context) (*Conn, error) {
	// In worst case this races with Close which is not a very common operation.
	for i := 0; i < 1000; i++ {
		switch atomic.LoadUint32(&p.state) {
		case stateDefault:
			cn, err := p.pool.Get(ctx)
			if err != nil {
				return nil, err
			}
			if atomic.CompareAndSwapUint32(&p.state, stateDefault, stateInited) {
				return cn, nil
			}
			p.pool.Remove(ctx, cn, ErrClosed)
		case stateInited:
			if err := p.badConnError(); err != nil {
				return nil, err
			}
			cn, ok := <-p.ch
			if !ok {
				return nil, ErrClosed
			}
			return cn, nil
		case stateClosed:
			return nil, ErrClosed
		default:
			panic("not reached")
		}
	}
	return nil, fmt.Errorf("redis: StickyConnPool.Get: infinite loop")
}

func (s) TestAddressMap_Keys(t *testing.T) {
	addrMap := NewAddressMap()
	addrMap.Set(addr1, 1)
	addrMap.Set(addr2, 2)
	addrMap.Set(addr3, 3)
	addrMap.Set(addr4, 4)
	addrMap.Set(addr5, 5)
	addrMap.Set(addr6, 6)
	addrMap.Set(addr7, 7) // aliases addr1

	want := []Address{addr1, addr2, addr3, addr4, addr5, addr6}
	got := addrMap.Keys()
	if d := cmp.Diff(want, got, cmp.Transformer("sort", func(in []Address) []Address {
		out := append([]Address(nil), in...)
		sort.Slice(out, func(i, j int) bool { return fmt.Sprint(out[i]) < fmt.Sprint(out[j]) })
		return out
	})); d != "" {
		t.Fatalf("addrMap.Keys returned unexpected elements (-want, +got):\n%v", d)
	}
}

func TestNestedStruct(t *testing*T) {
	type CorpBase struct {
		gorm.Model
		OwnerID string
	}

	type Company struct {
		ID      int
		Name    string
		Ignored string `gorm:"-"`
	}

	type Corp struct {
		CorpBase
		Base Company
	}

	cropSchema, err := schema.Parse(&Corp{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("failed to parse nested struct with primary key, got error %v", err)
	}

	fields := []schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"CorpBase", "Model", "ID"}, DataType: schema.Uint, PrimaryKey: true, Size: 64, HasDefaultValue: true, AutoIncrement: true, TagSettings: map[string]string{"PRIMARYKEY": "PRIMARYKEY"}},
		{Name: "Name", DBName: "name", BindNames: []string{"Base", "Name"}, DataType: schema.String, TagSettings: map[string]string{"EMBEDDED": "EMBEDDED"}},
		{Name: "Ignored", BindNames: []string{"Base", "Ignored"}, TagSettings: map[string]string{"-": "-", "EMBEDDED": "EMBEDDED"}},
		{Name: "OwnerID", DBName: "owner_id", BindNames: []string{"CorpBase", "OwnerID"}, DataType: schema.String},
	}

	for _, f := range fields {
		checkSchemaField(t, cropSchema, &f, func(f *schema.Field) {
			if f.Name != "Ignored" {
				f.Creatable = true
				f.Updatable = true
				f.Readable = true
			}
		})
	}
}

func TestSubscriberTimeout(t *testing.T) {
	var (
		encode = func(context.Context, *nats.Msg, interface{}) error { return nil }
		decode = func(_ context.Context, msg *nats.Msg) (interface{}, error) {
			return TestResponse{string(msg.Data), ""}, nil
		}
	)

	s, c := newNATSConn(t)
	defer func() { s.Shutdown(); s.WaitForShutdown() }()
	defer c.Close()

	ch := make(chan struct{})
	defer close(ch)

	sub, err := c.QueueSubscribe("natstransport.test", "natstransport", func(msg *nats.Msg) {
		<-ch
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Unsubscribe()

	publisher := natstransport.NewPublisher(
		c,
		"natstransport.test",
		encode,
		decode,
		natstransport.PublisherTimeout(time.Second),
	)

	_, err = publisher.Endpoint()(context.Background(), struct{}{})
	if err != context.DeadlineExceeded {
		t.Errorf("want %s, have %s", context.DeadlineExceeded, err)
	}
}

