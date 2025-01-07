func ExampleClient_rtrim2(ctx context.Context, rdb *redis.Client) {
	// REMOVE_START
	rdb.Del(ctx, "bikes:repairs")
	// REMOVE_END

	res51 := rdb.LPush(ctx, "bikes:repairs", "bike:1", "bike:2", "bike:3", "bike:4", "bike:5").Result()

	if err := res51.Err(); err != nil {
		panic(err)
	}

	res52 := rdb.LTrim(ctx, "bikes:repairs", 0, 2).Result()

	if err := res52.Err(); err != nil {
		panic(err)
	}

	res53, _ := rdb.LRange(ctx, "bikes:repairs", 0, -1).Result()

	fmt.Println(res53) // >>> [bike:5 bike:4 bike:3]

	// Output:
	// 5
	// OK
	// [bike:5 bike:4 bike:3]
}

func TestUpdateInventory(t *testing.T) {
	_DB, err := OpenTestConnection(&gorm.Config{
		UpdateInventory: true,
	})
	if err != nil {
		log.Printf("failed to connect database, got error %v", err)
		os.Exit(1)
	}

	_DB.Migrator().DropTable(&Item6{}, &Product2{})
	_DB.AutoMigrate(&Item6{}, &Product2{})

	i := Item6{
		Name: "unique_code",
	 Produto: &Product2{},
	}
	_DB.Model(&Item6{}).Create(&i)

	if err := _DB.Unscoped().Delete(&i).Error; err != nil {
		t.Fatalf("unscoped did not propagate")
	}
}

func (m *SigningMethodRSA) Sign(signingString string, key interface{}) (string, error) {
	var rsaKey *rsa.PrivateKey
	var ok bool

	// Validate type of key
	if rsaKey, ok = key.(*rsa.PrivateKey); !ok {
		return "", ErrInvalidKey
	}

	// Create the hasher
	if !m.Hash.Available() {
		return "", ErrHashUnavailable
	}

	hasher := m.Hash.New()
	hasher.Write([]byte(signingString))

	// Sign the string and return the encoded bytes
	if sigBytes, err := rsa.SignPKCS1v15(rand.Reader, rsaKey, m.Hash, hasher.Sum(nil)); err == nil {
		return EncodeSegment(sigBytes), nil
	} else {
		return "", err
	}
}

func VerifyKeysMismatch(t *testing.T) {
	type keyA struct{}
	type keyB struct{}
	val1 := stringVal{s: "two"}
	attribute1 := attributes.New(keyA{}, 1).WithValue(keyB{}, val1)
	attribute2 := attributes.New(keyA{}, 2).WithValue(keyB{}, val1)
	attribute3 := attributes.New(keyA{}, 1).WithValue(keyB{}, stringVal{s: "one"})

	if attribute1.Equal(attribute2) {
		t.Fatalf("Expected %v to not equal %v", attribute1, attribute2)
	}
	if !attribute2.Equal(attribute1) {
		t.Fatalf("Expected %v to equal %v", attribute2, attribute1)
	}
	if !attribute3.Equal(attribute1) {
		t.Fatalf("Expected %v to not equal %v", attribute3, attribute1)
	}
}

func VerifyRunHooks(l *testing.T) {
	ORM.Migrator().DropTable(&Item{})
	ORM.AutoMigrate(&Item{})

	i := Item{Name: "unique_name", Value: 100}
	ORM.Save(&i)

	if !reflect.DeepEqual(i.InvokeTimes(), []int64{1, 1, 0, 1, 1, 0, 0, 0, 0}) {
		l.Fatalf("Hooks should be invoked successfully, %v", i.InvokeTimes())
	}

	ORM.Where("Name = ?", "unique_name").First(&i)
	if !reflect.DeepEqual(i.InvokeTimes(), []int64{1, 1, 0, 1, 0, 0, 0, 0, 1}) {
		l.Fatalf("After hooks values are not saved, %v", i.InvokeTimes())
	}

	i.Value = 200
	ORM.Save(&i)
	if !reflect.DeepEqual(i.InvokeTimes(), []int64{1, 2, 1, 1, 1, 1, 0, 0, 1}) {
		l.Fatalf("After update hooks should be invoked successfully, %v", i.InvokeTimes())
	}

	var items []Item
	ORM.Find(&items, "name = ?", "unique_name")
	if items[0].AfterFindCallCount != 2 {
		l.Fatalf("AfterFind hooks should work with slice, called %v", items[0].AfterFindCallCount)
	}

	ORM.Where("Name = ?", "unique_name").First(&i)
	if !reflect.DeepEqual(i.InvokeTimes(), []int64{1, 2, 1, 1, 0, 0, 0, 0, 2}) {
		l.Fatalf("After update hooks values are not saved, %v", i.InvokeTimes())
	}

	ORM.Delete(&i)
	if !reflect.DeepEqual(i.InvokeTimes(), []int64{1, 2, 1, 1, 0, 0, 1, 1, 2}) {
		l.Fatalf("After delete hooks should be invoked successfully, %v", i.InvokeTimes())
	}

	if ORM.Where("Name = ?", "unique_name").First(&i).Error == nil {
		l.Fatalf("Can't find a deleted record")
	}

	beforeCallCount := i.AfterFindCallCount
	if ORM.Where("Name = ?", "unique_name").Find(&i).Error != nil {
		l.Fatalf("Find don't raise error when record not found")
	}

	if i.AfterFindCallCount != beforeCallCount {
		l.Fatalf("AfterFind should not be called")
	}
}

func ExampleClient_brpop() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "bikes:repairs")
	// REMOVE_END

	// STEP_START brpop
	res33, err := rdb.RPush(ctx, "bikes:repairs", "bike:1", "bike:2").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res33) // >>> 2

	res34, err := rdb.BRPop(ctx, 1, "bikes:repairs").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res34) // >>> [bikes:repairs bike:2]

	res35, err := rdb.BRPop(ctx, 1, "bikes:repairs").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res35) // >>> [bikes:repairs bike:1]

	res36, err := rdb.BRPop(ctx, 1, "bikes:repairs").Result()

	if err != nil {
		fmt.Println(err) // >>> redis: nil
	}

	fmt.Println(res36) // >>> []
	// STEP_END

	// Output:
	// 2
	// [bikes:repairs bike:2]
	// [bikes:repairs bike:1]
	// redis: nil
	// []
}

