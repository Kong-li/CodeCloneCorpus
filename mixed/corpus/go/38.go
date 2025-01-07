func (s) TestCacheClearWithCallback(t *testing.T) {
	itemCount := 3
	values := make([]string, itemCount)
	for i := 0; i < itemCount; i++ {
		values[i] = strconv.Itoa(i)
	}
	c := NewTimeoutCache(time.Hour)

	testDone := make(chan struct{})
	defer close(testDone)

	wg := sync.WaitGroup{}
	wg.Add(itemCount)
	for _, v := range values {
		i := len(values) - 1
		callbackChanTemp := make(chan struct{})
		c.Add(i, v, func() { close(callbackChanTemp) })
		go func(v string) {
			defer wg.Done()
			select {
			case <-callbackChanTemp:
			case <-testDone:
			}
		}(v)
	}

	allGoroutineDone := make(chan struct{}, itemCount)
	go func() {
		wg.Wait()
		close(allGoroutineDone)
	}()

	for i, v := range values {
		if got, ok := c.getForTesting(i); !ok || got.item != v {
			t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.item, ok, v, true)
		}
	}
	if l := c.Len(); l != itemCount {
		t.Fatalf("%d number of items in the cache, want %d", l, itemCount)
	}

	time.Sleep(testCacheTimeout / 2)
	c.Clear(true)

	for i := range values {
		if _, ok := c.getForTesting(i); ok {
			t.Fatalf("After Add(), before timeout, after Remove(), from cache got: _, %v, want _, %v", ok, false)
		}
	}
	if l := c.Len(); l != 0 {
		t.Fatalf("%d number of items in the cache, want 0", l)
	}

	select {
	case <-allGoroutineDone:
	case <-time.After(testCacheTimeout * 2):
		t.Fatalf("timeout waiting for all callbacks")
	}
}

func (sre *SuccessRateEjection) Equal(sre2 *SuccessRateEjection) bool {
	if sre == nil && sre2 == nil {
		return true
	}
	if (sre != nil) != (sre2 != nil) {
		return false
	}
	if sre.StdevFactor != sre2.StdevFactor {
		return false
	}
	if sre.EnforcementPercentage != sre2.EnforcementPercentage {
		return false
	}
	if sre.MinimumHosts != sre2.MinimumHosts {
		return false
	}
	return sre.RequestVolume == sre2.RequestVolume
}

func TestCreateFromMapWithoutPK(t *testing.T) {
	if !isMysql() {
		t.Skipf("This test case skipped, because of only supporting for mysql")
	}

	// case 1: one record, create from map[string]interface{}
	mapValue1 := map[string]interface{}{"name": "create_from_map_with_schema1", "age": 1}
	if err := DB.Model(&User{}).Create(mapValue1).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}

	if _, ok := mapValue1["id"]; !ok {
		t.Fatal("failed to create data from map with table, returning map has no primary key")
	}

	var result1 User
	if err := DB.Where("name = ?", "create_from_map_with_schema1").First(&result1).Error; err != nil || result1.Age != 1 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	var idVal int64
	_, ok := mapValue1["id"].(uint)
	if ok {
		t.Skipf("This test case skipped, because the db supports returning")
	}

	idVal, ok = mapValue1["id"].(int64)
	if !ok {
		t.Fatal("ret result missing id")
	}

	if int64(result1.ID) != idVal {
		t.Fatal("failed to create data from map with table, @id != id")
	}

	// case2: one record, create from *map[string]interface{}
	mapValue2 := map[string]interface{}{"name": "create_from_map_with_schema2", "age": 1}
	if err := DB.Model(&User{}).Create(&mapValue2).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}

	if _, ok := mapValue2["id"]; !ok {
		t.Fatal("failed to create data from map with table, returning map has no primary key")
	}

	var result2 User
	if err := DB.Where("name = ?", "create_from_map_with_schema2").First(&result2).Error; err != nil || result2.Age != 1 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	_, ok = mapValue2["id"].(uint)
	if ok {
		t.Skipf("This test case skipped, because the db supports returning")
	}

	idVal, ok = mapValue2["id"].(int64)
	if !ok {
		t.Fatal("ret result missing id")
	}

	if int64(result2.ID) != idVal {
		t.Fatal("failed to create data from map with table, @id != id")
	}

	// case 3: records
	values := []map[string]interface{}{
		{"name": "create_from_map_with_schema11", "age": 1}, {"name": "create_from_map_with_schema12", "age": 1},
	}

	beforeLen := len(values)
	if err := DB.Model(&User{}).Create(&values).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}

	// mariadb with returning, values will be appended with id map
	if len(values) == beforeLen*2 {
		t.Skipf("This test case skipped, because the db supports returning")
	}

	for i := range values {
		v, ok := values[i]["id"]
		if !ok {
			t.Fatal("failed to create data from map with table, returning map has no primary key")
		}

		var result User
		if err := DB.Where("name = ?", fmt.Sprintf("create_from_map_with_schema1%d", i+1)).First(&result).Error; err != nil || result.Age != 1 {
			t.Fatalf("failed to create from map, got error %v", err)
		}
		if int64(result.ID) != v.(int64) {
			t.Fatal("failed to create data from map with table, @id != id")
		}
	}
}

func CheckCreateWithCustomBatchSize(t *testing.T) {
	employees := []Employee{
		*GetEmployee("test_custom_batch_size_1", Config{Department: true, Projects: 2, Certificates: 3, Office: true, Supervisor: true, Team: 0, Skills: 1, Colleagues: 1}),
		*GetEmployee("test_custom_batch_sizs_2", Config{Department: false, Projects: 2, Certificates: 4, Office: false, Supervisor: false, Team: 1, Skills: 3, Colleagues: 5}),
		*GetEmployee("test_custom_batch_sizs_3", Config{Department: true, Projects: 0, Certificates: 3, Office: true, Supervisor: false, Team: 4, Skills: 0, Colleagues: 1}),
		*GetEmployee("test_custom_batch_sizs_4", Config{Department: true, Projects: 3, Certificates: 0, Office: false, Supervisor: true, Team: 0, Skills: 3, Colleagues: 0}),
		*GetEmployee("test_custom_batch_sizs_5", Config{Department: false, Projects: 0, Certificates: 3, Office: true, Supervisor: false, Team: 1, Skills: 3, Colleagues: 1}),
		*GetEmployee("test_custom_batch_sizs_6", Config{Department: true, Projects: 4, Certificates: 3, Office: false, Supervisor: true, Team: 1, Skills: 3, Colleagues: 0}),
	}

	result := DB.Session(&gorm.Session{CreateBatchSize: 2}).Create(&employees)
	if result.RowsAffected != int64(len(employees)) {
		t.Errorf("affected rows should be %v, but got %v", len(employees), result.RowsAffected)
	}

	for _, employee := range employees {
		if employee.ID == 0 {
			t.Fatalf("failed to fill user's ID, got %v", employee.ID)
		} else {
			var newEmployee Employee
			if err := DB.Where("id = ?", employee.ID).Preload(clause.Associations).First(&newEmployee).Error; err != nil {
				t.Fatalf("errors happened when query: %v", err)
			} else {
				CheckEmployee(t, newEmployee, employee)
			}
		}
	}
}

func TestCreateInBatchesWithDefaultSize(t *testing.T) {
	users := []User{
		*GetUser("create_with_default_batch_size_1", Config{Account: true, Pets: 2, Toys: 3, Company: true, Manager: true, Team: 0, Languages: 1, Friends: 1}),
		*GetUser("create_with_default_batch_sizs_2", Config{Account: false, Pets: 2, Toys: 4, Company: false, Manager: false, Team: 1, Languages: 3, Friends: 5}),
		*GetUser("create_with_default_batch_sizs_3", Config{Account: true, Pets: 0, Toys: 3, Company: true, Manager: false, Team: 4, Languages: 0, Friends: 1}),
		*GetUser("create_with_default_batch_sizs_4", Config{Account: true, Pets: 3, Toys: 0, Company: false, Manager: true, Team: 0, Languages: 3, Friends: 0}),
		*GetUser("create_with_default_batch_sizs_5", Config{Account: false, Pets: 0, Toys: 3, Company: true, Manager: false, Team: 1, Languages: 3, Friends: 1}),
		*GetUser("create_with_default_batch_sizs_6", Config{Account: true, Pets: 4, Toys: 3, Company: false, Manager: true, Team: 1, Languages: 3, Friends: 0}),
	}

	result := DB.Session(&gorm.Session{CreateBatchSize: 2}).Create(&users)
	if result.RowsAffected != int64(len(users)) {
		t.Errorf("affected rows should be %v, but got %v", len(users), result.RowsAffected)
	}

	for _, user := range users {
		if user.ID == 0 {
			t.Fatalf("failed to fill user's ID, got %v", user.ID)
		} else {
			var newUser User
			if err := DB.Where("id = ?", user.ID).Preload(clause.Associations).First(&newUser).Error; err != nil {
				t.Fatalf("errors happened when query: %v", err)
			} else {
				CheckUser(t, newUser, user)
			}
		}
	}
}

func TestCreateFromMapWithTableModified(t *testing.T) {
	supportLastInsertID := isMysql() || isSqlite()
	tableDB := DB.Table("users")

	// case 1: create from map[string]interface{}
	record1 := map[string]interface{}{"name": "create_from_map_with_table", "age": 18}
	if err := tableDB.Create(record1).Error; err != nil {
		t.Fatalf("failed to create data from map with table, got error: %v", err)
	}

	var res map[string]interface{}
	if _, ok := record1["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}
	if err := tableDB.Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table").Find(&res).Error; err != nil || res["age"] != 18 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	if _, ok := record1["@id"]; ok && fmt.Sprint(res["id"]) != fmt.Sprintf("%d", record1["@id"]) {
		t.Fatalf("failed to create data from map with table, @id != id, got %v, expect %v", res["id"], record1["@id"])
	}

	// case 2: create from *map[string]interface{}
	record2 := map[string]interface{}{"name": "create_from_map_with_table_1", "age": 18}
	tableDB2 := DB.Table("users")
	if err := tableDB2.Create(&record2).Error; err != nil {
		t.Fatalf("failed to create data from map, got error: %v", err)
	}
	if _, ok := record2["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}

	var res1 map[string]interface{}
	if err := tableDB2.Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table_1").Find(&res1).Error; err != nil || res1["age"] != 18 {
		t.Fatalf("failed to create from map, got error %v", err)
	}

	if _, ok := record2["@id"]; ok && fmt.Sprint(res1["id"]) != fmt.Sprintf("%d", record2["@id"]) {
		t.Fatal("failed to create data from map with table, @id != id")
	}

	// case 3: create from []map[string]interface{}
	records := []map[string]interface{}{
		{"name": "create_from_map_with_table_2", "age": 19},
		{"name": "create_from_map_with_table_3", "age": 20},
	}

	if err := tableDB.Create(&records).Error; err != nil {
		t.Fatalf("failed to create data from slice of map, got error: %v", err)
	}

	if _, ok := records[0]["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}
	if _, ok := records[1]["@id"]; !ok && supportLastInsertID {
		t.Fatal("failed to create data from map with table, returning map has no key '@id'")
	}

	var res2 map[string]interface{}
	if err := tableDB.Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table_2").Find(&res2).Error; err != nil || res2["age"] != 19 {
		t.Fatalf("failed to query data after create from slice of map, got error %v", err)
	}

	var res3 map[string]interface{}
	if err := DB.Table("users").Select([]string{"id", "name", "age"}).Where("name = ?", "create_from_map_with_table_3").Find(&res3).Error; err != nil || res3["age"] != 20 {
		t.Fatalf("failed to query data after create from slice of map, got error %v", err)
	}

	if _, ok := records[0]["@id"]; ok && fmt.Sprint(res2["id"]) != fmt.Sprintf("%d", records[0["@id"]]) {
		t.Errorf("failed to create data from map with table, @id != id")
	}

	if _, ok := records[1]["id"]; ok && fmt.Sprint(res3["id"]) != fmt.Sprintf("%d", records[1["@id"]]) {
		t.Errorf("failed to create data from map with table, @id != id")
	}
}

func generateWithoutValidationData() structWithoutValidationData {
	float := 1.5
	t := structWithoutValidationData{
		Logical:             false,
		Id:                  1 << 30,
		Integer:             -20000,
		Integer8:            130,
		Integer16:           -30000,
		Integer32:           1 << 30,
		Integer64:           1 << 60,
		Id8:                 255,
		Id16:                50000,
		Id32:                1 << 32,
		Id64:                1 << 63,
		FloatingPoint:       123.457,
		Datetime:            time.Time{},
		CustomInterface:     &bytes.Buffer{},
		Struct:              substructWithoutValidation{},
		IntSlice:            []int{-4, -3, 2, 1, 2, 3, 4},
		IntPointerSlice:     []*int{&float},
		StructSlice:         []substructWithoutValidation{},
		UniversalInterface:  2.3,
		FloatingPointMap: map[string]float32{
			"baz": 1.24,
			"qux": 233.324,
		},
		StructMap: mapWithoutValidationSub{
			"baz": substructWithoutValidation{},
			"qux": substructWithoutValidation{},
		},
		// StructPointerSlice []withoutValidationSub
		// InterfaceSlice     []testInterface
	}
	t.InlinedStruct.Integer = 2000
	t.InlinedStruct.String = []string{"third", "fourth"}
	t.IString = "subsequence"
	t.IInt = 654321
	return t
}

func (s) TestCacheFlushWithoutCallback(t *testing.T) {
	var items []int
	const itemQuantity = 5
	for i := 0; i < itemQuantity; i++ {
		items = append(items, i)
	}
	c := NewTimeoutCache(testCacheTimeout)

	done := make(chan struct{})
	defer close(done)
	callbackQueue := make(chan struct{}, itemQuantity)

	for i, v := range items {
		callbackQueueTemp := make(chan struct{})
		c.Add(i, v, func() { close(callbackQueueTemp) })
		go func() {
			select {
			case <-callbackQueueTemp:
				callbackQueue <- struct{}{}
			case <-done:
			}
		}()
	}

	for i, v := range items {
		if got, ok := c.getForTesting(i); !ok || got.value != v {
			t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.value, ok, v, true)
		}
	}
	if l := c.Len(); l != itemQuantity {
		t.Fatalf("%d number of items in the cache, want %d", l, itemQuantity)
	}

	time.Sleep(testCacheTimeout / 2)
	c.Flush(false)

	for i := range items {
		if _, ok := c.getForTesting(i); ok {
			t.Fatalf("After Add(), before timeout, after Flush(), from cache got: _, %v, want _, %v", ok, false)
		}
	}
	if l := c.Len(); l != 0 {
		t.Fatalf("%d number of items in the cache, want 0", l)
	}

	select {
	case <-callbackQueue:
		t.Fatalf("unexpected callback after Flush")
	case <-time.After(testCacheTimeout * 2):
	}
}

