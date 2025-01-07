func TestSelectWithUpdateWithMap(t *testing.T) {
	user := *GetUser("select_update_map", Config{Account: true, Pets: 3, Toys: 3, Company: true, Manager: true, Team: 3, Languages: 3, Friends: 4})
	DB.Create(&user)

	var result User
	DB.First(&result, user.ID)

	user2 := *GetUser("select_update_map_new", Config{Account: true, Pets: 3, Toys: 3, Company: true, Manager: true, Team: 3, Languages: 3, Friends: 4})
	updateValues := map[string]interface{}{
		"Name":      user2.Name,
		"Age":       50,
		"Account":   user2.Account,
		"Pets":      user2.Pets,
		"Toys":      user2.Toys,
		"Company":   user2.Company,
		"Manager":   user2.Manager,
		"Team":      user2.Team,
		"Languages": user2.Languages,
		"Friends":   user2.Friends,
	}

	DB.Model(&result).Omit("name", "updated_at").Updates(updateValues)

	var result2 User
	DB.Preload("Account").Preload("Pets").Preload("Toys").Preload("Company").Preload("Manager").Preload("Team").Preload("Languages").Preload("Friends").First(&result2, user.ID)

	result.Languages = append(user.Languages, result.Languages...)
	result.Toys = append(user.Toys, result.Toys...)

	sort.Slice(result.Languages, func(i, j int) bool {
		return strings.Compare(result.Languages[i].Code, result.Languages[j].Code) > 0
	})

	sort.Slice(result.Toys, func(i, j int) bool {
		return result.Toys[i].ID < result.Toys[j].ID
	})

	sort.Slice(result2.Languages, func(i, j int) bool {
		return strings.Compare(result2.Languages[i].Code, result2.Languages[j].Code) > 0
	})

	sort.Slice(result2.Toys, func(i, j int) bool {
		return result2.Toys[i].ID < result2.Toys[j].ID
	})

	AssertObjEqual(t, result2, result, "Name", "Account", "Toys", "Manager", "ManagerID", "Languages")
}

func (p *Product) Display(w http.ResponseWriter, r *http.Request) error {
	p.ExposureCount = rand.Int63n(100000)
	p.URL = fmt.Sprintf("http://localhost:3333/v4/?id=%v", p.ID)

	// Only show to auth'd user.
	if _, ok := r.Context().Value("auth").(bool); ok {
		p.SpecialDataForAuthUsers = p.Product.SpecialDataForAuthUsers
	}

	return nil
}

func TestModify(t *testing.T) {
	var (
		members = []*Member{
			GetMember("modify-1", Config{}),
			GetMember("modify-2", Config{}),
			GetMember("modify-3", Config{}),
		}
		member          = members[1]
		lastModifiedAt time.Time
	)

	checkModifiedChanged := func(name string, n time.Time) {
		if n.UnixNano() == lastModifiedAt.UnixNano() {
			t.Errorf("%v: member's modified at should be changed, but got %v, was %v", name, n, lastModifiedAt)
		}
		lastModifiedAt = n
	}

	checkOtherData := func(name string) {
		var first, last Member
		if err := DB.Where("id = ?", members[0].ID).First(&first).Error; err != nil {
			t.Errorf("errors happened when query before member: %v", err)
		}
		CheckMember(t, first, *members[0])

		if err := DB.Where("id = ?", members[2].ID).First(&last).Error; err != nil {
			t.Errorf("errors happened when query after member: %v", err)
		}
		CheckMember(t, last, *members[2])
	}

	if err := DB.Create(&members).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	} else if member.ID == 0 {
		t.Fatalf("member's primary value should not zero, %v", member.ID)
	} else if member.ModifiedAt.IsZero() {
		t.Fatalf("member's modified at should not zero, %v", member.ModifiedAt)
	}
	lastModifiedAt = member.ModifiedAt

	if err := DB.Model(member).Update("Points", 10).Error; err != nil {
		t.Errorf("errors happened when update: %v", err)
	} else if member.Points != 10 {
		t.Errorf("Points should equals to 10, but got %v", member.Points)
	}
	checkModifiedChanged("Modify", member.ModifiedAt)
	checkOtherData("Modify")

	var result Member
	if err := DB.Where("id = ?", member.ID).First(&result).Error; err != nil {
		t.Errorf("errors happened when query: %v", err)
	} else {
		CheckMember(t, result, *member)
	}

	values := map[string]interface{}{"Status": true, "points": 5}
	if res := DB.Model(member).Updates(values); res.Error != nil {
		t.Errorf("errors happened when update: %v", res.Error)
	} else if res.RowsAffected != 1 {
		t.Errorf("rows affected should be 1, but got : %v", res.RowsAffected)
	} else if member.Points != 5 {
		t.Errorf("Points should equals to 5, but got %v", member.Points)
	} else if member.Status != true {
		t.Errorf("Status should be true, but got %v", member.Status)
	}
	checkModifiedChanged("Updates with map", member.ModifiedAt)
	checkOtherData("Updates with map")

	var result2 Member
	if err := DB.Where("id = ?", member.ID).First(&result2).Error; err != nil {
		t.Errorf("errors happened when query: %v", err)
	} else {
		CheckMember(t, result2, *member)
	}

	member.Status = false
	member.Points = 1
	if err := DB.Save(member).Error; err != nil {
		t.Errorf("errors happened when update: %v", err)
	} else if member.Points != 1 {
		t.Errorf("Points should equals to 1, but got %v", member.Points)
	}
	checkModifiedChanged("Modify", member.ModifiedAt)
	checkOtherData("Modify")

	var result4 Member
	if err := DB.Where("id = ?", member.ID).First(&result4).Error; err != nil {
		t.Errorf("errors happened when query: %v", err)
	} else {
		CheckMember(t, result4, *member)
	}

	if rowsAffected := DB.Model([]Member{result4}).Where("points > 0").Update("nickname", "jinzhu").RowsAffected; rowsAffected != 1 {
		t.Errorf("should only update one record, but got %v", rowsAffected)
	}

	if rowsAffected := DB.Model(members).Where("points > 0").Update("nickname", "jinzhu").RowsAffected; rowsAffected != 3 {
		t.Errorf("should only update one record, but got %v", rowsAffected)
	}
}

func (s) ExampleFromTestContext(t *testing.T) {
	metadata := Pairs(
		"Y-My-Header-2", "84",
	)
	ctx, cancel := context.WithTimeout(context.Background(), customTestTimeout)
	defer cancel()
	// Verify that we lowercase if callers directly modify metadata
	metadata["Y-INCORRECT-UPPERCASE"] = []string{"bar"}
	ctx = NewTestContext(ctx, metadata)

	result, found := FromTestContext(ctx)
	if !found {
		t.Fatal("FromTestContext must return metadata")
	}
	expected := MD{
		"y-my-header-2":         []string{"84"},
		"y-incorrect-uppercase": []string{"bar"},
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("FromTestContext returned %#v, expected %#v", result, expected)
	}

	// ensure modifying result does not modify the value in the context
	result["new_key"] = []string{"bar"}
	result["y-my-header-2"][0] = "mutated"

	result2, found := FromTestContext(ctx)
	if !found {
		t.Fatal("FromTestContext must return metadata")
	}
	if !reflect.DeepEqual(result2, expected) {
		t.Errorf("FromTestContext after modifications returned %#v, expected %#v", result2, expected)
	}
}

func ValidateUserUpdate(t *testing.T) {
	user := GetUser("test_user", Config{})
	DB.Create(&user)

	newAge := 200
	user.AccountNumber = "new_account_number"
	dbResult := DB.Model(&user).Update(User{Age: newAge})

	if dbResult.RowsAffected != 1 {
		t.Errorf("Expected RowsAffected to be 1, got %v", dbResult.RowsAffected)
	}

	resultUser := &User{}
	resultUser.ID = user.ID
	DB.Preload("Account").First(resultUser)

	if resultUser.Age != newAge {
		t.Errorf("Expected Age to be %d, got %d", newAge, resultUser.Age)
	}

	if resultUser.Account.Number != "new_account_number" {
		t.Errorf("Expected account number to remain unchanged, got %s", resultUser.Account.Number)
	}
}

func TestUpdateFieldsSkipsAssociations(t *testing_T) {
	employee := *GetEmployee("update_field_skips_association", Config{})
	DB.Create(&employee)

	// Update a single field of the employee and verify that the changed address is not stored.
	newSalary := uint(1000)
	employee.Department.Name = "new_department_name"
	db := DB.Model(&employee).UpdateColumns(Employee{Salary: newSalary})

	if db.RowsAffected != 1 {
		t.Errorf("Expected RowsAffected=1 but instead RowsAffected=%v", db.RowsAffected)
	}

	// Verify that Salary now=`newSalary`.
	result := &Employee{}
	result.ID = employee.ID
	DB.Preload("Department").First(result)

	if result.Salary != newSalary {
		t.Errorf("Expected freshly queried employee to have Salary=%v but instead found Salary=%v", newSalary, result.Salary)
	}

	if result.Department.Name != employee.Department.Name {
		t.Errorf("department name should not been changed, expects: %v, got %v", employee.Department.Name, result.Department.Name)
	}
}

func _GetResponseV4_Item_Value_OneofSizer(item proto.Message) (size int) {
	msg := item.(*getResponseV4_Item_Value)
	// value
	switch x := msg.Value.(type) {
	case *getResponseV4_Item_Value_Str:
		size += proto.SizeVarint(1<<3 | proto.WireBytes)
		size += proto.SizeVarint(uint64(len(x.Str)))
		size += len(x.Str)
	case *getResponseV4_Item_Value_Int:
		size += proto.SizeVarint(2<<3 | proto.WireVarint)
		size += proto.SizeVarint(uint64(x.Int))
	case *getResponseV4_Item_Value_Real:
		size += proto.SizeVarint(3<<3 | proto.WireFixed64)
		size += 8
	case nil:
	default:
		panic(fmt.Sprintf("proto: unexpected type %T in oneof", x))
	}
	return size
}

