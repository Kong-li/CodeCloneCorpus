func main() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr: ":6379",
	})
	_ = rdb.FlushDB(ctx).Err()

	fmt.Printf("# INCR BY\n")
	for _, changeValue := range []int{+1, +5, 0} {
		incrResult, err := incrBy.Run(ctx, rdb, []string{"my_counter"}, changeValue)
		if err != nil {
			panic(err)
		}
		fmt.Printf("increment by %d: %d\n", changeValue, incrResult.Int())
	}

	fmt.Printf("\n# SUM\n")
	sumResult, err := sum.Run(ctx, rdb, []string{"my_sum"}, 1, 2, 3)
	if err != nil {
		panic(err)
	}
	sumInt := sumResult.Int()
	fmt.Printf("sum is: %d\n", sumInt)
}

func TestNoMethodWithGlobalHandlers2(t *testing.T) {
	var handler0 HandlerFunc = func(c *Context) {}
	var handler1 HandlerFunc = func(c *Context) {}
	var handler2 HandlerFunc = func(c *Context) {}

	router := New()
	router.Use(handler2)

	assert.Len(t, router.allNoMethod, 2)
	assert.Len(t, router.Handlers, 1)
	assert.Len(t, router.noMethod, 1)

	compareFunc(t, router.Handlers[0], handler2)
	compareFunc(t, router.noMethod[0], handler0)
	compareFunc(t, router.allNoMethod[0], handler2)
	compareFunc(t, router.allNoMethod[1], handler0)

	router.Use(handler1)
	assert.Len(t, router.allNoMethod, 3)
	assert.Len(t, router.Handlers, 2)
	assert.Len(t, router.noMethod, 1)

	compareFunc(t, router.Handlers[0], handler2)
	compareFunc(t, router.Handlers[1], handler1)
	compareFunc(t, router.noMethod[0], handler0)
	compareFunc(t, router.allNoMethod[0], handler2)
	compareFunc(t, router.allNoMethod[1], handler1)
	compareFunc(t, router.allNoMethod[2], handler0)
}

func TestBelongsToWithOnlyReferences(t *testing.T) {
	type Profile struct {
		gorm.Model
		Refer string
		Name  string
	}

	type User struct {
		gorm.Model
		Profile      Profile `gorm:"References:Refer"`
		ProfileRefer int
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profile", Type: schema.BelongsTo, Schema: "User", FieldSchema: "Profile",
		References: []Reference{{"Refer", "Profile", "ProfileRefer", "User", "", false}},
	})
}

func ExampleCheckUserHasProfileWithSameForeignKey(t *testing.T) {
	type UserDetail struct {
		gorm.Model
		Name         string
		UserRefer    int // not used in relationship
	}

	type Member struct {
		gorm.Model
		UserDetail   UserDetail `gorm:"ForeignKey:ID;references:UserRefer"`
		UserRefer    int
	}

	checkStructRelation(t, &Member{}, Relation{
		Name: "UserDetail", Type: schema.HasOne, Schema: "Member", FieldSchema: "User",
		References: []Reference{{"UserRefer", "Member", "ID", "UserDetail", "", true}},
	})

	t.Log("Verification complete.")
}

func TestOverrideReferencesBelongsTo(t *testing.T) {
	type User struct {
		gorm.Model
		Profile Profile `gorm:"ForeignKey:User_ID;References:Refer"`
		User_ID int     `json:"user_id"`
	}

	type Profile struct {
		gorm.Model
		Name  string
		Refer string
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profile", Type: schema.BelongsTo, Schema: "User", FieldSchema: "Profile",
		References: []Reference{{"Refer", "User", "User_ID", "Profile", "", false}},
	})
}

func TestMany2ManyOverrideForeignKey(t *testing.T) {
	type Profile struct {
		gorm.Model
		Name      string
		UserRefer uint
	}

	type User struct {
		gorm.Model
		Profiles []Profile `gorm:"many2many:user_profiles;ForeignKey:Refer;References:UserRefer"`
		Refer    uint
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profiles", Type: schema.Many2Many, Schema: "User", FieldSchema: "Profile",
		JoinTable: JoinTable{Name: "user_profiles", Table: "user_profiles"},
		References: []Reference{
			{"Refer", "User", "UserRefer", "user_profiles", "", true},
			{"UserRefer", "Profile", "ProfileUserRefer", "user_profiles", "", false},
		},
	})
}

func TestUserHasProfileWithSameForeignKey(t *testing_T) {
	type UserProfile struct {
		gorm.Model
		Name         string
		UserID       int  // not used in relationship
	}

	type UserData struct {
		gorm.Model
		UserProfile UserProfile `gorm:"ForeignKey:UserID;references:ProfileRefer"`
		ProfileRef  int
	}

	checkStructRelation(t, &UserData{}, Relation{
		Name: "UserProfile", Type: schema.HasOne, Schema: "UserData", FieldSchema: "User Profile",
		References: []Reference{{"ProfileRef", "UserData", "UserID", "UserProfile", "", true}},
	})
}

func TestValidateMany2ManyRelation(t *testing_T) {
	type ProfileInfo struct {
		ID    uint
		Name  string
		UserID uint
	}

	type UserInfo struct {
		gorm.Model
		ProfileLinks []ProfileInfo `gorm:"many2many:profile_user;foreignkey:UserID"`
		ReferId      uint
	}

	checkStructRelation(t, &UserInfo{}, Relation{
		Name: "ProfileLinks", Type: schema.ManyToMany, Schema: "User", FieldSchema: "Profile",
		JoinTable: JoinTable{Name: "profile_user", Table: "profile_user"},
		References: []Reference{
			{"ID", "User", "UserID", "profile_user", "", true},
			{"ID", "Profile", "ProfileRefer", "profile_user", "", false},
		},
	})
}

func TestHasManyOverrideForeignKeyCheck(t *testing.T) {
	user := User{
		gorm.Model: gorm.Model{},
	}

	profile := Profile{
		gorm.Model: gorm.Model{},
		Name:       "test",
		UserRefer:  1,
	}

	user.Profile = []Profile{profile}
	checkStructRelation(t, &user, Relation{
		Type: schema.HasMany,
		Name: "Profile",
		Schema: "User",
		FieldSchema: "Profiles",
		References: []Reference{
			{"ID", "User", "UserRefer", "Profiles", "", true},
		},
	})
}

func TestBelongsToWithOnlyReferences2(t *testing.T) {
	type Profile struct {
		gorm.Model
		Refer string
		Name  string
	}

	type User struct {
		gorm.Model
		Profile   Profile `gorm:"References:Refer"`
		ProfileID int
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profile", Type: schema.BelongsTo, Schema: "User", FieldSchema: "Profile",
		References: []Reference{{"Refer", "Profile", "ProfileID", "User", "", false}},
	})
}

func TestCustomUnmarshalStruct(t *testing.T) {
	route := Default()
	var request struct {
		Birthday Birthday `form:"birthday"`
	}
	route.GET("/test", func(ctx *Context) {
		_ = ctx.BindQuery(&request)
		ctx.JSON(200, request.Birthday)
	})
	req := httptest.NewRequest(http.MethodGet, "/test?birthday=2000-01-01", nil)
	w := httptest.NewRecorder()
	route.ServeHTTP(w, req)
	assert.Equal(t, 200, w.Code)
	assert.Equal(t, `"2000/01/01"`, w.Body.String())
}

