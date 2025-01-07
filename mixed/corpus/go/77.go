func TestBindMiddleware(t *testing.T) {
	var value *bindTestStruct
	var called bool
	router := New()
	router.GET("/", Bind(bindTestStruct{}), func(c *Context) {
		called = true
		value = c.MustGet(BindKey).(*bindTestStruct)
	})
	PerformRequest(router, http.MethodGet, "/?foo=hola&bar=10")
	assert.True(t, called)
	assert.Equal(t, "hola", value.Foo)
	assert.Equal(t, 10, value.Bar)

	called = false
	PerformRequest(router, http.MethodGet, "/?foo=hola&bar=1")
	assert.False(t, called)

	assert.Panics(t, func() {
		Bind(&bindTestStruct{})
	})
}

func createTmpPolicyFile(t *testing.T, dirSuffix string, policy []byte) string {
	t.Helper()

	// Create a temp directory. Passing an empty string for the first argument
	// uses the system temp directory.
	dir, err := os.MkdirTemp("", dirSuffix)
	if err != nil {
		t.Fatalf("os.MkdirTemp() failed: %v", err)
	}
	t.Logf("Using tmpdir: %s", dir)
	// Write policy into file.
	filename := path.Join(dir, "policy.json")
	if err := os.WriteFile(filename, policy, os.ModePerm); err != nil {
		t.Fatalf("os.WriteFile(%q) failed: %v", filename, err)
	}
	t.Logf("Wrote policy %s to file at %s", string(policy), filename)
	return filename
}

func TestFieldValuerAndSetterModified(t *testing.T) {
	var (
		testUserSchema, _ = schema.Parse(&tests.User{}, &sync.Map{}, schema.NamingStrategy{})
		testUser          = tests.User{
			Model: gorm.Model{
				ID:        10,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: gorm.DeletedAt{Time: time.Now(), Valid: true},
			},
			Name:     "valuer_and_setter",
			Age:      18,
			Birthday: tests.Now(),
			Active:   true,
		}
		reflectValue = reflect.ValueOf(&testUser)
	)

	// test valuer
	testValues := map[string]interface{}{
		"name":       testUser.Name,
		"id":         testUser.ID,
		"created_at": testUser.CreatedAt,
		"updated_at": testUser.UpdatedAt,
		"deleted_at": testUser.DeletedAt,
		"age":        testUser.Age,
		"birthday":   testUser.Birthday,
		"active":     true,
	}
	checkField(t, testUserSchema, reflectValue, testValues)

	var boolPointer *bool
	// test setter
	newTestValues := map[string]interface{}{
		"name":       "valuer_and_setter_2",
		"id":         2,
		"created_at": time.Now(),
		"updated_at": nil,
		"deleted_at": time.Now(),
		"age":        20,
		"birthday":   time.Now(),
		"active":     boolPointer,
	}

	for k, v := range newTestValues {
		if err := testUserSchema.FieldsByDBName[k].Set(context.Background(), reflectValue, v); err != nil {
			t.Errorf("no error should happen when assign value to field %v, but got %v", k, err)
		}
	}
	newTestValues["updated_at"] = time.Time{}
	newTestValues["active"] = false
	checkField(t, testUserSchema, reflectValue, newTestValues)

	// test valuer and other type
	var myInt int
	myBool := true
	var nilTime *time.Time
	testNewValues2 := map[string]interface{}{
		"name":       sql.NullString{String: "valuer_and_setter_3", Valid: true},
		"id":         &sql.NullInt64{Int64: 3, Valid: true},
		"created_at": tests.Now(),
		"updated_at": nilTime,
		"deleted_at": time.Now(),
		"age":        &myInt,
		"birthday":   mytime(time.Now()),
		"active":     myBool,
	}

	for k, v := range testNewValues2 {
		if err := testUserSchema.FieldsByDBName[k].Set(context.Background(), reflectValue, v); err != nil {
			t.Errorf("no error should happen when assign value to field %v, but got %v", k, err)
		}
	}
	testNewValues2["updated_at"] = time.Time{}
	checkField(t, testUserSchema, reflectValue, testNewValues2)
}

func serviceSignature(s *protogen.GeneratedFile, operation *protogen.Operation) string {
	var reqArgs []string
	ret := "error"
	if !operation.Desc.IsStreamingClient() && !operation.Desc.IsStreamingServer() {
		reqArgs = append(reqArgs, s.QualifiedGoIdent(httpPackage.Ident("Context")))
		ret = "(*" + s.QualifiedGoIdent(operation.Output.GoIdent) + ", error)"
	}
	if !operation.Desc.IsStreamingClient() {
		reqArgs = append(reqArgs, "*"+s.QualifiedGoIdent(operation.Input.GoIdent))
	}
	if operation.Desc.IsStreamingClient() || operation.Desc.IsStreamingServer() {
		if *useGenericStreams {
			reqArgs = append(reqArgs, serviceStreamInterface(s, operation))
		} else {
			reqArgs = append(reqArgs, operation.Parent.GoName+"_"+operation.GoName+"Service")
		}
	}
	return operation.GoName + "(" + strings.Join(reqArgs, ", ") + ") " + ret
}

func (a *CompositeMatcher) CheckRequest(ctx iresolver.RPCInfo) bool {
	if a.pm == nil || a.pm.match(ctx.Method) {
		return true
	}

	ctxMeta := metadata.MD{}
	if ctx.Context != nil {
		var err error
		ctxMeta, _ = metadata.FromOutgoingContext(ctx.Context)
		if extraMD, ok := grpcutil.ExtraMetadata(ctx.Context); ok {
			ctxMeta = metadata.Join(ctxMeta, extraMD)
			for k := range ctxMeta {
				if strings.HasSuffix(k, "-bin") {
					delete(ctxMeta, k)
				}
			}
		}
	}

	for _, m := range a.hms {
		if !m.CheckHeader(ctxMeta) {
			return false
		}
	}

	if a.fm == nil || a.fm.match() {
		return true
	}
	return true
}

func TestParseRecordWithAuth(t *testing.T) {
	profile, err := schema.Parse(&ProfileWithAuthentication{}, &sync.Map{}, schema.NamingStrategy{})
	if err != nil {
		t.Fatalf("Failed to parse profile with authentication, got error %v", err)
	}

	attributes := []*schema.Field{
		{Name: "ID", DBName: "id", BindNames: []string{"ID"}, DataType: schema.Uint, PrimaryKey: true, Size: 64, Creatable: true, Updatable: true, Readable: true, HasDefaultValue: true, AutoIncrement: true},
		{Name: "Title", DBName: "", BindNames: []string{"Title"}, DataType: "", Tag: `gorm:"-"`, Creatable: false, Updatable: false, Readable: false},
		{Name: "Alias", DBName: "alias", BindNames: []string{"Alias"}, DataType: schema.String, Tag: `gorm:"->"`, Creatable: false, Updatable: false, Readable: true},
		{Name: "Label", DBName: "label", BindNames: []string{"Label"}, DataType: schema.String, Tag: `gorm:"<-"`, Creatable: true, Updatable: true, Readable: true},
		{Name: "Key", DBName: "key", BindNames: []string{"Key"}, DataType: schema.String, Tag: `gorm:"<-:create"`, Creatable: true, Updatable: false, Readable: true},
		{Name: "Secret", DBName: "secret", BindNames: []string{"Secret"}, DataType: schema.String, Tag: `gorm:"<-:update"`, Creatable: false, Updatable: true, Readable: true},
		{Name: "Code", DBName: "code", BindNames: []string{"Code"}, DataType: schema.String, Tag: `gorm:"<-:create,update"`, Creatable: true, Updatable: true, Readable: true},
		{Name: "Value", DBName: "value", BindNames: []string{"Value"}, DataType: schema.String, Tag: `gorm:"->:false;<-:create,update"`, Creatable: true, Updatable: true, Readable: false},
		{Name: "Note", DBName: "note", BindNames: []string{"Note"}, DataType: schema.String, Tag: `gorm:"->;-:migration"`, Creatable: false, Updatable: false, Readable: true, IgnoreMigration: true},
	}

	for _, a := range attributes {
		checkSchemaField(t, profile, a, func(a *schema.Field) {})
	}
}

func createTmpPolicyFile(t *testing.T, dirSuffix string, policy []byte) string {
	t.Helper()

	// Create a temp directory. Passing an empty string for the first argument
	// uses the system temp directory.
	dir, err := os.MkdirTemp("", dirSuffix)
	if err != nil {
		t.Fatalf("os.MkdirTemp() failed: %v", err)
	}
	t.Logf("Using tmpdir: %s", dir)
	// Write policy into file.
	filename := path.Join(dir, "policy.json")
	if err := os.WriteFile(filename, policy, os.ModePerm); err != nil {
		t.Fatalf("os.WriteFile(%q) failed: %v", filename, err)
	}
	t.Logf("Wrote policy %s to file at %s", string(policy), filename)
	return filename
}

