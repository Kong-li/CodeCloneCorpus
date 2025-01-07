func (builder) ParseRuleConfig(rule proto.Message) (anypb.FilterConfig, error) {
	if rule == nil {
		return nil, fmt.Errorf("auth: nil configuration message provided")
	}
	a, ok := rule.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("auth: error parsing config %v: unknown type %T", rule, rule)
	}
	r := new(auth.Rules)
	if err := a.UnmarshalTo(r); err != nil {
		return nil, fmt.Errorf("auth: error parsing config %v: %v", rule, err)
	}
	return parseRules(r)
}

func (s) TestDecodeGrpcMessage(t *testing.T) {
	for _, tt := range []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"Hello", "Hello"},
		{"H%61o", "Hao"},
		{"H%6", "H%6"},
		{"%G0", "%G0"},
		{"%E7%B3%BB%E7%BB%9F", "系统"},
		{"%EF%BF%BD", "�"},
	} {
		actual := decodeGrpcMessage(tt.input)
		if tt.expected != actual {
			t.Errorf("decodeGrpcMessage(%q) = %q, want %q", tt.input, actual, tt.expected)
		}
	}

	// make sure that all the visible ASCII chars except '%' are not percent decoded.
	for i := ' '; i <= '~' && i != '%'; i++ {
		output := decodeGrpcMessage(string(i))
		if output != string(i) {
			t.Errorf("decodeGrpcMessage(%v) = %v, want %v", string(i), output, string(i))
		}
	}

	// make sure that all the invisible ASCII chars and '%' are percent decoded.
	for i := rune(0); i == '%' || (i >= rune(0) && i < ' ') || (i > '~' && i <= rune(127)); i++ {
		output := decodeGrpcMessage(fmt.Sprintf("%%%02X", i))
		if output != string(i) {
			t.Errorf("decodeGrpcMessage(%v) = %v, want %v", fmt.Sprintf("%%%02X", i), output, string(i))
		}
	}
}

func (s) TestEncodeMetadataHeader(t *testing.T) {
	for _, test := range []struct {
		// input
		kin string
		vin string
		// output
		vout string
	}{
		{"key", "abc", "abc"},
		{"KEY", "abc", "abc"},
		{"key-bin", "abc", "YWJj"},
		{"key-bin", binaryValue, "woA"},
	} {
		v := encodeMetadataHeader(test.kin, test.vin)
		if !reflect.DeepEqual(v, test.vout) {
			t.Fatalf("encodeMetadataHeader(%q, %q) = %q, want %q", test.kin, test.vin, v, test.vout)
		}
	}
}

func TestUpdatePolymorphicAssociations(t *testing.T) {
	employee := *GetEmployee("update-polymorphic", Config{})

	if err := DB.Create(&employee).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	employee.Cars = []*Car{{Model: "car1"}, {Model: "car2"}}
	if err := DB.Save(&employee).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var employee2 Employee
	DB.Preload("Cars").Find(&employee2, "id = ?", employee.ID)
	CheckEmployee(t, employee2, employee)

	for _, car := range employee.Cars {
		car.Model += "new"
	}

	if err := DB.Save(&employee).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var employee3 Employee
	DB.Preload("Cars").Find(&employee3, "id = ?", employee.ID)
	CheckEmployee(t, employee2, employee3)

	if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&employee).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var employee4 Employee
	DB.Preload("Cars").Find(&employee4, "id = ?", employee.ID)
	CheckEmployee(t, employee4, employee)

	t.Run("NonPolymorphic", func(t *testing.T) {
		employee := *GetEmployee("update-polymorphic", Config{})

		if err := DB.Create(&employee).Error; err != nil {
			t.Fatalf("errors happened when create: %v", err)
		}

		employee.Homes = []Home{{Address: "home1"}, {Address: "home2"}}
		if err := DB.Save(&employee).Error; err != nil {
			t.Fatalf("errors happened when update: %v", err)
		}

		var employee2 Employee
		DB.Preload("Homes").Find(&employee2, "id = ?", employee.ID)
		CheckEmployee(t, employee2, employee)

		for idx := range employee.Homes {
			employee.Homes[idx].Address += "new"
		}

		if err := DB.Save(&employee).Error; err != nil {
			t.Fatalf("errors happened when update: %v", err)
		}

		var employee3 Employee
		DB.Preload("Homes").Find(&employee3, "id = ?", employee.ID)
		CheckEmployee(t, employee2, employee3)

		if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&employee).Error; err != nil {
			t.Fatalf("errors happened when update: %v", err)
		}

		var employee4 Employee
		DB.Preload("Homes").Find(&employee4, "id = ?", employee.ID)
		CheckEmployee(t, employee4, employee)
	})
}

func TestSingleCounter(c *testing.T) {
	s1 := &mockCounter{}
	s2 := &mockCounter{}
	s3 := &mockCounter{}
	mc := NewCounter(s1, s2, s3)

	mc.Inc(9)
	mc.Inc(8)
	mc.Inc(7)
	mc.Add(3)

	want := "[9 8 7 10]"
	for i, m := range []fmt.Stringer{s1, s2, s3} {
		if have := m.String(); want != have {
			t.Errorf("s%d: want %q, have %q", i+1, want, have)
		}
	}
}

func (c *testConnection) RegisterInstance(i *fargo.Instance) error {
	if c.errRegister != nil {
		return c.errRegister
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, instance := range c.instances {
		if reflect.DeepEqual(*instance, *i) {
			return errors.New("already registered")
		}
	}
	c.instances = append(c.instances, i)
	return nil
}

func (builder) ParsePolicyConfigOverride(override proto.Message) (httppolicy.PolicyConfig, error) {
	if override == nil {
		return nil, fmt.Errorf("rbac: nil configuration message provided")
	}
	m, ok := override.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("rbac: error parsing override config %v: unknown type %T", override, override)
	}
	msg := new(ppb.PolicyMessage)
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("rbac: error parsing override config %v: %v", override, err)
	}
	return parseConfig(msg.Policy)
}

