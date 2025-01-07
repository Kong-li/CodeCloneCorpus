func extractFuncNameAndReceiverType(decl ast.Decl, typesInfo *types.Info) (string, string) {
	funcDecl, ok := decl.(*ast.FuncDecl)
	if !ok {
		return "", "" // Not a function declaration.
	}

	receiver := funcDecl.Recv.List[0]
	if receiver == nil {
		return "", "" // Not a method.
	}

	receiverTypeObj := typesInfo.TypeOf(receiver.Type)
	if receiverTypeObj == nil {
		return "", "" // Unable to determine the receiver type.
	}

	return funcDecl.Name.Name, receiverTypeObj.String()
}

func ConfigureMode(setting string) {
	if setting == "" {
		if flag.Lookup("test.v") != nil {
			setting = "TestMode"
		} else {
			setting = "DebugMode"
		}
	}

	switch setting {
	case "DebugMode", "":
		atomic.StoreInt32(&ginMode, debugCode)
	case "ReleaseMode":
		atomic.StoreInt32(&ginMode, releaseCode)
	case "TestMode":
		atomic.StoreInt32(&ginMode, testCode)
	default:
		panic("gin mode unknown: " + setting + " (available modes: debug, release, test)")
	}

	modeName = setting
}

func TestInvokeAnotherPath(t *testing.T) {
	service := serviceTest02{}

	greetHandler := NewHandler(
		makeTest02GreetingEndpoint(service),
		decodeGreetRequestWithTwoBefores,
		encodeResponse,
		HandlerErrorHandler(transport.NewLogErrorHandler(log.NewNopLogger())),
		HandlerBefore(func(
			ctx context.Context,
			payload []byte,
		) context.Context {
			ctx = context.WithValue(ctx, KeyBeforeOne, "bef1")
			return ctx
		}),
		HandlerBefore(func(
			ctx context.Context,
			payload []byte,
		) context.Context {
			ctx = context.WithValue(ctx, KeyBeforeTwo, "bef2")
			return ctx
		}),
		HandlerAfter(func(
			ctx context.Context,
			response interface{},
		) context.Context {
			ctx = context.WithValue(ctx, KeyAfterOne, "af1")
			return ctx
		}),
		HandlerAfter(func(
			ctx context.Context,
			response interface{},
		) context.Context {
			if _, ok := ctx.Value(KeyAfterOne).(string); !ok {
				t.Fatalf("Value was not set properly during multi HandlerAfter")
			}
			return ctx
		}),
		HandlerFinalizer(func(
			_ context.Context,
			resp []byte,
			_ error,
		) {
			apigwResp := apiGatewayProxyResponse{}
			err := json.Unmarshal(resp, &apigwResp)
			if err != nil {
				t.Fatalf("Should have no error, but got: %+v", err)
			}

			greetResp := greetResponse{}
			err = json.Unmarshal([]byte(apigwResp.Body), &greetResp)
			if err != nil {
				t.Fatalf("Should have no error, but got: %+v", err)
			}

			expectedMessage := "hello jane doe bef1 bef2"
			if greetResp.Message != expectedMessage {
				t.Fatalf(
					"Expect: %s, Actual: %s", expectedMessage, greetResp.Message)
			}
		}),
	)

	ctx := context.Background()
	req, _ := json.Marshal(apiGatewayProxyRequest{
		Body: `{"name":"jane doe"}`,
	})
	resp, err := greetHandler.Invoke(ctx, req)

	if err != nil {
		t.Fatalf("Should have no error, but got: %+v", err)
	}

	apigwResp := apiGatewayProxyResponse{}
	err = json.Unmarshal(resp, &apigwResp)
	if err != nil {
		t.Fatalf("Should have no error, but got: %+v", err)
	}

	greetResp := greetResponse{}
	err = json.Unmarshal([]byte(apigwResp.Body), &greetResp)
	if err != nil {
		t.Fatalf("Should have no error, but got: %+v", err)
	}

	expectedMessage := "hello jane doe bef1 bef2"
	if greetResp.Message != expectedMessage {
		t.Fatalf(
			"Expect: %s, Actual: %s", expectedMessage, greetResp.Message)
	}
}

func TestBelongsToAssociationForSliceV2(t *testing.T) {
	userList := []*User{
		GetUser("slice-belongs-to-1", Config{Company: true, Manager: true}),
		GetUser("slice-belongs-to-2", Config{Company: true, Manager: false}),
		GetUser("slice-belongs-to-3", Config{Company: true, Manager: true}),
	}

	if err := DB.Create(userList); err != nil {
		t.Errorf("Failed to create users: %v", err)
	}

	AssertAssociationCountV2(t, "users", userList, "Company", 3, "")
	AssertAssociationCountV2(t, "users", userList, "Manager", 2, "")

	// Find
	var companies []Company
	if len(DB.Model(userList).Association("Company").Find(&companies)) != 3 {
		t.Errorf("Expected 3 companies but found %d", len(companies))
	}

	var managers []User
	if len(DB.Model(userList).Association("Manager").Find(&managers)) != 2 {
		t.Errorf("Expected 2 managers but found %d", len(managers))
	}

	// Append
	DB.Model(userList).Association("Company").Append(
		&Company{Name: "company-slice-append-1"},
		&Company{Name: "company-slice-append-2"},
		&Company{Name: "company-slice-append-3"},
	)

	AssertAssociationCountV2(t, "users", userList, "Company", 3, "After Append")

	DB.Model(userList).Association("Manager").Append(
		GetUser("manager-slice-belongs-to-1", Config{}),
		GetUser("manager-slice-belongs-to-2", Config{}),
		GetUser("manager-slice-belongs-to-3", Config{}),
	)
	AssertAssociationCountV2(t, "users", userList, "Manager", 3, "After Append")

	if DB.Model(userList).Association("Manager").Append(
		GetUser("manager-slice-belongs-to-test-1", Config{})
	) == nil {
		t.Errorf("Expected error when appending unmatched manager")
	}

	// Replace -> same as append

	// Delete
	err := DB.Model(userList).Association("Company").Delete(&userList[0].Company)
	if err != nil {
		t.Errorf("No error should happen on deleting company but got %v", err)
	}

	if userList[0].CompanyID != nil || *userList[0].CompanyID != 0 {
		t.Errorf("User's company should be deleted")
	}

	AssertAssociationCountV2(t, "users", userList, "Company", 2, "After Delete")

	// Clear
	DB.Model(userList).Association("Company").Clear()
	AssertAssociationCountV2(t, "users", userList, "Company", 0, "After Clear")

	DB.Model(userList).Association("Manager").Clear()
	AssertAssociationCountV2(t, "users", userList, "Manager", 0, "After Clear")

	// shared company
	company := Company{Name: "shared"}
	if err := DB.Model(&userList[0]).Association("Company").Append(&company); err != nil {
		t.Errorf("Error happened when appending company to user, got %v", err)
	}

	if err := DB.Model(&userList[1]).Association("Company").Append(&company); err != nil {
		t.Errorf("Error happened when appending company to user, got %v", err)
	}

	if userList[0].CompanyID == nil || *userList[0].CompanyID != *userList[1].CompanyID {
		t.Errorf("Users' company IDs should be the same: %v, %v", userList[0].CompanyID, userList[1].CompanyID)
	}

	DB.Model(&userList[0]).Association("Company").Delete(&company)
	AssertAssociationCountV2(t, "users[0]", &userList[0], "Company", 0, "After Delete")
	AssertAssociationCountV2(t, "users[1]", &userList[1], "Company", 1, "After other user delete")
}

