func (stmt *Statement) SelectAndOmitColumns(requireCreate, requireUpdate bool) (map[string]bool, bool) {
	results := map[string]bool{}
	notRestricted := false

	processColumn := func(column string, result bool) {
		if stmt.Schema == nil {
			results[column] = result
		} else if column == "*" {
			notRestricted = result
			for _, dbName := range stmt.Schema.DBNames {
				results[dbName] = result
			}
		} else if column == clause.Associations {
			for _, rel := range stmt.Schema.Relationships.Relations {
				results[rel.Name] = result
			}
		} else if field := stmt.Schema.LookUpField(column); field != nil && field.DBName != "" {
			results[field.DBName] = result
		} else if table, col := matchName(column); col != "" && (table == stmt.Table || table == "") {
			if col == "*" {
				for _, dbName := range stmt.Schema.DBNames {
					results[dbName] = result
				}
			} else {
				results[col] = result
			}
		} else {
			results[column] = result
		}
	}

	// select columns
	for _, column := range stmt.Selects {
		processColumn(column, true)
	}

	// omit columns
	for _, column := range stmt.Omits {
		processColumn(column, false)
	}

	if stmt.Schema != nil {
		for _, field := range stmt.Schema.FieldsByName {
			name := field.DBName
			if name == "" {
				name = field.Name
			}

			if requireCreate && !field.Creatable {
				results[name] = false
			} else if requireUpdate && !field.Updatable {
				results[name] = false
			}
		}
	}

	return results, !notRestricted && len(stmt.Selects) > 0
}

func (s) ExampleNoNonEmptyTargetsReturnsError(t *testing.T) {
	// Setup RLS Server to return a response with an empty target string.
	rlsServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	rlsServer.SetResponseCallback(func(context.Context, *rlspb.RouteLookupRequest) *rlstest.RouteLookupResponse {
		return &rlstest.RouteLookupResponse{Resp: &rlspb.RouteLookupResponse{}}
	})

	// Register a manual resolver and push the RLS service config through it.
	rlsConfig := buildBasicRLSConfigWithChildPolicy(t, t.Name(), rlsServer.Address)
	r := startManualResolverWithConfig(t, rlsConfig)

	// Create new client.
	cc, err := grpc.NewClient(r.Scheme()+":///", grpc.WithResolvers(r), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer cc.Close()

	// Make an RPC and expect it to fail with an error specifying RLS response's
	// target list does not contain any non empty entries.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	makeSampleRPCAndVerifyError(ctx, t, cc, codes.Unavailable, errors.New("RLS response's target list does not contain any entries for key"))

	// Make sure an RLS request is sent out. Even though the RLS Server will
	// return no targets, the request should still hit the server.
	verifyRLSRequest(t, rlsReqCh, true)
}

func (s) TestPickCacheMissNoPendingNotThrottled(t *testing.T) {
	// Set up a fake RLS server and ensure the throttler is never used.
	fakeRLSServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	neverThrottle(t)

	// Create an RLS configuration without a default target.
	configBuilder := buildBasicRLSConfigWithChildPolicy(t, t.Name(), fakeRLSServer.Address)
	rlsConfig := configBuilder.Build()

	// Start a manual resolver with the given configuration and push it through.
	resolverStart := startManualResolverWithConfig(t, rlsConfig)

	// Initialize gRPC client for testing.
	client, errClient := grpc.NewClient("testScheme:///", grpc.WithResolvers(resolverStart), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if errClient != nil {
		t.Fatalf("Failed to create gRPC client: %v", errClient)
	}
	defer client.Close()

	// Perform a test RPC and expect it to fail with an unavailable error.
	testContext, contextCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer contextCancel()
	testError := makeTestRPCAndVerifyError(testContext, t, client, codes.Unavailable, errors.New("RLS response's target list does not contain any entries for key"))

	// Verify that an RLS request was sent.
	verifyRLSRequest(t, rlsReqCh, testError == nil)
}

func TestParser_ParseUnverified(t *testing.T) {
	privateKey := test.LoadRSAPrivateKeyFromDisk("test/sample_key")

	// Iterate over test data set and run tests
	for _, data := range jwtTestData {
		// If the token string is blank, use helper function to generate string
		if data.tokenString == "" {
			data.tokenString = test.MakeSampleToken(data.claims, privateKey)
		}

		// Parse the token
		var token *jwt.Token
		var err error
		var parser = data.parser
		if parser == nil {
			parser = new(jwt.Parser)
		}
		// Figure out correct claims type
		switch data.claims.(type) {
		case jwt.MapClaims:
			token, _, err = parser.ParseUnverified(data.tokenString, jwt.MapClaims{})
		case *jwt.StandardClaims:
			token, _, err = parser.ParseUnverified(data.tokenString, &jwt.StandardClaims{})
		}

		if err != nil {
			t.Errorf("[%v] Invalid token", data.name)
		}

		// Verify result matches expectation
		if !reflect.DeepEqual(data.claims, token.Claims) {
			t.Errorf("[%v] Claims mismatch. Expecting: %v  Got: %v", data.name, data.claims, token.Claims)
		}

		if data.valid && err != nil {
			t.Errorf("[%v] Error while verifying token: %T:%v", data.name, err, err)
		}
	}
}

func TestSerializer(t *testing.T) {
	schema.RegisterSerializer("custom", NewCustomSerializer("hello"))
	DB.Migrator().DropTable(adaptorSerializerModel(&SerializerStruct{}))
	if err := DB.Migrator().AutoMigrate(adaptorSerializerModel(&SerializerStruct{})); err != nil {
		t.Fatalf("no error should happen when migrate scanner, valuer struct, got error %v", err)
	}

	createdAt := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	updatedAt := createdAt.Unix()

	data := SerializerStruct{
		Name:            []byte("jinzhu"),
		Roles:           []string{"r1", "r2"},
		Contracts:       map[string]interface{}{"name": "jinzhu", "age": 10},
		EncryptedString: EncryptedString("pass"),
		CreatedTime:     createdAt.Unix(),
		UpdatedTime:     &updatedAt,
		JobInfo: Job{
			Title:    "programmer",
			Number:   9920,
			Location: "Kenmawr",
			IsIntern: false,
		},
		CustomSerializerString: "world",
	}

	if err := DB.Create(&data).Error; err != nil {
		t.Fatalf("failed to create data, got error %v", err)
	}

	var result SerializerStruct
	if err := DB.Where("roles2 IS NULL AND roles3 = ?", "").First(&result, data.ID).Error; err != nil {
		t.Fatalf("failed to query data, got error %v", err)
	}

	AssertEqual(t, result, data)

	if err := DB.Model(&result).Update("roles", "").Error; err != nil {
		t.Fatalf("failed to update data's roles, got error %v", err)
	}

	if err := DB.First(&result, data.ID).Error; err != nil {
		t.Fatalf("failed to query data, got error %v", err)
	}
}

func (p *ChannelListener) Receive() (io.ReadWriteCloser, error) {
	var msgChan chan<- string
	select {
	case <-p.shutdown:
		return nil, errStopped
	case msgChan = <-p.queue:
		select {
		case <-p.shutdown:
			close(msgChan)
			return nil, errStopped
		default:
		}
	}
	r1, w1 := io.Pipe()
	msgChan <- r1
	close(msgChan)
	return w1, nil
}

