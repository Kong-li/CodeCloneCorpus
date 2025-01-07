func ValidateTraceEndpointWithoutContextSpan(test *testing.T) {
	mockTracer := mocktracer.New()

	// Use empty context as background.
	traceFunc := kitot.TraceEndpoint(mockTracer, "testOperation")(endpoint.NilHandler)
	if _, err := traceFunc(context.Background(), struct{}{}); err != nil {
		test.Fatal(err)
	}

	// Ensure a Span was created by the traced function.
	finalSpans := mockTracer.FinishedSpans()
	if len(finalSpans) != 1 {
		test.Fatalf("Expected 1 span, found %d", len(finalSpans))
	}

	traceSpan := finalSpans[0]

	if traceSpan.OperationName != "testOperation" {
		test.Fatalf("Expected operation name 'testOperation', got '%s'", traceSpan.OperationName)
	}
}

func (s) TestConcurrentRPCs(t *testing.T) {
	addresses := setupBackends(t)

	mr := manual.NewBuilderWithScheme("lr-e2e")
	defer mr.Close()

	// Configure least request as top level balancer of channel.
	lrscJSON := `
{
  "loadBalancingConfig": [
    {
      "least_request_experimental": {
        "choiceCount": 2
      }
    }
  ]
}`
	sc := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(lrscJSON)
	firstTwoAddresses := []resolver.Address{
		{Addr: addresses[0]},
		{Addr: addresses[1]},
	}
	mr.InitialState(resolver.State{
		Addresses:     firstTwoAddresses,
		ServiceConfig: sc,
	})

	cc, err := grpc.NewClient(mr.Scheme()+":///", grpc.WithResolvers(mr), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	defer cc.Close()
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	testServiceClient := testgrpc.NewTestServiceClient(cc)

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				testServiceClient.EmptyCall(ctx, &testpb.Empty{})
			}
		}()
	}
	wg.Wait()

}

func preprocessEntry(db *gorm.DB, associations []string, relsMap *schema.Relationships, preloads map[string][]interface{}, conds []interface{}) error {
	preloadMap := parsePreloadMap(db.Statement.Schema, preloads)

	// rearrange keys for consistent iteration
	keys := make([]string, 0, len(preloadMap))
	for k := range preloadMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	isJoined := func(name string) (bool, []string) {
		for _, join := range associations {
			if _, ok := (*relsMap).Relations[join]; ok && name == join {
				return true, nil
			}
			joinParts := strings.SplitN(join, ".", 2)
			if len(joinParts) == 2 {
				if _, ok := (*relsMap).Relations[joinParts[0]]; ok && name == joinParts[0] {
					return true, []string{joinParts[1]}
				}
			}
		}
		return false, nil
	}

	for _, key := range keys {
		relation := (*relsMap).EmbeddedRelations[key]
		if relation != nil {
			preloadMapKey, ok := preloadMap[key]
			if !ok {
				continue
			}
			if err := preprocessEntry(db, associations, relations, preloadMapKey, conds); err != nil {
				return err
			}
		} else if rel := (*relsMap).Relations[key]; rel != nil {
			isJoinedVal, nestedJoins := isJoined(key)
			if isJoinedVal {
				valueKind := db.Statement.ReflectValue.Kind()
				switch valueKind {
				case reflect.Slice, reflect.Array:
					if db.Statement.ReflectValue.Len() > 0 {
						valType := rel.FieldSchema.Elem().Elem()
						values := valType.NewSlice(db.Statement.ReflectValue.Len())
						for i := 0; i < db.Statement.ReflectValue.Len(); i++ {
							valRef := rel.Field.ReflectValueOf(db.Statement.Context, db.Statement.ReflectValue.Index(i))
							if valRef.Kind() != reflect.Ptr {
								values = reflect.Append(values, valRef.Addr())
							} else if !valRef.IsNil() {
								values = reflect.Append(values, valRef)
							}
						}

						tx := preloadDB(db, values, values.Interface())
						if err := preprocessEntry(tx, nestedJoins, relsMap, preloadMap[key], conds); err != nil {
							return err
						}
					}
				case reflect.Struct, reflect.Pointer:
					valueRef := rel.Field.ReflectValueOf(db.Statement.Context, db.Statement.ReflectValue)
					tx := preloadDB(db, valueRef, valueRef.Interface())
					if err := preprocessEntry(tx, nestedJoins, relsMap, preloadMap[key], conds); err != nil {
						return err
					}
				default:
					return gorm.ErrInvalidData
				}
			} else {
				sessionCtx := &gorm.Session{Context: db.Statement.Context, SkipHooks: db.Statement.SkipHooks}
				tx := db.Table("").Session(sessionCtx)
				tx.Statement.ReflectValue = db.Statement.ReflectValue
				tx.Statement.Unscoped = db.Statement.Unscoped
				if err := preload(tx, rel, append(preloads[key], conds...), preloadMap[key]); err != nil {
					return err
				}
			}
		} else {
			return fmt.Errorf("%s: %w for schema %s", key, gorm.ErrUnsupportedRelation, db.Statement.Schema.Name)
		}
	}
	return nil
}

func parsePreloadMap(schema string, preloads map[string][]interface{}) map[string]interface{} {
	// Implementation remains the same as in original function
	return make(map[string]interface{})
}

func preloadDB(db *gorm.DB, reflectValue reflect.Value, value interface{}) *gorm.DB {
	// Implementation remains the same as in original function
	return db
}

func preload(tx *gorm.DB, rel *schema.Relationship, conds []interface{}, preloads map[string][]interface{}) error {
	// Implementation remains the same as in original function
	return nil
}

func file_grpc_testing_empty_proto_init() {
	if File_grpc_testing_empty_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_grpc_testing_empty_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   1,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_grpc_testing_empty_proto_goTypes,
		DependencyIndexes: file_grpc_testing_empty_proto_depIdxs,
		MessageInfos:      file_grpc_testing_empty_proto_msgTypes,
	}.Build()
	File_grpc_testing_empty_proto = out.File
	file_grpc_testing_empty_proto_rawDesc = nil
	file_grpc_testing_empty_proto_goTypes = nil
	file_grpc_testing_empty_proto_depIdxs = nil
}

func verifyRoundRobinCalls(ctx context.Context, client testgrpc.TestServiceClient, locations []resolver.Location) error {
	expectedLocationCount := make(map[string]int)
	for _, loc := range locations {
		expectedLocationCount[loc.Addr]++
	}
	actualLocationCount := make(map[string]int)
	for ; ctx.Err() == nil; <-time.After(time.Millisecond) {
		actualLocationCount = make(map[string]int)
		// Execute 3 cycles.
		var rounds [][]string
		for i := 0; i < 3; i++ {
			round := make([]string, len(locations))
			for c := 0; c < len(locations); c++ {
				var connection peer.Connection
				client.SimpleCall(ctx, &testpb.Empty{}, grpc.Peer(&connection))
				round[c] = connection.RemoteAddr().String()
			}
			rounds = append(rounds, round)
		}
		// Confirm the first cycle contains all addresses in locations.
		for _, loc := range rounds[0] {
			actualLocationCount[loc]++
		}
		if !cmp.Equal(actualLocationCount, expectedLocationCount) {
			continue
		}
		// Ensure all three cycles contain the same addresses.
		if !cmp.Equal(rounds[0], rounds[1]) || !cmp.Equal(rounds[0], rounds[2]) {
			continue
		}
		return nil
	}
	return fmt.Errorf("timeout while awaiting roundrobin allocation of calls among locations: %v; observed: %v", locations, actualLocationCount)
}

func TestTraceClient(t *testing.T) {
	tracer := mocktracer.New()

	// Empty/background context.
	tracedEndpoint := kitot.TraceClient(tracer, "testOp")(endpoint.Nop)
	if _, err := tracedEndpoint(context.Background(), struct{}{}); err != nil {
		t.Fatal(err)
	}

	// tracedEndpoint created a new Span.
	finishedSpans := tracer.FinishedSpans()
	if want, have := 1, len(finishedSpans); want != have {
		t.Fatalf("Want %v span(s), found %v", want, have)
	}

	span := finishedSpans[0]

	if want, have := "testOp", span.OperationName; want != have {
		t.Fatalf("Want %q, have %q", want, have)
	}

	if want, have := map[string]interface{}{
		otext.SpanKindRPCClient.Key: otext.SpanKindRPCClient.Value,
	}, span.Tags(); fmt.Sprint(want) != fmt.Sprint(have) {
		t.Fatalf("Want %q, have %q", want, have)
	}
}

