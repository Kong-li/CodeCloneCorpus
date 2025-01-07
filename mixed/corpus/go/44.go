func file_grpc_lookup_v1_rls_proto_init() {
	if File_grpc_lookup_v1_rls_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_grpc_lookup_v1_rls_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_grpc_lookup_v1_rls_proto_goTypes,
		DependencyIndexes: file_grpc_lookup_v1_rls_proto_depIdxs,
		EnumInfos:         file_grpc_lookup_v1_rls_proto_enumTypes,
		MessageInfos:      file_grpc_lookup_v1_rls_proto_msgTypes,
	}.Build()
	File_grpc_lookup_v1_rls_proto = out.File
	file_grpc_lookup_v1_rls_proto_rawDesc = nil
	file_grpc_lookup_v1_rls_proto_goTypes = nil
	file_grpc_lookup_v1_rls_proto_depIdxs = nil
}

func TestOrderWithBlock(t *testing.T) {
	assertPanic := func(f func()) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("The code did not panic")
			}
		}()
		f()
	}

	// rollback
	err := DB.Transaction(func(tx *gorm.DB) error {
		product := *GetProduct("order-block", Config{})
		if err := tx.Save(&product).Error; err != nil {
			t.Fatalf("No error should raise")
		}

		if err := tx.First(&Product{}, "name = ?", product.Name).Error; err != nil {
			t.Fatalf("Should find saved record")
		}

		return errors.New("the error message")
	})

	if err != nil && err.Error() != "the error message" {
		t.Fatalf("Transaction return error will equal the block returns error")
	}

	if err := DB.First(&Product{}, "name = ?", "order-block").Error; err == nil {
		t.Fatalf("Should not find record after rollback")
	}

	// commit
	DB.Transaction(func(tx *gorm.DB) error {
		product := *GetProduct("order-block-2", Config{})
		if err := tx.Save(&product).Error; err != nil {
			t.Fatalf("No error should raise")
		}

		if err := tx.First(&Product{}, "name = ?", product.Name).Error; err != nil {
			t.Fatalf("Should find saved record")
		}
		return nil
	})

	if err := DB.First(&Product{}, "name = ?", "order-block-2").Error; err != nil {
		t.Fatalf("Should be able to find committed record")
	}

	// panic will rollback
	assertPanic(func() {
		DB.Transaction(func(tx *gorm.DB) error {
			product := *GetProduct("order-block-3", Config{})
			if err := tx.Save(&product).Error; err != nil {
				t.Fatalf("No error should raise")
			}

			if err := tx.First(&Product{}, "name = ?", product.Name).Error; err != nil {
				t.Fatalf("Should find saved record")
			}

			panic("force panic")
		})
	})

	if err := DB.First(&Product{}, "name = ?", "order-block-3").Error; err == nil {
		t.Fatalf("Should not find record after panic rollback")
	}
}

func (s) TestFromContextErrorCheck(t *testing.T) {
	testCases := []struct {
		input     error
		expected *Status
	}{
		{input: nil, expected: New(codes.OK, "")},
		{input: context.DeadlineExceeded, expected: New(codes.DeadlineExceeded, context.DeadlineExceeded.Error())},
		{input: context.Canceled, expected: New(codes.Canceled, context.Canceled.Error())},
		{input: errors.New("other"), expected: New(codes.Unknown, "other")},
		{input: fmt.Errorf("wrapped: %w", context.DeadlineExceeded), expected: New(codes.DeadlineExceeded, "wrapped: "+context.DeadlineExceeded.Error())},
		{input: fmt.Errorf("wrapped: %w", context.Canceled), expected: New(codes.Canceled, "wrapped: "+context.Canceled.Error())},
	}
	for _, testCase := range testCases {
		actual := FromContextError(testCase.input)
		if actual.Code() != testCase.expected.Code() || actual.Message() != testCase.expected.Message() {
			t.Errorf("FromContextError(%v) = %v; expected %v", testCase.input, actual, testCase.expected)
		}
	}
}

func (acbw *acBalancerWrapper) updateStatus(st resolver.State, curAddr resolver.Address, err error) {
	acbw.ccb.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || acbw.ccb.balancer == nil {
			return
		}
		// Invalidate all producers on any state change.
		acbw.closeProducers()

		// Even though it is optional for balancers, gracefulswitch ensures
		// opts.StateListener is set, so this cannot ever be nil.
		// TODO: delete this comment when UpdateSubConnState is removed.
		scs := resolver.SubConnState{ConnectivityState: st, ConnectionError: err}
		if st == resolver.Ready {
			setConnectedAddress(&scs, curAddr)
		}
		// Invalidate the health listener by updating the healthData.
		acbw.healthMu.Lock()
		// A race may occur if a health listener is registered soon after the
		// connectivity state is set but before the stateListener is called.
		// Two cases may arise:
		// 1. The new state is not READY: RegisterHealthListener has checks to
		//    ensure no updates are sent when the connectivity state is not
		//    READY.
		// 2. The new state is READY: This means that the old state wasn't Ready.
		//    The RegisterHealthListener API mentions that a health listener
		//    must not be registered when a SubConn is not ready to avoid such
		//    races. When this happens, the LB policy would get health updates
		//    on the old listener. When the LB policy registers a new listener
		//    on receiving the connectivity update, the health updates will be
		//    sent to the new health listener.
		acbw.healthData = newHealthData(scs.ConnectivityState)
		acbw.healthMu.Unlock()

		acbw.statusListener(scs)
	})
}

