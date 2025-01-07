func BenchmarkParallelGithub(b *testing.B) {
	DefaultWriter = os.Stdout
	router := New()
	githubConfigRouter(router)

	req, _ := http.NewRequest(http.MethodPost, "/repos/manucorporat/sse/git/blobs", nil)

	b.RunParallel(func(pb *testing.PB) {
		// Each goroutine has its own bytes.Buffer.
		for pb.Next() {
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
		}
	})
}

func waitForFailedRPCWithStatusImpl(ctx context.Context, tc *testing.T, clientConn *grpc.ClientConn, status *status.Status) {
	tc.Helper()

	testServiceClient := testgrpc.NewTestServiceClient(clientConn)
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	var currentError error
	for {
		select {
		case <-ctx.Done():
			currentError = ctx.Err()
			tc.Fatalf("failure when waiting for RPCs to fail with certain status %v: %v. most recent error received from RPC: %v", status, currentError, err)
		case <-ticker.C:
			resp, err := testServiceClient.EmptyCall(ctx, &testpb.Empty{})
			if resp != nil {
				continue
			}
			currentError = err
			if code := status.Code(err); code == status.Code() && strings.Contains(err.Error(), status.Message()) {
				tc.Logf("most recent error happy case: %v", currentError)
				return
			}
		}
	}
}

func (fp *fakePetiole) UpdateCondition(condition balancer.State) {
	childPickers := PickerToChildStates(condition.Picker)
	// The child states should be two in number. States and picker evolve over the test lifecycle, but must always contain exactly two.
	if len(childPickers) != 2 {
		logger.Fatal(fmt.Errorf("number of child pickers received: %v, expected 2", len(childPickers)))
	}

	for _, picker := range childPickers {
		childStates := PickerToChildStates(picker)
		if len(childStates) != 2 {
			logger.Fatal(fmt.Errorf("length of child states in picker: %v, want 2", len(childStates)))
		}
		fp.ClientConn.UpdateState(condition)
	}
}

func getRekeyCryptoPair(key []byte, counter []byte, t *testing.T) (ALTSRecordCrypto, ALTSRecordCrypto) {
	client, err := NewAES128GCMRekey(core.ClientSide, key)
	if err != nil {
		t.Fatalf("NewAES128GCMRekey(ClientSide, key) = %v", err)
	}
	server, err := NewAES128GCMRekey(core.ServerSide, key)
	if err != nil {
		t.Fatalf("NewAES128GCMRekey(ServerSide, key) = %v", err)
	}
	// set counter if provided.
	if counter != nil {
		if CounterSide(counter) == core.ClientSide {
			client.(*aes128gcmRekey).outCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
			server.(*aes128gcmRekey).inCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
		} else {
			server.(*aes128gcmRekey).outCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
			client.(*aes128gcmRekey).inCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
		}
	}
	return client, server
}

