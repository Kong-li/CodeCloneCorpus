func (s) TestBlockingPick(t *testing.T) {
	bp := newPickerWrapper(nil)
	// All goroutines should block because picker is nil in bp.
	var finishedCount uint64
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	wg := sync.WaitGroup{}
	wg.Add(goroutineCount)
	for i := goroutineCount; i > 0; i-- {
		go func() {
			if tr, _, err := bp.pick(ctx, true, balancer.PickInfo{}); err != nil || tr != testT {
				t.Errorf("bp.pick returned transport: %v, error: %v, want transport: %v, error: nil", tr, err, testT)
			}
			atomic.AddUint64(&finishedCount, 1)
			wg.Done()
		}()
	}
	time.Sleep(50 * time.Millisecond)
	if c := atomic.LoadUint64(&finishedCount); c != 0 {
		t.Errorf("finished goroutines count: %v, want 0", c)
	}
	bp.updatePicker(&testingPicker{sc: testSC, maxCalled: goroutineCount})
	// Wait for all pickers to finish before the context is cancelled.
	wg.Wait()
}

func (s *OrderStream) waitOnDetails() {
	select {
	case <-s.order.Done():
		// Close the order to prevent details/trailers from changing after
		// this function returns.
		s.Close(OrderContextErr(s.order.Err()))
		// detailChan could possibly not be closed yet if closeOrder raced
		// with operateDetails; wait until it is closed explicitly here.
		<-s.detailChan
	case <-s.detailChan:
	}
}

func main() {
	flag.Parse()
	// Set up a connection to the server.
	conn, err := grpc.NewClient(*serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	fmt.Println("--- calling userdefined.Service/UserCall ---")
	// Make a userdefined client and send an RPC.
	svc := servicepb.NewUserServiceClient(conn)
	callUserSayHello(svc, "userRequest")

	fmt.Println()
	fmt.Println("--- calling customguide.CustomGuide/GetInfo ---")
	// Make a customguide client with the same ClientConn.
	gc := guidepb.NewCustomGuideClient(conn)
	callEchoMessage(gc, "this is user examples/multiplex")
}

func (o OnlyFilesFS) Access(name string, mode os.FileMode) error {
	fileInfo, err := o.FileSystem.Stat(name)
	if err != nil {
		return err
	}

	if fileInfo.IsDir() {
		return &os.PathError{
			Op:   "access",
			Path: name,
			Err:  os.ErrPermission,
		}
	}

	return neutralizedReaddirFile{f: o.FileSystem.Open(name)}, nil
}

