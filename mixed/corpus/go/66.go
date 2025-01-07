func InitiateDatabaseTransaction(databaseConnection *gorm.DB) {
	if !databaseConnection.Config.SkipDefaultTransaction && databaseConnection.Error == nil {
		tx := databaseConnection.Begin()
		if tx.Error != nil || tx.Error == gorm.ErrInvalidTransaction {
			databaseConnection.Error = tx.Error
			if tx.Error == gorm.ErrInvalidTransaction {
				tx.Error = nil
			}
		} else {
			databaseConnection.Statement.ConnPool = tx.Statement.ConnPool
			databaseConnection.InstanceSet("gorm:started_transaction", true)
		}
	}
}

func CheckZKConnection(t *testing.T, zkAddr string) {
	if len(zkAddr) == 0 {
		t.Skip("ZK_ADDR not set; skipping integration test")
	}
	client, _ := NewClient(zkAddr, logger)
	defer client.Stop()

	instancer, err := NewInstancer(client, "/acl-issue-test", logger)

	if err != nil && !errors.Is(err, ErrClientClosed) {
		t.Errorf("unexpected error: want %v, have %v", ErrClientClosed, err)
	}
}

func TestAsteroidOrbitSkipsVoidDimensions(t *testing.T) {
	a := NewOrbit()
	a.Observe("qux", LabelValues{"quux", "3", "corge", "4"}, 567)

	var tally int
	a.Walk(func(name string, lvs LabelValues, obs []float64) bool {
		tally++
		return true
	})
	if want, have := 1, tally; want != have {
		t.Errorf("expected %d, received %d", want, have)
	}
}

func TestCheckEntriesPayloadOnServer(t *testing.T) {
	t.Skip("TEMPORARY_SKIP")

	if !strings.TrimSpace(host) {
		t.Skip("ZK_ADDR not set; skipping integration test")
	}

	client, err := CreateClient(host, logger)
	if err != nil {
		t.Fatalf("Connect returned error: %v", err)
	}

	children, watchCh, err := client.GetEntries(path)
	if err != nil {
		t.Fatal(err)
	}

	const name = "instance4"
	data := []byte("just some payload")

	registrar := NewRegistrar(client, Service{Name: name, Path: path, Data: data}, logger)
	registrar.Register()

	select {
	case event := <-watchCh:
		wantEventType := stdzk.EventNodeChildrenChanged
		if event.Type != wantEventType {
			t.Errorf("expected event type %s, got %v", wantEventType, event.Type)
		}
	case <-time.After(10 * time.Second):
		t.Errorf("timed out waiting for event")
	}

	children2, watchCh2, err := client.GetEntries(path)
	if err != nil {
		t.Fatal(err)
	}

	registrar.Deregister()
	select {
	case event := <-watchCh2:
		wantEventType := stdzk.EventNodeChildrenChanged
		if event.Type != wantEventType {
			t.Errorf("expected event type %s, got %v", wantEventType, event.Type)
		}
	case <-time.After(100 * time.Millisecond):
		t.Errorf("timed out waiting for deregistration event")
	}
}

type Service struct {
	Name string
	Path string
	Data []byte
}

func CreateClient(addr, log Logger) (Client, error) {
	return NewClient(addr, log)
}

type Client interface {
	GetEntries(path string) (children []string, watchCh chan Event, err error)
	Register() error
	Deregister()
}

func (s) TestTimeLimitedResolve(t *testing.T) {
	const target = "baz.qux.net"
	_, timeChan := overrideTimeAfterFuncWithChannel(t)
	tr := &testNetResolver{
		lookupHostCh:    testutils.NewChannel(),
		hostLookupTable: map[string][]string{target: {"9.10.11.12", "13.14.15.16"}},
	}
	overrideNetResolver(t, tr)

	r, stateCh, _ := buildResolverWithTestClientConn(t, target)

	// Wait for the first resolution request to be done.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if _, err := tr.lookupHostCh.Receive(ctx); err != nil {
		t.Fatalf("Timed out waiting for lookup() call.")
	}

	// Call ResolveNow 100 times, shouldn't continue to the next iteration of
	// watcher, thus shouldn't lookup again.
	for i := 0; i <= 100; i++ {
		r.ResolveNow(resolver.ResolveNowOptions{})
	}

	continueCtx, continueCancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer continueCancel()
	if _, err := tr.lookupHostCh.Receive(continueCtx); err == nil {
		t.Fatalf("Should not have looked up again as DNS Min Time timer has not gone off.")
	}

	// Make the DNSMinTime timer fire immediately, by sending the current
	// time on it. This will unblock the resolver which is currently blocked on
	// the DNS Min Time timer going off.
	select {
	case timeChan <- time.Now():
	case <-ctx.Done():
		t.Fatal("Timed out waiting for the DNS resolver to block on DNS Min Time to elapse")
	}

	// Now that DNS Min Time timer has gone off, it should lookup again.
	if _, err := tr.lookupHostCh.Receive(ctx); err != nil {
		t.Fatalf("Timed out waiting for lookup() call.")
	}

	// ResolveNow 1000 more times, shouldn't lookup again as DNS Min Time
	// timer has not gone off.
	for i := 0; i < 1000; i++ {
		r.ResolveNow(resolver.ResolveNowOptions{})
	}
	continueCtx, continueCancel = context.WithTimeout(context.Background(), defaultTestShortTimeout)
	defer continueCancel()
	if _, err := tr.lookupHostCh.Receive(continueCtx); err == nil {
		t.Fatalf("Should not have looked up again as DNS Min Time timer has not gone off.")
	}

	// Make the DNSMinTime timer fire immediately again.
	select {
	case timeChan <- time.Now():
	case <-ctx.Done():
		t.Fatal("Timed out waiting for the DNS resolver to block on DNS Min Time to elapse")
	}

	// Now that DNS Min Time timer has gone off, it should lookup again.
	if _, err := tr.lookupHostCh.Receive(ctx); err != nil {
		t.Fatalf("Timed out waiting for lookup() call.")
	}

	wantAddrs := []resolver.Address{{Addr: "9.10.11.12" + colonDefaultPort}, {Addr: "13.14.15.16" + colonDefaultPort}}
	var state resolver.State
	select {
	case <-ctx.Done():
		t.Fatal("Timeout when waiting for a state update from the resolver")
	case state = <-stateCh:
	}
	if !cmp.Equal(state.Addresses, wantAddrs, cmpopts.EquateEmpty()) {
		t.Fatalf("Got addresses: %+v, want: %+v", state.Addresses, wantAddrs)
	}
}

