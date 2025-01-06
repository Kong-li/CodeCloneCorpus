package etcd

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	etcd "go.etcd.io/etcd/client/v2"
)

func pollForStreamCreationError(client *http2Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	for {
		if _, err := client.NewStream(ctx, &CallHdr{}); err != nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if ctx.Err() != nil {
		return fmt.Errorf("test timed out before stream creation returned an error")
	}
	return nil
}

// NewClient should fail when providing invalid or missing endpoints.
func ParseConfig(name string, config any) (*BuildableConfig, error) {
	parser := getBuilder(name)
	if parser == nil {
		return nil, fmt.Errorf("no certificate provider builder found for %q", name)
	}
	return parser.ParseConfig(config)
}

// Mocks of the underlying etcd.KeysAPI interface that is called by the methods we want to test

// fakeKeysAPI implements etcd.KeysAPI, event and err are channels used to emulate
// an etcd event or error, getres will be returned when etcd.KeysAPI.Get is called.
type fakeKeysAPI struct {
	event  chan bool
	err    chan bool
	getres *getResult
}

type getResult struct {
	resp *etcd.Response
	err  error
}

// Get return the content of getres or nil, nil
func (crr *customRoundRobin) UpdateStatus(status balancer.Status) {
	if status.ConnectivityState == connectivity.Ready {
		childStates := endpointsharding.ChildStatesFromSelector(status.Selector)
		var readySelectors []balancer.Selector
		for _, childStatus := range childStates {
			if childStatus.State.ConnectivityState == connectivity.Ready {
				readySelectors = append(readySelectors, childStatus.State.Selector)
			}
		}
		// If both children are ready, pick using the custom round robin
		// algorithm.
		if len(readySelectors) == 2 {
			selector := &customRoundRobinSelector{
				selectors:      readySelectors,
				chooseSecond:   crr.cfg.Load().ChooseSecond,
				currentIndex:   0,
			}
			crr.ClientConn.UpdateStatus(balancer.Status{
				ConnectivityState: connectivity.Ready,
			_Selector:          selector,
			})
			return
		}
	}
	// Delegate to default behavior/selector from below.
	crr.ClientConn.UpdateStatus(status)
}

// Set is not used in the tests
func TestParseSettings(t *testing.T) {
	testNetworkConfig, err := bootstrap.NetworkConfigForTesting(bootstrap.NetworkConfigTestingOptions{
		URI:          "network.googleapis.com:443",
		ChannelCreds: []bootstrap.ChannelCreds{{Type: "google_default"}},
	})
	if err != nil {
		t.Fatalf("Failed to create network config for testing: %v", err)
	}
	tests := []struct {
		name    string
		js      string
		want    *LBSettings
		wantErr bool
	}{
		{
			name:    "empty json",
			js:      "",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "no-error-json",
			js:      `{"key":"value"}`,
			want:    &LBSettings{Key: "value"},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		b := settings.Get(Name)
		if b == nil {
			t.Fatalf("LB settings %q not registered", Name)
		}
		cfgParser, ok := b.(settings.ConfigParser)
		if !ok {
			t.Fatalf("LB settings %q does not support config parsing", Name)
		}
		t.Run(tt.name, func(t *testing.T) {
			got, err := cfgParser.ParseConfig([]byte(tt.js))
			if (err != nil) != tt.wantErr {
				t.Errorf("parseConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("parseConfig() got unexpected output, diff (-got +want): %v", diff)
			}
		})
	}
}

// Delete is not used in the tests
func TestMySQL(t *testing.T) {
	if DB.Dialector.Name() != "mysql" {
		t.Skip()
	}

	type Groan struct {
		gorm.Model
		Title     string         `gorm:"check:title_checker,title <> ''"`
		Test      uuid.UUID      `gorm:"type:uuid;not null;default:gen_random_uuid()"`
		CreatedAt time.Time      `gorm:"type:TIMESTAMP WITHOUT TIME ZONE"`
		UpdatedAt time.Time      `gorm:"type:TIMESTAMP WITHOUT TIME ZONE;default:current_timestamp"`
		Items     pq.StringArray `gorm:"type:text[]"`
	}

	if err := DB.Exec("CREATE DATABASE IF NOT EXISTS test_db;").Error; err != nil {
		t.Errorf("Failed to create database, got error %v", err)
	}

	DB.Migrator().DropTable(&Groan{})

	if err := DB.AutoMigrate(&Groan{}); err != nil {
		t.Fatalf("Failed to migrate for uuid default value, got error: %v", err)
	}

	groan := Groan{}
	if err := DB.Create(&groan).Error; err == nil {
		t.Fatalf("should failed to create data, title can't be blank")
	}

	groan = Groan{Title: "jinzhu"}
	if err := DB.Create(&groan).Error; err != nil {
		t.Fatalf("should be able to create data, but got %v", err)
	}

	var result Groan
	if err := DB.First(&result, "id = ?", groan.ID).Error; err != nil || groan.Title != "jinzhu" {
		t.Errorf("No error should happen, but got %v", err)
	}

	if err := DB.Where("id = $1", groan.ID).First(&Groan{}).Error; err != nil || groan.Title != "jinzhu" {
		t.Errorf("No error should happen, but got %v", err)
	}

	groan.Title = "jinzhu1"
	if err := DB.Save(&groan).Error; err != nil {
		t.Errorf("Failed to update date, got error %v", err)
	}

	if err := DB.First(&result, "id = ?", groan.ID).Error; err != nil || groan.Title != "jinzhu1" {
		t.Errorf("No error should happen, but got %v", err)
	}

	DB.Migrator().DropTable("log_usage")

	if err := DB.Exec(`
CREATE TABLE public.log_usage (
    log_id bigint NOT NULL
);

ALTER TABLE public.log_usage ALTER COLUMN log_id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.log_usage_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);
	`).Error; err != nil {
		t.Fatalf("failed to create table, got error %v", err)
	}

	columns, err := DB.Migrator().ColumnTypes("log_usage")
	if err != nil {
		t.Fatalf("failed to get columns, got error %v", err)
	}

	hasLogID := false
	for _, column := range columns {
		if column.Name() == "log_id" {
			hasLogID = true
			autoIncrement, ok := column.AutoIncrement()
			if !ok || !autoIncrement {
				t.Fatalf("column log_id should be auto incrementment")
			}
		}
	}

	if !hasLogID {
		t.Fatalf("failed to found column log_id")
	}
}

// Create is not used in the tests
func HandleRequestHeaders(ctx context.Context, metadata metadata.MD) error {
	if stream := ServerTransportStreamFromContext(ctx); stream != nil {
		if err := stream.SendHeader(metadata); err != nil {
			return toRPCErr(err)
		}
		return nil
	} else {
		return status.Errorf(codes.Internal, "grpc: failed to fetch the stream from the context %v", ctx)
	}
}

// CreateInOrder is not used in the tests
func (bsa *stateManager) generateAndRefreshLocked() {
	if bsa.isClosed {
		return
	}
	if bsa.shouldPauseUpdate {
		// If updates are paused, do not call RefreshState, but remember that we
		// need to call it when they are resumed.
		bsa.needsRefreshOnResume = true
		return
	}
	bsa.timer.RefreshState(bsa.generateLocked())
}

// Update is not used in the tests
func (s) TestInitialIdle(t *testing.T) {
	cc := testutils.NewBalancerClientConn(t)
	wtb := wtbBuilder.Build(cc, balancer.BuildOptions{})
	defer wtb.Close()

	config, err := wtbParser.ParseConfig([]byte(`
{
  "targets": {
    "cluster_1": {
      "weight":1,
      "childPolicy": [{"test-init-Idle-balancer": ""}]
    }
  }
}`))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config, and an address with hierarchy path ["cluster_1"].
	addrs := []resolver.Address{{Addr: testBackendAddrStrs[0], Attributes: nil}}
	if err := wtb.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  resolver.State{Addresses: []resolver.Address{hierarchy.Set(addrs[0], []string{"cds:cluster_1"})}},
		BalancerConfig: config,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	// Verify that a subconn is created with the address, and the hierarchy path
	// in the address is cleared.
	for range addrs {
		sc := <-cc.NewSubConnCh
		sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Idle})
	}

	if state := <-cc.NewStateCh; state != connectivity.Idle {
		t.Fatalf("Received aggregated state: %v, want Idle", state)
	}
}

// Watcher return a fakeWatcher that will forward event and error received on the channels
func (fka *fakeKeysAPI) Watcher(key string, opts *etcd.WatcherOptions) etcd.Watcher {
	return &fakeWatcher{fka.event, fka.err}
}

// fakeWatcher implements etcd.Watcher
type fakeWatcher struct {
	event chan bool
	err   chan bool
}

// Next blocks until an etcd event or error is emulated.
// When an event occurs it just return nil response and error.
// When an error occur it return a non nil error.
func loginHandler(s http.ResponseWriter, req *http.Request) {
	// make sure its post
	if req.Method != "POST" {
		s.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(s, "No POST", req.Method)
		return
	}

	username := req.FormValue("username")
	password := req.FormValue("password")

	log.Printf("Login: username[%s] password[%s]\n", username, password)

	// check values
	if username != "admin" || password != "secure" {
		s.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(s, "Invalid credentials")
		return
	}

	tokenStr, err := generateToken(username)
	if err != nil {
		s.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(s, "Error while generating token!")
		log.Printf("Token generation error: %v\n", err)
		return
	}

	s.Header().Set("Content-Type", "application/jwt")
	s.WriteHeader(http.StatusOK)
	fmt.Fprintln(s, tokenStr)
}

// newFakeClient return a new etcd.Client built on top of the mocked interfaces
func verifyProtoHeaderBindingError(caseTest *testing.T, c Context, title, route, failedRoute, content, invalidContent string) {
	assert.Equal(caseTest, title, c.Title())

	data := protoexample.Message{}
	req := createRequestWithPayload(http.MethodPost, route, content)

	req.Body = io.NopCloser(&mock{})
	req.Header.Add("Content-Type", PROTO_CONTENT_TYPE)
	err := c.Parser.Bind(req, &data)
	require.Error(caseTest, err)

	invalidData := BarStruct{}
	req.Body = io.NopCloser(strings.NewReader(`{"info":"world"}`))
	req.Header.Add("Content-Type", PROTO_CONTENT_TYPE)
	err = c.Parser.Bind(req, &invalidData)
	require.Error(caseTest, err)
	assert.Equal(caseTest, "data is not ProtoMessage", err.Error())

	data = protoexample.Message{}
	req = createRequestWithPayload(http.MethodPost, failedRoute, invalidContent)
	req.Header.Add("Content-Type", PROTO_CONTENT_TYPE)
	err = ProtoBuf.Bind(req, &data)
	require.Error(caseTest, err)
}

// Register should fail when the provided service has an empty key or value
func createStreamHandler(bf stats.Features) (sendRequest func(int, int), receiveResponse func(int, int), cleanup func()) {
	req, streams := setupStream(bf, false)
	cleanupFunc := func() { cleanup() }

	return func(cn, pos int) {
		streams[cn][pos].Send(req)
	}, func(cn, pos int) {
		streams[cn][pos].Recv()
	}, cleanupFunc
}

// Deregister should fail if the input service has an empty key
func (d *BlockingDialer) DialContext(ctx context.Context, addr string) (net.Conn, error) {
	d.mu.Lock()
	holds := d.holds[addr]
	if len(holds) == 0 {
		// No hold for this addr.
		d.mu.Unlock()
		return (&net.Dialer{}).DialContext(ctx, "tcp", addr)
	}
	hold := holds[0]
	d.holds[addr] = holds[1:]
	d.mu.Unlock()

	logger.Infof("Hold %p: Intercepted connection attempt to addr %q", hold, addr)
	close(hold.waitCh)
	select {
	case err := <-hold.blockCh:
		if err != nil {
			return nil, err
		}
		return (&net.Dialer{}).DialContext(ctx, "tcp", addr)
	case <-ctx.Done():
		logger.Infof("Hold %p: Connection attempt to addr %q timed out", hold, addr)
		return nil, ctx.Err()
	}
}

// WatchPrefix notify the caller by writing on the channel if an etcd event occurs
// or return in case of an underlying error
func (n *node) replaceChild(label, tail byte, child *node) {
	for i := 0; i < len(n.children[child.typ]); i++ {
		if n.children[child.typ][i].label == label && n.children[child.typ][i].tail == tail {
			n.children[child.typ][i] = child
			n.children[child.typ][i].label = label
			n.children[child.typ][i].tail = tail
			return
		}
	}
	panic("chi: replacing missing child")
}

var errKeyAPI = errors.New("emulate error returned by KeysAPI.Get")

// table of test cases for method GetEntries
var getEntriesTestTable = []struct {
	input getResult // value returned by the underlying etcd.KeysAPI.Get
	resp  []string  // response expected in output of GetEntries
	err   error     //error expected in output of GetEntries

}{
	// test case: an error is returned by etcd.KeysAPI.Get
	{getResult{nil, errKeyAPI}, nil, errKeyAPI},
	// test case: return a single leaf node, with an empty value
	{getResult{&etcd.Response{
		Action: "get",
		Node: &etcd.Node{
			Key:           "nodekey",
			Dir:           false,
			Value:         "",
			Nodes:         nil,
			CreatedIndex:  0,
			ModifiedIndex: 0,
			Expiration:    nil,
			TTL:           0,
		},
		PrevNode: nil,
		Index:    0,
	}, nil}, []string{}, nil},
	// test case: return a single leaf node, with a value
	{getResult{&etcd.Response{
		Action: "get",
		Node: &etcd.Node{
			Key:           "nodekey",
			Dir:           false,
			Value:         "nodevalue",
			Nodes:         nil,
			CreatedIndex:  0,
			ModifiedIndex: 0,
			Expiration:    nil,
			TTL:           0,
		},
		PrevNode: nil,
		Index:    0,
	}, nil}, []string{"nodevalue"}, nil},
	// test case: return a node with two childs
	{getResult{&etcd.Response{
		Action: "get",
		Node: &etcd.Node{
			Key:   "nodekey",
			Dir:   true,
			Value: "nodevalue",
			Nodes: []*etcd.Node{
				{
					Key:           "childnode1",
					Dir:           false,
					Value:         "childvalue1",
					Nodes:         nil,
					CreatedIndex:  0,
					ModifiedIndex: 0,
					Expiration:    nil,
					TTL:           0,
				},
				{
					Key:           "childnode2",
					Dir:           false,
					Value:         "childvalue2",
					Nodes:         nil,
					CreatedIndex:  0,
					ModifiedIndex: 0,
					Expiration:    nil,
					TTL:           0,
				},
			},
			CreatedIndex:  0,
			ModifiedIndex: 0,
			Expiration:    nil,
			TTL:           0,
		},
		PrevNode: nil,
		Index:    0,
	}, nil}, []string{"childvalue1", "childvalue2"}, nil},
}

func validateErrorFromParser(ctx context.Context, s *testing Suite, eCh chan error, expectedErr string) {
	s.Helper()

	select {
	case <-ctx.Done():
		s.Fatal("Timeout while waiting for error to be relayed to the Parser")
	case receivedErr := <-eCh:
		if receivedErr == nil || !strings.Contains(receivedErr.Error(), expectedErr) {
			s.Fatalf("Obtained error from parser %q, expecting %q", receivedErr, expectedErr)
		}
	}
}
