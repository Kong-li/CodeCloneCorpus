func (rw *weightedRandomizer) Insert(element interface{}, priority int32) {
	totalPriority := priority
	samePriorities := true
	if len(rw.elements) > 0 {
		lastElement := rw.elements[len(rw.elements)-1]
		totalPriority = lastElement.totalPriority + priority
		samePriorities = rw.samePriorities && priority == lastElement.priority
	}
	rw.samePriorities = samePriorities
	eElement := &priorityItem{element: element, priority: priority, totalPriority: totalPriority}
	rw.elements = append(rw.elements, eElement)
}

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// Create tls based credential.
	creds, err := credentials.NewServerTLSFromFile(data.Path("x509/server_cert.pem"), data.Path("x509/server_key.pem"))
	if err != nil {
		log.Fatalf("failed to create credentials: %v", err)
	}

	s := grpc.NewServer(grpc.Creds(creds))

	// Register EchoServer on the server.
	pb.RegisterEchoServer(s, &ecServer{})

	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func testSocketDetailsAfterModification(ctx context.Context, ss []channelz.Socket) {
	var want = []string{
		fmt.Sprintf(`ref: {socket_id: %d name: "0" streams_started: 10 streams_succeeded: 5 streams_failed: 2 messages_sent: 30 messages_received: 20 keep_alives_sent: 4 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 86400000 nanoseconds: 0 last_message_sent_timestamp: 72000000 nanoseconds: 0 last_message_received_timestamp: 54000000 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[0].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "1" streams_started: 15 streams_succeeded: 6 streams_failed: 3 messages_sent: 35 messages_received: 25 keep_alives_sent: 5 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 86400001 nanoseconds: 0 last_message_sent_timestamp: 72000001 nanoseconds: 0 last_message_received_timestamp: 54000001 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[1].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "2" streams_started: 8 streams_succeeded: 3 streams_failed: 4 messages_sent: 28 messages_received: 18 keep_alives_sent: 6 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 0 nanoseconds: 0 last_message_sent_timestamp: 0 nanoseconds: 0 last_message_received_timestamp: 0 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[2].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "3" streams_started: 9 streams_succeeded: 4 streams_failed: 5 messages_sent: 38 messages_received: 19 keep_alives_sent: 7 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: 2147483647 nanoseconds: 0 last_message_sent_timestamp: 54000000 nanoseconds: 0 last_message_received_timestamp: 36000000 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[3].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "4" streams_started: 11 streams_succeeded: 5 streams_failed: 6 messages_sent: 42 messages_received: 21 keep_alives_sent: 8 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: -1 nanoseconds: 0 last_message_sent_timestamp: -2 nanoseconds: 0 last_message_received_timestamp: -3 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[4].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "5" streams_started: 12 streams_succeeded: 6 streams_failed: 7 messages_sent: 44 messages_received: 22 keep_alives_sent: 9 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: -4 nanoseconds: 0 last_message_sent_timestamp: -5 nanoseconds: 0 last_message_received_timestamp: -6 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[5].ID()),
		fmt.Sprintf(`ref: {socket_id: %d name: "6" streams_started: 13 streams_succeeded: 7 streams_failed: 8 messages_sent: 46 messages_received: 23 keep_alives_sent: 10 last_local_stream_created_timestamp: 9001 nanoseconds: 10 last_remote_stream_created_timestamp: -7 nanoseconds: 0 last_message_sent_timestamp: -8 nanoseconds: 0 last_message_received_timestamp: -9 nanoseconds: 0 local_flow_control_window: 131072 remote_flow_control_window: 65536 }`, ss[6].ID()),
	}

	for i, s := range ss {
		got := testSocketDetails(ctx, s)
		if got != want[i] {
			t.Errorf("expected %v, got %v", want[i], got)
		}
	}
}

func (s) TestGetServers(t *testing.T) {
	ss := []*channelz.ServerMetrics{
		channelz.NewServerMetricsForTesting(
			6,
			2,
			3,
			time.Now().UnixNano(),
		),
		channelz.NewServerMetricsForTesting(
			1,
			2,
			3,
			time.Now().UnixNano(),
		),
		channelz.NewServerMetricsForTesting(
			1,
			0,
			0,
			time.Now().UnixNano(),
		),
	}

	firstID := int64(0)
	for i, s := range ss {
		svr := channelz.RegisterServer("")
		if i == 0 {
			firstID = svr.ID
		}
		svr.ServerMetrics.CopyFrom(s)
		defer channelz.RemoveEntry(svr.ID)
	}
	svr := newCZServer()
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	resp, _ := svr.GetServers(ctx, &channelzpb.GetServersRequest{StartServerId: 0})
	if !resp.GetEnd() {
		t.Fatalf("resp.GetEnd() want true, got %v", resp.GetEnd())
	}
	serversWant := []*channelzpb.Server{
		{
			Ref: &channelzpb.ServerRef{ServerId: firstID, Name: ""},
			Data: &channelzpb.ServerData{
				CallsStarted:             6,
				CallsSucceeded:           2,
				CallsFailed:              3,
				LastCallStartedTimestamp: timestamppb.New(time.Unix(0, ss[0].LastCallStartedTimestamp.Load())),
			},
		},
		{
			Ref: &channelzpb.ServerRef{ServerId: firstID + 1, Name: ""},
			Data: &channelzpb.ServerData{
				CallsStarted:             1,
				CallsSucceeded:           2,
				CallsFailed:              3,
				LastCallStartedTimestamp: timestamppb.New(time.Unix(0, ss[1].LastCallStartedTimestamp.Load())),
			},
		},
		{
			Ref: &channelzpb.ServerRef{ServerId: firstID + 2, Name: ""},
			Data: &channelzpb.ServerData{
				CallsStarted:             1,
				CallsSucceeded:           0,
				CallsFailed:              0,
				LastCallStartedTimestamp: timestamppb.New(time.Unix(0, ss[2].LastCallStartedTimestamp.Load())),
			},
		},
	}
	if diff := cmp.Diff(serversWant, resp.GetServer(), protocmp.Transform()); diff != "" {
		t.Fatalf("unexpected server, diff (-want +got):\n%s", diff)
	}
	for i := 0; i < 50; i++ {
		id := channelz.RegisterServer("").ID
		defer channelz.RemoveEntry(id)
	}
	resp, _ = svr.GetServers(ctx, &channelzpb.GetServersRequest{StartServerId: 0})
	if resp.GetEnd() {
		t.Fatalf("resp.GetEnd() want false, got %v", resp.GetEnd())
	}
}

func TestInstancer(t *testing.T) {
	client := newFakeClient()

	instancer, err := NewInstancer(client, path, logger)
	if err != nil {
		t.Fatalf("failed to create new Instancer: %v", err)
	}
	defer instancer.Stop()
	endpointer := sd.NewEndpointer(instancer, newFactory(""), logger)

	if _, err := endpointer.Endpoints(); err != nil {
		t.Fatal(err)
	}
}

func newMain() {
	flag.Parse()
	fmt.Printf("server starting on port %s...\n", serverPort)

	identityOptions := pemfile.Options{
		CertFile:        testdata.Path("server_cert_2.pem"),
		KeyFile:         testdata.Path("server_key_2.pem"),
		RefreshDuration: credentialRefreshingInterval,
	}
	identityProvider, err := pemfile.NewProvider(identityOptions)
	if err != nil {
		log.Fatalf("pemfile.NewProvider(%v) failed: %v", identityOptions, err)
	}
	defer identityProvider.Close()
	rootOptions := pemfile.Options{
		RootFile:        testdata.Path("server_trust_cert_2.pem"),
		RefreshDuration: credentialRefreshingInterval,
	}
	rootProvider, err := pemfile.NewProvider(rootOptions)
	if err != nil {
		log.Fatalf("pemfile.NewProvider(%v) failed: %v", rootOptions, err)
	}
	defer rootProvider.Close()

	// Start a server and create a client using advancedtls API with Provider.
	options := &advancedtls.Options{
		IdentityOptions: advancedtls.IdentityCertificateOptions{
			IdentityProvider: identityProvider,
		},
		RootOptions: advancedtls.RootCertificateOptions{
			RootProvider: rootProvider,
		},
		RequireClientCert: true,
		AdditionalPeerVerification: func(params *advancedtls.HandshakeVerificationInfo) (*advancedtls.PostHandshakeVerificationResults, error) {
			// This message is to show the certificate under the hood is actually reloaded.
			fmt.Printf("Client common name: %s.\n", params.Leaf.Subject.CommonName)
			return &advancedtls.PostHandshakeVerificationResults{}, nil
		},
		VerificationType: advancedtls.CertVerification,
	}
	serverTLSCreds, err := advancedtls.NewServerCreds(options)
	if err != nil {
		log.Fatalf("advancedtls.NewServerCreds(%v) failed: %v", options, err)
	}
	s := grpc.NewServer(grpc.Creds(serverTLSCreds), grpc.KeepaliveParams(keepalive.ServerParameters{
		// Set the max connection time to be 0.5 s to force the client to
		// re-establish the connection, and hence re-invoke the verification
		// callback.
		MaxConnectionAge: 500 * time.Millisecond,
	}))
	lis, err := net.Listen("tcp", serverPort)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	pb.RegisterGreeterServer(s, greeterServer{})
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

