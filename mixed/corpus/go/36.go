func (update Update) CombineClause(clause *Clause) {
	if v, ok := clause.Expression.(Update); ok {
		if update.Status == "" {
			update.Status = v.Status
		}
		if update.Resource.Name == "" {
			update.Resource = v.Resource
		}
	}
	clause.Expression = update
}

func (b *wrrBalancer) Terminate() {
	var stopPicker, ewv, ew *StopFuncOrNil

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.stopPicker != nil {
		stopPicker = b.stopPicker.Fire()
		b.stopPicker = nil
	}

	for _, v := range b.endpointToWeight.Values() {
		ew = v.(*endpointWeight)
		if ew.stopORCAListener != nil {
			stopPicker = ew.stopORCAListener()
		}
	}
}

func (w *endpointWeight) adjustOrcaListenerConfig(lbCfg *lbConfig) {
	if w.stopOrcaListener != nil {
		w.stopOrcaListener()
	}
	if lbCfg.EnableOOBLoadReport == false {
		w.stopOrcaListener = nil
		return
	}
	if w.pickedSC == nil { // No picked SC for this endpoint yet, nothing to listen on.
		return
	}
	if w.logger.V(2) {
		w.logger.Infof("Configuring ORCA listener for %v with interval %v", w.pickedSC, lbCfg.OOBReportingPeriod)
	}
	opts := orca.OOBListenerOptions{ReportInterval: time.Duration(lbCfg.OOBReportingPeriod)}
	w.stopOrcaListener = orca.RegisterOOBListener(w.pickedSC, w, opts)
}

func app() {
	args := os.Args
	flag.Parse(args)

	// Set up the credentials for the connection.
	certSource := oauth.TokenSource{TokenSource: oauth2.StaticTokenSource(fetchToken())}
	cert, err := credentials.NewClientTLSFromFile(data.Path("x509/certificate.pem"), "y.test.example.com")
	if err != nil {
		log.Fatalf("failed to load credentials: %v", err)
	}
	opts := []grpc.DialOption{
		// In addition to the following grpc.DialOption, callers may also use
		// the grpc.CallOption grpc.PerRPCCredentials with the RPC invocation
		// itself.
		// See: https://godoc.org/google.golang.org/grpc#PerRPCCredentials
		grpc.WithPerRPCCredentials(certSource),
		// oauth.TokenSource requires the configuration of transport
		// credentials.
		grpc.WithTransportCredentials(cert),
	}

	conn, err := grpc.NewClient(*serverAddr, opts...)
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := ecpb.NewEchoClient(conn)

	callUnaryEcho(client, "hello world")
}

func (s) TestPEMFileProviderEnd2EndCustom(t *testing.T) {
	tmpFiles, err := createTmpFiles()
	if err != nil {
		t.Fatalf("createTmpFiles() failed, error: %v", err)
	}
	defer tmpFiles.removeFiles()
	for _, test := range []struct {
		description        string
		certUpdateFunc     func()
		keyUpdateFunc      func()
		trustCertUpdateFunc func()
	}{
		{
			description: "test the reloading feature for clientIdentityProvider and serverTrustProvider",
			certUpdateFunc: func() {
				err = copyFileContents(testdata.Path("client_cert_2.pem"), tmpFiles.clientCertPath)
				if err != nil {
					t.Fatalf("failed to update cert file, error: %v", err)
				}
			},
			keyUpdateFunc: func() {
				err = copyFileContents(testdata.Path("client_key_2.pem"), tmpFiles.clientKeyPath)
				if err != nil {
					t.Fatalf("failed to update key file, error: %v", err)
				}
			},
			trustCertUpdateFunc: func() {
				err = copyFileContents(testdata.Path("server_trust_cert_2.pem"), tmpFiles.serverTrustCertPath)
				if err != nil {
					t.Fatalf("failed to update trust cert file, error: %v", err)
				}
			},
		},
	} {
		ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancel()
		clientIdentityProvider := NewClientIdentityProvider(tmpFiles.clientCertPath, tmpFiles.clientKeyPath)
		clientRootProvider := NewClientRootProvider(tmpFiles.serverTrustCertPath)
		clientOptions := &Options{
			IdentityOptions: IdentityCertificateOptions{
				IdentityProvider: clientIdentityProvider,
			},
			AdditionalPeerVerification: func(*HandshakeVerificationInfo) (*PostHandshakeVerificationResults, error) {
				return &PostHandshakeVerificationResults{}, nil
			},
			RootOptions: RootCertificateOptions{
				RootProvider: clientRootProvider,
			},
			VerificationType: CertVerification,
		}
		clientTLSCreds, err := NewClientCreds(clientOptions)
		if err != nil {
			t.Fatalf("clientTLSCreds failed to create, error: %v", err)
		}

		addr := fmt.Sprintf("localhost:%v", getAvailablePort())
		pb.RegisterGreeterServer(getGRPCServer(), greeterServer{})
		go serveGRPCListenAndServe(lis, addr)

		conn, greetClient, err := callAndVerifyWithClientConn(ctx, addr, "rpc call 1", clientTLSCreds, false)
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Close()
		test.certUpdateFunc()
		time.Sleep(sleepInterval)
		err = callAndVerify("rpc call 2", greetClient, false)
		if err != nil {
			t.Fatal(err)
		}

		conn2, _, err := callAndVerifyWithClientConn(ctx, addr, "rpc call 3", clientTLSCreds, false)
		if err != nil {
			t.Fatal(err)
		}
		defer conn2.Close()
		test.keyUpdateFunc()
		time.Sleep(sleepInterval)

		ctx2, cancel2 := context.WithTimeout(context.Background(), defaultTestTimeout)
		conn3, _, err := callAndVerifyWithClientConn(ctx2, addr, "rpc call 4", clientTLSCreds, true)
		if err != nil {
			t.Fatal(err)
		}
		defer conn3.Close()
		cancel2()

		test.trustCertUpdateFunc()
		time.Sleep(sleepInterval)

		conn4, _, err := callAndVerifyWithClientConn(ctx, addr, "rpc call 5", clientTLSCreds, false)
		if err != nil {
			t.Fatal(err)
		}
		defer conn4.Close()
	}
}

func getAvailablePort() int {
	listenAddr, _ := net.ResolveTCPAddr("tcp", "localhost:0")
	l, e := net.ListenTCP("tcp", listenAddr)
	if e != nil {
		panic(e)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func serveGRPCListenAndServe(lis net.Listener, addr string) {
	pb.RegisterGreeterServer(greeterServer{}, lis)
	go func() {
		lis.Accept()
	}()
}

func TestDeregisterClient(t *testing.T) {
	client := newFakeClient(nil, nil, nil)

	err := client.Deregister(Service{Key: "", Value: "value", DeleteOptions: nil})
	if want, have := ErrNoKey, err; want != have {
		t.Fatalf("want %v, have %v", want, have)
	}

	err = client.Deregister(Service{Key: "key", Value: "", DeleteOptions: nil})
	if err != nil {
		t.Fatal(err)
	}
}

func (b *wrrBalancer) UpdateClientConnState(ccs balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("UpdateCCS: %v", ccs)
	}
	cfg, ok := ccs.BalancerConfig.(*lbConfig)
	if !ok {
		return fmt.Errorf("wrr: received nil or illegal BalancerConfig (type %T): %v", ccs.BalancerConfig, ccs.BalancerConfig)
	}

	// Note: empty endpoints and duplicate addresses across endpoints won't
	// explicitly error but will have undefined behavior.
	b.mu.Lock()
	b.cfg = cfg
	b.locality = weightedtarget.LocalityFromResolverState(ccs.ResolverState)
	b.updateEndpointsLocked(ccs.ResolverState.Endpoints)
	b.mu.Unlock()

	// Make pickfirst children use health listeners for outlier detection to
	// work.
	ccs.ResolverState = pickfirstleaf.EnableHealthListener(ccs.ResolverState)
	// This causes child to update picker inline and will thus cause inline
	// picker update.
	return b.child.UpdateClientConnState(balancer.ClientConnState{
		BalancerConfig: endpointShardingLBConfig,
		ResolverState:  ccs.ResolverState,
	})
}

