func (b *cdsBalancer) TerminateIdlePeriod() {
	b.serializer.TrySchedule(func(ctx context.Context) {
		if b.childLB == nil {
			b.logger.Warning("Received TerminateIdlePeriod without a child policy")
			return
		}
		childBalancer, exists := b.childLB.(balancer.ExitIdler)
		if !exists {
			return
		}
		childBalancer.ExitIdle()
	})
}

func (s) TestServerHandshakeModified(t *testing.T) {
	for _, testConfig := range []struct {
		delay              time.Duration
		handshakeCount     int
	}{
		{0 * time.Millisecond, 1},
		{100 * time.Millisecond, 10 * int(envconfig.ALTSMaxConcurrentHandshakes)},
	} {
		errorChannel := make(chan error)
		resetStats()

		testContext, cancelTest := context.WithTimeout(context.Background(), defaultTestTimeout)
		defer cancelTest()

		for j := 0; j < testConfig.handshakeCount; j++ {
			streamInstance := &testRPCStream{
				t:        t,
				isClient: false,
			}
			frame1 := testutil.MakeFrame("ClientInit")
			frame2 := testutil.MakeFrame("ClientFinished")
			inputBuffer := bytes.NewBuffer(frame1)
			inputBuffer.Write(frame2)
			outputBuffer := new(bytes.Buffer)
			testConnection := testutil.NewTestConn(inputBuffer, outputBuffer)
			serverHandshakerInstance := &altsHandshaker{
				stream:     streamInstance,
				conn:       testConnection,
				serverOpts: DefaultServerHandshakerOptions(),
				side:       core.ServerSide,
			}
			go func() {
				contextValue, contextErr := serverHandshakerInstance.ServerHandshake(testContext)
				if contextErr == nil && contextValue == nil {
					errorChannel <- errors.New("expected non-nil ALTS context")
					return
				}
				errorChannel <- contextErr
				serverHandshakerInstance.Close()
			}()
		}

		for k := 0; k < testConfig.handshakeCount; k++ {
			if err := <-errorChannel; err != nil {
				t.Errorf("ServerHandshake() = _, %v, want _, <nil>", err)
			}
		}

		if maxConcurrentCalls > int(envconfig.ALTSMaxConcurrentHandshakes) {
			t.Errorf("Observed %d concurrent handshakes; want <= %d", maxConcurrentCalls, envconfig.ALTSMaxConcurrentHandshakes)
		}
	}
}

func (lb *lbBalancer) UpdateClientConnState(ccs balancer.ClientConnState) error {
	if lb.logger.V(2) {
		lb.logger.Infof("UpdateClientConnState: %s", pretty.ToJSON(ccs))
	}
	gc, _ := ccs.BalancerConfig.(*grpclbServiceConfig)
	lb.handleServiceConfig(gc)

	backendAddrs := ccs.ResolverState.Addresses

	var remoteBalancerAddrs []resolver.Address
	if sd := grpclbstate.Get(ccs.ResolverState); sd != nil {
		// Override any balancer addresses provided via
		// ccs.ResolverState.Addresses.
		remoteBalancerAddrs = sd.BalancerAddresses
	}

	if len(backendAddrs)+len(remoteBalancerAddrs) == 0 {
		// There should be at least one address, either grpclb server or
		// fallback. Empty address is not valid.
		return balancer.ErrBadResolverState
	}

	if len(remoteBalancerAddrs) == 0 {
		if lb.ccRemoteLB != nil {
			lb.ccRemoteLB.close()
			lb.ccRemoteLB = nil
		}
	} else if lb.ccRemoteLB == nil {
		// First time receiving resolved addresses, create a cc to remote
		// balancers.
		if err := lb.newRemoteBalancerCCWrapper(); err != nil {
			return err
		}
		// Start the fallback goroutine.
		go lb.fallbackToBackendsAfter(lb.fallbackTimeout)
	}

	if lb.ccRemoteLB != nil {
		// cc to remote balancers uses lb.manualResolver. Send the updated remote
		// balancer addresses to it through manualResolver.
		lb.manualResolver.UpdateState(resolver.State{Addresses: remoteBalancerAddrs})
	}

	lb.mu.Lock()
	lb.resolvedBackendAddrs = backendAddrs
	if len(remoteBalancerAddrs) == 0 || lb.inFallback {
		// If there's no remote balancer address in ClientConn update, grpclb
		// enters fallback mode immediately.
		//
		// If a new update is received while grpclb is in fallback, update the
		// list of backends being used to the new fallback backends.
		lb.refreshSubConns(lb.resolvedBackendAddrs, true, lb.usePickFirst)
	}
	lb.mu.Unlock()
	return nil
}

func (s) TestNewClientHandshaker(t *testing.T) {
	conn := testutil.NewTestConn(nil, nil)
	clientConn := &grpc.ClientConn{}
	opts := &ClientHandshakerOptions{}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	hs, err := NewClientHandshaker(ctx, clientConn, conn, opts)
	if err != nil {
		t.Errorf("NewClientHandshaker returned unexpected error: %v", err)
	}
	expectedHs := &altsHandshaker{
		stream:     nil,
		conn:       conn,
		clientConn: clientConn,
		clientOpts: opts,
		serverOpts: nil,
		side:       core.ClientSide,
	}
	cmpOpts := []cmp.Option{
		cmp.AllowUnexported(altsHandshaker{}),
		cmpopts.IgnoreFields(altsHandshaker{}, "conn", "clientConn"),
	}
	if got, want := hs.(*altsHandshaker), expectedHs; !cmp.Equal(got, want, cmpOpts...) {
		t.Errorf("NewClientHandshaker() returned unexpected handshaker: got: %v, want: %v", got, want)
	}
	if hs.(*altsHandshaker).stream != nil {
		t.Errorf("NewClientHandshaker() returned handshaker with non-nil stream")
	}
	if hs.(*altsHandshaker).clientConn != clientConn {
		t.Errorf("NewClientHandshaker() returned handshaker with unexpected clientConn")
	}
	hs.Close()
}

func (b *systemBalancer) generateServersForGroup(groupName string, level int, servers []groupresolver.ServerMechanism, groupsSeen map[string]bool) ([]groupresolver.ServerMechanism, bool, error) {
	if level >= aggregateGroupMaxLevel {
		return servers, false, errExceedsMaxLevel
	}

	if groupsSeen[groupName] {
		// Server mechanism already seen through a different path.
		return servers, true, nil
	}
	groupsSeen[groupName] = true

	state, ok := b.watchers[groupName]
	if !ok {
		// If we have not seen this group so far, create a watcher for it, add
		// it to the map, start the watch and return.
		b.createAndAddWatcherForGroup(groupName)

		// And since we just created the watcher, we know that we haven't
		// resolved the group graph yet.
		return servers, false, nil
	}

	// A watcher exists, but no update has been received yet.
	if state.lastUpdate == nil {
		return servers, false, nil
	}

	var server groupresolver.ServerMechanism
	group := state.lastUpdate
	switch group.GroupType {
	case xdsresource.GroupTypeAggregate:
		// This boolean is used to track if any of the groups in the graph is
		// not yet completely resolved or returns errors, thereby allowing us to
		// traverse as much of the graph as possible (and start the associated
		// watches where required) to ensure that groupsSeen contains all
		// groups in the graph that we can traverse to.
		missingGroup := false
		var err error
		for _, child := range group.PrioritizedGroupNames {
			var ok bool
			servers, ok, err = b.generateServersForGroup(child, level+1, servers, groupsSeen)
			if err != nil || !ok {
				missingGroup = true
			}
		}
		return servers, !missingGroup, err
	case xdsresource.GroupTypeGRPC:
		server = groupresolver.ServerMechanism{
			Type:                  groupresolver.ServerMechanismTypeGRPC,
			GroupName:              group.GroupName,
			GRPCServiceName:        group.GRPCServiceName,
			MaxConcurrentStreams:   group.MaxStreams,
			TLSContext:             group.TLSContextConfig,
		}
	case xdsresource.GroupTypeHTTP:
		server = groupresolver.ServerMechanism{
			Type:         groupresolver.ServerMechanismTypeHTTP,
			GroupName:    group.GroupName,
			Hostname:     group.Hostname,
			Port:         group.Port,
			PathMatchers: group.PathMatchers,
		}
	}
	odJSON := group.OutlierDetection
	// "In the system LB policy, if the outlier_detection field is not set in
	// the Group resource, a "no-op" outlier_detection config will be
	// generated in the corresponding ServerMechanism config, with all
	// fields unset." - A50
	if odJSON == nil {
		// This will pick up top level defaults in Group Resolver
		// ParseConfig, but sre and fpe will be nil still so still a
		// "no-op" config.
		odJSON = json.RawMessage(`{}`)
	}
	server.OutlierDetection = odJSON

	server.TelemetryLabels = group.TelemetryLabels

	return append(servers, server), true, nil
}

func (b *cdsBalancer) onServiceUpdate(name string, update xdsresource.ServiceUpdate) {
	state := b.watchers[name]
	if state == nil {
		// We are currently not watching this service anymore. Return early.
		return
	}

	b.logger.Infof("Received Service resource: %s", pretty.ToJSON(update))

	// Update the watchers map with the update for the service.
	state.lastUpdate = &update

	// For an aggregate service, always use the security configuration on the
	// root service.
	if name == b.lbCfg.ServiceName {
		// Process the security config from the received update before building the
		// child policy or forwarding the update to it. We do this because the child
		// policy may try to create a new subConn inline. Processing the security
		// configuration here and setting up the handshakeInfo will make sure that
		// such attempts are handled properly.
		if err := b.handleSecurityConfig(update.SecurityCfg); err != nil {
			// If the security config is invalid, for example, if the provider
			// instance is not found in the bootstrap config, we need to put the
			// channel in transient failure.
			b.onServiceError(name, fmt.Errorf("received Service resource contains invalid security config: %v", err))
			return
		}
	}

	servicesSeen := make(map[string]bool)
	dms, ok, err := b.generateDMsForService(b.lbCfg.ServiceName, 0, nil, servicesSeen)
	if err != nil {
		b.onServiceError(b.lbCfg.ServiceName, fmt.Errorf("failed to generate discovery mechanisms: %v", err))
		return
	}
	if ok {
		if len(dms) == 0 {
			b.onServiceError(b.lbCfg.ServiceName, fmt.Errorf("aggregate service graph has no leaf services"))
			return
		}
		// Child policy is built the first time we resolve the service graph.
		if b.childLB == nil {
			childLB, err := newChildBalancer(b.ccw, b.bOpts)
			if err != nil {
				b.logger.Errorf("Failed to create child policy of type %s: %v", serviceresolver.Name, err)
				return
			}
			b.childLB = childLB
			b.logger.Infof("Created child policy %p of type %s", b.childLB, serviceresolver.Name)
		}

		// Prepare the child policy configuration, convert it to JSON, have it
		// parsed by the child policy to convert it into service config and push
		// an update to it.
		childCfg := &serviceresolver.LBConfig{
			DiscoveryMechanisms: dms,
			// The LB policy is configured by the root service.
			XDSLBPolicy: b.watchers[b.lbCfg.ServiceName].lastUpdate.LBPolicy,
		}
		cfgJSON, err := json.Marshal(childCfg)
		if err != nil {
			// Shouldn't happen, since we just prepared struct.
			b.logger.Errorf("cds_balancer: error marshalling prepared config: %v", childCfg)
			return
		}

		var sc serviceconfig.LoadBalancingConfig
		if sc, err = b.childConfigParser.ParseConfig(cfgJSON); err != nil {
			b.logger.Errorf("cds_balancer: serviceresolver config generated %v is invalid: %v", string(cfgJSON), err)
			return
		}

		ccState := balancer.ClientConnState{
			ResolverState:  xdsclient.SetClient(resolver.State{}, b.xdsClient),
			BalancerConfig: sc,
		}
		if err := b.childLB.UpdateClientConnState(ccState); err != nil {
			b.logger.Errorf("Encountered error when sending config {%+v} to child policy: %v", ccState, err)
		}
	}
	// We no longer need the services that we did not see in this iteration of
	// generateDMsForService().
	for service := range servicesSeen {
		state, ok := b.watchers[service]
		if ok {
			continue
		}
		state.cancelWatch()
		delete(b.watchers, service)
	}
}

func (c *tlsCreds) SecureConnection(rawNetConn net.Conn) (net.Conn, AuthInfo, error) {
	tlsConfig := c.config
	tlsServerConn := tls.Server(rawNetConn, tlsConfig)
	err := tlsServerConn.Handshake()
	if err != nil {
		tlsServerConn.Close()
		return nil, nil, err
	}
	connectionState := tlsServerConn.ConnectionState()
	var securityLevel SecurityLevel = PrivacyAndIntegrity

	negotiatedProtocol := connectionState.NegotiatedProtocol
	if negotiatedProtocol == "" {
		if envconfig.EnforceALPNEnabled {
			tlsServerConn.Close()
			return nil, nil, fmt.Errorf("credentials: cannot check peer: missing selected ALPN property")
		}
		if logger.V(2) {
			logger.Info("Allowing TLS connection from client with ALPN disabled. TLS connections with ALPN disabled will be disallowed in future grpc-go releases")
		}
	}

	tlsInfo := TLSInfo{
		State:     connectionState,
		SPIFFEID:  getSPIFFEIFromState(connectionState),
		CommonAuthInfo: CommonAuthInfo{
			SecurityLevel: securityLevel,
		},
	}
	return credinternal.WrapSyscallConn(rawNetConn, tlsServerConn), tlsInfo, nil
}

func getSPIFFEIFromState(state ConnectionState) *SPIFFEID {
	id := credinternal.SPIFFEIDFromState(state)
	return id
}

func (s) TestStatusDetails(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	for _, serverType := range []struct {
		name            string
		startServerFunc func(*stubserver.StubServer) error
	}{{
		name: "normal server",
		startServerFunc: func(ss *stubserver.StubServer) error {
			return ss.StartServer()
		},
	}, {
		name: "handler server",
		startServerFunc: func(ss *stubserver.StubServer) error {
			return ss.StartHandlerServer()
		},
	}} {
		t.Run(serverType.name, func(t *testing.T) {
			// Convenience function for making a status including details.
			detailErr := func(c codes.Code, m string) error {
				s, err := status.New(c, m).WithDetails(&testpb.SimpleRequest{
					Payload: &testpb.Payload{Body: []byte("detail msg")},
				})
				if err != nil {
					t.Fatalf("Error adding details: %v", err)
				}
				return s.Err()
			}

			serialize := func(err error) string {
				buf, _ := proto.Marshal(status.Convert(err).Proto())
				return string(buf)
			}

			testCases := []struct {
				name        string
				trailerSent metadata.MD
				errSent     error
				trailerWant []string
				errWant     error
				errContains error
			}{{
				name:        "basic without details",
				trailerSent: metadata.MD{},
				errSent:     status.Error(codes.Aborted, "test msg"),
				errWant:     status.Error(codes.Aborted, "test msg"),
			}, {
				name:        "basic without details passes through trailers",
				trailerSent: metadata.MD{"grpc-status-details-bin": []string{"random text"}},
				errSent:     status.Error(codes.Aborted, "test msg"),
				trailerWant: []string{"random text"},
				errWant:     status.Error(codes.Aborted, "test msg"),
			}, {
				name:        "basic without details conflicts with manual details",
				trailerSent: metadata.MD{"grpc-status-details-bin": []string{serialize(status.Error(codes.Canceled, "test msg"))}},
				errSent:     status.Error(codes.Aborted, "test msg"),
				trailerWant: []string{serialize(status.Error(codes.Canceled, "test msg"))},
				errContains: status.Error(codes.Internal, "mismatch"),
			}, {
				name:        "basic with details",
				trailerSent: metadata.MD{},
				errSent:     detailErr(codes.Aborted, "test msg"),
				trailerWant: []string{serialize(detailErr(codes.Aborted, "test msg"))},
				errWant:     detailErr(codes.Aborted, "test msg"),
			}, {
				name:        "basic with details discards user's trailers",
				trailerSent: metadata.MD{"grpc-status-details-bin": []string{"will be ignored"}},
				errSent:     detailErr(codes.Aborted, "test msg"),
				trailerWant: []string{serialize(detailErr(codes.Aborted, "test msg"))},
				errWant:     detailErr(codes.Aborted, "test msg"),
			}}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					// Start a simple server that returns the trailer and error it receives from
					// channels.
					ss := &stubserver.StubServer{
						UnaryCallF: func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
							grpc.SetTrailer(ctx, tc.trailerSent)
							return nil, tc.errSent
						},
					}
					if err := serverType.startServerFunc(ss); err != nil {
						t.Fatalf("Error starting endpoint server: %v", err)
					}
					if err := ss.StartClient(); err != nil {
						t.Fatalf("Error starting endpoint client: %v", err)
					}
					defer ss.Stop()

					trailerGot := metadata.MD{}
					_, errGot := ss.Client.UnaryCall(ctx, &testpb.SimpleRequest{}, grpc.Trailer(&trailerGot))
					gsdb := trailerGot["grpc-status-details-bin"]
					if !cmp.Equal(gsdb, tc.trailerWant) {
						t.Errorf("Trailer got: %v; want: %v", gsdb, tc.trailerWant)
					}
					if tc.errWant != nil && !testutils.StatusErrEqual(errGot, tc.errWant) {
						t.Errorf("Err got: %v; want: %v", errGot, tc.errWant)
					}
					if tc.errContains != nil && (status.Code(errGot) != status.Code(tc.errContains) || !strings.Contains(status.Convert(errGot).Message(), status.Convert(tc.errContains).Message())) {
						t.Errorf("Err got: %v; want: (Contains: %v)", errGot, tc.errWant)
					}
				})
			}
		})
	}
}

