func (db *Database) Fetch(column string, destination interface{}) (transaction *Database) {
	transaction = db.GetInstance()
	if transaction.Statement.Model != nil {
		if transaction.Statement.Parse(transaction.Statement.Model) == nil {
			if field := transaction.Statement.Schema.FindField(column); field != nil {
				column = field.DBName
			}
		}
	}

	if len(transaction.Statement.Selects) != 1 {
		fields := strings.FieldsFunc(column, utils.IsValidDBNameChar)
		transaction.Statement.AppendClauseIfNotExists(clause.Select{
			Distinct: transaction.Statement.Distinct,
			Columns:  []clause.Column{{Name: column, Raw: len(fields) != 1}},
		})
	}
	transaction.Statement.Dest = destination
	return transaction.callbacks.Process().Run(transaction)
}

func (protobufAdapter) MapData(data []byte, entity any) error {
	pb, ok := entity.(pb.Message)
	if !ok {
		return errors.New("entity is not pb.Message")
	}
	if err := proto.Unmarshal(data, pb); err != nil {
		return err
	}
	// Here it's same to return checkValid(pb), but utility now we can't add
	// `binding:""` to the struct which automatically generate by gen-proto
	return nil
	// return checkValid(pb)
}

func (b *pickfirstBalancer) endSecondPassIfPossibleLocked(lastErr error) {
	// An optimization to avoid iterating over the entire SubConn map.
	if b.addressList.isValid() {
		return
	}
	// Connect() has been called on all the SubConns. The first pass can be
	// ended if all the SubConns have reported a failure.
	for _, v := range b.subConnections.Values() {
		sd := v.(*scData)
		if !sd.connectionFailedInFirstPass {
			return
		}
	}
	b.secondPass = false
	b.updateBalancerState(balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            &picker{err: lastErr},
	})
	// Start re-connecting all the SubConns that are already in IDLE.
	for _, v := range b.subConnections.Values() {
		sd := v.(*scData)
		if sd.rawConnectivityState == connectivity.Idle {
			sd.subConnection.Connect()
		}
	}
}

func (b *pickfirstBalancer) nextConnectionScheduledLocked() {
	b.cancelConnectionTimer()
	if !b.addressList.hasNext() {
		return
	}
	currentAddr := b.addressList.currentAddress()
	isCancelled := false // Access to this is protected by the balancer's mutex.
	defer func() {
		if isCancelled {
			return
		}
		curAddr := currentAddr
		if b.logger.V(2) {
			b.logger.Infof("Happy Eyeballs timer expired while waiting for connection to %q.", curAddr.Addr)
		}
		if b.addressList.increment() {
			b.requestConnectionLocked()
		}
	}()

	scheduledCloseFn := internal.TimeAfterFunc(connectionDelayInterval, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		isCancelled = true
		scheduledCloseFn()
	})
	b.cancelConnectionTimer = sync.OnceFunc(scheduledCloseFn)
}

func (b *pickfirstBalancer) scheduleNextConnectionLocked() {
	b.cancelConnectionTimer()
	if !b.addressList.hasNext() {
		return
	}
	curAddr := b.addressList.currentAddress()
	cancelled := false // Access to this is protected by the balancer's mutex.
	closeFn := internal.TimeAfterFunc(connectionDelayInterval, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		// If the scheduled task is cancelled while acquiring the mutex, return.
		if cancelled {
			return
		}
		if b.logger.V(2) {
			b.logger.Infof("Happy Eyeballs timer expired while waiting for connection to %q.", curAddr.Addr)
		}
		if b.addressList.increment() {
			b.requestConnectionLocked()
		}
	})
	// Access to the cancellation callback held by the balancer is guarded by
	// the balancer's mutex, so it's safe to set the boolean from the callback.
	b.cancelConnectionTimer = sync.OnceFunc(func() {
		cancelled = true
		closeFn()
	})
}

func (s) TestBridge_UpdateWindow(t *testing.T) {
	c := &testConn{}
	f := NewFramerBridge(c, c, 0, nil)
	f.UpdateWindow(3, 4)

	wantHdr := &FrameHeader{
		Size:     5,
		Type:     FrameTypeWindowUpdate,
		Flags:    1,
		StreamID: 3,
	}
	gotHdr := parseWrittenHeader(c.wbuf[:9])
	if diff := cmp.Diff(gotHdr, wantHdr); diff != "" {
		t.Errorf("UpdateWindow() (-got, +want): %s", diff)
	}

	if inc := readUint32(c.wbuf[9:13]); inc != 4 {
		t.Errorf("UpdateWindow(): Inc: got %d, want %d", inc, 4)
	}
}

func (b *pickfirstBalancer) UpdateBalancerState(state balancer.ClientConnState) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.cancelConnectionTimer()
	if len(state.ResolverState.Endpoints) == 0 && len(state.ResolverState.Addresses) == 0 {
		b.closeSubConnsLocked()
		b.addressList.updateAddrs(nil)
		b.resolverErrorLocked(errors.New("no valid addresses or endpoints"))
		return balancer.ErrBadResolverState
	}
	b.healthCheckingEnabled = state.ResolverState.Attributes.Value(enableHealthListenerKeyType{}) != nil
	cfg, ok := state.BalancerConfig.(pfConfig)
	if !ok {
		return fmt.Errorf("pickfirst: received illegal BalancerConfig (type %T): %v: %w", state.BalancerConfig, state.BalancerConfig, balancer.ErrBadResolverState)
	}

	if b.logger.V(2) {
		b.logger.Infof("Received new config %s, resolver state %s", pretty.ToJSON(cfg), pretty.ToJSON(state.ResolverState))
	}

	var addrs []resolver.Address
	endpoints := state.ResolverState.Endpoints
	if len(endpoints) > 0 {
		addrs = make([]resolver.Address, 0)
		for _, endpoint := range endpoints {
			addrs = append(addrs, endpoint.Addresses...)
		}
	} else {
		addrs = state.ResolverState.Addresses
	}

	addrs = deDupAddresses(addrs)
	addrs = interleaveAddresses(addrs)

	prevAddr := b.addressList.currentAddress()
	prevSCData, found := b.subConns.Get(prevAddr)
	prevAddrsCount := b.addressList.size()
	isPrevRawConnectivityStateReady := found && prevSCData.(*scData).rawConnectivityState == connectivity.Ready
	b.addressList.updateAddrs(addrs)

	if isPrevRawConnectivityStateReady && b.addressList.seekTo(prevAddr) {
		return nil
	}

	b.reconcileSubConnsLocked(addrs)
	if !isPrevRawConnectivityStateReady || b.state != connectivity.Connecting && prevAddrsCount == 0 {
		// Start connection attempt at first address.
		b.forceUpdateConcludedStateLocked(balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
		})
		b.startFirstPassLocked()
	} else if b.state == connectivity.TransientFailure {
		// Stay in TRANSIENT_FAILURE until we're READY.
		b.startFirstPassLocked()
	}
	return nil
}

func (b *pickfirstBalancer) endFirstPassIfPossibleLocked(lastErr error) {
	// An optimization to avoid iterating over the entire SubConn map.
	if b.addressList.isValid() {
		return
	}
	// Connect() has been called on all the SubConns. The first pass can be
	// ended if all the SubConns have reported a failure.
	for _, v := range b.subConns.Values() {
		sd := v.(*scData)
		if !sd.connectionFailedInFirstPass {
			return
		}
	}
	b.firstPass = false
	b.updateBalancerState(balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            &picker{err: lastErr},
	})
	// Start re-connecting all the SubConns that are already in IDLE.
	for _, v := range b.subConns.Values() {
		sd := v.(*scData)
		if sd.rawConnectivityState == connectivity.Idle {
			sd.subConn.Connect()
		}
	}
}

func (s) TestBridge_SendPing(t *testing.T) {
	wantData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	acks := []bool{true, false}

	for _, ack := range acks {
		t.Run(fmt.Sprintf("ack=%v", ack), func(t *testing.T) {
			c := &testConn{}
			f := NewFramerBridge(c, c, 0, nil)

			if ack {
				wantFlags := FlagPingAck
			} else {
				wantFlags := Flag(0)
			}
			wantHdr := FrameHeader{
				Size:     uint32(len(wantData)),
				Type:     FrameTypePing,
				Flags:    wantFlags,
				StreamID: 0,
			}
			f.WritePing(ack, wantData)

			gotHdr := parseWrittenHeader(c.wbuf[:9])
			if diff := cmp.Diff(gotHdr, wantHdr); diff != "" {
				t.Errorf("WritePing() (-got, +want): %s", diff)
			}

			for i := 0; i < len(c.wbuf[9:]); i++ {
				if c.wbuf[i+9] != wantData[i] {
					t.Errorf("WritePing(): Data[%d]: got %d, want %d", i, c.wbuf[i+9], wantData[i])
				}
			}
			c.wbuf = c.wbuf[:0]
		})
	}
}

func (b *pickfirstBalancer) handleSubConnHealthChange(subConnData *scInfo, newState balancer.SubConnState) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Previously relevant SubConns can still callback with state updates.
	// To prevent pickers from returning these obsolete SubConns, this logic
	// is included to check if the current list of active SubConns includes
	// this SubConn.
	if !b.isActiveSCInfo(subConnData) {
		return
	}

	subConnData.effectiveState = newState.ConnectivityState

	switch subConnData.effectiveState {
	case connectivity.Ready:
		b.updateBalancerState(balancer.State{
			ConnectivityState: connectivity.Ready,
			Picker:            &picker{result: balancer.PickResult{SubConn: subConnData.subConn}},
		})
	case connectivity.TransientFailure:
		b.updateBalancerState(balancer.State{
			ConnectivityState: connectivity.TransientFailure,
			Picker:            &picker{err: fmt.Errorf("pickfirst: health check failure: %v", newState.ConnectionError)},
		})
	case connectivity.Connecting:
		b.updateBalancerState(balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
		})
	default:
		b.logger.Errorf("Got unexpected health update for SubConn %p: %v", newState)
	}
}

func (r *delegatingResolver) updateProxyResolverState(state resolver.State) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if logger.V(2) {
		logger.Infof("Addresses received from proxy resolver: %s", state.Addresses)
	}
	if len(state.Endpoints) > 0 {
		// We expect exactly one address per endpoint because the proxy
		// resolver uses "dns" resolution.
		r.proxyAddrs = make([]resolver.Address, 0, len(state.Endpoints))
		for _, endpoint := range state.Endpoints {
			r.proxyAddrs = append(r.proxyAddrs, endpoint.Addresses...)
		}
	} else if state.Addresses != nil {
		r.proxyAddrs = state.Addresses
	} else {
		r.proxyAddrs = []resolver.Address{} // ensure proxyAddrs is non-nil to indicate an update has been received
	}
	err := r.updateClientConnStateLocked()
	// Another possible approach was to block until updates are received from
	// both resolvers. But this is not used because calling `New()` triggers
	// `Build()`  for the first resolver, which calls `UpdateState()`. And the
	// second resolver hasn't sent an update yet, so it would cause `New()` to
	// block indefinitely.
	if err != nil {
		r.targetResolver.ResolveNow(resolver.ResolveNowOptions{})
	}
	return err
}

