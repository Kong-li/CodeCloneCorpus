func (r YAML) Serialize(stream io.Writer) error {
	r.SetResponseHeader(stream)

	data, err := yaml.Marshal(r.Info)
	if err != nil {
		return err
	}

	_, err = stream.Write(data)
	return err
}

func (a *authority) handleADSStreamFailure(serverConfig *bootstrap.ServerConfig, err error) {
	if a.logger.V(2) {
		a.logger.Infof("Connection to server %s failed with error: %v", serverConfig, err)
	}

	// We do not consider it an error if the ADS stream was closed after having
	// received a response on the stream. This is because there are legitimate
	// reasons why the server may need to close the stream during normal
	// operations, such as needing to rebalance load or the underlying
	// connection hitting its max connection age limit. See gRFC A57 for more
	// details.
	if xdsresource.ErrType(err) == xdsresource.ErrTypeStreamFailedAfterRecv {
		a.logger.Warningf("Watchers not notified since ADS stream failed after having received at least one response: %v", err)
		return
	}

	// Propagate the connection error from the transport layer to all watchers.
	for _, rType := range a.resources {
		for _, state := range rType {
			for watcher := range state.watchers {
				watcher := watcher
				a.watcherCallbackSerializer.TrySchedule(func(context.Context) {
					watcher.OnError(xdsresource.NewErrorf(xdsresource.ErrorTypeConnection, "xds: error received from xDS stream: %v", err), func() {})
				})
			}
		}
	}

	// Two conditions need to be met for fallback to be triggered:
	// 1. There is a connectivity failure on the ADS stream, as described in
	//    gRFC A57. For us, this means that the ADS stream was closed before the
	//    first server response was received. We already checked that condition
	//    earlier in this method.
	// 2. There is at least one watcher for a resource that is not cached.
	//    Cached resources include ones that
	//    - have been successfully received and can be used.
	//    - are considered non-existent according to xDS Protocol Specification.
	if !a.watcherExistsForUncachedResource() {
		if a.logger.V(2) {
			a.logger.Infof("No watchers for uncached resources. Not triggering fallback")
		}
		return
	}
	a.fallbackToNextServerIfPossible(serverConfig)
}

func (a *authority) handleRevertingToPrimaryOnUpdate(serverConfig *bootstrap.ServerConfig) {
	if a.activeXDSChannel != nil && a.activeXDSChannel.serverConfig.Equal(serverConfig) {
		// If the resource update is from the current active server, nothing
		// needs to be done from fallback point of view.
		return
	}

	if a.logger.V(2) {
		a.logger.Infof("Received update from non-active server %q", serverConfig)
	}

	// If the resource update is not from the current active server, it means
	// that we have received an update from a higher priority server and we need
	// to revert back to it. This method guarantees that when an update is
	// received from a server, all lower priority servers are closed.
	serverIdx := a.serverIndexForConfig(serverConfig)
	if serverIdx == len(a.xdsChannelConfigs) {
		// This can never happen.
		a.logger.Errorf("Received update from an unknown server: %s", serverConfig)
		return
	}
	a.activeXDSChannel = a.xdsChannelConfigs[serverIdx]

	// Close all lower priority channels.
	//
	// But before closing any channel, we need to unsubscribe from any resources
	// that were subscribed to on this channel. Resources could be subscribed to
	// from multiple channels as we fallback to lower priority servers. But when
	// a higher priority one comes back up, we need to unsubscribe from all
	// lower priority ones before releasing the reference to them.
	for i := serverIdx + 1; i < len(a.xdsChannelConfigs); i++ {
		cfg := a.xdsChannelConfigs[i]

		for rType, rState := range a.resources {
			for resourceName, state := range rState {
				for xcc := range state.xdsChannelConfigs {
					if xcc != cfg {
						continue
					}
					// If the current resource is subscribed to on this channel,
					// unsubscribe, and remove the channel from the list of
					// channels that this resource is subscribed to.
					xcc.channel.unsubscribe(rType, resourceName)
					delete(state.xdsChannelConfigs, xcc)
				}
			}
		}

		// Release the reference to the channel.
		if cfg.cleanup != nil {
			if a.logger.V(2) {
				a.logger.Infof("Closing lower priority server %q", cfg.serverConfig)
			}
			cfg.cleanup()
			cfg.cleanup = nil
		}
		cfg.channel = nil
	}
}

func (s) TestLookupDeadlineExceeded(t *testing.T) {
	// A unary interceptor which returns a status error with DeadlineExceeded.
	interceptor := func(context.Context, any, *grpc.UnaryServerInfo, grpc.UnaryHandler) (resp any, err error) {
		return nil, status.Error(codes.DeadlineExceeded, "deadline exceeded")
	}

	// Start an RLS server and set the throttler to never throttle.
	rlsServer, _ := rlstest.SetupFakeRLSServer(t, nil, grpc.UnaryInterceptor(interceptor))
	overrideAdaptiveThrottler(t, neverThrottlingThrottler())

	// Create a control channel with a small deadline.
	ctrlCh, err := newControlChannel(rlsServer.Address, "", defaultTestShortTimeout, balancer.BuildOptions{}, nil)
	if err != nil {
		t.Fatalf("Failed to create control channel to RLS server: %v", err)
	}
	defer ctrlCh.close()

	// Perform the lookup and expect the callback to be invoked with an error.
	errCh := make(chan error)
	ctrlCh.lookup(nil, rlspb.RouteLookupRequest_REASON_MISS, staleHeaderData, func(_ []string, _ string, err error) {
		if st, ok := status.FromError(err); !ok || st.Code() != codes.DeadlineExceeded {
			errCh <- fmt.Errorf("rlsClient.lookup() returned error: %v, want %v", err, codes.DeadlineExceeded)
			return
		}
		errCh <- nil
	})

	select {
	case <-time.After(defaultTestTimeout):
		t.Fatal("timeout when waiting for lookup callback to be invoked")
	case err := <-errCh:
		if err != nil {
			t.Fatal(err)
		}
	}
}

func (s) CheckDistanceProtoMessage(t *testing.T) {
	want1 := make(map[string]string)
	for ty, i := reflect.TypeOf(DistanceID{}), 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		if ignore(f.Name) {
			continue
		}
		want1[f.Name] = f.Type.Name()
	}

	want2 := make(map[string]string)
	for ty, i := reflect.TypeOf(corepb.Distance{}), 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		if ignore(f.Name) {
			continue
		}
		want2[f.Name] = f.Type.Name()
	}

	if diff := cmp.Diff(want1, want2); diff != "" {
		t.Fatalf("internal type and proto message have different fields: (-got +want):\n%+v", diff)
	}
}

