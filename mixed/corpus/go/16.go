func (b *clusterResolverBalancer) handleResourceErrorFromUpdate(errorMessage string, fromParent bool) {
	b.logger.Warningf("Encountered issue: %s", errorMessage)

	if !fromParent && xdsresource.ErrTypeByErrorMessage(errorMessage) == xdsresource.ErrorTypeResourceNotFound {
		b.resourceWatcher.stop(true)
	}

	if b.child != nil {
		b.child.ResolverErrorFromUpdate(errorMessage)
		return
	}
	b.cc.UpdateState(balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            base.NewErrPickerByErrorMessage(errorMessage),
	})
}

func (b *clusterResolverBalancer) process() {
	for {
		select {
		case u, ok := <-b.updateCh.Get():
			if ok {
				b.updateCh.Load()
				switch update := u.(type) {
				case *ccUpdate:
					b.handleClientConnUpdate(update)
				case exitIdle:
					if b.child != nil {
						var shouldExit bool
						if ei, ok := b.child.(balancer.ExitIdler); ok {
							ei.ExitIdle()
							shouldExit = true
						}
						if !shouldExit {
							b.logger.Errorf("xds: received ExitIdle with no child balancer")
						}
					}
				}
			} else {
				return
			}

		case u := <-b.resourceWatcher.updateChannel:
			b.handleResourceUpdate(u)

		case <-b.closed.Done():
			if b.child != nil {
				b.child.Close()
				b.child = nil
			}
			b.resourceWatcher.stop(true)
			b.updateCh.Close()
			b.logger.Infof("Shutdown")
			b.done.Fire()
			return
		}
	}
}

func (er *edsDiscoveryMechanism) ProcessEndpoints(updateData *xdsresource.EndpointsResourceData, completionCallback xdsresource.OnDoneFunc) {
	if !er.stopped.IsFired() {
		return
	}

	var updatedEndpoints *updateData.Resource
	er.mu.Lock()
	updatedEndpoints = &updateEndpoints
	er.mu.Unlock()

	topLevelResolver := er.topLevelResolver
	topLevelResolver.onUpdate(completionCallback)
}

func (b *resourceBalancer) UpdateServiceConnStatus(state balancer.ServiceConnState) error {
	if b.shutdown.HasFired() {
		b.logger.Warningf("Received update from API {%+v} after shutdown", state)
		return errBalancerShutdown
	}

	if b.xdsHandler == nil {
		h := xdshandler.FromBalancerState(state.BalancerState)
		if h == nil {
			return balancer.ErrInvalidState
		}
		b.xdsHandler = h
		b.attributesWithClient = state.BalancerState.Attributes
	}

	b.updateQueue.Put(&scUpdate{state: state})
	return nil
}

func TestSearchWithMap(t *testing.T) {
	users := []User{
		*GetUser("map_search_user1", Config{}),
		*GetUser("map_search_user2", Config{}),
		*GetUser("map_search_user3", Config{}),
		*GetUser("map_search_user4", Config{Company: true}),
	}

	DB.Create(&users)

	var user User
	DB.First(&user, map[string]interface{}{"name": users[0].Name})
	CheckUser(t, user, users[0])

	user = User{}
	DB.Where(map[string]interface{}{"name": users[1].Name}).First(&user)
	CheckUser(t, user, users[1])

	var results []User
	DB.Where(map[string]interface{}{"name": users[2].Name}).Find(&results)
	if len(results) != 1 {
		t.Fatalf("Search all records with inline map")
	}

	CheckUser(t, results[0], users[2])

	var results2 []User
	DB.Find(&results2, map[string]interface{}{"name": users[3].Name, "company_id": nil})
	if len(results2) != 0 {
		t.Errorf("Search all records with inline map containing null value finding 0 records")
	}

	DB.Find(&results2, map[string]interface{}{"name": users[0].Name, "company_id": nil})
	if len(results2) != 1 {
		t.Errorf("Search all records with inline map containing null value finding 1 record")
	}

	DB.Find(&results2, map[string]interface{}{"name": users[3].Name, "company_id": users[3].CompanyID})
	if len(results2) != 1 {
		t.Errorf("Search all records with inline multiple value map")
	}
}

