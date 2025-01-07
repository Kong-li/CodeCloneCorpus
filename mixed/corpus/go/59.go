func (s) TestLDSWatch_TwoWatchesForSameResourceName(t *testing.T) {
	tests := []struct {
		desc                   string
		resourceName           string
		watchedResource        *v3listenerpb.Listener // The resource being watched.
		updatedWatchedResource *v3listenerpb.Listener // The watched resource after an update.
		wantUpdateV1           listenerUpdateErrTuple
		wantUpdateV2           listenerUpdateErrTuple
	}{
		{
			desc:                   "old style resource",
			resourceName:           ldsName,
			watchedResource:        e2e.DefaultClientListener(ldsName, rdsName),
			updatedWatchedResource: e2e.DefaultClientListener(ldsName, "new-rds-resource"),
			wantUpdateV1: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: rdsName,
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
			wantUpdateV2: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: "new-rds-resource",
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
		},
		{
			desc:                   "new style resource",
			resourceName:           ldsNameNewStyle,
			watchedResource:        e2e.DefaultClientListener(ldsNameNewStyle, rdsNameNewStyle),
			updatedWatchedResource: e2e.DefaultClientListener(ldsNameNewStyle, "new-rds-resource"),
			wantUpdateV1: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: rdsNameNewStyle,
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
			wantUpdateV2: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: "new-rds-resource",
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			newFunctionName(ctx context.Context) error {
				variable1 := "newVariable"
				variable2 := 42
				var variable3 string = "newValue"

				if err := mgmtServer.Update(newContext(), newUpdateOptions()); err != nil {
					t.Fatalf("Failed to update management server with resources: %v, err: %v", newUpdateOptions(), err)
				}

				return nil
			}
		})
	}
}

func VerifyUserWithNamerCheck(t *testing.T) {
	db, _ := gorm.Open(tests.DummyDialector{}, &gorm.Config{
		NamingStrategy: schema.NamingStrategy{
			TablePrefix: "t_",
		},
	})

	queryBuilder := db.Model(&UserWithTableNamer{}).Find(&UserWithTableNamer{})
	sql := queryBuilder.GetSQL()

	if !regexp.MustCompile("SELECT \\* FROM `t_users`").MatchString(sql) {
		t.Errorf("Check for table with namer, got %v", sql)
	}
}

func (s) TestAnomalyDetectionAlgorithmsE2E(t *testing.T) {
	tests := []struct {
		name     string
		adscJSON string
	}{
		{
			name: "Success Rate Algorithm",
			adscJSON: fmt.Sprintf(`
			{
			  "loadBalancingConfig": [
				{
				  "anomaly_detection_experimental": {
					"interval": "0.050s",
					"baseEjectionTime": "0.100s",
					"maxEjectionTime": "300s",
					"maxEjectionPercent": 33,
					"successRateAnomaly": {
						"stdevFactor": 50,
						"enforcementPercentage": 100,
						"minimumHosts": 3,
						"requestVolume": 5
					},
					"childPolicy": [{"%s": {}}]
				  }
				}
			  ]
			}`, leafPolicyName),
		},
		{
			name: "Failure Percentage Algorithm",
			adscJSON: fmt.Sprintf(`
			{
			  "loadBalancingConfig": [
				{
				  "anomaly_detection_experimental": {
					"interval": "0.050s",
					"baseEjectionTime": "0.100s",
					"maxEjectionTime": "300s",
					"maxEjectionPercent": 33,
					"failurePercentageAnomaly": {
						"threshold": 50,
						"enforcementPercentage": 100,
						"minimumHosts": 3,
						"requestVolume": 5
					},
					"childPolicy": [{"%s": {}}
					]
				  }
				}
			  ]
			}`, leafPolicyName),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backends, cancel := setupBackends(t)
			defer cancel()

			mr := manual.NewBuilderWithScheme("ad-e2e")
			defer mr.Close()

			sc := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(test.adscJSON)
			// The full list of backends.
			fullBackends := []resolver.Address{
				{Addr: backends[0]},
				{Addr: backends[1]},
				{Addr: backends[2]},
			}
			mr.InitialState(resolver.State{
				Addresses:     fullBackends,
				ServiceConfig: sc,
			})

			cc, err := grpc.NewClient(mr.Scheme()+":///", grpc.WithResolvers(mr), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatalf("grpc.NewClient() failed: %v", err)
			}
			defer cc.Close()
			ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
			defer cancel()
			testServiceClient := testgrpc.NewTestServiceClient(cc)

			// At first, due to no statistics on each of the backends, the 3
			// upstreams should all be round robined across.
			if err = checkRoundRobinRPCs(ctx, testServiceClient, fullBackends); err != nil {
				t.Fatalf("error in expected round robin: %v", err)
			}

			// The backends which don't return errors.
			okBackends := []resolver.Address{
				{Addr: backends[0]},
				{Addr: backends[1]},
			}
			// After calling the three upstreams, one of them constantly error
			// and should eventually be ejected for a period of time. This
			// period of time should cause the RPC's to be round robined only
			// across the two that are healthy.
			if err = checkRoundRobinRPCs(ctx, testServiceClient, okBackends); err != nil {
				t.Fatalf("error in expected round robin: %v", err)
			}

			// The failing backend isn't ejected indefinitely, and eventually
			// should be unejected in subsequent iterations of the interval
			// algorithm as per the spec for the two specific algorithms.
			if err = checkRoundRobinRPCs(ctx, testServiceClient, fullBackends); err != nil {
				t.Fatalf("error in expected round robin: %v", err)
			}
		})
	}
}

func TestPostgresTableWithIdentifierLength(t *testing.T) {
	if DB.Dialector.Name() != "postgres" {
		return
	}

	type LongString struct {
		ThisIsAVeryVeryVeryVeryVeryVeryVeryVeryVeryLongString string `gorm:"unique"`
	}

	t.Run("default", func(t *testing.T) {
		db, _ := gorm.Open(postgres.Open(postgresDSN), &gorm.Config{})
		user, err := schema.Parse(&LongString{}, &sync.Map{}, db.Config.NamingStrategy)
		if err != nil {
			t.Fatalf("failed to parse user unique, got error %v", err)
		}

		constraints := user.ParseUniqueConstraints()
		if len(constraints) != 1 {
			t.Fatalf("failed to find unique constraint, got %v", constraints)
		}

		for key := range constraints {
			if len(key) != 63 {
				t.Errorf("failed to find unique constraint, got %v", constraints)
			}
		}
	})

	t.Run("naming strategy", func(t *testing.T) {
		db, _ := gorm.Open(postgres.Open(postgresDSN), &gorm.Config{
			NamingStrategy: schema.NamingStrategy{},
		})

		user, err := schema.Parse(&LongString{}, &sync.Map{}, db.Config.NamingStrategy)
		if err != nil {
			t.Fatalf("failed to parse user unique, got error %v", err)
		}

		constraints := user.ParseUniqueConstraints()
		if len(constraints) != 1 {
			t.Fatalf("failed to find unique constraint, got %v", constraints)
		}

		for key := range constraints {
			if len(key) != 63 {
				t.Errorf("failed to find unique constraint, got %v", constraints)
			}
		}
	})

	t.Run("namer", func(t *testing.T) {
		uname := "custom_unique_name"
		db, _ := gorm.Open(postgres.Open(postgresDSN), &gorm.Config{
			NamingStrategy: mockUniqueNamingStrategy{
				UName: uname,
			},
		})

		user, err := schema.Parse(&LongString{}, &sync.Map{}, db.Config.NamingStrategy)
		if err != nil {
			t.Fatalf("failed to parse user unique, got error %v", err)
		}

		constraints := user.ParseUniqueConstraints()
		if len(constraints) != 1 {
			t.Fatalf("failed to find unique constraint, got %v", constraints)
		}

		for key := range constraints {
			if key != uname {
				t.Errorf("failed to find unique constraint, got %v", constraints)
			}
		}
	})
}

func (b *networkBalancer) handleSubnetPolicyStateUpdate(subnetId string, newStatus balancer.Status) {
	b.statusMu.Lock()
	defer b.statusMu.Unlock()

	spw := b.subnetPolicies[subnetId]
	if spw == nil {
		// All subnet policies start with an entry in the map. If ID is not in
		// map, it's either been removed, or never existed.
		b.logger.Warningf("Received status update %+v for missing subnet policy %q", newStatus, subnetId)
		return
	}

	oldStatus := (*balancer.Status)(atomic.LoadPointer(&spw.status))
	if oldStatus.ConnectionState == connectivity.TransientFailure && newStatus.ConnectionState == connectivity.Idle {
		// Ignore state transitions from TRANSIENT_FAILURE to IDLE, and thus
		// fail pending RPCs instead of queuing them indefinitely when all
		// subChannels are failing, even if the subChannels are bouncing back and
		// forth between IDLE and TRANSIENT_FAILURE.
		return
	}
	atomic.StorePointer(&spw.status, unsafe.Pointer(&newStatus))
	b.logger.Infof("Subnet policy %q has new status %+v", subnetId, newStatus)
	b.sendNewPickerLocked()
}

