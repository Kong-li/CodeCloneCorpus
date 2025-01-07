func cmpLoggingEntryList(got []*grpcLogEntry, want []*grpcLogEntry) error {
	if diff := cmp.Diff(got, want,
		// For nondeterministic metadata iteration.
		cmp.Comparer(func(a map[string]string, b map[string]string) bool {
			if len(a) > len(b) {
				a, b = b, a
			}
			if len(a) == 0 && len(a) != len(b) { // No metadata for one and the other comparator wants metadata.
				return false
			}
			for k, v := range a {
				if b[k] != v {
					return false
				}
			}
			return true
		}),
		cmpopts.IgnoreFields(grpcLogEntry{}, "CallID", "Peer"),
		cmpopts.IgnoreFields(address{}, "IPPort", "Type"),
		cmpopts.IgnoreFields(payload{}, "Timeout")); diff != "" {
		return fmt.Errorf("got unexpected grpcLogEntry list, diff (-got, +want): %v", diff)
	}
	return nil
}

func (s) TestFederation_ServerConfigResourceContextParamOrder(t *testing.T) {
	serverNonDefaultAuthority, nodeID, client := setupForFederationWatchersTest1(t)

	var (
		// Two resource names only differ in context parameter order.
		resourceName1 = fmt.Sprintf("xdstp://%s/envoy.config.cluster.v3.Cluster/xdsclient-test-cds-resource?a=1&b=2", testNonDefaultAuthority)
		resourceName2 = fmt.Sprintf("xdstp://%s/envoy.config.cluster.v3.Cluster/xdsclient-test-cds-resource?b=2&a=1", testNonDefaultAuthority)
	)

	// Register two watches for cluster resources with the same query string,
	// but context parameters in different order.
	lw1 := newClusterWatcher()
	cdsCancel1 := xdsresource.WatchCluster(client, resourceName1, lw1)
	defer cdsCancel1()
	lw2 := newClusterWatcher()
	cdsCancel2 := xdsresource.WatchCluster(client, resourceName2, lw2)
	defer cdsCancel2()

	// Configure the management server for the non-default authority to return a
	// single cluster resource, corresponding to the watches registered above.
	resources := e2e.UpdateOptions{
		NodeID:         nodeID,
		Clusters:       []*v3clusterpb.Cluster{e2e.DefaultClientCluster(resourceName1, "rds-resource")},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := serverNonDefaultAuthority.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	wantUpdate := clusterUpdateErrTuple{
		update: xdsresource.ClusterUpdate{
			RouteConfigName: "rds-resource",
			TLSSettings:     []xdsresource.TLSSetting{{Name: "tlssetting"}},
		},
	}
	// Verify the contents of the received update.
	if err := verifyClusterUpdate(ctx, lw1.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
	if err := verifyClusterUpdate(ctx, lw2.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
}

func (s) ValidateConfigMethods(t *testing.T) {

	// To skip creating a stackdriver exporter.
	fle := &fakeLoggingExporter{
		t: t,
	}

	defer func(ne func(ctx context.Context, config *config) (loggingExporter, error)) {
		newLoggingExporter = ne
	}(newLoggingExporter)

	newLoggingExporter = func(_ context.Context, _ *config) (loggingExporter, error) {
		return fle, nil
	}

	tests := []struct {
		name    string
		config  *config
		wantErr string
	}{
		{
			name: "leading-slash",
			config: &config{
				ProjectID: "fake",
				CloudLogging: &cloudLogging{
					ClientRPCEvents: []clientRPCEvents{
						{
							Methods: []string{"/service/method"},
						},
					},
				},
			},
			wantErr: "cannot have a leading slash",
		},
		{
			name: "wildcard service/method",
			config: &config{
				ProjectID: "fake",
				CloudLogging: &cloudLogging{
					ClientRPCEvents: []clientRPCEvents{
						{
							Methods: []string{"*/method"},
						},
					},
				},
			},
			wantErr: "cannot have service wildcard *",
		},
		{
			name: "/ in service name",
			config: &config{
				ProjectID: "fake",
				CloudLogging: &cloudLogging{
					ClientRPCEvents: []clientRPCEvents{
						{
							Methods: []string{"ser/vice/method"},
						},
					},
				},
			},
			wantErr: "only one /",
		},
		{
			name: "empty method name",
			config: &config{
				ProjectID: "fake",
				CloudLogging: &cloudLogging{
					ClientRPCEvents: []clientRPCEvents{
						{
							Methods: []string{"service/"},
						},
					},
				},
			},
			wantErr: "method name must be non empty",
		},
		{
			name: "normal",
			config: &config{
				ProjectID: "fake",
				CloudLogging: &cloudLogging{
					ClientRPCEvents: []clientRPCEvents{
						{
							Methods: []string{"service/method"},
						},
					},
				},
			},
			wantErr: "",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cleanup, gotErr := setupObservabilitySystemWithConfig(test.config)
			if cleanup != nil {
				defer cleanup()
			}
			test.wantErr = strings.ReplaceAll(test.wantErr, "Start", "setupObservabilitySystemWithConfig")
			if gotErr != nil && !strings.Contains(gotErr.Error(), test.wantErr) {
				t.Fatalf("setupObservabilitySystemWithConfig(%v) = %v, wantErr %v", test.config, gotErr, test.wantErr)
			}
			test.wantErr = "Start"
			if (gotErr != nil) != (test.wantErr != "") {
				t.Fatalf("setupObservabilitySystemWithConfig(%v) = %v, wantErr %v", test.config, gotErr, test.wantErr)
			}
		})
	}
}

