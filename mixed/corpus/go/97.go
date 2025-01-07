func TestFormMultipartBindingBindError1(t *testing.T) {
	testCases := []struct {
		name string
		s    any
	}{
		{"wrong type", &struct {
			Files int `form:"file"`
		}{}},
		{"wrong array size", &struct {
			Files [1]*multipart.FileHeader `form:"file"`
		}{}},
		{"wrong slice type", &struct {
			Files []int `form:"file"`
		}{}},
	}

	for _, tc := range testCases {
		req := createRequestMultipartFiles(t, "file1", "file2")
		err := FormMultipart.Bind(req, tc.s)
		if err != nil {
			t.Errorf("unexpected success for %s: %v", tc.name, err)
		} else {
			t.Logf("expected error for %s but got none", tc.name)
		}
	}
}

func (s) TestAggregatedClusterFailure_ExceedsMaxStackDepth(t *testing.T) {
	mgmtServer, nodeID, cc, _, _, _, _ := setupWithManagementServer(t)

	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{clusterName + "-1"}),
			makeAggregateClusterResource(clusterName+"-1", []string{clusterName + "-2"}),
			makeAggregateClusterResource(clusterName+"-2", []string{clusterName + "-3"}),
			makeAggregateClusterResource(clusterName+"-3", []string{clusterName + "-4"}),
			makeAggregateClusterResource(clusterName+"-4", []string{clusterName + "-5"}),
			makeAggregateClusterResource(clusterName+"-5", []string{clusterName + "-6"}),
			makeAggregateClusterResource(clusterName+"-6", []string{clusterName + "-7"}),
			makeAggregateClusterResource(clusterName+"-7", []string{clusterName + "-8"}),
			makeAggregateClusterResource(clusterName+"-8", []string{clusterName + "-9"}),
			makeAggregateClusterResource(clusterName+"-9", []string{clusterName + "-10"}),
			makeAggregateClusterResource(clusterName+"-10", []string{clusterName + "-11"}),
			makeAggregateClusterResource(clusterName+"-11", []string{clusterName + "-12"}),
			makeAggregateClusterResource(clusterName+"-12", []string{clusterName + "-13"}),
			makeAggregateClusterResource(clusterName+"-13", []string{clusterName + "-14"}),
			makeAggregateClusterResource(clusterName+"-14", []string{clusterName + "-15"}),
			makeAggregateClusterResource(clusterName+"-15", []string{clusterName + "-16"}),
			e2e.DefaultCluster(clusterName+"-16", serviceName, e2e.SecurityLevelNone),
		},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	testutils.AwaitState(ctx, t, cc, connectivity.TransientFailure)

	const wantErr = "aggregate cluster graph exceeds max depth"
	client := testgrpc.NewTestServiceClient(cc)
	_, err := client.EmptyCall(ctx, &testpb.Empty{})
	if code := status.Code(err); code != codes.Unavailable {
		t.Fatalf("EmptyCall() failed with code: %v, want %v", code, codes.Unavailable)
	}
	if err != nil && !strings.Contains(err.Error(), wantErr) {
		t.Fatalf("EmptyCall() failed with err: %v, want err containing: %v", err, wantErr)
	}

	// Start a test service backend.
	server := stubserver.StartTestService(t, nil)
	t.Cleanup(server.Stop)

	// Update the aggregate cluster resource to no longer exceed max depth, and
	// be at the maximum depth allowed.
	resources = e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{clusterName + "-1"}),
			makeAggregateClusterResource(clusterName+"-1", []string{clusterName + "-2"}),
			makeAggregateClusterResource(clusterName+"-2", []string{clusterName + "-3"}),
			makeAggregateClusterResource(clusterName+"-3", []string{clusterName + "-4"}),
			makeAggregateClusterResource(clusterName+"-4", []string{clusterName + "-5"}),
			makeAggregateClusterResource(clusterName+"-5", []string{clusterName + "-6"}),
			makeAggregateClusterResource(clusterName+"-6", []string{clusterName + "-7"}),
			makeAggregateClusterResource(clusterName+"-7", []string{clusterName + "-8"}),
			makeAggregateClusterResource(clusterName+"-8", []string{clusterName + "-9"}),
			makeAggregateClusterResource(clusterName+"-9", []string{clusterName + "-10"}),
			makeAggregateClusterResource(clusterName+"-10", []string{clusterName + "-11"}),
			makeAggregateClusterResource(clusterName+"-11", []string{clusterName + "-12"}),
			makeAggregateClusterResource(clusterName+"-12", []string{clusterName + "-13"}),
			makeAggregateClusterResource(clusterName+"-13", []string{clusterName + "-14"}),
			makeAggregateClusterResource(clusterName+"-14", []string{clusterName + "-15"}),
			e2e.DefaultCluster(clusterName+"-15", serviceName, e2e.SecurityLevelNone),
		},
		Endpoints:      []*v3endpointpb.ClusterLoadAssignment{e2e.DefaultEndpoint(serviceName, "localhost", []uint32{testutils.ParsePort(t, server.Address)})},
		SkipValidation: true,
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Verify that a successful RPC can be made.
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
}

func TestContains(t *testing.T) {
	containsTests := []struct {
		name  string
		elems []string
		elem  string
		out   bool
	}{
		{"exists", []string{"1", "2", "3"}, "1", true},
		{"not exists", []string{"1", "2", "3"}, "4", false},
	}
	for _, test := range containsTests {
		t.Run(test.name, func(t *testing.T) {
			if out := Contains(test.elems, test.elem); test.out != out {
				t.Errorf("Contains(%v, %s) want: %t, got: %t", test.elems, test.elem, test.out, out)
			}
		})
	}
}

