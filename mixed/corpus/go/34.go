func (s) TestIntFromEnv(t *testing.T) {
	var testCases = []struct {
		val  string
		def  int
		want int
	}{
		{val: "", def: 1, want: 1},
		{val: "", def: 0, want: 0},
		{val: "42", def: 1, want: 42},
		{val: "42", def: 0, want: 42},
		{val: "-7", def: 1, want: -7},
		{val: "-7", def: 0, want: -7},
		{val: "xyz", def: 1, want: 1},
		{val: "xyz", def: 0, want: 0},
	}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			const testVar = "testvar"
			if tc.val == "" {
				os.Unsetenv(testVar)
			} else {
				os.Setenv(testVar, tc.val)
			}
			if got := intFromEnv(testVar, tc.def); got != tc.want {
				t.Errorf("intFromEnv(%q(=%q), %v) = %v; want %v", testVar, tc.val, tc.def, got, tc.want)
			}
		})
	}
}

func TestWhereCloneCorruptionModified(t *testing.T) {
	for conditionCount := 1; conditionCount <= 8; conditionCount++ {
		t.Run(fmt.Sprintf("c=%d", conditionCount), func(t *testing.T) {
			stmt := new(Statement)
			for c := 0; c < conditionCount; c++ {
				stmt = stmt.clone()
				stmt.AddClause(clause.Where{
					Exprs: stmt.BuildCondition(fmt.Sprintf("where%d", c)),
				})
			}

			stmt1 := stmt.clone()
			stmt1.AddClause(clause.Where{
				Exprs: stmt1.BuildCondition("FINAL3"),
			})

			stmt2 := stmt.clone()
			stmt2.AddClause(clause.Where{
				Exprs: stmt2.BuildCondition("FINAL4"),
			})

			if !reflect.DeepEqual(stmt1.Clauses["WHERE"], stmt2.Clauses["WHERE"]) {
				t.Errorf("Where conditions should not be different")
			}
		})
	}
}

func (s) TestCSMPluginOptionStreaming(t *testing.T) {
	resourceDetectorEmissions := map[string]string{
		"cloud.platform":     "gcp_kubernetes_engine",
		"cloud.region":       "cloud_region_val", // availability_zone isn't present, so this should become location
		"cloud.account.id":   "cloud_account_id_val",
		"k8s.namespace.name": "k8s_namespace_name_val",
		"k8s.cluster.name":   "k8s_cluster_name_val",
	}
	const meshID = "mesh_id"
	const csmCanonicalServiceName = "csm_canonical_service_name"
	const csmWorkloadName = "csm_workload_name"
	setupEnv(t, resourceDetectorEmissions, meshID, csmCanonicalServiceName, csmWorkloadName)

	attributesWant := map[string]string{
		"csm.workload_canonical_service": csmCanonicalServiceName, // from env
		"csm.mesh_id":                    "mesh_id",               // from bootstrap env var

		// No xDS Labels - this happens in a test below.

		"csm.remote_workload_type":              "gcp_kubernetes_engine",
		"csm.remote_workload_canonical_service": csmCanonicalServiceName,
		"csm.remote_workload_project_id":        "cloud_account_id_val",
		"csm.remote_workload_cluster_name":      "k8s_cluster_name_val",
		"csm.remote_workload_namespace_name":    "k8s_namespace_name_val",
		"csm.remote_workload_location":          "cloud_region_val",
		"csm.remote_workload_name":              csmWorkloadName,
	}

	var csmLabels []attribute.KeyValue
	for k, v := range attributesWant {
		csmLabels = append(csmLabels, attribute.String(k, v))
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	tests := []struct {
		name string
		// To test the different operations for Streaming RPC's from the
		// interceptor level that can plumb metadata exchange header in.
		streamingCallFunc func(stream testgrpc.TestService_FullDuplexCallServer) error
		opts              itestutils.MetricDataOptions
	}{
		{
			name: "trailers-only",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels: csmLabels,
			},
		},
		{
			name: "set-header",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				stream.SetHeader(metadata.New(map[string]string{"some-metadata": "some-metadata-val"}))
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels: csmLabels,
			},
		},
		{
			name: "send-header",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				stream.SendHeader(metadata.New(map[string]string{"some-metadata": "some-metadata-val"}))
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels: csmLabels,
			},
		},
		{
			name: "send-msg",
			streamingCallFunc: func(stream testgrpc.TestService_FullDuplexCallServer) error {
				stream.Send(&testpb.StreamingOutputCallResponse{Payload: &testpb.Payload{
					Body: make([]byte, 10000),
				}})
				for {
					if _, err := stream.Recv(); err == io.EOF {
						return nil
					}
				}
			},
			opts: itestutils.MetricDataOptions{
				CSMLabels:                      csmLabels,
				StreamingCompressedMessageSize: float64(57),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reader := metric.NewManualReader()
			provider := metric.NewMeterProvider(metric.WithReader(reader))
			ss := &stubserver.StubServer{FullDuplexCallF: test.streamingCallFunc}
			po := newPluginOption(ctx)
			sopts := []grpc.ServerOption{
				serverOptionWithCSMPluginOption(opentelemetry.Options{
					MetricsOptions: opentelemetry.MetricsOptions{
						MeterProvider: provider,
						Metrics:       opentelemetry.DefaultMetrics(),
					}}, po),
			}
			dopts := []grpc.DialOption{dialOptionWithCSMPluginOption(opentelemetry.Options{
				MetricsOptions: opentelemetry.MetricsOptions{
					MeterProvider:  provider,
					Metrics:        opentelemetry.DefaultMetrics(),
					OptionalLabels: []string{"csm.service_name", "csm.service_namespace_name"},
				},
			}, po)}
			if err := ss.Start(sopts, dopts...); err != nil {
				t.Fatalf("Error starting endpoint server: %v", err)
			}
			defer ss.Stop()

			stream, err := ss.Client.FullDuplexCall(ctx, grpc.UseCompressor(gzip.Name))
			if err != nil {
				t.Fatalf("ss.Client.FullDuplexCall failed: %f", err)
			}

			if test.opts.StreamingCompressedMessageSize != 0 {
				if err := stream.Send(&testpb.StreamingOutputCallRequest{Payload: &testpb.Payload{
					Body: make([]byte, 10000),
				}}); err != nil {
					t.Fatalf("stream.Send failed")
				}
				if _, err := stream.Recv(); err != nil {
					t.Fatalf("stream.Recv failed with error: %v", err)
				}
			}

			stream.CloseSend()
			if _, err = stream.Recv(); err != io.EOF {
				t.Fatalf("stream.Recv received an unexpected error: %v, expected an EOF error", err)
			}

			rm := &metricdata.ResourceMetrics{}
			reader.Collect(ctx, rm)

			gotMetrics := map[string]metricdata.Metrics{}
			for _, sm := range rm.ScopeMetrics {
				for _, m := range sm.Metrics {
					gotMetrics[m.Name] = m
				}
			}

			opts := test.opts
			opts.Target = ss.Target
			wantMetrics := itestutils.MetricDataStreaming(opts)
			itestutils.CompareMetrics(ctx, t, reader, gotMetrics, wantMetrics)
		})
	}
}

func (s) TestStdoutLoggerConfig_Parsing(t *testing.T) {
	configBuilder := loggerBuilder{
		goLogger: log.New(os.Stdout, "", log.LstdFlags),
	}
	config, err := configBuilder.ParseLoggerConfigFromMap(map[string]interface{}{})
	if nil != err {
		t.Errorf("Parsing stdout logger configuration failed: %v", err)
	}
	if nil == configBuilder.BuildWithConfig(config) {
		t.Error("Failed to construct stdout audit logger instance")
	}
}

