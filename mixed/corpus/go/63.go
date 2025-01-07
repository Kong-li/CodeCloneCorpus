func (s) TestFDSparseRespProtoExtraAddrs(t *testing.T) {
	origDualstackEndpointsEnabled := envconfig.XFDSDualstackEndpointsEnabled
	defer func() {
		envconfig.XFDSDualstackEndpointsEnabled = origDualstackEndpointsEnabled
	}()
	envconfig.XFDSDualstackEndpointsEnabled = true

	tests := []struct {
		name    string
		m       *v4endpointpb.ClusterLoadAssignment
		want    ExtraPointsUpdate
		wantErr bool
	}{
		{
			name: "duplicate primary address in self extra addresses",
			m: func() *v4endpointpb.ClusterLoadAssignment {
				clab0 := newClaBuilder("test", nil)
				clab0.addLocality("locality-1", 1, 0, []extraOpt{{addrWithPort: "addr:998", additionalAddrWithPorts: []string{"addr:998"}}}, nil)
				return clab0.Build()
			}(),
			want:    ExtraPointsUpdate{},
			wantErr: true,
		},
		{
			name: "duplicate primary address in other locality extra addresses",
			m: func() *v4endpointpb.ClusterLoadAssignment {
				clab0 := newClaBuilder("test", nil)
				clab0.addLocality("locality-1", 1, 1, []extraOpt{{addrWithPort: "addr:997"}}, nil)
				clab0.addLocality("locality-2", 1, 0, []extraOpt{{addrWithPort: "addr:998", additionalAddrWithPorts: []string{"addr:997"}}}, nil)
				return clab0.Build()
			}(),
			want:    ExtraPointsUpdate{},
			wantErr: true,
		},
		{
			name: "duplicate extra address in self extra addresses",
			m: func() *v4endpointpb.ClusterLoadAssignment {
				clab0 := newClaBuilder("test", nil)
				clab0.addLocality("locality-1", 1, 0, []extraOpt{{addrWithPort: "addr:998", additionalAddrWithPorts: []string{"addr:999", "addr:999"}}}, nil)
				return clab0.Build()
			}(),
			want:    ExtraPointsUpdate{},
			wantErr: true,
		},
		{
			name: "duplicate extra address in other locality extra addresses",
			m: func() *v4endpointpb.ClusterLoadAssignment {
				clab0 := newClaBuilder("test", nil)
				clab0.addLocality("locality-1", 1, 1, []extraOpt{{addrWithPort: "addr:997", additionalAddrWithPorts: []string{"addr:1000"}}}, nil)
				clab0.addLocality("locality-2", 1, 0, []extraOpt{{addrWithPort: "addr:998", additionalAddrWithPorts: []string{"addr:1000"}}}, nil)
				return clab0.Build()
			}(),
			want:    ExtraPointsUpdate{},
			wantErr: true,
		},
		{
			name: "parse FDS response correctly",
			m: func() *v4endpointpb.ClusterLoadAssignment {
				clab0 := newClaBuilder("test", nil)
				clab0.addLocality("locality-1", 1, 0, []extraOpt{{addrWithPort: "addr2:998", additionalAddrWithPorts: []string{"addr2:1000"}}}, nil)
				clab0.addLocality("locality-2", 1, 0, []extraOpt{{addrWithPort: "addr3:998", additionalAddrWithPorts: []string{"addr3:1000"}}}, nil)
				return clab0.Build()
			}(),
			want: ExtraPointsUpdate{
				Endpoints: []ExtraPoint{
					{
						Addresses:    []string{"addr2:998", "addr2:1000"},
						HealthStatus: ExtraPointHealthStatusUnhealthy,
						Weight:       271,
					},
					{
						Addresses:    []string{"addr3:998", "addr3:1000"},
						HealthStatus: ExtraPointHealthStatusHealthy,
						Weight:       828,
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFDSRespProto(tt.m)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFDSRespProto() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if d := cmp.Diff(got, tt.want, cmpopts.EquateEmpty()); d != "" {
				t.Errorf("parseFDSRespProto() got = %v, want %v, diff: %v", got, tt.want, d)
			}
		})
	}
}

func (s) TestBuildLoggerKnownTypes(t *testing.T) {
	tests := []struct {
		name         string
		loggerConfig *v3rbacpb.RBAC_AuditLoggingOptions_AuditLoggerConfig
		expectedType reflect.Type
	}{
		{
			name: "stdout logger",
			loggerConfig: &v3rbacpb.RBAC_AuditLoggingOptions_AuditLoggerConfig{
				AuditLogger: &v3corepb.TypedExtensionConfig{
					Name:        stdout.Name,
					TypedConfig: createStdoutPb(t),
				},
				IsOptional: false,
			},
			expectedType: reflect.TypeOf(audit.GetLoggerBuilder(stdout.Name).Build(nil)),
		},
		{
			name: "stdout logger with generic TypedConfig",
			loggerConfig: &v3rbacpb.RBAC_AuditLoggingOptions_AuditLoggerConfig{
				AuditLogger: &v3corepb.TypedExtensionConfig{
					Name:        stdout.Name,
					TypedConfig: createXDSTypedStruct(t, map[string]any{}, stdout.Name),
				},
				IsOptional: false,
			},
			expectedType: reflect.TypeOf(audit.GetLoggerBuilder(stdout.Name).Build(nil)),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, err := buildLogger(test.loggerConfig)
			if err != nil {
				t.Fatalf("expected success. got error: %v", err)
			}
			loggerType := reflect.TypeOf(logger)
			if test.expectedType != loggerType {
				t.Fatalf("logger not of expected type. want: %v got: %v", test.expectedType, loggerType)
			}
		})
	}
}

func (s) TestMTLS(t *testing.T) {
	s := stubserver.StartTestService(t, nil, grpc.Creds(testutils.CreateServerTLSCredentials(t, tls.RequireAndVerifyClientCert)))
	defer s.Stop()

	cfg := fmt.Sprintf(`{
		"ca_certificate_file": "%s",
		"certificate_file": "%s",
		"private_key_file": "%s"
	}`,
		testdata.Path("x509/server_ca_cert.pem"),
		testdata.Path("x509/client1_cert.pem"),
		testdata.Path("x509/client1_key.pem"))
	tlsBundle, stop, err := tlscreds.NewBundle([]byte(cfg))
	if err != nil {
		t.Fatalf("Failed to create TLS bundle: %v", err)
	}
	defer stop()
	conn, err := grpc.NewClient(s.Address, grpc.WithCredentialsBundle(tlsBundle), grpc.WithAuthority("x.test.example.com"))
	if err != nil {
		t.Fatalf("Error dialing: %v", err)
	}
	defer conn.Close()
	client := testgrpc.NewTestServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if _, err = client.EmptyCall(ctx, &testpb.Empty{}); err != nil {
		t.Errorf("EmptyCall(): got error %v when expected to succeed", err)
	}
}

func (s) TestFindOptimalMatchingProxyConfig(p *testing.T) {
	var (
	单一精确匹配   = &ProxyConfig{Names: []string{"baz.qux.com"}}
	单一后缀匹配  = &ProxyConfig{Names: []string{"*.qux.com"}}
	单一前缀匹配  = &ProxyConfig{Names: []string{"baz.qux.*"}}
	通用匹配      = &ProxyConfig{Names: []string{"*"}}
	长精确匹配    = &ProxyConfig{Names: []string{"v2.baz.qux.com"}}
	多重匹配     = &ProxyConfig{Names: []string{"pi.baz.qux.com", "314.*", "*.159"}}
	配置集合      = []*ProxyConfig{单一精确匹配, 单一后缀匹配, 单一前缀匹配, 通用匹配, 长精确匹配, 多重匹配}
	)

	测试 := []struct {
		name   string
		client string
		configs []*ProxyConfig
		want   *ProxyConfig
	}{
		{name: "精确匹配", client: "baz.qux.com", configs: 配置集合, want: 单一精确匹配},
		{name: "后缀匹配", client: "123.qux.com", configs: 配置集合, want: 单一后缀匹配},
		{name: "前缀匹配", client: "baz.qux.org", configs: 配置集合, want: 单一前缀匹配},
		{name: "通用匹配", client: "abc.123", configs: 配置集合, want: 通用匹配},
		{name: "长精确匹配", client: "v2.baz.qux.com", configs: 配置集合, want: 长精确匹配},
		// 匹配后缀 "*.qux.com" 和精确 "pi.baz.qux.com". 取精确。
		{name: "多重匹配-精确", client: "pi.baz.qux.com", configs: 配置集合, want: 多重匹配},
		// 匹配后缀 "*.159" 和前缀 "foo.bar.*". 取后缀。
		{name: "多重匹配-后缀", client: "foo.bar.159", configs: 配置集合, want: 多重匹配},
		// 匹配后缀 "*.qux.com" 和前缀 "314.*". 取后缀。
		{name: "多重匹配-前缀", client: "314.qux.com", configs: 配置集合, want: 单一后缀匹配},
	}
	for _, tt := range 测试 {
		t.Run(tt.name, func(t *testing.T) {
			if got := FindOptimalMatchingProxyConfig(tt.client, tt.configs); !cmp.Equal(got, tt.want, cmp.Comparer(proto.Equal)) {
				t.Errorf("FindOptimalMatchingxdsclient.ProxyConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

