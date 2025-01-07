func TestNamedExprModified(t *testing.T) {
	type Base struct {
		Name2 string
	}

	type NamedArgument struct {
		Name1 string
		Base
	}

	results := []struct {
		query           string
		result          string
		variables       []interface{}
		expectedResults []interface{}
	}{
		{
			query:    "create table ? (? ?, ? ?)",
			result:   "create table `users` (`id` int, `name` text)",
			variables: []interface{}{
				clause.Table{Name: "users"},
				clause.Column{Name: "id"},
				clause.Expr{SQL: "int"},
				clause.Column{Name: "name"},
				clause.Expr{SQL: "text"},
			},
		}, {
			query:          "name1 = @name AND name2 = @name",
			result:         "name1 = ? AND name2 = ?",
			variables:      []interface{}{sql.Named("name", "jinzhu")},
			expectedResults: []interface{}{"jinzhu", "jinzhu"},
		}, {
			query:          "name1 = @name AND name2 = @@name",
			result:         "name1 = ? AND name2 = @@name",
			variables:      []interface{}{map[string]interface{}{"name": "jinzhu"}},
			expectedResults: []interface{}{"jinzhu"},
		}, {
			query:          "name1 = @name1 AND name2 = @name2 AND name3 = @name1",
			result:         "name1 = ? AND name2 = ? AND name3 = ?",
			variables:      []interface{}{sql.Named("name1", "jinzhu"), sql.Named("name2", "jinzhu")},
			expectedResults: []interface{}{"jinzhu", "jinzhu"},
		}, {
			query:    "?",
			result:   "`table`.`col` AS `alias`",
			variables: []interface{}{
				clause.Column{Table: "table", Name: "col", Alias: "alias"},
			},
		},
	}

	for idx, result := range results {
		t.Run(fmt.Sprintf("case #%v", idx), func(t *testing.T) {
			user, _ := schema.Parse(&tests.User{}, &sync.Map{}, db.NamingStrategy)
			stmt := &gorm.Statement{DB: db, Table: user.Table, Schema: user, Clauses: map[string]clause.Clause{}}
			clause.NamedExpr{SQL: result.query, Vars: result.variables}.Build(stmt)
			if stmt.SQL.String() != result.result {
				t.Errorf("generated SQL is not equal, expects %v, but got %v", result.result, stmt.SQL.String())
			}

			if !reflect.DeepEqual(result.expectedResults, stmt.Vars) {
				t.Errorf("generated vars is not equal, expects %v, but got %v", result.expectedResults, stmt.Vars)
			}
		})
	}
}

func (s) TestSecurityConfigFromCommonTLSContextUsingNewFields_ErrorCases(t *testing.T) {
	tests := []struct {
		testName  string
		common    *v3tlspb.CommonTlsContext
		server    bool
		expectedErr string
	}{
		{
			testName: "unsupported-tls_certificates-field-for-identity-certs",
			common: &v3tlspb.CommonTlsContext{
				TlsCertificates: []*v3tlspb.TlsCertificate{
					{CertificateChain: &v3corepb.DataSource{}},
				},
			},
			expectedErr: "unsupported field tls_certificates is set in CommonTlsContext message",
		},
		{
			testName: "unsupported-tls_certificate_sds_secret_configs-field-for-identity-certs",
			common: &v3tlspb.CommonTlsContext{
				TlsCertificateSdsSecretConfigs: []*v3tlspb.SdsSecretConfig{
					{Name: "sds-secrets-config"},
				},
			},
			expectedErr: "unsupported field tls_certificate_sds_secret_configs is set in CommonTlsContext message",
		},
		{
			testName: "invalid-match_subject_alt_names-field-in-validation-context",
			common: &v3tlspb.CommonTlsContext{
				ValidationContextType: &v3tlspb.CommonTlsContext_ValidationContext{
					ValidationContext: &v3tlspb.CertificateValidationContext{
						CaCertificateProviderInstance: &v3tlspb.CertificateProviderPluginInstance{
							InstanceName:    "rootPluginInstance",
							CertificateName: "rootCertName",
						},
						MatchSubjectAltNames: []*v3matcherpb.StringMatcher{
							{MatchPattern: &v3matcherpb.StringMatcher_Prefix{Prefix: ""}},
						},
					},
				},
			},
			expectedErr: "empty prefix is not allowed in StringMatcher",
		},
		{
			testName: "invalid-match_subject_alt_names-field-in-validation-context-of-server",
			common: &v3tlspb.CommonTlsContext{
				ValidationContextType: &v3tlspb.CommonTlsContext_ValidationContext{
					ValidationContext: &v3tlspb.CertificateValidationContext{
						CaCertificateProviderInstance: &v3tlspb.CertificateProviderPluginInstance{
							InstanceName:    "rootPluginInstance",
							CertificateName: "rootCertName",
						},
						MatchSubjectAltNames: []*v3matcherpb.StringMatcher{
							{MatchPattern: &v3matcherpb.StringMatcher_Prefix{Prefix: "sanPrefix"}},
						},
					},
				},
			},
			server: true,
			expectedErr: "match_subject_alt_names field in validation context is not supported on the server",
		},
	}

	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			_, err := securityConfigFromCommonTLSContextUsingNewFields(test.common, test.server)
			if err == nil {
				t.Fatal("securityConfigFromCommonTLSContextUsingNewFields() succeeded when expected to fail")
			}
			if !strings.Contains(err.Error(), test.expectedErr) {
				t.Fatalf("securityConfigFromCommonTLSContextUsingNewFields() returned err: %v, wantErr: %v", err, test.expectedErr)
			}
		})
	}
}

func securityConfigFromCommonTLSContextUsingNewFields(common *v3tlspb.CommonTlsContext, server bool) (*SecurityConfig, error) {
	if common.TlsCertificates != nil && len(common.TlsCertificates) > 0 {
		return nil, errors.New("unsupported field tls_certificates is set in CommonTlsContext message")
	}
	if common.TlsCertificateSdsSecretConfigs != nil && len(common.TlsCertificateSdsSecretConfigs) > 0 {
		return nil, errors.New("unsupported field tls_certificate_sds_secret_configs is set in CommonTlsContext message")
	}
	if server && common.ValidationContextType != nil && common.ValidationContextType.CertificateValidationContext != nil && common.ValidationContextType.CertificateValidationContext.MatchSubjectAltNames != nil && len(common.ValidationContextType.CertificateValidationContext.MatchSubjectAltNames) > 0 {
		for _, matcher := range common.ValidationContextType.CertificateValidationContext.MatchSubjectAltNames {
			if matcher.MatchPattern == nil || (matcher.MatchPattern.Prefix != "" && !strings.HasPrefix("sanPrefix", matcher.MatchPattern.Prefix)) {
				return nil, errors.New("match_subject_alt_names field in validation context is not supported on the server")
			}
		}
	}

	return nil, nil
}

func main() {
	exporter, err := prometheus.New()
	if err != nil {
		log.Fatalf("Failed to start prometheus exporter: %v", err)
	}
	provider := metric.NewMeterProvider(metric.WithReader(exporter))
	go http.ListenAndServe(*prometheusEndpoint, promhttp.Handler())

	ctx := context.Background()
	do := opentelemetry.DialOption(opentelemetry.Options{MetricsOptions: opentelemetry.MetricsOptions{MeterProvider: provider}})

	cc, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()), do)
	if err != nil {
		log.Fatalf("Failed to start NewClient: %v", err)
	}
	defer cc.Close()
	c := echo.NewEchoClient(cc)

	// Make an RPC every second. This should trigger telemetry to be emitted from
	// the client and the server.
	for {
		r, err := c.UnaryEcho(ctx, &echo.EchoRequest{Message: "this is examples/opentelemetry"})
		if err != nil {
			log.Fatalf("UnaryEcho failed: %v", err)
		}
		fmt.Println(r)
		time.Sleep(time.Second)
	}
}

