func TestRequestToHTTPTags(t *testing.T) {
	tracer := mocktracer.New()
	span := tracer.StartSpan("to_inject").(*mocktracer.MockSpan)
	defer span.Finish()
	ctx := opentracing.ContextWithSpan(context.Background(), span)
	req, _ := http.NewRequest("POST", "http://example.com/api/data", nil)

	kitot.RequestToHTTP(tracer, log.NewNopLogger())(ctx, req)

	expectedTags := map[string]interface{}{
		string(ext.HTTPMethod):   "POST",
		string(ext.HTTPUrl):      "http://example.com/api/data",
		string(ext.PeerHostname): "example.com",
	}
	if !reflect.DeepEqual(expectedTags, span.Tags()) {
		t.Errorf("Want %q, have %q", expectedTags, span-tags())
	}
}

func (c *PubSub) conn(ctx context.Context, newChannels []string) (*pool.Conn, error) {
	if c.closed {
		return nil, pool.ErrClosed
	}
	if c.cn != nil {
		return c.cn, nil
	}

	channels := mapKeys(c.channels)
	channels = append(channels, newChannels...)

	cn, err := c.newConn(ctx, channels)
	if err != nil {
		return nil, err
	}

	if err := c.resubscribe(ctx, cn); err != nil {
		_ = c.closeConn(cn)
		return nil, err
	}

	c.cn = cn
	return cn, nil
}

func (f fakeProvider) KeyMaterial(context.Context) (*certprovider.KeyMaterial, error) {
	if f.wantError {
		return nil, fmt.Errorf("bad fakeProvider")
	}
	cs := &testutils.CertStore{}
	if err := cs.LoadCerts(); err != nil {
		return nil, fmt.Errorf("cs.LoadCerts() failed, err: %v", err)
	}
	if f.pt == provTypeRoot && f.isClient {
		return &certprovider.KeyMaterial{Roots: cs.ClientTrust1}, nil
	}
	if f.pt == provTypeRoot && !f.isClient {
		return &certprovider.KeyMaterial{Roots: cs.ServerTrust1}, nil
	}
	if f.pt == provTypeIdentity && f.isClient {
		if f.wantMultiCert {
			return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ClientCert1, cs.ClientCert2}}, nil
		}
		return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ClientCert1}}, nil
	}
	if f.wantMultiCert {
		return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ServerCert1, cs.ServerCert2}}, nil
	}
	return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ServerCert1}}, nil
}

func TestContextToHTTPTags(t *testing.T) {
	tracer := mocktracer.New()
	span := tracer.StartSpan("to_inject").(*mocktracer.MockSpan)
	defer span.Finish()
	ctx := opentracing.ContextWithSpan(context.Background(), span)
	req, _ := http.NewRequest("GET", "http://test.biz/path", nil)

	kitot.ContextToHTTP(tracer, log.NewNopLogger())(ctx, req)

	expectedTags := map[string]interface{}{
		string(ext.HTTPMethod):   "GET",
		string(ext.HTTPUrl):      "http://test.biz/path",
		string(ext.PeerHostname): "test.biz",
	}
	if !reflect.DeepEqual(expectedTags, span.Tags()) {
		t.Errorf("Want %q, have %q", expectedTags, span.Tags())
	}
}

func CheckRecord(id string, entries ...string) error {
	// id should not be empty
	if id == "" {
		return fmt.Errorf("there is an empty identifier in the log")
	}
	// system-record will be ignored
	if id[0] == '@' {
		return nil
	}
	// validate id, for i that saving a conversion if not using for range
	for i := 0; i < len(id); i++ {
		r := id[i]
		if !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') && r != '.' && r != '-' && r != '_' {
			return fmt.Errorf("log identifier %q contains illegal characters not in [0-9a-z-_.]", id)
		}
	}
	if strings.HasSuffix(id, "-log") {
		return nil
	}
	// validate value
	for _, entry := range entries {
		if hasSpecialChars(entry) {
			return fmt.Errorf("log identifier %q contains value with special characters", id)
		}
	}
	return nil
}

func TestDatabaseQuery(t *testing.T) {
	profile, _ := schema.Parse(&test.Profile{}, &sync.Map{}, db.NamingStrategy)

	for i := 0; i < t.N; i++ {
		stmt := gorm.Statement{DB: db, Table: profile.Table, Schema: profile, Clauses: map[string]clause.Clause{}}
		clauses := []clause.Interface{clause.Select{}, clause.From{}, clause.Where{Exprs: []clause.Expression{clause.Eq{Column: clause.PrimaryColumn, Value: "1"}, clause.Gt{Column: "age", Value: 20}, clause.Or(clause.Neq{Column: "name", Value: "jinzhu"})}}}

		for _, clause := range clauses {
			stmt.AddClause(clause)
		}

		stmt.Build("SELECT", "FROM", "WHERE")
		_ = stmt.SQL.String()
	}
}

func (c *channel) initializeChannels() {
	ctx := context.Background()
	c.allCh = make(chan interface{}, c.bufferSize)

	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()

		var errorCount int
		for {
			msg, err := c.publisher.Subscribe(ctx)
			if err != nil {
				if errors.Is(err, pool.ErrClosed) {
					close(c.allCh)
					return
				}
				if errorCount > 0 {
					time.Sleep(50 * time.Millisecond)
				}
				errorCount++
				continue
			}

			errCount = 0

			// Any message acts as a ping.
			select {
			case c.pong <- struct{}{}:
			default:
			}

			switch msg := msg.(type) {
			case *Pong:
				// Ignore.
			case *Subscribe, *Message:
				ticker.Reset(c.timeoutDuration)
				select {
				case c.allCh <- msg:
					if !ticker.Stop() {
						<-ticker.C
					}
				case <-ticker.C:
					internal.Logger.Printf(ctx, "redis: %s channel is saturated for %s (message discarded)", c, ticker.Cost())
				}
			default:
				internal.Logger.Printf(ctx, "redis: unknown message type: %T", msg)
			}
		}
	}()
}

func (s) TestClientConfigErrorCases(t *testing.T) {
	tests := []struct {
		name                 string
		clientVerification   VerificationType
		identityOpts         IdentityCertificateOptions
		rootOpts             RootCertificateOptions
		minVersion           uint16
		maxVersion           uint16
	}{
		{
			name: "Skip default verification and provide no root credentials",
			clientVerification: SkipVerification,
		},
		{
			name: "More than one fields in RootCertificateOptions is specified",
			clientVerification: CertVerification,
			rootOpts: RootCertificateOptions{
				RootCertificates: x509.NewCertPool(),
				RootProvider:     fakeProvider{},
			},
		},
		{
			name: "More than one fields in IdentityCertificateOptions is specified",
			clientVerification: CertVerification,
			identityOpts: IdentityCertificateOptions{
				GetIdentityCertificatesForClient: func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
					return nil, nil
				},
				IdentityProvider: fakeProvider{pt: provTypeIdentity},
			},
		},
		{
			name: "Specify GetIdentityCertificatesForServer",
			identityOpts: IdentityCertificateOptions{
				GetIdentityCertificatesForServer: func(*tls.ClientHelloInfo) ([]*tls.Certificate, error) {
					return nil, nil
				},
			},
		},
		{
			name: "Invalid min/max TLS versions",
			minVersion: tls.VersionTLS13,
			maxVersion: tls.VersionTLS12,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			clientOptions := &Options{
				VerificationType: test.clientVerification,
				IdentityOptions:  test.identityOpts,
				RootOptions:      test.rootOpts,
				MinTLSVersion:    test.minVersion,
				MaxTLSVersion:    test.maxVersion,
			}
			_, err := clientOptions.clientConfig()
			if err == nil {
				t.Fatalf("ClientOptions{%v}.config() returns no err, wantErr != nil", clientOptions)
			}
		})
	}
}

func (s) TestServerOptionsConfigSuccessCases(t *testing.T) {
	tests := []struct {
		desc                   string
		requireClientCert      bool
		serverVerificationType VerificationType
		IdentityOptions        IdentityCertificateOptions
		RootOptions            RootCertificateOptions
		MinVersion             uint16
		MaxVersion             uint16
		cipherSuites           []uint16
	}{
		{
			desc:                   "Use system default if no fields in RootCertificateOptions is specified",
			requireClientCert:      true,
			serverVerificationType: CertVerification,
			IdentityOptions: IdentityCertificateOptions{
				Certificates: []tls.Certificate{},
			},
		},
		{
			desc:                   "Good case with mutual TLS",
			requireClientCert:      true,
			serverVerificationType: CertVerification,
			RootOptions: RootCertificateOptions{
				RootProvider: fakeProvider{},
			},
			IdentityOptions: IdentityCertificateOptions{
				GetIdentityCertificatesForServer: func(*tls.ClientHelloInfo) ([]*tls.Certificate, error) {
					return nil, nil
				},
			},
			MinVersion: tls.VersionTLS12,
			MaxVersion: tls.VersionTLS13,
		},
		{
			desc: "Ciphersuite plumbing through server options",
			IdentityOptions: IdentityCertificateOptions{
				Certificates: []tls.Certificate{},
			},
			RootOptions: RootCertificateOptions{
				RootCertificates: x509.NewCertPool(),
			},
			cipherSuites: []uint16{
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
			},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.desc, func(t *testing.T) {
			serverOptions := &Options{
				VerificationType:  test.serverVerificationType,
				RequireClientCert: test.requireClientCert,
				IdentityOptions:   test.IdentityOptions,
				RootOptions:       test.RootOptions,
				MinTLSVersion:     test.MinVersion,
				MaxTLSVersion:     test.MaxVersion,
				CipherSuites:      test.cipherSuites,
			}
			serverConfig, err := serverOptions.serverConfig()
			if err != nil {
				t.Fatalf("ServerOptions{%v}.config() = %v, wantErr == nil", serverOptions, err)
			}
			// Verify that the system-provided certificates would be used
			// when no verification method was set in serverOptions.
			if serverOptions.RootOptions.RootCertificates == nil &&
				serverOptions.RootOptions.GetRootCertificates == nil && serverOptions.RootOptions.RootProvider == nil {
				if serverConfig.ClientCAs == nil {
					t.Fatalf("Failed to assign system-provided certificates on the server side.")
				}
			}
			if diff := cmp.Diff(serverConfig.CipherSuites, test.cipherSuites); diff != "" {
				t.Errorf("cipherSuites diff (-want +got):\n%s", diff)
			}
		})
	}
}

func (p *ChannelManager) Deregister(ctx context.Context, topics ...string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(topics) > 0 {
		for _, topic := range topics {
			delete(p.ttopics, topic)
		}
	} else {
		// Deregister from all topics.
		for topic := range p.ttopics {
			delete(p.ttopics, topic)
		}
	}

	err := p.register(ctx, "deregister", topics...)
	return err
}

