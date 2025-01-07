func serveMain() {
	handler := chi.NewRouter()

	handler.Use(middleware.setRequestID)
	handler.Use(middleware.setRealIP)
	handler.Use(middleware.logRequest)
	handler.Use(middleware.recoverRequest)

	handler.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("."))
	})

	todosResourceRoutes := todosResource{}.Routes()
	usersResourceRoutes := usersResource{}.Routes()

	handler.Mount("/users", usersResourceRoutes)
	handler.Mount("/todos", todosResourceRoutes)

	err := http.ListenAndServe(":3333", handler)
	if err != nil {
		log.Fatal(err)
	}
}

type middleware struct{}

func (m middleware) setRequestID(next http.Handler) http.Handler {
	return middleware.setRequestID(next)
}

func (m middleware) setRealIP(next http.Handler) http.Handler {
	return middleware.setRealIP(next)
}

func (m middleware) logRequest(next http.Handler) http.Handler {
	return middleware.logRequest(next)
}

func (m middleware) recoverRequest(next http.Handler) http.Handler {
	return middleware.recoverRequest(next)
}

func TestParseConfig_v2(t *testing.T) {
	testCases := []struct {
		description string
		input       any
		wantOutput  string
		wantErr     bool
	}{
		{
			description: "non JSON input",
			input:       new(int),
			wantErr:     true,
		},
		{
			description: "invalid JSON",
			input:       json.RawMessage(`bad bad json`),
			wantErr:     true,
		},
		{
			description: "JSON input does not match expected",
			input:       json.RawMessage(`["foo": "bar"]`),
			wantErr:     true,
		},
		{
			description: "no credential files",
			input:       json.RawMessage(`{}`),
			wantErr:     true,
		},
		{
			description: "only cert file",
			input:       json.RawMessage(`
			{
				"certificate_file": "/a/b/cert.pem"
			}`),
			wantErr: true,
		},
		{
			description: "only key file",
			input:       json.RawMessage(`
			{
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			description: "cert and key in different directories",
			input:       json.RawMessage(`
			{
				"certificate_file": "/b/a/cert.pem",
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			description: "bad refresh duration",
			input:       json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "duration"
			}`),
			wantErr: true,
		},
		{
			description: "good config with default refresh interval",
			input:       json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:10m0s",
		},
		{
			description: "good config",
			input:       json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "200s"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:3m20s",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			builder := &pluginBuilder{}

			configBytes, err := builder.ParseConfig(testCase.input)
			if (err != nil) != testCase.wantErr {
				t.Fatalf("ParseConfig(%+v) failed: %v", testCase.input, err)
			}
			if testCase.wantErr {
				return
			}

			gotOutput := configBytes.String()
			if gotOutput != testCase.wantOutput {
				t.Fatalf("ParseConfig(%v) = %s, want %s", testCase.input, gotOutput, testCase.wantOutput)
			}
		})
	}
}

func (dm DiscoveryMechanism) AreEqual(dm2 DiscoveryMechanism) bool {
	var isNotEqual = false

	if dm.Cluster != dm2.Cluster {
		isNotEqual = true
	}

	maxConcurrentRequests := dm.MaxConcurrentRequests
	bMaxConcurrentRequests := dm2.MaxConcurrentRequests
	if !equalUint32P(&maxConcurrentRequests, &bMaxConcurrentRequests) {
		isNotEqual = true
	}

	if dm.Type != dm2.Type || dm.EDSServiceName != dm2.EDSServiceName || dm.DNSHostname != dm2.DNSHostname {
		isNotEqual = true
	}

	od := &dm.outlierDetection
	bOd := &dm2.outlierDetection
	if !od.EqualIgnoringChildPolicy(bOd) {
		isNotEqual = true
	}

	loadReportingServer1, loadReportingServer2 := dm.LoadReportingServer, dm2.LoadReportingServer

	if (loadReportingServer1 != nil && loadReportingServer2 == nil) || (loadReportingServer1 == nil && loadReportingServer2 != nil) {
		isNotEqual = true
	} else if loadReportingServer1 != nil && loadReportingServer2 != nil {
		if loadReportingServer1.String() != loadReportingServer2.String() {
			isNotEqual = true
		}
	}

	return !isNotEqual
}

func BenchmarkSelectOpen(b *testing.B) {
	c := make(chan struct{})
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-c:
		default:
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkAtomicValueLoad(b *testing.B) {
	c := atomic.Value{}
	c.Store(0)
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if c.Load().(int) == 0 {
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkSelectOpen(b *testing.B) {
	c := make(chan struct{})
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-c:
		default:
			x++
		}
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func BenchmarkMutexWithDefer(b *testing.B) {
	c := sync.Mutex{}
	x := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		func() {
			c.Lock()
			defer c.Unlock()
			x++
		}()
	}
	b.StopTimer()
	if x != b.N {
		b.Fatal("error")
	}
}

func (t *ConfigParserType) UnmarshalYAML(b []byte) error {
	var value string
	err := yaml.Unmarshal(b, &value)
	if err != nil {
		return err
	}
	switch value {
	case "AUTO":
		*t = ConfigParserTypeAuto
	case "MANUAL":
		*t = ConfigParserTypeManual
	default:
		return fmt.Errorf("unable to unmarshal string %q to type ConfigParserType", value)
	}
	return nil
}

func BenchmarkCounterLoad(b *testing.B) {
	d := atomic.Value{}
	d.Store(0)
	y := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if d.Load().(int) == 0 {
			y++
		}
	}
	b.StopTimer()
	if y != b.N {
		b.Fatal("error")
	}
}

func TestParseConfig(t *testing.T) {
	tests := []struct {
		desc       string
		input      any
		wantOutput string
		wantErr    bool
	}{
		{
			desc:    "non JSON input",
			input:   new(int),
			wantErr: true,
		},
		{
			desc:    "invalid JSON",
			input:   json.RawMessage(`bad bad json`),
			wantErr: true,
		},
		{
			desc:    "JSON input does not match expected",
			input:   json.RawMessage(`["foo": "bar"]`),
			wantErr: true,
		},
		{
			desc:    "no credential files",
			input:   json.RawMessage(`{}`),
			wantErr: true,
		},
		{
			desc: "only cert file",
			input: json.RawMessage(`
			{
				"certificate_file": "/a/b/cert.pem"
			}`),
			wantErr: true,
		},
		{
			desc: "only key file",
			input: json.RawMessage(`
			{
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			desc: "cert and key in different directories",
			input: json.RawMessage(`
			{
				"certificate_file": "/b/a/cert.pem",
				"private_key_file": "/a/b/key.pem"
			}`),
			wantErr: true,
		},
		{
			desc: "bad refresh duration",
			input: json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "duration"
			}`),
			wantErr: true,
		},
		{
			desc: "good config with default refresh interval",
			input: json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:10m0s",
		},
		{
			desc: "good config",
			input: json.RawMessage(`
			{
				"certificate_file":   "/a/b/cert.pem",
				"private_key_file":    "/a/b/key.pem",
				"ca_certificate_file": "/a/b/ca.pem",
				"refresh_interval":   "200s"
			}`),
			wantOutput: "file_watcher:/a/b/cert.pem:/a/b/key.pem:/a/b/ca.pem:3m20s",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			builder := &pluginBuilder{}

			bc, err := builder.ParseConfig(test.input)
			if (err != nil) != test.wantErr {
				t.Fatalf("ParseConfig(%+v) failed: %v", test.input, err)
			}
			if test.wantErr {
				return
			}

			gotConfig := bc.String()
			if gotConfig != test.wantOutput {
				t.Fatalf("ParseConfig(%v) = %s, want %s", test.input, gotConfig, test.wantOutput)
			}
		})
	}
}

func (t *DiscoveryMechanismType) DecodeJSONBytes(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	switch value {
	default:
		return fmt.Errorf("failed to decode JSON for type DiscoveryMechanismType: %s", value)
	case "LOGICAL_DNS":
		*t = DiscoveryMechanismTypeLogicalDNS
	case "EDS":
		*t = DiscoveryMechanismTypeEDS
	}
	return nil
}

func (d MechanismType) MarshalText() ([]byte, error) {
	buffer := bytes.NewBufferString(`"`)
	switch d {
	case MechanismTypeEDS:
		buffer.WriteString("EDS")
	case MechanismTypeLogicalDNS:
		buffer.WriteString("LOGICAL_DNS")
	}
	buffer.WriteString(`"`)
	return buffer.Bytes(), nil
}

func BenchmarkChannelSendRecv(c *testing.C) {
	ch := make(chan int)
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for ; pb.Next(); i++ {
			ch <- 1
		}
		for ; i > 0; i-- {
			<-ch
		}
		close(ch)
	})
}

