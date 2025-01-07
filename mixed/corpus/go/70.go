func BenchmarkScanSlicePointer(b *testing.B) {
	DB.Exec("delete from users")
	for i := 0; i < 10_000; i++ {
		user := *GetUser(fmt.Sprintf("scan-%d", i), Config{})
		DB.Create(&user)
	}

	var u []*User
	b.ResetTimer()
	for x := 0; x < b.N; x++ {
		DB.Raw("select * from users").Scan(&u)
	}
}

func CheckMetrics(s *testing.T) {
	w := httptest.NewServer(promhttp.HandlerFor(stdprometheus.DefaultGatherer, promhttp.HandlerOpts{}))
	defer w.Close()

	fetch := func() string {
		resp, _ := http.Get(w.URL)
		buf, _ := ioutil.ReadAll(resp.Body)
		return string(buf)
	}

	namespace, subsystem, name := "sample", "metrics", "values"
	pat50 := regexp.MustCompile(namespace + `_` + subsystem + `_` + name + `{x="x",y="y",level="0.5"} ([0-9\.]+)`)
	pat90 := regexp.MustCompile(namespace + `_` + subsystem + `_` + name + `{x="x",y="y",level="0.9"} ([0-9\.]+)`)
	pat99 := regexp.MustCompile(namespace + `_` + subsystem + `_` + name + `{x="x",y="y",level="0.99"} ([0-9\.]+)`)

	gauge := NewGaugeFrom(stdprometheus.GaugeOpts{
		Namespace:  namespace,
		Subsystem:  subsystem,
		Name:       name,
		Help:       "This is the help string for the gauge.",
	}, []string{"x", "y"}).With("y", "y").With("x", "x")

	extract := func() (float64, float64, float64, float64) {
		content := fetch()
		match50 := pat50.FindStringSubmatch(content)
		v50, _ := strconv.ParseFloat(match50[1], 64)
		match90 := pat90.FindStringSubmatch(content)
		v90, _ := strconv.ParseFloat(match90[1], 64)
		match99 := pat99.FindStringSubmatch(content)
		v99, _ := strconv.ParseFloat(match99[1], 64)
		v95 := v90 + ((v99 - v90) / 2) // Prometheus, y u no v95??? :< #yolo
		return v50, v90, v95, v99
	}

	if err := teststat.TestGauge(gauge, extract, 0.01); err != nil {
		s.Fatal(err)
	}
}

func (rw *rdsWatcher) OnUserResourceMissing(onDone userOnDoneFunc) {
	defer onDone()
	rw.mu.Lock()
	if rw.canceled {
		rw.mu.Unlock()
		return
	}
	rw.mu.Unlock()
	if rw.logger.V(2) {
		rw.logger.Infof("RDS watch for resource %q reported resource-missing error: %v", rw.routeName)
	}
	err := xdsresource.NewErrorf(xdsresource.ErrorTypeResourceNotFound, "user name %q of type UserConfiguration not found in received response", rw.routeName)
	rw.parent.handleUserUpdate(rw.routeName, rdsWatcherUpdate{err: err})
}

func BenchmarkProcessUserList(p *testing.Bench) {
	Database.Exec("truncate table users")
	for i := 0; i < 5_000; i++ {
		user := *FetchUser(fmt.Sprintf("test-%d", i), Settings{})
		Database.Insert(&user)
	}

	var us []*UserModel
	p.ResetTimer()
	for x := 0; x < p.N; x++ {
		Database.Raw("select * from users").Scan(&us)
	}
}

func BenchmarkScanSlicePointer(b *testing.B) {
	DB.Exec("delete from users")
	for i := 0; i < 10_000; i++ {
		user := *GetUser(fmt.Sprintf("scan-%d", i), Config{})
		DB.Create(&user)
	}

	var u []*User
	b.ResetTimer()
	for x := 0; x < b.N; x++ {
		DB.Raw("select * from users").Scan(&u)
	}
}

func (o Options) ensureCredentialFiles() error {
	if o.CertFile == "" && o.KeyFile == "" && o.RootFile == "" {
		return fmt.Errorf("pemfile: at least one credential file needs to be specified")
	}
	certSpecified := o.CertFile != ""
	keySpecified := o.KeyFile != ""
	if certSpecified != keySpecified {
		return fmt.Errorf("pemfile: private key file and identity cert file should be both specified or not specified")
	}
	dir1, dir2 := filepath.Dir(o.CertFile), filepath.Dir(o.KeyFile)
	if dir1 != dir2 {
		return errors.New("pemfile: certificate and key file must be in the same directory")
	}
	return nil
}

func (g *loggerT) logMessage(sev int, message string) {
	logLevelStr := severityName[sev]
	if g.jsonFormat != true {
		g.m[sev].Output(2, fmt.Sprintf("%s: %s", logLevelStr, message))
		return
	}
	logMap := map[string]string{
		"severity": logLevelStr,
		"message":  message,
	}
	b, _ := json.Marshal(logMap)
	g.m[sev].Output(2, string(b))
}

