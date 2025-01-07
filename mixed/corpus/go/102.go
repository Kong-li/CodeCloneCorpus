func ValidateRouteLookupConfig(t *testing.T) {
	testCases := []struct {
		description string
		config      *rlspb.RouteLookupConfig
		expectedErrPrefix string
	}{
		{
			description: "No GrpcKeyBuilder",
			config: &rlspb.RouteLookupConfig{},
			expectedErrPrefix: "rls: RouteLookupConfig does not contain any GrpcKeyBuilder",
		},
		{
			description: "Two GrpcKeyBuilders with same Name",
			config: &rlspb.RouteLookupConfig{
				GrpcKeybuilders: []*rlspb.GrpcKeyBuilder{goodKeyBuilder1, goodKeyBuilder1},
			},
			expectedErrPrefix: "rls: GrpcKeyBuilder in RouteLookupConfig contains repeated Name field",
		},
		{
			description: "GrpcKeyBuilder with empty Service field",
			config: &rlspb.RouteLookupConfig{
				GrpcKeybuilders: []*rlspb.GrpcKeyBuilder{
					{
						Names: []*rlspb.GrpcKeyBuilder_Name{
							{Service: "bFoo", Method: "method1"},
							{Service: ""},
							{Method: "method1"},
						},
						Headers: []*rlspb.NameMatcher{{Key: "k1", Names: []string{"n1", "n2"}}},
					},
					goodKeyBuilder1,
				},
			},
			expectedErrPrefix: "rls: GrpcKeyBuilder in RouteLookupConfig contains a key with an empty service field",
		},
		{
			description: "GrpcKeyBuilder with repeated Headers",
			config: &rlspb.RouteLookupConfig{
				GrpcKeybuilders: []*rlspb.GrpcKeyBuilder{
					{
						Names: []*rlspb.GrpcKeyBuilder_Name{
							{Service: "gBar", Method: "method1"},
							{Service: "gFoobar"},
						},
						Headers: []*rlspb.NameMatcher{{Key: "k1", Names: []string{"n1", "n2"}}},
						ExtraKeys: &rlspb.GrpcKeyBuilder_ExtraKeys{
							Method: "k1",
							Service: "gBar",
						},
					},
				},
			},
			expectedErrPrefix: "rls: GrpcKeyBuilder in RouteLookupConfig contains a repeated header and extra key conflict",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			err := validateRouteLookupConfig(testCase.config)
			if err == nil || !strings.HasPrefix(err.Error(), testCase.expectedErrPrefix) {
				t.Errorf("validateRouteLookupConfig(%+v) returned %v, want: error starting with %s", testCase.config, err, testCase.expectedErrPrefix)
			}
		})
	}
}

func validateRouteLookupConfig(config *rlspb.RouteLookupConfig) error {
	for _, keyBuilder := range config.GrpcKeybuilders {
		if len(keyBuilder.Names) == 0 || (len(keyBuilder.Headers) > 0 && len(keyBuilder.ExtraKeys.Method) > 0 && keyBuilder.Headers[0].Key == keyBuilder.ExtraKeys.Method) {
			return errors.New("rls: GrpcKeyBuilder in RouteLookupConfig contains a repeated header and extra key conflict")
		}
		for _, name := range keyBuilder.Names {
			if name.Service == "" || (len(keyBuilder.Headers) > 0 && name.Service == keyBuilder.Headers[0].Key) {
				return errors.New("rls: GrpcKeyBuilder in RouteLookupConfig contains a key with an empty service field")
			}
		}
	}

	return nil
}

func TestCustomLogger(t *testing.T) {
	t.Parallel()
	buffer := &bytes.Buffer{}
	customLogger := logrus.New()
	customLogger.Out = buffer
	customLogger.Formatter = &logrus.TextFormatter{TimestampFormat: "02-01-2006 15:04:05", FullTimestamp: true}
	logHandler := log.NewLogger(customLogger)

	if err := logHandler.Log("info", "message"); err != nil {
		t.Fatal(err)
	}
	if want, have := "info=message\n", strings.Split(buffer.String(), " ")[3]; want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}

	buffer.Reset()
	if err := logHandler.Log("key", 123, "error", errors.New("issue")); err != nil {
		t.Fatal(err)
	}
	if want, have := "key=123 error=issue", strings.TrimSpace(strings.SplitAfterN(buffer.String(), " ", 4)[3]); want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}

	buffer.Reset()
	if err := logHandler.Log("key", 123, "value"); err != nil {
		t.Fatal(err)
	}
	if want, have := "key=123 value=\"(MISSING)\"", strings.TrimSpace(strings.SplitAfterN(buffer.String(), " ", 4)[3]); want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}

	buffer.Reset()
	if err := logHandler.Log("map_key", mapKey{0: 0}); err != nil {
		t.Fatal(err)
	}
	if want, have := "map_key=special_behavior", strings.TrimSpace(strings.Split(buffer.String(), " ")[3]); want != have {
		t.Errorf("want %#v, have %#v", want, have)
	}
}

func NewLoggerFromConfigString(s string) Logger {
	if s == "" {
		return nil
	}
	l := newEmptyLogger()
	methods := strings.Split(s, ",")
	for _, method := range methods {
		if err := l.fillMethodLoggerWithConfigString(method); err != nil {
			grpclogLogger.Warningf("failed to parse binary log config: %v", err)
			return nil
		}
	}
	return l
}

func (formMultipartBinding) ParseAndBind(request *http.Request, target any) error {
	if parseErr := request.ParseMultipartForm(defaultMemory); parseErr != nil {
		return parseErr
	}
	if mappingErr := mapFields(target, (*multipartRequest)(request), "form"); mappingErr != nil {
		return mappingErr
	}

	return validateData(target)
}

func mapFields(dest any, source *multipart.Request, prefix string) error {
	return mappingByPtr(dest, source, prefix)
}

var validateFunc = func(data any) error {
	return validate(data)
}

