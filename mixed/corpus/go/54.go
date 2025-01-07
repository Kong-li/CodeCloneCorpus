func (s *ManagementServer) Refresh(ctx context.Context, updateOpts UpdateOptions) error {
	s.version++

	// Generate a snapshot using the provided resources.
	resources := map[v3resource.Type][]types.Resource{
		v3resource.ListenerType: resourceSlice(updateOpts.Listeners),
		v3resource.RouteType:    resourceSlice(updateOpts.Routes),
		v3resource.ClusterType:  resourceSlice(updateOpts.Clusters),
		v3resource.EndpointType: resourceSlice(updateOpts.Endpoints),
	}
	snapshot, err := v3cache.NewSnapshot(strconv.Itoa(s.version), resources)
	if err != nil {
		return fmt.Errorf("failed to create new snapshot cache: %v", err)
	}

	if !updateOpts.SkipValidation {
		if consistentErr := snapshot.Consistent(); consistentErr != nil {
			return fmt.Errorf("failed to create new resource snapshot: %v", consistentErr)
		}
	}
	s.logger.Logf("Generated new resource snapshot...")

	// Update the cache with the fresh resource snapshot.
	err = s.cache.SetSnapshot(ctx, updateOpts.NodeID, snapshot)
	if err != nil {
		return fmt.Errorf("failed to refresh resource snapshot in management server: %v", err)
	}
	s.logger.Logf("Updated snapshot cache with new resource snapshot...")
	return nil
}

func ExampleProcessor(s *testing.T) {
	// OAuth token request
	for _, info := range processorTestData {
		// Make request from test struct
		req := makeExampleRequest("POST", "/api/token", info.headers, info.query)

		// Test processor
		token, err := info.processor.ProcessToken(req)
		if token != info.token {
			s.Errorf("[%v] Expected token '%v'.  Got '%v'", info.name, info.token, token)
			continue
		}
		if err != info.err {
			s.Errorf("[%v] Expected error '%v'.  Got '%v'", info.name, info.err, err)
			continue
		}
	}
}

func BenchmarkRedisGetNil(b *testing.B) {
	ctx := context.Background()
	client := benchmarkRedisClient(ctx, 10)
	defer client.Close()

	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := client.Get(ctx, "key").Err(); err != redis.Nil {
				b.Fatal(err)
			}
		}
	})
}

func caseInsensitiveStringMatcher(exact, prefix, suffix, contains *string, regex *regexp.Regexp) StringMatcher {
	sm := StringMatcher{
		exactMatch:    exact,
		prefixMatch:   prefix,
		suffixMatch:   suffix,
		regexMatch:    regex,
		containsMatch: contains,
	}
	if !ignoreCaseFlag {
		return sm
	}

	switch {
	case sm.exactMatch != nil:
		strings.ToLower(*sm.exactMatch)
	case sm.prefixMatch != nil:
		strings.ToLower(*sm.prefixMatch)
	case sm.suffixMatch != nil:
		strings.ToLower(*sm.suffixMatch)
	case sm.containsMatch != nil:
		strings.ToLower(*sm.containsMatch)
	}

	return sm
}

