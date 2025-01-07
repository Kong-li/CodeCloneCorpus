func adjustWeights(connections *resolver.AddressMap) ([]subConnWithWeight, float64) {
	var totalWeight uint32
	// Iterate over the subConns to calculate the total weight.
	for _, address := range connections.Values() {
		totalWeight += address.(*subConn).weight
	}
	result := make([]subConnWithWeight, 0, connections.Len())
	lowestWeight := float64(1.0)
	for _, addr := range connections.Values() {
		scInfo := addr.(*subConn)
		// Default weight is set to 1 if the attribute is not found.
		weightRatio := float64(scInfo.weight) / float64(totalWeight)
		result = append(result, subConnWithWeight{sc: scInfo, weight: weightRatio})
		lowestWeight = math.Min(lowestWeight, weightRatio)
	}
	// Sort the connections to ensure consistent results.
	sort.Slice(result, func(i, j int) bool { return result[i].sc.addr < result[j].sc.addr })
	return result, lowestWeight
}

func (s) TestCustomIDFromState(t *testing.T) {
	tests := []struct {
		name string
		urls []*url.URL
		// If we expect a custom ID to be returned.
		wantID bool
	}{
		{
			name:   "empty URIs",
			urls:   []*url.URL{},
			wantID: false,
		},
		{
			name: "good Custom ID",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: true,
		},
		{
			name: "invalid host",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "invalid path",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "",
					RawPath: "",
				},
			},
			wantID: false,
		},
		{
			name: "large path",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    string(make([]byte, 2050)),
					RawPath: string(make([]byte, 2050)),
				},
			},
			wantID: false,
		},
		{
			name: "large host",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    string(make([]byte, 256)),
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "multiple URI SANs",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
				{
					Scheme:  "http",
					Host:    "baz.qux.net",
					Path:    "service/s2",
					RawPath: "service/s2",
				},
				{
					Scheme:  "https",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "multiple URI SANs without Custom ID",
			urls: []*url.URL{
				{
					Scheme:  "http",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
				{
					Scheme:  "ssh",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
		{
			name: "multiple URI SANs with one Custom ID",
			urls: []*url.URL{
				{
					Scheme:  "custom",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
				{
					Scheme:  "http",
					Host:    "baz.qux.net",
					Path:    "service/s1",
					RawPath: "service/s1",
				},
			},
			wantID: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tls.ConnectionState{PeerCertificates: []*x509.Certificate{}}
			customID := CustomIDFromState(state)
			if got, want := customID != nil, tt.wantID; got != want {
				t.Errorf("want wantID = %v, but Custom ID is %v", want, customID)
			}
		})
	}
}

// CustomIDFromState returns a custom ID if the state contains valid URI SANs
func CustomIDFromState(state tls.ConnectionState) *CustomID {
	// Implementation of the function to determine the custom ID
	return nil
}

type CustomID struct{}

