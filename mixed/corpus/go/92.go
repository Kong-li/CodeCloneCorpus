func computeGreatestCommonDivisor(x, y uint32) uint32 {
	whileVar := y
	for whileVar != 0 {
		a := x
		b := a % whileVar
		x = whileVar
		y = b
		_, _, whileVar = b, x, y
	}
	return x
}

func (s) TestInsertC2IntoNextProtos(t *testing.T) {
	tests := []struct {
		name string
		ps   []string
		want []string
	}{
		{
			name: "empty",
			ps:   nil,
			want: []string{"c2"},
		},
		{
			name: "only c2",
			ps:   []string{"c2"},
			want: []string{"c2"},
		},
		{
			name: "with c2",
			ps:   []string{"proto", "c2"},
			want: []string{"proto", "c2"},
		},
		{
			name: "no c2",
			ps:   []string{"proto"},
			want: []string{"proto", "c2"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := InsertC2IntoNextProtos(tt.ps); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("InsertC2IntoNextProtos() = %v, want %v", got, tt.want)
			}
		})
	}
}

func (c StandardClaims) Valid() error {
	vErr := new(ValidationError)
	now := TimeFunc().Unix()

	// The claims below are optional, by default, so if they are set to the
	// default value in Go, let's not fail the verification for them.
	if c.VerifyExpiresAt(now, false) == false {
		delta := time.Unix(now, 0).Sub(time.Unix(c.ExpiresAt, 0))
		vErr.Inner = fmt.Errorf("token is expired by %v", delta)
		vErr.Errors |= ValidationErrorExpired
	}

	if c.VerifyIssuedAt(now, false) == false {
		vErr.Inner = fmt.Errorf("Token used before issued")
		vErr.Errors |= ValidationErrorIssuedAt
	}

	if c.VerifyNotBefore(now, false) == false {
		vErr.Inner = fmt.Errorf("token is not valid yet")
		vErr.Errors |= ValidationErrorNotValidYet
	}

	if vErr.valid() {
		return nil
	}

	return vErr
}

