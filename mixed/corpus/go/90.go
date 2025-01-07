func TestSelect(t *testing.T) {
	results := []struct {
		Clauses []clause.Interface
		Result  string
		Vars    []interface{}
	}{
		{
			[]clause.Interface{clause.Select{}, clause.From{}},
			"SELECT * FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Columns: []clause.Column{clause.PrimaryColumn},
			}, clause.From{}},
			"SELECT `users`.`id` FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Columns: []clause.Column{clause.PrimaryColumn},
			}, clause.Select{
				Columns: []clause.Column{{Name: "name"}},
			}, clause.From{}},
			"SELECT `name` FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Expression: clause.CommaExpression{
					Exprs: []clause.Expression{
						clause.NamedExpr{"?", []interface{}{clause.Column{Name: "id"}}},
						clause.NamedExpr{"?", []interface{}{clause.Column{Name: "name"}}},
						clause.NamedExpr{"LENGTH(?)", []interface{}{clause.Column{Name: "mobile"}}},
					},
				},
			}, clause.From{}},
			"SELECT `id`, `name`, LENGTH(`mobile`) FROM `users`", nil,
		},
		{
			[]clause.Interface{clause.Select{
				Expression: clause.CommaExpression{
					Exprs: []clause.Expression{
						clause.Expr{
							SQL: "? as name",
							Vars: []interface{}{
								clause.Eq{
									Column: clause.Column{Name: "age"},
									Value:  18,
								},
							},
						},
					},
				},
			}, clause.From{}},
			"SELECT `age` = ? as name FROM `users`",
			[]interface{}{18},
		},
	}

	for idx, result := range results {
		t.Run(fmt.Sprintf("case #%v", idx), func(t *testing.T) {
			checkBuildClauses(t, result.Clauses, result.Result, result.Vars)
		})
	}
}

func (m *SigningMethodHMAC) Verify(signingString, signature string, key interface{}) error {
	// Verify the key is the right type
	keyBytes, ok := key.([]byte)
	if !ok {
		return ErrInvalidKeyType
	}

	// Decode signature, for comparison
	sig, err := DecodeSegment(signature)
	if err != nil {
		return err
	}

	// Can we use the specified hashing method?
	if !m.Hash.Available() {
		return ErrHashUnavailable
	}

	// This signing method is symmetric, so we validate the signature
	// by reproducing the signature from the signing string and key, then
	// comparing that against the provided signature.
	hasher := hmac.New(m.Hash.New, keyBytes)
	hasher.Write([]byte(signingString))
	if !hmac.Equal(sig, hasher.Sum(nil)) {
		return ErrSignatureInvalid
	}

	// No validation errors.  Signature is good.
	return nil
}

