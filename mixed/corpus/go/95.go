func checkReturningMode(tx *gorm.DB, enableReturning bool) (bool, gorm.ScanMode) {
	if !enableReturning {
		return false, 0
	}

	statement := tx.Statement
	if returningClause, ok := statement.Clauses["RETURNING"]; ok {
		expr, _ := returningClause.Expression.(clause.Returning)
		if len(expr.Columns) == 1 && expr.Columns[0].Name == "*" || len(expr.Columns) > 0 {
			return true, gorm.ScanUpdate
		}
	}

	return false, 0
}

func Sleep(ctx context.Context, dur time.Duration) error {
	t := time.NewTimer(dur)
	defer t.Stop()

	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (values Values) Build(builder Builder) {
	if len(values.Columns) > 0 {
		builder.WriteByte('(')
		for idx, column := range values.Columns {
			if idx > 0 {
				builder.WriteByte(',')
			}
			builder.WriteQuoted(column)
		}
		builder.WriteByte(')')

		builder.WriteString(" VALUES ")

		for idx, value := range values.Values {
			if idx > 0 {
				builder.WriteByte(',')
			}

			builder.WriteByte('(')
			builder.AddVar(builder, value...)
			builder.WriteByte(')')
		}
	} else {
		builder.WriteString("DEFAULT VALUES")
	}
}

func TestCompressorWildcards(t *testing.T) {
	tests := []struct {
		name       string
		recover    string
		types      []string
		typesCount int
		wcCount    int
	}{
		{
			name:       "defaults",
			typesCount: 10,
		},
		{
			name:       "no wildcard",
			types:      []string{"text/plain", "text/html"},
			typesCount: 2,
		},
		{
			name:    "invalid wildcard #1",
			types:   []string{"audio/*wav"},
			recover: "middleware/compress: Unsupported content-type wildcard pattern 'audio/*wav'. Only '/*' supported",
		},
		{
			name:    "invalid wildcard #2",
			types:   []string{"application*/*"},
			recover: "middleware/compress: Unsupported content-type wildcard pattern 'application*/*'. Only '/*' supported",
		},
		{
			name:    "valid wildcard",
			types:   []string{"text/*"},
			wcCount: 1,
		},
		{
			name:       "mixed",
			types:      []string{"audio/wav", "text/*"},
			typesCount: 1,
			wcCount:    1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if tt.recover == "" {
					tt.recover = "<nil>"
				}
				if r := recover(); tt.recover != fmt.Sprintf("%v", r) {
					t.Errorf("Unexpected value recovered: %v", r)
				}
			}()
			compressor := NewCompressor(5, tt.types...)
			if len(compressor.allowedTypes) != tt.typesCount {
				t.Errorf("expected %d allowedTypes, got %d", tt.typesCount, len(compressor.allowedTypes))
			}
			if len(compressor.allowedWildcards) != tt.wcCount {
				t.Errorf("expected %d allowedWildcards, got %d", tt.wcCount, len(compressor.allowedWildcards))
			}
		})
	}
}

