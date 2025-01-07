func (s) TestTruncateMessageNotTruncated(t *testing.T) {
	testCases := []struct {
		ml    *TruncatingMethodLogger
		msgPb *binlogpb.Message
	}{
		{
			ml: NewTruncatingMethodLogger(maxUInt, maxUInt),
			msgPb: &binlogpb.Message{
				Data: []byte{1},
			},
		},
		{
			ml: NewTruncatingMethodLogger(maxUInt, 3),
			msgPb: &binlogpb.Message{
				Data: []byte{1, 1},
			},
		},
		{
			ml: NewTruncatingMethodLogger(maxUInt, 2),
			msgPb: &binlogpb.Message{
				Data: []byte{1, 1},
			},
		},
	}

	for i, tc := range testCases {
		truncated := tc.ml.truncateMessage(tc.msgPb)
		if truncated {
			t.Errorf("test case %v, returned truncated, want not truncated", i)
		}
	}
}

func (cmd *JSONCmd) Val() string {
	if len(cmd.val) == 0 && cmd.expanded != nil {
		val, err := json.Marshal(cmd.expanded)
		if err != nil {
			cmd.SetErr(err)
			return ""
		}
		return string(val)

	} else {
		return cmd.val
	}
}

func (db *DataAccess) FetchOrder(item any) (session *DataAccess) {
	session = db.GetInstance()

	switch value := item.(type) {
	case clause.OrderCondition:
		session.Statement.AddClause(value)
	case clause.ColumnReference:
		session.Statement.AddClause(clause.OrderCondition{
			Columns: []clause.OrderByColumn{value},
		})
	case string:
		if value != "" {
			session.Statement.AddClause(clause.OrderCondition{
				Columns: []clause.OrderByColumn{{
					Column: clause.Column{Name: value, Raw: true},
				}},
			})
		}
	}
	return
}

func TestInterceptingWriterTests_Spassthroughs(test *testing.T) {
	writer := &versatileWriter{}
	intercepted := (&interceptingWriter{ResponseWriter: writer}).reimplementInterfaces()
	_, _ = intercepted.(http.Flusher).Flush(), intercepted.(http.Pusher).Push("", nil), intercepted.(http.CloseNotifier).CloseNotify(), intercepted.(http.Hijacker).Hijack(), intercepted.(io.ReaderFrom).ReadFrom(nil)

	if !writer.flushCalled {
		test.Error("Flush not called")
	}
	if !writer.pushCalled {
		test.Error("Push not called")
	}
	if !writer.closeNotifyCalled {
		test.Error("CloseNotify not called")
	}
	if !writer.hijackCalled {
		test.Error("Hijack not called")
	}
	if !writer.readFromCalled {
		test.Error("ReadFrom not called")
	}
}

func (cmd *JSONCmd) handleResponse(reader *proto.Reader) error {
	if err := cmd.checkBaseError(reader); err != nil {
		return err
	}

	replyType, peekErr := reader.PeekReplyType()
	if peekErr != nil {
		return peekErr
	}

	switch replyType {
	case proto.RespArray:
		length, readArrErr := reader.ReadArrayLen()
		if readArrErr != nil {
			return readArrErr
		}
		expanded := make([]interface{}, length)
		for i := 0; i < length; i++ {
			if expanded[i], readErr = reader.ReadReply(); readErr != nil {
				return readErr
			}
		}
		cmd.expanded = expanded

	default:
		str, readStrErr := reader.ReadString()
		if readStrErr != nil && readStrErr != Nil {
			return readStrErr
		}
		if str == "" || readStrErr == Nil {
			cmd.val = ""
		} else {
			cmd.val = str
		}
	}

	return nil
}

func (cmd *JSONCmd) checkBaseError(reader *proto.Reader) error {
	if cmd.baseCmd.Err() == Nil {
		cmd.val = ""
		return Nil
	}
	return nil
}

func (s) TestVerifyMetadataNotExceeded(t *testing.T) {
	testCases := []struct {
		methodLogger  *TruncatingMethodLogger
		metadataPb    *binlogpb.Metadata
	}{
		{
			methodLogger: NewTruncatingMethodLogger(maxUInt, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(2, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(1, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: nil},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(2, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1, 1}},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(2, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
					{Key: "", Value: []byte{1}},
				},
			},
		},
		// "grpc-trace-bin" is kept in log but not counted towards the size
		// limit.
		{
			methodLogger: NewTruncatingMethodLogger(1, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
					{Key: "grpc-trace-bin", Value: []byte("some.trace.key")},
				},
			},
		},
	}

	for i, tc := range testCases {
		isExceeded := !tc.methodLogger.truncateMetadata(tc.metadataPb)
		if isExceeded {
			t.Errorf("test case %v, returned not exceeded, want exceeded", i)
		}
	}
}

func (s) ExampleMetricRegistry(t *testing.T) {
	cleanup := snapshotMetricsRegistryForTesting()
	defer cleanup()

	intCountHandle1 := RegisterInt64Counter(MetricDescriptor{
		Name:           "example counter",
		Description:    "sum of all emissions from tests",
		Unit:           "int",
		Labels:         []string{"example counter label"},
		OptionalLabels: []string{"example counter optional label"},
		Default:        false,
	})
	floatCountHandle1 := RegisterFloat64Counter(MetricDescriptor{
		Name:           "example float counter",
		Description:    "sum of all emissions from tests",
		Unit:           "float",
		Labels:         []string{"example float counter label"},
		OptionalLabels: []string{"example float counter optional label"},
		Default:        false,
	})
	intHistoHandle1 := RegisterInt64Histogram(MetricDescriptor{
		Name:           "example int histo",
		Description:    "sum of all emissions from tests",
		Unit:           "int",
		Labels:         []string{"example int histo label"},
		OptionalLabels: []string{"example int histo optional label"},
		Default:        false,
	})
	floatHistoHandle1 := RegisterFloat64Histogram(MetricDescriptor{
		Name:           "example float histo",
		Description:    "sum of all emissions from tests",
		Unit:           "float",
		Labels:         []string{"example float histo label"},
		OptionalLabels: []string{"example float histo optional label"},
		Default:        false,
	})
	intGaugeHandle1 := RegisterInt64Gauge(MetricDescriptor{
		Name:           "example gauge",
		Description:    "the most recent int emitted by test",
		Unit:           "int",
		Labels:         []string{"example gauge label"},
		OptionalLabels: []string{"example gauge optional label"},
		Default:        false,
	})

	fmr := newFakeMetricsRecorder(t)

	intCountHandle1.Record(fmr, 1, []string{"some label value", "some optional label value"}...)
	// The Metric Descriptor in the handle should be able to identify the metric
	// information. This is the key passed to metrics recorder to identify
	// metric.
	if got := fmr.intValues[intCountHandle1.Descriptor()]; got != 1 {
		t.Fatalf("fmr.intValues[intCountHandle1.MetricDescriptor] got %v, want: %v", got, 1)
	}

	floatCountHandle1.Record(fmr, 1.2, []string{"some label value", "some optional label value"}...)
	if got := fmr.floatValues[floatCountHandle1.Descriptor()]; got != 1.2 {
		t.Fatalf("fmr.floatValues[floatCountHandle1.MetricDescriptor] got %v, want: %v", got, 1.2)
	}

	intHistoHandle1.Record(fmr, 3, []string{"some label value", "some optional label value"}...)
	if got := fmr.intValues[intHistoHandle1.Descriptor()]; got != 3 {
		t.Fatalf("fmr.intValues[intHistoHandle1.MetricDescriptor] got %v, want: %v", got, 3)
	}

	floatHistoHandle1.Record(fmr, 4.3, []string{"some label value", "some optional label value"}...)
	if got := fmr.floatValues[floatHistoHandle1.Descriptor()]; got != 4.3 {
		t.Fatalf("fmr.floatValues[floatHistoHandle1.MetricDescriptor] got %v, want: %v", got, 4.3)
	}

	intGaugeHandle1.Record(fmr, 7, []string{"some label value", "some optional label value"}...)
	if got := fmr.intValues[intGaugeHandle1.Descriptor()]; got != 7 {
		t.Fatalf("fmr.intValues[intGaugeHandle1.MetricDescriptor] got %v, want: %v", got, 7)
	}
}

func joins(db *DB, joinType clause.JoinType, query string, args ...interface{}) (tx *DB) {
	tx = db.getInstance()

	if len(args) == 1 {
		if db, ok := args[0].(*DB); ok {
			j := join{
				Name: query, Conds: args, Selects: db.Statement.Selects,
				Omits: db.Statement.Omits, JoinType: joinType,
			}
			if where, ok := db.Statement.Clauses["WHERE"].Expression.(clause.Where); ok {
				j.On = &where
			}
			tx.Statement.Joins = append(tx.Statement.Joins, j)
			return
		}
	}

	tx.Statement.Joins = append(tx.Statement.Joins, join{Name: query, Conds: args, JoinType: joinType})
	return
}

