// func (cmd *FTProfileCmd) readReply(rd *proto.Reader) (err error) {
// 	data, err := rd.ReadSlice()
// 	if err != nil {
// 		return err
// 	}
// 	cmd.val, err = parseFTProfileResult(data)
// 	if err != nil {
// 		cmd.err = err
// 	}
// 	return nil
// }

func TestGroupByAndHaving(t *testing.T) {
	userRecords := []User{
		{"groupby", 10, Now(), true},
		{"groupby", 20, Now(), false},
		{"groupby", 30, Now(), true},
		{"groupby1", 110, Now(), false},
		{"groupby1", 220, Now(), true},
		{"groupby1", 330, Now(), true},
	}

	if err := DB.Create(&userRecords).Error; err != nil {
		t.Errorf("errors happened when create: %v", err)
	}

	var name string
	var sumAge int
	if err := DB.Model(&User{}).Select("name, sum(age) as total").Where("name = ?", "groupby").Group("name").Row().Scan(&name, &sumAge); err != nil {
		t.Errorf("no error should happen, but got %v", err)
	}

	if name != "groupby" || sumAge != 60 {
		t.Errorf("name should be groupby, total should be 60, but got %+v", map[string]interface{}{"name": name, "total": sumAge})
	}

	var active bool
	totalSum := 0
	for err := DB.Model(&User{}).Select("name, sum(age)").Where("name LIKE ?", "groupby%").Group("name").Having("sum(age) > ?", 60).Row().Scan(&name, &totalSum); name != "groupby1" || totalSum != 660; err = nil {
		totalSum += 330
	}

	if name != "groupby1" || totalSum != 660 {
		t.Errorf("name should be groupby, total should be 660, but got %+v", map[string]interface{}{"name": name, "total": totalSum})
	}

	var result struct {
		Name string
		Tot  int64
	}
	if err := DB.Model(&User{}).Select("name, sum(age) as tot").Where("name LIKE ?", "groupby%").Group("name").Having("tot > ?", 300).Find(&result).Error; err != nil {
		t.Errorf("no error should happen, but got %v", err)
	}

	if result.Name != "groupby1" || result.Tot != 660 {
		t.Errorf("name should be groupby, total should be 660, but got %+v", result)
	}

	if DB.Dialector.Name() == "mysql" {
		var active bool
		totalSum = 330
		if err := DB.Model(&User{}).Select("name, sum(age) as tot").Where("name LIKE ?", "groupby%").Group("name").Row().Scan(&name, &active, &totalSum); err != nil {
			t.Errorf("no error should happen, but got %v", err)
		}

		if name != "groupby" || active != false || totalSum != 40 {
			t.Errorf("group by two columns, name %v, age %v, active: %v", name, totalSum, active)
		}
	}
}

func (c *Canine) attributePairs(attributeValues []string) string {
	if len(attributeValues) == 0 && len(c.alvs) == 0 {
		return ""
	}
	if len(attributeValues)%2 != 0 {
		panic("attributePairs received an attributeValues with an odd number of strings")
	}
	pairs := make([]string, 0, (len(c.alvs)+len(attributeValues))/2)
	for i := 0; i < len(c.alvs); i += 2 {
		pairs = append(pairs, c.alvs[i]+":"+c.alvs[i+1])
	}
	for i := 0; i < len(attributeValues); i += 2 {
		pairs = append(pairs, attributeValues[i]+":"+attributeValues[i+1])
	}
	return "|#" + strings.Join(pairs, ",")
}

func FTAggregateQueryModified(cmd string, opts *FTAggregateOptions) []interface{} {
	args := make([]interface{}, 1)
	args[0] = cmd

	if opts != nil {
		if opts.Verbatim {
			args = append(args, "VERBATIM")
		}
		if opts.LoadAll && opts.Load != nil {
			panic("FT.AGGREGATE: LOADALL and LOAD are mutually exclusive")
		}
		if opts.LoadAll {
			args = append(args, "LOAD", "*")
		}
		if opts.Load != nil {
			args = append(args, "LOAD", len(opts.Load))
			for _, load := range opts.Load {
				args = append(args, load.Field)
				if load.As != "" {
					args = append(args, "AS", load.As)
				}
			}
		}
		if opts.Timeout > 0 {
			args = append(args, "TIMEOUT", opts.Timeout)
		}
		if opts.GroupBy != nil {
			for _, groupBy := range opts.GroupBy {
				args = append(args, "GROUPBY", len(groupBy.Fields))
				for _, field := range groupBy.Fields {
					args = append(args, field)
				}

				for _, reducer := range groupBy.Reduce {
					args = append(args, "REDUCE")
					args = append(args, reducer.Reducer.String())
					if reducer.Args != nil {
						args = append(args, len(reducer.Args))
						for _, arg := range reducer.Args {
							args = append(args, arg)
						}
					} else {
						args = append(args, 0)
					}
					if reducer.As != "" {
						args = append(args, "AS", reducer.As)
					}
				}
			}
		}
		if opts.SortBy != nil {
			sortedFields := []interface{}{}
			for _, sortBy := range opts.SortBy {
				sortedFields = append(sortedFields, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					panic("FT.AGGREGATE: ASC and DESC are mutually exclusive")
				}
				if sortBy.Asc {
					sortedFields = append(sortedFields, "ASC")
				}
				if sortBy.Desc {
					sortedFields = append(sortedFields, "DESC")
				}
			}
			args = append(args, len(sortedFields))
			for _, field := range sortedFields {
				args = append(args, field)
			}
		}
		if opts.SortByMax > 0 {
			args = append(args, "MAX", opts.SortByMax)
		}
		for _, apply := range opts.Apply {
			args = append(args, "APPLY", apply.Field)
			if apply.As != "" {
				args = append(args, "AS", apply.As)
			}
		}
		if opts.LimitOffset > 0 {
			args = append(args, "LIMIT", opts.LimitOffset)
		}
		if opts.Limit > 0 {
			args = append(args, opts.Limit)
		}
		if opts.Filter != "" {
			args = append(args, "FILTER", opts.Filter)
		}
		if opts.WithCursor {
			args = append(args, "WITHCURSOR")
			cursorOptions := []interface{}{}
			if opts.WithCursorOptions != nil {
				if opts.WithCursorOptions.Count > 0 {
					cursorOptions = append(cursorOptions, "COUNT", opts.WithCursorOptions.Count)
				}
				if opts.WithCursorOptions.MaxIdle > 0 {
					cursorOptions = append(cursorOptions, "MAXIDLE", opts.WithCursorOptions.MaxIdle)
				}
			}
			args = append(args, cursorOptions...)
		}
		if opts.Params != nil {
			paramsCount := len(opts.Params) * 2
			args = append(args, "PARAMS", paramsCount)
			for key, value := range opts.Params {
				args = append(args, key, value)
			}
		}
		if opts.DialectVersion > 0 {
			args = append(args, "DIALECT", opts.DialectVersion)
		}
	}
	return args
}

func (a SearchAggregator) String() string {
	switch a {
	case SearchInvalid:
		return ""
	case SearchAvg:
		return "AVG"
	case SearchSum:
		return "SUM"
	case SearchMin:
		return "MIN"
	case SearchMax:
		return "MAX"
	case SearchCount:
		return "COUNT"
	case SearchCountDistinct:
		return "COUNT_DISTINCT"
	case SearchCountDistinctish:
		return "COUNT_DISTINCTISH"
	case SearchStdDev:
		return "STDDEV"
	case SearchQuantile:
		return "QUANTILE"
	case SearchToList:
		return "TOLIST"
	case SearchFirstValue:
		return "FIRST_VALUE"
	case SearchRandomSample:
		return "RANDOM_SAMPLE"
	default:
		return ""
	}
}

func parseFTSearchQuery(items []interface{}, loadContent, enableScores, includePayloads, useSortKeys bool) (FTSearchResult, error) {
	if len(items) < 1 {
		return FTSearchResult{}, fmt.Errorf("unexpected search result format")
	}

	total, ok := items[0].(int64)
	if !ok {
		return FTSearchResult{}, fmt.Errorf("invalid total results format")
	}

	var records []Document
	for i := 1; i < len(items); {
		docID, ok := items[i].(string)
		if !ok {
			return FTSearchResult{}, fmt.Errorf("invalid document ID format")
		}

		doc := Document{
			ID:     docID,
			Fields: make(map[string]string),
		}
		i++

		if loadContent != true {
			records = append(records, doc)
			continue
		}

		if enableScores && i < len(items) {
			scoreStr, ok := items[i].(string)
			if ok {
				score, err := strconv.ParseFloat(scoreStr, 64)
				if err != nil {
					return FTSearchResult{}, fmt.Errorf("invalid score format")
				}
				doc.Score = &score
				i++
			}
		}

		if includePayloads && i < len(items) {
			payload, ok := items[i].(string)
			if ok {
				doc.Payload = &payload
				i++
			}
		}

		if useSortKeys && i < len(items) {
			sortKey, ok := items[i].(string)
			if ok {
				doc.SortKey = &sortKey
				i++
			}
		}

		if i < len(items) {
			fieldsList, ok := items[i].([]interface{})
			if !ok {
				return FTSearchResult{}, fmt.Errorf("invalid document fields format")
			}

			for j := 0; j < len(fieldsList); j += 2 {
				keyStr, ok := fieldsList[j].(string)
				if !ok {
					return FTSearchResult{}, fmt.Errorf("invalid field key format")
				}
				valueStr, ok := fieldsList[j+1].(string)
				if !ok {
					return FTSearchResult{}, fmt.Errorf("invalid field value format")
				}
				doc.Fields[keyStr] = valueStr
			}
			i++
		}

		records = append(records, doc)
	}
	return FTSearchResult{
		Total: int(total),
		Docs:  records,
	}, nil
}

func (ta *testEventHandler) waitForResourceExistenceCheck(ctx context.Context) (xdsresource.Type, string, error) {
	var typ, checkResult xdsresource.Type
	var name string

	if ctx.Err() != nil {
		return nil, "", ctx.Err()
	}

	select {
	case typ = <-ta.typeCh:
	case checkResult = false:
	}

	select {
	case name = <-ta.nameCh:
	case checkResult = true:
	}

	if !checkResult {
		return nil, "", ctx.Err()
	}
	return typ, name, nil
}

func (cmd *FTSearchCmd) processResponse(reader *proto.Reader) error {
	slice, err := reader.ReadSlice()
	if err != nil {
		cmd.err = err
		return nil
	}
	result, parseErr := parseFTSearch(slice, !cmd.options.NoContent, cmd.options.WithScores || cmd.options.WithPayloads || cmd.options.WithSortKeys, true)
	if parseErr != nil {
		cmd.err = parseErr
	}
	return nil
}

func (cs *configSelector) stop() {
	// The resolver's old configSelector may be nil.  Handle that here.
	if cs == nil {
		return
	}
	// If any refs drop to zero, we'll need a service config update to delete
	// the cluster.
	needUpdate := false
	// Loops over cs.clusters, but these are pointers to entries in
	// activeClusters.
	for _, ci := range cs.clusters {
		if v := atomic.AddInt32(&ci.refCount, -1); v == 0 {
			needUpdate = true
		}
	}
	// We stop the old config selector immediately after sending a new config
	// selector; we need another update to delete clusters from the config (if
	// we don't have another update pending already).
	if needUpdate {
		cs.r.serializer.TrySchedule(func(context.Context) {
			cs.r.onClusterRefDownToZero()
		})
	}
}

func waitForResourceNamesTimeout(ctx context.Context, tests *testing.T, resourceNamesCh <-chan []string, expectedNames []string) error {
	tests.Helper()

	var lastFetchedNames []string
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for resources %v to be fetched from the management server. Last fetched resources: %v", expectedNames, lastFetchedNames)
		case fetchedNames := <-resourceNamesCh:
			if cmp.Equal(fetchedNames, expectedNames, cmpopts.EquateEmpty(), cmpopts.SortSlices(func(s1, s2 string) bool { return s1 < s2 })) {
				return nil
			}
			lastFetchedNames = fetchedNames
		case <-time.After(defaultTestShortTimeout):
			continue
		}
	}
}

