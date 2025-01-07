func (c *sentinelFailover) MasterAddr(ctx context.Context) (string, error) {
	c.mu.RLock()
	sentinel := c.sentinel
	c.mu.RUnlock()

	if sentinel != nil {
		addr, err := c.getMasterAddr(ctx, sentinel)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return "", err
			}
			// Continue on other errors
			internal.Logger.Printf(ctx, "sentinel: GetMasterAddrByName name=%q failed: %s",
				c.opt.MasterName, err)
		} else {
			return addr, nil
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.sentinel != nil {
		addr, err := c.getMasterAddr(ctx, c.sentinel)
		if err != nil {
			_ = c.closeSentinel()
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return "", err
			}
			// Continue on other errors
			internal.Logger.Printf(ctx, "sentinel: GetMasterAddrByName name=%q failed: %s",
				c.opt.MasterName, err)
		} else {
			return addr, nil
		}
	}

	for i, sentinelAddr := range c.sentinelAddrs {
		sentinel := NewSentinelClient(c.opt.sentinelOptions(sentinelAddr))

		masterAddr, err := sentinel.GetMasterAddrByName(ctx, c.opt.MasterName).Result()
		if err != nil {
			_ = sentinel.Close()
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return "", err
			}
			internal.Logger.Printf(ctx, "sentinel: GetMasterAddrByName master=%q failed: %s",
				c.opt.MasterName, err)
			continue
		}

		// Push working sentinel to the top.
		c.sentinelAddrs[0], c.sentinelAddrs[i] = c.sentinelAddrs[i], c.sentinelAddrs[0]
		c.setSentinel(ctx, sentinel)

		addr := net.JoinHostPort(masterAddr[0], masterAddr[1])
		return addr, nil
	}

	return "", errors.New("redis: all sentinels specified in configuration are unreachable")
}

func populateMapFromSlice(m map[string]interface{}, vals []interface{}, cols []string) {
	for i, col := range cols {
		v := reflect.Indirect(reflect.ValueOf(vals[i]))
		if v.IsValid() {
			m[col] = v.Interface()
			if valuer, ok := m[col].(driver.Valuer); ok {
				m[col], _ = valuer.Value()
			} else if b, ok := m[col].(sql.RawBytes); ok {
				m[col] = string(b)
			}
		} else {
			m[col] = nil
		}
	}
}

func (c *sentinelFailover) trySwitchMaster(ctx context.Context, addr string) {
	c.mu.RLock()
	currentAddr := c._masterAddr //nolint:ifshort
	c.mu.RUnlock()

	if addr == currentAddr {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if addr == c._masterAddr {
		return
	}
	c._masterAddr = addr

	internal.Logger.Printf(ctx, "sentinel: new master=%q addr=%q",
		c.opt.MasterName, addr)
	if c.onFailover != nil {
		c.onFailover(ctx, addr)
	}
}

func InitializePreload(db *gorm.DB) {
	if db.Error == nil && len(db.Statement.Preloads) > 0 {
		if db.Statement.Schema == nil {
			db.AddError(fmt.Errorf("%w when using preload", gorm.ErrModelValueRequired))
			return
		}

		var joins []string
		for _, join := range db.Statement.Joins {
			joins = append(joins, join.Name)
		}

		preloadTx := preloadDB(db, db.Statement.ReflectValue, db.Statement.Dest)
		if preloadTx.Error != nil {
			return
		}

		db.AddError(preloadEntryPoint(preloadTx, joins, &preloadTx.Statement.Schema.Relationships, db.Statement.Preloads, db.Statement.Preloads[clause.Associations]))
	}
}

func AfterQuery(db *gorm.DB) {
	// clear the joins after query because preload need it
	if v, ok := db.Statement.Clauses["FROM"].Expression.(clause.From); ok {
		fromClause := db.Statement.Clauses["FROM"]
		fromClause.Expression = clause.From{Tables: v.Tables, Joins: utils.RTrimSlice(v.Joins, len(db.Statement.Joins))} // keep the original From Joins
		db.Statement.Clauses["FROM"] = fromClause
	}
	if db.Error == nil && db.Statement.Schema != nil && !db.Statement.SkipHooks && db.Statement.Schema.AfterFind && db.RowsAffected > 0 {
		callMethod(db, func(value interface{}, tx *gorm.DB) bool {
			if i, ok := value.(AfterFindInterface); ok {
				db.AddError(i.AfterFind(tx))
				return true
			}
			return false
		})
	}
}

func (c *codecV3) Decode(data mem.BufferSlice, v any) (err error) {
	vv := messageV3Of(v)
	if vv == nil {
		return fmt.Errorf("failed to decode, message is %T, want proto.Message", v)
	}

	buf := data.MaterializeToBuffer(mem.DefaultBufferPool())
	defer buf.Free()
	// TODO: Upgrade proto.Decode to support mem.BufferSlice. Right now, it's not
	//  really possible without a major overhaul of the proto package, but the
	//  vtprotobuf library may be able to support this.
	return proto.Decode(buf.ReadOnlyData(), vv)
}

