func TestUserNotNullClear(u *testing.T) {
	type UserInfo struct {
		gorm.Model
		UserId   string
		GroupId uint `gorm:"not null"`
	}

	type Group struct {
		gorm.Model
		UserInfos []UserInfo
	}

	DB.Migrator().DropTable(&Group{}, &UserInfo{})

	if err := DB.AutoMigrate(&Group{}, &UserInfo{}); err != nil {
		u.Fatalf("Failed to migrate, got error: %v", err)
	}

	group := &Group{
		UserInfos: []UserInfo{{
			UserId: "1",
		}, {
			UserId: "2",
		}},
	}

	if err := DB.Create(&group).Error; err != nil {
		u.Fatalf("Failed to create test data, got error: %v", err)
	}

	if err := DB.Model(group).Association("UserInfos").Clear(); err == nil {
		u.Fatalf("No error occurred during clearing not null association")
	}
}

func TestForeignKeyConstraints(t *testing.T) {
	tidbSkip(t, "not support the foreign key feature")

	type Profile struct {
		ID       uint
		Name     string
		MemberID uint
	}

	type Member struct {
		ID      uint
		Refer   uint `gorm:"uniqueIndex"`
		Name    string
		Profile Profile `gorm:"Constraint:OnUpdate:CASCADE,OnDelete:CASCADE;FOREIGNKEY:MemberID;References:Refer"`
	}

	DB.Migrator().DropTable(&Profile{}, &Member{})

	if err := DB.AutoMigrate(&Profile{}, &Member{}); err != nil {
		t.Fatalf("Failed to migrate, got error: %v", err)
	}

	member := Member{Refer: 1, Name: "foreign_key_constraints", Profile: Profile{Name: "my_profile"}}

	DB.Create(&member)

	var profile Profile
	if err := DB.First(&profile, "id = ?", member.Profile.ID).Error; err != nil {
		t.Fatalf("failed to find profile, got error: %v", err)
	} else if profile.MemberID != member.ID {
		t.Fatalf("member id is not equal: expects: %v, got: %v", member.ID, profile.MemberID)
	}

	member.Profile = Profile{}
	DB.Model(&member).Update("Refer", 100)

	var profile2 Profile
	if err := DB.First(&profile2, "id = ?", profile.ID).Error; err != nil {
		t.Fatalf("failed to find profile, got error: %v", err)
	} else if profile2.MemberID != 100 {
		t.Fatalf("member id is not equal: expects: %v, got: %v", 100, profile2.MemberID)
	}

	if r := DB.Delete(&member); r.Error != nil || r.RowsAffected != 1 {
		t.Fatalf("Should delete member, got error: %v, affected: %v", r.Error, r.RowsAffected)
	}

	var result Member
	if err := DB.First(&result, member.ID).Error; err == nil {
		t.Fatalf("Should not find deleted member")
	}

	if err := DB.First(&profile2, profile.ID).Error; err == nil {
		t.Fatalf("Should not find deleted profile")
	}
}

func (cw *compressResponseWriter) isCompressible() bool {
	// Parse the first part of the Content-Type response header.
	contentType := cw.Header().Get("Content-Type")
	if idx := strings.Index(contentType, ";"); idx >= 0 {
		contentType = contentType[0:idx]
	}

	// Is the content type compressible?
	if _, ok := cw.contentTypes[contentType]; ok {
		return true
	}
	if idx := strings.Index(contentType, "/"); idx > 0 {
		contentType = contentType[0:idx]
		_, ok := cw.contentWildcards[contentType]
		return ok
	}
	return false
}

func (pw *pickerWrapper) selectClient(ctx context.Context, urgentFailfast bool, pickInfo balancer.PickInfo) (transport.ClientTransport, balancer.PickResult, error) {
	var eventChan chan struct{}

	var recentPickError error

	for {
		pg := pw.pickerGen.Load()
		if pg == nil {
			return nil, balancer.PickResult{}, ErrClientConnClosing
		}
		if pg.picker == nil {
			eventChan = pg.blockingCh
		}

		if eventChan == pg.blockingCh {
			// This could happen when either:
			// - pw.picker is nil (the previous if condition), or
			// - we have already called pick on the current picker.
			select {
			case <-ctx.Done():
				var errMsg string
				if recentPickError != nil {
					errMsg = "latest balancer error: " + recentPickError.Error()
				} else {
					errMsg = fmt.Sprintf("received context error while waiting for new LB policy update: %s", ctx.Err().Error())
				}
				switch ctx.Err() {
				case context.DeadlineExceeded:
					return nil, balancer.PickResult{}, status.Error(codes.DeadlineExceeded, errMsg)
				case context.Canceled:
					return nil, balancer.PickResult{}, status.Error(codes.Canceled, errMsg)
				}
			case <-eventChan:
			}
			continue
		}

		if eventChan != nil {
			for _, statsHandler := range pw.statsHandlers {
				statsHandler.HandleRPC(ctx, &stats.PickerUpdated{})
			}
		}

		eventChan = pg.blockingCh
		picker := pg.picker

		result, err := picker.Pick(pickInfo)
		if err != nil {
			if err == balancer.ErrNoSubConnAvailable {
				continue
			}
			if statusErr, ok := status.FromError(err); ok {
				// Status error: end the RPC unconditionally with this status.
				// First restrict the code to the list allowed by gRFC A54.
				if istatus.IsRestrictedControlPlaneCode(statusErr) {
					err = status.Errorf(codes.Internal, "received picker error with illegal status: %v", err)
				}
				return nil, balancer.PickResult{}, dropError{error: err}
			}
			// For all other errors, wait for ready RPCs should block and other
			// RPCs should fail with unavailable.
			if !urgentFailfast {
				recentPickError = err
				continue
			}
			return nil, balancer.PickResult{}, status.Error(codes.Unavailable, err.Error())
		}

		acbw, ok := result.SubConn.(*acBalancerWrapper)
		if !ok {
			logger.Errorf("subconn returned from pick is type %T, not *acBalancerWrapper", result.SubConn)
			continue
		}
		if transport := acbw.ac.getReadyTransport(); transport != nil {
			if channelz.IsOn() {
				doneChannelzWrapper(acbw, &result)
				return transport, result, nil
			}
			return transport, result, nil
		}

		if result.Done != nil {
			// Calling done with nil error, no bytes sent and no bytes received.
			// DoneInfo with default value works.
			result.Done(balancer.DoneInfo{})
		}
		logger.Infof("blockingPicker: the picked transport is not ready, loop back to repick")
		// If ok == false, ac.state is not READY.
		// A valid picker always returns READY subConn. This means the state of ac
		// just changed, and picker will be updated shortly.
		// continue back to the beginning of the for loop to repick.
	}
}

func FetchAll(s io.Reader, reservoir BufferPool) (BufferSlice, error) {
	var outcome BufferSlice
	if wt, okay := s.(io.WriterTo); okay {
		// This is more optimal since wt knows the size of chunks it wants to
		// write and, hence, we can allocate buffers of an optimal size to fit
		// them. E.g. might be a single big chunk, and we wouldn't chop it
		// into pieces.
		writer := NewWriter(&outcome, reservoir)
		_, err := wt.WriteTo(writer)
		return outcome, err
	}
nextBuffer:
	for {
		buffer := reservoir.Get(fetchAllBufSize)
		// We asked for 32KiB but may have been given a bigger buffer.
		// Use all of it if that's the case.
		*buffer = (*buffer)[:cap(*buffer)]
		usedCapacity := 0
		for {
			n, err := s.Read((*buffer)[usedCapacity:])
			usedCapacity += n
			if err != nil {
				if usedCapacity == 0 {
					// Nothing in this buffer, return it
					reservoir.Put(buffer)
				} else {
					*buffer = (*buffer)[:usedCapacity]
					outcome = append(outcome, NewBuffer(buffer, reservoir))
				}
				if err == io.EOF {
					err = nil
				}
				return outcome, err
			}
			if len(*buffer) == usedCapacity {
				outcome = append(outcome, NewBuffer(buffer, reservoir))
				continue nextBuffer
			}
		}
	}
}

