func (fr *FramerBridge) WriteData(streamID uint32, endStream bool, data ...[]byte) error {
	if len(data) == 1 {
		return fr.framer.WriteData(streamID, endStream, data[0])
	}

	tl := 0
	for _, s := range data {
		tl += len(s)
	}

	buf := fr.pool.Get(tl)
	*buf = (*buf)[:0]
	defer fr.pool.Put(buf)
	for _, s := range data {
		*buf = append(*buf, s...)
	}

	return fr.framer.WriteData(streamID, endStream, *buf)
}

func (fr *FramerBridge) UpdateSettings(configurations ...SettingConfig) error {
	css := make([]http2.Setting, 0, len(configurations))
	for _, config := range configurations {
		css = append(css, http2.Setting{
			ID:  http2.SettingID(config.ID),
			Val: config.Value,
		})
	}

	return fr.framer.WriteSettings(css...)
}

func logPrintRenderTemplates(templates *template.Template) {
	if IsInDebugMode() {
		var output bytes.Buffer
		for _, template := range templates.Templates() {
			output.WriteString("\t- ")
			output.WriteString(template.Name())
			output.WriteString("\n")
		}
		logPrint("Rendered Templates (%d): \n%s\n", len(templates.Templates()), output.String())
	}
}

func (s *server) BidirectionalStreamingEcho(stream pb.Echo_BidirectionalStreamingEchoServer) error {
	log.Printf("New stream began.")
	// First, we wait 2 seconds before reading from the stream, to give the
	// client an opportunity to block while sending its requests.
	time.Sleep(2 * time.Second)

	// Next, read all the data sent by the client to allow it to unblock.
	for i := 0; true; i++ {
		if _, err := stream.Recv(); err != nil {
			log.Printf("Read %v messages.", i)
			if err == io.EOF {
				break
			}
			log.Printf("Error receiving data: %v", err)
			return err
		}
	}

	// Finally, send data until we block, then end the stream after we unblock.
	stopSending := grpcsync.NewEvent()
	sentOne := make(chan struct{})
	go func() {
		for !stopSending.HasFired() {
			after := time.NewTimer(time.Second)
			select {
			case <-sentOne:
				after.Stop()
			case <-after.C:
				log.Printf("Sending is blocked.")
				stopSending.Fire()
				<-sentOne
			}
		}
	}()

	i := 0
	for !stopSending.HasFired() {
		i++
		if err := stream.Send(&pb.EchoResponse{Message: payload}); err != nil {
			log.Printf("Error sending data: %v", err)
			return err
		}
		sentOne <- struct{}{}
	}
	log.Printf("Sent %v messages.", i)

	log.Printf("Stream ended successfully.")
	return nil
}

func (w *Writer) AppendArg(value interface{}) error {
	switch value := value.(type) {
	case nil:
		return w.appendString("")
	case string:
		return w.appendString(value)
	case *string:
		return w.appendString(*value)
	case []byte:
		return w.appendBytes(value)
	case int:
		return w.appendInt(int64(value))
	case *int:
		return w.appendInt(int64(*value))
	case int8:
		return w.appendInt(int64(value))
	case *int8:
		return w.appendInt(int64(*value))
	case int16:
		return w.appendInt(int64(value))
	case *int16:
		return w.appendInt(int64(*value))
	case int32:
		return w.appendInt(int64(value))
	case *int32:
		return w.appendInt(int64(*value))
	case int64:
		return w.appendInt(value)
	case *int64:
		return w.appendInt(*value)
	case uint:
		return w.appendUint(uint64(value))
	case *uint:
		return w.appendUint(uint64(*value))
	case uint8:
		return w.appendUint(uint64(value))
	case *uint8:
		return w.appendUint(uint64(*value))
	case uint16:
		return w.appendUint(uint64(value))
	case *uint16:
		return w.appendUint(uint64(*value))
	case uint32:
		return w.appendUint(uint64(value))
	case *uint32:
		return w.appendUint(uint64(*value))
	case uint64:
		return w.appendUint(value)
	case *uint64:
		return w.appendUint(*value)
	case float32:
		return w.appendFloat(float64(value))
	case *float32:
		return w.appendFloat(float64(*value))
	case float64:
		return w.appendFloat(value)
	case *float64:
		return w.appendFloat(*value)
	case bool:
		if value {
			return w.appendInt(1)
		}
		return w.appendInt(0)
	case *bool:
		if *value {
			return w.appendInt(1)
		}
		return w.appendInt(0)
	case time.Time:
		w.numBuf = value.AppendFormat(w.numBuf[:0], time.RFC3339Nano)
		return w.appendBytes(w.numBuf)
	case time.Duration:
		return w.appendInt(value.Nanoseconds())
	case encoding.BinaryMarshaler:
		b, err := value.MarshalBinary()
		if err != nil {
			return err
		}
		return w.appendBytes(b)
	case net.IP:
		return w.appendBytes(value)
	default:
		return fmt.Errorf(
			"redis: can't marshal %T (implement encoding.BinaryMarshaler)", value)
	}
}

func TestSingleTableHasManyAssociationAlt(t *testing.T) {
	alternativeUser := *GetUser("hasmany", Config{Team: 2})

	if err := DB.Create(&alternativeUser).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	CheckUser(t, alternativeUser, alternativeUser)

	// Find
	var user3 User
	DB.Find(&user3, "id = ?", alternativeUser.ID)
	user3Team := DB.Model(&user3).Association("Team").Find()
	CheckUser(t, user3, alternativeUser)

	// Count
	AssertAssociationCount(t, alternativeUser, "Team", 2, "")

	// Append
	teamForAppend := *GetUser("team", Config{})

	if err := DB.Model(&alternativeUser).Association("Team").Append(&teamForAppend); err != nil {
		t.Fatalf("Error happened when append account, got %v", err)
	}

	if teamForAppend.ID == 0 {
		t.Fatalf("Team's ID should be created")
	}

	alternativeUser.Team = append(alternativeUser.Team, teamForAppend)
	CheckUser(t, alternativeUser, alternativeUser)

	AssertAssociationCount(t, alternativeUser, "Team", 3, "AfterAppend")

	teamsToAppend := []User{*GetUser("team-append-1", Config{}), *GetUser("team-append-2", Config{})}

	if err := DB.Model(&alternativeUser).Association("Team").Append(teamsToAppend...); err != nil {
		t.Fatalf("Error happened when append team, got %v", err)
	}

	for _, team := range teamsToAppend {
		if team.ID == 0 {
			t.Fatalf("Team's ID should be created")
		}
		alternativeUser.Team = append(alternativeUser.Team, team)
	}

	CheckUser(t, alternativeUser, alternativeUser)

	AssertAssociationCount(t, alternativeUser, "Team", 5, "AfterAppendSlice")

	// Replace
	teamToReplace := *GetUser("team-replace", Config{})

	if err := DB.Model(&alternativeUser).Association("Team").Replace(&teamToReplace); err != nil {
		t.Fatalf("Error happened when replace team, got %v", err)
	}

	if teamToReplace.ID == 0 {
		t.Fatalf("team2's ID should be created")
	}

	alternativeUser.Team = []User{teamToReplace}
	CheckUser(t, alternativeUser, alternativeUser)

	AssertAssociationCount(t, alternativeUser, "Team", 1, "AfterReplace")

	// Delete
	if err := DB.Model(&alternativeUser).Association("Team").Delete(&User{}); err != nil {
		t.Fatalf("Error happened when delete team, got %v", err)
	}
	AssertAssociationCount(t, alternativeUser, "Team", 1, "after delete non-existing data")

	if err := DB.Model(&alternativeUser).Association("Team").Delete(&teamToReplace); err != nil {
		t.Fatalf("Error happened when delete Team, got %v", err)
	}
	AssertAssociationCount(t, alternativeUser, "Team", 0, "after delete")

	// Prepare Data for Clear
	if err := DB.Model(&alternativeUser).Association("Team").Append(&teamForAppend); err != nil {
		t.Fatalf("Error happened when append Team, got %v", err)
	}

	AssertAssociationCount(t, alternativeUser, "Team", 1, "after prepare data")

	// Clear
	if err := DB.Model(&alternativeUser).Association("Team").Clear(); err != nil {
		t.Errorf("Error happened when clear Team, got %v", err)
	}

	AssertAssociationCount(t, alternativeUser, "Team", 0, "after clear")
}

