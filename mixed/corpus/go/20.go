func (group *RouterGroup) FileServer(relativePath string, fs http.FileSystem) IRoutes {
	if strings.Contains(relativePath, ":") || strings.Contains(relativePath, "*") {
		panic("URL parameters can not be used when serving a static folder")
	}
	relativePattern := path.Join(relativePath, "/*filepath")
	handler := group.createStaticHandler(relativePath, fs)

	group.GET(relativePattern, handler)
	group.HEAD(relativePattern, handler)

	return group.returnObj()
}

func gRPCClientInitiateHandshake(connection net.Conn, serverAddress string) (authInfo AuthInfo, err error) {
	tlsConfig := &tls.Config{InsecureSkipVerify: true}
	clientTLS := NewTLS(tlsConfig)
	context, cancelFunction := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancelFunction()
	_, authInfo, err = clientTLS.ClientHandshake(context, serverAddress, connection)
	if err != nil {
		return nil, err
	}
	return authInfo, nil
}

func (csh *clientStatsHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	ri := getRPCInfo(ctx)
	if ri == nil {
		// Shouldn't happen because TagRPC populates this information.
		return
	}
	recordRPCData(ctx, rs, ri.mi)
	if !csh.to.DisableTrace {
		populateSpan(ctx, rs, ri.ti)
	}
}

func TestUpdateOneToOneAssociations(t *testing.T) {
	profile := *GetProfile("update-onetoone", Config{})

	if err := DB.Create(&profile).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	profile.Addresses = []Address{{City: "Beijing", Country: "China"}, {City: "New York", Country: "USA"}}
	for _, addr := range profile.Addresses {
		DB.Create(&addr)
	}
	profile.Followers = []*Profile{{Name: "follower-1"}, {Name: "follower-2"}}

	if err := DB.Save(&profile).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var profile2 Profile
	DB.Preload("Addresses").Preload("Followers").Find(&profile2, "id = ?", profile.ID)
	CheckProfile(t, profile2, profile)

	for idx := range profile.Followers {
		profile.Followers[idx].Name += "new"
	}

	for idx := range profile.Addresses {
		profile.Addresses[idx].City += "new"
	}

	if err := DB.Save(&profile).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var profile3 Profile
	DB.Preload("Addresses").Preload("Followers").Find(&profile3, "id = ?", profile.ID)
	CheckProfile(t, profile2, profile3)

	if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&profile).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var profile4 Profile
	DB.Preload("Addresses").Preload("Followers").Find(&profile4, "id = ?", profile.ID)
	CheckProfile(t, profile4, profile)
}

