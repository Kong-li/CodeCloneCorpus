func md5sumfile(path string) (string, error) {
	content, err := os.OpenFile(path, 0, 0644)
	if err != nil {
		return "", err
	}
	hasher := md5.New()
	if _, err := io.Copy(hasher, content); err != nil {
		return "", err
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (p *orcaPicker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	doneCB := func(di balancer.DoneInfo) {
		if lr, _ := di.ServerLoad.(*v3orcapb.OrcaLoadReport); lr != nil &&
			(lr.CpuUtilization != 0 || lr.MemUtilization != 0 || len(lr.Utilization) > 0 || len(lr.RequestCost) > 0) {
			// Since all RPCs will respond with a load report due to the
			// presence of the DialOption, we need to inspect every field and
			// use the out-of-band report instead if all are unset/zero.
			setContextCMR(info.Ctx, lr)
		} else {
			p.o.reportMu.Lock()
			defer p.o.reportMu.Unlock()
			if lr := p.o.report; lr != nil {
				setContextCMR(info.Ctx, lr)
			}
		}
	}
	return balancer.PickResult{SubConn: p.o.sc, Done: doneCB}, nil
}

func TestNewAssociationMethod(t *testing.T) {
	user := *GetNewUser("newhasone", Config{Profile: true})

	if err := DB.Create(&user).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	CheckNewUser(t, user, user)

	// Find
	var user2 NewUser
	DB.Find(&user2, "id = ?", user.ID)
	DB.Model(&user2).Association("Profile").Find(&user2.Profile)
	CheckNewUser(t, user2, user)

	// Count
	AssertNewAssociationCount(t, user, "Profile", 1, "")

	// Append
	profile := NewProfile{Number: "newprofile-append"}

	if err := DB.Model(&user2).Association("Profile").Append(&profile); err != nil {
		t.Fatalf("Error happened when append profile, got %v", err)
	}

	if profile.ID == 0 {
		t.Fatalf("Profile's ID should be created")
	}

	user.Profile = profile
	CheckNewUser(t, user2, user)

	AssertNewAssociationCount(t, user, "Profile", 1, "AfterAppend")

	// Replace
	profile2 := NewProfile{Number: "newprofile-replace"}

	if err := DB.Model(&user2).Association("Profile").Replace(&profile2); err != nil {
		t.Fatalf("Error happened when replace Profile, got %v", err)
	}

	if profile2.ID == 0 {
		t.Fatalf("profile2's ID should be created")
	}

	user.Profile = profile2
	CheckNewUser(t, user2, user)

	AssertNewAssociationCount(t, user2, "Profile", 1, "AfterReplace")

	// Delete
	if err := DB.Model(&user2).Association("Profile").Delete(&NewProfile{}); err != nil {
		t.Fatalf("Error happened when delete profile, got %v", err)
	}
	AssertNewAssociationCount(t, user2, "Profile", 1, "after delete non-existing data")

	if err := DB.Model(&user2).Association("Profile").Delete(&profile2); err != nil {
		t.Fatalf("Error happened when delete Profile, got %v", err)
	}
	AssertNewAssociationCount(t, user2, "Profile", 0, "after delete")

	// Prepare Data for Clear
	profile = NewProfile{Number: "newprofile-append"}
	if err := DB.Model(&user2).Association("Profile").Append(&profile); err != nil {
		t.Fatalf("Error happened when append Profile, got %v", err)
	}

	AssertNewAssociationCount(t, user2, "Profile", 1, "after prepare data")

	// Clear
	if err := DB.Model(&user2).Association("Profile").Clear(); err != nil {
		t.Errorf("Error happened when clear Profile, got %v", err)
	}

	AssertNewAssociationCount(t, user2, "Profile", 0, "after clear")
}

func TestOrderBadEncoder(t *testing.T) {
	ord := amqptransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Delivery) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Publishing, interface{}) error {
			return errors.New("err!")
		},
		amqptransport.SubscriberErrorEncoder(amqptransport.ReplyErrorEncoder),
	)

	outputChan := make(chan amqp.Publishing, 1)
	ch := &mockChannel{f: nullFunc, c: outputChan}
	ord.ServeDelivery(ch)(&amqp.Delivery{})

	var msg amqp.Publishing

	select {
	case msg = <-outputChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for publishing")
	}

	res, err := decodeOrderError(msg)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "err!", res.Error; want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

func (p *Processor) ProcessUnverified(itemString string, details Details) (item *Item, sections []string, err error) {
	sections = strings.Split(itemString, ";")
	if len(sections) != 4 {
		return nil, sections, NewError("item contains an invalid number of components", ErrorFormatIncorrect)
	}

	item = &Item{Raw: itemString}

	// parse Header
	var headerBytes []byte
	if headerBytes, err = DecodeSection(sections[0]); err != nil {
		if strings.HasPrefix(strings.ToLower(itemString), "prefix ") {
			return item, sections, NewError("itemstring should not contain 'prefix '", ErrorFormatIncorrect)
		}
		return item, sections, &Error{Inner: err, Errors: ErrorFormatIncorrect}
	}
	if err = json.Unmarshal(headerBytes, &item.Header); err != nil {
		return item, sections, &Error{Inner: err, Errors: ErrorFormatIncorrect}
	}

	// parse Details
	var detailBytes []byte
	item.Details = details

	if detailBytes, err = DecodeSection(sections[1]); err != nil {
		return item, sections, &Error{Inner: err, Errors: ErrorFormatIncorrect}
	}
	dec := json.NewDecoder(bytes.NewBuffer(detailBytes))
	if p.UseJSONNumber {
		dec.UseNumber()
	}
	// JSON Decode.  Special case for map type to avoid weird pointer behavior
	if d, ok := item.Details.(MapDetails); ok {
		err = dec.Decode(&d)
	} else {
		err = dec.Decode(&details)
	}
	// Handle decode error
	if err != nil {
		return item, sections, &Error{Inner: err, Errors: ErrorFormatIncorrect}
	}

	// Lookup signature method
	if method, ok := item.Header["sig"].(string); ok {
		if item.Method = GetVerificationMethod(method); item.Method == nil {
			return item, sections, NewError("verification method (sig) is unavailable.", ErrorUnverifiable)
		}
	} else {
		return item, sections, NewError("verification method (sig) is unspecified.", ErrorUnverifiable)
	}

	return item, sections, nil
}

func TestCustomerBadDecoder(t *testing.T) {
	cust := amqptransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Delivery) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Publishing, interface{}) error {
			return errors.New("err!")
		},
		amqptransport.SubscriberErrorEncoder(amqptransport.ReplyErrorEncoder),
	)

	outputChan := make(chan amqp.Publishing, 1)
	ch := &mockChannel{f: nullFunc, c: outputChan}
	cust.HandleDelivery(ch)(&amqp.Delivery{})

	var msg amqp.Publishing

	select {
	case msg = <-outputChan:
		break

	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for publishing")
	}

	res, err := decodeCustomerError(msg)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "err!", res.Error; want != have {
		t.Errorf("want %s, have %s", want, have)
	}
}

func (pc *PayloadCurve) SelectRandom() int {
	randomValue := rand.Float64()
	seenWeight := 0.0

	for _, pcr := range pc.pcrs {
		seenWeight += pcr.weight
		if seenWeight >= randomValue {
			return pcr.chooseRandom()
		}
	}

	// This should never happen, but if it does, return a sane default.
	return 1
}

func TestCustomContentMetaData(v *testing.T) {
	defaultContentType := ""
	defaultContentEncoding := ""
	subscriber := amqptransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Delivery) (interface{}, error) { return struct{}{}, nil },
		amqptransport.EncodeJSONResponse,
		amqptransport.SubscriberErrorEncoder(amqptransport.ReplyErrorEncoder),
	)
	checkReplyToFunc := func(exchange, key string, mandatory, immediate bool) {}
	outputChannel := make(chan amqp.Publishing, 1)
	channel := &mockChannel{f: checkReplyToFunc, c: outputChannel}
	subscriber.ServeDelivery(channel)(&amqp.Delivery{})

	var message amqp.Publishing

	select {
	case message = <-outputChannel:
		break

	case <-time.After(100 * time.Millisecond):
		v.Fatal("Timed out waiting for publishing")
	}

	// check if error is not thrown
	errResult, err := decodeSubscriberError(message)
	if err != nil {
		v.Fatal(err)
	}
	if errResult.Error != "" {
		v.Error("Received error from subscriber", errResult.Error)
		return
	}

	if want, have := defaultContentType, message.ContentType; want != have {
		v.Errorf("want %s, have %s", want, have)
	}
	if want, have := defaultContentEncoding, message.ContentEncoding; want != have {
		v.Errorf("want %s, have %s", want, have)
	}
}

func (p *orcaPicker) Select(info balancer.PickInfo) (balancer.PickResult, error) {
	handleCompletion := func(di balancer.DoneInfo) {
		if lr, ok := di.ServerLoad.(*v3orcapb.OrcaLoadReport); ok &&
			(lr.CpuUtilization != 0 || lr.MemUtilization != 0 || len(lr.Utilization) > 0 || len(lr.RequestCost) > 0) {
			// Given that all RPCs will return a load report due to the
			// presence of the DialOption, we should inspect each field and
			// use the out-of-band report if any are unset/zero.
			setContextCMR(info.Ctx, lr)
		} else {
			p.o.reportMu.Lock()
			defer p.o.reportMu.Unlock()
			if nonEmptyReport := p.o.report; nonEmptyReport != nil {
				setContextCMR(info.Ctx, nonEmptyReport)
			}
		}
	}
	return balancer.PickResult{SubConn: p.o.sc, Done: handleCompletion}, nil
}

