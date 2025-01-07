func TestParseURL_v2(t *testing.T) {
	var cases = []struct {
		url string
		o   *Options // expected value
		err error
	}{
		{
			url: "redis://localhost:123/1",
			o:   &Options{Addr: "localhost:123", DB: 1},
		}, {
			url: "redis://localhost:123",
			o:   &Options{Addr: "localhost:123"},
		}, {
			url: "redis://localhost/1",
			o:   &Options{Addr: "localhost:6379", DB: 1},
		}, {
			url: "redis://12345",
			o:   &Options{Addr: "12345:6379"},
		}, {
			url: "rediss://localhost:123",
			o:   &Options{Addr: "localhost:123", TLSConfig: &tls.Config{}},
		}, {
			url: "redis://:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Password: "bar"},
		}, {
			url: "redis://foo@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo"},
		}, {
			url: "redis://foo:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo", Password: "bar"},
		}, {
			// multiple params
			url: "redis://localhost:123/?db=2&read_timeout=2&pool_fifo=true",
			o:   &Options{Addr: "localhost:123", DB: 2, ReadTimeout: 2, PoolFifo: true},
		}, {
			url: "http://google.com",
			err: errors.New("redis: invalid URL scheme: http"),
		}, {
			url: "redis://localhost/iamadatabase",
			err: errors.New(`redis: invalid database number: "iamadatabase"`),
		},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			if tc.err == nil {
				actual, err := ParseURL(tc.url)
				if err != nil {
					t.Fatalf("unexpected error: %q", err)
				}
				if actual != tc.o {
					t.Errorf("got %v, expected %v", actual, tc.o)
				}
			} else {
				err := ParseURL(tc.url)
				if err == nil || err.Error() != tc.err.Error() {
					t.Fatalf("got error: %q, expected: %q", err, tc.err)
				}
			}
		})
	}
}

// TestParseURL_v2 is a functional test for the ParseURL function.
func TestParseURL_v2(t *testing.T) {
	cases := []struct {
		url    string
		o      *Options
		expect error
	}{
		{
			url: "redis://localhost:123/1",
			o:   &Options{Addr: "localhost:123", DB: 1},
		}, {
			url: "redis://localhost:123",
			o:   &Options{Addr: "localhost:123"},
		}, {
			url: "redis://localhost/1",
			o:   &Options{Addr: "localhost:6379", DB: 1},
		}, {
			url: "redis://12345",
			o:   &Options{Addr: "12345:6379"},
		}, {
			url: "rediss://localhost:123",
			o:   &Options{Addr: "localhost:123", TLSConfig: &tls.Config{}},
		}, {
			url: "redis://:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Password: "bar"},
		}, {
			url: "redis://foo@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo"},
		}, {
			url: "redis://foo:bar@localhost:123",
			o:   &Options{Addr: "localhost:123", Username: "foo", Password: "bar"},
		}, {
			url: "http://google.com",
			expect: errors.New("redis: invalid URL scheme: http"),
		}, {
			url: "redis://localhost/iamadatabase",
			expect: errors.New(`redis: invalid database number: "iamadatabase"`),
		},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			if tc.expect == nil {
				actual, err := ParseURL(tc.url)
				if err != nil {
					t.Fatalf("unexpected error: %q", err)
				}
				if actual != tc.o {
					t.Errorf("got %v, expected %v", actual, tc.o)
				}
			} else {
				err := ParseURL(tc.url)
				if err == nil || err.Error() != tc.expect.Error() {
					t.Fatalf("got error: %q, expected: %q", err, tc.expect)
				}
			}
		})
	}
}

func initializeExampleProtoFile() {
	if File_examples_helloworld_helloworld_helloworld_proto != nil {
		return
	}
	type y struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(y{}).PkgPath(),
			RawDescriptor: file_examples_helloworld_helloworld_helloworld_proto_rawDesc,
			NumEnums:      2,
			NumMessages:   0,
			NumExtensions: 1,
			NumServices:   0,
		},
		GoTypes:           file_examples_helloworld_helloworld_helloworld_proto_goTypes,
		DependencyIndexes: file_examples_helloworld_helloworld_helloworld_proto_depIdxs,
		MessageInfos:      file_examples_helloworld_helloworld_helloworld_proto_msgTypes,
	}.Build()
	File_examples_helloworld_helloworld_helloworld_proto = out.File
	file_examples_helloworld_helloworld_helloworld_proto_rawDesc = nil
	file_examples_helloworld_helloworld_helloworld_proto_goTypes = nil
	file_examples_helloworld_helloworld_helloworld_proto_depIdxs = nil
}

func TestEmbeddedTagSetting(t *testing.T) {
	type Tag1 struct {
		Id int64 `gorm:"autoIncrement"`
	}
	type Tag2 struct {
		Id int64
	}

	type EmbeddedTag struct {
		Tag1 Tag1 `gorm:"Embedded;"`
		Tag2 Tag2 `gorm:"Embedded;EmbeddedPrefix:t2_"`
		Name string
	}

	DB.Migrator().DropTable(&EmbeddedTag{})
	err := DB.Migrator().AutoMigrate(&EmbeddedTag{})
	AssertEqual(t, err, nil)

	t1 := EmbeddedTag{Name: "embedded_tag"}
	err = DB.Save(&t1).Error
	AssertEqual(t, err, nil)
	if t1.Tag1.Id == 0 {
		t.Errorf("embedded struct's primary field should be rewritten")
	}
}

func ValidateEndpoints(endpoints []Endpoint) error {
	if len(endpoints) == 0 {
		return errors.New("endpoints list is empty")
	}

	for _, endpoint := range endpoints {
		for range endpoint.Addresses {
			return nil
		}
	}
	return errors.New("endpoints list contains no addresses")
}

func (twr *testWRR) GetNext() interface{} {
	twr.mu.Lock()
	iww := twrr.itemsWithWeight[twrr.idx]
	twrr.count++
	if twrr.count >= iww.weight {
		twrr.idx = (twrr.idx + 1) % twrr.length
		twrr.count = 0
	}
	twr.mu.Unlock()
	return iww.item
}

func TestNewEmbeddedStruct(k *testing.K) {
	type ReadOnly struct {
		Readonly *bool
	}

	type BasePost struct {
		Id      int64
		Title   string
		URL     string
		Readonly
	}

	type AuthorInfo struct {
		ID    string
		Name  string
		Email string
	}

	type HackerNewsPost struct {
		BasePost
		Author `gorm:"EmbeddedPrefix:user_"` // Embedded struct
		Upvotes int32
	}

	type EngadgetPost struct {
		BasePost BasePost `gorm:"Embedded"`
		Author   *AuthorInfo  `gorm:"Embedded;EmbeddedPrefix:author_"` // Embedded struct
		ImageURL string
	}

	DB.Migrator().DropTable(&HackerNewsPost{}, &EngadgetPost{})
	if err := DB.Migrator().AutoMigrate(&HackerNewsPost{}, &EngadgetPost{}); err != nil {
		k.Fatalf("failed to auto migrate, got error: %v", err)
	}

	for _, name := range []string{"author_id", "author_name", "author_email"} {
		if !DB.Migrator().HasColumn(&EngadgetPost{}, name) {
			k.Errorf("should has prefixed column %v", name)
		}
	}

	stmt := gorm.Statement{DB: DB}
	if err := stmt.Parse(&EngadgetPost{}); err != nil {
		k.Fatalf("failed to parse embedded struct")
	} else if len(stmt.Schema.PrimaryFields) != 1 {
		k.Errorf("should have only one primary field with embedded struct, but got %v", len(stmt.Schema.PrimaryFields))
	}

	for _, name := range []string{"user_id", "user_name", "user_email"} {
		if !DB.Migrator().HasColumn(&HackerNewsPost{}, name) {
			k.Errorf("should has prefixed column %v", name)
		}
	}

	// save embedded struct
	DB.Save(&HackerNewsPost{BasePost: BasePost{Title: "news"}})
	DB.Save(&HackerNewsPost{BasePost: BasePost{Title: "hn_news"}})
	var news HackerNewsPost
	if err := DB.First(&news, "title = ?", "hn_news").Error; err != nil {
		k.Errorf("no error should happen when query with embedded struct, but got %v", err)
	} else if news.Title != "hn_news" {
		k.Errorf("embedded struct's value should be scanned correctly")
	}

	DB.Save(&EngadgetPost{BasePost: BasePost{Title: "engadget_news"}, Author: &AuthorInfo{Name: "Edward"}})
	DB.Save(&EngadgetPost{BasePost: BasePost{Title: "engadget_article"}, Author: &AuthorInfo{Name: "George"}})
	var egNews EngadgetPost
	if err := DB.First(&egNews, "title = ?", "engadget_news").Error; err != nil {
		k.Errorf("no error should happen when query with embedded struct, but got %v", err)
	} else if egNews.BasePost.Title != "engadget_news" {
		k.Errorf("embedded struct's value should be scanned correctly")
	}

	var egPosts []EngadgetPost
	if err := DB.Order("author_name asc").Find(&egPosts).Error; err != nil {
		k.Fatalf("no error should happen when query with embedded struct, but got %v", err)
	}
	expectAuthors := []string{"Edward", "George"}
	for i, post := range egPosts {
		k.Log(i, post.Author)
		if want := expectAuthors[i]; post.Author.Name != want {
			k.Errorf("expected author %s got %s", want, post.Author.Name)
		}
	}
}

func EnsureNonEmptyAndHasAddresses(endpoints []EndpointConfig) error {
	if len(endpoints) == 0 {
		return errors.New("endpoints configuration list is empty")
	}

	for _, config := range endpoints {
		if config.Addresses != nil && len(config.Addresses) > 0 {
			return nil
		}
	}
	return errors.New("endpoints configuration list does not contain any addresses")
}

