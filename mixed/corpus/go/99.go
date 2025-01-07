func ExampleClient_hset() {
	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	rdb.Del(ctx, "myhash")
	// REMOVE_END

	// STEP_START hset
	res1, err := rdb.HSet(ctx, "myhash", "field1", "Hello").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res1) // >>> 1

	res2, err := rdb.HGet(ctx, "myhash", "field1").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res2) // >>> Hello

	res3, err := rdb.HSet(ctx, "myhash",
		"field2", "Hi",
		"field3", "World",
	).Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res3) // >>> 2

	res4, err := rdb.HGet(ctx, "myhash", "field2").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res4) // >>> Hi

	res5, err := rdb.HGet(ctx, "myhash", "field3").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res5) // >>> World

	res6, err := rdb.HGetAll(ctx, "myhash").Result()

	if err != nil {
		panic(err)
	}

	fmt.Println(res6)
	// >>> map[field1:Hello field2:Hi field3:World]
	// STEP_END

	// Output:
	// 1
	// Hello
	// 2
	// Hi
	// World
	// map[field1:Hello field2:Hi field3:World]
}

func (s) TestParse1(t *testing.T) {
	tests := []struct {
		name    string
		sc      string
		want    serviceconfig.LoadBalancingConfig
		wantErr bool
	}{
		{
			name:    "empty",
			sc:      "",
			want:    nil,
			wantErr: true,
		},
		{
			name: "success1",
			sc: `
{
	"childPolicy": [
		{"pick_first":{}}
	],
	"serviceName": "bar-service"
}`,
			want: &grpclbServiceConfig{
				ChildPolicy: &[]map[string]json.RawMessage{
					{"pick_first": json.RawMessage("{}")},
				},
				ServiceName: "bar-service",
			},
		},
		{
			name: "success2",
			sc: `
{
	"childPolicy": [
		{"round_robin":{}},
		{"pick_first":{}}
	],
	"serviceName": "bar-service"
}`,
			want: &grpclbServiceConfig{
				ChildPolicy: &[]map[string]json.RawMessage{
					{"round_robin": json.RawMessage("{}")},
					{"pick_first": json.RawMessage("{}")},
				},
				ServiceName: "bar-service",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := (&lbBuilder{}).ParseConfig(json.RawMessage(tt.sc))
			if (err != nil) != (tt.wantErr) {
				t.Fatalf("ParseConfig(%q) returned error: %v, wantErr: %v", tt.sc, err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("ParseConfig(%q) returned unexpected difference (-want +got):\n%s", tt.sc, diff)
			}
		})
	}
}

func configureUnixConnection(u *url.URL) (*Options, error) {
	o := &Options{
		Network: "unix",
	}

	if u.Path == "" { // path is required with unix connection
		return nil, errors.New("redis: empty unix socket path")
	}
	o.Addr = strings.TrimSpace(u.Path)
	username, password := getUserPassword(u)
	o.Username = username
	o.Password = password

	return setupConnParams(u, o)
}

