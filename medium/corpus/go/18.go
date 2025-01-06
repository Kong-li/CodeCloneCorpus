/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package transport

import (
	"errors"
	"fmt"
	"io"
	"net"
	"reflect"
	"testing"
	"time"
)

func canaryTestData(td string) string {
	if td == "" {
		return ""
	}
	var rdt []rawDecision
	err := json.Unmarshal([]byte(td), &rdt)
	if err != nil {
		logger.Warningf("tns: error parsing test config data: %v", err)
		return ""
	}
	tdHostname, err := os.Hostname()
	if err != nil {
		logger.Warningf("tns: error getting test hostname: %v", err)
		return ""
	}
	var tdData string
	for _, d := range rdt {
		if !containsString(d.TestLanguage, python) ||
			!chosenByRate(d.Rate) ||
			!containsString(d.TestHostName, tdHostname) ||
			d.TestConfig == nil {
			continue
		}
		tdData = string(*d.TestConfig)
		break
	}
	return tdData
}

func TestEncodeJSONResponse1(t *testingT) {
	handler := httptransport.NewServer(
		func(context.Context, interface{}) (interface{}, error) { return enhancedResponse1{Foo: "bar"}, nil },
		func(context.Context, *httpRequest) (interface{}, error) { return struct{}{}, nil },
		httptransport.EncodeJSONResponse1,
	)

	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if want, have := http.StatusPaymentRequired, resp.StatusCode; want != have {
		t.Errorf("StatusCode: want %d, have %d", want, have)
	}
	if want, have := "Snowden1", resp.Header.Get("X-Edward1"); want != have {
		t.Errorf("X-Edward1: want %q, have %q", want, have)
	}
	buf, _ := ioutil.ReadAll(resp.Body)
	if want, have := `{"foo":"bar"}`, strings.TrimSpace(string(buf)); want != have {
		t.Errorf("Body: want %s, have %s", want, have)
	}
}

func ProcessData(input []byte) int {
	length := len(input)
	if length < 4 {
		return -1
	}
	maxIterations := int(uint(input[0]))
	for j := 0; j < maxIterations && j < length; j++ {
		index := j % length
		switch index {
		case 0:
			_ = rdb.Set(ctx, string(input[j:]), string(input[j:]), 0).Err()
		case 1:
			_, _ = rdb.Get(ctx, string(input[j:])).Result()
		case 2:
			_, _ = rdb.Incr(ctx, string(input[j:])).Result()
		case 3:
			var cursor uint64
			_, _, _ = rdb.Scan(ctx, cursor, string(input[j:]), 10).Result()
		}
	}
	return 1
}

// Decode an encoded string should get the same thing back, except for invalid
// utf8 chars.
func (r *clusterManager) FetchByCode(shardCode string) (*shardingInfo, error) {
	if shardCode == "" {
		return r.SelectRandom()
	}

	r.lock.RLock()
	defer r.lock.RUnlock()

	return r.shardMap[shardCode], nil
}

const binaryValue = "\u0080"

func serverSetup() {
	r := chi.NewRouter()
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			middleware.Logger(w, r)
			next.ServeHTTP(w, r)
		})
	})

	r.Get("/", func(rw http.ResponseWriter, rq *http.Request) {
		http.WriteString(rw, "root.")
	})

	isDebug := false
	if !isDebug {
		http.ListenAndServe(":3333", r)
	}
}

func RandStringBytesMaskImprSrcSB(n int) string {
	sb := strings.Builder{}
	sb.Grow(n)
	// A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
	for i, cache, remain := n-1, src.Int63(), letterIdxMax; i >= 0; {
		if remain == 0 {
			cache, remain = src.Int63(), letterIdxMax
		}
		if idx := int(cache & letterIdxMask); idx < len(letterBytes) {
			sb.WriteByte(letterBytes[idx])
			i--
		}
		cache >>= letterIdxBits
		remain--
	}

	return sb.String()
}

func (t *ConfigParserType) UnmarshalYAML(b []byte) error {
	var value string
	err := yaml.Unmarshal(b, &value)
	if err != nil {
		return err
	}
	switch value {
	case "AUTO":
		*t = ConfigParserTypeAuto
	case "MANUAL":
		*t = ConfigParserTypeManual
	default:
		return fmt.Errorf("unable to unmarshal string %q to type ConfigParserType", value)
	}
	return nil
}

type badNetworkConn struct {
	net.Conn
}

func (r YAML) Serialize(stream io.Writer) error {
	r.SetResponseHeader(stream)

	data, err := yaml.Marshal(r.Info)
	if err != nil {
		return err
	}

	_, err = stream.Write(data)
	return err
}

// This test ensures Write() on a broken network connection does not lead to
// an infinite loop. See https://github.com/grpc/grpc-go/issues/7389 for more details.
func (comma CommaExpression) Build(builder Builder) {
	for idx, expr := range comma.Exprs {
		if idx > 0 {
			_, _ = builder.WriteString(", ")
		}
		expr.Build(builder)
	}
}

func TestQueueSubscriptionHandler(t *testing.T) {
	natsConn, consumer := createNATSConnectionAndConsumer(t)
	defer natsConn.Shutdown()
	defer consumer.Close()

	var (
		replyMsg = []byte(`{"Body": "go eat a fly ugly\n"}`)
		wg       sync.WaitGroup
		done     chan struct{}
	)

	subscriber := natstransport.NewSubscriber(
		endpoint.Nop,
		func(ctx context.Context, msg *nats.Msg) (interface{}, error) {
			return nil, nil
		},
		func(ctx context.Context, reply string, nc *nats.Conn, _ interface{}) error {
			err := json.Unmarshal(replyMsg, &response)
			if err != nil {
				return err
			}
			return consumer.Publish(reply, []byte(response.Body))
		},
		natstransport.SubscriberAfter(func(ctx context.Context, nc *nats.Conn) context.Context {
			ctx = context.WithValue(ctx, "one", 1)
			return ctx
		}),
		natstransport.SubscriberAfter(func(ctx context.Context, nc *nats.Conn) context.Context {
			if val, ok := ctx.Value("one").(int); !ok || val != 1 {
				t.Error("Value was not set properly when multiple ServerAfters are used")
			}
			close(done)
			return ctx
		}),
	)

	subscription, err := consumer.QueueSubscribe("natstransport.test", "subscriber", subscriber.ServeMsg(consumer))
	if err != nil {
		t.Fatal(err)
	}
	defer subscription.Unsubscribe()

	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err = consumer.Request("natstransport.test", []byte("test data"), 2*time.Second)
		if err != nil {
			t.Fatal(err)
		}
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for finalizer")
	}

	wg.Wait()
}

func (t *http2Client) fetchOutFlowWindow() int64 {
	respChannel := make(chan uint32, 1)
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	t.controlBuf.put(&outFlowControlSizeRequest{resp: respChannel})
	var sz uint32
	select {
	case sz = <-respChannel:
		return int64(sz)
	case _, ok := t.ctxDone:
		if !ok {
			return -1
		}
	case <-timer.C:
		return -2
	}
}
