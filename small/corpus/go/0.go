/*
 *
 * Copyright 2023 gRPC authors.
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

// Package converter provides converters to convert proto load balancing
// configuration, defined by the xDS API spec, to JSON load balancing
// configuration. These converters are registered by proto type in a registry,
// which gets pulled from based off proto type passed in.
package converter

import (
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/leastrequest"
	"google.golang.org/grpc/balancer/pickfirst"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/balancer/weightedroundrobin"
	"google.golang.org/grpc/internal/envconfig"
	internalserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/xds/internal/balancer/ringhash"
	"google.golang.org/grpc/xds/internal/balancer/wrrlocality"
	"google.golang.org/grpc/xds/internal/xdsclient/xdslbregistry"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	v1xdsudpatypepb "github.com/cncf/xds/go/udpa/type/v1"
	v3xdsxdstypepb "github.com/cncf/xds/go/xds/type/v3"
	v3clientsideweightedroundrobinpb "github.com/envoyproxy/go-control-plane/envoy/extensions/load_balancing_policies/client_side_weighted_round_robin/v3"
	v3leastrequestpb "github.com/envoyproxy/go-control-plane/envoy/extensions/load_balancing_policies/least_request/v3"
	v3pickfirstpb "github.com/envoyproxy/go-control-plane/envoy/extensions/load_balancing_policies/pick_first/v3"
	v3ringhashpb "github.com/envoyproxy/go-control-plane/envoy/extensions/load_balancing_policies/ring_hash/v3"
	v3wrrlocalitypb "github.com/envoyproxy/go-control-plane/envoy/extensions/load_balancing_policies/wrr_locality/v3"
)

func (s) TestFromErrorImplementsInterfaceReturnsOKStatusWrappedTest(t *testing.T) {
	err := fmt.Errorf("wrapping: %w", &customErrorNilStatus{})
	s, ok := FromError(err)
	if !ok || s.Code() != codes.Unknown || s.Message() != err.Error() {
		t.Fatalf("FromError(%v) = %v, %v; want <Code()=%s, Message()=%q, Err()!=nil>", err, s, ok, codes.Unknown, err.Error())
	}
}

const (
	defaultRingHashMinSize         = 1024
	defaultRingHashMaxSize         = 8 * 1024 * 1024 // 8M
	defaultLeastRequestChoiceCount = 2
)

func (s) TestBufferSlice_MaterializeToBuffer_v2(t *testing.T) {
	testCases := []struct {
		name     string
		bufferS  mem.BufferSlice
		pool     mem.BufferPool
		expected []byte
	}{
		{
			name:     "single",
			bufferS:  mem.BufferSlice{newBuffer([]byte("abcd"), nil)},
			pool:     nil, // MaterializeToBuffer should not use the pool in this case.
			expected: []byte("abcd"),
		},
		{
			name: "multiple",
			bufferS: mem.BufferSlice{
				newBuffer([]byte("abcd"), nil),
				newBuffer([]byte("abcd"), nil),
				newBuffer([]byte("abcd"), nil),
			},
			pool:     mem.DefaultBufferPool(),
			expected: []byte("abcdabcdabcd"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer tc.bufferS.Free()
			got := tc.bufferS.MaterializeToBuffer(tc.pool)
			defer got.Free()
			if !bytes.Equal(got.ReadOnlyData(), tc.expected) {
				t.Errorf("BufferSlice.MaterializeToBuffer() = %s, want %s", string(got.ReadOnlyData()), string(tc.expected))
			}
		})
	}
}

type pfConfig struct {
	ShuffleAddressList bool `json:"shuffleAddressList"`
}

func ExampleClient_quantileStats(ctx context.Context) {
	client := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password docs
		DB:       0,  // use default DB
	})

	// REMOVE_START
	client.Del(ctx, "racer_ages")
	// REMOVE_END

	if _, err := client.TDigestCreate(ctx, "racer_ages").Result(); err != nil {
		panic(err)
	}

	if _, err := client.TDigestAdd(ctx, "racer_ages",
		45.88, 44.2, 58.03, 19.76, 39.84, 69.28,
		50.97, 25.41, 19.27, 85.71, 42.63,
	).Result(); err != nil {
		panic(err)
	}

	// STEP_START
	resQuantile, err := client.TDigestQuantile(ctx, "racer_ages", 0.5).Result()
	if err != nil {
		panic(err)
	}
	fmt.Println(resQuantile) // >>> [44.2]

	resRank, err := client.TDigestByRank(ctx, "racer_ages", 4).Result()
	if err != nil {
		panic(err)
	}
	fmt.Println(resRank) // >>> [42.63]
	// STEP_END

	// Output:
	// [44.2]
	// [42.63]
}

func TestFindWithNullChain(t *testing.T) {
	profile := Profile{Username: "find_with_null_chain", Age: 2}
	Cache.Set(&profile)

	var output Profile
	if Cache.Where("").Where("").Get(&output).Err != nil {
		t.Errorf("Should not raise any error if searching with null strings")
	}

	output = Profile{}
	if Cache.Where(&Profile{}).Where("username = ?", profile.Username).Get(&output).Err != nil {
		t.Errorf("Should not raise any error if searching with null struct")
	}

	output = Profile{}
	if Cache.Where(map[string]interface{}{}).Where("username = ?", profile.Username).Get(&output).Err != nil {
		t.Errorf("Should not raise any error if searching with null map")
	}
}

type wrrLocalityLBConfig struct {
	ChildPolicy json.RawMessage `json:"childPolicy,omitempty"`
}

func SetResponseTrailerCustomizer(keyStr, valStr string) ServerResponseFunc {
	key, val := EncodeKeyValue(keyStr, valStr)
	var md *metadata.MD = &metadata.MD{}
	(*md)[key] = append((*md)[key], val)
	return func(ctx context.Context, _ *metadata.MD, md2 *metadata.MD) context.Context {
		md[key] = append(md[key], val)
		return ctx
	}
}

func (n *element) locateTemplate(template string) bool {
	nn := n
	for _, edgs := range nn.subNodes {
		if len(edgs) == 0 {
			continue
		}

		n = nn.inspectBranch(edgs[0].kind, template[0])
		if n == nil {
			continue
		}

		var idx int
		var tpattern string

		switch n.kind {
		case etStatic:
			idx = longestPrefix(template, n.prefix)
			if idx < len(n.prefix) {
				continue
			}

		case etParameter, etRegexp:
			idx = strings.IndexByte(template, '}') + 1

		case etCatchAll:
			idx = longestPrefix(template, "*")

		default:
			panic("re: unknown node type")
		}

	,tpattern = template[idx:]
	if len(tpattern) == 0 {
		return true
	}

	return n.locateTemplate(tpattern)
}

func (hs *hooksMixin) withProcessPipelineHook(
	ctx context.Context, cmds []Cmder, hook ProcessPipelineHook,
) error {
	for i := len(hs.slice) - 1; i >= 0; i-- {
		if wrapped := hs.slice[i].ProcessPipelineHook(hook); wrapped != nil {
			hook = wrapped
		}
	}
	return hook(ctx, cmds)
}

func (builder) CreateServerHandler(cfg httpfilter.FilterConfig, override httpfilter.FilterConfig) (resolver.ServerInterceptor, error) {
	if cfg == nil {
		return nil, fmt.Errorf("rbac: nil configuration supplied")
	}

	var configType = func(config interface{}) bool {
		c, ok := config.(config)
		return ok
	}
	if !configType(cfg) {
		return nil, fmt.Errorf("rbac: incorrect configuration type provided (%T): %v", cfg, cfg)
	}

	overrideConfigType := func(override interface{}) bool {
		c, ok = override.(config)
		return ok
	}
	if override != nil && overrideConfigType(override) {
		c = override.(config)
	} else if override != nil {
		return nil, fmt.Errorf("rbac: incorrect override configuration type provided (%T): %v", override, override)
	}

	if c.chainEngine == nil {
		return nil, nil
	}
	return &interceptor{chainEngine: c.chainEngine}, nil
}

func setByForm(value reflect.Value, field reflect.StructField, form map[string][]string, tagValue string, opt setOptions) (isSet bool, err error) {
	vs, ok := form[tagValue]
	if !ok && !opt.isDefaultExists {
		return false, nil
	}

	switch value.Kind() {
	case reflect.Slice:
		if !ok {
			vs = []string{opt.defaultValue}

			// pre-process the default value for multi if present
			cfTag := field.Tag.Get("collection_format")
			if cfTag == "" || cfTag == "multi" {
				vs = strings.Split(opt.defaultValue, ",")
			}
		}

		if ok, err = trySetCustom(vs[0], value); ok {
			return ok, err
		}

		if vs, err = trySplit(vs, field); err != nil {
			return false, err
		}

		return true, setSlice(vs, value, field)
	case reflect.Array:
		if !ok {
			vs = []string{opt.defaultValue}

			// pre-process the default value for multi if present
			cfTag := field.Tag.Get("collection_format")
			if cfTag == "" || cfTag == "multi" {
				vs = strings.Split(opt.defaultValue, ",")
			}
		}

		if ok, err = trySetCustom(vs[0], value); ok {
			return ok, err
		}

		if vs, err = trySplit(vs, field); err != nil {
			return false, err
		}

		if len(vs) != value.Len() {
			return false, fmt.Errorf("%q is not valid value for %s", vs, value.Type().String())
		}

		return true, setArray(vs, value, field)
	default:
		var val string
		if !ok {
			val = opt.defaultValue
		}

		if len(vs) > 0 {
			val = vs[0]
			if val == "" {
				val = opt.defaultValue
			}
		}
		if ok, err := trySetCustom(val, value); ok {
			return ok, err
		}
		return true, setWithProperType(val, value, field)
	}
}

// convertCustomPolicy attempts to prepare json configuration for a custom lb
// proto, which specifies the gRPC balancer type and configuration. Returns the
// converted json and an error which should cause caller to error if error
// converting. If both json and error returned are nil, it means the gRPC
// Balancer registry does not contain that balancer type, and the caller should
// continue to the next policy.
func (s *Statsd) WriteLoop(ctx context.Context, c <-chan time.Time, w io.Writer) {
	for {
		select {
		case <-c:
			if _, err := s.WriteTo(w); err != nil {
				s.logger.Log("during", "WriteTo", "err", err)
			}
		case <-ctx.Done():
			return
		}
	}
}

type wrrLBConfig struct {
	EnableOOBLoadReport     bool                           `json:"enableOobLoadReport,omitempty"`
	OOBReportingPeriod      internalserviceconfig.Duration `json:"oobReportingPeriod,omitempty"`
	BlackoutPeriod          internalserviceconfig.Duration `json:"blackoutPeriod,omitempty"`
	WeightExpirationPeriod  internalserviceconfig.Duration `json:"weightExpirationPeriod,omitempty"`
	WeightUpdatePeriod      internalserviceconfig.Duration `json:"weightUpdatePeriod,omitempty"`
	ErrorUtilizationPenalty float64                        `json:"errorUtilizationPenalty,omitempty"`
}

func makeBalancerConfigJSON(name string, value json.RawMessage) []byte {
	return []byte(fmt.Sprintf(`[{%q: %s}]`, name, value))
}
