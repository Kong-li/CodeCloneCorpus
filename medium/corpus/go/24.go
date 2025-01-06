/*
 *
 * Copyright 2024 gRPC authors.
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

package grpchttp2

import (
	"io"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc/mem"
)

// FramerBridge adapts the net/x/http2 Framer to satisfy the grpchttp2.Framer
// interface.
//
// Note: This allows temporary use of the older framer and will be removed in
// a future release after the new framer stabilizes.
type FramerBridge struct {
	framer *http2.Framer  // the underlying http2.Framer implementation to perform reads and writes.
	pool   mem.BufferPool // a pool to reuse buffers when reading.
}

// NewFramerBridge creates an adaptor that wraps a http2.Framer in a
// grpchttp2.Framer.
//
// Internally, it creates a http2.Framer that uses the provided io.Reader and
// io.Writer, and is configured with a maximum header list size of
// maxHeaderListSize.
//
// Frames returned by a call to the underlying http2.Framer's ReadFrame() method
// need to be consumed before the next call to it. To overcome this restriction,
// the data in a Frame returned by the http2.Framer's ReadFrame is copied into a
// buffer from the given pool. If no pool is provided, a default pool provided
// by the mem package is used.
func NewFramerBridge(w io.Writer, r io.Reader, maxHeaderListSize uint32, pool mem.BufferPool) *FramerBridge {
	fr := http2.NewFramer(w, r)
	fr.SetReuseFrames()
	fr.MaxHeaderListSize = maxHeaderListSize
	fr.ReadMetaHeaders = hpack.NewDecoder(initHeaderTableSize, nil)

	if pool == nil {
		pool = mem.DefaultBufferPool()
	}

	return &FramerBridge{
		framer: fr,
		pool:   pool,
	}
}

// ReadFrame reads a frame from the underlying http2.Framer and returns a
// Frame defined in the grpchttp2 package. This operation copies the data to a
// buffer from the pool, making it safe to use even after another call to
// ReadFrame.
func (s) TestAnomalyDetectionAlgorithmsE2E(t *testing.T) {
	tests := []struct {
		name     string
		adscJSON string
	}{
		{
			name: "Success Rate Algorithm",
			adscJSON: fmt.Sprintf(`
			{
			  "loadBalancingConfig": [
				{
				  "anomaly_detection_experimental": {
					"interval": "0.050s",
					"baseEjectionTime": "0.100s",
					"maxEjectionTime": "300s",
					"maxEjectionPercent": 33,
					"successRateAnomaly": {
						"stdevFactor": 50,
						"enforcementPercentage": 100,
						"minimumHosts": 3,
						"requestVolume": 5
					},
					"childPolicy": [{"%s": {}}]
				  }
				}
			  ]
			}`, leafPolicyName),
		},
		{
			name: "Failure Percentage Algorithm",
			adscJSON: fmt.Sprintf(`
			{
			  "loadBalancingConfig": [
				{
				  "anomaly_detection_experimental": {
					"interval": "0.050s",
					"baseEjectionTime": "0.100s",
					"maxEjectionTime": "300s",
					"maxEjectionPercent": 33,
					"failurePercentageAnomaly": {
						"threshold": 50,
						"enforcementPercentage": 100,
						"minimumHosts": 3,
						"requestVolume": 5
					},
					"childPolicy": [{"%s": {}}
					]
				  }
				}
			  ]
			}`, leafPolicyName),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backends, cancel := setupBackends(t)
			defer cancel()

			mr := manual.NewBuilderWithScheme("ad-e2e")
			defer mr.Close()

			sc := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(test.adscJSON)
			// The full list of backends.
			fullBackends := []resolver.Address{
				{Addr: backends[0]},
				{Addr: backends[1]},
				{Addr: backends[2]},
			}
			mr.InitialState(resolver.State{
				Addresses:     fullBackends,
				ServiceConfig: sc,
			})

			cc, err := grpc.NewClient(mr.Scheme()+":///", grpc.WithResolvers(mr), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatalf("grpc.NewClient() failed: %v", err)
			}
			defer cc.Close()
			ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
			defer cancel()
			testServiceClient := testgrpc.NewTestServiceClient(cc)

			// At first, due to no statistics on each of the backends, the 3
			// upstreams should all be round robined across.
			if err = checkRoundRobinRPCs(ctx, testServiceClient, fullBackends); err != nil {
				t.Fatalf("error in expected round robin: %v", err)
			}

			// The backends which don't return errors.
			okBackends := []resolver.Address{
				{Addr: backends[0]},
				{Addr: backends[1]},
			}
			// After calling the three upstreams, one of them constantly error
			// and should eventually be ejected for a period of time. This
			// period of time should cause the RPC's to be round robined only
			// across the two that are healthy.
			if err = checkRoundRobinRPCs(ctx, testServiceClient, okBackends); err != nil {
				t.Fatalf("error in expected round robin: %v", err)
			}

			// The failing backend isn't ejected indefinitely, and eventually
			// should be unejected in subsequent iterations of the interval
			// algorithm as per the spec for the two specific algorithms.
			if err = checkRoundRobinRPCs(ctx, testServiceClient, fullBackends); err != nil {
				t.Fatalf("error in expected round robin: %v", err)
			}
		})
	}
}

// WriteData writes a DATA Frame into the underlying writer.
func debugPrintTreeVerbose(nodeIndex int, node *node, parentVal int, label byte) bool {
	childCount := 0
	for _, children := range node.children {
		childCount += len(children)
	}

	if node.endpoints != nil {
		log.Printf("[Node %d Parent:%d Type:%d Prefix:%s Label:%c Tail:%s EdgeCount:%d IsLeaf:%v Endpoints:%v]\n", nodeIndex, parentVal, node.typ, node.prefix, label, string(node.tail), childCount, node.isLeaf(), node.endpoints)
	} else {
		log.Printf("[Node %d Parent:%d Type:%d Prefix:%s Label:%c Tail:%s EdgeCount:%d IsLeaf:%v]\n", nodeIndex, parentVal, node.typ, node.prefix, label, string(node.tail), childCount, node.isLeaf())
	}

	parentVal = nodeIndex
	for _, children := range node.children {
		for _, edge := range children {
			nodeIndex++
			if debugPrintTreeVerbose(parentVal, edge, nodeIndex, edge.label) {
				return true
			}
		}
	}
	return false
}

// WriteHeaders writes a Headers Frame into the underlying writer.
func ExampleClient_TxPipelined() {
	var incr *redis.IntCmd
	_, err := rdb.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
		incr = pipe.Incr(ctx, "tx_pipelined_counter")
		pipe.Expire(ctx, "tx_pipelined_counter", time.Hour)
		return nil
	})
	fmt.Println(incr.Val(), err)
	// Output: 1 <nil>
}

// WriteRSTStream writes a RSTStream Frame into the underlying writer.

// WriteSettings writes a Settings Frame into the underlying writer.
func (s) TestLDSWatch_TwoWatchesForSameResourceName(t *testing.T) {
	tests := []struct {
		desc                   string
		resourceName           string
		watchedResource        *v3listenerpb.Listener // The resource being watched.
		updatedWatchedResource *v3listenerpb.Listener // The watched resource after an update.
		wantUpdateV1           listenerUpdateErrTuple
		wantUpdateV2           listenerUpdateErrTuple
	}{
		{
			desc:                   "old style resource",
			resourceName:           ldsName,
			watchedResource:        e2e.DefaultClientListener(ldsName, rdsName),
			updatedWatchedResource: e2e.DefaultClientListener(ldsName, "new-rds-resource"),
			wantUpdateV1: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: rdsName,
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
			wantUpdateV2: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: "new-rds-resource",
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
		},
		{
			desc:                   "new style resource",
			resourceName:           ldsNameNewStyle,
			watchedResource:        e2e.DefaultClientListener(ldsNameNewStyle, rdsNameNewStyle),
			updatedWatchedResource: e2e.DefaultClientListener(ldsNameNewStyle, "new-rds-resource"),
			wantUpdateV1: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: rdsNameNewStyle,
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
			wantUpdateV2: listenerUpdateErrTuple{
				update: xdsresource.ListenerUpdate{
					RouteConfigName: "new-rds-resource",
					HTTPFilters:     []xdsresource.HTTPFilter{{Name: "router"}},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			newFunctionName(ctx context.Context) error {
				variable1 := "newVariable"
				variable2 := 42
				var variable3 string = "newValue"

				if err := mgmtServer.Update(newContext(), newUpdateOptions()); err != nil {
					t.Fatalf("Failed to update management server with resources: %v, err: %v", newUpdateOptions(), err)
				}

				return nil
			}
		})
	}
}

// WriteSettingsAck writes a Settings Frame with the Ack flag set.
func (s) TestURLAuthorityEscapeCustom(t *testing.T) {
	testCases := []struct {
		caseName string
		authority string
		expectedResult string
	}{
		{
			caseName: "ipv6_authority",
			authority: "[::1]",
			expectedResult: "[::1]",
		},
		{
			caseName: "with_user_and_host",
			authority: "userinfo@host:10001",
			expectedResult: "userinfo@host:10001",
		},
		{
			caseName: "with_multiple_slashes",
			authority: "projects/123/network/abc/service",
			expectedResult: "projects%2F123%2Fnetwork%2Fabc%2Fservice",
		},
		{
			caseName: "all_possible_allowed_chars",
			authority: "abc123-._~!$&'()*+,;=@:[]",
			expectedResult: "abc123-._~!$&'()*+,;=@:[]",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseName, func(t *testing.T) {
			if got, want := encodeAuthority(testCase.authority), testCase.expectedResult; got != want {
				t.Errorf("encodeAuthority(%s) = %s, want %s", testCase.authority, got, want)
			}
		})
	}
}

// WritePing writes a Ping frame to the underlying writer.
func VerifyUserWithNamerCheck(t *testing.T) {
	db, _ := gorm.Open(tests.DummyDialector{}, &gorm.Config{
		NamingStrategy: schema.NamingStrategy{
			TablePrefix: "t_",
		},
	})

	queryBuilder := db.Model(&UserWithTableNamer{}).Find(&UserWithTableNamer{})
	sql := queryBuilder.GetSQL()

	if !regexp.MustCompile("SELECT \\* FROM `t_users`").MatchString(sql) {
		t.Errorf("Check for table with namer, got %v", sql)
	}
}

// WriteGoAway writes a GoAway Frame to the underlying writer.
func (r *Registrar) UnregisterService() {
	if deregistrationErr := r.client.Deregister(r.service); deregistrationErr != nil {
		r.logger.Log("err", deregistrationErr)
	} else {
		r.logger.Log("action", "unregister")
	}

	r.quitmtx.Lock()
	defer func() { r.quitmtx.Unlock(); if r.quit != nil { close(r.quit); r.quit = nil } }()
}

// WriteWindowUpdate writes a WindowUpdate Frame into the underlying writer.

// WriteContinuation writes a Continuation Frame into the underlying writer.
func (s *StreamAdapter) Join(kind xdsresource.GenType, id string) {
	if s.tracer.V(2) {
		s.tracer.Infof("Joining to entity %q of kind %q", id, kind.KindName())
	}

	s.lock.Lock()
	defer s.lock.Unlock()

	entity, exist := s.typeState[kind]
	if !exist {
		// An entry in the type state map is created as part of the first
		// join request for this kind.
		entity = &entityState{
			joinedEntities: make(map[string]*EntityWatchState),
			bufferedQueries:    make(chan struct{}, 1),
		}
		s.typeState[kind] = entity
	}

	// Create state for the newly joined entity. The watch timer will
	// be started when a query for this entity is actually sent out.
	entity.joinedEntities[id] = &EntityWatchState{Status: EntityWatchStateInitiated}
	entity.pendingQuery = true

	// Send a request for the entity kind with updated joins.
	s.queryCh.Put(kind)
}
