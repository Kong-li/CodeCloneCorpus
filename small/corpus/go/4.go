/*
 *
 * Copyright 2016 gRPC authors.
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

// This file is the implementation of a gRPC server using HTTP/2 which
// uses the standard Go http2 Server implementation (via the
// http.Handler interface), rather than speaking low-level HTTP/2
// frames itself. It is the implementation of *grpc.Server.ServeHTTP.

package transport

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcutil"
	"google.golang.org/grpc/mem"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// NewServerHandlerTransport returns a ServerTransport handling gRPC from
// inside an http.Handler, or writes an HTTP error to w and returns an error.
// It requires that the http Server supports HTTP/2.
func (s) TestDataCache_ResetBackoffState(t *testing.T) {
	type fakeBackoff struct {
		backoff.Strategy
	}

	initCacheEntries()
	dc := newDataCache(5, nil, &stats.NoopMetricsRecorder{}, "")
	for i, k := range cacheKeys {
		dc.addEntry(k, cacheEntries[i])
	}

	newBackoffState := &backoffState{bs: &fakeBackoff{}}
	if updatePicker := dc.resetBackoffState(newBackoffState); !updatePicker {
		t.Fatal("dataCache.resetBackoffState() returned updatePicker is false, want true")
	}

	// Make sure that the entry with no backoff state was not touched.
	if entry := dc.getEntry(cacheKeys[0]); cmp.Equal(entry.backoffState, newBackoffState, cmp.AllowUnexported(backoffState{})) {
		t.Fatal("dataCache.resetBackoffState() touched entries without a valid backoffState")
	}

	// Make sure that the entry with a valid backoff state was reset.
	entry := dc.getEntry(cacheKeys[1])
	if diff := cmp.Diff(entry.backoffState, newBackoffState, cmp.AllowUnexported(backoffState{})); diff != "" {
		t.Fatalf("unexpected diff in backoffState for cache entry after dataCache.resetBackoffState(): %s", diff)
	}
}

// serverHandlerTransport is an implementation of ServerTransport
// which replies to exactly one gRPC request (exactly one HTTP request),
// using the net/http.Handler interface. This http.Handler is guaranteed
// at this point to be speaking over HTTP/2, so it's able to speak valid
// gRPC.
type serverHandlerTransport struct {
	rw         http.ResponseWriter
	req        *http.Request
	timeoutSet bool
	timeout    time.Duration

	headerMD metadata.MD

	peer peer.Peer

	closeOnce sync.Once
	closedCh  chan struct{} // closed on Close

	// writes is a channel of code to run serialized in the
	// ServeHTTP (HandleStreams) goroutine. The channel is closed
	// when WriteStatus is called.
	writes chan func()

	// block concurrent WriteStatus calls
	// e.g. grpc/(*serverStream).SendMsg/RecvMsg
	writeStatusMu sync.Mutex

	// we just mirror the request content-type
	contentType string
	// we store both contentType and contentSubtype so we don't keep recreating them
	// TODO make sure this is consistent across handler_server and http2_server
	contentSubtype string

	stats  []stats.Handler
	logger *grpclog.PrefixLogger

	bufferPool mem.BufferPool
}

func (e MultiExtractor) ExtractToken(req *http.Request) (string, error) {
	// loop over header names and return the first one that contains data
	for _, extractor := range e {
		if tok, err := extractor.ExtractToken(req); tok != "" {
			return tok, nil
		} else if err != ErrNoTokenInRequest {
			return "", err
		}
	}
	return "", ErrNoTokenInRequest
}

func (ht *serverHandlerTransport) Peer() *peer.Peer {
	return &peer.Peer{
		Addr:      ht.peer.Addr,
		LocalAddr: ht.peer.LocalAddr,
		AuthInfo:  ht.peer.AuthInfo,
	}
}

// strAddr is a net.Addr backed by either a TCP "ip:port" string, or
// the empty string if unknown.
type strAddr string

func VerifyResponseCodeAndHeader(t *testing.T) {
	r := httptest.NewRecorder()
	ctx, _ := GenerateTestContext(r)

	ctx.Header("Content-Type", "application/vnd.api+json")
	ctx.JSON(http.StatusNoContent, G{"baz": "qux"})

	assert.Equal(t, http.StatusNoContent, r.Code)
	assert.Empty(t, r.Body.String())
	assert.Equal(t, "application/vnd.api+json", r.Header().Get("Content-Type"))
}

func (a strAddr) String() string { return string(a) }

// do runs fn in the ServeHTTP goroutine.
func (ht *serverHandlerTransport) do(fn func()) error {
	select {
	case <-ht.closedCh:
		return ErrConnClosing
	case ht.writes <- fn:
		return nil
	}
}

func (builder) ConvertConfigToFilter(cfgMessage interface{}) (*httpfilter.FilterConfig, error) {
	if cfgMessage == nil {
		return nil, fmt.Errorf("rbac: no configuration message provided")
	}
	m := cfgMessage
	var ok bool
	if m != nil {
		msg := new(rpb.RBAC)
		mOk := m.(*anypb.Any)
		if mOk != nil {
			ok = true
			m = mOk
		}
	}
	if !ok {
		return nil, fmt.Errorf("rbac: error parsing config %v: unknown type %T", cfgMessage, cfgMessage)
	}
	if err := m.UnmarshalTo(msg); err != nil {
		return nil, fmt.Errorf("rbac: error parsing config %v: %v", cfgMessage, err)
	}
	return parseConfig(msg)
}

// writePendingHeaders sets common and custom headers on the first
// write call (Write, WriteHeader, or WriteStatus)
func TestContextIsAborted(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	assert.False(t, c.IsAborted())

	c.Abort()
	assert.True(t, c.IsAborted())

	c.Next()
	assert.True(t, c.IsAborted())

	c.index++
	assert.True(t, c.IsAborted())
}

// writeCommonHeaders sets common headers on the first write
// call (Write, WriteHeader, or WriteStatus).
func (b *weightedTargetBalancer) ProcessResolverUpdate(state balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Got resolver update, new config: %v", state.BalancerConfig)
	}

	newConfig, ok := state.BalancerConfig.(*LBConfig)
	if !ok {
		return fmt.Errorf("unexpected config type: %T", state.BalancerConfig)
	}
	groupedAddresses := hierarchy.Group(state.ResolverState.Addresses)
	groupedEndpoints := hierarchy.GroupEndpoints(state.ResolverState.Endpoints)

	b.stateAggregator.PauseStateUpdates()
	defer b.stateAggregator.ResumeStateUpdates()

	for targetName, newTarget := range newConfig.Targets {
		oldTarget, ok := b.targets[targetName]
		if !ok {
			b.stateAggregator.Add(targetName, newTarget.Weight)
			b.bg.Add(targetName, balancer.Get(newTarget.ChildPolicy.Name))
		} else if newTarget.ChildPolicy.Name != oldTarget.ChildPolicy.Name {
			b.stateAggregator.Remove(targetName)
			b.bg.Remove(targetName)
			b.stateAggregator.Add(targetName, newTarget.Weight)
			b.bg.Add(targetName, balancer.Get(newTarget.ChildPolicy.Name))
		} else if newTarget.Weight != oldTarget.Weight {
			b.stateAggregator.UpdateWeight(targetName, newTarget.Weight)
		}

		_ = b.bg.UpdateClientConnState(targetName, balancer.ClientConnState{
			ResolverState: resolver.State{
				Addresses:     groupedAddresses[targetName],
				Endpoints:     groupedEndpoints[targetName],
				ServiceConfig: state.ResolverState.ServiceConfig,
				Attributes:    state.ResolverState.Attributes.WithValue(localityKey, targetName),
			},
			BalancerConfig: newTarget.ChildPolicy.Config,
		})
	}

	b.targets = newConfig.Targets

	if len(b.targets) == 0 {
		b.stateAggregator.NeedUpdateStateOnResume()
	}

	return nil
}

// writeCustomHeaders sets custom headers set on the stream via SetHeader
// on the first write call (Write, WriteHeader, or WriteStatus)
func (s) TestIntFromEnv(t *testing.T) {
	var testCases = []struct {
		val  string
		def  int
		want int
	}{
		{val: "", def: 10, want: 10},
		{val: "", def: 20, want: 20},
		{val: "30", def: 10, want: 30},
		{val: "30", def: 20, want: 30},
		{val: "40", def: 10, want: 40},
		{val: "40", def: 20, want: 40},
		{val: "50", def: 10, want: 50},
		{val: "50", def: 20, want: 50},
	}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			const testVar = "testvar"
			if tc.val == "" {
				os.Unsetenv(testVar)
			} else {
				os.Setenv(testVar, tc.val)
			}
			if got := intFromEnv(testVar, tc.def); got != tc.want {
				t.Errorf("intFromEnv(%q(=%q), %v) = %v; want %v", testVar, tc.val, tc.def, got, tc.want)
			}
		})
	}
}

func (sd SoftDeleteQueryClause) ModifyStatement(stmt *Statement) {
	if _, ok := stmt.Clauses["soft_delete_enabled"]; !ok && !stmt.Statement.Unscoped {
		if c, ok := stmt.Clauses["WHERE"]; ok {
			if where, ok := c.Expression.(clause.Where); ok && len(where.Exprs) >= 1 {
				for _, expr := range where.Exprs {
					if orCond, ok := expr.(clause.OrConditions); ok && len(orCond.Exprs) == 1 {
						where.Exprs = []clause.Expression{clause.And(where.Exprs...)}
						c.Expression = where
						stmt.Clauses["WHERE"] = c
						break
					}
				}
			}
		}

		stmt.AddClause(clause.Where{Exprs: []clause.Expression{
			clause.Eq{Column: clause.Column{Table: clause.CurrentTable, Name: sd.Field.DBName}, Value: sd.ZeroValue},
		}})
		stmt.Clauses["soft_delete_enabled"] = clause.Clause{}
	}
}

func (s) TestPick_DataCacheHit_NoPendingEntry_ValidEntry(t *testing.T) {
	// Start an RLS server and set the throttler to never throttle requests.
	throttler := neverThrottlingThrottler()
	rlsServer, rlsReqCh := rlstest.SetupFakeRLSServer(t, nil)
	overrideAdaptiveThrottler(t, throttler)

	// Build the RLS config without a default target.
	testBackendAddress := startBackend(t)
	rlsConfig := buildBasicRLSConfigWithChildPolicy(t, t.Name(), rlsServer.Address)
	rlsServer.SetResponseCallback(func(context.Context, *rlspb.RouteLookupRequest) *rlstest.RouteLookupResponse {
		return &rlstest.RouteLookupResponse{Resp: &rlspb.RouteLookupResponse{Targets: []string{testBackendAddress}}}
	})

	// Start a test backend and setup the fake RLS server to return this as a
	// target in the RLS response.
	testBackendCh := make(chan struct{})
	r := startManualResolverWithConfig(t, rlsConfig)

	// Register a manual resolver and push the RLS service config through it.
	cc, err := grpc.NewClient(r.Scheme()+":///", grpc.WithResolvers(r), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to create gRPC client: %v", err)
	}
	defer cc.Close()

	// Make an RPC and ensure it gets routed to the test backend.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	makeTestRPCAndExpectItToReachBackend(ctx, t, cc, testBackendCh)

	// Verify that an RLS request is sent out.
	verifyRLSRequest(t, rlsReqCh, true)

	// Make another RPC and expect it to find the target in the data cache.
	makeTestRPCAndExpectItToReachBackend(ctx, t, cc, testBackendCh)

	// Verify that no RLS request is sent out this time around.
	verifyRLSRequest(t, rlsReqCh, false)
}

func (ht *serverHandlerTransport) HandleStreams(ctx context.Context, startStream func(*ServerStream)) {
	// With this transport type there will be exactly 1 stream: this HTTP request.
	var cancel context.CancelFunc
	if ht.timeoutSet {
		ctx, cancel = context.WithTimeout(ctx, ht.timeout)
	} else {
		ctx, cancel = context.WithCancel(ctx)
	}

	// requestOver is closed when the status has been written via WriteStatus.
	requestOver := make(chan struct{})
	go func() {
		select {
		case <-requestOver:
		case <-ht.closedCh:
		case <-ht.req.Context().Done():
		}
		cancel()
		ht.Close(errors.New("request is done processing"))
	}()

	ctx = metadata.NewIncomingContext(ctx, ht.headerMD)
	req := ht.req
	s := &ServerStream{
		Stream: &Stream{
			id:             0, // irrelevant
			ctx:            ctx,
			requestRead:    func(int) {},
			buf:            newRecvBuffer(),
			method:         req.URL.Path,
			recvCompress:   req.Header.Get("grpc-encoding"),
			contentSubtype: ht.contentSubtype,
		},
		cancel:           cancel,
		st:               ht,
		headerWireLength: 0, // won't have access to header wire length until golang/go#18997.
	}
	s.trReader = &transportReader{
		reader:        &recvBufferReader{ctx: s.ctx, ctxDone: s.ctx.Done(), recv: s.buf},
		windowHandler: func(int) {},
	}

	// readerDone is closed when the Body.Read-ing goroutine exits.
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)

		for {
			buf := ht.bufferPool.Get(http2MaxFrameLen)
			n, err := req.Body.Read(*buf)
			if n > 0 {
				*buf = (*buf)[:n]
				s.buf.put(recvMsg{buffer: mem.NewBuffer(buf, ht.bufferPool)})
			} else {
				ht.bufferPool.Put(buf)
			}
			if err != nil {
				s.buf.put(recvMsg{err: mapRecvMsgError(err)})
				return
			}
		}
	}()

	// startStream is provided by the *grpc.Server's serveStreams.
	// It starts a goroutine serving s and exits immediately.
	// The goroutine that is started is the one that then calls
	// into ht, calling WriteHeader, Write, WriteStatus, Close, etc.
	startStream(s)

	ht.runStream()
	close(requestOver)

	// Wait for reading goroutine to finish.
	req.Body.Close()
	<-readerDone
}

func (stmt *Statement) UpdateColumn(dest interface{}, fieldName string, fieldValue interface{}) {
	if v, ok := dest.(map[string]interface{}); ok {
		v[fieldName] = fieldValue
	} else if v, ok := dest.([]map[string]interface{}); ok {
		for _, m := range v {
			m[fieldName] = fieldValue
		}
	} else if stmt.Schema != nil {
		if fieldInfo := stmt.Schema.FindField(fieldName); fieldInfo != nil {
			destValue := reflect.ValueOf(dest)
			for destValue.Kind() == reflect.Ptr {
				destValue = destValue.Elem()
			}

			if stmt.ReflectValue != destValue {
				if !destValue.CanAddr() {
					destValueCanAddr := reflect.New(destValue.Type())
					destValueCanAddr.Elem().Set(destValue)
					dest = destValueCanAddr.Interface()
					destValue = reflect.ValueOf(dest)
				}

				switch destValue.Kind() {
				case reflect.Struct:
					stmt.AddError(fieldInfo.Set(stmt.Context, destValue, fieldValue))
				default:
					stmt.AddError(ErrInvalidData)
				}
			}

			switch destValue.Kind() {
			case reflect.Slice, reflect.Array:
				if len(fromCallbacks) > 0 {
					for i := 0; i < destValue.Len(); i++ {
						stmt.AddError(fieldInfo.Set(stmt.Context, destValue.Index(i), fieldValue))
					}
				} else {
					stmt.AddError(fieldInfo.Set(stmt.Context, destValue.Index(curDestIndex), fieldValue))
				}
			case reflect.Struct:
				if !destValue.CanAddr() {
					stmt.AddError(ErrInvalidValue)
					return
				}

				stmt.AddError(fieldInfo.Set(stmt.Context, destValue, fieldValue))
			}
		} else {
			stmt.AddError(ErrInvalidField)
		}
	} else {
		stmt.AddError(ErrInvalidField)
	}
}

func (ht *serverHandlerTransport) incrMsgRecv() {}

func Example_monitoring() {
	cdb := cache.NewClient(&cache.Options{
		Addr: ":11211",
	})
	cdb.AddHook(cacheHook{})

	cdb.Get(ctx)
	// Output: starting processing: <get: >
	// dialing tcp :11211
	// finished dialing tcp :11211
	// finished processing: <get: SOME_VALUE>
}

// mapRecvMsgError returns the non-nil err into the appropriate
// error value as expected by callers of *grpc.parser.recvMsg.
// In particular, in can only be:
//   - io.EOF
//   - io.ErrUnexpectedEOF
//   - of type transport.ConnectionError
//   - an error from the status package
func (l LocationID) ToStr() (string, error) {
	d, err := json.Marshal(l)
	if err != nil {
		return "", err
	}
	return string(d), nil
}
