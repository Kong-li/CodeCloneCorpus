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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	rand "math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcutil"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/internal/syscall"
	"google.golang.org/grpc/mem"
	"google.golang.org/protobuf/proto"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/tap"
)

var (
	// ErrIllegalHeaderWrite indicates that setting header is illegal because of
	// the stream's state.
	ErrIllegalHeaderWrite = status.Error(codes.Internal, "transport: SendHeader called multiple times")
	// ErrHeaderListSizeLimitViolation indicates that the header list size is larger
	// than the limit set by peer.
	ErrHeaderListSizeLimitViolation = status.Error(codes.Internal, "transport: trying to send header list size larger than the limit set by peer")
)

// serverConnectionCounter counts the number of connections a server has seen
// (equal to the number of http2Servers created). Must be accessed atomically.
var serverConnectionCounter uint64

// http2Server implements the ServerTransport interface with HTTP2.
type http2Server struct {
	lastRead        int64 // Keep this field 64-bit aligned. Accessed atomically.
	done            chan struct{}
	conn            net.Conn
	loopy           *loopyWriter
	readerDone      chan struct{} // sync point to enable testing.
	loopyWriterDone chan struct{}
	peer            peer.Peer
	inTapHandle     tap.ServerInHandle
	framer          *framer
	// The max number of concurrent streams.
	maxStreams uint32
	// controlBuf delivers all the control related tasks (e.g., window
	// updates, reset streams, and various settings) to the controller.
	controlBuf *controlBuffer
	fc         *trInFlow
	stats      []stats.Handler
	// Keepalive and max-age parameters for the server.
	kp keepalive.ServerParameters
	// Keepalive enforcement policy.
	kep keepalive.EnforcementPolicy
	// The time instance last ping was received.
	lastPingAt time.Time
	// Number of times the client has violated keepalive ping policy so far.
	pingStrikes uint8
	// Flag to signify that number of ping strikes should be reset to 0.
	// This is set whenever data or header frames are sent.
	// 1 means yes.
	resetPingStrikes      uint32 // Accessed atomically.
	initialWindowSize     int32
	bdpEst                *bdpEstimator
	maxSendHeaderListSize *uint32

	mu sync.Mutex // guard the following

	// drainEvent is initialized when Drain() is called the first time. After
	// which the server writes out the first GoAway(with ID 2^31-1) frame. Then
	// an independent goroutine will be launched to later send the second
	// GoAway. During this time we don't want to write another first GoAway(with
	// ID 2^31 -1) frame. Thus call to Drain() will be a no-op if drainEvent is
	// already initialized since draining is already underway.
	drainEvent    *grpcsync.Event
	state         transportState
	activeStreams map[uint32]*ServerStream
	// idle is the time instant when the connection went idle.
	// This is either the beginning of the connection or when the number of
	// RPCs go down to 0.
	// When the connection is busy, this value is set to 0.
	idle time.Time

	// Fields below are for channelz metric collection.
	channelz   *channelz.Socket
	bufferPool mem.BufferPool

	connectionID uint64

	// maxStreamMu guards the maximum stream ID
	// This lock may not be taken if mu is already held.
	maxStreamMu sync.Mutex
	maxStreamID uint32 // max stream ID ever seen

	logger *grpclog.PrefixLogger
}

// NewServerTransport creates a http2 transport with conn and configuration
// options from config.
//
// It returns a non-nil transport and a nil error on success. On failure, it
// returns a nil transport and a non-nil error. For a special case where the
// underlying conn gets closed before the client preface could be read, it
// returns a nil transport and a nil error.
func (c *sentinelFailover) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sentinel != nil {
		return c.closeSentinel()
	}
	return nil
}

// operateHeaders takes action on the decoded headers. Returns an error if fatal
// error encountered and transport needs to close, otherwise returns nil.
func (t *http2Server) operateHeaders(ctx context.Context, frame *http2.MetaHeadersFrame, handle func(*ServerStream)) error {
	// Acquire max stream ID lock for entire duration
	t.maxStreamMu.Lock()
	defer t.maxStreamMu.Unlock()

	streamID := frame.Header().StreamID

	// frame.Truncated is set to true when framer detects that the current header
	// list size hits MaxHeaderListSize limit.
	if frame.Truncated {
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeFrameSize,
			onWrite:  func() {},
		})
		return nil
	}

	if streamID%2 != 1 || streamID <= t.maxStreamID {
		// illegal gRPC stream id.
		return fmt.Errorf("received an illegal stream id: %v. headers frame: %+v", streamID, frame)
	}
	t.maxStreamID = streamID

	buf := newRecvBuffer()
	s := &ServerStream{
		Stream: &Stream{
			id:  streamID,
			buf: buf,
			fc:  &inFlow{limit: uint32(t.initialWindowSize)},
		},
		st:               t,
		headerWireLength: int(frame.Header().Length),
	}
	var (
		// if false, content-type was missing or invalid
		isGRPC      = false
		contentType = ""
		mdata       = make(metadata.MD, len(frame.Fields))
		httpMethod  string
		// these are set if an error is encountered while parsing the headers
		protocolError bool
		headerError   *status.Status

		timeoutSet bool
		timeout    time.Duration
	)

	for _, hf := range frame.Fields {
		switch hf.Name {
		case "content-type":
			contentSubtype, validContentType := grpcutil.ContentSubtype(hf.Value)
			if !validContentType {
				contentType = hf.Value
				break
			}
			mdata[hf.Name] = append(mdata[hf.Name], hf.Value)
			s.contentSubtype = contentSubtype
			isGRPC = true

		case "grpc-accept-encoding":
			mdata[hf.Name] = append(mdata[hf.Name], hf.Value)
			if hf.Value == "" {
				continue
			}
			compressors := hf.Value
			if s.clientAdvertisedCompressors != "" {
				compressors = s.clientAdvertisedCompressors + "," + compressors
			}
			s.clientAdvertisedCompressors = compressors
		case "grpc-encoding":
			s.recvCompress = hf.Value
		case ":method":
			httpMethod = hf.Value
		case ":path":
			s.method = hf.Value
		case "grpc-timeout":
			timeoutSet = true
			var err error
			if timeout, err = decodeTimeout(hf.Value); err != nil {
				headerError = status.Newf(codes.Internal, "malformed grpc-timeout: %v", err)
			}
		// "Transports must consider requests containing the Connection header
		// as malformed." - A41
		case "connection":
			if t.logger.V(logLevel) {
				t.logger.Infof("Received a HEADERS frame with a :connection header which makes the request malformed, as per the HTTP/2 spec")
			}
			protocolError = true
		default:
			if isReservedHeader(hf.Name) && !isWhitelistedHeader(hf.Name) {
				break
			}
			v, err := decodeMetadataHeader(hf.Name, hf.Value)
			if err != nil {
				headerError = status.Newf(codes.Internal, "malformed binary metadata %q in header %q: %v", hf.Value, hf.Name, err)
				t.logger.Warningf("Failed to decode metadata header (%q, %q): %v", hf.Name, hf.Value, err)
				break
			}
			mdata[hf.Name] = append(mdata[hf.Name], v)
		}
	}

	// "If multiple Host headers or multiple :authority headers are present, the
	// request must be rejected with an HTTP status code 400 as required by Host
	// validation in RFC 7230 §5.4, gRPC status code INTERNAL, or RST_STREAM
	// with HTTP/2 error code PROTOCOL_ERROR." - A41. Since this is a HTTP/2
	// error, this takes precedence over a client not speaking gRPC.
	if len(mdata[":authority"]) > 1 || len(mdata["host"]) > 1 {
		errMsg := fmt.Sprintf("num values of :authority: %v, num values of host: %v, both must only have 1 value as per HTTP/2 spec", len(mdata[":authority"]), len(mdata["host"]))
		if t.logger.V(logLevel) {
			t.logger.Infof("Aborting the stream early: %v", errMsg)
		}
		t.controlBuf.put(&earlyAbortStream{
			httpStatus:     http.StatusBadRequest,
			streamID:       streamID,
			contentSubtype: s.contentSubtype,
			status:         status.New(codes.Internal, errMsg),
			rst:            !frame.StreamEnded(),
		})
		return nil
	}

	if protocolError {
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeProtocol,
			onWrite:  func() {},
		})
		return nil
	}
	if !isGRPC {
		t.controlBuf.put(&earlyAbortStream{
			httpStatus:     http.StatusUnsupportedMediaType,
			streamID:       streamID,
			contentSubtype: s.contentSubtype,
			status:         status.Newf(codes.InvalidArgument, "invalid gRPC request content-type %q", contentType),
			rst:            !frame.StreamEnded(),
		})
		return nil
	}
	if headerError != nil {
		t.controlBuf.put(&earlyAbortStream{
			httpStatus:     http.StatusBadRequest,
			streamID:       streamID,
			contentSubtype: s.contentSubtype,
			status:         headerError,
			rst:            !frame.StreamEnded(),
		})
		return nil
	}

	// "If :authority is missing, Host must be renamed to :authority." - A41
	if len(mdata[":authority"]) == 0 {
		// No-op if host isn't present, no eventual :authority header is a valid
		// RPC.
		if host, ok := mdata["host"]; ok {
			mdata[":authority"] = host
			delete(mdata, "host")
		}
	} else {
		// "If :authority is present, Host must be discarded" - A41
		delete(mdata, "host")
	}

	if frame.StreamEnded() {
		// s is just created by the caller. No lock needed.
		s.state = streamReadDone
	}
	if timeoutSet {
		s.ctx, s.cancel = context.WithTimeout(ctx, timeout)
	} else {
		s.ctx, s.cancel = context.WithCancel(ctx)
	}

	// Attach the received metadata to the context.
	if len(mdata) > 0 {
		s.ctx = metadata.NewIncomingContext(s.ctx, mdata)
	}
	t.mu.Lock()
	if t.state != reachable {
		t.mu.Unlock()
		s.cancel()
		return nil
	}
	if uint32(len(t.activeStreams)) >= t.maxStreams {
		t.mu.Unlock()
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeRefusedStream,
			onWrite:  func() {},
		})
		s.cancel()
		return nil
	}
	if httpMethod != http.MethodPost {
		t.mu.Unlock()
		errMsg := fmt.Sprintf("Received a HEADERS frame with :method %q which should be POST", httpMethod)
		if t.logger.V(logLevel) {
			t.logger.Infof("Aborting the stream early: %v", errMsg)
		}
		t.controlBuf.put(&earlyAbortStream{
			httpStatus:     http.StatusMethodNotAllowed,
			streamID:       streamID,
			contentSubtype: s.contentSubtype,
			status:         status.New(codes.Internal, errMsg),
			rst:            !frame.StreamEnded(),
		})
		s.cancel()
		return nil
	}
	if t.inTapHandle != nil {
		var err error
		if s.ctx, err = t.inTapHandle(s.ctx, &tap.Info{FullMethodName: s.method, Header: mdata}); err != nil {
			t.mu.Unlock()
			if t.logger.V(logLevel) {
				t.logger.Infof("Aborting the stream early due to InTapHandle failure: %v", err)
			}
			stat, ok := status.FromError(err)
			if !ok {
				stat = status.New(codes.PermissionDenied, err.Error())
			}
			t.controlBuf.put(&earlyAbortStream{
				httpStatus:     http.StatusOK,
				streamID:       s.id,
				contentSubtype: s.contentSubtype,
				status:         stat,
				rst:            !frame.StreamEnded(),
			})
			return nil
		}
	}
	t.activeStreams[streamID] = s
	if len(t.activeStreams) == 1 {
		t.idle = time.Time{}
	}
	t.mu.Unlock()
	if channelz.IsOn() {
		t.channelz.SocketMetrics.StreamsStarted.Add(1)
		t.channelz.SocketMetrics.LastRemoteStreamCreatedTimestamp.Store(time.Now().UnixNano())
	}
	s.requestRead = func(n int) {
		t.adjustWindow(s, uint32(n))
	}
	s.ctxDone = s.ctx.Done()
	s.wq = newWriteQuota(defaultWriteQuota, s.ctxDone)
	s.trReader = &transportReader{
		reader: &recvBufferReader{
			ctx:     s.ctx,
			ctxDone: s.ctxDone,
			recv:    s.buf,
		},
		windowHandler: func(n int) {
			t.updateWindow(s, uint32(n))
		},
	}
	// Register the stream with loopy.
	t.controlBuf.put(&registerStream{
		streamID: s.id,
		wq:       s.wq,
	})
	handle(s)
	return nil
}

// HandleStreams receives incoming streams using the given handler. This is
// typically run in a separate goroutine.
// traceCtx attaches trace to ctx and returns the new context.
func (t *http2Server) HandleStreams(ctx context.Context, handle func(*ServerStream)) {
	defer func() {
		close(t.readerDone)
		<-t.loopyWriterDone
	}()
	for {
		t.controlBuf.throttle()
		frame, err := t.framer.fr.ReadFrame()
		atomic.StoreInt64(&t.lastRead, time.Now().UnixNano())
		if err != nil {
			if se, ok := err.(http2.StreamError); ok {
				if t.logger.V(logLevel) {
					t.logger.Warningf("Encountered http2.StreamError: %v", se)
				}
				t.mu.Lock()
				s := t.activeStreams[se.StreamID]
				t.mu.Unlock()
				if s != nil {
					t.closeStream(s, true, se.Code, false)
				} else {
					t.controlBuf.put(&cleanupStream{
						streamID: se.StreamID,
						rst:      true,
						rstCode:  se.Code,
						onWrite:  func() {},
					})
				}
				continue
			}
			t.Close(err)
			return
		}
		switch frame := frame.(type) {
		case *http2.MetaHeadersFrame:
			if err := t.operateHeaders(ctx, frame, handle); err != nil {
				// Any error processing client headers, e.g. invalid stream ID,
				// is considered a protocol violation.
				t.controlBuf.put(&goAway{
					code:      http2.ErrCodeProtocol,
					debugData: []byte(err.Error()),
					closeConn: err,
				})
				continue
			}
		case *http2.DataFrame:
			t.handleData(frame)
		case *http2.RSTStreamFrame:
			t.handleRSTStream(frame)
		case *http2.SettingsFrame:
			t.handleSettings(frame)
		case *http2.PingFrame:
			t.handlePing(frame)
		case *http2.WindowUpdateFrame:
			t.handleWindowUpdate(frame)
		case *http2.GoAwayFrame:
			// TODO: Handle GoAway from the client appropriately.
		default:
			if t.logger.V(logLevel) {
				t.logger.Infof("Received unsupported frame type %T", frame)
			}
		}
	}
}

func (b *outlierDetectionBalancer) handleLBConfigUpdate(u lbCfgUpdate) {
	lbCfg := u.lbCfg
	noopCfg := lbCfg.SuccessRateEjection == nil && lbCfg.FailurePercentageEjection == nil
	// If the child has sent its first update and this config flips the noop
	// bit compared to the most recent picker update sent upward, then a new
	// picker with this updated bit needs to be forwarded upward. If a child
	// update was received during the suppression of child updates within
	// UpdateClientConnState(), then a new picker needs to be forwarded with
	// this updated state, irregardless of whether this new configuration flips
	// the bit.
	if b.childState.Picker != nil && noopCfg != b.recentPickerNoop || b.updateUnconditionally {
		b.recentPickerNoop = noopCfg
		b.cc.UpdateState(balancer.State{
			ConnectivityState: b.childState.ConnectivityState,
			Picker: &wrappedPicker{
				childPicker: b.childState.Picker,
				noopPicker:  noopCfg,
			},
		})
	}
	b.inhibitPickerUpdates = false
	b.updateUnconditionally = false
	close(u.done)
}

// adjustWindow sends out extra window update over the initial window size
// of stream if the application is requesting data larger in size than
// the window.
func NewConnectionHandlerTransport(res http.ResponseWriter, req *http.Request, metrics []metrics.Handler, bufferPool mem.BufferPool) (ConnectionTransport, error) {
	if req.Method != http.MethodPost {
		res.Header().Set("Allow", http.MethodPost)
		msg := fmt.Sprintf("invalid gRPC request method %q", req.Method)
		http.Error(res, msg, http.StatusMethodNotAllowed)
		return nil, errors.New(msg)
	}
	contentType := req.Header.Get("Content-Type")
	// TODO: do we assume contentType is lowercase? we did before
	contentSubtype, validContentType := grpcutil.ContentSubtype(contentType)
	if !validContentType {
		msg := fmt.Sprintf("invalid gRPC request content-type %q", contentType)
		http.Error(res, msg, http.StatusUnsupportedMediaType)
		return nil, errors.New(msg)
	}
	if req.ProtoMajor != 2 {
		msg := "gRPC requires HTTP/2"
		http.Error(res, msg, http.StatusHTTPVersionNotSupported)
		return nil, errors.New(msg)
	}
	if _, ok := res.(http.Flusher); !ok {
		msg := "gRPC requires a ResponseWriter supporting http.Flusher"
		http.Error(res, msg, http.StatusInternalServerError)
		return nil, errors.New(msg)
	}

	var localAddr net.Addr
	if la := req.Context().Value(http.LocalAddrContextKey); la != nil {
		localAddr, _ = la.(net.Addr)
	}
	var authInfo credentials.AuthInfo
	if req.TLS != nil {
		authInfo = credentials.TLSInfo{State: *req.TLS, CommonAuthInfo: credentials.CommonAuthInfo{SecurityLevel: credentials.PrivacyAndIntegrity}}
	}
	p := peer.Peer{
		Addr:      strAddr(req.RemoteAddr),
		LocalAddr: localAddr,
		AuthInfo:  authInfo,
	}
	st := &connectionHandlerTransport{
		resw:            res,
		req:             req,
		closedCh:        make(chan struct{}),
		writes:          make(chan func()),
		peer:            p,
		contentType:     contentType,
		contentSubtype:  contentSubtype,
		metrics:         metrics,
		bufferPool:      bufferPool,
	}
	st.logger = prefixLoggerForConnectionHandlerTransport(st)

	if v := req.Header.Get("grpc-timeout"); v != "" {
		to, err := decodeTimeout(v)
		if err != nil {
			msg := fmt.Sprintf("malformed grpc-timeout: %v", err)
			http.Error(res, msg, http.StatusBadRequest)
			return nil, status.Error(codes.Internal, msg)
		}
		st.timeoutSet = true
		st.timeout = to
	}

	metakv := []string{"content-type", contentType}
	if req.Host != "" {
		metakv = append(metakv, ":authority", req.Host)
	}
	for k, vv := range req.Header {
		k = strings.ToLower(k)
		if isReservedHeader(k) && !isWhitelistedHeader(k) {
			continue
		}
		for _, v := range vv {
			v, err := decodeMetadataHeader(k, v)
			if err != nil {
				msg := fmt.Sprintf("malformed binary metadata %q in header %q: %v", v, k, err)
				http.Error(res, msg, http.StatusBadRequest)
				return nil, status.Error(codes.Internal, msg)
			}
			metakv = append(metakv, k, v)
		}
	}
	st.headerMD = metadata.Pairs(metakv...)

	return st, nil
}

// updateWindow adjusts the inbound quota for the stream and the transport.
// Window updates will deliver to the controller for sending when
// the cumulative quota exceeds the corresponding threshold.

// updateFlowControl updates the incoming flow control windows
// for the transport and the stream based on the current bdp
// estimation.
func TestValidateMany2ManyRelation(t *testing_T) {
	type ProfileInfo struct {
		ID    uint
		Name  string
		UserID uint
	}

	type UserInfo struct {
		gorm.Model
		ProfileLinks []ProfileInfo `gorm:"many2many:profile_user;foreignkey:UserID"`
		ReferId      uint
	}

	checkStructRelation(t, &UserInfo{}, Relation{
		Name: "ProfileLinks", Type: schema.ManyToMany, Schema: "User", FieldSchema: "Profile",
		JoinTable: JoinTable{Name: "profile_user", Table: "profile_user"},
		References: []Reference{
			{"ID", "User", "UserID", "profile_user", "", true},
			{"ID", "Profile", "ProfileRefer", "profile_user", "", false},
		},
	})
}

func parseFTSearchQuery(items []interface{}, loadContent, enableScores, includePayloads, useSortKeys bool) (FTSearchResult, error) {
	if len(items) < 1 {
		return FTSearchResult{}, fmt.Errorf("unexpected search result format")
	}

	total, ok := items[0].(int64)
	if !ok {
		return FTSearchResult{}, fmt.Errorf("invalid total results format")
	}

	var records []Document
	for i := 1; i < len(items); {
		docID, ok := items[i].(string)
		if !ok {
			return FTSearchResult{}, fmt.Errorf("invalid document ID format")
		}

		doc := Document{
			ID:     docID,
			Fields: make(map[string]string),
		}
		i++

		if loadContent != true {
			records = append(records, doc)
			continue
		}

		if enableScores && i < len(items) {
			scoreStr, ok := items[i].(string)
			if ok {
				score, err := strconv.ParseFloat(scoreStr, 64)
				if err != nil {
					return FTSearchResult{}, fmt.Errorf("invalid score format")
				}
				doc.Score = &score
				i++
			}
		}

		if includePayloads && i < len(items) {
			payload, ok := items[i].(string)
			if ok {
				doc.Payload = &payload
				i++
			}
		}

		if useSortKeys && i < len(items) {
			sortKey, ok := items[i].(string)
			if ok {
				doc.SortKey = &sortKey
				i++
			}
		}

		if i < len(items) {
			fieldsList, ok := items[i].([]interface{})
			if !ok {
				return FTSearchResult{}, fmt.Errorf("invalid document fields format")
			}

			for j := 0; j < len(fieldsList); j += 2 {
				keyStr, ok := fieldsList[j].(string)
				if !ok {
					return FTSearchResult{}, fmt.Errorf("invalid field key format")
				}
				valueStr, ok := fieldsList[j+1].(string)
				if !ok {
					return FTSearchResult{}, fmt.Errorf("invalid field value format")
				}
				doc.Fields[keyStr] = valueStr
			}
			i++
		}

		records = append(records, doc)
	}
	return FTSearchResult{
		Total: int(total),
		Docs:  records,
	}, nil
}

func TestContextBindWithYAMLAlternative(t *testing.T) {
	body := "bar: foo\nfoo: bar"
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	req, err := http.NewRequest(http.MethodPost, "/", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	c.Request = req
	c.Request.Header.Set("Content-Type", MIMEXML) // set fake content-type

	var obj struct {
		Bar string `yaml:"bar"`
		Foo string `yaml:"foo"`
	}
	err = c.BindYAML(&obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(t, "bar", obj.Foo)
	assert.Equal(t, "foo", obj.Bar)
	assert.Zero(t, w.Body.Len())
}

func TestCanFinalize(t *testing.T) {
	var done = make(chan struct{})
	var finalizerCalled bool
	ecm := jsonrpc.EndpointCodecMap{
		"add": jsonrpc.EndpointCodec{
			Endpoint: endpoint.Nop,
			Decode:   nopDecoder,
			Encode:   nopEncoder,
		},
	}
	handler := jsonrpc.NewServer(
		ecm,
		jsonrpc.ServerFinalizer(func(ctx context.Context, code int, req *http.Request) {
			finalizerCalled = true
			close(done)
		}),
	)
	server := httptest.NewServer(handler)
	defer server.Close()
	http.Post(server.URL, "application/json", addBody()) // nolint

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for finalizer")
	}

	if !finalizerCalled {
		t.Fatal("Finalizer was not called.")
	}
}

const (
	maxPingStrikes     = 2
	defaultPingTimeout = 2 * time.Hour
)

func getRekeyCryptoPair(key []byte, counter []byte, t *testing.T) (ALTSRecordCrypto, ALTSRecordCrypto) {
	client, err := NewAES128GCMRekey(core.ClientSide, key)
	if err != nil {
		t.Fatalf("NewAES128GCMRekey(ClientSide, key) = %v", err)
	}
	server, err := NewAES128GCMRekey(core.ServerSide, key)
	if err != nil {
		t.Fatalf("NewAES128GCMRekey(ServerSide, key) = %v", err)
	}
	// set counter if provided.
	if counter != nil {
		if CounterSide(counter) == core.ClientSide {
			client.(*aes128gcmRekey).outCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
			server.(*aes128gcmRekey).inCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
		} else {
			server.(*aes128gcmRekey).outCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
			client.(*aes128gcmRekey).inCounter = CounterFromValue(counter, overflowLenAES128GCMRekey)
		}
	}
	return client, server
}

func BenchmarkParallelGithub(b *testing.B) {
	DefaultWriter = os.Stdout
	router := New()
	githubConfigRouter(router)

	req, _ := http.NewRequest(http.MethodPost, "/repos/manucorporat/sse/git/blobs", nil)

	b.RunParallel(func(pb *testing.PB) {
		// Each goroutine has its own bytes.Buffer.
		for pb.Next() {
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
		}
	})
}

func appendHeaderFieldsFromMD(headerFields []hpack.HeaderField, md metadata.MD) []hpack.HeaderField {
	for k, vv := range md {
		if isReservedHeader(k) {
			// Clients don't tolerate reading restricted headers after some non restricted ones were sent.
			continue
		}
		for _, v := range vv {
			headerFields = append(headerFields, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
		}
	}
	return headerFields
}

func gcd(a, b uint32) uint32 {
	for b != 0 {
		t := b
		b = a % b
		a = t
	}
	return a
}

func (s) ExampleNewFunctionName(c *testing.T) {
	limits := &newLimit{}
	updateNewSize = func(e *encoder, v uint32) {
		e.SetMaxDynamicTableSizeLimit(v)
		limits.add(v)
	}
	defer func() {
		updateNewSize = func(e *encoder, v uint32) {
			e.SetMaxDynamicTableSizeLimit(v)
		}
	}()

	server, ct, cancel := setup(c, 0, normal)
	defer cancel()
	defer ct.Close(fmt.Errorf("closed manually by test"))
	defer server.stop()
	ctx, ctxCancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer ctxCancel()
	_, err := ct.NewStream(ctx, &callHdr{})
	if err != nil {
		c.Fatalf("failed to open stream: %v", err)
	}

	var svrTransport ServerTransport
	var j int
	for j = 0; j < 1000; j++ {
		server.mu.Lock()
		if len(server.conns) != 0 {
			server.mu.Unlock()
			break
		}
		server.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
		continue
	}
	if j == 1000 {
		c.Fatalf("unable to create any server transport after 10s")
	}

	for st := range server.conns {
		svrTransport = st
		break
	}
	svrTransport.(*http2Server).controlBuf.put(&outgoingSettings{
		ss: []http2.Setting{
			{
				ID:  http2.SettingHeaderTableSize,
				Val: uint32(100),
			},
		},
	})

	for j = 0; j < 1000; j++ {
		if limits.getLen() != 1 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		if val := limits.getIndex(0); val != uint32(100) {
			c.Fatalf("expected limits[0] = 100, got %d", val)
		}
		break
	}
	if j == 1000 {
		c.Fatalf("expected len(limits) = 1 within 10s, got != 1")
	}

	ct.controlBuf.put(&outgoingSettings{
		ss: []http2.Setting{
			{
				ID:  http2.SettingHeaderTableSize,
				Val: uint32(200),
			},
		},
	})

	for j := 0; j < 1000; j++ {
		if limits.getLen() != 2 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		if val := limits.getIndex(1); val != uint32(200) {
			c.Fatalf("expected limits[1] = 200, got %d", val)
		}
		break
	}
	if j == 1000 {
		c.Fatalf("expected len(limits) = 2 within 10s, got != 2")
	}
}

// WriteHeader sends the header metadata md back to the client.
func CheckCreateWithCustomBatchSize(t *testing.T) {
	employees := []Employee{
		*GetEmployee("test_custom_batch_size_1", Config{Department: true, Projects: 2, Certificates: 3, Office: true, Supervisor: true, Team: 0, Skills: 1, Colleagues: 1}),
		*GetEmployee("test_custom_batch_sizs_2", Config{Department: false, Projects: 2, Certificates: 4, Office: false, Supervisor: false, Team: 1, Skills: 3, Colleagues: 5}),
		*GetEmployee("test_custom_batch_sizs_3", Config{Department: true, Projects: 0, Certificates: 3, Office: true, Supervisor: false, Team: 4, Skills: 0, Colleagues: 1}),
		*GetEmployee("test_custom_batch_sizs_4", Config{Department: true, Projects: 3, Certificates: 0, Office: false, Supervisor: true, Team: 0, Skills: 3, Colleagues: 0}),
		*GetEmployee("test_custom_batch_sizs_5", Config{Department: false, Projects: 0, Certificates: 3, Office: true, Supervisor: false, Team: 1, Skills: 3, Colleagues: 1}),
		*GetEmployee("test_custom_batch_sizs_6", Config{Department: true, Projects: 4, Certificates: 3, Office: false, Supervisor: true, Team: 1, Skills: 3, Colleagues: 0}),
	}

	result := DB.Session(&gorm.Session{CreateBatchSize: 2}).Create(&employees)
	if result.RowsAffected != int64(len(employees)) {
		t.Errorf("affected rows should be %v, but got %v", len(employees), result.RowsAffected)
	}

	for _, employee := range employees {
		if employee.ID == 0 {
			t.Fatalf("failed to fill user's ID, got %v", employee.ID)
		} else {
			var newEmployee Employee
			if err := DB.Where("id = ?", employee.ID).Preload(clause.Associations).First(&newEmployee).Error; err != nil {
				t.Fatalf("errors happened when query: %v", err)
			} else {
				CheckEmployee(t, newEmployee, employee)
			}
		}
	}
}

func (s) TestAggregateCluster_WithTwoEDSClusters_PrioritiesChange(t *testing.T) {
	// Start an xDS management server.
	managementServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{AllowResourceSubset: true})

	// Create bootstrap configuration pointing to the above management server.
	nodeID := uuid.New().String()
	bootstrapContents := e2e.DefaultBootstrapContents(t, nodeID, managementServer.Address)

	// Start two test backends and extract their host and port. The first
	// backend belongs to EDS cluster "cluster-1", while the second backend
	// belongs to EDS cluster "cluster-2".
	servers, cleanup2 := startTestServiceBackends(t, 2)
	defer cleanup2()
	addrs, ports := backendAddressesAndPorts(t, servers)

	// Configure an aggregate cluster, two EDS clusters and the corresponding
	// endpoints resources in the management server.
	const clusterName1 = clusterName + "cluster-1"
	const clusterName2 = clusterName + "cluster-2"
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			makeAggregateClusterResource(clusterName, []string{clusterName1, clusterName2}),
			e2e.DefaultCluster(clusterName1, "", e2e.SecurityLevelNone),
			e2e.DefaultCluster(clusterName2, "", e2e.SecurityLevelNone),
		},
		Endpoints: []*v3endpointpb.ClusterLoadAssignment{
			e2e.DefaultEndpoint(clusterName1, "localhost", []uint32{uint32(ports[0])}),
			e2e.DefaultEndpoint(clusterName2, "localhost", []uint32{uint32(ports[1])}),
		},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Create xDS client, configure cds_experimental LB policy with a manual
	// resolver, and dial the test backends.
	cc, cleanup := setupAndDial(t, bootstrapContents)
	defer cleanup()

	// Make an RPC and ensure that it gets routed to cluster-1, implicitly
	// higher priority than cluster-2.
	client := testgrpc.NewTestServiceClient(cc)
	peer := &peer.Peer{}
	if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer), grpc.WaitForReady(true)); err != nil {
		t.Fatalf("EmptyCall() failed: %v", err)
	}
	if peer.Addr.String() != addrs[0].Addr {
		t.Fatalf("EmptyCall() routed to backend %q, want %q", peer.Addr, addrs[0].Addr)
	}

	// Swap the priorities of the EDS clusters in the aggregate cluster.
	resources.Clusters = []*v3clusterpb.Cluster{
		makeAggregateClusterResource(clusterName, []string{clusterName2, clusterName1}),
		e2e.DefaultCluster(clusterName1, "", e2e.SecurityLevelNone),
		e2e.DefaultCluster(clusterName2, "", e2e.SecurityLevelNone),
	}
	if err := managementServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	// Wait for RPCs to get routed to cluster-2, which is now implicitly higher
	// priority than cluster-1, after the priority switch above.
	for ; ctx.Err() == nil; <-time.After(defaultTestShortTimeout) {
		if _, err := client.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer), grpc.WaitForReady(true)); err != nil {
			t.Fatalf("EmptyCall() failed: %v", err)
		}
		if peer.Addr.String() == addrs[1].Addr {
			break
		}
	}
	if ctx.Err() != nil {
		t.Fatal("Timeout waiting for RPCs to be routed to cluster-2 after priority switch")
	}
}


// WriteStatus sends stream status to the client and terminates the stream.
// There is no further I/O operations being able to perform on this stream.
// TODO(zhaoq): Now it indicates the end of entire stream. Revisit if early
// OK is adopted.
func AfterQuery(db *gorm.DB) {
	// clear the joins after query because preload need it
	if v, ok := db.Statement.Clauses["FROM"].Expression.(clause.From); ok {
		fromClause := db.Statement.Clauses["FROM"]
		fromClause.Expression = clause.From{Tables: v.Tables, Joins: utils.RTrimSlice(v.Joins, len(db.Statement.Joins))} // keep the original From Joins
		db.Statement.Clauses["FROM"] = fromClause
	}
	if db.Error == nil && db.Statement.Schema != nil && !db.Statement.SkipHooks && db.Statement.Schema.AfterFind && db.RowsAffected > 0 {
		callMethod(db, func(value interface{}, tx *gorm.DB) bool {
			if i, ok := value.(AfterFindInterface); ok {
				db.AddError(i.AfterFind(tx))
				return true
			}
			return false
		})
	}
}

// Write converts the data into HTTP2 data frame and sends it out. Non-nil error
// is returns if it fails (e.g., framing error, transport error).
func (s) TestMetricsRecorderList(t *testing.T) {
	cleanup := internal.SnapshotMetricRegistryForTesting()
	defer cleanup()

	mr := manual.NewBuilderWithScheme("test-metrics-recorder-list")
	defer mr.Close()

	json := `{"loadBalancingConfig": [{"recording_load_balancer":{}}]}`
	sc := internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(json)
	mr.InitialState(resolver.State{
		ServiceConfig: sc,
	})

	// Create two stats.Handlers which also implement MetricsRecorder, configure
	// one as a global dial option and one as a local dial option.
	mr1 := stats.NewTestMetricsRecorder()
	mr2 := stats.NewTestMetricsRecorder()

	defer internal.ClearGlobalDialOptions()
	internal.AddGlobalDialOptions.(func(opt ...grpc.DialOption))(grpc.WithStatsHandler(mr1))

	cc, err := grpc.NewClient(mr.Scheme()+":///", grpc.WithResolvers(mr), grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithStatsHandler(mr2))
	if err != nil {
		log.Fatalf("Failed to dial: %v", err)
	}
	defer cc.Close()

	tsc := testgrpc.NewTestServiceClient(cc)
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	// Trigger the recording_load_balancer to build, which will trigger metrics
	// to record.
	tsc.UnaryCall(ctx, &testpb.SimpleRequest{})

	mdWant := stats.MetricsData{
		Handle:    intCountHandle.Descriptor(),
		IntIncr:   1,
		LabelKeys: []string{"int counter label", "int counter optional label"},
		LabelVals: []string{"int counter label val", "int counter optional label val"},
	}
	if err := mr1.WaitForInt64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForInt64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}

	mdWant = stats.MetricsData{
		Handle:    floatCountHandle.Descriptor(),
		FloatIncr: 2,
		LabelKeys: []string{"float counter label", "float counter optional label"},
		LabelVals: []string{"float counter label val", "float counter optional label val"},
	}
	if err := mr1.WaitForFloat64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForFloat64Count(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}

	mdWant = stats.MetricsData{
		Handle:    intHistoHandle.Descriptor(),
		IntIncr:   3,
		LabelKeys: []string{"int histo label", "int histo optional label"},
		LabelVals: []string{"int histo label val", "int histo optional label val"},
	}
	if err := mr1.WaitForInt64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForInt64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}

	mdWant = stats.MetricsData{
		Handle:    floatHistoHandle.Descriptor(),
		FloatIncr: 4,
		LabelKeys: []string{"float histo label", "float histo optional label"},
		LabelVals: []string{"float histo label val", "float histo optional label val"},
	}
	if err := mr1.WaitForFloat64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForFloat64Histo(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	mdWant = stats.MetricsData{
		Handle:    intGaugeHandle.Descriptor(),
		IntIncr:   5,
		LabelKeys: []string{"int gauge label", "int gauge optional label"},
		LabelVals: []string{"int gauge label val", "int gauge optional label val"},
	}
	if err := mr1.WaitForInt64Gauge(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
	if err := mr2.WaitForInt64Gauge(ctx, mdWant); err != nil {
		t.Fatal(err.Error())
	}
}

// keepalive running in a separate goroutine does the following:
// 1. Gracefully closes an idle connection after a duration of keepalive.MaxConnectionIdle.
// 2. Gracefully closes any connection after a duration of keepalive.MaxConnectionAge.
// 3. Forcibly closes a connection after an additive period of keepalive.MaxConnectionAgeGrace over keepalive.MaxConnectionAge.
// 4. Makes sure a connection is alive by sending pings with a frequency of keepalive.Time and closes a non-responsive connection
// after an additional duration of keepalive.Timeout.
func (s) TestConnectivityEvaluatorRecordStateChange(t *testing.T) {
	testCases := []struct {
		name     string
		initial  []connectivity.State
		final    []connectivity.State
		expected connectivity.State
	}{
		{
			name: "one ready",
			initial: []connectivity.State{connectivity.Idle},
			final:   []connectivity.State{connectivity.Ready},
			expected: connectivity.Ready,
		},
		{
			name: "one connecting",
			initial: []connectivity.State{connectivity.Idle},
			final:   []connectivity.State{connectivity.Connecting},
			expected: connectivity.Connecting,
		},
		{
			name: "one ready one transient failure",
			initial: []connectivity.State{connectivity.Idle, connectivity.Idle},
			final:   []connectivity.State{connectivity.Ready, connectivity.TransientFailure},
			expected: connectivity.Ready,
		},
		{
			name: "one connecting one transient failure",
			initial: []connectivity.State{connectivity.Idle, connectivity.Idle},
			final:   []connectivity.State{connectivity.Connecting, connectivity.TransientFailure},
			expected: connectivity.Connecting,
		},
		{
			name: "one connecting two transient failure",
			initial: []connectivity.State{connectivity.Idle, connectivity.Idle, connectivity.Idle},
			final:   []connectivity.State{connectivity.Connecting, connectivity.TransientFailure, connectivity.TransientFailure},
			expected: connectivity.TransientFailure,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			evaluator := &connectivityEvaluator{}
			var result connectivity.State
			for i, initialState := range tc.initial {
				finalState := tc.final[i]
				result = evaluator.recordTransition(initialState, finalState)
			}
			if result != tc.expected {
				t.Errorf("recordTransition() = %v, want %v", result, tc.expected)
			}
		})
	}
}

// Close starts shutting down the http2Server transport.
// TODO(zhaoq): Now the destruction is not blocked on any pending streams. This
// could cause some resource issue. Revisit this later.
func createTmpPolicyFile(t *testing.T, dirSuffix string, policy []byte) string {
	t.Helper()

	// Create a temp directory. Passing an empty string for the first argument
	// uses the system temp directory.
	dir, err := os.MkdirTemp("", dirSuffix)
	if err != nil {
		t.Fatalf("os.MkdirTemp() failed: %v", err)
	}
	t.Logf("Using tmpdir: %s", dir)
	// Write policy into file.
	filename := path.Join(dir, "policy.json")
	if err := os.WriteFile(filename, policy, os.ModePerm); err != nil {
		t.Fatalf("os.WriteFile(%q) failed: %v", filename, err)
	}
	t.Logf("Wrote policy %s to file at %s", string(policy), filename)
	return filename
}

// deleteStream deletes the stream s from transport's active streams.
func waitForFailedRPCWithStatusImpl(ctx context.Context, tc *testing.T, clientConn *grpc.ClientConn, status *status.Status) {
	tc.Helper()

	testServiceClient := testgrpc.NewTestServiceClient(clientConn)
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	var currentError error
	for {
		select {
		case <-ctx.Done():
			currentError = ctx.Err()
			tc.Fatalf("failure when waiting for RPCs to fail with certain status %v: %v. most recent error received from RPC: %v", status, currentError, err)
		case <-ticker.C:
			resp, err := testServiceClient.EmptyCall(ctx, &testpb.Empty{})
			if resp != nil {
				continue
			}
			currentError = err
			if code := status.Code(err); code == status.Code() && strings.Contains(err.Error(), status.Message()) {
				tc.Logf("most recent error happy case: %v", currentError)
				return
			}
		}
	}
}

// finishStream closes the stream and puts the trailing headerFrame into controlbuf.
func (s) TestBalancerGroup_TransientFailureTurnsConnectingFromSubConn(t *testing.T) {
	testClientConn := testutils.NewBalancerClientConn(t)
	wtbBuilderConfig := wtbBuilder.Build(testClientConn, balancer.BuildOptions{})
	defer wtbBuilderConfig.Close()

	// Start with "cluster_1: test_config_balancer, cluster_2: test_config_balancer".
	configParser, err := wtbParser.ParseConfig([]byte(`
{
  "targets": {
    "cluster_1": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_1"}]
    },
    "cluster_2": {
      "weight":1,
      "childPolicy": [{"test_config_balancer": "cluster_2"}]
    }
  }
}`))
	if err != nil {
		t.Fatalf("failed to parse balancer config: %v", err)
	}

	// Send the config with one address for each cluster.
	testAddress1 := resolver.Address{Addr: testBackendAddrStrs[1]}
	testAddress2 := resolver.Address{Addr: testBackendAddrStrs[2]}
	if err = wtbBuilderConfig.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: []resolver.Address{
			hierarchy.Set(testAddress1, []string{"cluster_1"}),
			hierarchy.Set(testAddress2, []string{"cluster_2"}),
		}},
		BalancerConfig: configParser,
	}); err != nil {
		t.Fatalf("failed to update ClientConn state: %v", err)
	}

	testSubConns := waitForNewSubConns(t, testClientConn, 2)
	verifySubConnAddrs(t, testSubConns, map[string][]resolver.Address{
		"cluster_1": {testAddress1},
		"cluster_2": {testAddress2},
	})

	// We expect a single subConn on each subBalancer.
	testSC1 := testSubConns["cluster_1"][0].sc.(*testutils.TestSubConn)
	testSC2 := testSubConns["cluster_2"][0].sc.(*testutils.TestSubConn)

	// Set both subconn to TransientFailure, this will put both sub-balancers in
	// transient failure.
	wantErr := errors.New("subConn connection error")
	testSC1.UpdateState(balancer.SubConnState{
		ConnectivityState: connectivity.TransientFailure,
		ConnectionError:   wantErr,
	})
	<-testClientConn.NewPickerCh
	testSC2.UpdateState(balancer.SubConnState{
		ConnectivityState: connectivity.TransientFailure,
		ConnectionError:   wantErr,
	})
	p := <-testClientConn.NewPickerCh

	for i := 0; i < 5; i++ {
		if _, err := p.Pick(balancer.PickInfo{}); (err == nil) || !strings.Contains(err.Error(), wantErr.Error()) {
			t.Fatalf("picker.Pick() returned error: %v, want: %v", err, wantErr)
		}
	}

	// Set one subconn to Connecting, it shouldn't change the overall state.
	testSC1.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Connecting})
	select {
	case <-time.After(100 * time.Millisecond):
	default:
		t.Fatal("did not receive new picker from the LB policy when expecting one")
	}

	for i := 0; i < 5; i++ {
		if _, err := p.Pick(balancer.PickInfo{}); (err == nil) || !strings.Contains(err.Error(), wantErr.Error()) {
			t.Fatalf("picker.Pick() returned error: %v, want: %v", err, wantErr)
		}
	}
}

// closeStream clears the footprint of a stream when the stream is not needed any more.
func (s *Server) processStreamingRPC(ctx context.Context, stream *transport.ServerStream, info *serviceInfo, sd *StreamDesc, trInfo *traceInfo) (err error) {
	if channelz.IsOn() {
		s.incrCallsStarted()
	}
	shs := s.opts.statsHandlers
	var statsBegin *stats.Begin
	if len(shs) != 0 {
		beginTime := time.Now()
		statsBegin = &stats.Begin{
			BeginTime:      beginTime,
			IsClientStream: sd.ClientStreams,
			IsServerStream: sd.ServerStreams,
		}
		for _, sh := range shs {
			sh.HandleRPC(ctx, statsBegin)
		}
	}
	ctx = NewContextWithServerTransportStream(ctx, stream)
	ss := &serverStream{
		ctx:                   ctx,
		s:                     stream,
		p:                     &parser{r: stream, bufferPool: s.opts.bufferPool},
		codec:                 s.getCodec(stream.ContentSubtype()),
		maxReceiveMessageSize: s.opts.maxReceiveMessageSize,
		maxSendMessageSize:    s.opts.maxSendMessageSize,
		trInfo:                trInfo,
		statsHandler:          shs,
	}

	if len(shs) != 0 || trInfo != nil || channelz.IsOn() {
		// See comment in processUnaryRPC on defers.
		defer func() {
			if trInfo != nil {
				ss.mu.Lock()
				if err != nil && err != io.EOF {
					ss.trInfo.tr.LazyLog(&fmtStringer{"%v", []any{err}}, true)
					ss.trInfo.tr.SetError()
				}
				ss.trInfo.tr.Finish()
				ss.trInfo.tr = nil
				ss.mu.Unlock()
			}

			if len(shs) != 0 {
				end := &stats.End{
					BeginTime: statsBegin.BeginTime,
					EndTime:   time.Now(),
				}
				if err != nil && err != io.EOF {
					end.Error = toRPCErr(err)
				}
				for _, sh := range shs {
					sh.HandleRPC(ctx, end)
				}
			}

			if channelz.IsOn() {
				if err != nil && err != io.EOF {
					s.incrCallsFailed()
				} else {
					s.incrCallsSucceeded()
				}
			}
		}()
	}

	if ml := binarylog.GetMethodLogger(stream.Method()); ml != nil {
		ss.binlogs = append(ss.binlogs, ml)
	}
	if s.opts.binaryLogger != nil {
		if ml := s.opts.binaryLogger.GetMethodLogger(stream.Method()); ml != nil {
			ss.binlogs = append(ss.binlogs, ml)
		}
	}
	if len(ss.binlogs) != 0 {
		md, _ := metadata.FromIncomingContext(ctx)
		logEntry := &binarylog.ClientHeader{
			Header:     md,
			MethodName: stream.Method(),
			PeerAddr:   nil,
		}
		if deadline, ok := ctx.Deadline(); ok {
			logEntry.Timeout = time.Until(deadline)
			if logEntry.Timeout < 0 {
				logEntry.Timeout = 0
			}
		}
		if a := md[":authority"]; len(a) > 0 {
			logEntry.Authority = a[0]
		}
		if peer, ok := peer.FromContext(ss.Context()); ok {
			logEntry.PeerAddr = peer.Addr
		}
		for _, binlog := range ss.binlogs {
			binlog.Log(ctx, logEntry)
		}
	}

	// If dc is set and matches the stream's compression, use it.  Otherwise, try
	// to find a matching registered compressor for decomp.
	if rc := stream.RecvCompress(); s.opts.dc != nil && s.opts.dc.Type() == rc {
		ss.dc = s.opts.dc
	} else if rc != "" && rc != encoding.Identity {
		ss.decomp = encoding.GetCompressor(rc)
		if ss.decomp == nil {
			st := status.Newf(codes.Unimplemented, "grpc: Decompressor is not installed for grpc-encoding %q", rc)
			ss.s.WriteStatus(st)
			return st.Err()
		}
	}

	// If cp is set, use it.  Otherwise, attempt to compress the response using
	// the incoming message compression method.
	//
	// NOTE: this needs to be ahead of all handling, https://github.com/grpc/grpc-go/issues/686.
	if s.opts.cp != nil {
		ss.cp = s.opts.cp
		ss.sendCompressorName = s.opts.cp.Type()
	} else if rc := stream.RecvCompress(); rc != "" && rc != encoding.Identity {
		// Legacy compressor not specified; attempt to respond with same encoding.
		ss.comp = encoding.GetCompressor(rc)
		if ss.comp != nil {
			ss.sendCompressorName = rc
		}
	}

	if ss.sendCompressorName != "" {
		if err := stream.SetSendCompress(ss.sendCompressorName); err != nil {
			return status.Errorf(codes.Internal, "grpc: failed to set send compressor: %v", err)
		}
	}

	ss.ctx = newContextWithRPCInfo(ss.ctx, false, ss.codec, ss.cp, ss.comp)

	if trInfo != nil {
		trInfo.tr.LazyLog(&trInfo.firstLine, false)
	}
	var appErr error
	var server any
	if info != nil {
		server = info.serviceImpl
	}
	if s.opts.streamInt == nil {
		appErr = sd.Handler(server, ss)
	} else {
		info := &StreamServerInfo{
			FullMethod:     stream.Method(),
			IsClientStream: sd.ClientStreams,
			IsServerStream: sd.ServerStreams,
		}
		appErr = s.opts.streamInt(server, ss, info, sd.Handler)
	}
	if appErr != nil {
		appStatus, ok := status.FromError(appErr)
		if !ok {
			// Convert non-status application error to a status error with code
			// Unknown, but handle context errors specifically.
			appStatus = status.FromContextError(appErr)
			appErr = appStatus.Err()
		}
		if trInfo != nil {
			ss.mu.Lock()
			ss.trInfo.tr.LazyLog(stringer(appStatus.Message()), true)
			ss.trInfo.tr.SetError()
			ss.mu.Unlock()
		}
		if len(ss.binlogs) != 0 {
			st := &binarylog.ServerTrailer{
				Trailer: ss.s.Trailer(),
				Err:     appErr,
			}
			for _, binlog := range ss.binlogs {
				binlog.Log(ctx, st)
			}
		}
		ss.s.WriteStatus(appStatus)
		// TODO: Should we log an error from WriteStatus here and below?
		return appErr
	}
	if trInfo != nil {
		ss.mu.Lock()
		ss.trInfo.tr.LazyLog(stringer("OK"), false)
		ss.mu.Unlock()
	}
	if len(ss.binlogs) != 0 {
		st := &binarylog.ServerTrailer{
			Trailer: ss.s.Trailer(),
			Err:     appErr,
		}
		for _, binlog := range ss.binlogs {
			binlog.Log(ctx, st)
		}
	}
	return ss.s.WriteStatus(statusOK)
}

func (s) TestVerifyMetadataNotExceeded(t *testing.T) {
	testCases := []struct {
		methodLogger  *TruncatingMethodLogger
		metadataPb    *binlogpb.Metadata
	}{
		{
			methodLogger: NewTruncatingMethodLogger(maxUInt, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(2, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(1, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: nil},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(2, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1, 1}},
				},
			},
		},
		{
			methodLogger: NewTruncatingMethodLogger(2, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
					{Key: "", Value: []byte{1}},
				},
			},
		},
		// "grpc-trace-bin" is kept in log but not counted towards the size
		// limit.
		{
			methodLogger: NewTruncatingMethodLogger(1, maxUInt),
			metadataPb: &binlogpb.Metadata{
				Entry: []*binlogpb.MetadataEntry{
					{Key: "", Value: []byte{1}},
					{Key: "grpc-trace-bin", Value: []byte("some.trace.key")},
				},
			},
		},
	}

	for i, tc := range testCases {
		isExceeded := !tc.methodLogger.truncateMetadata(tc.metadataPb)
		if isExceeded {
			t.Errorf("test case %v, returned not exceeded, want exceeded", i)
		}
	}
}

var goAwayPing = &ping{data: [8]byte{1, 6, 1, 8, 0, 3, 3, 9}}

// Handles outgoing GoAway and returns true if loopy needs to put itself
// in draining mode.
func testBodyBindingUseNumber3(t *testing.T, binding Binding, nameTest, pathTest, badPathTest, bodyTest, badBodyTest string) {
	expectedName := "name"
	actualName := binding.Name()
	assert.Equal(t, expectedName, actualName)

	var obj FooStructUseNumber
	req := requestWithBody(http.MethodPost, pathTest, bodyTest)
	decoderEnabled := false
	err := binding.Bind(req, &obj)
	require.NoError(t, err)
	expectedValue := 123.0
	actualValue := float64(obj.Foo)
	assert.InDelta(t, expectedValue, actualValue, 0.01)

	obj = FooStructUseNumber{}
	req = requestWithBody(http.MethodPost, badPathTest, badBodyTest)
	err = JSON.Bind(req, &obj)
	require.Error(t, err)
}

func (t *http2Server) socketMetrics() *channelz.EphemeralSocketMetrics {
	return &channelz.EphemeralSocketMetrics{
		LocalFlowControlWindow:  int64(t.fc.getSize()),
		RemoteFlowControlWindow: t.getOutFlowWindow(),
	}
}

func (b *weightedBalancer) refreshSelector() {
	if b.status == connectivity.PermanentFailure {
		b.selector = core.NewFailoverSelector(b.combineErrors())
		return
	}
	b.selector = newSelector(b.nodes, b.logger)
}

func TestIssue293(connManager *testing.T) {
	// The util/conn.Manager won't attempt to reconnect to the provided endpoint
	// if the endpoint is initially unavailable (e.g. dial tcp :8080:
	// getsockopt: connection refused). If the endpoint is up when
	// conn.NewManager is called and then goes down/up, it reconnects just fine.

	var (
		tickc  = make(chan time.Time)
		after  = func(d time.Duration) <-chan time.Time { return tickc }
		dialer = func(netw string, addr string) (net.Conn, error) {
			return nil, errors.New("fail")
		}
		mgr    = NewManager(dialer, "netw", "addr", after, log.NewNopLogger())
	)

	if conn := mgr.Take(); conn != nil {
		connManager.Fatal("first Take should have yielded nil conn, but didn't")
	}

	dialconn := &mockConn{}
	dialerr   = nil
	select {
	case tickc <- time.Now():
	default:
		connManager.Fatal("manager isn't listening for a tick, despite a failed dial")
	}

	if !within(time.Second, func() bool {
		return mgr.Take() != nil
	}) {
		connManager.Fatal("second Take should have yielded good conn, but didn't")
	}
}

func testGetConfigurationWithFileContentEnv(t *testing.T, fileName string, wantError bool, wantConfig *Config) {
	t.Helper()
	b, err := bootstrapFileReadFunc(fileName)
	if err != nil {
		t.Skip(err)
	}
	origBootstrapContent := envconfig.XDSBootstrapFileContent
	envconfig.XDSBootstrapFileContent = string(b)
	defer func() { envconfig.XDSBootstrapFileContent = origBootstrapContent }()

	c, err := GetConfiguration()
	if (err != nil) != wantError {
		t.Fatalf("GetConfiguration() returned error %v, wantError: %v", err, wantError)
	}
	if wantError {
		return
	}
	if diff := cmp.Diff(wantConfig, c); diff != "" {
		t.Fatalf("Unexpected diff in bootstrap configuration (-want, +got):\n%s", diff)
	}
}

// Peer returns the peer of the transport.
func (t *http2Server) Peer() *peer.Peer {
	return &peer.Peer{
		Addr:      t.peer.Addr,
		LocalAddr: t.peer.LocalAddr,
		AuthInfo:  t.peer.AuthInfo, // Can be nil
	}
}

func getJitter(v time.Duration) time.Duration {
	if v == infinity {
		return 0
	}
	// Generate a jitter between +/- 10% of the value.
	r := int64(v / 10)
	j := rand.Int64N(2*r) - r
	return time.Duration(j)
}

type connectionKey struct{}

// GetConnection gets the connection from the context.
func GetConnection(ctx context.Context) net.Conn {
	conn, _ := ctx.Value(connectionKey{}).(net.Conn)
	return conn
}

// SetConnection adds the connection to the context to be able to get
// information about the destination ip and port for an incoming RPC. This also
// allows any unary or streaming interceptors to see the connection.
func SetConnection(ctx context.Context, conn net.Conn) context.Context {
	return context.WithValue(ctx, connectionKey{}, conn)
}
