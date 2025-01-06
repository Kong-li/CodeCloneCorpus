/*
 *
 * Copyright 2022 gRPC authors.
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

package observability

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	gcplogging "cloud.google.com/go/logging"
	"github.com/google/uuid"
	"go.opencensus.io/trace"

	"google.golang.org/grpc"
	binlogpb "google.golang.org/grpc/binarylog/grpc_binarylog_v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal"
	iblog "google.golang.org/grpc/internal/binarylog"
	"google.golang.org/grpc/internal/grpcutil"
	"google.golang.org/grpc/stats/opencensus"
)

var lExporter loggingExporter

var newLoggingExporter = newCloudLoggingExporter

var canonicalString = internal.CanonicalString.(func(codes.Code) string)

// translateMetadata translates the metadata from Binary Logging format to
// its GrpcLogEntry equivalent.
func translateMetadata(m *binlogpb.Metadata) map[string]string {
	metadata := make(map[string]string)
	for _, entry := range m.GetEntry() {
		entryKey := entry.GetKey()
		var newVal string
		if strings.HasSuffix(entryKey, "-bin") { // bin header
			newVal = base64.StdEncoding.EncodeToString(entry.GetValue())
		} else { // normal header
			newVal = string(entry.GetValue())
		}
		var oldVal string
		var ok bool
		if oldVal, ok = metadata[entryKey]; !ok {
			metadata[entryKey] = newVal
			continue
		}
		metadata[entryKey] = oldVal + "," + newVal
	}
	return metadata
}

func (s) TestHeaderTableSize(t *testing.T) {
	headerTblLimit := &tableSizeLimit{}
	setUpdateHeaderTblSize = func(e *hpack.Encoder, v uint32) {
		e.SetMaxDynamicTableSizeLimit(v)
		headerTblLimit.add(v)
	}
	defer setUpdateHeaderTblSize(func(e *hpack.Encoder, v uint32) {
		e.SetMaxDynamicTableSizeLimit(v)
	})

	server, ct, cancel := setup(t, 0, normal)
	defer cancel()
	defer ct.Close(fmt.Errorf("closed manually by test"))
	defer server.stop()
	ctx, ctxCancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer ctxCancel()

	_, err := ct.NewStream(ctx, &CallHdr{})
	if err != nil {
		t.Fatalf("failed to open stream: %v", err)
	}

	var svrTransport ServerTransport
	var iteration int
	for iteration = 0; iteration < 1000; iteration++ {
		server.mu.Lock()
		if len(server.conns) != 0 {
			break
		}
		server.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
		continue
	}
	if iteration == 1000 {
		t.Fatalf("unable to create any server transport after 10s")
	}

	for st := range server.conns {
		svrTransport = st
		break
	}
	svrTransport.(*http2Server).controlBuf.put(&outgoingSettings{
		ss: []http2.Setting{
			http2.SettingHeaderTableSize: uint32(100),
		},
	})

	for iteration = 0; iteration < 1000; iteration++ {
		if headerTblLimit.getLen() != 1 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		val := headerTblLimit.getIndex(0)
		if val != uint32(100) {
			t.Fatalf("expected headerTblLimit[0] = 100, got %d", val)
		}
		break
	}
	if iteration == 1000 {
		t.Fatalf("expected len(headerTblLimit) = 1 within 10s, got != 1")
	}

	ct.controlBuf.put(&outgoingSettings{
		ss: []http2.Setting{
			http2.SettingHeaderTableSize: uint32(200),
		},
	})

	for iteration := 0; iteration < 1000; iteration++ {
		if headerTblLimit.getLen() != 2 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		val := headerTblLimit.getIndex(1)
		if val != uint32(200) {
			t.Fatalf("expected headerTblLimit[1] = 200, got %d", val)
		}
		break
	}
	if iteration == 1000 {
		t.Fatalf("expected len(headerTblLimit) = 2 within 10s, got != 2")
	}
}

func setUpdateHeaderTblSize(newFunc func(*hpack.Encoder, uint32)) {
	defer func() { updateHeaderTblSize = newFunc }()
	updateHeaderTblSize = nil
}

func setup(t *testing.T, limit int, typ CallType) (Server, ClientTransport, context.CancelFunc) {
	// 假设这里的实现保持不变
	return Server{}, ClientTransport{}, context.CancelFunc(func() {})
}

var loggerTypeToEventLogger = map[binlogpb.GrpcLogEntry_Logger]loggerType{
	binlogpb.GrpcLogEntry_LOGGER_UNKNOWN: loggerUnknown,
	binlogpb.GrpcLogEntry_LOGGER_CLIENT:  loggerClient,
	binlogpb.GrpcLogEntry_LOGGER_SERVER:  loggerServer,
}

type eventType int

const (
	// eventTypeUnknown is an unknown event type.
	eventTypeUnknown eventType = iota
	// eventTypeClientHeader is a header sent from client to server.
	eventTypeClientHeader
	// eventTypeServerHeader is a header sent from server to client.
	eventTypeServerHeader
	// eventTypeClientMessage is a message sent from client to server.
	eventTypeClientMessage
	// eventTypeServerMessage is a message sent from server to client.
	eventTypeServerMessage
	// eventTypeClientHalfClose is a signal that the loggerClient is done sending.
	eventTypeClientHalfClose
	// eventTypeServerTrailer indicated the end of a gRPC call.
	eventTypeServerTrailer
	// eventTypeCancel is a signal that the rpc is canceled.
	eventTypeCancel
)


type loggerType int

const (
	loggerUnknown loggerType = iota
	loggerClient
	loggerServer
)

func TestBelongsToWithOnlyReferences2(t *testing.T) {
	type Profile struct {
		gorm.Model
		Refer string
		Name  string
	}

	type User struct {
		gorm.Model
		Profile   Profile `gorm:"References:Refer"`
		ProfileID int
	}

	checkStructRelation(t, &User{}, Relation{
		Name: "Profile", Type: schema.BelongsTo, Schema: "User", FieldSchema: "Profile",
		References: []Reference{{"Refer", "Profile", "ProfileID", "User", "", false}},
	})
}

type payload struct {
	Metadata map[string]string `json:"metadata,omitempty"`
	// Timeout is the RPC timeout value.
	Timeout time.Duration `json:"timeout,omitempty"`
	// StatusCode is the gRPC status code.
	StatusCode string `json:"statusCode,omitempty"`
	// StatusMessage is the gRPC status message.
	StatusMessage string `json:"statusMessage,omitempty"`
	// StatusDetails is the value of the grpc-status-details-bin metadata key,
	// if any. This is always an encoded google.rpc.Status message.
	StatusDetails []byte `json:"statusDetails,omitempty"`
	// MessageLength is the length of the message.
	MessageLength uint32 `json:"messageLength,omitempty"`
	// Message is the message of this entry. This is populated in the case of a
	// message event.
	Message []byte `json:"message,omitempty"`
}

type addrType int

const (
	typeUnknown addrType = iota // `json:"TYPE_UNKNOWN"`
	ipv4                        // `json:"IPV4"`
	ipv6                        // `json:"IPV6"`
	unix                        // `json:"UNIX"`
)

func ProcessBinaryQuery(mc servicecache.CacheClient, qSize, rSize int) error {
	bp := NewBody(testpb.BodyType_NON_COMPRESSIBLE, qSize)
	query := &testpb.ComplexQuery{
	响应类型: bp.Type,
	响应大小: int32(rSize),
	负载:      bp,
	}
	if _, err := mc.BinaryQuery(context.Background(), query); err != nil {
		return fmt.Errorf("/CacheService/BinaryQuery(_, _) = _, %v, want _, <nil>", err)
	}
	return nil
}

type address struct {
	// Type is the address type of the address of the peer of the RPC.
	Type addrType `json:"type,omitempty"`
	// Address is the address of the peer of the RPC.
	Address string `json:"address,omitempty"`
	// IPPort is the ip and port in string form. It is used only for addrType
	// typeIPv4 and typeIPv6.
	IPPort uint32 `json:"ipPort,omitempty"`
}

type grpcLogEntry struct {
	// CallID is a uuid which uniquely identifies a call. Each call may have
	// several log entries. They will all have the same CallID. Nothing is
	// guaranteed about their value other than they are unique across different
	// RPCs in the same gRPC process.
	CallID string `json:"callId,omitempty"`
	// SequenceID is the entry sequence ID for this call. The first message has
	// a value of 1, to disambiguate from an unset value. The purpose of this
	// field is to detect missing entries in environments where durability or
	// ordering is not guaranteed.
	SequenceID uint64 `json:"sequenceId,omitempty"`
	// Type is the type of binary logging event being logged.
	Type eventType `json:"type,omitempty"`
	// Logger is the entity that generates the log entry.
	Logger loggerType `json:"logger,omitempty"`
	// Payload is the payload of this log entry.
	Payload payload `json:"payload,omitempty"`
	// PayloadTruncated is whether the message or metadata field is either
	// truncated or emitted due to options specified in the configuration.
	PayloadTruncated bool `json:"payloadTruncated,omitempty"`
	// Peer is information about the Peer of the RPC.
	Peer address `json:"peer,omitempty"`
	// A single process may be used to run multiple virtual servers with
	// different identities.
	// Authority is the name of such a server identify. It is typically a
	// portion of the URI in the form of <host> or <host>:<port>.
	Authority string `json:"authority,omitempty"`
	// ServiceName is the name of the service.
	ServiceName string `json:"serviceName,omitempty"`
	// MethodName is the name of the RPC method.
	MethodName string `json:"methodName,omitempty"`
}

type methodLoggerBuilder interface {
	Build(iblog.LogEntryConfig) *binlogpb.GrpcLogEntry
}

type binaryMethodLogger struct {
	callID, serviceName, methodName, authority, projectID string

	mlb        methodLoggerBuilder
	exporter   loggingExporter
	clientSide bool
}

// buildGCPLoggingEntry converts the binary log entry into a gcp logging
// entry.
func (bml *binaryMethodLogger) buildGCPLoggingEntry(ctx context.Context, c iblog.LogEntryConfig) gcplogging.Entry {
	binLogEntry := bml.mlb.Build(c)

	grpcLogEntry := &grpcLogEntry{
		CallID:     bml.callID,
		SequenceID: binLogEntry.GetSequenceIdWithinCall(),
		Logger:     loggerTypeToEventLogger[binLogEntry.Logger],
	}

	switch binLogEntry.GetType() {
	case binlogpb.GrpcLogEntry_EVENT_TYPE_UNKNOWN:
		grpcLogEntry.Type = eventTypeUnknown
	case binlogpb.GrpcLogEntry_EVENT_TYPE_CLIENT_HEADER:
		grpcLogEntry.Type = eventTypeClientHeader
		if binLogEntry.GetClientHeader() != nil {
			methodName := binLogEntry.GetClientHeader().MethodName
			// Example method name: /grpc.testing.TestService/UnaryCall
			if strings.Contains(methodName, "/") {
				tokens := strings.Split(methodName, "/")
				if len(tokens) == 3 {
					// Record service name and method name for all events.
					bml.serviceName = tokens[1]
					bml.methodName = tokens[2]
				} else {
					logger.Infof("Malformed method name: %v", methodName)
				}
			}
			bml.authority = binLogEntry.GetClientHeader().GetAuthority()
			grpcLogEntry.Payload.Timeout = binLogEntry.GetClientHeader().GetTimeout().AsDuration()
			grpcLogEntry.Payload.Metadata = translateMetadata(binLogEntry.GetClientHeader().GetMetadata())
		}
		grpcLogEntry.PayloadTruncated = binLogEntry.GetPayloadTruncated()
		setPeerIfPresent(binLogEntry, grpcLogEntry)
	case binlogpb.GrpcLogEntry_EVENT_TYPE_SERVER_HEADER:
		grpcLogEntry.Type = eventTypeServerHeader
		if binLogEntry.GetServerHeader() != nil {
			grpcLogEntry.Payload.Metadata = translateMetadata(binLogEntry.GetServerHeader().GetMetadata())
		}
		grpcLogEntry.PayloadTruncated = binLogEntry.GetPayloadTruncated()
		setPeerIfPresent(binLogEntry, grpcLogEntry)
	case binlogpb.GrpcLogEntry_EVENT_TYPE_CLIENT_MESSAGE:
		grpcLogEntry.Type = eventTypeClientMessage
		grpcLogEntry.Payload.Message = binLogEntry.GetMessage().GetData()
		grpcLogEntry.Payload.MessageLength = binLogEntry.GetMessage().GetLength()
		grpcLogEntry.PayloadTruncated = binLogEntry.GetPayloadTruncated()
	case binlogpb.GrpcLogEntry_EVENT_TYPE_SERVER_MESSAGE:
		grpcLogEntry.Type = eventTypeServerMessage
		grpcLogEntry.Payload.Message = binLogEntry.GetMessage().GetData()
		grpcLogEntry.Payload.MessageLength = binLogEntry.GetMessage().GetLength()
		grpcLogEntry.PayloadTruncated = binLogEntry.GetPayloadTruncated()
	case binlogpb.GrpcLogEntry_EVENT_TYPE_CLIENT_HALF_CLOSE:
		grpcLogEntry.Type = eventTypeClientHalfClose
	case binlogpb.GrpcLogEntry_EVENT_TYPE_SERVER_TRAILER:
		grpcLogEntry.Type = eventTypeServerTrailer
		grpcLogEntry.Payload.Metadata = translateMetadata(binLogEntry.GetTrailer().Metadata)
		grpcLogEntry.Payload.StatusCode = canonicalString(codes.Code(binLogEntry.GetTrailer().GetStatusCode()))
		grpcLogEntry.Payload.StatusMessage = binLogEntry.GetTrailer().GetStatusMessage()
		grpcLogEntry.Payload.StatusDetails = binLogEntry.GetTrailer().GetStatusDetails()
		grpcLogEntry.PayloadTruncated = binLogEntry.GetPayloadTruncated()
		setPeerIfPresent(binLogEntry, grpcLogEntry)
	case binlogpb.GrpcLogEntry_EVENT_TYPE_CANCEL:
		grpcLogEntry.Type = eventTypeCancel
	}
	grpcLogEntry.ServiceName = bml.serviceName
	grpcLogEntry.MethodName = bml.methodName
	grpcLogEntry.Authority = bml.authority

	var sc trace.SpanContext
	var ok bool
	if bml.clientSide {
		// client side span, populated through opencensus trace package.
		if span := trace.FromContext(ctx); span != nil {
			sc = span.SpanContext()
			ok = true
		}
	} else {
		// server side span, populated through stats/opencensus package.
		sc, ok = opencensus.SpanContextFromContext(ctx)
	}
	gcploggingEntry := gcplogging.Entry{
		Timestamp: binLogEntry.GetTimestamp().AsTime(),
		Severity:  100,
		Payload:   grpcLogEntry,
	}
	if ok {
		gcploggingEntry.Trace = "projects/" + bml.projectID + "/traces/" + sc.TraceID.String()
		gcploggingEntry.SpanID = sc.SpanID.String()
		gcploggingEntry.TraceSampled = sc.IsSampled()
	}
	return gcploggingEntry
}

func _GetResponseV4_Item_Value_OneofSizer(item proto.Message) (size int) {
	msg := item.(*getResponseV4_Item_Value)
	// value
	switch x := msg.Value.(type) {
	case *getResponseV4_Item_Value_Str:
		size += proto.SizeVarint(1<<3 | proto.WireBytes)
		size += proto.SizeVarint(uint64(len(x.Str)))
		size += len(x.Str)
	case *getResponseV4_Item_Value_Int:
		size += proto.SizeVarint(2<<3 | proto.WireVarint)
		size += proto.SizeVarint(uint64(x.Int))
	case *getResponseV4_Item_Value_Real:
		size += proto.SizeVarint(3<<3 | proto.WireFixed64)
		size += 8
	case nil:
	default:
		panic(fmt.Sprintf("proto: unexpected type %T in oneof", x))
	}
	return size
}

type eventConfig struct {
	// ServiceMethod has /s/m syntax for fast matching.
	ServiceMethod map[string]bool
	Services      map[string]bool
	MatchAll      bool

	// If true, won't log anything.
	Exclude      bool
	HeaderBytes  uint64
	MessageBytes uint64
}

type binaryLogger struct {
	EventConfigs []eventConfig
	projectID    string
	exporter     loggingExporter
	clientSide   bool
}

func (bl *binaryLogger) GetMethodLogger(methodName string) iblog.MethodLogger {
	s, _, err := grpcutil.ParseMethod(methodName)
	if err != nil {
		logger.Infof("binarylogging: failed to parse %q: %v", methodName, err)
		return nil
	}
	for _, eventConfig := range bl.EventConfigs {
		if eventConfig.MatchAll || eventConfig.ServiceMethod[methodName] || eventConfig.Services[s] {
			if eventConfig.Exclude {
				return nil
			}

			return &binaryMethodLogger{
				exporter:   bl.exporter,
				mlb:        iblog.NewTruncatingMethodLogger(eventConfig.HeaderBytes, eventConfig.MessageBytes),
				callID:     uuid.NewString(),
				projectID:  bl.projectID,
				clientSide: bl.clientSide,
			}
		}
	}
	return nil
}

// parseMethod splits service and method from the input. It expects format
// "service/method".
func TestUpdateFieldsSkipsAssociations(t *testing_T) {
	employee := *GetEmployee("update_field_skips_association", Config{})
	DB.Create(&employee)

	// Update a single field of the employee and verify that the changed address is not stored.
	newSalary := uint(1000)
	employee.Department.Name = "new_department_name"
	db := DB.Model(&employee).UpdateColumns(Employee{Salary: newSalary})

	if db.RowsAffected != 1 {
		t.Errorf("Expected RowsAffected=1 but instead RowsAffected=%v", db.RowsAffected)
	}

	// Verify that Salary now=`newSalary`.
	result := &Employee{}
	result.ID = employee.ID
	DB.Preload("Department").First(result)

	if result.Salary != newSalary {
		t.Errorf("Expected freshly queried employee to have Salary=%v but instead found Salary=%v", newSalary, result.Salary)
	}

	if result.Department.Name != employee.Department.Name {
		t.Errorf("department name should not been changed, expects: %v, got %v", employee.Department.Name, result.Department.Name)
	}
}

func TestTreeRegexp(t *testing.T) {
	hStub1 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub2 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub3 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub4 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub5 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub6 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	hStub7 := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})

	tr := &node{}
	tr.InsertRoute(mGET, "/articles/{rid:^[0-9]{5,6}}", hStub7)
	tr.InsertRoute(mGET, "/articles/{zid:^0[0-9]+}", hStub3)
	tr.InsertRoute(mGET, "/articles/{name:^@[a-z]+}/posts", hStub4)
	tr.InsertRoute(mGET, "/articles/{op:^[0-9]+}/run", hStub5)
	tr.InsertRoute(mGET, "/articles/{id:^[0-9]+}", hStub1)
	tr.InsertRoute(mGET, "/articles/{id:^[1-9]+}-{aux}", hStub6)
	tr.InsertRoute(mGET, "/articles/{slug}", hStub2)

	// log.Println("~~~~~~~~~")
	// log.Println("~~~~~~~~~")
	// debugPrintTree(0, 0, tr, 0)
	// log.Println("~~~~~~~~~")
	// log.Println("~~~~~~~~~")

	tests := []struct {
		r string       // input request path
		h http.Handler // output matched handler
		k []string     // output param keys
		v []string     // output param values
	}{
		{r: "/articles", h: nil, k: []string{}, v: []string{}},
		{r: "/articles/12345", h: hStub7, k: []string{"rid"}, v: []string{"12345"}},
		{r: "/articles/123", h: hStub1, k: []string{"id"}, v: []string{"123"}},
		{r: "/articles/how-to-build-a-router", h: hStub2, k: []string{"slug"}, v: []string{"how-to-build-a-router"}},
		{r: "/articles/0456", h: hStub3, k: []string{"zid"}, v: []string{"0456"}},
		{r: "/articles/@pk/posts", h: hStub4, k: []string{"name"}, v: []string{"@pk"}},
		{r: "/articles/1/run", h: hStub5, k: []string{"op"}, v: []string{"1"}},
		{r: "/articles/1122", h: hStub1, k: []string{"id"}, v: []string{"1122"}},
		{r: "/articles/1122-yes", h: hStub6, k: []string{"id", "aux"}, v: []string{"1122", "yes"}},
	}

	for i, tt := range tests {
		rctx := NewRouteContext()

		_, handlers, _ := tr.FindRoute(rctx, mGET, tt.r)

		var handler http.Handler
		if methodHandler, ok := handlers[mGET]; ok {
			handler = methodHandler.handler
		}

		paramKeys := rctx.routeParams.Keys
		paramValues := rctx.routeParams.Values

		if fmt.Sprintf("%v", tt.h) != fmt.Sprintf("%v", handler) {
			t.Errorf("input [%d]: find '%s' expecting handler:%v , got:%v", i, tt.r, tt.h, handler)
		}
		if !stringSliceEqual(tt.k, paramKeys) {
			t.Errorf("input [%d]: find '%s' expecting paramKeys:(%d)%v , got:(%d)%v", i, tt.r, len(tt.k), tt.k, len(paramKeys), paramKeys)
		}
		if !stringSliceEqual(tt.v, paramValues) {
			t.Errorf("input [%d]: find '%s' expecting paramValues:(%d)%v , got:(%d)%v", i, tt.r, len(tt.v), tt.v, len(paramValues), paramValues)
		}
	}
}

func (c *ClusterClient) cmdsInfo(ctx context.Context) (map[string]*CommandInfo, error) {
	// Try 3 random nodes.
	const nodeLimit = 3

	addrs, err := c.nodes.Addrs()
	if err != nil {
		return nil, err
	}

	var firstErr error

	perm := rand.Perm(len(addrs))
	if len(perm) > nodeLimit {
		perm = perm[:nodeLimit]
	}

	for _, idx := range perm {
		addr := addrs[idx]

		node, err := c.nodes.GetOrCreate(addr)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}

		info, err := node.Client.Command(ctx).Result()
		if err == nil {
			return info, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}

	if firstErr == nil {
		panic("not reached")
	}
	return nil, firstErr
}


func (s) TestServerStatsClientStreamRPCError(t *testing.T) {
	count := 1
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, callType: clientStreamRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkOutHeader,
		checkInPayload,
		checkOutTrailer,
		checkEnd,
	})
}
