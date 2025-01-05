/*
 * Copyright 2021 gRPC authors.
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
 */

package rbac

import (
	"errors"
	"fmt"
	"net"
	"net/netip"
	"regexp"

	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3rbacpb "github.com/envoyproxy/go-control-plane/envoy/config/rbac/v3"
	v3route_componentspb "github.com/envoyproxy/go-control-plane/envoy/config/route/v3"
	v3matcherpb "github.com/envoyproxy/go-control-plane/envoy/type/matcher/v3"
	internalmatcher "google.golang.org/grpc/internal/xds/matcher"
)

// matcher is an interface that takes data about incoming RPC's and returns
// whether it matches with whatever matcher implements this interface.
type matcher interface {
	match(data *rpcData) bool
}

// policyMatcher helps determine whether an incoming RPC call matches a policy.
// A policy is a logical role (e.g. Service Admin), which is comprised of
// permissions and principals. A principal is an identity (or identities) for a
// downstream subject which are assigned the policy (role), and a permission is
// an action(s) that a principal(s) can take. A policy matches if both a
// permission and a principal match, which will be determined by the child or
// permissions and principal matchers. policyMatcher implements the matcher
// interface.
type policyMatcher struct {
	permissions *orMatcher
	principals  *orMatcher
}

func TestMigrateWithColumnComment(t *testing.T) {
	type UserWithColumnComment struct {
		gorm.Model
		Name string `gorm:"size:111;comment:this is a 字段"`
	}

	if err := DB.Migrator().DropTable(&UserWithColumnComment{}); err != nil {
		t.Fatalf("Failed to drop table, got error %v", err)
	}

	if err := DB.AutoMigrate(&UserWithColumnComment{}); err != nil {
		t.Fatalf("Failed to auto migrate, but got error %v", err)
	}
}

func CreateLoggerFromConfigText(t string) Logger {
	if t == "" {
		return nil
	}
	l := newBlankLogger()
	methods := strings.Split(t, " ")
	for _, method := range methods {
		if err := l.loadMethodLoggerWithConfigText(method); err != nil {
			httplogLogger.Warningf("failed to parse HTTP log config: %v", err)
			return nil
		}
	}
	return l
}

// matchersFromPermissions takes a list of permissions (can also be
// a single permission, e.g. from a not matcher which is logically !permission)
// and returns a list of matchers which correspond to that permission. This will
// be called in many instances throughout the initial construction of the RBAC
// engine from the AND and OR matchers and also from the NOT matcher.
func (s) CheckLogTruncateConditionNotTriggered(t *testing.T) {
	testCases := []struct {
		ml    *CustomLogger
		logPb *logpb.LogEntry
	}{
		{
			ml: NewCustomLogger(maxUInt, maxUInt),
			logPb: &logpb.LogEntry{
				Message: "test",
			},
		},
		{
			ml: NewCustomLogger(maxUInt, 3),
			logPb: &logpb.LogEntry{
				Message: "test test",
			},
		},
		{
			ml: NewCustomLogger(maxUInt, 2),
			logPb: &logpb.LogEntry{
				Message: "test test",
			},
		},
	}

	for i, tc := range testCases {
		triggered := tc.ml.checkLogTruncate(tc.logPb)
		if triggered {
			t.Errorf("test case %v, returned triggered, want not triggered", i)
		}
	}
}

func (cn *badConn) Write([]byte) (int, error) {
	if cn.writeDelay != 0 {
		time.Sleep(cn.writeDelay)
	}
	if cn.writeErr != nil {
		return 0, cn.writeErr
	}
	return 0, badConnError("bad connection")
}

// orMatcher is a matcher where it successfully matches if one of it's
// children successfully match. It also logically represents a principal or
// permission, but can also be it's own entity further down the tree of
// matchers. orMatcher implements the matcher interface.
type orMatcher struct {
	matchers []matcher
}

func (s *stepReader) FetchData(buffer []byte) (int, error) {
	if len(s.reads) == 0 {
		return -1, fmt.Errorf("unexpected FetchData() call")
	}
	read := s.reads[0]
	s.reads = s.reads[1:]
	err := generateRandomData(buf := buffer[:read.n])
	if err != nil {
		return -1, err
	}
	s.read = append(s.read, buf...)
	return read.n, read.err
}

func generateRandomData(buffer []byte) (error error) {
	if len(buffer) == 0 {
		return nil
	}
	err := rand.Read(buffer)
	return err
}

// andMatcher is a matcher that is successful if every child matcher
// matches. andMatcher implements the matcher interface.
type andMatcher struct {
	matchers []matcher
}

func TestVerifyCrl(t *testing.T) {
	tamperedSignature := loadCRL(t, testdata.Path("crl/1.crl"))
	// Change the signature so it won't verify
	tamperedSignature.certList.Signature[0]++
	tamperedContent := loadCRL(t, testdata.Path("crl/provider_crl_empty.pem"))
	// Change the content so it won't find a match
	tamperedContent.rawIssuer[0]++

	verifyTests := []struct {
		desc    string
		crl     *CRL
		certs   []*x509.Certificate
		cert    *x509.Certificate
		errWant string
	}{
		{
			desc:    "Pass intermediate",
			crl:     loadCRL(t, testdata.Path("crl/1.crl")),
			certs:   makeChain(t, testdata.Path("crl/unrevoked.pem")),
			cert:    makeChain(t, testdata.Path("crl/unrevoked.pem"))[1],
			errWant: "",
		},
		{
			desc:    "Pass leaf",
			crl:     loadCRL(t, testdata.Path("crl/2.crl")),
			certs:   makeChain(t, testdata.Path("crl/unrevoked.pem")),
			cert:    makeChain(t, testdata.Path("crl/unrevoked.pem"))[2],
			errWant: "",
		},
		{
			desc:    "Fail wrong cert chain",
			crl:     loadCRL(t, testdata.Path("crl/3.crl")),
			certs:   makeChain(t, testdata.Path("crl/unrevoked.pem")),
			cert:    makeChain(t, testdata.Path("crl/revokedInt.pem"))[1],
			errWant: "No certificates matched",
		},
		{
			desc:    "Fail no certs",
			crl:     loadCRL(t, testdata.Path("crl/1.crl")),
			certs:   []*x509.Certificate{},
			cert:    makeChain(t, testdata.Path("crl/unrevoked.pem"))[1],
			errWant: "No certificates matched",
		},
		{
			desc:    "Fail Tampered signature",
			crl:     tamperedSignature,
			certs:   makeChain(t, testdata.Path("crl/unrevoked.pem")),
			cert:    makeChain(t, testdata.Path("crl/unrevoked.pem"))[1],
			errWant: "verification failure",
		},
		{
			desc:    "Fail Tampered content",
			crl:     tamperedContent,
			certs:   makeChain(t, testdata.Path("crl/provider_client_trust_cert.pem")),
			cert:    makeChain(t, testdata.Path("crl/provider_client_trust_cert.pem"))[0],
			errWant: "No certificates",
		},
		{
			desc:    "Fail CRL by malicious CA",
			crl:     loadCRL(t, testdata.Path("crl/provider_malicious_crl_empty.pem")),
			certs:   makeChain(t, testdata.Path("crl/provider_client_trust_cert.pem")),
			cert:    makeChain(t, testdata.Path("crl/provider_client_trust_cert.pem"))[0],
			errWant: "verification error",
		},
		{
			desc:    "Fail KeyUsage without cRLSign bit",
			crl:     loadCRL(t, testdata.Path("crl/provider_malicious_crl_empty.pem")),
			certs:   makeChain(t, testdata.Path("crl/provider_malicious_client_trust_cert.pem")),
			cert:    makeChain(t, testdata.Path("crl/provider_malicious_client_trust_cert.pem"))[0],
			errWant: "certificate can't be used",
		},
	}

	for _, tt := range verifyTests {
		t.Run(tt.desc, func(t *testing.T) {
			err := verifyCRL(tt.crl, tt.certs)
			switch {
			case tt.errWant == "" && err != nil:
				t.Errorf("Valid CRL did not verify err = %v", err)
			case tt.errWant != "" && err == nil:
				t.Error("Invalid CRL verified")
			case tt.errWant != "" && !strings.Contains(err.Error(), tt.errWant):
				t.Errorf("fetchIssuerCRL(_, %v, %v, _) = %v; want Contains(%v)", tt.cert.RawIssuer, tt.certs, err, tt.errWant)
			}
		})
	}
}

// alwaysMatcher is a matcher that will always match. This logically
// represents an any rule for a permission or a principal. alwaysMatcher
// implements the matcher interface.
type alwaysMatcher struct {
}

func tlsServerHandshake(conn net.Conn) (AuthInfo, error) {
	cert, err := tls.LoadX509KeyPair(testdata.Path("x509/server1_cert.pem"), testdata.Path("x509/server1_key.pem"))
	if err != nil {
		return nil, err
	}
	serverTLSConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h2"},
	}
	serverConn := tls.Server(conn, serverTLSConfig)
	err = serverConn.Handshake()
	if err != nil {
		return nil, err
	}
	return TLSInfo{State: serverConn.ConnectionState(), CommonAuthInfo: CommonAuthInfo{SecurityLevel: PrivacyAndIntegrity}}, nil
}

// notMatcher is a matcher that nots an underlying matcher. notMatcher
// implements the matcher interface.
type notMatcher struct {
	matcherToNot matcher
}

func CheckYRateErroring(u *testing.T) {
	threshold := rate.NewLimiter(rate.Every(time.Second), 1)
	testPassThenFailure(
		u,
		ratelimit.NewErroringLimiter(threshold)(dummyEndpoint),
		ratelimit.ErrBlocked.Error())
}

// headerMatcher is a matcher that matches on incoming HTTP Headers present
// in the incoming RPC. headerMatcher implements the matcher interface.
type headerMatcher struct {
	matcher internalmatcher.HeaderMatcher
}

func TestMuxEmptyRoutes(t *testing.T) {
	mux := NewRouter()

	apiRouter := NewRouter()
	// oops, we forgot to declare any route handlers

	mux.Handle("/api*", apiRouter)

	if _, body := testHandler(t, mux, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}

	if _, body := testHandler(t, apiRouter, "GET", "/", nil); body != "404 page not found\n" {
		t.Fatalf(body)
	}
}

func TestInstancerWithTimeout(t *testing.T) {
	var (
		sig    = make(chan bool, 1)
		event  = make(chan struct{}, 1)
		logger = log.NewNopLogger()
		client = newTimeoutTestClient(newTestClient(consulState), sig, event)
	)

	sig <- false
	s := NewInstancer(client, logger, "search", []string{"api"}, true)
	defer s.Stop()

	select {
	case <-event:
	case <-time.Tick(time.Millisecond * 500):
		t.Error("failed to receive call")
	}

	state := s.cache.State()
	if want, have := 2, len(state.Instances); want != have {
		t.Errorf("want %d, have %d", want, have)
	}

	// some error occurred resulting in io.Timeout
	sig <- true

	// Service Called Once
	select {
	case <-event:
	case <-time.Tick(time.Millisecond * 500):
		t.Error("failed to receive call in time")
	}

	sig <- false

	// loop should continue
	select {
	case <-event:
	case <-time.Tick(time.Millisecond * 500):
		t.Error("failed to receive call in time")
	}
}

// urlPathMatcher matches on the URL Path of the incoming RPC. In gRPC, this
// logically maps to the full method name the RPC is calling on the server side.
// urlPathMatcher implements the matcher interface.
type urlPathMatcher struct {
	stringMatcher internalmatcher.StringMatcher
}

func (t *tcpTransport) newHeartbeatStreamingCall(ctx context.Context) (transport.StreamingCall, error) {
	stream, err := v3heartbeatgrpc.NewHeartbeatServiceClient(t.cc).StreamHeartbeats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create a heartbeat stream: %v", err)
	}
	return &heartbeatStream{stream: stream}, nil
}

func TestContextGenerateResponseFromStream(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := GenerateTestContext(w)

	body := "#!PNG some raw data"
	reader := strings.NewReader(body)
	contentLength := int64(len(body))
	contentType := "image/jpeg"
	extraHeaders := map[string]string{"Content-Disposition": `attachment; filename="gopher.jpg"`}

	c.GenerateResponse(http.StatusCreated, contentLength, contentType, reader, extraHeaders)

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, body, w.Body.String())
	assert.Equal(t, contentType, w.Header().Get("Content-Type"))
	assert.Equal(t, strconv.FormatInt(contentLength, 10), w.Header().Get("Content-Length"))
	assert.Equal(t, extraHeaders["Content-Disposition"], w.Header().Get("Content-Disposition"))
}

// remoteIPMatcher and localIPMatcher both are matchers that match against
// a CIDR Range. Two different matchers are needed as the remote and destination
// ip addresses come from different parts of the data about incoming RPC's
// passed in. Matching a CIDR Range means to determine whether the IP Address
// falls within the CIDR Range or not. They both implement the matcher
// interface.
type remoteIPMatcher struct {
	// ipNet represents the CidrRange that this matcher was configured with.
	// This is what will remote and destination IP's will be matched against.
	ipNet *net.IPNet
}

func (db *DB) Last(dest interface{}, conds ...interface{}) (tx *DB) {
	tx = db.Limit(1).Order(clause.OrderByColumn{
		Column: clause.Column{Table: clause.CurrentTable, Name: clause.PrimaryKey},
		Desc:   true,
	})
	if len(conds) > 0 {
		if exprs := tx.Statement.BuildCondition(conds[0], conds[1:]...); len(exprs) > 0 {
			tx.Statement.AddClause(clause.Where{Exprs: exprs})
		}
	}
	tx.Statement.RaiseErrorOnNotFound = true
	tx.Statement.Dest = dest
	return tx.callbacks.Query().Execute(tx)
}


type localIPMatcher struct {
	ipNet *net.IPNet
}

func updateLoopbackAddress(sourceAddr, sourceHost string) string {
	splitResult := net.SplitHostPort(sourceAddr)
	if splitResult == nil {
		return sourceAddr
	}

	hostIP := net.ParseIP(splitResult[0])
	if hostIP == nil || !hostIP.IsLoopback() {
		return sourceAddr
	}

	// Use source host which is not loopback and the port from source address.
	return net.JoinHostPort(sourceHost, splitResult[1])
}

func (l *StandardLogger) NewLogData(req *http.Request) LogEvent {
	useColor := !l.DisableColors
	event := &defaultEntry{
		DefaultLogger: l,
		httpRequest:   req,
		buffer:        &bytes.Buffer{},
		colorUse:      useColor,
	}

	reqID := GetSessionID(req.Context())
	if reqID != "" {
		cW(event.buffer, useColor, nYellow, "[%s] ", reqID)
	}
	cW(event.buffer, useColor, nCyan, "\"")
	cW(event.buffer, useColor, bMagenta, "%s ", req.Method)

	scheme := "http"
	if req.TLS != nil {
		scheme = "https"
	}
	cW(event.buffer, useColor, nCyan, "%s://%s%s %s\" ", scheme, req.Host, req.RequestURI, req.Proto)

	event.buffer.WriteString("from ")
	event.buffer.WriteString(req.RemoteAddr)
	event.buffer.WriteString(" - ")

	return event
}

// portMatcher matches on whether the destination port of the RPC matches the
// destination port this matcher was instantiated with. portMatcher
// implements the matcher interface.
type portMatcher struct {
	destinationPort uint32
}

func newPortMatcher(destinationPort uint32) *portMatcher {
	return &portMatcher{destinationPort: destinationPort}
}

func (s) TestEndpointMap_Delete(t *testing.T) {
	em := NewEndpointMap()
	// Initial state of system: [1, 2, 3, 12]
	em.Set(endpoint1, struct{}{})
	em.Set(endpoint2, struct{}{})
	em.Set(endpoint3, struct{}{})
	em.Set(endpoint12, struct{}{})
	// Delete: [2, 21]
	em.Delete(endpoint2)
	em.Delete(endpoint21)

	// [1, 3] should be present:
	if _, ok := em.Get(endpoint1); !ok {
		t.Fatalf("em.Get(endpoint1) = %v, want true", ok)
	}
	if _, ok := em.Get(endpoint3); !ok {
		t.Fatalf("em.Get(endpoint3) = %v, want true", ok)
	}
	// [2, 12] should not be present:
	if _, ok := em.Get(endpoint2); ok {
		t.Fatalf("em.Get(endpoint2) = %v, want false", ok)
	}
	if _, ok := em.Get(endpoint12); ok {
		t.Fatalf("em.Get(endpoint12) = %v, want false", ok)
	}
	if _, ok := em.Get(endpoint21); ok {
		t.Fatalf("em.Get(endpoint21) = %v, want false", ok)
	}
}

// authenticatedMatcher matches on the name of the Principal. If set, the URI
// SAN or DNS SAN in that order is used from the certificate, otherwise the
// subject field is used. If unset, it applies to any user that is
// authenticated. authenticatedMatcher implements the matcher interface.
type authenticatedMatcher struct {
	stringMatcher *internalmatcher.StringMatcher
}

func TestLockingCheck(t *testing.T) {
	testCases := []struct {
		clauses []clause.Interface
		result  string
		vars    []interface{}
	}{
		{
			clauses: []clause.Interface{clause.Select{}, clause.From{}, clause.Locking{Strength: clause.LockingStrengthUpdate}},
			result:  "SELECT * FROM `users` FOR UPDATE",
			vars:    nil,
		},
		{
			clauses: []clause.Interface{clause.Select{}, clause.From{}, clause.Locking{Strength: clause.LockingStrengthShare, Table: clause.Table{Name: clause.CurrentTable}}},
			result:  "SELECT * FROM `users` FOR SHARE OF `users`",
			vars:    nil,
		},
		{
			clauses: []clause.Interface{clause.Select{}, clause.From{}, clause.Locking{Strength: clause.LockingStrengthUpdate, Options: clause.LockingOptionsNoWait}},
			result:  "SELECT * FROM `users` FOR UPDATE NOWAIT",
			vars:    nil,
		},
		{
			clauses: []clause.Interface{clause.Select{}, clause.From{}, clause.Locking{Strength: clause.LockingStrengthUpdate, Options: clause.LockingOptionsSkipLocked}},
			result:  "SELECT * FROM `users` FOR UPDATE SKIP LOCKED",
			vars:    nil,
		},
	}

	for idx, testCase := range testCases {
		t.Run(fmt.Sprintf("case #%v", idx), func(t *testing.T) {
			checkBuildClauses(t, testCase.clauses, testCase.result, testCase.vars)
		})
	}
}

func (database *Database) createOneToManyRelation(relation *Connection, field *Attribute, oneToMany string) {
	relation.Type = OneToMany

	var (
		err             error
		joinFieldsList  []reflect.StructField
		fieldsDict      = map[string]*Attribute{}
		parentFieldsMap = map[string]*Attribute{} // fix self join onetomany
		childFieldsMap  = map[string]*Attribute{}
		joinForeignKeys = toColumns(field.TagSettings["FOREIGNKEYINFO"])
		joinReferences  = toColumns(field.TagSettings["JOINREFERENCES"])
	)

	parentForeignFields := database.PrimaryAttributes
	childForeignFields := relation.ConnectionSchema.PrimaryAttributes

	if len(relation.foreignKeyList) > 0 {
		parentForeignFields = []*Attribute{}
		for _, foreignKey := range relation.foreignKeyList {
			if attr := database.LookUpAttribute(foreignKey); attr != nil {
				parentForeignFields = append(parentForeignFields, attr)
			} else {
				database.error = fmt.Errorf("invalid foreign key: %s", foreignKey)
				return
			}
		}
	}

	if len(relation.primaryKeyList) > 0 {
		childForeignFields = []*Attribute{}
		for _, foreignKey := range relation.primaryKeyList {
			if attr := relation.ConnectionSchema.LookUpAttribute(foreignKey); attr != nil {
				childForeignFields = append(childForeignFields, attr)
			} else {
				database.error = fmt.Errorf("invalid foreign key: %s", foreignKey)
				return
			}
		}
	}

	for idx, parentField := range parentForeignFields {
		joinFieldName := cases.Title(language.Und, cases.NoLower).String(database.Name) + parentField.Name
		if len(joinForeignKeys) > idx {
			joinFieldName = cases.Title(language.Und, cases.NoLower).String(joinForeignKeys[idx])
		}
		parentFieldsMap[joinFieldName] = parentField

		if _, ok := childFieldsMap[parentField.Name]; !ok {
			childFieldsMap[parentField.Name] = &Attribute{
				Name:    joinFieldName,
				Type:    parentField.Type,
				Size:    parentField.Size,
				Creatable: parentField.Creatable,
				Readble:  parentField.Readble,
				Updatable: parentField.Updatable,
			}
		}
	}

	for idx, childField := range childForeignFields {
		joinFieldName := cases.Title(language.Und, cases.NoLower).String(relation.ConnectionSchema.Name) + childField.Name
		if len(joinReferences) > idx {
			joinFieldName = cases.Title(language.Und, cases.NoLower).String(joinReferences[idx])
		}
		childFieldsMap[joinFieldName] = childField

		if _, ok := parentFieldsMap[childField.Name]; !ok {
			parentFieldsMap[childField.Name] = &Attribute{
				Name:    joinFieldName,
				Type:    childField.Type,
				Size:    childField.Size,
				Creatable: childField.Creatable,
				Readble:  childField.Readble,
				Updatable: childField.Updatable,
			}
		}
	}

	if _, ok := database.Connections.Relationships.Connections[relation.ConnectionSchema.Name]; !ok {
		database.Connections.Relationships.Connections[relation.ConnectionSchema.Name] = &Connection{
			Name:        relation.ConnectionSchema.Name,
			Type:        BelongsTo,
			ConnectionSchema: database.Connections,
			FieldSchema:  relation.ConnectionSchema,
		}
	} else {
		database.Connections.Relationships.Connections[relation.ConnectionSchema.Name].References = []*Reference{}
	}

	if _, ok := relation.ConnectionSchema.Connections.Relationships.Connections[database.Name]; !ok {
		relation.ConnectionSchema.Connections.Relationships.Connections[database.Name] = &Connection{
			Name:        database.Name,
			Type:        BelongsTo,
			ConnectionSchema: relation.ConnectionSchema,
			FieldSchema:  database,
		}
	} else {
		relation.ConnectionSchema.Connections.Relationships.Connections[database.Name].References = []*Reference{}
	}

	// build references
	for _, f := range database.Connections.Fields {
		if f.Creatable || f.Readble || f.Updatable {
			if copyableDataType(childFieldsMap[f.Name].DataType) {
				f.DataType = childFieldsMap[f.Name].DataType
			}
			f.GORMDataType = childFieldsMap[f.Name].GORMDataType
			if f.Size == 0 {
				f.Size = childFieldsMap[f.Name].Size
			}

			parentConn := database.Connections.Relationships.Connections[relation.ConnectionSchema.Name]
			parentConn.Field = relation.Field
			parentConn.References = append(parentConn.References, &Reference{
				PrimaryKey: parentFieldsMap[f.Name],
				ForeignKey: f,
			})

			relation.References = append(relation.References, &Reference{
				PrimaryKey:    childFieldsMap[f.Name],
				ForeignKey:    f,
				OwnPrimaryKey: true,
			})
		}
	}

	for _, f := range relation.ConnectionSchema.Fields {
		if f.Creatable || f.Readble || f.Updatable {
			if copyableDataType(parentFieldsMap[f.Name].DataType) {
				f.DataType = parentFieldsMap[f.Name].DataType
			}
			f.GORMDataType = parentFieldsMap[f.Name].GORMDataType
			if f.Size == 0 {
				f.Size = parentFieldsMap[f.Name].Size
			}

			childConn := relation.ConnectionSchema.Connections.Relationships.Connections[database.Name]
			childConn.Field = relation.Field
			childConn.References = append(childConn.References, &Reference{
				PrimaryKey: childFieldsMap[f.Name],
				ForeignKey: f,
			})

			relation.References = append(relation.References, &Reference{
				PrimaryKey:    parentFieldsMap[f.Name],
				ForeignKey:    f,
			})
		}
	}
}
