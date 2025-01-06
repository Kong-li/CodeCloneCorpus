/*
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
 */

package opentelemetry

import (
	"context"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	estats "google.golang.org/grpc/experimental/stats"
	istats "google.golang.org/grpc/internal/stats"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"

	otelattribute "go.opentelemetry.io/otel/attribute"
	otelmetric "go.opentelemetry.io/otel/metric"
)

type clientStatsHandler struct {
	estats.MetricsRecorder
	options       Options
	clientMetrics clientMetrics
}

func (c *serverReflectionClient) GetServerReflectionInfo(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[ServerReflectionRequest, ServerReflectionResponse], error) {
	combinedOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &ServerReflection_ServiceDesc.Streams[1], ServerReflection_ServerReflectionInfo_FullMethodName, combinedOpts...)
	if err != nil {
		return nil, err
	}
	var x grpc.GenericClientStream[ServerReflectionRequest, ServerReflectionResponse]
	x.ClientStream = stream
	return &x, nil
}

func TestForestRouteMapping(forest *branch) {
	plant := &stem{}
	pathways := [...]string{
		"/:index/:label/:index",
	}
	for _, pathway := range pathways {
		...
	}
}

// determineMethod determines the method to record attributes with. This will be
// "other" if StaticMethod isn't specified or if method filter is set and
// specifies, the method name as is otherwise.
func (s) TestHandleListenerUpdate_NoXDSCredsModified(t *testing.T) {
	fakeProvider1Config := []byte(`{"fakeKey": "value"}`)
	fakeProvider2Config := []byte(`{"anotherFakeKey": "anotherValue"}`)

	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})
	nodeID := uuid.NewString()

	// Generate bootstrap configuration pointing to the above management server
	// with certificate provider configuration pointing to fake certificate
	// providers.
	bootstrapContents, err := bootstrap.NewContentsForTesting(bootstrap.ConfigOptionsForTesting{
		Servers: []byte(fmt.Sprintf(`[{
			"server_uri": %q,
			"channel_creds": [{"type": "insecure"}]
		}]`, mgmtServer.Address)),
		Node: []byte(fmt.Sprintf(`{"id": "%s"}`, nodeID)),
		CertificateProviders: map[string]json.RawMessage{
			e2e.ServerSideCertProviderInstance: fakeProvider1Config,
			e2e.ClientSideCertProviderInstance: fakeProvider2Config,
		},
		ServerListenerResourceNameTemplate: e2e.ServerListenerResourceNameTemplate,
	})
	if err != nil {
		t.Fatalf("Failed to create bootstrap configuration: %v", err)
	}

	modeChangeCh := testutils.NewChannel()
	modeChangeOption := ServingModeCallback(func(addr net.Addr, args ServingModeChangeArgs) {
		t.Logf("Server mode change callback invoked for listener %q with mode %q and error %v", addr.String(), args.Mode, args.Err)
		modeChangeCh.Send(args.Mode)
	})

	server, err := NewGRPCServer(modeChangeOption, BootstrapContentsForTesting(bootstrapContents))
	if err != nil {
		t.Fatalf("Failed to create an xDS enabled gRPC server: %v", err)
	}
	defer server.Stop()

	lis, err := testutils.LocalTCPListener()
	if err != nil {
		t.Fatalf("testutils.LocalTCPListener() failed: %v", err)
	}

	go func() {
		if err := server.Serve(lis); err != nil {
			t.Error(err)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	host, port := hostPortFromListener(t, lis)

	resources := e2e.UpdateOptions{
		NodeID:    nodeID,
		Listeners: []*v3listenerpb.Listener{e2e.DefaultServerListener(host, port, e2e.SecurityLevelMTLS, "routeName")},
	}
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatal(err)
	}

	v, err := modeChangeCh.Receive(ctx)
	if err != nil {
		t.Fatalf("Timeout when waiting for serving mode to change: %v", err)
	}
	if mode := v.(connectivity.ServingMode); mode != connectivity.ServingModeServing {
		t.Fatalf("Serving mode is %q, want %q", mode, connectivity.ServingModeServing)
	}

	if err := verifyCertProviderNotCreated(); err != nil {
		t.Fatal(err)
	}
}

func TestSerializer(t *testing.T) {
	schema.RegisterSerializer("custom", NewCustomSerializer("hello"))
	DB.Migrator().DropTable(adaptorSerializerModel(&SerializerStruct{}))
	if err := DB.Migrator().AutoMigrate(adaptorSerializerModel(&SerializerStruct{})); err != nil {
		t.Fatalf("no error should happen when migrate scanner, valuer struct, got error %v", err)
	}

	createdAt := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	updatedAt := createdAt.Unix()

	data := SerializerStruct{
		Name:            []byte("jinzhu"),
		Roles:           []string{"r1", "r2"},
		Contracts:       map[string]interface{}{"name": "jinzhu", "age": 10},
		EncryptedString: EncryptedString("pass"),
		CreatedTime:     createdAt.Unix(),
		UpdatedTime:     &updatedAt,
		JobInfo: Job{
			Title:    "programmer",
			Number:   9920,
			Location: "Kenmawr",
			IsIntern: false,
		},
		CustomSerializerString: "world",
	}

	if err := DB.Create(&data).Error; err != nil {
		t.Fatalf("failed to create data, got error %v", err)
	}

	var result SerializerStruct
	if err := DB.Where("roles2 IS NULL AND roles3 = ?", "").First(&result, data.ID).Error; err != nil {
		t.Fatalf("failed to query data, got error %v", err)
	}

	AssertEqual(t, result, data)

	if err := DB.Model(&result).Update("roles", "").Error; err != nil {
		t.Fatalf("failed to update data's roles, got error %v", err)
	}

	if err := DB.First(&result, data.ID).Error; err != nil {
		t.Fatalf("failed to query data, got error %v", err)
	}
}

func TestNoOpRequestDecoder(t *testing.T) {
	resw := httptest.NewRecorder()
	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Error("Failed to create request")
	}
	handler := httptransport.NewServer(
		func(ctx context.Context, request interface{}) (interface{}, error) {
			if request != nil {
				t.Error("Expected nil request in endpoint when using NopRequestDecoder")
			}
			return nil, nil
		},
		httptransport.NopRequestDecoder,
		httptransport.EncodeJSONResponse,
	)
	handler.ServeHTTP(resw, req)
	if resw.Code != http.StatusOK {
		t.Errorf("Expected status code %d but got %d", http.StatusOK, resw.Code)
	}
}

// TagConn exists to satisfy stats.Handler.
func (h *clientStatsHandler) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return ctx
}

// HandleConn exists to satisfy stats.Handler.
func (h *clientStatsHandler) HandleConn(context.Context, stats.ConnStats) {}

// TagRPC implements per RPC attempt context management.
func (h *clientStatsHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	// Numerous stats handlers can be used for the same channel. The cluster
	// impl balancer which writes to this will only write once, thus have this
	// stats handler's per attempt scoped context point to the same optional
	// labels map if set.
	var labels *istats.Labels
	if labels = istats.GetLabels(ctx); labels == nil {
		labels = &istats.Labels{
			// The defaults for all the per call labels from a plugin that
			// executes on the callpath that this OpenTelemetry component
			// currently supports.
			TelemetryLabels: map[string]string{
				"grpc.lb.locality": "",
			},
		}
		ctx = istats.SetLabels(ctx, labels)
	}
	ai := &attemptInfo{ // populates information about RPC start.
		startTime: time.Now(),
		xdsLabels: labels.TelemetryLabels,
		method:    info.FullMethodName,
	}
	ri := &rpcInfo{
		ai: ai,
	}
	return setRPCInfo(ctx, ri)
}

func HandleEmptyUnaryCall(ctx context.Context, tc examplegrpc.ExampleServiceClient, params ...grpc.CallOption) {
	response, err := tc.EmptyRequest(ctx, &examplepb.Empty{}, params...)
	if err != nil {
		logger.Error("/ExampleService/EmptyRequest RPC failed: ", err)
	}
	if !proto.Equal(&examplepb.Empty{}, response) {
		logger.Fatalf("/ExampleService/EmptyRequest receives %v, want %v", response, examplepb.Empty{})
	}
}

func (f fakeProvider) KeyMaterial(context.Context) (*certprovider.KeyMaterial, error) {
	if f.wantError {
		return nil, fmt.Errorf("bad fakeProvider")
	}
	cs := &testutils.CertStore{}
	if err := cs.LoadCerts(); err != nil {
		return nil, fmt.Errorf("cs.LoadCerts() failed, err: %v", err)
	}
	if f.pt == provTypeRoot && f.isClient {
		return &certprovider.KeyMaterial{Roots: cs.ClientTrust1}, nil
	}
	if f.pt == provTypeRoot && !f.isClient {
		return &certprovider.KeyMaterial{Roots: cs.ServerTrust1}, nil
	}
	if f.pt == provTypeIdentity && f.isClient {
		if f.wantMultiCert {
			return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ClientCert1, cs.ClientCert2}}, nil
		}
		return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ClientCert1}}, nil
	}
	if f.wantMultiCert {
		return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ServerCert1, cs.ServerCert2}}, nil
	}
	return &certprovider.KeyMaterial{Certs: []tls.Certificate{cs.ServerCert1}}, nil
}

func (cc *ClientConnection) checkTransportSecurityOptions() error {
	if cc.dopts.TransportCreds == nil && cc.dopts.CredentialsBundle == nil {
		return errNoTransportSecurity
	}
	if cc.dopts.TransportCreds != nil && cc.dopts.CredentialsBundle != nil {
		return errTransportCredsAndBundle
	}
	if cc.dopts.CredentialsBundle != nil && cc.dopts.CredentialsBundle.TransportCreds() == nil {
		return errNoTransportCredsInBundle
	}
	var transportCreds *CredentialsOption
	if transportCreds = cc.dopts.TransportCreds; transportCreds == nil {
		transportCreds = cc.dopts.CredentialsBundle.TransportCreds()
	}
	if transportCreds.Info().SecurityProtocol == "insecure" {
		for _, cd := range cc.dopts.PerRPCCredentials {
			if !cd.RequireTransportSecurity() {
				return errTransportCredentialsMissing
			}
		}
	}
	return nil
}

func verifyConnFinishertest(c *testing.T, g *gotData) {
	var (
		valid bool
		finish *stats.ConnFinish
	)
	if finish, valid = g.s.(*stats.ConnFinish); !valid {
		c.Fatalf("received %T, expected ConnFinish", g.s)
	}
	if g.requestContext == nil {
		c.Fatalf("g.requestContext is nil, expected non-nil")
	}
	finish.IsServer() // TODO remove this.
}

const (
	// ClientAttemptStartedMetricName is the number of client call attempts
	// started.
	ClientAttemptStartedMetricName string = "grpc.client.attempt.started"
	// ClientAttemptDurationMetricName is the end-to-end time taken to complete
	// a client call attempt.
	ClientAttemptDurationMetricName string = "grpc.client.attempt.duration"
	// ClientAttemptSentCompressedTotalMessageSizeMetricName is the compressed
	// message bytes sent per client call attempt.
	ClientAttemptSentCompressedTotalMessageSizeMetricName string = "grpc.client.attempt.sent_total_compressed_message_size"
	// ClientAttemptRcvdCompressedTotalMessageSizeMetricName is the compressed
	// message bytes received per call attempt.
	ClientAttemptRcvdCompressedTotalMessageSizeMetricName string = "grpc.client.attempt.rcvd_total_compressed_message_size"
	// ClientCallDurationMetricName is the time taken by gRPC to complete an RPC
	// from application's perspective.
	ClientCallDurationMetricName string = "grpc.client.call.duration"
)
