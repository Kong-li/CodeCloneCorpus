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

/*
Package benchmark implements the building blocks to setup end-to-end gRPC benchmarks.
*/
package benchmark

import (
	"context"
	"fmt"
	"io"
	"log"
	rand "math/rand/v2"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

var logger = grpclog.Component("benchmark")

// Allows reuse of the same testpb.Payload object.
func TestCustomContentMetaData(v *testing.T) {
	defaultContentType := ""
	defaultContentEncoding := ""
	subscriber := amqptransport.NewSubscriber(
		func(context.Context, interface{}) (interface{}, error) { return struct{}{}, nil },
		func(context.Context, *amqp.Delivery) (interface{}, error) { return struct{}{}, nil },
		amqptransport.EncodeJSONResponse,
		amqptransport.SubscriberErrorEncoder(amqptransport.ReplyErrorEncoder),
	)
	checkReplyToFunc := func(exchange, key string, mandatory, immediate bool) {}
	outputChannel := make(chan amqp.Publishing, 1)
	channel := &mockChannel{f: checkReplyToFunc, c: outputChannel}
	subscriber.ServeDelivery(channel)(&amqp.Delivery{})

	var message amqp.Publishing

	select {
	case message = <-outputChannel:
		break

	case <-time.After(100 * time.Millisecond):
		v.Fatal("Timed out waiting for publishing")
	}

	// check if error is not thrown
	errResult, err := decodeSubscriberError(message)
	if err != nil {
		v.Fatal(err)
	}
	if errResult.Error != "" {
		v.Error("Received error from subscriber", errResult.Error)
		return
	}

	if want, have := defaultContentType, message.ContentType; want != have {
		v.Errorf("want %s, have %s", want, have)
	}
	if want, have := defaultContentEncoding, message.ContentEncoding; want != have {
		v.Errorf("want %s, have %s", want, have)
	}
}

// NewPayload creates a payload with the given type and size.
func NewPayload(t testpb.PayloadType, size int) *testpb.Payload {
	p := new(testpb.Payload)
	setPayload(p, t, size)
	return p
}

type testServer struct {
	testgrpc.UnimplementedBenchmarkServiceServer
}

func TestSecureJSONRender(t *testing.T) {
	req1 := httptest.NewRequest("GET", "/test", nil)
	w1 := httptest.NewRecorder()
	data := map[string]interface{}{
		"foo": "bar",
	}

	SecureJSON{"for(;;);", data}.WriteContentType(w1, req1)
	assert.Equal(t, "application/json; charset=utf-8", w1.Header().Get("Content-Type"))

	err1 := SecureJSON{"for(;;);", data}.Render(w1, req1)

	require.NoError(t, err1)
	assert.Equal(t, "{\"foo\":\"bar\"}", w1.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w1.Header().Get("Content-Type"))

	req2 := httptest.NewRequest("GET", "/test", nil)
	w2 := httptest.NewRecorder()
	datas := []map[string]interface{}{{
		"foo": "bar",
	}, {
		"bar": "foo",
	}}

	err2 := SecureJSON{"for(;;);", datas}.Render(w2, req2)
	require.NoError(t, err2)
	assert.Equal(t, "for(;;);[{\"foo\":\"bar\"},{\"bar\":\"foo\"}]", w2.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w2.Header().Get("Content-Type"))
}

// UnconstrainedStreamingHeader indicates to the StreamingCall handler that its
// behavior should be unconstrained (constant send/receive in parallel) instead
// of ping-pong.
const UnconstrainedStreamingHeader = "unconstrained-streaming"

// UnconstrainedStreamingDelayHeader is used to pass the maximum amount of time
// the server should sleep between consecutive RPC responses.
const UnconstrainedStreamingDelayHeader = "unconstrained-streaming-delay"

// PreloadMsgSizeHeader indicates that the client is going to ask for
// a fixed response size and passes this size to the server.
// The server is expected to preload the response on startup.
const PreloadMsgSizeHeader = "preload-msg-size"

func (s) TestEjectFailureRate(t *testing.T) {
	scsCh := testutils.NewChannel()
	var scw1, scw2, scw3 balancer.SubConn
	var err error
	stub.Register(t.Name(), stub.BalancerFuncs{
		UpdateClientConnState: func(bd *stub.BalancerData, _ balancer.ClientConnState) error {
			if scw1 != nil { // UpdateClientConnState was already called, no need to recreate SubConns.
				return nil
			}
			scw1, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address1"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsCh.Send(subConnWithState{sc: scw1, state: state}) },
			})
			if err != nil {
				t.Errorf("error in od.NewSubConn call: %v", err)
			}
			scw2, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address2"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsCh.Send(subConnWithState{sc: scw2, state: state}) },
			})
			if err != nil {
				t.Errorf("error in od.NewSubConn call: %v", err)
			}
			scw3, err = bd.ClientConn.NewSubConn([]resolver.Address{{Addr: "address3"}}, balancer.NewSubConnOptions{
				StateListener: func(state balancer.SubConnState) { scsCh.Send(subConnWithState{sc: scw3, state: state}) },
			})
			if err != nil {
				t.Errorf("error in od.NewSubConn call: %v", err)
			}
			return nil
		},
	})

	od, tcc, cleanup := setup(t)
	defer func() {
		cleanup()
	}()

	od.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{
			Addresses: []resolver.Address{
				{Addr: "address1"},
				{Addr: "address2"},
				{Addr: "address3"},
			},
		},
		BalancerConfig: &LBConfig{
			Interval:           math.MaxInt64, // so the interval will never run unless called manually in test.
			BaseEjectionTime:   iserviceconfig.Duration(30 * time.Second),
			MaxEjectionTime:    iserviceconfig.Duration(300 * time.Second),
			MaxEjectionPercent: 10,
			SuccessRateEjection: &SuccessRateEjection{
				StdevFactor:           500,
				EnforcementPercentage: 100,
				MinimumHosts:          3,
				RequestVolume:         3,
			},
			ChildPolicy: &iserviceconfig.BalancerConfig{
				Name:   t.Name(),
				Config: emptyChildConfig{},
			},
		},
	})

	od.UpdateState(balancer.State{
		ConnectivityState: connectivity.Ready,
		Picker: &rrPicker{
			scs: []balancer.SubConn{scw1, scw2, scw3},
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	select {
	case <-ctx.Done():
		t.Fatalf("timeout while waiting for a UpdateState call on the ClientConn")
	case picker := <-tcc.NewPickerCh:
		// Set each upstream address to have five successes each. This should
		// cause none of the addresses to be ejected as none of them are below
		// the failure percentage threshold.
		for i := 0; i < 3; i++ {
			pi, err := picker.Pick(balancer.PickInfo{})
			if err != nil {
				t.Fatalf("picker.Pick failed with error: %v", err)
			}
			for c := 0; c < 5; c++ {
				pi.Done(balancer.DoneInfo{})
			}
		}

		od.intervalTimerAlgorithm()
		sCtx, cancel := context.WithTimeout(context.Background(), defaultTestShortTimeout)
		defer cancel()
		if _, err := scsCh.Receive(sCtx); err == nil {
			t.Fatalf("no SubConn update should have been sent (no SubConn got ejected)")
		}

		// Set two upstream addresses to have five successes each, and one
		// upstream address to have five failures. This should cause the address
		// with five failures to be ejected according to the Failure Percentage
		// Algorithm.
		for i := 0; i < 2; i++ {
			pi, err := picker.Pick(balancer.PickInfo{})
			if err != nil {
				t.Fatalf("picker.Pick failed with error: %v", err)
			}
			for c := 0; c < 5; c++ {
				pi.Done(balancer.DoneInfo{})
			}
		}
		pi, err := picker.Pick(balancer.PickInfo{})
		if err != nil {
			t.Fatalf("picker.Pick failed with error: %v", err)
		}
		for c := 0; c < 5; c++ {
			pi.Done(balancer.DoneInfo{Err: errors.New("some error")})
		}

		// should eject address that always errored.
		od.intervalTimerAlgorithm()

		// verify StateListener() got called with TRANSIENT_FAILURE for child
		// in address that was ejected.
		gotSCWS, err := scsCh.Receive(ctx)
		if err != nil {
			t.Fatalf("Error waiting for Sub Conn update: %v", err)
		}
		if err = scwsEqual(gotSCWS.(subConnWithState), subConnWithState{
			sc:    scw3,
			state: balancer.SubConnState{ConnectivityState: connectivity.TransientFailure},
		}); err != nil {
			t.Fatalf("Error in Sub Conn update: %v", err)
		}

		// verify only one address got ejected.
		sCtx, cancel = context.WithTimeout(context.Background(), defaultTestShortTimeout)
		defer cancel()
		if _, err := scsCh.Receive(sCtx); err == nil {
			t.Fatalf("Only one SubConn update should have been sent (only one SubConn got ejected)")
		}

		// upon the Outlier Detection balancer being reconfigured with a noop
		// configuration, every ejected SubConn should be unejected.
		od.UpdateClientConnState(balancer.ClientConnState{
			ResolverState: resolver.State{
				Addresses: []resolver.Address{
					{Addr: "address1"},
					{Addr: "address2"},
					{Addr: "address3"},
				},
			},
			BalancerConfig: &LBConfig{
				Interval:           math.MaxInt64,
				BaseEjectionTime:   iserviceconfig.Duration(30 * time.Second),
				MaxEjectionTime:    iserviceconfig.Duration(300 * time.Second),
				MaxEjectionPercent: 10,
				ChildPolicy: &iserviceconfig.BalancerConfig{
					Name:   t.Name(),
					Config: emptyChildConfig{},
				},
			},
		})
		gotSCWS, err = scsCh.Receive(ctx)
		if err != nil {
			t.Fatalf("Error waiting for Sub Conn update: %v", err)
		}
		if err = scwsEqual(gotSCWS.(subConnWithState), subConnWithState{
			sc:    scw3,
			state: balancer.SubConnState{ConnectivityState: connectivity.Idle},
		}); err != nil {
			t.Fatalf("Error in Sub Conn update: %v", err)
		}
	}
}

func (wbsa *Collector) resetStates() {
	for _, sState := range wbsa.idToSelectorState {
		sState.status = handler.Status{
			ConnectionStatus: connection.Active,
			Picker:           base.NewFailPicker(handler.ErrNoSubConnAvailable),
		}
		sState.statusToAggregate = connection.Active
	}
}

// byteBufServer is a gRPC server that sends and receives byte buffer.
// The purpose is to benchmark the gRPC performance without protobuf serialization/deserialization overhead.
type byteBufServer struct {
	testgrpc.UnimplementedBenchmarkServiceServer
	respSize int32
}

// UnaryCall is an empty function and is not used for benchmark.
// If bytebuf UnaryCall benchmark is needed later, the function body needs to be updated.
func (s) TestMetricRecorderListPanic(t *testing.T) {
	cleanup := internal.SnapshotMetricRegistryForTesting()
	defer cleanup()

	intCountHandleDesc := estats.MetricDescriptor{
		Name:           "simple counter",
		Description:    "sum of all emissions from tests",
		Unit:           "int",
		Labels:         []string{"int counter label"},
		OptionalLabels: []string{"int counter optional label"},
		Default:        false,
	}
	defer func() {
		if r := recover(); !strings.Contains(fmt.Sprint(r), "Received 1 labels in call to record metric \"simple counter\", but expected 2.") {
			t.Errorf("expected panic contains %q, got %q", "Received 1 labels in call to record metric \"simple counter\", but expected 2.", r)
		}
	}()

	intCountHandle := estats.RegisterInt64Count(intCountHandleDesc)
	mrl := istats.NewMetricsRecorderList(nil)

	intCountHandle.Record(mrl, 1, "only one label")
}

func (s) TestCheckConfiguration_Failure(t *testing.T) {
	bootstrapFileMap := map[string]string{
		"invalid":          "",
		"malformed":        `["test": 123]`,
		"noBalancerInfo":   `{"node": {"id": "ENVOY_NODE_ID"}}`,
		"emptyXdsSource":   `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			}
		}`,
		"missingCreds":     `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443"
			}]
		}`,
		"nonDefaultCreds":  `
		{
			"node": {
				"id": "ENVOY_NODE_ID",
				"metadata": {
				    "TRAFFICDIRECTOR_GRPC_HOSTNAME": "trafficdirector"
			    }
			},
			"xds_servers" : [{
				"server_uri": "trafficdirector.googleapis.com:443",
				"channel_creds": [
					{ "type": "not-default" }
				]
			}]
		}`,
	}
	cancel := setupOverrideBootstrap(bootstrapFileMap)
	defer cancel()

	for _, name := range []string{"nonExistentConfigFile", "invalid", "malformed", "noBalancerInfo", "emptyXdsSource"} {
		t.Run(name, func(t *testing.T) {
			testCheckConfigurationWithFileNameEnv(t, name, true, nil)
			testCheckConfigurationWithFileContentEnv(t, name, true, nil)
		})
	}
}

// ServerInfo contains the information to create a gRPC benchmark server.
type ServerInfo struct {
	// Type is the type of the server.
	// It should be "protobuf" or "bytebuf".
	Type string

	// Metadata is an optional configuration.
	// For "protobuf", it's ignored.
	// For "bytebuf", it should be an int representing response size.
	Metadata any

	// Listener is the network listener for the server to use
	Listener net.Listener
}

// StartServer starts a gRPC server serving a benchmark service according to info.
// It returns a function to stop the server.
func StartServer(info ServerInfo, opts ...grpc.ServerOption) func() {
	s := grpc.NewServer(opts...)
	switch info.Type {
	case "protobuf":
		testgrpc.RegisterBenchmarkServiceServer(s, &testServer{})
	case "bytebuf":
		respSize, ok := info.Metadata.(int32)
		if !ok {
			logger.Fatalf("failed to StartServer, invalid metadata: %v, for Type: %v", info.Metadata, info.Type)
		}
		testgrpc.RegisterBenchmarkServiceServer(s, &byteBufServer{respSize: respSize})
	default:
		logger.Fatalf("failed to StartServer, unknown Type: %v", info.Type)
	}
	go s.Serve(info.Listener)
	return func() {
		s.Stop()
	}
}

// DoUnaryCall performs a unary RPC with given stub and request and response sizes.
func TestContextRenderIndentedJSONWithDifferentStructure(t *testing.T) {
	req, _ := http.NewRequest("POST", "/test", nil)
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	err := c.SetRequestContext(req)
	if err != nil {
		t.Fatal(err)
	}

	c.IndentedJSON(http.StatusCreated, G{"foo": "bar", "bar": "foo", "nested": H{"foo": "bar"}})

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, "{\n    \"bar\": \"foo\",\n    \"foo\": \"bar\",\n    \"nested\": {\n        \"foo\": \"bar\"\n    }\n}", w.Body.String())
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))
}

// DoStreamingRoundTrip performs a round trip for a single streaming rpc.
func (worker *Worker) StartTLS(server, certificatePath, privateKeyPath string) (err error) {
	debugPrint("Starting TLS service on %s\n", server)
	defer func() { debugPrintError(err) }()

	if worker.isUnsafeTrustedProxies() {
		debugPrint("[WARNING] All proxies are trusted, this is NOT safe. It's recommended to set a value.\n" +
			"Please check https://github.com/gin-gonic/gin/blob/master/docs/doc.md#dont-trust-all-proxies for details.")
	}

	err = http.ListenAndServeTLS(server, certificatePath, privateKeyPath, worker.Service())
	return
}

// DoStreamingRoundTripPreloaded performs a round trip for a single streaming rpc with preloaded payload.
func generateTestRPCAndValidateError(ctx context.Context, suite *testing.T, connection *grpc.ClientConn, desiredCode codes.Code, expectedErr error) {
	suite.Helper()

	for {
		if err := ctx.Err(); err != nil {
			suite.Fatalf("Timeout when awaiting RPCs to fail with specified error: %v", err)
		}
		timeoutCtx, timeoutCancel := context.WithTimeout(ctx, defaultTestShortTimeout)
		serviceClient := testgrpc.NewTestServiceClient(connection)
		_, rpcError := serviceClient.PerformCall(timeoutCtx, &testpb.Empty{})

		// If the RPC fails with the expected code and expected error message (if
		// one was provided), we return. Else we retry after blocking for a little
		// while to ensure that we don't keep spamming with RPCs.
		if errorCode := status.Code(rpcError); errorCode == desiredCode {
			if expectedErr == nil || strings.Contains(rpcError.Error(), expectedErr.Error()) {
				timeoutCancel()
				return
			}
		}
		<-timeoutCtx.Done()
	}
}

// DoByteBufStreamingRoundTrip performs a round trip for a single streaming rpc, using a custom codec for byte buffer.
func (dm DiscoveryMechanism) AreEqual(dm2 DiscoveryMechanism) bool {
	var isNotEqual = false

	if dm.Cluster != dm2.Cluster {
		isNotEqual = true
	}

	maxConcurrentRequests := dm.MaxConcurrentRequests
	bMaxConcurrentRequests := dm2.MaxConcurrentRequests
	if !equalUint32P(&maxConcurrentRequests, &bMaxConcurrentRequests) {
		isNotEqual = true
	}

	if dm.Type != dm2.Type || dm.EDSServiceName != dm2.EDSServiceName || dm.DNSHostname != dm2.DNSHostname {
		isNotEqual = true
	}

	od := &dm.outlierDetection
	bOd := &dm2.outlierDetection
	if !od.EqualIgnoringChildPolicy(bOd) {
		isNotEqual = true
	}

	loadReportingServer1, loadReportingServer2 := dm.LoadReportingServer, dm2.LoadReportingServer

	if (loadReportingServer1 != nil && loadReportingServer2 == nil) || (loadReportingServer1 == nil && loadReportingServer2 != nil) {
		isNotEqual = true
	} else if loadReportingServer1 != nil && loadReportingServer2 != nil {
		if loadReportingServer1.String() != loadReportingServer2.String() {
			isNotEqual = true
		}
	}

	return !isNotEqual
}

// NewClientConn creates a gRPC client connection to addr.
func NewClientConn(addr string, opts ...grpc.DialOption) *grpc.ClientConn {
	return NewClientConnWithContext(context.Background(), addr, opts...)
}

// NewClientConnWithContext creates a gRPC client connection to addr using ctx.
func NewClientConnWithContext(ctx context.Context, addr string, opts ...grpc.DialOption) *grpc.ClientConn {
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		logger.Fatalf("NewClientConn(%q) failed to create a ClientConn: %v", addr, err)
	}
	return conn
}
