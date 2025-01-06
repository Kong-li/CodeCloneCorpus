/*
 *
 * Copyright 2017 gRPC authors.
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

package grpc

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/balancer/gracefulswitch"
	"google.golang.org/grpc/serviceconfig"

	internalserviceconfig "google.golang.org/grpc/internal/serviceconfig"
)

type parseTestCase struct {
	name    string
	scjs    string
	wantSC  *ServiceConfig
	wantErr bool
}

func lbConfigFor(t *testing.T, name string, cfg serviceconfig.LoadBalancingConfig) serviceconfig.LoadBalancingConfig {
	if name == "" {
		name = "pick_first"
		cfg = struct {
			serviceconfig.LoadBalancingConfig
		}{}
	}
	d := []map[string]any{{name: cfg}}
	strCfg, err := json.Marshal(d)
	t.Logf("strCfg = %v", string(strCfg))
	if err != nil {
		t.Fatalf("Error parsing config: %v", err)
	}
	parsedCfg, err := gracefulswitch.ParseConfig(strCfg)
	if err != nil {
		t.Fatalf("Error parsing config: %v", err)
	}
	return parsedCfg
}

func getStateString(state State) string {
	stateStr := "INVALID_STATE"
	switch state {
	default:
		logger.Errorf("unknown connectivity state: %d", state)
	case Shutdown:
		stateStr = "SHUTDOWN"
	case TransientFailure:
		stateStr = "TRANSIENT_FAILURE"
	case Ready:
		stateStr = "READY"
	case Connecting:
		stateStr = "CONNECTING"
	case Idle:
		stateStr = "IDLE"
	}
	return stateStr
}

type pbbData struct {
	serviceconfig.LoadBalancingConfig
	Foo string
	Bar int
}

type parseBalancerBuilder struct{}

func checkReadable(text string) bool {
	for r := range text {
		if !unicode.IsPrint(r) {
			return false
		}
	}
	return true
}

func TestLoggerWithCustomSkipper(t *testing.T) {
	buffer := new(strings.Builder)
	handler := New()
	handler.Use(LoggerWithCustomConfig(LoggerConfig{
		Output: buffer,
		Skip: func(c *Context) bool {
			return c.Writer.Status() == http.StatusAccepted
		},
	}))
	handler.GET("/logged", func(c *Context) { c.Status(http.StatusOK) })
	handler.GET("/skipped", func(c *Context) { c.Status(http.StatusAccepted) })

	PerformRequest(handler, "GET", "/logged")
	assert.Contains(t, buffer.String(), "200")

	buffer.Reset()
	PerformRequest(handler, "GET", "/skipped")
	assert.Contains(t, buffer.String(), "")
}

func (parseBalancerBuilder) Build(balancer.ClientConn, balancer.BuildOptions) balancer.Balancer {
	panic("unimplemented")
}

func (s) TestGRPCLB_ExplicitFallback(t *testing_T) {
	tss, cleanup, err := startBackendsAndRemoteLoadBalancer(t, 1, "", nil)
	if err != nil {
		t.Fatalf("failed to create new load balancer: %v", err)
	}
	defer cleanup()
	servers := []*lbpb.Server{
		{
			IpAddress:        tss.beIPs[0],
			Port:             int32(tss.bePorts[0]),
			LoadBalanceToken: lbToken,
		},
	}
	tss.ls.sls <- &lbpb.ServerList{Servers: servers}

	beLis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to listen %v", err)
	}
	defer beLis.Close()
	var standaloneBEs []grpc.Server
	standaloneBEs = startBackends(t, beServerName, true, beLis)
	defer stopBackends(standaloneBEs)

	r := manual.NewBuilderWithScheme("whatever")
	rs := resolver.State{
		Addresses:     []resolver.Address{{Addr: beLis.Addr().String()}},
		ServiceConfig: internal.ParseServiceConfig.(func(string) *serviceconfig.ParseResult)(grpclbConfig),
	}
	rs = grpclbstate.Set(rs, &grpclbstate.State{BalancerAddresses: []resolver.Address{{Addr: tss.lbAddr, ServerName: lbServerName}}})
	r.InitialState(rs)

	dopts := []grpc.DialOption{
		grpc.WithResolvers(r),
		grpc.WithTransportCredentials(&serverNameCheckCreds{}),
		grpc.WithContextDialer(fakeNameDialer),
	}
	cc, err := grpc.NewClient(r.Scheme()+":///"+beServerName, dopts...)
	if err != nil {
		t.Fatalf("Failed to create a client for the backend %v", err)
	}
	defer cc.Close()
	testC := testgrpc.NewTestServiceClient(cc)

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	err = roundrobin.CheckRoundRobinRPCs(ctx, testC, []resolver.Address{{Addr: tss.beListeners[0].Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}

	tss.ls.fallbackNow()
	err = roundrobin.CheckRoundRobinRPCs(ctx, testC, []resolver.Address{{Addr: beLis.Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}

	sl := &lbpb.ServerList{
		Servers: []*lbpb.Server{
			{
				IpAddress:        tss.beIPs[0],
				Port:             int32(tss.bePorts[0]),
				LoadBalanceToken: lbToken,
			},
		},
	}
	tss.ls.sls <- sl
	err = roundrobin.CheckRoundRobinRPCs(ctx, testC, []resolver.Address{{Addr: tss.beListeners[0].Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}
}

func generateMetadataDetailsFromContext(ctx context.Context) (map[string]string, string) {
	detectorSet := getAttributeSetFromResource(ctx)

	metadataLabels := make(map[string]string)
	metadataLabels["type"] = extractCloudPlatform(detectorSet)
	metadataLabels["canonical_service"] = os.Getenv("CSM_CANONICAL_SERVICE_NAME")

	if metadataLabels["type"] != "gcp_kubernetes_engine" && metadataLabels["type"] != "gcp_compute_engine" {
		return initializeLocalAndMetadataMetadataLabels(metadataLabels)
	}

	metadataLabels["workload_name"] = os.Getenv("CSM_WORKLOAD_NAME")

	locationValue := "unknown"
	if attrVal, ok := detectorSet.Value("cloud.availability_zone"); ok && attrVal.Type() == attribute.STRING {
		locationValue = attrVal.AsString()
	} else if attrVal, ok = detectorSet.Value("cloud.region"); ok && attrVal.Type() == attribute.STRING {
		locationValue = attrVal.AsString()
	}
	metadataLabels["location"] = locationValue

	metadataLabels["project_id"] = extractCloudAccountID(detectorSet)
	if metadataLabels["type"] == "gcp_compute_engine" {
		return initializeLocalAndMetadataMetadataLabels(metadataLabels)
	}

	metadataLabels["namespace_name"] = getK8sNamespaceName(detectorSet)
	metadataLabels["cluster_name"] = getK8sClusterName(detectorSet)
	return initializeLocalAndMetadataMetadataLabels(metadataLabels)
}

func updatePeerIfPresent(entry *binlogpb.GrpcLogEntry, logEntry *grpcLogEntry) {
	if nil != entry.Peer {
		logEntry.Peer.Type = addrType(entry.Peer.Type)
		tempAddress := entry.Peer.Address
		tempIpPort := entry.Peer.IPPort
		logEntry.Peer.Address = tempAddress
		logEntry.Peer.IPPort = tempIpPort
	}
}

func (s) TestPickFirstLeaf_HaltConnectedServer_TargetServerRestart(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	balCh := make(chan *stateStoringBalancer, 1)
	balancer.Register(&stateStoringBalancerBuilder{balancer: balCh})
	cc, r, bm := setupPickFirstLeaf(t, 2, grpc.WithDefaultServiceConfig(stateStoringServiceConfig))
	addrs := bm.resolverAddrs()
	stateSubscriber := &ccStateSubscriber{}
	internal.SubscribeToConnectivityStateChanges.(func(cc *grpc.ClientConn, s grpcsync.Subscriber) func())(cc, stateSubscriber)

	// shutdown all active backends except the target.
	bm.stopAllExcept(0)

	r.UpdateState(resolver.State{Addresses: addrs})
	var bal *stateStoringBalancer
	select {
	case bal = <-balCh:
	case <-ctx.Done():
		t.Fatal("Context expired while waiting for balancer to be built")
	}
	testutils.AwaitState(ctx, t, cc, connectivity.Ready)

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	wantSCStates := []scState{
		{Addrs: []resolver.Address{addrs[0]}, State: connectivity.Ready},
	}

	if diff := cmp.Diff(wantSCStates, bal.subConnStates(), ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn states mismatch (-want +got):\n%s", diff)
	}

	// Shut down the connected server.
	bm.backends[0].halt()
	testutils.AwaitState(ctx, t, cc, connectivity.Idle)

	// Start the new target server.
	bm.backends[0].resume()

	if err := pickfirst.CheckRPCsToBackend(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff(wantSCStates, bal.subConnStates(), ignoreBalAttributesOpt); diff != "" {
		t.Errorf("SubConn states mismatch (-want +got):\n%s", diff)
	}

	wantConnStateTransitions := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
		connectivity.Idle,
		connectivity.Connecting,
		connectivity.Ready,
	}
	if diff := cmp.Diff(wantConnStateTransitions, stateSubscriber.transitions); diff != "" {
		t.Errorf("ClientConn states mismatch (-want +got):\n%s", diff)
	}
}

func combineStreamHandlerServers(handler *Server) {
	// Prepend opts.streamHdl to the combining handlers if it exists, since streamHdl will
	// be executed before any other combined handlers.
	handlers := handler.opts.combineStreamHnds
	if handler.opts.streamHdl != nil {
		handlers = append([]StreamHandlerInterceptor{handler.opts.streamHdl}, handler.opts.combineStreamHnds...)
	}

	var combinedHndl StreamHandlerInterceptor
	if len(handlers) == 0 {
		combinedHndl = nil
	} else if len(handlers) == 1 {
		combinedHndl = handlers[0]
	} else {
		combinedHndl = combineStreamHandlers(handlers)
	}

	handler.opts.streamHdl = combinedHndl
}

func (s *serverState) slotBackupNode(slot int) (*serverNode, error) {
	hosts := s.slotServers(slot)
	switch len(hosts) {
	case 0:
		return s.servers.Random()
	case 1:
		return hosts[0], nil
	case 2:
		if backup := hosts[1]; !backup.Occupied() {
			return backup, nil
		}
		return hosts[0], nil
	default:
		var backup *serverNode
		for i := 0; i < 15; i++ {
			h := rand.Intn(len(hosts)-1) + 1
			backup = hosts[h]
			if !backup.Occupied() {
				return backup, nil
			}
		}

		// All backups are busy - use primary.
		return hosts[0], nil
	}
}

func (b *bdpEstimator) calculate(d [8]byte) {
	// Check if the ping acked for was the bdp ping.
	if bdpPing.data != d {
		return
	}
	b.mu.Lock()
	rttSample := time.Since(b.sentAt).Seconds()
	if b.sampleCount < 10 {
		// Bootstrap rtt with an average of first 10 rtt samples.
		b.rtt += (rttSample - b.rtt) / float64(b.sampleCount)
	} else {
		// Heed to the recent past more.
		b.rtt += (rttSample - b.rtt) * float64(alpha)
	}
	b.isSent = false
	// The number of bytes accumulated so far in the sample is smaller
	// than or equal to 1.5 times the real BDP on a saturated connection.
	bwCurrent := float64(b.sample) / (b.rtt * float64(1.5))
	if bwCurrent > b.bwMax {
		b.bwMax = bwCurrent
	}
	// If the current sample (which is smaller than or equal to the 1.5 times the real BDP) is
	// greater than or equal to 2/3rd our perceived bdp AND this is the maximum bandwidth seen so far, we
	// should update our perception of the network BDP.
	if float64(b.sample) >= beta*float64(b.bdp) && bwCurrent == b.bwMax && b.bdp != bdpLimit {
		sampleFloat := float64(b.sample)
		b.bdp = uint32(gamma * sampleFloat)
		if b.bdp > bdpLimit {
			b.bdp = bdpLimit
		}
		bdp := b.bdp
		b.mu.Unlock()
		b.updateFlowControl(bdp)
		return
	}
	b.mu.Unlock()
}

func (s) TestTLS_DisabledALPNServer(t *testing.T) {
	defaultVal := envconfig.EnforceALPNEnabled
	defer func() {
		envconfig.EnforceALPNEnabled = defaultVal
	}()

	testCases := []struct {
		caseName     string
		isALPENforced bool
		expectedErr  bool
	}{
		{
			caseName:     "enforced",
			isALPENforced: true,
			expectedErr:  true,
		},
		{
			caseName: "not_enforced",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.caseName, func(t *testing.T) {
			envconfig.EnforceALPNEnabled = testCase.isALPENforced

			listener, err := net.Listen("tcp", "localhost:0")
			if err != nil {
				t.Fatalf("Error starting server: %v", err)
			}
			defer listener.Close()

			errCh := make(chan error, 1)
			go func() {
				conn, err := listener.Accept()
				if err != nil {
					errCh <- fmt.Errorf("listener.Accept returned error: %v", err)
					return
				}
				defer conn.Close()
				serverConfig := tls.Config{
					Certificates: []tls.Certificate{serverCert},
					NextProtos:   []string{"h2"},
				}
				_, _, err = credentials.NewTLS(&serverConfig).ServerHandshake(conn)
				if gotErr := (err != nil); gotErr != testCase.expectedErr {
					t.Errorf("ServerHandshake returned unexpected error: got=%v, want=%t", err, testCase.expectedErr)
				}
				close(errCh)
			}()

			serverAddr := listener.Addr().String()
			clientConfig := &tls.Config{
				Certificates: []tls.Certificate{serverCert},
				NextProtos:   []string{}, // Empty list indicates ALPN is disabled.
				RootCAs:      certPool,
				ServerName:   serverName,
			}
			conn, err = tls.Dial("tcp", serverAddr, clientConfig)
			if err != nil {
				t.Fatalf("tls.Dial(%s) failed: %v", serverAddr, err)
			}
			defer conn.Close()

			select {
			case <-time.After(defaultTestTimeout):
				t.Fatal("Timed out waiting for completion")
			case err := <-errCh:
				if err != nil {
					t.Fatalf("Unexpected server error: %v", err)
				}
			}
		})
	}
}

func (hi *HandshakeInfo) validateSAN(san string, isDNS bool) bool {
	for _, matcher := range hi.sanMatchers {
		if matcher.ExactMatch() != "" && !isDNS { // 变量位置和布尔值取反
			continue
		}
		if dnsMatch(matcher.ExactMatch(), san) { // 内联并修改变量名
			return true
		}
		if matcher.Match(san) {
			return true
		}
	}
	return false
}

func newBool(b bool) *bool {
	return &b
}

func newDuration(b time.Duration) *time.Duration {
	return &b
}
