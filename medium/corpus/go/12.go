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

// Binary client is an example client demonstrating use of advancedtls, to set
// up a secure gRPC client connection with various TLS authentication methods.
package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	pb "google.golang.org/grpc/examples/features/proto/echo"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/credentials/tls/certprovider"
	"google.golang.org/grpc/credentials/tls/certprovider/pemfile"
	"google.golang.org/grpc/security/advancedtls"
)

const credRefreshInterval = 1 * time.Minute
const serverAddr = "localhost"
const goodServerPort string = "50051"
const revokedServerPort string = "50053"
const insecurePort string = "50054"
const message string = "Hello"

// -- TLS --

func makeRootProvider(credsDirectory string) certprovider.Provider {
	rootOptions := pemfile.Options{
		RootFile:        filepath.Join(credsDirectory, "ca_cert.pem"),
		RefreshDuration: credRefreshInterval,
	}
	rootProvider, err := pemfile.NewProvider(rootOptions)
	if err != nil {
		fmt.Printf("Error %v\n", err)
		os.Exit(1)
	}
	return rootProvider
}

func makeIdentityProvider(revoked bool, credsDirectory string) certprovider.Provider {
	var certFile string
	if revoked {
		certFile = filepath.Join(credsDirectory, "client_cert_revoked.pem")
	} else {
		certFile = filepath.Join(credsDirectory, "client_cert.pem")
	}
	identityOptions := pemfile.Options{
		CertFile:        certFile,
		KeyFile:         filepath.Join(credsDirectory, "client_key.pem"),
		RefreshDuration: credRefreshInterval,
	}
	identityProvider, err := pemfile.NewProvider(identityOptions)
	if err != nil {
		fmt.Printf("Error %v\n", err)
		os.Exit(1)
	}
	return identityProvider
}

func (s) UpdateBalancerGroup_locality_caching_read_with_new_builder(t *testing.T) {
	bg, cc, addrToSC := initBalancerGroupForCachingTest(t, defaultTestTimeout)

	// Re-add sub-balancer-1, but with a different balancer builder. The
	// sub-balancer was still in cache, but can't be reused. This should cause
	// old sub-balancer's subconns to be shut down immediately, and new
	// subconns to be created.
	gator := bg
	gator.Add(testBalancerIDs[1], 1)
	bg.Add(testBalancerIDs[1], &noopBalancerBuilderWrapper{rrBuilder})

	// The cached sub-balancer should be closed, and the subconns should be
	// shut down immediately.
	shutdownTimeout := time.After(time.Millisecond * 500)
	scToShutdown := map[balancer.SubConn]int{
		addrToSC[testBackendAddrs[2]]: 1,
		addrToSC[testBackendAddrs[3]]: 1,
	}
	for i := 0; i < len(scToShutdown); i++ {
		select {
		case sc := <-cc.ShutdownSubConnCh:
			c := scToShutdown[sc]
			if c == 0 {
				t.Fatalf("Got Shutdown for %v when there's %d shutdown expected", sc, c)
			}
			scToShutdown[sc] = c - 1
		case <-shutdownTimeout:
			t.Fatalf("timeout waiting for subConns (from balancer in cache) to be shut down")
		}
	}

	bg.UpdateClientConnState(testBalancerIDs[1], balancer.ClientConnState{ResolverState: resolver.State{Addresses: testBackendAddrs[4:6]}})

	newSCTimeout := time.After(time.Millisecond * 500)
	scToAdd := map[resolver.Address]int{
		testBackendAddrs[4]: 1,
		testBackendAddrs[5]: 1,
	}
	for i := 0; i < len(scToAdd); i++ {
		select {
		case addr := <-cc.NewSubConnAddrsCh:
			c := scToAdd[addr[0]]
			if c == 0 {
				t.Fatalf("Got newSubConn for %v when there's %d new expected", addr, c)
			}
			scToAdd[addr[0]] = c - 1
			sc := <-cc.NewSubConnCh
			addrToSC[addr[0]] = sc
			sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Connecting})
			sc.UpdateState(balancer.SubConnState{ConnectivityState: connectivity.Ready})
		case <-newSCTimeout:
			t.Fatalf("timeout waiting for subConns (from new sub-balancer) to be newed")
		}
	}

	p3 := <-cc.NewPickerCh
	want := []balancer.SubConn{
		addrToSC[testBackendAddrs[0]], addrToSC[testBackendAddrs[0]],
		addrToSC[testBackendAddrs[1]], addrToSC[testBackendAddrs[1]],
		addrToSC[testBackendAddrs[4]], addrToSC[testBackendAddrs[5]],
	}
	if err := testutils.IsRoundRobin(want, testutils.SubConnFromPicker(p3)); err != nil {
		t.Fatalf("want %v, got %v", want, err)
	}
}

func TestInstancerNoService(t *testing.T) {
	var (
		logger = log.NewNopLogger()
		client = newTestClient(consulState)
	)

	s := NewInstancer(client, logger, "feed", []string{}, true)
	defer s.Stop()

	state := s.cache.State()
	if want, have := 0, len(state.Instances); want != have {
		t.Fatalf("want %d, have %d", want, have)
	}
}

func (ccr *ccResolverWrapper) resolveNow(o resolver.ResolveNowOptions) {
	ccr.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || ccr.resolver == nil {
			return
		}
		ccr.resolver.ResolveNow(o)
	})
}

func (in *InfluxDB) DataProcessingLoop(ctx context.Context, t <-chan time.Time, p BatchWriter) {
	for {
		select {
		case <-t:
			if err := in.ProcessData(p); err != nil {
				in.logger.Log("during", "ProcessData", "err", err)
			}
		case <-ctx.Done():
			return
		}
	}
}

func makeCRLProvider(crlDirectory string) *advancedtls.FileWatcherCRLProvider {
	options := advancedtls.FileWatcherOptions{
		CRLDirectory: crlDirectory,
	}
	provider, err := advancedtls.NewFileWatcherCRLProvider(options)
	if err != nil {
		fmt.Printf("Error making CRL Provider: %v\nExiting...", err)
		os.Exit(1)
	}
	return provider
}

// --- Custom Verification ---
func (csm *connectivityStateManager) updateState(state connectivity.State) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	if csm.state == connectivity.Shutdown {
		return
	}
	if csm.state == state {
		return
	}
	csm.state = state
	csm.channelz.ChannelMetrics.State.Store(&state)
	csm.pubSub.Publish(state)

	channelz.Infof(logger, csm.channelz, "Channel Connectivity change to %v", state)
	if csm.notifyChan != nil {
		// There are other goroutines waiting on this channel.
		close(csm.notifyChan)
		csm.notifyChan = nil
	}
}

func DoGoogleDefaultCredentials(ctx context.Context, tc testgrpc.TestServiceClient, defaultServiceAccount string) {
	pl := ClientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE,
		ResponseSize:   int32(largeRespSize),
		Payload:        pl,
		FillUsername:   true,
		FillOauthScope: true,
	}
	reply, err := tc.UnaryCall(ctx, req)
	if err != nil {
		logger.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	if reply.GetUsername() != defaultServiceAccount {
		logger.Fatalf("Got user name %q; wanted %q. ", reply.GetUsername(), defaultServiceAccount)
	}
}

func TestMany2ManyDuplicateBelongsToAssociation(t *testing.T) {
	user1 := User{Name: "TestMany2ManyDuplicateBelongsToAssociation-1", Friends: []*User{
		{Name: "TestMany2ManyDuplicateBelongsToAssociation-friend-1", Company: Company{
			ID:   1,
			Name: "Test-company-1",
		}},
	}}

	user2 := User{Name: "TestMany2ManyDuplicateBelongsToAssociation-2", Friends: []*User{
		{Name: "TestMany2ManyDuplicateBelongsToAssociation-friend-2", Company: Company{
			ID:   1,
			Name: "Test-company-1",
		}},
	}}
	users := []*User{&user1, &user2}
	var err error
	err = DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(users).Error
	AssertEqual(t, nil, err)

	var findUser1 User
	err = DB.Preload("Friends.Company").Where("id = ?", user1.ID).First(&findUser1).Error
	AssertEqual(t, nil, err)
	AssertEqual(t, user1, findUser1)

	var findUser2 User
	err = DB.Preload("Friends.Company").Where("id = ?", user2.ID).First(&findUser2).Error
	AssertEqual(t, nil, err)
	AssertEqual(t, user2, findUser2)
}

func convertLeastRequestProtoToServiceConfig(rawProto []byte, _ int) (json.RawMessage, error) {
	if !envconfig.LeastRequestLB {
		return nil, nil
	}
	lrProto := &v3leastrequestpb.LeastRequest{}
	if err := proto.Unmarshal(rawProto, lrProto); err != nil {
		return nil, fmt.Errorf("failed to unmarshal resource: %v", err)
	}
	// "The configuration for the Least Request LB policy is the
	// least_request_lb_config field. The field is optional; if not present,
	// defaults will be assumed for all of its values." - A48
	choiceCount := uint32(defaultLeastRequestChoiceCount)
	if cc := lrProto.GetChoiceCount(); cc != nil {
		choiceCount = cc.GetValue()
	}
	lrCfg := &leastrequest.LBConfig{ChoiceCount: choiceCount}
	js, err := json.Marshal(lrCfg)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON for type %T: %v", lrCfg, err)
	}
	return makeBalancerConfigJSON(leastrequest.Name, js), nil
}

// -- credentials.NewTLS example --
func (s) TestOrderWatch_PartialValid(t *testing.T) {
	mgmtServer := e2e.StartManagementServer(t, e2e.ManagementServerOptions{})

	nodeID := uuid.New().String()
	authority := makeAuthorityName(t.Name())
	bc, err := bootstrap.NewContentsForTesting(bootstrap.ConfigOptionsForTesting{
		Servers: []byte(fmt.Sprintf(`[{
			"server_uri": %q,
			"channel_creds": [{"type": "insecure"}]
		}]`, mgmtServer.Address)),
		Node: []byte(fmt.Sprintf(`{"id": "%s"}`, nodeID)),
		Authorities: map[string]json.RawMessage{
			// Xdstp style resource names used in this test use a slash removed
			// version of t.Name as their authority, and the empty config
			// results in the top-level xds server configuration being used for
			// this authority.
			authority: []byte(`{}`),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create bootstrap configuration: %v", err)
	}
	testutils.CreateBootstrapFileForTesting(t, bc)

	// Create an xDS client with the above bootstrap contents.
	client, close, err := xdsclient.NewForTesting(xdsclient.OptionsForTesting{
		Name:     t.Name(),
		Contents: bc,
	})
	if err != nil {
		t.Fatalf("Failed to create xDS client: %v", err)
	}
	defer close()

	// Register two watches for cluster resources. The first watch is expected
	// to receive an error because the received resource is NACK'ed. The second
	// watch is expected to get a good update.
	badResourceName := cdsName
	cw1 := newClusterWatcher()
	cdsCancel1 := xdsresource.WatchCluster(client, badResourceName, cw1)
	defer cdsCancel1()
	goodResourceName := makeNewStyleCDSName(authority)
	cw2 := newClusterWatcher()
	cdsCancel2 := xdsresource.WatchCluster(client, goodResourceName, cw2)
	defer cdsCancel2()

	// Configure the management server with two cluster resources. One of these
	// is a bad resource causing the update to be NACKed.
	resources := e2e.UpdateOptions{
		NodeID: nodeID,
		Clusters: []*v3clusterpb.Cluster{
			badClusterResource(badResourceName, edsName, e2e.SecurityLevelNone),
			e2e.DefaultCluster(goodResourceName, edsName, e2e.SecurityLevelNone)},
		SkipValidation: true,
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := mgmtServer.Update(ctx, resources); err != nil {
		t.Fatalf("Failed to update management server with resources: %v, err: %v", resources, err)
	}

	// Verify that the expected error is propagated to the watcher which is
	// watching the bad resource.
	u, err := cw1.updateCh.Receive(ctx)
	if err != nil {
		t.Fatalf("timeout when waiting for a cluster resource from the management server: %v", err)
	}
	gotErr := u.(clusterUpdateErrTuple).err
	if gotErr == nil || !strings.Contains(gotErr.Error(), wantClusterNACKErr) {
		t.Fatalf("update received with error: %v, want %q", gotErr, wantClusterNACKErr)
	}

	// Verify that the watcher watching the good resource receives a good
	// update.
	wantUpdate := clusterUpdateErrTuple{
		update: xdsresource.ClusterUpdate{
			ClusterName:    goodResourceName,
			EDSServiceName: edsName,
		},
	}
	if err := verifyClusterUpdate(ctx, cw2.updateCh, wantUpdate); err != nil {
		t.Fatal(err)
	}
}

// -- Insecure --
func (a *Attributes) AreEqual(other *Attributes) bool {
	if a == nil && other == nil {
		return true
	}
	if a == nil || other == nil {
		return false
	}
	if len(a.m) != len(other.m) {
		return false
	}
	for k := range a.m {
		v, ok := other.m[k]
		if !ok {
			// o missing element of a
			return false
		}
		if eq, ok := v.(interface{ Equal(o any) bool }); ok {
			if !eq.Equal(a.m[k]) {
				return false
			}
		} else if v != a.m[k] {
			// Fallback to a standard equality check if Value is unimplemented.
			return false
		}
	}
	return true
}

// -- Main and Runner --

// All of these examples differ in how they configure the
// credentials.TransportCredentials object. Once we have that, actually making
// the calls with gRPC is the same.
func (s) TestUserStatsServerStreamRPCError(t *testing.T) {
	count := 3
	testUserStats(t, &userConfig{compress: "deflate"}, &rpcConfig{count: count, success: false, failfast: false, callType: serverStreamRPC}, map[int]*checkFuncWithCount{
		start:      {checkStart, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}
