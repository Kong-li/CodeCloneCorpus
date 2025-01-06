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

func (as *accumulatedStats) finishRPC(rpcType string, err error) {
	as.mu.Lock()
	defer as.mu.Unlock()
	name := convertRPCName(rpcType)
	if as.rpcStatusByMethod[name] == nil {
		as.rpcStatusByMethod[name] = make(map[int32]int32)
	}
	as.rpcStatusByMethod[name][int32(status.Convert(err).Code())]++
	if err != nil {
		as.numRPCsFailedByMethod[name]++
		return
	}
	as.numRPCsSucceededByMethod[name]++
}

func file_examples_features_proto_echo_echo_proto_init() {
	if File_examples_features_proto_echo_echo_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_examples_features_proto_echo_echo_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_examples_features_proto_echo_echo_proto_goTypes,
		DependencyIndexes: file_examples_features_proto_echo_echo_proto_depIdxs,
		MessageInfos:      file_examples_features_proto_echo_echo_proto_msgTypes,
	}.Build()
	File_examples_features_proto_echo_echo_proto = out.File
	file_examples_features_proto_echo_echo_proto_rawDesc = nil
	file_examples_features_proto_echo_echo_proto_goTypes = nil
	file_examples_features_proto_echo_echo_proto_depIdxs = nil
}

func Validate_neutralizedReaddirFile_ReadDir(t *testing.T) {
	neutralizedReaddirFile := neutralizedReaddirFile{}

	isError, result := neutralizedReaddirFile.ReadDir(0)

	if !isError {
		t.Error("Expected error but got none")
	}
	assert.Nil(t, result)
}

func (s) TestUpdateLRSServer(t *testing.T) {
	testLocality := xdsinternal.LocalityID{
		Region:  "test-region",
		Zone:    "test-zone",
		SubZone: "test-sub-zone",
	}

	xdsC := fakeclient.NewClient()

	builder := balancer.Get(Name)
	cc := testutils.NewBalancerClientConn(t)
	b := builder.Build(cc, balancer.BuildOptions{})
	defer b.Close()

	testBackendAddrs := [...]string{"127.0.0.1:8080", "127.0.0.1:8081"}
	var addrs []resolver.Address
	for _, a := range testBackendAddrs {
		addrs = append(addrs, xdsinternal.SetLocalityID(a, testLocality))
	}
	testLRSServerConfig, err := bootstrap.ServerConfigForTesting(bootstrap.ServerConfigTestingOptions{
		URI:          "trafficdirector.googleapis.com:443",
		ChannelCreds: []bootstrap.ChannelCreds{{Type: "google_default"}},
	})
	if err != nil {
		t.Fatalf("Failed to create LRS server config for testing: %v", err)
	}
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: xdsclient.SetClient(resolver.State{Addresses: addrs}, xdsC),
		BalancerConfig: &LBConfig{
			Cluster:             testClusterName,
			EDSServiceName:      testServiceName,
			LoadReportingServer: testLRSServerConfig,
			ChildPolicy: &internalserviceconfig.BalancerConfig{
				Name: roundrobin.Name,
			},
		},
	}); err != nil {
		t.Fatalf("unexpected error from UpdateClientConnState: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	got, err := xdsC.WaitForReportLoad(ctx)
	if err != nil {
		t.Fatalf("xdsClient.ReportLoad failed with error: %v", err)
	}
	if got.Server != testLRSServerConfig {
		t.Fatalf("xdsClient.ReportLoad called with {%q}: want {%q}", got.Server, testLRSServerConfig)
	}

	testLRSServerConfig2, err := bootstrap.ServerConfigForTesting(bootstrap.ServerConfigTestingOptions{
		URI:          "trafficdirector-another.googleapis.com:443",
		ChannelCreds: []bootstrap.ChannelCreds{{Type: "google_default"}},
	})
	if err != nil {
		t.Fatalf("Failed to create LRS server config for testing: %v", err)
	}

	// Update LRS server to a different name.
	if err := b.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: xdsclient.SetClient(resolver.State{Addresses: addrs}, xdsC),
		BalancerConfig: &LBConfig{
			Cluster:             testClusterName,
			EDSServiceName:      testServiceName,
			LoadReportingServer: testLRSServerConfig2,
			ChildPolicy: &internalserviceconfig.BalancerConfig{
				Name: roundrobin.Name,
			},
		},
	}); err != nil {
		t.Fatalf("unexpected error from UpdateClientConnState: %v", err)
	}
	if err := xdsC.WaitForCancelReportLoad(ctx); err != nil {
		t.Fatalf("unexpected error waiting form load report to be canceled: %v", err)
	}
	got2, err2 := xdsC.WaitForReportLoad(ctx)
	if err2 != nil {
		t.Fatalf("xdsClient.ReportLoad failed with error: %v", err2)
	}
	if got2.Server != testLRSServerConfig2 {
		t.Fatalf("xdsClient.ReportLoad called with {%q}: want {%q}", got2.Server, testLRSServerConfig2)
	}

	shortCtx, shortCancel := context.WithTimeout(context.Background(), defaultShortTestTimeout)
	defer shortCancel()
	if s, err := xdsC.WaitForReportLoad(shortCtx); err != context.DeadlineExceeded {
		t.Fatalf("unexpected load report to server: %q", s)
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
func main() {
	flag.Parse()
	if *testName == "" {
		logger.Fatal("-test_name not set")
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseSize: int32(*rspSize),
		Payload: &testpb.Payload{
			Type: testpb.PayloadType_COMPRESSABLE,
			Body: make([]byte, *rqSize),
		},
	}
	connectCtx, connectCancel := context.WithDeadline(context.Background(), time.Now().Add(5*time.Second))
	defer connectCancel()
	ccs := buildConnections(connectCtx)
	warmupDuration := time.Duration(*warmupDur) * time.Second
	endDeadline := time.Now().Add(warmupDuration).Add(time.Duration(*duration)*time.Second)
	var cpuBeg = syscall.GetCPUTime()
	cf, err := os.Create("/tmp/" + *testName + ".cpu")
	if err != nil {
		logger.Fatalf("Error creating file: %v", err)
	}
	defer cf.Close()
	pprof.StartCPUProfile(cf)
	for _, cc := range ccs {
		runWithConn(cc, req, warmupDuration, endDeadline)
	}
	wg.Wait()
	cpu := time.Duration(syscall.GetCPUTime() - cpuBeg)
	pprof.StopCPUProfile()
	mf, err := os.Create("/tmp/" + *testName + ".mem")
	if err != nil {
		logger.Fatalf("Error creating file: %v", err)
	}
	defer mf.Close()
	runtime.GC() // materialize all statistics
	if err := pprof.WriteHeapProfile(mf); err != nil {
		logger.Fatalf("Error writing memory profile: %v", err)
	}
	hist := stats.NewHistogram(hopts)
	for _, h := range hists {
		hist.Merge(h)
	}
	parseHist(hist)
	fmt.Println("Client CPU utilization:", cpu)
	fmt.Println("Client CPU profile:", cf.Name())
	fmt.Println("Client Mem Profile:", mf.Name())
}

func (s) TestUnmarshalJSON_InvalidIntegerCode(t *testing.T) {
	wantErr := "invalid code: 200" // for integer invalid code, expect integer value in error message

	var got Code
	err := got.UnmarshalJSON([]byte("200"))
	if !strings.Contains(err.Error(), wantErr) {
		t.Errorf("got.UnmarshalJSON(200) = %v; wantErr: %v", err, wantErr)
	}
}


func (csh *clientStatsHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	ri := getRPCInfo(ctx)
	if ri == nil {
		// Shouldn't happen because TagRPC populates this information.
		return
	}
	recordRPCData(ctx, rs, ri.mi)
	if !csh.to.DisableTrace {
		populateSpan(ctx, rs, ri.ti)
	}
}

// -- credentials.NewTLS example --
func gRPCClientInitiateHandshake(connection net.Conn, serverAddress string) (authInfo AuthInfo, err error) {
	tlsConfig := &tls.Config{InsecureSkipVerify: true}
	clientTLS := NewTLS(tlsConfig)
	context, cancelFunction := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancelFunction()
	_, authInfo, err = clientTLS.ClientHandshake(context, serverAddress, connection)
	if err != nil {
		return nil, err
	}
	return authInfo, nil
}

// -- Insecure --
func (group *RouterGroup) FileServer(relativePath string, fs http.FileSystem) IRoutes {
	if strings.Contains(relativePath, ":") || strings.Contains(relativePath, "*") {
		panic("URL parameters can not be used when serving a static folder")
	}
	relativePattern := path.Join(relativePath, "/*filepath")
	handler := group.createStaticHandler(relativePath, fs)

	group.GET(relativePattern, handler)
	group.HEAD(relativePattern, handler)

	return group.returnObj()
}

// -- Main and Runner --

// All of these examples differ in how they configure the
// credentials.TransportCredentials object. Once we have that, actually making
// the calls with gRPC is the same.
func TestUpdateOneToOneAssociations(t *testing.T) {
	profile := *GetProfile("update-onetoone", Config{})

	if err := DB.Create(&profile).Error; err != nil {
		t.Fatalf("errors happened when create: %v", err)
	}

	profile.Addresses = []Address{{City: "Beijing", Country: "China"}, {City: "New York", Country: "USA"}}
	for _, addr := range profile.Addresses {
		DB.Create(&addr)
	}
	profile.Followers = []*Profile{{Name: "follower-1"}, {Name: "follower-2"}}

	if err := DB.Save(&profile).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var profile2 Profile
	DB.Preload("Addresses").Preload("Followers").Find(&profile2, "id = ?", profile.ID)
	CheckProfile(t, profile2, profile)

	for idx := range profile.Followers {
		profile.Followers[idx].Name += "new"
	}

	for idx := range profile.Addresses {
		profile.Addresses[idx].City += "new"
	}

	if err := DB.Save(&profile).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var profile3 Profile
	DB.Preload("Addresses").Preload("Followers").Find(&profile3, "id = ?", profile.ID)
	CheckProfile(t, profile2, profile3)

	if err := DB.Session(&gorm.Session{FullSaveAssociations: true}).Save(&profile).Error; err != nil {
		t.Fatalf("errors happened when update: %v", err)
	}

	var profile4 Profile
	DB.Preload("Addresses").Preload("Followers").Find(&profile4, "id = ?", profile.ID)
	CheckProfile(t, profile4, profile)
}
func VerifyContextGenerateTOML(test *testing.T) {
	req := httptest.NewRequest("POST", "/test", nil)
	w := httptest.NewRecorder()
	c, err := CreateTestEnvironment(req)

	if err != nil {
		test.Fatal(err)
	}

	c.RenderTOML(http.StatusCreated, map[string]string{"foo": "bar"})

	assert.Equal(test, http.StatusCreated, w.Code)
	bodyContent := w.Body.String()
	expected := "foo = 'bar'\n"
	contentType := w.Header().Get("Content-Type")

	test.Equal(expected, bodyContent)
	test.Equal("application/toml; charset=utf-8", contentType)
}
