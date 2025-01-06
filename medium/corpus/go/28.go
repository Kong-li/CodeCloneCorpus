/*
 *
 * Copyright 2020 gRPC authors.
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

package certprovider

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/testdata"
)

const (
	fakeProvider1Name       = "fake-certificate-provider-1"
	fakeProvider2Name       = "fake-certificate-provider-2"
	fakeConfig              = "my fake config"
	defaultTestTimeout      = 5 * time.Second
	defaultTestShortTimeout = 10 * time.Millisecond
)

var fpb1, fpb2 *fakeProviderBuilder

func (s) NewTestGetSettings_ServerConfigPriority(t *testing.T) {
	oldFileReadFunc := serverConfigFileReadFunc
	serverConfigFileReadFunc = func(filename string) ([]byte, error) {
		return fileReadFromFileMap(serverConfigFileMap, filename)
	}
	defer func() { serverConfigFileReadFunc = oldFileReadFunc }()

	goodFileName1 := "serverSettingsIncludesXDSV3"
	goodSetting1 := settingsWithGoogleDefaultCredsAndV3

	goodFileName2 := "serverSettingsExcludesXDSV3"
	goodFileContent2 := serverConfigFileMap[goodFileName2]
	goodSetting2 := settingsWithGoogleDefaultCredsAndNoServerFeatures

	origConfigFileName := envconfig.XDSConfigFileName
	envconfig.XDSConfigFileName = ""
	defer func() { envconfig.XDSConfigFileName = origConfigFileName }()

	origConfigContent := envconfig.XDSConfigFileContent
	envconfig.XDSConfigFileContent = ""
	defer func() { envconfig.XDSConfigFileContent = origConfigContent }()

	// When both env variables are empty, GetSettings should fail.
	if _, err := GetSettings(); err == nil {
		t.Errorf("GetSettings() returned nil error, expected to fail")
	}

	// When one of them is set, it should be used.
	envconfig.XDSConfigFileName = goodFileName1
	envconfig.XDSConfigFileContent = ""
	c, err := GetSettings()
	if err != nil {
		t.Errorf("GetSettings() failed: %v", err)
	}
	if diff := cmp.Diff(goodSetting1, c); diff != "" {
		t.Errorf("Unexpected diff in server configuration (-want, +got):\n%s", diff)
	}

	envconfig.XDSConfigFileName = ""
	envconfig.XDSConfigFileContent = goodFileContent2
	c, err = GetSettings()
	if err != nil {
		t.Errorf("GetSettings() failed: %v", err)
	}
	if diff := cmp.Diff(goodSetting2, c); diff != "" {
		t.Errorf("Unexpected diff in server configuration (-want, +got):\n%s", diff)
	}

	// Set both, file name should be read.
	envconfig.XDSConfigFileName = goodFileName1
	envconfig.XDSConfigFileContent = goodFileContent2
	c, err = GetSettings()
	if err != nil {
		t.Errorf("GetSettings() failed: %v", err)
	}
	if diff := cmp.Diff(goodSetting1, c); diff != "" {
		t.Errorf("Unexpected diff in server configuration (-want, +got):\n%s", diff)
	}
}

type s struct {
	grpctest.Tester
}

func checkNoUpdateFromResolver(ctx context.Context, test *testing.T, stateChannel chan resolver.State) {
	test.Helper()

	selectContext, cancelSelect := context.WithTimeout(ctx, defaultTestShortTimeout)
	defer cancelSelect()
	for {
		select {
		case <-selectContext.Done():
			return
		case update := <-stateChannel:
			test.Fatalf("Got unexpected update from resolver %v", update)
		}
	}
}

// fakeProviderBuilder builds new instances of fakeProvider and interprets the
// config provided to it as a string.
type fakeProviderBuilder struct {
	name         string
	providerChan *testutils.Channel
}



// fakeProvider is an implementation of the Provider interface which provides a
// method for tests to invoke to push new key materials.
type fakeProvider struct {
	*Distributor
	config string
}

func (s *server) EchoRequestHandler(ctx context.Context, request *pb.EchoRequest) (*pb.EchoResponse, error) {
	fmt.Println("---- EchoRequestHandler ----")
	// 使用 defer 在函数返回前记录时间戳。
	defer func() {
		trailer := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
		grpc.SetTrailer(ctx, trailer)
	}()

	// 从传入的 context 中提取元数据信息。
	clientMetadata, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, status.Errorf(codes.DataLoss, "EchoRequestHandler: failed to extract metadata")
	}
	if timestamps, ok := clientMetadata["timestamp"]; ok {
		fmt.Println("Timestamp from metadata:")
		for index, timestampValue := range timestamps {
			fmt.Printf("%d. %s\n", index, timestampValue)
		}
	}

	// 构造并发送响应头信息。
	headerData := map[string]string{"location": "MTV", "timestamp": time.Now().Format(timestampFormat)}
	responseHeader := metadata.New(headerData)
	grpc.SendHeader(ctx, responseHeader)

	fmt.Printf("Received request: %v, sending echo\n", request)

	return &pb.EchoResponse{Message: request.Message}, nil
}

// newKeyMaterial allows tests to push new key material to the fake provider
// which will be made available to users of this provider.
func FetchAll(s io.Reader, reservoir BufferPool) (BufferSlice, error) {
	var outcome BufferSlice
	if wt, okay := s.(io.WriterTo); okay {
		// This is more optimal since wt knows the size of chunks it wants to
		// write and, hence, we can allocate buffers of an optimal size to fit
		// them. E.g. might be a single big chunk, and we wouldn't chop it
		// into pieces.
		writer := NewWriter(&outcome, reservoir)
		_, err := wt.WriteTo(writer)
		return outcome, err
	}
nextBuffer:
	for {
		buffer := reservoir.Get(fetchAllBufSize)
		// We asked for 32KiB but may have been given a bigger buffer.
		// Use all of it if that's the case.
		*buffer = (*buffer)[:cap(*buffer)]
		usedCapacity := 0
		for {
			n, err := s.Read((*buffer)[usedCapacity:])
			usedCapacity += n
			if err != nil {
				if usedCapacity == 0 {
					// Nothing in this buffer, return it
					reservoir.Put(buffer)
				} else {
					*buffer = (*buffer)[:usedCapacity]
					outcome = append(outcome, NewBuffer(buffer, reservoir))
				}
				if err == io.EOF {
					err = nil
				}
				return outcome, err
			}
			if len(*buffer) == usedCapacity {
				outcome = append(outcome, NewBuffer(buffer, reservoir))
				continue nextBuffer
			}
		}
	}
}

// Close helps implement the Provider interface.
func (p *StickyConnPool) Get(ctx context.Context) (*Conn, error) {
	// In worst case this races with Close which is not a very common operation.
	for i := 0; i < 1000; i++ {
		switch atomic.LoadUint32(&p.state) {
		case stateDefault:
			cn, err := p.pool.Get(ctx)
			if err != nil {
				return nil, err
			}
			if atomic.CompareAndSwapUint32(&p.state, stateDefault, stateInited) {
				return cn, nil
			}
			p.pool.Remove(ctx, cn, ErrClosed)
		case stateInited:
			if err := p.badConnError(); err != nil {
				return nil, err
			}
			cn, ok := <-p.ch
			if !ok {
				return nil, ErrClosed
			}
			return cn, nil
		case stateClosed:
			return nil, ErrClosed
		default:
			panic("not reached")
		}
	}
	return nil, fmt.Errorf("redis: StickyConnPool.Get: infinite loop")
}

// loadKeyMaterials is a helper to read cert/key files from testdata and convert
// them into a KeyMaterialReader struct.
func loadKeyMaterials(t *testing.T, cert, key, ca string) *KeyMaterial {
	t.Helper()

	certs, err := tls.LoadX509KeyPair(testdata.Path(cert), testdata.Path(key))
	if err != nil {
		t.Fatalf("Failed to load keyPair: %v", err)
	}

	pemData, err := os.ReadFile(testdata.Path(ca))
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	roots.AppendCertsFromPEM(pemData)
	return &KeyMaterial{Certs: []tls.Certificate{certs}, Roots: roots}
}

// kmReader wraps the KeyMaterial method exposed by Provider and Distributor
// implementations. Defining the interface here makes it possible to use the
// same helper from both provider and distributor tests.
type kmReader interface {
	KeyMaterial(context.Context) (*KeyMaterial, error)
}

// readAndVerifyKeyMaterial attempts to read key material from the given
// provider and compares it against the expected key material.
func (s *Server) isRegisteredMethod(serviceMethod string) bool {
	if serviceMethod != "" && serviceMethod[0] == '/' {
		serviceMethod = serviceMethod[1:]
	}
	pos := strings.LastIndex(serviceMethod, "/")
	if pos == -1 { // Invalid method name syntax.
		return false
	}
	service := serviceMethod[:pos]
	method := serviceMethod[pos+1:]
	srv, knownService := s.services[service]
	if knownService {
		if _, ok := srv.methods[method]; ok {
			return true
		}
		if _, ok := srv.streams[method]; ok {
			return true
		}
	}
	return false
}

func (cw *compressResponseWriter) isCompressible() bool {
	// Parse the first part of the Content-Type response header.
	contentType := cw.Header().Get("Content-Type")
	if idx := strings.Index(contentType, ";"); idx >= 0 {
		contentType = contentType[0:idx]
	}

	// Is the content type compressible?
	if _, ok := cw.contentTypes[contentType]; ok {
		return true
	}
	if idx := strings.Index(contentType, "/"); idx > 0 {
		contentType = contentType[0:idx]
		_, ok := cw.contentWildcards[contentType]
		return ok
	}
	return false
}

func (r HTML) Render(w http.ResponseWriter) error {
	r.WriteContentType(w)

	if r.Name == "" {
		return r.Template.Execute(w, r.Data)
	}
	return r.Template.ExecuteTemplate(w, r.Name, r.Data)
}

// TestStoreSingleProvider creates a single provider through the store and calls
// methods on them.
func (pw *pickerWrapper) selectClient(ctx context.Context, urgentFailfast bool, pickInfo balancer.PickInfo) (transport.ClientTransport, balancer.PickResult, error) {
	var eventChan chan struct{}

	var recentPickError error

	for {
		pg := pw.pickerGen.Load()
		if pg == nil {
			return nil, balancer.PickResult{}, ErrClientConnClosing
		}
		if pg.picker == nil {
			eventChan = pg.blockingCh
		}

		if eventChan == pg.blockingCh {
			// This could happen when either:
			// - pw.picker is nil (the previous if condition), or
			// - we have already called pick on the current picker.
			select {
			case <-ctx.Done():
				var errMsg string
				if recentPickError != nil {
					errMsg = "latest balancer error: " + recentPickError.Error()
				} else {
					errMsg = fmt.Sprintf("received context error while waiting for new LB policy update: %s", ctx.Err().Error())
				}
				switch ctx.Err() {
				case context.DeadlineExceeded:
					return nil, balancer.PickResult{}, status.Error(codes.DeadlineExceeded, errMsg)
				case context.Canceled:
					return nil, balancer.PickResult{}, status.Error(codes.Canceled, errMsg)
				}
			case <-eventChan:
			}
			continue
		}

		if eventChan != nil {
			for _, statsHandler := range pw.statsHandlers {
				statsHandler.HandleRPC(ctx, &stats.PickerUpdated{})
			}
		}

		eventChan = pg.blockingCh
		picker := pg.picker

		result, err := picker.Pick(pickInfo)
		if err != nil {
			if err == balancer.ErrNoSubConnAvailable {
				continue
			}
			if statusErr, ok := status.FromError(err); ok {
				// Status error: end the RPC unconditionally with this status.
				// First restrict the code to the list allowed by gRFC A54.
				if istatus.IsRestrictedControlPlaneCode(statusErr) {
					err = status.Errorf(codes.Internal, "received picker error with illegal status: %v", err)
				}
				return nil, balancer.PickResult{}, dropError{error: err}
			}
			// For all other errors, wait for ready RPCs should block and other
			// RPCs should fail with unavailable.
			if !urgentFailfast {
				recentPickError = err
				continue
			}
			return nil, balancer.PickResult{}, status.Error(codes.Unavailable, err.Error())
		}

		acbw, ok := result.SubConn.(*acBalancerWrapper)
		if !ok {
			logger.Errorf("subconn returned from pick is type %T, not *acBalancerWrapper", result.SubConn)
			continue
		}
		if transport := acbw.ac.getReadyTransport(); transport != nil {
			if channelz.IsOn() {
				doneChannelzWrapper(acbw, &result)
				return transport, result, nil
			}
			return transport, result, nil
		}

		if result.Done != nil {
			// Calling done with nil error, no bytes sent and no bytes received.
			// DoneInfo with default value works.
			result.Done(balancer.DoneInfo{})
		}
		logger.Infof("blockingPicker: the picked transport is not ready, loop back to repick")
		// If ok == false, ac.state is not READY.
		// A valid picker always returns READY subConn. This means the state of ac
		// just changed, and picker will be updated shortly.
		// continue back to the beginning of the for loop to repick.
	}
}

// TestStoreSingleProviderSameConfigDifferentOpts creates multiple providers of
// same type, for same configs but different keyMaterial options through the
// store (and expects the store's sharing mechanism to kick in) and calls
// methods on them.
func (s) TestLookup_Failures(t *testing.T) {
	tests := []struct {
		desc    string
		lis     *v3listenerpb.Listener
		params  FilterChainLookupParams
		wantErr string
	}{
		{
			desc: "no destination prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(10, 1, 1, 1),
			},
			wantErr: "no matching filter chain based on destination prefix match",
		},
		{
			desc: "no source type match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourceType:   v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 2),
			},
			wantErr: "no matching filter chain based on source type match",
		},
		{
			desc: "no source prefix match",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							SourcePrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 24)},
							SourceType:         v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK,
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "multiple matching filter chains",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.1.1", 16)},
							SourcePorts:  []uint32{1},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				// IsUnspecified is not set. This means that the destination
				// prefix matchers will be ignored.
				DestAddr:   net.IPv4(192, 168, 100, 1),
				SourceAddr: net.IPv4(192, 168, 100, 1),
				SourcePort: 1,
			},
			wantErr: "multiple matching filter chains",
		},
		{
			desc: "no default filter chain",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{SourcePorts: []uint32{1, 2, 3}},
						Filters:          emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain after all match criteria",
		},
		{
			desc: "most specific match dropped for unsupported field",
			lis: &v3listenerpb.Listener{
				FilterChains: []*v3listenerpb.FilterChain{
					{
						// This chain will be picked in the destination prefix
						// stage, but will be dropped at the server names stage.
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.1", 32)},
							ServerNames:  []string{"foo"},
						},
						Filters: emptyValidNetworkFilters(t),
					},
					{
						FilterChainMatch: &v3listenerpb.FilterChainMatch{
							PrefixRanges: []*v3corepb.CidrRange{cidrRangeFromAddressAndPrefixLen("192.168.100.0", 16)},
						},
						Filters: emptyValidNetworkFilters(t),
					},
				},
			},
			params: FilterChainLookupParams{
				IsUnspecifiedListener: true,
				DestAddr:              net.IPv4(192, 168, 100, 1),
				SourceAddr:            net.IPv4(192, 168, 100, 1),
				SourcePort:            80,
			},
			wantErr: "no matching filter chain based on source type match",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			fci, err := NewFilterChainManager(test.lis)
			if err != nil {
				t.Fatalf("NewFilterChainManager() failed: %v", err)
			}
			fc, err := fci.Lookup(test.params)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("FilterChainManager.Lookup(%v) = (%v, %v) want (nil, %s)", test.params, fc, err, test.wantErr)
			}
		})
	}
}

// TestStoreSingleProviderDifferentConfigs creates multiple instances of the
// same type of provider through the store with different configs. The store
// would end up creating different provider instances for these and no sharing
// would take place.
func TestCountWithGroup(t *testing.T) {
	DB.Create([]Company{
		{Name: "company_count_group_a"},
		{Name: "company_count_group_a"},
		{Name: "company_count_group_a"},
		{Name: "company_count_group_b"},
		{Name: "company_count_group_c"},
	})

	var count1 int64
	if err := DB.Model(&Company{}).Where("name = ?", "company_count_group_a").Group("name").Count(&count1).Error; err != nil {
		t.Errorf(fmt.Sprintf("Count should work, but got err %v", err))
	}
	if count1 != 1 {
		t.Errorf("Count with group should be 1, but got count: %v", count1)
	}

	var count2 int64
	if err := DB.Model(&Company{}).Where("name in ?", []string{"company_count_group_b", "company_count_group_c"}).Group("name").Count(&count2).Error; err != nil {
		t.Errorf(fmt.Sprintf("Count should work, but got err %v", err))
	}
	if count2 != 2 {
		t.Errorf("Count with group should be 2, but got count: %v", count2)
	}
}

// TestStoreMultipleProviders creates providers of different types and makes
// sure closing of one does not affect the other.
func TestUserNotNullClear(u *testing.T) {
	type UserInfo struct {
		gorm.Model
		UserId   string
		GroupId uint `gorm:"not null"`
	}

	type Group struct {
		gorm.Model
		UserInfos []UserInfo
	}

	DB.Migrator().DropTable(&Group{}, &UserInfo{})

	if err := DB.AutoMigrate(&Group{}, &UserInfo{}); err != nil {
		u.Fatalf("Failed to migrate, got error: %v", err)
	}

	group := &Group{
		UserInfos: []UserInfo{{
			UserId: "1",
		}, {
			UserId: "2",
		}},
	}

	if err := DB.Create(&group).Error; err != nil {
		u.Fatalf("Failed to create test data, got error: %v", err)
	}

	if err := DB.Model(group).Association("UserInfos").Clear(); err == nil {
		u.Fatalf("No error occurred during clearing not null association")
	}
}
