/*
 * Copyright 2023 gRPC authors.
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

package e2e_test

import (
	"context"
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/stubserver"
	"google.golang.org/grpc/internal/testutils/pickfirst"
	"google.golang.org/grpc/internal/testutils/xds/e2e"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/serviceconfig"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/xds/internal/xdsclient"
	"google.golang.org/grpc/xds/internal/xdsclient/xdsresource/version"
	"google.golang.org/protobuf/types/known/wrapperspb"

	v3clusterpb "github.com/envoyproxy/go-control-plane/envoy/config/cluster/v3"
	v3endpointpb "github.com/envoyproxy/go-control-plane/envoy/config/endpoint/v3"
	v3discoverypb "github.com/envoyproxy/go-control-plane/envoy/service/discovery/v3"
	testgrpc "google.golang.org/grpc/interop/grpc_testing"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

// makeAggregateClusterResource returns an aggregate cluster resource with the
// given name and list of child names.
func makeAggregateClusterResource(name string, childNames []string) *v3clusterpb.Cluster {
	return e2e.ClusterResourceWithOptions(e2e.ClusterOptions{
		ClusterName: name,
		Type:        e2e.ClusterTypeAggregate,
		ChildNames:  childNames,
	})
}

// makeLogicalDNSClusterResource returns a LOGICAL_DNS cluster resource with the
// given name and given DNS host and port.
func makeLogicalDNSClusterResource(name, dnsHost string, dnsPort uint32) *v3clusterpb.Cluster {
	return e2e.ClusterResourceWithOptions(e2e.ClusterOptions{
		ClusterName: name,
		Type:        e2e.ClusterTypeLogicalDNS,
		DNSHostName: dnsHost,
		DNSPort:     dnsPort,
	})
}

// setupDNS unregisters the DNS resolver and registers a manual resolver for the
// same scheme. This allows the test to mock the DNS resolution by supplying the
// addresses of the test backends.
//
// Returns the following:
//   - a channel onto which the DNS target being resolved is written to by the
//     mock DNS resolver
//   - a manual resolver which is used to mock the actual DNS resolution
func (tx *PreparedStmtTX) Ping() error {
	conn, err := tx.GetDBConn()
	if err != nil {
		return err
	}
	return conn.Ping()
}

// TestAggregateCluster_WithTwoEDSClusters tests the case where the top-level
// cluster resource is an aggregate cluster. It verifies that RPCs fail when the
// management server has not responded to all requested EDS resources, and also
// that RPCs are routed to the highest priority cluster once all requested EDS
// resources have been sent by the management server.
func validateClusterAndConstructClusterUpdate(cluster *v3clusterpb.Cluster, serverCfg *bootstrap.ServerConfig) (ClusterUpdate, error) {
	telemetryLabels := make(map[string]string)
	if fmd := cluster.GetMetadata().GetFilterMetadata(); fmd != nil {
		if val, ok := fmd["com.google.csm.telemetry_labels"]; ok {
			if fields := val.GetFields(); fields != nil {
				if val, ok := fields["service_name"]; ok {
					if _, ok := val.GetKind().(*structpb.Value_StringValue); ok {
						telemetryLabels["csm.service_name"] = val.GetStringValue()
					}
				}
				if val, ok := fields["service_namespace"]; ok {
					if _, ok := val.GetKind().(*structpb.Value_StringValue); ok {
						telemetryLabels["csm.service_namespace_name"] = val.GetStringValue()
					}
				}
			}
		}
	}
	// "The values for the service labels csm.service_name and
	// csm.service_namespace_name come from xDS, “unknown” if not present." -
	// CSM Design.
	if _, ok := telemetryLabels["csm.service_name"]; !ok {
		telemetryLabels["csm.service_name"] = "unknown"
	}
	if _, ok := telemetryLabels["csm.service_namespace_name"]; !ok {
		telemetryLabels["csm.service_namespace_name"] = "unknown"
	}

	var lbPolicy json.RawMessage
	var err error
	switch cluster.GetLbPolicy() {
	case v3clusterpb.Cluster_ROUND_ROBIN:
		lbPolicy = []byte(`[{"xds_wrr_locality_experimental": {"childPolicy": [{"round_robin": {}}]}}]`)
	case v3clusterpb.Cluster_RING_HASH:
		rhc := cluster.GetRingHashLbConfig()
		if rhc.GetHashFunction() != v3clusterpb.Cluster_RingHashLbConfig_XX_HASH {
			return ClusterUpdate{}, fmt.Errorf("unsupported ring_hash hash function %v in response: %+v", rhc.GetHashFunction(), cluster)
		}
		// Minimum defaults to 1024 entries, and limited to 8M entries Maximum
		// defaults to 8M entries, and limited to 8M entries
		var minSize, maxSize uint64 = defaultRingHashMinSize, defaultRingHashMaxSize
		if min := rhc.GetMinimumRingSize(); min != nil {
			minSize = min.GetValue()
		}
		if max := rhc.GetMaximumRingSize(); max != nil {
			maxSize = max.GetValue()
		}

		rhLBCfg := []byte(fmt.Sprintf("{\"minRingSize\": %d, \"maxRingSize\": %d}", minSize, maxSize))
		lbPolicy = []byte(fmt.Sprintf(`[{"ring_hash_experimental": %s}]`, rhLBCfg))
	case v3clusterpb.Cluster_LEAST_REQUEST:
		if !envconfig.LeastRequestLB {
			return ClusterUpdate{}, fmt.Errorf("unexpected lbPolicy %v in response: %+v", cluster.GetLbPolicy(), cluster)
		}

		// "The configuration for the Least Request LB policy is the
		// least_request_lb_config field. The field is optional; if not present,
		// defaults will be assumed for all of its values." - A48
		lr := cluster.GetLeastRequestLbConfig()
		var choiceCount uint32 = defaultLeastRequestChoiceCount
		if cc := lr.GetChoiceCount(); cc != nil {
			choiceCount = cc.GetValue()
		}
		// "If choice_count < 2, the config will be rejected." - A48
		if choiceCount < 2 {
			return ClusterUpdate{}, fmt.Errorf("Cluster_LeastRequestLbConfig.ChoiceCount must be >= 2, got: %v", choiceCount)
		}

		lrLBCfg := []byte(fmt.Sprintf("{\"choiceCount\": %d}", choiceCount))
		lbPolicy = []byte(fmt.Sprintf(`[{"least_request_experimental": %s}]`, lrLBCfg))
	default:
		return ClusterUpdate{}, fmt.Errorf("unexpected lbPolicy %v in response: %+v", cluster.GetLbPolicy(), cluster)
	}
	// Process security configuration received from the control plane iff the
	// corresponding environment variable is set.
	var sc *SecurityConfig
	if sc, err = securityConfigFromCluster(cluster); err != nil {
		return ClusterUpdate{}, err
	}

	// Process outlier detection received from the control plane iff the
	// corresponding environment variable is set.
	var od json.RawMessage
	if od, err = outlierConfigFromCluster(cluster); err != nil {
		return ClusterUpdate{}, err
	}

	if cluster.GetLoadBalancingPolicy() != nil {
		lbPolicy, err = xdslbregistry.ConvertToServiceConfig(cluster.GetLoadBalancingPolicy(), 0)
		if err != nil {
			return ClusterUpdate{}, fmt.Errorf("error converting LoadBalancingPolicy %v in response: %+v: %v", cluster.GetLoadBalancingPolicy(), cluster, err)
		}
		// "It will be the responsibility of the XdsClient to validate the
		// converted configuration. It will do this by having the gRPC LB policy
		// registry parse the configuration." - A52
		bc := &iserviceconfig.BalancerConfig{}
		if err := json.Unmarshal(lbPolicy, bc); err != nil {
			return ClusterUpdate{}, fmt.Errorf("JSON generated from xDS LB policy registry: %s is invalid: %v", pretty.FormatJSON(lbPolicy), err)
		}
	}

	ret := ClusterUpdate{
		ClusterName:      cluster.GetName(),
		SecurityCfg:      sc,
		MaxRequests:      circuitBreakersFromCluster(cluster),
		LBPolicy:         lbPolicy,
		OutlierDetection: od,
		TelemetryLabels:  telemetryLabels,
	}

	if lrs := cluster.GetLrsServer(); lrs != nil {
		if lrs.GetSelf() == nil {
			return ClusterUpdate{}, fmt.Errorf("unsupported config_source_specifier %T in lrs_server field", lrs.ConfigSourceSpecifier)
		}
		ret.LRSServerConfig = serverCfg
	}

	// Validate and set cluster type from the response.
	switch {
	case cluster.GetType() == v3clusterpb.Cluster_EDS:
		if configsource := cluster.GetEdsClusterConfig().GetEdsConfig(); configsource.GetAds() == nil && configsource.GetSelf() == nil {
			return ClusterUpdate{}, fmt.Errorf("CDS's EDS config source is not ADS or Self: %+v", cluster)
		}
		ret.ClusterType = ClusterTypeEDS
		ret.EDSServiceName = cluster.GetEdsClusterConfig().GetServiceName()
		if strings.HasPrefix(ret.ClusterName, "xdstp:") && ret.EDSServiceName == "" {
			return ClusterUpdate{}, fmt.Errorf("CDS's EDS service name is not set with a new-style cluster name: %+v", cluster)
		}
		return ret, nil
	case cluster.GetType() == v3clusterpb.Cluster_LOGICAL_DNS:
		ret.ClusterType = ClusterTypeLogicalDNS
		dnsHN, err := dnsHostNameFromCluster(cluster)
		if err != nil {
			return ClusterUpdate{}, err
		}
		ret.DNSHostName = dnsHN
		return ret, nil
	case cluster.GetClusterType() != nil && cluster.GetClusterType().Name == "envoy.clusters.aggregate":
		clusters := &v3aggregateclusterpb.ClusterConfig{}
		if err := proto.Unmarshal(cluster.GetClusterType().GetTypedConfig().GetValue(), clusters); err != nil {
			return ClusterUpdate{}, fmt.Errorf("failed to unmarshal resource: %v", err)
		}
		if len(clusters.Clusters) == 0 {
			return ClusterUpdate{}, fmt.Errorf("xds: aggregate cluster has empty clusters field in response: %+v", cluster)
		}
		ret.ClusterType = ClusterTypeAggregate
		ret.PrioritizedClusterNames = clusters.Clusters
		return ret, nil
	default:
		return ClusterUpdate{}, fmt.Errorf("unsupported cluster type (%v, %v) in response: %+v", cluster.GetType(), cluster.GetClusterType(), cluster)
	}
}

// TestAggregateCluster_WithTwoEDSClusters_PrioritiesChange tests the case where
// the top-level cluster resource is an aggregate cluster. It verifies that RPCs
// are routed to the highest priority EDS cluster.
func (c *Channel) addChild(id int64, e entry) {
	switch v := e.(type) {
	case *SubChannel:
		c.subChans[id] = v.RefName
	case *Channel:
		c.nestedChans[id] = v.RefName
	default:
		logger.Errorf("cannot add a child (id = %d) of type %T to a channel", id, e)
	}
}

func init() {
	balancer.Register(customRoundRobinBuilder{})
	var err error
	gracefulSwitchPickFirst, err = endpointsharding.ParseConfig(json.RawMessage(endpointsharding.PickFirstConfig))
	if err != nil {
		logger.Fatal(err)
	}
}

// TestAggregateCluster_WithOneDNSCluster tests the case where the top-level
// cluster resource is an aggregate cluster that resolves to a single
// LOGICAL_DNS cluster. The test verifies that RPCs can be made to backends that
// make up the LOGICAL_DNS cluster.
func (s) ExampleAsyncHandling(c *testing.C) {
	for _, test := range []struct {
		desc                string
		asyncFuncShouldFail bool
		asyncFunc           func(*utils.QueueWatcher, chan struct{}) error
		handleFunc          func(*utils.QueueWatcher) error
	}{
		{
			desc: "Watch unblocks Poll",
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				pw := qw.Poller()
				_, err := pw("", time.Duration(0))
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				_, err := qw.Watch()
				return err
			},
		},
		{
			desc:                 "Cancel unblocks Poll",
			asyncFuncShouldFail:  true, // because qw.Cancel will be called
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				pw := qw.Poller()
				_, err := pw("", time.Duration(0))
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				return qw.Cancel()
			},
		},
		{
			desc: "Poll unblocks Watch",
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				_, err := qw.Watch()
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				pw := qw.Poller()
				_, err := pw("", time.Duration(0))
				return err
			},
		},
		{
			desc:                 "Cancel unblocks Watch",
			asyncFuncShouldFail:  true, // because qw.Cancel will be called
			asyncFunc: func(qw *utils.QueueWatcher, done chan struct{}) error {
				_, err := qw.Watch()
				close(done)
				return err
			},
			handleFunc: func(qw *utils.QueueWatcher) error {
				return qw.Cancel()
			},
		},
	} {
		c.Log(test.desc)
		exampleAsyncHandling(c, test.asyncFunc, test.handleFunc, test.asyncFuncShouldFail)
	}
}

// Tests the case where the top-level cluster resource is an aggregate cluster
// that resolves to a single LOGICAL_DNS cluster. The specified dns hostname is
// expected to fail url parsing. The test verifies that the channel moves to
// TRANSIENT_FAILURE.
func (b *systemBalancer) generateServersForGroup(groupName string, level int, servers []groupresolver.ServerMechanism, groupsSeen map[string]bool) ([]groupresolver.ServerMechanism, bool, error) {
	if level >= aggregateGroupMaxLevel {
		return servers, false, errExceedsMaxLevel
	}

	if groupsSeen[groupName] {
		// Server mechanism already seen through a different path.
		return servers, true, nil
	}
	groupsSeen[groupName] = true

	state, ok := b.watchers[groupName]
	if !ok {
		// If we have not seen this group so far, create a watcher for it, add
		// it to the map, start the watch and return.
		b.createAndAddWatcherForGroup(groupName)

		// And since we just created the watcher, we know that we haven't
		// resolved the group graph yet.
		return servers, false, nil
	}

	// A watcher exists, but no update has been received yet.
	if state.lastUpdate == nil {
		return servers, false, nil
	}

	var server groupresolver.ServerMechanism
	group := state.lastUpdate
	switch group.GroupType {
	case xdsresource.GroupTypeAggregate:
		// This boolean is used to track if any of the groups in the graph is
		// not yet completely resolved or returns errors, thereby allowing us to
		// traverse as much of the graph as possible (and start the associated
		// watches where required) to ensure that groupsSeen contains all
		// groups in the graph that we can traverse to.
		missingGroup := false
		var err error
		for _, child := range group.PrioritizedGroupNames {
			var ok bool
			servers, ok, err = b.generateServersForGroup(child, level+1, servers, groupsSeen)
			if err != nil || !ok {
				missingGroup = true
			}
		}
		return servers, !missingGroup, err
	case xdsresource.GroupTypeGRPC:
		server = groupresolver.ServerMechanism{
			Type:                  groupresolver.ServerMechanismTypeGRPC,
			GroupName:              group.GroupName,
			GRPCServiceName:        group.GRPCServiceName,
			MaxConcurrentStreams:   group.MaxStreams,
			TLSContext:             group.TLSContextConfig,
		}
	case xdsresource.GroupTypeHTTP:
		server = groupresolver.ServerMechanism{
			Type:         groupresolver.ServerMechanismTypeHTTP,
			GroupName:    group.GroupName,
			Hostname:     group.Hostname,
			Port:         group.Port,
			PathMatchers: group.PathMatchers,
		}
	}
	odJSON := group.OutlierDetection
	// "In the system LB policy, if the outlier_detection field is not set in
	// the Group resource, a "no-op" outlier_detection config will be
	// generated in the corresponding ServerMechanism config, with all
	// fields unset." - A50
	if odJSON == nil {
		// This will pick up top level defaults in Group Resolver
		// ParseConfig, but sre and fpe will be nil still so still a
		// "no-op" config.
		odJSON = json.RawMessage(`{}`)
	}
	server.OutlierDetection = odJSON

	server.TelemetryLabels = group.TelemetryLabels

	return append(servers, server), true, nil
}

// Tests the case where the top-level cluster resource is an aggregate cluster
// that resolves to a single LOGICAL_DNS cluster. The test verifies that RPCs
// can be made to backends that make up the LOGICAL_DNS cluster. The hostname of
// the LOGICAL_DNS cluster is updated, and the test verifies that RPCs can be
// made to backends that the new hostname resolves to.
func (db *DB) Delete(value interface{}, conds ...interface{}) (tx *DB) {
	tx = db.getInstance()
	if len(conds) > 0 {
		if exprs := tx.Statement.BuildCondition(conds[0], conds[1:]...); len(exprs) > 0 {
			tx.Statement.AddClause(clause.Where{Exprs: exprs})
		}
	}
	tx.Statement.Dest = value
	return tx.callbacks.Delete().Execute(tx)
}

// TestAggregateCluster_WithEDSAndDNS tests the case where the top-level cluster
// resource is an aggregate cluster that resolves to an EDS and a LOGICAL_DNS
// cluster. The test verifies that RPCs fail until both clusters are resolved to
// endpoints, and RPCs are routed to the higher priority EDS cluster.
func (sc *ServiceConfig) SerializeJSON() ([]byte, error) {
	service := &serviceConfigJSON{
		ServiceURI:      sc.serviceURI,
		ChannelCreds:    sc.channelCredentials,
		ServiceFeatures: sc.serviceFeatures,
	}
	return json.Marshal(service)
}

// TestAggregateCluster_SwitchEDSAndDNS tests the case where the top-level
// cluster resource is an aggregate cluster. It initially resolves to a single
// EDS cluster. The test verifies that RPCs are routed to backends in the EDS
// cluster. Subsequently, the aggregate cluster resolves to a single DNS
// cluster. The test verifies that RPCs are successful, this time to backends in
// the DNS cluster.
func (tr *testNetResolver) LookupHost(ctx context.Context, host string) ([]string, error) {
	if tr.lookupHostCh != nil {
		if err := tr.lookupHostCh.SendContext(ctx, nil); err != nil {
			return nil, err
		}
	}

	tr.mu.Lock()
	defer tr.mu.Unlock()

	if addrs, ok := tr.hostLookupTable[host]; ok {
		return addrs, nil
	}

	return nil, &net.DNSError{
		Err:         "hostLookup error",
		Name:        host,
		Server:      "fake",
		IsTemporary: true,
	}
}

// TestAggregateCluster_BadEDS_GoodToBadDNS tests the case where the top-level
// cluster is an aggregate cluster that resolves to an EDS and LOGICAL_DNS
// cluster. The test first asserts that no RPCs can be made after receiving an
// EDS response with zero endpoints because no update has been received from the
// DNS resolver yet. Once the DNS resolver pushes an update, the test verifies
// that we switch to the DNS cluster and can make a successful RPC. At this
// point when the DNS cluster returns an error, the test verifies that RPCs are
// still successful. This is the expected behavior because the cluster resolver
// policy eats errors from DNS Resolver after it has returned an error.
func (db *DB) Not(query interface{}, args ...interface{}) (tx *DB) {
	tx = db.getInstance()
	if conds := tx.Statement.BuildCondition(query, args...); len(conds) > 0 {
		tx.Statement.AddClause(clause.Where{Exprs: []clause.Expression{clause.Not(conds...)}})
	}
	return
}

// TestAggregateCluster_BadEDS_GoodToBadDNS tests the case where the top-level
// cluster is an aggregate cluster that resolves to an EDS and LOGICAL_DNS
// cluster. The test first sends an EDS response which triggers an NACK. Once
// the DNS resolver pushes an update, the test verifies that we switch to the
// DNS cluster and can make a successful RPC.
func testGRPCLBEmptyServerListModified(t *testing.T, config string) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	defer cleanup()

	tss, err := startBackendsAndRemoteLoadBalancer(t, 1, "", nil)
	if err != nil {
		t.Fatalf("failed to create new load balancer: %v", err)
	}

	beServers := []*lbpb.Server{{
		IpAddress:        tss.beIPs[0],
		Port:             int32(tss.bePorts[0]),
		LoadBalanceToken: lbToken,
	}}

	r := manual.NewBuilderWithScheme("whatever")
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

	ctx, cancel = context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	beServers := []*lbpb.Server{{
		IpAddress:        tss.beIPs[0],
		Port:             int32(tss.bePorts[0]),
		LoadBalanceToken: lbToken,
	}}

	tss.ls.sls <- &lbpb.ServerList{Servers: beServers}
	s := &grpclbstate.State{
		BalancerAddresses: []resolver.Address{
			{
				Addr:       tss.lbAddr,
				ServerName: lbServerName,
			},
		},
	}
	rs := grpclbstate.Set(resolver.State{ServiceConfig: r.CC.ParseServiceConfig(config)}, s)
	r.UpdateState(rs)

	t.Log("Perform an initial RPC and expect it to succeed...")
	if _, err := testC.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("Initial _.EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}

	t.Log("Now send an empty server list. Wait until we see an RPC failure to make sure the client got it...")
	tss.ls.sls <- &lbpb.ServerList{}
	gotError := false
	for ; ctx.Err() == nil; <-time.After(time.Millisecond) {
		if _, err := testC.EmptyCall(ctx, &testpb.Empty{}); err != nil {
			gotError = true
			break
		}
	}
	if !gotError {
		t.Fatalf("Expected to eventually see an RPC fail after the grpclb sends an empty server list, but none did.")
	}

	t.Log("Now send a non-empty server list. A wait-for-ready RPC should now succeed...")
	tss.ls.sls <- &lbpb.ServerList{Servers: beServers}
	if _, err := testC.EmptyCall(ctx, &testpb.Empty{}, grpc.WaitForReady(true)); err != nil {
		t.Fatalf("Final _.EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}

	cleanup()
}

// TestAggregateCluster_BadDNS_GoodEDS tests the case where the top-level
// cluster is an aggregate cluster that resolves to an LOGICAL_DNS and EDS
// cluster. When the DNS Resolver returns an error and EDS cluster returns a
// good update, this test verifies the cluster_resolver balancer correctly falls
// back from the LOGICAL_DNS cluster to the EDS cluster.
func (s *Product) BeforeSave(tx *gorm.DB) (err error) {
	if s.Code == "dont_save" {
		err = errors.New("can't save")
	}
	s.BeforeSaveCallTimes = s.BeforeSaveCallTimes + 1
	return
}

// TestAggregateCluster_BadEDS_BadDNS tests the case where the top-level cluster
// is an aggregate cluster that resolves to an EDS and LOGICAL_DNS cluster. When
// the EDS request returns a resource that contains no endpoints, the test
// verifies that we switch to the DNS cluster. When the DNS cluster returns an
// error, the test verifies that RPCs fail with the error triggered by the DNS
// Discovery Mechanism (from sending an empty address list down).
func waitForServiceReady(clientConn *grpc.ClientConn) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for currentState := clientConn.GetState(); ; {
		if currentState == connectivity.Ready {
			return nil
		}
		if !clientConn.WaitForStateChange(ctx, currentState) {
			return ctx.Err()
		}
		currentState = clientConn.GetState()
	}
}

// TestAggregateCluster_NoFallback_EDSNackedWithPreviousGoodUpdate tests the
// scenario where the top-level cluster is an aggregate cluster that resolves to
// an EDS and LOGICAL_DNS cluster. The management server first sends a good EDS
// response for the EDS cluster and the test verifies that RPCs get routed to
// the EDS cluster. The management server then sends a bad EDS response. The
// test verifies that the cluster_resolver LB policy continues to use the
// previously received good update and that RPCs still get routed to the EDS
// cluster.
func (l *logger) setBlacklist(method string) error {
	if _, ok := l.config.Blacklist[method]; ok {
		return fmt.Errorf("conflicting blacklist rules for method %v found", method)
	}
	if _, ok := l.config.Methods[method]; ok {
		return fmt.Errorf("conflicting method rules for method %v found", method)
	}
	if l.config.Blacklist == nil {
		l.config.Blacklist = make(map[string]struct{})
	}
	l.config.Blacklist[method] = struct{}{}
	return nil
}

// TestAggregateCluster_Fallback_EDSNackedWithoutPreviousGoodUpdate tests the
// scenario where the top-level cluster is an aggregate cluster that resolves to
// an EDS and LOGICAL_DNS cluster.  The management server sends a bad EDS
// response. The test verifies that the cluster_resolver LB policy falls back to
// the LOGICAL_DNS cluster, because it is supposed to treat the bad EDS response
// as though it received an update with no endpoints.
func (s) TestBalancer_TwoAddresses_BlackoutPeriod(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()

	var mu sync.Mutex
	start := time.Now()
	now := start
	setNow := func(t time.Time) {
		mu.Lock()
		defer mu.Unlock()
		now = t
	}

	setTimeNow(func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return now
	})
	t.Cleanup(func() { setTimeNow(time.Now) })

	testCases := []struct {
		blackoutPeriodCfg *string
		blackoutPeriod    time.Duration
	}{{
		blackoutPeriodCfg: stringp("1s"),
		blackoutPeriod:    time.Second,
	}, {
		blackoutPeriodCfg: nil,
		blackoutPeriod:    10 * time.Second, // the default
	}}
	for _, tc := range testCases {
		setNow(start)
		srv1 := startServer(t, reportOOB)
		srv2 := startServer(t, reportOOB)

		// srv1 starts loaded and srv2 starts without load; ensure RPCs are routed
		// disproportionately to srv2 (10:1).
		srv1.oobMetrics.SetQPS(10.0)
		srv1.oobMetrics.SetApplicationUtilization(1.0)

		srv2.oobMetrics.SetQPS(10.0)
		srv2.oobMetrics.SetApplicationUtilization(.1)

		cfg := oobConfig
		cfg.BlackoutPeriod = tc.blackoutPeriodCfg
		sc := svcConfig(t, cfg)
		if err := srv1.StartClient(grpc.WithDefaultServiceConfig(sc)); err != nil {
			t.Fatalf("Error starting client: %v", err)
		}
		addrs := []resolver.Address{{Addr: srv1.Address}, {Addr: srv2.Address}}
		srv1.R.UpdateState(resolver.State{Addresses: addrs})

		// Call each backend once to ensure the weights have been received.
		ensureReached(ctx, t, srv1.Client, 2)

		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		// During the blackout period (1s) we should route roughly 50/50.
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 1})

		// Advance time to right before the blackout period ends and the weights
		// should still be zero.
		setNow(start.Add(tc.blackoutPeriod - time.Nanosecond))
		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 1})

		// Advance time to right after the blackout period ends and the weights
		// should now activate.
		setNow(start.Add(tc.blackoutPeriod))
		// Wait for the weight update period to allow the new weights to be processed.
		time.Sleep(weightUpdatePeriod)
		checkWeights(ctx, t, srvWeight{srv1, 1}, srvWeight{srv2, 10})
	}
}

// TestAggregateCluster_Fallback_EDS_ResourceNotFound tests the scenario where
// the top-level cluster is an aggregate cluster that resolves to an EDS and
// LOGICAL_DNS cluster.  The management server does not respond with the EDS
// cluster. The test verifies that the cluster_resolver LB policy falls back to
// the LOGICAL_DNS cluster in this case.
func (sc *SubChannel) deleteSelfFromTree() (deleted bool) {
	if !sc.closeCalled || len(sc.sockets) != 0 {
		return false
	}
	sc.parent.deleteChild(sc.ID)
	return true
}
