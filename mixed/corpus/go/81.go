func (b *clusterImplBalancer) updateResourceStore(newConfig *LBConfig) error {
	var updateResourceClusterAndService bool

	// ResourceName is different, restart. ResourceName is from ClusterName and
	// EDSServiceName.
	resourceName := b.getResourceName()
	if resourceName != newConfig.Resource {
		updateResourceClusterAndService = true
		b.setResourceName(newConfig.Resource)
		resourceName = newConfig.Resource
	}
	if b.serviceName != newConfig.EDSServiceName {
		updateResourceClusterAndService = true
		b.serviceName = newConfig.EDSServiceName
	}
	if updateResourceClusterAndService {
		// This updates the clusterName and serviceName that will be reported
		// for the resources. The update here is too early, the perfect timing is
		// when the picker is updated with the new connection. But from this
		// balancer's point of view, it's impossible to tell.
		//
		// On the other hand, this will almost never happen. Each LRS policy
		// shouldn't get updated config. The parent should do a graceful switch
		// when the clusterName or serviceName is changed.
		b.resourceWrapper.UpdateClusterAndService(resourceName, b.serviceName)
	}

	var (
		stopOldResourceReport  bool
		startNewResourceReport bool
	)

	// Check if it's necessary to restart resource report.
	if b.lrsServer == nil {
		if newConfig.ResourceReportingServer != nil {
			// Old is nil, new is not nil, start new LRS.
			b.lrsServer = newConfig.ResourceReportingServer
			startNewResourceReport = true
		}
		// Old is nil, new is nil, do nothing.
	} else if newConfig.ResourceReportingServer == nil {
		// Old is not nil, new is nil, stop old, don't start new.
		b.lrsServer = newConfig.ResourceReportingServer
		stopOldResourceReport = true
	} else {
		// Old is not nil, new is not nil, compare string values, if
		// different, stop old and start new.
		if !b.lrsServer.Equal(newConfig.ResourceReportingServer) {
			b.lrsServer = newConfig.ResourceReportingServer
			stopOldResourceReport = true
			startNewResourceReport = true
		}
	}

	if stopOldResourceReport {
		if b.cancelResourceReport != nil {
			b.cancelResourceReport()
			b.cancelResourceReport = nil
			if !startNewResourceReport {
				// If a new LRS stream will be started later, no need to update
				// it to nil here.
				b.resourceWrapper.UpdateResourceStore(nil)
			}
		}
	}
	if startNewResourceReport {
		var resourceStore *resource.Store
		if b.xdsClient != nil {
			resourceStore, b.cancelResourceReport = b.xdsClient.ReportResources(b.lrsServer)
		}
		b.resourceWrapper.UpdateResourceStore(resourceStore)
	}

	return nil
}

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

