    private void normalize() {
        // Version 0 only supported a single host and port and the protocol was always plaintext
        // Version 1 added support for multiple endpoints, each with its own security protocol
        // Version 2 added support for rack
        // Version 3 added support for listener name, which we can infer from the security protocol for older versions
        if (version() < 3) {
            for (UpdateMetadataBroker liveBroker : data.liveBrokers()) {
                // Set endpoints so that callers can rely on it always being present
                if (version() == 0 && liveBroker.endpoints().isEmpty()) {
                    SecurityProtocol securityProtocol = SecurityProtocol.PLAINTEXT;
                    liveBroker.setEndpoints(singletonList(new UpdateMetadataEndpoint()
                        .setHost(liveBroker.v0Host())
                        .setPort(liveBroker.v0Port())
                        .setSecurityProtocol(securityProtocol.id)
                        .setListener(ListenerName.forSecurityProtocol(securityProtocol).value())));
                } else {
                    for (UpdateMetadataEndpoint endpoint : liveBroker.endpoints()) {
                        // Set listener so that callers can rely on it always being present
                        if (endpoint.listener().isEmpty())
                            endpoint.setListener(listenerNameFromSecurityProtocol(endpoint));
                    }
                }
            }
        }

        if (version() >= 5) {
            for (UpdateMetadataTopicState topicState : data.topicStates()) {
                for (UpdateMetadataPartitionState partitionState : topicState.partitionStates()) {
                    // Set the topic name so that we can always present the ungrouped view to callers
                    partitionState.setTopicName(topicState.topicName());
                }
            }
        }
    }

public synchronized ResponseFuture<Void> potentialExitCluster(String exitReason) {
    ResponseFuture<Void> future = null;

    // Starting from 2.3, only dynamic members will send ExitGroupRequest to the broker,
    // consumer with valid group.instance.id is viewed as static member that never sends ExitGroup,
    // and the membership expiration is only controlled by session timeout.
    if (isFlexibleMember() && !coordinatorUnavailable() &&
        status != MemberStatus.UNASSOCIATED && generation.hasValidId()) {
        // this is a minimal effort attempt to exit the cluster. we do not
        // attempt any resending if the request fails or times out.
        log.info("Generation {} sending ExitGroup request to coordinator {} due to {}",
            generation.id, coordinator, exitReason);
        ExitGroupRequest.Builder request = new ExitGroupRequest.Builder(
            rebalanceConfig.clusterId,
            Collections.singletonList(new ParticipantIdentity().setGroupId(generation.id).setReason(JoinGroupRequest.maybeTruncateReason(exitReason)))
        );

        future = client.send(coordinator, request).compose(new ExitGroupResponseHandler(generation));
        client.pollNoInterrupt();
    }

    resetGenerationOnExitCluster();

    return future;
}

public boolean areContentsEqual(Object comparison) {
    if (this == comparison) {
        return true;
    } else if (comparison instanceof DataSummary) {
        DataSummary other = (DataSummary) comparison;
        return getTotalSize() == other.getTotalSize() &&
               getFilePathCount() == other.getFilePathCount() &&
               getDirectoryPathCount() == other.getDirectoryPathCount() &&
               getSnapshotTotalSize() == other.getSnapshotTotalSize() &&
               getSnapshotFilePathCount() == other.getSnapshotFilePathCount() &&
               getSnapshotDirectoryPathCount() == other.getSnapshotDirectoryPathCount() &&
               getSnapshotSpaceOccupied() == other.getSnapshotSpaceOccupied() &&
               getDataRedundancyPolicy().equals(other.getDataRedundancyPolicy()) &&
               super.areContentsEqual(comparison);
    } else {
        return super.areContentsEqual(comparison);
    }
}

