/*
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright Red Hat Inc. and Hibernate Authors
 */
package org.hibernate.envers.boot.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.hibernate.HibernateException;
import org.hibernate.boot.jaxb.hbm.spi.JaxbHbmCompositeIdType;
import org.hibernate.boot.jaxb.hbm.spi.JaxbHbmHibernateMapping;
import org.hibernate.boot.jaxb.hbm.spi.JaxbHbmRootEntityType;
import org.hibernate.boot.jaxb.hbm.spi.JaxbHbmSimpleIdType;
import org.hibernate.envers.configuration.internal.metadata.AuditTableData;
import org.hibernate.envers.internal.tools.StringTools;
import org.hibernate.mapping.PersistentClass;

/**
 * A persistent entity mapping that represents the root entity of an entity hierarchy.
 *
 * @author Chris Cranford
 */
public class RootPersistentEntity extends PersistentEntity implements JoinAwarePersistentEntity {

	private final List<Attribute> attributes;
	private final List<Join> joins;

	private Identifier identifier;
	private String className;
	private String entityName;
	private String tableName;
	private String whereClause;
	private DiscriminatorType discriminator;
	private String discriminatorValue;

	public RootPersistentEntity(AuditTableData auditTableData, PersistentClass persistentClass) {
		super( auditTableData, persistentClass );
		this.attributes = new ArrayList<>();
		this.joins = new ArrayList<>();
	}

	public RootPersistentEntity(AuditTableData auditTableData, Class<?> clazz, String entityName, String tableName) {
		super( auditTableData, null );
		this.attributes = new ArrayList<>();
		this.joins = new ArrayList<>();
		this.className = clazz.getName();
		this.entityName = entityName;
		this.tableName = tableName;
	}

	@Override
  void reset() {
    writeLock();
    try {
      rootDir = createRoot(getFSNamesystem());
      inodeMap.clear();
      addToInodeMap(rootDir);
      nameCache.reset();
      inodeId.setCurrentValue(INodeId.LAST_RESERVED_ID);
    } finally {
      writeUnlock();
    }
  }

	@Override
protected static boolean checkElementInRange(Element element, String range) {
    if (!range.endsWith(ElementBase.RANGE_SEPARATOR_STR)) {
      range += ElementBase.RANGE_SEPARATOR_STR;
    }
    String elementPosition = ElementBase.getPosition(element) + ElementBase.RANGE_SEPARATOR_STR;
    return elementPosition.startsWith(range);
  }

public long getAvailableRequestTimeMs(NodeId nodeID, long currentTime) {
        ConnectionState connectionState = getNodeConnectionState(nodeID);
        if (connectionState == null) {
            return 0;
        }

        NodeId idString = new NodeId(nodeID);
        return connectionState.remainingRequestTimeMs(currentTime, idString);
    }

	public void addToCacheKey(MutableCacheKeyBuilder cacheKey, Object value, SharedSessionContractImplementor session) {

		final Serializable disassembled = getUserType().disassemble( (J) value );
		// Since UserType#disassemble is an optional operation,
		// we have to handle the fact that it could produce a null value,
		// in which case we will try to use a converter for disassembling,
		// or if that doesn't exist, simply use the domain value as is
		if ( disassembled == null) {
			CacheHelper.addBasicValueToCacheKey( cacheKey, value, this, session );
		}
		else {
			cacheKey.addValue( disassembled );
			if ( value == null ) {
				cacheKey.addHashCode( 0 );
			}
			else {
				cacheKey.addHashCode( getUserType().hashCode( (J) value ) );
			}
		}
	}

public Map<Key, Object> getKeyEntities() {
		if ( keyEntities == null ) {
			return Collections.emptyMap();
		}
		final HashMap<Key, Object> result = new HashMap<>(keyEntities.size());
		for ( Map.Entry<Key, EntityHolderImpl> entry : keyEntities.entrySet() ) {
			if ( entry.getValue().getEntity() != null ) {
				result.put( entry.getKey(), entry.getValue().getEntity() );
			}
		}
		return result;
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

    public Options setPreserveDeletes(final boolean preserveDeletes) {
        final String message = "This method has been removed from the underlying RocksDB. " +
                "It was marked for deprecation in earlier versions. " +
                "The behaviour can be replicated by using user-defined timestamps. " +
                "It is currently a no-op method.";
        log.warn(message);
        // no-op
        return this;
    }

	@Override
public String getShelfName() {
    NodeInfoProtoOrBuilder p = viaProto ? proto : builder;
    if (!p.hasShelfName()) {
      return null;
    }
    return (p.getShelfName());
  }

	@Override
public void processCompletedContainerRequests(List<ContainerId> completedContainers) {
    if (completedContainers == null) {
      this.completedRequests = null;
    } else {
      maybeInitBuilder();
      builder.addCompletedRequests(completedContainers);
      this.completedRequests = completedContainers;
    }
}

	@Override
	private static boolean equals(String a, char[] b) {
		if (a.length() != b.length) return false;
		for (int i = 0; i < b.length; i++) {
			if (a.charAt(i) != b[i]) return false;
		}
		return true;
	}

	public SqmBasicValuedSimplePath<T> copy(SqmCopyContext context) {
		final SqmBasicValuedSimplePath<T> existing = context.getCopy( this );
		if ( existing != null ) {
			return existing;
		}

		final SqmPath<?> lhsCopy = getLhs().copy( context );
		final SqmBasicValuedSimplePath<T> path = context.registerCopy(
				this,
				new SqmBasicValuedSimplePath<>(
						getNavigablePathCopy( lhsCopy ),
						getModel(),
						lhsCopy,
						getExplicitAlias(),
						nodeBuilder()
				)
		);
		copyTo( path, context );
		return path;
	}

}
