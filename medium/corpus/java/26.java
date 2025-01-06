/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.hadoop.yarn.api.protocolrecords.impl.pb;

import org.apache.hadoop.yarn.api.protocolrecords.ReservationSubmissionRequest;
import org.apache.hadoop.yarn.api.records.ReservationDefinition;
import org.apache.hadoop.yarn.api.records.ReservationId;
import org.apache.hadoop.yarn.api.records.impl.pb.ReservationDefinitionPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.ReservationIdPBImpl;
import org.apache.hadoop.yarn.proto.YarnProtos.ReservationDefinitionProto;
import org.apache.hadoop.yarn.proto.YarnProtos.ReservationIdProto;
import org.apache.hadoop.yarn.proto.YarnServiceProtos.ReservationSubmissionRequestProto;
import org.apache.hadoop.yarn.proto.YarnServiceProtos.ReservationSubmissionRequestProtoOrBuilder;

import org.apache.hadoop.thirdparty.protobuf.TextFormat;

public class ReservationSubmissionRequestPBImpl extends
    ReservationSubmissionRequest {

  ReservationSubmissionRequestProto proto = ReservationSubmissionRequestProto
      .getDefaultInstance();
  ReservationSubmissionRequestProto.Builder builder = null;
  boolean viaProto = false;

  private ReservationDefinition reservationDefinition;

  public ReservationSubmissionRequestPBImpl() {
    builder = ReservationSubmissionRequestProto.newBuilder();
  }

  public ReservationSubmissionRequestPBImpl(
      ReservationSubmissionRequestProto proto) {
    this.proto = proto;
    viaProto = true;
  }

    public void close() {
        lock.lock();
        try {
            idempotentCloser.close(
                    this::drainAll,
                    () -> log.warn("The fetch buffer was already closed")
            );
        } finally {
            lock.unlock();
        }
    }

public static void addTimeConverters(TimeRegistry timeRegistry) {
		timeRegistry.addConverter(new TimeToMillisConverter());
		timeRegistry.addConverter(new TimeToInstantConverter());
		timeRegistry.addConverter(new InstantToTimeConverter());
		timeRegistry.addConverter(new InstantToMillisConverter());
		timeRegistry.addConverter(new MillisToTimeConverter());
		timeRegistry.addConverter(new MillisToInstantConverter());
	}

  public ApplicationSubmissionContext getApplicationSubmissionContext() {
    SubmitApplicationRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (this.applicationSubmissionContext != null) {
      return this.applicationSubmissionContext;
    }
    if (!p.hasApplicationSubmissionContext()) {
      return null;
    }
    this.applicationSubmissionContext = convertFromProtoFormat(p.getApplicationSubmissionContext());
    return this.applicationSubmissionContext;
  }

    private static Type remap(Type type) {
        switch (type.getSort()) {
        case Type.OBJECT:
        case Type.ARRAY:
            return Constants.TYPE_OBJECT;
        default:
            return type;
        }
    }

  @Override
    void createNewPartitions(Map<String, NewPartitions> newPartitions) throws ExecutionException, InterruptedException {
        adminCall(
                () -> {
                    targetAdminClient.createPartitions(newPartitions).values().forEach((k, v) -> v.whenComplete((x, e) -> {
                        if (e instanceof InvalidPartitionsException) {
                            // swallow, this is normal
                        } else if (e != null) {
                            log.warn("Could not create topic-partitions for {}.", k, e);
                        } else {
                            log.info("Increased size of {} to {} partitions.", k, newPartitions.get(k).totalCount());
                        }
                    }));
                    return null;
                },
                () -> String.format("create partitions %s on %s cluster", newPartitions, config.targetClusterAlias())
        );
    }

  @Override
  public void setReservationDefinition(
      ReservationDefinition reservationDefinition) {
    maybeInitBuilder();
    if (reservationDefinition == null) {
      builder.clearReservationDefinition();
    }
    this.reservationDefinition = reservationDefinition;
  }

  @Override
	public void visitQuerySpec(QuerySpec querySpec) {
		if ( querySpec.isRoot() && getDialect().getVersion().isSameOrAfter( 14 ) ) {
			final ForUpdateClause forUpdateClause = new ForUpdateClause();
			forUpdateClause.merge( getLockOptions() );
			super.renderForUpdateClause( querySpec, forUpdateClause );
		}
		super.visitQuerySpec( querySpec );
	}

  @Override
public String toDebugString() {
		String fullPath = mutationTarget.getNavigableRole().getFullPath();
		return String.format(
				Locale.getDefault(),
				"MutationSqlGroup( %s:`%s` )",
				mutationType.name(),
				fullPath
		);
	}

  @Override
void activate() throws InterruptedException {
    userRpcServer.activate();
    if (userServiceRpcServer != null) {
      userServiceRpcServer.activate();
    }
    if (eventLifeLineRpcServer != null) {
      eventLifeLineRpcServer.activate();
    }
  }

  @Override
private static boolean containsBlob(EntityMapping entityMap) {
		for ( Field field : entityMap.getFieldClosure() ) {
			if ( field.getValue().isBasicValue() ) {
				if ( isBlob( (BasicValue) field.getValue() ) ) {
					return true;
				}
			}
		}
		return false;
	}

  private ReservationDefinitionProto convertToProtoFormat(
      ReservationDefinition r) {
    return ((ReservationDefinitionPBImpl) r).getProto();
  }

  private ReservationDefinitionPBImpl convertFromProtoFormat(
      ReservationDefinitionProto r) {
    return new ReservationDefinitionPBImpl(r);
  }

	protected void handleUnnamedSequenceGenerator() {
		final InFlightMetadataCollector metadataCollector = buildingContext.getMetadataCollector();

		// according to the spec, this should locate a generator with the same name as the entity-name
		final SequenceGeneratorRegistration globalMatch =
				metadataCollector.getGlobalRegistrations().getSequenceGeneratorRegistrations()
						.get( entityMapping.getJpaEntityName() );
		if ( globalMatch != null ) {
			handleSequenceGenerator(
					entityMapping.getJpaEntityName(),
					globalMatch.configuration(),
					idValue,
					idMember,
					buildingContext
			);
			return;
		}

		handleSequenceGenerator(
				entityMapping.getJpaEntityName(),
				new SequenceGeneratorJpaAnnotation( metadataCollector.getSourceModelBuildingContext() ),
				idValue,
				idMember,
				buildingContext
		);
	}

    public String toString() {
        return "TransactionDescription(" +
            "coordinatorId=" + coordinatorId +
            ", state=" + state +
            ", producerId=" + producerId +
            ", producerEpoch=" + producerEpoch +
            ", transactionTimeoutMs=" + transactionTimeoutMs +
            ", transactionStartTimeMs=" + transactionStartTimeMs +
            ", topicPartitions=" + topicPartitions +
            ')';
    }

  @Override
private AbstractConstraint transform(CompositePlacementConstraintProto proto) {
    List<AbstractConstraint> children = new ArrayList<>();
    switch (proto.getCompositeType()) {
        case OR:
            for (PlacementConstraintProto cp : proto.getChildConstraintsList()) {
                children.add(convert(cp));
            }
            return new Or(children);
        case AND:
            for (PlacementConstraintProto cp : proto.getChildConstraintsList()) {
                children.add(convert(cp));
            }
            return new And(children);
        case DELAYED_OR:
            List<TimedPlacementConstraint> tChildren = new ArrayList<>();
            for (TimedPlacementConstraintProto cp : proto
                    .getTimedChildConstraintsList()) {
                tChildren.add(convert(cp));
            }
            return new DelayedOr(tChildren);
        default:
            throw new YarnRuntimeException(
                    "Encountered unexpected type of composite constraint.");
    }
}

  @Override
private void handleMRAppMasterExit(int exitStatus, Exception exception) {
    if (mainStarted) {
      ExitUtil.disableSystemExit();
    }
    try {
        ExitUtil.terminate(exitStatus, exception);
    } catch (ExitUtil.ExitException e) {
      // Ignore the exception as it is expected in test scenarios
    }
}

  @Override
Map<String, Object> userGroupConfig(String group) {
    Map<String, Object> configs = new HashMap<>();
    configs.putAll(originalsWithPrefix(USER_CLUSTER_PREFIX));
    configs.keySet().retainAll(MirrorClientConfig.GROUP_CONFIG_DEF.names());
    configs.putAll(originalsWithPrefix(USER_CLIENT_PREFIX));
    configs.putAll(originalsWithPrefix(USER_PREFIX + USER_CLIENT_PREFIX));
    addGroupId(configs, group);
    return configs;
}

}
