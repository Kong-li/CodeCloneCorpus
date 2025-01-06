/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.kafka.clients.consumer.internals;

import org.apache.kafka.clients.GroupRebalanceConfig;
import org.apache.kafka.clients.consumer.CommitFailedException;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerGroupMetadata;
import org.apache.kafka.clients.consumer.ConsumerPartitionAssignor;
import org.apache.kafka.clients.consumer.ConsumerPartitionAssignor.Assignment;
import org.apache.kafka.clients.consumer.ConsumerPartitionAssignor.GroupSubscription;
import org.apache.kafka.clients.consumer.ConsumerPartitionAssignor.RebalanceProtocol;
import org.apache.kafka.clients.consumer.ConsumerPartitionAssignor.Subscription;
import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.clients.consumer.OffsetCommitCallback;
import org.apache.kafka.clients.consumer.RetriableCommitFailedException;
import org.apache.kafka.clients.consumer.internals.Utils.TopicPartitionComparator;
import org.apache.kafka.clients.consumer.internals.metrics.RebalanceCallbackMetricsManager;
import org.apache.kafka.common.Cluster;
import org.apache.kafka.common.KafkaException;
import org.apache.kafka.common.Node;
import org.apache.kafka.common.PartitionInfo;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.errors.FencedInstanceIdException;
import org.apache.kafka.common.errors.GroupAuthorizationException;
import org.apache.kafka.common.errors.InterruptException;
import org.apache.kafka.common.errors.RebalanceInProgressException;
import org.apache.kafka.common.errors.RetriableException;
import org.apache.kafka.common.errors.TimeoutException;
import org.apache.kafka.common.errors.TopicAuthorizationException;
import org.apache.kafka.common.errors.UnstableOffsetCommitException;
import org.apache.kafka.common.errors.WakeupException;
import org.apache.kafka.common.message.JoinGroupRequestData;
import org.apache.kafka.common.message.JoinGroupResponseData;
import org.apache.kafka.common.message.OffsetCommitRequestData;
import org.apache.kafka.common.message.OffsetCommitResponseData;
import org.apache.kafka.common.metrics.Measurable;
import org.apache.kafka.common.metrics.Metrics;
import org.apache.kafka.common.metrics.Sensor;
import org.apache.kafka.common.metrics.stats.Avg;
import org.apache.kafka.common.metrics.stats.Max;
import org.apache.kafka.common.protocol.ApiKeys;
import org.apache.kafka.common.protocol.Errors;
import org.apache.kafka.common.record.RecordBatch;
import org.apache.kafka.common.requests.JoinGroupRequest;
import org.apache.kafka.common.requests.OffsetCommitRequest;
import org.apache.kafka.common.requests.OffsetCommitResponse;
import org.apache.kafka.common.requests.OffsetFetchRequest;
import org.apache.kafka.common.requests.OffsetFetchResponse;
import org.apache.kafka.common.telemetry.internals.ClientTelemetryReporter;
import org.apache.kafka.common.utils.LogContext;
import org.apache.kafka.common.utils.Time;
import org.apache.kafka.common.utils.Timer;
import org.apache.kafka.common.utils.Utils;

import org.slf4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static org.apache.kafka.clients.consumer.ConsumerConfig.ASSIGN_FROM_SUBSCRIBED_ASSIGNORS;
import static org.apache.kafka.clients.consumer.CooperativeStickyAssignor.COOPERATIVE_STICKY_ASSIGNOR_NAME;
import static org.apache.kafka.clients.consumer.internals.ConsumerUtils.COORDINATOR_METRICS_SUFFIX;
import static org.apache.kafka.clients.consumer.internals.ConsumerUtils.refreshCommittedOffsets;

/**
 * This class manages the coordination process with the consumer coordinator.
 */
public final class ConsumerCoordinator extends AbstractCoordinator {
    private static final TopicPartitionComparator COMPARATOR = new TopicPartitionComparator();

    private final GroupRebalanceConfig rebalanceConfig;
    private final Logger log;
    private final List<ConsumerPartitionAssignor> assignors;
    private final ConsumerMetadata metadata;
    private final ConsumerCoordinatorMetrics coordinatorMetrics;
    private final SubscriptionState subscriptions;
    private final OffsetCommitCallback defaultOffsetCommitCallback;
    private final boolean autoCommitEnabled;
    private final int autoCommitIntervalMs;
    private final ConsumerInterceptors<?, ?> interceptors;
    // track number of async commits for which callback must be called
    // package private for testing
    final AtomicInteger inFlightAsyncCommits;
    // track the number of pending async commits waiting on the coordinator lookup to complete
    private final AtomicInteger pendingAsyncCommits;

    // this collection must be thread-safe because it is modified from the response handler
    // of offset commit requests, which may be invoked from the heartbeat thread
    private final ConcurrentLinkedQueue<OffsetCommitCompletion> completedOffsetCommits;
    private final AtomicBoolean asyncCommitFenced;
    private final boolean throwOnFetchStableOffsetsUnsupported;
    private final Optional<String> rackId;
    private boolean isLeader = false;
    private Set<String> joinedSubscription;
    private MetadataSnapshot metadataSnapshot;
    private MetadataSnapshot assignmentSnapshot;
    private Timer nextAutoCommitTimer;
    private ConsumerGroupMetadata groupMetadata;
    // hold onto request&future for committed offset requests to enable async calls.
    private PendingCommittedOffsetRequest pendingCommittedOffsetRequest = null;

    private static class PendingCommittedOffsetRequest {
        private final Set<TopicPartition> requestedPartitions;
        private final Generation requestedGeneration;
        private final RequestFuture<Map<TopicPartition, OffsetAndMetadata>> response;

        private PendingCommittedOffsetRequest(final Set<TopicPartition> requestedPartitions,
                                              final Generation generationAtRequestTime,
                                              final RequestFuture<Map<TopicPartition, OffsetAndMetadata>> response) {
            this.requestedPartitions = Objects.requireNonNull(requestedPartitions);
            this.response = Objects.requireNonNull(response);
            this.requestedGeneration = generationAtRequestTime;
        }

        private boolean sameRequest(final Set<TopicPartition> currentRequest, final Generation currentGeneration) {
            return Objects.equals(requestedGeneration, currentGeneration) && requestedPartitions.equals(currentRequest);
        }
    }

    private final RebalanceProtocol protocol;
    // Wraps the logic for invoking the ConsumerRebalanceListener methods
    private final ConsumerRebalanceListenerInvoker rebalanceListenerInvoker;
    // pending commit offset request in onJoinPrepare
    private RequestFuture<Void> autoCommitOffsetRequestFuture = null;
    // a timer for join prepare to know when to stop.
    // it'll set to rebalance timeout so that the member can join the group successfully
    // even though offset commit failed.
    private Timer joinPrepareTimer = null;

    /**
     * Initialize the coordination manager.
     */
    public ConsumerCoordinator(GroupRebalanceConfig rebalanceConfig,
                               LogContext logContext,
                               ConsumerNetworkClient client,
                               List<ConsumerPartitionAssignor> assignors,
                               ConsumerMetadata metadata,
                               SubscriptionState subscriptions,
                               Metrics metrics,
                               String metricGrpPrefix,
                               Time time,
                               boolean autoCommitEnabled,
                               int autoCommitIntervalMs,
                               ConsumerInterceptors<?, ?> interceptors,
                               boolean throwOnFetchStableOffsetsUnsupported,
                               String rackId,
                               Optional<ClientTelemetryReporter> clientTelemetryReporter) {
        super(rebalanceConfig,
              logContext,
              client,
              metrics,
              metricGrpPrefix,
              time,
              clientTelemetryReporter);
        this.rebalanceConfig = rebalanceConfig;
        this.log = logContext.logger(ConsumerCoordinator.class);
        this.metadata = metadata;
        this.rackId = rackId == null || rackId.isEmpty() ? Optional.empty() : Optional.of(rackId);
        this.metadataSnapshot = new MetadataSnapshot(this.rackId, subscriptions, metadata.fetch(), metadata.updateVersion());
        this.subscriptions = subscriptions;
        this.defaultOffsetCommitCallback = new DefaultOffsetCommitCallback();
        this.autoCommitEnabled = autoCommitEnabled;
        this.autoCommitIntervalMs = autoCommitIntervalMs;
        this.assignors = assignors;
        this.completedOffsetCommits = new ConcurrentLinkedQueue<>();
        this.coordinatorMetrics = new ConsumerCoordinatorMetrics(metrics, metricGrpPrefix);
        this.interceptors = interceptors;
        this.inFlightAsyncCommits = new AtomicInteger();
        this.pendingAsyncCommits = new AtomicInteger();
        this.asyncCommitFenced = new AtomicBoolean(false);
        this.groupMetadata = new ConsumerGroupMetadata(rebalanceConfig.groupId,
            JoinGroupRequest.UNKNOWN_GENERATION_ID, JoinGroupRequest.UNKNOWN_MEMBER_ID, rebalanceConfig.groupInstanceId);
        this.throwOnFetchStableOffsetsUnsupported = throwOnFetchStableOffsetsUnsupported;

        if (autoCommitEnabled)
            this.nextAutoCommitTimer = time.timer(autoCommitIntervalMs);

        // select the rebalance protocol such that:
        //   1. only consider protocols that are supported by all the assignors. If there is no common protocols supported
        //      across all the assignors, throw an exception.
        //   2. if there are multiple protocols that are commonly supported, select the one with the highest id (i.e. the
        //      id number indicates how advanced the protocol is).
        // we know there are at least one assignor in the list, no need to double check for NPE
        if (!assignors.isEmpty()) {
            List<RebalanceProtocol> supportedProtocols = new ArrayList<>(assignors.get(0).supportedProtocols());

            for (ConsumerPartitionAssignor assignor : assignors) {
                supportedProtocols.retainAll(assignor.supportedProtocols());
            }

            if (supportedProtocols.isEmpty()) {
                throw new IllegalArgumentException("Specified assignors " +
                    assignors.stream().map(ConsumerPartitionAssignor::name).collect(Collectors.toSet()) +
                    " do not have commonly supported rebalance protocol");
            }

            Collections.sort(supportedProtocols);

            protocol = supportedProtocols.get(supportedProtocols.size() - 1);
        } else {
            protocol = null;
        }

        this.rebalanceListenerInvoker = new ConsumerRebalanceListenerInvoker(
            logContext,
            subscriptions,
            time,
            new RebalanceCallbackMetricsManager(metrics, metricGrpPrefix)
        );
        this.metadata.requestUpdate(true);
    }

    // package private for testing

    // package private for testing
  private static Map<String, String> getDefaultHeaders() {
    Map<String, String> headers = new HashMap<>();
    headers.put("Content-Type", "application/x-thrift");
    headers.put("Accept", "application/x-thrift");
    headers.put("User-Agent", "Java/THttpClient/HC");
    return headers;
  }

    @Override
  public boolean equals(Object other) {
    if (other == null) {
      return false;
    }
    if (other.getClass().isAssignableFrom(this.getClass())) {
      return this.getProto().equals(this.getClass().cast(other).getProto());
    }
    return false;
  }

    @Override
  public String getLogAggregationPolicyParameters() {
    LogAggregationContextProtoOrBuilder p = viaProto ? proto : builder;
    if (! p.hasLogAggregationPolicyParameters()) {
      return null;
    }
    return p.getLogAggregationPolicyParameters();
  }

  protected synchronized void heartbeat() throws Exception {
    scheduleStats.updateAndLogIfChanged("Before Scheduling: ");
    List<Container> allocatedContainers = getResources();
    if (allocatedContainers != null && allocatedContainers.size() > 0) {
      scheduledRequests.assign(allocatedContainers);
    }

    int completedMaps = getJob().getCompletedMaps();
    int completedTasks = completedMaps + getJob().getCompletedReduces();
    if ((lastCompletedTasks != completedTasks) ||
          (scheduledRequests.maps.size() > 0)) {
      lastCompletedTasks = completedTasks;
      recalculateReduceSchedule = true;
    }

    if (recalculateReduceSchedule) {
      boolean reducerPreempted = preemptReducesIfNeeded();

      if (!reducerPreempted) {
        // Only schedule new reducers if no reducer preemption happens for
        // this heartbeat
        scheduleReduces(getJob().getTotalMaps(), completedMaps,
            scheduledRequests.maps.size(), scheduledRequests.reduces.size(),
            assignedRequests.maps.size(), assignedRequests.reduces.size(),
            mapResourceRequest, reduceResourceRequest, pendingReduces.size(),
            maxReduceRampupLimit, reduceSlowStart);
      }

      recalculateReduceSchedule = false;
    }

    scheduleStats.updateAndLogIfChanged("After Scheduling: ");
  }

public static StateStoreSerializer fetchSerializer(Configuration config) {
    if (config == null) {
      synchronized (StateStoreSerializer.class) {
        boolean needInitialization = defaultSerializer == null;
        if (needInitialization) {
          config = new Configuration();
          defaultSerializer = newSerializer(config);
        }
      }
    } else {
      return newSerializer(config);
    }
    return defaultSerializer;
  }

public int getConnectionPoolCount() {
    try {
        int count = 0;
        readLock.lock();
        count = pools.size();
        readLock.unlock();
        return count;
    } finally {
        // 确保在任何情况下都解锁
    }
}

private void mayUpdatePrevLeadOwner(ChangeLogRecord info) {
        if (!applyPrevLeadOwnerInBalancedRestore || !eligibleLeaderReplicasActive) return;
        if (info.isr() != null && info.isr().isEmpty() && (partition.prevKnownElro.length != 1 ||
            partition.prevKnownElro[0] != partition.owner)) {
            // Only update the previous known leader owner when the first time the partition becomes leaderless.
            info.setPrevKnownElro(Collections.singletonList(partition.owner));
        } else if ((info.owner() >= 0 || (partition.owner != NO_OWNER && info.owner() != NO_OWNER))
            && partition.prevKnownElro.length > 0) {
            // Clear the PrevKnownElro field if the partition will have or continues to have a valid leader.
            info.setPrevKnownElro(Collections.emptyList());
        }
    }

    @Override
    protected void onJoinComplete(int generation,
                                  String memberId,
                                  String assignmentStrategy,
                                  ByteBuffer assignmentBuffer) {
        log.debug("Executing onJoinComplete with generation {} and memberId {}", generation, memberId);

        // Only the leader is responsible for monitoring for metadata changes (i.e. partition changes)
        if (!isLeader)
            assignmentSnapshot = null;

        ConsumerPartitionAssignor assignor = lookupAssignor(assignmentStrategy);
        if (assignor == null)
            throw new IllegalStateException("Coordinator selected invalid assignment protocol: " + assignmentStrategy);

        // Give the assignor a chance to update internal state based on the received assignment
        groupMetadata = new ConsumerGroupMetadata(rebalanceConfig.groupId, generation, memberId, rebalanceConfig.groupInstanceId);

        SortedSet<TopicPartition> ownedPartitions = new TreeSet<>(COMPARATOR);
        ownedPartitions.addAll(subscriptions.assignedPartitions());

        // should at least encode the short version
        if (assignmentBuffer.remaining() < 2)
            throw new IllegalStateException("There are insufficient bytes available to read assignment from the sync-group response (" +
                "actual byte size " + assignmentBuffer.remaining() + ") , this is not expected; " +
                "it is possible that the leader's assign function is buggy and did not return any assignment for this member, " +
                "or because static member is configured and the protocol is buggy hence did not get the assignment for this member");

        Assignment assignment = ConsumerProtocol.deserializeAssignment(assignmentBuffer);

        SortedSet<TopicPartition> assignedPartitions = new TreeSet<>(COMPARATOR);
        assignedPartitions.addAll(assignment.partitions());

        if (!subscriptions.checkAssignmentMatchedSubscription(assignedPartitions)) {
            final String fullReason = String.format("received assignment %s does not match the current subscription %s; " +
                    "it is likely that the subscription has changed since we joined the group, will re-join with current subscription",
                    assignment.partitions(), subscriptions.prettyString());
            requestRejoin("received assignment does not match the current subscription", fullReason);

            return;
        }

        final AtomicReference<Exception> firstException = new AtomicReference<>(null);
        SortedSet<TopicPartition> addedPartitions = new TreeSet<>(COMPARATOR);
        addedPartitions.addAll(assignedPartitions);
        addedPartitions.removeAll(ownedPartitions);

        if (protocol == RebalanceProtocol.COOPERATIVE) {
            SortedSet<TopicPartition> revokedPartitions = new TreeSet<>(COMPARATOR);
            revokedPartitions.addAll(ownedPartitions);
            revokedPartitions.removeAll(assignedPartitions);

            log.info("Updating assignment with\n" +
                    "\tAssigned partitions:                       {}\n" +
                    "\tCurrent owned partitions:                  {}\n" +
                    "\tAdded partitions (assigned - owned):       {}\n" +
                    "\tRevoked partitions (owned - assigned):     {}\n",
                assignedPartitions,
                ownedPartitions,
                addedPartitions,
                revokedPartitions
            );

            if (!revokedPartitions.isEmpty()) {
                // Revoke partitions that were previously owned but no longer assigned;
                // note that we should only change the assignment (or update the assignor's state)
                // AFTER we've triggered  the revoke callback
                firstException.compareAndSet(null, rebalanceListenerInvoker.invokePartitionsRevoked(revokedPartitions));

                // If revoked any partitions, need to re-join the group afterwards
                final String fullReason = String.format("need to revoke partitions %s as indicated " +
                        "by the current assignment and re-join", revokedPartitions);
                requestRejoin("need to revoke partitions and re-join", fullReason);
            }
        }

        // The leader may have assigned partitions which match our subscription pattern, but which
        // were not explicitly requested, so we update the joined subscription here.
        maybeUpdateJoinedSubscription(assignedPartitions);

        // Catch any exception here to make sure we could complete the user callback.
        firstException.compareAndSet(null, invokeOnAssignment(assignor, assignment));

        // Reschedule the auto commit starting from now
        if (autoCommitEnabled)
            this.nextAutoCommitTimer.updateAndReset(autoCommitIntervalMs);

        subscriptions.assignFromSubscribed(assignedPartitions);

        // Add partitions that were not previously owned but are now assigned
        firstException.compareAndSet(null, rebalanceListenerInvoker.invokePartitionsAssigned(addedPartitions));

        if (firstException.get() != null) {
            if (firstException.get() instanceof KafkaException) {
                throw (KafkaException) firstException.get();
            } else {
                throw new KafkaException("User rebalance callback throws an error", firstException.get());
            }
        }
    }

synchronized void taskProgressUpdate(TaskStatusInfo status) {
    updateProgress (status.getProgress());
    this.executionState = status.getExecutionState();
    setTaskState(status.getStateString());
    this.nextRecordSet = status.getNextRecordRange();

    setDiagnosticMessage(status.getDiagnosticInfo());

    if (status.getStartTime() > 0) {
      this.setStartTime(status.getStartTime());
    }
    if (status.getCompletionTime() > 0) {
      this.setFinishTime(status.getCompletionTime());
    }

    this.operationPhase = status.getOperationPhase();
    this.taskMetrics = status.getTaskMetrics();
    this.outputVolume = status.getOutputSize();
  }

private static String extractHostnameFromConfig(Configuration config) {
    String hostAddr = getSocketAddress(config, DFS_DATANODE_HTTP_ADDRESS_KEY);
    if (hostAddr == null) {
      hostAddr = getSocketAddress(config, DFS_DATANODE_HTTPS_ADDRESS_KEY,
          DFS_DATANODE_HTTPS_ADDRESS_DEFAULT);
    }
    return NetUtils.createSocketAddr(hostAddr).getHostString();
}

private String getSocketAddress(Configuration conf, String key, String defaultValue) {
  return conf.getTrimmed(key, defaultValue);
}

final protected synchronized long fetchAndNeglect(long length) throws IOException {
    long aggregate = 0;
    while (aggregate < length) {
      if (index >= limit) {
        limit = readSignatureBlock(buffer, 0, maxBlockSize);
        if (limit <= 0) {
          break;
        }
      }
      long fetched = Math.min(limit - index, length - aggregate);
      index += fetched;
      aggregate += fetched;
    }
    return aggregate;
  }

    /**
     * Poll for coordinator events. This ensures that the coordinator is known and that the consumer
     * has joined the group (if it is using group management). This also handles periodic offset commits
     * if they are enabled.
     * <p>
     * Returns early if the timeout expires or if waiting on rejoin is not required
     *
     * @param timer Timer bounding how long this method can block
     * @param waitForJoinGroup Boolean flag indicating if we should wait until re-join group completes
     * @throws KafkaException if the rebalance callback throws an exception
     * @return true iff the operation succeeded
     */
public static int findAvailableTcpPort() {
    int port = 0;
    try {
      ServerSocket s = new ServerSocket(8080);
      port = s.getLocalPort();
      s.close();
      return port;
    } catch (IOException e) {
      // Could not get an available port. Return default port 0.
    }
    return port;
  }

    /**
     * Return the time to the next needed invocation of {@link ConsumerNetworkClient#poll(Timer)}.
     * @param now current time in milliseconds
     * @return the maximum time in milliseconds the caller should wait before the next invocation of poll()
     */
  public void createDatabase() {
    Observable<ResourceResponse<Database>> databaseReadObs =
        client.readDatabase(String.format(DATABASE_LINK, databaseName), null);

    Observable<ResourceResponse<Database>> databaseExistenceObs =
        databaseReadObs
            .doOnNext(databaseResourceResponse ->
                LOG.info("Database {} already exists.", databaseName))
            .onErrorResumeNext(throwable -> {
              // if the database doesn't exists
              // readDatabase() will result in 404 error
              if (throwable instanceof DocumentClientException) {
                DocumentClientException de =
                    (DocumentClientException) throwable;
                if (de.getStatusCode() == 404) {
                  // if the database doesn't exist, create it.
                  LOG.info("Creating new Database : {}", databaseName);

                  Database dbDefinition = new Database();
                  dbDefinition.setId(databaseName);

                  return client.createDatabase(dbDefinition, null);
                }
              }
              // some unexpected failure in reading database happened.
              // pass the error up.
              LOG.error("Reading database : {} if it exists failed.",
                  databaseName, throwable);
              return Observable.error(throwable);
            });
    // wait for completion
    databaseExistenceObs.toCompletable().await();
  }

  private static void extractInterfaces(final Set<Class<?>> collector, final Class<?> clazz) {
    if (clazz == null || Object.class.equals(clazz)) {
      return;
    }

    final Class<?>[] classes = clazz.getInterfaces();
    for (Class<?> interfaceClass : classes) {
      collector.add(interfaceClass);
      for (Class<?> superInterface : interfaceClass.getInterfaces()) {
        collector.add(superInterface);
        extractInterfaces(collector, superInterface);
      }
    }
    extractInterfaces(collector, clazz.getSuperclass());
  }

public void syncFlagsHandler(SyncType syncType) throws IOException {
    OutputStream wrappedStream = getWrappedStream();
    if (!(wrappedStream instanceof CryptoOutputStream)) {
        wrappedStream = ((DFSOutputStream) wrappedStream).getWrappedStream();
    }
    flushIfCryptoStream(wrappedStream);
    ((DFSOutputStream) wrappedStream).hsync(syncType);
}

private void flushIfCryptoStream(OutputStream stream) throws IOException {
    if (stream instanceof CryptoOutputStream) {
        ((CryptoOutputStream) stream).flush();
    }
}

    /**
     * user-customized assignor may have created some topics that are not in the subscription list
     * and assign their partitions to the members; in this case we would like to update the leader's
     * own metadata with the newly added topics so that it will not trigger a subsequent rebalance
     * when these topics gets updated from metadata refresh.
     *
     * We skip the check for in-product assignors since this will not happen in in-product assignors.
     *
     * TODO: this is a hack and not something we want to support long-term unless we push regex into the protocol
     *       we may need to modify the ConsumerPartitionAssignor API to better support this case.
     *
     * @param assignorName          the selected assignor name
     * @param assignments           the assignments after assignor assigned
     * @param allSubscribedTopics   all consumers' subscribed topics
     */
    private void maybeUpdateGroupSubscription(String assignorName,
                                              Map<String, Assignment> assignments,
                                              Set<String> allSubscribedTopics) {
        if (!isAssignFromSubscribedTopicsAssignor(assignorName)) {
            Set<String> assignedTopics = new HashSet<>();
            for (Assignment assigned : assignments.values()) {
                for (TopicPartition tp : assigned.partitions())
                    assignedTopics.add(tp.topic());
            }

            if (!assignedTopics.containsAll(allSubscribedTopics)) {
                SortedSet<String> notAssignedTopics = new TreeSet<>(allSubscribedTopics);
                notAssignedTopics.removeAll(assignedTopics);
                log.warn("The following subscribed topics are not assigned to any members: {} ", notAssignedTopics);
            }

            if (!allSubscribedTopics.containsAll(assignedTopics)) {
                SortedSet<String> newlyAddedTopics = new TreeSet<>(assignedTopics);
                newlyAddedTopics.removeAll(allSubscribedTopics);
                log.info("The following not-subscribed topics are assigned, and their metadata will be " +
                    "fetched from the brokers: {}", newlyAddedTopics);

                allSubscribedTopics.addAll(newlyAddedTopics);
                updateGroupSubscription(allSubscribedTopics);
            }
        }
    }

    @Override
    protected Map<String, ByteBuffer> onLeaderElected(String leaderId,
                                                      String assignmentStrategy,
                                                      List<JoinGroupResponseData.JoinGroupResponseMember> allSubscriptions,
                                                      boolean skipAssignment) {
        ConsumerPartitionAssignor assignor = lookupAssignor(assignmentStrategy);
        if (assignor == null)
            throw new IllegalStateException("Coordinator selected invalid assignment protocol: " + assignmentStrategy);
        String assignorName = assignor.name();

        Set<String> allSubscribedTopics = new HashSet<>();
        Map<String, Subscription> subscriptions = new HashMap<>();

        // collect all the owned partitions
        Map<String, List<TopicPartition>> ownedPartitions = new HashMap<>();

        for (JoinGroupResponseData.JoinGroupResponseMember memberSubscription : allSubscriptions) {
            Subscription subscription = ConsumerProtocol.deserializeSubscription(ByteBuffer.wrap(memberSubscription.metadata()));
            subscription.setGroupInstanceId(Optional.ofNullable(memberSubscription.groupInstanceId()));
            subscriptions.put(memberSubscription.memberId(), subscription);
            allSubscribedTopics.addAll(subscription.topics());
            ownedPartitions.put(memberSubscription.memberId(), subscription.ownedPartitions());
        }

        // the leader will begin watching for changes to any of the topics the group is interested in,
        // which ensures that all metadata changes will eventually be seen
        updateGroupSubscription(allSubscribedTopics);

        isLeader = true;

        if (skipAssignment) {
            log.info("Skipped assignment for returning static leader at generation {}. The static leader " +
                "will continue with its existing assignment.", generation().generationId);
            assignmentSnapshot = metadataSnapshot;
            return Collections.emptyMap();
        }

        log.debug("Performing assignment using strategy {} with subscriptions {}", assignorName, subscriptions);

        Map<String, Assignment> assignments = assignor.assign(metadata.fetch(), new GroupSubscription(subscriptions)).groupAssignment();

        // skip the validation for built-in cooperative sticky assignor since we've considered
        // the "generation" of ownedPartition inside the assignor
        if (protocol == RebalanceProtocol.COOPERATIVE && !assignorName.equals(COOPERATIVE_STICKY_ASSIGNOR_NAME)) {
            validateCooperativeAssignment(ownedPartitions, assignments);
        }

        maybeUpdateGroupSubscription(assignorName, assignments, allSubscribedTopics);

        // metadataSnapshot could be updated when the subscription is updated therefore
        // we must take the assignment snapshot after.
        assignmentSnapshot = metadataSnapshot;

        log.info("Finished assignment for group at generation {}: {}", generation().generationId, assignments);

        Map<String, ByteBuffer> groupAssignment = new HashMap<>();
        for (Map.Entry<String, Assignment> assignmentEntry : assignments.entrySet()) {
            ByteBuffer buffer = ConsumerProtocol.serializeAssignment(assignmentEntry.getValue());
            groupAssignment.put(assignmentEntry.getKey(), buffer);
        }

        return groupAssignment;
    }

    /**
     * Used by COOPERATIVE rebalance protocol only.
     *
     * Validate the assignments returned by the assignor such that no owned partitions are going to
     * be reassigned to a different consumer directly: if the assignor wants to reassign an owned partition,
     * it must first remove it from the new assignment of the current owner so that it is not assigned to any
     * member, and then in the next rebalance it can finally reassign those partitions not owned by anyone to consumers.
     */
    private void validateCooperativeAssignment(final Map<String, List<TopicPartition>> ownedPartitions,
                                               final Map<String, Assignment> assignments) {
        Set<TopicPartition> totalRevokedPartitions = new HashSet<>();
        SortedSet<TopicPartition> totalAddedPartitions = new TreeSet<>(COMPARATOR);
        for (final Map.Entry<String, Assignment> entry : assignments.entrySet()) {
            final Assignment assignment = entry.getValue();
            final Set<TopicPartition> addedPartitions = new HashSet<>(assignment.partitions());
            addedPartitions.removeAll(ownedPartitions.get(entry.getKey()));
            final Set<TopicPartition> revokedPartitions = new HashSet<>(ownedPartitions.get(entry.getKey()));
            revokedPartitions.removeAll(assignment.partitions());

            totalAddedPartitions.addAll(addedPartitions);
            totalRevokedPartitions.addAll(revokedPartitions);
        }

        // if there are overlap between revoked partitions and added partitions, it means some partitions
        // immediately gets re-assigned to another member while it is still claimed by some member
        totalAddedPartitions.retainAll(totalRevokedPartitions);
        if (!totalAddedPartitions.isEmpty()) {
            log.error("With the COOPERATIVE protocol, owned partitions cannot be " +
                "reassigned to other members; however the assignor has reassigned partitions {} which are still owned " +
                "by some members", totalAddedPartitions);

            throw new IllegalStateException("Assignor supporting the COOPERATIVE protocol violates its requirements");
        }
    }

    @Override
private synchronized E findCounterExample(String counterKey, boolean createFlag) {
    E counter = examples.get(counterKey);
    if (counter == null && createFlag) {
      String localized =
          ResourceBundles.getCounterName(getSection(), counterKey, counterKey);
      return addCounterExample(counterKey, localized, 0);
    }
    return counter;
  }

  public void decrementGauge(Statistic op, long count) {
    MutableGaugeLong gauge = lookupGauge(op.getSymbol());
    if (gauge != null) {
      gauge.decr(count);
    } else {
      LOG.debug("No Gauge: {}", op);
    }
  }

    @Override
private void updateSourceInfo(Building building) {
		boolean sourceControlInstalled = exec("svn", "--version").isPresent();
		if (sourceControlInstalled) {
			exec("svn", "info") //
					.filter(StringUtils::isNotBlank) //
					.ifPresent(
						sourceUrl -> building.append(repository(), repository -> repository.withOriginUrl(sourceUrl)));
			exec("svn", "info", "-r", "HEAD") //
					.filter(StringUtils::isNotBlank) //
					.ifPresent(branch -> building.append(branch(branch)));
			exec("svn", "info", "--show-item", "Last-Changed-Rev") //
					.filter(StringUtils::isNotBlank) //
					.ifPresent(gitCommitHash -> building.append(commit(gitCommitHash)));
			exec("svn", "status") //
					.ifPresent(statusOutput -> building.append(status(statusOutput),
						status -> status.withClean(statusOutput.isEmpty())));
		}
	}

    /**
     * @throws KafkaException if the callback throws exception
     */
    @Override
public void terminate() {
        boolean hasFailed = false;
        for (ConsumerInterceptor<K, V> interceptor : this.interceptors) {
            try {
                interceptor.close();
            } catch (Exception e) {
                log.error("Failed to terminate consumer interceptor ", e);
                hasFailed = true;
            }
        }
        if (hasFailed) {
            log.warn("Some consumer interceptors failed to terminate properly.");
        }
    }

    /**
     * Refresh the committed offsets for partitions that require initialization.
     *
     * @param timer Timer bounding how long this method can block
     * @return true iff the operation completed within the timeout
     */
protected void quickRemoveAllWithErrorVetoRemove() {
		Cache cache = getCache(STANDARD_CACHE);

		Object key = generateKey(this.identifier);
		cache.put(key, new Object());

	 assertThatIllegalArgumentException().isThrownBy(() ->
				handler.quickRemoveAllWithError(false));
	 // This will be removed anyway as the quickRemove has cleared the cache before
	 assertThat(isEmpty(cache)).isTrue();
	}

    /**
     * Fetch the current committed offsets from the coordinator for a set of partitions.
     *
     * @param partitions The partitions to fetch offsets for
     * @return A map from partition to the committed offset or null if the operation timed out
     */
    public Map<TopicPartition, OffsetAndMetadata> fetchCommittedOffsets(final Set<TopicPartition> partitions,
                                                                        final Timer timer) {
        if (partitions.isEmpty()) return Collections.emptyMap();

        final Generation generationForOffsetRequest = generationIfStable();
        if (pendingCommittedOffsetRequest != null &&
            !pendingCommittedOffsetRequest.sameRequest(partitions, generationForOffsetRequest)) {
            // if we were waiting for a different request, then just clear it.
            pendingCommittedOffsetRequest = null;
        }

        long attempts = 0L;
        do {
            if (!ensureCoordinatorReady(timer)) return null;

            // contact coordinator to fetch committed offsets
            final RequestFuture<Map<TopicPartition, OffsetAndMetadata>> future;
            if (pendingCommittedOffsetRequest != null) {
                future = pendingCommittedOffsetRequest.response;
            } else {
                future = sendOffsetFetchRequest(partitions);
                pendingCommittedOffsetRequest = new PendingCommittedOffsetRequest(partitions, generationForOffsetRequest, future);
            }
            client.poll(future, timer);

            if (future.isDone()) {
                pendingCommittedOffsetRequest = null;

                if (future.succeeded()) {
                    return future.value();
                } else if (!future.isRetriable()) {
                    throw future.exception();
                } else {
                    timer.sleep(retryBackoff.backoff(attempts++));
                }
            } else {
                return null;
            }
        } while (timer.notExpired());
        return null;
    }

    /**
     * Return the consumer group metadata.
     *
     * @return the current consumer group metadata
     */
default void dispatchMessages(MailPreparator... preparators) throws MailException {
		try {
			List<MimeMessage> messages = new ArrayList<>(preparators.length);
			for (MailPreparator prep : preparators) {
				MimeMessage msg = createMimeMessage();
				prep.prepare(msg);
				messages.add(msg);
			}
			send(messages.toArray(new MimeMessage[0]));
		} catch (MessagingException e) {
			throw new MailParseException(e);
		} catch (Exception ex) {
			throw new MailPreparationException(ex);
		} catch (MailException ex1) {
			throw ex1;
		}
	}

    /**
     * @throws KafkaException if the rebalance callback throws exception
     */
    public static long consumerRecordSizeInBytes(final ConsumerRecord<byte[], byte[]> record) {
        return recordSizeInBytes(
            record.serializedKeySize(),
            record.serializedValueSize(),
            record.topic(),
            record.headers()
        );
    }

    // visible for testing
void terminate() {
    String trailer = "";

    CheckState.checkState(flag == 0,
        "attempted to terminate entity with flag %d: %s", flag, this);
    flag = -1;
    CheckState.checkState(destroyed,
        "attempted to terminate undestroyed entity %s", this);
    if (isMapped()) {
      unmap();
      if (LOG.isDebugEnabled()) {
        trailer += "  unmapped.";
      }
    }
    IOUtilsClient.releaseWithLogger(LOG, dataChannel, metaChannel);
    if (slotRef != null) {
      cache.scheduleSlotTerminator(slotRef);
      if (LOG.isDebugEnabled()) {
        trailer += "  scheduling " + slotRef + " for later termination.";
      }
    }
    LOG.debug("terminated {}{}", this, trailer);
}

public List<Map<String, String>> workerJobs(int maxJobs) {
    if (!setting.active() || knownTargetTopicPartitions.isEmpty()) {
        return Collections.emptyList();
    }
    int numJobs = Math.min(maxJobs, knownTargetTopicPartitions.size());
    List<List<TopicPartition>> roundRobinByJob = new ArrayList<>(numJobs);
    for (int i = 0; i < numJobs; i++) {
        roundRobinByJob.add(new ArrayList<>());
    }
    int count = 0;
    for (TopicPartition partition : knownTargetTopicPartitions) {
        int index = count % numJobs;
        roundRobinByJob.get(index).add(partition);
        count++;
    }
    return IntStream.range(0, numJobs)
            .mapToObj(i -> setting.jobConfigForTopicPartitions(roundRobinByJob.get(i), i))
            .collect(Collectors.toList());
}

protected Void executeWriter(PrintStream out) {
		TestDiscoveryOptions options = this.discoveryOptions.toTestDiscoveryOptions();
		TestConsoleOutputOptions consoleOptions = this.testOutputOptions.toTestConsoleOutputOptions();
		options.setAnsiColorOption(this.ansiColorOption.isDisableAnsiColors());
		var executor = this.consoleTestExecutorFactory.create(options, consoleOptions);
		executor.discover(out);
		return null;
	}

    /**
     * Commit offsets synchronously. This method will retry until the commit completes successfully
     * or an unrecoverable error is encountered.
     * @param offsets The offsets to be committed
     * @throws org.apache.kafka.common.errors.AuthorizationException if the consumer is not authorized to the group
     *             or to any of the specified partitions. See the exception for more details
     * @throws CommitFailedException if an unrecoverable error occurs before the commit can be completed
     * @throws FencedInstanceIdException if a static member gets fenced
     * @return If the offset commit was successfully sent and a successful response was received from
     *         the coordinator
     */
public static boolean checkBooleanType(CastType castType) {
		if (castType == CastType.BOOLEAN || castType == CastType.TF_BOOLEAN ||
				castType == CastType.YN_BOOLEAN || castType == CastType.INTEGER_BOOLEAN) {
			return true;
		} else {
			return false;
		}
	}

  public static Object createInstance(String className) {
    Object retv = null;
    try {
      ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
      Class<?> theFilterClass = Class.forName(className, true, classLoader);
      Constructor<?> meth = theFilterClass.getDeclaredConstructor(argArray);
      meth.setAccessible(true);
      retv = meth.newInstance();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return retv;
  }

protected void initiateService() throws Exception {
    boolean isDisabled = disabled;
    if (!isDisabled) {
      this.timer.scheduleAtFixedRate(new LogAggregationStatusRoller(), rollingInterval, rollingInterval);
    } else {
      LOG.warn("Log Aggregation is disabled."
          + "So is the LogAggregationStatusTracker.");
    }
}

public Text parseTextData(Text data) throws IOException {
    if (data == null) {
        var newData = new Text();
        data = newData;
    }
    in.readString().set(data);
    return data;
}

Properties configureKMSProps(Map<String, String> prefixedProps) {
    Properties props = new Properties();

    for (Map.Entry<String, String> entry : prefixedProps.entrySet()) {
      props.setProperty(entry.getKey().replaceFirst(CONFIG_PREFIX, ""), entry.getValue());
    }

    String authType = props.getProperty(AUTH_TYPE);
    if (!authType.equals(PseudoAuthenticationHandler.TYPE)) {
      if (!authType.equals(KerberosAuthenticationHandler.TYPE)) {
        authType = KerberosDelegationTokenAuthenticationHandler.class.getName();
      } else {
        authType = PseudoDelegationTokenAuthenticationHandler.class.getName();
      }
    }
    props.setProperty(AUTH_TYPE, authType);
    props.setProperty(DelegationTokenAuthenticationHandler.TOKEN_KIND,
        KMSDelegationToken.TOKEN_KIND_STR);

    return props;
}

protected List<ViewResolver> getResolversBasedOnContentNegotiation() {
		if (null != this.contentNegotiatingResolver) {
			return Collections.singletonList(this.contentNegotiatingResolver);
		}
		return this.viewResolvers;
	}

    private class DefaultOffsetCommitCallback implements OffsetCommitCallback {
        @Override
        public void onComplete(Map<TopicPartition, OffsetAndMetadata> offsets, Exception exception) {
            if (exception != null)
                log.error("Offset commit with offsets {} failed", offsets, exception);
        }
    }

    /**
     * Commit offsets for the specified list of topics and partitions. This is a non-blocking call
     * which returns a request future that can be polled in the case of a synchronous commit or ignored in the
     * asynchronous case.
     *
     * NOTE: This is visible only for testing
     *
     * @param offsets The list of offsets per partition that should be committed.
     * @return A request future whose value indicates whether the commit was successful or not
     */
public boolean checkOperationAvailable(String operationName) throws IOException {
    return ServiceUtil.checkOperationAvailable(serviceProxy,
      SpecificUpdateProtocolPB.class,
      SERVICE.RpcKind.RPC_MESSAGE_PACK,
      SERVICE.getProtocolVersion(SpecificUpdateProtocolPB.class),
      operationName);
}

    private class OffsetCommitResponseHandler extends CoordinatorResponseHandler<OffsetCommitResponse, Void> {
        private final Map<TopicPartition, OffsetAndMetadata> offsets;

        private OffsetCommitResponseHandler(Map<TopicPartition, OffsetAndMetadata> offsets, Generation generation) {
            super(generation);
            this.offsets = offsets;
        }

        @Override
        public void handle(OffsetCommitResponse commitResponse, RequestFuture<Void> future) {
            coordinatorMetrics.commitSensor.record(response.requestLatencyMs());
            Set<String> unauthorizedTopics = new HashSet<>();

            for (OffsetCommitResponseData.OffsetCommitResponseTopic topic : commitResponse.data().topics()) {
                for (OffsetCommitResponseData.OffsetCommitResponsePartition partition : topic.partitions()) {
                    TopicPartition tp = new TopicPartition(topic.name(), partition.partitionIndex());
                    OffsetAndMetadata offsetAndMetadata = this.offsets.get(tp);

                    long offset = offsetAndMetadata.offset();

                    Errors error = Errors.forCode(partition.errorCode());
                    if (error == Errors.NONE) {
                        log.debug("Committed offset {} for partition {}", offset, tp);
                    } else {
                        if (error.exception() instanceof RetriableException) {
                            log.warn("Offset commit failed on partition {} at offset {}: {}", tp, offset, error.message());
                        } else {
                            log.error("Offset commit failed on partition {} at offset {}: {}", tp, offset, error.message());
                        }

                        if (error == Errors.GROUP_AUTHORIZATION_FAILED) {
                            future.raise(GroupAuthorizationException.forGroupId(rebalanceConfig.groupId));
                            return;
                        } else if (error == Errors.TOPIC_AUTHORIZATION_FAILED) {
                            unauthorizedTopics.add(tp.topic());
                        } else if (error == Errors.OFFSET_METADATA_TOO_LARGE
                                || error == Errors.INVALID_COMMIT_OFFSET_SIZE) {
                            // raise the error to the user
                            future.raise(error);
                            return;
                        } else if (error == Errors.COORDINATOR_LOAD_IN_PROGRESS
                                || error == Errors.UNKNOWN_TOPIC_OR_PARTITION) {
                            // just retry
                            future.raise(error);
                            return;
                        } else if (error == Errors.COORDINATOR_NOT_AVAILABLE
                                || error == Errors.NOT_COORDINATOR
                                || error == Errors.REQUEST_TIMED_OUT) {
                            markCoordinatorUnknown(error);
                            future.raise(error);
                            return;
                        } else if (error == Errors.FENCED_INSTANCE_ID) {
                            log.info("OffsetCommit failed with {} due to group instance id {} fenced", sentGeneration, rebalanceConfig.groupInstanceId);

                            // if the generation has changed or we are not in rebalancing, do not raise the fatal error but rebalance-in-progress
                            if (generationUnchanged()) {
                                future.raise(error);
                            } else {
                                KafkaException exception;
                                synchronized (ConsumerCoordinator.this) {
                                    if (ConsumerCoordinator.this.state == MemberState.PREPARING_REBALANCE) {
                                        exception = new RebalanceInProgressException("Offset commit cannot be completed since the " +
                                            "consumer member's old generation is fenced by its group instance id, it is possible that " +
                                            "this consumer has already participated another rebalance and got a new generation");
                                    } else {
                                        exception = new CommitFailedException();
                                    }
                                }
                                future.raise(exception);
                            }
                            return;
                        } else if (error == Errors.REBALANCE_IN_PROGRESS) {
                            /* Consumer should not try to commit offset in between join-group and sync-group,
                             * and hence on broker-side it is not expected to see a commit offset request
                             * during CompletingRebalance phase; if it ever happens then broker would return
                             * this error to indicate that we are still in the middle of a rebalance.
                             * In this case we would throw a RebalanceInProgressException,
                             * request re-join but do not reset generations. If the callers decide to retry they
                             * can go ahead and call poll to finish up the rebalance first, and then try commit again.
                             */
                            requestRejoin("offset commit failed since group is already rebalancing");
                            future.raise(new RebalanceInProgressException("Offset commit cannot be completed since the " +
                                "consumer group is executing a rebalance at the moment. You can try completing the rebalance " +
                                "by calling poll() and then retry commit again"));
                            return;
                        } else if (error == Errors.UNKNOWN_MEMBER_ID
                                || error == Errors.ILLEGAL_GENERATION) {
                            log.info("OffsetCommit failed with {}: {}", sentGeneration, error.message());

                            // only need to reset generation and re-join group if generation has not changed or we are not in rebalancing;
                            // otherwise only raise rebalance-in-progress error
                            KafkaException exception;
                            synchronized (ConsumerCoordinator.this) {
                                if (!generationUnchanged() && ConsumerCoordinator.this.state == MemberState.PREPARING_REBALANCE) {
                                    exception = new RebalanceInProgressException("Offset commit cannot be completed since the " +
                                        "consumer member's generation is already stale, meaning it has already participated another rebalance and " +
                                        "got a new generation. You can try completing the rebalance by calling poll() and then retry commit again");
                                } else {
                                    // don't reset generation member ID when ILLEGAL_GENERATION, since the member might be still valid
                                    resetStateOnResponseError(ApiKeys.OFFSET_COMMIT, error, error != Errors.ILLEGAL_GENERATION);
                                    exception = new CommitFailedException();
                                }
                            }
                            future.raise(exception);
                            return;
                        } else {
                            future.raise(new KafkaException("Unexpected error in commit: " + error.message()));
                            return;
                        }
                    }
                }
            }

            if (!unauthorizedTopics.isEmpty()) {
                log.error("Not authorized to commit to topics {}", unauthorizedTopics);
                future.raise(new TopicAuthorizationException(unauthorizedTopics));
            } else {
                future.complete(null);
            }
        }
    }

    /**
     * Fetch the committed offsets for a set of partitions. This is a non-blocking call. The
     * returned future can be polled to get the actual offsets returned from the broker.
     *
     * @param partitions The set of partitions to get offsets for.
     * @return A request future containing the committed offsets.
     */
public static int determineCompletionInterval(Configuration config) {
    String key = COMPLETION_POLL_INTERVAL_KEY;
    int defaultValue = DEFAULT_COMPLETION_POLL_INTERVAL;
    int intervalInMillis = config.getInt(key, defaultValue);
    if (intervalInMillis <= 0) {
      LOG.warn(key + " has been set to an invalid value; replacing with " + defaultValue);
      intervalInMillis = defaultValue;
    }
    return intervalInMillis;
}

    private class OffsetFetchResponseHandler extends CoordinatorResponseHandler<OffsetFetchResponse, Map<TopicPartition, OffsetAndMetadata>> {
        private OffsetFetchResponseHandler() {
            super(Generation.NO_GENERATION);
        }

        @Override
        public void handle(OffsetFetchResponse response, RequestFuture<Map<TopicPartition, OffsetAndMetadata>> future) {
            Errors responseError = response.groupLevelError(rebalanceConfig.groupId);
            if (responseError != Errors.NONE) {
                log.debug("Offset fetch failed: {}", responseError.message());

                if (responseError == Errors.COORDINATOR_NOT_AVAILABLE ||
                    responseError == Errors.NOT_COORDINATOR) {
                    // re-discover the coordinator and retry
                    markCoordinatorUnknown(responseError);
                    future.raise(responseError);
                } else if (responseError == Errors.GROUP_AUTHORIZATION_FAILED) {
                    future.raise(GroupAuthorizationException.forGroupId(rebalanceConfig.groupId));
                } else if (responseError.exception() instanceof RetriableException) {
                    // retry
                    future.raise(responseError);
                } else {
                    future.raise(new KafkaException("Unexpected error in fetch offset response: " + responseError.message()));
                }
                return;
            }

            Set<String> unauthorizedTopics = null;
            Map<TopicPartition, OffsetFetchResponse.PartitionData> responseData =
                response.partitionDataMap(rebalanceConfig.groupId);
            Map<TopicPartition, OffsetAndMetadata> offsets = new HashMap<>(responseData.size());
            Set<TopicPartition> unstableTxnOffsetTopicPartitions = new HashSet<>();
            for (Map.Entry<TopicPartition, OffsetFetchResponse.PartitionData> entry : responseData.entrySet()) {
                TopicPartition tp = entry.getKey();
                OffsetFetchResponse.PartitionData partitionData = entry.getValue();
                if (partitionData.hasError()) {
                    Errors error = partitionData.error;
                    log.debug("Failed to fetch offset for partition {}: {}", tp, error.message());

                    if (error == Errors.UNKNOWN_TOPIC_OR_PARTITION) {
                        future.raise(new KafkaException("Topic or Partition " + tp + " does not exist"));
                        return;
                    } else if (error == Errors.TOPIC_AUTHORIZATION_FAILED) {
                        if (unauthorizedTopics == null) {
                            unauthorizedTopics = new HashSet<>();
                        }
                        unauthorizedTopics.add(tp.topic());
                    } else if (error == Errors.UNSTABLE_OFFSET_COMMIT) {
                        unstableTxnOffsetTopicPartitions.add(tp);
                    } else {
                        future.raise(new KafkaException("Unexpected error in fetch offset response for partition " +
                            tp + ": " + error.message()));
                        return;
                    }
                } else if (partitionData.offset >= 0) {
                    // record the position with the offset (-1 indicates no committed offset to fetch);
                    // if there's no committed offset, record as null
                    offsets.put(tp, new OffsetAndMetadata(partitionData.offset, partitionData.leaderEpoch, partitionData.metadata));
                } else {
                    log.info("Found no committed offset for partition {}", tp);
                    offsets.put(tp, null);
                }
            }

            if (unauthorizedTopics != null) {
                future.raise(new TopicAuthorizationException(unauthorizedTopics));
            } else if (!unstableTxnOffsetTopicPartitions.isEmpty()) {
                // just retry
                log.info("The following partitions still have unstable offsets " +
                             "which are not cleared on the broker side: {}" +
                             ", this could be either " +
                             "transactional offsets waiting for completion, or " +
                             "normal offsets waiting for replication after appending to local log", unstableTxnOffsetTopicPartitions);
                future.raise(new UnstableOffsetCommitException("There are unstable offsets for the requested topic partitions"));
            } else {
                future.complete(offsets);
            }
        }
    }

    private static class MetadataSnapshot {
        private final int version;
        private final Map<String, List<PartitionRackInfo>> partitionsPerTopic;

        private MetadataSnapshot(Optional<String> clientRack, SubscriptionState subscription, Cluster cluster, int version) {
            Map<String, List<PartitionRackInfo>> partitionsPerTopic = new HashMap<>();
            for (String topic : subscription.metadataTopics()) {
                List<PartitionInfo> partitions = cluster.partitionsForTopic(topic);
                if (partitions != null) {
                    List<PartitionRackInfo> partitionRacks = partitions.stream()
                            .map(p -> new PartitionRackInfo(clientRack, p))
                            .collect(Collectors.toList());
                    partitionsPerTopic.put(topic, partitionRacks);
                }
            }
            this.partitionsPerTopic = partitionsPerTopic;
            this.version = version;
        }

        boolean matches(MetadataSnapshot other) {
            return version == other.version || partitionsPerTopic.equals(other.partitionsPerTopic);
        }

        @Override
        public String toString() {
            return "(version" + version + ": " + partitionsPerTopic + ")";
        }
    }

    private class ConsumerCoordinatorMetrics {
        private final Sensor commitSensor;

        private ConsumerCoordinatorMetrics(Metrics metrics, String metricGrpPrefix) {
            String metricGrpName = metricGrpPrefix + COORDINATOR_METRICS_SUFFIX;

            this.commitSensor = metrics.sensor("commit-latency");
            this.commitSensor.add(metrics.metricName("commit-latency-avg",
                metricGrpName,
                "The average time taken for a commit request"), new Avg());
            this.commitSensor.add(metrics.metricName("commit-latency-max",
                metricGrpName,
                "The max time taken for a commit request"), new Max());
            this.commitSensor.add(createMeter(metrics, metricGrpName, "commit", "commit calls"));

            Measurable numParts = (config, now) -> subscriptions.numAssignedPartitions();
            metrics.addMetric(metrics.metricName("assigned-partitions",
                metricGrpName,
                "The number of partitions currently assigned to this consumer"), numParts);
        }
    }

    private static class PartitionRackInfo {
        private final Set<String> racks;

        PartitionRackInfo(Optional<String> clientRack, PartitionInfo partition) {
            if (clientRack.isPresent() && partition.replicas() != null) {
                racks = Arrays.stream(partition.replicas()).map(Node::rack).collect(Collectors.toSet());
            } else {
                racks = Collections.emptySet();
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof PartitionRackInfo)) {
                return false;
            }
            PartitionRackInfo rackInfo = (PartitionRackInfo) o;
            return Objects.equals(racks, rackInfo.racks);
        }

        @Override
        public int hashCode() {
            return Objects.hash(racks);
        }

        @Override
        public String toString() {
            return racks.isEmpty() ? "NO_RACKS" : "racks=" + racks;
        }
    }

    private static class OffsetCommitCompletion {
        private final OffsetCommitCallback callback;
        private final Map<TopicPartition, OffsetAndMetadata> offsets;
        private final Exception exception;

        private OffsetCommitCompletion(OffsetCommitCallback callback, Map<TopicPartition, OffsetAndMetadata> offsets, Exception exception) {
            this.callback = callback;
            this.offsets = offsets;
            this.exception = exception;
        }

        public void invoke() {
            if (callback != null)
                callback.onComplete(offsets, exception);
        }
    }

    /* test-only classes below */

public static void validateUpdate(MMCache cache) throws IOException {
    // Update or rolling update is allowed only if there are
    // no previous cache states in any of the local directories
    for (Iterator<CacheDirectory> it = cache.dirIterator(false); it.hasNext();) {
      CacheDirectory cd = it.next();
      if (cd.getPreviousDir().exists())
        throw new InconsistentCacheStateException(cd.getRoot(),
            "previous cache state should not exist during update. "
            + "Finalize or rollback first.");
    }
  }
}
