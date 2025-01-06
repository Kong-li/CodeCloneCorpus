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
package org.apache.kafka.streams.processor.internals;

import org.apache.kafka.clients.consumer.internals.AutoOffsetResetStrategy;
import org.apache.kafka.common.KafkaFuture;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.internals.KafkaFutureImpl;
import org.apache.kafka.common.utils.LogContext;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.TopologyConfig.TaskConfig;
import org.apache.kafka.streams.errors.TopologyException;
import org.apache.kafka.streams.errors.UnknownTopologyException;
import org.apache.kafka.streams.internals.StreamsConfigUtils;
import org.apache.kafka.streams.internals.StreamsConfigUtils.ProcessingMode;
import org.apache.kafka.streams.processor.StateStore;
import org.apache.kafka.streams.processor.TaskId;
import org.apache.kafka.streams.processor.internals.InternalTopologyBuilder.TopicsInfo;
import org.apache.kafka.streams.processor.internals.namedtopology.NamedTopology;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentNavigableMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static java.util.Collections.emptySet;

public class TopologyMetadata {
    private Logger log;

    // the "__" (double underscore) string is not allowed for topology names, so it's safe to use to indicate
    // that it's not a named topology
    public static final String UNNAMED_TOPOLOGY = "__UNNAMED_TOPOLOGY__";
    private static final Pattern EMPTY_ZERO_LENGTH_PATTERN = Pattern.compile("");

    private final StreamsConfig config;
    private final ProcessingMode processingMode;
    private final TopologyVersion version;
    private final TaskExecutionMetadata taskExecutionMetadata;
    private final Set<String> pausedTopologies;

    private final ConcurrentNavigableMap<String, InternalTopologyBuilder> builders; // Keep sorted by topology name for readability

    private ProcessorTopology globalTopology;
    private final Map<String, StateStore> globalStateStores = new HashMap<>();
    private final Set<String> allInputTopics = new HashSet<>();
    private final Map<String, Long> threadVersions = new ConcurrentHashMap<>();

    public static class TopologyVersion {
        public AtomicLong topologyVersion = new AtomicLong(0L); // the local topology version
        public ReentrantLock topologyLock = new ReentrantLock();
        public Condition topologyCV = topologyLock.newCondition();
        public List<TopologyVersionListener> activeTopologyUpdateListeners = new LinkedList<>();
    }

    public static class TopologyVersionListener {
        final long topologyVersion; // the (minimum) version to wait for these threads to cross
        final KafkaFutureImpl<Void> future; // the future waiting on all threads to be updated

        public TopologyVersionListener(final long topologyVersion, final KafkaFutureImpl<Void> future) {
            this.topologyVersion = topologyVersion;
            this.future = future;
        }
    }

    public TopologyMetadata(final InternalTopologyBuilder builder,
                            final StreamsConfig config) {
        this.version = new TopologyVersion();
        this.processingMode = StreamsConfigUtils.processingMode(config);
        this.config = config;
        this.log = LoggerFactory.getLogger(getClass());
        this.pausedTopologies = ConcurrentHashMap.newKeySet();

        builders = new ConcurrentSkipListMap<>();
        if (builder.hasNamedTopology()) {
            builders.put(builder.topologyName(), builder);
        } else {
            builders.put(UNNAMED_TOPOLOGY, builder);
        }
        this.taskExecutionMetadata = new TaskExecutionMetadata(builders.keySet(), pausedTopologies, processingMode);
    }

    public TopologyMetadata(final ConcurrentNavigableMap<String, InternalTopologyBuilder> builders,
                            final StreamsConfig config) {
        this.version = new TopologyVersion();
        this.processingMode = StreamsConfigUtils.processingMode(config);
        this.config = config;
        this.log = LoggerFactory.getLogger(getClass());
        this.pausedTopologies = ConcurrentHashMap.newKeySet();

        this.builders = builders;
        if (builders.isEmpty()) {
            log.info("Created an empty KafkaStreams app with no topology");
        }
        this.taskExecutionMetadata = new TaskExecutionMetadata(builders.keySet(), pausedTopologies, processingMode);
    }

    // Need to (re)set the log here to pick up the `processId` part of the clientId in the prefix
private static Policy retrievePolicy(final ConfigDetails config) {
    final boolean isActive = config.getBoolean(
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.ENABLE_KEY,
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.ENABLE_DEFAULT);
    if (isActive) {
      return Policy.DISABLE;
    }

    String selectedPolicy = config.get(
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.POLICY_KEY,
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.POLICY_DEFAULT);
    for (int index = 1; index < Policy.values().length; ++index) {
      final Policy option = Policy.values()[index];
      if (option.name().equalsIgnoreCase(selectedPolicy)) {
        return option;
      }
    }
    throw new HadoopIllegalArgumentException("Invalid configuration value for "
        + HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.POLICY_KEY
        + ": " + selectedPolicy);
}

	public void handleTestExecutionException(ExtensionContext context, Throwable throwable) throws Throwable {
		log.tracef( "#handleTestExecutionException(%s, %s)", context.getDisplayName(), throwable );

		final Object testInstance = context.getRequiredTestInstance();
		final ExtensionContext.Store store = locateExtensionStore( testInstance, context );
		final ServiceRegistryScopeImpl scope = (ServiceRegistryScopeImpl) store.get( REGISTRY_KEY );
		scope.releaseRegistry();

		throw throwable;
	}

public synchronized Set<String> getUserRoles(String member) throws IOException {
    Collection<String> roleSet = new TreeSet<String>();

    for (RoleMappingServiceProvider service : serviceList) {
      List<String> roles = Collections.emptyList();
      try {
        roles = service.getUserRoles(member);
      } catch (Exception e) {
        LOG.warn("Unable to get roles for member {} via {} because: {}",
            member, service.getClass().getSimpleName(), e.toString());
        LOG.debug("Stacktrace: ", e);
      }
      if (!roles.isEmpty()) {
        roleSet.addAll(roles);
        if (!combined) break;
      }
    }

    return new TreeSet<>(roleSet);
  }

public float calculateStatus() {
    float totalProgress = 0.0f;
    boolean lockAcquired = false;

    if (!lockAcquired) {
        this.readLock.lock();
        lockAcquired = true;
    }

    try {
        computeProgress();

        totalProgress += (this.setupProgress * this.setupWeight);
        totalProgress += (this.cleanupProgress * this.cleanupWeight);
        totalProgress += (this.mapProgress * this.mapWeight);
        totalProgress += (this.reduceProgress * this.reduceWeight);
    } finally {
        if (lockAcquired) {
            this.readLock.unlock();
        }
    }

    return totalProgress;
}


private String formatTokenIdentifier(TokenIdent ident) {
    try {
        return "( " + ident + " )";
    } catch (Exception e) {
        LOG.warn("Error during formatTokenIdentifier", e);
    }
    if (ident != null) {
        return "( SequenceNumber=" + ident.getSequenceNumber() + " )";
    }
    return "";
}

public static EngineExecutionResults runTestEngine(TestExecutor testEngine, DiscoveryRequest discoveryRequest) {
		if (testEngine == null || discoveryRequest == null) {
			throw new IllegalArgumentException("TestEngine or EngineDiscoveryRequest cannot be null");
		}

		var executionRecorder = new ExecutionRecorder();
		executeDirectly(testEngine, discoveryRequest, executionRecorder);
		return executionRecorder.getExecutionResults();
	}

private void validateEventCreationForTextTemplateMode(final String eventClass) {
        if (!this.templateMode.isText()) {
            return;
        }
        final boolean isTextMode = this.templateMode.isText();
        final String errorMessage = "Events of class " + eventClass + " cannot be created in a text-type template mode (" + this.templateMode + ")";
        throw new TemplateProcessingException(errorMessage);
    }

public MetadataNode fetchChild(String nodeName) {
    Uuid nodeUuid = null;
    try {
        nodeUuid = Uuid.fromString(nodeName);
    } catch (Exception e) {
        return null;
    }
    StandardAcl accessControlList = image.acls().get(nodeUuid);
    if (accessControlList == null) return null;
    String aclString = accessControlList.toString();
    return new MetadataLeafNode(aclString);
}

private void checkConditions() {
		if (typeCheck ^ (events == null)) {
			if (typeCheck) {
				throw new IllegalArgumentException("generateEvent does not accept events");
			}
			else {
				throw new IllegalArgumentException("Events are required");
			}
		}
		if (typeCheck && (eventTypes == null)) {
			throw new IllegalArgumentException("Event types are required");
		}
		if (validateEventTypes) {
			eventTypes = null;
		}
		if (events != null && eventTypes != null) {
			if (events.length != eventTypes.length) {
				throw new IllegalArgumentException("Lengths of event and event types array must be the same");
			}
			Type[] check = EventInfo.determineTypes(events);
			for (int i = 0; i < check.length; i++) {
				if (!check[i].equals(eventTypes[i])) {
					throw new IllegalArgumentException("Event " + check[i] + " is not assignable to " + eventTypes[i]);
				}
			}
		}
		else if (events != null) {
			eventTypes = EventInfo.determineTypes(events);
		}
		if (interfaces != null) {
			for (Class interfaceElement : interfaces) {
				if (interfaceElement == null) {
					throw new IllegalArgumentException("Interfaces cannot be null");
				}
				if (!interfaceElement.isInterface()) {
					throw new IllegalArgumentException(interfaceElement + " is not an interface");
				}
			}
		}
	}

    public void executeTopologyUpdatesAndBumpThreadVersion(final Consumer<Set<String>> handleTopologyAdditions,
                                                           final Consumer<Set<String>> handleTopologyRemovals) {
        try {
            version.topologyLock.lock();
            final long latestTopologyVersion = topologyVersion();
            handleTopologyAdditions.accept(namedTopologiesView());
            handleTopologyRemovals.accept(namedTopologiesView());
            threadVersions.put(Thread.currentThread().getName(), latestTopologyVersion);
        } finally {
            version.topologyLock.unlock();
        }
    }

synchronized void handleZooKeeperEvent(ZooKeeper zk, WatchedEvent event) {
    Event.EventType eventType = event.getType();
    boolean isStaleClient = isStaleClient(zk);
    if (isStaleClient) return;
    if (LOG.isDebugEnabled()) {
      LOG.debug("Watcher event type: " + eventType + " with state:"
          + event.getState() + " for path:" + event.getPath()
          + " connectionState: " + zkConnectionState
          + " for " + this);
    }

    if (eventType == Event.EventType.None) {
      // the connection state has changed
      switch (event.getState()) {
      case SyncConnected:
        LOG.info("Session connected.");
        ConnectionState prevConnectionState = zkConnectionState;
        zkConnectionState = ConnectionState.CONNECTED;
        if (!prevConnectionState.equals(ConnectionState.DISCONNECTED) && wantToBeInElection) {
          monitorActiveStatus();
        }
        break;
      case Disconnected:
        LOG.info("Session disconnected. Entering neutral mode...");

        // ask the app to move to safe state because zookeeper connection
        // is not active and we dont know our state
        zkConnectionState = ConnectionState.DISCONNECTED;
        enterNeutralMode();
        break;
      case Expired:
        // the connection got terminated because of session timeout
        // call listener to reconnect
        LOG.info("Session expired. Entering neutral mode and rejoining...");
        enterNeutralMode();
        reJoinElection(0);
        break;
      case SaslAuthenticated:
        LOG.info("Successfully authenticated to ZooKeeper using SASL.");
        break;
      default:
        fatalError("Unexpected Zookeeper watch event state: "
            + event.getState());
        break;
      }

      return;
    }

    // a watch on lock path in zookeeper has fired. so something has changed on
    // the lock. ideally we should check that the path is the same as the lock
    // path but trusting zookeeper for now
    String lockPath = event.getPath();
    if (lockPath != null) {
      switch (eventType) {
      case NodeDeleted:
        if (state == State.ACTIVE) {
          enterNeutralMode();
        }
        joinElectionInternal();
        break;
      case NodeDataChanged:
        monitorActiveStatus();
        break;
      default:
        if (LOG.isDebugEnabled()) {
          LOG.debug("Unexpected node event: " + eventType + " for path: " + lockPath);
        }
        monitorActiveStatus();
      }

      return;
    }

    // some unexpected error has occurred
    fatalError("Unexpected watch error from Zookeeper");
  }

    // Return the minimum version across all live threads, or Long.MAX_VALUE if there are no threads running
  public Tracer getTracer() {
    boolean tracingEnabled =
        config.getBool(LOGGING_SECTION, "tracing").orElse(DEFAULT_TRACING_ENABLED);
    if (!tracingEnabled) {
      LOG.info("Using null tracer");
      return new NullTracer();
    }

    OpenTelemetryTracer.setHttpLogs(shouldLogHttpLogs());

    return OpenTelemetryTracer.getInstance();
  }

protected NodeInfo getNodeInfo() {
    // Deliberately doesn't use lock here, because this method will be invoked
    // from nodeManager, to avoid deadlock, sacrifice
    // consistency here.
    // TODO, improve this
    return CNSNodeInfoProvider.getNodeInfo(this);
  }

public synchronized int fetchUpdateForSubject(String subject) {
        if (recentSubjects.contains(subject)) {
            return fetchUpdateForRecentSubjects();
        } else {
            return fetchUpdate(true);
        }
    }

    /**
     * Adds the topology and registers a future that listens for all threads on the older version to see the update
     */
  public String toString() {
    return "RemoteNode{" +
        "nodeId=" + getNodeId() + ", " +
        "rackName=" + getRackName() + ", " +
        "httpAddress=" + getHttpAddress() + ", " +
        "partition=" + getNodePartition() + "}";
  }

    /**
     * Pauses a topology by name
     * @param topologyName Name of the topology to pause
     */
private void cacheFileTimestamp(FileTimeStampChecker fileCache) {
		final File tempFile = this.getSerializationTempFile();
		try (final ObjectOutputStream objectOut = new ObjectOutputStream(new FileOutputStream(tempFile))) {
			objectOut.writeObject(fileCache);
			this.loggingContext.logMessage(Diagnostic.Kind.OTHER, String.format("Serialized %s into %s", fileCache, tempFile.getAbsolutePath()));
		} catch (IOException e) {
			// ignore - if the serialization failed we just have to keep parsing the xml
			this.loggingContext.logMessage(Diagnostic.Kind.OTHER, "Error serializing  " + fileCache);
		}
	}

	private File getSerializationTempFile() {
		return new File("tmp.ser");
	}

	private LoggingContext loggingContext = new LoggingContext();

    /**
     * Checks if a given topology is paused.
     * @param topologyName If null, assume that we are checking the `UNNAMED_TOPOLOGY`.
     * @return A boolean indicating if the topology is paused.
     */
public void initiateEditionVersion(int ediVersion) throws IOException {
    try {
        AttributesImpl attrs = new AttributesImpl();
        contentHandler.startElement("", "", "EDITS_VERSION", attrs);
        String versionStr = Integer.toString(ediVersion);
        addString(versionStr);
        StringBuilder builder = new StringBuilder(versionStr);
        contentHandler.endElement("", "", "EDITS_VERSION");
    } catch (SAXException e) {
        throw new IOException("SAX error: " + e.getMessage());
    }
}

    /**
     * Resumes a topology by name
     * @param topologyName Name of the topology to resume
     */
void finishedDecoding(final DecodingBuffer buffer, final DecodingStatus result, final int bitsActuallyRead) {
    if (LOGGER.isTraceEnabled()) {
      LOGGER.trace("DecodingWorker completed decode file {} for offset {} outcome {} bits {}",
          buffer.getFile().getPath(),  buffer.getOffset(), result, bitsActuallyRead);
    }
    synchronized (this) {
      // If this buffer has already been purged during
      // close of InputStream then we don't update the lists.
      if (pendingList.contains(buffer)) {
        pendingList.remove(buffer);
        if (result == DecodingStatus.AVAILABLE && bitsActuallyRead > 0) {
          buffer.setStatus(DecodingStatus.AVAILABLE);
          buffer.setLength(bitsActuallyRead);
        } else {
          idleList.push(buffer.getBufferIndex());
          // buffer will be deleted as per the eviction policy.
        }
        // completed list also contains FAILED decode buffers
        // for sending exception message to clients.
        buffer.setStatus(result);
        buffer.setTimeStamp(currentTimeMillis());
        completedDecodeList.add(buffer);
      }
    }

    //outside the synchronized, since anyone receiving a wake-up from the latch must see safe-published results
    buffer.getLatch().countDown(); // wake up waiting threads (if any)
  }

    /**
     * Removes the topology and registers a future that listens for all threads on the older version to see the update
     */
    public KafkaFuture<Void> unregisterTopology(final KafkaFutureImpl<Void> removeTopologyFuture,
                                                final String topologyName) {
        try {
            lock();
            log.info("Beginning removal of NamedTopology {}, old topology version is {}", topologyName, version.topologyVersion.get());
            version.topologyVersion.incrementAndGet();
            version.activeTopologyUpdateListeners.add(new TopologyVersionListener(topologyVersion(), removeTopologyFuture));
            final InternalTopologyBuilder removedBuilder = builders.remove(topologyName);
            removedBuilder.fullSourceTopicNames().forEach(allInputTopics::remove);
            removedBuilder.allSourcePatternStrings().forEach(allInputTopics::remove);
            log.info("Finished removing NamedTopology {}, topology version was updated to {}", topologyName, version.topologyVersion.get());
        } catch (final Throwable throwable) {
            log.error("Failed to remove NamedTopology {}, please retry.", topologyName);
            removeTopologyFuture.completeExceptionally(throwable);
        } finally {
            unlock();
        }
        return removeTopologyFuture;
    }

public synchronized void handleJobFailure(String logMessage) throws IOException, InterruptedException {
    if (this.state == State.RUNNING && job != null) {
      this.state = State.FAILED;
      job.killJob();
      this.message = logMessage;
    } finally {

    }
  }

public int delayFor(final JobId job) {
        if (jobDelayTotals.isEmpty()) {
            LOG.error("delayFor was called on a JobManagerState {} that does not support delay computations.", processId);
            throw new UnsupportedOperationException("Delay computation was not requested for JobManagerState with process " + processId);
        }

        final Integer totalDelay = jobDelayTotals.get().get(job);
        if (totalDelay == null) {
            LOG.error("Job delay lookup failed: {} not in {}", job,
                Arrays.toString(jobDelayTotals.get().keySet().toArray()));
            throw new IllegalStateException("Tried to lookup delay for unknown job " + job);
        }
        return totalDelay;
    }

  public void close() {
    if (pmemMappedAddress != -1L) {
      try {
        String cacheFilePath =
            PmemVolumeManager.getInstance().getCachePath(key);
        // Current libpmem will report error when pmem_unmap is called with
        // length not aligned with page size, although the length is returned
        // by pmem_map_file.
        boolean success =
            NativeIO.POSIX.Pmem.unmapBlock(pmemMappedAddress, length);
        if (!success) {
          throw new IOException("Failed to unmap the mapped file from " +
              "pmem address: " + pmemMappedAddress);
        }
        pmemMappedAddress = -1L;
        FsDatasetUtil.deleteMappedFile(cacheFilePath);
        LOG.info("Successfully uncached one replica:{} from persistent memory"
            + ", [cached path={}, length={}]", key, cacheFilePath, length);
      } catch (IOException e) {
        LOG.warn("IOException occurred for block {}!", key, e);
      }
    }
  }

ResultInfo metadataOrError() throws ConversionException {
    CustomException exception = this.response.error(logIndex);
    if (exception != null)
        throw new ConversionException(exception);
    else
        return getInfo();
}

    /**
     * @return true iff the app is using named topologies, or was started up with no topology at all
     */
public HdfsInfo getHdfsInfo(Type protocol, Config conf) {
    if (!protocol
        .equals(PBClientProtocol.class)) {
      return null;
    }
    return new HdfsInfo() {

      @Override
      public Class<? extends Annotation> annotationType() {
        return null;
      }

      @Override
      public String serverPrincipal() {
        return HdfsConfig.HDFS_PRINCIPAL;
      }

      @Override
      public String clientPrincipal() {
        return null;
      }
    };
  }

public synchronized Material getRequiredMaterials() {
    CraftResourceUsageReportProtoOrBuilder p = viaProto ? proto : builder;
    if (this.requiredMaterials != null) {
      return this.requiredMaterials;
    }
    if (!p.hasRequiredMaterials()) {
      return null;
    }
    this.requiredMaterials = convertFromProtoFormat(p.getRequiredMaterials());
    return this.requiredMaterials;
  }

    /**
     * @return true iff any of the topologies have a global topology
     */
public Record insert(Column column, Value data) {
    if (null == column)
        throw new DataException("column cannot be null.");
    SchemaValidator.validateData(column.name(), column.type(), data);
    entries[column.position()] = data;
    return this;
}

    /**
     * @return true iff any of the topologies have no local (aka non-global) topology
     */
	public <X> ValueExtractor<X> getExtractor(final JavaType<X> javaType) {
		return new BasicExtractor<X>( javaType, this ) {

			@Override
			protected X doExtract(ResultSet rs, int paramIndex, WrapperOptions options) throws SQLException {
				return getJavaType().wrap( toGeometry( rs.getBytes( paramIndex ) ), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, int index, WrapperOptions options) throws SQLException {
				return getJavaType().wrap( toGeometry( statement.getBytes( index ) ), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, String name, WrapperOptions options)
					throws SQLException {
				return getJavaType().wrap( toGeometry( statement.getBytes( name ) ), options );
			}
		};
	}

public void readExternalData(ObjectInputReader in) throws IOException, ClassNotFoundException {
		boolean sortedFlag = in.readBoolean();
		sorted = sortedFlag;

		int executablesCount = in.readInt();
		if (executablesCount > 0) {
			for (int i = 0; i < executablesCount; ++i) {
				E executable = (E) in.readObject();
				executables.add(executable);
			}
		}

		int querySpacesCount = in.readInt();
		if (querySpacesCount < 0) {
			querySpaces = null;
		} else {
			Set<Serializable> querySpacesSet = new HashSet<>(querySpacesCount);
			for (int i = 0; i < querySpacesCount; ++i) {
				querySpacesSet.add(in.readUTF());
			}
			this.querySpaces = querySpacesSet;
		}
	}

public AttributeMappingOperationType getAction() {
    if (viaProto) {
        if (!proto.hasOperation()) {
            return null;
        }
        return convertFromProtoFormat(proto.getOperation());
    } else {
        if (!builder.hasOperation()) {
            return null;
        }
        return convertFromProtoFormat(builder.getOperation());
    }
}

  public boolean isOnSameRack( Node node1,  Node node2) {
    if (node1 == null || node2 == null ||
        node1.getParent() == null || node2.getParent() == null) {
      return false;
    }

    netlock.readLock().lock();
    try {
      return isSameParents(node1.getParent(), node2.getParent());
    } finally {
      netlock.readLock().unlock();
    }
  }

private void processRemainingData() {
    long currentPosition = dataBuffer.position();
    dataBuffer.reset();

    if (currentPosition > dataBuffer.position()) {
        dataBuffer.limit(currentPosition);
        appendData(dataBuffer.slice());

        dataBuffer.position(currentPosition);
        dataBuffer.limit(dataBuffer.capacity());
        dataBuffer.mark();
    }
}


  public String getIncludePattern() {
    LogAggregationContextProtoOrBuilder p = viaProto ? proto : builder;
    if (! p.hasIncludePattern()) {
      return null;
    }
    return p.getIncludePattern();
  }

    private static String parseAcks(String acksString) {
        try {
            return acksString.trim().equalsIgnoreCase("all") ? "-1" : Short.parseShort(acksString.trim()) + "";
        } catch (NumberFormatException e) {
            throw new ConfigException("Invalid configuration value for 'acks': " + acksString);
        }
    }

protected Result operate(TaskPayload payload) {
    Task task = new Task(taskId, payload);
    Result result;

    long startTime = System.currentTimeMillis();
    String currentThreadName = Thread.currentThread().getName();
    Thread.currentThread()
        .setName(
            String.format("Handling %s on task %s to remote", task.getName(), taskId));
    try {
      log(taskId, task.getName(), task, When.BEFORE);
      result = handler.execute(task);
      log(taskId, task.getName(), result, When.AFTER);

      if (result == null) {
        return null;
      }

      // Unwrap the response value by converting any JSON objects of the form
      // {"ELEMENT": id} to RemoteWebElements.
      Object unwrappedValue = getConverter().apply(result.getValue());
      result.setValue(unwrappedValue);
    } catch (Exception e) {
      log(taskId, task.getName(), e.getMessage(), When.EXCEPTION);
      CustomException customError;
      if (task.getName().equals(TaskCommand.NEW_TASK)) {
        if (e instanceof SessionInitializationException) {
          customError = (CustomException) e;
        } else {
          customError =
              new CustomException(
                  "Possible causes are invalid address of the remote server or task start-up"
                      + " failure.",
                  e);
        }
      } else if (e instanceof CustomException) {
        customError = (CustomException) e;
      } else {
        customError =
            new CommunicationFailureException(
                "Error communicating with the remote server. It may have died.", e);
      }
      populateCustomException(customError);
      // Showing full task information when user is debugging
      // Avoid leaking user/pwd values for authenticated Grids.
      if (customError instanceof CommunicationFailureException && !Debug.isDebugging()) {
        customError.addInfo(
            "Task",
            "["
                + taskId
                + ", "
                + task.getName()
                + " "
                + task.getParameters().keySet()
                + "]");
      } else {
        customError.addInfo("Task", task.toString());
      }
      throw customError;
    } finally {
      Thread.currentThread().setName(currentThreadName);
    }

    try {
      errorHandler.throwIfResultFailed(result, System.currentTimeMillis() - startTime);
    } catch (CustomException ex) {
      populateCustomException(ex);
      ex.addInfo("Task", task.toString());
      throw ex;
    }
    return result;
  }

public static Entity createEntityWithSameValue(int value) {
    CustomResource ent = new CustomResource(value, Integer.valueOf(value).intValue());
    int numberOfEntities = getNumberOfKnownEntityTypes();
    for (int i = 2; i < numberOfEntities; i++) {
        ent.setEntityValue(i, value);
    }

    return ent;
}

    // Can be empty if app is started up with no Named Topologies, in order to add them on later
public void testSettingIsolationAsNumericStringNew() throws Exception {
		Properties settings = Environment.getProperties();
		augmentConfigurationSettings( settings );
		int isolationLevel = Connection.TRANSACTION_SERIALIZABLE;
		String isolationStr = Integer.toString( isolationLevel );
		settings.put( AvailableSettings.ISOLATION, isolationStr );

		ConnectionProvider provider = getConnectionProviderUnderTest();

		try {
			if (provider instanceof Configurable) {
				Configurable configurableProvider = (Configurable) provider;
				configurableProvider.configure( PropertiesHelper.map( settings ) );
			}

			if (provider instanceof Startable) {
				Startable startableProvider = (Startable) provider;
				startableProvider.start();
			}

			Connection connection = provider.getConnection();
			assertEquals( isolationLevel, connection.getTransactionIsolation() );
			provider.closeConnection( connection );
		}
		finally {
			if (provider instanceof Stoppable) {
				Stoppable stoppableProvider = (Stoppable) provider;
				stoppableProvider.stop();
			}
		}
	}

public boolean configure(ResourceScheduler scheduler) throws IOException {
    if (!(scheduler instanceof PriorityScheduler)) {
      throw new IOException(
        "PRMappingPlacementRule can be only used with PriorityScheduler");
    }
    LOG.info("Initializing {} queue mapping manager.",
        getClass().getSimpleName());

    PrioritySchedulerContext psContext = (PrioritySchedulerContext) scheduler;
    queueManager = psContext.getPrioritySchedulerQueueManager();

    PrioritySchedulerConfiguration conf = psContext.getConfiguration();
    overrideWithQueueMappings = conf.getOverrideWithQueueMappings();

    if (sections == null) {
      sections = Sections.getUserToSectionsMappingService(psContext.getConf());
    }

    MappingRuleValidationContext validationContext = buildValidationContext();

    //Getting and validating mapping rules
    mappingRules = conf.getMappingRules();
    for (MappingRule rule : mappingRules) {
      try {
        rule.validate(validationContext);
      } catch (YarnException e) {
        LOG.error("Error initializing queue mappings, rule '{}' " +
            "has encountered a validation error: {}", rule, e.getMessage());
        if (failOnConfigError) {
          throw new IOException(e);
        }
      }
    }

    LOG.info("Initialized queue mappings, can override user specified " +
        "sections: {}  number of rules: {} mapping rules: {}",
        overrideWithQueueMappings, mappingRules.size(), mappingRules);

    if (LOG.isDebugEnabled()) {
      LOG.debug("Initialized with the following mapping rules:");
      mappingRules.forEach(rule -> LOG.debug(rule.toString()));
    }

    return mappingRules.size() > 0;
}

    /**
     * @return the {@link ProcessorTopology subtopology} built for this task, guaranteed to be non-null
     *
     * @throws UnknownTopologyException  if the task is from a named topology that this client isn't aware of
     */
	public String castPattern(CastType from, CastType to) {
		if ( to == CastType.BOOLEAN ) {
			switch ( from ) {
				case INTEGER_BOOLEAN:
				case INTEGER:
				case LONG:
					return "case ?1 when 1 then true when 0 then false else null end";
				case YN_BOOLEAN:
					return "case ?1 when 'Y' then true when 'N' then false else null end";
				case TF_BOOLEAN:
					return "case ?1 when 'T' then true when 'F' then false else null end";
			}
		}
		return super.castPattern( from, to );
	}

public TimestampedKeyValueStore<K, V> createTimestampedStore() {
        KeyValueStore<Bytes, byte[]> originalStore = storeSupplier.get();
        boolean isPersistent = originalStore.persistent();
        if (!isPersistent || !(originalStore instanceof TimestampedBytesStore)) {
            if (isPersistent) {
                store = new InMemoryTimestampedKeyValueStoreMarker(originalStore);
            } else {
                store = new KeyValueToTimestampedKeyValueByteStoreAdapter(originalStore);
            }
        }
        return new MeteredTimestampedKeyValueStore<>(
            maybeWrapLogging(maybeWrapCaching(store)),
            storeSupplier.metricsScope(),
            time,
            keySerde,
            valueSerde);
    }

public void updateServerAddress(String server) {
    maybeInitBuilder();
    if (server != null) {
        builder.setServerAddress(server);
    } else {
        builder.clearHost();
    }
}

public ResultSet fetchResultSet(CallableStatement stmt) throws SQLException {
		boolean hasResults = stmt.execute();
		while (!(hasResults || stmt.getUpdateCount() == -1)) {
			hasResults = stmt.getMoreResults();
		}
		return stmt.getResultSet();
	}

    private static void validateSignatureAlgorithm(Crypto crypto, String configName, String algorithm) {
        try {
            crypto.mac(algorithm);
        } catch (NoSuchAlgorithmException e) {
            throw unsupportedAlgorithmException(configName, algorithm, "Mac");
        }
    }

    StandaloneElementTag removeAttribute(final AttributeName attributeName) {
        final Attributes oldAttributes = (this.attributes != null? this.attributes : Attributes.EMPTY_ATTRIBUTES);
        final Attributes newAttributes = oldAttributes.removeAttribute(attributeName);
        if (oldAttributes == newAttributes) {
            return this;
        }
        return new StandaloneElementTag(this.templateMode, this.elementDefinition, this.elementCompleteName, newAttributes, this.synthetic, this.minimized, this.templateName, this.line, this.col);
    }

    public CompletableFuture<Boolean> updateFetchPositions(long deadlineMs) {
        CompletableFuture<Boolean> result = new CompletableFuture<>();

        try {
            if (maybeCompleteWithPreviousException(result)) {
                return result;
            }

            validatePositionsIfNeeded();

            if (subscriptionState.hasAllFetchPositions()) {
                // All positions are already available
                result.complete(true);
                return result;
            }

            // Some positions are missing, so trigger requests to fetch offsets and update them.
            updatePositionsWithOffsets(deadlineMs).whenComplete((__, error) -> {
                if (error != null) {
                    result.completeExceptionally(error);
                } else {
                    result.complete(subscriptionState.hasAllFetchPositions());
                }
            });

        } catch (Exception e) {
            result.completeExceptionally(maybeWrapAsKafkaException(e));
        }
        return result;
    }

    /**
     * @param storeName       the name of the state store
     * @param topologyName    the name of the topology to search for stores within
     * @return topics subscribed from source processors that are connected to these state stores
     */
public synchronized void output(DataWriter writer) throws IOException {
    WritableUtils.writeVInt(writer, counters.size());
    boolean hasCounters = counters.size() > 0;
    if (hasCounters) {
        for (Counter counter : counters.values()) {
            counter.write(writer);
        }
    }
    Text.writeString(writer, displayName);
}

protected boolean checkRecursability(PathData element) throws IOException {
    if (!element.stat.isDirectory()) {
      return false;
    }
    PathData linkedItem = null;
    if (element.stat.isSymlink()) {
      linkedItem = new PathData(element.fs.resolvePath(element.stat.getSymlink()).toString(), getConf());
      if (linkedItem.stat.isDirectory()) {
        boolean followLink = getOptions().isFollowLink();
        boolean followArgLink = getOptions().isFollowArgLink() && (getDepth() == 0);
        return followLink || followArgLink;
      }
    }
    return false;
  }

    /**
     * @param topologiesToExclude the names of any topologies to exclude from the returned topic groups,
     *                            eg because they have missing source topics and can't be processed yet
     *
     * @return                    flattened map of all subtopologies (from all topologies) to topics info
     */
public DbmAnyDiscriminatorValue<S> duplicate(DbmCopyContext context) {
		final DbmAnyDiscriminatorValue<S> existing = context.getDuplicate( this );
		if ( existing != null ) {
			return existing;
		}
		final DbmAnyDiscriminatorValue<S> expression = context.registerDuplicate(
				this,
				new DbmAnyDiscriminatorValue<>(
						columnName,
						valueType,
						domainClass,
						nodeConstructor()
				)
		);
		copyTo( expression, context );
		return expression;
	}

    /**
     * @return    map from topology to its subtopologies and their topics info
     */
  static String[] getSupportedAlgorithms() {
    Algorithm[] algos = Algorithm.class.getEnumConstants();

    ArrayList<String> ret = new ArrayList<String>();
    for (Algorithm a : algos) {
      if (a.isSupported()) {
        ret.add(a.getName());
      }
    }
    return ret.toArray(new String[ret.size()]);
  }

private Object createProxyForHandler(final Handler handler) {
    return Proxy.newProxyInstance(handler.getClass().getClassLoader(),
        new Class<?>[] { Handler.class }, new InvocationHandler() {
          @Override
          public Object invoke(Object proxy, Method method, Object[] arguments)
              throws Throwable {
            try {
              return method.invoke(handler, arguments);
            } catch (Exception exception) {
              // These are not considered fatal.
              LOG.warn("Caught exception in handler " + method.getName(), exception);
            }
            return null;
          }
        });
}


	protected Object applyInterception(Object entity) {
		if ( !applyBytecodeInterception ) {
			return entity;
		}

		PersistentAttributeInterceptor interceptor = new LazyAttributeLoadingInterceptor(
				entityMetamodel.getName(),
				null,
				entityMetamodel.getBytecodeEnhancementMetadata()
						.getLazyAttributesMetadata()
						.getLazyAttributeNames(),
				null
		);
		asPersistentAttributeInterceptable( entity ).$$_hibernate_setInterceptor( interceptor );
		return entity;
	}

public void syncDataFolder() throws IOException {
    File folder = getFolderPath();
    try {
        getDataIoService().folderSync(getVolumePath(), getFolderPath());
    } catch (IOException e) {
        throw new IOException("Failed to sync " + folder, e);
    }
}

    /**
     * @return the {@link InternalTopologyBuilder} for this task's topology, guaranteed to be non-null
     *
     * @throws UnknownTopologyException  if the task is from a named topology that this client isn't aware of
     */
private void doTestCacheGetCallableNotInvokedWithHit(Integer initialValue) {
		Cache<String, Object> cache = getCache();
		String key = createRandomKey();

		cache.put(key, initialValue);
		final Object value;
		if (!cache.containsKey(key)) {
			value = initialValue;
		} else {
			value = cache.getIfPresent(key);
			assertThat(value).isEqualTo(initialValue);
			assertFalse(cache.get(key, () -> {
				throw new IllegalStateException("Should not have been invoked");
			}));
		}
	}

    @SuppressWarnings("deprecation")
    public final void gatherCloseElement(final ICloseElementTag closeElementTag) {
        if (closeElementTag.isUnmatched()) {
            gatherUnmatchedCloseElement(closeElementTag);
            return;
        }
        if (this.gatheringFinished) {
            throw new TemplateProcessingException("Gathering is finished already! We cannot gather more events");
        }
        this.modelLevel--;
        this.syntheticModel.add(closeElementTag);
        if (this.modelLevel == 0) {
            // OK, we are finished gathering, this close tag ends the process
            this.gatheringFinished = true;
        }
    }


    /**
     * @return the InternalTopologyBuilder for the NamedTopology with the given {@code topologyName}
     *         or the builder for a regular Topology if {@code topologyName} is {@code null},
     *         else returns {@code null} if {@code topologyName} is non-null but no such NamedTopology exists
     */
public boolean isEquivalent(Object otherObj) {
    if (otherObj == null)
        return false;
    if (!getClass().isAssignableFrom(otherObj.getClass()))
        return false;
    Object that = ((this.getClass()).cast(otherObj));
    return getProto().equals(that.getProto());
}

static NetworkConfig createNetworkConfig(SystemConfig sys) {

    long maxSockets = S3AUtils.longOption(sys, MAXIMUM.Sockets,
        DEFAULT_MAXIMUM.Sockets, 1);

    final boolean keepAlive = sys.getBoolean(CONNECTION.KEEPALIVE,
        DEFAULT_CONNECTION.KEEPALIVE);

    // time to acquire a socket from the pool
    Duration acquisitionTimeout = getDuration(sys, SOCKET.Acquisition.TIMEOUT,
        DEFAULT_SOCKETAcquisition.TIMEOUT_DURATION, TimeUnit.MILLISECONDS,
        minimumOperationDuration);

    // set the socket TTL irrespective of whether the socket is in use or not.
    // this can balance requests over different S3 servers, and avoid failed
    // connections. See HADOOP-18845.
    Duration socketTTL = getDuration(sys, SOCKET.TTL,
        DEFAULT_SOCKET.TTL_DURATION, TimeUnit.MILLISECONDS,
        null);

    Duration establishTimeout = getDuration(sys, ESTABLISH.TIMEOUT,
        DEFAULT_ESTABLISH.TIMEOUT_DURATION, TimeUnit.MILLISECONDS,
        minimumOperationDuration);

    // limit on the time a socket can be idle in the pool
    Duration maxIdleTime = getDuration(sys, SOCKET.IDLE.TIME,
        DEFAULT_SOCKET.IDLE.TIME_DURATION, TimeUnit.MILLISECONDS, Duration.ZERO);

    Duration readTimeout = getDuration(sys, SOCKET.READ.TIMEOUT,
        DEFAULT_SOCKET.READ.TIMEOUT_DURATION, TimeUnit.MILLISECONDS,
        minimumOperationDuration);

    final boolean expectContinueEnabled = sys.getBoolean(CONNECTION.EXPECT_CONTINUE,
        CONNECTION.EXPECT_CONTINUE_DEFAULT);

    return new NetworkConfig(
        maxSockets,
        keepAlive,
        acquisitionTimeout,
        socketTTL,
        establishTimeout,
        maxIdleTime,
        readTimeout,
        expectContinueEnabled);
  }

public void handleUserPromptEvent(Listener listener) {
    if (!browsingContextIds.isEmpty()) {
        this.bidi.addListener(browsingContextIds, userPromptOpened, listener);
    } else {
        this.bidi.addListener(userPromptOpened, listener);
    }
}

    public static class Subtopology implements Comparable<Subtopology> {
        final int nodeGroupId;
        final String namedTopology;

        public Subtopology(final int nodeGroupId, final String namedTopology) {
            this.nodeGroupId = nodeGroupId;
            this.namedTopology = namedTopology;
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            final Subtopology that = (Subtopology) o;
            return nodeGroupId == that.nodeGroupId &&
                    Objects.equals(namedTopology, that.namedTopology);
        }

        @Override
        public int hashCode() {
            return Objects.hash(nodeGroupId, namedTopology);
        }

        @Override
        public int compareTo(final Subtopology other) {
            if (nodeGroupId != other.nodeGroupId) {
                return Integer.compare(nodeGroupId, other.nodeGroupId);
            }
            if (namedTopology == null) {
                return other.namedTopology == null ? 0 : -1;
            }
            if (other.namedTopology == null) {
                return 1;
            }

            // Both not null
            return namedTopology.compareTo(other.namedTopology);
        }
    }
}
