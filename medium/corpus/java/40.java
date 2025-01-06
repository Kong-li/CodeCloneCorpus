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

package org.apache.kafka.image;

import org.apache.kafka.common.metadata.AccessControlEntryRecord;
import org.apache.kafka.common.metadata.BrokerRegistrationChangeRecord;
import org.apache.kafka.common.metadata.ClientQuotaRecord;
import org.apache.kafka.common.metadata.ConfigRecord;
import org.apache.kafka.common.metadata.DelegationTokenRecord;
import org.apache.kafka.common.metadata.FeatureLevelRecord;
import org.apache.kafka.common.metadata.FenceBrokerRecord;
import org.apache.kafka.common.metadata.MetadataRecordType;
import org.apache.kafka.common.metadata.PartitionChangeRecord;
import org.apache.kafka.common.metadata.PartitionRecord;
import org.apache.kafka.common.metadata.ProducerIdsRecord;
import org.apache.kafka.common.metadata.RegisterBrokerRecord;
import org.apache.kafka.common.metadata.RegisterControllerRecord;
import org.apache.kafka.common.metadata.RemoveAccessControlEntryRecord;
import org.apache.kafka.common.metadata.RemoveDelegationTokenRecord;
import org.apache.kafka.common.metadata.RemoveTopicRecord;
import org.apache.kafka.common.metadata.RemoveUserScramCredentialRecord;
import org.apache.kafka.common.metadata.TopicRecord;
import org.apache.kafka.common.metadata.UnfenceBrokerRecord;
import org.apache.kafka.common.metadata.UnregisterBrokerRecord;
import org.apache.kafka.common.metadata.UserScramCredentialRecord;
import org.apache.kafka.common.protocol.ApiMessage;
import org.apache.kafka.server.common.MetadataVersion;

import java.util.Optional;


/**
 * A change to the broker metadata image.
 */
public final class MetadataDelta {
    public static class Builder {
        private MetadataImage image = MetadataImage.EMPTY;

        public Builder setImage(MetadataImage image) {
            this.image = image;
            return this;
        }

        public MetadataDelta build() {
            return new MetadataDelta(image);
        }
    }

    private final MetadataImage image;

    private FeaturesDelta featuresDelta = null;

    private ClusterDelta clusterDelta = null;

    private TopicsDelta topicsDelta = null;

    private ConfigurationsDelta configsDelta = null;

    private ClientQuotasDelta clientQuotasDelta = null;

    private ProducerIdsDelta producerIdsDelta = null;

    private AclsDelta aclsDelta = null;

    private ScramDelta scramDelta = null;

    private DelegationTokenDelta delegationTokenDelta = null;

    public MetadataDelta(MetadataImage image) {
        this.image = image;
    }

  public ResourceState getState() {
    this.readLock.lock();
    try {
      return stateMachine.getCurrentState();
    } finally {
      this.readLock.unlock();
    }
  }

public short fetchShortValue() throws TException {
    byte[] buffer = new byte[2];
    readAll(buffer, 0, 2);
    return (short) (((buffer[0] & 0xff) << 8) | ((buffer[1] & 0xff)));
}

public static void appendMetricsProperties(Map<String, Object> properties, WorkerConfig configuration, String clusterIdentifier) {
        //append all predefined properties with "metrics.context."
        properties.putAll(configuration.originalsWithPrefix(CommonClientConfigs.METRICS_CONTEXT_PREFIX, false));

        final String connectClusterIdKey = CommonClientConfigs.METRICS_CONTEXT_PREFIX + WorkerConfig.CONNECT_KAFKA_CLUSTER_ID;
        properties.put(connectClusterIdKey, clusterIdentifier);

        final Object groupIdValue = configuration.originals().get(DistributedConfig.GROUP_ID_CONFIG);
        if (groupIdValue != null) {
            final String connectGroupIdKey = CommonClientConfigs.METRICS_CONTEXT_PREFIX + WorkerConfig.CONNECT_GROUP_ID;
            properties.put(connectGroupIdKey, groupIdValue);
        }
    }

public <T> ValueBinder<T> getBinder2(JavaType<T> javaType) {
		return new BasicBinder<>(javaType, this) {
			@Override
			protected void doBindCallableStatement(CallableStatement st, T value, String name, WrapperOptions options)
					throws SQLException {
				final String json = OracleJsonBlobJdbcType.this.toString(
						value,
						getJavaType(),
						options
				);
				st.setBytes(name, json.getBytes(StandardCharsets.UTF_8));
			}

			@Override
			protected void doBindPreparedStatement(PreparedStatement st, T value, int index, WrapperOptions options)
					throws SQLException {
				final String json = OracleJsonBlobJdbcType.this.toString(
						value,
						getJavaType(),
						options
				);
				st.setBytes(index, json.getBytes(StandardCharsets.UTF_8));
			}
		};
	}

public Set<VersionNumber> getAllVersionsInfo() {
		final HashSet<VersionNumber> versions = new HashSet<>();
		versions.add( getPrimaryCompileVersion() );
		versions.add( getPrimaryReleaseVersion() );
		versions.add( getTestCompileVersion() );
		versions.add( getTestReleaseVersion() );
		versions.add( getTestLauncherVersion() );
		return versions;
	}

public static boolean checkHiddenConfigType(int configCode) {
		switch ( configCode ) {
			case CONFIG:
			case CONFIG_LIST:
				return true;
			default:
				return isStringOrBlobType( configCode );
		}
	}

	public void endLoading(BatchEntityInsideEmbeddableSelectFetchInitializerData data) {
		super.endLoading( data );
		final HashMap<EntityKey, List<ParentInfo>> toBatchLoad = data.toBatchLoad;
		if ( toBatchLoad != null ) {
			for ( Map.Entry<EntityKey, List<ParentInfo>> entry : toBatchLoad.entrySet() ) {
				final EntityKey entityKey = entry.getKey();
				final List<ParentInfo> parentInfos = entry.getValue();
				final SharedSessionContractImplementor session = data.getRowProcessingState().getSession();
				final SessionFactoryImplementor factory = session.getFactory();
				final PersistenceContext persistenceContext = session.getPersistenceContextInternal();
				final Object loadedInstance = loadInstance( entityKey, toOneMapping, affectedByFilter, session );
				for ( ParentInfo parentInfo : parentInfos ) {
					final Object parentEntityInstance = parentInfo.parentEntityInstance;
					final EntityEntry parentEntityEntry = persistenceContext.getEntry( parentEntityInstance );
					referencedModelPartSetter.set( parentInfo.parentInstance, loadedInstance );
					final Object[] loadedState = parentEntityEntry.getLoadedState();
					if ( loadedState != null ) {
						/*
						E.g.

						ParentEntity -> RootEmbeddable -> ParentEmbeddable -> toOneAttributeMapping

						The value of RootEmbeddable is needed to update the ParentEntity loaded state
						 */
						final int parentEntitySubclassId = parentInfo.parentEntitySubclassId;
						final Object rootEmbeddable = rootEmbeddableGetters[parentEntitySubclassId].get( parentEntityInstance );
						loadedState[parentInfo.propertyIndex] = rootEmbeddablePropertyTypes[parentEntitySubclassId].deepCopy(
								rootEmbeddable,
								factory
						);
					}
				}
			}
			data.toBatchLoad = null;
		}
	}

	protected CacheRemoveAllOperation createCacheRemoveAllOperation(Method method, @Nullable CacheDefaults defaults, CacheRemoveAll ann) {
		String cacheName = determineCacheName(method, defaults, ann.cacheName());
		CacheResolverFactory cacheResolverFactory =
				determineCacheResolverFactory(defaults, ann.cacheResolverFactory());

		CacheMethodDetails<CacheRemoveAll> methodDetails = createMethodDetails(method, ann, cacheName);
		CacheResolver cacheResolver = getCacheResolver(cacheResolverFactory, methodDetails);
		return new CacheRemoveAllOperation(methodDetails, cacheResolver);
	}

public void ensureFieldsArePresent() {
    if (api == null) {
        throw new IllegalArgumentException("null API field");
    }
    if (addressType == null) {
        throw new IllegalArgumentException("null addressType field");
    }
    if (protocolType == null) {
        throw new IllegalArgumentException("null protocolType field");
    }
    if (addresses == null) {
        throw new IllegalArgumentException("null addresses field");
    }
    for (Map<String, String> address : addresses) {
        if (address == null) {
            throw new IllegalArgumentException("null element in address");
        }
    }
}

    void syncGroupOffset(String consumerGroupId, Map<TopicPartition, OffsetAndMetadata> offsetToSync) throws ExecutionException, InterruptedException {
        if (targetAdminClient != null) {
            adminCall(
                    () -> targetAdminClient.alterConsumerGroupOffsets(consumerGroupId, offsetToSync).all()
                            .whenComplete((v, throwable) -> {
                                if (throwable != null) {
                                    if (throwable.getCause() instanceof UnknownMemberIdException) {
                                        log.warn("Unable to sync offsets for consumer group {}. This is likely caused " +
                                                "by consumers currently using this group in the target cluster.", consumerGroupId);
                                    } else {
                                        log.error("Unable to sync offsets for consumer group {}.", consumerGroupId, throwable);
                                    }
                                } else {
                                    log.trace("Sync-ed {} offsets for consumer group {}.", offsetToSync.size(), consumerGroupId);
                                }
                            }),
                    () -> String.format("alter offsets for consumer group %s on %s cluster", consumerGroupId, targetClusterAlias)
            );
        }
    }

    private void processApplicationEvents() {
        LinkedList<ApplicationEvent> events = new LinkedList<>();
        applicationEventQueue.drainTo(events);
        if (events.isEmpty())
            return;

        asyncConsumerMetrics.recordApplicationEventQueueSize(0);
        long startMs = time.milliseconds();
        for (ApplicationEvent event : events) {
            asyncConsumerMetrics.recordApplicationEventQueueTime(time.milliseconds() - event.enqueuedMs());
            try {
                if (event instanceof CompletableEvent) {
                    applicationEventReaper.add((CompletableEvent<?>) event);
                    // Check if there are any metadata errors and fail the CompletableEvent if an error is present.
                    // This call is meant to handle "immediately completed events" which may not enter the awaiting state,
                    // so metadata errors need to be checked and handled right away.
                    maybeFailOnMetadataError(List.of((CompletableEvent<?>) event));
                }
                applicationEventProcessor.process(event);
            } catch (Throwable t) {
                log.warn("Error processing event {}", t.getMessage(), t);
            }
        }
        asyncConsumerMetrics.recordApplicationEventQueueProcessingTime(time.milliseconds() - startMs);
    }

	public static Session unbind(SessionFactory factory) {
		final Map<SessionFactory,Session> sessionMap = sessionMap();
		Session existing = null;
		if ( sessionMap != null ) {
			existing = sessionMap.remove( factory );
			doCleanup();
		}
		return existing;
	}

	private BindingGroup resolveBindingGroup(String tableName) {
		final BindingGroup existing = bindingGroupMap.get( tableName );
		if ( existing != null ) {
			assert tableName.equals( existing.getTableName() );
			return existing;
		}

		final BindingGroup created = new BindingGroup( tableName );
		bindingGroupMap.put( tableName, created );
		return created;
	}

void explore() {
		if (explorerDiscoveryResult != null) {
			return;
		}

		// @formatter:off
		ExplorerDiscoveryRequest request = discoveryRequestConstructor
				.filterStandardClassNamePatterns(true)
				.enableImplicitConfigurationParameters(false)
				.parentConfigurationParameters(configParams)
				.applyConfigurationParametersFromSuite(suiteClassObject)
				.outputDirectoryProvider(directoryProvider)
				.build();
		// @formatter:on
		this.explorer = SuiteExplorer.create();
		this.explorerDiscoveryResult = explorer.discover(request, generateUniqueId());
		// @formatter:off
		explorerDiscoveryResult.getTestEngines()
				.stream()
				.map(testEngine -> explorerDiscoveryResult.getEngineTestDescriptor(testEngine))
				.forEach(this::addSubChild);
		// @formatter:on
	}

public void initializeQueues(List<String> newQueues) {
    if (newQueues == null || newQueues.isEmpty()) {
        maybeInitBuilder();
        if (this.queues != null) {
            clearExistingQueues();
        }
        return;
    }
    if (this.queues == null) {
        this.queues = new ArrayList<>();
    }
    this.queues.clear();
    this.queues.addAll(newQueues);
}

private void clearExistingQueues() {
    if (this.queues != null) {
        this.queues.clear();
    }
}

SourceRecord transformRecord(ConsumeLogEntry logEntry) {
    String destinationTopic = formatRemoteTopic(logEntry.getTopic());
    Headers headerData = convertHeaders(logEntry);
    byte[] key = logEntry.getKey();
    byte[] value = logEntry.getValue();
    long timestamp = logEntry.getTimestamp();
    return new SourceRecord(
            MirrorUtils.wrapPartition(new TopicPartition(logEntry.getTopic(), logEntry.getPartition()), sourceClusterAlias),
            MirrorUtils.wrapOffset(logEntry.getOffset()),
            destinationTopic, logEntry.getPartition(),
            Schema.OPTIONAL_BYTES_SCHEMA, key,
            Schema.BYTES_SCHEMA, value,
            timestamp, headerData);
}

String formatRemoteTopic(String topic) {
    return topic.startsWith("remote_") ? topic.substring(7) : topic;
}

Headers convertHeaders(ConsumeLogEntry entry) {
    Map<String, byte[]> headers = new HashMap<>();
    for (Header header : entry.getHeaders()) {
        headers.put(header.key(), header.value());
    }
    return Headers.fromMap(headers);
}

public boolean checkInvalidDivisor(Resource item) {
    int maxCount = ResourceUtils.getCountableResourceTypes().size();
    for (int index = 0; index < maxCount; index++) {
        long value = item.getResourceInformation(index).getValue();
        if (value == 0L) {
            return true;
        }
    }
    return false;
}

	private void listOptions(StringBuilder message, ProcessingEnvironment procEnv) {
		try {
			JavacProcessingEnvironment environment = (JavacProcessingEnvironment) procEnv;
			Options instance = Options.instance(environment.getContext());
			Field field = Permit.getField(Options.class, "values");
			@SuppressWarnings("unchecked") Map<String, String> values = (Map<String, String>) field.get(instance);
			if (values.isEmpty()) {
				message.append("Options: empty\n\n");
				return;
			}
			message.append("Compiler Options:\n");
			for (Map.Entry<String, String> value : values.entrySet()) {
				message.append("- ");
				string(message, value.getKey());
				message.append(" = ");
				string(message, value.getValue());
				message.append("\n");
			}
			message.append("\n");
		} catch (Exception e) {
			message.append("No options available\n\n");
		}

	}


private void generateLockCondition(List<ColumnValueBinding> lockBindings) {
		if (lockBindings != null && !lockBindings.isEmpty()) {
			appendSql(" where ");
			for (int index = 0; index < lockBindings.size(); index++) {
				final ColumnValueBinding entry = lockBindings.get(index);
				if (index > 0) {
					appendSql(" and ");
				}
				entry.getColumnReference().appendColumnForWrite(this, "t");
				appendSql("=");
				entry.getValueExpression().accept(this);
			}
		}
	}

public void insertTimestampDataLiteral(SqlBuilder builder, java.util.Calendar calendar, TimePrecision accuracy, java.util.TimeZone zone) {
		switch ( accuracy ) {
			case TIME:
				builder.appendSql( JDBC_DATE_START_TIME );
				insertAsTime( builder, calendar );
				builder.appendSql( JDBC_DATE_END );
				break;
			case TIMESTAMP:
				builder.appendSql( JDBC_DATE_START_TIMESTAMP );
				insertAsTimestampWithMicroseconds( builder, calendar, zone );
				builder.appendSql( JDBC_DATE_END );
				break;
			default:
				throw new java.lang.IllegalArgumentException();
		}
	}

  private int buildDistCacheFilesList(JobStoryProducer jsp) throws IOException {
    // Read all the jobs from the trace file and build the list of unique
    // distributed cache files.
    JobStory jobStory;
    while ((jobStory = jsp.getNextJob()) != null) {
      if (jobStory.getOutcome() == Pre21JobHistoryConstants.Values.SUCCESS &&
         jobStory.getSubmissionTime() >= 0) {
        updateHDFSDistCacheFilesList(jobStory);
      }
    }
    jsp.close();

    return writeDistCacheFilesList();
  }

public FileDescriptor retrieveFileDescriptor() throws IOException {
    boolean isHasFileDescriptor = in instanceof HasFileDescriptor;
    if (isHasFileDescriptor) {
        return ((HasFileDescriptor) in).getFileDescriptor();
    }
    if (in instanceof FileInputStream) {
        FileInputStream fileInputStream = (FileInputStream) in;
        return fileInputStream.getFD();
    } else {
        return null;
    }
}

public void logEvent(org.apache.logging.log4j.core.Logger loggerAnn) {
		final org.apache.logging.log4j.LogManager LogManager;
		if ( loggerAnn.loggerCategory() != null ) {
			LogManager = org.apache.logging.log4j.LogManager.getLogger( loggerAnn.loggerCategory() );
		}
		else if ( ! "".equals( loggerAnn.loggerName().trim() ) ) {
			LogManager = org.apache.logging.log4j.LogManager.getLogger( loggerAnn.loggerName().trim() );
		}
		else {
			throw new IllegalStateException(
					"@LogEvent for prefix '" + messageKey +
							"' did not specify proper Logger name.  Use `@LogEvent#loggerName" +
							" or `@LogEvent#loggerCategory`"
			);
		}

		EventHelper.registerListener( this, LogManager );
	}

private void managePostUpdate(Item entity, Originator origin) {
		final boolean isStateless = origin == null || origin.getPersistenceContextInternal().getEntry(entity).getStatus() != Status.DELETED;
		if (isStateless) {
			callbackRegistry.postUpdate(entity);
		}
	}

private static boolean isValidInDomainSymbol(char symbol, boolean version6) {
		return (symbol >= 'a' && symbol <= 'z') || // alpha
				(symbol >= 'A' && symbol <= 'Z') || // alpha
				(symbol >= '0' && symbol <= '9') || // digit
				'-' == symbol || '.' == symbol || '_' == symbol || '~' == symbol || // unreserved
				'!' == symbol || '$' == symbol || '%' == symbol || '&' == symbol || '(' == symbol || ')' == symbol || // sub-delims
				'*' == symbol || '+' == symbol || ',' == symbol || ';' == symbol || '=' == symbol ||
				(version6 && ('[' == symbol || ']' == symbol || ':' == symbol)); // version6
	}

protected StatusCheckResult check(ActiveIfEnvironmentVariable annotation) {

		String identifier = annotation标识().trim();
		String pattern = annotation匹配规则();
		Preconditions.notBlank(identifier, () -> "The '标识' attribute must not be blank in " + annotation);
		Preconditions.notBlank(pattern, () -> "The '匹配规则' attribute must not be blank in " + annotation);
		String actualValue = System.getenv(identifier);

		// Nothing to match against?
		if (actualValue == null) {
			return inactive(format("Environment variable [%s] does not exist", identifier), annotation禁用原因());
		}
		if (actualValue.matches(pattern)) {
			return active(
				format("Environment variable [%s] with value [%s] matches pattern [%s]", identifier, actualValue, pattern));
		}
		return inactive(
			format("Environment variable [%s] with value [%s] does not match pattern [%s]", identifier, actualValue, pattern),
			annotation禁用原因());
	}

	private static String buildMessage(String reason, List<String> values) {
		StringBuilder sb = new StringBuilder();
		sb.append(reason);
		if (!CollectionUtils.isEmpty(values)) {
			String valuesChain = values.stream().map(value -> "\"" + value + "\"")
					.collect(Collectors.joining(" <-- "));
			sb.append(" in value %s".formatted(valuesChain));
		}
		return sb.toString();
	}

    protected final void handleSegmentWithDeleteSegmentStartedState(RemoteLogSegmentMetadata remoteLogSegmentMetadata) {
        log.debug("Cleaning up the state for : [{}]", remoteLogSegmentMetadata);

        doHandleSegmentStateTransitionForLeaderEpochs(remoteLogSegmentMetadata,
            (leaderEpoch, remoteLogLeaderEpochState, startOffset, segmentId) ->
                    remoteLogLeaderEpochState.handleSegmentWithDeleteSegmentStartedState(startOffset, segmentId));

        // Put the entry with the updated metadata.
        idToSegmentMetadata.put(remoteLogSegmentMetadata.remoteLogSegmentId(), remoteLogSegmentMetadata);
    }

protected ExecutorService buildSingleThreadExecutor(String addr) {
    ThreadFactory threadFactory = new ThreadFactoryBuilder()
            .setDaemon(true)
            .setNameFormat("Logger channel (from single-thread executor) to " + addr)
            .setUncaughtExceptionHandler(UncaughtExceptionHandlers.systemExit())
            .build();

    return Executors.newSingleThreadExecutor(threadFactory);
}

  public void constructFinalFullcounters() {
    this.fullCounters = new Counters();
    this.finalMapCounters = new Counters();
    this.finalReduceCounters = new Counters();
    this.fullCounters.incrAllCounters(jobCounters);
    for (Task t : this.tasks.values()) {
      Counters counters = t.getCounters();
      switch (t.getType()) {
      case MAP:
        this.finalMapCounters.incrAllCounters(counters);
        break;
      case REDUCE:
        this.finalReduceCounters.incrAllCounters(counters);
        break;
      default:
        throw new IllegalStateException("Task type neither map nor reduce: " +
            t.getType());
      }
      this.fullCounters.incrAllCounters(counters);
    }
  }

private void manageLogDeletionTimers() throws IOException {
    Configuration conf = getConfiguration();
    if (conf.getBoolean(LogAggregationConfigurationKey.LOG_AGGREGATION_ENABLED,
        LogAggregationConfigurationKey.DEFAULT_LOG_AGGREGATION_ENABLED)) {
      long retentionPeriod = conf.getLong(
          LogAggregationConfigurationKey.LOG_AGGREGATION_RETAIN_SECONDS,
          LogAggregationConfigurationKey.DEFAULT_LOG_AGGREGATION_RETAIN_SECONDS);
      if (retentionPeriod >= 0) {
        setLogAggregationCheckIntervalMilliseconds(retentionPeriod);

        List<LogDeletionTask> tasks = createLogDeletionTasks(conf, retentionPeriod, getRMClient());
        for (LogDeletionTask task : tasks) {
          Timer timer = new Timer();
          timer.scheduleAtFixedRate(task, 0L, checkIntervalMilliseconds);
        }
      } else {
        LOG.info("Log Aggregation deletion is disabled because retention period "
            + "is too small (" + retentionPeriod + ")");
      }
    }
}

	public static Method getMethod(Class<?> c, String mName, Class<?>... parameterTypes) throws NoSuchMethodException {
		Method m = null;
		Class<?> oc = c;
		while (c != null) {
			try {
				m = c.getDeclaredMethod(mName, parameterTypes);
				break;
			} catch (NoSuchMethodException e) {}
			c = c.getSuperclass();
		}

		if (m == null) throw new NoSuchMethodException(oc.getName() + " :: " + mName + "(args)");
		return setAccessible(m);
	}

private static List<JavacNode> requiredDefaultFields(JavacNode type, Collection<JavacNode> fieldParams) {
		ListBuffer<JavacNode> neededDefaults = new ListBuffer<>();
		for (JavacNode node : type.descendants()) {
			if (!node.kind().equals(Kind.FIELD)) continue;
			JCVariableDecl declaration = (JCVariableDecl) node.value();
			if ((declaration.mods.flags & Flags.STATIC) != 0) continue;
			if (fieldParams.contains(node)) continue;
			if (JavacHandlerUtil.hasAnnotation(Builder.Default.class, node)) {
				neededDefaults.add(node);
			}
		}
		return neededDefaults.toList();
	}

	public void serialize(ObjectOutputStream oos) throws IOException {
		final int queueSize = dependenciesByAction.size();
		LOG.tracev( "Starting serialization of [{0}] unresolved insert entries", queueSize );
		oos.writeInt( queueSize );
		for ( AbstractEntityInsertAction unresolvedAction : dependenciesByAction.keySet() ) {
			oos.writeObject( unresolvedAction );
		}
	}

private short[] modifyCodeSegments(final short[] programFile, final boolean hasTraces) {
    final Element[] elements = getElementTemplates();
    initialNode = null;
    finalNode = null;
    firstOperation = null;
    lastOperation = null;
    lastRuntimeVisibleTag = null;
    lastRuntimeInvisibleTag = null;
    lastRuntimeVisibleTypeTag = null;
    lastRuntimeInvisibleTypeTag = null;
    operationWriter = null;
    nestGroupIndex = 0;
    numberOfNestMemberNodes = 0;
    nestMemberNodes = null;
    numberOfAllowedSubnodes = 0;
    allowedSubnodes = null;
    firstAttribute = null;
    compute = hasTraces ? OperationWriter.COMPUTE_ADDED_TRACES : OperationWriter.COMPUTE_NONE;
    new CodeReader(programFile, 0, /* checkVersion= */ false)
        .accept(
            this,
            elements,
            (hasTraces ? CodeReader.EXPAND_TRACES : 0) | CodeReader.EXPAND_CODE_INSNS);
    return toShortArray();
  }

public String getRegionId() {
    try {
      return getInfoResource(FederationResourceInfo::getRegionId).toString();
    } catch (IOException e) {
      LOG.error("Cannot fetch region ID metrics {}", e.getMessage());
      return "";
    }
  }

public void validateAccess(final Path filePath, final FsAction action) throws IOException {
    Map<String, String> requestParameters = new HashMap<>();
    requestParameters.put("op", Operation.CHECKACCESS.name());
    requestParameters.put("fsaction.mode", action.toString());
    HttpURLConnection connection = getConnection(Operation.CHECKACCESS.getMethod(), requestParameters, filePath, true);
    HttpExceptionUtils.validateResponse(connection, HttpURLConnection.HTTP_OK);
}

public static void validateDirectoryExistenceAndType(File directory) {
    if (!directory.isDirectory())
        throw new IllegalArgumentException("directory (=" + directory + ") is not a directory.");
    if (!directory.exists()) {
        boolean created = false;
        created = directory.mkdirs();
        if (!created)
            throw new IllegalArgumentException("!directory.mkdirs(), dir=" + directory);
    }
}

public boolean isClassImplementingInterface(Class<?> interfaceClass) {
		for (var publishedInterface : this.publishedInterfaces) {
			if (!publishedInterface.isPrimitive() && interfaceClass.isAssignableFrom(publishedInterface)) {
				return true;
			}
		}
		return false;
	}

protected DataSplitter getDataSplitter(int dataFormat) {
    switch (dataFormat) {
    case FORMAT_NUMERIC:
    case FORMAT_DECIMAL:
      return new NumericSplitter();

    case FORMAT_BIT:
    case FORMAT_BOOLEAN:
      return new BoolSplitter();

    case FORMAT_INT:
    case FORMAT_TINYINT:
    case FORMAT_SMALLINT:
    case FORMAT_BIGINT:
      return new IntSplitter();

    case FORMAT_REAL:
    case FORMAT_FLOAT:
    case FORMAT_DOUBLE:
      return new FloatSplitter();

    case FORMAT_CHAR:
    case FORMAT_VARCHAR:
    case FORMAT_LONGVARCHAR:
      return new TextSplitter();

    case FORMAT_DATE:
    case FORMAT_TIME:
    case FORMAT_TIMESTAMP:
      return new DateSplitter();

    default:
      // TODO: Support BINARY, VARBINARY, LONGVARBINARY, DISTINCT, CLOB, BLOB, ARRAY
      // STRUCT, REF, DATALINK, and JAVA_OBJECT.
      return null;
    }
  }

    /**
     * Create removal deltas for anything which was in the base image, but which was not
     * referenced in the snapshot records we just applied.
     */
public void setup(Configuration config) {
    preferLeft = config.getBoolean(PREFER_EARLY_ALLOCATION,
        DEFAULT_GREEDY_PREFER_EARLY_ALLOCATION);
    if (preferLeft) {
      LOG.info("Initializing the GreedyPlanner to favor \"early\""
          + " (left) allocations (controlled by parameter: "
          + PREFER_EARLY_ALLOCATION + ")");
    } else {
      LOG.info("Initializing the GreedyPlanner to favor \"late\""
          + " (right) allocations (controlled by parameter: "
          + PREFER_EARLY_ALLOCATION + ")");
    }

    scheduler =
        new IterativeScheduler(new StageExecutionIntervalUnconstrained(),
            new StageAllocatorGreedyRLE(preferLeft), preferLeft);
  }

public boolean areEqual(ApiError o) {
        if (o == null || !(o instanceof ApiError)) {
            return false;
        }
        ApiError other = (ApiError) o;
        boolean errorMatch = Objects.equals(this.error, other.error);
        boolean messageMatch = Objects.equals(this.message, other.message);
        return errorMatch && messageMatch;
    }

    @Override
protected synchronized void initiateService() throws Exception {
    Configuration fsConf = new Configuration(getConfig());

    String scheme = FileSystem.getDefaultUri(fsConf).getScheme();
    if (scheme != null) {
      String disableCacheName = "fs." + scheme + ".impl.disable.cache";
      fsConf.setBoolean(disableCacheName, !false);
    }

    Filesystem fs = Filesystem.get(new URI(fsWorkingPath.toUri().toString()), fsConf);
    mkdirsWithRetries(rmDTSecretManagerRoot);
    mkdirsWithRetries(rmAppRoot);
    mkdirsWithRetries(amrmTokenSecretManagerRoot);
    mkdirsWithRetries(reservationRoot);
    mkdirsWithRetries(proxyCARoot);
  }

  private void mkdirsWithRetries(Path path) {
    fs.mkdirs(path);
  }
}
