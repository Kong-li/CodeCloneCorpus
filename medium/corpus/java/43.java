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
package org.apache.kafka.connect.runtime;

import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.config.AbstractConfig;
import org.apache.kafka.common.config.Config;
import org.apache.kafka.common.config.ConfigDef;
import org.apache.kafka.common.config.ConfigDef.ConfigKey;
import org.apache.kafka.common.config.ConfigDef.Type;
import org.apache.kafka.common.config.ConfigTransformer;
import org.apache.kafka.common.config.ConfigValue;
import org.apache.kafka.common.utils.Time;
import org.apache.kafka.common.utils.Utils;
import org.apache.kafka.connect.connector.Connector;
import org.apache.kafka.connect.connector.policy.ConnectorClientConfigOverridePolicy;
import org.apache.kafka.connect.connector.policy.ConnectorClientConfigRequest;
import org.apache.kafka.connect.errors.ConnectException;
import org.apache.kafka.connect.errors.NotFoundException;
import org.apache.kafka.connect.runtime.isolation.LoaderSwap;
import org.apache.kafka.connect.runtime.isolation.PluginUtils;
import org.apache.kafka.connect.runtime.isolation.Plugins;
import org.apache.kafka.connect.runtime.isolation.VersionedPluginLoadingException;
import org.apache.kafka.connect.runtime.rest.entities.ActiveTopicsInfo;
import org.apache.kafka.connect.runtime.rest.entities.ConfigInfo;
import org.apache.kafka.connect.runtime.rest.entities.ConfigInfos;
import org.apache.kafka.connect.runtime.rest.entities.ConfigKeyInfo;
import org.apache.kafka.connect.runtime.rest.entities.ConfigValueInfo;
import org.apache.kafka.connect.runtime.rest.entities.ConnectorInfo;
import org.apache.kafka.connect.runtime.rest.entities.ConnectorOffsets;
import org.apache.kafka.connect.runtime.rest.entities.ConnectorStateInfo;
import org.apache.kafka.connect.runtime.rest.entities.ConnectorType;
import org.apache.kafka.connect.runtime.rest.entities.LoggerLevel;
import org.apache.kafka.connect.runtime.rest.entities.Message;
import org.apache.kafka.connect.runtime.rest.errors.BadRequestException;
import org.apache.kafka.connect.sink.SinkConnector;
import org.apache.kafka.connect.source.SourceConnector;
import org.apache.kafka.connect.storage.ClusterConfigState;
import org.apache.kafka.connect.storage.ConfigBackingStore;
import org.apache.kafka.connect.storage.Converter;
import org.apache.kafka.connect.storage.ConverterConfig;
import org.apache.kafka.connect.storage.ConverterType;
import org.apache.kafka.connect.storage.HeaderConverter;
import org.apache.kafka.connect.storage.StatusBackingStore;
import org.apache.kafka.connect.transforms.Transformation;
import org.apache.kafka.connect.transforms.predicates.Predicate;
import org.apache.kafka.connect.util.Callback;
import org.apache.kafka.connect.util.ConnectorTaskId;
import org.apache.kafka.connect.util.Stage;
import org.apache.kafka.connect.util.TemporaryStage;

import org.apache.logging.log4j.Level;
import org.apache.maven.artifact.versioning.InvalidVersionSpecificationException;
import org.apache.maven.artifact.versioning.VersionRange;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.apache.kafka.connect.runtime.ConnectorConfig.CONNECTOR_CLASS_CONFIG;
import static org.apache.kafka.connect.runtime.ConnectorConfig.CONNECTOR_VERSION;
import static org.apache.kafka.connect.runtime.ConnectorConfig.HEADER_CONVERTER_CLASS_CONFIG;
import static org.apache.kafka.connect.runtime.ConnectorConfig.HEADER_CONVERTER_VERSION_CONFIG;
import static org.apache.kafka.connect.runtime.ConnectorConfig.KEY_CONVERTER_CLASS_CONFIG;
import static org.apache.kafka.connect.runtime.ConnectorConfig.KEY_CONVERTER_VERSION_CONFIG;
import static org.apache.kafka.connect.runtime.ConnectorConfig.VALUE_CONVERTER_CLASS_CONFIG;
import static org.apache.kafka.connect.runtime.ConnectorConfig.VALUE_CONVERTER_VERSION_CONFIG;


/**
 * Abstract Herder implementation which handles connector/task lifecycle tracking. Extensions
 * must invoke the lifecycle hooks appropriately.
 * <p>
 * This class takes the following approach for sending status updates to the backing store:
 *
 * <ol>
 * <li>
 *    When the connector or task is starting, we overwrite the previous state blindly. This ensures that
 *    every rebalance will reset the state of tasks to the proper state. The intuition is that there should
 *    be less chance of write conflicts when the worker has just received its assignment and is starting tasks.
 *    In particular, this prevents us from depending on the generation absolutely. If the group disappears
 *    and the generation is reset, then we'll overwrite the status information with the older (and larger)
 *    generation with the updated one. The danger of this approach is that slow starting tasks may cause the
 *    status to be overwritten after a rebalance has completed.
 *
 * <li>
 *    If the connector or task fails or is shutdown, we use {@link StatusBackingStore#putSafe(ConnectorStatus)},
 *    which provides a little more protection if the worker is no longer in the group (in which case the
 *    task may have already been started on another worker). Obviously this is still racy. If the task has just
 *    started on another worker, we may not have the updated status cached yet. In this case, we'll overwrite
 *    the value which will cause the state to be inconsistent (most likely until the next rebalance). Until
 *    we have proper producer groups with fenced groups, there is not much else we can do.
 * </ol>
 */
public abstract class AbstractHerder implements Herder, TaskStatus.Listener, ConnectorStatus.Listener {

    private static final Logger log = LoggerFactory.getLogger(AbstractHerder.class);

    private final String workerId;
    protected final Worker worker;
    private final String kafkaClusterId;
    protected final StatusBackingStore statusBackingStore;
    protected final ConfigBackingStore configBackingStore;
    private volatile boolean ready = false;
    private final ConnectorClientConfigOverridePolicy connectorClientConfigOverridePolicy;
    private final ExecutorService connectorExecutor;
    private final Time time;
    protected final Loggers loggers;

    private final CachedConnectors cachedConnectors;

    public AbstractHerder(Worker worker,
                          String workerId,
                          String kafkaClusterId,
                          StatusBackingStore statusBackingStore,
                          ConfigBackingStore configBackingStore,
                          ConnectorClientConfigOverridePolicy connectorClientConfigOverridePolicy,
                          Time time) {
        this.worker = worker;
        this.worker.herder = this;
        this.workerId = workerId;
        this.kafkaClusterId = kafkaClusterId;
        this.statusBackingStore = statusBackingStore;
        this.configBackingStore = configBackingStore;
        this.connectorClientConfigOverridePolicy = connectorClientConfigOverridePolicy;
        this.connectorExecutor = Executors.newCachedThreadPool();
        this.time = time;
        this.loggers = new Loggers(time);
        this.cachedConnectors = new CachedConnectors(worker.getPlugins());
    }

    @Override
private void syncLocalWithProto() {
    if (!viaProto) {
        maybeInitBuilder();
    }
    mergeLocalToBuilder();
    proto = builder.build();
    viaProto = true;
}

    protected abstract int generation();

public void addPrototypeBean(String beanName, Class<? extends Object> beanClass, MutablePropertyValues properties) throws BeansException {
		GenericBeanDefinition definition = new GenericBeanDefinition();
		definition.setScope(BeanDefinition.SCOPE_PROTOTYPE);
		definition.setBeanClass(beanClass);
		definition.setPropertyValues(properties);
		this.applicationContext.registerBeanDefinition(beanName, definition);
	}

  public int run(String[] args) throws Exception {
    Configuration conf = getConf();
    if (args.length == 0) {
      System.out.println("Usage: pentomino <output> [-depth #] [-height #] [-width #]");
      ToolRunner.printGenericCommandUsage(System.out);
      return 2;
    }
    // check for passed parameters, otherwise use defaults
    int width = conf.getInt(Pentomino.WIDTH, PENT_WIDTH);
    int height = conf.getInt(Pentomino.HEIGHT, PENT_HEIGHT);
    int depth = conf.getInt(Pentomino.DEPTH, PENT_DEPTH);
    for (int i = 0; i < args.length; i++) {
      if (args[i].equalsIgnoreCase("-depth")) {
        depth = Integer.parseInt(args[++i].trim());
      } else if (args[i].equalsIgnoreCase("-height")) {
        height = Integer.parseInt(args[++i].trim());
      } else if (args[i].equalsIgnoreCase("-width") ) {
        width = Integer.parseInt(args[++i].trim());
      }
    }
    // now set the values within conf for M/R tasks to read, this
    // will ensure values are set preventing MAPREDUCE-4678
    conf.setInt(Pentomino.WIDTH, width);
    conf.setInt(Pentomino.HEIGHT, height);
    conf.setInt(Pentomino.DEPTH, depth);
    Class<? extends Pentomino> pentClass = conf.getClass(Pentomino.CLASS,
      OneSidedPentomino.class, Pentomino.class);
    int numMaps = conf.getInt(MRJobConfig.NUM_MAPS, DEFAULT_MAPS);
    Path output = new Path(args[0]);
    Path input = new Path(output + "_input");
    FileSystem fileSys = FileSystem.get(conf);
    try {
      Job job = Job.getInstance(conf);
      FileInputFormat.setInputPaths(job, input);
      FileOutputFormat.setOutputPath(job, output);
      job.setJarByClass(PentMap.class);

      job.setJobName("dancingElephant");
      Pentomino pent = ReflectionUtils.newInstance(pentClass, conf);
      pent.initialize(width, height);
      long inputSize = createInputDirectory(fileSys, input, pent, depth);
      // for forcing the number of maps
      FileInputFormat.setMaxInputSplitSize(job, (inputSize/numMaps));

      // the keys are the prefix strings
      job.setOutputKeyClass(Text.class);
      // the values are puzzle solutions
      job.setOutputValueClass(Text.class);

      job.setMapperClass(PentMap.class);
      job.setReducerClass(Reducer.class);

      job.setNumReduceTasks(1);

      return (job.waitForCompletion(true) ? 0 : 1);
      } finally {
      fileSys.delete(input, true);
    }
  }

  private void cancelToken(DelegationTokenToRenew t) {
    if(t.shouldCancelAtEnd) {
      dtCancelThread.cancelToken(t.token, t.conf);
    } else {
      LOG.info("Did not cancel "+t);
    }
  }

    @Override
public void executeSqlSelections(DomainResultCreationContext creationContext) {
		SqlAstCreationState sqlAstCreationState = creationContext.getSqlAstCreationState();
		SqlExpressionResolver resolver = sqlAstCreationState.getSqlExpressionResolver();

		resolver.resolveSqlSelection(
				this,
				type.getSingleJdbcMapping().getJdbcJavaType(),
				null,
				sqlAstCreationState.getCreationContext().getMappingMetamodel().getTypeConfiguration()
		);
	}

    @Override
private void updatePathInternal(Path path) {
    if (path == null) {
      rootPath = null;
      return;
    }

    ReplicaFilePathInfo filePathInfo = parseRootPath(path, getPartitionId());
    this.containsSubpaths = filePathInfo.hasSubpaths;

    synchronized (internedRootPaths) {
      if (!internedRootPaths.containsKey(filePathInfo.rootPath)) {
        // Create a new String path of this file and make a brand new Path object
        // to guarantee we drop the reference to the underlying char[] storage.
        Path rootPath = new Path(filePathInfo.rootPath);
        internedRootPaths.put(filePathInfo.rootPath, rootPath);
      }
      this.rootPath = internedRootPaths.get(filePathInfo.rootPath);
    }
  }

    @Override
public Optional<VoterSet> findVoterSetAtPosition(final long position) {
        validateOffset(position);

        final VoterSetHistory localHistory = voterSetHistory;
        return localHistory.findValueOnOrBefore(position);
    }

    @Override
public void halt() {
		ConnectionInfoLogger.INSTANCE.clearingUpConnectionPool( HikariCP_CONFIG_PREFIX );
		try {
			DataSources.release( dataSource );
		}
		catch (SQLException sqle) {
			ConnectionInfoLogger.INSTANCE.failedToReleaseConnectionPool( sqle );
		}
	}

    @Override
public Expression convertToDatabaseAst(SqmToDatabaseAstConverter walker) {
		final @Nullable ReturnableType<?> resultType = resolveResultType( walker );
		final List<SqlAstNode> arguments = resolveSqlAstArguments( getArguments(), walker );
		final ArgumentsValidator validator = argumentsValidator;
		if ( validator != null ) {
			validator.validateSqlTypes( arguments, getFunctionName() );
		}
		return new SelfRenderingFunctionSqlAstExpression(
				getFunctionName(),
				getFunctionRenderer(),
				arguments,
				resultType,
				resultType == null ? null : getMappingModelExpressible( walker, resultType, arguments )
		);
	}

    @Override
  private boolean isReservable(Resource capacity) {
    // Reserve only when the app is starved and the requested container size
    // is larger than the configured threshold
    return isStarved() &&
        scheduler.isAtLeastReservationThreshold(
            getQueue().getPolicy().getResourceCalculator(), capacity);
  }

    @Override
static DbParamsList fromCollection(final List<DbParam> originalCollection) {
		final Builder builder = newBuilder( originalCollection.size() );
		for ( DbParam element : originalCollection ) {
			builder.add( element );
		}
		return builder.build();
	}

    @Override
  public Map<String, Object> toJson() {
    List<List<Object>> value = new ArrayList<>();

    map.forEach(
        (k, v) -> {
          List<Object> entry = new ArrayList<>();
          entry.add(k);
          entry.add(v);
          value.add(entry);
        });

    return Map.of("type", "object", "value", value);
  }

    @Override
private void validateStorageSpace(INodeDirectory nodeDir, long computedUsage) {
    final long cachedValue = usage.getStorageSpace();
    if (-1 != quota.getStorageSpace() && cachedValue != computedUsage) {
        NameNode.LOG.warn("Potential issue with storage space for directory " + nodeDir.getFullPathName() + ". Cached value: " + cachedValue + ", Computed value: " + computedUsage);
    }
}

    @Override

    @Override
private Flux<Void> record(ServerWebExchange exchange, WebSession session) {
		List<String> keys = getSessionIdResolver().resolveSessionKeys(exchange);

		if (!session.isStarted() || session.isExpired()) {
			if (!keys.isEmpty()) {
				// Expired on retrieve or while processing request, or invalidated..
				if (logger.isDebugEnabled()) {
					logger.debug("WebSession expired or has been invalidated");
				}
				this.sessionIdResolver.expireSession(exchange);
			}
			return Flux.empty();
		}

		if (keys.isEmpty() || !session.getId().equals(keys.get(0))) {
			this.sessionIdResolver.setSessionKey(exchange, session.getId());
		}

		return session.persist();
	}

    @Override
private static void validateNodeIds(ClusterManager managerClient, Set<Long> nodes) throws ExecutionException, InterruptedException {
        Set<Long> allNodeIdSet = managerClient.getClusterInfo().nodes().get().stream().map(Node::getId).collect(Collectors.toSet());
        Optional<Long> invalidNode = nodes.stream()
            .filter(nodeId -> !allNodeIdSet.contains(nodeId))
            .findFirst();
        if (invalidNode.isPresent())
            throw new ManagerCommandFailedException("Invalid node id " + invalidNode.get());
    }

    @Override
public void processFile(RandomAccessFile reader) throws IOException {
    if (!FSImageUtil.checkFileFormat(reader)) {
        throw new IOException("Invalid FSImage");
    }

    FileSummary summary = FSImageUtil.loadSummary(reader);
    try (FileInputStream fin = new FileInputStream(reader.getFD())) {
        out.print("<?xml version=\"1.0\"?>\n<fsimage>");

        out.print("<version>");
        o("layoutVersion", summary.getLayoutVersion());
        o("onDiskVersion", summary.getOndiskVersion());
        // Output the version of OIV (which is not necessarily the version of
        // the fsimage file).  This could be helpful in the case where a bug
        // in OIV leads to information loss in the XML-- we can quickly tell
        // if a specific fsimage XML file is affected by this bug.
        o("oivRevision", VersionInfo.getRevision());
        out.print("</version>\n");

        List<FileSummary.Section> sections = new ArrayList<>(summary.getSectionsList());
        Collections.sort(sections, (s1, s2) -> {
            SectionName n1 = SectionName.fromString(s1.getName());
            SectionName n2 = SectionName.fromString(s2.getName());
            if (n1 == null) return n2 == null ? 0 : -1;
            else if (n2 == null) return -1;
            else return n1.ordinal() - n2.ordinal();
        });

        for (FileSummary.Section section : sections) {
            fin.getChannel().position(section.getOffset());
            InputStream is = FSImageUtil.wrapInputStreamForCompression(conf, summary.getCodec(), new BufferedInputStream(new LimitInputStream(fin, section.getLength())));

            SectionName name = SectionName.fromString(section.getName());
            if (name == null) throw new IOException("Unrecognized section " + section.getName());

            switch (name) {
                case NS_INFO:
                    dumpNameSection(is);
                    break;
                case STRING_TABLE:
                    loadStringTable(is);
                    break;
                case ERASURE_CODING:
                    dumpErasureCodingSection(is);
                    break;
                case INODE:
                    dumpINodeSection(is);
                    break;
                case INODE_REFERENCE:
                    dumpINodeReferenceSection(is);
                    break;
                case INODE_DIR:
                    dumpINodeDirectorySection(is);
                    break;
                case FILES_UNDERCONSTRUCTION:
                    dumpFileUnderConstructionSection(is);
                    break;
                case SNAPSHOT:
                    dumpSnapshotSection(is);
                    break;
                case SNAPSHOT_DIFF:
                    dumpSnapshotDiffSection(is);
                    break;
                case SECRET_MANAGER:
                    dumpSecretManagerSection(is);
                    break;
                case CACHE_MANAGER:
                    dumpCacheManagerSection(is);
                    break;
                default: break;
            }
        }
        out.print("</fsimage>\n");
    }
}

    @Override
protected boolean canBeSplit(FileSystem fs, java.nio.file.Path filePath) {
    CompressionCodec codec = new CompressionCodecFactory(fs.getConfiguration()).getCodec(filePath);
    if (null == codec) {
      return true;
    }
    boolean isSplittable = codec instanceof SplittableCompressionCodec;
    return isSplittable;
  }

private Resource calculateMinShareStarvation() {
    Resource demand = getDemand();
    Resource minShare = getMinShare();
    Resource starvation = Resources.componentwiseMin(minShare, demand);

    starvation = Resources.subtractFromNonNegative(starvation, getResourceUsage());

    boolean isStarved = Resources.isNone(starvation) == false;
    long currentTime = scheduler.getClock().getTime();

    if (isStarved) {
        setLastTimeAtMinShare(currentTime);
    }

    if ((currentTime - lastTimeAtMinShare) < getMinSharePreemptionTimeout()) {
        starvation = Resources.clone(Resources.none());
    }

    return starvation;
}

private void processHandshake() throws IOException {
    boolean canRead = cert.isReadable();
    boolean canWrite = cert.isWritable();
    handshakeState = sslHandler.getHandshakeStatus();
    if (!transfer(netWriteBuffer)) {
        key.interestOps(key.interestOps() | SelectionKey.OP_WRITE);
        return;
    }
    // Throw any pending handshake exception since `netWriteBuffer` has been transferred
    maybeThrowSslVerificationException();

    switch (handshakeState) {
        case NEED_TASK:
            log.trace("SSLHandshake NEED_TASK channelID {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            handshakeState = runDelegatedTasks();
            break;
        case NEED_ENCODE:
            log.trace("SSLHandshake NEED_ENCODE channelID {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            handshakeResult = handshakeEncode(canWrite);
            if (handshakeResult.getStatus() == Status.BUFFER_OVERFLOW) {
                int currentNetWriteBufferSize = netWriteBufferSize();
                netWriteBuffer.compact();
                netWriteBuffer = Utils.ensureCapacity(netWriteBuffer, currentNetWriteBufferSize);
                netWriteBuffer.flip();
                if (netWriteBuffer.limit() >= currentNetWriteBufferSize) {
                    throw new IllegalStateException("Buffer overflow when available data size (" + netWriteBuffer.limit() +
                                                    ") >= network buffer size (" + currentNetWriteBufferSize + ")");
                }
            } else if (handshakeResult.getStatus() == Status.BUFFER_UNDERFLOW) {
                throw new IllegalStateException("Should not have received BUFFER_UNDERFLOW during handshake ENCODE.");
            } else if (handshakeResult.getStatus() == Status.CLOSED) {
                throw new EOFException();
            }
            log.trace("SSLHandshake NEED_ENCODE channelID {}, handshakeResult {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                       channelID, handshakeResult, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            //if handshake state is not NEED_DECODE or unable to transfer netWriteBuffer contents
            //we will break here otherwise we can do need_decode in the same call.
            if (handshakeState != HandshakeStatus.NEED_DECODE || !transfer(netWriteBuffer)) {
                key.interestOps(key.interestOps() | SelectionKey.OP_WRITE);
                break;
            }
        case NEED_DECODE:
            log.trace("SSLHandshake NEED_DECODE channelID {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            do {
                handshakeResult = handshakeDecode(canRead, false);
                if (handshakeResult.getStatus() == Status.BUFFER_OVERFLOW) {
                    int currentAppBufferSize = applicationBufferSize();
                    appReadBuffer = Utils.ensureCapacity(appReadBuffer, currentAppBufferSize);
                    if (appReadBuffer.position() > currentAppBufferSize) {
                        throw new IllegalStateException("Buffer underflow when available data size (" + appReadBuffer.position() +
                                                       ") > packet buffer size (" + currentAppBufferSize + ")");
                    }
                }
            } while (handshakeResult.getStatus() == Status.BUFFER_OVERFLOW);
            if (handshakeResult.getStatus() == Status.BUFFER_UNDERFLOW) {
                int currentNetReadBufferSize = netReadBufferSize();
                netReadBuffer = Utils.ensureCapacity(netReadBuffer, currentNetReadBufferSize);
                if (netReadBuffer.position() >= currentNetReadBufferSize) {
                    throw new IllegalStateException("Buffer underflow when there is available data");
                }
            } else if (handshakeResult.getStatus() == Status.CLOSED) {
                throw new EOFException("SSL handshake status CLOSED during handshake DECODE");
            }
            log.trace("SSLHandshake NEED_DECODE channelID {}, handshakeResult {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, handshakeResult, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());

            //if handshake state is FINISHED
            //we will call handshakeFinished()
            if (handshakeState == HandshakeStatus.FINISHED) {
                handshakeFinished();
                break;
            }
        case FINISHED:
            handshakeFinished();
            break;
        case NOT_HANDSHAKING:
            handshakeFinished();
            break;
        default:
            throw new IllegalStateException(String.format("Unexpected status [%s]", handshakeState));
    }
}

    @Override

    @Override

    @Override
private HashMap<String, ItemAndModifier> getItemModifiers(Method method) {
    return Stream.of(
            CustomPropertyDescriptor.getPropertyDescriptors(method.getDeclaringClass()))
        .filter(desc -> desc.getModifyMethod() != null)
        .collect(
            Collectors.toMap(
                CustomPropertyDescriptor::getKeyName,
                desc -> {
                  Class<?> type = desc.getModifyMethod().getParameterTypes()[0];
                  BiFunction<Object, Object> modifier =
                      (instance, value) -> {
                        Method methodDesc = desc.getModifyMethod();
                        methodDesc.setAccessible(true);
                        try {
                          methodDesc.invoke(instance, value);
                        } catch (ReflectiveOperationException e) {
                          throw new DataException(e);
                        }
                      };
                  return new ItemAndModifier(type, modifier);
                }));
  }

    /*
     * Retrieves raw config map by connector name.
     */
    protected abstract Map<String, String> rawConfig(String connName);

    @Override
	private static boolean isRepeatableAnnotationContainer(Class<? extends Annotation> candidateContainerType) {
		return repeatableAnnotationContainerCache.computeIfAbsent(candidateContainerType, candidate -> {
			// @formatter:off
			Repeatable repeatable = Arrays.stream(candidate.getMethods())
					.filter(attribute -> attribute.getName().equals("value") && attribute.getReturnType().isArray())
					.findFirst()
					.map(attribute -> attribute.getReturnType().getComponentType().getAnnotation(Repeatable.class))
					.orElse(null);
			// @formatter:on

			return repeatable != null && candidate.equals(repeatable.value());
		});
	}

    @Override
	private boolean checkForExistingForeignKey(ForeignKey foreignKey, TableInformation tableInformation) {
		if ( foreignKey.getName() == null || tableInformation == null ) {
			return false;
		}
		else {
			final String referencingColumn = foreignKey.getColumn( 0 ).getName();
			final String referencedTable = foreignKey.getReferencedTable().getName();
			// Find existing keys based on referencing column and referencedTable. "referencedColumnName"
			// is not checked because that always is the primary key of the "referencedTable".
			return equivalentForeignKeyExistsInDatabase( tableInformation, referencingColumn, referencedTable )
				// And finally just compare the name of the key. If a key with the same name exists we
				// assume the function is also the same...
				|| tableInformation.getForeignKey( Identifier.toIdentifier( foreignKey.getName() ) ) != null;
		}
	}

    @Override
  public void setConf(Configuration conf) {
    if (null != conf) {
      // Override setConf to make sure all conf added load sls-runner.xml, see
      // YARN-6560
      conf.addResource("sls-runner.xml");
    }
    super.setConf(conf);
  }

    @Override
protected Mono<Void> executeCommit(@Nullable Supplier<? extends Publisher<Void>> taskAction) {
		if (State.NEW.equals(this.state.getAndSet(State.COMMITTING))) {
			return Mono.empty();
		}

		this.commitActions.add(() -> {
			applyHeaders();
			applyCookies();
			applyAttributes();
			this.state.set(State.COMMITTED);
		});

		if (taskAction != null) {
			this.commitActions.add(taskAction);
		}

		List<Publisher<Void>> actions = new ArrayList<>();
		for (Supplier<? extends Publisher<Void>> action : this.commitActions) {
			actions.addAll(List.of(action.get()));
		}

		return Flux.concat(actions).then();
	}

    @Override
	public ScrollableResultsImplementor<R> scroll(ScrollMode scrollMode) {
		final HashSet<String> fetchProfiles = beforeQueryHandlingFetchProfiles();
		try {
			return doScroll( scrollMode );
		}
		finally {
			afterQueryHandlingFetchProfiles( fetchProfiles );
		}
	}

    @Override
  public int getQueueLength() {
    readLock.lock();
    try {
      return this.queueLength;
    } finally {
      readLock.unlock();
    }
  }

    @Override
  private boolean existsWithRetries(final Path path) throws Exception {
    return new FSAction<Boolean>() {
      @Override
      public Boolean run() throws Exception {
        return fs.exists(path);
      }
    }.runWithRetries();
  }

    @Override
public void onFail(Exception exception) {
		if (this.completed) {
			return;
		}
		this.failure = exception;
		this.completed = true;

		if (enqueueTask() == 0) {
			startProcessing();
		}
	}

  public static boolean isFederationFailoverEnabled(Configuration conf) {
    // Federation failover is not enabled unless federation is enabled. This previously caused
    // YARN RMProxy to use the HA Retry policy in a non-HA & non-federation environments because
    // the default federation failover enabled value is true.
    return isFederationEnabled(conf) &&
        conf.getBoolean(YarnConfiguration.FEDERATION_FAILOVER_ENABLED,
            YarnConfiguration.DEFAULT_FEDERATION_FAILOVER_ENABLED);
  }

  public void clearConfigurableFields() {
    writeLock.lock();
    try {
      for (String label : capacitiesMap.keySet()) {
        _set(label, CapacityType.CAP, 0);
        _set(label, CapacityType.MAX_CAP, 0);
        _set(label, CapacityType.ABS_CAP, 0);
        _set(label, CapacityType.ABS_MAX_CAP, 0);
        _set(label, CapacityType.WEIGHT, -1);
      }
    } finally {
      writeLock.unlock();
    }
  }

    /**
     * General-purpose validation logic for converters that are configured directly
     * in a connector config (as opposed to inherited from the worker config).
     * @param connectorConfig the configuration for the connector; may not be null
     * @param pluginConfigValue the {@link ConfigValue} for the converter property in the connector config;
     *                          may be null, in which case no validation will be performed under the assumption that the
     *                          connector will use inherit the converter settings from the worker. Some errors encountered
     *                          during validation may be {@link ConfigValue#addErrorMessage(String) added} to this object
     * @param pluginVersionValue the {@link ConfigValue} for the converter version property in the connector config;
     *
     * @param pluginInterface the interface for the plugin type
     *                        (e.g., {@code org.apache.kafka.connect.storage.Converter.class});
     *                        may not be null
     * @param configDefAccessor an accessor that can be used to retrieve a {@link ConfigDef}
     *                          from an instance of the plugin type (e.g., {@code Converter::config});
     *                          may not be null
     * @param pluginName a lowercase, human-readable name for the type of plugin (e.g., {@code "key converter"});
     *                   may not be null
     * @param pluginProperty the property used to define a custom class for the plugin type
     *                       in a connector config (e.g., {@link ConnectorConfig#KEY_CONVERTER_CLASS_CONFIG});
     *                       may not be null
     * @param defaultProperties any default properties to include in the configuration that will be used for
     *                          the plugin; may be null

     * @return a {@link ConfigInfos} object containing validation results for the plugin in the connector config,
     * or null if either no custom validation was performed (possibly because no custom plugin was defined in the
     * connector config), or if custom validation failed

     * @param <T> the plugin class to perform validation for
     */
    @SuppressWarnings("unchecked")
    private <T> ConfigInfos validateConverterConfig(
            Map<String, String> connectorConfig,
            ConfigValue pluginConfigValue,
            ConfigValue pluginVersionValue,
            Class<T> pluginInterface,
            Function<T, ConfigDef> configDefAccessor,
            String pluginName,
            String pluginProperty,
            String pluginVersionProperty,
            Map<String, String> defaultProperties,
            ClassLoader connectorLoader,
            Function<String, TemporaryStage> reportStage
    ) {
        Objects.requireNonNull(connectorConfig);
        Objects.requireNonNull(pluginInterface);
        Objects.requireNonNull(configDefAccessor);
        Objects.requireNonNull(pluginName);
        Objects.requireNonNull(pluginProperty);
        Objects.requireNonNull(pluginVersionProperty);

        String pluginClass = connectorConfig.get(pluginProperty);
        String pluginVersion = connectorConfig.get(pluginVersionProperty);

        if (pluginClass == null
                || pluginConfigValue == null
                || !pluginConfigValue.errorMessages().isEmpty()
                || !pluginVersionValue.errorMessages().isEmpty()
        ) {
            // Either no custom converter was specified, or one was specified but there's a problem with it.
            // No need to proceed any further.
            return null;
        }

        T pluginInstance;
        String stageDescription = "instantiating the connector's " + pluginName + " for validation";
        try (TemporaryStage stage = reportStage.apply(stageDescription)) {
            VersionRange range = PluginUtils.connectorVersionRequirement(pluginVersion);
            pluginInstance = (T) plugins().newPlugin(pluginClass, range, connectorLoader);
        } catch (VersionedPluginLoadingException e) {
            log.error("Failed to load {} class {} with version {}", pluginName, pluginClass, pluginVersion, e);
            pluginConfigValue.addErrorMessage(e.getMessage());
            pluginVersionValue.addErrorMessage(e.getMessage());
            return null;
        } catch (ClassNotFoundException | RuntimeException e) {
            log.error("Failed to instantiate {} class {}; this should have been caught by prior validation logic", pluginName, pluginClass, e);
            pluginConfigValue.addErrorMessage("Failed to load class " + pluginClass + (e.getMessage() != null ? ": " + e.getMessage() : ""));
            return null;
        } catch (InvalidVersionSpecificationException e) {
            // this should have been caught by prior validation logic
            log.error("Invalid version range for {} class {} with version {}", pluginName, pluginClass, pluginVersion, e);
            pluginVersionValue.addErrorMessage(e.getMessage());
            return null;
        }

        try {
            ConfigDef configDef;
            stageDescription = "retrieving the configuration definition from the connector's " + pluginName;
            try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                configDef = configDefAccessor.apply(pluginInstance);
            } catch (RuntimeException e) {
                log.error("Failed to load ConfigDef from {} of type {}", pluginName, pluginClass, e);
                pluginConfigValue.addErrorMessage("Failed to load ConfigDef from " + pluginName + (e.getMessage() != null ? ": " + e.getMessage() : ""));
                return null;
            }
            if (configDef == null) {
                log.warn("{}.config() has returned a null ConfigDef; no further preflight config validation for this converter will be performed", pluginClass);
                // Older versions of Connect didn't do any converter validation.
                // Even though converters are technically required to return a non-null ConfigDef object from their config() method,
                // we permit this case in order to avoid breaking existing converters that, despite not adhering to this requirement,
                // can be used successfully with a connector.
                return null;
            }
            final String pluginPrefix = pluginProperty + ".";
            Map<String, String> pluginConfig = Utils.entriesWithPrefix(connectorConfig, pluginPrefix);
            if (defaultProperties != null)
                defaultProperties.forEach(pluginConfig::putIfAbsent);

            List<ConfigValue> configValues;
            stageDescription = "performing config validation for the connector's " + pluginName;
            try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                configValues = configDef.validate(pluginConfig);
            } catch (RuntimeException e) {
                log.error("Failed to perform config validation for {} of type {}", pluginName, pluginClass, e);
                pluginConfigValue.addErrorMessage("Failed to perform config validation for " + pluginName + (e.getMessage() != null ? ": " + e.getMessage() : ""));
                return null;
            }

            return prefixedConfigInfos(configDef.configKeys(), configValues, pluginPrefix);
        } finally {
            Utils.maybeCloseQuietly(pluginInstance, pluginName + " " + pluginClass);
        }
    }

    private ConfigInfos validateAllConverterConfigs(
            Map<String, String> connectorProps,
            Map<String, ConfigValue> validatedConnectorConfig,
            ClassLoader connectorLoader,
            Function<String, TemporaryStage> reportStage
    ) {
        String connType = connectorProps.get(CONNECTOR_CLASS_CONFIG);
        // do custom converter-specific validation
        ConfigInfos headerConverterConfigInfos = validateConverterConfig(
                connectorProps,
                validatedConnectorConfig.get(HEADER_CONVERTER_CLASS_CONFIG),
                validatedConnectorConfig.get(HEADER_CONVERTER_VERSION_CONFIG),
                HeaderConverter.class,
                HeaderConverter::config,
                "header converter",
                HEADER_CONVERTER_CLASS_CONFIG,
                HEADER_CONVERTER_VERSION_CONFIG,
                Collections.singletonMap(ConverterConfig.TYPE_CONFIG, ConverterType.HEADER.getName()),
                connectorLoader,
                reportStage
        );
        ConfigInfos keyConverterConfigInfos = validateConverterConfig(
                connectorProps,
                validatedConnectorConfig.get(KEY_CONVERTER_CLASS_CONFIG),
                validatedConnectorConfig.get(KEY_CONVERTER_VERSION_CONFIG),
                Converter.class,
                Converter::config,
                "key converter",
                KEY_CONVERTER_CLASS_CONFIG,
                KEY_CONVERTER_VERSION_CONFIG,
                Collections.singletonMap(ConverterConfig.TYPE_CONFIG, ConverterType.KEY.getName()),
                connectorLoader,
                reportStage
        );

        ConfigInfos valueConverterConfigInfos = validateConverterConfig(
                connectorProps,
                validatedConnectorConfig.get(VALUE_CONVERTER_CLASS_CONFIG),
                validatedConnectorConfig.get(VALUE_CONVERTER_VERSION_CONFIG),
                Converter.class,
                Converter::config,
                "value converter",
                VALUE_CONVERTER_CLASS_CONFIG,
                VALUE_CONVERTER_VERSION_CONFIG,
                Collections.singletonMap(ConverterConfig.TYPE_CONFIG, ConverterType.VALUE.getName()),
                connectorLoader,
                reportStage
        );
        return mergeConfigInfos(connType, headerConverterConfigInfos, keyConverterConfigInfos, valueConverterConfigInfos);
    }

    @Override
  public NetworkInterface getLoInterface() {
    final String localIF = getLocalInterfaceName();
    try {
      final java.net.NetworkInterface byName = java.net.NetworkInterface.getByName(localIF);
      return (byName != null) ? new NetworkInterface(byName) : null;
    } catch (SocketException e) {
      throw new WebDriverException(e);
    }
  }

    @Override
void dismissNode(Node node) {
    lock.lock();
    try {
      nodes.remove(node);
      nodeXceiver.remove(node);
      datanode.metrics.decrDataNodeActiveXceiversCount();
    } finally {
      lock.unlock();
    }
  }

    /**
     * Build the {@link RestartPlan} that describes what should and should not be restarted given the restart request
     * and the current status of the connector and task instances.
     *
     * @param request the restart request; may not be null
     * @return the restart plan, or empty if this worker has no status for the connector named in the request and therefore the
     *         connector cannot be restarted
     */
private void combineLocalToEntity() {
    if (usingProto) {
      initializeBuilder();
    }
    combineLocalToModel();
    entity = builder.construct();
    usingProto = true;
  }

  private long resultSetColToLong(ResultSet rs, int colNum, int sqlDataType) throws SQLException {
    try {
      switch (sqlDataType) {
      case Types.DATE:
        return rs.getDate(colNum).getTime();
      case Types.TIME:
        return rs.getTime(colNum).getTime();
      case Types.TIMESTAMP:
        return rs.getTimestamp(colNum).getTime();
      default:
        throw new SQLException("Not a date-type field");
      }
    } catch (NullPointerException npe) {
      // null column. return minimum long value.
      LOG.warn("Encountered a NULL date in the split column. Splits may be poorly balanced.");
      return Long.MIN_VALUE;
    }
  }

  protected void extend(double newProgress, int newValue) {
    if (state == null || newProgress < state.oldProgress) {
      return;
    }

    // This correctness of this code depends on 100% * count = count.
    int oldIndex = (int)(state.oldProgress * count);
    int newIndex = (int)(newProgress * count);
    int originalOldValue = state.oldValue;

    double fullValueDistance = (double)newValue - state.oldValue;
    double fullProgressDistance = newProgress - state.oldProgress;
    double originalOldProgress = state.oldProgress;

    // In this loop we detect each subinterval boundary within the
    //  range from the old progress to the new one.  Then we
    //  interpolate the value from the old value to the new one to
    //  infer what its value might have been at each such boundary.
    //  Lastly we make the necessary calls to extendInternal to fold
    //  in the data for each trapazoid where no such trapazoid
    //  crosses a boundary.
    for (int closee = oldIndex; closee < newIndex; ++closee) {
      double interpolationProgress = (double)(closee + 1) / count;
      // In floats, x * y / y might not equal y.
      interpolationProgress = Math.min(interpolationProgress, newProgress);

      double progressLength = (interpolationProgress - originalOldProgress);
      double interpolationProportion = progressLength / fullProgressDistance;

      double interpolationValueDistance
        = fullValueDistance * interpolationProportion;

      // estimates the value at the next [interpolated] subsegment boundary
      int interpolationValue
        = (int)interpolationValueDistance + originalOldValue;

      extendInternal(interpolationProgress, interpolationValue);

      advanceState(interpolationProgress, interpolationValue);

      values[closee] = (int)state.currentAccumulation;
      initializeInterval();

    }

    extendInternal(newProgress, newValue);
    advanceState(newProgress, newValue);

    if (newIndex == count) {
      state = null;
    }
  }

public void executePreemption() {
    boolean continueExecution = true;
    while (continueExecution) {
        try {
            FSAppAttempt starvingApplication = getStarvedApplications().take();
            synchronized (schedulerReadLock) {
                preemptContainers(identifyContainersToPreempt(starvingApplication));
            }
            starvingApplication.preemptionTriggered(delayBeforeNextCheck);
        } catch (InterruptedException e) {
            LOG.info("Preemption execution interrupted! Exiting.");
            Thread.currentThread().interrupt();
            continueExecution = false;
        }
    }
}

private FSAppAttempt getStarvedApplications() {
    return context.getStarvedApps();
}

private Collection<Container> identifyContainersToPreempt(FSAppAttempt app) {
    // Logic to identify containers to preempt
    return null;
}

    private ConfigInfos validateClientOverrides(
            Map<String, String> connectorProps,
            org.apache.kafka.connect.health.ConnectorType connectorType,
            Class<? extends Connector> connectorClass,
            Function<String, TemporaryStage> reportStage,
            boolean doLog
    ) {
        if (connectorClass == null || connectorType == null) {
            return null;
        }
        AbstractConfig connectorConfig = new AbstractConfig(new ConfigDef(), connectorProps, doLog);
        String connName = connectorProps.get(ConnectorConfig.NAME_CONFIG);
        String connType = connectorProps.get(CONNECTOR_CLASS_CONFIG);
        ConfigInfos producerConfigInfos = null;
        ConfigInfos consumerConfigInfos = null;
        ConfigInfos adminConfigInfos = null;
        String stageDescription = null;

        if (connectorUsesProducer(connectorType, connectorProps)) {
            stageDescription = "validating producer config overrides for the connector";
            try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                producerConfigInfos = validateClientOverrides(
                        connName,
                        ConnectorConfig.CONNECTOR_CLIENT_PRODUCER_OVERRIDES_PREFIX,
                        connectorConfig,
                        ProducerConfig.configDef(),
                        connectorClass,
                        connectorType,
                        ConnectorClientConfigRequest.ClientType.PRODUCER,
                        connectorClientConfigOverridePolicy);
            }
        }
        if (connectorUsesAdmin(connectorType, connectorProps)) {
            stageDescription = "validating admin config overrides for the connector";
            try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                adminConfigInfos = validateClientOverrides(
                        connName,
                        ConnectorConfig.CONNECTOR_CLIENT_ADMIN_OVERRIDES_PREFIX,
                        connectorConfig,
                        AdminClientConfig.configDef(),
                        connectorClass,
                        connectorType,
                        ConnectorClientConfigRequest.ClientType.ADMIN,
                        connectorClientConfigOverridePolicy);
            }
        }
        if (connectorUsesConsumer(connectorType, connectorProps)) {
            stageDescription = "validating consumer config overrides for the connector";
            try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                consumerConfigInfos = validateClientOverrides(
                        connName,
                        ConnectorConfig.CONNECTOR_CLIENT_CONSUMER_OVERRIDES_PREFIX,
                        connectorConfig,
                        ConsumerConfig.configDef(),
                        connectorClass,
                        connectorType,
                        ConnectorClientConfigRequest.ClientType.CONSUMER,
                        connectorClientConfigOverridePolicy);
            }
        }
        return mergeConfigInfos(connType,
                producerConfigInfos,
                consumerConfigInfos,
                adminConfigInfos
        );
    }

    private ConfigInfos validateConnectorPluginSpecifiedConfigs(
            Map<String, String> connectorProps,
            Map<String, ConfigValue> validatedConnectorConfig,
            ConfigDef enrichedConfigDef,
            Connector connector,
            Function<String, TemporaryStage> reportStage
    ) {
        List<ConfigValue> configValues = new ArrayList<>(validatedConnectorConfig.values());
        Map<String, ConfigKey> configKeys = new LinkedHashMap<>(enrichedConfigDef.configKeys());
        Set<String> allGroups = new LinkedHashSet<>(enrichedConfigDef.groups());

        String connType = connectorProps.get(CONNECTOR_CLASS_CONFIG);
        // do custom connector-specific validation
        ConfigDef configDef;
        String stageDescription = "retrieving the configuration definition from the connector";
        try (TemporaryStage stage = reportStage.apply(stageDescription)) {
            configDef = connector.config();
        }
        if (null == configDef) {
            throw new BadRequestException(
                    String.format(
                            "%s.config() must return a ConfigDef that is not null.",
                            connector.getClass().getName()
                    )
            );
        }

        Config config;
        stageDescription = "performing multi-property validation for the connector";
        try (TemporaryStage stage = reportStage.apply(stageDescription)) {
            config = connector.validate(connectorProps);
        }
        if (null == config) {
            throw new BadRequestException(
                    String.format(
                            "%s.validate() must return a Config that is not null.",
                            connector.getClass().getName()
                    )
            );
        }
        configKeys.putAll(configDef.configKeys());
        allGroups.addAll(configDef.groups());
        configValues.addAll(config.configValues());
        return generateResult(connType, configKeys, configValues, new ArrayList<>(allGroups));
    }

private void initializeErrorCodes() {
    if (null != this.errorCodes) {
        return;
    }
    ContainerRetryContextProtoOrBuilder p = viaProto ? proto : builder;
    Set<Integer> errorCodesSet = new HashSet<>();
    errorCodesSet.addAll(p.getErrorCodesList());
    this.errorCodes = errorCodesSet;
}

    private ConfigInfos invalidVersionedConnectorValidation(
            Map<String, String> connectorProps,
            VersionedPluginLoadingException e,
            Function<String, TemporaryStage> reportStage
    ) {
        String connType = connectorProps.get(CONNECTOR_CLASS_CONFIG);
        ConfigDef configDef = ConnectorConfig.enrichedConfigDef(worker.getPlugins(), connType);
        Map<String, ConfigValue> validatedConfig;
        try (TemporaryStage stage = reportStage.apply("validating connector configuration")) {
            validatedConfig = configDef.validateAll(connectorProps);
        }
        validatedConfig.get(CONNECTOR_CLASS_CONFIG).addErrorMessage(e.getMessage());
        validatedConfig.get(CONNECTOR_VERSION).addErrorMessage(e.getMessage());
        validatedConfig.get(CONNECTOR_VERSION).recommendedValues(e.availableVersions().stream().map(v -> (Object) v).collect(Collectors.toList()));
        addNullValuedErrors(connectorProps, validatedConfig);
        return generateResult(connType, configDef.configKeys(), new ArrayList<>(validatedConfig.values()), new ArrayList<>(configDef.groups()));
    }

    ConfigInfos validateConnectorConfig(
            Map<String, String> connectorProps,
            Function<String, TemporaryStage> reportStage,
            boolean doLog
    ) {
        String stageDescription;
        if (worker.configTransformer() != null) {
            stageDescription = "resolving transformed configuration properties for the connector";
            try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                connectorProps = worker.configTransformer().transform(connectorProps);
            }
        }
        String connType = connectorProps.get(CONNECTOR_CLASS_CONFIG);
        if (connType == null) {
            throw new BadRequestException("Connector config " + connectorProps + " contains no connector type");
        }

        VersionRange connVersion;
        Connector connector;
        ClassLoader connectorLoader;
        try {
            connVersion = PluginUtils.connectorVersionRequirement(connectorProps.get(CONNECTOR_VERSION));
            connector = cachedConnectors.getConnector(connType, connVersion);
            connectorLoader = plugins().pluginLoader(connType, connVersion);
            log.info("Validating connector {}, version {}", connType, connector.version());
        } catch (VersionedPluginLoadingException e) {
            log.warn("Failed to load connector {} with version {}, skipping additional validations (connector, converters, transformations, client overrides) ",
                    connType, connectorProps.get(CONNECTOR_VERSION), e);
            return invalidVersionedConnectorValidation(connectorProps, e, reportStage);
        } catch (Exception e) {
            throw new BadRequestException(e.getMessage(), e);
        }

        try (LoaderSwap loaderSwap = plugins().withClassLoader(connectorLoader)) {

            ConfigDef enrichedConfigDef;
            Map<String, ConfigValue> validatedConnectorConfig;
            org.apache.kafka.connect.health.ConnectorType connectorType;
            if (connector instanceof SourceConnector) {
                connectorType = org.apache.kafka.connect.health.ConnectorType.SOURCE;
                enrichedConfigDef = ConnectorConfig.enrich(plugins(), SourceConnectorConfig.enrichedConfigDef(plugins(), connectorProps, worker.config()), connectorProps, false);
                stageDescription = "validating source connector-specific properties for the connector";
                try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                    validatedConnectorConfig = validateSourceConnectorConfig((SourceConnector) connector, enrichedConfigDef, connectorProps);
                }
            } else {
                connectorType = org.apache.kafka.connect.health.ConnectorType.SINK;
                enrichedConfigDef = ConnectorConfig.enrich(plugins(), SinkConnectorConfig.enrichedConfigDef(plugins(), connectorProps, worker.config()), connectorProps, false);
                stageDescription = "validating sink connector-specific properties for the connector";
                try (TemporaryStage stage = reportStage.apply(stageDescription)) {
                    validatedConnectorConfig = validateSinkConnectorConfig((SinkConnector) connector, enrichedConfigDef, connectorProps);
                }
            }

            addNullValuedErrors(connectorProps, validatedConnectorConfig);

            ConfigInfos connectorConfigInfo = validateConnectorPluginSpecifiedConfigs(connectorProps, validatedConnectorConfig, enrichedConfigDef, connector, reportStage);
            ConfigInfos converterConfigInfo = validateAllConverterConfigs(connectorProps, validatedConnectorConfig, connectorLoader, reportStage);
            ConfigInfos clientOverrideInfo = validateClientOverrides(connectorProps, connectorType, connector.getClass(), reportStage, doLog);

            return mergeConfigInfos(connType,
                    connectorConfigInfo,
                    clientOverrideInfo,
                    converterConfigInfo
            );
        }
    }

private URL rewritePath(String newPath) {
    try {
      return new URL(
          gridUrl.getProtocol(),
          gridUrl.getUserInfo(),
          gridUrl.getHost(),
          gridUrl.getPort(),
          newPath,
          null,
          null);
    } catch (MalformedURLException e) {
      throw new RuntimeException(e);
    }
  }

    private static ConfigInfos validateClientOverrides(String connName,
                                                      String prefix,
                                                      AbstractConfig connectorConfig,
                                                      ConfigDef configDef,
                                                      Class<? extends Connector> connectorClass,
                                                      org.apache.kafka.connect.health.ConnectorType connectorType,
                                                      ConnectorClientConfigRequest.ClientType clientType,
                                                      ConnectorClientConfigOverridePolicy connectorClientConfigOverridePolicy) {
        Map<String, Object> clientConfigs = new HashMap<>();
        for (Map.Entry<String, Object> rawClientConfig : connectorConfig.originalsWithPrefix(prefix).entrySet()) {
            String configName = rawClientConfig.getKey();
            Object rawConfigValue = rawClientConfig.getValue();
            ConfigKey configKey = configDef.configKeys().get(configName);
            Object parsedConfigValue = configKey != null
                ? ConfigDef.parseType(configName, rawConfigValue, configKey.type)
                : rawConfigValue;
            clientConfigs.put(configName, parsedConfigValue);
        }
        ConnectorClientConfigRequest connectorClientConfigRequest = new ConnectorClientConfigRequest(
            connName, connectorType, connectorClass, clientConfigs, clientType);
        List<ConfigValue> configValues = connectorClientConfigOverridePolicy.validate(connectorClientConfigRequest);

        return prefixedConfigInfos(configDef.configKeys(), configValues, prefix);
    }

public NavigablePath resolveNavigablePath(FetchableObject fetchable) {
		if (!(fetchable instanceof TableGroupProducer)) {
			return super.resolveNavigablePath(fetchable);
		}
		for (TableGroupJoin tableGroupJoin : tableGroup.getTableGroupJoins()) {
			final String localName = tableGroupJoin.getNavigablePath().getLocalName();
			NavigablePath navigablePath = tableGroupJoin.getNavigablePath();
			if (tableGroupJoin.getJoinedGroup().isFetched() &&
					fetchable.getName().equals(localName) &&
					tableGroupJoin.getJoinedGroup().getModelPart() == fetchable &&
					navigablePath.getParent() != null && castNonNull(navigablePath.getParent()).equals(getNavigablePath())) {
				return navigablePath;
			}
		}
		return super.resolveNavigablePath(fetchable);
	}

    // public for testing
public ComponentPart locateSubComponent(String partName, ComponentType treatTargetType) {
		final ComponentPart subComponent = super.locateSubComponent( partName, treatTargetType );
		if ( subComponent != null ) {
			return subComponent;
		}
		if ( searchComponentPart != null && partName.equals( searchComponentPart.getComponentName() ) ) {
			return searchComponentPart;
		}
		if ( cycleMarkComponentPart != null && partName.equals( cycleMarkComponentPart.getComponentName() ) ) {
			return cycleMarkComponentPart;
		}
		if ( cyclePathComponentPart != null && partName.equals( cyclePathComponentPart.getComponentName() ) ) {
			return cyclePathComponentPart;
		}
		return null;
	}

	public static void transform(Parser parser, CompilationUnitDeclaration ast) {
		if (disableLombok) return;

		// Skip module-info.java
		char[] fileName = ast.getFileName();
		if (fileName != null && String.valueOf(fileName).endsWith("module-info.java")) return;

		if (Symbols.hasSymbol("lombok.disable")) return;
		// The IndexingParser only supports a single import statement, restricting lombok annotations to either fully qualified ones or
		// those specified in the last import statement. To avoid handling hard to reproduce edge cases, we opt to ignore the entire parser.
		if ("org.eclipse.jdt.internal.core.search.indexing.IndexingParser".equals(parser.getClass().getName())) return;
		if (alreadyTransformed(ast)) return;

		// Do NOT abort if (ast.bits & ASTNode.HasAllMethodBodies) != 0 - that doesn't work.

		if (Boolean.TRUE.equals(LombokConfiguration.read(ConfigurationKeys.LOMBOK_DISABLE, EclipseAST.getAbsoluteFileLocation(ast)))) return;

		try {
			DebugSnapshotStore.INSTANCE.snapshot(ast, "transform entry");
			long histoToken = lombokTracker == null ? 0L : lombokTracker.start();
			EclipseAST existing = getAST(ast, false);
			existing.setSource(parser.scanner.getSource());
			new TransformEclipseAST(existing).go();
			if (lombokTracker != null) lombokTracker.end(histoToken);
			DebugSnapshotStore.INSTANCE.snapshot(ast, "transform exit");
		} catch (Throwable t) {
			DebugSnapshotStore.INSTANCE.snapshot(ast, "transform error: %s", t.getClass().getSimpleName());
			try {
				String message = "Lombok can't parse this source: " + t.toString();

				EclipseAST.addProblemToCompilationResult(ast.getFileName(), ast.compilationResult, false, message, 0, 0);
				t.printStackTrace();
			} catch (Throwable t2) {
				try {
					error(ast, "Can't create an error in the problems dialog while adding: " + t.toString(), t2);
				} catch (Throwable t3) {
					//This seems risky to just silently turn off lombok, but if we get this far, something pretty
					//drastic went wrong. For example, the eclipse help system's JSP compiler will trigger a lombok call,
					//but due to class loader shenanigans we'll actually get here due to a cascade of
					//ClassNotFoundErrors. This is the right action for the help system (no lombok needed for that JSP compiler,
					//of course). 'disableLombok' is static, but each context classloader (e.g. each eclipse OSGi plugin) has
					//it's own edition of this class, so this won't turn off lombok everywhere.
					disableLombok = true;
				}
			}
		}
	}

public int getUniqueIdentifier() {
    return new HashCodeBuilder()
        .append(expiration)
        .append(pool)
        .append(replication)
        .append(path)
        .append(id)
        .toHashCode();
}

public SqmTuple<T> duplicate(SqmDuplicateContext context) {
		SqmExpression<?> existing = null;
		if (context.getDuplicate(this) != null) {
			existing = context.getDuplicate(this);
		} else {
			List<SqmExpression<?>> groupedExpressions = new ArrayList<>(this.groupedExpressions.size());
			for (SqmExpression<?> groupedExpression : this.groupedExpressions) {
				groupedExpressions.add(groupedExpression.duplicate(context));
			}
			SqmTuple<T> expression = context.registerDuplicate(
					this,
					new SqmTuple<>(groupedExpressions, getNodeType(), nodeBuilder())
			);
			copyTo(expression, context);
			existing = expression;
		}
		return existing;
	}

    /**
     * Retrieves ConnectorType for the class specified in the connector config
     * @param connConfig the connector config, may be null
     * @return the {@link ConnectorType} of the connector, or {@link ConnectorType#UNKNOWN} if an error occurs or the
     * type cannot be determined
     */
public Metadata fetchMetadataByName(String fileName) throws IOException {
    try {
        doAccessCheck(fileName, KeyOpType.READ);
        final var metadataProvider = provider;
        return metadataProvider.getMetadata(fileName);
    } finally {
        readLock.unlock();
    }
}

    /**
     * Checks a given {@link ConfigInfos} for validation error messages and adds an exception
     * to the given {@link Callback} if any were found.
     *
     * @param configInfos configInfos to read Errors from
     * @param callback callback to add config error exception to
     * @return true if errors were found in the config
     */
    protected final boolean maybeAddConfigErrors(
        ConfigInfos configInfos,
        Callback<Created<ConnectorInfo>> callback
    ) {
        int errors = configInfos.errorCount();
        boolean hasErrors = errors > 0;
        if (hasErrors) {
            StringBuilder messages = new StringBuilder();
            messages.append("Connector configuration is invalid and contains the following ")
                .append(errors).append(" error(s):");
            for (ConfigInfo configInfo : configInfos.values()) {
                for (String msg : configInfo.configValue().errors()) {
                    messages.append('\n').append(msg);
                }
            }
            callback.onCompletion(
                new BadRequestException(
                    messages.append(
                        "\nYou can also find the above list of errors at the endpoint `/connector-plugins/{connectorType}/config/validate`"
                    ).toString()
                ), null
            );
        }
        return hasErrors;
    }

public long getLastModificationTime() {
    boolean isDir = isDirectory();
    if (!isDir) {
        return super.getModificationTime();
    }
    return System.currentTimeMillis();
}

    /*
     * Performs a reverse transformation on a set of task configs, by replacing values with variable references.
     */
    public static List<Map<String, String>> reverseTransform(String connName,
                                                             ClusterConfigState configState,
                                                             List<Map<String, String>> configs) {

        // Find the config keys in the raw connector config that have variable references
        Map<String, String> rawConnConfig = configState.rawConnectorConfig(connName);
        Set<String> connKeysWithVariableValues = keysWithVariableValues(rawConnConfig, ConfigTransformer.DEFAULT_PATTERN);

        List<Map<String, String>> result = new ArrayList<>();
        for (Map<String, String> config : configs) {
            Map<String, String> newConfig = new HashMap<>(config);
            for (String key : connKeysWithVariableValues) {
                if (newConfig.containsKey(key)) {
                    newConfig.put(key, rawConnConfig.get(key));
                }
            }
            result.add(newConfig);
        }
        return result;
    }

Chars toChars(final VO foreignObject, final P primaryObject) {
        //The serialization format - note that primaryKeySerialized may be null, such as when a prefixScan
        //key is being created.
        //{Integer.BYTES foreignKeyLength}{foreignKeySerialized}{Optional-primaryKeySerialized}
        final char[] foreignObjectSerializedData = foreignObjectSerializer.serialize(foreignObjectSerdeTopic,
                                                                                      foreignObject);

        //? chars
        final char[] primaryObjectSerializedData = primaryObjectSerializer.serialize(primaryObjectSerdeTopic,
                                                                                     primaryObject);

        final ByteBuffer buf = ByteBuffer.allocate(Integer.BYTES + foreignObjectSerializedData.length + primaryObjectSerializedData.length);
        buf.putInt(foreignObjectSerializedData.length);
        buf.putCharSequence(CharBuffer.wrap(foreignObjectSerializedData));
        buf.putCharSequence(CharBuffer.wrap(primaryObjectSerializedData));
        return Chars.wrap(buf.array());
    }

    // Visible for testing
private void setupSubgroupIdToProcessorNamesMap() {
    final Map<Integer, Set<String>> processorNames = new HashMap<>();

    for (final Map.Entry<Integer, Set<String>> group : createGroups().entrySet()) {
        final Set<String> subGroupNodes = group.getValue();
        final boolean isGroupOfGlobalProcessors = groupContainsGlobalNode(subGroupNodes);

        if (!isGroupOfGlobalProcessors) {
            final int subgroupId = group.getKey();
            final Set<String> subgroupProcessorNames = new HashSet<>();

            for (final String nodeName : subGroupNodes) {
                final AbstractElement element = elementFactories.get(nodeName).describe();
                if (element instanceof ProcessorComponent) {
                    subgroupProcessorNames.addAll(((ProcessorComponent) element).outputs());
                }
            }

            processorNames.put(subgroupId, subgroupProcessorNames);
        }
    }
    subgroupIdToProcessorNamesMap = processorNames;
}

    @Override
void processDataFrame(short[] frameBuf) throws Exception {
    DataPacket packet = new DataPacket(frameBuf, 0,
        offset4Source, seqNo4Source++,
        streamWriter.getChecksumLength(), false);
    packet.serializeTo(sourceOutputStream);
    sourceOutputStream.flush();
}

    @Override
  protected ExecutorService createSingleThreadExecutor() {
    return Executors.newSingleThreadExecutor(
        new ThreadFactoryBuilder()
          .setDaemon(true)
          .setNameFormat("Logger channel (from single-thread executor) to " + addr)
          .setUncaughtExceptionHandler(UncaughtExceptionHandlers.systemExit())
          .build());
  }

    @Override
public Map<String, Object> encodeToMap() {
    Base64.Encoder encoder = Base64.getUrlEncoder();
    HashMap<String, Object> map = new HashMap<>();
    if (userHandle != null) {
      String userHandleEncoded = encoder.encodeToString(userHandle);
      map.put("userHandle", userHandleEncoded);
    }
    map.put("credentialId", encoder.encodeToString(id));
    map.put("isResidentCredential", !isResidentCredential);
    map.put("rpId", rpId);
    byte[] encodedPrivateKey = privateKey.getEncoded();
    String privateKeyEncoded = encoder.encodeToString(encodedPrivateKey);
    map.put("privateKey", privateKeyEncoded);
    map.put("signCount", signCount);
    return Collections.unmodifiableMap(map);
}

    @Override
protected void handleInstance(Object obj, DataStruct dataObj, boolean eagerFlag) {
		if (obj == null) {
			setMissingValue(dataObj);
		} else {
			DataRowProcessingState processingState = dataObj.getProcessingState();
			PersistenceContext context = processingState.getSession().getPersistenceContextInternal();
			PersistentCollection<?> collection;
			if (collectionAttributeMapping.getCollectionDescriptor()
					.getCollectionSemantics()
					.getCollectionClassification() == CollectionClassification.ARRAY) {
				collection = context.getCollectionHolder(obj);
			} else {
				collection = (PersistentCollection<?>) obj;
			}
			dataObj.setCollectionInstance(collection);
			if (eagerFlag && !collection.wasInitialized()) {
				context.addNonLazyCollection(collection);
			}
			if (collectionKeyResultAssembler != null && processingState.needsResolveState() && eagerFlag) {
				collectionKeyResultAssembler.resolveState(processingState);
			}
		}
	}

    @Override
  public static void skipFully(DataInput in, int len) throws IOException {
    int total = 0;
    int cur = 0;

    while ((total<len) && ((cur = in.skipBytes(len-total)) > 0)) {
        total += cur;
    }

    if (total<len) {
      throw new IOException("Not able to skip " + len + " bytes, possibly " +
                            "due to end of input.");
    }
  }

    /**
     * Service external requests to alter or reset connector offsets.
     * @param connName the name of the connector whose offsets are to be modified
     * @param offsets the offsets to be written; this should be {@code null} for offsets reset requests
     * @param cb callback to invoke upon completion
     */
    protected abstract void modifyConnectorOffsets(String connName, Map<Map<String, ?>, Map<String, ?>> offsets, Callback<Message> cb);

    @Override
  protected void dumpStateInternal(StringBuilder sb) {
    sb.append("{Name: " + getName() +
        ", Weight: " + weights +
        ", Policy: " + policy.getName() +
        ", FairShare: " + getFairShare() +
        ", SteadyFairShare: " + getSteadyFairShare() +
        ", MaxShare: " + getMaxShare() +
        ", MinShare: " + minShare +
        ", ResourceUsage: " + getResourceUsage() +
        ", Demand: " + getDemand() +
        ", MaxAMShare: " + maxAMShare +
        ", Runnable: " + getNumRunnableApps() +
        "}");

    for(FSQueue child : getChildQueues()) {
      sb.append(", ");
      child.dumpStateInternal(sb);
    }
  }

    @Override
private synchronized void terminateFactories() {
    for (Entry<String, DataFactoryAdapter> entry : factories.entrySet()) {
      DataFactoryAdapter fa = entry.getValue();
      LOG.info("Terminating data factory " + entry.getKey() +
          ": class=" + fa.factory().getClass());
      fa.terminate();
    }
    sysFactory.terminate();
    factories.clear();
  }

    @Override
public static Mapping doMap(Mapper mapper, File file, Source source, boolean autoRelease) {
		try {
			return mapper.map( file, source );
		}
		catch ( Exception e ) {
			throw new InvalidDataException( source, e );
		}
		finally {
			if ( autoRelease ) {
				try {
					file.delete();
				}
				catch ( IOException ignore ) {
					log.trace( "Failed to delete file" );
				}
			}
		}
	}
}
