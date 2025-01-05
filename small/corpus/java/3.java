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

package org.apache.hadoop.yarn.server.api.protocolrecords.impl.pb;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.NodeAttribute;
import org.apache.hadoop.yarn.api.records.NodeId;
import org.apache.hadoop.yarn.api.records.NodeLabel;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.impl.pb.ApplicationIdPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.NodeIdPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.NodeLabelPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.NodeAttributePBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.ProtoUtils;
import org.apache.hadoop.yarn.api.records.impl.pb.ResourcePBImpl;
import org.apache.hadoop.yarn.proto.YarnProtos.ApplicationIdProto;
import org.apache.hadoop.yarn.proto.YarnProtos.NodeIdProto;
import org.apache.hadoop.yarn.proto.YarnProtos.NodeLabelProto;
import org.apache.hadoop.yarn.proto.YarnProtos.NodeAttributeProto;
import org.apache.hadoop.yarn.proto.YarnProtos.ResourceProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonProtos.NodeStatusProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.LogAggregationReportProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.NMContainerStatusProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.NodeLabelsProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.NodeLabelsProto.Builder;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.NodeAttributesProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.RegisterNodeManagerRequestProto;
import org.apache.hadoop.yarn.proto.YarnServerCommonServiceProtos.RegisterNodeManagerRequestProtoOrBuilder;
import org.apache.hadoop.yarn.server.api.protocolrecords.LogAggregationReport;
import org.apache.hadoop.yarn.server.api.protocolrecords.NMContainerStatus;
import org.apache.hadoop.yarn.server.api.protocolrecords.RegisterNodeManagerRequest;
import org.apache.hadoop.yarn.server.api.records.NodeStatus;
import org.apache.hadoop.yarn.server.api.records.impl.pb.NodeStatusPBImpl;

public class RegisterNodeManagerRequestPBImpl extends RegisterNodeManagerRequest {
  RegisterNodeManagerRequestProto proto = RegisterNodeManagerRequestProto.getDefaultInstance();
  RegisterNodeManagerRequestProto.Builder builder = null;
  boolean viaProto = false;

  private Resource resource = null;
  private NodeId nodeId = null;
  private List<NMContainerStatus> containerStatuses = null;
  private List<ApplicationId> runningApplications = null;
  private Set<NodeLabel> labels = null;
  private Set<NodeAttribute> attributes = null;

  private List<LogAggregationReport> logAggregationReportsForApps = null;

  /** Physical resources in the node. */
  private Resource physicalResource = null;
  private NodeStatus nodeStatus;

  public RegisterNodeManagerRequestPBImpl() {
    builder = RegisterNodeManagerRequestProto.newBuilder();
  }

  public RegisterNodeManagerRequestPBImpl(RegisterNodeManagerRequestProto proto) {
    this.proto = proto;
    viaProto = true;
  }

private ConcreteConstraint transform(CompositeLocationConstraintProto proto) {
    switch (proto.getCompositeType()) {
    case AND:
    case OR:
      List<ConcreteConstraint> subConstraints = new ArrayList<>();
      for (LocationConstraintProto lc : proto.getChildConstraintsList()) {
        subConstraints.add(transform(lc));
      }
      return (proto.getCompositeType() == AND) ? new And(subConstraints)
          : new Or(subConstraints);
    case DELAYED_OR:
      List<TimedLocationConstraint> tSubConstraints = new ArrayList<>();
      for (TimedLocationConstraintProto lc : proto
          .getTimedChildConstraintsList()) {
        tSubConstraints.add(transform(lc));
      }
      return new DelayedOr(tSubConstraints);
    default:
      throw new YarnRuntimeException(
          "Encountered unexpected type of composite location constraint.");
    }
  }

  public List<NMToken> getNMTokensFromPreviousAttempts() {
    if (nmTokens != null) {
      return nmTokens;
    }
    initLocalNewNMTokenList();
    return nmTokens;
  }

public static void logErrorResultGraph(String title, List<ErrorResult<?>> errorResults) {
		if ( !ERROR_LOGGER.isDebugEnabled() ) {
			return;
		}

		final ErrorResultGraphPrinter graphPrinter = new ErrorResultGraphPrinter( title );
		graphPrinter.visitErrorResults( errorResults );
	}

  private LogAggregationReportProto convertToProtoFormat(
      LogAggregationReport value) {
    return ((LogAggregationReportPBImpl) value).getProto();
  }

	static Builder create(HttpStatusCode statusCode, List<HttpMessageReader<?>> messageReaders) {
		return create(statusCode, new ExchangeStrategies() {
			@Override
			public List<HttpMessageReader<?>> messageReaders() {
				return messageReaders;
			}
			@Override
			public List<HttpMessageWriter<?>> messageWriters() {
				// not used in the response
				return Collections.emptyList();
			}
		});
	}


    private Object getFieldOrDefault(BoundField field) {
        Object value = this.values[field.index];
        if (value != null)
            return value;
        else if (field.def.hasDefaultValue)
            return field.def.defaultValue;
        else if (field.def.type.isNullable())
            return null;
        else
            throw new SchemaException("Missing value for field '" + field.def.name + "' which has no default value.");
    }

public static void runSchemaUpdate(String[] commandLineArgs) {
		try {
			var parsedArgs = CommandLineArgs.parseCommandLineArgs(commandLineArgs);
			var serviceRegistry = buildStandardServiceRegistry(parsedArgs);

			try {
				var metadata = buildMetadata(parsedArgs, serviceRegistry);

				SchemaUpdate()
						.setOutputFile(parsedArgs.outputFile)
						.setDelimiter(parsedArgs.delimiter)
						.execute(parsedArgs.targetTypes, metadata, serviceRegistry);
			}
			finally {
				StandardServiceRegistryBuilder.destroy(serviceRegistry);
			}
		}
		catch (Exception e) {
			LOG.unableToRunSchemaUpdate(e);
		}
	}


  @Override
    public void removeMetricsRecorder(final RocksDBMetricsRecorder metricsRecorder) {
        final RocksDBMetricsRecorder removedMetricsRecorder =
            metricsRecordersToTrigger.remove(metricsRecorderName(metricsRecorder));
        if (removedMetricsRecorder == null) {
            throw new IllegalStateException("No RocksDB metrics recorder for store "
                + "\"" + metricsRecorder.storeName() + "\" of task " + metricsRecorder.taskId() + " could be found. "
                + "This is a bug in Kafka Streams.");
        }
    }

  @Override
public TransferOwnerRequestData.OwnerCollection toTransferOwnerRequest() {
    TransferOwnerRequestData.OwnerCollection owners =
        new TransferOwnerRequestData.OwnerCollection(nodes.size());
    for (Map.Entry<NodeName, InetSocketAddress> entry : nodes.entrySet()) {
        owners.add(
            new TransferOwnerRequestData.Owner()
                .setName(entry.getKey().value())
                .setHost(entry.getValue().getHostString())
                .setPort(entry.getValue().getPort())
        );
    }
    return owners;
}

  @Override
private Map<String, MessageQueue> getMessageQueuesMap(List<MessageQueue> queues) {
    Map<String, MessageQueue> queueMap = new HashMap<String, MessageQueue>();
    for (MessageQueue queue : queues) {
      queueMap.put(queue.getQueuePath(), queue);
    }
    return queueMap;
  }

  @Override
private YDR configure(short cid, YDR input, YDR output) {
    ServiceMapping service = ServiceRequest.mapping(input);
    String identifier = ServiceMapping.key(service);
    if (LOG.isDebugEnabled()) {
      LOG.debug("Service config identifier=" + identifier);
    }

    registry.put(identifier, service);
    return ServiceResponse.shortReply(output, cid, service.getPort());
}

  @Override
    public long residentMemorySizeEstimate() {
        long size = 0;
        size += Long.BYTES; // value.context.timestamp
        size += Long.BYTES; // value.context.offset
        if (topic != null) {
            size += topic.toCharArray().length;
        }
        size += Integer.BYTES; // partition
        for (final Header header : headers) {
            size += header.key().toCharArray().length;
            final byte[] value = header.value();
            if (value != null) {
                size += value.length;
            }
        }
        return size;
    }

  @Override
public String generateFinalAttributeDeclaration() {
		String attributeValue = "\"" + getPropertyName() + "\"";
		return "public static final " + hostingEntity.importType(String.class.getName()) +
				" " + StringUtil.getUpperUnderscoreCaseFromLowerCamelCase(getPropertyName()) + " = " + attributeValue + ";";
	}

  @Override
public F getItem(int position) {
		int listSize = this.collection.size();
		F item;
		if (position < listSize) {
			item = this.collection.get(position);
			if (item == null) {
				item = this.factory.createItem(position);
				this.collection.set(position, item);
			}
		}
		else {
			for (int x = listSize; x < position; x++) {
				this.collection.add(null);
			}
			item = this.factory.createItem(position);
			this.collection.add(item);
		}
		return item;
	}

public void configureSecurityWhitelist(Configuration conf) {
    super.setConf(conf);
    String defaultFixedFile = FIXEDWHITELIST_DEFAULT_LOCATION;
    String fixedFile = conf.get(HADOOP_SECURITY_SASL_FIXEDWHITELIST_FILE, defaultFixedFile);
    String variableFile = null;

    if (!conf.getBoolean(HADOOP_SECURITY_SASL_VARIABLEWHITELIST_ENABLE, true)) {
        variableFile = conf.get(HADOOP_SECURITY_SASL_VARIABLEWHITELIST_FILE,
            VARIABLEWHITELIST_DEFAULT_LOCATION);
        long expiryTime = 3600 * 1000;
    } else {
        variableFile = conf.get(HADOOP_SECURITY_SASL_VARIABLEWHITELIST_FILE,
            VARIABLEWHITELIST_DEFAULT_LOCATION);
        long cacheSecs = conf.getLong(HADOOP_SECURITY_SASL_VARIABLEWHITELIST_CACHE_SECS, 3600) * 1000;
        variableFile = conf.get(HADOOP_SECURITY_SASL_VARIABLEWHITELIST_FILE,
            VARIABLEWHITELIST_DEFAULT_LOCATION);
        expiryTime = cacheSecs;
    }

    String combinedLocation = fixedFile + (variableFile == null ? "" : "," + variableFile);
    whiteList = new CombinedIPWhiteList(combinedLocation, expiryTime);

    this.saslProps = getSaslProperties(conf);
}

  @Override
    private void generateAccessor(String name, String type) {
        buffer.printf("public %s %s() {%n", type, name);
        buffer.incrementIndent();
        buffer.printf("return this.%s;%n", name);
        buffer.decrementIndent();
        buffer.printf("}%n");
    }

protected Object getBeanType(Element element) {
		if (!SYSTEM_PROPERTIES_MODE_DEFAULT.equals(element.getAttribute(SYSTEM_PROPERTIES_MODE_ATTRIBUTE))) {
			return org.springframework.beans.factory.config.PropertyPlaceholderConfigurer.class;
		}
		return PropertySourcesPlaceholderConfigurer.class;
	}

  @Override
static void processDecoding(byte[] source, int startSrc, int length, byte[] destination, int startDest) {
    if (length > 0) {
        destination[startDest] =
            (byte)(((DECODE_TABLE[source[startSrc] & 0xFF]) << 2)
                   | ((DECODE_TABLE[source[startSrc + 1] & 0xFF]) >>> 4));
        if (length > 2) {
            int temp1 = DECODE_TABLE[source[startSrc + 1] & 0xFF];
            destination[startDest + 1] =
                (byte)((((temp1 << 4) & 0xF0)
                       | ((DECODE_TABLE[source[startSrc + 2] & 0xFF]) >>> 2)));
            if (length > 3) {
                int temp2 = DECODE_TABLE[source[startSrc + 2] & 0xFF];
                destination[startDest + 2] =
                    (byte)((((temp2 << 6) & 0xC0)
                           | (DECODE_TABLE[source[startSrc + 3] & 0xFF])));
            }
        }
    }
}

synchronized List<Block> invalidateWorkItems(final DatanodeDetails dd) {
    final long delay = calculateInvalidationDelay();
    if (delay > 0L) {
      BlockManager.LOG.debug("Block removal is postponed during NameNode initialization. "
              + "The removal will commence after {} ms.", delay);
      return Collections.emptyList();
    }

    int pendingLimit = blockRemovalLimit;
    List<Block> toRemove = new ArrayList<>();

    if (dataToBlocksMap.get(dd) != null) {
      pendingLimit = selectBlocksToRemoveByLimit(dataToBlocksMap.get(dd),
          toRemove, totalBlocks, pendingLimit);
    }
    if ((pendingLimit > 0) && (dataToECBlocksMap.get(dd) != null)) {
      selectBlocksToRemoveByLimit(dataToECBlocksMap.get(dd),
          toRemove, totalECBlocks, pendingLimit);
    }

    if (!toRemove.isEmpty()) {
      if (getBlocksCount(dd) == 0) {
        deregisterNode(dd);
      }
      dd.addBlocksForRemoval(toRemove);
    }
    return toRemove;
  }

  private long calculateInvalidationDelay() {
    return getInvalidationDelay();
  }

  private int selectBlocksToRemoveByLimit(Map<DatanodeDetails, List<Block>> nodeToBlocksMap,
                                         List<Block> result, int maxCount, int limit) {
    if (nodeToBlocksMap != null && !nodeToBlocksMap.isEmpty()) {
      for (Block block : nodeToBlocksMap.values()) {
        if (limit > 0) {
          result.add(block);
          limit--;
        }
      }
    }
    return limit;
  }

  private void deregisterNode(DatanodeDetails dd) {
    remove(dd);
  }

  @Override
  public synchronized void setContainerStatuses(
      List<NMContainerStatus> containerReports) {
    if (containerReports == null) {
      return;
    }
    initContainerRecoveryReports();
    this.containerStatuses.addAll(containerReports);
  }

  @Override
public static List<JCAnnotation> extractCopyableAnnotations(JavacNode currentNode) {
		List<TypeName> configuredCopyable = currentNode.getAst().readConfiguration(ConfigurationKeys.COPYABLE_ANNOTATIONS);

		ListBuffer<JCAnnotation> annotationsList = new ListBuffer<>();
		for (JavacNode child : currentNode.down()) {
			if (child.getKind() == Kind.ANNOTATION) {
				JCAnnotation annotation = (JCAnnotation) child.get();
				String typeName = getTypeName(annotation.annotationType);
				boolean isMatch = false;
				for (TypeName type : configuredCopyable) if (type != null && typeMatches(type.toString(), currentNode, typeName)) {
					annotationsList.append(annotation);
					isMatch = true;
					break;
				}
				if (!isMatch) for (String baseAnnotation : BASE_COPYABLE_ANNOTATIONS) if (typeMatches(baseAnnotation, currentNode, typeName)) {
					annotationsList.append(annotation);
					break;
				}
			}
		}

		return copyAnnotations(annotationsList.toList(), currentNode.getTreeMaker());
	}

  @Override
private static <U> U createInstance(QueryContext queryContext, Class<U> instanceClassToProxy) {
    try {
      try {
        Constructor<U> constructor = instanceClassToProxy.getConstructor(Browser.class);
        return constructor.newInstance(queryContext);
      } catch (NoSuchMethodException e) {
        return instanceClassToProxy.getDeclaredConstructor().newInstance();
      }
    } catch (ReflectiveOperationException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
	private static String urlDecode(String in) {
		try {
			return URLDecoder.decode(in, "UTF-8");
		} catch (UnsupportedEncodingException e) {
			throw new InternalError("UTF-8 not supported");
		}
	}

  @Override
private void displayUsageGuide(CmdParser<CmdOptions> parser, String infoMessage, PrintWriter output) {
		if (infoMessage != null && !infoMessage.isEmpty()) {
			output.println(infoMessage);
			output.println("-------------------------------------------------");
		}
		String helpText = parser.createCommandLineHelp("java -jar lombok.jar initProject");
		output.println(helpText);
	}

  @Override
private void handleTasksCommitRecord(String connectorName, SchemaAndValue value) {
        synchronized (lock) {
            Map<String, String> appliedConnectorConfig = connectorConfigs.get(connectorName);
            List<ConnectorTaskId> updatedTasks = new ArrayList<>();
            if (appliedConnectorConfig == null) {
                processConnectorRemoval(connectorName);
                log.debug(
                        "Ignoring task configs for connector {}; it appears that the connector was deleted previously "
                                + "and that log compaction has since removed any trace of its previous configurations "
                                + "from the config topic",
                        connectorName
                );
                return;
            }

            if (!(value.value() instanceof Map)) {
                log.error("Ignoring connector tasks configuration commit for connector '{}' because it is in the wrong format: {}", connectorName, className(value.value()));
                return;
            }
            @SuppressWarnings("unchecked")
            int newTaskCount = intValue(((Map<String, Object>) value.value()).get("tasks"));
            Map<ConnectorTaskId, Map<String, String>> deferred = deferredTaskUpdates.get(connectorName);

            Set<Integer> taskIdSet = taskIds(connectorName, deferred);
            boolean complete = completeTaskIdSet(taskIdSet, newTaskCount);
            if (!complete) {
                log.debug("We have an incomplete set of task configs for connector '{}' probably due to compaction. So we are not doing anything with the new configuration.", connectorName);
                inconsistent.add(connectorName);
            } else {
                appliedConnectorConfigs.put(
                        connectorName,
                        new AppliedConnectorConfig(appliedConnectorConfig)
                );
                if (deferred != null) {
                    taskConfigs.putAll(deferred);
                    updatedTasks.addAll(deferred.keySet());
                    connectorTaskConfigGenerations.compute(connectorName, (ignored, generation) -> generation != null ? generation + 1 : 0);
                }
                inconsistent.remove(connectorName);

                if (deferred != null)
                    deferred.clear();

                connectorTaskCounts.put(connectorName, newTaskCount);
            }

            connectorsPendingFencing.add(connectorName);
            if (started)
                updateListener.onTaskConfigUpdate(updatedTasks);
        }
    }

  @Override
private <U> Map<NodeId, Set<U>> generateNodeRolesInfoPerNode(Class<U> clazz) {
    readLock.lock();
    try {
      Map<NodeId, Set<U>> nodeToRoles = new HashMap<>();
      for (Entry<String, Server> entry : serverCollections.entrySet()) {
        String serverName = entry.getKey();
        Server server = entry.getValue();
        for (NodeId nodeId : server.rms.keySet()) {
          if (clazz.isAssignableFrom(String.class)) {
            Set<String> nodeRoles = getRolesByNode(nodeId);
            if (nodeRoles == null || nodeRoles.isEmpty()) {
              continue;
            }
            nodeToRoles.put(nodeId, (Set<U>) nodeRoles);
          } else {
            Set<NodeRole> nodeRoles = getRolesInfoByNode(nodeId);
            if (nodeRoles == null || nodeRoles.isEmpty()) {
              continue;
            }
            nodeToRoles.put(nodeId, (Set<U>) nodeRoles);
          }
        }
        if (!server.roles.isEmpty()) {
          if (clazz.isAssignableFrom(String.class)) {
            nodeToRoles.put(NodeId.newInstance(serverName, WILDCARD_PORT),
                (Set<U>) server.roles);
          } else {
            nodeToRoles.put(NodeId.newInstance(serverName, WILCARD_PORT),
                (Set<U>) createNodeRoleFromRoleNames(server.roles));
          }
        }
      }
      return Collections.unmodifiableMap(nodeToRoles);
    } finally {
      readLock.unlock();
    }
  }

  @Override
private void addRemoteResourcesToConfig() {
    maybeInitBuilder();
    builder.clearResources();
    if (resourceList == null) {
      return;
    }
    Iterable<ResourceReportProto> iterable =
        new Iterable<ResourceReportProto>() {
          @Override
          public Iterator<ResourceReportProto> iterator() {
            return new Iterator<ResourceReportProto>() {

              Iterator<ResourceReport> iter = resourceList.iterator();

              @Override
              public boolean hasNext() {
                return iter.hasNext();
              }

              @Override
              public ResourceReportProto next() {
                return convertToConfigFormat(iter.next());
              }

              @Override
              public void remove() {
                throw new UnsupportedOperationException();

              }
            };

          }
        };
    builder.addAllResources(iterable);
  }

  @Override
	protected String insertBeforeForUpdate(String limitOffsetClause, String sqlStatement) {
		Matcher forUpdateMatcher = getForUpdatePattern().matcher( sqlStatement );
		if ( forUpdateMatcher.find() ) {
			return new StringBuilder( sqlStatement )
					.insert( forUpdateMatcher.start(), limitOffsetClause )
					.toString();
		}
		else {
			return sqlStatement;
		}
	}

  @Override
public void setResourceUpdateType(ResourceUpdateType resourceUpdateType) {
    maybeInitBuilder();
    if (resourceUpdateType == null) {
      builder.clearResourceUpdateType();
      return;
    }
    builder.setResourceUpdateType(ProtoUtils.convertToProtoFormat(resourceUpdateType));
  }

  @Override
private void setupFields(ArrayString fields, ArrayString order, ArrayList<String> items) {
		for ( int j = 0, count = items.size(); j < count; j++ ) {
			final String itemDesc = items.get( j );
			final String temp = itemDesc.toLowerCase(Locale.ROOT);
			if ( temp.endsWith( " desc" ) ) {
				fields[j] = itemDesc.substring( 0, itemDesc.length() - 5 );
				order[j] = "desc";
			}
			else if ( temp.endsWith( " asc" ) ) {
				fields[j] = itemDesc.substring( 0, itemDesc.length() - 4 );
				order[j] = "asc";
			}
			else {
				fields[j] = itemDesc;
				order[j] = null;
			}
		}
	}

void validateBeforeProceeding(String action) {
    if (isLocked()) {
        throw new RuntimeException(
            String.format(
                "%s is not allowed. System is already locked: id = %d; logPath = %s",
                action,
                lockId,
                tempLogPath
            )
        );
    }
}

  @Override
protected void updateCurrentKeyId(int newKey) {
    int currentKeyId = 0;
    try {
        synchronized (this.apiLock.writeLock()) {
            currentKeyId = newKey;
        }
    } finally {
        this.apiLock.writeLock().unlock();
    }
}

  @Override
  public synchronized void setNodeAttributes(
      Set<NodeAttribute> nodeAttributes) {
    maybeInitBuilder();
    builder.clearNodeAttributes();
    this.attributes = nodeAttributes;
  }

    public boolean cancel(boolean mayInterruptIfRunning) {
        synchronized (this) {
            if (isDone()) {
                return false;
            }
            if (mayInterruptIfRunning) {
                this.cancelled = true;
                finishedLatch.countDown();
                return true;
            }
        }
        try {
            finishedLatch.await();
        } catch (InterruptedException e) {
            throw new ConnectException("Interrupted while waiting for task to complete", e);
        }
        return false;
    }

    public String toString() {
        return "ResignedState(" +
            "localId=" + localId +
            ", epoch=" + epoch +
            ", voters=" + voters +
            ", electionTimeoutMs=" + electionTimeoutMs +
            ", unackedVoters=" + unackedVoters +
            ", preferredSuccessors=" + preferredSuccessors +
            ')';
    }

public boolean compare(User other) {
    if (other == null) {
      return false;
    }
    if (other.getClass().isAssignableFrom(this.getClass())) {
      return this.getInfo().equals(this.getClass().cast(other). getInfo());
    }
    return false;
  }

  private static NodeAttributePBImpl convertFromProtoFormat(
      NodeAttributeProto p) {
    return new NodeAttributePBImpl(p);
  }

	public @Nullable Object getObject() throws IllegalAccessException {
		if (this.fieldObject == null) {
			throw new FactoryBeanNotInitializedException();
		}
		ReflectionUtils.makeAccessible(this.fieldObject);
		if (this.targetObject != null) {
			// instance field
			return this.fieldObject.get(this.targetObject);
		}
		else {
			// class field
			return this.fieldObject.get(null);
		}
	}

  private static ApplicationIdPBImpl convertFromProtoFormat(
      ApplicationIdProto p) {
    return new ApplicationIdPBImpl(p);
  }

Token<CredentialIdentifier> credentialToken() throws IOException {
    String credential = param(CredentialsParam.NAME);
    if (credential == null) {
      return null;
    }
    final Token<CredentialIdentifier> token = new
      Token<CredentialIdentifier>();
    token.decodeFromUrlString(credential);
    URI nnUri = URI.create(SecurityURI_SCHEME + "://" + secureNodeId());
    boolean isLogical = SecureUtilClient.isLogicalUri(config, nnUri);
    if (isLogical) {
      token.setService(
          SecureUtilClient.buildTokenServiceForLogicalUri(nnUri, SecurityURI_SCHEME));
    } else {
      token.setService(SecurityUtil.buildCredentialService(nnUri));
    }
    return token;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder("FileDeletionTask :");
    sb.append("  id : ").append(getTaskId());
    sb.append("  user : ").append(getUser());
    sb.append("  subDir : ").append(
        subDir == null ? "null" : subDir.toString());
    sb.append("  baseDir : ");
    if (baseDirs == null || baseDirs.size() == 0) {
      sb.append("null");
    } else {
      for (Path baseDir : baseDirs) {
        sb.append(baseDir.toString()).append(',');
      }
    }
    return sb.toString().trim();
  }

    public SuppressedInternal<K> buildFinalResultsSuppression(final Duration gracePeriod) {
        return new SuppressedInternal<>(
            name,
            gracePeriod,
            bufferConfig,
            TimeDefinitions.WindowEndTimeDefinition.instance(),
            true
        );
    }

public boolean checkAnyUninitializedProperties() {
		if ( lazyProps.isEmpty() ) {
			return false;
		}

		if ( initializedLazyProps == null ) {
			return true;
		}

		for ( String propName : lazyProps ) {
			if ( !initializedLazyProps.contains( propName ) ) {
				return true;
			}
		}

		return false;
	}

	public void conditionallyExecuteBatch(BatchKey key) {
		if ( currentBatch != null && !currentBatch.getKey().equals( key ) ) {
			JdbcBatchLogging.BATCH_LOGGER.debugf( "Conditionally executing batch - %s", currentBatch.getKey() );
			try {
				currentBatch.execute();
			}
			finally {
				currentBatch.release();
			}
		}
	}

  private static NMContainerStatusPBImpl convertFromProtoFormat(
      NMContainerStatusProto c) {
    return new NMContainerStatusPBImpl(c);
  }

  private static NMContainerStatusProto convertToProtoFormat(
      NMContainerStatus c) {
    return ((NMContainerStatusPBImpl)c).getProto();
  }

  @Override
  public synchronized List<LogAggregationReport>
      getLogAggregationReportsForApps() {
    if (this.logAggregationReportsForApps != null) {
      return this.logAggregationReportsForApps;
    }
    initLogAggregationReportsForApps();
    return logAggregationReportsForApps;
  }

  static <T> T findFirstValidInput(T[] inputs) {
    for (T input : inputs) {
      if (input != null) {
        return input;
      }
    }

    throw new HadoopIllegalArgumentException(
        "Invalid inputs are found, all being null");
  }

  private LogAggregationReport convertFromProtoFormat(
      LogAggregationReportProto logAggregationReport) {
    return new LogAggregationReportPBImpl(logAggregationReport);
  }

  @Override
  public synchronized void setLogAggregationReportsForApps(
      List<LogAggregationReport> logAggregationStatusForApps) {
    if(logAggregationStatusForApps == null) {
      builder.clearLogAggregationReportsForApps();
    }
    this.logAggregationReportsForApps = logAggregationStatusForApps;
  }

public void handleAliases(TypeValueProcessor processor) {
		Assert.notNull(processor, "TypeValueProcessor must not be null");
		synchronized (this.aliasSet) {
			List<String> aliasIdsCopy = new ArrayList<>(this.aliasIds);
			aliasIdsCopy.forEach(alias -> {
				String registeredId = this.aliasSet.get(alias);
				if (registeredId != null) {
					String resolvedAlias = processor.resolveTypeValue(alias);
					String resolvedId = processor.resolveTypeValue(registeredId);
					if (resolvedAlias == null || resolvedId == null || resolvedAlias.equals(resolvedId)) {
						this.aliasSet.remove(alias);
						this.aliasIds.remove(alias);
					}
					else if (!resolvedAlias.equals(alias)) {
						String existingId = this.aliasSet.get(resolvedAlias);
						if (existingId != null) {
							if (existingId.equals(resolvedId)) {
								// Pointing to existing alias - just remove placeholder
								this.aliasSet.remove(alias);
								this.aliasIds.remove(alias);
								return;
							}
							throw new IllegalStateException(
									"Cannot register resolved alias '" + resolvedAlias + "' (original: '" + alias +
									"') for id '" + resolvedId + "': It is already registered for id '" +
									existingId + "'.");
						}
						checkForAliasCycle(resolvedId, resolvedAlias);
						this.aliasSet.remove(alias);
						this.aliasIds.remove(alias);
						this.aliasSet.put(resolvedAlias, resolvedId);
						this.aliasIds.add(resolvedAlias);
					}
					else if (!registeredId.equals(resolvedId)) {
						this.aliasSet.put(alias, resolvedId);
						this.aliasIds.add(alias);
					}
				}
			});
		}
	}

	private @Nullable String identifierNameToUse(@Nullable String identifierName) {
		if (identifierName == null) {
			return null;
		}
		else if (isStoresUpperCaseIdentifiers()) {
			return identifierName.toUpperCase(Locale.ROOT);
		}
		else if (isStoresLowerCaseIdentifiers()) {
			return identifierName.toLowerCase(Locale.ROOT);
		}
		else {
			return identifierName;
		}
	}
}
