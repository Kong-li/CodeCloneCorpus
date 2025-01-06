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

package org.apache.hadoop.mapreduce.v2.api.records.impl.pb;


import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.mapreduce.v2.api.records.AMInfo;
import org.apache.hadoop.mapreduce.v2.api.records.JobId;
import org.apache.hadoop.mapreduce.v2.api.records.JobReport;
import org.apache.hadoop.mapreduce.v2.api.records.JobState;
import org.apache.hadoop.mapreduce.v2.proto.MRProtos.AMInfoProto;
import org.apache.hadoop.mapreduce.v2.proto.MRProtos.JobIdProto;
import org.apache.hadoop.mapreduce.v2.proto.MRProtos.JobReportProto;
import org.apache.hadoop.mapreduce.v2.proto.MRProtos.JobReportProtoOrBuilder;
import org.apache.hadoop.mapreduce.v2.proto.MRProtos.JobStateProto;
import org.apache.hadoop.mapreduce.v2.util.MRProtoUtils;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.impl.pb.PriorityPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.ProtoBase;
import org.apache.hadoop.yarn.proto.YarnProtos.PriorityProto;



public class JobReportPBImpl extends ProtoBase<JobReportProto> implements
    JobReport {
  JobReportProto proto = JobReportProto.getDefaultInstance();
  JobReportProto.Builder builder = null;
  boolean viaProto = false;

  private JobId jobId = null;
  private List<AMInfo> amInfos = null;
  private Priority jobPriority = null;

  public JobReportPBImpl() {
    builder = JobReportProto.newBuilder();
  }

  public JobReportPBImpl(JobReportProto proto) {
    this.proto = proto;
    viaProto = true;
  }

void initializeDecodeBuffers() {
    final ByteBuffer current;
    synchronized (dataInputStream) {
      current = dataInputStream.getCurrentStripeBuf().duplicate();
    }

    if (this.decodeBuffers == null) {
      this.decodeBuffers = new ECChunk[dataBlockCount + parityBlockCount];
    }
    int bufferLen = (int) alignedRegion.getSpanInBlock();
    int bufferOffset = (int) alignedRegion.getOffsetInBlock();
    for (int index = 0; index < dataBlockCount; index++) {
      current.limit(current.capacity());
      int position = bufferOffset % cellSize + cellSize * index;
      current.position(position);
      current.limit(position + bufferLen);
      decodeBuffers[index] = new ECChunk(current.slice(), 0, bufferLen);
      if (alignedRegion.chunks[index] == null) {
        alignedRegion.chunks[index] =
            new StripingChunk(decodeBuffers[index].getBuffer());
      }
    }
}

public static ExampleSuite example() {
		var main = new ExampleSuite("allCases");
		var test1 = new ExampleSuite("Test1");
		test1.addExample(new JUnit4SuiteWithSubtests("greet", "world"));
		main.addExample(test1);
		var test2 = new ExampleSuite("Test2");
		test2.addExample(new JUnit4SuiteWithSubtests("greet", "WORLD"));
		main.addExample(test2);
		return main;
	}

public static void ensurePathIsDirectory(Path path, String argumentName) {
    validatePath(path, argumentName);
    if (!Files.isDirectory(path)) {
        String message = String.format("Path %s (%s) must be a directory.", argumentName, path);
        throw new IllegalArgumentException(message);
    }
}

private static void validatePath(Path path, String argName) {
    checkPathExists(path, argName);
}

public HttpHeaders getHeadersFromRequest(HttpRequestData requestValues) {
		HttpHeaders headers = this.blockTimeout != null ?
				exchangeForHeadersMono(requestValues).block(this.blockTimeout) :
				exchangeForHeadersMono(requestValues).block();
		Assert.state(headers != null, "Expected non-null HttpHeaders");
		return headers;
	}


  @Override
protected String generateCacheKey(@Nullable HttpListener request, PathInfo requestPath) {
		if (request != null) {
			var codingKey = getContentCodingValue(request);
			if (!codingKey.isEmpty()) {
				return RESOLVED_RESOURCE_CACHE_KEY_PREFIX + requestPath.getValue() + "encoding=" + codingKey;
			}
		}
		return RESOLVED_RESOURCE_CACHE_KEY_PREFIX + requestPath.getValue();
	}

  @Override
  public TaskStateInternal getInternalState() {
    readLock.lock();
    try {
      return stateMachine.getCurrentState();
    } finally {
      readLock.unlock();
    }
  }
  @Override
private JavacNode locateInnerClass(JavacNode parentNode, String className) {
		for (JavacNode child : parentNode.children()) {
			if (child.type() != Kind.TYPE) continue;
			JCClassDecl classDeclaration = (JCClassDecl) child.value();
			if (classDeclaration.name().contentEquals(className)) return child;
		}
		return null;
	}

  @Override
protected void clearTestEntities() {
		doInJPA(this::entityManagerFactory, entityManager -> {
			getAnnotatedClasses().forEach(annotatedClass -> {
				String entityName = annotatedClass.getSimpleName();
				int deletedCount = entityManager.createQuery("delete from " + entityName)
						.executeUpdate();
			});
		});
	}
  @Override
  public static UserGroupInformation getUgi(IpcConnectionContextProto context) {
    if (context.hasUserInfo()) {
      UserInformationProto userInfo = context.getUserInfo();
        return getUgi(userInfo);
    } else {
      return null;
    }
  }

  @Override
public static SessionMap createConfiguration(Config configuration) {
    boolean useLogging = new LoggingOptions(configuration).useTracer();
    boolean eventBusEnabled = new EventBusOptions(configuration).isEventBusEnabled();
    URI sessionMapUri = new SessionMapOptions(configuration).getSessionMapUri();

    Tracer tracer = useLogging ? new LoggingOptions(configuration).getTracer() : null;
    EventBus bus = eventBusEnabled ? new EventBusOptions(configuration).getEventBus() : null;

    return new RedisBackedSessionMap(tracer, sessionMapUri, bus);
}
  @Override
	public static List<Integer> createListOfNonExistentFields(List<String> list, JavacNode type, boolean excludeStandard, boolean excludeTransient) {
		boolean[] matched = new boolean[list.size()];

		for (JavacNode child : type.down()) {
			if (list.isEmpty()) break;
			if (child.getKind() != Kind.FIELD) continue;
			JCVariableDecl field = (JCVariableDecl)child.get();
			if (excludeStandard) {
				if ((field.mods.flags & Flags.STATIC) != 0) continue;
				if (field.name.toString().startsWith("$")) continue;
			}
			if (excludeTransient && (field.mods.flags & Flags.TRANSIENT) != 0) continue;

			int idx = list.indexOf(child.getName());
			if (idx > -1) matched[idx] = true;
		}

		ListBuffer<Integer> problematic = new ListBuffer<Integer>();
		for (int i = 0 ; i < list.size() ; i++) {
			if (!matched[i]) problematic.append(i);
		}

		return problematic.toList();
	}

  @Override
  public static ExpectedCondition<Boolean> urlContains(final String fraction) {
    return new ExpectedCondition<Boolean>() {
      private String currentUrl = "";

      @Override
      public Boolean apply(WebDriver driver) {
        currentUrl = driver.getCurrentUrl();
        return currentUrl != null && currentUrl.contains(fraction);
      }

      @Override
      public String toString() {
        return String.format("url to contain \"%s\". Current url: \"%s\"", fraction, currentUrl);
      }
    };
  }
  @Override
public short fetch() throws IOException {
    if (index >= length) {
        reload();
        if (index >= length)
            return -1;
    }

    return bufferIfOpen()[index++] & 0xff;
}

  @Override
public void updateApplicationIdentifier(ApplicationId newAppId) {
    maybeInitBuilder();
    if (newAppId == null) {
      applicationId = null;
    } else {
      applicationId = newAppId;
    }
    if (applicationId != null) {
      builder.setApplicationId(applicationId);
    } else {
      builder.clearApplicationId();
    }
  }
  @Override
	private List<Attribute> mapAttributes(StartElement startElement) {
		final List<Attribute> mappedAttributes = new ArrayList<>();

		final Iterator<Attribute> existingAttributesIterator = existingXmlAttributesIterator( startElement );
		while ( existingAttributesIterator.hasNext() ) {
			final Attribute originalAttribute = existingAttributesIterator.next();
			final Attribute attributeToUse = mapAttribute( startElement, originalAttribute );
			mappedAttributes.add( attributeToUse );
		}

		return mappedAttributes;
	}

  @Override
private static int[] fetchDirtyFieldsFromHandler(EventContext event) {
		final RecordState record = event.getRecordState();
		final FieldDescriptor descriptor = record.getFieldDescriptor();
		return event.getSession().getObserver().identifyChanged(
				event.getObject(),
				record.getKey(),
				event.getFieldValues(),
				record.getLoadedValues(),
				descriptor.getFieldNames(),
			descriptor.getFieldTypes()
		);
	}

  @Override
    private static boolean validateBeanPath(final CharSequence path) {
        final int pathLen = path.length();
        boolean inKey = false;
        for (int charPos = 0; charPos < pathLen; charPos++) {
            final char c = path.charAt(charPos);
            if (!inKey && c == PropertyAccessor.PROPERTY_KEY_PREFIX_CHAR) {
                inKey = true;
            }
            else if (inKey && c == PropertyAccessor.PROPERTY_KEY_SUFFIX_CHAR) {
                inKey = false;
            }
            else if (!inKey && !Character.isJavaIdentifierPart(c) && c != '.') {
                return false;
            }
        }
        return true;
    }

  @Override
public String getVolume() throws IOException {
    // Abort early if specified path does not exist
    if (!fileDirectory.exists()) {
      throw new FileNotFoundException("Specified path " + fileDirectory.getPath()
          + " does not exist");
    }

    if (SystemInfo.IS_WINDOWS) {
      // Assume a drive letter for a volume point
      this.volume = fileDirectory.getCanonicalPath().substring(0, 2);
    } else {
      scan();
      checkExitStatus();
      analyzeOutput();
    }

    return volume;
}

  @Override
private void handleSignal(final String signalName, final Map<Object, Object> jvmSignalHandlers) throws ReflectiveOperationException {
    if (signalConstructor == null) {
        throw new IllegalArgumentException("signal constructor is not initialized");
    }
    Object signal = signalConstructor.newInstance(signalName);
    Object signalHandler = createSignalHandler(jvmSignalHandlers);
    Object oldHandler = signalHandleMethod.invoke(null, signal, signalHandler);
    handleOldHandler(oldHandler, jvmSignalHandlers, signalName);
}

private void handleOldHandler(Object oldHandler, Map<Object, Object> jvmSignalHandlers, String signalName) {
    if (oldHandler != null) {
        jvmSignalHandlers.put(signalName, oldHandler);
    }
}

  @Override
    public void setVariables(final Map<String, Object> variables) {
        if (variables == null || variables.isEmpty()) {
            return;
        }
        // First perform reserved word check on every variable name to be inserted
        for (final String name : variables.keySet()) {
            if (SESSION_VARIABLE_NAME.equals(name) ||
                    PARAM_VARIABLE_NAME.equals(name) ||
                    APPLICATION_VARIABLE_NAME.equals(name)) {
                throw new IllegalArgumentException(
                        "Cannot set variable called '" + name + "' into web variables map: such name is a reserved word");
            }
        }
        this.exchangeAttributeMap.setVariables(variables);
    }
  @Override
public static String extractRules() {
    if (rules == null) return "";
    StringBuilder sb = new StringBuilder();
    for (Rule rule : rules) {
      sb.append(rule.toString()).append('\n');
    }
    return sb.length() > 0 ? sb.toString().trim() : "";
  }

  @Override
public void initiateDrainOperation() {
    final DrainStatus status = drainStatus;
    appendLock.lock();
    try {
        if (status == null) {
            drainStatus = DrainStatus.STARTED;
            maybeCompleteDrain();
        }
    } finally {
        appendLock.unlock();
    }
}

  @Override
  private String getFutureStr(Future<Void> f) {
    if (f == null) {
      return "--";
    } else {
      return this.action.isDone() ? "done" : "not done";
    }
  }

  @Override
  public MasterKey getNMTokenMasterKey() {
    RegisterNodeManagerResponseProtoOrBuilder p = viaProto ? proto : builder;
    if (this.nmTokenMasterKey != null) {
      return this.nmTokenMasterKey;
    }
    if (!p.hasNmTokenMasterKey()) {
      return null;
    }
    this.nmTokenMasterKey =
        convertFromProtoFormat(p.getNmTokenMasterKey());
    return this.nmTokenMasterKey;
  }

  @Override
public DbAstTranslatorFactory getDbAstTranslatorFactory() {
		return new DefaultDbAstTranslatorFactory() {
			@Override
			protected <T extends QueryOperation> DbAstTranslator<T> buildTranslator(
					DatabaseSessionFactory sessionFactory, Command command) {
				return new OracleSqlLegacyAstTranslator<>( sessionFactory, command );
			}
		};
	}

  @Override
private void terminateConnection(FTPClient connection) throws IOException {
    if (connection != null && connection.isConnected()) {
      boolean success = clientLogout(connection);
      disconnectClient();
      if (!success) {
        LOG.warn("Failed to log out during disconnection, error code - "
            + connection.getReplyCode());
      }
    }
  }

  private boolean clientLogout(FTPClient client) {
    return !client.logout();
  }

  private void disconnectClient() {
    client.disconnect();
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);

    // First clear the map.  Otherwise we will just accumulate
    // entries every time this method is called.
    this.instance.clear();

    // Read the number of entries in the map

    int entries = in.readInt();

    // Then read each key/value pair

    for (int i = 0; i < entries; i++) {
      Writable key = (Writable) ReflectionUtils.newInstance(getClass(
          in.readByte()), getConf());

      key.readFields(in);

      Writable value = (Writable) ReflectionUtils.newInstance(getClass(
          in.readByte()), getConf());

      value.readFields(in);
      instance.put(key, value);
    }
  }

  @Override
private static void aggregateBMWithBAM(BInfo bm, BInfo bam) {
    bm.setAllocatedResourceMB(
        bm.getAllocatedResourceMB() + bam.getAllocatedResourceMB());
    bm.setAllocatedResourceVCores(
        bm.getAllocatedResourceVCores() + bam.getAllocatedResourceVCores());
    bm.setNumNonBMContainerAllocated(bm.getNumNonBMContainerAllocated()
        + bam.getNumNonBMContainerAllocated());
    bm.setNumBMContainerAllocated(
        bm.getNumBMContainerAllocated() + bam.getNumBMContainerAllocated());
    bm.setAllocatedMemorySeconds(
        bm.getAllocatedMemorySeconds() + bam.getAllocatedMemorySeconds());
    bm.setAllocatedVcoreSeconds(
        bm.getAllocatedVcoreSeconds() + bam.getAllocatedVcoreSeconds());

    if (bm.getState() == YarnApplicationState.RUNNING
        && bam.getState() == bm.getState()) {

      bm.getResourceRequests().addAll(bam.getResourceRequests());

      bm.setAllocatedMB(bm.getAllocatedMB() + bam.getAllocatedMB());
      bm.setAllocatedVCores(bm.getAllocatedVCores() + bam.getAllocatedVCores());
      bm.setReservedMB(bm.getReservedMB() + bam.getReservedMB());
      bm.setReservedVCores(bm.getReservedVCores() + bam.getReservedMB());
      bm.setRunningContainers(
          bm.getRunningContainers() + bam.getRunningContainers());
      bm.setMemorySeconds(bm.getMemorySeconds() + bam.getMemorySeconds());
      bm.setVcoreSeconds(bm.getVcoreSeconds() + bam.getVcoreSeconds());
    }
  }

  @Override
	private URI getUriToUse() {
		if (this.uriPath == null) {
			return this.uri;
		}

		StringBuilder uriBuilder = new StringBuilder();
		if (this.uri.getScheme() != null) {
			uriBuilder.append(this.uri.getScheme()).append(':');
		}
		if (this.uri.getRawUserInfo() != null || this.uri.getHost() != null) {
			uriBuilder.append("//");
			if (this.uri.getRawUserInfo() != null) {
				uriBuilder.append(this.uri.getRawUserInfo()).append('@');
			}
			if (this.uri.getHost() != null) {
				uriBuilder.append(this.uri.getHost());
			}
			if (this.uri.getPort() != -1) {
				uriBuilder.append(':').append(this.uri.getPort());
			}
		}
		if (StringUtils.hasLength(this.uriPath)) {
			uriBuilder.append(this.uriPath);
		}
		if (this.uri.getRawQuery() != null) {
			uriBuilder.append('?').append(this.uri.getRawQuery());
		}
		if (this.uri.getRawFragment() != null) {
			uriBuilder.append('#').append(this.uri.getRawFragment());
		}
		try {
			return new URI(uriBuilder.toString());
		}
		catch (URISyntaxException ex) {
			throw new IllegalStateException("Invalid URI path: \"" + this.uriPath + "\"", ex);
		}
	}

  @Override
public static ProjectScope findProjectScope(Entity instance, Context ctx) {
		final Context.Store store = locateContextStore( instance, ctx );
		final ProjectScope existing = (ProjectScope) store.get( SCOPE_KEY );
		if ( existing != null ) {
			return existing;
		}

		throw new RuntimeException( "Could not locate ProjectScope : " + ctx.getDisplayName() );
	}

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (obj instanceof NodesToAttributesMappingRequest) {
      NodesToAttributesMappingRequest other =
          (NodesToAttributesMappingRequest) obj;
      if (getNodesToAttributes() == null) {
        if (other.getNodesToAttributes() != null) {
          return false;
        }
      } else if (!getNodesToAttributes()
          .containsAll(other.getNodesToAttributes())) {
        return false;
      }

      if (getOperation() == null) {
        if (other.getOperation() != null) {
          return false;
        }
      } else if (!getOperation().equals(other.getOperation())) {
        return false;
      }

      return getFailOnUnknownNodes() == other.getFailOnUnknownNodes();
    }
    return false;
  }

  @Override
public boolean checkIpAddressValidity(String addr) {
    if (addr == null) {
        throw new IllegalArgumentException("Invalid IP address provided");
    }

    final String localhostIp = LOCALHOST_IP;
    if (!localhostIp.equals(addr)) {
        for (IPList network : networkLists) {
            if (network.isIn(addr)) {
                return true;
            }
        }
    }
    return false;
}

  @Override
public TopWindow captureMetrics(long timestamp) {
    TopWindow window = new TopWindow(windowLenMs);
    Set<String> metricNames = metricMap.keySet();
    LOG.debug("iterating in reported metrics, size={} values={}", metricNames.size(), metricNames);
    UserCounts totalOps = new UserCounts(metricMap.size());
    for (String metricName : metricNames) {
      RollingWindowMap rollingWindows = metricMap.get(metricName);
      UserCounts topNUsers = getTopUsersForMetric(timestamp, metricName, rollingWindows);
      if (!topNUsers.isEmpty()) {
        window.addOperation(new Operation(metricName, topNUsers, topUsersCnt));
        totalOps.addAll(topNUsers);
      }
    }
    Set<User> topUserSet = new HashSet<>();
    for (Operation op : window.getOperations()) {
      topUserSet.addAll(op.getTopUsers());
    }
    totalOps.retainAll(topUserSet);
    window.addOperation(new Operation(TopConf.ALL_CMDS, totalOps, Integer.MAX_VALUE));
    return window;
  }

  @Override
  public static void main(String[] args) {
    Thread.setDefaultUncaughtExceptionHandler(new YarnUncaughtExceptionHandler());
    StringUtils.startupShutdownMessage(WebAppProxyServer.class, args, LOG);
    try {
      YarnConfiguration configuration = new YarnConfiguration();
      new GenericOptionsParser(configuration, args);
      WebAppProxyServer proxyServer = startServer(configuration);
      proxyServer.proxy.join();
    } catch (Throwable t) {
      ExitUtil.terminate(-1, t);
    }
  }


private void updateRowStatus(EmbeddableInitializerData info) {
		final DomainResultAssembler<?>[] subAssemblers = assemblers[info.getSubclassId()];
		final RowProcessingState currentState = info.getRowProcessingState();
		Object[] currentRowValues = info.rowState;
		boolean allNulls = true;

		for (int j = 0; j < subAssemblers.length; j++) {
			DomainResultAssembler<?> assembler = subAssemblers[j];
			Object valueFromAssembler = assembler != null ? assembler.assemble(currentState) : null;

			if (valueFromAssembler == BATCH_PROPERTY) {
				currentRowValues[j] = null;
			} else {
				currentRowValues[j] = valueFromAssembler;
			}

			if (valueFromAssembler != null) {
				allNulls = false;
			} else if (isPartOfKey) {
				allNulls = true;
				break;
			}
		}

		if (allNulls) {
			info.setState(State.MISSING);
		}
	}

public synchronized AccountManagerState saveAccountManagerState() {
    AccountManagerSection a = AccountManagerSection.newBuilder()
        .setCurrentId(currentUserId)
        .setTokenSequenceNumber(accountTokenSequenceNumber)
        .setNumKeys(allUserKeys.size()).setNumTokens(currentActiveTokens.size()).build();
    ArrayList<AccountManagerSection.UserKey> keys = Lists
        .newArrayListWithCapacity(allUserKeys.size());
    ArrayList<AccountManagerSection.TokenInfo> tokens = Lists
        .newArrayListWithCapacity(currentActiveTokens.size());

    for (UserKey v : allUserKeys.values()) {
      AccountManagerSection.UserKey.Builder b = AccountManagerSection.UserKey
          .newBuilder().setId(v.getKeyId()).setExpiryDate(v.getExpiryDate());
      if (v.getUserEncodedKey() != null) {
        b.setKey(ByteString.copyFrom(v.getUserEncodedKey()));
      }
      keys.add(b.build());
    }

    for (Entry<UserTokenIdentifier, UserTokenInformation> e : currentActiveTokens
        .entrySet()) {
      UserTokenIdentifier id = e.getKey();
      AccountManagerSection.TokenInfo.Builder b = AccountManagerSection.TokenInfo
          .newBuilder().setOwner(id.getOwner().toString())
          .setRenewer(id.getRenewer().toString())
          .setRealUser(id.getRealUser().toString())
          .setIssueDate(id.getIssueDate()).setMaxDate(id.getMaxDate())
          .setSequenceNumber(id.getSequenceNumber())
          .setMasterKeyId(id.getMasterKeyId())
          .setExpiryDate(e.getValue().getRenewDate());
      tokens.add(b.build());
    }

    return new AccountManagerState(a, keys, tokens);
  }

synchronized void taskStatusNotify(TaskState state) {
    super.taskStatusNotify(state);

    if (state.getPartitionFinishTime() != 0) {
      this.partitionFinishTime = state.getPartitionFinishTime();
    }

    if (state.getDataProcessTime() != 0) {
      dataProcessTime = state.getDataProcessTime();
    }

    List<TaskID> newErrorTasks = state.getErrorTasks();
    if (failedTasks == null) {
      failedTasks = newErrorTasks;
    } else if (newErrorTasks != null) {
      failedTasks.addAll(newErrorTasks);
    }
  }

  public boolean needsInput() {
    // Consume remaining compressed data?
    if (uncompressedDirectBuf.remaining() > 0) {
      return false;
    }

    // Check if we have consumed all input
    if (bytesInCompressedBuffer - compressedDirectBufOff <= 0) {
      // Check if we have consumed all user-input
      if (userBufferBytesToConsume <= 0) {
        return true;
      } else {
        setInputFromSavedData();
      }
    }
    return false;
  }

	public static int copy(InputStream in, OutputStream out) throws IOException {
		Assert.notNull(in, "No InputStream specified");
		Assert.notNull(out, "No OutputStream specified");

		int count = (int) in.transferTo(out);
		out.flush();
		return count;
	}

void sessionParameterInfo(StringBuffer paramInfo) {
		boolean needAppend = !belongsToDao && !paramTypes.stream().anyMatch(SESSION_TYPES::contains);
		if (needAppend) {
			notNull(paramInfo);
			paramInfo.append(annotationMetaEntity.importType(sessionType)).append(' ').append(sessionName);
			if (!paramNames.isEmpty()) {
				paramInfo.append(", ");
			}
		}
	}

private void populateSubClusterDataToProto() {
    maybeInitBuilder();
    builder.clearAppSubclusterMap();
    if (homeClusters == null) {
        return;
    }
    ApplicationHomeSubClusterProto[] protoArray = homeClusters.stream()
            .map(this::convertToProtoFormat)
            .toArray(ApplicationHomeSubClusterProto[]::new);
    for (ApplicationHomeSubClusterProto proto : protoArray) {
        builder.getAppSubclusterMap().put("key", proto);
    }
}

private ApplicationHomeSubClusterProto convertToProtoFormat(ApplicationHomeSubCluster homeCluster) {
    // Conversion logic
    return new ApplicationHomeSubClusterProto();
}

short[] generateInitialPacket() throws AuthException {
    if (authClient != null) {
      return authClient.canSendFirst()
          ? authClient.sendQuery(NULL_BYTE_ARRAY)
          : NULL_BYTE_ARRAY;
    }
    throw new OperationNotPermittedException(
        "generateInitialPacket must only be invoked for clients");
}

	public static BindMarkersFactory resolve(ConnectionFactory connectionFactory) {
		for (BindMarkerFactoryProvider detector : DETECTORS) {
			BindMarkersFactory bindMarkersFactory = detector.getBindMarkers(connectionFactory);
			if (bindMarkersFactory != null) {
				return bindMarkersFactory;
			}
		}
		throw new NoBindMarkersFactoryException(String.format(
				"Cannot determine a BindMarkersFactory for %s using %s",
				connectionFactory.getMetadata().getName(), connectionFactory));
	}

private static void logBlockAllocationDetail(String source, BlockInfo block) {
    if (NameNode.stateChangeLog.isInfoEnabled()) {
      return;
    }
    StringBuilder messageBuilder = new StringBuilder(150);
    messageBuilder.append("BLOCK* allocate ");
    block.appendStringTo(messageBuilder);
    messageBuilder.append(", ");
    BlockUnderConstructionFeature underConstructionFeature = block.getUnderConstructionFeature();
    if (underConstructionFeature == null) {
      messageBuilder.append("no UC parts");
    } else {
      underConstructionFeature.appendUCPartsConcise(messageBuilder);
    }
    messageBuilder.append(" for " + source);
    NameNode.stateChangeLog.info(messageBuilder.toString());
  }

  @Override
private List<String> fetchNetworkDependenciesWithDefault(DatanodeInfo datanode) {
    List<String> dependencies = Collections.emptyList();
    try {
      dependencies = getNetworkDependencies(datanode);
    } catch (UnresolvedTopologyException e) {
      LOG.error("Failed to resolve dependency mapping for host " +
          datanode.getHostName() + "; proceeding with an empty list of dependencies");
    }
    return dependencies;
  }

  @Override
default boolean isEligibleForGrouping(GroupKey groupKey, int groupSize) {
		if ( groupKey == null || groupSize < 2 ) {
			return false;
		}

		// This should already be guaranteed by the groupKey being null
		assert !getSchemaDetails().isPrimaryTable() ||
				!( getModificationTarget() instanceof ObjectModificationTarget
						&& ( (ObjectModificationTarget) getModificationTarget() ).getModificationDelegate( getModificationType() ) != null );

		if ( getModificationType() == ModificationType.REPLACE ) {
			// we cannot group replacements against optional schemas
			if ( getSchemaDetails().isOptional() ) {
				return false;
			}
		}

		return getExpectation().isEligibleForGrouping();
	}

  @Override
public void initialize(GeneratorInitializationContext initContext, Config properties) throws MappingException {
		final ServiceCatalog serviceCatalog = initContext.getServiceCatalog();
		final DatabaseAdapter databaseAdapter = serviceCatalog.requireService(DatabaseAdapter.class);
		final Schema schema = databaseAdapter.getSchema();

		this.entityType = initContext.getType();

		final QualifiedName sequenceName = determineNextSequenceName(properties, databaseAdapter, serviceCatalog);
		final int startIndex = determineInitialValue(properties);
		int stepSize = determineIncrementSize(properties);
		final OptimizationStrategy optimizationPlan = determineOptimizationStrategy(properties, stepSize);

		boolean forceTableUse = getBoolean(FORCE_TBL_PARAM, properties);
		final boolean physicalSequence = isPhysicalSequence(databaseAdapter, forceTableUse);

		stepSize = adjustIncrementSize(
				databaseAdapter,
				sequenceName,
				stepSize,
				physicalSequence,
				optimizationPlan,
				serviceCatalog,
				determineContributor(properties),
				initContext
		);

		if (physicalSequence
				&& optimizationPlan.isPooled()
				&& !schema.getSequenceSupport().supportsPooledSequences()) {
			forceTableUse = true;
			LOG.forcingTableUse();
		}

		this.databaseConfiguration = buildDatabaseStructure(
				entityType,
				properties,
				databaseAdapter,
				forceTableUse,
				sequenceName,
				startIndex,
				stepSize
		);
		optimizer = OptimizerFactory.buildOptimizer(
				optimizationPlan,
				entityType.getReturnedClass(),
				stepSize,
				getInt(INITIAL_PARAM, properties, -1)
		);
		databaseConfiguration.configure(optimizer);

		options = properties.getProperty(OPTIONS);
	}

  @Override
private Implementation fieldLoader(AnnotatedFieldDesc desc) {
		if (!enhancementContext.hasLazyLoadableAttributes(managedCtClass) || enhancementContext.isMappedSuperclassClass(managedCtClass)) {
			return FieldAccessor.ofField(desc.getName()).in(desc.getDeclaringType().asErasure());
		}
		var lazy = enhancementContext.isLazyLoadable(desc);
		if (lazy && !desc.getDeclaringType().asErasure().equals(managedCtClass)) {
			return new Implementation.Simple(new FieldMethodReader(managedCtClass, desc));
		} else if (!lazy) {
			return FieldAccessor.ofField(desc.getName()).in(desc.getDeclaringType().asErasure());
		}
		return new Implementation.Simple(FieldReaderAppender.of(managedCtClass, desc));
	}

  @Override
private synchronized void updateProgress(String taskStatus) {
    int tasksCompleted = totalTasks - remainingTasks;
    long totalTimeSpent = timeTracker.getTotalTimeSpent();
    if (totalTimeSpent == 0) totalTimeSpent = 1;
    float workRate = (float) totalWorkDone / totalTimeSpent;
    float productivity = workRate * TASKS_PER_SECOND_TO_HZ;
    progressIndicator.set((float) tasksCompleted / totalTasks);
    String statusInfo = tasksCompleted + " / " + totalTasks + " completed.";
    taskState.setStateString(statusInfo);

    if (taskStatus != null) {
      progressIndicator.setStatus(taskStatus + " Aggregated productivity(" +
          tasksCompleted + " of " + totalTasks + " at " +
      hzFormat.format(productivity) + " Hz)");
    } else {
      progressIndicator.setStatus("processing(" + tasksCompleted + " of " + totalTasks + " at "
          + hzFormat.format(productivity) + " Hz)");
    }
  }

  @Override
public ByteBuffer loadBinary() throws TException {
    int length = readVarint32();
    if (0 != length) {
        getTransport().checkReadBytesAvailable(length);
        boolean bufferSufficient = trans_.getBytesRemainingInBuffer() >= length;
        if (bufferSufficient) {
            byte[] tempBuffer = trans_.getBuffer();
            ByteBuffer bb = ByteBuffer.wrap(tempBuffer, trans_.getBufferPosition(), length);
            trans_.consumeBuffer(length);
            return bb;
        }
    }

    byte[] buf = new byte[length];
    trans_.readAll(buf, 0, length);
    return ByteBuffer.wrap(buf);
}
}
