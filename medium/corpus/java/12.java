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

package org.apache.hadoop.yarn.server.resourcemanager.recovery.records.impl.pb;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.hadoop.io.DataInputByteBuffer;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.impl.pb.ApplicationAttemptIdPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.ContainerPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.ProtoUtils;
import org.apache.hadoop.yarn.proto.YarnProtos.FinalApplicationStatusProto;
import org.apache.hadoop.yarn.proto.YarnServerResourceManagerRecoveryProtos.ApplicationAttemptStateDataProto;
import org.apache.hadoop.yarn.proto.YarnServerResourceManagerRecoveryProtos.ApplicationAttemptStateDataProtoOrBuilder;
import org.apache.hadoop.yarn.proto.YarnServerResourceManagerRecoveryProtos.RMAppAttemptStateProto;
import org.apache.hadoop.yarn.server.resourcemanager.recovery.records.ApplicationAttemptStateData;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.attempt.RMAppAttemptState;

import org.apache.hadoop.thirdparty.protobuf.TextFormat;

public class ApplicationAttemptStateDataPBImpl extends
    ApplicationAttemptStateData {
  private static final Logger LOG =
      LoggerFactory.getLogger(ApplicationAttemptStateDataPBImpl.class);
  ApplicationAttemptStateDataProto proto =
      ApplicationAttemptStateDataProto.getDefaultInstance();
  ApplicationAttemptStateDataProto.Builder builder = null;
  boolean viaProto = false;

  private ApplicationAttemptId attemptId = null;
  private Container masterContainer = null;
  private ByteBuffer appAttemptTokens = null;

  private Map<String, Long> resourceSecondsMap;
  private Map<String, Long> preemptedResourceSecondsMap;

  public ApplicationAttemptStateDataPBImpl() {
    builder = ApplicationAttemptStateDataProto.newBuilder();
  }

  public ApplicationAttemptStateDataPBImpl(
      ApplicationAttemptStateDataProto proto) {
    this.proto = proto;
    viaProto = true;
  }

  @Override

  public void append() throws IOException {
    HdfsCompatUtil.createFile(fs(), path, 128);
    FSDataOutputStream out = null;
    byte[] data = new byte[64];
    try {
      out = fs().append(path);
      out.write(data);
      out.close();
      out = null;
      FileStatus fileStatus = fs().getFileStatus(path);
      Assert.assertEquals(128 + 64, fileStatus.getLen());
    } finally {
      IOUtils.closeStream(out);
    }
  }

	public void fileEntryPublished(TestIdentifier testIdentifier, FileEntry entry) {
		String id = inProgressIds.get(testIdentifier.getUniqueIdObject());
		eventsFileWriter.append(reported(id, Instant.now()), //
			reported -> reported.append(attachments(), attachments -> attachments.append(file(entry.getTimestamp()), //
				file -> {
					file.withPath(outputDir.relativize(entry.getPath()).toString());
					entry.getMediaType().ifPresent(file::withMediaType);
				})));
	}


  @Override
private void configureApplicationTimeouts() {
    if (this.applicationTimeouts != null) return;
    ApplicationStateDataProtoOrBuilder provider = viaProto ? proto : builder;
    List<ApplicationTimeoutMapProto> timeoutsList = provider.getApplicationTimeoutsList();
    int listSize = timeoutsList.size();
    this.applicationTimeouts = new HashMap<>(listSize);
    for (ApplicationTimeoutMapProto timeoutInfo : timeoutsList) {
        Long timeoutValue = timeoutInfo.getTimeout();
        ApplicationTimeoutType timeoutType =
            ProtoUtils.convertFromProtoFormat(timeoutInfo.getApplicationTimeoutType());
        this.applicationTimeouts.put(timeoutType, timeoutValue);
    }
}

  @Override
private Set<String> analyze(List<String> projects, boolean includeProjects) {
		Queue<Item> toAnalyze = new ArrayDeque<>();
		for (String project : projects) {
			String[] split = project.split(":");
			Item item = new Item();
			item.space = split[0];
			item.name = split[1];
			item.versionRange = VersionRange.ANY;
			toAnalyze.add(item);
		}
		Set<Project> analyzed = new HashSet<>();
		while (!toAnalyze.isEmpty()) {
			Item next = toAnalyze.poll();

			// Skip already analyzed
			if (analyzed.stream().anyMatch(p -> p.matches(next))) {
				continue;
			}

			List<Project> matchingProjects = repository.projects.stream().filter(p -> p.matches(next)).collect(Collectors.toList());
			// Skip unknown
			if (matchingProjects.isEmpty()) {
				System.out.println("Skipping unknown project " + next);
				continue;
			}

			// Skip JDK dependencies
			boolean jdkDependency = matchingProjects.stream().anyMatch(p -> p.id.equals("a.jre.javase"));
			if (jdkDependency) {
				continue;
			}

			if (matchingProjects.size() > 1) {
				System.out.println("Ambiguous analysis for " + next + ": " + matchingProjects.toString() + ", picking first");
			}

			Project project = matchingProjects.get(0);
			analyzed.add(project);

			if (includeProjects && project.dependencies != null) {
				for (Item dependency : project.dependencies) {
					if (dependency.optional) continue;
					if (!matchesFilter(dependency.filter)) continue;

					toAnalyze.add(dependency);
				}
			}
		}

		return analyzed.stream().map(p -> p.toString() + ".jar").collect(Collectors.toSet());
	}

  @Override
  public static void main(String[] args) throws Exception {
    if (args.length < 4) {
      System.out.println("Arguments: <WORKDIR> <MINIKDCPROPERTIES> " +
              "<KEYTABFILE> [<PRINCIPALS>]+");
      System.exit(1);
    }
    File workDir = new File(args[0]);
    if (!workDir.exists()) {
      throw new RuntimeException("Specified work directory does not exists: "
              + workDir.getAbsolutePath());
    }
    Properties conf = createConf();
    File file = new File(args[1]);
    if (!file.exists()) {
      throw new RuntimeException("Specified configuration does not exists: "
              + file.getAbsolutePath());
    }
    Properties userConf = new Properties();
    InputStreamReader r = null;
    try {
      r = new InputStreamReader(new FileInputStream(file),
          StandardCharsets.UTF_8);
      userConf.load(r);
    } finally {
      if (r != null) {
        r.close();
      }
    }
    for (Map.Entry<?, ?> entry : userConf.entrySet()) {
      conf.put(entry.getKey(), entry.getValue());
    }
    final MiniKdc miniKdc = new MiniKdc(conf, workDir);
    miniKdc.start();
    File krb5conf = new File(workDir, "krb5.conf");
    if (miniKdc.getKrb5conf().renameTo(krb5conf)) {
      File keytabFile = new File(args[2]).getAbsoluteFile();
      String[] principals = new String[args.length - 3];
      System.arraycopy(args, 3, principals, 0, args.length - 3);
      miniKdc.createPrincipal(keytabFile, principals);
      System.out.println();
      System.out.println("Standalone MiniKdc Running");
      System.out.println("---------------------------------------------------");
      System.out.println("  Realm           : " + miniKdc.getRealm());
      System.out.println("  Running at      : " + miniKdc.getHost() + ":" +
              miniKdc.getHost());
      System.out.println("  krb5conf        : " + krb5conf);
      System.out.println();
      System.out.println("  created keytab  : " + keytabFile);
      System.out.println("  with principals : " + Arrays.asList(principals));
      System.out.println();
      System.out.println(" Do <CTRL-C> or kill <PID> to stop it");
      System.out.println("---------------------------------------------------");
      System.out.println();
      Runtime.getRuntime().addShutdownHook(new Thread() {
        @Override
        public void run() {
          miniKdc.stop();
        }
      });
    } else {
      throw new RuntimeException("Cannot rename KDC's krb5conf to "
              + krb5conf.getAbsolutePath());
    }
  }

  @Override
public String toDebugString() {
    boolean hasData = fetchedData != null;
    int startOffset = logStartOffset;
    long endOffset = logEndOffset;
    return "LogReadInfo(" +
            "hasData=" + hasData +
            ", divergingEpoch=" + divergingEpoch +
            ", highWatermark=" + highWatermark +
            ", startOffset=" + startOffset +
            ", endOffset=" + endOffset +
            ", lastStableOffset=" + lastStableOffset +
            ')';
}

  @Override
public void processMethod(EclipseNode node, AbstractMethodDeclaration declaration, List<DeclaredException> exceptions) {
		if (!declaration.isAbstract()) {
			node.addError("@SneakyThrows can only be used on concrete methods.");
			return;
		}

		Statement[] statements = declaration.statements;
		if (statements == null || statements.length == 0) {
			if (declaration instanceof ConstructorDeclaration) {
				ConstructorCall constructorCall = ((ConstructorDeclaration) declaration).constructorCall;
				boolean hasExplicitConstructorCall = constructorCall != null && !constructorCall.isImplicitSuper() && !constructorCall.isImplicitThis();

				if (hasExplicitConstructorCall) {
					node.addWarning("Calls to sibling / super constructors are always excluded from @SneakyThrows; @SneakyThrows has been ignored because there is no other code in this constructor.");
				} else {
					node.addWarning("This method or constructor is empty; @SneakyThrows has been ignored.");
				}

				return;
			}
		}

		for (DeclaredException exception : exceptions) {
			if (statements != null && statements.length > 0) {
				statements = new Statement[] { buildTryCatchBlock(statements, exception, exception.node, declaration) };
			}
		}

		declaration.statements = statements;
		node.up().rebuild();
	}

	private Statement buildTryCatchBlock(Statement[] originalStatements, DeclaredException exception, Node node, AbstractMethodDeclaration methodDeclaration) {
		TryCatchBlock tryCatchBlock = new TryCatchBlock();
		tryCatchBlock.setStatements(originalStatements);
		tryCatchBlock.addException(exception.exceptionType);

		return tryCatchBlock;
	}

  @Override
public void addFunctions(FunctionContributions functionContributions) {
		final String className = this.getClass().getCanonicalName();
		HSMessageLogger.SPATIAL_MSG_LOGGER.functionContributions(className);
		SqlServerSqmFunctionDescriptors functions = new SqlServerSqmFunctionDescriptors(functionContributions);
		SqmFunctionRegistry functionRegistry = functionContributions.getFunctionRegistry();
		for (Map.Entry<String, Object> entry : functions.asMap().entrySet()) {
			functionRegistry.register(entry.getKey(), (SqmFunctionDescriptor) entry.getValue());
			if (entry.getValue() instanceof SqmFunctionDescriptorWithAltName && !((SqmFunctionDescriptorWithAltName) entry.getValue()).getAltName().isEmpty()) {
				functionRegistry.registerAlternateKey(((SqmFunctionDescriptorWithAltName) entry.getValue()).getAltName(), entry.getKey());
			}
		}
	}

  @Override
    public boolean maybePunctuateSystemTime() {
        final long systemTime = time.milliseconds();

        final boolean punctuated = systemTimePunctuationQueue.maybePunctuate(systemTime, PunctuationType.WALL_CLOCK_TIME, this);

        if (punctuated) {
            commitNeeded = true;
        }

        return punctuated;
    }

  @Override
public int getLatestProcessLogId() throws IOException {
    rpcServer.checkOperation(OperationCategory.WRITE);

    RemoteMethod method =
        new RemoteMethod(LoggerProtocol.class, "getLatestProcessLogId");
    return rpcServer.invokeAtAvailableNs(method, int.class);
  }

  @Override
    static int mainNoExit(String... args) {
        try {
            execute(args);
            return 0;
        } catch (Throwable e) {
            System.err.println(e.getMessage());
            System.err.println(Utils.stackTrace(e));
            return 1;
        }
    }

  @Override
protected void handleSetAssignment(AssignNode assignment) {
		final Statement currentStatement = getStatementStack().peekCurrent();
		UpdateStatement statement;
		if (!(currentStatement instanceof UpdateStatement)
				|| !hasNonTrivialFromClause(((statement = (UpdateStatement) currentStatement).getFromClause()))) {
			super.visitSetAssignment(assignment);
		} else {
			visitSetAssignmentEmulateJoin(assignment, statement);
		}
	}

  @Override
public boolean isEqual(@Nullable Object obj) {
		if (this != obj) {
			return false;
		}
		if (obj == null || getClass() != obj.getClass()) {
			return false;
		}
		var content = getContent();
		var otherContent = ((AbstractMessageCondition<?>) obj).getContent();
		return content.equals(otherContent);
	}

  @Override
public boolean processTimerTaskEntry(TimerTaskEntry entry) {
    long exp = entry.expirationMs;

    if (entry.cancelled()) {
        return false;
    } else if (exp < System.currentTimeMillis() + 1000) { // 修改tickMs为硬编码值
        return false;
    } else if (exp < System.currentTimeMillis() + interval) {
        long virtualId = exp / 1000; // 修改tickMs为硬编码值
        int bucketIndex = (int) (virtualId % wheelSize);
        TimerTaskList bucket = buckets[bucketIndex];
        boolean added = bucket.add(entry);

        if (!added && bucket.setExpiration(virtualId * 1000)) { // 修改tickMs为硬编码值
            queue.offer(bucket);
        }

        return added;
    } else {
        if (overflowWheel == null) addOverflowWheel();
        return overflowWheel.add(entry);
    }
}

  @Override
private void addResourceConstraintMap() {
    maybeInitBuilder();
    builder.clearResourceConstraints();
    if (this.resourceConstraints == null) {
      return;
    }
    List<YarnProtos.ResourceConstraintMapEntryProto> protoList =
        new ArrayList<>();
    for (Map.Entry<Set<String>, ResourceConstraint> entry :
        this.resourceConstraints.entrySet()) {
      protoList.add(
          YarnProtos.ResourceConstraintMapEntryProto.newBuilder()
              .addAllAllocationTags(entry.getKey())
              .setResourceConstraint(
                  new ResourceConstraintToProtoConverter(
                      entry.getValue()).convert())
              .build());
    }
    builder.addAllResourceConstraints(protoList);
  }

  @Override
public void setupFunctionLibrary(FunctionContributions contributions) {
		super.initializeFunctionRegistry(contributions);

		CommonFunctionCreator creator = new CommonFunctionCreator(contributions);

		if (contributions != null) {
			creator.unnestSybasease();
			int maxSeriesSize = getMaximumSeriesSize();
			creator.generateSeriesSybasease(maxSeriesSize);
			creator.xmltableSybasease();
		}
	}

  @Override
public static void logWarning(String message, Throwable exception) {
		if (logger != null) {
			try {
				logger.warning(message, exception);
			} catch (Throwable t) {
				logger = new TerminalLogger();
				logger.warning(message, exception);
			}
		} else {
			init();
			logger.warning(message, exception);
		}
	}

  @Override
private static Expression createDefaultExpr(TypeReference refType, int start, int end) {
		if (refType instanceof ArrayTypeReference) return new NullLiteral(start, end);
		else if (!Arrays.equals(TypeConstants.BOOLEAN, refType.getLastToken())) {
			if ((Arrays.equals(TypeConstants.CHAR, refType.getLastToken()) ||
					Arrays.equals(TypeConstants.BYTE, refType.getLastToken()) ||
					Arrays.equals(TypeConstants.SHORT, refType.getLastToken()) ||
					Arrays.equals(TypeConstants.INT, refType.getLastToken()))) {
				return IntLiteral.buildIntLiteral(new char[] {'0'}, start, end);
			} else if (Arrays.equals(TypeConstants.LONG, refType.getLastToken())) {
				return LongLiteral.buildLongLiteral(new char[] {'0', 'L'}, start, end);
			} else if (Arrays.equals(TypeConstants.FLOAT, refType.getLastToken())) {
				return new FloatLiteral(new char[] {'0', 'F'}, start, end);
			} else if (Arrays.equals(TypeConstants.DOUBLE, refType.getLastToken())) {
				return new DoubleLiteral(new char[] {'0', 'D'}, start, end);
			}
		}
		return new NullLiteral(start, end);
	}

  @Override
public DbmFunctionPath<U> duplicate(DbmCopyContext context) {
		final DbmFunctionPath<U> existing = context.getDuplicate( this );
		if ( existing != null ) {
			return existing;
		}

		final DbmFunctionPath<U> path = context.registerDuplicate(
				this,
				new DbmFunctionPath<>( getNavigablePath(), (DbmFunction<?>) function.duplicate( context ) )
		);
		duplicateTo( path, context );
		return path;
	}

  @Override
    private RequestFuture<Map<TopicPartition, OffsetAndMetadata>> sendOffsetFetchRequest(Set<TopicPartition> partitions) {
        Node coordinator = checkAndGetCoordinator();
        if (coordinator == null)
            return RequestFuture.coordinatorNotAvailable();

        log.debug("Fetching committed offsets for partitions: {}", partitions);
        // construct the request
        OffsetFetchRequest.Builder requestBuilder =
            new OffsetFetchRequest.Builder(this.rebalanceConfig.groupId, true, new ArrayList<>(partitions), throwOnFetchStableOffsetsUnsupported);

        // send the request with a callback
        return client.send(coordinator, requestBuilder)
                .compose(new OffsetFetchResponseHandler());
    }

  @Override
public static void checkSettings(Map<String, String> params, Map<String, ConfigElement> validatedConfig) {
        final String topicNames = params.get(TOPIC_CONFIG);
        final String topicRegex = params.get(TOPIC_REGEX_CONFIG);
        final String deadLetterQueueTopic = params.getOrDefault(DEAD_LETTER_QUEUE_TOPIC_NAME_CONFIG, "").trim();
        final boolean hasTopicNamesConfig = !Utils.isEmpty(topicNames);
        final boolean hasTopicRegexConfig = !Utils.isEmpty(topicRegex);
        final boolean hasDeadLetterQueueTopicConfig = !Utils.isEmpty(deadLetterQueueTopic);

        if (hasTopicNamesConfig && hasTopicRegexConfig) {
            String warningMessage = TOPIC_CONFIG + " and " + TOPIC_REGEX_CONFIG + " are mutually exclusive options, but both are set.";
            addWarning(validatedConfig, TOPIC_CONFIG, topicNames, warningMessage);
            addWarning(validatedConfig, TOPIC_REGEX_CONFIG, topicRegex, warningMessage);
        }

        if (!hasTopicNamesConfig && !hasTopicRegexConfig) {
            String warningMessage = "Must configure one of " + TOPIC_CONFIG + " or " + TOPIC_REGEX_CONFIG;
            addWarning(validatedConfig, TOPIC_CONFIG, topicNames, warningMessage);
            addWarning(validatedConfig, TOPIC_REGEX_CONFIG, topicRegex, warningMessage);
        }

        if (hasDeadLetterQueueTopicConfig) {
            if (hasTopicNamesConfig) {
                List<String> topics = parseTopicNamesList(params);
                if (topics.contains(deadLetterQueueTopic)) {
                    String warningMessage = String.format(
                            "The dead letter queue topic '%s' may not be included in the list of topics ('%s=%s') consumed by the connector",
                            deadLetterQueueTopic, TOPIC_CONFIG, topics
                    );
                    addWarning(validatedConfig, TOPIC_CONFIG, topicNames, warningMessage);
                }
            }
            if (hasTopicRegexConfig) {
                Pattern pattern = Pattern.compile(topicRegex);
                if (pattern.matcher(deadLetterQueueTopic).matches()) {
                    String warningMessage = String.format(
                            "The dead letter queue topic '%s' may not be matched by the regex for the topics ('%s=%s') consumed by the connector",
                            deadLetterQueueTopic, TOPIC_REGEX_CONFIG, topicRegex
                    );
                    addWarning(validatedConfig, TOPIC_REGEX_CONFIG, topicRegex, warningMessage);
                }
            }
        }
    }

  @Override
public void processVarInsn(final int instType, final int localIndex) {
    lastBytecodeOffset = code.length;
    // Add the instruction to the bytecode of the method.
    if (localIndex < 4 && instType != Opcodes.RET) {
        int optimizedOpcode;
        if (instType < Opcodes.ISTORE) {
            optimizedOpcode = Constants.ILOAD_0 + ((instType - Opcodes.ILOAD) << 2) + localIndex;
        } else {
            optimizedOpcode = Constants.ISTORE_0 + ((instType - Opcodes.ISTORE) << 2) + localIndex;
        }
        code.putByte(optimizedOpcode);
    } else if (localIndex >= 256) {
        code.putByte(Constants.WIDE).put12(instType, localIndex);
    } else {
        code.put11(instType, localIndex);
    }

    // If needed, update the maximum stack size and number of locals, and stack map frames.
    if (currentBasicBlock != null) {
        if ((compute == COMPUTE_ALL_FRAMES || compute == COMPUTE_INSERTED_FRAMES)) {
            currentBasicBlock.frame.execute(instType, localIndex, null, null);
        } else {
            if (instType == Opcodes.RET) {
                // No stack size delta.
                currentBasicBlock.flags |= Label.FLAG_SUBROUTINE_END;
                currentBasicBlock.outputStackSize = (short) relativeStackSize;
                endCurrentBasicBlockWithNoSuccessor();
            } else { // xLOAD or xSTORE
                int size = relativeStackSize + STACK_SIZE_DELTA[instType];
                if (size > maxRelativeStackSize) {
                    maxRelativeStackSize = size;
                }
                relativeStackSize = size;
            }
        }
    }

    if (compute != COMPUTE_NOTHING) {
        int currentMaxLocals;
        if (instType == Opcodes.LLOAD
            || instType == Opcodes.DLOAD
            || instType == Opcodes.LSTORE
            || instType == Opcodes.DSTORE) {
            currentMaxLocals = localIndex + 2;
        } else {
            currentMaxLocals = localIndex + 1;
        }
        if (currentMaxLocals > maxLocals) {
            maxLocals = currentMaxLocals;
        }
    }

    // If there are exception handler blocks, each instruction within a handler range is, in
    // theory, a basic block.
    if (opcode >= Opcodes.ISTORE && compute == COMPUTE_ALL_FRAMES && firstHandler != null) {
        visitLabel(new Label());
    }
}

  @Override
  private void checkAsyncCall() throws IOException {
    if (isAsynchronousMode()) {
      if (asyncCallCounter.incrementAndGet() > maxAsyncCalls) {
        asyncCallCounter.decrementAndGet();
        String errMsg = String.format(
            "Exceeded limit of max asynchronous calls: %d, " +
            "please configure %s to adjust it.",
            maxAsyncCalls,
            CommonConfigurationKeys.IPC_CLIENT_ASYNC_CALLS_MAX_KEY);
        throw new AsyncCallLimitExceededException(errMsg);
      }
    }
  }

  @Override
protected void initService(Configuration config) throws Exception {
    int maxThreads = config.getInt(MAX_SHUFFLE_THREADS,
                                    DEFAULT_MAX_SHUFFLE_THREADS);

    if (maxThreads == 0) {
      maxThreads = Runtime.getRuntime().availableProcessors() * 2;
    }

    ThreadFactoryBuilder workerBuilderFactory = new ThreadFactoryBuilder()
        .setNameFormat("ShuffleHandler Netty Worker #%d")
        .setDaemon(true);

    ThreadFactory bossFactory = new ThreadFactoryBuilder()
        .setNameFormat("ShuffleHandler Netty Boss #%d")
        .build();

    int workerThreads = maxThreads;
    ThreadFactory workerFactory = workerBuilderFactory.setPriority(Thread.MIN_PRIORITY).build();

    NioEventLoopGroup bossGroup = new NioEventLoopGroup(1, bossFactory);
    NioEventLoopGroup workerGroup = new NioEventLoopGroup(workerThreads, workerFactory);

    super.serviceInit(new Configuration(config));
  }

  @Override
  public Path getFullPath() {
    String parentFullPathStr =
        (parentFullPath == null || parentFullPath.length == 0) ?
            null : DFSUtilClient.bytes2String(parentFullPath);
    if (parentFullPathStr == null
        && dirStatus.getLocalNameInBytes().length == 0) {
      // root
      return new Path("/");
    } else {
      return parentFullPathStr == null ? new Path(dirStatus.getLocalName())
          : new Path(parentFullPathStr, dirStatus.getLocalName());
    }
  }

  @Override
public void logOutput(OutputStream out) throws IOException {
		if (isModified()) {
			return;
		}

		if (!formatPreferences.useDelombokComments()) {
			out.write("// Generated by delombok at ");
			out.write(String.valueOf(new Date()));
			out.write(System.getProperty("line.separator"));
		}

		List<CommentInfo> comments_ = convertComments((List<? extends CommentInfo>) comments);
		int[] textBlockStarts_ = convertTextBlockStarts(textBlockStarts);
		FormatPreferences preferences = new FormatPreferenceScanner().scan(formatPreferences, getContent());
		compilationUnit.accept(new PrettyPrinter(out, compilationUnit, comments_, textBlockStarts_, preferences));
	}

	private List<CommentInfo> convertComments(List<? extends CommentInfo> comments) {
		return (comments instanceof com.sun.tools.javac.util.List) ? (List<CommentInfo>) comments : com.sun.tools.javac.util.List.from(comments.toArray(new CommentInfo[0]));
	}

	private int[] convertTextBlockStarts(Set<Integer> textBlockStarts) {
		int[] result = new int[textBlockStarts.size()];
		int idx = 0;
		for (int tbs : textBlockStarts) result[idx++] = tbs;
		return result;
	}

  @Override
public <K> K executeAs(User user, Callable<K> callable) throws CompletionException {
    try {
        return performAs(user, callable::call);
    } catch (PrivilegedActionException e) {
        throw new CompletionException(e.getCause());
    }
}

  @Override
	public Validator mvcValidator() {
		Validator validator = getValidator();
		if (validator == null) {
			if (ClassUtils.isPresent("jakarta.validation.Validator", getClass().getClassLoader())) {
				try {
					validator = new OptionalValidatorFactoryBean();
				}
				catch (Throwable ex) {
					throw new BeanInitializationException("Failed to create default validator", ex);
				}
			}
			else {
				validator = new NoOpValidator();
			}
		}
		return validator;
	}

  @Override
  CSQueue getByFullName(String fullName) {
    if (fullName == null) {
      return null;
    }

    try {
      modificationLock.readLock().lock();
      return fullNameQueues.getOrDefault(fullName, null);
    } finally {
      modificationLock.readLock().unlock();
    }
  }


  @Override
  public void updateAttempt(TaskAttemptStatus status, long timestamp) {
    super.updateAttempt(status, timestamp);
    TaskAttemptId attemptID = status.id;

    float progress = status.progress;

    incorporateReading(attemptID, progress, timestamp);
  }

  @Override
	private void addPropertyAuditingOverrides(MemberDetails memberDetails, PropertyAuditingData propertyData) {
		final AuditOverride annotationOverride = memberDetails.getDirectAnnotationUsage( AuditOverride.class );
		if ( annotationOverride != null ) {
			propertyData.addAuditingOverride( annotationOverride );
		}
		final AuditOverrides annotationOverrides = memberDetails.getDirectAnnotationUsage( AuditOverrides.class );
		if ( annotationOverrides != null ) {
			propertyData.addAuditingOverrides( annotationOverrides );
		}
	}

  private static String RM_APP_ATTEMPT_PREFIX = "RMATTEMPT_";
private boolean checkNextCondition() {
        boolean hasMore = false;
        try {
            hasMore = currentIterator.hasNext();
        } catch (final InvalidStateStoreException e) {
            // already closed so ignore
        }
        return !hasMore ^ hasNextCondition.hasNext(currentIterator);
    }
private <E> E createInstance(Class<E> clazz, PersistentClass pc, AuditTableData atd) {
		try {
			var constructor = clazz.getDeclaredConstructor(AuditTableData.class, PersistentClass.class);
			return constructor.newInstance(atd, pc);
		} catch (Exception e) {
			throw new EnversMappingException("Cannot create entity of type " + clazz.getName());
		}
	}

public static TokenData fromRecord(TokenRecord record) {
        List<KafkaPrincipal> validators = new ArrayList<>();
        for (String validatorString : record.validators()) {
            validators.add(SecurityUtils.parseKafkaPrincipal(validatorString));
        }
        return new TokenData(TokenInformation.fromRecord(
            record.tokenNumber(),
            SecurityUtils.parseKafkaPrincipal(record.requester()),
            SecurityUtils.parseKafkaPrincipal(record.user()),
            validators,
            record.issueTime(),
            record.maxTime(),
            record.expiryTime()));
    }
public static void addStandardFormatters(FormatterRegistry formatterRegistry) {
		// Standard handling of numeric values
		formatterRegistry.addFormatterForFieldAnnotation(new NumericFormatAnnotationFormatterFactory());

		// Standard handling of monetary values
		if (jsr354Present) {
			formatterRegistry.addFormatter(new CurrencyUnitFormatter());
			formatterRegistry.addFormatter(new MonetaryAmountFormatter());
			formatterRegistry.addFormatterForFieldAnnotation(new Jsr354NumericFormatAnnotationFormatterFactory());
		}

		// Standard handling of date-time values

		// just handling JSR-310 specific date and time types
		new DateTimeFormatterRegistrar().registerFormatters(formatterRegistry);

		// regular DateFormat-based Date, Calendar, Long converters
		new DateFormatterRegistrar().registerFormatters(formatterRegistry);
	}

  @Override
  public void flush() throws TTransportException {
    // Extract request and reset buffer
    byte[] data = requestBuffer_.toByteArray();
    requestBuffer_.reset();

    try {
      // Create connection object
      connection = (HttpConnection)Connector.open(url_);

      // Make the request
      connection.setRequestMethod("POST");
      connection.setRequestProperty("Content-Type", "application/x-thrift");
      connection.setRequestProperty("Accept", "application/x-thrift");
      connection.setRequestProperty("User-Agent", "JavaME/THttpClient");

      connection.setRequestProperty("Connection", "Keep-Alive");
      connection.setRequestProperty("Keep-Alive", "5000");
      connection.setRequestProperty("Http-version", "HTTP/1.1");
      connection.setRequestProperty("Cache-Control", "no-transform");

      if (customHeaders_ != null) {
        for (Enumeration e = customHeaders_.keys() ; e.hasMoreElements() ;) {
          String key = (String)e.nextElement();
          String value = (String)customHeaders_.get(key);
          connection.setRequestProperty(key, value);
        }
      }

      OutputStream os = connection.openOutputStream();
      os.write(data);
      os.close();

      int responseCode = connection.getResponseCode();
      if (responseCode != HttpConnection.HTTP_OK) {
        throw new TTransportException("HTTP Response code: " + responseCode);
      }

      // Read the responses
      inputStream_ = connection.openInputStream();
    } catch (IOException iox) {
      System.out.println(iox.toString());
      throw new TTransportException(iox);
    }
  }

  @Override
public UserGroupInformation getUGIValue(final HttpEnvironment context) {
    Configuration conf = (Configuration) servletContext.getAttribute(JspHelper.CURRENT_CONF);
    try {
        boolean isKerberos = true;
        return JspHelper.getUGI(servletContext, request, conf, isKerberos, false);
    } catch (IOException e) {
        throw new SecurityException(SecurityUtil.FAILED_TO_GET_UGI_MSG_HEADER + " " + e, e);
    }
}

  private static ByteBuffer convertCredentialsToByteBuffer(
      Credentials credentials) {
    ByteBuffer appAttemptTokens = null;
    DataOutputBuffer dob = new DataOutputBuffer();
    try {
      if (credentials != null) {
        credentials.writeTokenStorageToStream(dob);
        appAttemptTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());
      }
      return appAttemptTokens;
    } catch (IOException e) {
      LOG.error("Failed to convert Credentials to ByteBuffer.");
      assert false;
      return null;
    } finally {
      IOUtils.closeStream(dob);
    }
  }

  private static Credentials convertCredentialsFromByteBuffer(
      ByteBuffer appAttemptTokens) {
    DataInputByteBuffer dibb = new DataInputByteBuffer();
    try {
      Credentials credentials = null;
      if (appAttemptTokens != null) {
        credentials = new Credentials();
        appAttemptTokens.rewind();
        dibb.reset(appAttemptTokens);
        credentials.readTokenStorageStream(dibb);
      }
      return credentials;
    } catch (IOException e) {
      LOG.error("Failed to convert Credentials from ByteBuffer.");
      assert false;
      return null;
    } finally {
      IOUtils.closeStream(dibb);
    }
  }

  @Override
	public <X> ValueExtractor<X> getExtractor(JavaType<X> javaType) {
		return new BasicExtractor<>( javaType, this ) {
			@Override
			protected X doExtract(ResultSet rs, int paramIndex, WrapperOptions options) throws SQLException {
				final XmlAsStringArrayJdbcType jdbcType = (XmlAsStringArrayJdbcType) getJdbcType();
				final String value;
				if ( jdbcType.nationalized && options.getDialect().supportsNationalizedMethods() ) {
					value = rs.getNString( paramIndex );
				}
				else {
					value = rs.getString( paramIndex );
				}
				return jdbcType.fromString( value, getJavaType(), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, int index, WrapperOptions options)
					throws SQLException {
				final XmlAsStringArrayJdbcType jdbcType = (XmlAsStringArrayJdbcType) getJdbcType();
				final String value;
				if ( jdbcType.nationalized && options.getDialect().supportsNationalizedMethods() ) {
					value = statement.getNString( index );
				}
				else {
					value = statement.getString( index );
				}
				return jdbcType.fromString( value, getJavaType(), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, String name, WrapperOptions options)
					throws SQLException {
				final XmlAsStringArrayJdbcType jdbcType = (XmlAsStringArrayJdbcType) getJdbcType();
				final String value;
				if ( jdbcType.nationalized && options.getDialect().supportsNationalizedMethods() ) {
					value = statement.getNString( name );
				}
				else {
					value = statement.getString( name );
				}
				return jdbcType.fromString( value, getJavaType(), options );
			}

		};
	}

  @Override
  public int getHeaderFlag(int i) {
    if (proto.getFlagCount() > 0) {
      return proto.getFlag(i);
    } else {
      return combineHeader(ECN.DISABLED, proto.getReply(i), SLOW.DISABLED);
    }
  }

  @Override
static TopicsDelta buildInitialTopicsDelta(int totalTopicCount, int partitionsPerTopic, int replicationFactor, int numReplicasPerBroker) {
    int brokers = getNumBrokers(totalTopicCount, partitionsPerTopic, replicationFactor, numReplicasPerBroker);
    TopicsDelta topicsDelta = new TopicsDelta(TopicsImage.EMPTY);
    final AtomicInteger leaderId = new AtomicInteger(0);

    for (int topicIndex = 0; topicIndex < totalTopicCount; topicIndex++) {
        Uuid topicUuid = Uuid.randomUuid();
        topicsDelta.replay(new TopicRecord().setName("topic" + topicIndex).setTopicId(topicUuid));

        for (int partitionIndex = 0; partitionIndex < partitionsPerTopic; partitionIndex++) {
            List<Integer> replicaList = getReplicas(totalTopicCount, partitionsPerTopic, replicationFactor, numReplicasPerBroker, leaderId.get());
            List<Integer> inSyncReplicaSet = new ArrayList<>(replicaList);
            topicsDelta.replay(new PartitionRecord()
                .setPartitionId(partitionIndex)
                .setTopicId(topicUuid)
                .setReplicas(replicaList)
                .setIsr(inSyncReplicaSet)
                .setRemovingReplicas(Collections.emptyList())
                .setAddingReplicas(Collections.emptyList())
                .setLeader(leaderId.get()));
            leaderId.set((1 + leaderId.get()) % brokers);
        }
    }

    return topicsDelta;
}

  @Override
  public void setPreemptedResourceSecondsMap(
      Map<String, Long> preemptedResourceSecondsMap) {
    maybeInitBuilder();
    builder.clearPreemptedResourceUsageMap();
    this.preemptedResourceSecondsMap = preemptedResourceSecondsMap;
    if (preemptedResourceSecondsMap != null) {
      builder.addAllPreemptedResourceUsageMap(ProtoUtils
          .convertMapToStringLongMapProtoList(preemptedResourceSecondsMap));
    }
  }

  @Override
public static <T> SqmSelectStatement<T>[] divide(SqmSelectStatement<T> query) {
		// We only allow unmapped polymorphism in a very restricted way.  Specifically,
		// the unmapped polymorphic reference can only be a root and can be the only
		// root.  Use that restriction to locate the unmapped polymorphic reference
		final SqmRoot<?> ref = findUnmappedPolymorphicReference(query.getQueryPart());

		if ( ref == null ) {
			@SuppressWarnings("unchecked")
			SqmSelectStatement<T>[] stmts = new SqmSelectStatement[] { query };
			return stmts;
		}

		final SqmPolymorphicRootDescriptor<T> descriptor = (SqmPolymorphicRootDescriptor<T>) ref.getReferencedPathSource();
		final Set<EntityDomainType<? extends T>> implementors = descriptor.getImplementors();
		@SuppressWarnings("unchecked")
		final SqmSelectStatement<T>[] expanded = new SqmSelectStatement[implementors.size()];

		int i = 0;
		for ( EntityDomainType<?> mappedDesc : implementors ) {
			expanded[i++] = copyQuery(query, ref, mappedDesc);
		}

		return expanded;
	}

  @Override
public void setSocketTimeoutPeriod(int timeout) {
    SocketChannel socketChannel = socketChannel_;
    if (socketChannel != null) {
        try {
            socketChannel.socket().setSoTimeout(timeout);
        } catch (SocketException e) {
            LOGGER.warn("Failed to set the socket's timeout period.", e);
        }
    }
}
}
