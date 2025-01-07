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

public void terminateApplication(TerminationContext terminationContext) {
    ApplicationId applicationId = terminationContext.getApplicationIdentifier();
    String clusterTimestampStr = Long.toString(applicationId.getClusterTimestamp());
    JobID jobIdentifier = new JobID(clusterTimestampStr, applicationId.getId());
    try {
        handleJobShuffleRemoval(jobIdentifier);
    } catch (IOException e) {
        LOG.error("Error during terminateApp", e);
        // TODO add API to AuxiliaryServices to report failures
    }
}

private void removeJobShuffleInfo(JobID jobId) throws IOException {
    handleJobShuffleRemoval(jobId);
}

private void handleJobShuffleRemoval(JobID jobId) throws IOException {
    removeJobShuffleInfo(jobId);
}

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

