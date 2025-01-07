    public static void main(String[] args) {
        try {
            if (args.length != 3) {
                Utils.printHelp("This example takes 3 parameters (i.e. 6 3 10000):%n" +
                    "- partition: number of partitions for input and output topics (required)%n" +
                    "- instances: number of application instances (required)%n" +
                    "- records: total number of records (required)");
                return;
            }

            int numPartitions = Integer.parseInt(args[0]);
            int numInstances = Integer.parseInt(args[1]);
            int numRecords = Integer.parseInt(args[2]);

            // stage 1: clean any topics left from previous runs
            Utils.recreateTopics(KafkaProperties.BOOTSTRAP_SERVERS, numPartitions, INPUT_TOPIC, OUTPUT_TOPIC);

            // stage 2: send demo records to the input-topic
            CountDownLatch producerLatch = new CountDownLatch(1);
            Producer producerThread = new Producer(
                    "producer",
                    KafkaProperties.BOOTSTRAP_SERVERS,
                    INPUT_TOPIC,
                    false,
                    null,
                    true,
                    numRecords,
                    -1,
                    producerLatch);
            producerThread.start();
            if (!producerLatch.await(2, TimeUnit.MINUTES)) {
                Utils.printErr("Timeout after 2 minutes waiting for data load");
                producerThread.shutdown();
                return;
            }

            // stage 3: read from input-topic, process once and write to the output-topic
            CountDownLatch processorsLatch = new CountDownLatch(numInstances);
            List<ExactlyOnceMessageProcessor> processors = IntStream.range(0, numInstances)
                .mapToObj(id -> new ExactlyOnceMessageProcessor(
                        "processor-" + id,
                        KafkaProperties.BOOTSTRAP_SERVERS,
                        INPUT_TOPIC,
                        OUTPUT_TOPIC,
                        processorsLatch))
                .collect(Collectors.toList());
            processors.forEach(ExactlyOnceMessageProcessor::start);
            if (!processorsLatch.await(2, TimeUnit.MINUTES)) {
                Utils.printErr("Timeout after 2 minutes waiting for record copy");
                processors.forEach(ExactlyOnceMessageProcessor::shutdown);
                return;
            }

            // stage 4: check consuming records from the output-topic
            CountDownLatch consumerLatch = new CountDownLatch(1);
            Consumer consumerThread = new Consumer(
                    "consumer",
                    KafkaProperties.BOOTSTRAP_SERVERS,
                    OUTPUT_TOPIC,
                    GROUP_NAME,
                    Optional.empty(),
                    true,
                    numRecords,
                    consumerLatch);
            consumerThread.start();
            if (!consumerLatch.await(2, TimeUnit.MINUTES)) {
                Utils.printErr("Timeout after 2 minutes waiting for output read");
                consumerThread.shutdown();
            }
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

public int computeHash() {
    final int prime = 31;
    int hashValue = 1;
    if (containerID != null) {
        hashValue = prime * hashValue + containerID.hashCode();
    }
    if (containerMgrAddress != null) {
        hashValue = prime * hashValue + containerMgrAddress.hashCode();
    }
    if (containerToken != null) {
        hashValue = prime * hashValue + containerToken.hashCode();
    }
    if (taskAttemptID != null) {
        hashValue = prime * hashValue + taskAttemptID.hashCode();
    }
    boolean shouldDumpThreads = dumpContainerThreads;
    if (shouldDumpThreads) {
        hashValue++;
    }
    return hashValue;
}

public SqmTreatedSetJoin<O, T, S> duplicate(SqmCopyContext context) {
		final SqmTreatedSetJoin<O, T, S> existing = context.getDuplicate( this );
		if (existing != null) {
			return existing;
		}
		SqmTreatedSetJoin<O, T, S> path = new SqmTreatedSetJoin<>(
				getNavigablePath(),
				wrappedPath.duplicate(context),
				treatTarget,
				getExplicitAlias(),
				isFetched()
		);
		context.registerDuplicate(this, path);
		copyTo(path, context);
		return path;
	}

private void anonymize(StatePool statePool, Config config) {
    FileNameState fState = (FileNameState) statePool.getState(getClass());
    if (fState == null) {
      fState = new FileNameState();
      statePool.addState(getClass(), fState);
    }

    String[] files = StringUtils.split(filePath);
    String[] anonymizedFileNames = new String[files.length];
    int index = 0;
    for (String file : files) {
      anonymizedFileNames[index++] =
        anonymize(statePool, config, fState, file);
    }

    anonymizedFilePath = StringUtils.arrayToString(anonymizedFileNames);
  }

