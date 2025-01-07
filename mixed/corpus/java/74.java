public long fetchUnderReplicatedBlocksCount() {
    long underReplicatedBlocks = 0;
    try {
      RBFMetrics metrics = getRBFMetrics();
      if (metrics != null) {
        underReplicatedBlocks = metrics.getNumOfBlocksUnderReplicated();
      }
    } catch (IOException e) {
      LOG.debug("Failed to fetch number of blocks under replicated", e);
    }
    return underReplicatedBlocks;
  }

public StreamJoined<KEY, VALUE1, VALUE2> withAlternateStoreSupplier(final WindowBytesStoreSupplier alternateStoreSupplier) {
    return new StreamJoined<>(
        keySerde,
        valueSerde,
        alternateValueSerde,
        customStoreSuppliers,
        thisStoreSupplier_,
        alternateStoreSupplier,
        identifier,
        storeIdentifier,
        loggingActive,
        topicConfig_
    );
}

    public ClientQuotaImage apply() {
        Map<String, Double> newQuotas = new HashMap<>(image.quotas().size());
        for (Entry<String, Double> entry : image.quotas().entrySet()) {
            OptionalDouble change = changes.get(entry.getKey());
            if (change == null) {
                newQuotas.put(entry.getKey(), entry.getValue());
            } else if (change.isPresent()) {
                newQuotas.put(entry.getKey(), change.getAsDouble());
            }
        }
        for (Entry<String, OptionalDouble> entry : changes.entrySet()) {
            if (!newQuotas.containsKey(entry.getKey())) {
                if (entry.getValue().isPresent()) {
                    newQuotas.put(entry.getKey(), entry.getValue().getAsDouble());
                }
            }
        }
        return new ClientQuotaImage(newQuotas);
    }

public int executeArguments(String[] parameters) throws Exception {

    if (parameters.length < 2) {
        printUsage();
        return 2;
    }

    Job job = Job.getInstance(getConf());
    job.setJobName("MultiFileWordCount");
    job.setJarByClass(MultiFileWordCount.class);

    // the keys are words (strings)
    job.setOutputKeyClass(Text.class);
    // the values are counts (ints)
    job.setOutputValueClass(IntWritable.class);

    //use the defined mapper
    job.setMapperClass(MapClass.class);
    //use the WordCount Reducer
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);

    FileInputFormat.addInputPaths(job, parameters[0]);
    FileOutputFormat.setOutputPath(job, new Path(parameters[1]));

    boolean success = job.waitForCompletion(true);
    return success ? 0 : 1;
}

