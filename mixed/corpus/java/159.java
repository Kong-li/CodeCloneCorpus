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

