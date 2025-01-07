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

