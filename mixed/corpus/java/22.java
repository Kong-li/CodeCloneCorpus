private void processReadBlock(DataInputStream in) throws IOException {
    final ProcessReadBlockProto proto = ProcessReadBlockProto.parseFrom(vintPrefixed(in));
    final DatanodeInfo[] destinations = PBHelperClient.convert(proto.getDestinationsList());
    TraceScope traceSpan = continueTraceSegment(proto.getHeader(),
        proto.getClass().getSimpleName());
    try {
      readBlock(PBHelperClient.convert(proto.getHeader().getBaseHeader().getBlock()),
          PBHelperClient.convertStorageType(proto.getStorageType()),
          PBHelperClient.convert(proto.getHeader().getBaseHeader().getToken()),
          proto.getHeader().getClientName(),
          destinations,
          PBHelperClient.convertStorageTypes(proto.getTargetStorageTypesList(), destinations.length),
          PBHelperClient.convert(proto.getSource()),
          fromProto(proto.getStage()),
          proto.getPipelineSize(),
          proto.getMinBytesRcvd(), proto.getMaxBytesRcvd(),
          proto.getLatestGenerationStamp(),
          fromProto(proto.getRequestedChecksum()),
          (proto.hasCachingStrategy() ?
              getCachingStrategy(proto.getCachingStrategy()) :
            CachingStrategy.newDefaultStrategy()),
          (proto.hasAllowLazyPersist() ? proto.getAllowLazyPersist() : false),
          (proto.hasPinning() ? proto.getPinning(): false),
          (PBHelperClient.convertBooleanList(proto.getTargetPinningsList())),
          proto.getStorageId(),
          proto.getTargetStorageIdsList().toArray(new String[0]));
    } finally {
     if (traceSpan != null) traceSpan.close();
    }
  }

public QualifiedTableName wrapIdentifier() {
		var catalogName = getCatalogName();
		if (catalogName == null) {
			return new QualifiedTableName(null, null, null);
		}
		catalogName = new Identifier(catalogName.getText(), true);

		var schemaName = getSchemaName();
		if (schemaName != null) {
			schemaName = new Identifier(schemaName.getText(), false);
		}

		var tableName = getTableName();
		if (tableName != null && !tableName.isEmpty()) {
			tableName = new Identifier(tableName.getText(), true);
		}

		return new QualifiedTableName(catalogName, schemaName, tableName);
	}

public void executeDatabaseFilters(DataResultInitializationContext initializationContext) {
		final QueryAstInitializationState queryAstInitializationState = initializationContext.getQueryAstInitializationState();
		final ExpressionEvaluator expressionEvaluator = queryAstInitializationState.getExpressionEvaluator();

		expressionEvaluator.resolveDatabaseFilter(
				this,
				column.getTypeMapping().getDatabaseType(),
				null,
				queryAstInitializationState_INITIALIZATION_CONTEXT.getMetadataModel().getColumnConfiguration()
		);
	}

public void updateMetrics(Set<TimelineMetric> metricsInput) {
    if (null == real) {
        this.metrics = metricsInput;
    } else {
        setEntityMetrics(metricsInput);
    }
}

private void setEntityMetrics(Set<TimelineMetric> entityMetrics) {
    real.setMetrics(entityMetrics);
}

