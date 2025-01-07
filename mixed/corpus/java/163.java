protected TimelineEntity parseTimelineResult(Result timelineResult) throws IOException {
    FlowActivityRowKey rowKey = FlowActivityRowKey.parseRowKey(timelineResult.getRow());

    String userId = rowKey.getUserId();
    Long timestamp = rowKey.getDayTimestamp();
    String flowName = rowKey.getFlowName();

    FlowActivityEntity activityEntity = new FlowActivityEntity(
        getContext().getClusterId(), timestamp, userId, flowName);

    Map<Long, Object> runIdsMap = ColumnRWHelper.readResults(timelineResult,
        FlowActivityColumnPrefix.RUN_ID, longKeyConverter);

    for (var entry : runIdsMap.entrySet()) {
      Long runId = entry.getKey();
      String version = (String)entry.getValue();

      FlowRunEntity runEntity = new FlowRunEntity();
      runEntity.setUser(userId);
      runEntity.setName(flowName);
      runEntity.setRunId(runId);
      runEntity.setVersion(version);

      activityEntity.addFlowRun(runEntity);
    }

    activityEntity.getInfo().put(TimelineReaderUtils.FROMID_KEY,
        rowKey.getRowKeyAsString());

    return activityEntity;
}

	public QueryParameterImplementor<?> getQueryParameter(String name) {
		final QueryParameterImplementor<?> parameter = findQueryParameter( name );
		if ( parameter != null ) {
			return parameter;
		}
		else {
			final String errorMessage = String.format(
					Locale.ROOT,
					"No parameter named ':%s' in query with named parameters [%s]",
					name,
					String.join( ", ", getNamedParameterNames() )
			);
			throw new IllegalArgumentException(
					errorMessage,
					new UnknownParameterException( errorMessage )
			);
		}
	}

protected void initializeSequence(Database db) {
		int incrementSize = this.getSourceIncrementSize();

		Namespace ns = db.locateNamespace(
				logicalQualifiedSequenceName.getCatalogName(),
				logicalQualifiedSequenceName.getSchemaName()
		);
		Sequence seq = ns.locateSequence(logicalQualifiedSequenceName.getObjectName());
		if (seq == null) {
			seq = ns.createSequence(
					logicalQualifiedSequenceName.getObjectName(),
					physicalName -> new Sequence(
							contributor,
							ns.getPhysicalName().getCatalog(),
							ns.getPhysicalName().getSchema(),
							physicalName,
							initialValue,
							incrementSize,
							options
					)
			);
		} else {
			seq.validate(initialValue, incrementSize);
		}

		physicalSequenceName = seq.getName();
	}

