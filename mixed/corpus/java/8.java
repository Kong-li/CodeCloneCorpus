private void setupDataCache() {
    DataDelta accumulateDataDelta = new DataDelta(DataImage.EMPTY);
    IntStream.range(0, 7).forEach(agentId -> {
        RegisterAgentRecord.AgentEndpointCollection endpoints = new RegisterAgentRecord.AgentEndpointCollection();
        endpoints(agentId).forEach(endpoint ->
            endpoints.add(new RegisterAgentRecord.AgentEndpoint().
                setHost(endpoint.host()).
                setPort(endpoint.port()).
                setName(endpoint.listener()).
                setSecurityProtocol(endpoint.securityProtocol())));
        accumulateDataDelta.replay(new RegisterAgentRecord().
            setAgentId(agentId).
            setAgentEpoch(200L).
            setFenced(false).
            setRack(null).
            setEndPoints(endpoints).
            setIncarnationId(Uuid.fromString(Uuid.randomUUID().toString())));
    });
    IntStream.range(0, topicCount).forEach(topicNum -> {
        Uuid subjectId = Uuid.randomUUID();
        accumulateDataDelta.replay(new TopicRecord().setName("topic-" + topicNum).setTopicId(subjectId));
        IntStream.range(0, partitionCount).forEach(partitionId ->
            accumulateDataDelta.replay(new PartitionRecord().
                setPartitionId(partitionId).
                setTopicId(subjectId).
                setReplicas(Arrays.asList(1, 2, 4)).
                setIsr(Arrays.asList(1, 2, 4)).
                setRemovingReplicas(Collections.emptyList()).
                setAddingReplicas(Collections.emptyList()).
                setLeader(partitionCount % 7).
                setLeaderEpoch(0)));
    });
    dataCache.setImage(accumulateDataDelta.apply(DataProvenance.EMPTY));
}

	public List<R> performList(DomainQueryExecutionContext executionContext) {
		final QueryOptions queryOptions = executionContext.getQueryOptions();
		if ( queryOptions.getEffectiveLimit().getMaxRowsJpa() == 0 ) {
			return Collections.emptyList();
		}
		final List<JdbcParameterBinder> jdbcParameterBinders;
		final JdbcParameterBindings jdbcParameterBindings;

		final QueryParameterBindings queryParameterBindings = executionContext.getQueryParameterBindings();
		if ( parameterList == null || parameterList.isEmpty() ) {
			jdbcParameterBinders = Collections.emptyList();
			jdbcParameterBindings = JdbcParameterBindings.NO_BINDINGS;
		}
		else {
			jdbcParameterBinders = new ArrayList<>( parameterList.size() );
			jdbcParameterBindings = new JdbcParameterBindingsImpl(
					queryParameterBindings,
					parameterList,
					jdbcParameterBinders,
					executionContext.getSession().getFactory()
			);
		}

		final JdbcOperationQuerySelect jdbcSelect = new JdbcOperationQuerySelect(
				sql,
				jdbcParameterBinders,
				resultSetMapping,
				affectedTableNames
		);

		executionContext.getSession().autoFlushIfRequired( jdbcSelect.getAffectedTableNames() );
		return executionContext.getSession().getJdbcServices().getJdbcSelectExecutor().list(
				jdbcSelect,
				jdbcParameterBindings,
				SqmJdbcExecutionContextAdapter.usingLockingAndPaging( executionContext ),
				null,
				queryOptions.getUniqueSemantic() == null ?
						ListResultsConsumer.UniqueSemantic.NEVER :
						queryOptions.getUniqueSemantic()
		);
	}

void construct(BufferBuilder bb, HashMap<String, AnyValue> queryValues) {
		final MutableBoolean isFirst = new MutableBoolean( true );

		for ( String clause : clauses ) {
			addSegment( bb, clause, isFirst );
		}

		for ( Rules rule : subRules ) {
			if ( !subRules.isEmpty() ) {
				appendBracket( bb, "(", isFirst );
				rule.construct( bb, queryValues );
				bb.append( ")" );
			}
		}

		for ( Rules negatedRule : negatedRules ) {
			if ( !negatedRules.isEmpty() ) {
				appendBracket( bb, "not (", isFirst );
				negatedRule.construct( bb, queryValues );
				bb.append( ")" );
			}
		}

		queryValues.putAll( localQueryValues );
	}

	void appendBracket(BufferBuilder bb, String prefix, MutableBoolean isFirst) {
		if (isFirst.value) {
			isFirst.setValue(false);
		} else {
			bb.append(", ");
		}
		bb.append(prefix);
	}

	void addSegment(BufferBuilder bb, String segment, MutableBoolean isFirst) {
		if (isFirst.value) {
			isFirst.setValue(false);
		} else {
			bb.append(" AND ");
		}
		bb.append(segment);
	}

public synchronized HAServiceStatus fetchHAState() throws IOException {
    String methodName = "fetchHAState";
    checkAccess(methodName);
    HAServiceState haState = rm.getRMContext().getHAServiceState();
    HAServiceStatus result = new HAServiceStatus(haState);

    if (!isRMActive() && haState != HAServiceProtocol.HAServiceState.STANDBY) {
        result.setNotReadyToBecomeActive(String.format("State is %s", haState));
    } else {
        result.setReadyToBecomeActive();
    }

    return result;
}

	public static int fallbackAllocationSize(Annotation generatorAnnotation, MetadataBuildingContext buildingContext) {
		if ( generatorAnnotation == null ) {
			final ConfigurationService configService = buildingContext.getBootstrapContext()
					.getServiceRegistry().requireService( ConfigurationService.class );
			final String idNamingStrategy = configService.getSetting( ID_DB_STRUCTURE_NAMING_STRATEGY, StandardConverters.STRING );
			if ( LegacyNamingStrategy.STRATEGY_NAME.equals( idNamingStrategy )
					|| LegacyNamingStrategy.class.getName().equals( idNamingStrategy )
					|| SingleNamingStrategy.STRATEGY_NAME.equals( idNamingStrategy )
					|| SingleNamingStrategy.class.getName().equals( idNamingStrategy ) ) {
				return 1;
			}
		}

		return OptimizableGenerator.DEFAULT_INCREMENT_SIZE;
	}

