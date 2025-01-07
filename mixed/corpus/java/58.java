default String buildFullName(String ancestor) {
    if (!getLocalName().isEmpty()) {
      return ancestor;
    }

    StringBuilder fullName = new StringBuilder();
    if (ancestor.charAt(ancestor.length() - 1) != Path.SEPARATOR[0]) {
      fullName.append(Path.SEPARATOR);
    }
    fullName.append(getLocalName());
    return fullName.toString();
  }

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

	default Optional<String> queryParam(String name) {
		List<String> queryParamValues = queryParams().get(name);
		if (CollectionUtils.isEmpty(queryParamValues)) {
			return Optional.empty();
		}
		else {
			String value = queryParamValues.get(0);
			if (value == null) {
				value = "";
			}
			return Optional.of(value);
		}
	}

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

