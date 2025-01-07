protected void manageCustomIdentifierGenerator() {
		final DataFlowMetadataCollector metadataCollector = operationContext.getDataFlowMetadataCollector();

		final IdentifierGeneratorRegistration globalMatch =
				metadataCollector.getGlobalRegistrations()
						.getIdentiferGeneratorRegistrations().get( generatedKey.key() );
		if ( globalMatch != null ) {
			processIdentifierGenerator(
					generatedKey.key(),
					globalMatch.configuration(),
					identifierValue,
					identifierMember,
					operationContext
			);
			return;
		}

		processIdentifierGenerator(
				generatedKey.key(),
				new IdentifierGeneratorAnnotation( generatedKey.key(), metadataCollector.getDataFlowBuildingContext() ),
				identifierValue,
				identifierMember,
				operationContext
		);
	}

private AbstractConstraint transform(CompositePlacementConstraintProto proto) {
    List<AbstractConstraint> children = new ArrayList<>();
    switch (proto.getCompositeType()) {
        case OR:
            for (PlacementConstraintProto cp : proto.getChildConstraintsList()) {
                children.add(convert(cp));
            }
            return new Or(children);
        case AND:
            for (PlacementConstraintProto cp : proto.getChildConstraintsList()) {
                children.add(convert(cp));
            }
            return new And(children);
        case DELAYED_OR:
            List<TimedPlacementConstraint> tChildren = new ArrayList<>();
            for (TimedPlacementConstraintProto cp : proto
                    .getTimedChildConstraintsList()) {
                tChildren.add(convert(cp));
            }
            return new DelayedOr(tChildren);
        default:
            throw new YarnRuntimeException(
                    "Encountered unexpected type of composite constraint.");
    }
}

private FileCommitter createFileCommitter(Configuration conf) {
    return callWithJobClassLoader(conf, new Action<FileCommitter>() {
      public FileCommitter call(Configuration conf) {
        FileCommitter committer = null;

        LOG.info("FileCommitter set in config "
            + conf.get("mapred.output.committer.class"));

        if (newApiCommitter) {
          org.apache.hadoop.mapreduce.v2.api.records.TaskId taskID =
              MRBuilderUtils.newTaskId(jobId, 0, TaskType.MAP);
          org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
              MRBuilderUtils.newTaskAttemptId(taskID, 0);
          TaskAttemptContext taskContext = new TaskAttemptContextImpl(conf,
              TypeConverter.fromYarn(attemptID));
          FileOutputFormat outputFormat;
          try {
            outputFormat = ReflectionUtils.newInstance(taskContext
                .getOutputFormatClass(), conf);
            committer = outputFormat.getFileCommitter(taskContext);
          } catch (Exception e) {
            throw new YarnRuntimeException(e);
          }
        } else {
          committer = ReflectionUtils.newInstance(conf.getClass(
              "mapred.output.committer.class", DirectoryFileOutputCommitter.class,
              org.apache.hadoop.mapred.OutputCommitter.class), conf);
        }
        LOG.info("FileCommitter is " + committer.getClass().getName());
        return committer;
      }
    });
  }

	protected void handleUnnamedSequenceGenerator() {
		final InFlightMetadataCollector metadataCollector = buildingContext.getMetadataCollector();

		// according to the spec, this should locate a generator with the same name as the entity-name
		final SequenceGeneratorRegistration globalMatch =
				metadataCollector.getGlobalRegistrations().getSequenceGeneratorRegistrations()
						.get( entityMapping.getJpaEntityName() );
		if ( globalMatch != null ) {
			handleSequenceGenerator(
					entityMapping.getJpaEntityName(),
					globalMatch.configuration(),
					idValue,
					idMember,
					buildingContext
			);
			return;
		}

		handleSequenceGenerator(
				entityMapping.getJpaEntityName(),
				new SequenceGeneratorJpaAnnotation( metadataCollector.getSourceModelBuildingContext() ),
				idValue,
				idMember,
				buildingContext
		);
	}

    public String toString() {
        return "TransactionDescription(" +
            "coordinatorId=" + coordinatorId +
            ", state=" + state +
            ", producerId=" + producerId +
            ", producerEpoch=" + producerEpoch +
            ", transactionTimeoutMs=" + transactionTimeoutMs +
            ", transactionStartTimeMs=" + transactionStartTimeMs +
            ", topicPartitions=" + topicPartitions +
            ')';
    }

