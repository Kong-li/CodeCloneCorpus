	private static boolean discoverTypeWithoutReflection(ClassDetails classDetails, MemberDetails memberDetails) {
		if ( memberDetails.hasDirectAnnotationUsage( Target.class ) ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( Basic.class ) ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( Type.class ) ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( JavaType.class ) ) {
			return true;
		}

		final OneToOne oneToOneAnn = memberDetails.getDirectAnnotationUsage( OneToOne.class );
		if ( oneToOneAnn != null ) {
			return oneToOneAnn.targetEntity() != void.class;
		}

		final OneToMany oneToManyAnn = memberDetails.getDirectAnnotationUsage( OneToMany.class );
		if ( oneToManyAnn != null ) {
			return oneToManyAnn.targetEntity() != void.class;
		}

		final ManyToOne manyToOneAnn = memberDetails.getDirectAnnotationUsage( ManyToOne.class );
		if ( manyToOneAnn != null ) {
			return manyToOneAnn.targetEntity() != void.class;
		}

		final ManyToMany manyToManyAnn = memberDetails.getDirectAnnotationUsage( ManyToMany.class );
		if ( manyToManyAnn != null ) {
			return manyToManyAnn.targetEntity() != void.class;
		}

		if ( memberDetails.hasDirectAnnotationUsage( Any.class ) ) {
			return true;
		}

		final ManyToAny manToAnyAnn = memberDetails.getDirectAnnotationUsage( ManyToAny.class );
		if ( manToAnyAnn != null ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( JdbcTypeCode.class ) ) {
			return true;
		}

		if ( memberDetails.getType().determineRawClass().isImplementor( Class.class ) ) {
			// specialized case for @Basic attributes of type Class (or Class<?>, etc.).
			// we only really care about the Class part
			return true;
		}

		return false;
	}

private static String entityInfo(PersistEvent event, Object info, EntityEntry entry) {
		if ( event.getEntityInfo() != null ) {
			return event.getEntityInfo();
		}
		else {
			// changes event.entityInfo by side effect!
			final String infoName = event.getSession().bestGuessEntityInfo( info, entry );
			event.setEntityInfo( infoName );
			return infoName;
		}
	}

public static long getLastIncludedLogTime(RawSnapshotReader dataReader) {
        RecordsSnapshotReader<ByteBuffer> recordsSnapshotReader = RecordsSnapshotReader.of(
            dataReader,
            IdentitySerde.INSTANCE,
            new BufferSupplier.GrowableBufferSupplier(),
            KafkaRaftClient.MAX_BATCH_SIZE_BYTES,
            false
        );
        try (recordsSnapshotReader) {
            return recordsSnapshotReader.getLastIncludedLogTime();
        }
    }

public <Y> ValueExtractor<Y> getExtractor(final JavaType<Y> javaType) {
		return new BasicExtractor<Y>( javaType, this ) {

			private Y doExtract(ResultSet rs, int columnIndex, WrapperOptions options) throws SQLException {
				if (!this.determineCrsIdFromDatabase()) {
					return javaType.wrap(HANASpatialUtils.toGeometry(rs.getObject(columnIndex)), options);
				} else {
					throw new UnsupportedOperationException("First need to refactor HANASpatialUtils");
					//return getJavaTypeDescriptor().wrap( HANASpatialUtils.toGeometry( rs, paramIndex ), options );
				}
			}

			private Y doExtract(CallableStatement statement, int index, WrapperOptions options) throws SQLException {
				return javaType.wrap(HANASpatialUtils.toGeometry(statement.getObject(index)), options);
			}

			private Y doExtract(CallableStatement statement, String columnName, WrapperOptions options)
					throws SQLException {
				return javaType.wrap(HANASpatialUtils.toGeometry(statement.getObject(columnName)), options);
			}
		};
	}

public UserFetchRequest.Builder newUserFetchBuilder(String userId, FetchConfig fetchConfig) {
    List<UserPartition> added = new ArrayList<>();
    List<UserPartition> removed = new ArrayList<>();
    List<UserPartition> replaced = new ArrayList<>();

    if (nextMetadata.isNewSession()) {
        // Add any new partitions to the session
        for (Entry<UserId, UserPartition> entry : nextPartitions.entrySet()) {
            UserId userIdentity = entry.getKey();
            UserPartition userPartition = entry.getValue();
            sessionPartitions.put(userIdentity, userPartition);
        }

        // If it's a new session, all the partitions must be added to the request
        added.addAll(sessionPartitions.values());
    } else {
        // Iterate over the session partitions, tallying which were added
        Iterator<Entry<UserId, UserPartition>> partitionIterator = sessionPartitions.entrySet().iterator();
        while (partitionIterator.hasNext()) {
            Entry<UserId, UserPartition> entry = partitionIterator.next();
            UserId userIdentity = entry.getKey();
            UserPartition prevData = entry.getValue();
            UserPartition nextData = nextPartitions.remove(userIdentity);
            if (nextData != null) {
                // If the user ID does not match, the user has been recreated
                if (!prevData.equals(nextData)) {
                    nextPartitions.put(userIdentity, nextData);
                    entry.setValue(nextData);
                    replaced.add(prevData);
                }
            } else {
                // This partition is not in the builder, so we need to remove it from the session
                partitionIterator.remove();
                removed.add(prevData);
            }
        }

        // Add any new partitions to the session
        for (Entry<UserId, UserPartition> entry : nextPartitions.entrySet()) {
            UserId userIdentity = entry.getKey();
            UserPartition userPartition = entry.getValue();
            sessionPartitions.put(userIdentity, userPartition);
            added.add(userPartition);
        }
    }

    if (log.isDebugEnabled()) {
        log.debug("Build UserFetch {} for node {}. Added {}, removed {}, replaced {} out of {}",
                nextMetadata, node,
                userPartitionsToLogString(added),
                userPartitionsToLogString(removed),
                userPartitionsToLogString(replaced),
                userPartitionsToLogString(sessionPartitions.values()));
    }

    // The replaced user-partitions need to be removed, and their replacements are already added
    removed.addAll(replaced);

    Map<UserPartition, List<UserFetchRequestData.AcknowledgementBatch>> acknowledgementBatches = new HashMap<>();
    nextAcknowledgements.forEach((partition, acknowledgements) -> acknowledgementBatches.put(partition, acknowledgements.getAcknowledgementBatches()
            .stream().map(AcknowledgementBatch::toUserFetchRequest)
            .collect(Collectors.toList())));

    nextPartitions = new LinkedHashMap<>();
    nextAcknowledgements = new LinkedHashMap<>();

    return UserFetchRequest.Builder.forConsumer(
            userId, nextMetadata, fetchConfig.maxWaitMs,
            fetchConfig.minBytes, fetchConfig.maxBytes, fetchConfig.fetchSize,
            added, removed, acknowledgementBatches);
}

