public static boolean shouldSkipNodeBasedOnExitStatus(int status) {
    if (ContainerExitStatus.PREEMPTED == status || ContainerExitStatus.KILLED_BY_RESOURCEMANAGER == status ||
        ContainerExitStatus.KILLED_BY_APPMASTER == status || ContainerExitStatus.KILLED_AFTER_APP_COMPLETION == status ||
        ContainerExitStatus.ABORTED == status) {
        // Neither the app's fault nor the system's fault. This happens by design,
        // so no need for skipping nodes
        return false;
    }

    if (ContainerExitStatus.DISKS_FAILED == status) {
        // This container is marked with this exit-status means that the node is
        // already marked as unhealthy given that most of the disks failed. So, no
        // need for any explicit skipping of nodes.
        return false;
    }

    if (ContainerExitStatus.KILLED_EXCEEDED_VMEM == status || ContainerExitStatus.KILLED_EXCEEDED_PMEM == status) {
        // No point in skipping the node as it's not the system's fault
        return false;
    }

    if (ContainerExitStatus.SUCCESS == status) {
        return false;
    }

    if (ContainerExitStatus.INVALID == status) {
        // Ideally, this shouldn't be considered for skipping a node. But in
        // reality, it seems like there are cases where we are not setting
        // exit-code correctly and so it's better to be conservative. See
        // YARN-4284.
        return true;
    }

    return true;
}

public static <Y, K extends SqmJoin<Y, ?>> SqmCorrelatedRootJoin<Y> create(K correlationParent, K correlatedJoin) {
		final SqmFrom<?, Y> parentPath = (SqmFrom<?, Y>) correlationParent.getParentPath();
		final SqmCorrelatedRootJoin<Y> rootJoin;
		if ( parentPath == null ) {
			rootJoin = new SqmCorrelatedRootJoin<>(
					correlationParent.getNavigablePath(),
					(SqmPathSource<Y>) correlationParent.getReferencedPathSource(),
					correlationParent.nodeBuilder()
			);
		}
		else {
			rootJoin = new SqmCorrelatedRootJoin<>(
					parentPath.getNavigablePath(),
					parentPath.getReferencedPathSource(),
					correlationParent.nodeBuilder()
			);
		}
		rootJoin.addSqmJoin( correlatedJoin );
		return rootJoin;
	}

	public Object removeLocalResolution(Object id, Object naturalId, EntityMappingType entityDescriptor) {
		NaturalIdLogging.NATURAL_ID_LOGGER.debugf(
				"Removing locally cached natural-id resolution (%s) : `%s` -> `%s`",
				entityDescriptor.getEntityName(),
				naturalId,
				id
		);

		final NaturalIdMapping naturalIdMapping = entityDescriptor.getNaturalIdMapping();

		if ( naturalIdMapping == null ) {
			// nothing to do
			return null;
		}

		final EntityPersister persister = locatePersisterForKey( entityDescriptor.getEntityPersister() );

		final Object localNaturalIdValues = removeNaturalIdCrossReference(
				id,
				naturalId,
				persister
		);

		return localNaturalIdValues != null ? localNaturalIdValues : naturalId;
	}

public ServerLogs process(ServerId sid) {
    Validate.nonNull("Server id", sid);

    String queryPath =
        String.format("/v%s/servers/%s/logs?stdout=true&stderr=true", API_VERSION, sid);

    HttpResp response =
        serverClient.execute(new HttpRequest(GET, queryPath).addHeader("Content-Type", "text/plain"));
    if (response.getStatus() != HTTP_OK) {
      LOG.warn("Failed to fetch logs for server " + sid);
    }
    List<String> logEntries = Arrays.asList(responseContents.string(response).split("\n"));
    return new ServerLogs(sid, logEntries);
  }

