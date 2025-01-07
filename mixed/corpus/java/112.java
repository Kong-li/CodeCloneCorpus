protected void operationInit() throws Exception {
    if (this.clusterContext.isHAEnabled()) {
      switchToStandby(false);
    }

    launchApplication();
    if (getConfiguration().getBoolean(ApplicationConfig.IS_MINI_APPLICATION,
        false)) {
      int port = application.port();
      ApplicationUtils.setRMAppPort(conf, port);
    }

    // Refresh node state before the operation startup to reflect the unregistered
    // nodemanagers as LOST if the tracking for unregistered nodes flag is enabled.
    // For HA setup, refreshNodes is already being called before the active
    // transition.
    Configuration appConf = getConfiguration();
    if (!this.clusterContext.isHAEnabled() && appConf.getBoolean(
        ApplicationConfig.ENABLE_TRACKING_FOR_UNREGISTERED_NODES,
        ApplicationConfig.DEFAULT_ENABLE_TRACKING_FOR_UNREGISTERED_NODES)) {
      this.clusterContext.getNodeStateManager().refreshNodes(appConf);
    }

    super.operationInit();

    // Non HA case, start after RM services are started.
    if (!this.clusterContext.isHAEnabled()) {
      switchToActive();
    }
}

public void initializeService(Configuration config) {
    this.serviceConfig = config;
    String baseDir = serviceConfig.get(NM_RUNC_IMAGE_TOPLEVEL_DIR,
        DEFAULT_NM_RUNC_IMAGE_TOPLEVEL_DIR);
    String layersPath = baseDir + "/layers/";
    String configPath = baseDir + "/config/";
    FileStatusCacheLoader cacheLoader = new FileStatusCacheLoader() {
      @Override
      public FileStatus get(@Nonnull Path path) throws Exception {
        return statBlob(path);
      }
    };
    int maxStatCacheSize = serviceConfig.getInt(NM_RUNC_STAT_CACHE_SIZE,
        DEFAULT_RUNC_STAT_CACHE_SIZE);
    long statCacheTimeoutSecs = serviceConfig.getInt(NM_RUNC_STAT_CACHE_TIMEOUT,
        DEFAULT_NM_RUNC_STAT_CACHE_TIMEOUT);
    this.statCache = CacheBuilder.newBuilder().maximumSize(maxStatCacheSize)
        .refreshAfterWrite(statCacheTimeoutSecs, TimeUnit.SECONDS)
        .build(cacheLoader);
  }

  class FileStatusCacheLoader extends CacheLoader<Path, FileStatus> {
    @Override
    public FileStatus load(@Nonnull Path path) throws Exception {
      return statBlob(path);
    }
  }

    void syncGroupOffset(String consumerGroupId, Map<TopicPartition, OffsetAndMetadata> offsetToSync) throws ExecutionException, InterruptedException {
        if (targetAdminClient != null) {
            adminCall(
                    () -> targetAdminClient.alterConsumerGroupOffsets(consumerGroupId, offsetToSync).all()
                            .whenComplete((v, throwable) -> {
                                if (throwable != null) {
                                    if (throwable.getCause() instanceof UnknownMemberIdException) {
                                        log.warn("Unable to sync offsets for consumer group {}. This is likely caused " +
                                                "by consumers currently using this group in the target cluster.", consumerGroupId);
                                    } else {
                                        log.error("Unable to sync offsets for consumer group {}.", consumerGroupId, throwable);
                                    }
                                } else {
                                    log.trace("Sync-ed {} offsets for consumer group {}.", offsetToSync.size(), consumerGroupId);
                                }
                            }),
                    () -> String.format("alter offsets for consumer group %s on %s cluster", consumerGroupId, targetClusterAlias)
            );
        }
    }

	private <T> T instantiateListener(Class<T> listenerClass) {
		try {
			//noinspection deprecation
			return listenerClass.newInstance();
		}
		catch ( Exception e ) {
			throw new EventListenerRegistrationException(
					"Unable to instantiate specified event listener class: " + listenerClass.getName(),
					e
			);
		}
	}

    public KeyValueIterator<Windowed<K>, V> all() {
        return new MeteredWindowedKeyValueIterator<>(
            wrapped().all(),
            fetchSensor,
            iteratorDurationSensor,
            streamsMetrics,
            serdes::keyFrom,
            serdes::valueFrom,
            time,
            numOpenIterators,
            openIterators
        );
    }

