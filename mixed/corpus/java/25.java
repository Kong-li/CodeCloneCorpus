public void secureWriteAccess() {
    if (!usesFencableWriter || fencableProducer != null) return;

    try {
        FencableProducer producer = createFencableProducer();
        producer.initTransactions();
        fencableProducer = producer;
    } catch (Exception e) {
        relinquishWritePrivileges();
        throw new ConnectException("Failed to create and initialize secure producer for config topic", e);
    }
}

  protected void reregisterCollectors() {
    Map<ApplicationId, AppCollectorData> knownCollectors
        = context.getKnownCollectors();
    if (knownCollectors == null) {
      return;
    }
    ConcurrentMap<ApplicationId, AppCollectorData> registeringCollectors
        = context.getRegisteringCollectors();
    for (Map.Entry<ApplicationId, AppCollectorData> entry
        : knownCollectors.entrySet()) {
      Application app = context.getApplications().get(entry.getKey());
      if ((app != null)
          && !ApplicationState.FINISHED.equals(app.getApplicationState())) {
        registeringCollectors.putIfAbsent(entry.getKey(), entry.getValue());
        AppCollectorData data = entry.getValue();
        LOG.debug("{} : {}@<{}, {}>", entry.getKey(), data.getCollectorAddr(),
            data.getRMIdentifier(), data.getVersion());
      } else {
        LOG.debug("Remove collector data for done app {}", entry.getKey());
      }
    }
    knownCollectors.clear();
  }

public static void logInfo(Logger logger, Predicate<Boolean> messageConditioner) {
		if (logger.isInfoEnabled()) {
			boolean infoEnabled = logger.isDebugEnabled();
			String logMessage = messageConditioner.test(infoEnabled) ? "Info Log Message" : "Debug Log Message";
			if (infoEnabled) {
				logger.debug(logMessage);
			} else {
				logger.info(logMessage);
			}
		}
	}

