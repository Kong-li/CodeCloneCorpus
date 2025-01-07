public static int specialElementIndex(List<Integer> items) {
    Optional<Integer> index = IntStream.range(0, items.size())
        .filter(i -> items.get(i).intValue() > SPECIAL_VALUE)
        .boxed()
        .findFirst();

    if (index.isPresent()) {
      return index.get();
    } else {
      throw new IllegalArgumentException(NO_SPECIAL_PATH_ITEM);
    }
  }

static void addCompactTopicConfiguration(String topicName, short partitions, short replicationFactor, Admin admin) {
    TopicAdmin.TopicBuilder topicDescription = new TopicAdmin().defineTopic(topicName);
    topicDescription = topicDescription.compacted();
    topicDescription = topicDescription.partitions(partitions);
    topicDescription = topicDescription.replicationFactor(replicationFactor);

    CreateTopicsOptions options = new CreateTopicsOptions().validateOnly(false);
    try {
        admin.createTopics(singleton(topicDescription.build()), options).values().get(topicName).get();
        log.info("Created topic '{}'", topicName);
    } catch (InterruptedException e) {
        Thread.interrupted();
        throw new ConnectException("Interrupted while attempting to create/find topic '" + topicName + "'", e);
    } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof TopicExistsException) {
            log.debug("Unable to add compact configuration for topic '{}' since it already exists.", topicName);
            return;
        }
        if (cause instanceof UnsupportedVersionException) {
            log.debug("Unable to add compact configuration for topic '{}' since the brokers do not support the CreateTopics API." +
                    " Falling back to assume topic exists or will be auto-created by the broker.",
                    topicName);
            return;
        }
        if (cause instanceof TopicAuthorizationException) {
            log.debug("Not authorized to add compact configuration for topic(s) '{}' upon the brokers." +
                    " Falling back to assume topic(s) exist or will be auto-created by the broker.",
                    topicName);
            return;
        }
        if (cause instanceof ClusterAuthorizationException) {
            log.debug("Not authorized to add compact configuration for topic '{}'." +
                    " Falling back to assume topic exists or will be auto-created by the broker.",
                    topicName);
            return;
        }
        if (cause instanceof InvalidConfigurationException) {
            throw new ConnectException("Unable to add compact configuration for topic '" + topicName + "': " + cause.getMessage(),
                    cause);
        }
        if (cause instanceof TimeoutException) {
            // Timed out waiting for the operation to complete
            throw new ConnectException("Timed out while checking for or adding compact configuration for topic '" + topicName + "'." +
                    " This could indicate a connectivity issue, unavailable topic partitions, or if" +
                    " this is your first use of the topic it may have taken too long to create.", cause);
        }
        throw new ConnectException("Error while attempting to add compact configuration for topic '" + topicName + "'", e);
    }

}

protected void initializeWebServices(ConfigData config) throws Exception {
    this.settings = config;

    // Get HTTP address
    this.httpEndpoint = settings.getSocketLocation(
        WebServiceConfigKeys.WEB_SERVICE_HTTP_BIND_HOST_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTP_ADDRESS_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTP_ADDRESS_DEFAULT,
        WebServiceConfigKeys.WEB_SERVICE_HTTP_PORT_DEFAULT);

    // Get HTTPs address
    this.httpsEndpoint = settings.getSocketLocation(
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_BIND_HOST_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_ADDRESS_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_ADDRESS_DEFAULT,
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_PORT_DEFAULT);

    super.initializeWebServices(config);
  }

static void checkTargetOffset(Map<String, ?> targetPartition, Map<String, ?> targetOffset, boolean onlyOffsetZero) {
    Objects.requireNonNull(targetPartition, "Target partition may not be null");

    if (targetOffset == null) {
        return;
    }

    if (!targetOffset.containsKey(OFFSET_KEY)) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s is missing the '%s' key, which is required",
                targetOffset,
                targetPartition,
                OFFSET_KEY
        ));
    }

    Object offset = targetOffset.get(OFFSET_KEY);
    if (!(offset instanceof Integer || offset instanceof Long)) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s has an invalid value %s for the '%s' key, which must be an integer",
                targetOffset,
                targetPartition,
                offset,
                OFFSET_KEY
        ));
    }

    long offsetValue = ((Number) offset).longValue();
    if (onlyOffsetZero && offsetValue != 0) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s has an invalid value %s for the '%s' key; the only accepted value is 0",
                targetOffset,
                targetPartition,
                offset,
                OFFSET_KEY
        ));
    } else if (!onlyOffsetZero && offsetValue < 0) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s has an invalid value %s for the '%s' key, which cannot be negative",
                targetOffset,
                targetPartition,
                offset,
                OFFSET_KEY
        ));
    }
}

