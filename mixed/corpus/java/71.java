	public AbstractHandlerMapping getHandlerMapping() {
		Map<String, Object> urlMap = new LinkedHashMap<>();
		for (ServletWebSocketHandlerRegistration registration : this.registrations) {
			MultiValueMap<HttpRequestHandler, String> mappings = registration.getMappings();
			mappings.forEach((httpHandler, patterns) -> {
				for (String pattern : patterns) {
					urlMap.put(pattern, httpHandler);
				}
			});
		}
		WebSocketHandlerMapping hm = new WebSocketHandlerMapping();
		hm.setUrlMap(urlMap);
		hm.setOrder(this.order);
		if (this.urlPathHelper != null) {
			hm.setUrlPathHelper(this.urlPathHelper);
		}
		return hm;
	}

public Future<Map<String, UserGroupDescription>> getAll() {
        return Future.allOf(userFutures.values().toArray(new Future[0])).thenApply(
            nil -> {
                Map<String, UserGroupDescription> descriptions = new HashMap<>(userFutures.size());
                userFutures.forEach((key, future) -> {
                    try {
                        descriptions.put(key, future.get());
                    } catch (InterruptedException | ExecutionException e) {
                        // This should be unreachable, since the Future#allOf already ensured
                        // that all of the futures completed successfully.
                        throw new RuntimeException(e);
                    }
                });
                return descriptions;
            });
    }

    public void close() {
        Arrays.asList(
            NUM_OFFSETS,
            NUM_CLASSIC_GROUPS,
            NUM_CLASSIC_GROUPS_PREPARING_REBALANCE,
            NUM_CLASSIC_GROUPS_COMPLETING_REBALANCE,
            NUM_CLASSIC_GROUPS_STABLE,
            NUM_CLASSIC_GROUPS_DEAD,
            NUM_CLASSIC_GROUPS_EMPTY
        ).forEach(registry::removeMetric);

        Arrays.asList(
            classicGroupCountMetricName,
            consumerGroupCountMetricName,
            consumerGroupCountEmptyMetricName,
            consumerGroupCountAssigningMetricName,
            consumerGroupCountReconcilingMetricName,
            consumerGroupCountStableMetricName,
            consumerGroupCountDeadMetricName,
            shareGroupCountMetricName,
            shareGroupCountEmptyMetricName,
            shareGroupCountStableMetricName,
            shareGroupCountDeadMetricName
        ).forEach(metrics::removeMetric);

        Arrays.asList(
            OFFSET_COMMITS_SENSOR_NAME,
            OFFSET_EXPIRED_SENSOR_NAME,
            OFFSET_DELETIONS_SENSOR_NAME,
            CLASSIC_GROUP_COMPLETED_REBALANCES_SENSOR_NAME,
            CONSUMER_GROUP_REBALANCES_SENSOR_NAME,
            SHARE_GROUP_REBALANCES_SENSOR_NAME
        ).forEach(metrics::removeSensor);
    }

public void output(DataWrite out) throws IOException {

    // First write out the size of the class array and any classes that are
    // "unknown" classes

    out.writeByte(unknownClasses);

    for (byte i = 1; i <= unknownClasses; i++) {
      out.writeByte(i);
      out.writeUTF(getClassType(i).getTypeName());
    }
  }

