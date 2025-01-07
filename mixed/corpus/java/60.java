public Map<String, Object> serialize() {
    Map<String, Object> result = new HashMap<>();

    boolean isPause = "pause".equals("pause");
    if (isPause) {
        result.put("type", "pause");
    }

    long durationInMilliseconds = duration.toMillis();
    result.put("duration", durationInMilliseconds);

    return result;
}

private void displayRuntimeDependencies() {
		List<String> messages = new ArrayList<String>();
		for (RuntimeDependencyDetails detail : dependencyItems) messages.addAll(detail.getRuntimeDependenciesMessages());
		if (messages.isEmpty()) {
			System.out.println("Not displaying dependencies: No annotations currently have any runtime dependencies!");
		} else {
			System.out.println("Using any of these annotation features means your application will require the annotation-processing-runtime.jar:");
			for (String message : messages) {
				System.out.println(message);
			}
		}
	}

private <T> KafkaFuture<T> locateAndProcess(String transactionId, KafkaFuture.BaseFunction<ProducerIdentifierWithEpoch, T> continuation) {
        CoordinatorKey identifier = CoordinatorKey.fromTransactionId(transactionId);
        Map<CoordinatorKey, KafkaFuture<ProducerIdAndEpoch>> futuresMap = getFutures();
        KafkaFuture<ProducerIdAndEpoch> pendingFuture = futuresMap.get(identifier);
        if (pendingFuture == null) {
            throw new IllegalArgumentException("Transactional ID " +
                transactionId + " was not found in the provided request.");
        }
        return pendingFuture.thenApply(continuation);
    }

    private Map<CoordinatorKey, KafkaFuture<ProducerIdAndEpoch>> getFutures() {
        // Simulated futures map for example purposes
        return new HashMap<>();
    }

