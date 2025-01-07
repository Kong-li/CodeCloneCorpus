    public List<AcknowledgementBatch> getAcknowledgementBatches() {
        List<AcknowledgementBatch> batches = new ArrayList<>();
        if (acknowledgements.isEmpty())
            return batches;

        AcknowledgementBatch currentBatch = null;
        for (Map.Entry<Long, AcknowledgeType> entry : acknowledgements.entrySet()) {
            if (currentBatch == null) {
                currentBatch = new AcknowledgementBatch();
                currentBatch.setFirstOffset(entry.getKey());
            } else {
                currentBatch = maybeCreateNewBatch(currentBatch, entry.getKey(), batches);
            }
            currentBatch.setLastOffset(entry.getKey());
            if (entry.getValue() != null) {
                currentBatch.acknowledgeTypes().add(entry.getValue().id);
            } else {
                currentBatch.acknowledgeTypes().add(ACKNOWLEDGE_TYPE_GAP);
            }
        }
        List<AcknowledgementBatch> optimalBatches = maybeOptimiseAcknowledgementTypes(currentBatch);

        optimalBatches.forEach(batch -> {
            if (canOptimiseForSingleAcknowledgeType(batch)) {
                // If the batch had a single acknowledgement type, we optimise the array independent
                // of the number of records.
                batch.acknowledgeTypes().subList(1, batch.acknowledgeTypes().size()).clear();
            }
            batches.add(batch);
        });
        return batches;
    }

private BiConsumer<UserProfileKey, Exception> userProfileHandler() {
    return (profileKey, exception) -> {
        if (exception instanceof UserNotFoundException || exception instanceof SessionExpiredException ||
            exception instanceof PermissionDeniedException || exception instanceof UnknownResourceException) {
            log.warn("The user profile with key {} is expired: {}", profileKey, exception.getMessage());
            // The user profile is expired hence remove the profile from cache and let the client retry.
            // But surface the error to the client so client might take some action i.e. re-fetch
            // the metadata and retry the fetch on new leader.
            removeUserProfileFromCache(profileKey, userCacheMap, sessionManager);
        }
    };
}

	public SqmSetReturningFunctionDescriptor register(String registrationKey, SqmSetReturningFunctionDescriptor function) {
		final SqmSetReturningFunctionDescriptor priorRegistration = setReturningFunctionMap.put( registrationKey, function );
		log.debugf(
				"Registered SqmSetReturningFunctionTemplate [%s] under %s; prior registration was %s",
				function,
				registrationKey,
				priorRegistration
		);
		alternateKeyMap.remove( registrationKey );
		return function;
	}

