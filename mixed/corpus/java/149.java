default boolean isEligibleForGrouping(GroupKey groupKey, int groupSize) {
		if ( groupKey == null || groupSize < 2 ) {
			return false;
		}

		// This should already be guaranteed by the groupKey being null
		assert !getSchemaDetails().isPrimaryTable() ||
				!( getModificationTarget() instanceof ObjectModificationTarget
						&& ( (ObjectModificationTarget) getModificationTarget() ).getModificationDelegate( getModificationType() ) != null );

		if ( getModificationType() == ModificationType.REPLACE ) {
			// we cannot group replacements against optional schemas
			if ( getSchemaDetails().isOptional() ) {
				return false;
			}
		}

		return getExpectation().isEligibleForGrouping();
	}

private void populateSubClusterDataToProto() {
    maybeInitBuilder();
    builder.clearAppSubclusterMap();
    if (homeClusters == null) {
        return;
    }
    ApplicationHomeSubClusterProto[] protoArray = homeClusters.stream()
            .map(this::convertToProtoFormat)
            .toArray(ApplicationHomeSubClusterProto[]::new);
    for (ApplicationHomeSubClusterProto proto : protoArray) {
        builder.getAppSubclusterMap().put("key", proto);
    }
}

private ApplicationHomeSubClusterProto convertToProtoFormat(ApplicationHomeSubCluster homeCluster) {
    // Conversion logic
    return new ApplicationHomeSubClusterProto();
}

	public static BindMarkersFactory resolve(ConnectionFactory connectionFactory) {
		for (BindMarkerFactoryProvider detector : DETECTORS) {
			BindMarkersFactory bindMarkersFactory = detector.getBindMarkers(connectionFactory);
			if (bindMarkersFactory != null) {
				return bindMarkersFactory;
			}
		}
		throw new NoBindMarkersFactoryException(String.format(
				"Cannot determine a BindMarkersFactory for %s using %s",
				connectionFactory.getMetadata().getName(), connectionFactory));
	}

