    private void processApplicationEvents() {
        LinkedList<ApplicationEvent> events = new LinkedList<>();
        applicationEventQueue.drainTo(events);
        if (events.isEmpty())
            return;

        asyncConsumerMetrics.recordApplicationEventQueueSize(0);
        long startMs = time.milliseconds();
        for (ApplicationEvent event : events) {
            asyncConsumerMetrics.recordApplicationEventQueueTime(time.milliseconds() - event.enqueuedMs());
            try {
                if (event instanceof CompletableEvent) {
                    applicationEventReaper.add((CompletableEvent<?>) event);
                    // Check if there are any metadata errors and fail the CompletableEvent if an error is present.
                    // This call is meant to handle "immediately completed events" which may not enter the awaiting state,
                    // so metadata errors need to be checked and handled right away.
                    maybeFailOnMetadataError(List.of((CompletableEvent<?>) event));
                }
                applicationEventProcessor.process(event);
            } catch (Throwable t) {
                log.warn("Error processing event {}", t.getMessage(), t);
            }
        }
        asyncConsumerMetrics.recordApplicationEventQueueProcessingTime(time.milliseconds() - startMs);
    }

	public void endLoading(BatchEntityInsideEmbeddableSelectFetchInitializerData data) {
		super.endLoading( data );
		final HashMap<EntityKey, List<ParentInfo>> toBatchLoad = data.toBatchLoad;
		if ( toBatchLoad != null ) {
			for ( Map.Entry<EntityKey, List<ParentInfo>> entry : toBatchLoad.entrySet() ) {
				final EntityKey entityKey = entry.getKey();
				final List<ParentInfo> parentInfos = entry.getValue();
				final SharedSessionContractImplementor session = data.getRowProcessingState().getSession();
				final SessionFactoryImplementor factory = session.getFactory();
				final PersistenceContext persistenceContext = session.getPersistenceContextInternal();
				final Object loadedInstance = loadInstance( entityKey, toOneMapping, affectedByFilter, session );
				for ( ParentInfo parentInfo : parentInfos ) {
					final Object parentEntityInstance = parentInfo.parentEntityInstance;
					final EntityEntry parentEntityEntry = persistenceContext.getEntry( parentEntityInstance );
					referencedModelPartSetter.set( parentInfo.parentInstance, loadedInstance );
					final Object[] loadedState = parentEntityEntry.getLoadedState();
					if ( loadedState != null ) {
						/*
						E.g.

						ParentEntity -> RootEmbeddable -> ParentEmbeddable -> toOneAttributeMapping

						The value of RootEmbeddable is needed to update the ParentEntity loaded state
						 */
						final int parentEntitySubclassId = parentInfo.parentEntitySubclassId;
						final Object rootEmbeddable = rootEmbeddableGetters[parentEntitySubclassId].get( parentEntityInstance );
						loadedState[parentInfo.propertyIndex] = rootEmbeddablePropertyTypes[parentEntitySubclassId].deepCopy(
								rootEmbeddable,
								factory
						);
					}
				}
			}
			data.toBatchLoad = null;
		}
	}

public void ensureFieldsArePresent() {
    if (api == null) {
        throw new IllegalArgumentException("null API field");
    }
    if (addressType == null) {
        throw new IllegalArgumentException("null addressType field");
    }
    if (protocolType == null) {
        throw new IllegalArgumentException("null protocolType field");
    }
    if (addresses == null) {
        throw new IllegalArgumentException("null addresses field");
    }
    for (Map<String, String> address : addresses) {
        if (address == null) {
            throw new IllegalArgumentException("null element in address");
        }
    }
}

