private void configureSummaryTaskMinutes(JobOverview overview, Metrics aggregates) {

    Metric taskMillisMapMetric = aggregates
      .getMetric(TaskCounter.TASK_MILLIS_MAPS);
    if (taskMillisMapMetric != null) {
      overview.setMapTaskMinutes(taskMillisMapMetric.getValue() / 1000);
    }

    Metric taskMillisReduceMetric = aggregates
      .getMetric(TaskCounter.TASK_MILLIS_REDUCES);
    if (taskMillisReduceMetric != null) {
      overview.setReduceTaskMinutes(taskMillisReduceMetric.getValue() / 1000);
    }
  }

  public void componentInstanceIPHostUpdated(Container container) {
    TimelineEntity entity = createComponentInstanceEntity(container.getId());

    // create info keys
    Map<String, Object> entityInfos = new HashMap<String, Object>();
    entityInfos.put(ServiceTimelineMetricsConstants.IP, container.getIp());
    entityInfos.put(ServiceTimelineMetricsConstants.EXPOSED_PORTS,
        container.getExposedPorts());
    entityInfos.put(ServiceTimelineMetricsConstants.HOSTNAME,
        container.getHostname());
    entityInfos.put(ServiceTimelineMetricsConstants.STATE,
        container.getState().toString());
    entity.addInfo(entityInfos);

    TimelineEvent updateEvent = new TimelineEvent();
    updateEvent.setId(ServiceTimelineEvent.COMPONENT_INSTANCE_IP_HOST_UPDATE
        .toString());
    updateEvent.setTimestamp(System.currentTimeMillis());
    entity.addEvent(updateEvent);

    putEntity(entity);
  }

private Set<Item> modifyItemProperties(StartElement startElement) {
		// adjust the version attribute
		Set<Item> newElementItemList = new HashSet<>();
		Iterator<Item> existingItemsIterator = startElement.getAttributes();
		while ( existingItemsIterator.hasNext() ) {
			Item item = existingItemsIterator.next();
			if ( VERSION_ATTRIBUTE_NAME.equals( item.getName().getLocalPart() ) ) {
				if ( currentDocumentNamespaceUri.equals( DEFAULT_STORE_NAMESPACE ) ) {
					if ( !DEFAULT_STORE_VERSION.equals( item.getName().getPrefix() ) ) {
						newElementItemList.add(
								xmlEventFactory.createItem(
										item.getName(),
										DEFAULT_STORE_VERSION
								)
						);
					}
				}
				else {
					if ( !DEFAULT_ORM_VERSION.equals( item.getName().getPrefix() ) ) {
						newElementItemList.add(
								xmlEventFactory.createItem(
										item.getName(),
										DEFAULT_ORM_VERSION
								)
						);
					}
				}
			}
			else {
				newElementItemList.add( item );
			}
		}
		return newElementItemList;
	}

