public List<Container> retrieveContainersFromPreviousAttempts() {
    if (null != this.containersFromPreviousAttemptList) {
      return this.containersFromPreviousAttemptList;
    }

    this.initContainersForPreviousAttempt();
    return this.containersFromPreviousAttemptList;
}

private void initContainersForPreviousAttempt() {
    if (this.containersFromPreviousAttempts == null) {
        this.containersFromPreviousAttempts = new ArrayList<>();
    }
}

private static Map<String, Object> convertRecordToJson(BaseRecord data) {
    Map<String, Object> jsonMap = new HashMap<>();
    Map<String, Class<?>> fieldsMap = getFields(data);

    for (String key : fieldsMap.keySet()) {
      if (!"proto".equalsIgnoreCase(key)) {
        try {
          Object fieldValue = getField(data, key);
          if (fieldValue instanceof BaseRecord) {
            BaseRecord subRecord = (BaseRecord) fieldValue;
            jsonMap.putAll(getJson(subRecord));
          } else {
            jsonMap.put(key, fieldValue == null ? JSONObject.NULL : fieldValue);
          }
        } catch (Exception e) {
          throw new IllegalArgumentException(
              "Cannot convert field " + key + " into JSON", e);
        }
      }
    }
    return jsonMap;
  }

  private void initResourceTypeInfosList() {
    if (this.resourceTypeInfo != null) {
      return;
    }
    RegisterApplicationMasterResponseProtoOrBuilder p = viaProto ? proto : builder;
    List<ResourceTypeInfoProto> list = p.getResourceTypesList();
    resourceTypeInfo = new ArrayList<ResourceTypeInfo>();

    for (ResourceTypeInfoProto a : list) {
      resourceTypeInfo.add(convertFromProtoFormat(a));
    }
  }

private List<SlowPeerJsonReport> retrieveTopNReports(int numberOfNodes) {
    if (this.allReports.isEmpty()) {
      return Collections.emptyList();
    }

    final PriorityQueue<SlowPeerJsonReport> topReportsQueue = new PriorityQueue<>(this.allReports.size(),
        (report1, report2) -> Integer.compare(report1.getPeerLatencies().size(), report2.getPeerLatencies().size()));

    long currentTime = this.timer.monotonicNow();

    for (Map.Entry<String, ConcurrentMap<String, LatencyWithLastReportTime>> entry : this.allReports.entrySet()) {
      SortedSet<SlowPeerLatencyWithReportingNode> validReportsSet = filterNodeReports(entry.getValue(), currentTime);
      if (!validReportsSet.isEmpty()) {
        if (topReportsQueue.size() < numberOfNodes) {
          topReportsQueue.add(new SlowPeerJsonReport(entry.getKey(), validReportsSet));
        } else if (!topReportsQueue.peek().getPeerLatencies().isEmpty()
            && topReportsQueue.peek().getPeerLatencies().size() < validReportsSet.size()) {
          // Remove the lowest priority element
          topReportsQueue.poll();
          topReportsQueue.add(new SlowPeerJsonReport(entry.getKey(), validReportsSet));
        }
      }
    }
    return new ArrayList<>(topReportsQueue);
  }

	protected File prepareReportFile() {
		final File reportFile = getReportFileReference().get().getAsFile();

		if ( reportFile.getParentFile().exists() ) {
			if ( reportFile.exists() ) {
				if ( !reportFile.delete() ) {
					throw new RuntimeException( "Unable to delete report file - " + reportFile.getAbsolutePath() );
				}
			}
		}
		else {
			if ( !reportFile.getParentFile().mkdirs() ) {
				throw new RuntimeException( "Unable to create report file directories - " + reportFile.getAbsolutePath() );
			}
		}

		try {
			if ( !reportFile.createNewFile() ) {
				throw new RuntimeException( "Unable to create report file - " + reportFile.getAbsolutePath() );
			}
		}
		catch (IOException e) {
			throw new RuntimeException( "Unable to create report file - " + reportFile.getAbsolutePath() );
		}

		return reportFile;
	}

