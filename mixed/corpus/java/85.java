public void finalizeJournal() throws IOException {
    boolean needsErrorHandling = true;
    mapJournalsAndReportErrors(journal -> {
        if (needsErrorHandling) {
            try {
                journal.close();
            } catch (IOException e) {
                // handle error
            }
        }
    }, "close journal");
    closed = true;
}

  protected synchronized void heartbeat() throws Exception {
    scheduleStats.updateAndLogIfChanged("Before Scheduling: ");
    List<Container> allocatedContainers = getResources();
    if (allocatedContainers != null && allocatedContainers.size() > 0) {
      scheduledRequests.assign(allocatedContainers);
    }

    int completedMaps = getJob().getCompletedMaps();
    int completedTasks = completedMaps + getJob().getCompletedReduces();
    if ((lastCompletedTasks != completedTasks) ||
          (scheduledRequests.maps.size() > 0)) {
      lastCompletedTasks = completedTasks;
      recalculateReduceSchedule = true;
    }

    if (recalculateReduceSchedule) {
      boolean reducerPreempted = preemptReducesIfNeeded();

      if (!reducerPreempted) {
        // Only schedule new reducers if no reducer preemption happens for
        // this heartbeat
        scheduleReduces(getJob().getTotalMaps(), completedMaps,
            scheduledRequests.maps.size(), scheduledRequests.reduces.size(),
            assignedRequests.maps.size(), assignedRequests.reduces.size(),
            mapResourceRequest, reduceResourceRequest, pendingReduces.size(),
            maxReduceRampupLimit, reduceSlowStart);
      }

      recalculateReduceSchedule = false;
    }

    scheduleStats.updateAndLogIfChanged("After Scheduling: ");
  }

