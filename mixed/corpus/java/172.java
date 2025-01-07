  private long resultSetColToLong(ResultSet rs, int colNum, int sqlDataType) throws SQLException {
    try {
      switch (sqlDataType) {
      case Types.DATE:
        return rs.getDate(colNum).getTime();
      case Types.TIME:
        return rs.getTime(colNum).getTime();
      case Types.TIMESTAMP:
        return rs.getTimestamp(colNum).getTime();
      default:
        throw new SQLException("Not a date-type field");
      }
    } catch (NullPointerException npe) {
      // null column. return minimum long value.
      LOG.warn("Encountered a NULL date in the split column. Splits may be poorly balanced.");
      return Long.MIN_VALUE;
    }
  }

  protected void extend(double newProgress, int newValue) {
    if (state == null || newProgress < state.oldProgress) {
      return;
    }

    // This correctness of this code depends on 100% * count = count.
    int oldIndex = (int)(state.oldProgress * count);
    int newIndex = (int)(newProgress * count);
    int originalOldValue = state.oldValue;

    double fullValueDistance = (double)newValue - state.oldValue;
    double fullProgressDistance = newProgress - state.oldProgress;
    double originalOldProgress = state.oldProgress;

    // In this loop we detect each subinterval boundary within the
    //  range from the old progress to the new one.  Then we
    //  interpolate the value from the old value to the new one to
    //  infer what its value might have been at each such boundary.
    //  Lastly we make the necessary calls to extendInternal to fold
    //  in the data for each trapazoid where no such trapazoid
    //  crosses a boundary.
    for (int closee = oldIndex; closee < newIndex; ++closee) {
      double interpolationProgress = (double)(closee + 1) / count;
      // In floats, x * y / y might not equal y.
      interpolationProgress = Math.min(interpolationProgress, newProgress);

      double progressLength = (interpolationProgress - originalOldProgress);
      double interpolationProportion = progressLength / fullProgressDistance;

      double interpolationValueDistance
        = fullValueDistance * interpolationProportion;

      // estimates the value at the next [interpolated] subsegment boundary
      int interpolationValue
        = (int)interpolationValueDistance + originalOldValue;

      extendInternal(interpolationProgress, interpolationValue);

      advanceState(interpolationProgress, interpolationValue);

      values[closee] = (int)state.currentAccumulation;
      initializeInterval();

    }

    extendInternal(newProgress, newValue);
    advanceState(newProgress, newValue);

    if (newIndex == count) {
      state = null;
    }
  }

public void executePreemption() {
    boolean continueExecution = true;
    while (continueExecution) {
        try {
            FSAppAttempt starvingApplication = getStarvedApplications().take();
            synchronized (schedulerReadLock) {
                preemptContainers(identifyContainersToPreempt(starvingApplication));
            }
            starvingApplication.preemptionTriggered(delayBeforeNextCheck);
        } catch (InterruptedException e) {
            LOG.info("Preemption execution interrupted! Exiting.");
            Thread.currentThread().interrupt();
            continueExecution = false;
        }
    }
}

private FSAppAttempt getStarvedApplications() {
    return context.getStarvedApps();
}

private Collection<Container> identifyContainersToPreempt(FSAppAttempt app) {
    // Logic to identify containers to preempt
    return null;
}

