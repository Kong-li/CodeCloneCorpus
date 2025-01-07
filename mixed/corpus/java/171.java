  public void clearConfigurableFields() {
    writeLock.lock();
    try {
      for (String label : capacitiesMap.keySet()) {
        _set(label, CapacityType.CAP, 0);
        _set(label, CapacityType.MAX_CAP, 0);
        _set(label, CapacityType.ABS_CAP, 0);
        _set(label, CapacityType.ABS_MAX_CAP, 0);
        _set(label, CapacityType.WEIGHT, -1);
      }
    } finally {
      writeLock.unlock();
    }
  }

public void onFail(Exception exception) {
		if (this.completed) {
			return;
		}
		this.failure = exception;
		this.completed = true;

		if (enqueueTask() == 0) {
			startProcessing();
		}
	}

