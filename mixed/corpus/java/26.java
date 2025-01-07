private static Policy retrievePolicy(final ConfigDetails config) {
    final boolean isActive = config.getBoolean(
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.ENABLE_KEY,
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.ENABLE_DEFAULT);
    if (isActive) {
      return Policy.DISABLE;
    }

    String selectedPolicy = config.get(
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.POLICY_KEY,
        HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.POLICY_DEFAULT);
    for (int index = 1; index < Policy.values().length; ++index) {
      final Policy option = Policy.values()[index];
      if (option.name().equalsIgnoreCase(selectedPolicy)) {
        return option;
      }
    }
    throw new HadoopIllegalArgumentException("Invalid configuration value for "
        + HdfsClientConfigKeys.BlockWrite.ReplaceDatanodeOnFailure.POLICY_KEY
        + ": " + selectedPolicy);
}

public synchronized Set<String> getUserRoles(String member) throws IOException {
    Collection<String> roleSet = new TreeSet<String>();

    for (RoleMappingServiceProvider service : serviceList) {
      List<String> roles = Collections.emptyList();
      try {
        roles = service.getUserRoles(member);
      } catch (Exception e) {
        LOG.warn("Unable to get roles for member {} via {} because: {}",
            member, service.getClass().getSimpleName(), e.toString());
        LOG.debug("Stacktrace: ", e);
      }
      if (!roles.isEmpty()) {
        roleSet.addAll(roles);
        if (!combined) break;
      }
    }

    return new TreeSet<>(roleSet);
  }

  public void constructFinalFullcounters() {
    this.fullCounters = new Counters();
    this.finalMapCounters = new Counters();
    this.finalReduceCounters = new Counters();
    this.fullCounters.incrAllCounters(jobCounters);
    for (Task t : this.tasks.values()) {
      Counters counters = t.getCounters();
      switch (t.getType()) {
      case MAP:
        this.finalMapCounters.incrAllCounters(counters);
        break;
      case REDUCE:
        this.finalReduceCounters.incrAllCounters(counters);
        break;
      default:
        throw new IllegalStateException("Task type neither map nor reduce: " +
            t.getType());
      }
      this.fullCounters.incrAllCounters(counters);
    }
  }

public float calculateStatus() {
    float totalProgress = 0.0f;
    boolean lockAcquired = false;

    if (!lockAcquired) {
        this.readLock.lock();
        lockAcquired = true;
    }

    try {
        computeProgress();

        totalProgress += (this.setupProgress * this.setupWeight);
        totalProgress += (this.cleanupProgress * this.cleanupWeight);
        totalProgress += (this.mapProgress * this.mapWeight);
        totalProgress += (this.reduceProgress * this.reduceWeight);
    } finally {
        if (lockAcquired) {
            this.readLock.unlock();
        }
    }

    return totalProgress;
}

