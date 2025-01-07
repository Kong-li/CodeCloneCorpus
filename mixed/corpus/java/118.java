  private static List<Node> coreResolve(List<String> hostNames) {
    List<Node> nodes = new ArrayList<Node>(hostNames.size());
    List<String> rNameList = dnsToSwitchMapping.resolve(hostNames);
    if (rNameList == null || rNameList.isEmpty()) {
      for (String hostName : hostNames) {
        nodes.add(new NodeBase(hostName, NetworkTopology.DEFAULT_RACK));
      }
      LOG.info("Got an error when resolve hostNames. Falling back to "
          + NetworkTopology.DEFAULT_RACK + " for all.");
    } else {
      for (int i = 0; i < hostNames.size(); i++) {
        if (Strings.isNullOrEmpty(rNameList.get(i))) {
          // fallback to use default rack
          nodes.add(new NodeBase(hostNames.get(i),
              NetworkTopology.DEFAULT_RACK));
          LOG.debug("Could not resolve {}. Falling back to {}",
              hostNames.get(i), NetworkTopology.DEFAULT_RACK);
        } else {
          nodes.add(new NodeBase(hostNames.get(i), rNameList.get(i)));
          LOG.debug("Resolved {} to {}", hostNames.get(i), rNameList.get(i));
        }
      }
    }
    return nodes;
  }

  public void write(DataOutput out) throws IOException {
    conf.write(out);
    Text.writeString(out, src.toString());
    Text.writeString(out, dst.toString());
    Text.writeString(out, mount);
    out.writeBoolean(forceCloseOpenFiles);
    out.writeBoolean(useMountReadOnly);
    out.writeInt(mapNum);
    out.writeInt(bandwidthLimit);
    out.writeInt(trashOpt.ordinal());
    out.writeLong(delayDuration);
    out.writeInt(diffThreshold);
  }

private void haltProcessors() {
    workerPool.shutdown();
    boolean controlled = getSettings().getBoolean(
        HadoopConfiguration.JobHistoryServer.JHS_RECOVERY_CONTROLLED,
        HadoopConfiguration.JobHistoryServer.DEFAULT_JHS_RECOVERY_CONTROLLED);
    // if recovery on restart is supported then leave outstanding processes
    // to the next start
    boolean needToStop = context.getJobStateStore().canResume()
        && !context.getDecommissioned() && controlled;
    // kindly request to end
    for (JobProcessor processor : jobProcessors.values()) {
      if (needToStop) {
        processor.terminateProcessing();
      } else {
        processor.completeProcessing();
      }
    }
    while (!workerPool.isTerminated()) { // wait for all workers to finish
      for (JobId jobId : jobProcessors.keySet()) {
        LOG.info("Waiting for processing to complete for " + jobId);
      }
      try {
        if (!workerPool.awaitTermination(30, TimeUnit.SECONDS)) {
          workerPool.shutdownNow(); // send interrupt to hasten them along
        }
      } catch (InterruptedException e) {
        LOG.warn("Processing halt interrupted!");
        break;
      }
    }
    for (JobId jobId : jobProcessors.keySet()) {
      LOG.warn("Some data may not have been processed for " + jobId);
    }
  }

