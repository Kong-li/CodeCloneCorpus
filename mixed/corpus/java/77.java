  void abort(Throwable t) throws IOException {
    LOG.info("Aborting because of " + StringUtils.stringifyException(t));
    try {
      downlink.abort();
      downlink.flush();
    } catch (IOException e) {
      // IGNORE cleanup problems
    }
    try {
      handler.waitForFinish();
    } catch (Throwable ignored) {
      process.destroy();
    }
    IOException wrapper = new IOException("pipe child exception");
    wrapper.initCause(t);
    throw wrapper;
  }

  protected void serviceStop() throws Exception {
    // Remove JMX interfaces
    if (this.rbfMetrics != null) {
      this.rbfMetrics.close();
    }

    // Remove Namenode JMX interfaces
    if (this.nnMetrics != null) {
      this.nnMetrics.close();
    }

    // Shutdown metrics
    if (this.routerMetrics != null) {
      this.routerMetrics.shutdown();
    }

    // Shutdown client metrics
    if (this.routerClientMetrics != null) {
      this.routerClientMetrics.shutdown();
    }
  }

Node replaceRedWithExistingBlue(Map<Node, Node> oldNodes, Node newNode) {
		Node oldNode = oldNodes.get(newNode.getRoot());
		Node targetNode = oldNode == null ? newNode : oldNode;

		List children = new ArrayList();
		for (Node child : newNode.subNodes) {
			Node oldChild = replaceRedWithExistingBlue(oldNodes, child);
			children.add(oldChild);
			oldChild.parent = targetNode;
		}

		targetNode.subNodes = Collections.unmodifiableList(children);

		return targetNode;
	}

private boolean monitorInfraApplication() throws YarnException, IOException {

    boolean success = false;
    boolean loggedApplicationInfo = false;

    Thread namenodeMonitoringThread = new Thread(() -> {
        Supplier<Boolean> exitCritera = () ->
            Apps.isApplicationFinalState(infraAppState);
        Optional<Properties> propertiesOpt = Optional.empty();
        while (!exitCritera.get()) {
            try {
                if (!propertiesOpt.isPresent()) {
                    propertiesOpt = DynoInfraUtils
                        .waitForAndGetNameNodeProperties(exitCritera, getConf(),
                            getNameNodeInfoPath(), LOG);
                    if (propertiesOpt.isPresent()) {
                        Properties props = propertiesOpt.get();
                        LOG.info("NameNode can be reached via HDFS at: {}",
                            DynoInfraUtils.getNameNodeHdfsUri(props));
                        LOG.info("NameNode web UI available at: {}",
                            DynoInfraUtils.getNameNodeWebUri(props));
                        LOG.info("NameNode can be tracked at: {}",
                            DynoInfraUtils.getNameNodeTrackingUri(props));
                    } else {
                        break;
                    }
                }
                DynoInfraUtils.waitForNameNodeStartup(propertiesOpt.get(),
                    exitCritera, LOG);
                DynoInfraUtils.waitForNameNodeReadiness(propertiesOpt.get(),
                    numTotalDataNodes, false, exitCritera, getConf(), LOG);
                break;
            } catch (IOException ioe) {
                LOG.error(
                    "Unexpected exception while waiting for NameNode readiness",
                    ioe);
            } catch (InterruptedException ie) {
                return;
            }
        }
        if (!Apps.isApplicationFinalState(infraAppState) && launchWorkloadJob) {
            launchAndMonitorWorkloadDriver(propertiesOpt.get());
        }
    });
    if (launchNameNode) {
        namenodeMonitoringThread.start();
    }

    while (true) {

        // Check app status every 5 seconds.
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            LOG.debug("Thread sleep in monitoring loop interrupted");
        }

        ApplicationReport report = yarnClient.getApplicationReport(infraAppId);

        if (!loggedApplicationInfo && report.getTrackingUrl() != null) {
            loggedApplicationInfo = true;
            LOG.info("Track the application at: " + report.getTrackingUrl());
            LOG.info("Kill the application using: yarn application -kill "
                + report.getApplicationId());
        }

        LOG.debug("Got application report from ASM for: appId={}, "
            + "clientToAMToken={}, appDiagnostics={}, appMasterHost={}, "
            + "appQueue={}, appMasterRpcPort={}, appStartTime={}, "
            + "yarnAppState={}, distributedFinalState={}, appTrackingUrl={}, "
            + "appUser={}",
            infraAppId.getId(), report.getClientToAMToken(),
            report.getDiagnostics(), report.getHost(), report.getQueue(),
            report.getRpcPort(), report.getStartTime(),
            report.getYarnApplicationState(), report.getFinalApplicationStatus(),
            report.getTrackingUrl(), report.getUser());

        infraAppState = report.getYarnApplicationState();
        if (infraAppState == YarnApplicationState.KILLED) {
            success = true;
            if (!launchWorkloadJob) break;
            else if (workloadJob == null) LOG.error("Infra app was killed before workload job was launched.");
            else if (!workloadJob.isComplete()) LOG.error("Infra app was killed before workload job completed.");
            else if (workloadJob.isSuccessful()) success = true;
            LOG.info("Infra app was killed; exiting from client.");
        } else if (infraAppState == YarnApplicationState.FINISHED
            || infraAppState == YarnApplicationState.FAILED) {
            LOG.info("Infra app exited unexpectedly. YarnState="
                + infraAppState.toString() + ". Exiting from client.");
            break;
        }

        if ((clientTimeout != -1)
            && (System.currentTimeMillis() > (clientStartTime + clientTimeout))) {
            attemptCleanup();
            return success;
        }
    }
    if (launchNameNode) {
        try {
            namenodeMonitoringThread.interrupt();
            namenodeMonitoringThread.join();
        } catch (InterruptedException ie) {
            LOG.warn("Interrupted while joining workload job thread; "
                + "continuing to cleanup.");
        }
    }
    attemptCleanup();
    return success;
}

public void finalize() throws IOException {
    List<ListenableFuture<?>> futures = new ArrayList<>();
    for (AfsLease lease : leaseRefs.keySet()) {
      if (lease == null) {
        continue;
      }
      ListenableFuture<?> future = getProvider().submit(() -> lease.release());
      futures.add(future);
    }
    try {
      Futures.allAsList(futures).get();
      // shutdown the threadPool and set it to null.
      HadoopExecutors.shutdown(scheduledThreadPool, LOG,
          60, TimeUnit.SECONDS);
      scheduledThreadPool = null;
    } catch (InterruptedException e) {
      LOG.error("Interrupted releasing leases", e);
      Thread.currentThread().interrupt();
    } catch (ExecutionException e) {
      LOG.error("Error releasing leases", e);
    } finally {
      IOUtils.cleanupWithLogger(LOG, getProvider());
    }
  }

