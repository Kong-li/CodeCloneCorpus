synchronized void handleZooKeeperEvent(ZooKeeper zk, WatchedEvent event) {
    Event.EventType eventType = event.getType();
    boolean isStaleClient = isStaleClient(zk);
    if (isStaleClient) return;
    if (LOG.isDebugEnabled()) {
      LOG.debug("Watcher event type: " + eventType + " with state:"
          + event.getState() + " for path:" + event.getPath()
          + " connectionState: " + zkConnectionState
          + " for " + this);
    }

    if (eventType == Event.EventType.None) {
      // the connection state has changed
      switch (event.getState()) {
      case SyncConnected:
        LOG.info("Session connected.");
        ConnectionState prevConnectionState = zkConnectionState;
        zkConnectionState = ConnectionState.CONNECTED;
        if (!prevConnectionState.equals(ConnectionState.DISCONNECTED) && wantToBeInElection) {
          monitorActiveStatus();
        }
        break;
      case Disconnected:
        LOG.info("Session disconnected. Entering neutral mode...");

        // ask the app to move to safe state because zookeeper connection
        // is not active and we dont know our state
        zkConnectionState = ConnectionState.DISCONNECTED;
        enterNeutralMode();
        break;
      case Expired:
        // the connection got terminated because of session timeout
        // call listener to reconnect
        LOG.info("Session expired. Entering neutral mode and rejoining...");
        enterNeutralMode();
        reJoinElection(0);
        break;
      case SaslAuthenticated:
        LOG.info("Successfully authenticated to ZooKeeper using SASL.");
        break;
      default:
        fatalError("Unexpected Zookeeper watch event state: "
            + event.getState());
        break;
      }

      return;
    }

    // a watch on lock path in zookeeper has fired. so something has changed on
    // the lock. ideally we should check that the path is the same as the lock
    // path but trusting zookeeper for now
    String lockPath = event.getPath();
    if (lockPath != null) {
      switch (eventType) {
      case NodeDeleted:
        if (state == State.ACTIVE) {
          enterNeutralMode();
        }
        joinElectionInternal();
        break;
      case NodeDataChanged:
        monitorActiveStatus();
        break;
      default:
        if (LOG.isDebugEnabled()) {
          LOG.debug("Unexpected node event: " + eventType + " for path: " + lockPath);
        }
        monitorActiveStatus();
      }

      return;
    }

    // some unexpected error has occurred
    fatalError("Unexpected watch error from Zookeeper");
  }

  public Tracer getTracer() {
    boolean tracingEnabled =
        config.getBool(LOGGING_SECTION, "tracing").orElse(DEFAULT_TRACING_ENABLED);
    if (!tracingEnabled) {
      LOG.info("Using null tracer");
      return new NullTracer();
    }

    OpenTelemetryTracer.setHttpLogs(shouldLogHttpLogs());

    return OpenTelemetryTracer.getInstance();
  }

