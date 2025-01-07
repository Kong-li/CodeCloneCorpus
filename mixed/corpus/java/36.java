static NetworkConfig createNetworkConfig(SystemConfig sys) {

    long maxSockets = S3AUtils.longOption(sys, MAXIMUM.Sockets,
        DEFAULT_MAXIMUM.Sockets, 1);

    final boolean keepAlive = sys.getBoolean(CONNECTION.KEEPALIVE,
        DEFAULT_CONNECTION.KEEPALIVE);

    // time to acquire a socket from the pool
    Duration acquisitionTimeout = getDuration(sys, SOCKET.Acquisition.TIMEOUT,
        DEFAULT_SOCKETAcquisition.TIMEOUT_DURATION, TimeUnit.MILLISECONDS,
        minimumOperationDuration);

    // set the socket TTL irrespective of whether the socket is in use or not.
    // this can balance requests over different S3 servers, and avoid failed
    // connections. See HADOOP-18845.
    Duration socketTTL = getDuration(sys, SOCKET.TTL,
        DEFAULT_SOCKET.TTL_DURATION, TimeUnit.MILLISECONDS,
        null);

    Duration establishTimeout = getDuration(sys, ESTABLISH.TIMEOUT,
        DEFAULT_ESTABLISH.TIMEOUT_DURATION, TimeUnit.MILLISECONDS,
        minimumOperationDuration);

    // limit on the time a socket can be idle in the pool
    Duration maxIdleTime = getDuration(sys, SOCKET.IDLE.TIME,
        DEFAULT_SOCKET.IDLE.TIME_DURATION, TimeUnit.MILLISECONDS, Duration.ZERO);

    Duration readTimeout = getDuration(sys, SOCKET.READ.TIMEOUT,
        DEFAULT_SOCKET.READ.TIMEOUT_DURATION, TimeUnit.MILLISECONDS,
        minimumOperationDuration);

    final boolean expectContinueEnabled = sys.getBoolean(CONNECTION.EXPECT_CONTINUE,
        CONNECTION.EXPECT_CONTINUE_DEFAULT);

    return new NetworkConfig(
        maxSockets,
        keepAlive,
        acquisitionTimeout,
        socketTTL,
        establishTimeout,
        maxIdleTime,
        readTimeout,
        expectContinueEnabled);
  }

