private long loadSecrets(ServerState context) throws IOException {
    FileStatus[] statuses = fileSystem.listStatus(secretKeysPath);
    long count = 0;
    for (FileStatus status : statuses) {
        String fileName = status.getPath().getName();
        if (fileName.startsWith(SECRET_MASTER_KEY_FILE_PREFIX)) {
            loadSecretMasterKey(context, status.getPath(), status.getLen());
            ++count;
        } else {
            LOGGER.warn("Skipping unexpected file in server secret state: " + status.getPath());
        }
    }
    return count;
}

public void monitor(Clock clock, Condition checkCondition, boolean ignoreAlert) {
    // there may be alerts which need to be triggered if we alerted the previous call to monitor
    triggerPendingResolvedEvents();

    lock.lock();
    try {
        // Handle async disconnections prior to attempting any sends
        processPendingDisconnects();

        // send all the requests we can send now
        long checkDelayMs = attemptSend(clock.currentTimeMillis());

        // check whether the monitoring is still needed by the caller. Note that if the expected completion
        // condition becomes satisfied after the call to shouldBlock() (because of a triggered alert handler),
        // the client will be woken up.
        if (pendingAlerts.isEmpty() && (checkCondition == null || checkCondition.shouldBlock())) {
            // if there are no requests in flight, do not block longer than the retry backoff
            long checkTimeout = Math.min(clock.remainingTime(), checkDelayMs);
            if (client.inFlightRequestCount() == 0)
                checkTimeout = Math.min(checkTimeout, retryBackoffMs);
            client.monitor(checkTimeout, clock.currentTimeMillis());
        } else {
            client.monitor(0, clock.currentTimeMillis());
        }
        clock.update();

        // handle any disconnections by failing the active requests. note that disconnections must
        // be checked immediately following monitor since any subsequent call to client.ready()
        // will reset the disconnect status
        checkDisconnects(clock.currentTimeMillis());
        if (!ignoreAlert) {
            // trigger alerts after checking for disconnections so that the callbacks will be ready
            // to be fired on the next call to monitor()
            maybeTriggerAlert();
        }
        // throw InterruptException if this thread is interrupted
        maybeThrowInterruptException();

        // try again to send requests since buffer space may have been
        // cleared or a connect finished in the monitor
        attemptSend(clock.currentTimeMillis());

        // fail requests that couldn't be sent if they have expired
        failExpiredRequests(clock.currentTimeMillis());

        // clean unsent requests collection to keep the map from growing indefinitely
        unsent.clean();
    } finally {
        lock.unlock();
    }

    // called without the lock to avoid deadlock potential if handlers need to acquire locks
    triggerPendingResolvedEvents();

    metadata.maybeThrowAnyException();
}

long attemptSend(long currentTime) {
    long delayLimit = maxPollTimeoutMs;

    for (Node location : pendingRequests.keys()) {
        var outstanding = pendingRequests.get(location);
        if (!outstanding.isEmpty()) {
            delayLimit = Math.min(delayLimit, client.calculateDelay(location, currentTime));
        }

        ClientRequest nextRequest;
        while ((nextRequest = outstanding.poll()) != null) {
            if (client.isReady(location, currentTime)) {
                client.send(nextRequest, currentTime);
            } else {
                break; // move to the next node
            }
        }
    }

    return delayLimit;
}

private static String convertLogMessage(String userOrPath, Object identifier) {
		assert userOrPath != null;

		StringBuilder builder = new StringBuilder();

		builder.append( userOrPath );
		builder.append( '#' );

		if ( identifier == null ) {
			builder.append( EMPTY );
		}
		else {
			builder.append( identifier );
		}

		return builder.toString();
	}

