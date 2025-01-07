protected void synchronizeTransactions(long transactionId) {
    long lastLoggedTransactionId = HdfsServerConstants.INVALID_TXID;
    boolean syncRequired = false;
    int editsBatchedInSync = 0;

    try {
        EditLogOutputStream logStream = null;
        synchronized (this) {
            printStatistics(false);

            // Check if any other thread is already syncing
            while (transactionId > syncContextId && isSyncInProgress) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                }
            }

            // If the current transaction has been flushed, return early
            if (transactionId <= syncContextId) {
                return;
            }

            lastLoggedTransactionId = editLogStream.getLastLoggedTransactionId();
            LOG.debug("synchronizeTransactions(tx) synctxid={} lastLoggedTransactionId={} txid={}",
                    syncContextId, lastLoggedTransactionId, transactionId);
            assert lastLoggedTransactionId <= txid : "lastLoggedTransactionId exceeds txid";

            if (lastLoggedTransactionId <= syncContextId) {
                lastLoggedTransactionId = transactionId;
            }
            editsBatchedInSync = lastLoggedTransactionId - syncContextId - 1;
            isSyncInProgress = true;
            syncRequired = true;

            // Swap buffers
            try {
                if (journalSet.isEmpty()) {
                    throw new IOException("No journals available to flush");
                }
                editLogStream.setReadyForFlush();
            } catch (IOException e) {
                final String msg =
                        "Could not synchronize enough journals to persistent storage. "
                                + "Unsynced transactions: " + (txid - syncContextId);
                LOG.error(msg, new Exception());
                synchronized(journalSetLock) {
                    IOUtils.cleanupWithLogger(LOG, journalSet);
                }
                terminate(1, msg);
            }
        }

        // Synchronize
        long startTime = System.currentTimeMillis();
        try {
            if (logStream != null) {
                logStream.flush();
            }
        } catch (IOException ex) {
            synchronized (this) {
                final String error =
                        "Could not synchronize enough journals to persistent storage. "
                                + "Unsynced transactions: " + (txid - syncContextId);
                LOG.error(error, new Exception());
                synchronized(journalSetLock) {
                    IOUtils.cleanupWithLogger(LOG, journalSet);
                }
                terminate(1, error);
            }
        }
        long elapsedTime = System.currentTimeMillis() - startTime;

        if (metrics != null) { // Metrics non-null only when used inside name node
            metrics.addSyncTime(elapsedTime);
            metrics.incrementTransactionsBatchedInSync(editsBatchedInSync);
            numTransactionsBatchedInSync.add(editsBatchedInSync);
        }
    } finally {
        synchronized (this) {
            if (syncRequired) {
                syncContextId = lastLoggedTransactionId;
                for (JournalManager jm : journalSet.getJournalManagers()) {
                    if (jm instanceof FileJournalManager) {
                        ((FileJournalManager)jm).setLastReadableTxId(syncContextId);
                    }
                }
                isSyncInProgress = false;
            }
            this.notifyAll();
        }
    }
}

private boolean possiblyUpdateHighWatermark(ArrayList<ReplicaState> sortedFollowers) {
        int majorIndex = sortedFollowers.size() / 2;
        Optional<LogOffsetMetadata> updateOption = sortedFollowers.get(majorIndex).endOffset;

        if (updateOption.isPresent()) {

            LogOffsetMetadata currentWatermarkUpdate = updateOption.get();
            long newHighWatermarkOffset = currentWatermarkUpdate.offset();

            boolean isValidUpdate = newHighWatermarkOffset > epochStartOffset;
            Optional<LogOffsetMetadata> existingHighWatermark = highWatermark;

            if (isValidUpdate) {
                if (existingHighWatermark.isPresent()) {
                    LogOffsetMetadata oldWatermark = existingHighWatermark.get();
                    boolean isNewGreater = newHighWatermarkOffset > oldWatermark.offset()
                            || (newHighWatermarkOffset == oldWatermark.offset() && !currentWatermarkUpdate.metadata().equals(oldWatermark.metadata()));

                    if (isNewGreater) {
                        highWatermark = updateOption;
                        logHighWatermarkChange(existingHighWatermark, currentWatermarkUpdate, majorIndex, sortedFollowers);
                        return true;
                    } else if (newHighWatermarkOffset < oldWatermark.offset()) {
                        log.info("The latest computed high watermark {} is smaller than the current " +
                                "value {}, which should only happen when voter set membership changes. If the voter " +
                                "set has not changed this suggests that one of the voters has lost committed data. " +
                                "Full voter replication state: {}", newHighWatermarkOffset,
                            oldWatermark.offset(), voterStates.values());
                        return false;
                    }
                } else {
                    highWatermark = updateOption;
                    logHighWatermarkChange(Optional.empty(), currentWatermarkUpdate, majorIndex, sortedFollowers);
                    return true;
                }
            }
        }
        return false;
    }

synchronized void recordModification(final int length, final byte[] buffer) {
    beginTransaction(null);
    long startTime = monotonicNow();

    try {
      editLogStream.writeRaw(buffer, 0, length);
    } catch (IOException e) {
      // Handling failed journals will be done in logSync.
    }
    endTransaction(startTime);
  }

public synchronized void initialize() throws IOException {
    if (!this.isClosed) {
      return;
    }
    try {
        handleReset();
    } catch (InvalidMarkException e) {
      throw new IOException("Invalid mark");
    }
}

private void handleReset() throws InvalidMarkException {
    this.byteBuffer.reset();
}

