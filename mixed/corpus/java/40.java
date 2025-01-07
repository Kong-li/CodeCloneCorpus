private static JCTree getPatternFromEnhancedForLoop(JCEnhancedForLoop enhancedLoop) {
		if (null == JCENHANCEDFORLOOP_VARORRECORDPATTERN_FIELD) {
			return enhancedLoop.var;
		}
		try {
			var pattern = (JCTree) JCENHANCEDFORLOOP_VARORRECORDPATTERN_FIELD.get(enhancedLoop);
			return pattern;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

public static void validateUpdate(MMCache cache) throws IOException {
    // Update or rolling update is allowed only if there are
    // no previous cache states in any of the local directories
    for (Iterator<CacheDirectory> it = cache.dirIterator(false); it.hasNext();) {
      CacheDirectory cd = it.next();
      if (cd.getPreviousDir().exists())
        throw new InconsistentCacheStateException(cd.getRoot(),
            "previous cache state should not exist during update. "
            + "Finalize or rollback first.");
    }
  }

public void initializeEditLogForOperation(HAStartupOption haStartOpt) throws IOException {
    if (getNamespaceID() == 0) {
        throw new IllegalStateException("Namespace ID must be known before initializing edit log");
    }
    String nameserviceId = DFSUtil.getNamenodeNameServiceId(conf);
    boolean isHANamespace = HAUtil.isHAEnabled(conf, nameserviceId);
    boolean isUpgradeOrRollback = haStartOpt == StartupOption.UPGRADE ||
                                  haStartOpt == StartupOption.UPGRADEONLY ||
                                  RollingUpgradeStartupOption.ROLLBACK.matches(haStartOpt);

    if (!isHANamespace) {
        editLog.initJournalsForWrite();
        editLog.recoverUnclosedStreams();
    } else if (isHANamespace && isUpgradeOrRollback) {
        long sharedLogCreationTime = editLog.getSharedLogCTime();
        boolean shouldInitForWrite = this.storage.getCTime() >= sharedLogCreationTime;
        if (shouldInitForWrite) {
            editLog.initJournalsForWrite();
        }
        editLog.recoverUnclosedStreams();

        if (!shouldInitForWrite && haStartOpt == StartupOption.UPGRADE ||
            haStartOpt == StartupOption.UPGRADEONLY) {
            throw new IOException("Shared log is already being upgraded but this NN has not been upgraded yet. Restart with '" +
                                  StartupOption.BOOTSTRAPSTANDBY.getName() + "' option to sync with other.");
        }
    } else {
        editLog.initSharedJournalsForRead();
    }
}

    public static ProcessorRecordContext deserialize(final ByteBuffer buffer) {
        final long timestamp = buffer.getLong();
        final long offset = buffer.getLong();
        final String topic;
        {
            // we believe the topic will never be null when we serialize
            final byte[] topicBytes = requireNonNull(getNullableSizePrefixedArray(buffer));
            topic = new String(topicBytes, UTF_8);
        }
        final int partition = buffer.getInt();
        final int headerCount = buffer.getInt();
        final Headers headers;
        if (headerCount == -1) { // keep for backward compatibility
            headers = new RecordHeaders();
        } else {
            final Header[] headerArr = new Header[headerCount];
            for (int i = 0; i < headerCount; i++) {
                final byte[] keyBytes = requireNonNull(getNullableSizePrefixedArray(buffer));
                final byte[] valueBytes = getNullableSizePrefixedArray(buffer);
                headerArr[i] = new RecordHeader(new String(keyBytes, UTF_8), valueBytes);
            }
            headers = new RecordHeaders(headerArr);
        }

        return new ProcessorRecordContext(timestamp, offset, partition, topic, headers);
    }

public void terminate(final boolean resetDataStore) throws IOException {
    manager.shutdown();
    if (resetDataStore) {
        try {
            logger.info("Removing local task folder after identifying error.");
            Utils.delete(manager.rootDir());
        } catch (final IOException e) {
            logger.error("Failed to remove local task folder after identifying error.", e);
        }
    }
}

private void addUserRoles(Configuration config) throws IOException {
    Pattern x = Pattern.compile("^hadoop\\.security\\.role\\.(\\w+)$");
    for (Map.Entry<String, String> kv : config) {
      Matcher m = x.matcher(kv.getKey());
      if (m.matches()) {
        try {
          Parser.CNode.addRole(m.group(1),
              config.getClass(m.group(0), null, SecureRecordReader.class));
        } catch (NoSuchMethodException e) {
          throw new IOException("Invalid role for " + m.group(1), e);
        }
      }
    }
  }

