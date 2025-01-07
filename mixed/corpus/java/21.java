    Node selectReadReplica(final TopicPartition partition, final Node leaderReplica, final long currentTimeMs) {
        Optional<Integer> nodeId = subscriptions.preferredReadReplica(partition, currentTimeMs);

        if (nodeId.isPresent()) {
            Optional<Node> node = nodeId.flatMap(id -> metadata.fetch().nodeIfOnline(partition, id));
            if (node.isPresent()) {
                return node.get();
            } else {
                log.trace("Not fetching from {} for partition {} since it is marked offline or is missing from our metadata," +
                        " using the leader instead.", nodeId, partition);
                // Note that this condition may happen due to stale metadata, so we clear preferred replica and
                // refresh metadata.
                requestMetadataUpdate(metadata, subscriptions, partition);
                return leaderReplica;
            }
        } else {
            return leaderReplica;
        }
    }

public synchronized void terminate() throws IOException {
    for (LinkInfo<T> li : nodeNetworkLinks) {
      if (li.link != null) {
        if (li.link instanceof Dismissable) {
          ((Dismissable)li.link).dismiss();
        } else {
          Network.stopLink(li.link);
        }
        // Set to null to avoid the failoverLink having to re-do the dismiss
        // if it is sharing a link instance
        li.link = null;
      }
    }
    failoverLink.terminate();
    nnMonitoringThreadPool.shutdown();
  }

void configureAccessControlLists(Configuration settings) {
    Map<String, HashMap<KeyOperationType, AccessControlSet>> temporaryKeyAcls = new HashMap<>();
    Map<String, String> allKeyACLs = settings.getValByRegex(KMSConfig.KEY_ACL_PREFIX_REGEX);

    for (Map.Entry<String, String> keyACL : allKeyACLS.entrySet()) {
        final String entryKey = keyACL.getKey();
        if (entryKey.startsWith(KMSConfig.KEY_ACL_PREFIX) && entryKey.contains(".")) {
            final int keyNameStartIndex = KMSConfig.KEY_ACL_PREFIX.length();
            final int keyNameEndIndex = entryKey.lastIndexOf(".");

            if (keyNameStartIndex < keyNameEndIndex) {
                final String aclString = keyACL.getValue();
                final String keyName = entryKey.substring(keyNameStartIndex, keyNameEndIndex);
                final String operationTypeStr = entryKey.substring(keyNameEndIndex + 1);

                try {
                    final KeyOperationType operationType = KeyOperationType.valueOf(operationTypeStr);

                    HashMap<KeyOperationType, AccessControlSet> aclMap;
                    if (temporaryKeyAcls.containsKey(keyName)) {
                        aclMap = temporaryKeyAcls.get(keyName);
                    } else {
                        aclMap = new HashMap<>();
                        temporaryKeyAcls.put(keyName, aclMap);
                    }

                    aclMap.put(operationType, new AccessControlSet(aclString));
                    LOG.info("KEY_NAME '{}' KEY_OP '{}' ACL '{}'", keyName, operationType, aclString);
                } catch (IllegalArgumentException e) {
                    LOG.warn("Invalid key Operation '{}'", operationTypeStr);
                }
            } else {
                LOG.warn("Invalid key name '{}'", entryKey);
            }
        }
    }

    final Map<KeyOperationType, AccessControlSet> defaultACLs = new HashMap<>();
    final Map<KeyOperationType, AccessControlSet> whitelistACLs = new HashMap<>();

    for (KeyOperationType operation : KeyOperationType.values()) {
        parseAclsWithPrefix(settings, KMSConfig.DEFAULT_KEY_ACL_PREFIX, operation, defaultACLs);
        parseAclsWithPrefix(settings, KMSConfig.WHITELIST_KEY_ACL_PREFIX, operation, whitelistACLs);
    }

    defaultKeyACLs = defaultACLs;
    whitelistKeyACLs = whitelistACLs;
}

  private boolean shouldFindObserver() {
    // lastObserverProbeTime > 0 means we tried, but did not find any
    // Observers yet
    // If lastObserverProbeTime <= 0, previous check found observer, so
    // we should not skip observer read.
    if (lastObserverProbeTime > 0) {
      return Time.monotonicNow() - lastObserverProbeTime
          >= observerProbeRetryPeriodMs;
    }
    return true;
  }

public void initializeTopicsAndRegexes(int topicCount, int regexCount) {
        Random random = new Random();

        MetadataDelta delta = new MetadataDelta(MetadataImage.EMPTY);
        for (int i = 0; i < topicCount; i++) {
            String topicName =
                WORDS.get(random.nextInt(WORDS.size())) + "_" +
                WORDS.get(random.nextInt(WORDS.size())) + "_" +
                i;

            delta.replay(TopicRecord.builder()
                    .topicId(Uuid.randomUuid())
                    .name(topicName)
                    .build());
        }
        this.image = delta.apply(MetadataProvenance.EMPTY);

        Set<String> regexes = new HashSet<>();
        for (int i = 0; i < regexCount; i++) {
            regexes.add(".*" + WORDS.get(random.nextInt(WORDS.size())) + ".*");
        }
        this.regexes = regexes;
    }

  synchronized public void insert(long v) {
    buffer[bufferCount] = v;
    bufferCount++;

    count++;

    if (bufferCount == buffer.length) {
      insertBatch();
      compress();
    }
  }

