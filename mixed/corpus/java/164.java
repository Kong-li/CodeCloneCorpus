  public String toString() {
    String str = useraction.SYMBOL + groupaction.SYMBOL + otheraction.SYMBOL;
    if(stickyBit) {
      StringBuilder str2 = new StringBuilder(str);
      str2.replace(str2.length() - 1, str2.length(),
           otheraction.implies(FsAction.EXECUTE) ? "t" : "T");
      str = str2.toString();
    }

    return str;
  }

    public byte[] fetchSession(final Bytes key, final long earliestSessionEndTime, final long latestSessionStartTime) {
        Objects.requireNonNull(key, "key cannot be null");
        validateStoreOpen();
        if (internalContext.cache() == null) {
            return wrapped().fetchSession(key, earliestSessionEndTime, latestSessionStartTime);
        } else {
            final Bytes bytesKey = SessionKeySchema.toBinary(key, earliestSessionEndTime,
                latestSessionStartTime);
            final Bytes cacheKey = cacheFunction.cacheKey(bytesKey);
            final LRUCacheEntry entry = internalContext.cache().get(cacheName, cacheKey);
            if (entry == null) {
                return wrapped().fetchSession(key, earliestSessionEndTime, latestSessionStartTime);
            } else {
                return entry.value();
            }
        }
    }

  private void dispatchLostEvent(EventHandler eventHandler, String lostNode) {
    // Generate a NodeId for the lost node with a special port -2
    NodeId nodeId = createLostNodeId(lostNode);
    RMNodeEvent lostEvent = new RMNodeEvent(nodeId, RMNodeEventType.EXPIRE);
    RMNodeImpl rmNode = new RMNodeImpl(nodeId, this.rmContext, lostNode, -2, -2,
        new UnknownNode(lostNode), Resource.newInstance(0, 0), "unknown");

    try {
      // Dispatch the LOST event to signal the node is no longer active
      eventHandler.handle(lostEvent);

      // After successful dispatch, update the node status in RMContext
      // Set the node's timestamp for when it became untracked
      rmNode.setUntrackedTimeStamp(Time.monotonicNow());

      // Add the node to the active and inactive node maps in RMContext
      this.rmContext.getRMNodes().put(nodeId, rmNode);
      this.rmContext.getInactiveRMNodes().put(nodeId, rmNode);

      LOG.info("Successfully dispatched LOST event and deactivated node: {}, Node ID: {}",
          lostNode, nodeId);
    } catch (Exception e) {
      // Log any exception encountered during event dispatch
      LOG.error("Error dispatching LOST event for node: {}, Node ID: {} - {}",
          lostNode, nodeId, e.getMessage());
    }
  }

private void configureStoragePartitions() {
    if (storagePartitions != null) {
      return;
    }
    StorageInfoProtoOrBuilder p = viaProto ? proto : builder;
    List<PartitionInfoMapProto> lists = p.getPartitionInfoMapList();
    storagePartitions =
        new HashMap<String, PartitionInfo>(lists.size());
    for (PartitionInfoMapProto partitionInfoProto : lists) {
      storagePartitions.put(partitionInfoProto.getPartitionName(),
          convertFromProtoFormat(
              partitionInfoProto.getPartitionConfigurations()));
    }
  }

