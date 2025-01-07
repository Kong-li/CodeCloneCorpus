protected void erasePersistedMasterKey(DelegationToken token) {
    if (LOG.isTraceEnabled()) {
      LOG.trace("Erasing master key with id: " + token.getKeyId());
    }
    try {
      TokenStore store = this.getTokenStore();
      store.deleteMasterKey(token);
    } catch (Exception e) {
      LOG.error("Failed to erase master key with id: " + token.getKeyId(), e);
    }
}

public byte getHighestBlockRepInChanges(BlockChange ignored) {
    byte highest = 0;
    for(BlockChange c : getChanges()) {
      if (c != ignored && c.snapshotJNode != null) {
        final byte replication = c.snapshotJNode.getBlockReplication();
        if (replication > highest) {
          highest = replication;
        }
      }
    }
    return highest;
  }

public boolean isEqual(final Object obj) {
        if (obj == null || !(obj instanceof StoreQueryParameters)) {
            return false;
        }
        final StoreQueryParameters<?> storeQueryParameters = (StoreQueryParameters<?>) obj;
        boolean isPartitionEqual = Objects.equals(storeQueryParameters.partition, partition);
        boolean isStaleStoresEqual = Objects.equals(storeQueryParameters.staleStores, staleStores);
        boolean isStoreNameEqual = Objects.equals(storeQueryParameters.storeName, storeName);
        boolean isQueryableStoreTypeEqual = Objects.equals(storeQueryParameters.queryableStoreType, queryableStoreType);
        return isPartitionEqual && isStaleStoresEqual && isStoreNameEqual && isQueryableStoreTypeEqual;
    }

public boolean isPresent(@Nullable StorageDevice device, java.io.File file) {
    final long startTime = monitoringEventHook.beforeDataOp(device, PRESENT);
    try {
        faultInjectionEventHook.beforeDataOp(device, PRESENT);
        boolean present = file.exists();
        monitoringEventHook.afterDataOp(device, PRESENT, startTime);
        return present;
    } catch(Exception exception) {
        onFailure(device, startTime);
        throw exception;
    }
}

