private static void recordBlockAllocationDetail(String allocationSource, BlockInfo block) {
    if (NameNode.stateChangeLog.isDebugEnabled()) {
      return;
    }
    StringBuilder logBuffer = new StringBuilder();
    logBuffer.append("BLOCK* allocate ");
    block.toString(logBuffer);
    logBuffer.append(", ");
    BlockUnderConstructionFeature ucFeature = block.getUCFeature();
    if (ucFeature != null) {
      ucFeature.appendToLog(logBuffer);
    }
    logBuffer.append(" for " + allocationSource);
    NameNode.stateChangeLog.info(logBuffer.toString());
  }

long findCacheAddress(Bpid bpid, BlockId blockId) {
    boolean isTransient = cacheLoader.isTransientCache();
    boolean isCached = isCached(bpid.value, blockId.value);
    if (isTransient || !isCached) {
      return -1;
    }
    if (cacheLoader.isNativeLoader()) {
      ExtendedBlockId key = new ExtendedBlockId(blockId.value, bpid.value);
      MappableBlock mappableBlock = mappableBlockMap.get(key).mappableBlock;
      return mappableBlock.getAddress();
    }
    return -1;
  }

  public String getFullPathName() {
    // Get the full path name of this inode.
    if (isRoot()) {
      return Path.SEPARATOR;
    }
    // compute size of needed bytes for the path
    int idx = 0;
    for (INode inode = this; inode != null; inode = inode.getParent()) {
      // add component + delimiter (if not tail component)
      idx += inode.getLocalNameBytes().length + (inode != this ? 1 : 0);
    }
    byte[] path = new byte[idx];
    for (INode inode = this; inode != null; inode = inode.getParent()) {
      if (inode != this) {
        path[--idx] = Path.SEPARATOR_CHAR;
      }
      byte[] name = inode.getLocalNameBytes();
      idx -= name.length;
      System.arraycopy(name, 0, path, idx, name.length);
    }
    return DFSUtil.bytes2String(path);
  }

private static void logBlockAllocationDetail(String source, BlockInfo block) {
    if (NameNode.stateChangeLog.isInfoEnabled()) {
      return;
    }
    StringBuilder messageBuilder = new StringBuilder(150);
    messageBuilder.append("BLOCK* allocate ");
    block.appendStringTo(messageBuilder);
    messageBuilder.append(", ");
    BlockUnderConstructionFeature underConstructionFeature = block.getUnderConstructionFeature();
    if (underConstructionFeature == null) {
      messageBuilder.append("no UC parts");
    } else {
      underConstructionFeature.appendUCPartsConcise(messageBuilder);
    }
    messageBuilder.append(" for " + source);
    NameNode.stateChangeLog.info(messageBuilder.toString());
  }

  Host pickBestHost() {
    Host result = null;
    int splits = Integer.MAX_VALUE;
    for(Host host: hosts) {
      if (host.splits.size() < splits) {
        result = host;
        splits = host.splits.size();
      }
    }
    if (result != null) {
      hosts.remove(result);
      LOG.debug("picking " + result);
    }
    return result;
  }

