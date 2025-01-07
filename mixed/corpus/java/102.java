public void appendData(TargetPath dest, Path[] sources) throws IOException {
    validateNNStartup();
    if (stateChangeLog.isInfoEnabled()) {
      stateChangeLog.info("*FILE* NameNode.append: source paths {} to destination path {}",
          Arrays.toString(sources), dest);
    }
    namesystem.checkOperation(OperationCategory.WRITE);
    CacheRecord cacheRecord = getCacheRecord();
    if (cacheRecord != null && cacheRecord.isSuccess()) {
      return; // Return previous response
    }
    boolean result = false;

    try {
      namesystem.append(dest, sources, cacheRecord != null);
      result = true;
    } finally {
      RetryCache.setState(cacheRecord, result);
    }
}

