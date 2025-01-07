private static String extractHostnameFromConfig(Configuration config) {
    String hostAddr = getSocketAddress(config, DFS_DATANODE_HTTP_ADDRESS_KEY);
    if (hostAddr == null) {
      hostAddr = getSocketAddress(config, DFS_DATANODE_HTTPS_ADDRESS_KEY,
          DFS_DATANODE_HTTPS_ADDRESS_DEFAULT);
    }
    return NetUtils.createSocketAddr(hostAddr).getHostString();
}

private String getSocketAddress(Configuration conf, String key, String defaultValue) {
  return conf.getTrimmed(key, defaultValue);
}

public int getConnectionPoolCount() {
    try {
        int count = 0;
        readLock.lock();
        count = pools.size();
        readLock.unlock();
        return count;
    } finally {
        // 确保在任何情况下都解锁
    }
}

public static StateStoreSerializer fetchSerializer(Configuration config) {
    if (config == null) {
      synchronized (StateStoreSerializer.class) {
        boolean needInitialization = defaultSerializer == null;
        if (needInitialization) {
          config = new Configuration();
          defaultSerializer = newSerializer(config);
        }
      }
    } else {
      return newSerializer(config);
    }
    return defaultSerializer;
  }

public static int findAvailableTcpPort() {
    int port = 0;
    try {
      ServerSocket s = new ServerSocket(8080);
      port = s.getLocalPort();
      s.close();
      return port;
    } catch (IOException e) {
      // Could not get an available port. Return default port 0.
    }
    return port;
  }

final protected synchronized long fetchAndNeglect(long length) throws IOException {
    long aggregate = 0;
    while (aggregate < length) {
      if (index >= limit) {
        limit = readSignatureBlock(buffer, 0, maxBlockSize);
        if (limit <= 0) {
          break;
        }
      }
      long fetched = Math.min(limit - index, length - aggregate);
      index += fetched;
      aggregate += fetched;
    }
    return aggregate;
  }

