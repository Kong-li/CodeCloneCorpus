  private static URI createHttpUri(String rawHost) {
    int slashIndex = rawHost.indexOf('/');
    String host = slashIndex == -1 ? rawHost : rawHost.substring(0, slashIndex);
    String path = slashIndex == -1 ? null : rawHost.substring(slashIndex);

    try {
      return new URI("http", host, path, null);
    } catch (URISyntaxException e) {
      throw new UncheckedIOException(new IOException(e));
    }
  }

protected int processPoliciesXmlAndSave(String configXml) {
    try {
        List<FederationQueueWeight> queueWeights = parseConfigByXml(configXml);
        MemoryPageUtils<FederationQueueWeight> memoryUtil = new MemoryPageUtils<>(15);
        queueWeights.forEach(weight -> memoryUtil.addToMemory(weight));
        int pageCount = memoryUtil.getPages();
        for (int index = 0; index < pageCount; index++) {
            List<FederationQueueWeight> weights = memoryUtil.readFromMemory(index);
            BatchSaveFederationQueuePoliciesRequest request = BatchSaveFederationQueuePoliciesRequest.newInstance(weights);
            ResourceManagerAdministrationProtocol adminService = createAdminService();
            BatchSaveFederationQueuePoliciesResponse result = adminService.batchSaveFederationQueuePolicies(request);
            System.out.println("page <" + (index + 1) + "> : " + result.getMessage());
        }
    } catch (Exception exception) {
        LOG.error("BatchSaveFederationQueuePolicies error", exception);
    }
    return EXIT_ERROR;
}

private void cleanKeysWithSpecificPrefix(String keyPrefix) throws IOException {
    WriteBatch batch = null;
    LeveldbIterator iter = null;
    try {
      iter = new LeveldbIterator(db);
      try {
        batch = db.createWriteBatch();
        iter.seek(toBytes(keyPrefix));
        while (iter.hasNext()) {
          byte[] currentKey = iter.next().getKey();
          String keyStr = toJavaString(currentKey);
          if (!keyStr.startsWith(keyPrefix)) {
            break;
          }
          batch.delete(currentKey);
          LOG.debug("clean {} from leveldb", keyStr);
        }
        db.write(batch);
      } catch (DBException e) {
        throw new IOException(e);
      } finally {
        closeIfNotNull(batch);
      }
    } catch (DBException e) {
      throw new IOException(e);
    } finally {
      closeIfNotNull(iter);
    }
  }

private byte[] toBytes(String prefix) {
    return bytes(prefix);
}

private String toJavaString(byte[] key) {
    return asString(key);
}

private void closeIfNotNull(WriteBatch batch) {
    if (batch != null) {
        batch.close();
    }
}

private void closeIfNotNull(LeveldbIterator iter) {
    if (iter != null) {
        iter.close();
    }
}

