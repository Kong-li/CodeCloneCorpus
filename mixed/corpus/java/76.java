public List<T> fetchNItems(int num) {
    if (num >= size) {
        return extractAllElements();
    }
    List<T> resultList = new ArrayList<>(num);
    if (num == 0) {
        return resultList;
    }

    boolean finished = false;
    int currentBucketIndex = 0;

    while (!finished) {
        LinkedElement<T> currentNode = entries[currentBucketIndex];
        while (currentNode != null) {
            resultList.add(currentNode.element);
            currentNode = currentNode.next;
            entries[currentBucketIndex] = currentNode;
            size--;
            modification++;
            if (--num == 0) {
                finished = true;
                break;
            }
        }
        currentBucketIndex++;
    }

    reduceStorageIfRequired();
    return resultList;
}

public long getClearedCount() {
    long clearedCount = 0;

    for (CBSection infoBlock : blockSections) {
      if (infoBlock.isCleared()) clearedCount++;
    }

    for (CBSection checkBlock : checkSections) {
      if (checkBlock.isCleared()) clearedCount++;
    }

    return clearedCount;
}

protected QJournalService createConnector() throws IOException {
    final Configuration confCopy = new Configuration(conf);

    // Need to set NODELAY or else batches larger than MTU can trigger
    // 40ms nailing delays.
    confCopy.setBoolean(CommonConfigurationKeysPublic.IPC_CLIENT_TCPNODELAY_KEY, true);
    RPC.setProtocolEngine(confCopy,
        QJournalServicePB.class, ProtobufRpcEngine3.class);
    return SecurityUtil.doAsLoginUser(
        (PrivilegedExceptionAction<QJournalService>) () -> {
          RPC.setProtocolEngine(confCopy,
              QJournalServicePB.class, ProtobufRpcEngine3.class);
          QJournalServicePB pbproxy = RPC.getProxy(
              QJournalServicePB.class,
              RPC.getProtocolVersion(QJournalServicePB.class),
              addr, confCopy);
          return new QJournalServiceTranslatorPB(pbproxy);
        });
  }

