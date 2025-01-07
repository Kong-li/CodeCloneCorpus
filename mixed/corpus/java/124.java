  private int doRun() throws IOException {
    // find the active NN
    NamenodeProtocol proxy = null;
    NamespaceInfo nsInfo = null;
    boolean isUpgradeFinalized = false;
    boolean isRollingUpgrade = false;
    RemoteNameNodeInfo proxyInfo = null;
    for (int i = 0; i < remoteNNs.size(); i++) {
      proxyInfo = remoteNNs.get(i);
      InetSocketAddress otherIpcAddress = proxyInfo.getIpcAddress();
      proxy = createNNProtocolProxy(otherIpcAddress);
      try {
        // Get the namespace from any active NN. If you just formatted the primary NN and are
        // bootstrapping the other NNs from that layout, it will only contact the single NN.
        // However, if there cluster is already running and you are adding a NN later (e.g.
        // replacing a failed NN), then this will bootstrap from any node in the cluster.
        nsInfo = getProxyNamespaceInfo(proxy);
        isUpgradeFinalized = proxy.isUpgradeFinalized();
        isRollingUpgrade = proxy.isRollingUpgrade();
        break;
      } catch (IOException ioe) {
        LOG.warn("Unable to fetch namespace information from remote NN at " + otherIpcAddress
            + ": " + ioe.getMessage());
        if (LOG.isDebugEnabled()) {
          LOG.debug("Full exception trace", ioe);
        }
      }
    }

    if (nsInfo == null) {
      LOG.error(
          "Unable to fetch namespace information from any remote NN. Possible NameNodes: "
              + remoteNNs);
      return ERR_CODE_FAILED_CONNECT;
    }

    if (!checkLayoutVersion(nsInfo, isRollingUpgrade)) {
      if(isRollingUpgrade) {
        LOG.error("Layout version on remote node in rolling upgrade ({}, {})"
            + " is not compatible based on minimum compatible version ({})",
            nsInfo.getLayoutVersion(), proxyInfo.getIpcAddress(),
            HdfsServerConstants.MINIMUM_COMPATIBLE_NAMENODE_LAYOUT_VERSION);
      } else {
        LOG.error("Layout version on remote node ({}) does not match this "
            + "node's service layout version ({})", nsInfo.getLayoutVersion(),
            nsInfo.getServiceLayoutVersion());
      }
      return ERR_CODE_INVALID_VERSION;
    }

    System.out.println(
        "=====================================================\n" +
        "About to bootstrap Standby ID " + nnId + " from:\n" +
        "           Nameservice ID: " + nsId + "\n" +
        "        Other Namenode ID: " + proxyInfo.getNameNodeID() + "\n" +
        "  Other NN's HTTP address: " + proxyInfo.getHttpAddress() + "\n" +
        "  Other NN's IPC  address: " + proxyInfo.getIpcAddress() + "\n" +
        "             Namespace ID: " + nsInfo.getNamespaceID() + "\n" +
        "            Block pool ID: " + nsInfo.getBlockPoolID() + "\n" +
        "               Cluster ID: " + nsInfo.getClusterID() + "\n" +
        "           Layout version: " + nsInfo.getLayoutVersion() + "\n" +
        "   Service Layout version: " + nsInfo.getServiceLayoutVersion() + "\n" +
        "       isUpgradeFinalized: " + isUpgradeFinalized + "\n" +
        "         isRollingUpgrade: " + isRollingUpgrade + "\n" +
        "=====================================================");

    NNStorage storage = new NNStorage(conf, dirsToFormat, editUrisToFormat);

    if (!isUpgradeFinalized) {
      // the remote NameNode is in upgrade state, this NameNode should also
      // create the previous directory. First prepare the upgrade and rename
      // the current dir to previous.tmp.
      LOG.info("The active NameNode is in Upgrade. " +
          "Prepare the upgrade for the standby NameNode as well.");
      if (!doPreUpgrade(storage, nsInfo)) {
        return ERR_CODE_ALREADY_FORMATTED;
      }
    } else if (!format(storage, nsInfo, isRollingUpgrade)) { // prompt the user to format storage
      return ERR_CODE_ALREADY_FORMATTED;
    }

    // download the fsimage from active namenode
    int download = downloadImage(storage, proxy, proxyInfo, isRollingUpgrade);
    if (download != 0) {
      return download;
    }

    // finish the upgrade: rename previous.tmp to previous
    if (!isUpgradeFinalized) {
      doUpgrade(storage);
    }

    if (inMemoryAliasMapEnabled) {
      return formatAndDownloadAliasMap(aliasMapPath, proxyInfo);
    } else {
      LOG.info("Skipping InMemoryAliasMap bootstrap as it was not configured");
    }
    return 0;
  }

