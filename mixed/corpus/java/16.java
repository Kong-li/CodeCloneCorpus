synchronized public void refreshMaps() throws IOException {
    if (checkUnsupportedPlatform()) {
      return;
    }

    boolean initMap = constructCompleteMapAtStartup;
    if (initMap) {
      loadComprehensiveMaps();
      // set constructCompleteMapAtStartup to false for testing purposes, allowing incremental updates after initial construction
      constructCompleteMapAtStartup = false;
    } else {
      updateStaticAssociations();
      clearIdentifierMaps();
    }
  }

