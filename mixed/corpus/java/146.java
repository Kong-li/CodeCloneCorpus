  public void readFields(DataInput in) throws IOException {
    super.readFields(in);

    // First clear the map.  Otherwise we will just accumulate
    // entries every time this method is called.
    this.instance.clear();

    // Read the number of entries in the map

    int entries = in.readInt();

    // Then read each key/value pair

    for (int i = 0; i < entries; i++) {
      Writable key = (Writable) ReflectionUtils.newInstance(getClass(
          in.readByte()), getConf());

      key.readFields(in);

      Writable value = (Writable) ReflectionUtils.newInstance(getClass(
          in.readByte()), getConf());

      value.readFields(in);
      instance.put(key, value);
    }
  }

private static void aggregateBMWithBAM(BInfo bm, BInfo bam) {
    bm.setAllocatedResourceMB(
        bm.getAllocatedResourceMB() + bam.getAllocatedResourceMB());
    bm.setAllocatedResourceVCores(
        bm.getAllocatedResourceVCores() + bam.getAllocatedResourceVCores());
    bm.setNumNonBMContainerAllocated(bm.getNumNonBMContainerAllocated()
        + bam.getNumNonBMContainerAllocated());
    bm.setNumBMContainerAllocated(
        bm.getNumBMContainerAllocated() + bam.getNumBMContainerAllocated());
    bm.setAllocatedMemorySeconds(
        bm.getAllocatedMemorySeconds() + bam.getAllocatedMemorySeconds());
    bm.setAllocatedVcoreSeconds(
        bm.getAllocatedVcoreSeconds() + bam.getAllocatedVcoreSeconds());

    if (bm.getState() == YarnApplicationState.RUNNING
        && bam.getState() == bm.getState()) {

      bm.getResourceRequests().addAll(bam.getResourceRequests());

      bm.setAllocatedMB(bm.getAllocatedMB() + bam.getAllocatedMB());
      bm.setAllocatedVCores(bm.getAllocatedVCores() + bam.getAllocatedVCores());
      bm.setReservedMB(bm.getReservedMB() + bam.getReservedMB());
      bm.setReservedVCores(bm.getReservedVCores() + bam.getReservedMB());
      bm.setRunningContainers(
          bm.getRunningContainers() + bam.getRunningContainers());
      bm.setMemorySeconds(bm.getMemorySeconds() + bam.getMemorySeconds());
      bm.setVcoreSeconds(bm.getVcoreSeconds() + bam.getVcoreSeconds());
    }
  }

private void terminateConnection(FTPClient connection) throws IOException {
    if (connection != null && connection.isConnected()) {
      boolean success = clientLogout(connection);
      disconnectClient();
      if (!success) {
        LOG.warn("Failed to log out during disconnection, error code - "
            + connection.getReplyCode());
      }
    }
  }

  private boolean clientLogout(FTPClient client) {
    return !client.logout();
  }

  private void disconnectClient() {
    client.disconnect();
  }

  public MasterKey getNMTokenMasterKey() {
    RegisterNodeManagerResponseProtoOrBuilder p = viaProto ? proto : builder;
    if (this.nmTokenMasterKey != null) {
      return this.nmTokenMasterKey;
    }
    if (!p.hasNmTokenMasterKey()) {
      return null;
    }
    this.nmTokenMasterKey =
        convertFromProtoFormat(p.getNmTokenMasterKey());
    return this.nmTokenMasterKey;
  }

