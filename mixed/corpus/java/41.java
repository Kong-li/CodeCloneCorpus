private void updateBuilderWithLocalValues() {
    boolean hasContainerId = this.existingContainerId != null;
    boolean hasTargetCapability = this.targetCapability != null;

    if (hasContainerId) {
      builder.setContainerId(
          ProtoUtils.convertToProtoFormat(this.existingContainerId));
    }

    if (hasTargetCapability) {
      builder.setCapability(
          ProtoUtils.convertToProtoFormat(this.targetCapability));
    }
}

private void cleanupGlobalCleanerPidFile(Configuration conf, FileSystem fs) {
    String root = conf.get(YarnConfiguration.SHARED_CACHE_ROOT,
            YarnConfiguration.DEFAULT_SHARED_CACHE_ROOT);

    Path pidPath = new Path(root, GLOBAL_CLEANER_PID);

    try {
        fs.delete(pidPath, false);
        LOG.info("Removed the global cleaner pid file at " + pidPath.toString());
    } catch (IOException e) {
        LOG.error(
                "Unable to remove the global cleaner pid file! The file may need "
                        + "to be removed manually.", e);
    }
}

public void freeMemory(BufferPool buffer) {
    try {
      ((SupportsEnhancedBufferAccess)input).freeMemory(buffer);
    }
    catch (ClassCastException e) {
      BufferManager bufferManager = activePools.remove( buffer);
      if (bufferManager == null) {
        throw new IllegalArgumentException("attempted to free a buffer " +
            "that was not allocated by this handler.");
      }
      bufferManager.returnBuffer(buffer);
    }
  }

public FileDescriptor retrieveFileDescriptor() throws IOException {
    boolean isHasFileDescriptor = in instanceof HasFileDescriptor;
    if (isHasFileDescriptor) {
        return ((HasFileDescriptor) in).getFileDescriptor();
    }
    if (in instanceof FileInputStream) {
        FileInputStream fileInputStream = (FileInputStream) in;
        return fileInputStream.getFD();
    } else {
        return null;
    }
}

private static int calculateDuration(Config config) {
    int durationInSeconds =
        config.getInt(NetworkConfiguration.DNS_CACHE_TTL_SECS,
            NetworkConfiguration.DEFAULT_DNS_CACHE_TTL_SECS);
    // non-positive value is invalid; use the default
    if (durationInSeconds <= 0) {
      throw new NetworkIllegalArgumentException("Non-positive duration value: "
          + durationInSeconds
          + ". The cache TTL must be greater than or equal to zero.");
    }
    return durationInSeconds;
  }

