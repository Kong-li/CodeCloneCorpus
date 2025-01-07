public void initiateDrainOperation() {
    final DrainStatus status = drainStatus;
    appendLock.lock();
    try {
        if (status == null) {
            drainStatus = DrainStatus.STARTED;
            maybeCompleteDrain();
        }
    } finally {
        appendLock.unlock();
    }
}

private void handleSignal(final String signalName, final Map<Object, Object> jvmSignalHandlers) throws ReflectiveOperationException {
    if (signalConstructor == null) {
        throw new IllegalArgumentException("signal constructor is not initialized");
    }
    Object signal = signalConstructor.newInstance(signalName);
    Object signalHandler = createSignalHandler(jvmSignalHandlers);
    Object oldHandler = signalHandleMethod.invoke(null, signal, signalHandler);
    handleOldHandler(oldHandler, jvmSignalHandlers, signalName);
}

private void handleOldHandler(Object oldHandler, Map<Object, Object> jvmSignalHandlers, String signalName) {
    if (oldHandler != null) {
        jvmSignalHandlers.put(signalName, oldHandler);
    }
}

public String getVolume() throws IOException {
    // Abort early if specified path does not exist
    if (!fileDirectory.exists()) {
      throw new FileNotFoundException("Specified path " + fileDirectory.getPath()
          + " does not exist");
    }

    if (SystemInfo.IS_WINDOWS) {
      // Assume a drive letter for a volume point
      this.volume = fileDirectory.getCanonicalPath().substring(0, 2);
    } else {
      scan();
      checkExitStatus();
      analyzeOutput();
    }

    return volume;
}

