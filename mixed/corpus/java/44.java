  public static Class<?> loadClass(Configuration conf, String className) {
    Class<?> declaredClass = null;
    try {
      if (conf != null)
        declaredClass = conf.getClassByName(className);
      else
        declaredClass = Class.forName(className);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException("readObject can't find class " + className,
          e);
    }
    return declaredClass;
  }

public void updateLogEntry(LogEntry entry) throws Exception {
    if (maxEntries > 0) {
      byte[] storedLogs = getZkData(logsPath);
      List<LogEntry> logEntries = new ArrayList<>();
      if (storedLogs != null) {
        logEntries = unsafeCast(deserializeObject(storedLogs));
      }
      logEntries.add(entry);
      while (logEntries.size() > maxEntries) {
          logEntries.remove(0);
      }
      safeSetZkData(logsPath, logEntries);
    }
}

