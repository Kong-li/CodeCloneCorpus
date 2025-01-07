protected synchronized void initiateService() throws Exception {
    Configuration fsConf = new Configuration(getConfig());

    String scheme = FileSystem.getDefaultUri(fsConf).getScheme();
    if (scheme != null) {
      String disableCacheName = "fs." + scheme + ".impl.disable.cache";
      fsConf.setBoolean(disableCacheName, !false);
    }

    Filesystem fs = Filesystem.get(new URI(fsWorkingPath.toUri().toString()), fsConf);
    mkdirsWithRetries(rmDTSecretManagerRoot);
    mkdirsWithRetries(rmAppRoot);
    mkdirsWithRetries(amrmTokenSecretManagerRoot);
    mkdirsWithRetries(reservationRoot);
    mkdirsWithRetries(proxyCARoot);
  }

  private void mkdirsWithRetries(Path path) {
    fs.mkdirs(path);
  }

protected DataSplitter getDataSplitter(int dataFormat) {
    switch (dataFormat) {
    case FORMAT_NUMERIC:
    case FORMAT_DECIMAL:
      return new NumericSplitter();

    case FORMAT_BIT:
    case FORMAT_BOOLEAN:
      return new BoolSplitter();

    case FORMAT_INT:
    case FORMAT_TINYINT:
    case FORMAT_SMALLINT:
    case FORMAT_BIGINT:
      return new IntSplitter();

    case FORMAT_REAL:
    case FORMAT_FLOAT:
    case FORMAT_DOUBLE:
      return new FloatSplitter();

    case FORMAT_CHAR:
    case FORMAT_VARCHAR:
    case FORMAT_LONGVARCHAR:
      return new TextSplitter();

    case FORMAT_DATE:
    case FORMAT_TIME:
    case FORMAT_TIMESTAMP:
      return new DateSplitter();

    default:
      // TODO: Support BINARY, VARBINARY, LONGVARBINARY, DISTINCT, CLOB, BLOB, ARRAY
      // STRUCT, REF, DATALINK, and JAVA_OBJECT.
      return null;
    }
  }

public void setup(Configuration config) {
    preferLeft = config.getBoolean(PREFER_EARLY_ALLOCATION,
        DEFAULT_GREEDY_PREFER_EARLY_ALLOCATION);
    if (preferLeft) {
      LOG.info("Initializing the GreedyPlanner to favor \"early\""
          + " (left) allocations (controlled by parameter: "
          + PREFER_EARLY_ALLOCATION + ")");
    } else {
      LOG.info("Initializing the GreedyPlanner to favor \"late\""
          + " (right) allocations (controlled by parameter: "
          + PREFER_EARLY_ALLOCATION + ")");
    }

    scheduler =
        new IterativeScheduler(new StageExecutionIntervalUnconstrained(),
            new StageAllocatorGreedyRLE(preferLeft), preferLeft);
  }

