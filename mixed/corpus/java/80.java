private static void integrateBinaryCredentials(CredsManager manager, Config config) {
    String binaryTokenPath =
        config.get(MRJobConfig.MAPREDUCE_JOB_CREDENTIALS_BINARY);
    if (binaryTokenPath != null) {
      Credentials binary;
      try {
        FileSystem fs = FileSystem.getLocal(config);
        Path path = new Path(fs.makeQualified(new Path(binaryTokenPath)).toString());
        binary = Credentials.readTokenStorageFile(path, config);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      // merge the existing credentials with those from the binary file
      manager.mergeCredentials(binary);
    }
  }

private static FileSystemManager createFileSystemService(Properties config) {
    Class<? extends FileSystemManager> defaultFileManagerClass;
    try {
      defaultFileManagerClass =
          (Class<? extends FileSystemManager>) Class
              .forName(HdfsConfiguration.DEFAULT_FILESYSTEM_MANAGER_CLASS);
    } catch (Exception e) {
      throw new HdfsRuntimeException("Invalid default file system manager class"
          + HdfsConfiguration.DEFAULT_FILESYSTEM_MANAGER_CLASS, e);
    }

    FileSystemManager manager =
        ReflectionUtils.newInstance(config.getProperty(
            HdfsConfiguration.FILESYSTEM_MANAGER_CLASS,
            defaultFileManagerClass.getName()), config);
    return manager;
  }

public static RunningJob executeJob(Configuration conf) throws IOException {
    JobClient jobClient = new JobClient(conf);
    RunningJob runningJob = jobClient.submitJob(conf);
    boolean isSuccess = true;
    try {
      isSuccess &= jcMonitorAndPrintJob(conf, runningJob);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    if (!isSuccess) {
      throw new IOException("Job failed!");
    }
    return runningJob;
}

private static boolean jcMonitorAndPrintJob(Configuration conf, RunningJob job) throws InterruptedException {
  return JobClient.runJobMonitoring(conf, job);
}

private void processLogFiles() throws IOException {
        // process logs in ascending order because transactional data from one log may depend on the
        // logs that come before it
        File[] files = directory.listFiles();
        if (files == null) files = new File[0];
        List<File> sortedFiles = Arrays.stream(files).filter(File::isFile).sorted().collect(Collectors.toList());
        for (File file : sortedFiles) {
            if (LogUtils.isIndexFile(file)) {
                // if it is an index file, make sure it has a corresponding .log file
                long offset = LogUtils.offsetFromFile(file);
                File logFile = LogUtils.logFile(directory, offset);
                if (!logFile.exists()) {
                    logger.warn("Found an orphaned index file {}, with no corresponding log file.", file.getAbsolutePath());
                    Files.deleteIfExists(file.toPath());
                }
            } else if (LogUtils.isLogFile(file)) {
                // if it's a log file, process the corresponding log segment
                long baseOffset = LogUtils.offsetFromFile(file);
                boolean newIndexFileCreated = !LogUtils.timeIndexFile(directory, baseOffset).exists();
                LogSegment segment = LogSegment.open(directory, baseOffset, configuration, timestamp, true, 0, false, "");
                try {
                    segment.validate(newIndexFileCreated);
                } catch (NoSuchFileException nsfe) {
                    if (hadCleanShutdown || segment.baseOffset() < recoveryPointCheckpoint) {
                        logger.error("Could not find offset index file corresponding to log file {}, recovering segment and rebuilding index files...", segment.log().file().getAbsolutePath());
                    }
                    recoverSegment(segment);
                } catch (CorruptIndexException cie) {
                    logger.warn("Found a corrupted index file corresponding to log file {} due to {}, recovering segment and rebuilding index files...", segment.log().file().getAbsolutePath(), cie.getMessage());
                    recoverSegment(segment);
                }
                segments.add(segment);
            }
        }
    }

