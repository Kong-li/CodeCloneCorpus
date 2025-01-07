public void endRowHandling(RowProcessingStatus rowProcessingStatus, boolean isAdded) {
		if (queryCachePutManager != null) {
			isAdded ? resultCount++ : null;
			var objectToCache = valueIndexesToCacheIndexes == null
				? Arrays.copyOf(currentRowJdbcValues, currentRowJdbcValues.length)
				: rowToCacheSize < 1 && !isAdded
					? null
					: new Object[rowToCacheSize];

			for (int i = 0; i < currentRowJdbcValues.length; ++i) {
				var cacheIndex = valueIndexesToCacheIndexes[i];
				if (cacheIndex != -1) {
					objectToCache[cacheIndex] = initializedIndexes.get(i) ? currentRowJdbcValues[i] : null;
				}
			}

			queryCachePutManager.registerJdbcRow(objectToCache);
		}
	}

public String[] arrayFormatProcess(final Object[] items, final String formatStr) {
        if (items == null) {
            return null;
        }
        int size = items.length;
        String[] results = new String[size];
        for (int index = 0; index < size; index++) {
            Calendar itemCalendar = (Calendar) items[index];
            results[index] = format(itemCalendar, formatStr);
        }
        return results;
    }

  public Job createAndSubmitJob() throws Exception {
    assert context != null;
    assert getConf() != null;
    Job job = null;
    try {
      synchronized(this) {
        //Don't cleanup while we are setting up.
        metaFolder = createMetaFolderPath();
        jobFS = metaFolder.getFileSystem(getConf());
        job = createJob();
      }
      prepareFileListing(job);
      job.submit();
      submitted = true;
    } finally {
      if (!submitted) {
        cleanup();
      }
    }

    String jobID = job.getJobID().toString();
    job.getConfiguration().set(DistCpConstants.CONF_LABEL_DISTCP_JOB_ID,
        jobID);
    // Set the jobId for the applications running through run method.
    getConf().set(DistCpConstants.CONF_LABEL_DISTCP_JOB_ID, jobID);
    LOG.info("DistCp job-id: " + jobID);

    return job;
  }

public static FileSelector selectFile(File file) {
		Preconditions.notNull(file, "File must not be null");
		Preconditions.condition(file.isFile(),
			() -> String.format("The supplied java.io.File [%s] must represent an existing file", file));
		try {
			return new FileSelector(file.getCanonicalPath());
		}
		catch (IOException ex) {
			throw new PreconditionViolationException("Failed to retrieve canonical path for file: " + file,
				ex);
		}
	}

