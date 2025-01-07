public void initiateEditionVersion(int ediVersion) throws IOException {
    try {
        AttributesImpl attrs = new AttributesImpl();
        contentHandler.startElement("", "", "EDITS_VERSION", attrs);
        String versionStr = Integer.toString(ediVersion);
        addString(versionStr);
        StringBuilder builder = new StringBuilder(versionStr);
        contentHandler.endElement("", "", "EDITS_VERSION");
    } catch (SAXException e) {
        throw new IOException("SAX error: " + e.getMessage());
    }
}

public int delayFor(final JobId job) {
        if (jobDelayTotals.isEmpty()) {
            LOG.error("delayFor was called on a JobManagerState {} that does not support delay computations.", processId);
            throw new UnsupportedOperationException("Delay computation was not requested for JobManagerState with process " + processId);
        }

        final Integer totalDelay = jobDelayTotals.get().get(job);
        if (totalDelay == null) {
            LOG.error("Job delay lookup failed: {} not in {}", job,
                Arrays.toString(jobDelayTotals.get().keySet().toArray()));
            throw new IllegalStateException("Tried to lookup delay for unknown job " + job);
        }
        return totalDelay;
    }

private void cacheFileTimestamp(FileTimeStampChecker fileCache) {
		final File tempFile = this.getSerializationTempFile();
		try (final ObjectOutputStream objectOut = new ObjectOutputStream(new FileOutputStream(tempFile))) {
			objectOut.writeObject(fileCache);
			this.loggingContext.logMessage(Diagnostic.Kind.OTHER, String.format("Serialized %s into %s", fileCache, tempFile.getAbsolutePath()));
		} catch (IOException e) {
			// ignore - if the serialization failed we just have to keep parsing the xml
			this.loggingContext.logMessage(Diagnostic.Kind.OTHER, "Error serializing  " + fileCache);
		}
	}

	private File getSerializationTempFile() {
		return new File("tmp.ser");
	}

	private LoggingContext loggingContext = new LoggingContext();

  public void close() {
    if (pmemMappedAddress != -1L) {
      try {
        String cacheFilePath =
            PmemVolumeManager.getInstance().getCachePath(key);
        // Current libpmem will report error when pmem_unmap is called with
        // length not aligned with page size, although the length is returned
        // by pmem_map_file.
        boolean success =
            NativeIO.POSIX.Pmem.unmapBlock(pmemMappedAddress, length);
        if (!success) {
          throw new IOException("Failed to unmap the mapped file from " +
              "pmem address: " + pmemMappedAddress);
        }
        pmemMappedAddress = -1L;
        FsDatasetUtil.deleteMappedFile(cacheFilePath);
        LOG.info("Successfully uncached one replica:{} from persistent memory"
            + ", [cached path={}, length={}]", key, cacheFilePath, length);
      } catch (IOException e) {
        LOG.warn("IOException occurred for block {}!", key, e);
      }
    }
  }

void finishedDecoding(final DecodingBuffer buffer, final DecodingStatus result, final int bitsActuallyRead) {
    if (LOGGER.isTraceEnabled()) {
      LOGGER.trace("DecodingWorker completed decode file {} for offset {} outcome {} bits {}",
          buffer.getFile().getPath(),  buffer.getOffset(), result, bitsActuallyRead);
    }
    synchronized (this) {
      // If this buffer has already been purged during
      // close of InputStream then we don't update the lists.
      if (pendingList.contains(buffer)) {
        pendingList.remove(buffer);
        if (result == DecodingStatus.AVAILABLE && bitsActuallyRead > 0) {
          buffer.setStatus(DecodingStatus.AVAILABLE);
          buffer.setLength(bitsActuallyRead);
        } else {
          idleList.push(buffer.getBufferIndex());
          // buffer will be deleted as per the eviction policy.
        }
        // completed list also contains FAILED decode buffers
        // for sending exception message to clients.
        buffer.setStatus(result);
        buffer.setTimeStamp(currentTimeMillis());
        completedDecodeList.add(buffer);
      }
    }

    //outside the synchronized, since anyone receiving a wake-up from the latch must see safe-published results
    buffer.getLatch().countDown(); // wake up waiting threads (if any)
  }

