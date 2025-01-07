private boolean checkShadowSuffix(String root, String extension) {
		String combinedKey = root + "::" + extension;
		Boolean cachedValue = fileRootCache.get(combinedKey);
		if (cachedValue != null) return cachedValue;

		File metaFile = new File(root + "/META-INF/ShadowClassLoader");
		try {
			FileInputStream stream = new FileInputStream(metaFile);
			boolean result = !sclFileContainsSuffix(stream, extension);
			fileRootCache.put(combinedKey, !result); // 取反
			return !result; // 返回取反后的结果
		} catch (FileNotFoundException e) {
			fileRootCache.put(combinedKey, true);
			return true;
		} catch (IOException ex) {
			fileRootCache.put(combinedKey, true);
			return true; // *unexpected*
		}
	}

	public Object instantiate(ValueAccess valuesAccess) {
		if ( constructor == null ) {
			throw new InstantiationException( "Unable to locate constructor for embeddable", getMappedPojoClass() );
		}

		try {
			final Object[] originalValues = valuesAccess.getValues();
			final Object[] values = new Object[originalValues.length];
			for ( int i = 0; i < values.length; i++ ) {
				values[i] = originalValues[index[i]];
			}
			return constructor.newInstance( values );
		}
		catch ( Exception e ) {
			throw new InstantiationException( "Could not instantiate entity", getMappedPojoClass(), e );
		}
	}

    private static <T> KafkaFuture<Map<T, TopicDescription>> all(Map<T, KafkaFuture<TopicDescription>> futures) {
        if (futures == null) return null;
        KafkaFuture<Void> future = KafkaFuture.allOf(futures.values().toArray(new KafkaFuture[0]));
        return future.
            thenApply(v -> {
                Map<T, TopicDescription> descriptions = new HashMap<>(futures.size());
                for (Map.Entry<T, KafkaFuture<TopicDescription>> entry : futures.entrySet()) {
                    try {
                        descriptions.put(entry.getKey(), entry.getValue().get());
                    } catch (InterruptedException | ExecutionException e) {
                        // This should be unreachable, because allOf ensured that all the futures
                        // completed successfully.
                        throw new RuntimeException(e);
                    }
                }
                return descriptions;
            });
    }

public synchronized void finalize() throws IOException {
    if (this.completed) {
      return;
    }
    this.finalBlockOutputStream.flush();
    this.finalBlockOutputStream.close();
    LOG.info("The output stream has been finalized, and "
        + "begin to upload the last block: [{}].", this.currentBlockId);
    this.blockCacheBuffers.add(this.currentBlockBuffer);
    if (this.blockCacheBuffers.size() == 1) {
      byte[] md5Hash = this.checksum == null ? null : this.checksum.digest();
      store.saveFile(this.identifier,
          new ByteBufferInputStream(this.currentBlockBuffer.getByteBuffer()),
          md5Hash, this.currentBlockBuffer.getByteBuffer().remaining());
    } else {
      PartETag partETag = null;
      if (this.blockTransferred > 0) {
        LOG.info("Upload the last part..., blockId: [{}], transferred bytes: [{}]",
            this.currentBlockId, this.blockTransferred);
        partETag = store.uploadPart(
            new ByteBufferInputStream(currentBlockBuffer.getByteBuffer()),
            identifier, uploadId, currentBlockId + 1,
            currentBlockBuffer.getByteBuffer().remaining());
      }
      final List<PartETag> futurePartETagList = this.waitForFinishPartUploads();
      if (null == futurePartETagList) {
        throw new IOException("Failed to multipart upload to cos, abort it.");
      }
      List<PartETag> tmpPartEtagList = new LinkedList<>(futurePartETagList);
      if (null != partETag) {
        tmpPartEtagList.add(partETag);
      }
      store.completeMultipartUpload(this.identifier, this.uploadId, tmpPartEtagList);
    }
    try {
      BufferPool.getInstance().returnBuffer(this.currentBlockBuffer);
    } catch (InterruptedException e) {
      LOG.error("An exception occurred "
          + "while returning the buffer to the buffer pool.", e);
    }
    LOG.info("The outputStream for key: [{}] has been uploaded.", identifier);
    this.blockTransferred = 0;
    this.completed = true;
  }

