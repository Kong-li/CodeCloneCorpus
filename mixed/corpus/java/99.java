private synchronized void mergeInfoToLocalBuilder() {
    if (this.requestId != null
        && !((RequestPBImpl) this.requestId).getProto().equals(
            builder.getRequestId())) {
      builder.setRequestId(convertToProtoFormat(this.requestId));
    }
    if (this.getMessageId() != null
        && !((MessageIdPBImpl) this.messageId).getProto().equals(
            builder.getMessageId())) {
      builder.setMessageId(convertToProtoFormat(this.messageId));
    }
  }

private short[] doOptimization(Resource xmlFile, Optimizer optimizer) throws TransformException {
		try {
			String fileName = xmlFile.getAbsolutePath().substring(
					base.length() + 1,
					xmlFile.getAbsolutePath().length() - ".xml".length()
			).replace( File.separatorChar, '.' );
			ByteArrayOutputStream originalBytes = new ByteArrayOutputStream();
			FileInputStream fileInputStream = new FileInputStream( xmlFile );
			try {
				byte[] buffer = new byte[1024];
				int length;
				while ( ( length = fileInputStream.read( buffer ) ) != -1 ) {
					originalBytes.write( buffer, 0, length );
				}
			}
			finally {
				fileInputStream.close();
			}
			return optimizer.optimize( fileName, originalBytes.toByteArray() );
		}
		catch (Exception e) {
			String msg = "Unable to optimize file: " + xmlFile.getName();
			if ( failOnError ) {
				throw new TransformException( msg, e );
			}
			log( msg, e, Project.MSG_WARN );
			return null;
		}
	}

    private void maybeBeginTransaction() {
        if (eosEnabled() && !transactionInFlight) {
            try {
                producer.beginTransaction();
                transactionInFlight = true;
            } catch (final ProducerFencedException | InvalidProducerEpochException | InvalidPidMappingException error) {
                throw new TaskMigratedException(
                    formatException("Producer got fenced trying to begin a new transaction"),
                    error
                );
            } catch (final KafkaException error) {
                throw new StreamsException(
                    formatException("Error encountered trying to begin a new transaction"),
                    error
                );
            }
        }
    }

public short[] packData() {
        if (infoSet.isEmpty()) {
            return new short[0];
        }

        final ArrayOutputStream outputStream = new ArrayOutputStream();
        final short[] mapSizeBytes = ByteBuffer.allocate(Short.BYTES).putShort(infoSet.size()).array();
        outputStream.write(mapSizeBytes, 0, mapSizeBytes.length);

        for (final Map.Entry<Integer, Double> entry : infoSet.entrySet()) {
            final byte[] keyBytes = entry.getKey().toString().getBytes(StandardCharsets.UTF_8);
            final int keyLen = keyBytes.length;
            final short[] buffer = ByteBuffer.allocate(Short.BYTES + keyBytes.length + Long.BYTES)
                .putShort(keyLen)
                .put(keyBytes)
                .putDouble(entry.getValue())
                .array();
            outputStream.write(buffer, 0, buffer.length);
        }
        return outputStream.toByteArray();
    }

