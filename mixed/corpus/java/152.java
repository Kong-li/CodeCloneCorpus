protected JsonNode parseByteBufferToJson(ByteBuffer byteBuffer) {
        try {
            CoordinatorRecordType recordType = CoordinatorRecordType.fromId(byteBuffer.getShort());
            if (recordType == CoordinatorRecordType.GROUP_METADATA) {
                ByteBufferAccessor accessor = new ByteBufferAccessor(byteBuffer);
                return GroupMetadataKeyJsonConverter.write(new GroupMetadataKey(accessor, (short) 0), (short) 0);
            } else {
                return NullNode.getInstance();
            }
        } catch (UnsupportedVersionException e) {
            return NullNode.getInstance();
        }
    }

public Iterator<Song> fetchLongPlaylistSongs() {

        final List<Song> baseList =
                this.jdbcTemplate.query(
                    QUERY_GET_ALL_SONGS,
                    (resultSet, i) -> {
                        return new Song(
                                Integer.valueOf(resultSet.getInt("songID")),
                                resultSet.getString("songTitle"),
                                resultSet.getString("artistName"),
                                resultSet.getString("albumName"));
                    });

        return new Iterator<Song>() {

            private static final int COUNT = 400;

            private int counter = 0;
            private Iterator<Song> currentIterator = null;

            @Override
            public boolean hasNext() {
                if (this.currentIterator != null && this.currentIterator.hasNext()) {
                    return true;
                }
                if (this.counter < COUNT) {
                    this.currentIterator = baseList.iterator();
                    this.counter++;
                    return true;
                }
                return false;
            }

            @Override
            public Song next() {
                return this.currentIterator.next();
            }

        };

    }

public void handleDataStreamer() {
    TraceScope scope = null;
    while (!streamerClosed && dfsClient.clientRunning) {
      if (errorState.hasError()) {
        closeResponder();
      }

      DFSPacket packet;
      try {
        boolean shouldSleep = processDatanodeOrExternalError();

        synchronized (dataQueue) {
          while ((!shouldStop() && dataQueue.isEmpty()) || shouldSleep) {
            long timeout = 1000;
            if (stage == BlockConstructionStage.DATA_STREAMING) {
              timeout = sendHeartbeat();
            }
            try {
              dataQueue.wait(timeout);
            } catch (InterruptedException e) {
              LOG.debug("Thread interrupted", e);
            }
            shouldSleep = false;
          }
          if (shouldStop()) {
            continue;
          }
          packet = dataQueue.pollFirst(); // regular data packet
          SpanContext[] parents = packet.getTraceParents();
          if (parents != null && parents.length > 0) {
            scope = dfsClient.getTracer().
                newScope("dataStreamer", parents[0], false);
            //scope.getSpan().setParents(parents);
          }
        }

        try {
          backOffIfNecessary();
        } catch (InterruptedException e) {
          LOG.debug("Thread interrupted", e);
        }

        if (stage == BlockConstructionStage.PIPELINE_SETUP_CREATE) {
          setupPipelineForCreate();
          initDataStreaming();
        } else if (stage == BlockConstructionStage.PIPELINE_SETUP_APPEND) {
          setupPipelineForAppendOrRecovery();
          if (streamerClosed) {
            continue;
          }
          initDataStreaming();
        }

        long lastByteOffsetInBlock = packet.getLastByteOffsetBlock();
        if (lastByteOffsetInBlock > stat.getBlockSize()) {
          throw new IOException("BlockSize " + stat.getBlockSize() +
              " < lastByteOffsetInBlock, " + this + ", " + packet);
        }

        if (packet.isLastPacketInBlock()) {
          waitForAllAcks();
          if(shouldStop()) {
            continue;
          }
          stage = BlockConstructionStage.PIPELINE_CLOSE;
        }

        SpanContext spanContext = null;
        synchronized (dataQueue) {
          // move packet from dataQueue to ackQueue
          if (!packet.isHeartbeatPacket()) {
            if (scope != null) {
              packet.setSpan(scope.span());
              spanContext = scope.span().getContext();
              scope.close();
            }
            scope = null;
            dataQueue.removeFirst();
            ackQueue.addLast(packet);
            packetSendTime.put(packet.getId(), System.currentTimeMillis());
          }
        }

        if (progress != null) { progress.progress(); }

        if (artificialSlowdown != 0 && dfsClient.clientRunning) {
          Thread.sleep(artificialSlowdown);
        }
      } catch (Throwable e) {
        if (!errorState.isRestartingNode()) {
          if (e instanceof QuotaExceededException) {
            LOG.debug("DataStreamer Quota Exception", e);
          } else {
            LOG.warn("DataStreamer Exception", e);
          }
        }
        lastException.set(e);
        assert !(e instanceof NullPointerException);
        errorState.setInternalError();
        if (!errorState.isNodeMarked()) {
          streamerClosed = true;
        }
      } finally {
        if (scope != null) {
          scope.close();
          scope = null;
        }
      }
    }
    closeInternal();
  }

