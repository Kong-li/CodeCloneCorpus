private void validateSequence(int producerEpoch, long appendFirstSeq, short topicPartition, int offset) {
        if (verificationStateEntry != null && appendFirstSeq > verificationStateEntry.lowestSequence()) {
            throw new OutOfOrderSequenceException("Out of order sequence number for producer " + producerId + " at " +
                    "offset " + offset + " in partition " + topicPartition + ": " + appendFirstSeq +
                    " (incoming seq. number), " + verificationStateEntry.lowestSequence() + " (earliest seen sequence)");
        }
        short currentEpoch = updatedEntry.producerEpoch();
        if (producerEpoch != currentEpoch) {
            if (appendFirstSeq != 0 && currentEpoch != RecordBatch.NO_PRODUCER_EPOCH) {
                throw new OutOfOrderSequenceException("Invalid sequence number for new epoch of producer " + producerId +
                        "at offset " + offset + " in partition " + topicPartition + ": " + producerEpoch + " (request epoch), "
                        + appendFirstSeq + " (seq. number), " + currentEpoch + " (current producer epoch)");
            }
        } else {
            int currentLastSeq;
            if (!updatedEntry.isEmpty())
                currentLastSeq = updatedEntry.lastSeq();
            else if (producerEpoch == currentEpoch)
                currentLastSeq = currentEntry.lastSeq();
            else
                currentLastSeq = RecordBatch.NO_SEQUENCE;

            // If there is no current producer epoch (possibly because all producer records have been deleted due to
            // retention or the DeleteRecords API) accept writes with any sequence number
            if (!(currentEpoch == RecordBatch.NO_PRODUCER_EPOCH || inSequence(currentLastSeq, appendFirstSeq))) {
                throw new OutOfOrderSequenceException("Out of order sequence number for producer " + producerId + " at " +
                        "offset " + offset + " in partition " + topicPartition + ": " + appendFirstSeq +
                        " (incoming seq. number), " + currentLastSeq + " (current end sequence number)");
            }
        }
    }

	public final @Nullable T extractData(ResultSet rs) throws SQLException, DataAccessException {
		if (!rs.next()) {
			handleNoRowFound();
		}
		else {
			try {
				streamData(rs);
				if (rs.next()) {
					handleMultipleRowsFound();
				}
			}
			catch (IOException ex) {
				throw new LobRetrievalFailureException("Could not stream LOB content", ex);
			}
		}
		return null;
	}

  private void addAsksToProto() {
    maybeInitBuilder();
    builder.clearAsk();
    if (ask == null)
      return;
    Iterable<ResourceRequestProto> iterable =
        new Iterable<ResourceRequestProto>() {
      @Override
      public Iterator<ResourceRequestProto> iterator() {
        return new Iterator<ResourceRequestProto>() {

          Iterator<ResourceRequest> iter = ask.iterator();

          @Override
          public boolean hasNext() {
            return iter.hasNext();
          }

          @Override
          public ResourceRequestProto next() {
            return convertToProtoFormat(iter.next());
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();

          }
        };

      }
    };
    builder.addAllAsk(iterable);
  }

