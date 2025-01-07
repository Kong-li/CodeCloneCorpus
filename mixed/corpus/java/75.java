public synchronized long movePointer(long newOffset) throws IOException {
    verifyOpen();

    if (newOffset < 0) {
        throw new EOFException(FSExceptionMessages.NEGATIVE_SEEK);
    }

    if (newOffset > size) {
        throw new EOFException(FSExceptionMessages.CANNOT_SEEK_PAST_EOF);
    }

    int newPosition = (int)(position() + newOffset);
    byteBuffer.position(newPosition);

    return newPosition;
}

public void logRecordDetails(ConsumerRecord<byte[], byte[]> record, PrintStream out) {
    defaultWriter.writeTo(record, out);
    String timestamp = consumerRecordHasTimestamp(record)
            ? getFormattedTimestamp(record) + ", "
            : "";
    String keyDetails = "key:" + (record.key() == null ? "null" : new String(record.key(), StandardCharsets.UTF_8) + ", ");
    String valueDetails = "value:" + (record.value() == null ? "null" : new String(record.value(), StandardCharsets.UTF_8));
    LOG.info(timestamp + keyDetails + valueDetails);

    boolean hasTimestamp = consumerRecordHasTimestamp(record);
    String formattedTs = hasTimestamp ? getFormattedTimestamp(record) + ", " : "";
    LOG.info(formattedTs + "key:" + (record.key() == null ? "null" : new String(record.key(), StandardCharsets.UTF_8) + ", ") +
              "value:" + (record.value() == null ? "null" : new String(record.value(), StandardCharsets.UTF_8)));
}

private boolean consumerRecordHasTimestamp(ConsumerRecord<byte[], byte[]> record) {
    return record.timestampType() != TimestampType.NO_TIMESTAMP_TYPE;
}

private String getFormattedTimestamp(ConsumerRecord<byte[], byte[]> record) {
    return record.timestampType() + ":" + record.timestamp();
}

public final void deregisterListener(Event<?> event) {
		EventMetadata metadata = event.getMetadata();

		DeliveryType deliveryType = EventHeaderAccessor.getDeliveryType(metadata);
		if (!DeliveryType.UNSUBSCRIBE.equals(deliveryType)) {
			throw new IllegalArgumentException("Expected UNSUBSCRIBE: " + event);
		}

		String listenerId = EventHeaderAccessor.getListenerId(metadata);
		if (listenerId == null) {
			if (logger.isWarnEnabled()) {
				logger.warn("No listenerId in " + event);
			}
			return;
		}

		String subscriptionKey = EventHeaderAccessor.getSubscriptionKey(metadata);
		if (subscriptionKey == null) {
			if (logger.isWarnEnabled()) {
				logger.warn("No subscriptionKey " + event);
			}
			return;
		}

		removeListenerInternal(listenerId, subscriptionKey, event);
	}

