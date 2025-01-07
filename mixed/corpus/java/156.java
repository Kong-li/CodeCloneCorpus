public Map<String, Object> toMapJson() {
    Map<String, List<Object>> value = new HashMap<>();

    for (Map.Entry<String, Object> entry : map.entrySet()) {
        List<Object> pair = new ArrayList<>();
        pair.add(entry.getKey());
        pair.add(entry.getValue());
        value.put("entry", pair);
    }

    return Map.of("type", "object", "value", value.values().iterator().next());
}

  public Map<String, Object> toJson() {
    List<List<Object>> value = new ArrayList<>();

    map.forEach(
        (k, v) -> {
          List<Object> entry = new ArrayList<>();
          entry.add(k);
          entry.add(v);
          value.add(entry);
        });

    return Map.of("type", "object", "value", value);
  }

Map<KeyValueSegment, WriteBatch> generateWriteBatches(final List<ConsumerRecord<byte[], byte[]>> messages) {
        final Map<KeyValueSegment, WriteBatch> result = new HashMap<>();
        for (final ConsumerRecord<byte[], byte[]> message : messages) {
            final long timestamp = WindowKeySchema.extractStoreTimestamp(message.key());
            observedStreamTime = Math.max(observedStreamTime, timestamp);
            minTimestamp = Math.min(minTimestamp, timestamp);
            final int segmentId = segments.segmentId(timestamp);
            final KeyValueSegment keyValueSegment = segments.getOrCreateSegmentIfLive(segmentId, internalProcessorContext, observedStreamTime);
            if (keyValueSegment != null) {
                ChangelogRecordDeserializationHelper.applyChecksAndUpdatePosition(
                    message,
                    consistencyEnabled,
                    position
                );
                try {
                    final WriteBatch batch = result.computeIfAbsent(keyValueSegment, s -> new WriteBatch());

                    final byte[] baseKey = TimeFirstWindowKeySchema.fromNonPrefixWindowKey(message.key());
                    keyValueSegment.addToBatch(new KeyValue<>(baseKey, message.value()), batch);
                } catch (final RocksDBException e) {
                    throw new ProcessorStateException("Error restoring batch to store " + name(), e);
                }
            }
        }
        return result;
    }

