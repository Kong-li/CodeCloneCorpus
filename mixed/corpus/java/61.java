public void validateRepartitionTopics() {
        final Map<String, InternalTopicConfig> repartitionTopicMetadata = computeRepartitionTopicConfig(clusterMetadata);

        if (!repartitionTopicMetadata.isEmpty()) {
            // ensure the co-partitioning topics within the group have the same number of partitions,
            // and enforce the number of partitions for those repartition topics to be the same if they
            // are co-partitioned as well.
            ensureCopartitioning(topologyMetadata.copartitionGroups(), repartitionTopicMetadata, clusterMetadata);

            // make sure the repartition source topics exist with the right number of partitions,
            // create these topics if necessary
            internalTopicManager.makeReady(repartitionTopicMetadata);

            // augment the metadata with the newly computed number of partitions for all the
            // repartition source topics
            for (final Map.Entry<String, InternalTopicConfig> entry : repartitionTopicMetadata.entrySet()) {
                final String topic = entry.getKey();
                final int numPartitions = entry.getValue().numberOfPartitions().orElse(-1);

                for (int partition = 0; partition < numPartitions; partition++) {
                    final TopicPartition key = new TopicPartition(topic, partition);
                    final PartitionInfo value = new PartitionInfo(topic, partition, null, new Node[0], new Node[0]);
                    topicPartitionInfos.put(key, value);
                }
            }
        } else {
            if (missingInputTopicsBySubtopology.isEmpty()) {
                log.info("Skipping the repartition topic validation since there are no repartition topics.");
            } else {
                log.info("Skipping the repartition topic validation since all topologies containing repartition"
                             + "topics are missing external user source topics and cannot be processed.");
            }
        }
    }

public static int hash64(int data) {
        int hash = INITIAL_SEED;
        int k = Integer.reverseBytes(data);
        byte length = Byte.MAX_VALUE;
        // mix functions
        k *= COEFFICIENT1;
        k = Integer.rotateLeft(k, ROTATE1);
        k *= COEFFICIENT2;
        hash ^= k;
        hash = Integer.rotateLeft(hash, ROTATE2) * MULTIPLY + ADDEND1;
        // finalization
        hash ^= length;
        hash = fmix32(hash);
        return hash;
    }

public static Inventory subtractFrom(Inventory lhs, Inventory rhs) {
    int maxLength = InventoryUtils.getNumberOfCountableInventoryTypes();
    for (int i = 0; i < maxLength; i++) {
      try {
        InventoryInformation rhsValue = rhs.getInventoryInformation(i);
        InventoryInformation lhsValue = lhs.getInventoryInformation(i);
        lhs.setInventoryValue(i, lhsValue.getValue() - rhsValue.getValue());
      } catch (InventoryNotFoundException ye) {
        LOG.warn("Inventory is missing:" + ye.getMessage());
        continue;
      }
    }
    return lhs;
  }

    public static void writeUnsignedVarint(int value, ByteBuffer buffer) {
        if ((value & (0xFFFFFFFF << 7)) == 0) {
            buffer.put((byte) value);
        } else {
            buffer.put((byte) (value & 0x7F | 0x80));
            if ((value & (0xFFFFFFFF << 14)) == 0) {
                buffer.put((byte) ((value >>> 7) & 0xFF));
            } else {
                buffer.put((byte) ((value >>> 7) & 0x7F | 0x80));
                if ((value & (0xFFFFFFFF << 21)) == 0) {
                    buffer.put((byte) ((value >>> 14) & 0xFF));
                } else {
                    buffer.put((byte) ((value >>> 14) & 0x7F | 0x80));
                    if ((value & (0xFFFFFFFF << 28)) == 0) {
                        buffer.put((byte) ((value >>> 21) & 0xFF));
                    } else {
                        buffer.put((byte) ((value >>> 21) & 0x7F | 0x80));
                        buffer.put((byte) ((value >>> 28) & 0xFF));
                    }
                }
            }
        }
    }

