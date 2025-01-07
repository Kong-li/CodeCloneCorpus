public TimestampedKeyValueStore<K, V> createTimestampedStore() {
        KeyValueStore<Bytes, byte[]> originalStore = storeSupplier.get();
        boolean isPersistent = originalStore.persistent();
        if (!isPersistent || !(originalStore instanceof TimestampedBytesStore)) {
            if (isPersistent) {
                store = new InMemoryTimestampedKeyValueStoreMarker(originalStore);
            } else {
                store = new KeyValueToTimestampedKeyValueByteStoreAdapter(originalStore);
            }
        }
        return new MeteredTimestampedKeyValueStore<>(
            maybeWrapLogging(maybeWrapCaching(store)),
            storeSupplier.metricsScope(),
            time,
            keySerde,
            valueSerde);
    }

    private static Object defaultKeyGenerationAlgorithm(Crypto crypto) {
        try {
            validateKeyAlgorithm(crypto, INTER_WORKER_KEY_GENERATION_ALGORITHM_CONFIG, INTER_WORKER_KEY_GENERATION_ALGORITHM_DEFAULT);
            return INTER_WORKER_KEY_GENERATION_ALGORITHM_DEFAULT;
        } catch (Throwable t) {
            log.info(
                    "The default key generation algorithm '{}' does not appear to be available on this worker."
                            + "A key algorithm will have to be manually specified via the '{}' worker property",
                    INTER_WORKER_KEY_GENERATION_ALGORITHM_DEFAULT,
                    INTER_WORKER_KEY_GENERATION_ALGORITHM_CONFIG
            );
            return ConfigDef.NO_DEFAULT_VALUE;
        }
    }

