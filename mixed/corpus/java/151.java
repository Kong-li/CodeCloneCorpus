    public synchronized <K, V> void increment(KafkaProducer<K, V> producer) {
        // Increment the message tracker.
        messageTracker += 1;

        // Compare the tracked message count with the throttle limits.
        if (messageTracker >= flushSize) {
            try {
                producer.flush();
            } catch (InterruptException e) {
                // Ignore flush interruption exceptions.
            }
            calculateFlushSize();
        }
    }

