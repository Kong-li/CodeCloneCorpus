public void removeCredentialEntryByIdentifier(String entryIdentifier) throws IOException {
    writeLock.lock();
    try {
        if (!keyStore.containsAlias(entryIdentifier)) {
            throw new IOException("Credential " + entryIdentifier + " does not exist in " + this);
        }
        keyStore.deleteEntry(entryIdentifier);
        changed = true;
    } catch (KeyStoreException e) {
        throw new IOException("Problem removing " + entryIdentifier + " from " + this, e);
    } finally {
        writeLock.unlock();
    }
}

    public void close() {
        lock.lock();
        try {
            idempotentCloser.close(
                    this::drainAll,
                    () -> log.warn("The fetch buffer was already closed")
            );
        } finally {
            lock.unlock();
        }
    }

