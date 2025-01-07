  private static void extractInterfaces(final Set<Class<?>> collector, final Class<?> clazz) {
    if (clazz == null || Object.class.equals(clazz)) {
      return;
    }

    final Class<?>[] classes = clazz.getInterfaces();
    for (Class<?> interfaceClass : classes) {
      collector.add(interfaceClass);
      for (Class<?> superInterface : interfaceClass.getInterfaces()) {
        collector.add(superInterface);
        extractInterfaces(collector, superInterface);
      }
    }
    extractInterfaces(collector, clazz.getSuperclass());
  }

public void terminate() {
        boolean hasFailed = false;
        for (ConsumerInterceptor<K, V> interceptor : this.interceptors) {
            try {
                interceptor.close();
            } catch (Exception e) {
                log.error("Failed to terminate consumer interceptor ", e);
                hasFailed = true;
            }
        }
        if (hasFailed) {
            log.warn("Some consumer interceptors failed to terminate properly.");
        }
    }

public void syncFlagsHandler(SyncType syncType) throws IOException {
    OutputStream wrappedStream = getWrappedStream();
    if (!(wrappedStream instanceof CryptoOutputStream)) {
        wrappedStream = ((DFSOutputStream) wrappedStream).getWrappedStream();
    }
    flushIfCryptoStream(wrappedStream);
    ((DFSOutputStream) wrappedStream).hsync(syncType);
}

private void flushIfCryptoStream(OutputStream stream) throws IOException {
    if (stream instanceof CryptoOutputStream) {
        ((CryptoOutputStream) stream).flush();
    }
}

  public void createDatabase() {
    Observable<ResourceResponse<Database>> databaseReadObs =
        client.readDatabase(String.format(DATABASE_LINK, databaseName), null);

    Observable<ResourceResponse<Database>> databaseExistenceObs =
        databaseReadObs
            .doOnNext(databaseResourceResponse ->
                LOG.info("Database {} already exists.", databaseName))
            .onErrorResumeNext(throwable -> {
              // if the database doesn't exists
              // readDatabase() will result in 404 error
              if (throwable instanceof DocumentClientException) {
                DocumentClientException de =
                    (DocumentClientException) throwable;
                if (de.getStatusCode() == 404) {
                  // if the database doesn't exist, create it.
                  LOG.info("Creating new Database : {}", databaseName);

                  Database dbDefinition = new Database();
                  dbDefinition.setId(databaseName);

                  return client.createDatabase(dbDefinition, null);
                }
              }
              // some unexpected failure in reading database happened.
              // pass the error up.
              LOG.error("Reading database : {} if it exists failed.",
                  databaseName, throwable);
              return Observable.error(throwable);
            });
    // wait for completion
    databaseExistenceObs.toCompletable().await();
  }

