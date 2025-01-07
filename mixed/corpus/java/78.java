  public void setDelegationTokenSeqNum(int seqNum) {
    Connection connection = null;
    try {
      connection = getConnection(false);
      FederationQueryRunner runner = new FederationQueryRunner();
      runner.updateSequenceTable(connection, YARN_ROUTER_SEQUENCE_NUM, seqNum);
    } catch (Exception e) {
      throw new RuntimeException("Could not update sequence table!!", e);
    } finally {
      // Return to the pool the CallableStatement
      try {
        FederationStateStoreUtils.returnToPool(LOG, null, connection);
      } catch (YarnException e) {
        LOG.error("close connection error.", e);
      }
    }
  }

	protected Mono<Void> doCommit(@Nullable Supplier<? extends Mono<Void>> writeAction) {
		Flux<Void> allActions = Flux.empty();
		if (this.state.compareAndSet(State.NEW, State.COMMITTING)) {
			if (!this.commitActions.isEmpty()) {
				allActions = Flux.concat(Flux.fromIterable(this.commitActions).map(Supplier::get))
						.doOnError(ex -> {
							if (this.state.compareAndSet(State.COMMITTING, State.COMMIT_ACTION_FAILED)) {
								getHeaders().clearContentHeaders();
							}
						});
			}
		}
		else if (this.state.compareAndSet(State.COMMIT_ACTION_FAILED, State.COMMITTING)) {
			// Skip commit actions
		}
		else {
			return Mono.empty();
		}

		allActions = allActions.concatWith(Mono.fromRunnable(() -> {
			applyStatusCode();
			applyHeaders();
			applyCookies();
			this.state.set(State.COMMITTED);
		}));

		if (writeAction != null) {
			allActions = allActions.concatWith(writeAction.get());
		}

		return allActions.then();
	}

public void await(long duration) throws InterruptedException {
    long end = System.currentTimeMillis() + duration;
    boolean timeoutFlag = true;
    while (System.currentTimeMillis() < end) {
        if (Thread.interrupted()) {
            throw new InterruptedException();
        }
        if (!handler.isEmpty()) {
            timeoutFlag = false;
            break;
        }
        Thread.sleep(50);
    }
    if (timeoutFlag) {
        throw new TimeoutException(
                String.format("Operation timed out after waiting for %d ms.", duration));
    }

    // Ensure syserr and sysout are processed
}

public static boolean canBeConvertedToStream(Class<?> clazz) {
		if (clazz == null || clazz == Void.class) {
			return false;
		}
		boolean isAssignableFrom = Stream.class.isAssignableFrom(clazz)
				|| DoubleStream.class.isAssignableFrom(clazz)
				|| IntStream.class.isAssignableFrom(clazz)
				|| LongStream.class.isAssignableFrom(clazz)
				|| Iterable.class.isAssignableFrom(clazz)
				|| Iterator.class.isAssignableFrom(clazz);
		return isAssignableFrom || Object[].class.isAssignableFrom(clazz) || clazz.isArray() && clazz.getComponentType().isPrimitive();
	}

