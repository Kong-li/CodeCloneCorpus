    public CompletableFuture<Boolean> updateFetchPositions(long deadlineMs) {
        CompletableFuture<Boolean> result = new CompletableFuture<>();

        try {
            if (maybeCompleteWithPreviousException(result)) {
                return result;
            }

            validatePositionsIfNeeded();

            if (subscriptionState.hasAllFetchPositions()) {
                // All positions are already available
                result.complete(true);
                return result;
            }

            // Some positions are missing, so trigger requests to fetch offsets and update them.
            updatePositionsWithOffsets(deadlineMs).whenComplete((__, error) -> {
                if (error != null) {
                    result.completeExceptionally(error);
                } else {
                    result.complete(subscriptionState.hasAllFetchPositions());
                }
            });

        } catch (Exception e) {
            result.completeExceptionally(maybeWrapAsKafkaException(e));
        }
        return result;
    }

public DbmAnyDiscriminatorValue<S> duplicate(DbmCopyContext context) {
		final DbmAnyDiscriminatorValue<S> existing = context.getDuplicate( this );
		if ( existing != null ) {
			return existing;
		}
		final DbmAnyDiscriminatorValue<S> expression = context.registerDuplicate(
				this,
				new DbmAnyDiscriminatorValue<>(
						columnName,
						valueType,
						domainClass,
						nodeConstructor()
				)
		);
		copyTo( expression, context );
		return expression;
	}

  static String[] getSupportedAlgorithms() {
    Algorithm[] algos = Algorithm.class.getEnumConstants();

    ArrayList<String> ret = new ArrayList<String>();
    for (Algorithm a : algos) {
      if (a.isSupported()) {
        ret.add(a.getName());
      }
    }
    return ret.toArray(new String[ret.size()]);
  }

protected boolean checkRecursability(PathData element) throws IOException {
    if (!element.stat.isDirectory()) {
      return false;
    }
    PathData linkedItem = null;
    if (element.stat.isSymlink()) {
      linkedItem = new PathData(element.fs.resolvePath(element.stat.getSymlink()).toString(), getConf());
      if (linkedItem.stat.isDirectory()) {
        boolean followLink = getOptions().isFollowLink();
        boolean followArgLink = getOptions().isFollowArgLink() && (getDepth() == 0);
        return followLink || followArgLink;
      }
    }
    return false;
  }

