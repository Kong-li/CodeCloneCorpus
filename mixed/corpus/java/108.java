public static void main(String[] args) throws Exception {
        ConsoleConsumerOptions options = new ConsoleConsumerOptions(args);
        try {
            run(options);
        } catch (AuthenticationException e) {
            LOG.error("Auth failed: consumer process ending", e);
            Exit.exit(1);
        } catch (Throwable t) {
            if (!t.getMessage().isEmpty()) {
                LOG.error("Error running consumer: ", t);
            }
            Exit.exit(1);
        }
    }

  private V getDoneValue(Object obj) throws ExecutionException {
    // While this seems like it might be too branch-y, simple benchmarking
    // proves it to be unmeasurable (comparing done AbstractFutures with
    // immediateFuture)
    if (obj instanceof Cancellation) {
      throw cancellationExceptionWithCause(
          "Task was cancelled.", ((Cancellation) obj).cause);
    } else if (obj instanceof Failure) {
      throw new ExecutionException(((Failure) obj).exception);
    } else if (obj == NULL) {
      return null;
    } else {
      @SuppressWarnings("unchecked") // this is the only other option
          V asV = (V) obj;
      return asV;
    }
  }

  public void addListener(Runnable listener, Executor executor) {
    Preconditions.checkNotNull(listener, "Runnable was null.");
    Preconditions.checkNotNull(executor, "Executor was null.");
    Listener oldHead = listeners;
    if (oldHead != Listener.TOMBSTONE) {
      Listener newNode = new Listener(listener, executor);
      do {
        newNode.next = oldHead;
        if (ATOMIC_HELPER.casListeners(this, oldHead, newNode)) {
          return;
        }
        oldHead = listeners; // re-read
      } while (oldHead != Listener.TOMBSTONE);
    }
    // If we get here then the Listener TOMBSTONE was set, which means the
    // future is done, call the listener.
    executeListener(listener, executor);
  }

public TargetRepository<?> getTarget(PluginTarget target) throws IOException {
    TargetRepository<?> targetRepository;
    switch (target.type()) {
        case PROJECT:
            targetRepository = new ProjectRepository(target);
            break;
        case LIBRARY:
            targetRepository = new LibraryRepository(target);
            break;
        case EXECUTABLE:
            targetRepository = new ExecutableRepository(target);
            break;
        case INTERFACE_HIERARCHY:
            targetRepository = new InterfaceHierarchyRepository(target);
            break;
        default:
            throw new IllegalStateException("Unknown target type " + target.type());
    }
    repositories.add(targetRepository);
    return targetRepository;
}

