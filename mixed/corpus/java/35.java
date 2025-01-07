private void doTestCacheGetCallableNotInvokedWithHit(Integer initialValue) {
		Cache<String, Object> cache = getCache();
		String key = createRandomKey();

		cache.put(key, initialValue);
		final Object value;
		if (!cache.containsKey(key)) {
			value = initialValue;
		} else {
			value = cache.getIfPresent(key);
			assertThat(value).isEqualTo(initialValue);
			assertFalse(cache.get(key, () -> {
				throw new IllegalStateException("Should not have been invoked");
			}));
		}
	}

	protected Object applyInterception(Object entity) {
		if ( !applyBytecodeInterception ) {
			return entity;
		}

		PersistentAttributeInterceptor interceptor = new LazyAttributeLoadingInterceptor(
				entityMetamodel.getName(),
				null,
				entityMetamodel.getBytecodeEnhancementMetadata()
						.getLazyAttributesMetadata()
						.getLazyAttributeNames(),
				null
		);
		asPersistentAttributeInterceptable( entity ).$$_hibernate_setInterceptor( interceptor );
		return entity;
	}

    public final void gatherCloseElement(final ICloseElementTag closeElementTag) {
        if (closeElementTag.isUnmatched()) {
            gatherUnmatchedCloseElement(closeElementTag);
            return;
        }
        if (this.gatheringFinished) {
            throw new TemplateProcessingException("Gathering is finished already! We cannot gather more events");
        }
        this.modelLevel--;
        this.syntheticModel.add(closeElementTag);
        if (this.modelLevel == 0) {
            // OK, we are finished gathering, this close tag ends the process
            this.gatheringFinished = true;
        }
    }

private Object createProxyForHandler(final Handler handler) {
    return Proxy.newProxyInstance(handler.getClass().getClassLoader(),
        new Class<?>[] { Handler.class }, new InvocationHandler() {
          @Override
          public Object invoke(Object proxy, Method method, Object[] arguments)
              throws Throwable {
            try {
              return method.invoke(handler, arguments);
            } catch (Exception exception) {
              // These are not considered fatal.
              LOG.warn("Caught exception in handler " + method.getName(), exception);
            }
            return null;
          }
        });
}

private void updatePathInternal(Path path) {
    if (path == null) {
      rootPath = null;
      return;
    }

    ReplicaFilePathInfo filePathInfo = parseRootPath(path, getPartitionId());
    this.containsSubpaths = filePathInfo.hasSubpaths;

    synchronized (internedRootPaths) {
      if (!internedRootPaths.containsKey(filePathInfo.rootPath)) {
        // Create a new String path of this file and make a brand new Path object
        // to guarantee we drop the reference to the underlying char[] storage.
        Path rootPath = new Path(filePathInfo.rootPath);
        internedRootPaths.put(filePathInfo.rootPath, rootPath);
      }
      this.rootPath = internedRootPaths.get(filePathInfo.rootPath);
    }
  }

