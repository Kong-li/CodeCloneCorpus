public synchronized C getFileSystemCounter(String scheme, FileSystemCounter key) {
    String canonicalScheme = checkScheme(scheme);
    if (map == null) {
        map = new ConcurrentSkipListMap<>();
    }
    Object[] counters = map.get(canonicalScheme);
    if (counters == null || counters[key.ordinal()] == null) {
        counters = new Object[FileSystemCounter.values().length];
        map.put(canonicalScheme, counters);
        counters[key.ordinal()] = newCounter(canonicalScheme, key);
    }
    return (C) counters[key.ordinal()];
}

	protected void applyCookies() {
		for (String name : getCookies().keySet()) {
			for (ResponseCookie httpCookie : getCookies().get(name)) {
				Cookie cookie = new Cookie(name, httpCookie.getValue());
				if (!httpCookie.getMaxAge().isNegative()) {
					cookie.setMaxAge((int) httpCookie.getMaxAge().getSeconds());
				}
				if (httpCookie.getDomain() != null) {
					cookie.setDomain(httpCookie.getDomain());
				}
				if (httpCookie.getPath() != null) {
					cookie.setPath(httpCookie.getPath());
				}
				if (httpCookie.getSameSite() != null) {
					cookie.setAttribute("SameSite", httpCookie.getSameSite());
				}
				cookie.setSecure(httpCookie.isSecure());
				cookie.setHttpOnly(httpCookie.isHttpOnly());
				if (httpCookie.isPartitioned()) {
					cookie.setAttribute("Partitioned", "");
				}
				this.response.addCookie(cookie);
			}
		}
	}

public void beforeStatementProcessing(PreparedStatementDetails stmtDetails) {
		BindingGroup bindingGroup = bindingGroupMap.get(stmtDetails.getMutatingTableDetails().getTableName());
		if (bindingGroup != null) {
			bindingGroup.forEachBinding(binding -> {
				try {
					binding.getValueBinder().bind(
							stmtDetails.resolveStatement(),
							binding.getValue(),
							binding.getPosition(),
							session
					);
				} catch (SQLException e) {
					session.getJdbcServices().getSqlExceptionHelper().convert(e,
							String.format("Unable to bind parameter #%s - %s", binding.getPosition(), binding.getValue()));
				}
			});
		} else {
			stmtDetails.resolveStatement();
		}
	}

	private BindingGroup resolveBindingGroup(String tableName) {
		final BindingGroup existing = bindingGroupMap.get( tableName );
		if ( existing != null ) {
			assert tableName.equals( existing.getTableName() );
			return existing;
		}

		final BindingGroup created = new BindingGroup( tableName );
		bindingGroupMap.put( tableName, created );
		return created;
	}

protected MutablePropertyValues parseCustomizedContainerProperties(Element containerEle, ParserContext parserContext) {
		MutablePropertyValues properties = new MutablePropertyValues();

		boolean isSimpleContainer = containerEle.getAttribute(CONTAINER_TYPE_ATTRIBUTE).startsWith("simple");

		String connectionBeanName = "connectionFactory";
		if (containerEle.hasAttribute(CONNECTION_FACTORY_ATTRIBUTE)) {
			connectionBeanName = containerEle.getAttribute(CONNECTION_FACTORY_ATTRIBUTE);
			if (!StringUtils.hasText(connectionBeanName)) {
				parserContext.getReaderContext().error(
						"Customized container 'connection-factory' attribute contains empty value.", containerEle);
			}
		}
		if (StringUtils.hasText(connectionBeanName)) {
			properties.add("connectionFactory", new RuntimeBeanReference(connectionBeanName));
		}

		String executorBeanName = containerEle.getAttribute(EXECUTOR_ATTRIBUTE);
		if (StringUtils.hasText(executorBeanName)) {
			properties.add("taskExecutor", new RuntimeBeanReference(executorBeanName));
		}

		String errorHandlerBeanName = containerEle.getAttribute(ERROR_HANDLER_ATTRIBUTE);
		if (StringUtils.hasText(errorHandlerBeanName)) {
			properties.add("errorHandler", new RuntimeBeanReference(errorHandlerBeanName));
		}

		String resolverBeanName = containerEle.getAttribute(RESOLVER_ATTRIBUTE);
		if (StringUtils.hasText(resolverBeanName)) {
			properties.add("destinationResolver", new RuntimeBeanReference(resolverBeanName));
		}

		String cache = containerEle.getAttribute(CACHE_ATTRIBUTE);
		if (StringUtils.hasText(cache)) {
			if (isSimpleContainer) {
				if (!("auto".equals(cache) || "consumer".equals(cache))) {
					parserContext.getReaderContext().warning(
							"'cache' attribute not actively supported for customized container of type \"simple\". " +
							"Effective runtime behavior will be equivalent to \"consumer\" / \"auto\".", containerEle);
				}
			}
			else {
				properties.add("cacheLevelName", "CACHE_" + cache.toUpperCase(Locale.ROOT));
			}
		}

		Integer acknowledgeMode = parseAcknowledgeMode(containerEle, parserContext);
		if (acknowledgeMode != null) {
			if (acknowledgeMode == Session.SESSION_TRANSACTED) {
				properties.add("sessionTransacted", Boolean.TRUE);
			}
			else {
			properties.add("sessionAcknowledgeMode", acknowledgeMode);
			}
		}

		String transactionManagerBeanName = containerEle.getAttribute(TRANSACTION_MANAGER_ATTRIBUTE);
		if (StringUtils.hasText(transactionManagerBeanName)) {
			if (isSimpleContainer) {
				parserContext.getReaderContext().error(
						"'transaction-manager' attribute not supported for customized container of type \"simple\".", containerEle);
			}
			else {
				properties.add("transactionManager", new RuntimeBeanReference(transactionManagerBeanName));
			}
		}

		String concurrency = containerEle.getAttribute(CONCURRENCY_ATTRIBUTE);
		if (StringUtils.hasText(concurrency)) {
			properties.add("concurrency", concurrency);
		}

		String prefetch = containerEle.getAttribute(PREFETCH_ATTRIBUTE);
		if (StringUtils.hasText(prefetch)) {
			if (!isSimpleContainer) {
				properties.add("maxMessagesPerTask", prefetch);
			}
		}

		String phase = containerEle.getAttribute(PHASE_ATTRIBUTE);
		if (StringUtils.hasText(phase)) {
			properties.add("phase", phase);
		}

		String receiveTimeout = containerEle.getAttribute(RECEIVE_TIMEOUT_ATTRIBUTE);
		if (StringUtils.hasText(receiveTimeout)) {
			if (!isSimpleContainer) {
				properties.add("receiveTimeout", receiveTimeout);
			}
		}

		String backOffBeanName = containerEle.getAttribute(BACK_OFF_ATTRIBUTE);
		if (StringUtils.hasText(backOffBeanName)) {
			if (!isSimpleContainer) {
				properties.add("backOff", new RuntimeBeanReference(backOffBeanName));
			}
		}
		else { // No need to consider this if back-off is set
			String recoveryInterval = containerEle.getAttribute(RECOVERY_INTERVAL_ATTRIBUTE);
			if (StringUtils.hasText(recoveryInterval)) {
				if (!isSimpleContainer) {
					properties.add("recoveryInterval", recoveryInterval);
				}
			}
		}

		return properties;
	}

