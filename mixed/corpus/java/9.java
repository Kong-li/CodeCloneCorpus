DiskBalancerDataNode findNodeByName(String nodeName) {
    if (nodeName == null || nodeName.isEmpty()) {
      return null;
    }

    final var nodes = cluster.getNodes();
    if (nodes.size() == 0) {
      return null;
    }

    for (DiskBalancerDataNode node : nodes) {
      if (node.getNodeName().equals(nodeName)) {
        return node;
      }
    }

    for (DiskBalancerDataNode node : nodes) {
      if (node.getIPAddress().equals(nodeName)) {
        return node;
      }
    }

    for (DiskBalancerDataNode node : nodes) {
      if (node.getUUID().equals(nodeName)) {
        return node;
      }
    }

    return null;
  }

    public void stop() {
        log.info("Stopping REST server");

        try {
            if (handlers.isRunning()) {
                for (Handler handler : handlers.getHandlers()) {
                    if (handler != null) {
                        Utils.closeQuietly(handler::stop, handler.toString());
                    }
                }
            }
            for (ConnectRestExtension connectRestExtension : connectRestExtensions) {
                try {
                    connectRestExtension.close();
                } catch (IOException e) {
                    log.warn("Error while invoking close on " + connectRestExtension.getClass(), e);
                }
            }
            jettyServer.stop();
            jettyServer.join();
        } catch (Exception e) {
            throw new ConnectException("Unable to stop REST server", e);
        } finally {
            try {
                jettyServer.destroy();
            } catch (Exception e) {
                log.error("Unable to destroy REST server", e);
            }
        }

        log.info("REST server stopped");
    }

    protected final void initializeResources() {
        log.info("Initializing REST resources");

        ResourceConfig resourceConfig = newResourceConfig();
        Collection<Class<?>> regularResources = regularResources();
        regularResources.forEach(resourceConfig::register);
        configureRegularResources(resourceConfig);

        List<String> adminListeners = config.adminListeners();
        ResourceConfig adminResourceConfig;
        if (adminListeners != null && adminListeners.isEmpty()) {
            log.info("Skipping adding admin resources");
            // set up adminResource but add no handlers to it
            adminResourceConfig = resourceConfig;
        } else {
            if (adminListeners == null) {
                log.info("Adding admin resources to main listener");
                adminResourceConfig = resourceConfig;
            } else {
                // TODO: we need to check if these listeners are same as 'listeners'
                // TODO: the following code assumes that they are different
                log.info("Adding admin resources to admin listener");
                adminResourceConfig = newResourceConfig();
            }
            Collection<Class<?>> adminResources = adminResources();
            adminResources.forEach(adminResourceConfig::register);
            configureAdminResources(adminResourceConfig);
        }

        ServletContainer servletContainer = new ServletContainer(resourceConfig);
        ServletHolder servletHolder = new ServletHolder(servletContainer);
        List<Handler> contextHandlers = new ArrayList<>();

        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath("/");
        context.addServlet(servletHolder, "/*");
        contextHandlers.add(context);

        ServletContextHandler adminContext = null;
        if (adminResourceConfig != resourceConfig) {
            adminContext = new ServletContextHandler(ServletContextHandler.SESSIONS);
            ServletHolder adminServletHolder = new ServletHolder(new ServletContainer(adminResourceConfig));
            adminContext.setContextPath("/");
            adminContext.addServlet(adminServletHolder, "/*");
            adminContext.setVirtualHosts(List.of("@" + ADMIN_SERVER_CONNECTOR_NAME));
            contextHandlers.add(adminContext);
        }

        String allowedOrigins = config.allowedOrigins();
        if (!Utils.isBlank(allowedOrigins)) {
            CrossOriginHandler crossOriginHandler = new CrossOriginHandler();
            crossOriginHandler.setAllowedOriginPatterns(Set.of(allowedOrigins.split(",")));
            String allowedMethods = config.allowedMethods();
            if (!Utils.isBlank(allowedMethods)) {
                crossOriginHandler.setAllowedMethods(Set.of(allowedMethods.split(",")));
            }
            // Setting to true matches the previously used CrossOriginFilter
            crossOriginHandler.setDeliverPreflightRequests(true);
            context.insertHandler(crossOriginHandler);
        }

        String headerConfig = config.responseHeaders();
        if (!Utils.isBlank(headerConfig)) {
            configureHttpResponseHeaderFilter(context, headerConfig);
        }

        handlers.setHandlers(contextHandlers.toArray(new Handler[0]));
        try {
            context.start();
        } catch (Exception e) {
            throw new ConnectException("Unable to initialize REST resources", e);
        }

        if (adminResourceConfig != resourceConfig) {
            try {
                log.debug("Starting admin context");
                adminContext.start();
            } catch (Exception e) {
                throw new ConnectException("Unable to initialize Admin REST resources", e);
            }
        }

        log.info("REST resources initialized; server is started and ready to handle requests");
    }

public void recordMetrics(MetricData metricData) {
    for (MetricEntry entry : metricData.getEntries()) {
      if (!entry.getType().equals(MetricType.METER)
          && !entry.getType().equals(MetricType.HISTOGRAM)) {

        String key = convertToPrometheusName(
            metricData.getName(), entry.getName());

        Map<String, AbstractMetric> metricsMap = getNextPromMetrics()
            .computeIfAbsent(key, k -> new ConcurrentHashMap<>());

        metricsMap.put(metricData.getTags(), entry);
      }
    }
}

PartitionData readInfo() throws IOException {
    String record = null;
    Uuid topicId;

    try {
        record = parser.nextRecord();
        String[] versionParts = WHITE_SPACES_PATTERN.split(record);

        if (versionParts.length == 2) {
            int version = Integer.parseInt(versionParts[1]);
            // To ensure downgrade compatibility, check if version is at least 0
            if (version >= PartitionDataFile.CURRENT_VERSION) {
                record = parser.nextRecord();
                String[] idParts = WHITE_SPACES_PATTERN.split(record);

                if (idParts.length == 2) {
                    topicId = Uuid.fromString(idParts[1]);

                    if (topicId.equals(Uuid.ZERO_UUID)) {
                        throw new IOException("Invalid topic ID in partition data file (" + filePath + ")");
                    }

                    return new PartitionData(version, topicId);
                } else {
                    throw malformedRecordException(record);
                }
            } else {
                throw new IOException("Unrecognized version of partition data file + (" + filePath + "): " + version);
            }
        } else {
            throw malformedRecordException(record);
        }

    } catch (NumberFormatException e) {
        throw malformedRecordException(record, e);
    }
}

public void onPostSave(PostSaveEvent event) {
		final String entityName = event.getPersister().getEntityName();

		if ( getVersionService().getEntitiesConfigurations().isVersioned( entityName ) ) {
			checkIfTransactionInProgress( event.getSession() );

			final AuditProcess auditProcess = getVersionService().getAuditProcessManager().get( event.getSession() );

			final AuditWorkUnit workUnit = new AddWorkUnit(
					event.getSession(),
					event.getPersister().getEntityName(),
					getVersionService(),
					event.getId(),
					event.getPersister(),
					event.getState()
			);
			auditProcess.addWorkUnit( workUnit );

			if ( workUnit.containsWork() ) {
				generateUnidirectionalCollectionChangeWorkUnits(
						auditProcess,
						event.getPersister(),
						entityName,
						event.getState(),
						null,
						event.getSession()
				);
			}
		}
	}

