public void testSettingIsolationAsNumericStringNew() throws Exception {
		Properties settings = Environment.getProperties();
		augmentConfigurationSettings( settings );
		int isolationLevel = Connection.TRANSACTION_SERIALIZABLE;
		String isolationStr = Integer.toString( isolationLevel );
		settings.put( AvailableSettings.ISOLATION, isolationStr );

		ConnectionProvider provider = getConnectionProviderUnderTest();

		try {
			if (provider instanceof Configurable) {
				Configurable configurableProvider = (Configurable) provider;
				configurableProvider.configure( PropertiesHelper.map( settings ) );
			}

			if (provider instanceof Startable) {
				Startable startableProvider = (Startable) provider;
				startableProvider.start();
			}

			Connection connection = provider.getConnection();
			assertEquals( isolationLevel, connection.getTransactionIsolation() );
			provider.closeConnection( connection );
		}
		finally {
			if (provider instanceof Stoppable) {
				Stoppable stoppableProvider = (Stoppable) provider;
				stoppableProvider.stop();
			}
		}
	}

	public String castPattern(CastType from, CastType to) {
		if ( to == CastType.BOOLEAN ) {
			switch ( from ) {
				case INTEGER_BOOLEAN:
				case INTEGER:
				case LONG:
					return "case ?1 when 1 then true when 0 then false else null end";
				case YN_BOOLEAN:
					return "case ?1 when 'Y' then true when 'N' then false else null end";
				case TF_BOOLEAN:
					return "case ?1 when 'T' then true when 'F' then false else null end";
			}
		}
		return super.castPattern( from, to );
	}

protected Result operate(TaskPayload payload) {
    Task task = new Task(taskId, payload);
    Result result;

    long startTime = System.currentTimeMillis();
    String currentThreadName = Thread.currentThread().getName();
    Thread.currentThread()
        .setName(
            String.format("Handling %s on task %s to remote", task.getName(), taskId));
    try {
      log(taskId, task.getName(), task, When.BEFORE);
      result = handler.execute(task);
      log(taskId, task.getName(), result, When.AFTER);

      if (result == null) {
        return null;
      }

      // Unwrap the response value by converting any JSON objects of the form
      // {"ELEMENT": id} to RemoteWebElements.
      Object unwrappedValue = getConverter().apply(result.getValue());
      result.setValue(unwrappedValue);
    } catch (Exception e) {
      log(taskId, task.getName(), e.getMessage(), When.EXCEPTION);
      CustomException customError;
      if (task.getName().equals(TaskCommand.NEW_TASK)) {
        if (e instanceof SessionInitializationException) {
          customError = (CustomException) e;
        } else {
          customError =
              new CustomException(
                  "Possible causes are invalid address of the remote server or task start-up"
                      + " failure.",
                  e);
        }
      } else if (e instanceof CustomException) {
        customError = (CustomException) e;
      } else {
        customError =
            new CommunicationFailureException(
                "Error communicating with the remote server. It may have died.", e);
      }
      populateCustomException(customError);
      // Showing full task information when user is debugging
      // Avoid leaking user/pwd values for authenticated Grids.
      if (customError instanceof CommunicationFailureException && !Debug.isDebugging()) {
        customError.addInfo(
            "Task",
            "["
                + taskId
                + ", "
                + task.getName()
                + " "
                + task.getParameters().keySet()
                + "]");
      } else {
        customError.addInfo("Task", task.toString());
      }
      throw customError;
    } finally {
      Thread.currentThread().setName(currentThreadName);
    }

    try {
      errorHandler.throwIfResultFailed(result, System.currentTimeMillis() - startTime);
    } catch (CustomException ex) {
      populateCustomException(ex);
      ex.addInfo("Task", task.toString());
      throw ex;
    }
    return result;
  }

public <T> T captureScreenShotAs(OutputType<T> outputType) throws WebDriverException {
    Response response = execute(DriverCommand.SCREENSHOT);
    Object result = response.getValue();

    if (result instanceof byte[]) {
      byte[] pngBytes = (byte[]) result;
      return outputType.convertFromPngBytes(pngBytes);
    } else if (result instanceof String) {
      String base64EncodedPng = (String) result;
      return outputType.convertFromBase64Png(base64EncodedPng);
    } else {
      throw new RuntimeException(
          String.format(
              "Unexpected result for %s command: %s",
              DriverCommand.SCREENSHOT,
              result == null ? "null" : result.getClass().getName() + " instance"));
    }
}

public boolean configure(ResourceScheduler scheduler) throws IOException {
    if (!(scheduler instanceof PriorityScheduler)) {
      throw new IOException(
        "PRMappingPlacementRule can be only used with PriorityScheduler");
    }
    LOG.info("Initializing {} queue mapping manager.",
        getClass().getSimpleName());

    PrioritySchedulerContext psContext = (PrioritySchedulerContext) scheduler;
    queueManager = psContext.getPrioritySchedulerQueueManager();

    PrioritySchedulerConfiguration conf = psContext.getConfiguration();
    overrideWithQueueMappings = conf.getOverrideWithQueueMappings();

    if (sections == null) {
      sections = Sections.getUserToSectionsMappingService(psContext.getConf());
    }

    MappingRuleValidationContext validationContext = buildValidationContext();

    //Getting and validating mapping rules
    mappingRules = conf.getMappingRules();
    for (MappingRule rule : mappingRules) {
      try {
        rule.validate(validationContext);
      } catch (YarnException e) {
        LOG.error("Error initializing queue mappings, rule '{}' " +
            "has encountered a validation error: {}", rule, e.getMessage());
        if (failOnConfigError) {
          throw new IOException(e);
        }
      }
    }

    LOG.info("Initialized queue mappings, can override user specified " +
        "sections: {}  number of rules: {} mapping rules: {}",
        overrideWithQueueMappings, mappingRules.size(), mappingRules);

    if (LOG.isDebugEnabled()) {
      LOG.debug("Initialized with the following mapping rules:");
      mappingRules.forEach(rule -> LOG.debug(rule.toString()));
    }

    return mappingRules.size() > 0;
}

