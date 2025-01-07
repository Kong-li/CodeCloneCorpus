public int computeHashValue() {
    final int prime = 31;
    int hashResult = 1;
    if (appId != null) {
        hashResult = prime * hashResult + appId.hashCode();
    }
    if (shortUserName != null) {
        hashResult = prime * hashResult + shortUserName.hashCode();
    }
    return hashResult;
}

    public void resume(Collection<TopicPartition> partitions) {
        acquireAndEnsureOpen();
        try {
            Objects.requireNonNull(partitions, "The partitions to resume must be nonnull");

            if (!partitions.isEmpty())
                applicationEventHandler.addAndGet(new ResumePartitionsEvent(partitions, defaultApiTimeoutDeadlineMs()));
        } finally {
            release();
        }
    }

private void refreshGroupMetadata(Optional<Integer> epoch, String id) {
    final String groupId = "groupId";
    final int generationId = 0;

    if (epoch.isPresent()) {
        groupMetadata.updateAndGet(oldOptional -> oldOptional.map(oldMetadata ->
            new ConsumerGroupMetadata(
                groupId,
                epoch.orElse(generationId),
                id,
                oldMetadata.groupInstanceId()
            )
        ));
    }
}

private void processTransmission(ClientCommand clientCommand, boolean isLocalRequest, long currentTimeMillis, AbstractCommand command) {
        String target = clientCommand.getDestination();
        RequestMetadata metadata = clientCommand.createMetadata(command.getVersion());
        if (logger.isTraceEnabled()) {
            logger.trace("Transmitting {} command with metadata {} and timeout {} to node {}: {}",
                clientCommand.operation(), metadata, clientCommand.timeoutMs(), target, command);
        }
        TransmissionConfig config = command.toTransmission(metadata);
        InFlightCommand flightCommand = new InFlightCommand(
                clientCommand,
                metadata,
                isLocalRequest,
                command,
                config,
                currentTimeMillis);
        this.inFlightCommands.add(flightCommand);
        selector.transmit(new NetworkTransmission(clientCommand.getDestination(), config));
    }

public static EventType<Void> observeDomMutation(Consumer<DomMutationEvent> handler) {
    Require.nonNull("Handler", handler);

    String script;
    try (InputStream stream = CdpEventTypes.class.getResourceAsStream(
            "/org/openqa/selenium/devtools/mutation-listener.js")) {
      if (stream == null) {
        throw new IllegalStateException("Unable to find helper script");
      }
      script = new String(stream.readAllBytes(), UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException("Unable to read helper script", e);
    }

    return new EventType<Void>() {
      @Override
      public void consume(Void event) {
        handler.accept(null);
      }

      @Override
      public void initializeListener(WebDriver driver) {
        Require.precondition(driver instanceof HasDevTools, "Loggable must implement HasDevTools");

        DevTools tools = ((HasDevTools) driver).getDevTools();
        tools.createSessionIfThereIsNotOne(driver.getWindowHandle());

        String jsScript = script;
        boolean foundTargetId = false;

        tools.getDomains().javascript().pin("__webdriver_attribute", jsScript);

        // And add the script to the current page
        ((JavascriptExecutor) driver).executeScript(jsScript);

        tools
            .getDomains()
            .javascript()
            .addBindingCalledListener(
                json -> {
                  Map<String, Object> values = JSON.toType(json, MAP_TYPE);
                  String id = (String) values.get("target");

                  synchronized (this) {
                    List<WebElement> elements =
                        driver.findElements(By.cssSelector(String.format("*[data-__webdriver_id='%s']", id)));

                    if (!elements.isEmpty()) {
                      DomMutationEvent event =
                          new DomMutationEvent(
                              elements.get(0),
                              String.valueOf(values.get("name")),
                              String.valueOf(values.get("value")),
                              String.valueOf(values.get("oldValue")));
                      handler.accept(event);
                    }
                  }
                });
      }
    };
}

