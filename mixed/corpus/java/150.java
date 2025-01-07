public ByteBuffer loadBinary() throws TException {
    int length = readVarint32();
    if (0 != length) {
        getTransport().checkReadBytesAvailable(length);
        boolean bufferSufficient = trans_.getBytesRemainingInBuffer() >= length;
        if (bufferSufficient) {
            byte[] tempBuffer = trans_.getBuffer();
            ByteBuffer bb = ByteBuffer.wrap(tempBuffer, trans_.getBufferPosition(), length);
            trans_.consumeBuffer(length);
            return bb;
        }
    }

    byte[] buf = new byte[length];
    trans_.readAll(buf, 0, length);
    return ByteBuffer.wrap(buf);
}

public void initialize(GeneratorInitializationContext initContext, Config properties) throws MappingException {
		final ServiceCatalog serviceCatalog = initContext.getServiceCatalog();
		final DatabaseAdapter databaseAdapter = serviceCatalog.requireService(DatabaseAdapter.class);
		final Schema schema = databaseAdapter.getSchema();

		this.entityType = initContext.getType();

		final QualifiedName sequenceName = determineNextSequenceName(properties, databaseAdapter, serviceCatalog);
		final int startIndex = determineInitialValue(properties);
		int stepSize = determineIncrementSize(properties);
		final OptimizationStrategy optimizationPlan = determineOptimizationStrategy(properties, stepSize);

		boolean forceTableUse = getBoolean(FORCE_TBL_PARAM, properties);
		final boolean physicalSequence = isPhysicalSequence(databaseAdapter, forceTableUse);

		stepSize = adjustIncrementSize(
				databaseAdapter,
				sequenceName,
				stepSize,
				physicalSequence,
				optimizationPlan,
				serviceCatalog,
				determineContributor(properties),
				initContext
		);

		if (physicalSequence
				&& optimizationPlan.isPooled()
				&& !schema.getSequenceSupport().supportsPooledSequences()) {
			forceTableUse = true;
			LOG.forcingTableUse();
		}

		this.databaseConfiguration = buildDatabaseStructure(
				entityType,
				properties,
				databaseAdapter,
				forceTableUse,
				sequenceName,
				startIndex,
				stepSize
		);
		optimizer = OptimizerFactory.buildOptimizer(
				optimizationPlan,
				entityType.getReturnedClass(),
				stepSize,
				getInt(INITIAL_PARAM, properties, -1)
		);
		databaseConfiguration.configure(optimizer);

		options = properties.getProperty(OPTIONS);
	}

private ECSchema parseSchema(Element schemaElement) {
    Map<String, String> options = new HashMap<>();
    NodeList nodes = schemaElement.getChildNodes();

    for (int i = 0; i < nodes.getLength(); i++) {
        Node node = nodes.item(i);
        if (node instanceof Element) {
            Element fieldElement = (Element) node;
            String name = fieldElement.getTagName();
            if ("k".equals(name)) {
                name = "numDataUnits";
            } else if ("m".equals(name)) {
                name = "numParityUnits";
            }

            Text textNode = (Text) fieldElement.getFirstChild();
            if (textNode != null) {
                String value = textNode.getData().trim();
                options.put(name, value);
            } else {
                throw new IllegalArgumentException("Value of <" + name + "> is null");
            }
        }
    }

    return new ECSchema(options);
}

private synchronized void updateProgress(String taskStatus) {
    int tasksCompleted = totalTasks - remainingTasks;
    long totalTimeSpent = timeTracker.getTotalTimeSpent();
    if (totalTimeSpent == 0) totalTimeSpent = 1;
    float workRate = (float) totalWorkDone / totalTimeSpent;
    float productivity = workRate * TASKS_PER_SECOND_TO_HZ;
    progressIndicator.set((float) tasksCompleted / totalTasks);
    String statusInfo = tasksCompleted + " / " + totalTasks + " completed.";
    taskState.setStateString(statusInfo);

    if (taskStatus != null) {
      progressIndicator.setStatus(taskStatus + " Aggregated productivity(" +
          tasksCompleted + " of " + totalTasks + " at " +
      hzFormat.format(productivity) + " Hz)");
    } else {
      progressIndicator.setStatus("processing(" + tasksCompleted + " of " + totalTasks + " at "
          + hzFormat.format(productivity) + " Hz)");
    }
  }

private Implementation fieldLoader(AnnotatedFieldDesc desc) {
		if (!enhancementContext.hasLazyLoadableAttributes(managedCtClass) || enhancementContext.isMappedSuperclassClass(managedCtClass)) {
			return FieldAccessor.ofField(desc.getName()).in(desc.getDeclaringType().asErasure());
		}
		var lazy = enhancementContext.isLazyLoadable(desc);
		if (lazy && !desc.getDeclaringType().asErasure().equals(managedCtClass)) {
			return new Implementation.Simple(new FieldMethodReader(managedCtClass, desc));
		} else if (!lazy) {
			return FieldAccessor.ofField(desc.getName()).in(desc.getDeclaringType().asErasure());
		}
		return new Implementation.Simple(FieldReaderAppender.of(managedCtClass, desc));
	}

