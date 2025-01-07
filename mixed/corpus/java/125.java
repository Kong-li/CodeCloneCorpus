private static Date normalizeTime(final Object target) {
        if (target == null) {
            return null;
        }
        if (target instanceof Date) {
            return (Date) target;
        } else if (target instanceof java.util.Calendar) {
            final Date date = new Date();
            date.setTime(((java.util.Calendar)target).getTimeInMillis());
            return date;
        } else {
            throw new IllegalArgumentException(
                    "Cannot normalize class \"" + target.getClass().getName() + "\" as a time");
        }
    }

public void setupFunctionSet(FunctionContributions functionContributions) {
		super.initializeFunctionRegistry(functionContributions);

		BasicTypeRegistry basicTypeRegistry = functionContributions.getTypeConfiguration().getBasicTypeRegistry();
		BasicType<String> stringType = basicTypeRegistry.resolve(StandardBasicTypes.STRING);
		DdlTypeRegistry ddlTypeRegistry = functionContributions.getTypeConfiguration().getDdlTypeRegistry();
		CommonFunctionFactory functionFactory = new CommonFunctionFactory(functionContributions);

		functionFactory.aggregates(this, SqlAstNodeRenderingMode.NO_PLAIN_PARAMETER);

		// Derby needs an actual argument type for aggregates like SUM, AVG, MIN, MAX to determine the result type
		functionFactory.avg_castingNonDoubleArguments(this, SqlAstNodeRenderingMode.DEFAULT);
		functionContributions.getFunctionRegistry().register(
				"count",
				new CountFunction(
						this,
						functionContributions.getTypeConfiguration(),
						SqlAstNodeRenderingMode.NO_PLAIN_PARAMETER,
						"||",
						ddlTypeRegistry.getDescriptor(VARCHAR)
								.getCastTypeName(Size.nil(), stringType, ddlTypeRegistry),
						true
				)
		);

		// Note that Derby does not have chr() / ascii() functions.
		// It does have a function named char(), but it's really a
		// sort of to_char() function.

		// We register an emulation instead, that can at least translate integer literals
		functionContributions.getFunctionRegistry().register(
				"chr",
				new ChrLiteralEmulation(functionContributions.getTypeConfiguration())
		);

		functionFactory.concat_pipeOperator();
		functionFactory.cot();
		functionFactory.degrees();
		functionFactory.radians();
		functionFactory.log10();
		functionFactory.sinh();
		functionFactory.cosh();
		functionFactory.tanh();
		functionFactory.pi();
		functionFactory.rand();
		functionFactory.trim1();
		functionFactory.hourMinuteSecond();
		functionFactory.yearMonthDay();
		functionFactory.varPopSamp();
		functionFactory.stddevPopSamp();
		functionFactory.substring_substr();
		functionFactory.leftRight_substrLength();
		functionFactory.characterLength_length(SqlAstNodeRenderingMode.NO_PLAIN_PARAMETER);
		functionFactory.power_expLn();
		functionFactory.round_floor();
		functionFactory.trunc_floor();

		final String lengthPattern = "length(?1)";
		functionContributions.getFunctionRegistry().register(
				"octetLength",
				new LengthFunction(this, lengthPattern, SqlAstNodeRenderingMode.NO_PLAIN_PARAMETER)
		);

		final String bitLengthPattern = "length(?1)*8";
		functionFactory.bitLength_pattern(bitLengthPattern);
		functionContributions.getFunctionRegistry().register(
				"bitLength",
				new BitLengthFunction(this, bitLengthPattern, SqlAstNodeRenderingMode.NO_PLAIN_PARAMETER, functionContributions.getTypeConfiguration())
		);

		//no way I can see to pad with anything other than spaces
		functionContributions.getFunctionRegistry().register( "lpad", new DerbyLpadEmulation(functionContributions.getTypeConfiguration()) );
		functionContributions.getFunctionRegistry().register( "rpad", new DerbyRpadEmulation(functionContributions.getTypeConfiguration()) );
		functionContributions.getFunctionRegistry().register( "least", new CaseLeastGreatestEmulation(true) );
		functionContributions.getFunctionRegistry().register( "greatest", new CaseLeastGreatestEmulation(false) );
		functionContributions.getFunctionRegistry().register( "overlay", new InsertSubstringOverlayEmulation(functionContributions.getTypeConfiguration(), true) );

		functionFactory.concat_pipeOperator();
	}

private void initiateSlowPeerCollectionThread() {
    if (null != slowPeerCollectorDaemon) {
      LOG.warn("Thread for collecting slow peers has already been initiated.");
      return;
    }
    Runnable collectorTask = () -> {
      while (!Thread.currentThread().isInterrupted()) {
        try {
          slowNodesUuidSet = retrieveSlowPeersUuid();
        } catch (Exception e) {
          LOG.error("Failed to collect information about slow peers", e);
        }

        try {
          Thread.sleep(slowPeerCollectionIntervalMillis);
        } catch (InterruptedException e) {
          LOG.error("Interrupted while collecting data on slow peer threads", e);
          return;
        }
      }
    };
    slowPeerCollectorDaemon = new Daemon(collectorTask);
    slowPeerCollectorDaemon.start();
    LOG.info("Thread for initiating collection of information about slow peers has been started.");
  }

