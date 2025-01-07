public String toDebugString() {
    boolean hasData = fetchedData != null;
    int startOffset = logStartOffset;
    long endOffset = logEndOffset;
    return "LogReadInfo(" +
            "hasData=" + hasData +
            ", divergingEpoch=" + divergingEpoch +
            ", highWatermark=" + highWatermark +
            ", startOffset=" + startOffset +
            ", endOffset=" + endOffset +
            ", lastStableOffset=" + lastStableOffset +
            ')';
}

public void handlePostCommit(final boolean forceCheckpoint) {
        switch (currentTaskState()) {
            case INITIALIZED:
                // We should never write a checkpoint for an INITIALIZED task as we may overwrite an existing checkpoint
                // with empty uninitialized offsets
                log.debug("Skipped writing checkpoint for {} task", currentTaskState());

                break;

            case RECOVERING:
            case PAUSED:
                maybeCreateCheckpoint(forceCheckpoint);
                log.debug("Completed commit for {} task with force checkpoint {}", currentTaskState(), forceCheckpoint);

                break;

            case OPERATING:
                if (forceCheckpoint || !endOfStreamEnabled) {
                    maybeCreateCheckpoint(forceCheckpoint);
                }
                log.debug("Completed commit for {} task with eos {}, force checkpoint {}", currentTaskState(), endOfStreamEnabled, forceCheckpoint);

                break;

            case TERMINATED:
                throw new IllegalStateException("Illegal state " + currentTaskState() + " while post committing active task " + taskId);

            default:
                throw new IllegalStateException("Unknown state " + currentTaskState() + " while post committing active task " + taskId);
        }

        clearCommitIndicators();
    }

  public static void main(String[] args) throws Exception {
    if (args.length < 4) {
      System.out.println("Arguments: <WORKDIR> <MINIKDCPROPERTIES> " +
              "<KEYTABFILE> [<PRINCIPALS>]+");
      System.exit(1);
    }
    File workDir = new File(args[0]);
    if (!workDir.exists()) {
      throw new RuntimeException("Specified work directory does not exists: "
              + workDir.getAbsolutePath());
    }
    Properties conf = createConf();
    File file = new File(args[1]);
    if (!file.exists()) {
      throw new RuntimeException("Specified configuration does not exists: "
              + file.getAbsolutePath());
    }
    Properties userConf = new Properties();
    InputStreamReader r = null;
    try {
      r = new InputStreamReader(new FileInputStream(file),
          StandardCharsets.UTF_8);
      userConf.load(r);
    } finally {
      if (r != null) {
        r.close();
      }
    }
    for (Map.Entry<?, ?> entry : userConf.entrySet()) {
      conf.put(entry.getKey(), entry.getValue());
    }
    final MiniKdc miniKdc = new MiniKdc(conf, workDir);
    miniKdc.start();
    File krb5conf = new File(workDir, "krb5.conf");
    if (miniKdc.getKrb5conf().renameTo(krb5conf)) {
      File keytabFile = new File(args[2]).getAbsoluteFile();
      String[] principals = new String[args.length - 3];
      System.arraycopy(args, 3, principals, 0, args.length - 3);
      miniKdc.createPrincipal(keytabFile, principals);
      System.out.println();
      System.out.println("Standalone MiniKdc Running");
      System.out.println("---------------------------------------------------");
      System.out.println("  Realm           : " + miniKdc.getRealm());
      System.out.println("  Running at      : " + miniKdc.getHost() + ":" +
              miniKdc.getHost());
      System.out.println("  krb5conf        : " + krb5conf);
      System.out.println();
      System.out.println("  created keytab  : " + keytabFile);
      System.out.println("  with principals : " + Arrays.asList(principals));
      System.out.println();
      System.out.println(" Do <CTRL-C> or kill <PID> to stop it");
      System.out.println("---------------------------------------------------");
      System.out.println();
      Runtime.getRuntime().addShutdownHook(new Thread() {
        @Override
        public void run() {
          miniKdc.stop();
        }
      });
    } else {
      throw new RuntimeException("Cannot rename KDC's krb5conf to "
              + krb5conf.getAbsolutePath());
    }
  }

    public boolean maybePunctuateSystemTime() {
        final long systemTime = time.milliseconds();

        final boolean punctuated = systemTimePunctuationQueue.maybePunctuate(systemTime, PunctuationType.WALL_CLOCK_TIME, this);

        if (punctuated) {
            commitNeeded = true;
        }

        return punctuated;
    }

public void processMethod(EclipseNode node, AbstractMethodDeclaration declaration, List<DeclaredException> exceptions) {
		if (!declaration.isAbstract()) {
			node.addError("@SneakyThrows can only be used on concrete methods.");
			return;
		}

		Statement[] statements = declaration.statements;
		if (statements == null || statements.length == 0) {
			if (declaration instanceof ConstructorDeclaration) {
				ConstructorCall constructorCall = ((ConstructorDeclaration) declaration).constructorCall;
				boolean hasExplicitConstructorCall = constructorCall != null && !constructorCall.isImplicitSuper() && !constructorCall.isImplicitThis();

				if (hasExplicitConstructorCall) {
					node.addWarning("Calls to sibling / super constructors are always excluded from @SneakyThrows; @SneakyThrows has been ignored because there is no other code in this constructor.");
				} else {
					node.addWarning("This method or constructor is empty; @SneakyThrows has been ignored.");
				}

				return;
			}
		}

		for (DeclaredException exception : exceptions) {
			if (statements != null && statements.length > 0) {
				statements = new Statement[] { buildTryCatchBlock(statements, exception, exception.node, declaration) };
			}
		}

		declaration.statements = statements;
		node.up().rebuild();
	}

	private Statement buildTryCatchBlock(Statement[] originalStatements, DeclaredException exception, Node node, AbstractMethodDeclaration methodDeclaration) {
		TryCatchBlock tryCatchBlock = new TryCatchBlock();
		tryCatchBlock.setStatements(originalStatements);
		tryCatchBlock.addException(exception.exceptionType);

		return tryCatchBlock;
	}

public void addFunctions(FunctionContributions functionContributions) {
		final String className = this.getClass().getCanonicalName();
		HSMessageLogger.SPATIAL_MSG_LOGGER.functionContributions(className);
		SqlServerSqmFunctionDescriptors functions = new SqlServerSqmFunctionDescriptors(functionContributions);
		SqmFunctionRegistry functionRegistry = functionContributions.getFunctionRegistry();
		for (Map.Entry<String, Object> entry : functions.asMap().entrySet()) {
			functionRegistry.register(entry.getKey(), (SqmFunctionDescriptor) entry.getValue());
			if (entry.getValue() instanceof SqmFunctionDescriptorWithAltName && !((SqmFunctionDescriptorWithAltName) entry.getValue()).getAltName().isEmpty()) {
				functionRegistry.registerAlternateKey(((SqmFunctionDescriptorWithAltName) entry.getValue()).getAltName(), entry.getKey());
			}
		}
	}

