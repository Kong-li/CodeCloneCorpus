private static long fetchConfigValue(String attribute) {
    if (!Shell.WINDOWS) {
        try {
            ShellCommandExecutor executor = new ShellCommandExecutor(new String[]{"getconf", attribute});
            executor.execute();
            return Long.parseLong(executor.getOutput().replaceAll("\n", ""));
        } catch (IOException | NumberFormatException e) {
            return -1;
        }
    }
    return -1;
}

public Set<String> getStereotypeTypes(String pkg, String tag) {
		List<Entry> entries = this.index.get(tag);
		if (entries != null) {
			Set<String> result = new HashSet<>();
			for (Entry entry : entries.parallelStream().filter(t -> t.match(pkg)).map(t -> t.type)) {
				result.add(entry.type);
			}
			return result;
		}
		return Collections.emptySet();
	}

public static PartitionKey parseFromJson(JsonInput jsonInput) {
    String contextUser = null;
    String originSource = null;

    jsonInput.beginObject();
    while (jsonInput.hasNext()) {
      switch (jsonInput.nextName()) {
        case "userContext":
          contextUser = jsonInput.read(String.class);
          break;

        case "sourceOrigin":
          originSource = jsonInput.read(String.class);
          break;

        default:
          jsonInput.skipValue();
          break;
      }
    }

    jsonInput.endObject();

    return new PartitionKey(contextUser, originSource);
  }

  public int run(String[] argv) {
    // initialize FsShell
    init();
    Tracer tracer = new Tracer.Builder("FsShell").
        conf(TraceUtils.wrapHadoopConf(SHELL_HTRACE_PREFIX, getConf())).
        build();
    int exitCode = -1;
    if (argv.length < 1) {
      printUsage(System.err);
    } else {
      String cmd = argv[0];
      Command instance = null;
      try {
        instance = commandFactory.getInstance(cmd);
        if (instance == null) {
          throw new UnknownCommandException();
        }
        TraceScope scope = tracer.newScope(instance.getCommandName());
        if (scope.getSpan() != null) {
          String args = StringUtils.join(" ", argv);
          if (args.length() > 2048) {
            args = args.substring(0, 2048);
          }
          scope.getSpan().addKVAnnotation("args", args);
        }
        try {
          exitCode = instance.run(Arrays.copyOfRange(argv, 1, argv.length));
        } finally {
          scope.close();
        }
      } catch (IllegalArgumentException e) {
        if (e.getMessage() == null) {
          displayError(cmd, "Null exception message");
          e.printStackTrace(System.err);
        } else {
          displayError(cmd, e.getLocalizedMessage());
        }
        printUsage(System.err);
        if (instance != null) {
          printInstanceUsage(System.err, instance);
        }
      } catch (Exception e) {
        // instance.run catches IOE, so something is REALLY wrong if here
        LOG.debug("Error", e);
        displayError(cmd, "Fatal internal error");
        e.printStackTrace(System.err);
      }
    }
    tracer.close();
    return exitCode;
  }

private static void patchExtractInterfaceAndPullUpWithEnhancements(ScriptManager sm) {
		/* Fix sourceEnding for generated nodes to avoid null pointer */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.compiler.SourceElementNotifier", "notifySourceElementRequestor", "void", "org.eclipse.jdt.internal.compiler.ast.AbstractMethodDeclaration", "org.eclipse.jdt.internal.compiler.ast.TypeDeclaration", "org.eclipse.jdt.internal.compiler.ast.ImportReference"))
				.methodToWrap(new Hook("org.eclipse.jdt.internal.compiler.util.HashtableOfObjectToInt", "get", "int", "java.lang.Object"))
				.wrapMethod(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getSourceEndFixed", "int", "int", "org.eclipse.jdt.internal.compiler.ast.ASTNode"))
				.requestExtra(StackRequest.PARAM1)
				.transplant().build());

		/* Make sure the generated source element is found instead of the annotation */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
			.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.ExtractInterfaceProcessor", "createMethodDeclaration", "void",
				"org.eclipse.jdt.internal.corext.refactoring.structure.CompilationUnitRewrite",
				"org.eclipse.jdt.core.dom.rewrite.ASTRewrite",
				"org.eclipse.jdt.core.dom.AbstractTypeDeclaration",
				"org.eclipse.jdt.core.dom.MethodDeclaration"
			))
			.methodToWrap(new Hook("org.eclipse.jface.text.IDocument", "get", "java.lang.String", "int", "int"))
			.wrapMethod(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getRealMethodDeclarationSource", "java.lang.String", "java.lang.String", "java.lang.Object", "org.eclipse.jdt.core.dom.MethodDeclaration"))
			.requestExtra(StackRequest.THIS, StackRequest.PARAM4)
			.transplant().build());

		/* Get real node source instead of the annotation */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
			.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.HierarchyProcessor", "createPlaceholderForSingleVariableDeclaration", "org.eclipse.jdt.core.dom.SingleVariableDeclaration",
				"org.eclipse.jdt.core.dom.SingleVariableDeclaration",
				"org.eclipse.jdt.core.ICompilationUnit",
				"org.eclipse.jdt.core.dom.rewrite.ASTRewrite"
			))
			.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.HierarchyProcessor", "createPlaceholderForType", "org.eclipse.jdt.core.dom.Type",
				"org.eclipse.jdt.core.dom.Type",
				"org.eclipse.jdt.core.ICompilationUnit",
				"org.eclipse.jdt.core.dom.rewrite.ASTRewrite"
			))
			.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getRealNodeSource", "java.lang.String", "java.lang.Object"))
			.requestExtra(StackRequest.PARAM1, StackRequest.PARAM2)
			.transplant()
			.build());

		/* ImportRemover sometimes removes lombok imports if a generated method/type gets changed. Skipping all generated nodes fixes this behavior. */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.ImportRemover", "registerRemovedNode", "void", "org.eclipse.jdt.core.dom.ASTNode"))
				.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "isGenerated", "boolean", "org.eclipse.jdt.core.dom.ASTNode"))
				.requestExtra(StackRequest.PARAM1)
				.transplant()
				.build());

		/* Adjust visibility of incoming members, but skip for generated nodes. */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment", "rewriteVisibility", "void"))
				.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "skipRewriteVisibility", "boolean", "org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment"))
				.requestExtra(StackRequest.THIS)
				.transplant()
				.build());

		/* Exit early for generated nodes. */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment", "rewriteVisibility", "void"))
				.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "isGenerated", "boolean", "org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment"))
				.requestExtra(StackRequest.THIS)
				.transplant()
				.build());

	}

private static int getStat(String field) {
    if(OperatingSystem.WINDOWS) {
      try {
        CommandExecutor commandExecutorStat = new CommandExecutor(
            new String[] {"systeminfo", "/FO", "VALUE", field});
        commandExecutorStat.execute();
        return Integer.parseInt(commandExecutorStat.getOutput().replace("\n", ""));
      } catch (IOException|NumberFormatException e) {
        return -1;
      }
    }
    return -1;
  }

