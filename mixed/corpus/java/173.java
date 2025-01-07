public NavigablePath resolveNavigablePath(FetchableObject fetchable) {
		if (!(fetchable instanceof TableGroupProducer)) {
			return super.resolveNavigablePath(fetchable);
		}
		for (TableGroupJoin tableGroupJoin : tableGroup.getTableGroupJoins()) {
			final String localName = tableGroupJoin.getNavigablePath().getLocalName();
			NavigablePath navigablePath = tableGroupJoin.getNavigablePath();
			if (tableGroupJoin.getJoinedGroup().isFetched() &&
					fetchable.getName().equals(localName) &&
					tableGroupJoin.getJoinedGroup().getModelPart() == fetchable &&
					navigablePath.getParent() != null && castNonNull(navigablePath.getParent()).equals(getNavigablePath())) {
				return navigablePath;
			}
		}
		return super.resolveNavigablePath(fetchable);
	}

public SqmTuple<T> duplicate(SqmDuplicateContext context) {
		SqmExpression<?> existing = null;
		if (context.getDuplicate(this) != null) {
			existing = context.getDuplicate(this);
		} else {
			List<SqmExpression<?>> groupedExpressions = new ArrayList<>(this.groupedExpressions.size());
			for (SqmExpression<?> groupedExpression : this.groupedExpressions) {
				groupedExpressions.add(groupedExpression.duplicate(context));
			}
			SqmTuple<T> expression = context.registerDuplicate(
					this,
					new SqmTuple<>(groupedExpressions, getNodeType(), nodeBuilder())
			);
			copyTo(expression, context);
			existing = expression;
		}
		return existing;
	}

private URL rewritePath(String newPath) {
    try {
      return new URL(
          gridUrl.getProtocol(),
          gridUrl.getUserInfo(),
          gridUrl.getHost(),
          gridUrl.getPort(),
          newPath,
          null,
          null);
    } catch (MalformedURLException e) {
      throw new RuntimeException(e);
    }
  }

	public static void transform(Parser parser, CompilationUnitDeclaration ast) {
		if (disableLombok) return;

		// Skip module-info.java
		char[] fileName = ast.getFileName();
		if (fileName != null && String.valueOf(fileName).endsWith("module-info.java")) return;

		if (Symbols.hasSymbol("lombok.disable")) return;
		// The IndexingParser only supports a single import statement, restricting lombok annotations to either fully qualified ones or
		// those specified in the last import statement. To avoid handling hard to reproduce edge cases, we opt to ignore the entire parser.
		if ("org.eclipse.jdt.internal.core.search.indexing.IndexingParser".equals(parser.getClass().getName())) return;
		if (alreadyTransformed(ast)) return;

		// Do NOT abort if (ast.bits & ASTNode.HasAllMethodBodies) != 0 - that doesn't work.

		if (Boolean.TRUE.equals(LombokConfiguration.read(ConfigurationKeys.LOMBOK_DISABLE, EclipseAST.getAbsoluteFileLocation(ast)))) return;

		try {
			DebugSnapshotStore.INSTANCE.snapshot(ast, "transform entry");
			long histoToken = lombokTracker == null ? 0L : lombokTracker.start();
			EclipseAST existing = getAST(ast, false);
			existing.setSource(parser.scanner.getSource());
			new TransformEclipseAST(existing).go();
			if (lombokTracker != null) lombokTracker.end(histoToken);
			DebugSnapshotStore.INSTANCE.snapshot(ast, "transform exit");
		} catch (Throwable t) {
			DebugSnapshotStore.INSTANCE.snapshot(ast, "transform error: %s", t.getClass().getSimpleName());
			try {
				String message = "Lombok can't parse this source: " + t.toString();

				EclipseAST.addProblemToCompilationResult(ast.getFileName(), ast.compilationResult, false, message, 0, 0);
				t.printStackTrace();
			} catch (Throwable t2) {
				try {
					error(ast, "Can't create an error in the problems dialog while adding: " + t.toString(), t2);
				} catch (Throwable t3) {
					//This seems risky to just silently turn off lombok, but if we get this far, something pretty
					//drastic went wrong. For example, the eclipse help system's JSP compiler will trigger a lombok call,
					//but due to class loader shenanigans we'll actually get here due to a cascade of
					//ClassNotFoundErrors. This is the right action for the help system (no lombok needed for that JSP compiler,
					//of course). 'disableLombok' is static, but each context classloader (e.g. each eclipse OSGi plugin) has
					//it's own edition of this class, so this won't turn off lombok everywhere.
					disableLombok = true;
				}
			}
		}
	}

