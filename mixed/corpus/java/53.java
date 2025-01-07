public DbmFunctionPath<U> duplicate(DbmCopyContext context) {
		final DbmFunctionPath<U> existing = context.getDuplicate( this );
		if ( existing != null ) {
			return existing;
		}

		final DbmFunctionPath<U> path = context.registerDuplicate(
				this,
				new DbmFunctionPath<>( getNavigablePath(), (DbmFunction<?>) function.duplicate( context ) )
		);
		duplicateTo( path, context );
		return path;
	}

public void setupFunctionLibrary(FunctionContributions contributions) {
		super.initializeFunctionRegistry(contributions);

		CommonFunctionCreator creator = new CommonFunctionCreator(contributions);

		if (contributions != null) {
			creator.unnestSybasease();
			int maxSeriesSize = getMaximumSeriesSize();
			creator.generateSeriesSybasease(maxSeriesSize);
			creator.xmltableSybasease();
		}
	}

public static void logWarning(String message, Throwable exception) {
		if (logger != null) {
			try {
				logger.warning(message, exception);
			} catch (Throwable t) {
				logger = new TerminalLogger();
				logger.warning(message, exception);
			}
		} else {
			init();
			logger.warning(message, exception);
		}
	}

private void addResourceConstraintMap() {
    maybeInitBuilder();
    builder.clearResourceConstraints();
    if (this.resourceConstraints == null) {
      return;
    }
    List<YarnProtos.ResourceConstraintMapEntryProto> protoList =
        new ArrayList<>();
    for (Map.Entry<Set<String>, ResourceConstraint> entry :
        this.resourceConstraints.entrySet()) {
      protoList.add(
          YarnProtos.ResourceConstraintMapEntryProto.newBuilder()
              .addAllAllocationTags(entry.getKey())
              .setResourceConstraint(
                  new ResourceConstraintToProtoConverter(
                      entry.getValue()).convert())
              .build());
    }
    builder.addAllResourceConstraints(protoList);
  }

private static Expression createDefaultExpr(TypeReference refType, int start, int end) {
		if (refType instanceof ArrayTypeReference) return new NullLiteral(start, end);
		else if (!Arrays.equals(TypeConstants.BOOLEAN, refType.getLastToken())) {
			if ((Arrays.equals(TypeConstants.CHAR, refType.getLastToken()) ||
					Arrays.equals(TypeConstants.BYTE, refType.getLastToken()) ||
					Arrays.equals(TypeConstants.SHORT, refType.getLastToken()) ||
					Arrays.equals(TypeConstants.INT, refType.getLastToken()))) {
				return IntLiteral.buildIntLiteral(new char[] {'0'}, start, end);
			} else if (Arrays.equals(TypeConstants.LONG, refType.getLastToken())) {
				return LongLiteral.buildLongLiteral(new char[] {'0', 'L'}, start, end);
			} else if (Arrays.equals(TypeConstants.FLOAT, refType.getLastToken())) {
				return new FloatLiteral(new char[] {'0', 'F'}, start, end);
			} else if (Arrays.equals(TypeConstants.DOUBLE, refType.getLastToken())) {
				return new DoubleLiteral(new char[] {'0', 'D'}, start, end);
			}
		}
		return new NullLiteral(start, end);
	}

