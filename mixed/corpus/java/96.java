public Expression calculateFinalExpression(ASTNode src, Expression value) {
		int start = src.sourceStart, end = src.sourceEnd;
		long key = (long)start << 32 | end;
		SingleNameReference resultRef = new SingleNameReference("result", key);
		setGeneratedBy(resultRef, src);
		SingleNameReference primeRef = new SingleNameReference("PRIME", key);
		setGeneratedBy(primeRef, src);

		BinaryExpression multiplyPrime = new BinaryExpression(resultRef, primeRef, OperatorIds.MULTIPLY);
		multiplyPrime.sourceStart = start; multiplyPrime.sourceEnd = end;
		setGeneratedBy(multiplyPrime, src);

		BinaryExpression addValue = new BinaryExpression(multiplyPrime, value, OperatorIds.PLUS);
		addValue.sourceStart = start; addValue.sourceEnd = end;
		setGeneratedBy(addValue, src);

		resultRef = new SingleNameReference("result", key);
		Assignment assignment = new Assignment(resultRef, addValue, end);
		assignment.sourceStart = start; assignment.sourceEnd = assignment.statementEnd = end;
		setGeneratedBy(assignment, src);

		return assignment;
	}

	public MethodDeclaration createCanEqual(EclipseNode type, ASTNode source, List<Annotation> onParam) {
		/* protected boolean canEqual(final java.lang.Object other) {
		 *     return other instanceof Outer.Inner.MyType;
		 * }
		 */
		int pS = source.sourceStart; int pE = source.sourceEnd;
		long p = (long)pS << 32 | pE;

		char[] otherName = "other".toCharArray();

		MethodDeclaration method = new MethodDeclaration(((CompilationUnitDeclaration) type.top().get()).compilationResult);
		setGeneratedBy(method, source);
		method.modifiers = toEclipseModifier(AccessLevel.PROTECTED);
		method.returnType = TypeReference.baseTypeReference(TypeIds.T_boolean, 0);
		method.returnType.sourceStart = pS; method.returnType.sourceEnd = pE;
		setGeneratedBy(method.returnType, source);
		method.selector = "canEqual".toCharArray();
		method.thrownExceptions = null;
		method.typeParameters = null;
		method.bits |= Eclipse.ECLIPSE_DO_NOT_TOUCH_FLAG;
		method.bodyStart = method.declarationSourceStart = method.sourceStart = source.sourceStart;
		method.bodyEnd = method.declarationSourceEnd = method.sourceEnd = source.sourceEnd;
		TypeReference objectRef = new QualifiedTypeReference(TypeConstants.JAVA_LANG_OBJECT, new long[] { p, p, p });
		setGeneratedBy(objectRef, source);
		method.arguments = new Argument[] {new Argument(otherName, 0, objectRef, Modifier.FINAL)};
		method.arguments[0].sourceStart = pS; method.arguments[0].sourceEnd = pE;
		if (!onParam.isEmpty()) method.arguments[0].annotations = onParam.toArray(new Annotation[0]);
		EclipseHandlerUtil.createRelevantNullableAnnotation(type, method.arguments[0], method);
		setGeneratedBy(method.arguments[0], source);

		SingleNameReference otherRef = new SingleNameReference(otherName, p);
		setGeneratedBy(otherRef, source);

		TypeReference typeReference = createTypeReference(type, p, source, false);
		setGeneratedBy(typeReference, source);

		InstanceOfExpression instanceOf = new InstanceOfExpression(otherRef, typeReference);
		instanceOf.sourceStart = pS; instanceOf.sourceEnd = pE;
		setGeneratedBy(instanceOf, source);

		ReturnStatement returnStatement = new ReturnStatement(instanceOf, pS, pE);
		setGeneratedBy(returnStatement, source);

		method.statements = new Statement[] {returnStatement};
		if (getCheckerFrameworkVersion(type).generatePure()) method.annotations = new Annotation[] { generateNamedAnnotation(source, CheckerFrameworkVersion.NAME__PURE) };
		return method;
	}

public void initialize() {
        booleanCasesMap = new HashMap<>();
        booleanCasesMap.put("convertToBoolean", successfulCases(Values::convertToBoolean));
        booleanCasesMap.put("convertToByte", successfulCases(Values::convertToByte));
        booleanCasesMap.put("convertToDate", successfulCases(Values::convertToDate));
        booleanCasesMap.put("convertToDecimal", successfulCases((schema, object) -> Values.convertToDecimal(schema, object, 1)));
        booleanCasesMap.put("convertToDouble", successfulCases(Values::convertToDouble));
        booleanCasesMap.put("convertToFloat", successfulCases(Values::convertToFloat));
        booleanCasesMap.put("convertToShort", successfulCases(Values::convertToShort));
        booleanCasesMap.put("convertToList", successfulCases(Values::convertToList));
        booleanCasesMap.put("convertToMap", successfulCases(Values::convertToMap));
        booleanCasesMap.put("convertToLong", successfulCases(Values::convertToLong));
        booleanCasesMap.put("convertToInteger", successfulCases(Values::convertToInteger));
        booleanCasesMap.put("convertToStruct", successfulCases(Values::convertToStruct));
        booleanCasesMap.put("convertToTime", successfulCases(Values::convertToTime));
        booleanCasesMap.put("convertToTimestamp", successfulCases(Values::convertToTimestamp));
        booleanCasesMap.put("convertToString", successfulCases(Values::convertToString));
        parseStringCases = casesToString(Values::parseString);
    }

	public void collectValueIndexesToCache(BitSet valueIndexes) {
		if ( collectionKeyResult != null ) {
			collectionKeyResult.collectValueIndexesToCache( valueIndexes );
		}
		if ( !getFetchedMapping().getCollectionDescriptor().useShallowQueryCacheLayout() ) {
			collectionValueKeyResult.collectValueIndexesToCache( valueIndexes );
			for ( Fetch fetch : fetches ) {
				fetch.collectValueIndexesToCache( valueIndexes );
			}
		}
	}

private void runStrategy() {
    Preconditions.checkState(lock.isHeldByCurrentThread());
    this.taskExecutor.setExecutable();
    if (this.coordinator.isTerminated()) {
      this.coordinator = Executors.newSingleThreadExecutor();
    }

    this.promise = coordinator.submit(new Runnable() {
      @Override
      public void run() {
        Thread.currentThread().setName("ResourceBalancerThread");
        LOG.info("Executing Resource balancer strategy. Strategy File: {}, Strategy ID: {}",
            strategyFile, strategyID);
        for (Map.Entry<ResourcePair, ResourceBalancerWorkItem> entry :
            taskMap.entrySet()) {
          taskExecutor.setExecutable();
          taskExecutor.transferResources(entry.getKey(), entry.getValue());
        }
      }
    });
  }

