private String[] gatherDocuments() {
		List<String> docs = new LinkedList<>();
		for ( DocumentSet docSet : documentSets ) {
			final FolderScanner fs = docSet.getFolderScanner( getProject() );
			final String[] fsDocs = fs.getIncludedDocuments();
			for ( String fsDocName : fsDocs ) {
				File d = new File( fsDocName );
				if ( !d.isFile() ) {
					d = new File( fs.getBaseDir(), fsDocName );
				}

				docs.add( d.getAbsolutePath() );
			}
		}
		return ArrayHelper.toStringArray( docs );
	}

boolean isClientStateWithMoreCapacity(ClientState comparison) {
    if (capacity <= 0L) {
            throw new IllegalStateException("Current ClientState's capacity must be greater than 0.");
        }

        if (comparison.capacity <= 0L) {
            throw new IllegalStateException("Other ClientState's capacity must be greater than 0");
        }

        double currentLoad = (double) assignedTaskCount() / capacity;
        double otherLoad = (double) comparison.assignedTaskCount() / comparison.capacity;

        if (currentLoad > otherLoad) {
            return false;
        } else if (currentLoad < otherLoad) {
            return true;
        } else {
            return capacity > comparison.capacity;
        }
    }

	public MethodDeclaration createWith(TypeDeclaration parent, EclipseNode fieldNode, String name, int modifier, EclipseNode sourceNode, List<Annotation> onMethod, List<Annotation> onParam, boolean makeAbstract) {
		ASTNode source = sourceNode.get();
		if (name == null) return null;
		FieldDeclaration field = (FieldDeclaration) fieldNode.get();
		int pS = source.sourceStart, pE = source.sourceEnd;
		long p = (long) pS << 32 | pE;
		MethodDeclaration method = new MethodDeclaration(parent.compilationResult);
		AnnotationValues<Accessors> accessors = getAccessorsForField(fieldNode);
		if (makeAbstract) modifier |= ClassFileConstants.AccAbstract | ExtraCompilerModifiers.AccSemicolonBody;
		if (shouldMakeFinal(fieldNode, accessors)) modifier |= ClassFileConstants.AccFinal;
		method.modifiers = modifier;
		method.returnType = cloneSelfType(fieldNode, source);
		if (method.returnType == null) return null;

		Annotation[] deprecated = null, checkerFramework = null;
		if (isFieldDeprecated(fieldNode)) deprecated = new Annotation[] { generateDeprecatedAnnotation(source) };
		if (getCheckerFrameworkVersion(fieldNode).generateSideEffectFree()) checkerFramework = new Annotation[] { generateNamedAnnotation(source, CheckerFrameworkVersion.NAME__SIDE_EFFECT_FREE) };

		method.annotations = copyAnnotations(source, onMethod.toArray(new Annotation[0]), checkerFramework, deprecated);
		Argument param = new Argument(field.name, p, copyType(field.type, source), ClassFileConstants.AccFinal);
		param.sourceStart = pS; param.sourceEnd = pE;
		method.arguments = new Argument[] { param };
		method.selector = name.toCharArray();
		method.binding = null;
		method.thrownExceptions = null;
		method.typeParameters = null;
		method.bits |= ECLIPSE_DO_NOT_TOUCH_FLAG;

		Annotation[] copyableAnnotations = findCopyableAnnotations(fieldNode);

		if (!makeAbstract) {
			List<Expression> args = new ArrayList<Expression>();
			for (EclipseNode child : fieldNode.up().down()) {
				if (child.getKind() != Kind.FIELD) continue;
				FieldDeclaration childDecl = (FieldDeclaration) child.get();
				// Skip fields that start with $
				if (childDecl.name != null && childDecl.name.length > 0 && childDecl.name[0] == '$') continue;
				long fieldFlags = childDecl.modifiers;
				// Skip static fields.
				if ((fieldFlags & ClassFileConstants.AccStatic) != 0) continue;
				// Skip initialized final fields.
				if (((fieldFlags & ClassFileConstants.AccFinal) != 0) && childDecl.initialization != null) continue;
				if (child.get() == fieldNode.get()) {
					args.add(new SingleNameReference(field.name, p));
				} else {
					args.add(createFieldAccessor(child, FieldAccess.ALWAYS_FIELD, source));
				}
			}

			AllocationExpression constructorCall = new AllocationExpression();
			constructorCall.arguments = args.toArray(new Expression[0]);
			constructorCall.type = cloneSelfType(fieldNode, source);

			Expression identityCheck = new EqualExpression(
					createFieldAccessor(fieldNode, FieldAccess.ALWAYS_FIELD, source),
					new SingleNameReference(field.name, p),
					OperatorIds.EQUAL_EQUAL);
			ThisReference thisRef = new ThisReference(pS, pE);
			Expression conditional = new ConditionalExpression(identityCheck, thisRef, constructorCall);
			Statement returnStatement = new ReturnStatement(conditional, pS, pE);
			method.bodyStart = method.declarationSourceStart = method.sourceStart = source.sourceStart;
			method.bodyEnd = method.declarationSourceEnd = method.sourceEnd = source.sourceEnd;

			List<Statement> statements = new ArrayList<Statement>(5);
			if (hasNonNullAnnotations(fieldNode)) {
				Statement nullCheck = generateNullCheck(field, sourceNode, null);
				if (nullCheck != null) statements.add(nullCheck);
			}
			statements.add(returnStatement);

			method.statements = statements.toArray(new Statement[0]);
		}
		param.annotations = copyAnnotations(source, copyableAnnotations, onParam.toArray(new Annotation[0]));

		EclipseHandlerUtil.createRelevantNonNullAnnotation(fieldNode, method);

		method.traverse(new SetGeneratedByVisitor(source), parent.scope);
		copyJavadoc(fieldNode, method, CopyJavadoc.WITH);
		return method;
	}

public static void ensurePathIsDirectory(Path path, String argumentName) {
    validatePath(path, argumentName);
    if (!Files.isDirectory(path)) {
        String message = String.format("Path %s (%s) must be a directory.", argumentName, path);
        throw new IllegalArgumentException(message);
    }
}

private static void validatePath(Path path, String argName) {
    checkPathExists(path, argName);
}

