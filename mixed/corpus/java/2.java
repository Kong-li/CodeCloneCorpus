	public SqmBasicValuedSimplePath<T> copy(SqmCopyContext context) {
		final SqmBasicValuedSimplePath<T> existing = context.getCopy( this );
		if ( existing != null ) {
			return existing;
		}

		final SqmPath<?> lhsCopy = getLhs().copy( context );
		final SqmBasicValuedSimplePath<T> path = context.registerCopy(
				this,
				new SqmBasicValuedSimplePath<>(
						getNavigablePathCopy( lhsCopy ),
						getModel(),
						lhsCopy,
						getExplicitAlias(),
						nodeBuilder()
				)
		);
		copyTo( path, context );
		return path;
	}

private void createNamedGetterMethodForBuilder(WorkerJob job, WorkerFieldData wfd, boolean obsolescence, String suffix) {
		TypeDeclaration td = (TypeDeclaration) job.workerType.get();
		EclipseNode fieldNode = wfd.createdFields.get(0);
		AbstractMethodDeclaration[] existing = td.methods;
		if (existing == null) existing = EMPTY_METHODS;
		int len = existing.length;
		String getterSuffix = suffix.isEmpty() ? "get" : suffix;
		String getterName;
		if (job.oldSyntax) {
			getterName = suffix.isEmpty() ? new String(wfd.name) : HandlerUtil.buildAccessorName(job.sourceNode, getterSuffix, new String(wfd.name));
		} else {
			getterName = HandlerUtil.buildAccessorName(job.sourceNode, getterSuffix, new String(wfd.name));
		}

		for (int i = 0; i < len; i++) {
			if (!(existing[i] instanceof MethodDeclaration)) continue;
			char[] existingName = existing[i].selector;
			if (Arrays.equals(getterName.toCharArray(), existingName) && !isTolerate(fieldNode, existing[i])) return;
		}

		List<Annotation> methodAnnsList = Collections.<Annotation>emptyList();
		Annotation[] methodAnns = EclipseHandlerUtil.findCopyableToGetterAnnotations(wfd.originalFieldNode);
		if (methodAnns != null && methodAnns.length > 0) methodAnnsList = Arrays.asList(methodAnns);
		ASTNode source = job.sourceNode.get();
		MethodDeclaration getter = HandleGetter.createGetter(td, obsolescence, fieldNode, getterName, wfd.name, wfd.nameOfGetFlag, job.oldSyntax, toEclipseModifier(job.accessInners),
			job.sourceNode, methodAnnsList, wfd.annotations != null ? Arrays.asList(copyAnnotations(source, wfd.annotations)) : Collections.<Annotation>emptyList());
		if (job.sourceNode.up().getKind() == Kind.METHOD) {
			copyJavadocFromParam(wfd.originalFieldNode.up(), getter, td, new String(wfd.name));
		} else {
			copyJavadoc(wfd.originalFieldNode, getter, td, CopyJavadoc.GETTER, true);
		}
		injectMethod(job.workerType, getter);
	}

private MethodDeclaration generateBuilderMethod(BuilderJob job, TypeParameter[] typeParams, String prefix) {
		int start = job.source.sourceStart;
		int end = job.source.sourceEnd;
		long position = job.getPos();

		MethodDeclaration method = job.createNewMethodDeclaration();
		method.selector = BUILDER_METHOD_NAME;
		method.modifiers = toEclipseModifier(job.accessOuters);
		method.bits |= ECLIPSE_DO_NOT_TOUCH_FLAG;

		method.returnType = job.createBuilderTypeReference();
		if (job.checkerFramework.generateUnique()) {
			int length = method.returnType.getTypeName().length;
			method.returnType.annotations = new Annotation[length][];
			method.returnType.annotations[length - 1] = new Annotation[]{generateNamedAnnotation(job.source, CheckerFrameworkVersion.NAME__UNIQUE)};
		}

		List<Statement> statements = new ArrayList<>();
		for (BuilderFieldData field : job.builderFields) {
			String setterName = new String(field.name);
			setterName = HandlerUtil.buildAccessorName(job.sourceNode, !prefix.isEmpty() ? prefix : job.oldFluent ? "" : "set", setterName);

			MessageSend messageSend = new MessageSend();
			Expression[] expressions = new Expression[field.singularData == null ? 1 : 2];

			if (field.obtainVia != null && !field.obtainVia.field().isEmpty()) {
				char[] fieldName = field.obtainVia.field().toCharArray();
				for (int i = 0; i < expressions.length; i++) {
					FieldReference ref = new FieldReference(fieldName, 0);
					ref.receiver = new ThisReference(0, 0);
					expressions[i] = ref;
				}
			} else {
				String methodName = field.obtainVia.method();
				boolean isStatic = field.obtainVia.isStatic();
				MessageSend invokeExpr = new MessageSend();

				if (isStatic) {
					if (typeParams != null && typeParams.length > 0) {
						invokeExpr.typeArguments = new TypeReference[typeParams.length];
						for (int j = 0; j < typeParams.length; j++) {
							invokeExpr.typeArguments[j] = new SingleTypeReference(typeParams[j].name, 0);
						}
					}

					invokeExpr.receiver = generateNameReference(job.parentType, 0);
				} else {
					invokeExpr.receiver = new ThisReference(0, 0);
				}

				invokeExpr.selector = methodName.toCharArray();
				if (isStatic) invokeExpr.arguments = new Expression[]{new ThisReference(0, 0)};
				for (int i = 0; i < expressions.length; i++) {
					expressions[i] = new SingleNameReference(field.name, 0L);
				}
			}

			LocalDeclaration var = new LocalDeclaration(BUILDER_TEMP_VAR, start, end);
			var.modifiers |= ClassFileConstants.AccFinal;
			var.type = job.createBuilderTypeReference();
			var.initialization = messageSend;

			if (field.singularData != null) {
				messageSend.target = new SingleNameReference(field.name, 0L);
				statements.add(var);
				statements.add(new ReturnStatement(messageSend, start, end));
			} else {
				statements.add(var);
				statements.add(new ReturnStatement(invokeExpr, start, end));
			}
		}

		createRelevantNonNullAnnotation(job.parentType, method);
		method.traverse(new SetGeneratedByVisitor(job.source), ((TypeDeclaration) job.parentType.get()).scope);

		method.statements = statements.toArray(new Statement[0]);
		return method;
	}

public static void initializeRegistries(RegistryPrimer.Entries entries, DevelopmentContext developmentContext) {
		OrmAnnotationHelper.processOrmAnnotations( entries::addAnnotation );

		developmentContext.getDescriptorRegistry().findDescriptor( UniqueId.class );

//		if ( developmentContext instanceof JandexDevelopmentContext ) {
//			final IndexView jandexIndex = developmentContext.as( JandexDevelopmentContext.class ).getJandexIndex();
//			if ( jandexIndex == null ) {
//				return;
//			}
//
//			final ClassDetailsStore classDetailsStore = developmentContext.getClassDetailsStore();
//			final AnnotationDescriptorRegistry annotationDescriptorRegistry = developmentContext.getAnnotationDescriptorRegistry();
//
//			for ( ClassInfo knownClass : jandexIndex.getKnownClasses() ) {
//				final String className = knownClass.name().toString();
//
//				if ( knownClass.isAnnotation() ) {
//					// it is always safe to load the annotation classes - we will never be enhancing them
//					//noinspection rawtypes
//					final Class annotationClass = developmentContext
//							.getClassLoading()
//							.classForName( className );
//					//noinspection unchecked
//					annotationDescriptorRegistry.resolveDescriptor(
//							annotationClass,
//							(t) -> JdkBuilders.buildAnnotationDescriptor( annotationClass, developmentContext )
//					);
//				}
//
//				resolveClassDetails(
//						className,
//						classDetailsStore,
//						() -> new JandexClassDetails( knownClass, developmentContext )
//				);
//			}
//		}
	}

private void displayNotification(Tone tone, String margin, String info) {
		String[] segments = info.split("\\R");
		writer.print(" ");
		writer.print(format(tone, segments[0]));
		if (segments.length > 1) {
			for (int j = 1; j < segments.length; j++) {
				writer.println();
				writer.print(margin);
				if (StringUtils.isNotBlank(segments[j])) {
					String padding = theme.gap();
					writer.print(format(tone, padding + segments[j]));
				}
			}
		}
	}

