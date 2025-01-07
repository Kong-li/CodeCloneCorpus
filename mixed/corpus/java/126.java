private static void createDelegateMethods(EclipseNode classNode, Collection<BindingPair> methodPairs, DelegateHandler delegateReceiver) {
		CompilationUnitDeclaration cu = (CompilationUnitDeclaration) classNode.top().get();
		List<MethodDeclaration> insertedMethods = new ArrayList<>();
		for (BindingPair pair : methodPairs) {
			EclipseNode annNode = classNode.getAst().get(pair.responsible);
			MethodDeclaration methodDecl = createDelegateMethod(pair.fieldName, classNode, pair, cu.compilationResult, annNode, delegateReceiver);
			if (methodDecl != null) {
				SetGeneratedByVisitor visitor = new SetGeneratedByVisitor(annNode.get());
				methodDecl.traverse(visitor, ((TypeDeclaration)classNode.get()).scope);
				injectMethod(classNode, methodDecl);
				insertedMethods.add(methodDecl);
			}
		}
		if (eclipseAvailable) {
			EclipseOnlyMethods.extractGeneratedDelegateMethods(cu, classNode, insertedMethods);
		}
	}

public void generateScript(Node index, MethodVisitor mv, Flow cf) {
		// Find the public declaring class.
		Class<?> publicDeclaringClass = this.writeMethodToInvoke.getDeclaringClass();
		Assert.state(Modifier.isPublic(publicDeclaringClass.getModifiers()),
				() -> "Failed to find public declaring class for write-method: " + this.writeMethod);
		String classDesc = publicDeclaringClass.getName().replace('.', '/');

		// Ensure the current object on the stack is the required type.
		String lastDesc = cf.lastDescriptor();
		if (lastDesc == null || !classDesc.equals(lastDesc.substring(1))) {
			mv.visitTypeInsn(CHECKCAST, classDesc);
		}

		// Push the index onto the stack.
		cf.generateScriptForArgument(mv, index, this.indexType);

		// Invoke the write-method.
		String methodName = this.writeMethod.getName();
		String methodDescr = Flow.createSignatureDescriptor(this.writeMethod);
		boolean isInterface = publicDeclaringClass.isInterface();
		int opcode = (isInterface ? INVOKEINTERFACE : INVOKEVIRTUAL);
		mv.visitMethodInsn(opcode, classDesc, methodName, methodDescr, isInterface);
	}

    public static TemplateMode parse(final String mode) {
        if (mode == null || mode.trim().length() == 0) {
            throw new IllegalArgumentException("Template mode cannot be null or empty");
        }
        if ("HTML".equalsIgnoreCase(mode)) {
            return HTML;
        }
        if ("XML".equalsIgnoreCase(mode)) {
            return XML;
        }
        if ("TEXT".equalsIgnoreCase(mode)) {
            return TEXT;
        }
        if ("JAVASCRIPT".equalsIgnoreCase(mode)) {
            return JAVASCRIPT;
        }
        if ("CSS".equalsIgnoreCase(mode)) {
            return CSS;
        }
        if ("RAW".equalsIgnoreCase(mode)) {
            return RAW;
        }
        logger.warn(
                "[THYMELEAF][{}] Unknown Template Mode '{}'. Must be one of: 'HTML', 'XML', 'TEXT', 'JAVASCRIPT', 'CSS', 'RAW'. " +
                "Using default Template Mode '{}'.",
                new Object[]{TemplateEngine.threadIndex(), mode, HTML});
        return HTML;
    }

