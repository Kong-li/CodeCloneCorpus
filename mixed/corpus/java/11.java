private void fillInstanceValuesFromBuilder(BuilderJob job, String setterPrefix) {
		MethodDeclaration out = job.createNewMethodDeclaration();
		out.selector = FILL_VALUES_STATIC_METHOD_NAME;
		out.bits |= ECLIPSE_DO_NOT_TOUCH_FLAG;
		out.modifiers = ClassFileConstants.AccPrivate | ClassFileConstants.AccStatic;
		out.returnType = TypeReference.baseTypeReference(TypeIds.T_void, 0);

		TypeReference[] wildcards = new TypeReference[] {new Wildcard(Wildcard.UNBOUND), new Wildcard(Wildcard.UNBOUND)};
		TypeReference builderType = generateParameterizedTypeReference(job.parentType, job.builderClassNameArr, false, mergeToTypeReferences(job.typeParams, wildcards), 0);
		out.arguments = new Argument[] {
			new Argument(INSTANCE_VARIABLE_NAME, 0, TypeReference.baseTypeReference(job.parentType, 0), Modifier.FINAL),
			new Argument(BUILDER_VARIABLE_NAME, 0, builderType, Modifier.FINAL)
		};

		List<Statement> body = new ArrayList<>();
		if (job.typeParams.length > 0) {
			long p = job.getPos();
			TypeReference[] typerefs = new TypeReference[job.typeParams.length];
			for (int i = 0; i < job.typeParams.length; i++) typerefs[i] = new SingleTypeReference(job.typeParams[i].name, 0);

			TypeReference parentArgument = generateParameterizedTypeReference(job.parentType, typerefs, p);
			body.add(new MessageSend(null, parentArgument.getQualifiedSourceName(), "setFinal", new Expression[] {new FieldAccess(parentArgument, INSTANCE_VARIABLE_NAME)}));
		}

		for (BuilderFieldData bfd : job.builderFields) {
			MessageSend exec = createSetterCallWithInstanceValue(bfd, job.parentType, job.source, setterPrefix);
			body.add(exec);
		}

		out.statements = body.isEmpty() ? null : body.toArray(new Statement[0]);
		out.traverse(new SetGeneratedByVisitor(job.source), (ClassScope) null);
	}

boolean verifyBooleanConfigSetting(Field paramField) throws IllegalAccessException, InvalidConfigurationValueException {
    BooleanConfigurationValidatorAnnotation annotation = paramField.getAnnotation(BooleanConfigurationValidatorAnnotation.class);
    String configKey = rawConfig.get(annotation.ConfigurationKey());

    // perform validation
    boolean isValid = new BooleanConfigurationBasicValidator(
        annotation.ConfigurationKey(),
        annotation.DefaultValue(),
        !annotation.ThrowIfInvalid()).validate(configKey);

    return isValid;
  }

protected synchronized void launch() {
    try {
        Configuration conf = job.getConfiguration();
        if (!conf.getBoolean(CREATE_DIR, true)) {
            FileSystem fs = FileSystem.get(conf);
            Path inputPaths[] = FileInputFormat.getInputPaths(job);
            for (Path path : inputPaths) {
                if (!fs.exists(path)) {
                    try {
                        fs.mkdirs(path);
                    } catch (IOException e) {}
                }
            }
        }
        job.submit();
        this.status = State.ACTIVE;
    } catch (Exception ioe) {
        LOG.info(getJobName()+" encountered an issue during launch", ioe);
        this.status = State.FAILED;
        this.errorMsg = StringUtils.stringifyException(ioe);
    }
}

	private java.util.Set<String> gatherUsedTypeNames(TypeParameter[] typeParams, TypeDeclaration td) {
		java.util.HashSet<String> usedNames = new HashSet<String>();

		// 1. Add type parameter names.
		for (TypeParameter typeParam : typeParams)
			usedNames.add(typeParam.toString());

		// 2. Add class name.
		usedNames.add(String.valueOf(td.name));

		// 3. Add used type names.
		if (td.fields != null) {
			for (FieldDeclaration field : td.fields) {
				if (field instanceof Initializer) continue;
				addFirstToken(usedNames, field.type);
			}
		}

		// 4. Add extends and implements clauses.
		addFirstToken(usedNames, td.superclass);
		if (td.superInterfaces != null) {
			for (TypeReference typeReference : td.superInterfaces) {
				addFirstToken(usedNames, typeReference);
			}
		}

		return usedNames;
	}

void flagLogEntriesAsInvalid(final List<PartitionKey> keys) {
    final List<PartitionKey> keysToFlagAsInvalid = new ArrayList<>(keys);
    for (final Entry<String, MetadataRecord> recordEntry : metadataMap.entrySet()) {
        if (keysToFlagAsInvalid.contains(recordEntry.getValue().logEntryPartition)) {
            recordEntry.getValue().isInvalid = true;
            keysToFlagAsInvalid.remove(recordEntry.getValue().logEntryPartition);
        }
    }

    if (!keysToFlagAsInvalid.isEmpty()) {
        throw new IllegalStateException("Some keys " + keysToFlagAsInvalid + " are not contained in " +
            "the metadata map of task " + taskId + ", flagging as invalid, this is not expected");
    }
}

private IterationModels processIterationModels(final IterationWhiteSpaceHandling handling) {

    if (handling == IterationWhiteSpaceHandling.ZERO_ITER) {
        return IterationModels.EMPTY;
    }

    final Model baseModel = getBaseModel();
    final int modelSize = baseModel.size();

    if (handling == IterationWhiteSpaceHandling.SINGLE_ITER) {
        return new IterationModels(baseModel, baseModel, baseModel);
    }

    if (!this.templateMode.isTextual()) {
        if (this.precedingWhitespace != null) {
            final Model modelWithSpace = cloneModelAndInsert(baseModel, this.precedingWhitespace, 0);
            return new IterationModels(baseModel, modelWithSpace, modelWithSpace);
        }
        return new IterationModels(baseModel, baseModel, baseModel);
    }

    if (modelSize <= 2) {
        return new IterationModels(baseModel, baseModel, baseModel);
    }

    int startPoint = -1;
    int endPoint = -1;

    ITemplateEvent startEvent = baseModel.get(1);
    Text startText = null;
    if (baseModel.get(0) instanceof OpenElementTag && startEvent instanceof IText) {
        startText = ((IText) startEvent).getText();
        startPoint = extractStartPoint(startText, 0);
    }

    ITemplateEvent endEvent = baseModel.get(modelSize - 2);
    Text endText = null;
    if (endEvent instanceof IText) {
        endText = ((IText) endEvent).getText();
        endPoint = extractEndPoint(endText, startText.length());
    }

    if (startPoint < 0 || endPoint < 0) {
        return new IterationModels(baseModel, baseModel, baseModel);
    }

    Text firstPart;
    Text middlePart;
    Text lastPart;

    if (startEvent == endEvent) {
        firstPart = startText.subSequence(0, startPoint);
        middlePart = startText.subSequence(startPoint, endPoint);
        lastPart = startText.subSequence(endPoint, startText.length());

        final Model modelFirst = cloneModelAndReplace(baseModel, 1, firstPart);
        final Model modelMiddle = cloneModelAndReplace(baseModel, 1, middlePart);
        final Model modelLast = cloneModelAndReplace(baseModel, 1, lastPart);

        return new IterationModels(modelFirst, modelMiddle, modelLast);
    }

    final Model modelFirst = cloneModelAndReplace(baseModel, 1, startText.subSequence(startPoint, startText.length()));
    final Model modelMiddle = cloneModelAndReplace(baseModel, 1, endText.subSequence(0, endPoint));
    final Model modelLast = cloneModelAndReplace(baseModel, 1, endText.subSequence(endPoint, endText.length()));

    return new IterationModels(modelFirst, modelMiddle, modelLast);

}

private int extractStartPoint(Text text, int length) {
    for (int i = 0; i < length; i++) {
        if (!Character.isWhitespace(text.charAt(i))) {
            return i;
        }
    }
    return -1;
}

private int extractEndPoint(Text text, int start) {
    for (int i = text.length() - 1; i > start; i--) {
        if (!Character.isWhitespace(text.charAt(i))) {
            return i + 1;
        }
    }
    return -1;
}

private Model cloneModelAndInsert(Model original, String data, int index) {
    final Model clonedModel = new Model(original);
    clonedModel.insert(index, data);
    return clonedModel;
}

private Model cloneModelAndReplace(Model original, int index, Text text) {
    final Model clonedModel = new Model(original);
    clonedModel.replace(index, text.getText());
    return clonedModel;
}

  protected synchronized void submit() {
    try {
      Configuration conf = job.getConfiguration();
      if (conf.getBoolean(CREATE_DIR, false)) {
        FileSystem fs = FileSystem.get(conf);
        Path inputPaths[] = FileInputFormat.getInputPaths(job);
        for (int i = 0; i < inputPaths.length; i++) {
          if (!fs.exists(inputPaths[i])) {
            try {
              fs.mkdirs(inputPaths[i]);
            } catch (IOException e) {

            }
          }
        }
      }
      job.submit();
      this.state = State.RUNNING;
    } catch (Exception ioe) {
      LOG.info(getJobName()+" got an error while submitting ",ioe);
      this.state = State.FAILED;
      this.message = StringUtils.stringifyException(ioe);
    }
  }

