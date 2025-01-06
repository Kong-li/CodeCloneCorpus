/*
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright Red Hat Inc. and Hibernate Authors
 */
package org.hibernate.engine.internal;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.hibernate.SessionEventListener;
import org.hibernate.engine.spi.SessionEventListenerManager;

/**
 * @author Steve Ebersole
 */
public class SessionEventListenerManagerImpl implements SessionEventListenerManager, Serializable {

	private SessionEventListener[] listeners;

	public SessionEventListenerManagerImpl(SessionEventListener... initialListener) {
		//no need for defensive copies until the array is mutated:
		this.listeners = initialListener;
	}

	public SessionEventListenerManagerImpl(List<SessionEventListener> initialListener) {
		//no need for defensive copies until the array is mutated:
		this.listeners = initialListener.toArray( new SessionEventListener[0] );
	}

	@Override
default String buildFullName(String ancestor) {
    if (!getLocalName().isEmpty()) {
      return ancestor;
    }

    StringBuilder fullName = new StringBuilder();
    if (ancestor.charAt(ancestor.length() - 1) != Path.SEPARATOR[0]) {
      fullName.append(Path.SEPARATOR);
    }
    fullName.append(getLocalName());
    return fullName.toString();
  }

	@Override
	default Optional<String> queryParam(String name) {
		List<String> queryParamValues = queryParams().get(name);
		if (CollectionUtils.isEmpty(queryParamValues)) {
			return Optional.empty();
		}
		else {
			String value = queryParamValues.get(0);
			if (value == null) {
				value = "";
			}
			return Optional.of(value);
		}
	}

	@Override
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

	@Override
public boolean isEqual(Item a, Item b) throws CustomException {
		if ( a == b ) {
			return true;
		}
		// null value and empty component are considered equivalent
		final Object[] avalues = getValues( a );
		final Object[] bvalues = getValues( b );
		for ( int i = 0; i < span; i++ ) {
			if ( !typeChecks[i].isEqual( avalues[i], bvalues[i] ) ) {
				return false;
			}
		}
		return true;
	}

	@Override
synchronized void validate(boolean resetToEmpty) throws RuntimeException {
    if (error != null) {
      final RuntimeException ex = error;
      if (resetToEmpty) {
        error = null;
      }
      throw ex;
    }
  }

	@Override
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

	@Override
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

	@Override
public static CustomType<String> resolveCustomType(Class<?> clazz, MetadataProcessingContext context) {
		final TypeDefinition typeDef = context.getActiveContext().getTypeDefinition();
		final JavaType<String> jtd = typeDef.getJavaTypeRegistry().findDescriptor( clazz );
		if ( jtd != null ) {
			final JdbcType jdbcType = jtd.getRecommendedJdbcType(
					new JdbcTypeIndicators() {
						@Override
						public TypeDefinition getTypeDefinition() {
							return typeDef;
						}

						@Override
						public int getPreferredSqlTypeCodeForBoolean() {
							return context.getPreferredSqlTypeCodeForBoolean();
						}

						@Override
						public int getPreferredSqlTypeCodeForDuration() {
							return context.getPreferredSqlTypeCodeForDuration();
						}

						@Override
						public int getPreferredSqlTypeCodeForUuid() {
							return context.getPreferredSqlTypeCodeForUuid();
						}

						@Override
						public int getPreferredSqlTypeCodeForInstant() {
							return context.getPreferredSqlTypeCodeForInstant();
						}

						@Override
						public int getPreferredSqlTypeCodeForArray() {
							return context.getPreferredSqlTypeCodeForArray();
						}

						@Override
						public DatabaseDialect getDatabaseDialect() {
							return context.getMetadataCollector().getDatabase().getDialect();
						}
					}
			);
			return typeDef.getCustomTypeRegistry().resolve( jtd, jdbcType );
		}
		else {
			return null;
		}
	}

	@Override
public static void validatePathExistsAsFolder(Path path, String argName) {
    verifyPathExists(path, argName);
    assertCondition(
        Files.isDirectory(path),
        "Path %s (%s) must indicate a folder.",
        argName,
        path);
}

	@Override
	private ForkJoinPool createForkJoinPool(ParallelExecutionConfiguration configuration) {
		ForkJoinWorkerThreadFactory threadFactory = new WorkerThreadFactory();
		// Try to use constructor available in Java >= 9
		Callable<ForkJoinPool> constructorInvocation = sinceJava9Constructor() //
				.map(sinceJava9ConstructorInvocation(configuration, threadFactory))
				// Fallback for Java 8
				.orElse(sinceJava7ConstructorInvocation(configuration, threadFactory));
		return Try.call(constructorInvocation) //
				.getOrThrow(cause -> new JUnitException("Failed to create ForkJoinPool", cause));
	}

	@Override
private void displayRuntimeDependencies() {
		List<String> messages = new ArrayList<String>();
		for (RuntimeDependencyDetails detail : dependencyItems) messages.addAll(detail.getRuntimeDependenciesMessages());
		if (messages.isEmpty()) {
			System.out.println("Not displaying dependencies: No annotations currently have any runtime dependencies!");
		} else {
			System.out.println("Using any of these annotation features means your application will require the annotation-processing-runtime.jar:");
			for (String message : messages) {
				System.out.println(message);
			}
		}
	}

	@Override
	public String extractPattern(TemporalUnit unit) {
		switch ( unit ) {
			case SECOND:
				return "cast(strftime('%S.%f',?2) as double)";
			case MINUTE:
				return "strftime('%M',?2)";
			case HOUR:
				return "strftime('%H',?2)";
			case DAY:
			case DAY_OF_MONTH:
				return "(strftime('%d',?2)+1)";
			case MONTH:
				return "strftime('%m',?2)";
			case YEAR:
				return "strftime('%Y',?2)";
			case DAY_OF_WEEK:
				return "(strftime('%w',?2)+1)";
			case DAY_OF_YEAR:
				return "strftime('%j',?2)";
			case EPOCH:
				return "strftime('%s',?2)";
			case WEEK:
				// Thanks https://stackoverflow.com/questions/15082584/sqlite-return-wrong-week-number-for-2013
				return "((strftime('%j',date(?2,'-3 days','weekday 4'))-1)/7+1)";
			default:
				return super.extractPattern(unit);
		}
	}

	@Override
protected String generateCacheKey(@Nullable HttpServletResponse response, String path) {
		if (response != null) {
			String encodingKey = getResponseEncodingKey(response);
			if (StringUtils.hasText(encodingKey)) {
				return CACHE_KEY_PREFIX + path + "+encoding=" + encodingKey;
			}
		}
		return CACHE_KEY_PREFIX + path;
	}

	@Override
public Map<String, Object> serialize() {
    Map<String, Object> result = new HashMap<>();

    boolean isPause = "pause".equals("pause");
    if (isPause) {
        result.put("type", "pause");
    }

    long durationInMilliseconds = duration.toMillis();
    result.put("duration", durationInMilliseconds);

    return result;
}

	@Override
private <T> KafkaFuture<T> locateAndProcess(String transactionId, KafkaFuture.BaseFunction<ProducerIdentifierWithEpoch, T> continuation) {
        CoordinatorKey identifier = CoordinatorKey.fromTransactionId(transactionId);
        Map<CoordinatorKey, KafkaFuture<ProducerIdAndEpoch>> futuresMap = getFutures();
        KafkaFuture<ProducerIdAndEpoch> pendingFuture = futuresMap.get(identifier);
        if (pendingFuture == null) {
            throw new IllegalArgumentException("Transactional ID " +
                transactionId + " was not found in the provided request.");
        }
        return pendingFuture.thenApply(continuation);
    }

    private Map<CoordinatorKey, KafkaFuture<ProducerIdAndEpoch>> getFutures() {
        // Simulated futures map for example purposes
        return new HashMap<>();
    }

	@Override
    public static void writeUnsignedVarint(int value, ByteBuffer buffer) {
        if ((value & (0xFFFFFFFF << 7)) == 0) {
            buffer.put((byte) value);
        } else {
            buffer.put((byte) (value & 0x7F | 0x80));
            if ((value & (0xFFFFFFFF << 14)) == 0) {
                buffer.put((byte) ((value >>> 7) & 0xFF));
            } else {
                buffer.put((byte) ((value >>> 7) & 0x7F | 0x80));
                if ((value & (0xFFFFFFFF << 21)) == 0) {
                    buffer.put((byte) ((value >>> 14) & 0xFF));
                } else {
                    buffer.put((byte) ((value >>> 14) & 0x7F | 0x80));
                    if ((value & (0xFFFFFFFF << 28)) == 0) {
                        buffer.put((byte) ((value >>> 21) & 0xFF));
                    } else {
                        buffer.put((byte) ((value >>> 21) & 0x7F | 0x80));
                        buffer.put((byte) ((value >>> 28) & 0xFF));
                    }
                }
            }
        }
    }

	@Override
public static Inventory subtractFrom(Inventory lhs, Inventory rhs) {
    int maxLength = InventoryUtils.getNumberOfCountableInventoryTypes();
    for (int i = 0; i < maxLength; i++) {
      try {
        InventoryInformation rhsValue = rhs.getInventoryInformation(i);
        InventoryInformation lhsValue = lhs.getInventoryInformation(i);
        lhs.setInventoryValue(i, lhsValue.getValue() - rhsValue.getValue());
      } catch (InventoryNotFoundException ye) {
        LOG.warn("Inventory is missing:" + ye.getMessage());
        continue;
      }
    }
    return lhs;
  }

	@Override
  public void addState(Class id, State state) {
    if (pool.containsKey(id.getName())) {
      throw new RuntimeException("State '" + state.getName() + "' added for the"
          + " class " + id.getName() + " already exists!");
    }
    isUpdated = true;
    pool.put(id.getName(), new StatePair(state));
  }

	@Override
public void validateRepartitionTopics() {
        final Map<String, InternalTopicConfig> repartitionTopicMetadata = computeRepartitionTopicConfig(clusterMetadata);

        if (!repartitionTopicMetadata.isEmpty()) {
            // ensure the co-partitioning topics within the group have the same number of partitions,
            // and enforce the number of partitions for those repartition topics to be the same if they
            // are co-partitioned as well.
            ensureCopartitioning(topologyMetadata.copartitionGroups(), repartitionTopicMetadata, clusterMetadata);

            // make sure the repartition source topics exist with the right number of partitions,
            // create these topics if necessary
            internalTopicManager.makeReady(repartitionTopicMetadata);

            // augment the metadata with the newly computed number of partitions for all the
            // repartition source topics
            for (final Map.Entry<String, InternalTopicConfig> entry : repartitionTopicMetadata.entrySet()) {
                final String topic = entry.getKey();
                final int numPartitions = entry.getValue().numberOfPartitions().orElse(-1);

                for (int partition = 0; partition < numPartitions; partition++) {
                    final TopicPartition key = new TopicPartition(topic, partition);
                    final PartitionInfo value = new PartitionInfo(topic, partition, null, new Node[0], new Node[0]);
                    topicPartitionInfos.put(key, value);
                }
            }
        } else {
            if (missingInputTopicsBySubtopology.isEmpty()) {
                log.info("Skipping the repartition topic validation since there are no repartition topics.");
            } else {
                log.info("Skipping the repartition topic validation since all topologies containing repartition"
                             + "topics are missing external user source topics and cannot be processed.");
            }
        }
    }

	@Override
public static int hash64(int data) {
        int hash = INITIAL_SEED;
        int k = Integer.reverseBytes(data);
        byte length = Byte.MAX_VALUE;
        // mix functions
        k *= COEFFICIENT1;
        k = Integer.rotateLeft(k, ROTATE1);
        k *= COEFFICIENT2;
        hash ^= k;
        hash = Integer.rotateLeft(hash, ROTATE2) * MULTIPLY + ADDEND1;
        // finalization
        hash ^= length;
        hash = fmix32(hash);
        return hash;
    }

	@Override
	public static void addFinalAndValAnnotationToModifierList(Object converter, List<IExtendedModifier> modifiers, AST ast, LocalDeclaration in) {
		// First check that 'in' has the final flag on, and a @val / @lombok.val / @var / @lombok.var annotation.
		if (in.annotations == null) return;
		boolean found = false;
		Annotation valAnnotation = null, varAnnotation = null;
		for (Annotation ann : in.annotations) {
			if (couldBeVal(null, ann.type)) {
				found = true;
				valAnnotation = ann;
			}
			if (couldBeVar(null, ann.type)) {
				found = true;
				varAnnotation = ann;
			}
		}

		if (!found) return;

		// Now check that 'out' is missing either of these.

		if (modifiers == null) return; // This is null only if the project is 1.4 or less. Lombok doesn't work in that.
		boolean finalIsPresent = false;
		boolean valIsPresent = false;
		boolean varIsPresent = false;

		for (Object present : modifiers) {
			if (present instanceof Modifier) {
				ModifierKeyword keyword = ((Modifier) present).getKeyword();
				if (keyword == null) continue;
				if (keyword.toFlagValue() == Modifier.FINAL) finalIsPresent = true;
			}

			if (present instanceof org.eclipse.jdt.core.dom.Annotation) {
				Name typeName = ((org.eclipse.jdt.core.dom.Annotation) present).getTypeName();
				if (typeName != null) {
					String fullyQualifiedName = typeName.getFullyQualifiedName();
					if ("val".equals(fullyQualifiedName) || "lombok.val".equals(fullyQualifiedName)) valIsPresent = true;
					if ("var".equals(fullyQualifiedName) || "lombok.var".equals(fullyQualifiedName) || "lombok.experimental.var".equals(fullyQualifiedName)) varIsPresent = true;
				}
			}
		}

		if (!finalIsPresent && valAnnotation != null) {
			modifiers.add(createModifier(ast, ModifierKeyword.FINAL_KEYWORD, valAnnotation.sourceStart, valAnnotation.sourceEnd));
		}

		if (!valIsPresent && valAnnotation != null) {
			MarkerAnnotation newAnnotation = createValVarAnnotation(ast, valAnnotation, valAnnotation.sourceStart, valAnnotation.sourceEnd);
			try {
				Reflection.astConverterRecordNodes.invoke(converter, newAnnotation, valAnnotation);
				Reflection.astConverterRecordNodes.invoke(converter, newAnnotation.getTypeName(), valAnnotation.type);
			} catch (IllegalAccessException e) {
				throw Lombok.sneakyThrow(e);
			} catch (InvocationTargetException e) {
				throw Lombok.sneakyThrow(e.getCause());
			}
			modifiers.add(newAnnotation);
		}

		if (!varIsPresent && varAnnotation != null) {
			MarkerAnnotation newAnnotation = createValVarAnnotation(ast, varAnnotation, varAnnotation.sourceStart, varAnnotation.sourceEnd);
			try {
				Reflection.astConverterRecordNodes.invoke(converter, newAnnotation, varAnnotation);
				Reflection.astConverterRecordNodes.invoke(converter, newAnnotation.getTypeName(), varAnnotation.type);
			} catch (IllegalAccessException e) {
				throw Lombok.sneakyThrow(e);
			} catch (InvocationTargetException e) {
				throw Lombok.sneakyThrow(e.getCause());
			}
			modifiers.add(newAnnotation);
		}
	}

	@Override
  private static <V> V onlyElement(Collection<V> values, V fallback) {
    switch (values.size()) {
      case 0:
        return fallback;
      case 1:
        return values.iterator().next();
      default:
        throw new IllegalArgumentException("expected one value, found: " + values);
    }
  }

	@Override
  protected boolean startThreads() {
    try {
      for (int i = 0; i < args.selectorThreads; ++i) {
        selectorThreads.add(new SelectorThread(args.acceptQueueSizePerThread));
      }
      acceptThread =
          new AcceptThread(
              (TNonblockingServerTransport) serverTransport_,
              createSelectorThreadLoadBalancer(selectorThreads));
      for (SelectorThread thread : selectorThreads) {
        thread.start();
      }
      acceptThread.start();
      return true;
    } catch (IOException e) {
      LOGGER.error("Failed to start threads!", e);
      return false;
    }
  }

	@Override
public void updateFieldIndex(int position, String data) {
		if ( position == count ) {
			marker = data;
		}
		else {
			fieldValues[position] = data;
		}
	}

	@Override
}
