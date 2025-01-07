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

private void mayUpdatePrevLeadOwner(ChangeLogRecord info) {
        if (!applyPrevLeadOwnerInBalancedRestore || !eligibleLeaderReplicasActive) return;
        if (info.isr() != null && info.isr().isEmpty() && (partition.prevKnownElro.length != 1 ||
            partition.prevKnownElro[0] != partition.owner)) {
            // Only update the previous known leader owner when the first time the partition becomes leaderless.
            info.setPrevKnownElro(Collections.singletonList(partition.owner));
        } else if ((info.owner() >= 0 || (partition.owner != NO_OWNER && info.owner() != NO_OWNER))
            && partition.prevKnownElro.length > 0) {
            // Clear the PrevKnownElro field if the partition will have or continues to have a valid leader.
            info.setPrevKnownElro(Collections.emptyList());
        }
    }

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

    private ElectionResult electAnyLeader() {
        if (isValidNewLeader(partition.leader)) {
            // Don't consider a new leader since the current leader meets all the constraints
            return new ElectionResult(partition.leader, false);
        }

        Optional<Integer> onlineLeader = targetReplicas.stream()
            .filter(this::isValidNewLeader)
            .findFirst();
        if (onlineLeader.isPresent()) {
            return new ElectionResult(onlineLeader.get(), false);
        }

        if (canElectLastKnownLeader()) {
            return new ElectionResult(partition.lastKnownElr[0], true);
        }

        if (election == Election.UNCLEAN) {
            // Attempt unclean leader election
            Optional<Integer> uncleanLeader = targetReplicas.stream()
                .filter(isAcceptableLeader::test)
                .findFirst();
            if (uncleanLeader.isPresent()) {
                return new ElectionResult(uncleanLeader.get(), true);
            }
        }

        return new ElectionResult(NO_LEADER, false);
    }

