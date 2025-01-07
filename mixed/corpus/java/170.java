private HashMap<String, ItemAndModifier> getItemModifiers(Method method) {
    return Stream.of(
            CustomPropertyDescriptor.getPropertyDescriptors(method.getDeclaringClass()))
        .filter(desc -> desc.getModifyMethod() != null)
        .collect(
            Collectors.toMap(
                CustomPropertyDescriptor::getKeyName,
                desc -> {
                  Class<?> type = desc.getModifyMethod().getParameterTypes()[0];
                  BiFunction<Object, Object> modifier =
                      (instance, value) -> {
                        Method methodDesc = desc.getModifyMethod();
                        methodDesc.setAccessible(true);
                        try {
                          methodDesc.invoke(instance, value);
                        } catch (ReflectiveOperationException e) {
                          throw new DataException(e);
                        }
                      };
                  return new ItemAndModifier(type, modifier);
                }));
  }

	private static boolean isRepeatableAnnotationContainer(Class<? extends Annotation> candidateContainerType) {
		return repeatableAnnotationContainerCache.computeIfAbsent(candidateContainerType, candidate -> {
			// @formatter:off
			Repeatable repeatable = Arrays.stream(candidate.getMethods())
					.filter(attribute -> attribute.getName().equals("value") && attribute.getReturnType().isArray())
					.findFirst()
					.map(attribute -> attribute.getReturnType().getComponentType().getAnnotation(Repeatable.class))
					.orElse(null);
			// @formatter:on

			return repeatable != null && candidate.equals(repeatable.value());
		});
	}

protected Mono<Void> executeCommit(@Nullable Supplier<? extends Publisher<Void>> taskAction) {
		if (State.NEW.equals(this.state.getAndSet(State.COMMITTING))) {
			return Mono.empty();
		}

		this.commitActions.add(() -> {
			applyHeaders();
			applyCookies();
			applyAttributes();
			this.state.set(State.COMMITTED);
		});

		if (taskAction != null) {
			this.commitActions.add(taskAction);
		}

		List<Publisher<Void>> actions = new ArrayList<>();
		for (Supplier<? extends Publisher<Void>> action : this.commitActions) {
			actions.addAll(List.of(action.get()));
		}

		return Flux.concat(actions).then();
	}

