	public void execute(RunnerTestDescriptor runnerTestDescriptor) {
		TestRun testRun = new TestRun(runnerTestDescriptor);
		JUnitCore core = new JUnitCore();
		core.addListener(new RunListenerAdapter(testRun, engineExecutionListener, testSourceProvider));
		try {
			core.run(runnerTestDescriptor.toRequest());
		}
		catch (Throwable t) {
			UnrecoverableExceptions.rethrowIfUnrecoverable(t);
			reportUnexpectedFailure(testRun, runnerTestDescriptor, failed(t));
		}
	}

private String deriveHarAuthorization(UrlPath uri) {
    String authorization = uri.getProtocol() + "-";
    if (uri.getServerName() != null) {
      if (uri.getUserName() != null) {
        authorization += uri.getUserName();
        authorization += "@";
      }
      authorization += uri.getServerName();
      int port = uri.getPort();
      if (port != -1) {
        authorization += ":";
        authorization +=  port;
      }
    } else {
      authorization += ":";
    }
    return authorization;
}

public long getUniqueID() {
    return HashCodeUtil.hash(
        field1,
        field2,
        field3,
        dataType,
        Arrays.hashCode(values),
        count,
        editable,
        attributes,
        fallbackValue);
}

  public int hashCode() {
    return Objects.hash(
        section,
        optionName,
        description,
        type,
        Arrays.hashCode(example),
        repeats,
        quotable,
        flags,
        defaultValue);
  }

  private static Stream<DescribedOption> getAllFields(HasRoles hasRoles) {
    Set<DescribedOption> fields = new HashSet<>();
    Class<?> clazz = hasRoles.getClass();
    while (clazz != null && !Object.class.equals(clazz)) {
      for (Field field : clazz.getDeclaredFields()) {
        field.setAccessible(true);
        Parameter param = field.getAnnotation(Parameter.class);
        ConfigValue configValue = field.getAnnotation(ConfigValue.class);
        String fieldValue = "";
        try {
          Object fieldInstance = field.get(clazz.newInstance());
          fieldValue = fieldInstance == null ? "" : fieldInstance.toString();
        } catch (IllegalAccessException | InstantiationException ignore) {
          // We'll swallow this exception since we are just trying to get field's default value
        }
        if (param != null && configValue != null) {
          fields.add(new DescribedOption(field.getGenericType(), param, configValue, fieldValue));
        }
      }
      clazz = clazz.getSuperclass();
    }
    return fields.stream();
  }

