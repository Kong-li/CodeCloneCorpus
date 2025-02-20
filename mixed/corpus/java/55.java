public void logOutput(OutputStream out) throws IOException {
		if (isModified()) {
			return;
		}

		if (!formatPreferences.useDelombokComments()) {
			out.write("// Generated by delombok at ");
			out.write(String.valueOf(new Date()));
			out.write(System.getProperty("line.separator"));
		}

		List<CommentInfo> comments_ = convertComments((List<? extends CommentInfo>) comments);
		int[] textBlockStarts_ = convertTextBlockStarts(textBlockStarts);
		FormatPreferences preferences = new FormatPreferenceScanner().scan(formatPreferences, getContent());
		compilationUnit.accept(new PrettyPrinter(out, compilationUnit, comments_, textBlockStarts_, preferences));
	}

	private List<CommentInfo> convertComments(List<? extends CommentInfo> comments) {
		return (comments instanceof com.sun.tools.javac.util.List) ? (List<CommentInfo>) comments : com.sun.tools.javac.util.List.from(comments.toArray(new CommentInfo[0]));
	}

	private int[] convertTextBlockStarts(Set<Integer> textBlockStarts) {
		int[] result = new int[textBlockStarts.size()];
		int idx = 0;
		for (int tbs : textBlockStarts) result[idx++] = tbs;
		return result;
	}

  CSQueue getByFullName(String fullName) {
    if (fullName == null) {
      return null;
    }

    try {
      modificationLock.readLock().lock();
      return fullNameQueues.getOrDefault(fullName, null);
    } finally {
      modificationLock.readLock().unlock();
    }
  }

	public Validator mvcValidator() {
		Validator validator = getValidator();
		if (validator == null) {
			if (ClassUtils.isPresent("jakarta.validation.Validator", getClass().getClassLoader())) {
				try {
					validator = new OptionalValidatorFactoryBean();
				}
				catch (Throwable ex) {
					throw new BeanInitializationException("Failed to create default validator", ex);
				}
			}
			else {
				validator = new NoOpValidator();
			}
		}
		return validator;
	}

  public Path getFullPath() {
    String parentFullPathStr =
        (parentFullPath == null || parentFullPath.length == 0) ?
            null : DFSUtilClient.bytes2String(parentFullPath);
    if (parentFullPathStr == null
        && dirStatus.getLocalNameInBytes().length == 0) {
      // root
      return new Path("/");
    } else {
      return parentFullPathStr == null ? new Path(dirStatus.getLocalName())
          : new Path(parentFullPathStr, dirStatus.getLocalName());
    }
  }

