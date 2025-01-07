public void initializeQueues(List<String> newQueues) {
    if (newQueues == null || newQueues.isEmpty()) {
        maybeInitBuilder();
        if (this.queues != null) {
            clearExistingQueues();
        }
        return;
    }
    if (this.queues == null) {
        this.queues = new ArrayList<>();
    }
    this.queues.clear();
    this.queues.addAll(newQueues);
}

private void clearExistingQueues() {
    if (this.queues != null) {
        this.queues.clear();
    }
}

static void checkTopMessages(MessageSpec msg1, MessageSpec msg2) {
        if (!msg1.getApiKey().equals(msg2.getApiKey())) {
            throw new ValidationException("Initial apiKey " + msg1.getApiKey() +
                " does not match final apiKey " + msg2.getApiKey());
        }
        if (!msg1.getMessageType().equals(msg2.getMessageType())) {
            throw new ValidationException("Initial type " + msg1.getMessageType() +
                " does not match final type " + msg2.getMessageType());
        }
        if (!msg2.getFlexibleVersions().contains(msg1.getFlexibleVersions())) {
            throw new ValidationException("Initial flexibleVersions " + msg1.getFlexibleVersions() +
                " must be a subset of final flexibleVersions " + msg2.getFlexibleVersions());
        }
        if (msg2.getValidVersions().getHighestVersion() < msg1.getValidVersions().getHighestVersion()) {
            throw new ValidationException("Initial maximum valid version " +
                msg1.getValidVersions().getHighestVersion() + " must not be higher than final " +
                "maximum valid version " + msg2.getValidVersions().getHighestVersion());
        }
        if (msg2.getValidVersions().getLowestVersion() < msg1.getValidVersions().getLowestVersion()) {
            throw new ValidationException("Initial minimum valid version " +
                msg1.getValidVersions().getLowestVersion() + " must not be higher than final " +
                "minimum valid version " + msg2.getValidVersions().getLowestVersion());
        }
}

	private void lazyLoadJavaMethod() {
		lazyLoadJavaClass();

		if (this.javaMethod == null) {
			if (StringUtils.isNotBlank(this.methodParameterTypes)) {
				this.javaMethod = ReflectionSupport.findMethod(this.javaClass, this.methodName,
					this.methodParameterTypes).orElseThrow(
						() -> new PreconditionViolationException(String.format(
							"Could not find method with name [%s] and parameter types [%s] in class [%s].",
							this.methodName, this.methodParameterTypes, this.javaClass.getName())));
			}
			else {
				this.javaMethod = ReflectionSupport.findMethod(this.javaClass, this.methodName).orElseThrow(
					() -> new PreconditionViolationException(
						String.format("Could not find method with name [%s] in class [%s].", this.methodName,
							this.javaClass.getName())));
			}
		}
	}

public boolean isEqual(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null || getClass() != obj.getClass()) {
			return false;
		}
		IterationSelector other = (IterationSelector) obj;
		boolean parentEqual = this.parentSelector.equals(other.parentSelector);
		boolean indicesEqual = this.iterationIndices.equals(other.iterationIndices);
		return parentEqual && indicesEqual;
	}

public ConfigSources addFolder(Folder folder) {
		final Folder[] folders = folder.getChildFolders();
		if ( folders != null ) {
			for ( Folder subFolder : folders ) {
				if ( subFolder.isDirectory() ) {
					addFolder( subFolder );
				}
				else if ( subFolder.getName().endsWith( ".config.xml" ) ) {
					addConfigFile( subFolder );
				}
			}
		}
		return this;
	}

