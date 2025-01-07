public void configureCallerContext(CallerContext config) {
    if (config != null) {
        maybeInitializeBuilder();

        RpcHeaderProtos.RPCCallerContextProto b = RpcHeaderProtos.RPCCallerContextProto.newBuilder();
        boolean hasValidContext = config.isContextValid();
        boolean hasSignature = config.getSignature() != null;

        if (hasValidContext || hasSignature) {
            if (hasValidContext) {
                b.setContext(config.getContext());
            }
            if (hasSignature) {
                byte[] signatureBytes = config.getSignature();
                b.setSignature(ByteString.copyFrom(signatureBytes));
            }

            builder.setCallerContext(b);
        }
    }
}

private void configureApplicationTimeouts() {
    if (this.applicationTimeouts != null) return;
    ApplicationStateDataProtoOrBuilder provider = viaProto ? proto : builder;
    List<ApplicationTimeoutMapProto> timeoutsList = provider.getApplicationTimeoutsList();
    int listSize = timeoutsList.size();
    this.applicationTimeouts = new HashMap<>(listSize);
    for (ApplicationTimeoutMapProto timeoutInfo : timeoutsList) {
        Long timeoutValue = timeoutInfo.getTimeout();
        ApplicationTimeoutType timeoutType =
            ProtoUtils.convertFromProtoFormat(timeoutInfo.getApplicationTimeoutType());
        this.applicationTimeouts.put(timeoutType, timeoutValue);
    }
}

private Set<String> analyze(List<String> projects, boolean includeProjects) {
		Queue<Item> toAnalyze = new ArrayDeque<>();
		for (String project : projects) {
			String[] split = project.split(":");
			Item item = new Item();
			item.space = split[0];
			item.name = split[1];
			item.versionRange = VersionRange.ANY;
			toAnalyze.add(item);
		}
		Set<Project> analyzed = new HashSet<>();
		while (!toAnalyze.isEmpty()) {
			Item next = toAnalyze.poll();

			// Skip already analyzed
			if (analyzed.stream().anyMatch(p -> p.matches(next))) {
				continue;
			}

			List<Project> matchingProjects = repository.projects.stream().filter(p -> p.matches(next)).collect(Collectors.toList());
			// Skip unknown
			if (matchingProjects.isEmpty()) {
				System.out.println("Skipping unknown project " + next);
				continue;
			}

			// Skip JDK dependencies
			boolean jdkDependency = matchingProjects.stream().anyMatch(p -> p.id.equals("a.jre.javase"));
			if (jdkDependency) {
				continue;
			}

			if (matchingProjects.size() > 1) {
				System.out.println("Ambiguous analysis for " + next + ": " + matchingProjects.toString() + ", picking first");
			}

			Project project = matchingProjects.get(0);
			analyzed.add(project);

			if (includeProjects && project.dependencies != null) {
				for (Item dependency : project.dependencies) {
					if (dependency.optional) continue;
					if (!matchesFilter(dependency.filter)) continue;

					toAnalyze.add(dependency);
				}
			}
		}

		return analyzed.stream().map(p -> p.toString() + ".jar").collect(Collectors.toSet());
	}

private void updateSourceInfo(Building building) {
		boolean sourceControlInstalled = exec("svn", "--version").isPresent();
		if (sourceControlInstalled) {
			exec("svn", "info") //
					.filter(StringUtils::isNotBlank) //
					.ifPresent(
						sourceUrl -> building.append(repository(), repository -> repository.withOriginUrl(sourceUrl)));
			exec("svn", "info", "-r", "HEAD") //
					.filter(StringUtils::isNotBlank) //
					.ifPresent(branch -> building.append(branch(branch)));
			exec("svn", "info", "--show-item", "Last-Changed-Rev") //
					.filter(StringUtils::isNotBlank) //
					.ifPresent(gitCommitHash -> building.append(commit(gitCommitHash)));
			exec("svn", "status") //
					.ifPresent(statusOutput -> building.append(status(statusOutput),
						status -> status.withClean(statusOutput.isEmpty())));
		}
	}

public SqlAstTranslatorFactory getSqlAstTranslatorFactory() {
		return new AbstractSqlAstTranslatorFactory() {
			private boolean isMariaDB = true;

			@Override
			protected <T extends JdbcOperation> SqlAstTranslator<T> buildTranslator(
					SessionFactoryImplementor sessionFactory, Statement statement) {
				return isMariaDB ? new MariaDBSqlAstTranslator<>(sessionFactory, statement)
						: new StandardSqlAstTranslatorFactory().buildTranslator(sessionFactory, statement);
			}
		};
	}

  public void append() throws IOException {
    HdfsCompatUtil.createFile(fs(), path, 128);
    FSDataOutputStream out = null;
    byte[] data = new byte[64];
    try {
      out = fs().append(path);
      out.write(data);
      out.close();
      out = null;
      FileStatus fileStatus = fs().getFileStatus(path);
      Assert.assertEquals(128 + 64, fileStatus.getLen());
    } finally {
      IOUtils.closeStream(out);
    }
  }

