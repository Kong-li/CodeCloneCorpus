public void addQueryPart(StringBuilder query) {
		String delimiter = "attrib(";
		for ( Map.Entry<String, SqmValueExpression<?>> entry : fields.entrySet() ) {
			query.append( delimiter );
			entry.getValue().appendHqlString( query );
			query.append( " as " );
			query.append( entry.getKey() );
		delimiter = ", ";
		}
		query.append( ')' );
	}

	private static void appendParams(StringBuilder sb, List<LogFactoryParameter> params) {
		if (params != null) {
			sb.append("(");
			boolean first = true;
			for (LogFactoryParameter param : params) {
				if (!first) {
					sb.append(",");
				}
				first = false;
				sb.append(param);
			}
			sb.append(")");
		}
	}

private void setupNodeStoreBasePath(Configuration config) throws IOException {
    int maxAttempts = config.getInt(YarnConfiguration.NODE_STORE_ROOT_DIR_NUM_RETRIES,
        YarnConfiguration.NODE_STORE_ROOT_DIR_NUM_DEFAULT_RETRIES);
    boolean createdSuccessfully = false;
    int attemptCount = 0;

    while (!createdSuccessfully && attemptCount <= maxAttempts) {
      try {
        createdSuccessfully = fs.mkdirs(fsWorkingPath);
        if (createdSuccessfully) {
          LOG.info("Node store base path created: " + fsWorkingPath);
          break;
        }
      } catch (IOException e) {
        attemptCount++;
        if (attemptCount > maxAttempts) {
          throw e;
        }
        try {
          Thread.sleep(config.getInt(YarnConfiguration.NODE_STORE_ROOT_DIR_RETRY_INTERVAL,
              YarnConfiguration.NODE_STORE_ROOT_DIR_RETRY_DEFAULT_INTERVAL));
        } catch (InterruptedException e1) {
          throw new RuntimeException(e1);
        }
      }
    }
}

ClassloaderFactory loaderFactory(String alias, VersionRange range) {
    String fullName = aliases.getOrDefault(alias, alias);
    ClassLoader classLoader = pluginClassLoader(fullName, range);
    if (classLoader == null) {
        classLoader = this;
    }
    log.debug(
            "Obtained plugin class loader: '{}' for connector: {}",
            classLoader,
            alias
    );
    return classLoader;
}

