public DomainResult<T> generateDomainResult(String resultVar, DomainResultCreationContext ctx) {
		final SqlAstCreationState state = ctx.getSqlAstCreationState();
		final SqlExpressionResolver resolver = state.getSqlExpressionResolver();

		SqlSelection selection = resolver.resolveSqlSelection(
				this,
				jdbcMapping.getJdbcJavaType(),
				null,
				state.getCreationContext().getMappingMetamodel().getTypeConfiguration()
		);

		return new BasicResult<>(selection.getValuesArrayPosition(), resultVar, jdbcMapping);
	}

  public static boolean isHealthy(URI uri) {
    //check scheme
    final String scheme = uri.getScheme();
    if (!HdfsConstants.HDFS_URI_SCHEME.equalsIgnoreCase(scheme)) {
      throw new IllegalArgumentException("The scheme is not "
          + HdfsConstants.HDFS_URI_SCHEME + ", uri=" + uri);
    }

    final Configuration conf = new Configuration();
    //disable FileSystem cache
    conf.setBoolean(String.format("fs.%s.impl.disable.cache", scheme), true);
    //disable client retry for rpc connection and rpc calls
    conf.setBoolean(HdfsClientConfigKeys.Retry.POLICY_ENABLED_KEY, false);
    conf.setInt(
        CommonConfigurationKeysPublic.IPC_CLIENT_CONNECT_MAX_RETRIES_KEY, 0);

    try (DistributedFileSystem fs =
             (DistributedFileSystem) FileSystem.get(uri, conf)) {
      final boolean safemode = fs.setSafeMode(SafeModeAction.SAFEMODE_GET);
      if (LOG.isDebugEnabled()) {
        LOG.debug("Is namenode in safemode? {}; uri={}", safemode, uri);
      }
      return !safemode;
    } catch (IOException e) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Got an exception for uri={}", uri, e);
      }
      return false;
    }
  }

public XmlMappingDiscriminatorType create() {
		final XmlMappingDiscriminatorType mapping = new XmlMappingDiscriminatorType();
		mapping.setDiscriminatorType( discriminator.getType().getDisplayName() );

		final Iterator<Property> iterator = discriminator.getProperties().iterator();
		while ( iterator.hasNext() ) {
			final Property property = iterator.next();
			if ( property.isExpression() ) {
				mapping.setFormula( FormulaAdapter.from( property ).create() );
			}
			else {
				mapping.setColumn( ColumnAdapter.from( property ).create() );
			}
		}

		return mapping;
	}

