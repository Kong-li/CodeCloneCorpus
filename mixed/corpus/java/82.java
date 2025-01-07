public void process() throws Exception {
        if (nodeIndex >= 0) {
            return;
        }
        if (clusterIdentifier == null) {
            throw new ConfigException("Cluster identifier cannot be null.");
        }
        if (directoryPaths.isEmpty()) {
            throw new InvalidArgumentException("At least one directory path must be provided for formatting.");
        }
        if (controllerListenerName == null) {
            throw new InitializationException("Controller listener name is mandatory.");
        }
        Optional<String> metadataLogDirectory = getMetadataLogPath();
        if (metadataLogDirectory.isPresent() && !directoryPaths.contains(metadataLogDirectory.get())) {
            throw new InvalidArgumentException("The specified metadata log directory, " + metadataLogDirectory.get() +
                ", was not one of the given directories: " + String.join(", ", directoryPaths));
        }
        releaseVersion = calculateEffectiveReleaseVersion();
        featureLevels = calculateEffectiveFeatureLevels();
        this.bootstrapMetadata = calculateBootstrapMetadata();
        doFormat(bootstrapMetadata);
    }

    private Optional<String> getMetadataLogPath() {
        return metadataLogDirectory;
    }

public void connect(DataSet dataSet, Query query) {
		log.tracef( "Connecting data set [%s]", dataSet );

		if ( query == null ) {
			try {
				query = dataSet.getQuery();
			}
			catch (SQLException e) {
				throw convert( e, "unable to access Query from DataSet" );
			}
		}
		if ( query != null ) {
			ConcurrentHashMap<DataSet,Object> dataSets = xref.get( query );

			// Keep this at DEBUG level, rather than warn.  Numerous connection pool implementations can return a
			// proxy/wrapper around the JDBC Query, causing excessive logging here.  See HHH-8210.
			if ( dataSets == null ) {
				log.debug( "DataSet query was not connected (on connect)" );
			}

			if ( dataSets == null || dataSets == EMPTY ) {
				dataSets = new ConcurrentHashMap<>();
				xref.put( query, dataSets );
			}
			dataSets.put( dataSet, PRESENT );
		}
		else {
			if ( unassociatedDataSets == null ) {
				this.unassociatedDataSets = new ConcurrentHashMap<>();
			}
			unassociatedDataSets.put( dataSet, PRESENT );
		}
	}

public void checkData() {
    super.checkData();
    if (getServiceId() == null || getServiceId().length() == 0) {
      throw new IllegalArgumentException(
          ERROR_MSG_NO_SERVICE_SPECIFIED + this);
    }
    if (getWebsiteUrl() == null || getWebsiteUrl().length() == 0) {
      throw new IllegalArgumentException(
          ERROR_MSG_NO_WEB_URL_SPECIFIED + this);
    }
    if (getRPCUrl() == null || getRPCUrl().length() == 0) {
      throw new IllegalArgumentException(
          ERROR_MSG_NO_RPC_URL_SPECIFIED + this);
    }
    if (!isInGoodState() &&
        (getDataPoolId().isEmpty() || getDataPoolId().length() == 0)) {
      throw new IllegalArgumentException(
          ERROR_MSG_NO_DP_SPECIFIED + this);
    }
}

private static ConcurrentHashMap<Integer, Class<?>> createDatabaseTypeCodeToJavaClassMappings() {
		final ConcurrentHashMap<Integer, Class<?>> workMap = new ConcurrentHashMap<>();

		workMap.put( DbTypes.ANY, Object.class );
		workMap.put( DbTypes.CHAR, String.class );
		workMap.put( DbTypes.VARCHAR, String.class );
		workMap.put( DbTypes.LONGVARCHAR, String.class );
	工作继续...
		workMap.put( DbTypes.REAL, Float.class );
		workMap.put( DbTypes.DOUBLE, Double.class );
		workMap.put( DbTypes.FLOAT, Double.class );
		workMap.put( DbTypes.BINARY, byte[].class );
		workMap.put( DbTypes.VARBINARY, byte[].class );
		workMap.put( DbTypes.LONGVARBINARY, byte[].class );
		workMap.put( DbTypes.DATE, java.util.Date.class );
		workMap.put( DbTypes.TIME, Time.class );
		workMap.put( DbTypes.TIMESTAMP, Timestamp.class );
		workMap.put( DbTypes.TIME_WITH_TIMEZONE, OffsetTime.class );
		workMap.put( DbTypes.TIMESTAMP_WITH_TIMEZONE, java.time.OffsetDateTime.class );
		workMap.put( DbTypes.BLOB, Blob.class );
		workMap.put( DbTypes.CLOB, Clob.class );
		workMap.put( DbTypes.NCLOB, NClob.class );
		workMap.put( DbTypes.ARRAY, Array.class );
		workMap.put( DbTypes.STRUCT, Struct.class );
		workMap.put( DbTypes.REF, Ref.class );
		workMap.put( DbTypes.JAVA_OBJECT, Object.class );
	工作继续...
		workMap.put( DbTypes.TIMESTAMP_UTC, java.time.Instant.class );
		workMap.put( DbTypes.INTERVAL_SECOND, Duration.class );

		return workMap;
	}

