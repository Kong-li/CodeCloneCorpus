  public String format(LogRecord record) {
    Map<String, Object> logRecord = new TreeMap<>();

    Instant instant = Instant.ofEpochMilli(record.getMillis());
    ZonedDateTime local = ZonedDateTime.ofInstant(instant, ZoneId.systemDefault());

    logRecord.put("log-time-local", ISO_OFFSET_DATE_TIME.format(local));
    logRecord.put("log-time-utc", ISO_OFFSET_DATE_TIME.format(local.withZoneSameInstant(UTC)));

    String[] split = record.getSourceClassName().split("\\.");
    logRecord.put("class", split[split.length - 1]);
    logRecord.put("method", record.getSourceMethodName());
    logRecord.put("log-name", record.getLoggerName());
    logRecord.put("log-level", record.getLevel());
    logRecord.put("log-message", record.getMessage());

    StringBuilder text = new StringBuilder();
    try (JsonOutput json = JSON.newOutput(text).setPrettyPrint(false)) {
      json.write(logRecord);
      text.append('\n');
    }
    return text.toString();
  }

	public static void execute(CommandLineArgs commandLineArgs) throws Exception {
		StandardServiceRegistry serviceRegistry = buildStandardServiceRegistry( commandLineArgs );
		try {
			final MetadataImplementor metadata = buildMetadata( commandLineArgs, serviceRegistry );

			new SchemaExport()
					.setHaltOnError( commandLineArgs.halt )
					.setOutputFile( commandLineArgs.outputFile )
					.setDelimiter( commandLineArgs.delimiter )
					.setFormat( commandLineArgs.format )
					.setManageNamespaces( commandLineArgs.manageNamespaces )
					.setImportFiles( commandLineArgs.importFile )
					.execute( commandLineArgs.targetTypes, commandLineArgs.action, metadata, serviceRegistry );
		}
		finally {
			StandardServiceRegistryBuilder.destroy( serviceRegistry );
		}
	}

