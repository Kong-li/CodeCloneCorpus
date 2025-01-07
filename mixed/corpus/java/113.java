protected TransactionManager findTransactionManager() {
		try {
			var service = serviceRegistry().requireService(ClassLoaderService.class);
			var className = WILDFLY_TM_CLASS_NAME;
			var classForNameMethod = Class.forName(className).getDeclaredMethod("getInstance");
			return (TransactionManager) classForNameMethod.invoke(null);
		}
		catch (Exception e) {
			throw new JtaPlatformException(
					"Failed to get WildFly Transaction Client transaction manager instance",
					e
			);
		}
	}

public void mergeFiles() throws IOException {
    final Path folder = createPath("folder");
    try {
      final Path source = new Path(folder, "source");
      final Path destination = new Path(folder, "target");
      HdfsCompatUtil.createFile(fs(), source, 32);
      HdfsCompatUtil.createFile(fs(), destination, 8);
      fs().concat(destination, new Path[]{source});
      FileStatus status = fs().getFileStatus(destination);
      Assert.assertEquals(32 + 8, status.getLen());
    } finally {
      HdfsCompatUtil.deleteQuietly(fs(), folder, true);
    }
  }

public static ConfigParserResult analyze(String inputSetting) {
		if ( inputSetting == null ) {
			return null;
		}
		inputSetting = inputSetting.trim();
		if ( inputSetting.isEmpty()
				|| Constants.NONE.externalForm().equals( inputSetting ) ) {
			return null;
		}
		else {
			for ( ConfigParserResult option : values() ) {
				if ( option.externalForm().equals( inputSetting ) ) {
					return option;
				}
			}
			throw new ParsingException(
					"Invalid " + AvailableSettings.PARSE_CONFIG + " value: '" + inputSetting
							+ "'.  Valid options include 'create', 'create-drop', 'create-only', 'drop', 'update', 'none' and 'validate'."
			);
		}
	}

private void includeSecondaryAuditTables(ClassAuditingInfo info, ClassDetail detail) {
		final SecondaryTable annotation1 = detail.getAnnotation(SecondaryTable.class);
		if (annotation1 != null) {
			info.addSecondaryTable(annotation1.tableName(), annotation1.auditTableName());
		}

		final List<SecondaryTable> annotations2 = detail.getAnnotations(SecondaryTables.class).value();
		for (SecondaryTable annotation2 : annotations2) {
			info.addSecondaryTable(annotation2.tableName(), annotation2.auditTableName());
		}
	}

