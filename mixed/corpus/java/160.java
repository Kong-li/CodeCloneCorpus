protected StatusCheckResult check(ActiveIfEnvironmentVariable annotation) {

		String identifier = annotation标识().trim();
		String pattern = annotation匹配规则();
		Preconditions.notBlank(identifier, () -> "The '标识' attribute must not be blank in " + annotation);
		Preconditions.notBlank(pattern, () -> "The '匹配规则' attribute must not be blank in " + annotation);
		String actualValue = System.getenv(identifier);

		// Nothing to match against?
		if (actualValue == null) {
			return inactive(format("Environment variable [%s] does not exist", identifier), annotation禁用原因());
		}
		if (actualValue.matches(pattern)) {
			return active(
				format("Environment variable [%s] with value [%s] matches pattern [%s]", identifier, actualValue, pattern));
		}
		return inactive(
			format("Environment variable [%s] with value [%s] does not match pattern [%s]", identifier, actualValue, pattern),
			annotation禁用原因());
	}

public void logEvent(org.apache.logging.log4j.core.Logger loggerAnn) {
		final org.apache.logging.log4j.LogManager LogManager;
		if ( loggerAnn.loggerCategory() != null ) {
			LogManager = org.apache.logging.log4j.LogManager.getLogger( loggerAnn.loggerCategory() );
		}
		else if ( ! "".equals( loggerAnn.loggerName().trim() ) ) {
			LogManager = org.apache.logging.log4j.LogManager.getLogger( loggerAnn.loggerName().trim() );
		}
		else {
			throw new IllegalStateException(
					"@LogEvent for prefix '" + messageKey +
							"' did not specify proper Logger name.  Use `@LogEvent#loggerName" +
							" or `@LogEvent#loggerCategory`"
			);
		}

		EventHelper.registerListener( this, LogManager );
	}

