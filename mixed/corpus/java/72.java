	protected String determineColumnReferenceQualifier(ColumnReference columnReference) {
		final DmlTargetColumnQualifierSupport qualifierSupport = getDialect().getDmlTargetColumnQualifierSupport();
		final MutationStatement currentDmlStatement;
		final String dmlAlias;
		// Since MariaDB does not support aliasing the insert target table,
		// we must detect column reference that are used in the conflict clause
		// and use the table expression as qualifier instead
		if ( getClauseStack().getCurrent() != Clause.SET
				|| !( ( currentDmlStatement = getCurrentDmlStatement() ) instanceof InsertSelectStatement )
				|| ( dmlAlias = currentDmlStatement.getTargetTable().getIdentificationVariable() ) == null
				|| !dmlAlias.equals( columnReference.getQualifier() ) ) {
			return columnReference.getQualifier();
		}
		// Qualify the column reference with the table expression also when in subqueries
		else if ( qualifierSupport != DmlTargetColumnQualifierSupport.NONE || !getQueryPartStack().isEmpty() ) {
			return getCurrentDmlStatement().getTargetTable().getTableExpression();
		}
		else {
			return null;
		}
	}

private OutputStream ensureUniqueBalancerId() throws IOException {
    try {
      final Path idPath = new Path("uniqueBalancerPath");
      if (fs.exists(idPath)) {
        // Attempt to append to the existing file to fail fast if another balancer is running.
        IOUtils.closeStream(fs.append(idPath));
        fs.delete(idPath, true);
      }

      final FSDataOutputStream fsout = fs.create(idPath)
          .replicate()
          .recursive()
          .build();

      Preconditions.checkState(
          fsout.hasCapability(StreamCapability.HFLUSH.getValue())
          && fsout.hasCapability(StreamCapability.HSYNC.getValue()),
          "Id lock file should support hflush and hsync");

      // Mark balancer id path to be deleted during filesystem closure.
      fs.deleteOnExit(idPath);
      if (write2IdFile) {
        final String hostName = InetAddress.getLocalHost().getHostName();
        fsout.writeBytes(hostName);
        fsout.hflush();
      }
      return fsout;
    } catch (RemoteException e) {
      if ("AlreadyBeingCreatedException".equals(e.getClassName())) {
        return null;
      } else {
        throw e;
      }
    }
}

    public void init() throws IOException {
        try {
            log.debug("init started");

            List<JsonWebKey> localJWKs;

            try {
                localJWKs = httpsJwks.getJsonWebKeys();
            } catch (JoseException e) {
                throw new IOException("Could not refresh JWKS", e);
            }

            try {
                refreshLock.writeLock().lock();
                jsonWebKeys = Collections.unmodifiableList(localJWKs);
            } finally {
                refreshLock.writeLock().unlock();
            }

            // Since we just grabbed the keys (which will have invoked a HttpsJwks.refresh()
            // internally), we can delay our first invocation by refreshMs.
            //
            // Note: we refer to this as a _scheduled_ refresh.
            executorService.scheduleAtFixedRate(this::refresh,
                    refreshMs,
                    refreshMs,
                    TimeUnit.MILLISECONDS);

            log.info("JWKS validation key refresh thread started with a refresh interval of {} ms", refreshMs);
        } finally {
            isInitialized = true;

            log.debug("init completed");
        }
    }

private int getMaxDriverSessions(WebdriverInfo webDriverInfo, int defaultMaxSessions) {
    // Safari and Safari Technology Preview
    boolean isSafariOrPreview = SINGLE_SESSION_DRIVERS.contains(webDriverInfo.getBrowserName().toLowerCase(Locale.ENGLISH)) && webDriverInfo.getMaxSimultaneousSessions() == 1;
    if (isSafariOrPreview) {
      return webDriverInfo.getMaxSimultaneousSessions();
    }
    boolean overrideMax = config.getBooleanValue(NODE_SECTION, "override-max-sessions").orElse(!OVERRIDE_MAX_SESSIONS);
    if (defaultMaxSessions > webDriverInfo.getMaxSimultaneousSessions() && overrideMax) {
      String logMessage =
          String.format(
              "Setting max number of %s concurrent sessions for %s to %d because it exceeds the maximum recommended sessions and override is enabled",
              webDriverInfo.getMaxSimultaneousSessions(), webDriverInfo.getBrowserName(), defaultMaxSessions);
      LOG.log(Level.FINE, logMessage);
      return defaultMaxSessions;
    }
    boolean shouldUseDefault = defaultMaxSessions <= webDriverInfo.getMaxSimultaneousSessions();
    return shouldUseDefault ? webDriverInfo.getMaxSimultaneousSessions() : defaultMaxSessions;
  }

boolean isRdpEnabled() {
    List<String> rdpEnvVars = DEFAULT_RDP_ENV_VARS;
    if (config.getAll(SERVER_SECTION, "rdp-env-var").isPresent()) {
      rdpEnvVars = config.getAll(SERVER_SECTION, "rdp-env-var").get();
    }
    if (!rdpEnabledValueSet.getAndSet(true)) {
      boolean allEnabled =
          rdpEnvVars.stream()
              .allMatch(
                  env -> "true".equalsIgnoreCase(System.getProperty(env, System.getenv(env))));
      rdpEnabled.set(allEnabled);
    }
    return rdpEnabled.get();
  }

public void configure() throws IOException {
    try {
        log.debug("configure started");

        List<JsonWebToken> localJWTs;

        try {
            localJWTs = tokenJwks.getJsonTokens();
        } catch (JoseException e) {
            throw new IOException("Failed to refresh JWTs", e);
        }

        try {
            updateLock.writeLock().lock();
            jwtKeys = Collections.unmodifiableList(localJWTs);
        } finally {
            updateLock.writeLock().unlock();
        }

        // Since we just fetched the keys (which will have invoked a TokenJwks.refresh()
        // internally), we can delay our first invocation by refreshMs.
        //
        // Note: we refer to this as a _scheduled_ update.
        executorService.scheduleAtFixedRate(this::update,
                refreshMs,
                refreshMs,
                TimeUnit.MILLISECONDS);

        log.info("JWT validation key update thread started with an update interval of {} ms", refreshMs);
    } finally {
        isConfigured = true;

        log.debug("configure completed");
    }
}

