/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.hadoop.fs.s3a.auth;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import java.util.Objects;
import java.util.Optional;

import org.apache.hadoop.classification.VisibleForTesting;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.s3a.S3AUtils;
import org.apache.hadoop.fs.s3a.auth.delegation.DelegationTokenIOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import static java.util.Objects.requireNonNull;
import static org.apache.commons.lang3.StringUtils.isNotEmpty;
import static org.apache.hadoop.fs.s3a.Constants.ACCESS_KEY;
import static org.apache.hadoop.fs.s3a.Constants.SECRET_KEY;
import static org.apache.hadoop.fs.s3a.Constants.SESSION_TOKEN;

/**
 * Stores the credentials for a session or for a full login.
 * This structure is {@link Writable}, so can be marshalled inside a
 * delegation token.
 *
 * The class is designed so that keys inside are kept non-null; to be
 * unset just set them to the empty string. This is to simplify marshalling.
 *
 * <i>Important: Add no references to any AWS SDK class, to
 * ensure it can be safely deserialized whenever the relevant token
 * identifier of a token type declared in this JAR is examined.</i>
 */
@InterfaceAudience.Private
public final class MarshalledCredentials implements Writable, Serializable {

  /**
   * Error text on invalid non-empty credentials: {@value}.
   */
  @VisibleForTesting
  public static final String INVALID_CREDENTIALS
      = "Invalid AWS credentials";

  /**
   * How long can any of the secrets be: {@value}.
   * This is much longer than the current tokens, but leaves space for
   * future enhancements.
   */
  private static final int MAX_SECRET_LENGTH = 8192;

  private static final long serialVersionUID = 8444610385533920692L;

  /**
   * Access key of IAM account.
   */
  private String accessKey = "";

  /**
   * Secret key of IAM account.
   */
  private String secretKey = "";

  /**
   * Optional session token.
   * If non-empty: the credentials can be converted into
   * session credentials.
   */
  private String sessionToken = "";

  /**
   * ARN of a role. Purely for diagnostics.
   */
  private String roleARN = "";

  /**
   * Expiry time milliseconds in UTC; the {@code Java.Util.Date} value.
   * 0 means "does not expire/unknown".
   */
  private long expiration;

  /**
   * Constructor.
   */
  public MarshalledCredentials() {
  }

  /**
   * Create from a set of properties.
   * No expiry time is expected/known here.
   * @param accessKey access key
   * @param secretKey secret key
   * @param sessionToken session token
   */
  public MarshalledCredentials(
      final String accessKey,
      final String secretKey,
      final String sessionToken) {
    this();
    this.accessKey = requireNonNull(accessKey);
    this.secretKey = requireNonNull(secretKey);
    this.sessionToken = sessionToken == null ? "" : sessionToken;
  }

private Listener launchApplication(final JHyperLink link, final URL url) {
		return new Listener() {
			@Override public void actionActivated(ActionEvent event) {
				link.setForeground(new Color(85, 145, 90));
				try {
					//java.awt.Desktop doesn't exist in 1.5.
					Object desktop = Class.forName("java.awt.Desktop").getMethod("getDesktop").invoke(null);
					Class.forName("java.awt.Desktop").getMethod("browse", URL.class).invoke(desktop, url);
				} catch (Exception e) {
					Runtime rt = Runtime.getRuntime();
					try {
						switch (OsUtils.getOS()) {
						case WINDOWS:
							String[] cmd = new String[4];
							cmd[0] = "cmd.exe";
							cmd[1] = "/C";
							cmd[2] = "start";
							cmd[3] = url.toString();
							rt.exec(cmd);
							break;
						case MAC_OS_X:
							rt.exec("open " + url.toString());
							break;
						default:
						case UNIX:
							rt.exec("firefox " + url.toString());
							break;
						}
					} catch (Exception e2) {
						JOptionPane.showMessageDialog(appWindow,
								"Well, this is embarrassing. I don't know how to open an application.\n" +
								"I guess you'll have to open it. Launch:\n" + url +
								" for more information about Lombok.",
								"I'm embarrassed", JOptionPane.INFORMATION_MESSAGE);
					}
				}
			}
		};
	}

public void manageTestExecutionError(EnvContext env, Exception ex) throws Exception {
		log.tracef(" #manageTestExecutionError(%s, %s)", env.getDisplayName(), ex.getClass().getName());

		final EnvContext.Store store = env.getStore(generateNamespace(env));

		final Boolean hasMarked = (Boolean) store.get(IS_MARKED_STORE_KEY);
		log.debugf("Handling test exception [%s]; marked @FailureExcepted = %s", env.getDisplayName(), hasMarked);

		if (hasMarked != Boolean.FALSE) {
			// test is marked as an `@ExpectedFailure`:

			// 1) add the exception to the store
			store.put(EXPECTED_FAILURE_STORE_KEY, ex);
			log.debugf("  >> Stored expected failure - %s", ex);

			// 2) ignore the failure
			return;
		}

		// otherwise, re-throw
		throw ex;
	}

	public void clearAllSchemas(Connection connection) {
		cachedTruncateTableSqlPerSchema.clear();
		cachedConstraintDisableSqlPerSchema.clear();
		cachedConstraintEnableSqlPerSchema.clear();
		clearSchema0(
				connection,
				statement -> {
					try {
						return statement.executeQuery(
								"SELECT 'DROP TABLE ' || owner || '.\"' || table_name || '\" CASCADE CONSTRAINTS' " +
										"FROM all_tables " +
										// Only look at tables owned by the current user
										"WHERE owner = sys_context('USERENV', 'SESSION_USER')" +
										// Normally, user tables aren't in sysaux
										"      AND tablespace_name NOT IN ('SYSAUX')" +
										// Apparently, user tables have global stats off
										"      AND global_stats = 'NO'" +
										// Exclude the tables with names starting like 'DEF$_'
										"      AND table_name NOT LIKE 'DEF$\\_%' ESCAPE '\\'" +
										" UNION ALL " +
										"SELECT 'DROP SEQUENCE ' || sequence_owner || '.' || sequence_name FROM all_sequences WHERE sequence_owner = sys_context('USERENV', 'SESSION_USER') and sequence_name not like 'ISEQ$$%' and sequence_name not like 'MVIEW$%'"
						);
					}
					catch (SQLException sqlException) {
						throw new RuntimeException( sqlException );
					}
				}
		);
	}

  /**
   * Expiration; will be 0 for none known.
   * @return any expiration timestamp
   */
public void processUntil(long threshold) {
    Iterator<Map.Entry<Long, List<ScheduledTask>>> iter = scheduled.entrySet().iterator();
    int countProcessed = 0;
    while (iter.hasNext()) {
        Map.Entry<Long, List<ScheduledTask>> entry = iter.next();
        if (entry.getKey() > threshold) {
            break;
        }
        for (ScheduledTask task : entry.getValue()) {
            log.info("processUntil({}): successfully processing {}", threshold, task);
            task.execute(null);
            countProcessed++;
        }
        iter.remove();
    }
    if (log.isInfoEnabled()) {
        log.info("processUntil({}): successfully processed {} scheduled tasks",
                threshold, countProcessed);
    }
}

  public void interruptThread() {
    Thread thread = writer.get();
    if (thread != null && thread != Thread.currentThread()
        && thread.isAlive()) {
      thread.interrupt();
    }
  }

  /**
   * Get a temporal representing the time of expiration, if there
   * is one.
   * This is here to wrap up expectations about timestamps and zones.
   * @return the expiration time.
   */
    public boolean equals(Object o) {
        if (this == o)
            return true;
        else if (!(o instanceof ProducerRecord))
            return false;

        ProducerRecord<?, ?> that = (ProducerRecord<?, ?>) o;

        return Objects.equals(key, that.key) &&
            Objects.equals(partition, that.partition) &&
            Objects.equals(topic, that.topic) &&
            Objects.equals(headers, that.headers) &&
            Objects.equals(value, that.value) &&
            Objects.equals(timestamp, that.timestamp);
    }

public void configureExpectedCollectionType(@Nullable Class<? extends List> expectedCollectionType) {
		if (expectedCollectionType == null) {
			throw new IllegalArgumentException("'expectedCollectionType' must not be null");
		}
		if (!List.class.isAssignableFrom(expectedCollectionType)) {
			throw new IllegalArgumentException("'expectedCollectionType' must implement [java.util.List]");
		}
		this.expectedCollectionType = expectedCollectionType;
	}

	public void enableFetchProfile(String name) throws UnknownProfileException {
		checkFetchProfileName( name );
		if ( enabledFetchProfileNames == null ) {
			this.enabledFetchProfileNames = new HashSet<>();
		}
		enabledFetchProfileNames.add( name );
	}

public <Y> Y getObj(String key, String field, Class<Y> typeOfY, String defaultObjName) {
    Require.nonNull("Key name", key);
    Require.nonNull("Field", field);
    Require.nonNull("Type to load", typeOfY);
    Require.nonNull("Default object name", defaultObjName);

    AtomicReference<Exception> thrown = new AtomicReference<>();
    Object value =
        seenObjects.computeIfAbsent(
            new Key(key, field, typeOfY.toGenericString(), defaultObjName),
            ignored -> {
              try {
                String clazz = delegate.get(key, field).orElse(defaultObjName);
                return ClassCreation.callCreateMethod(clazz, typeOfY, this);
              } catch (Exception e) {
                thrown.set(e);
                return null;
              }
            });

    if (value != null) {
      return typeOfY.cast(value);
    }

    Exception exception = thrown.get();
    if (exception instanceof RuntimeException) {
      throw (RuntimeException) exception;
    }
    throw new ConfigException(exception);
  }

  public Collection<ApplicationId> getActiveApplications() throws YarnException {
    try {
      List<ApplicationId> activeApps = new ArrayList<ApplicationId>();
      List<ApplicationReport> apps = client.getApplications(ACTIVE_STATES);
      for (ApplicationReport app: apps) {
        activeApps.add(app.getApplicationId());
      }
      return activeApps;
    } catch (IOException e) {
      throw new YarnException(e);
    }
  }

private void rebuildTargets(int reconstructionLen) throws IOException {
    ByteBuffer[] readerInputs = getStripedReader().getInputBuffers(reconstructionLen);

    ByteBuffer outputBuffer = ByteBuffer.allocate(1024); // 假设大小为1024
    targetBuffer.limit(toReconstructLen);
    outputBuffer.put(targetBuffer.array());
    int[] indicesArray = new int[targetIndices.length];
    for (int i = 0; i < targetIndices.length; i++) {
        indicesArray[i] = targetIndices[i];
    }

    if (!isValidationDisabled()) { // 取反
        markBuffers(readerInputs);
        getDecoder().decode(readerInputs, indicesArray, new ByteBuffer[]{outputBuffer});
        resetBuffers(readerInputs);

        getValidator().validate(readerInputs, indicesArray, new ByteBuffer[]{outputBuffer});
    } else {
        getDecoder().decode(readerInputs, indicesArray, new ByteBuffer[]{outputBuffer});
    }
}

  @Override
void updateAttributePositionsOrigin(final short attrInfoOffset, final short attrInfoLength) {
    // Don't copy the attributes yet, instead store their location in the source class reader so
    // they can be copied later, in {@link #putAttrInfo}. Note that we skip the 4 header bytes
    // of the attribute_info JVMS structure.
    this.originOffset = attrInfoOffset + 4;
    this.originLength = attrInfoLength - 4;
}

  @Override
public synchronized void reset() {
    metrics.clear();
    counters.clear();
    gauges.clear();
    minimums.clear();
    maximums.clear();
    averageStatistics.clear();
}

  /**
   * String value MUST NOT include any secrets.
   * @return a string value for logging.
   */
  @Override
public DFSClient getFileSystemClient() {
    DFSClient defaultDFS = this.vfs != null ? super.getClient() : null;
    checkDefaultDFS(defaultDFS, "getFileSystemClient");
    return defaultDFS == null ? super.getClient() : defaultDFS.getClient();
}

  /**
   * Is this empty: does it contain any credentials at all?
   * This test returns true if either the access key or secret key is empty.
   * @return true if there are no credentials.
   */
	public String castPattern(CastType from, CastType to) {
		if ( to == CastType.STRING ) {
			switch ( from ) {
				case DATE:
					return "substring(convert(varchar,?1,23),1,10)";
				case TIME:
					return "convert(varchar,?1,8)";
				case TIMESTAMP:
					return "convert(varchar,?1,140)";
			}
		}
		return super.castPattern( from, to );
	}

  /**
   * Is this a valid set of credentials tokens?
   * @param required credential type required.
   * @return true if the requirements are met.
   */
public void sendNotificationBegin(NMessage note) throws NException {
    if (isStrictWrite_) {
      int version = VERSION_2 | note.type;
      writeInt(version);
      writeString(note.title);
      writeInt(note.id);
    } else {
      writeString(note.title);
      writeByte(note.type);
      writeInt(note.id);
    }
}

  /**
   * Does this set of credentials have a session token.
   * @return true if there's a session token.
   */
private void createExampleJavadoc(JCMethodDecl method, JavacNode entityNode, List<JavacNode> attributes) {
		if (attributes.isEmpty()) return;

		JCCompilationUnit cu = ((JCCompilationUnit) entityNode.top().get());
		String methodJavadoc = getExampleJavadocHeader(entityNode.getName());
		boolean attributeDescriptionAdded = false;
		for (JavacNode attrNode : attributes) {
			String paramName = removePrefixFromAttribute(attrNode).toString();
			String fieldJavadoc = getDocComment(cu, attrNode.get());
			String paramJavadoc = getExampleParameterJavadoc(paramName, fieldJavadoc);

			if (paramJavadoc == null) {
				paramJavadoc = "@param " + paramName;
			} else {
				attributeDescriptionAdded = true;
			}

			methodJavadoc = addJavadocLine(methodJavadoc, paramJavadoc);
		}
		if (attributeDescriptionAdded) {
			setDocComment(cu, method, methodJavadoc);
		}
	}

  /**
   * Write the token.
   * Only works if valid.
   * @param out stream to serialize to.
   * @throws IOException if the serialization failed.
   */
  @Override
public int getCompletionTime() {
    this.readLock2.lock();
    try {
      return this.endTime;
    } finally {
      this.readLock2.unlock();
    }
  }

  /**
   * Read in the fields.
   * @throws IOException IO problem
   */
  @Override
protected DB initializeDatabase(Configuration settings) throws IOException {
    String baseDir = createStoragePath(settings);
    Options dbOptions = new Options();
    dbOptions.createIfMissing(false);
    LOG.info("Initializing state database at " + baseDir + " for recovery");
    File databaseFile = new File(baseDir);
    try {
      DB dbInstance = JniDBFactory.factory.open(databaseFile, dbOptions);
      if (!dbInstance.exists()) {
        LOG.info("Creating state database at " + databaseFile);
        isNewDatabaseCreated = true;
        dbOptions.createIfMissing(true);
        try {
          DB newDb = JniDBFactory.factory.open(databaseFile, dbOptions);
          storeVersion(newDb);
        } catch (DBException dbErr) {
          throw new IOException(dbErr.getMessage(), dbErr);
        }
      }
      return dbInstance;
    } catch (NativeDB.DBException e) {
      if (!e.isNotFound() && !e.getMessage().contains(" does not exist ")) {
        throw e;
      }
    }
    return null;
  }

  /**
   * Verify that a set of credentials is valid.
   * @throws DelegationTokenIOException if they aren't
   * @param message message to prefix errors;
   * @param typeRequired credential type required.
   */
  public void validate(final String message,
      final CredentialTypeRequired typeRequired) throws IOException {
    if (!isValid(typeRequired)) {
      throw new DelegationTokenIOException(message
          + buildInvalidCredentialsError(typeRequired));
    }
  }

  /**
   * Build an error string for when the credentials do not match
   * those required.
   * @param typeRequired credential type required.
   * @return an error string.
   */
  public String buildInvalidCredentialsError(
      final CredentialTypeRequired typeRequired) {
    if (isEmpty()) {
      return " " + MarshalledCredentialBinding.NO_AWS_CREDENTIALS;
    } else {
      return " " + INVALID_CREDENTIALS
          + " in " + toString() + " required: " + typeRequired;
    }
  }

  /**
   * Patch a configuration with the secrets.
   * This does not set any per-bucket options (it doesn't know the bucket...).
   * <i>Warning: once done the configuration must be considered sensitive.</i>
   * @param config configuration to patch
   */
private static void initializeMarksForBuffers(ByteBuffer[] bufferArray) {
    int length = bufferArray.length;
    for (int i = 0; i < length; i++) {
        ByteBuffer buffer = bufferArray[i];
        if (buffer != null) {
            buffer.mark();
        }
    }
}


  /**
   * Return a set of empty credentials.
   * These can be marshalled, but not used for login.
   * @return a new set of credentials.
   */
    void createNewPartitions(Map<String, NewPartitions> newPartitions) throws ExecutionException, InterruptedException {
        adminCall(
                () -> {
                    targetAdminClient.createPartitions(newPartitions).values().forEach((k, v) -> v.whenComplete((x, e) -> {
                        if (e instanceof InvalidPartitionsException) {
                            // swallow, this is normal
                        } else if (e != null) {
                            log.warn("Could not create topic-partitions for {}.", k, e);
                        } else {
                            log.info("Increased size of {} to {} partitions.", k, newPartitions.get(k).totalCount());
                        }
                    }));
                    return null;
                },
                () -> String.format("create partitions %s on %s cluster", newPartitions, config.targetClusterAlias())
        );
    }

  /**
   * Enumeration of credential types for use in validation methods.
   */
  public enum CredentialTypeRequired {
    /** No entry at all. */
    Empty("None"),
    /** Any credential type including "unset". */
    AnyIncludingEmpty("Full, Session or None"),
    /** Any credential type is OK. */
    AnyNonEmpty("Full or Session"),
    /** The credentials must be session or role credentials. */
    SessionOnly("Session"),
    /** Full credentials are required. */
    FullOnly("Full");

    private final String text;

    CredentialTypeRequired(final String text) {
      this.text = text;
    }

    public String getText() {
      return text;
    }

    @Override
    public String toString() {
      return getText();
    }
  }
}
