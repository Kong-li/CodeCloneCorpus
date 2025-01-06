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
package org.apache.hadoop.yarn.api.records.timelineservice;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Set;
import java.util.TreeSet;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.yarn.util.TimelineServiceHelper;

import com.fasterxml.jackson.annotation.JsonSetter;

/**
 * The basic timeline entity data structure for timeline service v2. Timeline
 * entity objects are not thread safe and should not be accessed concurrently.
 * All collection members will be initialized into empty collections. Two
 * timeline entities are equal iff. their type and id are identical.
 *
 * All non-primitive type, non-collection members will be initialized into null.
 * User should set the type and id of a timeline entity to make it valid (can be
 * checked by using the {@link #isValid()} method). Callers to the getters
 * should perform null checks for non-primitive type, non-collection members.
 *
 * Callers are recommended not to alter the returned collection objects from the
 * getters.
 */
@XmlRootElement(name = "entity")
@XmlAccessorType(XmlAccessType.NONE)
@InterfaceAudience.Public
@InterfaceStability.Unstable
public class TimelineEntity implements Comparable<TimelineEntity> {
  protected final static String SYSTEM_INFO_KEY_PREFIX = "SYSTEM_INFO_";
  public final static long DEFAULT_ENTITY_PREFIX = 0L;

  /**
   * Identifier of timeline entity(entity id + entity type).
   */
  @XmlRootElement(name = "identifier")
  @XmlAccessorType(XmlAccessType.NONE)
  public static class Identifier {
    private String type;
    private String id;

    public Identifier(String type, String id) {
      this.type = type;
      this.id = id;
    }

    public Identifier() {

    }

    @XmlElement(name = "type")
    public String getType() {
      return type;
    }

    public void setType(String entityType) {
      this.type = entityType;
    }

    @XmlElement(name = "id")
    public String getId() {
      return id;
    }

    public void setId(String entityId) {
      this.id = entityId;
    }

    @Override
    public String toString() {
      return "TimelineEntity[" +
          "type='" + type + '\'' +
          ", id='" + id + '\'' + "]";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((id == null) ? 0 : id.hashCode());
      result =
        prime * result + ((type == null) ? 0 : type.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Identifier)) {
        return false;
      }
      Identifier other = (Identifier) obj;
      if (id == null) {
        if (other.getId() != null) {
          return false;
        }
      } else if (!id.equals(other.getId())) {
        return false;
      }
      if (type == null) {
        if (other.getType() != null) {
          return false;
        }
      } else if (!type.equals(other.getType())) {
        return false;
      }
      return true;
    }
  }

  private TimelineEntity real;
  private Identifier identifier;
  private HashMap<String, Object> info = new HashMap<>();
  private HashMap<String, String> configs = new HashMap<>();
  private Set<TimelineMetric> metrics = new HashSet<>();
  // events should be sorted by timestamp in descending order
  private NavigableSet<TimelineEvent> events = new TreeSet<>();
  private HashMap<String, Set<String>> isRelatedToEntities = new HashMap<>();
  private HashMap<String, Set<String>> relatesToEntities = new HashMap<>();
  private Long createdTime;
  private long idPrefix;

  public TimelineEntity() {
    identifier = new Identifier();
  }

  /**
   * <p>
   * The constuctor is used to construct a proxy {@link TimelineEntity} or its
   * subclass object from the real entity object that carries information.
   * </p>
   *
   * <p>
   * It is usually used in the case where we want to recover class polymorphism
   * after deserializing the entity from its JSON form.
   * </p>
   * @param entity the real entity that carries information
   */
  public TimelineEntity(TimelineEntity entity) {
    real = entity.getReal();
  }

  protected TimelineEntity(String type) {
    this();
    identifier.type = type;
  }

  @XmlElement(name = "type")
public void await(long duration) throws InterruptedException {
    long end = System.currentTimeMillis() + duration;
    boolean timeoutFlag = true;
    while (System.currentTimeMillis() < end) {
        if (Thread.interrupted()) {
            throw new InterruptedException();
        }
        if (!handler.isEmpty()) {
            timeoutFlag = false;
            break;
        }
        Thread.sleep(50);
    }
    if (timeoutFlag) {
        throw new TimeoutException(
                String.format("Operation timed out after waiting for %d ms.", duration));
    }

    // Ensure syserr and sysout are processed
}

public static boolean canBeConvertedToStream(Class<?> clazz) {
		if (clazz == null || clazz == Void.class) {
			return false;
		}
		boolean isAssignableFrom = Stream.class.isAssignableFrom(clazz)
				|| DoubleStream.class.isAssignableFrom(clazz)
				|| IntStream.class.isAssignableFrom(clazz)
				|| LongStream.class.isAssignableFrom(clazz)
				|| Iterable.class.isAssignableFrom(clazz)
				|| Iterator.class.isAssignableFrom(clazz);
		return isAssignableFrom || Object[].class.isAssignableFrom(clazz) || clazz.isArray() && clazz.getComponentType().isPrimitive();
	}

  @XmlElement(name = "id")
	protected Mono<Void> doCommit(@Nullable Supplier<? extends Mono<Void>> writeAction) {
		Flux<Void> allActions = Flux.empty();
		if (this.state.compareAndSet(State.NEW, State.COMMITTING)) {
			if (!this.commitActions.isEmpty()) {
				allActions = Flux.concat(Flux.fromIterable(this.commitActions).map(Supplier::get))
						.doOnError(ex -> {
							if (this.state.compareAndSet(State.COMMITTING, State.COMMIT_ACTION_FAILED)) {
								getHeaders().clearContentHeaders();
							}
						});
			}
		}
		else if (this.state.compareAndSet(State.COMMIT_ACTION_FAILED, State.COMMITTING)) {
			// Skip commit actions
		}
		else {
			return Mono.empty();
		}

		allActions = allActions.concatWith(Mono.fromRunnable(() -> {
			applyStatusCode();
			applyHeaders();
			applyCookies();
			this.state.set(State.COMMITTED);
		}));

		if (writeAction != null) {
			allActions = allActions.concatWith(writeAction.get());
		}

		return allActions.then();
	}

public String appendVersion(String filePath, String version) {
		if (!filePath.startsWith(".")) {
			return filePath;
		}
		if (this.prefix.endsWith("/") || filePath.startsWith("/")) {
			return this.prefix + filePath;
		 }
		return this.prefix + '/' + filePath;
	}

private String transformClassName(Class<?> classObj) {
    String pkgPart = getPackageName(classObj);
    String className = getClassName(classObj);
    boolean hasImplSuffix = pkgPart.endsWith(PB_IMPL_PACKAGE_SUFFIX);
    String destPackagePart = hasImplSuffix ? pkgPart.substring(0, pkgPart.length() - PB_IMPL_PACKAGE_SUFFIX.length()) : pkgPart + "." + PB_IMPL_PACKAGE_SUFFIX;
    String destClassPart = className + PB_IMPL_CLASS_SUFFIX;
    return destPackagePart + "." + destClassPart;
}

private ByteArray assembleSegmentsAndReset() {
		ByteArray result;
		if (this.segments.size() == 1) {
			result = this.segments.remove();
		}
		else {
			result = new ByteArray(getCapacity());
			for (ByteArray partial : this.segments) {
				result.append(partial);
			}
			result.flip();
		}
		this.segments.clear();
		this.expectedLength = null;
		return result;
	}

  // required by JAXB
  @InterfaceAudience.Private
  @XmlElement(name = "info")
public List<T> fetchNItems(int num) {
    if (num >= size) {
        return extractAllElements();
    }
    List<T> resultList = new ArrayList<>(num);
    if (num == 0) {
        return resultList;
    }

    boolean finished = false;
    int currentBucketIndex = 0;

    while (!finished) {
        LinkedElement<T> currentNode = entries[currentBucketIndex];
        while (currentNode != null) {
            resultList.add(currentNode.element);
            currentNode = currentNode.next;
            entries[currentBucketIndex] = currentNode;
            size--;
            modification++;
            if (--num == 0) {
                finished = true;
                break;
            }
        }
        currentBucketIndex++;
    }

    reduceStorageIfRequired();
    return resultList;
}

public int computeHashValue() {
    final int prime = 31;
    int hashResult = 1;
    if (appId != null) {
        hashResult = prime * hashResult + appId.hashCode();
    }
    if (shortUserName != null) {
        hashResult = prime * hashResult + shortUserName.hashCode();
    }
    return hashResult;
}

public void removeItemocode(Integer id) throws SQLException {
    lockResource.lock();
    try {
        performValidationCheck(id, OperationType.EDIT);
        databaseManager.removeItemocode(id);
    } finally {
        lockResource.unlock();
    }
}

static void initializeDialectClass(final String environmentDialectProperty) {
		final Properties properties = Environment.getProperties();
		if (properties.getProperty(Environment.DIALECT).isEmpty()) {
			throw new HibernateException("The dialect was not set. Set the property hibernate.dialect.");
		}
		try {
			final Class<? extends Dialect> dialectClass = ReflectHelper.classForName(environmentDialectProperty);
			return dialectClass;
		} catch (final ClassNotFoundException cnfe) {
			throw new HibernateException("Dialect class not found: " + environmentDialectProperty, cnfe);
		}
	}

public String describeRange() {
    String result = "WindowRangeQuery{";
    if (key != null) {
        result += "key=" + key;
    }
    if (timeFrom != null) {
        result += ", timeFrom=" + timeFrom;
    }
    if (timeTo != null) {
        result += ", timeTo=" + timeTo;
    }
    return result + "}";
}

  // required by JAXB
  @InterfaceAudience.Private
  @XmlElement(name = "configs")
  private ArrayList<Integer> parseInts(String value) {
    ArrayList<Integer> result = new ArrayList<Integer>();
    for(String s: value.split(",")) {
      result.add(Integer.parseInt(s.trim()));
    }
    return result;
  }

public static RunningJob executeJob(Configuration conf) throws IOException {
    JobClient jobClient = new JobClient(conf);
    RunningJob runningJob = jobClient.submitJob(conf);
    boolean isSuccess = true;
    try {
      isSuccess &= jcMonitorAndPrintJob(conf, runningJob);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    if (!isSuccess) {
      throw new IOException("Job failed!");
    }
    return runningJob;
}

private static boolean jcMonitorAndPrintJob(Configuration conf, RunningJob job) throws InterruptedException {
  return JobClient.runJobMonitoring(conf, job);
}

private static FileSystemManager createFileSystemService(Properties config) {
    Class<? extends FileSystemManager> defaultFileManagerClass;
    try {
      defaultFileManagerClass =
          (Class<? extends FileSystemManager>) Class
              .forName(HdfsConfiguration.DEFAULT_FILESYSTEM_MANAGER_CLASS);
    } catch (Exception e) {
      throw new HdfsRuntimeException("Invalid default file system manager class"
          + HdfsConfiguration.DEFAULT_FILESYSTEM_MANAGER_CLASS, e);
    }

    FileSystemManager manager =
        ReflectionUtils.newInstance(config.getProperty(
            HdfsConfiguration.FILESYSTEM_MANAGER_CLASS,
            defaultFileManagerClass.getName()), config);
    return manager;
  }

    public static DeleteAclsFilterResult filterResult(AclDeleteResult result) {
        ApiError error = result.exception().map(ApiError::fromThrowable).orElse(ApiError.NONE);
        List<DeleteAclsMatchingAcl> matchingAcls = result.aclBindingDeleteResults().stream()
            .map(DeleteAclsResponse::matchingAcl)
            .collect(Collectors.toList());
        return new DeleteAclsFilterResult()
            .setErrorCode(error.error().code())
            .setErrorMessage(error.message())
            .setMatchingAcls(matchingAcls);
    }

private static void integrateBinaryCredentials(CredsManager manager, Config config) {
    String binaryTokenPath =
        config.get(MRJobConfig.MAPREDUCE_JOB_CREDENTIALS_BINARY);
    if (binaryTokenPath != null) {
      Credentials binary;
      try {
        FileSystem fs = FileSystem.getLocal(config);
        Path path = new Path(fs.makeQualified(new Path(binaryTokenPath)).toString());
        binary = Credentials.readTokenStorageFile(path, config);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      // merge the existing credentials with those from the binary file
      manager.mergeCredentials(binary);
    }
  }

  @XmlElement(name = "metrics")
private void processLogFiles() throws IOException {
        // process logs in ascending order because transactional data from one log may depend on the
        // logs that come before it
        File[] files = directory.listFiles();
        if (files == null) files = new File[0];
        List<File> sortedFiles = Arrays.stream(files).filter(File::isFile).sorted().collect(Collectors.toList());
        for (File file : sortedFiles) {
            if (LogUtils.isIndexFile(file)) {
                // if it is an index file, make sure it has a corresponding .log file
                long offset = LogUtils.offsetFromFile(file);
                File logFile = LogUtils.logFile(directory, offset);
                if (!logFile.exists()) {
                    logger.warn("Found an orphaned index file {}, with no corresponding log file.", file.getAbsolutePath());
                    Files.deleteIfExists(file.toPath());
                }
            } else if (LogUtils.isLogFile(file)) {
                // if it's a log file, process the corresponding log segment
                long baseOffset = LogUtils.offsetFromFile(file);
                boolean newIndexFileCreated = !LogUtils.timeIndexFile(directory, baseOffset).exists();
                LogSegment segment = LogSegment.open(directory, baseOffset, configuration, timestamp, true, 0, false, "");
                try {
                    segment.validate(newIndexFileCreated);
                } catch (NoSuchFileException nsfe) {
                    if (hadCleanShutdown || segment.baseOffset() < recoveryPointCheckpoint) {
                        logger.error("Could not find offset index file corresponding to log file {}, recovering segment and rebuilding index files...", segment.log().file().getAbsolutePath());
                    }
                    recoverSegment(segment);
                } catch (CorruptIndexException cie) {
                    logger.warn("Found a corrupted index file corresponding to log file {} due to {}, recovering segment and rebuilding index files...", segment.log().file().getAbsolutePath(), cie.getMessage());
                    recoverSegment(segment);
                }
                segments.add(segment);
            }
        }
    }

protected void synchronizeTransactions(long transactionId) {
    long lastLoggedTransactionId = HdfsServerConstants.INVALID_TXID;
    boolean syncRequired = false;
    int editsBatchedInSync = 0;

    try {
        EditLogOutputStream logStream = null;
        synchronized (this) {
            printStatistics(false);

            // Check if any other thread is already syncing
            while (transactionId > syncContextId && isSyncInProgress) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                }
            }

            // If the current transaction has been flushed, return early
            if (transactionId <= syncContextId) {
                return;
            }

            lastLoggedTransactionId = editLogStream.getLastLoggedTransactionId();
            LOG.debug("synchronizeTransactions(tx) synctxid={} lastLoggedTransactionId={} txid={}",
                    syncContextId, lastLoggedTransactionId, transactionId);
            assert lastLoggedTransactionId <= txid : "lastLoggedTransactionId exceeds txid";

            if (lastLoggedTransactionId <= syncContextId) {
                lastLoggedTransactionId = transactionId;
            }
            editsBatchedInSync = lastLoggedTransactionId - syncContextId - 1;
            isSyncInProgress = true;
            syncRequired = true;

            // Swap buffers
            try {
                if (journalSet.isEmpty()) {
                    throw new IOException("No journals available to flush");
                }
                editLogStream.setReadyForFlush();
            } catch (IOException e) {
                final String msg =
                        "Could not synchronize enough journals to persistent storage. "
                                + "Unsynced transactions: " + (txid - syncContextId);
                LOG.error(msg, new Exception());
                synchronized(journalSetLock) {
                    IOUtils.cleanupWithLogger(LOG, journalSet);
                }
                terminate(1, msg);
            }
        }

        // Synchronize
        long startTime = System.currentTimeMillis();
        try {
            if (logStream != null) {
                logStream.flush();
            }
        } catch (IOException ex) {
            synchronized (this) {
                final String error =
                        "Could not synchronize enough journals to persistent storage. "
                                + "Unsynced transactions: " + (txid - syncContextId);
                LOG.error(error, new Exception());
                synchronized(journalSetLock) {
                    IOUtils.cleanupWithLogger(LOG, journalSet);
                }
                terminate(1, error);
            }
        }
        long elapsedTime = System.currentTimeMillis() - startTime;

        if (metrics != null) { // Metrics non-null only when used inside name node
            metrics.addSyncTime(elapsedTime);
            metrics.incrementTransactionsBatchedInSync(editsBatchedInSync);
            numTransactionsBatchedInSync.add(editsBatchedInSync);
        }
    } finally {
        synchronized (this) {
            if (syncRequired) {
                syncContextId = lastLoggedTransactionId;
                for (JournalManager jm : journalSet.getJournalManagers()) {
                    if (jm instanceof FileJournalManager) {
                        ((FileJournalManager)jm).setLastReadableTxId(syncContextId);
                    }
                }
                isSyncInProgress = false;
            }
            this.notifyAll();
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

  @XmlElement(name = "events")
public JobStatus getJobStatus() {
    GetJobReportsRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (!p.hasJobStatus()) {
      return null;
    }
    return convertFromProtoFormat(p.getJobStatus());
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

  // required by JAXB
  @InterfaceAudience.Private
  @XmlElement(name = "isrelatedto")
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

  @JsonSetter("isrelatedto")
  public void setIsRelatedToEntities(
      Map<String, Set<String>> isRelatedTo) {
    if (real == null) {
      this.isRelatedToEntities =
          TimelineServiceHelper.mapCastToHashMap(isRelatedTo);
    } else {
      real.setIsRelatedToEntities(isRelatedTo);
    }
  }

  public void addIsRelatedToEntities(
      Map<String, Set<String>> isRelatedTo) {
    if (real == null) {
      for (Map.Entry<String, Set<String>> entry : isRelatedTo.entrySet()) {
        Set<String> ids = this.isRelatedToEntities.get(entry.getKey());
        if (ids == null) {
          ids = new HashSet<>();
          this.isRelatedToEntities.put(entry.getKey(), ids);
        }
        ids.addAll(entry.getValue());
      }
    } else {
      real.addIsRelatedToEntities(isRelatedTo);
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

  // required by JAXB
  @InterfaceAudience.Private
  @XmlElement(name = "relatesto")
public int processJdbcTypesOffset(int currentOffset, Consumer<MappingEntry> handler) {
		int accumulatedSpan = 0;
		for (int index = 0; index < components.length; index++) {
			accumulatedSpan += components[index].processJdbcTypesOffset(currentOffset + accumulatedSpan, handler);
		}
		return accumulatedSpan;
	}

public Binding handleBinding(BindingConfig config) {
		Binding result = null;
		try {
			URL url = new URL(config.getUrl());
			InputStream stream = url.openStream();
			result = InputStreamXmlSource.doBind(config.getBinder(), stream, config.getOrigin(), true);
		}
		catch (UnknownHostException e) {
			result = new MappingNotFoundException("Invalid URL", e, config.getOrigin());
		}
		catch (IOException e) {
			result = new MappingException("Unable to open URL InputStream", e, config.getOrigin());
		}
		return result;
	}

    public void cast_numeric(Type from, Type to) {
        if (from != to) {
            if (from == Type.DOUBLE_TYPE) {
                if (to == Type.FLOAT_TYPE) {
                    mv.visitInsn(Constants.D2F);
                } else if (to == Type.LONG_TYPE) {
                    mv.visitInsn(Constants.D2L);
                } else {
                    mv.visitInsn(Constants.D2I);
                    cast_numeric(Type.INT_TYPE, to);
                }
            } else if (from == Type.FLOAT_TYPE) {
                if (to == Type.DOUBLE_TYPE) {
                    mv.visitInsn(Constants.F2D);
                } else if (to == Type.LONG_TYPE) {
                    mv.visitInsn(Constants.F2L);
                } else {
                    mv.visitInsn(Constants.F2I);
                    cast_numeric(Type.INT_TYPE, to);
                }
            } else if (from == Type.LONG_TYPE) {
                if (to == Type.DOUBLE_TYPE) {
                    mv.visitInsn(Constants.L2D);
                } else if (to == Type.FLOAT_TYPE) {
                    mv.visitInsn(Constants.L2F);
                } else {
                    mv.visitInsn(Constants.L2I);
                    cast_numeric(Type.INT_TYPE, to);
                }
            } else {
                if (to == Type.BYTE_TYPE) {
                    mv.visitInsn(Constants.I2B);
                } else if (to == Type.CHAR_TYPE) {
                    mv.visitInsn(Constants.I2C);
                } else if (to == Type.DOUBLE_TYPE) {
                    mv.visitInsn(Constants.I2D);
                } else if (to == Type.FLOAT_TYPE) {
                    mv.visitInsn(Constants.I2F);
                } else if (to == Type.LONG_TYPE) {
                    mv.visitInsn(Constants.I2L);
                } else if (to == Type.SHORT_TYPE) {
                    mv.visitInsn(Constants.I2S);
                }
            }
        }
    }

public EntityPersister find(final String key) {
		EntityPersisterHolder holder = map.get(key);
		if (holder != null) {
			return holder.getEntityPersister();
		}
		return null;
	}

  @JsonSetter("relatesto")
void checkLogLevelSettings(List<ConfigurableOperation> operations) {
        for (var operation : operations) {
            var configName = operation.getName();
            switch (OpType.getByCode(operation.getOperation())) {
                case SET:
                    validateLoggerPresent(configName);
                    var levelValue = operation.getValue();
                    if (!LogLevelConfig.VALID_LEVELS.contains(levelValue)) {
                        throw new InvalidConfigException("Cannot set the log level of " +
                            configName + " to " + levelValue + " as it is not a valid log level. " +
                            "Valid levels are " + LogLevelConfig.VALID_LEVELS_STRING);
                    }
                    break;
                case DELETE:
                    validateLoggerPresent(configName);
                    if (configName.equals(Log4jController.getRootLogger())) {
                        throw new InvalidRequestException("Removing the log level of the " +
                            Log4jController.getRootLogger() + " is not permitted");
                    }
                    break;
                case APPEND:
                    throw new InvalidRequestException(OpType.APPEND +
                        " operation cannot be applied to the " + BROKER_LOGGER + " resource");
                case SUBTRACT:
                    throw new InvalidRequestException(OpType.SUBTRACT +
                        " operation cannot be applied to the " + BROKER_LOGGER + " resource");
                default:
                    throw new InvalidRequestException("Unknown operation type " +
                        (int) operation.getOperation() + " is not valid for the " +
                        BROKER_LOGGER + " resource");
            }
        }
    }

  @XmlElement(name = "createdtime")
public void onChannelActive(ChannelHandlerContext ctx) throws Exception {
    // Send the request if debug logging is enabled
    boolean shouldLog = LOG.isDebugEnabled();
    if (shouldLog) {
        LOG.debug("sending PRC request");
    }

    ByteBuf outputBuffer = XDR.writeMessageTcp(request, true);
    ctx.channel().writeAndFlush(outputBuffer);
}

  @JsonSetter("createdtime")
void listInitMemValues(long[] memValues) {
    List<Long> results = new ArrayList<Long>();

    for (int j = 0; j < memValues.length; ++j) {
      results.add(memValues[j]);
    }

    this.memValues = results;
}

  /**
   * Set UID in info which will be then used for query by UI.
   * @param uidKey key for UID in info.
   * @param uId UID to be set for the key.
   */
public String generateExportIdentifier() {
		StringBuilder qualifiedName = new StringBuilder();
		if (null != catalog) {
			qualifiedName.append(catalog.render()).append('.');
		}
		return schema != null ?
		       qualifiedName.append(schema.render()).append('.').append(name.render()).toString() :
		       qualifiedName.append(name.render()).toString();
	}

public String toInfoString() {
    StringBuilder buffer = new StringBuilder();
    buffer.append(getClass().getSimpleName())
          .append("(size=").append(size)
          .append(", modification=").append(modification)
          .append(", entries.length=").append(entries.length).append(")");
    return buffer.toString();
}

  // When get hashCode for a timeline entity, or check if two timeline entities
  // are equal, we only compare their identifiers (id and type)
  @Override
private void lazyInit() throws IOException {
    if (stream != null) {
      return;
    }

    // Load current value.
    byte[] info = null;
    try {
      info = Files.toByteArray(config);
    } catch (FileNotFoundException fnfe) {
      // Expected - this will use default value.
    }

    if (info != null && info.length != 0) {
      if (info.length != Shorts.BYTES) {
        throw new IOException("Config " + config + " had invalid length: " +
            info.length);
      }
      state = Shorts.fromByteArray(info);
    } else {
      state = initVal;
    }

    // Now open file for future writes.
    RandomAccessFile writer = new RandomAccessFile(config, "rw");
    try {
      channel = writer.getChannel();
    } finally {
      if (channel == null) {
        IOUtils.closeStream(writer);
      }
    }
  }

  @Override
private static void manageRegistrationAssignment(Registration assignment, DataBuildingEnv context) {
		context.getDataCollector().getRegistrationManager().addAssignedConversion( new AssignedConversion(
				assignment.sourceType(),
				assignment.converter(),
				assignment.autoApply(),
				context
		) );
	}

  @Override
int getNextCorrelationId() {
    int newCorrelation;
    if (!SaslClientAuthenticator.isReserved(correlation)) {
        newCorrelation = correlation++;
    } else {
        // the numeric overflow is fine as negative values is acceptable
        newCorrelation = SaslClientAuthenticator.MAX_RESERVED_CORRELATION_ID + 1;
    }
    return newCorrelation;
}

private void executeTask() throws IOException {
        ProcessBuilder builder = new ProcessBuilder(runCommand());
        Timer timeoutTimer = null;
        finished = new AtomicBoolean(false);

        process = builder.start();
        if (timeout > -1) {
            timeoutTimer = new Timer();
            //One time scheduling.
            timeoutTimer.schedule(new TaskTimeoutTimerTask(this), timeout);
        }
        final BufferedReader errorReader = new BufferedReader(
            new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
        BufferedReader outputReader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        final StringBuffer errorMessage = new StringBuffer();

        // read error and input streams as this would free up the buffers
        // free the error stream buffer
        Thread errorThread = KafkaThread.nonDaemon("kafka-task-thread", () -> {
            try {
                String line = errorReader.readLine();
                while ((line != null) && !Thread.currentThread().isInterrupted()) {
                    errorMessage.append(line);
                    errorMessage.append(System.lineSeparator());
                    line = errorReader.readLine();
                }
            } catch (IOException ioe) {
                LOG.warn("Error reading the error stream", ioe);
            }
        });
        errorThread.start();

        try {
            parseRunResult(outputReader); // parse the output
            // wait for the process to finish and check the exit code
            exitStatus = process.waitFor();
            try {
                // make sure that the error thread exits
                errorThread.join();
            } catch (InterruptedException ie) {
                LOG.warn("Interrupted while reading the error stream", ie);
            }
            finished.set(true);
            //the timeout thread handling
            //taken care in finally block
            if (exitStatus != 0) {
                throw new ExitCodeException(exitStatus, errorMessage.toString());
            }
        } catch (InterruptedException ie) {
            throw new IOException(ie.toString());
        } finally {
            if (timeoutTimer != null)
                timeoutTimer.cancel();

            // close the input stream
            try {
                outputReader.close();
            } catch (IOException ioe) {
                LOG.warn("Error while closing the input stream", ioe);
            }
            if (!finished.get())
                errorThread.interrupt();

            try {
                errorReader.close();
            } catch (IOException ioe) {
                LOG.warn("Error while closing the error stream", ioe);
            }

            process.destroy();
        }
    }

public static String unescapeCustomCode(final String inputText) {

        if (inputText == null) {
            return null;
        }

        StringBuilder customBuilder = null;

        final int startOffset = 0;
        final int maxLimit = inputText.length();

        int currentOffset = startOffset;
        int referencePoint = startOffset;

        for (int index = startOffset; index < maxLimit; index++) {

            final char character = inputText.charAt(index);

            /*
             * Check the need for an unescape operation at this point
             */

            if (character != CUSTOM_ESCAPE_PREFIX || (index + 1) >= maxLimit) {
                continue;
            }

            int codeValue = -1;

            if (character == CUSTOM_ESCAPE_PREFIX) {

                final char nextChar = inputText.charAt(index + 1);

                if (nextChar == CUSTOM_ESCAPE_UHEXA_PREFIX2) {
                    // This can be a uhexa escape, we need exactly four more characters

                    int startHexIndex = index + 2;
                    // First, discard any additional 'u' characters, which are allowed
                    while (startHexIndex < maxLimit) {
                        final char cf = inputText.charAt(startHexIndex);
                        if (cf != CUSTOM_ESCAPE_UHEXA_PREFIX2) {
                            break;
                        }
                        startHexIndex++;
                    }
                    int hexStart = startHexIndex;
                    // Parse the hexadecimal digits
                    while (hexStart < (index + 5) && hexStart < maxLimit) {
                        final char cf = inputText.charAt(hexStart);
                        if (!((cf >= '0' && cf <= '9') || (cf >= 'A' && cf <= 'F') || (cf >= 'a' && cf <= 'f'))) {
                            break;
                        }
                        hexStart++;
                    }

                    if ((hexStart - index) < 5) {
                        // We weren't able to consume the required four hexa chars, leave it as slash+'u', which
                        // is invalid, and let the corresponding Java parser fail.
                        index++;
                        continue;
                    }

                    codeValue = parseIntFromReference(inputText, index + 2, hexStart, 16);

                    // Fast-forward to the first char after the parsed codepoint
                    referencePoint = hexStart - 1;

                    // Don't continue here, just let the unescape code below do its job

                } else if (nextChar == CUSTOM_ESCAPE_PREFIX && index + 2 < maxLimit && inputText.charAt(index + 2) == CUSTOM_ESCAPE_UHEXA_PREFIX2){
                    // This unicode escape is actually escaped itself, so we don't need to perform the real unescaping,
                    // but we need to merge the "\\" into "\"

                    if (customBuilder == null) {
                        customBuilder = new StringBuilder(maxLimit + 5);
                    }

                    if (index - currentOffset > 0) {
                        customBuilder.append(inputText, currentOffset, index);
                    }

                    customBuilder.append('\\');

                    currentOffset = index + 3;

                    index++;
                    continue;

                } else {

                    // Other escape sequences will not be processed in this unescape step.
                    index++;
                    continue;

                }

            }


            /*
             * At this point we know for sure we will need some kind of unescape, so we
             * can increase the offset and initialize the string builder if needed, along with
             * copying to it all the contents pending up to this point.
             */

            if (customBuilder == null) {
                customBuilder = new StringBuilder(maxLimit + 5);
            }

            if (index - currentOffset > 0) {
                customBuilder.append(inputText, currentOffset, index);
            }

            index = referencePoint;
            currentOffset = index + 1;

            /*
             * --------------------------
             *
             * Peform the real unescape
             *
             * --------------------------
             */

            if (codeValue > '\uFFFF') {
                customBuilder.append(Character.toChars(codeValue));
            } else {
                customBuilder.append((char)codeValue);
            }

        }


        /*
         * -----------------------------------------------------------------------------------------------
         * Final cleaning: return the original String object if no unescape was actually needed. Otherwise
         *                 append the remaining escaped text to the string builder and return.
         * -----------------------------------------------------------------------------------------------
         */

        if (customBuilder == null) {
            return inputText;
        }

        if (maxLimit - currentOffset > 0) {
            customBuilder.append(inputText, currentOffset, maxLimit);
        }

        return customBuilder.toString();

    }

  @XmlElement(name = "idprefix")
protected void setUsageId(Long businessUsageId) {
    maybeInitBuilder();
    if (businessUsageId == null) {
      builder.clearUsageId();
      return;
    }
    builder.setUsageId(businessUsageId);
  }

  /**
   * Sets idPrefix for an entity.
   * <p>
   * <b>Note</b>: Entities will be stored in the order of idPrefix specified.
   * If users decide to set idPrefix for an entity, they <b>MUST</b> provide
   * the same prefix for every update of this entity.
   * </p>
   * Example: <blockquote><pre>
   * TimelineEntity entity = new TimelineEntity();
   * entity.setIdPrefix(value);
   * </pre></blockquote>
   * Users can use {@link TimelineServiceHelper#invertLong(long)} to invert
   * the prefix if necessary.
   *
   * @param entityIdPrefix prefix for an entity.
   */
  @JsonSetter("idprefix")
public void finalizeJournal() throws IOException {
    boolean needsErrorHandling = true;
    mapJournalsAndReportErrors(journal -> {
        if (needsErrorHandling) {
            try {
                journal.close();
            } catch (IOException e) {
                // handle error
            }
        }
    }, "close journal");
    closed = true;
}
}
