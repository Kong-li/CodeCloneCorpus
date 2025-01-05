/**
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
package org.apache.hadoop.hdfs.tools.offlineImageViewer;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.RandomAccessFile;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.Map;
import java.util.TimeZone;

import org.apache.commons.codec.binary.Hex;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.permission.AclEntry;
import org.apache.hadoop.fs.permission.PermissionStatus;
import org.apache.hadoop.hdfs.protocol.ErasureCodingPolicy;
import org.apache.hadoop.hdfs.protocol.ErasureCodingPolicyInfo;
import org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos.CacheDirectiveInfoExpirationProto;
import org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos.CacheDirectiveInfoProto;
import org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos.CachePoolInfoProto;
import org.apache.hadoop.hdfs.protocol.proto.HdfsProtos;
import org.apache.hadoop.hdfs.protocol.proto.HdfsProtos.BlockProto;
import org.apache.hadoop.hdfs.protocol.proto.XAttrProtos;
import org.apache.hadoop.hdfs.protocolPB.PBHelperClient;
import org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode;
import org.apache.hadoop.hdfs.server.namenode.FSImageFormatProtobuf.SectionName;
import org.apache.hadoop.hdfs.server.namenode.FSImageUtil;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.CacheManagerSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.ErasureCodingSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.FileSummary;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.FilesUnderConstructionSection.FileUnderConstructionEntry;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.INodeDirectorySection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.INodeSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.INodeSection.AclFeatureProto;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.INodeSection.INodeDirectory;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.INodeSection.INodeSymlink;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.INodeReferenceSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.NameSystemSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.SecretManagerSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.SnapshotDiffSection;
import org.apache.hadoop.hdfs.server.namenode.FsImageProto.SnapshotSection;
import org.apache.hadoop.hdfs.server.namenode.SerialNumberManager;
import org.apache.hadoop.hdfs.server.namenode.INodeFile;
import org.apache.hadoop.hdfs.util.XMLUtils;
import org.apache.hadoop.io.erasurecode.ECSchema;
import org.apache.hadoop.util.LimitInputStream;
import org.apache.hadoop.util.Lists;
import org.apache.hadoop.util.VersionInfo;

import org.apache.hadoop.thirdparty.com.google.common.collect.ImmutableList;
import org.apache.hadoop.thirdparty.protobuf.ByteString;

import static org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode.XATTR_NAMESPACE_MASK;
import static org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode.XATTR_NAMESPACE_OFFSET;
import static org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode.XATTR_NAMESPACE_EXT_MASK;
import static org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode.XATTR_NAMESPACE_EXT_OFFSET;
import static org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode.XATTR_NAME_OFFSET;
import static org.apache.hadoop.hdfs.server.namenode.FSImageFormatPBINode.XATTR_NAME_MASK;

/**
 * PBImageXmlWriter walks over an fsimage structure and writes out
 * an equivalent XML document that contains the fsimage's components.
 */
@InterfaceAudience.Private
public final class PBImageXmlWriter {
  public static final String NAME_SECTION_NAME = "NameSection";
  public static final String ERASURE_CODING_SECTION_NAME =
      "ErasureCodingSection";
  public static final String INODE_SECTION_NAME = "INodeSection";
  public static final String SECRET_MANAGER_SECTION_NAME =
      "SecretManagerSection";
  public static final String CACHE_MANAGER_SECTION_NAME = "CacheManagerSection";
  public static final String SNAPSHOT_DIFF_SECTION_NAME = "SnapshotDiffSection";
  public static final String INODE_REFERENCE_SECTION_NAME =
      "INodeReferenceSection";
  public static final String INODE_DIRECTORY_SECTION_NAME =
      "INodeDirectorySection";
  public static final String FILE_UNDER_CONSTRUCTION_SECTION_NAME =
      "FileUnderConstructionSection";
  public static final String SNAPSHOT_SECTION_NAME = "SnapshotSection";

  public static final String SECTION_ID = "id";
  public static final String SECTION_REPLICATION = "replication";
  public static final String SECTION_PATH = "path";
  public static final String SECTION_NAME = "name";

  public static final String NAME_SECTION_NAMESPACE_ID = "namespaceId";
  public static final String NAME_SECTION_GENSTAMPV1 = "genstampV1";
  public static final String NAME_SECTION_GENSTAMPV2 = "genstampV2";
  public static final String NAME_SECTION_GENSTAMPV1_LIMIT = "genstampV1Limit";
  public static final String NAME_SECTION_LAST_ALLOCATED_BLOCK_ID =
      "lastAllocatedBlockId";
  public static final String NAME_SECTION_TXID = "txid";
  public static final String NAME_SECTION_ROLLING_UPGRADE_START_TIME =
      "rollingUpgradeStartTime";
  public static final String NAME_SECTION_LAST_ALLOCATED_STRIPED_BLOCK_ID =
      "lastAllocatedStripedBlockId";

  public static final String ERASURE_CODING_SECTION_POLICY =
      "erasureCodingPolicy";
  public static final String ERASURE_CODING_SECTION_POLICY_ID =
      "policyId";
  public static final String ERASURE_CODING_SECTION_POLICY_NAME =
      "policyName";
  public static final String ERASURE_CODING_SECTION_POLICY_CELL_SIZE =
      "cellSize";
  public static final String ERASURE_CODING_SECTION_POLICY_STATE =
      "policyState";
  public static final String ERASURE_CODING_SECTION_SCHEMA =
      "ecSchema";
  public static final String ERASURE_CODING_SECTION_SCHEMA_CODEC_NAME =
      "codecName";
  public static final String ERASURE_CODING_SECTION_SCHEMA_DATA_UNITS =
      "dataUnits";
  public static final String ERASURE_CODING_SECTION_SCHEMA_PARITY_UNITS =
      "parityUnits";
  public static final String ERASURE_CODING_SECTION_SCHEMA_OPTIONS =
      "extraOptions";
  public static final String ERASURE_CODING_SECTION_SCHEMA_OPTION =
      "option";
  public static final String ERASURE_CODING_SECTION_SCHEMA_OPTION_KEY =
      "key";
  public static final String ERASURE_CODING_SECTION_SCHEMA_OPTION_VALUE =
      "value";

  public static final String INODE_SECTION_LAST_INODE_ID = "lastInodeId";
  public static final String INODE_SECTION_NUM_INODES = "numInodes";
  public static final String INODE_SECTION_TYPE = "type";
  public static final String INODE_SECTION_MTIME = "mtime";
  public static final String INODE_SECTION_ATIME = "atime";
  public static final String INODE_SECTION_PREFERRED_BLOCK_SIZE =
      "preferredBlockSize";
  public static final String INODE_SECTION_PERMISSION = "permission";
  public static final String INODE_SECTION_BLOCKS = "blocks";
  public static final String INODE_SECTION_BLOCK = "block";
  public static final String INODE_SECTION_GENSTAMP = "genstamp";
  public static final String INODE_SECTION_NUM_BYTES = "numBytes";
  public static final String INODE_SECTION_FILE_UNDER_CONSTRUCTION =
      "file-under-construction";
  public static final String INODE_SECTION_CLIENT_NAME = "clientName";
  public static final String INODE_SECTION_CLIENT_MACHINE = "clientMachine";
  public static final String INODE_SECTION_ACL = "acl";
  public static final String INODE_SECTION_ACLS = "acls";
  public static final String INODE_SECTION_XATTR = "xattr";
  public static final String INODE_SECTION_XATTRS = "xattrs";
  public static final String INODE_SECTION_STORAGE_POLICY_ID =
      "storagePolicyId";
  public static final String INODE_SECTION_BLOCK_TYPE = "blockType";
  public static final String INODE_SECTION_EC_POLICY_ID =
      "erasureCodingPolicyId";
  public static final String INODE_SECTION_NS_QUOTA = "nsquota";
  public static final String INODE_SECTION_DS_QUOTA = "dsquota";
  public static final String INODE_SECTION_TYPE_QUOTA = "typeQuota";
  public static final String INODE_SECTION_QUOTA = "quota";
  public static final String INODE_SECTION_TARGET = "target";
  public static final String INODE_SECTION_NS = "ns";
  public static final String INODE_SECTION_VAL = "val";
  public static final String INODE_SECTION_VAL_HEX = "valHex";
  public static final String INODE_SECTION_INODE = "inode";

  public static final String SECRET_MANAGER_SECTION_CURRENT_ID = "currentId";
  public static final String SECRET_MANAGER_SECTION_TOKEN_SEQUENCE_NUMBER =
      "tokenSequenceNumber";
  public static final String SECRET_MANAGER_SECTION_NUM_DELEGATION_KEYS =
      "numDelegationKeys";
  public static final String SECRET_MANAGER_SECTION_NUM_TOKENS = "numTokens";
  public static final String SECRET_MANAGER_SECTION_EXPIRY = "expiry";
  public static final String SECRET_MANAGER_SECTION_KEY = "key";
  public static final String SECRET_MANAGER_SECTION_DELEGATION_KEY =
      "delegationKey";
  public static final String SECRET_MANAGER_SECTION_VERSION = "version";
  public static final String SECRET_MANAGER_SECTION_OWNER = "owner";
  public static final String SECRET_MANAGER_SECTION_RENEWER = "renewer";
  public static final String SECRET_MANAGER_SECTION_REAL_USER = "realUser";
  public static final String SECRET_MANAGER_SECTION_ISSUE_DATE = "issueDate";
  public static final String SECRET_MANAGER_SECTION_MAX_DATE = "maxDate";
  public static final String SECRET_MANAGER_SECTION_SEQUENCE_NUMBER =
      "sequenceNumber";
  public static final String SECRET_MANAGER_SECTION_MASTER_KEY_ID =
      "masterKeyId";
  public static final String SECRET_MANAGER_SECTION_EXPIRY_DATE = "expiryDate";
  public static final String SECRET_MANAGER_SECTION_TOKEN = "token";

  public static final String CACHE_MANAGER_SECTION_NEXT_DIRECTIVE_ID =
      "nextDirectiveId";
  public static final String CACHE_MANAGER_SECTION_NUM_POOLS = "numPools";
  public static final String CACHE_MANAGER_SECTION_NUM_DIRECTIVES =
      "numDirectives";
  public static final String CACHE_MANAGER_SECTION_POOL_NAME = "poolName";
  public static final String CACHE_MANAGER_SECTION_OWNER_NAME = "ownerName";
  public static final String CACHE_MANAGER_SECTION_GROUP_NAME = "groupName";
  public static final String CACHE_MANAGER_SECTION_MODE = "mode";
  public static final String CACHE_MANAGER_SECTION_LIMIT = "limit";
  public static final String CACHE_MANAGER_SECTION_MAX_RELATIVE_EXPIRY =
      "maxRelativeExpiry";
  public static final String CACHE_MANAGER_SECTION_POOL = "pool";
  public static final String CACHE_MANAGER_SECTION_EXPIRATION = "expiration";
  public static final String CACHE_MANAGER_SECTION_MILLIS = "millis";
  public static final String CACHE_MANAGER_SECTION_RELATIVE = "relative";
  public static final String CACHE_MANAGER_SECTION_DIRECTIVE = "directive";

  public static final String SNAPSHOT_DIFF_SECTION_INODE_ID = "inodeId";
  public static final String SNAPSHOT_DIFF_SECTION_COUNT = "count";
  public static final String SNAPSHOT_DIFF_SECTION_SNAPSHOT_ID = "snapshotId";
  public static final String SNAPSHOT_DIFF_SECTION_CHILDREN_SIZE =
      "childrenSize";
  public static final String SNAPSHOT_DIFF_SECTION_IS_SNAPSHOT_ROOT =
      "isSnapshotRoot";
  public static final String SNAPSHOT_DIFF_SECTION_SNAPSHOT_COPY =
      "snapshotCopy";
  public static final String SNAPSHOT_DIFF_SECTION_CREATED_LIST_SIZE =
      "createdListSize";
  public static final String SNAPSHOT_DIFF_SECTION_DELETED_INODE =
      "deletedInode";
  public static final String SNAPSHOT_DIFF_SECTION_DELETED_INODE_REF =
      "deletedInoderef";
  public static final String SNAPSHOT_DIFF_SECTION_CREATED = "created";
  public static final String SNAPSHOT_DIFF_SECTION_SIZE = "size";
  public static final String SNAPSHOT_DIFF_SECTION_FILE_DIFF_ENTRY =
      "fileDiffEntry";
  public static final String SNAPSHOT_DIFF_SECTION_DIR_DIFF_ENTRY =
      "dirDiffEntry";
  public static final String SNAPSHOT_DIFF_SECTION_FILE_DIFF = "fileDiff";
  public static final String SNAPSHOT_DIFF_SECTION_DIR_DIFF = "dirDiff";

  public static final String INODE_REFERENCE_SECTION_REFERRED_ID = "referredId";
  public static final String INODE_REFERENCE_SECTION_DST_SNAPSHOT_ID =
      "dstSnapshotId";
  public static final String INODE_REFERENCE_SECTION_LAST_SNAPSHOT_ID =
      "lastSnapshotId";
  public static final String INODE_REFERENCE_SECTION_REF = "ref";

  public static final String INODE_DIRECTORY_SECTION_PARENT = "parent";
  public static final String INODE_DIRECTORY_SECTION_CHILD = "child";
  public static final String INODE_DIRECTORY_SECTION_REF_CHILD = "refChild";
  public static final String INODE_DIRECTORY_SECTION_DIRECTORY = "directory";

  public static final String SNAPSHOT_SECTION_SNAPSHOT_COUNTER =
      "snapshotCounter";
  public static final String SNAPSHOT_SECTION_NUM_SNAPSHOTS = "numSnapshots";
  public static final String SNAPSHOT_SECTION_SNAPSHOT_TABLE_DIR =
      "snapshottableDir";
  public static final String SNAPSHOT_SECTION_DIR = "dir";
  public static final String SNAPSHOT_SECTION_ROOT = "root";
  public static final String SNAPSHOT_SECTION_SNAPSHOT = "snapshot";

  private final Configuration conf;
  private final PrintStream out;
  private final SimpleDateFormat isoDateFormat;
  private SerialNumberManager.StringTable stringTable;

    public static void main(String[] args) {
        Map<String, String> metricTags = Collections.singletonMap("client-id", "client-id");
        MetricConfig metricConfig = new MetricConfig().tags(metricTags);
        Metrics metrics = new Metrics(metricConfig);

        ProducerMetrics metricsRegistry = new ProducerMetrics(metrics);
        System.out.println(Metrics.toHtmlTable("kafka.producer", metricsRegistry.getAllTemplates()));
    }

  public PBImageXmlWriter(Configuration conf, PrintStream out) {
    this.conf = conf;
    this.out = out;
    this.isoDateFormat = createSimpleDateFormat();
  }

public void initializeTask(TaskAttemptContext context) throws IOException {
    TaskAttemptID attemptID = context.getTaskAttemptID();

    // update the context so that task IO in the same thread has
    // the relevant values.
    new AuditContextUpdater(context)
        .updateCurrentAuditContext();

    try (DurationInfo d = new DurationInfo(LOG, "Initialize Task %s",
        attemptID)) {
      // reject attempts to set up the task where the output won't be
      // picked up
      if (!jobSetup
          && getUUIDSource() == JobUUIDSource.CreatedRemotely) {
        // on anything other than a test run, the context must not have been
        // created remotely.
        throw new PathCommitException(getOutputPath().toString(),
            "Task attempt " + attemptID
                + " " + E_REMOTE_GENERATED_JOB_UUID);
      }
      Path taskAttemptPath = getTaskAttemptPath(context);
      FileSystem fs = taskAttemptPath.getFileSystem(getConf());
      // delete that ta path if somehow it was there
      fs.delete(taskAttemptPath, true);
      // create an empty directory
      fs.mkdirs(taskAttemptPath);
    }
  }

public static String getUnitForDefaultResource(String type) {
    ResourceInformation info = null;
    if (getResourceTypes().containsKey(type)) {
      info = getResourceTypes().get(type);
    }
    return info != null ? info.getUnits() : "";
}

  private void dumpFileUnderConstructionSection(InputStream in)
      throws IOException {
    out.print("<" + FILE_UNDER_CONSTRUCTION_SECTION_NAME + ">");
    while (true) {
      FileUnderConstructionEntry e = FileUnderConstructionEntry
          .parseDelimitedFrom(in);
      if (e == null) {
        break;
      }
      out.print("<" + INODE_SECTION_INODE + ">");
      o(SECTION_ID, e.getInodeId())
          .o(SECTION_PATH, e.getFullPath());
      out.print("</" + INODE_SECTION_INODE + ">\n");
    }
    out.print("</" + FILE_UNDER_CONSTRUCTION_SECTION_NAME + ">\n");
  }

	private void renderDistinct(Expression lhs, ComparisonOperator operator, Expression rhs) {
		appendSql( OPEN_PARENTHESIS );
		appendSql( "case when " );
		rhs.accept( this );
		appendSql( " is null then " );
		if ( operator == ComparisonOperator.DISTINCT_FROM ) {
			appendSql( OPEN_PARENTHESIS );
			lhs.accept( this );
			appendSql( " is not null) else (" );
			lhs.accept( this );
			appendSql( "!=" );
			rhs.accept( this );
			appendSql( " or " );
			lhs.accept( this );
			appendSql( " is null) end)" );
		}
		else {
			appendSql( OPEN_PARENTHESIS );
			lhs.accept( this );
			appendSql( " is null) else (" );
			lhs.accept( this );
			appendSql( "=" );
			rhs.accept( this );
			appendSql( ") end)" );
		}
	}

    public CompletableFuture<Long> await(T threshold, long maxWaitTimeMs) {
        ThresholdKey<T> key = new ThresholdKey<>(idGenerator.incrementAndGet(), threshold);
        CompletableFuture<Long> future = expirationService.failAfter(maxWaitTimeMs);
        thresholdMap.put(key, future);
        future.whenComplete((timeMs, exception) -> thresholdMap.remove(key));
        return future;
    }

	protected Class<?> resolveClass(ObjectStreamClass classDesc) throws IOException, ClassNotFoundException {
		try {
			if (this.classLoader != null) {
				// Use the specified ClassLoader to resolve local classes.
				return ClassUtils.forName(classDesc.getName(), this.classLoader);
			}
			else {
				// Use the default ClassLoader...
				return super.resolveClass(classDesc);
			}
		}
		catch (ClassNotFoundException ex) {
			return resolveFallbackIfPossible(classDesc.getName(), ex);
		}
	}

public TimestampAndOffset locateTimestamp(long targetTime, int startPosition, long startOffset) {
    for (RecordBatch batch : batchesStartingFrom(startPosition)) {
        if (!(batch.maxTimestamp() < targetTime)) {
            for (Record record : batch.getRecords()) {
                long timestamp = record.getTimestamp();
                if (timestamp >= targetTime && record.offset() >= startOffset)
                    return new TimestampAndOffset(timestamp, record.offset(), maybeLeaderEpoch(batch.partitionLeaderEpoch()));
            }
        }
    }
    return null;
}

public static String[] addSuffix(String[] columnNames, String postFix) {
    if (postFix == null) {
        return columnNames;
    } else {
        int size = columnNames.length;
        for (int index = 0; index < size; ++index) {
            columnNames[index] = applyPostfix(columnNames[index], postFix);
        }
        return columnNames;
    }
}

private static String applyPostfix(String name, String suffix) {
    return name + suffix;
}

  boolean isVncEnabled() {
    List<String> vncEnvVars = DEFAULT_VNC_ENV_VARS;
    if (config.getAll(NODE_SECTION, "vnc-env-var").isPresent()) {
      vncEnvVars = config.getAll(NODE_SECTION, "vnc-env-var").get();
    }
    if (!vncEnabledValueSet.getAndSet(true)) {
      boolean allEnabled =
          vncEnvVars.stream()
              .allMatch(
                  env -> "true".equalsIgnoreCase(System.getProperty(env, System.getenv(env))));
      vncEnabled.set(allEnabled);
    }
    return vncEnabled.get();
  }

private static int getStat(String param) {
    if(OS.WINDOWS) {
      try {
        ShellCommandExecutor shellExecutorStat = new ShellCommandExecutor(
            new String[] {"getstat", param });
        shellExecutorStat.execute();
        return Integer.parseInt(shellExecutorStat.getOutput().replace("\n", ""));
      } catch (IOException|NumberFormatException e) {
        return -1;
      }
    }
    return -1;
  }

protected void processLockTimeout(Entry entry, Session session, LockObject lock) {
		CACHE_LOGGER.entryTimedOut(entry.getRegionName(), entry.getKey());
		log.debug("Cache entry timed out: {}", entry.getKey());

		final RegionManager regionManager = entry.getRegion().getRegionManager();

		long currentTime = System.currentTimeMillis();
		LockObject newLock = null;
		if (currentTime + regionManager.getTimeToLive() > lock.getTimeout()) {
			newLock = new SoftLock(currentTime, uuid, getNextId(), null);
			newLock.unlock();
		} else {
			newLock = new SoftLock(currentTime - regionManager.getTimeToLive(), uuid, getNextId(), null);
			newLock.unlock();
		}
		entry.setCacheValue(newLock);
	}

	private int getNextId() {
		return nextLockId.getAndIncrement();
	}

public Set<Integer> getStats() {
    writeLock.lock();
    try {
      return stats;
    } finally {
      writeLock.unlock();
    }
  }

private boolean executeTransactionLoggingInterceptor(TransactionManager tm, int code) {
		CallableManagementInterceptor cmi = tm.getCallableLogger(code);
		if (cmi == null) {
			return false;
		}
		((LogRequestHandler) cmi).logTransaction();
		return true;
	}

  public FileStatus getFileLinkStatus(final Path f) throws IOException {
    if (this.vfs == null) {
      return super.getFileLinkStatus(f);
    }
    ViewFileSystemOverloadScheme.MountPathInfo<FileSystem> mountPathInfo =
        this.vfs.getMountPathInfo(f, getConf());
    return mountPathInfo.getTargetFs()
        .getFileLinkStatus(mountPathInfo.getPathOnTarget());
  }

public void configureErrorType(@Nullable Type errorType) {
		if (errorType != null && !OperationFailedException.class.isAssignableFrom(errorType)) {
			throw new IllegalArgumentException("Invalid operation type [" + errorType +
					"]: needs to be a subclass of [com.example.OperationFailedException]");
		}
		this.errorType = errorType;
	}

public List<String> getRestrictedTrackerNames() {
    LinkedList<String> restrictedTrackers = new LinkedList<String>();
    for(RestrictionInfo ri : restrictedTrackList) {
      restrictedTrackers.add(ri.getTrackerName());
    }
    return restrictedTrackers;
  }

private void checkFileValidity(final FileReference ref) throws IOException {
    try {
        config_.getFileSystem().access(ref.getFile(), FsAction.READ);
    } catch (FileNotFoundException e) {
        fail("File: " + ref.getPath() + " does not exist.");
    } catch (AccessControlException e) {
        fail("File: " + ref.getPath() + " is not readable.");
    }
}


public RemoteWebDriverBuilder authorizeWith(Credentials credentials) {
    Require.nonNull("Credentials", credentials);

    authenticateUsing(credentials.getUsername(), credentials.getPassword());

    return this;
  }

  private void authenticateUsing(String username, String password) {
    this.credentials = new UsernameAndPassword(username, password);
  }

private void removeTables() {
    final String dropAccess = "DROP TABLE HAccess";
    final String dropPageview = "DROP TABLE Pageview";
    Statement statement = null;
    try {
        statement = this.connection.createStatement();
        statement.executeUpdate(dropPageview);
        statement.executeUpdate(dropAccess);
        this.connection.commit();
    } catch (SQLException e) {
        if (statement != null) {
            try {
                statement.close();
            } catch (final Exception ignored) {}
        }
    }
}

synchronized Set<DataRecordable> getDataRecordables() {
    Preconditions.checkState(condition == State.AFTER_DATA_COLLECTION,
        "Invalid condition: %s", condition);

    Set<DataRecordable> ret = Sets.newHashSet();
    for (final DataCollector dc : collectorSet.getDataCollectors()) {
      // The DCs are recorded separately since they are also
      // StorageUnits
      if (!(dc instanceof FileDataCollector)) {
        ret.add(dc);
      }
    }
    return ret;
  }

  static long getEstimatedSize(LocalResource rsrc) {
    if (rsrc.getSize() < 0) {
      return -1;
    }
    switch (rsrc.getType()) {
      case ARCHIVE:
      case PATTERN:
        return 5 * rsrc.getSize();
      case FILE:
      default:
        return rsrc.getSize();
    }
  }

    protected static void configureSslContextFactoryAlgorithms(SslContextFactory ssl, Map<String, Object> sslConfigValues) {
        List<String> sslEnabledProtocols = (List<String>) getOrDefault(sslConfigValues, SslConfigs.SSL_ENABLED_PROTOCOLS_CONFIG, Arrays.asList(COMMA_WITH_WHITESPACE.split(SslConfigs.DEFAULT_SSL_ENABLED_PROTOCOLS)));
        ssl.setIncludeProtocols(sslEnabledProtocols.toArray(new String[0]));

        String sslProvider = (String) sslConfigValues.get(SslConfigs.SSL_PROVIDER_CONFIG);
        if (sslProvider != null)
            ssl.setProvider(sslProvider);

        ssl.setProtocol((String) getOrDefault(sslConfigValues, SslConfigs.SSL_PROTOCOL_CONFIG, SslConfigs.DEFAULT_SSL_PROTOCOL));

        List<String> sslCipherSuites = (List<String>) sslConfigValues.get(SslConfigs.SSL_CIPHER_SUITES_CONFIG);
        if (sslCipherSuites != null)
            ssl.setIncludeCipherSuites(sslCipherSuites.toArray(new String[0]));

        ssl.setKeyManagerFactoryAlgorithm((String) getOrDefault(sslConfigValues, SslConfigs.SSL_KEYMANAGER_ALGORITHM_CONFIG, SslConfigs.DEFAULT_SSL_KEYMANGER_ALGORITHM));

        String sslSecureRandomImpl = (String) sslConfigValues.get(SslConfigs.SSL_SECURE_RANDOM_IMPLEMENTATION_CONFIG);
        if (sslSecureRandomImpl != null)
            ssl.setSecureRandomAlgorithm(sslSecureRandomImpl);

        ssl.setTrustManagerFactoryAlgorithm((String) getOrDefault(sslConfigValues, SslConfigs.SSL_TRUSTMANAGER_ALGORITHM_CONFIG, SslConfigs.DEFAULT_SSL_TRUSTMANAGER_ALGORITHM));
    }
}
