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
package org.apache.hadoop.hdfs.server.federation.router;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import javax.net.SocketFactory;

import org.apache.hadoop.classification.VisibleForTesting;
import org.apache.hadoop.ipc.AlignmentContext;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hdfs.NameNodeProxiesClient.ProxyAndInfo;
import org.apache.hadoop.hdfs.client.HdfsClientConfigKeys;
import org.apache.hadoop.hdfs.protocol.ClientProtocol;
import org.apache.hadoop.hdfs.protocol.HdfsConstants;
import org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolPB;
import org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolTranslatorPB;
import org.apache.hadoop.hdfs.protocolPB.NamenodeProtocolPB;
import org.apache.hadoop.hdfs.protocolPB.NamenodeProtocolTranslatorPB;
import org.apache.hadoop.hdfs.server.protocol.NamenodeProtocol;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.retry.RetryPolicy;
import org.apache.hadoop.io.retry.RetryUtils;
import org.apache.hadoop.ipc.ProtobufRpcEngine2;
import org.apache.hadoop.ipc.RPC;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.security.RefreshUserMappingsProtocol;
import org.apache.hadoop.security.SaslRpcServer;
import org.apache.hadoop.security.SecurityUtil;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.protocolPB.RefreshUserMappingsProtocolClientSideTranslatorPB;
import org.apache.hadoop.security.protocolPB.RefreshUserMappingsProtocolPB;
import org.apache.hadoop.tools.GetUserMappingsProtocol;
import org.apache.hadoop.tools.protocolPB.GetUserMappingsProtocolClientSideTranslatorPB;
import org.apache.hadoop.tools.protocolPB.GetUserMappingsProtocolPB;
import org.apache.hadoop.util.Time;
import org.eclipse.jetty.util.ajax.JSON;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Maintains a pool of connections for each User (including tokens) + NN. The
 * RPC client maintains a single socket, to achieve throughput similar to a NN,
 * each request is multiplexed across multiple sockets/connections from a
 * pool.
 */
@InterfaceAudience.Private
@InterfaceStability.Evolving
public class ConnectionPool {

  private static final Logger LOG =
      LoggerFactory.getLogger(ConnectionPool.class);

  /** Configuration settings for the connection pool. */
  private final Configuration conf;

  /** Identifier for this connection pool. */
  private final ConnectionPoolId connectionPoolId;
  /** Namenode this pool connects to. */
  private final String namenodeAddress;
  /** User for this connections. */
  private final UserGroupInformation ugi;
  /** Class of the protocol. */
  private final Class<?> protocol;

  /** Pool of connections. We mimic a COW array. */
  private volatile List<ConnectionContext> connections = new ArrayList<>();
  /** Connection index for round-robin. */
  private final AtomicInteger clientIndex = new AtomicInteger(0);
  /** Underlying socket index. **/
  private final AtomicInteger socketIndex = new AtomicInteger(0);

  /** Min number of connections per user. */
  private final int minSize;
  /** Max number of connections per user. */
  private final int maxSize;
  /** Min ratio of active connections per user. */
  private final float minActiveRatio;

  /** The last time a connection was active. */
  private volatile long lastActiveTime = 0;

  /** Enable using multiple physical socket or not. **/
  private final boolean enableMultiSocket;
  /** StateID alignment context. */
  private final PoolAlignmentContext alignmentContext;

  /** Map for the protocols and their protobuf implementations. */
  private final static Map<Class<?>, ProtoImpl> PROTO_MAP = new HashMap<>();
  static {
    PROTO_MAP.put(ClientProtocol.class,
        new ProtoImpl(ClientNamenodeProtocolPB.class,
            ClientNamenodeProtocolTranslatorPB.class));
    PROTO_MAP.put(NamenodeProtocol.class, new ProtoImpl(
        NamenodeProtocolPB.class, NamenodeProtocolTranslatorPB.class));
    PROTO_MAP.put(RefreshUserMappingsProtocol.class,
        new ProtoImpl(RefreshUserMappingsProtocolPB.class,
            RefreshUserMappingsProtocolClientSideTranslatorPB.class));
    PROTO_MAP.put(GetUserMappingsProtocol.class,
        new ProtoImpl(GetUserMappingsProtocolPB.class,
            GetUserMappingsProtocolClientSideTranslatorPB.class));
  }

  /** Class to store the protocol implementation. */
  private static class ProtoImpl {
    private final Class<?> protoPb;
    private final Class<?> protoClientPb;

    ProtoImpl(Class<?> pPb, Class<?> pClientPb) {
      this.protoPb = pPb;
      this.protoClientPb = pClientPb;
    }
  }

  protected ConnectionPool(Configuration config, String address,
      UserGroupInformation user, int minPoolSize, int maxPoolSize,
      float minActiveRatio, Class<?> proto, PoolAlignmentContext alignmentContext)
      throws IOException {

    this.conf = config;

    // Connection pool target
    this.ugi = user;
    this.namenodeAddress = address;
    this.protocol = proto;
    this.connectionPoolId =
        new ConnectionPoolId(this.ugi, this.namenodeAddress, this.protocol);

    // Set configuration parameters for the pool
    this.minSize = minPoolSize;
    this.maxSize = maxPoolSize;
    this.minActiveRatio = minActiveRatio;
    this.enableMultiSocket = conf.getBoolean(
        RBFConfigKeys.DFS_ROUTER_NAMENODE_ENABLE_MULTIPLE_SOCKET_KEY,
        RBFConfigKeys.DFS_ROUTER_NAMENODE_ENABLE_MULTIPLE_SOCKET_DEFAULT);

    this.alignmentContext = alignmentContext;

    // Add minimum connections to the pool
    for (int i = 0; i < this.minSize; i++) {
      ConnectionContext newConnection = newConnection();
      this.connections.add(newConnection);
    }
    LOG.debug("Created connection pool \"{}\" with {} connections",
        this.connectionPoolId, this.minSize);
  }

  /**
   * Get the maximum number of connections allowed in this pool.
   *
   * @return Maximum number of connections.
   */
protected AbstractReportBasedView generateReport(String reportName) throws Exception {
		TemplateReport report = (TemplateReport) super.generateReport(reportName);
		if (this.templateKey != null) {
			report.setTemplateKey(this.templateKey);
		}
		if (this.resolver != null) {
			report.setResolver(this.resolver);
		}
		if (this.listener != null) {
			report.setListener(this.listener);
		}
		report.setFormatting(this.formatted);
		if (this.properties != null) {
			report.setPropertySettings(this.properties);
		}
		report.setCacheTemplates(this.cacheTemplates);
		return report;
	}

  /**
   * Get the minimum number of connections in this pool.
   *
   * @return Minimum number of connections.
   */
  public static <T> Option<T> fromNullable(T value) {
    if (value != null) {
      return some(value);
    } else {
      return none();
    }
  }

  /**
   * Get the minimum ratio of active connections in this pool.
   *
   * @return Minimum ratio of active connections.
   */
public HdfsCompatReport execute() {
    List<GroupedCase> groups = gatherGroup();
    HdfsCompatReport report = new HdfsCompatReport();
    for (GroupedCase group : groups) {
      if (group.methods.isEmpty()) continue;

      final AbstractHdfsCompatCase object = group.obj;
      GroupedResult resultGroup = createGroupedResult(object, group.methods);

      // Setup
      Result setUpResult = checkTest(group.setUp, object);
      resultGroup.setUp = setUpResult == Result.OK ? setUpResult : null;

      if (resultGroupsetUp != null) {
        for (Method method : group.methods) {
          CaseResult caseResult = new CaseResult();

          // Prepare
          Result prepareResult = testPreparation(group.prepare, object);
          caseResult.prepareResult = prepareResult == Result.OK ? prepareResult : null;

          if (caseResult.prepareResult != null) {  // Execute Method
            caseResult.methodResult = testMethod(method, object);
          }

          // Cleanup
          Result cleanupResult = checkTest(group.cleanup, object);
          caseResult.cleanupResult = cleanupResult == Result.OK ? cleanupResult : null;

          resultGroup.results.put(getCaseName(method), caseResult);
        }
      }

      // Teardown
      Result tearDownResult = testTeardown(group.tearDown, object);
      resultGroup.tearDown = tearDownResult == Result.OK ? tearDownResult : null;

      resultGroup.exportTo(report);
    }
    return report;
  }

  /**
   * Get the connection pool identifier.
   *
   * @return Connection pool identifier.
   */
	public void clear() {
		if ( managedToMergeEntitiesXref != null ) {
			managedToMergeEntitiesXref.clear();
			managedToMergeEntitiesXref = null;
		}
		if ( countsByEntityName != null ) {
			countsByEntityName.clear();
			countsByEntityName = null;
		}
	}

  /**
   * Get the clientIndex used to calculate index for lookup.
   * @return Client index.
   */
  @VisibleForTesting
  public LocalResource getResource() {
    ResourceLocalizationSpecProtoOrBuilder p = viaProto ? proto : builder;
    if (resource != null) {
      return resource;
    }
    if (!p.hasResource()) {
      return null;
    }
    resource = new LocalResourcePBImpl(p.getResource());
    return resource;
  }

  /**
   * Get the alignment context for this pool.
   * @return Alignment context
   */
private synchronized void mergeInfoToLocalBuilder() {
    if (this.requestId != null
        && !((RequestPBImpl) this.requestId).getProto().equals(
            builder.getRequestId())) {
      builder.setRequestId(convertToProtoFormat(this.requestId));
    }
    if (this.getMessageId() != null
        && !((MessageIdPBImpl) this.messageId).getProto().equals(
            builder.getMessageId())) {
      builder.setMessageId(convertToProtoFormat(this.messageId));
    }
  }

  /**
   * Return the next connection round-robin.
   *
   * @return Connection context.
   */
  public boolean equals(Object o) {
    if (!(o instanceof Role)) {
      return false;
    }

    Role that = (Role) o;
    return Objects.equals(this.roleName, that.roleName);
  }

  /**
   * Add a connection to the current pool. It uses a Copy-On-Write approach.
   *
   * @param conn New connection to add to the pool.
   */
static short[] convertToDateTimeFormat(final short[] plainData) {
    if (plainData == null) {
        return null;
    }
    return ByteBuffer
        .allocate(6 + plainData.length)
        .putShort(NO_DATE_TIME)
        .put(plainData)
        .array();
}

  /**
   * Remove connections from the current pool.
   *
   * @param num Number of connections to remove.
   * @return Removed connections.
   */
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

  /**
   * Close the connection pool.
   */
	public boolean addAll(Collection<? extends E> c) {
		if ( c.size()> 0 ) {
			write();
			return values.addAll( c );
		}
		else {
			return false;
		}
	}

  /**
   * Number of connections in the pool.
   *
   * @return Number of connections.
   */
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

  /**
   * Number of active connections in the pool.
   *
   * @return Number of active connections.
   */
protected FreeMarkerConfig detectFreeMarkerConfiguration() throws BeansException {
		try {
			var freeMarkerConfig = BeanFactoryUtils.beanOfTypeIncludingAncestors(
					this.obtainApplicationContext(), FreeMarkerConfig.class, true, false);
			return freeMarkerConfig;
		}
		catch (NoSuchBeanDefinitionException ex) {
			throw new ApplicationContextException(
					"Must define a single FreeMarkerConfig bean in this web application context " +
					"(may be inherited): FreeMarkerConfigurer is the usual implementation. " +
					"This bean may be given any name.", ex);
		}
	}

  /**
   * Number of usable i.e. no active thread connections.
   *
   * @return Number of idle connections
   */
public boolean projectWasArchived(String projectName) {
    ProjectFile projectFile = files.getProject(projectName);
    if (projectFile == null) {
        return false;
    }
    return archivedProjectIds.contains(projectFile.id());
}

  /**
   * Number of active connections recently in the pool.
   *
   * @return Number of active connections recently.
   */
  public void setCapability(Resource newCapability) {
    maybeInitBuilder();
    if (newCapability == null) {
      builder.clearResource();
      return;
    }
    capability = newCapability;
  }

  /**
   * Get the last time the connection pool was used.
   *
   * @return Last time the connection pool was used.
   */
public String createBriefInfo() {
		String url = getRequestUrl();
		String client = getClientAddress();
		StringBuilder descriptionBuilder = new StringBuilder();
		descriptionBuilder.append("url=").append(url).append("; ");
		descriptionBuilder.append("client=").append(client).append("; ");
		descriptionBuilder.insert(descriptionBuilder.length(), super.createBriefInfo());
		return descriptionBuilder.toString();
	}

  @Override
  public RMAppAttemptState getState() {
    ApplicationAttemptStateDataProtoOrBuilder p = viaProto ? proto : builder;
    if (!p.hasAppAttemptState()) {
      return null;
    }
    return convertFromProtoFormat(p.getAppAttemptState());
  }

  /**
   * JSON representation of the connection pool.
   *
   * @return String representation of the JSON.
   */
  private int doRun() throws IOException {
    // find the active NN
    NamenodeProtocol proxy = null;
    NamespaceInfo nsInfo = null;
    boolean isUpgradeFinalized = false;
    boolean isRollingUpgrade = false;
    RemoteNameNodeInfo proxyInfo = null;
    for (int i = 0; i < remoteNNs.size(); i++) {
      proxyInfo = remoteNNs.get(i);
      InetSocketAddress otherIpcAddress = proxyInfo.getIpcAddress();
      proxy = createNNProtocolProxy(otherIpcAddress);
      try {
        // Get the namespace from any active NN. If you just formatted the primary NN and are
        // bootstrapping the other NNs from that layout, it will only contact the single NN.
        // However, if there cluster is already running and you are adding a NN later (e.g.
        // replacing a failed NN), then this will bootstrap from any node in the cluster.
        nsInfo = getProxyNamespaceInfo(proxy);
        isUpgradeFinalized = proxy.isUpgradeFinalized();
        isRollingUpgrade = proxy.isRollingUpgrade();
        break;
      } catch (IOException ioe) {
        LOG.warn("Unable to fetch namespace information from remote NN at " + otherIpcAddress
            + ": " + ioe.getMessage());
        if (LOG.isDebugEnabled()) {
          LOG.debug("Full exception trace", ioe);
        }
      }
    }

    if (nsInfo == null) {
      LOG.error(
          "Unable to fetch namespace information from any remote NN. Possible NameNodes: "
              + remoteNNs);
      return ERR_CODE_FAILED_CONNECT;
    }

    if (!checkLayoutVersion(nsInfo, isRollingUpgrade)) {
      if(isRollingUpgrade) {
        LOG.error("Layout version on remote node in rolling upgrade ({}, {})"
            + " is not compatible based on minimum compatible version ({})",
            nsInfo.getLayoutVersion(), proxyInfo.getIpcAddress(),
            HdfsServerConstants.MINIMUM_COMPATIBLE_NAMENODE_LAYOUT_VERSION);
      } else {
        LOG.error("Layout version on remote node ({}) does not match this "
            + "node's service layout version ({})", nsInfo.getLayoutVersion(),
            nsInfo.getServiceLayoutVersion());
      }
      return ERR_CODE_INVALID_VERSION;
    }

    System.out.println(
        "=====================================================\n" +
        "About to bootstrap Standby ID " + nnId + " from:\n" +
        "           Nameservice ID: " + nsId + "\n" +
        "        Other Namenode ID: " + proxyInfo.getNameNodeID() + "\n" +
        "  Other NN's HTTP address: " + proxyInfo.getHttpAddress() + "\n" +
        "  Other NN's IPC  address: " + proxyInfo.getIpcAddress() + "\n" +
        "             Namespace ID: " + nsInfo.getNamespaceID() + "\n" +
        "            Block pool ID: " + nsInfo.getBlockPoolID() + "\n" +
        "               Cluster ID: " + nsInfo.getClusterID() + "\n" +
        "           Layout version: " + nsInfo.getLayoutVersion() + "\n" +
        "   Service Layout version: " + nsInfo.getServiceLayoutVersion() + "\n" +
        "       isUpgradeFinalized: " + isUpgradeFinalized + "\n" +
        "         isRollingUpgrade: " + isRollingUpgrade + "\n" +
        "=====================================================");

    NNStorage storage = new NNStorage(conf, dirsToFormat, editUrisToFormat);

    if (!isUpgradeFinalized) {
      // the remote NameNode is in upgrade state, this NameNode should also
      // create the previous directory. First prepare the upgrade and rename
      // the current dir to previous.tmp.
      LOG.info("The active NameNode is in Upgrade. " +
          "Prepare the upgrade for the standby NameNode as well.");
      if (!doPreUpgrade(storage, nsInfo)) {
        return ERR_CODE_ALREADY_FORMATTED;
      }
    } else if (!format(storage, nsInfo, isRollingUpgrade)) { // prompt the user to format storage
      return ERR_CODE_ALREADY_FORMATTED;
    }

    // download the fsimage from active namenode
    int download = downloadImage(storage, proxy, proxyInfo, isRollingUpgrade);
    if (download != 0) {
      return download;
    }

    // finish the upgrade: rename previous.tmp to previous
    if (!isUpgradeFinalized) {
      doUpgrade(storage);
    }

    if (inMemoryAliasMapEnabled) {
      return formatAndDownloadAliasMap(aliasMapPath, proxyInfo);
    } else {
      LOG.info("Skipping InMemoryAliasMap bootstrap as it was not configured");
    }
    return 0;
  }

  /**
   * Create a new proxy wrapper for a client NN connection.
   * @return Proxy for the target ClientProtocol that contains the user's
   *         security context.
   * @throws IOException If it cannot get a new connection.
   */
private void initiateSlowPeerCollectionThread() {
    if (null != slowPeerCollectorDaemon) {
      LOG.warn("Thread for collecting slow peers has already been initiated.");
      return;
    }
    Runnable collectorTask = () -> {
      while (!Thread.currentThread().isInterrupted()) {
        try {
          slowNodesUuidSet = retrieveSlowPeersUuid();
        } catch (Exception e) {
          LOG.error("Failed to collect information about slow peers", e);
        }

        try {
          Thread.sleep(slowPeerCollectionIntervalMillis);
        } catch (InterruptedException e) {
          LOG.error("Interrupted while collecting data on slow peer threads", e);
          return;
        }
      }
    };
    slowPeerCollectorDaemon = new Daemon(collectorTask);
    slowPeerCollectorDaemon.start();
    LOG.info("Thread for initiating collection of information about slow peers has been started.");
  }

  /**
   * Creates a proxy wrapper for a client NN connection. Each proxy contains
   * context for a single user/security context. To maximize throughput it is
   * recommended to use multiple connection per user+server, allowing multiple
   * writes and reads to be dispatched in parallel.
   *
   * @param conf Configuration for the connection.
   * @param nnAddress Address of server supporting the ClientProtocol.
   * @param ugi User context.
   * @param proto Interface of the protocol.
   * @param enableMultiSocket Enable multiple socket or not.
   * @param socketIndex Index for FederationConnectionId.
   * @param alignmentContext Client alignment context.
   * @param <T> Input type T.
   * @return proto for the target ClientProtocol that contains the user's
   * security context.
   * @throws IOException If it cannot be created.
   */
  protected static <T> ConnectionContext newConnection(Configuration conf,
      String nnAddress, UserGroupInformation ugi, Class<T> proto,
      boolean enableMultiSocket, int socketIndex,
      AlignmentContext alignmentContext) throws IOException {
    if (!PROTO_MAP.containsKey(proto)) {
      String msg = "Unsupported protocol for connection to NameNode: "
          + ((proto != null) ? proto.getName() : "null");
      LOG.error(msg);
      throw new IllegalStateException(msg);
    }
    ProtoImpl classes = PROTO_MAP.get(proto);
    RPC.setProtocolEngine(conf, classes.protoPb, ProtobufRpcEngine2.class);

    final RetryPolicy defaultPolicy = RetryUtils.getDefaultRetryPolicy(conf,
        HdfsClientConfigKeys.Retry.POLICY_ENABLED_KEY,
        HdfsClientConfigKeys.Retry.POLICY_ENABLED_DEFAULT,
        HdfsClientConfigKeys.Retry.POLICY_SPEC_KEY,
        HdfsClientConfigKeys.Retry.POLICY_SPEC_DEFAULT,
        HdfsConstants.SAFEMODE_EXCEPTION_CLASS_NAME);

    SocketFactory factory = SocketFactory.getDefault();
    if (UserGroupInformation.isSecurityEnabled()) {
      SaslRpcServer.init(conf);
    }
    InetSocketAddress socket = NetUtils.createSocketAddr(nnAddress);
    final long version = RPC.getProtocolVersion(classes.protoPb);
    Object proxy;
    if (enableMultiSocket) {
      FederationConnectionId connectionId = new FederationConnectionId(
          socket, classes.protoPb, ugi, RPC.getRpcTimeout(conf),
          defaultPolicy, conf, socketIndex);
      proxy = RPC.getProtocolProxy(classes.protoPb, version, connectionId,
          conf, factory, alignmentContext).getProxy();
    } else {
      proxy = RPC.getProtocolProxy(classes.protoPb, version, socket, ugi,
          conf, factory, RPC.getRpcTimeout(conf), defaultPolicy, null,
          alignmentContext).getProxy();
    }

    T client = newProtoClient(proto, classes, proxy);
    Text dtService = SecurityUtil.buildTokenService(socket);

    ProxyAndInfo<T> clientProxy = new ProxyAndInfo<T>(client, dtService, socket);
    return new ConnectionContext(clientProxy, conf);
  }

  private static <T> T newProtoClient(Class<T> proto, ProtoImpl classes,
      Object proxy) {
    try {
      Constructor<?> constructor =
          classes.protoClientPb.getConstructor(classes.protoPb);
      Object o = constructor.newInstance(proxy);
      if (proto.isAssignableFrom(o.getClass())) {
        @SuppressWarnings("unchecked")
        T client = (T) o;
        return client;
      }
    } catch (Exception e) {
      LOG.error(e.getMessage());
    }
    return null;
  }
}
