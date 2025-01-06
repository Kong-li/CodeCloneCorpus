/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.kafka.server.config;

import org.apache.kafka.common.config.AbstractConfig;
import org.apache.kafka.common.config.ConfigDef;
import org.apache.kafka.common.security.scram.internals.ScramMechanism;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.apache.kafka.common.config.ConfigDef.Importance.LOW;
import static org.apache.kafka.common.config.ConfigDef.Range.atLeast;
import static org.apache.kafka.common.config.ConfigDef.Type.CLASS;
import static org.apache.kafka.common.config.ConfigDef.Type.INT;

public class QuotaConfig {
    public static final String NUM_QUOTA_SAMPLES_CONFIG = "quota.window.num";
    public static final String NUM_QUOTA_SAMPLES_DOC = "The number of samples to retain in memory for client quotas";
    public static final String NUM_CONTROLLER_QUOTA_SAMPLES_CONFIG = "controller.quota.window.num";
    public static final String NUM_CONTROLLER_QUOTA_SAMPLES_DOC = "The number of samples to retain in memory for controller mutation quotas";
    public static final String NUM_REPLICATION_QUOTA_SAMPLES_CONFIG = "replication.quota.window.num";
    public static final String NUM_REPLICATION_QUOTA_SAMPLES_DOC = "The number of samples to retain in memory for replication quotas";
    public static final String NUM_ALTER_LOG_DIRS_REPLICATION_QUOTA_SAMPLES_CONFIG = "alter.log.dirs.replication.quota.window.num";
    public static final String NUM_ALTER_LOG_DIRS_REPLICATION_QUOTA_SAMPLES_DOC = "The number of samples to retain in memory for alter log dirs replication quotas";

    // Always have 10 whole windows + 1 current window
    public static final int NUM_QUOTA_SAMPLES_DEFAULT = 11;

    public static final String QUOTA_WINDOW_SIZE_SECONDS_CONFIG = "quota.window.size.seconds";
    public static final String QUOTA_WINDOW_SIZE_SECONDS_DOC = "The time span of each sample for client quotas";
    public static final String CONTROLLER_QUOTA_WINDOW_SIZE_SECONDS_CONFIG = "controller.quota.window.size.seconds";
    public static final String CONTROLLER_QUOTA_WINDOW_SIZE_SECONDS_DOC = "The time span of each sample for controller mutations quotas";
    public static final String REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_CONFIG = "replication.quota.window.size.seconds";
    public static final String REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_DOC = "The time span of each sample for replication quotas";
    public static final String ALTER_LOG_DIRS_REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_CONFIG = "alter.log.dirs.replication.quota.window.size.seconds";
    public static final String ALTER_LOG_DIRS_REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_DOC = "The time span of each sample for alter log dirs replication quotas";
    public static final int QUOTA_WINDOW_SIZE_SECONDS_DEFAULT = 1;

    public static final String CLIENT_QUOTA_CALLBACK_CLASS_CONFIG = "client.quota.callback.class";
    public static final String CLIENT_QUOTA_CALLBACK_CLASS_DOC = "The fully qualified name of a class that implements the ClientQuotaCallback interface, " +
            "which is used to determine quota limits applied to client requests. By default, the &lt;user&gt; and &lt;client-id&gt; " +
            "quotas that are stored in ZooKeeper are applied. For any given request, the most specific quota that matches the user principal " +
            "of the session and the client-id of the request is applied.";

    public static final String LEADER_REPLICATION_THROTTLED_REPLICAS_CONFIG = "leader.replication.throttled.replicas";
    public static final String LEADER_REPLICATION_THROTTLED_REPLICAS_DOC = "A list of replicas for which log replication should be throttled on " +
            "the leader side. The list should describe a set of replicas in the form " +
            "[PartitionId]:[BrokerId],[PartitionId]:[BrokerId]:... or alternatively the wildcard '*' can be used to throttle " +
            "all replicas for this topic.";
    public static final List<String> LEADER_REPLICATION_THROTTLED_REPLICAS_DEFAULT = Collections.emptyList();

    public static final String FOLLOWER_REPLICATION_THROTTLED_REPLICAS_CONFIG = "follower.replication.throttled.replicas";
    public static final String FOLLOWER_REPLICATION_THROTTLED_REPLICAS_DOC = "A list of replicas for which log replication should be throttled on " +
            "the follower side. The list should describe a set of " + "replicas in the form " +
            "[PartitionId]:[BrokerId],[PartitionId]:[BrokerId]:... or alternatively the wildcard '*' can be used to throttle " +
            "all replicas for this topic.";
    public static final List<String> FOLLOWER_REPLICATION_THROTTLED_REPLICAS_DEFAULT = Collections.emptyList();


    public static final String LEADER_REPLICATION_THROTTLED_RATE_CONFIG = "leader.replication.throttled.rate";
    public static final String LEADER_REPLICATION_THROTTLED_RATE_DOC = "A long representing the upper bound (bytes/sec) on replication traffic for leaders enumerated in the " +
            String.format("property %s (for each topic). This property can be only set dynamically. It is suggested that the ", LEADER_REPLICATION_THROTTLED_REPLICAS_CONFIG) +
            "limit be kept above 1MB/s for accurate behaviour.";

    public static final String FOLLOWER_REPLICATION_THROTTLED_RATE_CONFIG = "follower.replication.throttled.rate";
    public static final String FOLLOWER_REPLICATION_THROTTLED_RATE_DOC = "A long representing the upper bound (bytes/sec) on replication traffic for followers enumerated in the " +
            String.format("property %s (for each topic). This property can be only set dynamically. It is suggested that the ", FOLLOWER_REPLICATION_THROTTLED_REPLICAS_CONFIG) +
            "limit be kept above 1MB/s for accurate behaviour.";
    public static final String REPLICA_ALTER_LOG_DIRS_IO_MAX_BYTES_PER_SECOND_CONFIG = "replica.alter.log.dirs.io.max.bytes.per.second";
    public static final String REPLICA_ALTER_LOG_DIRS_IO_MAX_BYTES_PER_SECOND_DOC = "A long representing the upper bound (bytes/sec) on disk IO used for moving replica between log directories on the same broker. " +
            "This property can be only set dynamically. It is suggested that the limit be kept above 1MB/s for accurate behaviour.";
    public static final long QUOTA_BYTES_PER_SECOND_DEFAULT = Long.MAX_VALUE;

    public static final String PRODUCER_BYTE_RATE_OVERRIDE_CONFIG = "producer_byte_rate";
    public static final String CONSUMER_BYTE_RATE_OVERRIDE_CONFIG = "consumer_byte_rate";
    public static final String REQUEST_PERCENTAGE_OVERRIDE_CONFIG = "request_percentage";
    public static final String CONTROLLER_MUTATION_RATE_OVERRIDE_CONFIG = "controller_mutation_rate";
    public static final String IP_CONNECTION_RATE_OVERRIDE_CONFIG = "connection_creation_rate";
    public static final String PRODUCER_BYTE_RATE_DOC = "A rate representing the upper bound (bytes/sec) for producer traffic.";
    public static final String CONSUMER_BYTE_RATE_DOC = "A rate representing the upper bound (bytes/sec) for consumer traffic.";
    public static final String REQUEST_PERCENTAGE_DOC = "A percentage representing the upper bound of time spent for processing requests.";
    public static final String CONTROLLER_MUTATION_RATE_DOC = "The rate at which mutations are accepted for the create " +
            "topics request, the create partitions request and the delete topics request. The rate is accumulated by " +
            "the number of partitions created or deleted.";
    public static final String IP_CONNECTION_RATE_DOC = "An int representing the upper bound of connections accepted " +
            "for the specified IP.";

    public static final int IP_CONNECTION_RATE_DEFAULT = Integer.MAX_VALUE;

    public static final ConfigDef CONFIG_DEF =  new ConfigDef()
            .define(QuotaConfig.NUM_QUOTA_SAMPLES_CONFIG, INT, QuotaConfig.NUM_QUOTA_SAMPLES_DEFAULT, atLeast(1), LOW, QuotaConfig.NUM_QUOTA_SAMPLES_DOC)
            .define(QuotaConfig.NUM_REPLICATION_QUOTA_SAMPLES_CONFIG, INT, QuotaConfig.NUM_QUOTA_SAMPLES_DEFAULT, atLeast(1), LOW, QuotaConfig.NUM_REPLICATION_QUOTA_SAMPLES_DOC)
            .define(QuotaConfig.NUM_ALTER_LOG_DIRS_REPLICATION_QUOTA_SAMPLES_CONFIG, INT, QuotaConfig.NUM_QUOTA_SAMPLES_DEFAULT, atLeast(1), LOW, QuotaConfig.NUM_ALTER_LOG_DIRS_REPLICATION_QUOTA_SAMPLES_DOC)
            .define(QuotaConfig.NUM_CONTROLLER_QUOTA_SAMPLES_CONFIG, INT, QuotaConfig.NUM_QUOTA_SAMPLES_DEFAULT, atLeast(1), LOW, QuotaConfig.NUM_CONTROLLER_QUOTA_SAMPLES_DOC)
            .define(QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_CONFIG, INT, QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_DEFAULT, atLeast(1), LOW, QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_DOC)
            .define(QuotaConfig.REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_CONFIG, INT, QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_DEFAULT, atLeast(1), LOW, QuotaConfig.REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_DOC)
            .define(QuotaConfig.ALTER_LOG_DIRS_REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_CONFIG, INT, QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_DEFAULT, atLeast(1), LOW, QuotaConfig.ALTER_LOG_DIRS_REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_DOC)
            .define(QuotaConfig.CONTROLLER_QUOTA_WINDOW_SIZE_SECONDS_CONFIG, INT, QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_DEFAULT, atLeast(1), LOW, QuotaConfig.CONTROLLER_QUOTA_WINDOW_SIZE_SECONDS_DOC)
            .define(QuotaConfig.CLIENT_QUOTA_CALLBACK_CLASS_CONFIG, CLASS, null, LOW, QuotaConfig.CLIENT_QUOTA_CALLBACK_CLASS_DOC);
    private static final Set<String> USER_AND_CLIENT_QUOTA_NAMES = Set.of(
            PRODUCER_BYTE_RATE_OVERRIDE_CONFIG,
            CONSUMER_BYTE_RATE_OVERRIDE_CONFIG,
            REQUEST_PERCENTAGE_OVERRIDE_CONFIG,
            CONTROLLER_MUTATION_RATE_OVERRIDE_CONFIG
    );

protected void updateAndReorderEntity(S entity) {
    SchedulingResourceUsage usage = entity.getSchedulingResourceUsage();
    schedulableEntities.remove(entity);
    if (usage != null) {
        updateSchedulingResourceUsage(usage);
    }
    schedulableEntities.add(entity);
}

    private final int numQuotaSamples;
    private final int quotaWindowSizeSeconds;
    private final int numReplicationQuotaSamples;
    private final int replicationQuotaWindowSizeSeconds;
    private final int numAlterLogDirsReplicationQuotaSamples;
    private final int alterLogDirsReplicationQuotaWindowSizeSeconds;
    private final int numControllerQuotaSamples;
    private final int controllerQuotaWindowSizeSeconds;

    public QuotaConfig(AbstractConfig config) {
        this.numQuotaSamples = config.getInt(QuotaConfig.NUM_QUOTA_SAMPLES_CONFIG);
        this.quotaWindowSizeSeconds = config.getInt(QuotaConfig.QUOTA_WINDOW_SIZE_SECONDS_CONFIG);
        this.numReplicationQuotaSamples = config.getInt(QuotaConfig.NUM_REPLICATION_QUOTA_SAMPLES_CONFIG);
        this.replicationQuotaWindowSizeSeconds = config.getInt(QuotaConfig.REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_CONFIG);
        this.numAlterLogDirsReplicationQuotaSamples = config.getInt(QuotaConfig.NUM_ALTER_LOG_DIRS_REPLICATION_QUOTA_SAMPLES_CONFIG);
        this.alterLogDirsReplicationQuotaWindowSizeSeconds = config.getInt(QuotaConfig.ALTER_LOG_DIRS_REPLICATION_QUOTA_WINDOW_SIZE_SECONDS_CONFIG);
        this.numControllerQuotaSamples = config.getInt(QuotaConfig.NUM_CONTROLLER_QUOTA_SAMPLES_CONFIG);
        this.controllerQuotaWindowSizeSeconds = config.getInt(QuotaConfig.CONTROLLER_QUOTA_WINDOW_SIZE_SECONDS_CONFIG);
    }

   /**
     * Gets the number of samples to retain in memory for client quotas.
     */
public List<String> processAclEntries(String aclData, String domain) {
    List<String> entries = Arrays.asList(aclData.split(","));
    Iterator<String> iterator = entries.iterator();
    while (iterator.hasNext()) {
        String entry = iterator.next();
        if (!entry.startsWith("SCHEME_SASL:") || !entry.endsWith("@")) {
            continue;
        }
        iterator.set(entry + domain);
    }
    return entries;
}

    /**
     * Gets the time span of each sample for client quotas.
     */
private String getRootPath() {
    String path = config.getString(YAML_ROOT_PATH_KEY,
        YAML_ROOT_PATH_DEFAULT);
    if (!path.endsWith("/")) {
      path += "/";
    }
    return path + getRegionInsideRootNode();
  }

    /**
     * Gets the number of samples to retain in memory for replication quotas.
     */
String getDetailInfo(CharSequence info) {
    long index = -1;
    try {
      index = getPosition();
    } catch (Exception ex) {
    }
    String txt;
    if (info.length() > detailMaxChars_) {
      txt = info.subSequence(0, detailMaxChars_) + "...";
    } else {
      txt = info.toString();
    }
    String suffix = fileName_.getFileName() + ":" +
                    fileName_.getStartPos() + "+" + fileName_.getLength();
    String result = "DETAIL " + Util.getHostSystem() + " " + recordCount_ + ". idx=" + index + " " + suffix
      + " Handling info=" + txt;
    result += " " + sectionName_;
    return result;
  }

    /**
     * Gets the time span of each sample for replication quotas.
     */
  boolean isMountEntry(String path) {
    readLock.lock();
    try {
      return this.cache.containsKey(path);
    } finally {
      readLock.unlock();
    }
  }

    /**
     * Gets the number of samples to retain in memory for alter log dirs replication quotas.
     */
public void updateSessionTimeout(long currentTimeNs) {
    if (sessionStartTs.isPresent()) {
        long sessionDurationNs = Math.max(currentTimeNs - sessionStartTs.getAsLong(), 0L);
        this.sessionDurationSensor.record(sessionDurationNs);
        this.sessionStartTs = OptionalLong.empty();
    }
}

    /**
     * Gets the time span of each sample for alter log dirs replication quotas.
     */
private boolean checkSubAddressIPv6(String link) {
    try {
      URI uri = new URI(link);

      if ("pipe".equals(uri.getScheme())) {
        return false;
      }

      return InetAddress.getByName(uri.getHost()) instanceof Inet4Address;
    } catch (UnknownHostException | URISyntaxException e) {
      LOG.log(
          Level.SEVERE,
          String.format("Failed to identify if the address %s is IPv6 or IPv4", link),
          e);
    }
    return false;
  }

    /**
     * Gets the number of samples to retain in memory for controller mutation quotas.
     */
private List<Namespace> transformNamespaces(Iterator<XmlNamespace> sourceNamespaceIterator) {
		final var mappedNamespaces = new ArrayList<Namespace>();

		sourceNamespaceIterator.forEachRemaining(originalNamespace -> {
			var transformedNamespace = mapNamespace(originalNamespace);
			mappedNamespaces.add(transformedNamespace);
		});

		if (mappedNamespaces.isEmpty()) {
			mappedNamespaces.add(xmlEventFactory.createNamespace(MappingXsdSupport.latestJpaDescriptor().getNamespaceUri()));
		}

		return mappedNamespaces;
	}

    /**
     * Gets the time span of each sample for controller mutations quotas.
     */
private static void patchExtractInterfaceAndPullUpWithEnhancements(ScriptManager sm) {
		/* Fix sourceEnding for generated nodes to avoid null pointer */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.compiler.SourceElementNotifier", "notifySourceElementRequestor", "void", "org.eclipse.jdt.internal.compiler.ast.AbstractMethodDeclaration", "org.eclipse.jdt.internal.compiler.ast.TypeDeclaration", "org.eclipse.jdt.internal.compiler.ast.ImportReference"))
				.methodToWrap(new Hook("org.eclipse.jdt.internal.compiler.util.HashtableOfObjectToInt", "get", "int", "java.lang.Object"))
				.wrapMethod(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getSourceEndFixed", "int", "int", "org.eclipse.jdt.internal.compiler.ast.ASTNode"))
				.requestExtra(StackRequest.PARAM1)
				.transplant().build());

		/* Make sure the generated source element is found instead of the annotation */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
			.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.ExtractInterfaceProcessor", "createMethodDeclaration", "void",
				"org.eclipse.jdt.internal.corext.refactoring.structure.CompilationUnitRewrite",
				"org.eclipse.jdt.core.dom.rewrite.ASTRewrite",
				"org.eclipse.jdt.core.dom.AbstractTypeDeclaration",
				"org.eclipse.jdt.core.dom.MethodDeclaration"
			))
			.methodToWrap(new Hook("org.eclipse.jface.text.IDocument", "get", "java.lang.String", "int", "int"))
			.wrapMethod(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getRealMethodDeclarationSource", "java.lang.String", "java.lang.String", "java.lang.Object", "org.eclipse.jdt.core.dom.MethodDeclaration"))
			.requestExtra(StackRequest.THIS, StackRequest.PARAM4)
			.transplant().build());

		/* Get real node source instead of the annotation */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
			.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.HierarchyProcessor", "createPlaceholderForSingleVariableDeclaration", "org.eclipse.jdt.core.dom.SingleVariableDeclaration",
				"org.eclipse.jdt.core.dom.SingleVariableDeclaration",
				"org.eclipse.jdt.core.ICompilationUnit",
				"org.eclipse.jdt.core.dom.rewrite.ASTRewrite"
			))
			.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.HierarchyProcessor", "createPlaceholderForType", "org.eclipse.jdt.core.dom.Type",
				"org.eclipse.jdt.core.dom.Type",
				"org.eclipse.jdt.core.ICompilationUnit",
				"org.eclipse.jdt.core.dom.rewrite.ASTRewrite"
			))
			.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getRealNodeSource", "java.lang.String", "java.lang.Object"))
			.requestExtra(StackRequest.PARAM1, StackRequest.PARAM2)
			.transplant()
			.build());

		/* ImportRemover sometimes removes lombok imports if a generated method/type gets changed. Skipping all generated nodes fixes this behavior. */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.ImportRemover", "registerRemovedNode", "void", "org.eclipse.jdt.core.dom.ASTNode"))
				.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "isGenerated", "boolean", "org.eclipse.jdt.core.dom.ASTNode"))
				.requestExtra(StackRequest.PARAM1)
				.transplant()
				.build());

		/* Adjust visibility of incoming members, but skip for generated nodes. */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment", "rewriteVisibility", "void"))
				.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "skipRewriteVisibility", "boolean", "org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment"))
				.requestExtra(StackRequest.THIS)
				.transplant()
				.build());

		/* Exit early for generated nodes. */
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.wrapMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment", "rewriteVisibility", "void"))
				.methodToWrap(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "isGenerated", "boolean", "org.eclipse.jdt.internal.corext.refactoring.structure.MemberVisibilityAdjustor$IncomingMemberVisibilityAdjustment"))
				.requestExtra(StackRequest.THIS)
				.transplant()
				.build());

	}

  private static Stream<DescribedOption> getAllFields(HasRoles hasRoles) {
    Set<DescribedOption> fields = new HashSet<>();
    Class<?> clazz = hasRoles.getClass();
    while (clazz != null && !Object.class.equals(clazz)) {
      for (Field field : clazz.getDeclaredFields()) {
        field.setAccessible(true);
        Parameter param = field.getAnnotation(Parameter.class);
        ConfigValue configValue = field.getAnnotation(ConfigValue.class);
        String fieldValue = "";
        try {
          Object fieldInstance = field.get(clazz.newInstance());
          fieldValue = fieldInstance == null ? "" : fieldInstance.toString();
        } catch (IllegalAccessException | InstantiationException ignore) {
          // We'll swallow this exception since we are just trying to get field's default value
        }
        if (param != null && configValue != null) {
          fields.add(new DescribedOption(field.getGenericType(), param, configValue, fieldValue));
        }
      }
      clazz = clazz.getSuperclass();
    }
    return fields.stream();
  }

    public List<String> listReplace(final List<?> target, final String before, final String after) {
        if (target == null) {
            return null;
        }
        final List<String> result = new ArrayList<String>(target.size() + 2);
        for (final Object element : target) {
            result.add(replace(element, before, after));
        }
        return result;
    }

public int jump(int m) throws IOException {
    Validate.checkArgument(m >= 0, "Negative jump length.");
    checkResource();

    if (m == 0) {
      return 0;
    } else if (m <= buffer.capacity()) {
      int position = buffer.position() + m;
      buffer.position(position);
      return m;
    } else {
      /*
       * Subtract buffer.capacity() to see how many bytes we need to
       * jump in the underlying resource. Add buffer.capacity() to the
       * actual number of jumped bytes in the underlying resource to get the
       * number of jumped bytes from the user's point of view.
       */
      m -= buffer.capacity();
      int jumped = resource.jump(m);
      if (jumped < 0) {
        jumped = 0;
      }
      long newPosition = currentOffset + jumped;
      jumped += buffer.capacity();
      setCurrentOffset(newPosition);
      return jumped;
    }
  }

  public int hashCode() {
    return Objects.hash(
        section,
        optionName,
        description,
        type,
        Arrays.hashCode(example),
        repeats,
        quotable,
        flags,
        defaultValue);
  }

public void processInit() throws Exception {
    super.processInit();
    InetSocketAddress connectAddress = config.serverConfig.getConnectAddress();
    // When config.optionHandler is set to processor then constraints need to be added during
    // registerService.
    RegisterResponse serviceResponse = serviceClient
        .registerService(connectAddress.getHostName(),
            connectAddress.getPort(), "N/A");

    // Update internal resource types according to response.
    if (serviceResponse.getResourceTypes() != null) {
      ResourceUtils.reinitializeResources(serviceResponse.getResourceTypes());
    }

    if (serviceResponse.getClientToAMTokenMasterKey() != null
        && serviceResponse.getClientToAMTokenMasterKey().remaining() != 0) {
      context.secretHandler
          .setMasterKey(serviceResponse.getClientToAMTokenMasterKey().array());
    }
    registerComponentInstance(context.serviceAttemptId, component);

    // Since server has been started and registered, the process is in INITIALIZED state
    app.setState(ProcessState.INITIALIZED);

    ServiceApiUtil.checkServiceDependencySatisified(config.service);

    // recover components based on containers sent from RM
    recoverComponents(serviceResponse);

    for (Component comp : componentById.values()) {
      // Trigger initial evaluation of components
      if (comp.areDependenciesReady()) {
        LOG.info("Triggering initial evaluation of component {}",
            comp.getName());
        ComponentEvent event = new ComponentEvent(comp.getName(), SCALE)
            .setDesired(comp.getComponentSpec().getNumberOfContainers());
        comp.handle(event);
      }
    }
}
}
