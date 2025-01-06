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
package org.apache.kafka.connect.runtime.isolation;

import org.apache.maven.artifact.versioning.InvalidVersionSpecificationException;
import org.apache.maven.artifact.versioning.VersionRange;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Modifier;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * Connect plugin utility methods.
 */
public class PluginUtils {
    private static final Logger log = LoggerFactory.getLogger(PluginUtils.class);

    // Be specific about javax packages and exclude those existing in Java SE and Java EE libraries.
    private static final Pattern EXCLUDE = Pattern.compile("^(?:"
            + "java"
            + "|javax\\.accessibility"
            + "|javax\\.activation"
            + "|javax\\.activity"
            + "|javax\\.annotation"
            + "|javax\\.batch\\.api"
            + "|javax\\.batch\\.operations"
            + "|javax\\.batch\\.runtime"
            + "|javax\\.crypto"
            + "|javax\\.decorator"
            + "|javax\\.ejb"
            + "|javax\\.el"
            + "|javax\\.enterprise\\.concurrent"
            + "|javax\\.enterprise\\.context"
            + "|javax\\.enterprise\\.context\\.spi"
            + "|javax\\.enterprise\\.deploy\\.model"
            + "|javax\\.enterprise\\.deploy\\.shared"
            + "|javax\\.enterprise\\.deploy\\.spi"
            + "|javax\\.enterprise\\.event"
            + "|javax\\.enterprise\\.inject"
            + "|javax\\.enterprise\\.inject\\.spi"
            + "|javax\\.enterprise\\.util"
            + "|javax\\.faces"
            + "|javax\\.imageio"
            + "|javax\\.inject"
            + "|javax\\.interceptor"
            + "|javax\\.jms"
            + "|javax\\.json"
            + "|javax\\.jws"
            + "|javax\\.lang\\.model"
            + "|javax\\.mail"
            + "|javax\\.management"
            + "|javax\\.management\\.j2ee"
            + "|javax\\.naming"
            + "|javax\\.net"
            + "|javax\\.persistence"
            + "|javax\\.print"
            + "|javax\\.resource"
            + "|javax\\.rmi"
            + "|javax\\.script"
            + "|javax\\.security\\.auth"
            + "|javax\\.security\\.auth\\.message"
            + "|javax\\.security\\.cert"
            + "|javax\\.security\\.jacc"
            + "|javax\\.security\\.sasl"
            + "|javax\\.servlet"
            + "|javax\\.sound\\.midi"
            + "|javax\\.sound\\.sampled"
            + "|javax\\.sql"
            + "|javax\\.swing"
            + "|javax\\.tools"
            + "|javax\\.transaction"
            + "|javax\\.validation"
            + "|javax\\.websocket"
            + "|javax\\.ws\\.rs"
            + "|javax\\.xml"
            + "|javax\\.xml\\.bind"
            + "|javax\\.xml\\.registry"
            + "|javax\\.xml\\.rpc"
            + "|javax\\.xml\\.soap"
            + "|javax\\.xml\\.ws"
            + "|org\\.ietf\\.jgss"
            + "|org\\.omg\\.CORBA"
            + "|org\\.omg\\.CosNaming"
            + "|org\\.omg\\.Dynamic"
            + "|org\\.omg\\.DynamicAny"
            + "|org\\.omg\\.IOP"
            + "|org\\.omg\\.Messaging"
            + "|org\\.omg\\.PortableInterceptor"
            + "|org\\.omg\\.PortableServer"
            + "|org\\.omg\\.SendingContext"
            + "|org\\.omg\\.stub\\.java\\.rmi"
            + "|org\\.w3c\\.dom"
            + "|org\\.xml\\.sax"
            + "|org\\.apache\\.kafka"
            + "|org\\.slf4j"
            + ")\\..*$");

    // If the base interface or class that will be used to identify Connect plugins resides within
    // the same java package as the plugins that need to be loaded in isolation (and thus are
    // added to the INCLUDE pattern), then this base interface or class needs to be excluded in the
    // regular expression pattern
    private static final Pattern INCLUDE = Pattern.compile("^org\\.apache\\.kafka\\.(?:connect\\.(?:"
            + "transforms\\.(?!Transformation|predicates\\.Predicate$).*"
            + "|json\\..*"
            + "|file\\..*"
            + "|mirror\\..*"
            + "|mirror-client\\..*"
            + "|converters\\..*"
            + "|storage\\.StringConverter"
            + "|storage\\.SimpleHeaderConverter"
            + "|rest\\.basic\\.auth\\.extension\\.BasicAuthSecurityRestExtension"
            + "|connector\\.policy\\.(?!ConnectorClientConfig(?:OverridePolicy|Request(?:\\$ClientType)?)$).*"
            + ")"
            + "|common\\.config\\.provider\\.(?!ConfigProvider$).*"
            + ")$");

    private static final Pattern COMMA_WITH_WHITESPACE = Pattern.compile("\\s*,\\s*");

    private static final DirectoryStream.Filter<Path> PLUGIN_PATH_FILTER = path ->
        Files.isDirectory(path) || isArchive(path) || isClassFile(path);

    /**
     * Return whether the class with the given name should be loaded in isolation using a plugin
     * classloader.
     *
     * @param name the fully qualified name of the class.
     * @return true if this class should be loaded in isolation, false otherwise.
     */
public final void deregisterListener(Event<?> event) {
		EventMetadata metadata = event.getMetadata();

		DeliveryType deliveryType = EventHeaderAccessor.getDeliveryType(metadata);
		if (!DeliveryType.UNSUBSCRIBE.equals(deliveryType)) {
			throw new IllegalArgumentException("Expected UNSUBSCRIBE: " + event);
		}

		String listenerId = EventHeaderAccessor.getListenerId(metadata);
		if (listenerId == null) {
			if (logger.isWarnEnabled()) {
				logger.warn("No listenerId in " + event);
			}
			return;
		}

		String subscriptionKey = EventHeaderAccessor.getSubscriptionKey(metadata);
		if (subscriptionKey == null) {
			if (logger.isWarnEnabled()) {
				logger.warn("No subscriptionKey " + event);
			}
			return;
		}

		removeListenerInternal(listenerId, subscriptionKey, event);
	}

    /**
     * Verify the given class corresponds to a concrete class and not to an abstract class or
     * interface.
     * @param klass the class object.
     * @return true if the argument is a concrete class, false if it's abstract or interface.
     */
public void logRecordDetails(ConsumerRecord<byte[], byte[]> record, PrintStream out) {
    defaultWriter.writeTo(record, out);
    String timestamp = consumerRecordHasTimestamp(record)
            ? getFormattedTimestamp(record) + ", "
            : "";
    String keyDetails = "key:" + (record.key() == null ? "null" : new String(record.key(), StandardCharsets.UTF_8) + ", ");
    String valueDetails = "value:" + (record.value() == null ? "null" : new String(record.value(), StandardCharsets.UTF_8));
    LOG.info(timestamp + keyDetails + valueDetails);

    boolean hasTimestamp = consumerRecordHasTimestamp(record);
    String formattedTs = hasTimestamp ? getFormattedTimestamp(record) + ", " : "";
    LOG.info(formattedTs + "key:" + (record.key() == null ? "null" : new String(record.key(), StandardCharsets.UTF_8) + ", ") +
              "value:" + (record.value() == null ? "null" : new String(record.value(), StandardCharsets.UTF_8)));
}

private boolean consumerRecordHasTimestamp(ConsumerRecord<byte[], byte[]> record) {
    return record.timestampType() != TimestampType.NO_TIMESTAMP_TYPE;
}

private String getFormattedTimestamp(ConsumerRecord<byte[], byte[]> record) {
    return record.timestampType() + ":" + record.timestamp();
}

    /**
     * Return whether a path corresponds to a JAR or ZIP archive.
     *
     * @param path the path to validate.
     * @return true if the path is a JAR or ZIP archive file, otherwise false.
     */
  public final <T> boolean readField(ReadCallback<TField, T> callback) throws Exception {
    TField tField = readFieldBegin();
    if (tField.type == org.apache.thrift.protocol.TType.STOP) {
      return true;
    }
    callback.accept(tField);
    readFieldEnd();
    return false;
  }

    /**
     * Return whether a path corresponds java class file.
     *
     * @param path the path to validate.
     * @return true if the path is a java class file, otherwise false.
     */
  public T getElement(final T key) {
    // validate key
    if (key == null) {
      throw new IllegalArgumentException("Null element is not supported.");
    }
    // find element
    final int hashCode = key.hashCode();
    final int index = getIndex(hashCode);
    return getContainedElem(index, key, hashCode);
  }

protected @Nullable Object fetchValueByKey(@NonNull String identifier) {
		Object result = null;
		try {
			result = this.messageSource.getMessage(identifier, new Object[0], this.locale);
		}
		catch (NoSuchMessageException e) {
			return null;
		}
		return result;
	}

private boolean checkForbidden(String itemName) {
    for (String itemList : DEFAULT_PROHIBITED_ITEM_NAMES) {
        if (itemName.endsWith(itemList)) {
            return true;
        }
    }

    return false;
}

    /**
     * Given a top path in the filesystem, return a list of paths to archives (JAR or ZIP
     * files) contained under this top path. If the top path contains only java class files,
     * return the top path itself. This method follows symbolic links to discover archives and
     * returns the such archives as absolute paths.
     *
     * @param topPath the path to use as root of plugin search.
     * @return a list of potential plugin paths, or empty list if no such paths exist.
     * @throws IOException
     */
protected QJournalService createConnector() throws IOException {
    final Configuration confCopy = new Configuration(conf);

    // Need to set NODELAY or else batches larger than MTU can trigger
    // 40ms nailing delays.
    confCopy.setBoolean(CommonConfigurationKeysPublic.IPC_CLIENT_TCPNODELAY_KEY, true);
    RPC.setProtocolEngine(confCopy,
        QJournalServicePB.class, ProtobufRpcEngine3.class);
    return SecurityUtil.doAsLoginUser(
        (PrivilegedExceptionAction<QJournalService>) () -> {
          RPC.setProtocolEngine(confCopy,
              QJournalServicePB.class, ProtobufRpcEngine3.class);
          QJournalServicePB pbproxy = RPC.getProxy(
              QJournalServicePB.class,
              RPC.getProtocolVersion(QJournalServicePB.class),
              addr, confCopy);
          return new QJournalServiceTranslatorPB(pbproxy);
        });
  }

public long getClearedCount() {
    long clearedCount = 0;

    for (CBSection infoBlock : blockSections) {
      if (infoBlock.isCleared()) clearedCount++;
    }

    for (CBSection checkBlock : checkSections) {
      if (checkBlock.isCleared()) clearedCount++;
    }

    return clearedCount;
}

  void abort(Throwable t) throws IOException {
    LOG.info("Aborting because of " + StringUtils.stringifyException(t));
    try {
      downlink.abort();
      downlink.flush();
    } catch (IOException e) {
      // IGNORE cleanup problems
    }
    try {
      handler.waitForFinish();
    } catch (Throwable ignored) {
      process.destroy();
    }
    IOException wrapper = new IOException("pipe child exception");
    wrapper.initCause(t);
    throw wrapper;
  }

public void finalize() throws IOException {
    List<ListenableFuture<?>> futures = new ArrayList<>();
    for (AfsLease lease : leaseRefs.keySet()) {
      if (lease == null) {
        continue;
      }
      ListenableFuture<?> future = getProvider().submit(() -> lease.release());
      futures.add(future);
    }
    try {
      Futures.allAsList(futures).get();
      // shutdown the threadPool and set it to null.
      HadoopExecutors.shutdown(scheduledThreadPool, LOG,
          60, TimeUnit.SECONDS);
      scheduledThreadPool = null;
    } catch (InterruptedException e) {
      LOG.error("Interrupted releasing leases", e);
      Thread.currentThread().interrupt();
    } catch (ExecutionException e) {
      LOG.error("Error releasing leases", e);
    } finally {
      IOUtils.cleanupWithLogger(LOG, getProvider());
    }
  }

    /**
     * Return the simple class name of a plugin as {@code String}.
     *
     * @param plugin the plugin descriptor.
     * @return the plugin's simple class name.
     */
  protected void serviceStop() throws Exception {
    // Remove JMX interfaces
    if (this.rbfMetrics != null) {
      this.rbfMetrics.close();
    }

    // Remove Namenode JMX interfaces
    if (this.nnMetrics != null) {
      this.nnMetrics.close();
    }

    // Shutdown metrics
    if (this.routerMetrics != null) {
      this.routerMetrics.shutdown();
    }

    // Shutdown client metrics
    if (this.routerClientMetrics != null) {
      this.routerClientMetrics.shutdown();
    }
  }

    /**
     * Remove the plugin type name at the end of a plugin class name, if such suffix is present.
     * This method is meant to be used to extract plugin aliases.
     *
     * @param plugin the plugin descriptor.
     * @return the pruned simple class name of the plugin.
     */

public String retrieveHeadRoomAlpha() {
    boolean hasAlpha = this.viaProto ? this.proto.hasHeadRoomAlpha() : this.builder.hasHeadRoomAlpha();
    if (hasAlpha) {
        return this.viaProto ? this.proto.getHeadRoomAlpha() : this.builder.getHeadRoomAlpha();
    }
    return null;
}

private boolean monitorInfraApplication() throws YarnException, IOException {

    boolean success = false;
    boolean loggedApplicationInfo = false;

    Thread namenodeMonitoringThread = new Thread(() -> {
        Supplier<Boolean> exitCritera = () ->
            Apps.isApplicationFinalState(infraAppState);
        Optional<Properties> propertiesOpt = Optional.empty();
        while (!exitCritera.get()) {
            try {
                if (!propertiesOpt.isPresent()) {
                    propertiesOpt = DynoInfraUtils
                        .waitForAndGetNameNodeProperties(exitCritera, getConf(),
                            getNameNodeInfoPath(), LOG);
                    if (propertiesOpt.isPresent()) {
                        Properties props = propertiesOpt.get();
                        LOG.info("NameNode can be reached via HDFS at: {}",
                            DynoInfraUtils.getNameNodeHdfsUri(props));
                        LOG.info("NameNode web UI available at: {}",
                            DynoInfraUtils.getNameNodeWebUri(props));
                        LOG.info("NameNode can be tracked at: {}",
                            DynoInfraUtils.getNameNodeTrackingUri(props));
                    } else {
                        break;
                    }
                }
                DynoInfraUtils.waitForNameNodeStartup(propertiesOpt.get(),
                    exitCritera, LOG);
                DynoInfraUtils.waitForNameNodeReadiness(propertiesOpt.get(),
                    numTotalDataNodes, false, exitCritera, getConf(), LOG);
                break;
            } catch (IOException ioe) {
                LOG.error(
                    "Unexpected exception while waiting for NameNode readiness",
                    ioe);
            } catch (InterruptedException ie) {
                return;
            }
        }
        if (!Apps.isApplicationFinalState(infraAppState) && launchWorkloadJob) {
            launchAndMonitorWorkloadDriver(propertiesOpt.get());
        }
    });
    if (launchNameNode) {
        namenodeMonitoringThread.start();
    }

    while (true) {

        // Check app status every 5 seconds.
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            LOG.debug("Thread sleep in monitoring loop interrupted");
        }

        ApplicationReport report = yarnClient.getApplicationReport(infraAppId);

        if (!loggedApplicationInfo && report.getTrackingUrl() != null) {
            loggedApplicationInfo = true;
            LOG.info("Track the application at: " + report.getTrackingUrl());
            LOG.info("Kill the application using: yarn application -kill "
                + report.getApplicationId());
        }

        LOG.debug("Got application report from ASM for: appId={}, "
            + "clientToAMToken={}, appDiagnostics={}, appMasterHost={}, "
            + "appQueue={}, appMasterRpcPort={}, appStartTime={}, "
            + "yarnAppState={}, distributedFinalState={}, appTrackingUrl={}, "
            + "appUser={}",
            infraAppId.getId(), report.getClientToAMToken(),
            report.getDiagnostics(), report.getHost(), report.getQueue(),
            report.getRpcPort(), report.getStartTime(),
            report.getYarnApplicationState(), report.getFinalApplicationStatus(),
            report.getTrackingUrl(), report.getUser());

        infraAppState = report.getYarnApplicationState();
        if (infraAppState == YarnApplicationState.KILLED) {
            success = true;
            if (!launchWorkloadJob) break;
            else if (workloadJob == null) LOG.error("Infra app was killed before workload job was launched.");
            else if (!workloadJob.isComplete()) LOG.error("Infra app was killed before workload job completed.");
            else if (workloadJob.isSuccessful()) success = true;
            LOG.info("Infra app was killed; exiting from client.");
        } else if (infraAppState == YarnApplicationState.FINISHED
            || infraAppState == YarnApplicationState.FAILED) {
            LOG.info("Infra app exited unexpectedly. YarnState="
                + infraAppState.toString() + ". Exiting from client.");
            break;
        }

        if ((clientTimeout != -1)
            && (System.currentTimeMillis() > (clientStartTime + clientTimeout))) {
            attemptCleanup();
            return success;
        }
    }
    if (launchNameNode) {
        try {
            namenodeMonitoringThread.interrupt();
            namenodeMonitoringThread.join();
        } catch (InterruptedException ie) {
            LOG.warn("Interrupted while joining workload job thread; "
                + "continuing to cleanup.");
        }
    }
    attemptCleanup();
    return success;
}

    private static class DirectoryEntry {
        final DirectoryStream<Path> stream;
        final Iterator<Path> iterator;

        DirectoryEntry(DirectoryStream<Path> stream) {
            this.stream = stream;
            this.iterator = stream.iterator();
        }
    }

Node replaceRedWithExistingBlue(Map<Node, Node> oldNodes, Node newNode) {
		Node oldNode = oldNodes.get(newNode.getRoot());
		Node targetNode = oldNode == null ? newNode : oldNode;

		List children = new ArrayList();
		for (Node child : newNode.subNodes) {
			Node oldChild = replaceRedWithExistingBlue(oldNodes, child);
			children.add(oldChild);
			oldChild.parent = targetNode;
		}

		targetNode.subNodes = Collections.unmodifiableList(children);

		return targetNode;
	}

    protected final void initializeResources() {
        log.info("Initializing REST resources");

        ResourceConfig resourceConfig = newResourceConfig();
        Collection<Class<?>> regularResources = regularResources();
        regularResources.forEach(resourceConfig::register);
        configureRegularResources(resourceConfig);

        List<String> adminListeners = config.adminListeners();
        ResourceConfig adminResourceConfig;
        if (adminListeners != null && adminListeners.isEmpty()) {
            log.info("Skipping adding admin resources");
            // set up adminResource but add no handlers to it
            adminResourceConfig = resourceConfig;
        } else {
            if (adminListeners == null) {
                log.info("Adding admin resources to main listener");
                adminResourceConfig = resourceConfig;
            } else {
                // TODO: we need to check if these listeners are same as 'listeners'
                // TODO: the following code assumes that they are different
                log.info("Adding admin resources to admin listener");
                adminResourceConfig = newResourceConfig();
            }
            Collection<Class<?>> adminResources = adminResources();
            adminResources.forEach(adminResourceConfig::register);
            configureAdminResources(adminResourceConfig);
        }

        ServletContainer servletContainer = new ServletContainer(resourceConfig);
        ServletHolder servletHolder = new ServletHolder(servletContainer);
        List<Handler> contextHandlers = new ArrayList<>();

        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath("/");
        context.addServlet(servletHolder, "/*");
        contextHandlers.add(context);

        ServletContextHandler adminContext = null;
        if (adminResourceConfig != resourceConfig) {
            adminContext = new ServletContextHandler(ServletContextHandler.SESSIONS);
            ServletHolder adminServletHolder = new ServletHolder(new ServletContainer(adminResourceConfig));
            adminContext.setContextPath("/");
            adminContext.addServlet(adminServletHolder, "/*");
            adminContext.setVirtualHosts(List.of("@" + ADMIN_SERVER_CONNECTOR_NAME));
            contextHandlers.add(adminContext);
        }

        String allowedOrigins = config.allowedOrigins();
        if (!Utils.isBlank(allowedOrigins)) {
            CrossOriginHandler crossOriginHandler = new CrossOriginHandler();
            crossOriginHandler.setAllowedOriginPatterns(Set.of(allowedOrigins.split(",")));
            String allowedMethods = config.allowedMethods();
            if (!Utils.isBlank(allowedMethods)) {
                crossOriginHandler.setAllowedMethods(Set.of(allowedMethods.split(",")));
            }
            // Setting to true matches the previously used CrossOriginFilter
            crossOriginHandler.setDeliverPreflightRequests(true);
            context.insertHandler(crossOriginHandler);
        }

        String headerConfig = config.responseHeaders();
        if (!Utils.isBlank(headerConfig)) {
            configureHttpResponseHeaderFilter(context, headerConfig);
        }

        handlers.setHandlers(contextHandlers.toArray(new Handler[0]));
        try {
            context.start();
        } catch (Exception e) {
            throw new ConnectException("Unable to initialize REST resources", e);
        }

        if (adminResourceConfig != resourceConfig) {
            try {
                log.debug("Starting admin context");
                adminContext.start();
            } catch (Exception e) {
                throw new ConnectException("Unable to initialize Admin REST resources", e);
            }
        }

        log.info("REST resources initialized; server is started and ready to handle requests");
    }

  public void setDelegationTokenSeqNum(int seqNum) {
    Connection connection = null;
    try {
      connection = getConnection(false);
      FederationQueryRunner runner = new FederationQueryRunner();
      runner.updateSequenceTable(connection, YARN_ROUTER_SEQUENCE_NUM, seqNum);
    } catch (Exception e) {
      throw new RuntimeException("Could not update sequence table!!", e);
    } finally {
      // Return to the pool the CallableStatement
      try {
        FederationStateStoreUtils.returnToPool(LOG, null, connection);
      } catch (YarnException e) {
        LOG.error("close connection error.", e);
      }
    }
  }

public List<String> convertToFormattedDecimalList(final Collection<? extends Number> values, final Integer minLength, final Integer fractionDigits, final String separator) {
        if (values == null) {
            return null;
        }
        List<String> resultList = new ArrayList<>(values.size() + 2);
        for (final Number value : values) {
            final String formattedValue = formatDecimal(value, minLength, fractionDigits, separator);
            resultList.add(formattedValue);
        }
        return resultList;
    }

    private String formatDecimal(final Number number, final Integer minIntegerDigits, final Integer decimalPlaces, final String pointType) {
        // 实现格式化逻辑
        return null;
    }
}
