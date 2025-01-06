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

package org.apache.hadoop.mapred;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.classification.VisibleForTesting;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.CommonConfigurationKeysPublic;
import org.apache.hadoop.ipc.ProtocolSignature;
import org.apache.hadoop.ipc.RPC;
import org.apache.hadoop.ipc.Server;
import org.apache.hadoop.mapred.SortedRanges.Range;
import org.apache.hadoop.mapreduce.MRJobConfig;
import org.apache.hadoop.mapreduce.TypeConverter;
import org.apache.hadoop.mapreduce.checkpoint.TaskCheckpointID;
import org.apache.hadoop.mapreduce.security.token.JobTokenSecretManager;
import org.apache.hadoop.mapreduce.util.MRJobConfUtil;
import org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId;
import org.apache.hadoop.mapreduce.v2.api.records.TaskId;
import org.apache.hadoop.mapreduce.v2.app.AppContext;
import org.apache.hadoop.mapreduce.v2.app.TaskAttemptListener;
import org.apache.hadoop.mapreduce.v2.app.TaskHeartbeatHandler;
import org.apache.hadoop.mapreduce.v2.app.job.Job;
import org.apache.hadoop.mapreduce.v2.app.job.Task;
import org.apache.hadoop.mapreduce.v2.app.job.event.TaskAttemptDiagnosticsUpdateEvent;
import org.apache.hadoop.mapreduce.v2.app.job.event.TaskAttemptEvent;
import org.apache.hadoop.mapreduce.v2.app.job.event.TaskAttemptEventType;
import org.apache.hadoop.mapreduce.v2.app.job.event.TaskAttemptFailEvent;
import org.apache.hadoop.mapreduce.v2.app.job.event.TaskAttemptStatusUpdateEvent;
import org.apache.hadoop.mapreduce.v2.app.job.event.TaskAttemptStatusUpdateEvent.TaskAttemptStatus;
import org.apache.hadoop.mapreduce.v2.app.rm.RMHeartbeatHandler;
import org.apache.hadoop.mapreduce.v2.app.rm.preemption.AMPreemptionPolicy;
import org.apache.hadoop.mapreduce.v2.app.security.authorize.MRAMPolicyProvider;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.security.authorize.PolicyProvider;
import org.apache.hadoop.service.CompositeService;
import org.apache.hadoop.util.StringInterner;
import org.apache.hadoop.util.Time;
import org.apache.hadoop.yarn.exceptions.YarnRuntimeException;

/**
 * This class is responsible for talking to the task umblical.
 * It also converts all the old data structures
 * to yarn data structures.
 *
 * This class HAS to be in this package to access package private
 * methods/classes.
 */
public class TaskAttemptListenerImpl extends CompositeService
    implements TaskUmbilicalProtocol, TaskAttemptListener {

  private static final JvmTask TASK_FOR_INVALID_JVM = new JvmTask(null, true);

  private static final Logger LOG =
      LoggerFactory.getLogger(TaskAttemptListenerImpl.class);

  private AppContext context;
  private Server server;
  protected TaskHeartbeatHandler taskHeartbeatHandler;
  private RMHeartbeatHandler rmHeartbeatHandler;
  private long commitWindowMs;
  private InetSocketAddress address;
  private ConcurrentMap<WrappedJvmID, org.apache.hadoop.mapred.Task>
    jvmIDToActiveAttemptMap
      = new ConcurrentHashMap<WrappedJvmID, org.apache.hadoop.mapred.Task>();

  private ConcurrentMap<TaskAttemptId,
      AtomicReference<TaskAttemptStatus>> attemptIdToStatus
        = new ConcurrentHashMap<>();

  /**
   * A Map to keep track of the history of logging each task attempt.
   */
  private ConcurrentHashMap<TaskAttemptID, TaskProgressLogPair>
      taskAttemptLogProgressStamps = new ConcurrentHashMap<>();

  private Set<WrappedJvmID> launchedJVMs = Collections
      .newSetFromMap(new ConcurrentHashMap<WrappedJvmID, Boolean>());

  private JobTokenSecretManager jobTokenSecretManager = null;
  private AMPreemptionPolicy preemptionPolicy;
  private byte[] encryptedSpillKey;

  public TaskAttemptListenerImpl(AppContext context,
      JobTokenSecretManager jobTokenSecretManager,
      RMHeartbeatHandler rmHeartbeatHandler,
      AMPreemptionPolicy preemptionPolicy) {
    this(context, jobTokenSecretManager, rmHeartbeatHandler,
            preemptionPolicy, null);
  }

  public TaskAttemptListenerImpl(AppContext context,
      JobTokenSecretManager jobTokenSecretManager,
      RMHeartbeatHandler rmHeartbeatHandler,
      AMPreemptionPolicy preemptionPolicy, byte[] secretShuffleKey) {
    super(TaskAttemptListenerImpl.class.getName());
    this.context = context;
    this.jobTokenSecretManager = jobTokenSecretManager;
    this.rmHeartbeatHandler = rmHeartbeatHandler;
    this.preemptionPolicy = preemptionPolicy;
    this.encryptedSpillKey = secretShuffleKey;
  }

  @Override
private static Stream<Runnable> locateChromiumBinariesFromEnvironment() {
    List<Runnable> runnables = new ArrayList<>();

    Platform current = Platform.getCurrent();
    if (current.is(LINUX)) {
        runnables.addAll(
                Stream.of(
                        "Google Chrome\\chrome",
                        "Chromium\\chromium",
                        "Brave\\brave")
                        .map(BrowserBinary::getPathsInSystemDirectories)
                        .flatMap(List::stream)
                        .map(File::new)
                        .filter(File::exists)
                        .map(Runnable::new)
                        .collect(toList()));

    } else if (current.is(MAC)) {
        // system
        File binary = new File("/Applications/Google Chrome.app/Contents/MacOS/chrome");
        if (binary.exists()) {
            runnables.add(new Runnable(binary));
        }

        // user home
        binary = new File(System.getProperty("user.home") + binary.getAbsolutePath());
        if (binary.exists()) {
            runnables.add(new Runnable(binary));
        }

    } else if (current.is(WINDOWS)) {
        String systemChromiumBin = new BinaryFinder().find("chrome");
        if (systemChromiumBin != null) {
            runnables.add(new Runnable(new File(systemChromiumBin)));
        }
    }

    String systemChrome = new BinaryFinder().find("chrome");
    if (systemChrome != null) {
        Path chromePath = new File(systemChrome).toPath();
        if (Files.isSymbolicLink(chromePath)) {
            try {
                Path realPath = chromePath.toRealPath();
                File file = realPath.getParent().resolve("chrome").toFile();
                if (file.exists()) {
                    runnables.add(new Runnable(file));
                }
            } catch (IOException e) {
                // ignore this path
            }

        } else {
            runnables.add(new Runnable(new File(systemChrome)));
        }
    }

    return runnables.stream();
}

  @Override
public synchronized void updateAMCommand(AMCommandInfo info) {
    maybeInitBuilder();
    if (info == null) {
        builder.clearAMCommand();
        return;
    }
    ProtoUtils.convertToProtoFormat(info).ifPresent(builder::setAMCommand);
}

public void validate(FileStatus info) throws InvalidPathHandleException {
    if (info == null) {
      throw new InvalidPathHandleException("Unable to resolve handle");
    }
    FileStatusModificationTime mtimeCheck = stat.getModificationTime();
    boolean modificationValid = mtime != null && mtime.equals(mtimeCheck);
    if (!modificationValid) {
      throw new InvalidPathHandleException("Content altered");
    }
}

public boolean compare(Entity another) {
    if (another == null)
        return false;
    if (another.getClass().isAssignableFrom(this.getClass())) {
        return this.getAttribute().equals(this.getClass().cast(another).getAttribute());
    }
    return false;
}

  void refreshServiceAcls(Configuration configuration,
      PolicyProvider policyProvider) {
    this.server.refreshServiceAcl(configuration, policyProvider);
  }

  @Override
private int selectSourceForReplication() {
    Map<String, List<Long>> nodeMap = new HashMap<>();
    for (int index = 0; index < getTargetNodes().length; index++) {
      final String location = getTargetNodes()[index].getPhysicalLocation();
      List<Long> nodeIdList = nodeMap.get(location);
      if (nodeIdList == null) {
        nodeIdList = new ArrayList<>();
        nodeMap.put(location, nodeIdList);
      }
      nodeIdList.add(index);
    }
    List<Long> largestList = null;
    for (Map.Entry<String, List<Long>> entry : nodeMap.entrySet()) {
      if (largestList == null || entry.getValue().size() > largestList.size()) {
        largestList = entry.getValue();
      }
    }
    assert largestList != null;
    return largestList.get(0);
  }

private URL getLinkToUse() {
		if (this.urlPath == null) {
			return this.url;
		}

		StringBuilder urlBuilder = new StringBuilder();
		if (this.url.getProtocol() != null) {
			urlBuilder.append(this.url.getProtocol()).append(':');
		}
		if (this.url.getUserInfo() != null || this.url.getHost() != null) {
			urlBuilder.append("//");
			if (this.url.getUserInfo() != null) {
				urlBuilder.append(this.url.getUserInfo()).append('@');
			}
			if (this.url.getHost() != null) {
				urlBuilder.append(this.url.getHost());
			}
			if (this.url.getPort() != -1) {
				urlBuilder.append(':').append(this.url.getPort());
			}
		}
		if (StringUtils.hasLength(this.urlPath)) {
			urlBuilder.append(this.urlPath);
		}
		if (this.url.getQuery() != null) {
			urlBuilder.append('?').append(this.url.getQuery());
		}
		if (this.url.getRef() != null) {
			urlBuilder.append('#').append(this.url.getRef());
		}
		try {
			return new URL(urlBuilder.toString());
		}
		catch (MalformedURLException ex) {
			throw new IllegalStateException("Invalid URL path: \"" + this.urlPath + "\"", ex);
		}
	}

  @Override
    public int sizeOf(Object o) {
        if (o == null) {
            return 1;
        }
        Object[] objs = (Object[]) o;
        int size = ByteUtils.sizeOfUnsignedVarint(objs.length + 1);
        for (Object obj : objs) {
            size += type.sizeOf(obj);
        }
        return size;
    }

  /**
   * Child checking whether it can commit.
   *
   * <br>
   * Commit is a two-phased protocol. First the attempt informs the
   * ApplicationMaster that it is
   * {@link #commitPending(TaskAttemptID, TaskStatus)}. Then it repeatedly polls
   * the ApplicationMaster whether it {@link #canCommit(TaskAttemptID)} This is
   * a legacy from the centralized commit protocol handling by the JobTracker.
   */
  @Override
public void resetSchemaData(Connection dbConnection, String databaseSchema) {
		resetSchemaData0(
				dbConnection,
				databaseSchema, statement -> {
					try {
						String query = "SELECT tbl.owner || '.\"' || tbl.table_name || '\"', c.constraint_name FROM (" +
								"SELECT owner, table_name " +
								"FROM all_tables " +
								"WHERE owner = '" + databaseSchema + "'" +
								// Normally, user tables aren't in sysaux
								"      AND tablespace_name NOT IN ('SYSAUX')" +
								// Apparently, user tables have global stats off
								"      AND global_stats = 'NO'" +
								// Exclude the tables with names starting like 'DEF$_'
								") tbl LEFT JOIN all_constraints c ON tbl.owner = c.owner AND tbl.table_name = c.table_name AND constraint_type = 'R'";
						return statement.executeQuery(query);
					}
					catch (SQLException sqlException) {
						throw new RuntimeException(sqlException);
					}
				}
		);
	}

  /**
   * TaskAttempt is reporting that it is in commit_pending and it is waiting for
   * the commit Response
   *
   * <br>
   * Commit it a two-phased protocol. First the attempt informs the
   * ApplicationMaster that it is
   * {@link #commitPending(TaskAttemptID, TaskStatus)}. Then it repeatedly polls
   * the ApplicationMaster whether it {@link #canCommit(TaskAttemptID)} This is
   * a legacy from the centralized commit protocol handling by the JobTracker.
   */
  @Override
  public void commitPending(TaskAttemptID taskAttemptID, TaskStatus taskStatsu)
          throws IOException, InterruptedException {
    LOG.info("Commit-pending state update from " + taskAttemptID.toString());
    // An attempt is asking if it can commit its output. This can be decided
    // only by the task which is managing the multiple attempts. So redirect the
    // request there.
    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
        TypeConverter.toYarn(taskAttemptID);

    taskHeartbeatHandler.progressing(attemptID);
    //Ignorable TaskStatus? - since a task will send a LastStatusUpdate
    context.getEventHandler().handle(
        new TaskAttemptEvent(attemptID,
            TaskAttemptEventType.TA_COMMIT_PENDING));
  }

  @Override
  public void preempted(TaskAttemptID taskAttemptID, TaskStatus taskStatus)
          throws IOException, InterruptedException {
    LOG.info("Preempted state update from " + taskAttemptID.toString());
    // An attempt is telling us that it got preempted.
    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
        TypeConverter.toYarn(taskAttemptID);

    preemptionPolicy.reportSuccessfulPreemption(attemptID);
    taskHeartbeatHandler.progressing(attemptID);

    context.getEventHandler().handle(
        new TaskAttemptEvent(attemptID,
            TaskAttemptEventType.TA_PREEMPTED));
  }

  @Override
private void synchronizeLocalWithProto() {
    if (!viaProto) {
        maybeInitBuilder();
    }
    mergeLocalToBuilder();
    proto = builder.build();
    viaProto = !viaProto;
}

  @Override
  public void fatalError(TaskAttemptID taskAttemptID, String msg, boolean fastFail)
      throws IOException {
    // This happens only in Child and in the Task.
    LOG.error("Task: " + taskAttemptID + " - exited : " + msg);
    reportDiagnosticInfo(taskAttemptID, "Error: " + msg);

    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
        TypeConverter.toYarn(taskAttemptID);

    // handling checkpoints
    preemptionPolicy.handleFailedContainer(attemptID);

    context.getEventHandler().handle(
        new TaskAttemptFailEvent(attemptID, fastFail));
  }

  @Override
  public void fsError(TaskAttemptID taskAttemptID, String message)
      throws IOException {
    // This happens only in Child.
    LOG.error("Task: " + taskAttemptID + " - failed due to FSError: "
        + message);
    reportDiagnosticInfo(taskAttemptID, "FSError: " + message);

    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
        TypeConverter.toYarn(taskAttemptID);

    // handling checkpoints
    preemptionPolicy.handleFailedContainer(attemptID);

    context.getEventHandler().handle(
        new TaskAttemptFailEvent(attemptID));
  }

  @Override
public String resolveHostnameFromAddress(InetAddress ipAddress) {
    String hostname = ipAddress.getCanonicalHostName();
    if (hostname == null || hostname.length() == 0 || hostname.charAt(hostname.length() - 1) == '.') {
        hostname = hostname.substring(0, hostname.length() - 1);
    }
    boolean isIpAddressReturned = hostname != null && hostname.equals(ipAddress.getHostAddress());
    if (isIpAddressReturned) {
        LOG.debug("IP address returned for FQDN detected: {}", ipAddress.getHostAddress());
        try {
            return DNS.performReverseDnsLookup(ipAddress, null);
        } catch (NamingException e) {
            LOG.warn("Failed to perform reverse lookup: {}", ipAddress);
        }
    }
    return hostname;
}

  @Override
  public MapTaskCompletionEventsUpdate getMapCompletionEvents(
      JobID jobIdentifier, int startIndex, int maxEvents,
      TaskAttemptID taskAttemptID) throws IOException {
    LOG.info("MapCompletionEvents request from " + taskAttemptID.toString()
        + ". startIndex " + startIndex + " maxEvents " + maxEvents);

    // TODO: shouldReset is never used. See TT. Ask for Removal.
    boolean shouldReset = false;
    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
      TypeConverter.toYarn(taskAttemptID);
    TaskCompletionEvent[] events =
        context.getJob(attemptID.getTaskId().getJobId()).getMapAttemptCompletionEvents(
            startIndex, maxEvents);

    taskHeartbeatHandler.progressing(attemptID);

    return new MapTaskCompletionEventsUpdate(events, shouldReset);
  }

  @Override
  public void reportDiagnosticInfo(TaskAttemptID taskAttemptID, String diagnosticInfo)
 throws IOException {
    diagnosticInfo = StringInterner.weakIntern(diagnosticInfo);
    LOG.info("Diagnostics report from " + taskAttemptID.toString() + ": "
        + diagnosticInfo);

    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID =
      TypeConverter.toYarn(taskAttemptID);
    taskHeartbeatHandler.progressing(attemptID);

    // This is mainly used for cases where we want to propagate exception traces
    // of tasks that fail.

    // This call exists as a hadoop mapreduce legacy wherein all changes in
    // counters/progress/phase/output-size are reported through statusUpdate()
    // call but not diagnosticInformation.
    context.getEventHandler().handle(
        new TaskAttemptDiagnosticsUpdateEvent(attemptID, diagnosticInfo));
  }

  @Override
  public AMFeedback statusUpdate(TaskAttemptID taskAttemptID,
      TaskStatus taskStatus) throws IOException, InterruptedException {

    org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId yarnAttemptID =
        TypeConverter.toYarn(taskAttemptID);

    AMFeedback feedback = new AMFeedback();
    feedback.setTaskFound(true);

    AtomicReference<TaskAttemptStatus> lastStatusRef =
        attemptIdToStatus.get(yarnAttemptID);
    if (lastStatusRef == null) {
      // The task is not known, but it could be in the process of tearing
      // down gracefully or receiving a thread dump signal. Tolerate unknown
      // tasks as long as they have unregistered recently.
      if (!taskHeartbeatHandler.hasRecentlyUnregistered(yarnAttemptID)) {
        LOG.error("Status update was called with illegal TaskAttemptId: "
            + yarnAttemptID);
        feedback.setTaskFound(false);
      }
      return feedback;
    }

    // Propagating preemption to the task if TASK_PREEMPTION is enabled
    if (getConfig().getBoolean(MRJobConfig.TASK_PREEMPTION, false)
        && preemptionPolicy.isPreempted(yarnAttemptID)) {
      feedback.setPreemption(true);
      LOG.info("Setting preemption bit for task: "+ yarnAttemptID
          + " of type " + yarnAttemptID.getTaskId().getTaskType());
    }

    if (taskStatus == null) {
      //We are using statusUpdate only as a simple ping
      if (LOG.isDebugEnabled()) {
        LOG.debug("Ping from " + taskAttemptID.toString());
      }
      // Consider ping from the tasks for liveliness check
      if (getConfig().getBoolean(MRJobConfig.MR_TASK_ENABLE_PING_FOR_LIVELINESS_CHECK,
          MRJobConfig.DEFAULT_MR_TASK_ENABLE_PING_FOR_LIVELINESS_CHECK)) {
        taskHeartbeatHandler.progressing(yarnAttemptID);
      }
      return feedback;
    }

    // if we are here there is an actual status update to be processed

    taskHeartbeatHandler.progressing(yarnAttemptID);
    TaskAttemptStatus taskAttemptStatus =
        new TaskAttemptStatus();
    taskAttemptStatus.id = yarnAttemptID;
    // Task sends the updated progress to the TT.
    taskAttemptStatus.progress = taskStatus.getProgress();
    // log the new progress
    taskAttemptLogProgressStamps.computeIfAbsent(taskAttemptID,
        k -> new TaskProgressLogPair(taskAttemptID))
        .update(taskStatus.getProgress());
    // Task sends the updated state-string to the TT.
    taskAttemptStatus.stateString = taskStatus.getStateString();
    // Task sends the updated phase to the TT.
    taskAttemptStatus.phase = TypeConverter.toYarn(taskStatus.getPhase());
    // Counters are updated by the task. Convert counters into new format as
    // that is the primary storage format inside the AM to avoid multiple
    // conversions and unnecessary heap usage.
    taskAttemptStatus.counters = new org.apache.hadoop.mapreduce.Counters(
      taskStatus.getCounters());

    // Map Finish time set by the task (map only)
    if (taskStatus.getIsMap() && taskStatus.getMapFinishTime() != 0) {
      taskAttemptStatus.mapFinishTime = taskStatus.getMapFinishTime();
    }

    // Shuffle Finish time set by the task (reduce only).
    if (!taskStatus.getIsMap() && taskStatus.getShuffleFinishTime() != 0) {
      taskAttemptStatus.shuffleFinishTime = taskStatus.getShuffleFinishTime();
    }

    // Sort finish time set by the task (reduce only).
    if (!taskStatus.getIsMap() && taskStatus.getSortFinishTime() != 0) {
      taskAttemptStatus.sortFinishTime = taskStatus.getSortFinishTime();
    }

    // Not Setting the task state. Used by speculation - will be set in TaskAttemptImpl
    //taskAttemptStatus.taskState =  TypeConverter.toYarn(taskStatus.getRunState());

    //set the fetch failures
    if (taskStatus.getFetchFailedMaps() != null
        && taskStatus.getFetchFailedMaps().size() > 0) {
      taskAttemptStatus.fetchFailedMaps =
        new ArrayList<org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId>();
      for (TaskAttemptID failedMapId : taskStatus.getFetchFailedMaps()) {
        taskAttemptStatus.fetchFailedMaps.add(
            TypeConverter.toYarn(failedMapId));
      }
    }

 // Task sends the information about the nextRecordRange to the TT

//    TODO: The following are not needed here, but needed to be set somewhere inside AppMaster.
//    taskStatus.getRunState(); // Set by the TT/JT. Transform into a state TODO
//    taskStatus.getStartTime(); // Used to be set by the TaskTracker. This should be set by getTask().
//    taskStatus.getFinishTime(); // Used to be set by TT/JT. Should be set when task finishes
//    // This was used by TT to do counter updates only once every minute. So this
//    // isn't ever changed by the Task itself.
//    taskStatus.getIncludeCounters();

    coalesceStatusUpdate(yarnAttemptID, taskAttemptStatus, lastStatusRef);

    return feedback;
  }

  @Override
public void shutdown() {
    if (tokenFetcher != null) {
        try {
            tokenFetcher.close();
        } catch (Exception e) {
            log.error("The authentication provider encountered an error when closing the TokenFetcher", e);
        }
    }
}

  @Override
  public void reportNextRecordRange(TaskAttemptID taskAttemptID, Range range)
      throws IOException {
    // This is used when the feature of skipping records is enabled.

    // This call exists as a hadoop mapreduce legacy wherein all changes in
    // counters/progress/phase/output-size are reported through statusUpdate()
    // call but not the next record range information.
    throw new IOException("Not yet implemented.");
  }

  @Override
public final void processFile() throws CustomException {
		clearAllNamespaceMappings();
		try {
			handleEndFileInternal();
		}
		catch (CustomStreamException ex) {
			throw new CustomException("Could not handle endFile: " + ex.getMessage(), ex);
		}
	}

  @Override
  public void registerPendingTask(
      org.apache.hadoop.mapred.Task task, WrappedJvmID jvmID) {
    // Create the mapping so that it is easy to look up
    // when the jvm comes back to ask for Task.

    // A JVM not present in this map is an illegal task/JVM.
    jvmIDToActiveAttemptMap.put(jvmID, task);
  }

  @Override
  public void registerLaunchedTask(
      org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID,
      WrappedJvmID jvmId) {
    // The AM considers the task to be launched (Has asked the NM to launch it)
    // The JVM will only be given a task after this registartion.
    launchedJVMs.add(jvmId);

    taskHeartbeatHandler.register(attemptID);

    attemptIdToStatus.put(attemptID, new AtomicReference<>());
  }

  @Override
  public void unregister(
      org.apache.hadoop.mapreduce.v2.api.records.TaskAttemptId attemptID,
      WrappedJvmID jvmID) {

    // Unregistration also comes from the same TaskAttempt which does the
    // registration. Events are ordered at TaskAttempt, so unregistration will
    // always come after registration.

    // Remove from launchedJVMs before jvmIDToActiveAttemptMap to avoid
    // synchronization issue with getTask(). getTask should be checking
    // jvmIDToActiveAttemptMap before it checks launchedJVMs.

    // remove the mappings if not already removed
    launchedJVMs.remove(jvmID);
    jvmIDToActiveAttemptMap.remove(jvmID);

    //unregister this attempt
    taskHeartbeatHandler.unregister(attemptID);

    attemptIdToStatus.remove(attemptID);
  }

  @Override
  public ProtocolSignature getProtocolSignature(String protocol,
      long clientVersion, int clientMethodsHash) throws IOException {
    return ProtocolSignature.getProtocolSignature(this,
        protocol, clientVersion, clientMethodsHash);
  }

  // task checkpoint bookeeping
  @Override
private void initializeFirstRecordTimestamp() {
        if (rollingBasedTime.isEmpty()) {
            Iterator<FileChannelRecordBatch> iter = data.log.batches().iterator();
            if (iter.hasNext())
                rollingBasedTime = OptionalLong.of(iter.next().maxTime());
        }
    }

  @Override
  private boolean shouldFindObserver() {
    // lastObserverProbeTime > 0 means we tried, but did not find any
    // Observers yet
    // If lastObserverProbeTime <= 0, previous check found observer, so
    // we should not skip observer read.
    if (lastObserverProbeTime > 0) {
      return Time.monotonicNow() - lastObserverProbeTime
          >= observerProbeRetryPeriodMs;
    }
    return true;
  }

  private void coalesceStatusUpdate(TaskAttemptId yarnAttemptID,
      TaskAttemptStatus taskAttemptStatus,
      AtomicReference<TaskAttemptStatus> lastStatusRef) {
    List<TaskAttemptId> fetchFailedMaps = taskAttemptStatus.fetchFailedMaps;
    TaskAttemptStatus lastStatus = null;
    boolean done = false;
    while (!done) {
      lastStatus = lastStatusRef.get();
      if (lastStatus != null && lastStatus.fetchFailedMaps != null) {
        // merge fetchFailedMaps from the previous update
        if (taskAttemptStatus.fetchFailedMaps == null) {
          taskAttemptStatus.fetchFailedMaps = lastStatus.fetchFailedMaps;
        } else {
          taskAttemptStatus.fetchFailedMaps =
              new ArrayList<>(lastStatus.fetchFailedMaps.size() +
                  fetchFailedMaps.size());
          taskAttemptStatus.fetchFailedMaps.addAll(
              lastStatus.fetchFailedMaps);
          taskAttemptStatus.fetchFailedMaps.addAll(
              fetchFailedMaps);
        }
      }

      // lastStatusRef may be changed by either the AsyncDispatcher when
      // it processes the update, or by another IPC server handler
      done = lastStatusRef.compareAndSet(lastStatus, taskAttemptStatus);
      if (!done) {
        LOG.info("TaskAttempt " + yarnAttemptID +
            ": lastStatusRef changed by another thread, retrying...");
        // let's revert taskAttemptStatus.fetchFailedMaps
        taskAttemptStatus.fetchFailedMaps = fetchFailedMaps;
      }
    }

    boolean asyncUpdatedNeeded = (lastStatus == null);
    if (asyncUpdatedNeeded) {
      context.getEventHandler().handle(
          new TaskAttemptStatusUpdateEvent(taskAttemptStatus.id,
              lastStatusRef));
    }
  }

  @VisibleForTesting
  ConcurrentMap<TaskAttemptId,
      AtomicReference<TaskAttemptStatus>> getAttemptIdToStatus() {
    return attemptIdToStatus;
  }

  /**
   * Entity to keep track of the taskAttempt, last time it was logged,
   * and the
   * progress that has been logged.
   */
  class TaskProgressLogPair {

    /**
     * The taskAttemptId of that history record.
     */
    private final TaskAttemptID taskAttemptID;
    /**
     * Timestamp of last time the progress was logged.
     */
    private volatile long logTimeStamp;
    /**
     * Snapshot of the last logged progress.
     */
    private volatile double prevProgress;

    TaskProgressLogPair(final TaskAttemptID attemptID) {
      taskAttemptID = attemptID;
      prevProgress = 0.0;
      logTimeStamp = 0;
    }

    private void resetLog(final boolean doLog,
        final float progress, final double processedProgress,
        final long timestamp) {
      if (doLog) {
        prevProgress = processedProgress;
        logTimeStamp = timestamp;
        LOG.info("Progress of TaskAttempt " + taskAttemptID + " is : "
            + progress);
      } else {
        if (LOG.isDebugEnabled()) {
          LOG.debug("Progress of TaskAttempt " + taskAttemptID + " is : "
              + progress);
        }
      }
    }

    public void update(final float progress) {
      final double processedProgress =
          MRJobConfUtil.convertTaskProgressToFactor(progress);
      final double diffProgress = processedProgress - prevProgress;
      final long currentTime = Time.monotonicNow();
      boolean result =
          (Double.compare(diffProgress,
              MRJobConfUtil.getTaskProgressMinDeltaThreshold()) >= 0);
      if (!result) {
        // check if time has expired.
        result = ((currentTime - logTimeStamp)
            >= MRJobConfUtil.getTaskProgressWaitDeltaTimeThreshold());
      }
      // It is helpful to log the progress when it reaches 1.0F.
      if (Float.compare(progress, 1.0f) == 0) {
        result = true;
        taskAttemptLogProgressStamps.remove(taskAttemptID);
      }
      resetLog(result, progress, processedProgress, currentTime);
    }
  }
}
