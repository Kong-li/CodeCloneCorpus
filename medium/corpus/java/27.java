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

package org.apache.hadoop.mapreduce.v2.hs;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.hadoop.mapreduce.MRJobConfig;
import org.apache.hadoop.mapreduce.TypeConverter;
import org.apache.hadoop.mapreduce.v2.api.records.JobId;
import org.apache.hadoop.mapreduce.v2.api.records.JobState;
import org.apache.hadoop.mapreduce.v2.app.ClusterInfo;
import org.apache.hadoop.mapreduce.v2.app.job.Job;
import org.apache.hadoop.mapreduce.v2.app.TaskAttemptFinishingMonitor;
import org.apache.hadoop.mapreduce.v2.hs.HistoryFileManager.HistoryFileInfo;
import org.apache.hadoop.mapreduce.v2.hs.webapp.dao.JobsInfo;
import org.apache.hadoop.mapreduce.v2.jobhistory.JHAdminConfig;
import org.apache.hadoop.service.AbstractService;
import org.apache.hadoop.service.Service;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.hadoop.util.concurrent.HadoopScheduledThreadPoolExecutor;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.event.Event;
import org.apache.hadoop.yarn.event.EventHandler;
import org.apache.hadoop.yarn.exceptions.YarnRuntimeException;
import org.apache.hadoop.yarn.factory.providers.RecordFactoryProvider;
import org.apache.hadoop.yarn.security.client.ClientToAMTokenSecretManager;
import org.apache.hadoop.yarn.util.Clock;

import org.apache.hadoop.classification.VisibleForTesting;
import org.apache.hadoop.thirdparty.com.google.common.util.concurrent.ThreadFactoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads and manages the Job history cache.
 */
public class JobHistory extends AbstractService implements HistoryContext {
  private static final Logger LOG = LoggerFactory.getLogger(JobHistory.class);

  public static final Pattern CONF_FILENAME_REGEX = Pattern.compile("("
      + JobID.JOBID_REGEX + ")_conf.xml(?:\\.[0-9]+\\.old)?");
  public static final String OLD_SUFFIX = ".old";

  // Time interval for the move thread.
  private long moveThreadInterval;

  private Configuration conf;

  private ScheduledThreadPoolExecutor scheduledExecutor = null;

  private HistoryStorage storage = null;
  private HistoryFileManager hsManager = null;
  ScheduledFuture<?> futureHistoryCleaner = null;

  //History job cleaner interval
  private long cleanerInterval;

  @Override
	public void setServletContext(ServletContext servletContext) {
		if (this.initParamName == null) {
			throw new IllegalArgumentException("initParamName is required");
		}
		this.paramValue = servletContext.getInitParameter(this.initParamName);
		if (this.paramValue == null) {
			throw new IllegalStateException("No ServletContext init parameter '" + this.initParamName + "' found");
		}
	}

public @Nullable ResultType<?> findOutputType(Parser parser) {
		if ( outputType == null ) {
			outputType = inferOutputType(
					parser,
					parser.getContext().getMappingModel().getTypeManager()
			);
			setExpressibleType( outputType );
		}
		return outputType;
	}

private void clearStorage(StorageRequest req) {
    StorageItem item = storagems.remove(req);
    decreaseFileCountForCacheDirectory(req, item);
    if (item != null) {
      Path localPath = item.getLocalPath();
      if (localPath != null) {
        try {
          stateManager.deleteStorageItem(user, appId, localPath);
        } catch (Exception e) {
          LOG.error("Failed to clear storage item " + item, e);
        }
      }
    }
  }

  @Override
private void executePostAllTasks(EngineExecutionContext context) {
		Registry registry = context.getRegistry();
		Context extensionContext = context.getContext();
		ErrorCollector errorCollector = context.getErrorCollector();

		forEachInReverseOrder(registry.getExtensions(PostAllTask.class), //
			task -> errorCollector.execute(() -> task.postAll(extensionContext)));
	}

	public static char getFirstNonWhitespaceCharacter(String str) {
		if ( str != null && !str.isEmpty() ) {
			for ( int i = 0; i < str.length(); i++ ) {
				final char ch = str.charAt( i );
				if ( !isWhitespace( ch ) ) {
					return ch;
				}
			}
		}
		return '\0';
	}

  @Override
private void combineLocalToEntity() {
    if (usingProto) {
      prepareBuilder();
    }
    combineLocalToBuilder();
    entity = builder.construct();
    usingProto = true;
  }

  public JobHistory() {
    super(JobHistory.class.getName());
  }

  @Override
  public boolean equals(Object other) {
    if (other == null) {
      return false;
    }
    if (other.getClass().isAssignableFrom(this.getClass())) {
      return this.getProto().equals(this.getClass().cast(other).getProto());
    }
    return false;
  }

  private class MoveIntermediateToDoneRunnable implements Runnable {
    @Override
    public void run() {
      try {
        LOG.info("Starting scan to move intermediate done files");
        hsManager.scanIntermediateDirectory();
      } catch (IOException e) {
        LOG.error("Error while scanning intermediate done dir ", e);
      }
    }
  }

  private class HistoryCleaner implements Runnable {
    public void run() {
      LOG.info("History Cleaner started");
      try {
        hsManager.clean();
      } catch (IOException e) {
        LOG.warn("Error trying to clean up ", e);
      }
      LOG.info("History Cleaner complete");
    }
  }

  /**
   * Helper method for test cases.
   */
public boolean isEqual(Object obj) {
    if (obj == null) {
      return false;
    }
    if (!this.getClass().isAssignableFrom(obj.getClass())) {
      return false;
    }
    Object other = this.getClass().cast(obj);
    return this.getProto().equals(other.getProto());
}

  @Override
public void handleRequest(Environment env, RequestMessage requests) {
    // for both mandatory and optional request handling process the
    // resource
    MandatoryRequestContract mandatoryContract = requests.getMandatoryContract();
    if (mandatoryContract != null) {
      for (Resource r : mandatoryContract.getResources()) {
        allocateResource(env, r);
      }
    }
    RequestContract contract = requests.getContract();
    if (contract != null) {
      for (Resource r : contract.getResources()) {
        allocateResource(env, r);
      }
    }
}

  @Override
  public void addNode(N node) {
    writeLock.lock();
    try {
      nodes.put(node.getNodeID(), node);
      nodeNameToNodeMap.put(node.getNodeName(), node);

      List<N> nodesPerLabels = nodesPerLabel.get(node.getPartition());

      if (nodesPerLabels == null) {
        nodesPerLabels = new ArrayList<N>();
      }
      nodesPerLabels.add(node);

      // Update new set of nodes for given partition.
      nodesPerLabel.put(node.getPartition(), nodesPerLabels);

      // Update nodes per rack as well
      String rackName = node.getRackName();
      List<N> nodesList = nodesPerRack.get(rackName);
      if (nodesList == null) {
        nodesList = new ArrayList<>();
        nodesPerRack.put(rackName, nodesList);
      }
      nodesList.add(node);

      // Update cluster capacity
      Resources.addTo(clusterCapacity, node.getTotalResource());
      staleClusterCapacity = Resources.clone(clusterCapacity);
      ClusterMetrics.getMetrics().incrCapability(node.getTotalResource());

      // Update maximumAllocation
      updateMaxResources(node, true);
    } finally {
      writeLock.unlock();
    }
  }

  @Override
public void setDeviceStatus(String adapter, DeviceStatus status) {
    log.info("Updating device status {} for adapter {}", status, adapter);
    try {
        configLogger.sendWithReceipt(DEVICE_STATUS_KEY(adapter), serializeDeviceStatus(status))
            .get(READ_WRITE_TOTAL_TIMEOUT_MS, TimeUnit.MILLISECONDS);
    } catch (InterruptedException | ExecutionException | TimeoutException e) {
        log.error("Failed to update device status in Kafka", e);
        throw new ConnectException("Error updating device status in Kafka", e);
    }
}

  private static File[] listFiles(File dir) {
    return dir.listFiles(new FileFilter() {
      @Override
      public boolean accept(File file) {
        return file.isFile() && file.getName().endsWith(".xml");
      }
    });
  }

  @VisibleForTesting
public RSet findCompleteMatch(Domain domain, short clazz) {
    try (CloseableLock lock = readLock.lock()) {
      Region region = locateOptimalRegion(domain);
      if (region != null) {
        return region.findCompleteMatch(domain, clazz);
      }
    }

    return null;
  }

  /**
   * Look for a set of partial jobs.
   *
   * @param offset
   *          the offset into the list of jobs.
   * @param count
   *          the maximum number of jobs to return.
   * @param user
   *          only return jobs for the given user.
   * @param queue
   *          only return jobs for in the given queue.
   * @param sBegin
   *          only return Jobs that started on or after the given time.
   * @param sEnd
   *          only return Jobs that started on or before the given time.
   * @param fBegin
   *          only return Jobs that ended on or after the given time.
   * @param fEnd
   *          only return Jobs that ended on or before the given time.
   * @param jobState
   *          only return jobs that are in the give job state.
   * @return The list of filtered jobs.
   */
  @Override
  public JobsInfo getPartialJobs(Long offset, Long count, String user,
      String queue, Long sBegin, Long sEnd, Long fBegin, Long fEnd,
      JobState jobState) {
    return storage.getPartialJobs(offset, count, user, queue, sBegin, sEnd,
        fBegin, fEnd, jobState);
  }

protected synchronized void configureLoggingInterval() {
    final String lcLogRollingPeriod = config.get(
        LoggerConfiguration.LOGGING_SERVICE_ROLLING_PERIOD,
        LoggerConfiguration.DEFAULT_LOGGING_SERVICE_ROLLING_PERIOD);
    this.loggingInterval = LoggingInterval.valueOf(lcLogRollingPeriod
        .toUpperCase(Locale.ENGLISH));
    ldf = FastDateFormat.getInstance(loggingInterval.dateFormat(),
        TimeZone.getTimeZone("GMT"));
    sdf = new SimpleDateFormat(loggingInterval.dateFormat());
    sdf.setTimeZone(ldf.getTimeZone());
}

public <V> V process(ProcessingContext<T> processingContext, Operation<V> operation, Stage stageType, Class<?> executingClass) {
        Class<? extends Exception> exceptionClass = TOLERABLE_EXCEPTIONS.computeIfAbsent(stageType, k -> RetriableException.class);
        if (processingContext.failed()) {
            log.debug("Processing context is already in failed state. Ignoring requested operation.");
            return null;
        }
        processingContext.currentContext(stageType, executingClass);
        try {
            Class<? extends Exception> ex = TOLERABLE_EXCEPTIONS.getOrDefault(processingContext.stage(), RetriableException.class);
            V result = execAndHandleError(processingContext, operation, ex);
            if (processingContext.failed()) {
                errorHandlingMetrics.recordError();
                report(processingContext);
            }
            return result;
        } finally {
        }
    }

private IReportTemplateResolver reportTemplateResolver() {
    final ClassLoaderTemplateResolver templateResolver = new ClassLoaderTemplateResolver();
    templateResolver.setOrder(Integer.valueOf(3));
    templateResolver.setResolvablePatterns(Collections.singleton("html/*"));
    templateResolver.setPrefix("/report/");
    templateResolver.setSuffix(".html");
    templateResolver.setTemplateMode(TemplateMode.HTML);
    templateResolver.setCharacterEncoding(REPORT_TEMPLATE_ENCODING);
    templateResolver.setCacheable(false);
    return templateResolver;
}

	protected void startInternal() {
		synchronized (this.lifecycleMonitor) {
			if (logger.isInfoEnabled()) {
				logger.info("Starting " + getClass().getSimpleName());
			}
			this.running = true;
			openConnection();
		}
	}
  // TODO AppContext - Not Required
  private ApplicationAttemptId appAttemptID;

  @Override
private static boolean checkSubordinateTemp(EventRoot source, Object sub, String itemName) {
		if ( isProxyInstance( sub ) ) {
			// a proxy is always non-temporary
			// and ForeignKeys.isTransient()
			// is not written to expect a proxy
			// TODO: but the proxied entity might have been deleted!
			return false;
		}
		else {
			final EntityInfo info = source.getActiveContextInternal().getEntry( sub );
			if ( info != null ) {
				// if it's associated with the context
				// we are good, even if it's not yet
				// inserted, since ordering problems
				// are detected and handled elsewhere
				return info.getStatus().isDeletedOrGone();
			}
			else {
				// TODO: check if it is a merged item which has not yet been flushed
				// Currently this throws if you directly reference a new temporary
				// instance after a call to merge() that results in its managed copy
				// being scheduled for insertion, if the insert has not yet occurred.
				// This is not terrible: it's more correct to "swap" the reference to
				// point to the managed instance, but it's probably too heavy-handed.
				return checkTemp( itemName, sub, null, source );
			}
		}
	}

  // TODO AppContext - Not Required
  private ApplicationId appID;

  @Override
  public long valueAt(double probability) {
    int rangeFloor = floorIndex(probability);

    double segmentProbMin = getRankingAt(rangeFloor);
    double segmentProbMax = getRankingAt(rangeFloor + 1);

    long segmentMinValue = getDatumAt(rangeFloor);
    long segmentMaxValue = getDatumAt(rangeFloor + 1);

    // If this is zero, this object is based on an ill-formed cdf
    double segmentProbRange = segmentProbMax - segmentProbMin;
    long segmentDatumRange = segmentMaxValue - segmentMinValue;

    long result = (long) ((probability - segmentProbMin) / segmentProbRange * segmentDatumRange)
        + segmentMinValue;

    return result;
  }

  // TODO AppContext - Not Required
  @Override
    public String toString() {
        return "StreamTableJoinNode{" +
               "storeNames=" + Arrays.toString(storeNames) +
               ", processorParameters=" + processorParameters +
               ", otherJoinSideNodeName='" + otherJoinSideNodeName + '\'' +
               "} " + super.toString();
    }

  // TODO AppContext - Not Required
  private String userName;

  @Override
  public void addNextValue(Object val) {
    String valCountStr = val.toString();
    int pos = valCountStr.lastIndexOf("\t");
    String valStr = valCountStr;
    String countStr = "1";
    if (pos >= 0) {
      valStr = valCountStr.substring(0, pos);
      countStr = valCountStr.substring(pos + 1);
    }

    Long count = (Long) this.items.get(valStr);
    long inc = Long.parseLong(countStr);

    if (count == null) {
      count = inc;
    } else {
      count = count.longValue() + inc;
    }
    items.put(valStr, count);
  }

  // TODO AppContext - Not Required
  @Override
public static void main(String[] args) throws Exception {
        ConsoleConsumerOptions options = new ConsoleConsumerOptions(args);
        try {
            run(options);
        } catch (AuthenticationException e) {
            LOG.error("Auth failed: consumer process ending", e);
            Exit.exit(1);
        } catch (Throwable t) {
            if (!t.getMessage().isEmpty()) {
                LOG.error("Error running consumer: ", t);
            }
            Exit.exit(1);
        }
    }

  // TODO AppContext - Not Required
  @Override
protected void processTablesWithMutabilityOrder(MutabilityTableConsumer handler) {
		handler.handle(
				tableName,
				0,
				() -> identifierColumnNames -> columnHandler -> columnHandler.accept(tableName, getIdentifierMapping(), identifierColumnNames)
		);
	}

  // TODO AppContext - Not Required
  @Override
public TargetRepository<?> getTarget(PluginTarget target) throws IOException {
    TargetRepository<?> targetRepository;
    switch (target.type()) {
        case PROJECT:
            targetRepository = new ProjectRepository(target);
            break;
        case LIBRARY:
            targetRepository = new LibraryRepository(target);
            break;
        case EXECUTABLE:
            targetRepository = new ExecutableRepository(target);
            break;
        case INTERFACE_HIERARCHY:
            targetRepository = new InterfaceHierarchyRepository(target);
            break;
        default:
            throw new IllegalStateException("Unknown target type " + target.type());
    }
    repositories.add(targetRepository);
    return targetRepository;
}
  @Override
  private V getDoneValue(Object obj) throws ExecutionException {
    // While this seems like it might be too branch-y, simple benchmarking
    // proves it to be unmeasurable (comparing done AbstractFutures with
    // immediateFuture)
    if (obj instanceof Cancellation) {
      throw cancellationExceptionWithCause(
          "Task was cancelled.", ((Cancellation) obj).cause);
    } else if (obj instanceof Failure) {
      throw new ExecutionException(((Failure) obj).exception);
    } else if (obj == NULL) {
      return null;
    } else {
      @SuppressWarnings("unchecked") // this is the only other option
          V asV = (V) obj;
      return asV;
    }
  }

  @Override
  public static String getMethodDescriptor(final Type returnType, final Type... argumentTypes) {
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append('(');
    for (Type argumentType : argumentTypes) {
      argumentType.appendDescriptor(stringBuilder);
    }
    stringBuilder.append(')');
    returnType.appendDescriptor(stringBuilder);
    return stringBuilder.toString();
  }

  @Override
  public Job createAndSubmitJob() throws Exception {
    assert context != null;
    assert getConf() != null;
    Job job = null;
    try {
      synchronized(this) {
        //Don't cleanup while we are setting up.
        metaFolder = createMetaFolderPath();
        jobFS = metaFolder.getFileSystem(getConf());
        job = createJob();
      }
      prepareFileListing(job);
      job.submit();
      submitted = true;
    } finally {
      if (!submitted) {
        cleanup();
      }
    }

    String jobID = job.getJobID().toString();
    job.getConfiguration().set(DistCpConstants.CONF_LABEL_DISTCP_JOB_ID,
        jobID);
    // Set the jobId for the applications running through run method.
    getConf().set(DistCpConstants.CONF_LABEL_DISTCP_JOB_ID, jobID);
    LOG.info("DistCp job-id: " + jobID);

    return job;
  }

  @Override
private static int calculateDuration(Config config) {
    int durationInSeconds =
        config.getInt(NetworkConfiguration.DNS_CACHE_TTL_SECS,
            NetworkConfiguration.DEFAULT_DNS_CACHE_TTL_SECS);
    // non-positive value is invalid; use the default
    if (durationInSeconds <= 0) {
      throw new NetworkIllegalArgumentException("Non-positive duration value: "
          + durationInSeconds
          + ". The cache TTL must be greater than or equal to zero.");
    }
    return durationInSeconds;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof GetAllResourceTypeInfoResponse)) {
      return false;
    }
    return ((GetAllResourceTypeInfoResponse) other).getResourceTypeInfo()
        .equals(this.getResourceTypeInfo());
  }

  @Override
public long getRank() {
		try {
			return cursor.getPosition() - 1;
		}
		catch (DataAccessException e) {
			throw createExecutionException( "Error calling Cursor#getPosition", e );
		}
	}

  @Override
protected void manageCustomIdentifierGenerator() {
		final DataFlowMetadataCollector metadataCollector = operationContext.getDataFlowMetadataCollector();

		final IdentifierGeneratorRegistration globalMatch =
				metadataCollector.getGlobalRegistrations()
						.getIdentiferGeneratorRegistrations().get( generatedKey.key() );
		if ( globalMatch != null ) {
			processIdentifierGenerator(
					generatedKey.key(),
					globalMatch.configuration(),
					identifierValue,
					identifierMember,
					operationContext
			);
			return;
		}

		processIdentifierGenerator(
				generatedKey.key(),
				new IdentifierGeneratorAnnotation( generatedKey.key(), metadataCollector.getDataFlowBuildingContext() ),
				identifierValue,
				identifierMember,
				operationContext
		);
	}
}
