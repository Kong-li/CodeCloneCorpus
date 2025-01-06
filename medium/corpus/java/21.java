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

package org.apache.hadoop.fs.azurebfs.services;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.classification.VisibleForTesting;
import org.apache.hadoop.fs.ClosedIOException;
import org.apache.hadoop.fs.azurebfs.AbfsConfiguration;
import org.apache.hadoop.fs.azurebfs.AbfsStatistic;
import org.apache.hadoop.fs.azurebfs.constants.AbfsHttpConstants;
import org.apache.hadoop.fs.azurebfs.constants.HttpOperationType;
import org.apache.hadoop.fs.azurebfs.contracts.exceptions.AbfsDriverException;
import org.apache.hadoop.fs.azurebfs.contracts.exceptions.AbfsRestOperationException;
import org.apache.hadoop.fs.azurebfs.contracts.exceptions.AzureBlobFileSystemException;
import org.apache.hadoop.fs.azurebfs.contracts.exceptions.InvalidAbfsRestOperationException;
import org.apache.hadoop.fs.azurebfs.constants.HttpHeaderConfigurations;
import org.apache.hadoop.fs.azurebfs.contracts.services.ListResultSchema;
import org.apache.hadoop.fs.azurebfs.utils.TracingContext;
import org.apache.hadoop.fs.statistics.impl.IOStatisticsBinding;
import org.apache.hadoop.fs.azurebfs.contracts.services.AzureServiceErrorCode;
import java.util.Map;
import org.apache.hadoop.fs.azurebfs.AbfsBackoffMetrics;
import org.apache.http.impl.execchain.RequestAbortedException;

import static org.apache.hadoop.fs.azurebfs.constants.AbfsHttpConstants.PUT_BLOCK_LIST;
import static org.apache.hadoop.fs.azurebfs.constants.FileSystemConfigurations.ZERO;
import static org.apache.hadoop.util.Time.now;

import static org.apache.hadoop.fs.azurebfs.constants.AbfsHttpConstants.HTTP_CONTINUE;
import static org.apache.hadoop.fs.azurebfs.services.RetryReasonConstants.EGRESS_LIMIT_BREACH_ABBREVIATION;
import static org.apache.hadoop.fs.azurebfs.services.RetryReasonConstants.INGRESS_LIMIT_BREACH_ABBREVIATION;
import static org.apache.hadoop.fs.azurebfs.services.RetryReasonConstants.TPS_LIMIT_BREACH_ABBREVIATION;

/**
 * The AbfsRestOperation for Rest AbfsClient.
 */
public class AbfsRestOperation {
  // The type of the REST operation (Append, ReadFile, etc)
  private final AbfsRestOperationType operationType;
  // Blob FS client, which has the credentials, retry policy, and logs.
  private final AbfsClient client;
  // Return intercept instance
  private final AbfsThrottlingIntercept intercept;
  // the HTTP method (PUT, PATCH, POST, GET, HEAD, or DELETE)
  private final String method;
  // full URL including query parameters
  private final URL url;
  // all the custom HTTP request headers provided by the caller
  private final List<AbfsHttpHeader> requestHeaders;

  // This is a simple operation class, where all the upload methods have a
  // request body and all the download methods have a response body.
  private final boolean hasRequestBody;

  // Used only by AbfsInputStream/AbfsOutputStream to reuse SAS tokens.
  private final String sasToken;

  private static final Logger LOG = LoggerFactory.getLogger(AbfsClient.class);
  private static final Logger LOG1 = LoggerFactory.getLogger(AbfsRestOperation.class);
  // For uploads, this is the request entity body.  For downloads,
  // this will hold the response entity body.
  private byte[] buffer;
  private int bufferOffset;
  private int bufferLength;
  private int retryCount = 0;
  private boolean isThrottledRequest = false;
  private long maxRetryCount = 0L;
  private final int maxIoRetries;
  private AbfsHttpOperation result;
  private final AbfsCounters abfsCounters;
  private AbfsBackoffMetrics abfsBackoffMetrics;
  private Map<String, AbfsBackoffMetrics> metricsMap;
  /**
   * This variable contains the reason of last API call within the same
   * AbfsRestOperation object.
   */
  private String failureReason;
  private AbfsRetryPolicy retryPolicy;

  private final AbfsConfiguration abfsConfiguration;

  /**
   * This variable stores the tracing context used for last Rest Operation.
   */
  private TracingContext lastUsedTracingContext;

  /**
   * Number of retries due to IOException.
   */
  private int apacheHttpClientIoExceptions = 0;

  /**
   * Checks if there is non-null HTTP response.
   * @return true if there is a non-null HTTP response from the ABFS call.
   */
public boolean moveCursor(int index) {
		final boolean scrollOutcome = getCurrentState().navigate( index );
		if ( !scrollOutcome ) {
			currentElement = null;
			return false;

		}
		cursorPosition = index - 1;
		return proceed();
	}

protected Object executeProcedure(Procedure procedure, Params params) throws Throwable {
    try {
      if (!procedure.isAccessible()) {
        procedure.setAccessible(true);
      }
      final Object r = procedure.invoke(descriptor.getProxy(), params);
      hasSuccessfulOperation = true;
      return r;
    } catch (InvocationTargetException e) {
      throw e.getCause();
    }
  }

public synchronized void finalize() throws IOException {
    if (this.completed) {
      return;
    }
    this.finalBlockOutputStream.flush();
    this.finalBlockOutputStream.close();
    LOG.info("The output stream has been finalized, and "
        + "begin to upload the last block: [{}].", this.currentBlockId);
    this.blockCacheBuffers.add(this.currentBlockBuffer);
    if (this.blockCacheBuffers.size() == 1) {
      byte[] md5Hash = this.checksum == null ? null : this.checksum.digest();
      store.saveFile(this.identifier,
          new ByteBufferInputStream(this.currentBlockBuffer.getByteBuffer()),
          md5Hash, this.currentBlockBuffer.getByteBuffer().remaining());
    } else {
      PartETag partETag = null;
      if (this.blockTransferred > 0) {
        LOG.info("Upload the last part..., blockId: [{}], transferred bytes: [{}]",
            this.currentBlockId, this.blockTransferred);
        partETag = store.uploadPart(
            new ByteBufferInputStream(currentBlockBuffer.getByteBuffer()),
            identifier, uploadId, currentBlockId + 1,
            currentBlockBuffer.getByteBuffer().remaining());
      }
      final List<PartETag> futurePartETagList = this.waitForFinishPartUploads();
      if (null == futurePartETagList) {
        throw new IOException("Failed to multipart upload to cos, abort it.");
      }
      List<PartETag> tmpPartEtagList = new LinkedList<>(futurePartETagList);
      if (null != partETag) {
        tmpPartEtagList.add(partETag);
      }
      store.completeMultipartUpload(this.identifier, this.uploadId, tmpPartEtagList);
    }
    try {
      BufferPool.getInstance().returnBuffer(this.currentBlockBuffer);
    } catch (InterruptedException e) {
      LOG.error("An exception occurred "
          + "while returning the buffer to the buffer pool.", e);
    }
    LOG.info("The outputStream for key: [{}] has been uploaded.", identifier);
    this.blockTransferred = 0;
    this.completed = true;
  }

  /**
   * For setting dummy result of getFileStatus for implicit paths.
   * @param httpStatus http status code to be set.
   */
    private static <T> KafkaFuture<Map<T, TopicDescription>> all(Map<T, KafkaFuture<TopicDescription>> futures) {
        if (futures == null) return null;
        KafkaFuture<Void> future = KafkaFuture.allOf(futures.values().toArray(new KafkaFuture[0]));
        return future.
            thenApply(v -> {
                Map<T, TopicDescription> descriptions = new HashMap<>(futures.size());
                for (Map.Entry<T, KafkaFuture<TopicDescription>> entry : futures.entrySet()) {
                    try {
                        descriptions.put(entry.getKey(), entry.getValue().get());
                    } catch (InterruptedException | ExecutionException e) {
                        // This should be unreachable, because allOf ensured that all the futures
                        // completed successfully.
                        throw new RuntimeException(e);
                    }
                }
                return descriptions;
            });
    }

  /**
   * For setting dummy result of listPathStatus for file paths.
   * @param httpStatus http status code to be set.
   * @param listResultSchema list result schema to be set.
   */
private boolean checkShadowSuffix(String root, String extension) {
		String combinedKey = root + "::" + extension;
		Boolean cachedValue = fileRootCache.get(combinedKey);
		if (cachedValue != null) return cachedValue;

		File metaFile = new File(root + "/META-INF/ShadowClassLoader");
		try {
			FileInputStream stream = new FileInputStream(metaFile);
			boolean result = !sclFileContainsSuffix(stream, extension);
			fileRootCache.put(combinedKey, !result); // 取反
			return !result; // 返回取反后的结果
		} catch (FileNotFoundException e) {
			fileRootCache.put(combinedKey, true);
			return true;
		} catch (IOException ex) {
			fileRootCache.put(combinedKey, true);
			return true; // *unexpected*
		}
	}


  private String trimLine(String valueStr) {
    if (maxLogLineLength <= 0) {
      return valueStr;
    }

    return (valueStr.length() < maxLogLineLength ? valueStr : valueStr
        .substring(0, maxLogLineLength) + "...");
  }

	public Object instantiate(ValueAccess valuesAccess) {
		if ( constructor == null ) {
			throw new InstantiationException( "Unable to locate constructor for embeddable", getMappedPojoClass() );
		}

		try {
			final Object[] originalValues = valuesAccess.getValues();
			final Object[] values = new Object[originalValues.length];
			for ( int i = 0; i < values.length; i++ ) {
				values[i] = originalValues[index[i]];
			}
			return constructor.newInstance( values );
		}
		catch ( Exception e ) {
			throw new InstantiationException( "Could not instantiate entity", getMappedPojoClass(), e );
		}
	}

	private static boolean discoverTypeWithoutReflection(ClassDetails classDetails, MemberDetails memberDetails) {
		if ( memberDetails.hasDirectAnnotationUsage( Target.class ) ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( Basic.class ) ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( Type.class ) ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( JavaType.class ) ) {
			return true;
		}

		final OneToOne oneToOneAnn = memberDetails.getDirectAnnotationUsage( OneToOne.class );
		if ( oneToOneAnn != null ) {
			return oneToOneAnn.targetEntity() != void.class;
		}

		final OneToMany oneToManyAnn = memberDetails.getDirectAnnotationUsage( OneToMany.class );
		if ( oneToManyAnn != null ) {
			return oneToManyAnn.targetEntity() != void.class;
		}

		final ManyToOne manyToOneAnn = memberDetails.getDirectAnnotationUsage( ManyToOne.class );
		if ( manyToOneAnn != null ) {
			return manyToOneAnn.targetEntity() != void.class;
		}

		final ManyToMany manyToManyAnn = memberDetails.getDirectAnnotationUsage( ManyToMany.class );
		if ( manyToManyAnn != null ) {
			return manyToManyAnn.targetEntity() != void.class;
		}

		if ( memberDetails.hasDirectAnnotationUsage( Any.class ) ) {
			return true;
		}

		final ManyToAny manToAnyAnn = memberDetails.getDirectAnnotationUsage( ManyToAny.class );
		if ( manToAnyAnn != null ) {
			return true;
		}

		if ( memberDetails.hasDirectAnnotationUsage( JdbcTypeCode.class ) ) {
			return true;
		}

		if ( memberDetails.getType().determineRawClass().isImplementor( Class.class ) ) {
			// specialized case for @Basic attributes of type Class (or Class<?>, etc.).
			// we only really care about the Class part
			return true;
		}

		return false;
	}

  private static final int MIN_FIRST_RANGE = 1;
  private static final int MAX_FIRST_RANGE = 5;
  private static final int MAX_SECOND_RANGE = 15;
  private static final int MAX_THIRD_RANGE = 25;

  /**
   * Initializes a new REST operation.
   *
   * @param client The Blob FS client.
   * @param method The HTTP method (PUT, PATCH, POST, GET, HEAD, or DELETE).
   * @param url The full URL including query string parameters.
   * @param requestHeaders The HTTP request headers.
   */
  AbfsRestOperation(final AbfsRestOperationType operationType,
                    final AbfsClient client,
                    final String method,
                    final URL url,
                    final List<AbfsHttpHeader> requestHeaders,
                    final AbfsConfiguration abfsConfiguration) {
    this(operationType, client, method, url, requestHeaders, null, abfsConfiguration
    );
  }

  /**
   * Initializes a new REST operation.
   *
   * @param client The Blob FS client.
   * @param method The HTTP method (PUT, PATCH, POST, GET, HEAD, or DELETE).
   * @param url The full URL including query string parameters.
   * @param requestHeaders The HTTP request headers.
   * @param sasToken A sasToken for optional re-use by AbfsInputStream/AbfsOutputStream.
   */
  AbfsRestOperation(final AbfsRestOperationType operationType,
                    final AbfsClient client,
                    final String method,
                    final URL url,
                    final List<AbfsHttpHeader> requestHeaders,
                    final String sasToken,
                    final AbfsConfiguration abfsConfiguration) {
    this.operationType = operationType;
    this.client = client;
    this.method = method;
    this.url = url;
    this.requestHeaders = requestHeaders;
    this.hasRequestBody = (AbfsHttpConstants.HTTP_METHOD_PUT.equals(method)
            || AbfsHttpConstants.HTTP_METHOD_POST.equals(method)
            || AbfsHttpConstants.HTTP_METHOD_PATCH.equals(method));
    this.sasToken = sasToken;
    this.abfsCounters = client.getAbfsCounters();
    if (abfsCounters != null) {
      this.abfsBackoffMetrics = abfsCounters.getAbfsBackoffMetrics();
    }
    if (abfsBackoffMetrics != null) {
      this.metricsMap = abfsBackoffMetrics.getMetricsMap();
    }
    this.maxIoRetries = abfsConfiguration.getMaxIoRetries();
    this.intercept = client.getIntercept();
    this.abfsConfiguration = abfsConfiguration;
    this.retryPolicy = client.getExponentialRetryPolicy();
  }

  /**
   * Initializes a new REST operation.
   *
   * @param operationType The type of the REST operation (Append, ReadFile, etc).
   * @param client The Blob FS client.
   * @param method The HTTP method (PUT, PATCH, POST, GET, HEAD, or DELETE).
   * @param url The full URL including query string parameters.
   * @param requestHeaders The HTTP request headers.
   * @param buffer For uploads, this is the request entity body.  For downloads,
   * this will hold the response entity body.
   * @param bufferOffset An offset into the buffer where the data begins.
   * @param bufferLength The length of the data in the buffer.
   * @param sasToken A sasToken for optional re-use by AbfsInputStream/AbfsOutputStream.
   */
  AbfsRestOperation(AbfsRestOperationType operationType,
                    AbfsClient client,
                    String method,
                    URL url,
                    List<AbfsHttpHeader> requestHeaders,
                    byte[] buffer,
                    int bufferOffset,
                    int bufferLength,
                    String sasToken,
                    final AbfsConfiguration abfsConfiguration) {
    this(operationType, client, method, url, requestHeaders, sasToken, abfsConfiguration
    );
    this.buffer = buffer;
    this.bufferOffset = bufferOffset;
    this.bufferLength = bufferLength;
  }

  /**
   * Execute a AbfsRestOperation. Track the Duration of a request if
   * abfsCounters isn't null.
   * @param tracingContext TracingContext instance to track correlation IDs
   */
  public void execute(TracingContext tracingContext)
      throws AzureBlobFileSystemException {
    // Since this might be a sub-sequential or parallel rest operation
    // triggered by a single file system call, using a new tracing context.
    lastUsedTracingContext = createNewTracingContext(tracingContext);
    try {
      if (abfsCounters != null) {
        abfsCounters.getLastExecutionTime().set(now());
      }
      client.timerOrchestrator(TimerFunctionality.RESUME, null);
      IOStatisticsBinding.trackDurationOfInvocation(abfsCounters,
          AbfsStatistic.getStatNameFromHttpCall(method),
          () -> completeExecute(lastUsedTracingContext));
    } catch (AzureBlobFileSystemException aze) {
      throw aze;
    } catch (IOException e) {
      throw new UncheckedIOException("Error while tracking Duration of an "
          + "AbfsRestOperation call", e);
    }
  }

  /**
   * Executes the REST operation with retry, by issuing one or more
   * HTTP operations.
   * @param tracingContext TracingContext instance to track correlation IDs
   */
  void completeExecute(TracingContext tracingContext)
      throws AzureBlobFileSystemException {
    // see if we have latency reports from the previous requests
    String latencyHeader = getClientLatency();
    if (latencyHeader != null && !latencyHeader.isEmpty()) {
      AbfsHttpHeader httpHeader =
              new AbfsHttpHeader(HttpHeaderConfigurations.X_MS_ABFS_CLIENT_LATENCY, latencyHeader);
      requestHeaders.add(httpHeader);
    }

    // By Default Exponential Retry Policy Will be used
    retryCount = 0;
    retryPolicy = client.getExponentialRetryPolicy();
    LOG.debug("First execution of REST operation - {}", operationType);
    long sleepDuration = 0L;
    if (abfsBackoffMetrics != null) {
      synchronized (this) {
        abfsBackoffMetrics.incrementTotalNumberOfRequests();
      }
    }
    while (!executeHttpOperation(retryCount, tracingContext)) {
      try {
        ++retryCount;
        tracingContext.setRetryCount(retryCount);
        long retryInterval = retryPolicy.getRetryInterval(retryCount);
        LOG.debug("Rest operation {} failed with failureReason: {}. Retrying with retryCount = {}, retryPolicy: {} and sleepInterval: {}",
            operationType, failureReason, retryCount, retryPolicy.getAbbreviation(), retryInterval);
        if (abfsBackoffMetrics != null) {
          updateBackoffTimeMetrics(retryCount, sleepDuration);
        }
        Thread.sleep(retryInterval);
      } catch (InterruptedException ex) {
        Thread.currentThread().interrupt();
      }
    }
    if (abfsBackoffMetrics != null) {
      updateBackoffMetrics(retryCount, result.getStatusCode());
    }
    int status = result.getStatusCode();
    /*
      If even after exhausting all retries, the http status code has an
      invalid value it qualifies for InvalidAbfsRestOperationException.
      All http status code less than 1xx range are considered as invalid
      status codes.
     */
    if (status < HTTP_CONTINUE) {
      throw new InvalidAbfsRestOperationException(null, retryCount);
    }

    if (status >= HttpURLConnection.HTTP_BAD_REQUEST) {
      throw new AbfsRestOperationException(result.getStatusCode(), result.getStorageErrorCode(),
          result.getStorageErrorMessage(), null, result);
    }
    LOG.trace("{} REST operation complete", operationType);
  }

  @VisibleForTesting
public <Y> ValueExtractor<Y> getExtractor(final JavaType<Y> javaType) {
		return new BasicExtractor<Y>( javaType, this ) {

			private Y doExtract(ResultSet rs, int columnIndex, WrapperOptions options) throws SQLException {
				if (!this.determineCrsIdFromDatabase()) {
					return javaType.wrap(HANASpatialUtils.toGeometry(rs.getObject(columnIndex)), options);
				} else {
					throw new UnsupportedOperationException("First need to refactor HANASpatialUtils");
					//return getJavaTypeDescriptor().wrap( HANASpatialUtils.toGeometry( rs, paramIndex ), options );
				}
			}

			private Y doExtract(CallableStatement statement, int index, WrapperOptions options) throws SQLException {
				return javaType.wrap(HANASpatialUtils.toGeometry(statement.getObject(index)), options);
			}

			private Y doExtract(CallableStatement statement, String columnName, WrapperOptions options)
					throws SQLException {
				return javaType.wrap(HANASpatialUtils.toGeometry(statement.getObject(columnName)), options);
			}
		};
	}

  @VisibleForTesting
private static String entityInfo(PersistEvent event, Object info, EntityEntry entry) {
		if ( event.getEntityInfo() != null ) {
			return event.getEntityInfo();
		}
		else {
			// changes event.entityInfo by side effect!
			final String infoName = event.getSession().bestGuessEntityInfo( info, entry );
			event.setEntityInfo( infoName );
			return infoName;
		}
	}

  /**
   * Executes a single HTTP operation to complete the REST operation.  If it
   * fails, there may be a retry.  The retryCount is incremented with each
   * attempt.
   */
  private boolean executeHttpOperation(final int retryCount,
    TracingContext tracingContext) throws AzureBlobFileSystemException {
    final AbfsHttpOperation httpOperation;
    // Used to avoid CST Metric Update in Case of UnknownHost/IO Exception.
    boolean wasKnownExceptionThrown = false;

    try {
      // initialize the HTTP request and open the connection
      httpOperation = createHttpOperation();
      incrementCounter(AbfsStatistic.CONNECTIONS_MADE, 1);
      tracingContext.constructHeader(httpOperation, failureReason, retryPolicy.getAbbreviation());

      signRequest(httpOperation, hasRequestBody ? bufferLength : 0);

    } catch (IOException e) {
      LOG.debug("Auth failure: {}, {}", method, url);
      throw new AbfsRestOperationException(-1, null,
          "Auth failure: " + e.getMessage(), e);
    }

    try {
      // dump the headers
      AbfsIoUtils.dumpHeadersToDebugLog("Request Headers",
          httpOperation.getRequestProperties());
      intercept.sendingRequest(operationType, abfsCounters);
      if (hasRequestBody) {
        httpOperation.sendPayload(buffer, bufferOffset, bufferLength);
        incrementCounter(AbfsStatistic.SEND_REQUESTS, 1);
        if (!(operationType.name().equals(PUT_BLOCK_LIST))) {
          incrementCounter(AbfsStatistic.BYTES_SENT, bufferLength);
        }
      }
      httpOperation.processResponse(buffer, bufferOffset, bufferLength);
      if (!isThrottledRequest && httpOperation.getStatusCode()
          >= HttpURLConnection.HTTP_INTERNAL_ERROR) {
        isThrottledRequest = true;
        AzureServiceErrorCode serviceErrorCode =
            AzureServiceErrorCode.getAzureServiceCode(
                httpOperation.getStatusCode(),
                httpOperation.getStorageErrorCode(),
                httpOperation.getStorageErrorMessage());
        LOG1.trace("Service code is " + serviceErrorCode + " status code is "
            + httpOperation.getStatusCode() + " error code is "
            + httpOperation.getStorageErrorCode()
            + " error message is " + httpOperation.getStorageErrorMessage());
        if (abfsBackoffMetrics != null) {
          synchronized (this) {
            if (serviceErrorCode.equals(
                    AzureServiceErrorCode.INGRESS_OVER_ACCOUNT_LIMIT)
                    || serviceErrorCode.equals(
                    AzureServiceErrorCode.EGRESS_OVER_ACCOUNT_LIMIT)) {
              abfsBackoffMetrics.incrementNumberOfBandwidthThrottledRequests();
            } else if (serviceErrorCode.equals(
                    AzureServiceErrorCode.TPS_OVER_ACCOUNT_LIMIT)) {
              abfsBackoffMetrics.incrementNumberOfIOPSThrottledRequests();
            } else {
              abfsBackoffMetrics.incrementNumberOfOtherThrottledRequests();
            }
          }
        }
      }
        incrementCounter(AbfsStatistic.GET_RESPONSES, 1);
      //Only increment bytesReceived counter when the status code is 2XX.
      if (httpOperation.getStatusCode() >= HttpURLConnection.HTTP_OK
          && httpOperation.getStatusCode() <= HttpURLConnection.HTTP_PARTIAL) {
        incrementCounter(AbfsStatistic.BYTES_RECEIVED,
            httpOperation.getBytesReceived());
      } else if (httpOperation.getStatusCode() == HttpURLConnection.HTTP_UNAVAILABLE) {
        incrementCounter(AbfsStatistic.SERVER_UNAVAILABLE, 1);
      }

      // If no exception occurred till here it means http operation was successfully complete and
      // a response from server has been received which might be failure or success.
      // If any kind of exception has occurred it will be caught below.
      // If request failed to determine failure reason and retry policy here.
      // else simply return with success after saving the result.
      LOG.debug("HttpRequest: {}: {}", operationType, httpOperation);

      int status = httpOperation.getStatusCode();
      failureReason = RetryReason.getAbbreviation(null, status, httpOperation.getStorageErrorMessage());
      retryPolicy = client.getRetryPolicy(failureReason);

      if (retryPolicy.shouldRetry(retryCount, httpOperation.getStatusCode())) {
        return false;
      }

      // If the request has succeeded or failed with non-retrial error, save the operation and return.
      result = httpOperation;

    } catch (UnknownHostException ex) {
      wasKnownExceptionThrown = true;
      String hostname = null;
      hostname = httpOperation.getHost();
      failureReason = RetryReason.getAbbreviation(ex, null, null);
      retryPolicy = client.getRetryPolicy(failureReason);
      LOG.warn("Unknown host name: {}. Retrying to resolve the host name...",
          hostname);
      if (httpOperation instanceof AbfsAHCHttpOperation) {
        registerApacheHttpClientIoException();
      }
      if (abfsBackoffMetrics != null) {
        synchronized (this) {
          abfsBackoffMetrics.incrementNumberOfNetworkFailedRequests();
        }
      }
      if (!retryPolicy.shouldRetry(retryCount, -1)) {
        updateBackoffMetrics(retryCount, httpOperation.getStatusCode());
        throw new InvalidAbfsRestOperationException(ex, retryCount);
      }
      return false;
    } catch (IOException ex) {
      wasKnownExceptionThrown = true;
      if (LOG.isDebugEnabled()) {
        LOG.debug("HttpRequestFailure: {}, {}", httpOperation, ex);
      }
      if (abfsBackoffMetrics != null) {
        synchronized (this) {
          abfsBackoffMetrics.incrementNumberOfNetworkFailedRequests();
        }
      }
      failureReason = RetryReason.getAbbreviation(ex, -1, "");
      retryPolicy = client.getRetryPolicy(failureReason);
      if (httpOperation instanceof AbfsAHCHttpOperation) {
        registerApacheHttpClientIoException();
        if (ex instanceof RequestAbortedException
            && ex.getCause() instanceof ClosedIOException) {
          throw new AbfsDriverException((IOException) ex.getCause());
        }
      }
      if (!retryPolicy.shouldRetry(retryCount, -1)) {
        updateBackoffMetrics(retryCount, httpOperation.getStatusCode());
        throw new InvalidAbfsRestOperationException(ex, retryCount);
      }
      return false;
    } finally {
      int statusCode = httpOperation.getStatusCode();
      // Update Metrics only if Succeeded or Throttled due to account limits.
      // Also Update in case of any unhandled exception is thrown.
      if (shouldUpdateCSTMetrics(statusCode) && !wasKnownExceptionThrown) {
        intercept.updateMetrics(operationType, httpOperation);
      }
    }

    return true;
  }

  /**
   * Registers switch off of ApacheHttpClient in case of IOException retries increases
   * more than the threshold.
   */
public UserFetchRequest.Builder newUserFetchBuilder(String userId, FetchConfig fetchConfig) {
    List<UserPartition> added = new ArrayList<>();
    List<UserPartition> removed = new ArrayList<>();
    List<UserPartition> replaced = new ArrayList<>();

    if (nextMetadata.isNewSession()) {
        // Add any new partitions to the session
        for (Entry<UserId, UserPartition> entry : nextPartitions.entrySet()) {
            UserId userIdentity = entry.getKey();
            UserPartition userPartition = entry.getValue();
            sessionPartitions.put(userIdentity, userPartition);
        }

        // If it's a new session, all the partitions must be added to the request
        added.addAll(sessionPartitions.values());
    } else {
        // Iterate over the session partitions, tallying which were added
        Iterator<Entry<UserId, UserPartition>> partitionIterator = sessionPartitions.entrySet().iterator();
        while (partitionIterator.hasNext()) {
            Entry<UserId, UserPartition> entry = partitionIterator.next();
            UserId userIdentity = entry.getKey();
            UserPartition prevData = entry.getValue();
            UserPartition nextData = nextPartitions.remove(userIdentity);
            if (nextData != null) {
                // If the user ID does not match, the user has been recreated
                if (!prevData.equals(nextData)) {
                    nextPartitions.put(userIdentity, nextData);
                    entry.setValue(nextData);
                    replaced.add(prevData);
                }
            } else {
                // This partition is not in the builder, so we need to remove it from the session
                partitionIterator.remove();
                removed.add(prevData);
            }
        }

        // Add any new partitions to the session
        for (Entry<UserId, UserPartition> entry : nextPartitions.entrySet()) {
            UserId userIdentity = entry.getKey();
            UserPartition userPartition = entry.getValue();
            sessionPartitions.put(userIdentity, userPartition);
            added.add(userPartition);
        }
    }

    if (log.isDebugEnabled()) {
        log.debug("Build UserFetch {} for node {}. Added {}, removed {}, replaced {} out of {}",
                nextMetadata, node,
                userPartitionsToLogString(added),
                userPartitionsToLogString(removed),
                userPartitionsToLogString(replaced),
                userPartitionsToLogString(sessionPartitions.values()));
    }

    // The replaced user-partitions need to be removed, and their replacements are already added
    removed.addAll(replaced);

    Map<UserPartition, List<UserFetchRequestData.AcknowledgementBatch>> acknowledgementBatches = new HashMap<>();
    nextAcknowledgements.forEach((partition, acknowledgements) -> acknowledgementBatches.put(partition, acknowledgements.getAcknowledgementBatches()
            .stream().map(AcknowledgementBatch::toUserFetchRequest)
            .collect(Collectors.toList())));

    nextPartitions = new LinkedHashMap<>();
    nextAcknowledgements = new LinkedHashMap<>();

    return UserFetchRequest.Builder.forConsumer(
            userId, nextMetadata, fetchConfig.maxWaitMs,
            fetchConfig.minBytes, fetchConfig.maxBytes, fetchConfig.fetchSize,
            added, removed, acknowledgementBatches);
}

  /**
   * Sign an operation.
   * @param httpOperation operation to sign
   * @param bytesToSign how many bytes to sign for shared key auth.
   * @throws IOException failure
   */
  @VisibleForTesting
public static long getLastIncludedLogTime(RawSnapshotReader dataReader) {
        RecordsSnapshotReader<ByteBuffer> recordsSnapshotReader = RecordsSnapshotReader.of(
            dataReader,
            IdentitySerde.INSTANCE,
            new BufferSupplier.GrowableBufferSupplier(),
            KafkaRaftClient.MAX_BATCH_SIZE_BYTES,
            false
        );
        try (recordsSnapshotReader) {
            return recordsSnapshotReader.getLastIncludedLogTime();
        }
    }

  /**
   * Creates new object of {@link AbfsHttpOperation} with the url, method, and
   * requestHeaders fields of the AbfsRestOperation object.
   */
  @VisibleForTesting
private static Map<String, Object> convertRecordToJson(BaseRecord data) {
    Map<String, Object> jsonMap = new HashMap<>();
    Map<String, Class<?>> fieldsMap = getFields(data);

    for (String key : fieldsMap.keySet()) {
      if (!"proto".equalsIgnoreCase(key)) {
        try {
          Object fieldValue = getField(data, key);
          if (fieldValue instanceof BaseRecord) {
            BaseRecord subRecord = (BaseRecord) fieldValue;
            jsonMap.putAll(getJson(subRecord));
          } else {
            jsonMap.put(key, fieldValue == null ? JSONObject.NULL : fieldValue);
          }
        } catch (Exception e) {
          throw new IllegalArgumentException(
              "Cannot convert field " + key + " into JSON", e);
        }
      }
    }
    return jsonMap;
  }

private List<SlowPeerJsonReport> retrieveTopNReports(int numberOfNodes) {
    if (this.allReports.isEmpty()) {
      return Collections.emptyList();
    }

    final PriorityQueue<SlowPeerJsonReport> topReportsQueue = new PriorityQueue<>(this.allReports.size(),
        (report1, report2) -> Integer.compare(report1.getPeerLatencies().size(), report2.getPeerLatencies().size()));

    long currentTime = this.timer.monotonicNow();

    for (Map.Entry<String, ConcurrentMap<String, LatencyWithLastReportTime>> entry : this.allReports.entrySet()) {
      SortedSet<SlowPeerLatencyWithReportingNode> validReportsSet = filterNodeReports(entry.getValue(), currentTime);
      if (!validReportsSet.isEmpty()) {
        if (topReportsQueue.size() < numberOfNodes) {
          topReportsQueue.add(new SlowPeerJsonReport(entry.getKey(), validReportsSet));
        } else if (!topReportsQueue.peek().getPeerLatencies().isEmpty()
            && topReportsQueue.peek().getPeerLatencies().size() < validReportsSet.size()) {
          // Remove the lowest priority element
          topReportsQueue.poll();
          topReportsQueue.add(new SlowPeerJsonReport(entry.getKey(), validReportsSet));
        }
      }
    }
    return new ArrayList<>(topReportsQueue);
  }

  @VisibleForTesting
  private void initializeNewPlans(Configuration conf) {
    LOG.info("Refreshing Reservation system");
    writeLock.lock();
    try {
      // Create a plan corresponding to every new reservable queue
      Set<String> planQueueNames = scheduler.getPlanQueues();
      for (String planQueueName : planQueueNames) {
        if (!plans.containsKey(planQueueName)) {
          Plan plan = initializePlan(planQueueName);
          plans.put(planQueueName, plan);
        } else {
          LOG.warn("Plan based on reservation queue {} already exists.",
              planQueueName);
        }
      }
      // Update the plan follower with the active plans
      if (planFollower != null) {
        planFollower.setPlans(plans.values());
      }
    } catch (YarnException e) {
      LOG.warn("Exception while trying to refresh reservable queues", e);
    } finally {
      writeLock.unlock();
    }
  }

  @VisibleForTesting
	protected File prepareReportFile() {
		final File reportFile = getReportFileReference().get().getAsFile();

		if ( reportFile.getParentFile().exists() ) {
			if ( reportFile.exists() ) {
				if ( !reportFile.delete() ) {
					throw new RuntimeException( "Unable to delete report file - " + reportFile.getAbsolutePath() );
				}
			}
		}
		else {
			if ( !reportFile.getParentFile().mkdirs() ) {
				throw new RuntimeException( "Unable to create report file directories - " + reportFile.getAbsolutePath() );
			}
		}

		try {
			if ( !reportFile.createNewFile() ) {
				throw new RuntimeException( "Unable to create report file - " + reportFile.getAbsolutePath() );
			}
		}
		catch (IOException e) {
			throw new RuntimeException( "Unable to create report file - " + reportFile.getAbsolutePath() );
		}

		return reportFile;
	}

  /**
   * Incrementing Abfs counters with a long value.
   *
   * @param statistic the Abfs statistic that needs to be incremented.
   * @param value     the value to be incremented by.
   */
    public void shutdown() throws InterruptedException {
        try {
            super.shutdown();
        } finally {
            client.close();
        }
    }

  /**
   * Updates the count metrics based on the provided retry count.
   * @param retryCount The retry count used to determine the metrics category.
   *
   * This method increments the number of succeeded requests for the specified retry count.
   */
public List<Container> retrieveContainersFromPreviousAttempts() {
    if (null != this.containersFromPreviousAttemptList) {
      return this.containersFromPreviousAttemptList;
    }

    this.initContainersForPreviousAttempt();
    return this.containersFromPreviousAttemptList;
}

private void initContainersForPreviousAttempt() {
    if (this.containersFromPreviousAttempts == null) {
        this.containersFromPreviousAttempts = new ArrayList<>();
    }
}

  /**
   * Updates backoff time metrics based on the provided retry count and sleep duration.
   * @param retryCount    The retry count used to determine the metrics category.
   * @param sleepDuration The duration of sleep during backoff.
   *
   * This method calculates and updates various backoff time metrics, including minimum, maximum,
   * and total backoff time, as well as the total number of requests for the specified retry count.
   */
  private static <V> V onlyElement(Collection<V> values, V fallback) {
    switch (values.size()) {
      case 0:
        return fallback;
      case 1:
        return values.iterator().next();
      default:
        throw new IllegalArgumentException("expected one value, found: " + values);
    }
  }

  /**
   * Generates a key based on the provided retry count to categorize metrics.
   *
   * @param retryCount The retry count used to determine the key.
   * @return A string key representing the metrics category for the given retry count.
   *
   * This method categorizes retry counts into different ranges and assigns a corresponding key.
   */
	protected void applyCookies() {
		for (String name : getCookies().keySet()) {
			for (ResponseCookie httpCookie : getCookies().get(name)) {
				Cookie cookie = new Cookie(name, httpCookie.getValue());
				if (!httpCookie.getMaxAge().isNegative()) {
					cookie.setMaxAge((int) httpCookie.getMaxAge().getSeconds());
				}
				if (httpCookie.getDomain() != null) {
					cookie.setDomain(httpCookie.getDomain());
				}
				if (httpCookie.getPath() != null) {
					cookie.setPath(httpCookie.getPath());
				}
				if (httpCookie.getSameSite() != null) {
					cookie.setAttribute("SameSite", httpCookie.getSameSite());
				}
				cookie.setSecure(httpCookie.isSecure());
				cookie.setHttpOnly(httpCookie.isHttpOnly());
				if (httpCookie.isPartitioned()) {
					cookie.setAttribute("Partitioned", "");
				}
				this.response.addCookie(cookie);
			}
		}
	}

  /**
   * Updating Client Side Throttling Metrics for relevant response status codes.
   * Following criteria is used to decide based on status code and failure reason.
   * <ol>
   *   <li>Case 1: Status code in 2xx range: Successful Operations should contribute</li>
   *   <li>Case 2: Status code in 3xx range: Redirection Operations should not contribute</li>
   *   <li>Case 3: Status code in 4xx range: User Errors should not contribute</li>
   *   <li>
   *     Case 4: Status code is 503: Throttling Error should contribute as following:
   *     <ol>
   *       <li>Case 4.a: Ingress Over Account Limit: Should Contribute</li>
   *       <li>Case 4.b: Egress Over Account Limit: Should Contribute</li>
   *       <li>Case 4.c: TPS Over Account Limit: Should Contribute</li>
   *       <li>Case 4.d: Other Server Throttling: Should not contribute</li>
   *     </ol>
   *   </li>
   *   <li>Case 5: Status code in 5xx range other than 503: Should not contribute</li>
   * </ol>
   * @param statusCode
   * @return
   */
public void beforeStatementProcessing(PreparedStatementDetails stmtDetails) {
		BindingGroup bindingGroup = bindingGroupMap.get(stmtDetails.getMutatingTableDetails().getTableName());
		if (bindingGroup != null) {
			bindingGroup.forEachBinding(binding -> {
				try {
					binding.getValueBinder().bind(
							stmtDetails.resolveStatement(),
							binding.getValue(),
							binding.getPosition(),
							session
					);
				} catch (SQLException e) {
					session.getJdbcServices().getSqlExceptionHelper().convert(e,
							String.format("Unable to bind parameter #%s - %s", binding.getPosition(), binding.getValue()));
				}
			});
		} else {
			stmtDetails.resolveStatement();
		}
	}

  /**
   * Creates a new Tracing context before entering the retry loop of a rest operation.
   * This will ensure all rest operations have unique
   * tracing context that will be used for all the retries.
   * @param tracingContext original tracingContext.
   * @return tracingContext new tracingContext object created from original one.
   */
  @VisibleForTesting
public synchronized C getFileSystemCounter(String scheme, FileSystemCounter key) {
    String canonicalScheme = checkScheme(scheme);
    if (map == null) {
        map = new ConcurrentSkipListMap<>();
    }
    Object[] counters = map.get(canonicalScheme);
    if (counters == null || counters[key.ordinal()] == null) {
        counters = new Object[FileSystemCounter.values().length];
        map.put(canonicalScheme, counters);
        counters[key.ordinal()] = newCounter(canonicalScheme, key);
    }
    return (C) counters[key.ordinal()];
}

  /**
   * Returns the tracing contest used for last rest operation made.
   * @return tracingContext lasUserTracingContext.
   */
  @VisibleForTesting
private static int getStat(String field) {
    if(OperatingSystem.WINDOWS) {
      try {
        CommandExecutor commandExecutorStat = new CommandExecutor(
            new String[] {"systeminfo", "/FO", "VALUE", field});
        commandExecutorStat.execute();
        return Integer.parseInt(commandExecutorStat.getOutput().replace("\n", ""));
      } catch (IOException|NumberFormatException e) {
        return -1;
      }
    }
    return -1;
  }

    public void setDialectsByPrefix(final Map<String,IDialect> dialects) {
        Validate.notNull(dialects, "Dialect map cannot be null");
        checkNotInitialized();
        this.dialectConfigurations.clear();
        for (final Map.Entry<String,IDialect> dialectEntry : dialects.entrySet()) {
            addDialect(dialectEntry.getKey(), dialectEntry.getValue());
        }
    }
}
