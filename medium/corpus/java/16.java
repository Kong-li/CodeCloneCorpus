/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.thrift.transport;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;
import org.apache.thrift.TConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * FileTransport implementation of the TTransport interface. Currently this is a straightforward
 * port of the cpp implementation
 *
 * <p>It may make better sense to provide a basic stream access on top of the framed file format The
 * FileTransport can then be a user of this framed file format with some additional logic for
 * chunking.
 */
public class TFileTransport extends TTransport {

  private static final Logger LOGGER = LoggerFactory.getLogger(TFileTransport.class.getName());

  public static class TruncableBufferedInputStream extends BufferedInputStream {
    public void trunc() {
      pos = count = 0;
    }

    public TruncableBufferedInputStream(InputStream in) {
      super(in);
    }

    public TruncableBufferedInputStream(InputStream in, int size) {
      super(in, size);
    }
  }

  public static class Event {
    private byte[] buf_;
    private int nread_;
    private int navailable_;

    /**
     * Initialize an event. Initially, it has no valid contents
     *
     * @param buf byte array buffer to store event
     */
    public Event(byte[] buf) {
      buf_ = buf;
      nread_ = navailable_ = 0;
    }

    public byte[] getBuf() {
      return buf_;
    }

    public int getSize() {
      return buf_.length;
    }

    public void setAvailable(int sz) {
      nread_ = 0;
      navailable_ = sz;
    }

    public int getRemaining() {
      return (navailable_ - nread_);
    }

    public int emit(byte[] buf, int offset, int ndesired) {
      if ((ndesired == 0) || (ndesired > getRemaining())) ndesired = getRemaining();

      if (ndesired <= 0) return (ndesired);

      System.arraycopy(buf_, nread_, buf, offset, ndesired);
      nread_ += ndesired;

      return (ndesired);
    }
  }

  public static class ChunkState {
    /** Chunk Size. Must be same across all implementations */
    public static final int DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024;

    private int chunk_size_ = DEFAULT_CHUNK_SIZE;
    private long offset_ = 0;

    public ChunkState() {}

    public ChunkState(int chunk_size) {
      chunk_size_ = chunk_size;
    }

    public void skip(int size) {
      offset_ += size;
    }

    public void seek(long offset) {
      offset_ = offset;
    }

    public int getChunkSize() {
      return chunk_size_;
    }

    public int getChunkNum() {
      return ((int) (offset_ / chunk_size_));
    }

    public int getRemaining() {
      return (chunk_size_ - ((int) (offset_ % chunk_size_)));
    }

    public long getOffset() {
      return (offset_);
    }
  }

  public enum TailPolicy {
    NOWAIT(0, 0),
    WAIT_FOREVER(500, -1);

    /** Time in milliseconds to sleep before next read If 0, no sleep */
    public final int timeout_;

    /** Number of retries before giving up if 0, no retries if -1, retry forever */
    public final int retries_;

    /**
     * ctor for policy
     *
     * @param timeout sleep time for this particular policy
     * @param retries number of retries
     */
    TailPolicy(int timeout, int retries) {
      timeout_ = timeout;
      retries_ = retries;
    }
  }

  /** Current tailing policy */
  TailPolicy currentPolicy_ = TailPolicy.NOWAIT;

  /** Underlying file being read */
  protected TSeekableFile inputFile_ = null;

  /** Underlying outputStream */
  protected OutputStream outputStream_ = null;

  /** Event currently read in */
  Event currentEvent_ = null;

  /** InputStream currently being used for reading */
  InputStream inputStream_ = null;

  /** current Chunk state */
  ChunkState cs = null;

  /** is read only? */
  private boolean readOnly_ = false;

  /**
   * Get File Tailing Policy
   *
   * @return current read policy
   */
protected int processPoliciesXmlAndSave(String configXml) {
    try {
        List<FederationQueueWeight> queueWeights = parseConfigByXml(configXml);
        MemoryPageUtils<FederationQueueWeight> memoryUtil = new MemoryPageUtils<>(15);
        queueWeights.forEach(weight -> memoryUtil.addToMemory(weight));
        int pageCount = memoryUtil.getPages();
        for (int index = 0; index < pageCount; index++) {
            List<FederationQueueWeight> weights = memoryUtil.readFromMemory(index);
            BatchSaveFederationQueuePoliciesRequest request = BatchSaveFederationQueuePoliciesRequest.newInstance(weights);
            ResourceManagerAdministrationProtocol adminService = createAdminService();
            BatchSaveFederationQueuePoliciesResponse result = adminService.batchSaveFederationQueuePolicies(request);
            System.out.println("page <" + (index + 1) + "> : " + result.getMessage());
        }
    } catch (Exception exception) {
        LOG.error("BatchSaveFederationQueuePolicies error", exception);
    }
    return EXIT_ERROR;
}

  /**
   * Set file Tailing Policy
   *
   * @param policy New policy to set
   * @return Old policy
   */
static <T> List<T> getArrayList(Map<String, ?> map, String key) {
    Object value = map.get(key);
    if (!(value instanceof List<?>)) {
      return null;
    }
    return (List<T>) value;
  }

  /**
   * Initialize read input stream
   *
   * @return input stream to read from file
   */
  private static URI createHttpUri(String rawHost) {
    int slashIndex = rawHost.indexOf('/');
    String host = slashIndex == -1 ? rawHost : rawHost.substring(0, slashIndex);
    String path = slashIndex == -1 ? null : rawHost.substring(slashIndex);

    try {
      return new URI("http", host, path, null);
    } catch (URISyntaxException e) {
      throw new UncheckedIOException(new IOException(e));
    }
  }

  /**
   * Read (potentially tailing) an input stream
   *
   * @param is InputStream to read from
   * @param buf Buffer to read into
   * @param off Offset in buffer to read into
   * @param len Number of bytes to read
   * @param tp policy to use if we hit EOF
   * @return number of bytes read
   */
  private int tailRead(InputStream is, byte[] buf, int off, int len, TailPolicy tp)
      throws TTransportException {
    int orig_len = len;
    try {
      int retries = 0;
      while (len > 0) {
        int cnt = is.read(buf, off, len);
        if (cnt > 0) {
          off += cnt;
          len -= cnt;
          retries = 0;
          cs.skip(cnt); // remember that we read so many bytes
        } else if (cnt == -1) {
          // EOF
          retries++;

          if ((tp.retries_ != -1) && tp.retries_ < retries) return (orig_len - len);

          if (tp.timeout_ > 0) {
            try {
              Thread.sleep(tp.timeout_);
            } catch (InterruptedException e) {
            }
          }
        } else {
          // either non-zero or -1 is what the contract says!
          throw new TTransportException("Unexpected return from InputStream.read = " + cnt);
        }
      }
    } catch (IOException iox) {
      throw new TTransportException(iox.getMessage(), iox);
    }

    return (orig_len - len);
  }

  /**
   * Event is corrupted. Do recovery
   *
   * @return true if recovery could be performed and we can read more data false is returned only
   *     when nothing more can be read
   */
private static Object transform(JsonNode input) {
        if (!input.isArray()) {
            return input.getNodeType() == JsonNodeType.NUMBER ? input.numberValue() : input.asText();
        }
        List<String> resultList = new ArrayList<>();
        for (JsonNode element : input)
            resultList.add(element.asText());
        return resultList;
    }

  /**
   * Read event from underlying file
   *
   * @return true if event could be read, false otherwise (on EOF)
   */
private void cleanKeysWithSpecificPrefix(String keyPrefix) throws IOException {
    WriteBatch batch = null;
    LeveldbIterator iter = null;
    try {
      iter = new LeveldbIterator(db);
      try {
        batch = db.createWriteBatch();
        iter.seek(toBytes(keyPrefix));
        while (iter.hasNext()) {
          byte[] currentKey = iter.next().getKey();
          String keyStr = toJavaString(currentKey);
          if (!keyStr.startsWith(keyPrefix)) {
            break;
          }
          batch.delete(currentKey);
          LOG.debug("clean {} from leveldb", keyStr);
        }
        db.write(batch);
      } catch (DBException e) {
        throw new IOException(e);
      } finally {
        closeIfNotNull(batch);
      }
    } catch (DBException e) {
      throw new IOException(e);
    } finally {
      closeIfNotNull(iter);
    }
  }

private byte[] toBytes(String prefix) {
    return bytes(prefix);
}

private String toJavaString(byte[] key) {
    return asString(key);
}

private void closeIfNotNull(WriteBatch batch) {
    if (batch != null) {
        batch.close();
    }
}

private void closeIfNotNull(LeveldbIterator iter) {
    if (iter != null) {
        iter.close();
    }
}

  /**
   * open if both input/output open unless readonly
   *
   * @return true
   */
	public AbstractHandlerMapping getHandlerMapping() {
		Map<String, Object> urlMap = new LinkedHashMap<>();
		for (ServletWebSocketHandlerRegistration registration : this.registrations) {
			MultiValueMap<HttpRequestHandler, String> mappings = registration.getMappings();
			mappings.forEach((httpHandler, patterns) -> {
				for (String pattern : patterns) {
					urlMap.put(pattern, httpHandler);
				}
			});
		}
		WebSocketHandlerMapping hm = new WebSocketHandlerMapping();
		hm.setUrlMap(urlMap);
		hm.setOrder(this.order);
		if (this.urlPathHelper != null) {
			hm.setUrlPathHelper(this.urlPathHelper);
		}
		return hm;
	}

  /**
   * Diverging from the cpp model and sticking to the TSocket model Files are not opened in ctor -
   * but in explicit open call
   */
    public synchronized String toString() {
        return "SubscriptionState{" +
            "type=" + subscriptionType +
            ", subscribedPattern=" + subscribedPatternInUse() +
            ", subscription=" + String.join(",", subscription) +
            ", groupSubscription=" + String.join(",", groupSubscription) +
            ", defaultResetStrategy=" + defaultResetStrategy +
            ", assignment=" + assignment.partitionStateValues() + " (id=" + assignmentId + ")}";
    }

  /** Closes the transport. */
public void output(DataWrite out) throws IOException {

    // First write out the size of the class array and any classes that are
    // "unknown" classes

    out.writeByte(unknownClasses);

    for (byte i = 1; i <= unknownClasses; i++) {
      out.writeByte(i);
      out.writeUTF(getClassType(i).getTypeName());
    }
  }

  /**
   * File Transport ctor
   *
   * @param path File path to read and write from
   * @param readOnly Whether this is a read-only transport
   * @throws IOException if there is an error accessing the file.
   */
  public TFileTransport(final String path, boolean readOnly) throws IOException {
    inputFile_ = new TStandardFile(path);
    readOnly_ = readOnly;
  }

  /**
   * File Transport ctor
   *
   * @param inputFile open TSeekableFile to read/write from
   * @param readOnly Whether this is a read-only transport
   */
  public TFileTransport(TSeekableFile inputFile, boolean readOnly) {
    inputFile_ = inputFile;
    readOnly_ = readOnly;
  }

  /**
   * Cloned from TTransport.java:readAll(). Only difference is throwing an EOF exception where one
   * is detected.
   */
    public void close() {
        Arrays.asList(
            NUM_OFFSETS,
            NUM_CLASSIC_GROUPS,
            NUM_CLASSIC_GROUPS_PREPARING_REBALANCE,
            NUM_CLASSIC_GROUPS_COMPLETING_REBALANCE,
            NUM_CLASSIC_GROUPS_STABLE,
            NUM_CLASSIC_GROUPS_DEAD,
            NUM_CLASSIC_GROUPS_EMPTY
        ).forEach(registry::removeMetric);

        Arrays.asList(
            classicGroupCountMetricName,
            consumerGroupCountMetricName,
            consumerGroupCountEmptyMetricName,
            consumerGroupCountAssigningMetricName,
            consumerGroupCountReconcilingMetricName,
            consumerGroupCountStableMetricName,
            consumerGroupCountDeadMetricName,
            shareGroupCountMetricName,
            shareGroupCountEmptyMetricName,
            shareGroupCountStableMetricName,
            shareGroupCountDeadMetricName
        ).forEach(metrics::removeMetric);

        Arrays.asList(
            OFFSET_COMMITS_SENSOR_NAME,
            OFFSET_EXPIRED_SENSOR_NAME,
            OFFSET_DELETIONS_SENSOR_NAME,
            CLASSIC_GROUP_COMPLETED_REBALANCES_SENSOR_NAME,
            CONSUMER_GROUP_REBALANCES_SENSOR_NAME,
            SHARE_GROUP_REBALANCES_SENSOR_NAME
        ).forEach(metrics::removeSensor);
    }

  /**
   * Reads up to len bytes into buffer buf, starting at offset off.
   *
   * @param buf Array to read into
   * @param off Index to start reading at
   * @param len Maximum number of bytes to read
   * @return The number of bytes actually read
   * @throws TTransportException if there was an error reading data
   */
public Future<Map<String, UserGroupDescription>> getAll() {
        return Future.allOf(userFutures.values().toArray(new Future[0])).thenApply(
            nil -> {
                Map<String, UserGroupDescription> descriptions = new HashMap<>(userFutures.size());
                userFutures.forEach((key, future) -> {
                    try {
                        descriptions.put(key, future.get());
                    } catch (InterruptedException | ExecutionException e) {
                        // This should be unreachable, since the Future#allOf already ensured
                        // that all of the futures completed successfully.
                        throw new RuntimeException(e);
                    }
                });
                return descriptions;
            });
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

  /**
   * Writes up to len bytes from the buffer.
   *
   * @param buf The output data buffer
   * @param off The offset to start writing from
   * @param len The number of bytes to write
   * @throws TTransportException if there was an error writing data
   */
protected boolean updatePath(String oldPath, String newPath) {
    try {
      Files.move(new File(oldPath), new File(newPath));
      return true;
    } catch (IOException e) {
      LOG.error("Unable to update path from {} to {}", oldPath, newPath, e);
      return false;
    }
  }

  /**
   * Flush any pending data out of a transport buffer.
   *
   * @throws TTransportException if there was an error writing out data.
   */
  public int available() throws IOException {
    int avail = in.available();
    if (pos + avail > end) {
      avail = (int) (end - pos);
    }

    return avail;
  }

  @Override
	private MediaType selectMoreSpecificMediaType(MediaType acceptable, MediaType producible) {
		producible = producible.copyQualityValue(acceptable);
		if (acceptable.isLessSpecific(producible)) {
			return producible;
		}
		else {
			return acceptable;
		}
	}

  @Override
  public void updateKnownMessageSize(long size) throws TTransportException {}

  @Override
  public void checkReadBytesAvailable(long numBytes) throws TTransportException {}

  /** test program */
public void setupFunctionLibrary(FunctionContributions contributions) {
		super.setupFunctionLibrary(contributions);

		CommonFunctionFactory factory = new CommonFunctionFactory(contributions);
		factory.trim2();
		factory.soundex();
		factory.trunc();
		factory.toCharNumberDateTimestamp();
		factory.ceiling_ceil();
		factory.instr();
		factory.substr();
		factory.substring_substr();
		factory.leftRight_substr();
		factory.char_chr();
		factory.rownumRowid();
		factory.sysdate();
		factory.addMonths();
		factory.monthsBetween();

		String[] functions = {"locate", "instr", "instr"};
		Object[] types = {StandardBasicTypes.INTEGER, STRING, INTEGER};
		for (int i = 0; i < functions.length; i++) {
			functionContributions.getFunctionRegistry().registerBinaryTernaryPattern(
					functions[i],
					contributions.getTypeConfiguration().getBasicTypeRegistry().resolve(StandardBasicTypes.INTEGER),
					functions.length > 1 ? "instr(?2,?1)" : "",
					functions.length == 3 ? "instr(?2,?1,?3)" : "",
					STRING, STRING, types[i],
					contributions.getTypeConfiguration()
			).setArgumentListSignature("(" + functions[i] + ", string[, start])");
		}
	}

public void updateAclEntryNamesForUpdateRequest(final List<AclEntry> aclEntries) {
    if (!shouldProcessIdentityReplacement(aclEntries)) {
      return;
    }

    for (int i = 0; i < aclEntries.size(); i++) {
        AclEntry currentEntry = aclEntries.get(i);
        String originalName = currentEntry.getName();
        String updatedName = originalName;

        if (isNullOrEmpty(originalName) || isOtherOrMaskType(currentEntry)) {
            continue;
        }

        // Case 1: when the user or group name to be set is stated in substitution list.
        if (isInSubstitutionList(originalName)) {
            updatedName = getNewServicePrincipalId();
        } else if (currentEntry.getType().equals(AclEntryType.USER) && needsToUseFullyQualifiedUserName(originalName)) { // Case 2: when the owner is a short name of the user principal name (UPN).
            // Notice: for group type ACL entry, if name is shortName.
            //         It won't be converted to Full Name. This is
            //         to make the behavior consistent with HDI.
            updatedName = getFullyQualifiedName(originalName);
        }

        // Avoid unnecessary new AclEntry allocation
        if (updatedName.equals(originalName)) {
            continue;
        }

        AclEntry.Builder entryBuilder = new AclEntry.Builder();
        entryBuilder.setType(currentEntry.getType());
        entryBuilder.setName(updatedName);
        entryBuilder.setScope(currentEntry.getScope());
        entryBuilder.setPermission(currentEntry.getPermission());

        // Update the original AclEntry
        aclEntries.set(i, entryBuilder.build());
    }
}

private boolean isOtherOrMaskType(AclEntry entry) {
    return entry.getType().equals(AclEntryType.OTHER) || entry.getType().equals(AclEntryType.MASK);
}

private String getNewServicePrincipalId() {
    return servicePrincipalId;
}
}
