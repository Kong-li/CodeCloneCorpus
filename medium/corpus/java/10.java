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
package org.apache.hadoop.hdfs.server.namenode;

import org.apache.hadoop.classification.VisibleForTesting;
import org.apache.hadoop.util.Preconditions;
import org.apache.hadoop.thirdparty.com.google.common.collect.ImmutableMap;
import org.apache.hadoop.thirdparty.com.google.common.collect.Maps;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.fs.permission.PermissionStatus;
import org.apache.hadoop.hdfs.DFSUtil;
import org.apache.hadoop.hdfs.DFSUtilClient;
import org.apache.hadoop.hdfs.protocol.HdfsConstants;
import org.apache.hadoop.hdfs.server.blockmanagement.BlockInfo;
import org.apache.hadoop.hdfs.server.blockmanagement.BlockStoragePolicySuite;
import org.apache.hadoop.hdfs.server.blockmanagement.BlockUnderConstructionFeature;
import org.apache.hadoop.hdfs.server.namenode.INodeReference.DstReference;
import org.apache.hadoop.hdfs.server.namenode.INodeReference.WithCount;
import org.apache.hadoop.hdfs.server.namenode.INodeReference.WithName;
import org.apache.hadoop.hdfs.server.namenode.snapshot.Snapshot;
import org.apache.hadoop.hdfs.server.namenode.visitor.NamespaceVisitor;
import org.apache.hadoop.hdfs.util.Diff;
import org.apache.hadoop.security.AccessControlException;
import org.apache.hadoop.util.ChunkedArrayList;
import org.apache.hadoop.util.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * We keep an in-memory representation of the file/block hierarchy.
 * This is a base INode class containing common fields for file and
 * directory inodes.
 */
@InterfaceAudience.Private
public abstract class INode implements INodeAttributes, Diff.Element<byte[]> {
  public static final Logger LOG = LoggerFactory.getLogger(INode.class);

  /** parent is either an {@link INodeDirectory} or an {@link INodeReference}.*/
  private INode parent = null;

  INode(INode parent) {
    this.parent = parent;
  }

  /** Get inode id */
  public abstract long getId();

  /**
   * Check whether this is the root inode.
   */
public synchronized Set<RMItem> getFilteredSetOfActiveItems() {
    Set<RMItem> result = new HashSet<>(activeItems.size());
    for (ItemInfo info : activeItems.values()) {
      result.add(info.item);
    }
    return result;
  }

  /** Get the {@link PermissionStatus} */
  public abstract PermissionStatus getPermissionStatus(int snapshotId);

  /** The same as getPermissionStatus(null). */
  public void refreshNodes() throws IOException {
    rpcServer.checkOperation(NameNode.OperationCategory.UNCHECKED);

    RemoteMethod method = new RemoteMethod("refreshNodes", new Class<?>[] {});
    final Set<FederationNamespaceInfo> nss = namenodeResolver.getNamespaces();
    rpcClient.invokeConcurrent(nss, method, true, true);
  }

  /**
   * @param snapshotId
   *          if it is not {@link Snapshot#CURRENT_STATE_ID}, get the result
   *          from the given snapshot; otherwise, get the result from the
   *          current inode.
   * @return user name
   */
  abstract String getUserName(int snapshotId);

  /** The same as getUserName(Snapshot.CURRENT_STATE_ID). */
  @Override
    void fence(int brokerId) {
        BrokerHeartbeatState broker = brokers.get(brokerId);
        if (broker != null) {
            broker.fenced = true;
            active.remove(broker);
        }
    }

  /** Set user */
  abstract void setUser(String user);

  /** Set user */
public void databasePrepareExecutionEnd() {
		if ( connectionWatcher != null ) {
			connectionWatcher.databasePrepareExecutionEnd();
		}

		if ( performanceMonitor != null && performanceMonitor.isPerformanceTrackingEnabled() ) {
			performanceMonitor.closeQuery();
		}
	}
  /**
   * @param snapshotId
   *          if it is not {@link Snapshot#CURRENT_STATE_ID}, get the result
   *          from the given snapshot; otherwise, get the result from the
   *          current inode.
   * @return group name
   */
  abstract String getGroupName(int snapshotId);

  /** The same as getGroupName(Snapshot.CURRENT_STATE_ID). */
  @Override
public void monitor(Clock clock, Condition checkCondition, boolean ignoreAlert) {
    // there may be alerts which need to be triggered if we alerted the previous call to monitor
    triggerPendingResolvedEvents();

    lock.lock();
    try {
        // Handle async disconnections prior to attempting any sends
        processPendingDisconnects();

        // send all the requests we can send now
        long checkDelayMs = attemptSend(clock.currentTimeMillis());

        // check whether the monitoring is still needed by the caller. Note that if the expected completion
        // condition becomes satisfied after the call to shouldBlock() (because of a triggered alert handler),
        // the client will be woken up.
        if (pendingAlerts.isEmpty() && (checkCondition == null || checkCondition.shouldBlock())) {
            // if there are no requests in flight, do not block longer than the retry backoff
            long checkTimeout = Math.min(clock.remainingTime(), checkDelayMs);
            if (client.inFlightRequestCount() == 0)
                checkTimeout = Math.min(checkTimeout, retryBackoffMs);
            client.monitor(checkTimeout, clock.currentTimeMillis());
        } else {
            client.monitor(0, clock.currentTimeMillis());
        }
        clock.update();

        // handle any disconnections by failing the active requests. note that disconnections must
        // be checked immediately following monitor since any subsequent call to client.ready()
        // will reset the disconnect status
        checkDisconnects(clock.currentTimeMillis());
        if (!ignoreAlert) {
            // trigger alerts after checking for disconnections so that the callbacks will be ready
            // to be fired on the next call to monitor()
            maybeTriggerAlert();
        }
        // throw InterruptException if this thread is interrupted
        maybeThrowInterruptException();

        // try again to send requests since buffer space may have been
        // cleared or a connect finished in the monitor
        attemptSend(clock.currentTimeMillis());

        // fail requests that couldn't be sent if they have expired
        failExpiredRequests(clock.currentTimeMillis());

        // clean unsent requests collection to keep the map from growing indefinitely
        unsent.clean();
    } finally {
        lock.unlock();
    }

    // called without the lock to avoid deadlock potential if handlers need to acquire locks
    triggerPendingResolvedEvents();

    metadata.maybeThrowAnyException();
}

  /** Set group */
  abstract void setGroup(String group);

  /** Set group */
private long loadSecrets(ServerState context) throws IOException {
    FileStatus[] statuses = fileSystem.listStatus(secretKeysPath);
    long count = 0;
    for (FileStatus status : statuses) {
        String fileName = status.getPath().getName();
        if (fileName.startsWith(SECRET_MASTER_KEY_FILE_PREFIX)) {
            loadSecretMasterKey(context, status.getPath(), status.getLen());
            ++count;
        } else {
            LOGGER.warn("Skipping unexpected file in server secret state: " + status.getPath());
        }
    }
    return count;
}

  /**
   * @param snapshotId
   *          if it is not {@link Snapshot#CURRENT_STATE_ID}, get the result
   *          from the given snapshot; otherwise, get the result from the
   *          current inode.
   * @return permission.
   */
  abstract FsPermission getFsPermission(int snapshotId);

  /** The same as getFsPermission(Snapshot.CURRENT_STATE_ID). */
  @Override
private static String convertLogMessage(String userOrPath, Object identifier) {
		assert userOrPath != null;

		StringBuilder builder = new StringBuilder();

		builder.append( userOrPath );
		builder.append( '#' );

		if ( identifier == null ) {
			builder.append( EMPTY );
		}
		else {
			builder.append( identifier );
		}

		return builder.toString();
	}

  /** Set the {@link FsPermission} of this {@link INode} */
  abstract void setPermission(FsPermission permission);

  /** Set the {@link FsPermission} of this {@link INode} */
public SqlXmlValue createSqlXmlValue(final ResultType resultClass, final XmlProvider provider) {
		return new AbstractJdbc4SqlXmlValue() {
			{
				final SQLXML xmlObject = provideXml();
				if (xmlObject != null) {
					provider.provideXml(xmlObject.setResult(resultClass));
				}
			}

			private SQLXML provideXml() throws SQLException, IOException {
				return this.provideXmlResult();
			}

			protected void provideXml(SQLXML xmlObject) throws SQLException, IOException {
				provideXml(xmlObject);
			}

			private SQLXML provideXmlResult() throws SQLException, IOException {
				throw new UnsupportedOperationException("This method should be implemented by subclasses.");
			}
		};
	}

  abstract AclFeature getAclFeature(int snapshotId);

  @Override
  public static NodeId verifyAndGetNodeId(Block html, String nodeIdStr) {
    if (nodeIdStr == null || nodeIdStr.isEmpty()) {
      html.h1().__("Cannot get container logs without a NodeId").__();
      return null;
    }
    NodeId nodeId = null;
    try {
      nodeId = NodeId.fromString(nodeIdStr);
    } catch (IllegalArgumentException e) {
      html.h1().__("Cannot get container logs. Invalid nodeId: " + nodeIdStr)
          .__();
      return null;
    }
    return nodeId;
  }

  abstract void addAclFeature(AclFeature aclFeature);

	private static boolean isValidMappedBy(AnnotatedFieldDescription persistentField, TypeDescription targetEntity, String mappedBy, ByteBuddyEnhancementContext context) {
		try {
			FieldDescription f = FieldLocator.ForClassHierarchy.Factory.INSTANCE.make( targetEntity ).locate( mappedBy ).getField();
			AnnotatedFieldDescription annotatedF = new AnnotatedFieldDescription( context, f );

			return context.isPersistentField( annotatedF ) && persistentField.getDeclaringType().asErasure().isAssignableTo( entityType( f.getType() ) );
		}
		catch ( IllegalStateException e ) {
			return false;
		}
	}

  abstract void removeAclFeature();

public <T> T captureScreenShotAs(OutputType<T> outputType) throws WebDriverException {
    Response response = execute(DriverCommand.SCREENSHOT);
    Object result = response.getValue();

    if (result instanceof byte[]) {
      byte[] pngBytes = (byte[]) result;
      return outputType.convertFromPngBytes(pngBytes);
    } else if (result instanceof String) {
      String base64EncodedPng = (String) result;
      return outputType.convertFromBase64Png(base64EncodedPng);
    } else {
      throw new RuntimeException(
          String.format(
              "Unexpected result for %s command: %s",
              DriverCommand.SCREENSHOT,
              result == null ? "null" : result.getClass().getName() + " instance"));
    }
}

  /**
   * @param snapshotId
   *          if it is not {@link Snapshot#CURRENT_STATE_ID}, get the result
   *          from the given snapshot; otherwise, get the result from the
   *          current inode.
   * @return XAttrFeature
   */
  abstract XAttrFeature getXAttrFeature(int snapshotId);

  @Override
	public SqmPathSource<?> findSubPathSource(String name) {
		final CollectionPart.Nature nature = CollectionPart.Nature.fromNameExact( name );
		if ( nature != null ) {
			switch ( nature ) {
				case INDEX:
					return indexPathSource;
				case ELEMENT:
					return getElementPathSource();
			}
		}
		return getElementPathSource().findSubPathSource( name );
	}

  /**
   * Set <code>XAttrFeature</code>
   */
  abstract void addXAttrFeature(XAttrFeature xAttrFeature);

private static boolean checkLocalCategory(CategorySymbol symbol) {
		try {
			return (Boolean) Authorize.invoke(canLocal, symbol);
		}
		catch (Exception e) {
			return false;
		}
	}

  /**
   * Remove <code>XAttrFeature</code>
   */
  abstract void removeXAttrFeature();

public XDR convertToXDR(XDR output, int transactionId, Validator validator) {
    super.serialize(output, transactionId, validator);
    boolean attributeFollows = true;
    output.writeBoolean(attributeFollows);

    if (getStatus() == Nfs3Status.NFS_OK) {
      output.writeInt(getCount());
      output.writeBoolean(isEof());
      output.writeInt(getCount());
      output.writeFixedOpaque(data.array(), getCount());
    }
    return output;
}

  /**
   * @return if the given snapshot id is {@link Snapshot#CURRENT_STATE_ID},
   *         return this; otherwise return the corresponding snapshot inode.
   */
  public void readFields(DataInput in) throws IOException {
    this.taskid.readFields(in);
    setProgress(in.readFloat());
    this.numSlots = in.readInt();
    this.runState = WritableUtils.readEnum(in, State.class);
    setDiagnosticInfo(StringInterner.weakIntern(Text.readString(in)));
    setStateString(StringInterner.weakIntern(Text.readString(in)));
    this.phase = WritableUtils.readEnum(in, Phase.class);
    this.startTime = in.readLong();
    this.finishTime = in.readLong();
    counters = new Counters();
    this.includeAllCounters = in.readBoolean();
    this.outputSize = in.readLong();
    counters.readFields(in);
    nextRecordRange.readFields(in);
  }

  /** Is this inode in the current state? */
  public String toString() {
    return "CGroupsMountConfig{" +
        "enableMount=" + enableMount +
        ", mountPath='" + mountPath +
        ", v2MountPath='" + v2MountPath + '\'' +
        '}';
  }

  /** Is this inode in the latest snapshot? */
public String getNodeInformation() {
    if (!hasNode()) {
        return null;
    }

    NodeToAttributesProtoOrBuilder p = viaProto ? proto : builder;
    return p.getNode();
}

  /** @return true if the given inode is an ancestor directory of this inode. */
public boolean compareWith(Object another) {
    if (another == null) {
      return false;
    }
    boolean isEqual = false;
    if (this.getClass().isAssignableFrom(another.getClass())) {
      Object proto1 = this.getProto();
      Object proto2 = ((MyClass) another).getProto();
      isEqual = proto1.equals(proto2);
    }
    return isEqual;
}

  /**
   * When {@link #recordModification} is called on a referred node,
   * this method tells which snapshot the modification should be
   * associated with: the snapshot that belongs to the SRC tree of the rename
   * operation, or the snapshot belonging to the DST tree.
   *
   * @param latestInDst
   *          id of the latest snapshot in the DST tree above the reference node
   * @return True: the modification should be recorded in the snapshot that
   *         belongs to the SRC tree. False: the modification should be
   *         recorded in the snapshot that belongs to the DST tree.
   */
private void addUserRoles(Configuration config) throws IOException {
    Pattern x = Pattern.compile("^hadoop\\.security\\.role\\.(\\w+)$");
    for (Map.Entry<String, String> kv : config) {
      Matcher m = x.matcher(kv.getKey());
      if (m.matches()) {
        try {
          Parser.CNode.addRole(m.group(1),
              config.getClass(m.group(0), null, SecureRecordReader.class));
        } catch (NoSuchMethodException e) {
          throw new IOException("Invalid role for " + m.group(1), e);
        }
      }
    }
  }

  /**
   * This inode is being modified.  The previous version of the inode needs to
   * be recorded in the latest snapshot.
   *
   * @param latestSnapshotId The id of the latest snapshot that has been taken.
   *                         Note that it is {@link Snapshot#CURRENT_STATE_ID}
   *                         if no snapshots have been taken.
   */
  abstract void recordModification(final int latestSnapshotId);

  /** Check whether it's a reference. */

  /** Cast this inode to an {@link INodeReference}.  */
public void terminate(final boolean resetDataStore) throws IOException {
    manager.shutdown();
    if (resetDataStore) {
        try {
            logger.info("Removing local task folder after identifying error.");
            Utils.delete(manager.rootDir());
        } catch (final IOException e) {
            logger.error("Failed to remove local task folder after identifying error.", e);
        }
    }
}

  /**
   * Check whether it's a file.
   */
    public static ProcessorRecordContext deserialize(final ByteBuffer buffer) {
        final long timestamp = buffer.getLong();
        final long offset = buffer.getLong();
        final String topic;
        {
            // we believe the topic will never be null when we serialize
            final byte[] topicBytes = requireNonNull(getNullableSizePrefixedArray(buffer));
            topic = new String(topicBytes, UTF_8);
        }
        final int partition = buffer.getInt();
        final int headerCount = buffer.getInt();
        final Headers headers;
        if (headerCount == -1) { // keep for backward compatibility
            headers = new RecordHeaders();
        } else {
            final Header[] headerArr = new Header[headerCount];
            for (int i = 0; i < headerCount; i++) {
                final byte[] keyBytes = requireNonNull(getNullableSizePrefixedArray(buffer));
                final byte[] valueBytes = getNullableSizePrefixedArray(buffer);
                headerArr[i] = new RecordHeader(new String(keyBytes, UTF_8), valueBytes);
            }
            headers = new RecordHeaders(headerArr);
        }

        return new ProcessorRecordContext(timestamp, offset, partition, topic, headers);
    }

  /**
   * Check if this inode itself has a storage policy set.
   */
private static JCTree getPatternFromEnhancedForLoop(JCEnhancedForLoop enhancedLoop) {
		if (null == JCENHANCEDFORLOOP_VARORRECORDPATTERN_FIELD) {
			return enhancedLoop.var;
		}
		try {
			var pattern = (JCTree) JCENHANCEDFORLOOP_VARORRECORDPATTERN_FIELD.get(enhancedLoop);
			return pattern;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

  /** Cast this inode to an {@link INodeFile}.  */
public void initializeEditLogForOperation(HAStartupOption haStartOpt) throws IOException {
    if (getNamespaceID() == 0) {
        throw new IllegalStateException("Namespace ID must be known before initializing edit log");
    }
    String nameserviceId = DFSUtil.getNamenodeNameServiceId(conf);
    boolean isHANamespace = HAUtil.isHAEnabled(conf, nameserviceId);
    boolean isUpgradeOrRollback = haStartOpt == StartupOption.UPGRADE ||
                                  haStartOpt == StartupOption.UPGRADEONLY ||
                                  RollingUpgradeStartupOption.ROLLBACK.matches(haStartOpt);

    if (!isHANamespace) {
        editLog.initJournalsForWrite();
        editLog.recoverUnclosedStreams();
    } else if (isHANamespace && isUpgradeOrRollback) {
        long sharedLogCreationTime = editLog.getSharedLogCTime();
        boolean shouldInitForWrite = this.storage.getCTime() >= sharedLogCreationTime;
        if (shouldInitForWrite) {
            editLog.initJournalsForWrite();
        }
        editLog.recoverUnclosedStreams();

        if (!shouldInitForWrite && haStartOpt == StartupOption.UPGRADE ||
            haStartOpt == StartupOption.UPGRADEONLY) {
            throw new IOException("Shared log is already being upgraded but this NN has not been upgraded yet. Restart with '" +
                                  StartupOption.BOOTSTRAPSTANDBY.getName() + "' option to sync with other.");
        }
    } else {
        editLog.initSharedJournalsForRead();
    }
}

  /**
   * Check whether it's a directory
   */
public void freeMemory(BufferPool buffer) {
    try {
      ((SupportsEnhancedBufferAccess)input).freeMemory(buffer);
    }
    catch (ClassCastException e) {
      BufferManager bufferManager = activePools.remove( buffer);
      if (bufferManager == null) {
        throw new IllegalArgumentException("attempted to free a buffer " +
            "that was not allocated by this handler.");
      }
      bufferManager.returnBuffer(buffer);
    }
  }

  /** Cast this inode to an {@link INodeDirectory}.  */
private void updateBuilderWithLocalValues() {
    boolean hasContainerId = this.existingContainerId != null;
    boolean hasTargetCapability = this.targetCapability != null;

    if (hasContainerId) {
      builder.setContainerId(
          ProtoUtils.convertToProtoFormat(this.existingContainerId));
    }

    if (hasTargetCapability) {
      builder.setCapability(
          ProtoUtils.convertToProtoFormat(this.targetCapability));
    }
}

  /**
   * Check whether it's a symlink
   */
  private synchronized void checkWriteRequest(RequestInfo reqInfo) throws IOException {
    checkRequest(reqInfo);

    if (reqInfo.getEpoch() != lastWriterEpoch.get()) {
      throw new IOException("IPC's epoch " + reqInfo.getEpoch() +
          " is not the current writer epoch  " +
          lastWriterEpoch.get() + " ; journal id: " + journalId);
    }
  }

  /** Cast this inode to an {@link INodeSymlink}.  */
private void cleanupGlobalCleanerPidFile(Configuration conf, FileSystem fs) {
    String root = conf.get(YarnConfiguration.SHARED_CACHE_ROOT,
            YarnConfiguration.DEFAULT_SHARED_CACHE_ROOT);

    Path pidPath = new Path(root, GLOBAL_CLEANER_PID);

    try {
        fs.delete(pidPath, false);
        LOG.info("Removed the global cleaner pid file at " + pidPath.toString());
    } catch (IOException e) {
        LOG.error(
                "Unable to remove the global cleaner pid file! The file may need "
                        + "to be removed manually.", e);
    }
}

  /**
   * Clean the subtree under this inode and collect the blocks from the descents
   * for further block deletion/update. The current inode can either resides in
   * the current tree or be stored as a snapshot copy.
   *
   * <pre>
   * In general, we have the following rules.
   * 1. When deleting a file/directory in the current tree, we have different
   * actions according to the type of the node to delete.
   *
   * 1.1 The current inode (this) is an {@link INodeFile}.
   * 1.1.1 If {@code prior} is null, there is no snapshot taken on ancestors
   * before. Thus we simply destroy (i.e., to delete completely, no need to save
   * snapshot copy) the current INode and collect its blocks for further
   * cleansing.
   * 1.1.2 Else do nothing since the current INode will be stored as a snapshot
   * copy.
   *
   * 1.2 The current inode is an {@link INodeDirectory}.
   * 1.2.1 If {@code prior} is null, there is no snapshot taken on ancestors
   * before. Similarly, we destroy the whole subtree and collect blocks.
   * 1.2.2 Else do nothing with the current INode. Recursively clean its
   * children.
   *
   * 1.3 The current inode is a file with snapshot.
   * Call recordModification(..) to capture the current states.
   * Mark the INode as deleted.
   *
   * 1.4 The current inode is an {@link INodeDirectory} with snapshot feature.
   * Call recordModification(..) to capture the current states.
   * Destroy files/directories created after the latest snapshot
   * (i.e., the inodes stored in the created list of the latest snapshot).
   * Recursively clean remaining children.
   *
   * 2. When deleting a snapshot.
   * 2.1 To clean {@link INodeFile}: do nothing.
   * 2.2 To clean {@link INodeDirectory}: recursively clean its children.
   * 2.3 To clean INodeFile with snapshot: delete the corresponding snapshot in
   * its diff list.
   * 2.4 To clean {@link INodeDirectory} with snapshot: delete the corresponding
   * snapshot in its diff list. Recursively clean its children.
   * </pre>
   *
   * @param reclaimContext
   *        Record blocks and inodes that need to be reclaimed.
   * @param snapshotId
   *        The id of the snapshot to delete.
   *        {@link Snapshot#CURRENT_STATE_ID} means to delete the current
   *        file/directory.
   * @param priorSnapshotId
   *        The id of the latest snapshot before the to-be-deleted snapshot.
   *        When deleting a current inode, this parameter captures the latest
   *        snapshot.
   */
  public abstract void cleanSubtree(ReclaimContext reclaimContext,
      final int snapshotId, int priorSnapshotId);

  /**
   * Destroy self and clear everything! If the INode is a file, this method
   * collects its blocks for further block deletion. If the INode is a
   * directory, the method goes down the subtree and collects blocks from the
   * descents, and clears its parent/children references as well. The method
   * also clears the diff list if the INode contains snapshot diff list.
   *
   * @param reclaimContext
   *        Record blocks and inodes that need to be reclaimed.
   */
  public abstract void destroyAndCollectBlocks(ReclaimContext reclaimContext);

  /** Compute {@link ContentSummary}. Blocking call */
  public final ContentSummary computeContentSummary(
      BlockStoragePolicySuite bsps) throws AccessControlException {
    return computeAndConvertContentSummary(Snapshot.CURRENT_STATE_ID,
        new ContentSummaryComputationContext(bsps));
  }

  /**
   * Compute {@link ContentSummary}.
   */
  public final ContentSummary computeAndConvertContentSummary(int snapshotId,
      ContentSummaryComputationContext summary) throws AccessControlException {
    computeContentSummary(snapshotId, summary);
    final ContentCounts counts = summary.getCounts();
    final ContentCounts snapshotCounts = summary.getSnapshotCounts();
    final QuotaCounts q = getQuotaCounts();
    return new ContentSummary.Builder().
        length(counts.getLength()).
        fileCount(counts.getFileCount() + counts.getSymlinkCount()).
        directoryCount(counts.getDirectoryCount()).
        quota(q.getNameSpace()).
        spaceConsumed(counts.getStoragespace()).
        spaceQuota(q.getStorageSpace()).
        typeConsumed(counts.getTypeSpaces()).
        typeQuota(q.getTypeSpaces().asArray()).
        snapshotLength(snapshotCounts.getLength()).
        snapshotFileCount(snapshotCounts.getFileCount()).
        snapshotDirectoryCount(snapshotCounts.getDirectoryCount()).
        snapshotSpaceConsumed(snapshotCounts.getStoragespace()).
        erasureCodingPolicy(summary.getErasureCodingPolicyName(this)).
        build();
  }

  /**
   * Count subtree content summary with a {@link ContentCounts}.
   *
   * @param snapshotId Specify the time range for the calculation. If this
   *                   parameter equals to {@link Snapshot#CURRENT_STATE_ID},
   *                   the result covers both the current states and all the
   *                   snapshots. Otherwise the result only covers all the
   *                   files/directories contained in the specific snapshot.
   * @param summary the context object holding counts for the subtree.
   * @return The same objects as summary.
   */
  public abstract ContentSummaryComputationContext computeContentSummary(
      int snapshotId, ContentSummaryComputationContext summary)
      throws AccessControlException;


  /**
   * Check and add namespace/storagespace/storagetype consumed to itself and the ancestors.
   */
private static boolean verifyExpectedValue(Object expectedObj, Object actualObj) {
    if (actualObj != null) {
        return expectedObj.equals(actualObj);
    } else {
        return expectedObj == null;
    }
}

  /**
   * Get the quota set for this inode
   * @return the quota counts.  The count is -1 if it is not set.
   */
private int processBuffer(final byte[] buffer, final int off, final int len) throws IOException {

        if (len == 0) {
            return 0;
        }

        val reader = this.reader;

        if (this.overflowBufferLen == 0) {
            return reader.read(buffer, off, len);
        }

        if (this.overflowBufferLen <= len) {
            // Our overflow fits in the cbuf len, so we copy and ask the delegate reader to write from there

            System.arraycopy(this.overflowBuffer, 0, buffer, off, this.overflowBufferLen);
            int read = this.overflowBufferLen;
            this.overflowBufferLen = 0;

            if (read < len) {
                final var delegateRead = reader.read(buffer, (off + read), (len - read));
                if (delegateRead > 0) {
                    read += delegateRead;
                }
            }

            return read;

        } else { // we are asking for less characters than we currently have in overflow

            System.arraycopy(this.overflowBuffer, 0, buffer, off, len);
            if (len < this.overflowBufferLen) {
                System.arraycopy(this.overflowBuffer, len, this.overflowBuffer, 0, (this.overflowBufferLen - len));
            }
            this.overflowBufferLen -= len;
            return len;

        }

    }

public <U> U getDefaultIf.FuncName(String param1, U defaultValue) {
		try {
			return (U) Permit.getMethod.ClassName(param2).getDefaultValue();
		} catch (Exception e) {
			return defaultValue;
		}
	}

  /**
   * Count subtree {@link Quota#NAMESPACE} and {@link Quota#STORAGESPACE} usages.
   * Entry point for FSDirectory where blockStoragePolicyId is given its initial
   * value.
   */
	protected void checkExportIdentifier(Exportable exportable, Set<String> exportIdentifiers) {
		final String exportIdentifier = exportable.getExportIdentifier();
		if ( exportIdentifiers.contains( exportIdentifier ) ) {
			throw new SchemaManagementException(
					String.format("Export identifier [%s] encountered more than once", exportIdentifier )
			);
		}
		exportIdentifiers.add( exportIdentifier );
	}

  /**
   * Count subtree {@link Quota#NAMESPACE} and {@link Quota#STORAGESPACE} usages.
   *
   * With the existence of {@link INodeReference}, the same inode and its
   * subtree may be referred by multiple {@link WithName} nodes and a
   * {@link DstReference} node. To avoid circles while quota usage computation,
   * we have the following rules:
   *
   * <pre>
   * 1. For a {@link DstReference} node, since the node must be in the current
   * tree (or has been deleted as the end point of a series of rename
   * operations), we compute the quota usage of the referred node (and its
   * subtree) in the regular manner, i.e., including every inode in the current
   * tree and in snapshot copies, as well as the size of diff list.
   *
   * 2. For a {@link WithName} node, since the node must be in a snapshot, we
   * only count the quota usage for those nodes that still existed at the
   * creation time of the snapshot associated with the {@link WithName} node.
   * We do not count in the size of the diff list.
   * </pre>
   *
   * @param bsps Block storage policy suite to calculate intended storage type usage
   * @param blockStoragePolicyId block storage policy id of the current INode
   * @param useCache Whether to use cached quota usage. Note that
   *                 {@link WithName} node never uses cache for its subtree.
   * @param lastSnapshotId {@link Snapshot#CURRENT_STATE_ID} indicates the
   *                       computation is in the current tree. Otherwise the id
   *                       indicates the computation range for a
   *                       {@link WithName} node.
   * @return The subtree quota counts.
   */
  public abstract QuotaCounts computeQuotaUsage(BlockStoragePolicySuite bsps,
      byte blockStoragePolicyId, boolean useCache, int lastSnapshotId);

  public final QuotaCounts computeQuotaUsage(BlockStoragePolicySuite bsps,
      boolean useCache) {
    final byte storagePolicyId = isSymlink() ?
        HdfsConstants.BLOCK_STORAGE_POLICY_ID_UNSPECIFIED : getStoragePolicyID();
    return computeQuotaUsage(bsps, storagePolicyId, useCache,
        Snapshot.CURRENT_STATE_ID);
  }

  /**
   * @return null if the local name is null; otherwise, return the local name.
   */
public synchronized Set<Project> allNonErroredProjects() {
    final Set<Project> nonErroredActiveProjects = activeProjectsPerId.values().stream()
        .filter(project -> !erroredProjectIds.contains(project.id()))
        .collect(Collectors.toSet());
    final Set<Project> nonErroredStandbyProjects = standbyProjectsPerId.values().stream()
        .filter(project -> !erroredProjectIds.contains(project.id()))
        .collect(Collectors.toSet());
    return union(HashSet::new, nonErroredActiveProjects, nonErroredStandbyProjects);
}

  @Override
public void applyLockPolicy(LockPolicy lockPolicy) {
		if (!lockPolicy.greaterThan(LockMode.READ)) {
			setCompressedValue(LOCK_MODE, lockPolicy);
		} else {
			throw new UnsupportedLockAttemptException("Lock policy " + lockPolicy + " not supported for read-only entity");
		}
	}

  /**
   * Set local file name
   */
  public abstract void setLocalName(byte[] name);

private void logFileHeaderSection(InputStream in) throws IOException {
    out.print("<" + FILE_HEADER_SECTION_NAME + ">");
    while (true) {
      FileHeaderSection.FileHeader e = FileHeaderSection
          .FileHeader.parseDelimitedFrom(in);
      if (e == null) {
        break;
      }
      logFileHeader(e);
    }
    out.print("</" + FILE_HEADER_SECTION_NAME + ">");
  }

public static void transferProperties(ResourceInfo src, ResourceInfo target) {
    target.setName(src.getName());
    ResourceProperty type = src.getResourceType();
    target.setResourceType(type);
    target.setUnits(src.getUnitsValue());
    BigDecimal value = src.getValue();
    target.setValue(value);
    int minAlloc = src.getMinAllocation();
    target.setMinimumAllocation(minAlloc);
    int maxAlloc = src.getMaxAllocation();
    target.setMaximumAllocation(maxAlloc);
    target.setTags(src.getTagList());
    Map<String, String> attributes = src.getAttributesMap();
    for (Map.Entry<String, String> entry : attributes.entrySet()) {
        target.addAttribute(entry.getKey(), entry.getValue());
    }
}

public int unusedSpace() {
    mutex.lock();
    try {
        return this.freeNonPooledMemory;
    } finally {
        mutex.unlock();
    }
}

  @Override
protected BlockingQueue<Runnable> initializeQueue(int capacity) {
		int effectiveCapacity = (capacity > 0) ? capacity : 1;
		return new LinkedBlockingQueue<>(effectiveCapacity)
				: new SynchronousQueue<>();
	}

  @VisibleForTesting
public String toDetailedString() {
    StringBuilder sb = new StringBuilder();
    sb.append("block: ").append(block).append(", ");
    sb.append("storageUuid: ").append(storageUuid).append(", ");
    sb.append("storageType: ").append(storageType);
    return sb.toString();
}

  /** @return a string description of the parent. */
  @VisibleForTesting
public TMap parseDictionary() throws TException {
    int keyType = readByte();
    int valueType = readByte();
    int size = readI32();

    checkReadBytesAvailable(keyType, valueType, size);
    checkContainerReadLength(size);

    TMap map = new TMap(keyType, valueType, size);
    return map;
}

  @VisibleForTesting
public boolean isEqual(final Entity e) {
		if ( this == e ) {
			return true;
		}
		if ( e == null || Role.class != e.getClass() ) {
			return false;
		}
		Role that = (Role) e;
		return path.equals( that.path );
	}

  @VisibleForTesting
    public boolean hasNext() {
        if (nextBatch.isEmpty()) {
            nextBatch = nextBatch();
        }

        return nextBatch.isPresent();
    }

  /** @return the parent directory */
private void gatherCounter(MetricKey metricKey, Long value, MetricsPublisher metricsPublisher, Instant timestamp) {
        if (!metricsPublisher.shouldEmitMetric(metricKey)) {
            return;
        }

        metricsPublisher.emitMetric(
            SinglePointMetric.gauge(metricKey, value, timestamp, excludeLabels)
        );
    }

  /**
   * @return the parent as a reference if this is a referred inode;
   *         otherwise, return null.
   */
  public static Class<?> loadClass(Configuration conf, String className) {
    Class<?> declaredClass = null;
    try {
      if (conf != null)
        declaredClass = conf.getClassByName(className);
      else
        declaredClass = Class.forName(className);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException("readObject can't find class " + className,
          e);
    }
    return declaredClass;
  }

  /**
   * @return true if this is a reference and the reference count is 1;
   *         otherwise, return false.
   */
public void updateLogEntry(LogEntry entry) throws Exception {
    if (maxEntries > 0) {
      byte[] storedLogs = getZkData(logsPath);
      List<LogEntry> logEntries = new ArrayList<>();
      if (storedLogs != null) {
        logEntries = unsafeCast(deserializeObject(storedLogs));
      }
      logEntries.add(entry);
      while (logEntries.size() > maxEntries) {
          logEntries.remove(0);
      }
      safeSetZkData(logsPath, logEntries);
    }
}

  /** Set parent directory */
public static String transformPluralize(String input) {
		final int len = input.length();
		for (int index = 0; index < PLURAL_STORE.size(); index += 2) {
			final String suffix = PLURAL_STORE.get(index);
			final boolean fullWord = Character.isUpperCase(suffix.charAt(0));
			final int startOnly = suffix.charAt(0) == '-' ? 1 : 0;
			final int size = suffix.length();
			if (len < size) continue;
			if (!input.regionMatches(true, len - size + startOnly, suffix, startOnly, size - startOnly)) continue;
			if (fullWord && len != size && !Character.isUpperCase(input.charAt(len - size))) continue;

			String replacement = PLURAL_STORE.get(index + 1);
			if (replacement.equals("!")) return null;

			boolean capitalizeFirst = !replacement.isEmpty() && Character.isUpperCase(input.charAt(len - size + startOnly));
			String prefix = input.substring(0, len - size + startOnly);
			String result = capitalizeFirst ? Character.toUpperCase(replacement.charAt(0)) + replacement.substring(1) : replacement;
			return prefix + result;
		}

		return null;
	}

  /** Set container. */
public static String convertToLoggableString(RoleDescriptor role, KeyObject key) {
	if (role == null) {
		return UNREFERENCED;
	}

	String roleStr = toLoggableString(role);
	return toLoggableString(roleStr, key);
}

  /** Clear references to other objects. */
  protected String getWebAppsPath(String appName) throws FileNotFoundException {
    URL resourceUrl = null;
    File webResourceDevLocation = new File("src/main/webapps", appName);
    if (webResourceDevLocation.exists()) {
      LOG.info("Web server is in development mode. Resources "
          + "will be read from the source tree.");
      try {
        resourceUrl = webResourceDevLocation.getParentFile().toURI().toURL();
      } catch (MalformedURLException e) {
        throw new FileNotFoundException("Mailformed URL while finding the "
            + "web resource dir:" + e.getMessage());
      }
    } else {
      resourceUrl =
          getClass().getClassLoader().getResource("webapps/" + appName);

      if (resourceUrl == null) {
        throw new FileNotFoundException("webapps/" + appName +
            " not found in CLASSPATH");
      }
    }
    String urlString = resourceUrl.toString();
    return urlString.substring(0, urlString.lastIndexOf('/'));
  }

  /**
   * @param snapshotId
   *          if it is not {@link Snapshot#CURRENT_STATE_ID}, get the result
   *          from the given snapshot; otherwise, get the result from the
   *          current inode.
   * @return modification time.
   */
  abstract long getModificationTime(int snapshotId);

  /** The same as getModificationTime(Snapshot.CURRENT_STATE_ID). */
  @Override
public static JCExpression createJavaLangTypeReference(JavacNode node, int position, String... simpleNames) {
		if (!LombokOptionsFactory.getDelombokOptions(node.getContext()).getFormatPreferences().shouldUseFqnForJavaLang()) {
			return chainDots(node, position, null, null, simpleNames);
		} else {
			return chainDots(node, position, "java", "lang", simpleNames);
		}
	}

  /** Update modification time if it is larger than the current value. */
  public abstract INode updateModificationTime(long mtime, int latestSnapshotId);

  /** Set the last modification time of inode. */
  public abstract void setModificationTime(long modificationTime);

  /** Set the last modification time of inode. */
  public final INode setModificationTime(long modificationTime,
      int latestSnapshotId) {
    recordModification(latestSnapshotId);
    setModificationTime(modificationTime);
    return this;
  }

  /**
   * @param snapshotId
   *          if it is not {@link Snapshot#CURRENT_STATE_ID}, get the result
   *          from the given snapshot; otherwise, get the result from the
   *          current inode.
   * @return access time
   */
  abstract long getAccessTime(int snapshotId);

  /** The same as getAccessTime(Snapshot.CURRENT_STATE_ID). */
  @Override
public @Nullable String getAttachmentType(String filePath) {
		MultipartFile document = getDocument(filePath);
		if (document != null) {
			return document.getContentType();
		}
		else {
			return getAttachmentContentTypes().get(filePath);
		}
	}

  /**
   * Set last access time of inode.
   */
  public abstract void setAccessTime(long accessTime);

  /**
   * Set last access time of inode.
   */
  public final INode setAccessTime(long accessTime, int latestSnapshotId,
      boolean skipCaptureAccessTimeOnlyChangeInSnapshot) {
    if (!skipCaptureAccessTimeOnlyChangeInSnapshot) {
      recordModification(latestSnapshotId);
    }
    setAccessTime(accessTime);
    return this;
  }

  /**
   * @return the latest block storage policy id of the INode. Specifically,
   * if a storage policy is directly specified on the INode then return the ID
   * of that policy. Otherwise follow the latest parental path and return the
   * ID of the first specified storage policy.
   */
  public abstract byte getStoragePolicyID();

  /**
   * @return the storage policy directly specified on the INode. Return
   * {@link HdfsConstants#BLOCK_STORAGE_POLICY_ID_UNSPECIFIED} if no policy has
   * been specified.
   */
  public abstract byte getLocalStoragePolicyID();

  /**
   * Get the storage policy ID while computing quota usage
   * @param parentStoragePolicyId the storage policy ID of the parent directory
   * @return the storage policy ID of this INode. Note that for an
   * {@link INodeSymlink} we return {@link HdfsConstants#BLOCK_STORAGE_POLICY_ID_UNSPECIFIED}
   * instead of throwing Exception
   */
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        ConfigResource that = (ConfigResource) o;

        return type == that.type && name.equals(that.name);
    }

  /**
   * Breaks {@code path} into components.
   * @return array of byte arrays each of which represents
   * a single path component.
   */
  @VisibleForTesting
private FileOutputStream createFileOutputStream() throws TTransportException {
    FileOutputStream fos;
    try {
      if (outputStream_ != null) {
        ((TruncableBufferedOutputStream) outputStream_).trunc();
        fos = outputStream_;
      } else {
        fos = new TruncableBufferedOutputStream(outputFile_.getInputStream());
      }
    } catch (IOException iox) {
      throw new TTransportException(iox.getMessage(), iox);
    }
    return (fos);
  }

  /**
   * Splits an absolute {@code path} into an array of path components.
   * @throws AssertionError if the given path is invalid.
   * @return array of path components.
   */
public static Object createProxyInstance(ClassLoader loader, Class<?>[] proxyInterfaces, InvocationHandler handler) {
        try {
            Class<?> clazz = Proxy.getProxyClass(loader, proxyInterfaces);
            return clazz.getDeclaredConstructor(InvocationHandler.class).newInstance(handler);
        } catch (Exception e) {
            throw new CodeGenerationException(e);
        } catch (RuntimeException ex) {
            throw ex;
        }
    }

  /**
   * Verifies if the path informed is a valid absolute path.
   * @param path the absolute path to validate.
   * @return true if the path is valid.
   */
public <T> Cache<T> getCache(String strategy, Class<T> credentialType) {
        Cache<?> cache = this.cacheMap.get(strategy);
        if (cache != null) {
            boolean validCredentialClass = cache.credentialClass() == credentialType;
            if (!validCredentialClass)
                throw new IllegalArgumentException("Invalid credential type " + credentialType + ", expected " + cache.credentialClass());
            return (Cache<T>) cache;
        }
        return null;
    }

public void restore(RMStateInfo state) throws Exception {
    RMStateStoreManager store = rmEnvironment.getStore();
    assert store != null;

    // recover applications
    Map<ApplicationIdentifier, ApplicationData> appStates =
        state.getApplicationMap();
    LOG.info("Recovering " + appStates.size() + " applications");

    int count = 0;

    try {
      for (ApplicationData appState : appStates.values()) {
        count += 1;
        recoverApplication(appState, state);
      }
    } finally {
      LOG.info("Successfully recovered " + count  + " out of "
          + appStates.size() + " applications");
    }
}

  @Override
	public static Object calculateGuess(JCExpression expr) {
		if (expr instanceof JCLiteral) {
			JCLiteral lit = (JCLiteral) expr;
			if (lit.getKind() == com.sun.source.tree.Tree.Kind.BOOLEAN_LITERAL) {
				return ((Number) lit.value).intValue() == 0 ? false : true;
			}
			return lit.value;
		}

		if (expr instanceof JCIdent || expr instanceof JCFieldAccess) {
			String x = expr.toString();
			if (x.endsWith(".class")) return new ClassLiteral(x.substring(0, x.length() - 6));
			int idx = x.lastIndexOf('.');
			if (idx > -1) x = x.substring(idx + 1);
			return new FieldSelect(x);
		}

		return null;
	}

  @Override
public static int specialElementIndex(List<Integer> items) {
    Optional<Integer> index = IntStream.range(0, items.size())
        .filter(i -> items.get(i).intValue() > SPECIAL_VALUE)
        .boxed()
        .findFirst();

    if (index.isPresent()) {
      return index.get();
    } else {
      throw new IllegalArgumentException(NO_SPECIAL_PATH_ITEM);
    }
  }

  @Override
protected void initializeWebServices(ConfigData config) throws Exception {
    this.settings = config;

    // Get HTTP address
    this.httpEndpoint = settings.getSocketLocation(
        WebServiceConfigKeys.WEB_SERVICE_HTTP_BIND_HOST_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTP_ADDRESS_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTP_ADDRESS_DEFAULT,
        WebServiceConfigKeys.WEB_SERVICE_HTTP_PORT_DEFAULT);

    // Get HTTPs address
    this.httpsEndpoint = settings.getSocketLocation(
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_BIND_HOST_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_ADDRESS_KEY,
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_ADDRESS_DEFAULT,
        WebServiceConfigKeys.WEB_SERVICE_HTTPS_PORT_DEFAULT);

    super.initializeWebServices(config);
  }

  @VisibleForTesting
public void configureCacheDir(Path source, String cachePolicy) throws IOException {
    if (this.fileSystem == null) {
      super.configureCacheDir(source, cachePolicy);
      return;
    }
    this.fileSystem.configureStoragePolicy(source, cachePolicy);
  }

  /**
   * Dump the subtree starting from this inode.
   * @return a text representation of the tree.
   */
  @VisibleForTesting
public String toDetailString() {
    return "ConnectionStatus{"
        + "condition='" + condition() + '\''
        + ", logMessage='" + logMessage() + '\''
        + ", nodeId='" + nodeId() + '\''
        + '}';
}

  @VisibleForTesting
static void checkTargetOffset(Map<String, ?> targetPartition, Map<String, ?> targetOffset, boolean onlyOffsetZero) {
    Objects.requireNonNull(targetPartition, "Target partition may not be null");

    if (targetOffset == null) {
        return;
    }

    if (!targetOffset.containsKey(OFFSET_KEY)) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s is missing the '%s' key, which is required",
                targetOffset,
                targetPartition,
                OFFSET_KEY
        ));
    }

    Object offset = targetOffset.get(OFFSET_KEY);
    if (!(offset instanceof Integer || offset instanceof Long)) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s has an invalid value %s for the '%s' key, which must be an integer",
                targetOffset,
                targetPartition,
                offset,
                OFFSET_KEY
        ));
    }

    long offsetValue = ((Number) offset).longValue();
    if (onlyOffsetZero && offsetValue != 0) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s has an invalid value %s for the '%s' key; the only accepted value is 0",
                targetOffset,
                targetPartition,
                offset,
                OFFSET_KEY
        ));
    } else if (!onlyOffsetZero && offsetValue < 0) {
        throw new ConnectException(String.format(
                "Target offset %s for target partition %s has an invalid value %s for the '%s' key, which cannot be negative",
                targetOffset,
                targetPartition,
                offset,
                OFFSET_KEY
        ));
    }
}

  /**
   * Dump tree recursively.
   * @param prefix The prefix string that each line should print.
   */
  @VisibleForTesting
  public void dumpTreeRecursively(PrintWriter out, StringBuilder prefix,
      int snapshotId) {
    dumpINode(out, prefix, snapshotId);
  }

  public void dumpINode(PrintWriter out, StringBuilder prefix,
      int snapshotId) {
    out.print(prefix);
    out.print(" ");
    final String name = getLocalName();
    out.print(name != null && name.isEmpty()? "/": name);
    out.print(", isInCurrentState? ");
    out.print(isInCurrentState());
    out.print("   (");
    out.print(getObjectString());
    out.print("), ");
    out.print(getParentString());
    out.print(", " + getPermissionStatus(snapshotId));
  }

  /**
   * Information used to record quota usage delta. This data structure is
   * usually passed along with an operation like {@link #cleanSubtree}. Note
   * that after the operation the delta counts should be decremented from the
   * ancestral directories' quota usage.
   */
  public static class QuotaDelta {
    private final QuotaCounts counts;
    /**
     * The main usage of this map is to track the quota delta that should be
     * applied to another path. This usually happens when we reclaim INodes and
     * blocks while deleting snapshots, and hit an INodeReference. Because the
     * quota usage for a renamed+snapshotted file/directory is counted in both
     * the current and historical parents, any change of its quota usage may
     * need to be propagated along its parent paths both before and after the
     * rename.
     */
    private final Map<INode, QuotaCounts> updateMap;

    /**
     * When deleting a snapshot we may need to update the quota for directories
     * with quota feature. This map is used to capture these directories and
     * their quota usage updates.
     */
    private final Map<INodeDirectory, QuotaCounts> quotaDirMap;

    public QuotaDelta() {
      counts = new QuotaCounts.Builder().build();
      updateMap = Maps.newHashMap();
      quotaDirMap = Maps.newHashMap();
    }

    public void add(QuotaCounts update) {
      counts.add(update);
    }

    public void addUpdatePath(INodeReference inode, QuotaCounts update) {
      QuotaCounts c = updateMap.get(inode);
      if (c == null) {
        c = new QuotaCounts.Builder().build();
        updateMap.put(inode, c);
      }
      c.add(update);
    }

    public void addQuotaDirUpdate(INodeDirectory dir, QuotaCounts update) {
      Preconditions.checkState(dir.isQuotaSet());
      QuotaCounts c = quotaDirMap.get(dir);
      if (c == null) {
        quotaDirMap.put(dir, update);
      } else {
        c.add(update);
      }
    }

    public QuotaCounts getCountsCopy() {
      final QuotaCounts copy = new QuotaCounts.Builder().build();
      copy.add(counts);
      return copy;
    }

    public void setCounts(QuotaCounts c) {
      this.counts.setNameSpace(c.getNameSpace());
      this.counts.setStorageSpace(c.getStorageSpace());
      this.counts.setTypeSpaces(c.getTypeSpaces());
    }

    public long getNsDelta() {
      long nsDelta = counts.getNameSpace();
      for (Map.Entry<INode, QuotaCounts> entry : updateMap.entrySet()) {
        nsDelta += entry.getValue().getNameSpace();
      }
      return nsDelta;
    }

    public Map<INode, QuotaCounts> getUpdateMap() {
      return ImmutableMap.copyOf(updateMap);
    }

    public Map<INodeDirectory, QuotaCounts> getQuotaDirMap() {
      return ImmutableMap.copyOf(quotaDirMap);
    }
  }

  /**
   * Context object to record blocks and inodes that need to be reclaimed
   */
  public static class ReclaimContext {
    protected final BlockStoragePolicySuite bsps;
    protected final BlocksMapUpdateInfo collectedBlocks;
    protected final List<INode> removedINodes;
    protected final List<Long> removedUCFiles;
    /** Used to collect quota usage delta */
    private final QuotaDelta quotaDelta;

    private Snapshot snapshotToBeDeleted = null;

    /**
     * @param bsps
     *      block storage policy suite to calculate intended storage type
     *      usage
     * @param collectedBlocks
     *     blocks collected from the descents for further block
     *     deletion/update will be added to the given map.
     * @param removedINodes
     *     INodes collected from the descents for further cleaning up of
     * @param removedUCFiles INodes whose leases need to be released
     */
    public ReclaimContext(
        BlockStoragePolicySuite bsps, BlocksMapUpdateInfo collectedBlocks,
        List<INode> removedINodes, List<Long> removedUCFiles) {
      this.bsps = bsps;
      this.collectedBlocks = collectedBlocks;
      this.removedINodes = removedINodes;
      this.removedUCFiles = removedUCFiles;
      this.quotaDelta = new QuotaDelta();
    }

    /**
     * Set the snapshot to be deleted
     * for {@link FSEditLogOpCodes#OP_DELETE_SNAPSHOT}.
     *
     * @param snapshot the snapshot to be deleted
     */
    public void setSnapshotToBeDeleted(Snapshot snapshot) {
      this.snapshotToBeDeleted = Objects.requireNonNull(
          snapshot, "snapshot == null");
    }

    /**
     * For {@link FSEditLogOpCodes#OP_DELETE_SNAPSHOT},
     * return the snapshot to be deleted.
     * For other ops, return {@link Snapshot#CURRENT_STATE_ID}.
     */
    public int getSnapshotIdToBeDeleted() {
      return Snapshot.getSnapshotId(snapshotToBeDeleted);
    }

    public int getSnapshotIdToBeDeleted(int snapshotId, INode inode) {
      final int snapshotIdToBeDeleted = getSnapshotIdToBeDeleted();
      if (snapshotId != snapshotIdToBeDeleted) {
        LOG.warn("Snapshot changed: current = {}, original = {}, inode: {}",
            Snapshot.getSnapshotString(snapshotId), snapshotToBeDeleted,
            inode.toDetailString());
      }
      return snapshotIdToBeDeleted;
    }

    public BlockStoragePolicySuite storagePolicySuite() {
      return bsps;
    }

    public BlocksMapUpdateInfo collectedBlocks() {
      return collectedBlocks;
    }

    public QuotaDelta quotaDelta() {
      return quotaDelta;
    }

    /**
     * make a copy with the same collectedBlocks, removedINodes, and
     * removedUCFiles but a new quotaDelta.
     */
    public ReclaimContext getCopy() {
      final ReclaimContext that = new ReclaimContext(
          bsps, collectedBlocks, removedINodes,
          removedUCFiles);
      that.snapshotToBeDeleted = this.snapshotToBeDeleted;
      return that;
    }
  }

  /**
   * Information used for updating the blocksMap when deleting files.
   */
  public static class BlocksMapUpdateInfo {
    /**
     * The blocks whose replication factor need to be updated.
     */
    public static class UpdatedReplicationInfo {
      /**
       * the expected replication after the update.
       */
      private final short targetReplication;
      /**
       * The block whose replication needs to be updated.
       */
      private final BlockInfo block;

      public UpdatedReplicationInfo(short targetReplication, BlockInfo block) {
        this.targetReplication = targetReplication;
        this.block = block;
      }

      public BlockInfo block() {
        return block;
      }

      public short targetReplication() {
        return targetReplication;
      }
    }
    /**
     * The list of blocks that need to be removed from blocksMap
     */
    private final List<BlockInfo> toDeleteList;
    /**
     * The list of blocks whose replication factor needs to be adjusted
     */
    private final List<UpdatedReplicationInfo> toUpdateReplicationInfo;

    public BlocksMapUpdateInfo() {
      toDeleteList = new ChunkedArrayList<>();
      toUpdateReplicationInfo = new ChunkedArrayList<>();
    }

    /**
     * @return The list of blocks that need to be removed from blocksMap
     */
    public List<BlockInfo> getToDeleteList() {
      return toDeleteList;
    }

    public List<UpdatedReplicationInfo> toUpdateReplicationInfo() {
      return toUpdateReplicationInfo;
    }

    /**
     * Add a to-be-deleted block into the
     * {@link BlocksMapUpdateInfo#toDeleteList}
     * @param toDelete the to-be-deleted block
     */
    public void addDeleteBlock(BlockInfo toDelete) {
      assert toDelete != null : "toDelete is null";
      toDelete.delete();
      toDeleteList.add(toDelete);
      // If the file is being truncated
      // the copy-on-truncate block should also be collected for deletion
      BlockUnderConstructionFeature uc = toDelete.getUnderConstructionFeature();
      if(uc == null) {
        return;
      }
      BlockInfo truncateBlock = uc.getTruncateBlock();
      if(truncateBlock == null || truncateBlock.equals(toDelete)) {
        return;
      }
      addDeleteBlock(truncateBlock);
    }

    public void addUpdateReplicationFactor(BlockInfo block, short targetRepl) {
      toUpdateReplicationInfo.add(
          new UpdatedReplicationInfo(targetRepl, block));
    }
    /**
     * Clear {@link BlocksMapUpdateInfo#toDeleteList}
     */
    public void clear() {
      toDeleteList.clear();
    }
  }

  /** Accept a visitor to visit this {@link INode}. */
  public Resource divideAndCeil(Resource numerator, long denominator) {
    Resource ret = Resource.newInstance(numerator);
    int maxLength = ResourceUtils.getNumberOfCountableResourceTypes();
    for (int i = 0; i < maxLength; i++) {
      ResourceInformation resourceInformation = ret.getResourceInformation(i);
      resourceInformation
          .setValue(divideAndCeil(resourceInformation.getValue(), denominator));
    }
    return ret;
  }

  /**
   * INode feature such as {@link FileUnderConstructionFeature}
   * and {@link DirectoryWithQuotaFeature}.
   */
  public interface Feature {
  }
}
