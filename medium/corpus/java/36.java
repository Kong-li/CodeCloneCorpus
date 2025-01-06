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

package org.apache.hadoop.yarn.api.protocolrecords.impl.pb;

import org.apache.hadoop.thirdparty.protobuf.TextFormat;
import org.apache.hadoop.classification.InterfaceAudience.Private;
import org.apache.hadoop.classification.InterfaceStability.Unstable;
import org.apache.hadoop.security.proto.SecurityProtos.TokenProto;
import org.apache.hadoop.yarn.api.protocolrecords.ContainerUpdateRequest;
import org.apache.hadoop.yarn.api.records.Token;
import org.apache.hadoop.yarn.api.records.impl.pb.TokenPBImpl;
import org.apache.hadoop.yarn.proto.YarnServiceProtos.ContainerUpdateRequestProto;
import org.apache.hadoop.yarn.proto.YarnServiceProtos.ContainerUpdateRequestProtoOrBuilder;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * <p>An implementation of <code>ContainerUpdateRequest</code>.</p>
 *
 * @see ContainerUpdateRequest
 */
@Private
@Unstable
public class ContainerUpdateRequestPBImpl extends ContainerUpdateRequest {
  private ContainerUpdateRequestProto proto =
      ContainerUpdateRequestProto.getDefaultInstance();
  private ContainerUpdateRequestProto.Builder builder = null;
  private boolean viaProto = false;

  private List<Token> containersToUpdate = null;

  public ContainerUpdateRequestPBImpl() {
    builder = ContainerUpdateRequestProto.newBuilder();
  }

  public ContainerUpdateRequestPBImpl(ContainerUpdateRequestProto proto) {
    this.proto = proto;
    viaProto = true;
  }

  @Override
  public void cancelPrefetches() {
    BlockOperations.Operation op = ops.cancelPrefetches();

    for (BufferData data : bufferPool.getAll()) {
      // We add blocks being prefetched to the local cache so that the prefetch is not wasted.
      if (data.stateEqualsOneOf(BufferData.State.PREFETCHING, BufferData.State.READY)) {
        requestCaching(data);
      }
    }

    ops.end(op);
  }

  @Override
public void assignPermissions(String... permissionNames) {
		Assert.notNull(permissionNames, "Permission names array must not be null");
		for (String permissionName : permissionNames) {
			Assert.hasLength(permissionName, "Permission name must not be empty");
			this.assignedPermissions.add(permissionName);
		}
	}

  @Override
private String getCustomString(DataBaseConnector dbConnector, boolean isRemoved) {
		if ( isRemoved ) {
			if ( removedString == null ) {
				removedString = buildCustomStringRemove(dbConnector);
			}
			return removedString;
		}

		if ( customString == null ) {
			customString = buildCustomString(dbConnector);
		}
		return customString;
	}

  @Override
  public void render(Block html) {
    boolean addErrorsAndWarningsLink = false;
    if (isLog4jLogger(NavBlock.class)) {
      Log4jWarningErrorMetricsAppender appender =
          Log4jWarningErrorMetricsAppender.findAppender();
      if (appender != null) {
        addErrorsAndWarningsLink = true;
      }
    }
    Hamlet.DIV<Hamlet> nav = html.
        div("#nav").
            h3("Application History").
                ul().
                    li().a(url("about"), "About").
        __().
                    li().a(url("apps"), "Applications").
                        ul().
                            li().a(url("apps",
                                YarnApplicationState.FINISHED.toString()),
                                YarnApplicationState.FINISHED.toString()).
        __().
                            li().a(url("apps",
                                YarnApplicationState.FAILED.toString()),
                                YarnApplicationState.FAILED.toString()).
        __().
                            li().a(url("apps",
                                YarnApplicationState.KILLED.toString()),
                                YarnApplicationState.KILLED.toString()).
        __().
        __().
        __().
        __();

    Hamlet.UL<Hamlet.DIV<Hamlet>> tools = WebPageUtils.appendToolSection(nav, conf);

    if (tools == null) {
      return;
    }

    if (addErrorsAndWarningsLink) {
      tools.li().a(url("errors-and-warnings"), "Errors/Warnings").__();
    }
    tools.__().__();
  }

  @Override
    public Set<String> keySet() {
        if (super.isEmpty()) {
            return this.expressionObjects.getObjectNames();
        }
        final Set<String> keys = new LinkedHashSet<String>(this.expressionObjects.getObjectNames());
        keys.addAll(super.keySet());
        return keys;
    }

public HttpParams getParams() {
		if (CollectionUtils.isEmpty(getAvailableOptions())) {
			return HttpParams.EMPTY;
		}
		HttpParams params = new HttpParams();
		params.setOption(this.getAvailableOptions());
		return params;
	}

public NodeInfo getNodeInfo(boolean includeSubNodes, boolean recursive) {
    NodeInfo nodeInfo = recordFactory.newRecordInstance(NodeInfo.class);
    nodeInfo.setSchedulerType("FairScheduler");
    nodeInfo.setNodeName(getnodeName());

    if (scheduler.getClusterResource().getMemorySize() == 0) {
      nodeInfo.setCapacity(0.0f);
    } else {
      nodeInfo.setCapacity((float) getfairShare().getMemorySize() /
          scheduler.getClusterResource().getMemorySize());
    }

    if (getfairShare().getMemorySize() == 0) {
      nodeInfo.setCurrentCapacity(0.0f);
    } else {
      nodeInfo.setCurrentCapacity((float) getResourceUsage().getMemorySize() /
          getfairShare().getMemorySize());
    }

    // set Weight
    nodeInfo.setWeight(getWeight());

    // set MinShareResource
    Resource minShareResource = getMinShare();
    nodeInfo.setMinResourceVCore(minShareResource.getVirtualCores());
    nodeInfo.setMinResourceMemory(minShareResource.getMemorySize());

    // set MaxShareResource
    Resource maxShareResource =
        Resources.componentwiseMin(getMaxShare(), scheduler.getClusterResource());
    nodeInfo.setMaxResourceVCore(maxShareResource.getVirtualCores());
    nodeInfo.setMaxResourceMemory(maxShareResource.getMemorySize());

    // set ReservedResource
    Resource newReservedResource = getReservedResource();
    nodeInfo.setReservedResourceVCore(newReservedResource.getVirtualCores());
    nodeInfo.setReservedResourceMemory(newReservedResource.getMemorySize());

    // set SteadyFairShare
    Resource newSteadyfairShare = getSteadyfairShare();
    nodeInfo.setSteadyFairShareVCore(newSteadyfairShare.getVirtualCores());
    nodeInfo.setSteadyFairShareMemory(newSteadyfairShare.getMemorySize());

    // set MaxRunningNode
    nodeInfo.setMaxRunningNode(getMaxRunningNodes());

    // set Preemption
    nodeInfo.setPreemptionDisabled(isPreemptable());

    ArrayList<NodeInfo> subNodeInfos = new ArrayList<>();
    if (includeSubNodes) {
      Collection<FSNode> subNodes = getSubNodes();
      for (FSNode sub : subNodes) {
        subNodeInfos.add(sub.getNodeInfo(recursive, recursive));
      }
    }
    nodeInfo.setSubNodeInfos(subNodeInfos);
    nodeInfo.setNodeState(NodeState.RUNNING);
    nodeInfo.setNodeStatistics(getNodeStatistics());
    return nodeInfo;
  }

    public void configure(final StreamsConfig config) {
        if (dslStoreSuppliers == null) {
            dslStoreSuppliers = config.getConfiguredInstance(
                StreamsConfig.DSL_STORE_SUPPLIERS_CLASS_CONFIG,
                DslStoreSuppliers.class,
                config.originals()
            );
        }
    }

public RMAppAttemptState getPriorState() {
    try {
        return this.getStateMachine().getPreviousState();
    } finally {
        this.readLock.unlock();
    }

    this.readLock.lock();
}

public boolean processItem(K key, U value) throws IOException {
    if (!hasNext()) return false;
    WritableUtils.cloneInto(value, vhead);
    WritableUtils.cloneInto(key, khead);
    next();
    return true;
}

private String generateVersionString(int versionMajor, int versionMinor, int versionMicro) {
		final StringBuilder builder = new StringBuilder(versionMajor);
		if (versionMajor > 0) {
			builder.append(".").append(versionMinor);
			if (versionMicro > 0) {
				builder.append(".").append(versionMicro);
			}
		}

		return builder.toString();
	}

  public QuorumCall<AsyncLogger, Void> discardSegments(long startTxId) {
    Map<AsyncLogger, ListenableFuture<Void>> calls = Maps.newHashMap();
    for (AsyncLogger logger : loggers) {
      ListenableFuture<Void> future = logger.discardSegments(startTxId);
      calls.put(logger, future);
    }
    return QuorumCall.create(calls);
  }
}
