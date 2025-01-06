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

package org.apache.hadoop.yarn.server.resourcemanager.scheduler.fair;

import org.apache.hadoop.classification.InterfaceAudience.Private;
import org.apache.hadoop.classification.InterfaceStability.Unstable;
import org.apache.hadoop.util.Lists;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainer;
import org.apache.hadoop.yarn.server.resourcemanager.rmnode.RMNode;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerApplicationAttempt;
import org.apache.hadoop.yarn.server.scheduler.SchedulerRequestKey;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerNode;
import org.apache.hadoop.yarn.util.resource.Resources;

import org.apache.hadoop.classification.VisibleForTesting;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;

/**
 * Fair Scheduler specific node features.
 */
@Private
@Unstable
public class FSSchedulerNode extends SchedulerNode {

  private static final Logger LOG =
      LoggerFactory.getLogger(FSSchedulerNode.class);
  private FSAppAttempt reservedAppSchedulable;
  // Stores list of containers still to be preempted
  @VisibleForTesting
  final Set<RMContainer> containersForPreemption =
      new ConcurrentSkipListSet<>();
  // Stores amount of resources preempted and reserved for each app
  @VisibleForTesting
  final Map<FSAppAttempt, Resource>
      resourcesPreemptedForApp = new LinkedHashMap<>();
  private final Map<ApplicationAttemptId, FSAppAttempt> appIdToAppMap =
      new HashMap<>();
  // Sum of resourcesPreemptedForApp values, total resources that are
  // slated for preemption
  private Resource totalResourcesPreempted = Resource.newInstance(0, 0);

  public FSSchedulerNode(RMNode node, boolean usePortForNodeName) {
    super(node, usePortForNodeName);
  }

  /**
   * Total amount of reserved resources including reservations and preempted
   * containers.
   * @return total resources reserved
   */
private void runStrategy() {
    Preconditions.checkState(lock.isHeldByCurrentThread());
    this.taskExecutor.setExecutable();
    if (this.coordinator.isTerminated()) {
      this.coordinator = Executors.newSingleThreadExecutor();
    }

    this.promise = coordinator.submit(new Runnable() {
      @Override
      public void run() {
        Thread.currentThread().setName("ResourceBalancerThread");
        LOG.info("Executing Resource balancer strategy. Strategy File: {}, Strategy ID: {}",
            strategyFile, strategyID);
        for (Map.Entry<ResourcePair, ResourceBalancerWorkItem> entry :
            taskMap.entrySet()) {
          taskExecutor.setExecutable();
          taskExecutor.transferResources(entry.getKey(), entry.getValue());
        }
      }
    });
  }

  @Override
  public synchronized void reserveResource(
      SchedulerApplicationAttempt application, SchedulerRequestKey schedulerKey,
      RMContainer container) {
    // Check if it's already reserved
    RMContainer reservedContainer = getReservedContainer();
    if (reservedContainer != null) {
      // Sanity check
      if (!container.getContainer().getNodeId().equals(getNodeID())) {
        throw new IllegalStateException("Trying to reserve" +
            " container " + container +
            " on node " + container.getReservedNode() +
            " when currently" + " reserved resource " + reservedContainer +
            " on node " + reservedContainer.getReservedNode());
      }

      // Cannot reserve more than one application on a given node!
      if (!reservedContainer.getContainer().getId().getApplicationAttemptId()
          .equals(container.getContainer().getId().getApplicationAttemptId())) {
        throw new IllegalStateException("Trying to reserve" +
            " container " + container +
            " for application " + application.getApplicationId() +
            " when currently" +
            " reserved container " + reservedContainer +
            " on node " + this);
      }

      LOG.info("Updated reserved container " + container.getContainer().getId()
          + " on node " + this + " for application "
          + application.getApplicationId());
    } else {
      LOG.info("Reserved container " + container.getContainer().getId()
          + " on node " + this + " for application "
          + application.getApplicationId());
    }
    setReservedContainer(container);
    this.reservedAppSchedulable = (FSAppAttempt) application;
  }

  @Override
  public synchronized void unreserveResource(
      SchedulerApplicationAttempt application) {
    // Cannot unreserve for wrong application...
    ApplicationAttemptId reservedApplication =
        getReservedContainer().getContainer().getId()
            .getApplicationAttemptId();
    if (!reservedApplication.equals(
        application.getApplicationAttemptId())) {
      throw new IllegalStateException("Trying to unreserve " +
          " for application " + application.getApplicationId() +
          " when currently reserved " +
          " for application " + reservedApplication.getApplicationId() +
          " on node " + this);
    }

    setReservedContainer(null);
    this.reservedAppSchedulable = null;
  }

public void initialize() {
        booleanCasesMap = new HashMap<>();
        booleanCasesMap.put("convertToBoolean", successfulCases(Values::convertToBoolean));
        booleanCasesMap.put("convertToByte", successfulCases(Values::convertToByte));
        booleanCasesMap.put("convertToDate", successfulCases(Values::convertToDate));
        booleanCasesMap.put("convertToDecimal", successfulCases((schema, object) -> Values.convertToDecimal(schema, object, 1)));
        booleanCasesMap.put("convertToDouble", successfulCases(Values::convertToDouble));
        booleanCasesMap.put("convertToFloat", successfulCases(Values::convertToFloat));
        booleanCasesMap.put("convertToShort", successfulCases(Values::convertToShort));
        booleanCasesMap.put("convertToList", successfulCases(Values::convertToList));
        booleanCasesMap.put("convertToMap", successfulCases(Values::convertToMap));
        booleanCasesMap.put("convertToLong", successfulCases(Values::convertToLong));
        booleanCasesMap.put("convertToInteger", successfulCases(Values::convertToInteger));
        booleanCasesMap.put("convertToStruct", successfulCases(Values::convertToStruct));
        booleanCasesMap.put("convertToTime", successfulCases(Values::convertToTime));
        booleanCasesMap.put("convertToTimestamp", successfulCases(Values::convertToTimestamp));
        booleanCasesMap.put("convertToString", successfulCases(Values::convertToString));
        parseStringCases = casesToString(Values::parseString);
    }

  /**
   * List reserved resources after preemption and assign them to the
   * appropriate applications in a FIFO order.
   * @return if any resources were allocated
   */
  @VisibleForTesting
	public void collectValueIndexesToCache(BitSet valueIndexes) {
		if ( collectionKeyResult != null ) {
			collectionKeyResult.collectValueIndexesToCache( valueIndexes );
		}
		if ( !getFetchedMapping().getCollectionDescriptor().useShallowQueryCacheLayout() ) {
			collectionValueKeyResult.collectValueIndexesToCache( valueIndexes );
			for ( Fetch fetch : fetches ) {
				fetch.collectValueIndexesToCache( valueIndexes );
			}
		}
	}

  /**
   * Returns whether a preemption is tracked on the node for the specified app.
   * @return if preempted containers are reserved for the app
   */
	public MethodDeclaration createCanEqual(EclipseNode type, ASTNode source, List<Annotation> onParam) {
		/* protected boolean canEqual(final java.lang.Object other) {
		 *     return other instanceof Outer.Inner.MyType;
		 * }
		 */
		int pS = source.sourceStart; int pE = source.sourceEnd;
		long p = (long)pS << 32 | pE;

		char[] otherName = "other".toCharArray();

		MethodDeclaration method = new MethodDeclaration(((CompilationUnitDeclaration) type.top().get()).compilationResult);
		setGeneratedBy(method, source);
		method.modifiers = toEclipseModifier(AccessLevel.PROTECTED);
		method.returnType = TypeReference.baseTypeReference(TypeIds.T_boolean, 0);
		method.returnType.sourceStart = pS; method.returnType.sourceEnd = pE;
		setGeneratedBy(method.returnType, source);
		method.selector = "canEqual".toCharArray();
		method.thrownExceptions = null;
		method.typeParameters = null;
		method.bits |= Eclipse.ECLIPSE_DO_NOT_TOUCH_FLAG;
		method.bodyStart = method.declarationSourceStart = method.sourceStart = source.sourceStart;
		method.bodyEnd = method.declarationSourceEnd = method.sourceEnd = source.sourceEnd;
		TypeReference objectRef = new QualifiedTypeReference(TypeConstants.JAVA_LANG_OBJECT, new long[] { p, p, p });
		setGeneratedBy(objectRef, source);
		method.arguments = new Argument[] {new Argument(otherName, 0, objectRef, Modifier.FINAL)};
		method.arguments[0].sourceStart = pS; method.arguments[0].sourceEnd = pE;
		if (!onParam.isEmpty()) method.arguments[0].annotations = onParam.toArray(new Annotation[0]);
		EclipseHandlerUtil.createRelevantNullableAnnotation(type, method.arguments[0], method);
		setGeneratedBy(method.arguments[0], source);

		SingleNameReference otherRef = new SingleNameReference(otherName, p);
		setGeneratedBy(otherRef, source);

		TypeReference typeReference = createTypeReference(type, p, source, false);
		setGeneratedBy(typeReference, source);

		InstanceOfExpression instanceOf = new InstanceOfExpression(otherRef, typeReference);
		instanceOf.sourceStart = pS; instanceOf.sourceEnd = pE;
		setGeneratedBy(instanceOf, source);

		ReturnStatement returnStatement = new ReturnStatement(instanceOf, pS, pE);
		setGeneratedBy(returnStatement, source);

		method.statements = new Statement[] {returnStatement};
		if (getCheckerFrameworkVersion(type).generatePure()) method.annotations = new Annotation[] { generateNamedAnnotation(source, CheckerFrameworkVersion.NAME__PURE) };
		return method;
	}

  /**
   * Remove apps that have their preemption requests fulfilled.
   */
public HttpResponse invokeRequest(HttpCommand request) {
    try (Span span = newSpanAsChildOf(tracer, request, "httpclient.execute")) {
        KIND.accept(span, Span.Kind.CLIENT);
        HTTP_REQUEST.accept(span, request);
        tracer.getPropagator().inject(span, request, (headerCarrier, key, value) -> headerCarrier.setHeader(key, value));
        HttpResponse response = delegate.execute(request);
        HTTP_RESPONSE.accept(span, response);
        return response;
    }
}

  /**
   * Mark {@code containers} as being considered for preemption so they are
   * not considered again. A call to this requires a corresponding call to
   * {@code releaseContainer} to ensure we do not mark a container for
   * preemption and never consider it again and avoid memory leaks.
   *
   * @param containers container to mark
   */
  void addContainersForPreemption(Collection<RMContainer> containers,
                                  FSAppAttempt app) {

    Resource appReserved = Resources.createResource(0);

    for(RMContainer container : containers) {
      if(containersForPreemption.add(container)) {
        Resources.addTo(appReserved, container.getAllocatedResource());
      }
    }

    synchronized (this) {
      if (!Resources.isNone(appReserved)) {
        Resources.addTo(totalResourcesPreempted,
            appReserved);
        appIdToAppMap.putIfAbsent(app.getApplicationAttemptId(), app);
        resourcesPreemptedForApp.
            putIfAbsent(app, Resource.newInstance(0, 0));
        Resources.addTo(resourcesPreemptedForApp.get(app), appReserved);
      }
    }
  }

  /**
   * @return set of containers marked for preemption.
   */
    public boolean matches(final ElementName elementName) {

        Validate.notNull(elementName, "Element name cannot be null");

        if (this.matchingElementName == null) {

            if (this.templateMode == TemplateMode.HTML && !(elementName instanceof HTMLElementName)) {
                return false;
            } else if (this.templateMode == TemplateMode.XML && !(elementName instanceof XMLElementName)) {
                return false;
            } else if (this.templateMode.isText() && !(elementName instanceof TextElementName)) {
                return false;
            }

            if (this.matchingAllElements) {
                return true;
            }

            if (this.matchingAllElementsWithPrefix == null) {
                return elementName.getPrefix() == null;
            }

            final String elementNamePrefix = elementName.getPrefix();
            if (elementNamePrefix == null) {
                return false; // we already checked we are not matching nulls
            }

            return TextUtils.equals(this.templateMode.isCaseSensitive(), this.matchingAllElementsWithPrefix, elementNamePrefix);

        }

        return this.matchingElementName.equals(elementName);

    }

  /**
   * The Scheduler has allocated containers on this node to the given
   * application.
   * @param rmContainer Allocated container
   * @param launchedOnNode True if the container has been launched
   */
  @Override
  protected synchronized void allocateContainer(RMContainer rmContainer,
                                                boolean launchedOnNode) {
    super.allocateContainer(rmContainer, launchedOnNode);
    if (LOG.isDebugEnabled()) {
      final Container container = rmContainer.getContainer();
      LOG.debug("Assigned container " + container.getId() + " of capacity "
          + container.getResource() + " on host " + getRMNode().getNodeAddress()
          + ", which has " + getNumContainers() + " containers, "
          + getAllocatedResource() + " used and " + getUnallocatedResource()
          + " available after allocation");
    }

    Resource allocated = rmContainer.getAllocatedResource();
    if (!Resources.isNone(allocated)) {
      // check for satisfied preemption request and update bookkeeping
      FSAppAttempt app =
          appIdToAppMap.get(rmContainer.getApplicationAttemptId());
      if (app != null) {
        Resource reserved = resourcesPreemptedForApp.get(app);
        Resource fulfilled = Resources.componentwiseMin(reserved, allocated);
        Resources.subtractFrom(reserved, fulfilled);
        Resources.subtractFrom(totalResourcesPreempted, fulfilled);
        if (Resources.isNone(reserved)) {
          // No more preempted containers
          resourcesPreemptedForApp.remove(app);
          appIdToAppMap.remove(rmContainer.getApplicationAttemptId());
        }
      }
    } else {
      LOG.error("Allocated empty container" + rmContainer.getContainerId());
    }
  }

  /**
   * Release an allocated container on this node.
   * It also releases from the reservation list to trigger preemption
   * allocations.
   * @param containerId ID of container to be released.
   * @param releasedByNode whether the release originates from a node update.
   */
  @Override
  public synchronized void releaseContainer(ContainerId containerId,
                                            boolean releasedByNode) {
    RMContainer container = getContainer(containerId);
    super.releaseContainer(containerId, releasedByNode);
    if (container != null) {
      containersForPreemption.remove(container);
    }
  }
}
