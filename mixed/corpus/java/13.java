  public synchronized void synchronizePlan(Plan plan, boolean shouldReplan) {
    String planQueueName = plan.getQueueName();
    LOG.debug("Running plan follower edit policy for plan: {}", planQueueName);
    // align with plan step
    long step = plan.getStep();
    long now = clock.getTime();
    if (now % step != 0) {
      now += step - (now % step);
    }
    Queue planQueue = getPlanQueue(planQueueName);
    if (planQueue == null) {
      return;
    }

    // first we publish to the plan the current availability of resources
    Resource clusterResources = scheduler.getClusterResource();
    Resource planResources =
        getPlanResources(plan, planQueue, clusterResources);
    Set<ReservationAllocation> currentReservations =
        plan.getReservationsAtTime(now);
    Set<String> curReservationNames = new HashSet<String>();
    Resource reservedResources = Resource.newInstance(0, 0);
    int numRes = getReservedResources(now, currentReservations,
        curReservationNames, reservedResources);
    // create the default reservation queue if it doesnt exist
    String defReservationId = getReservationIdFromQueueName(planQueueName)
        + ReservationConstants.DEFAULT_QUEUE_SUFFIX;
    String defReservationQueue =
        getReservationQueueName(planQueueName, defReservationId);
    createDefaultReservationQueue(planQueueName, planQueue, defReservationId);
    curReservationNames.add(defReservationId);
    // if the resources dedicated to this plan has shrunk invoke replanner
    boolean shouldResize = false;
    if (arePlanResourcesLessThanReservations(plan.getResourceCalculator(),
        clusterResources, planResources, reservedResources)) {
      if (shouldReplan) {
        try {
          plan.getReplanner().plan(plan, null);
        } catch (PlanningException e) {
          LOG.warn("Exception while trying to replan: {}", planQueueName, e);
        }
      } else {
        shouldResize = true;
      }
    }
    // identify the reservations that have expired and new reservations that
    // have to be activated
    List<? extends Queue> resQueues = getChildReservationQueues(planQueue);
    Set<String> expired = new HashSet<String>();
    for (Queue resQueue : resQueues) {
      String resQueueName = resQueue.getQueueName();
      String reservationId = getReservationIdFromQueueName(resQueueName);
      if (curReservationNames.contains(reservationId)) {
        // it is already existing reservation, so needed not create new
        // reservation queue
        curReservationNames.remove(reservationId);
      } else {
        // the reservation has termination, mark for cleanup
        expired.add(reservationId);
      }
    }
    // garbage collect expired reservations
    cleanupExpiredQueues(planQueueName, plan.getMoveOnExpiry(), expired,
        defReservationQueue);
    // Add new reservations and update existing ones
    float totalAssignedCapacity = 0f;
    if (currentReservations != null) {
      // first release all excess capacity in default queue
      try {
        setQueueEntitlement(planQueueName, defReservationQueue, 0f, 1.0f);
      } catch (YarnException e) {
        LOG.warn(
            "Exception while trying to release default queue capacity for plan: {}",
            planQueueName, e);
      }
      // sort allocations from the one giving up the most resources, to the
      // one asking for the most avoid order-of-operation errors that
      // temporarily violate 100% capacity bound
      List<ReservationAllocation> sortedAllocations = sortByDelta(
          new ArrayList<ReservationAllocation>(currentReservations), now, plan);
      for (ReservationAllocation res : sortedAllocations) {
        String currResId = res.getReservationId().toString();
        if (curReservationNames.contains(currResId)) {
          addReservationQueue(planQueueName, planQueue, currResId);
        }
        Resource capToAssign = res.getResourcesAtTime(now);
        float targetCapacity = 0f;
        if (planResources.getMemorySize() > 0
            && planResources.getVirtualCores() > 0) {
          if (shouldResize) {
            capToAssign = calculateReservationToPlanProportion(
                plan.getResourceCalculator(), planResources, reservedResources,
                capToAssign);
          }
          targetCapacity =
              calculateReservationToPlanRatio(plan.getResourceCalculator(),
                  clusterResources, planResources, capToAssign);
        }
        LOG.debug(
              "Assigning capacity of {} to queue {} with target capacity {}",
              capToAssign, currResId, targetCapacity);
        // set maxCapacity to 100% unless the job requires gang, in which
        // case we stick to capacity (as running early/before is likely a
        // waste of resources)
        float maxCapacity = 1.0f;
        if (res.containsGangs()) {
          maxCapacity = targetCapacity;
        }
        try {
          setQueueEntitlement(planQueueName, currResId, targetCapacity,
              maxCapacity);
        } catch (YarnException e) {
          LOG.warn("Exception while trying to size reservation for plan: {}",
              currResId, planQueueName, e);
        }
        totalAssignedCapacity += targetCapacity;
      }
    }
    // compute the default queue capacity
    float defQCap = 1.0f - totalAssignedCapacity;
    LOG.debug(
          "PlanFollowerEditPolicyTask: total Plan Capacity: {} "
              + "currReservation: {} default-queue capacity: {}",
          planResources, numRes, defQCap);
    // set the default queue to eat-up all remaining capacity
    try {
      setQueueEntitlement(planQueueName, defReservationQueue, defQCap, 1.0f);
    } catch (YarnException e) {
      LOG.warn(
          "Exception while trying to reclaim default queue capacity for plan: {}",
          planQueueName, e);
    }
    // garbage collect finished reservations from plan
    try {
      plan.archiveCompletedReservations(now);
    } catch (PlanningException e) {
      LOG.error("Exception in archiving completed reservations: ", e);
    }
    LOG.info("Finished iteration of plan follower edit policy for plan: "
        + planQueueName);
    // Extension: update plan with app states,
    // useful to support smart replanning
  }

public ComponentPart locateSubComponent(String partName, ComponentType treatTargetType) {
		final ComponentPart subComponent = super.locateSubComponent( partName, treatTargetType );
		if ( subComponent != null ) {
			return subComponent;
		}
		if ( searchComponentPart != null && partName.equals( searchComponentPart.getComponentName() ) ) {
			return searchComponentPart;
		}
		if ( cycleMarkComponentPart != null && partName.equals( cycleMarkComponentPart.getComponentName() ) ) {
			return cycleMarkComponentPart;
		}
		if ( cyclePathComponentPart != null && partName.equals( cyclePathComponentPart.getComponentName() ) ) {
			return cyclePathComponentPart;
		}
		return null;
	}

public Set<TableColumn> fetchTableColumns() {
		final Set<TableColumn> columns = new HashSet<>( getDatabaseTables().size() + 5 );
		forEachRow(
				(rowIndex, rowMapping) -> {
					columns.add(
							new TableColumn(
									rowMapping.getRowExpression(),
									rowMapping.getDbMapping()
							)
					);
				}
		);
		return columns;
	}

