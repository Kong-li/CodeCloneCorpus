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

