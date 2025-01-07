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

