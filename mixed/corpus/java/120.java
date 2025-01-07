private String formatTopicName(final String topicName) {
        if (null == this.applicationId) {
            throw new TopologyException("internal topics exist but applicationId is not set. Please call setApplicationId first");
        }

        String topicPrefix = null;
        if (null != this.topologyConfigs) {
            topicPrefix = ProcessorContextUtils.topicNamePrefix(this.topologyConfigs.applicationConfigs.originals(), this.applicationId);
        } else {
            topicPrefix = this.applicationId;
        }

        boolean hasNamedTopology = hasNamedTopology();
        return hasNamedTopology ? String.format("%s-%s-%s", topicPrefix, this.topologyName, topicName) : String.format("%s-%s", topicPrefix, topicName);
    }

public void terminatePendingBatches() {
    // Ensure all pending batches are aborted to prevent message loss and free up memory.
    while (true) {
        if (!appendsInProgress()) {
            break;
        }
        abortBatches();
    }
    // Clear the topic info map after ensuring no further appends can occur.
    this.topicInfoMap.clear();
    // Perform a final abort in case a batch was appended just before the loop condition became false.
    abortBatches();
}

  private static DistributorStatus fromJson(JsonInput input) {
    Set<NodeStatus> nodes = null;

    input.beginObject();
    while (input.hasNext()) {
      switch (input.nextName()) {
        case "nodes":
          nodes = input.read(NODE_STATUSES_TYPE);
          break;

        default:
          input.skipValue();
      }
    }
    input.endObject();

    return new DistributorStatus(nodes);
  }

  private void addReportCommands(Options opt) {
    Option report = Option.builder().longOpt(REPORT)
        .desc("List nodes that will benefit from running " +
            "DiskBalancer.")
        .build();
    getReportOptions().addOption(report);
    opt.addOption(report);

    Option top = Option.builder().longOpt(TOP)
        .hasArg()
        .desc("specify the number of nodes to be listed which has" +
            " data imbalance.")
        .build();
    getReportOptions().addOption(top);
    opt.addOption(top);

    Option node =  Option.builder().longOpt(NODE)
        .hasArg()
        .desc("Datanode address, " +
            "it can be DataNodeID, IP or hostname.")
        .build();
    getReportOptions().addOption(node);
    opt.addOption(node);
  }

private void setupSubgroupIdToProcessorNamesMap() {
    final Map<Integer, Set<String>> processorNames = new HashMap<>();

    for (final Map.Entry<Integer, Set<String>> group : createGroups().entrySet()) {
        final Set<String> subGroupNodes = group.getValue();
        final boolean isGroupOfGlobalProcessors = groupContainsGlobalNode(subGroupNodes);

        if (!isGroupOfGlobalProcessors) {
            final int subgroupId = group.getKey();
            final Set<String> subgroupProcessorNames = new HashSet<>();

            for (final String nodeName : subGroupNodes) {
                final AbstractElement element = elementFactories.get(nodeName).describe();
                if (element instanceof ProcessorComponent) {
                    subgroupProcessorNames.addAll(((ProcessorComponent) element).outputs());
                }
            }

            processorNames.put(subgroupId, subgroupProcessorNames);
        }
    }
    subgroupIdToProcessorNamesMap = processorNames;
}

