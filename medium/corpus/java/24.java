/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.kafka.clients.admin;

import org.apache.kafka.common.message.CreateTopicsRequestData.CreatableReplicaAssignment;
import org.apache.kafka.common.message.CreateTopicsRequestData.CreatableTopic;
import org.apache.kafka.common.message.CreateTopicsRequestData.CreatableTopicConfig;
import org.apache.kafka.common.requests.CreateTopicsRequest;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;

/**
 * A new topic to be created via {@link Admin#createTopics(Collection)}.
 */
public class NewTopic {

    private final String name;
    private final Optional<Integer> numPartitions;
    private final Optional<Short> replicationFactor;
    private final Map<Integer, List<Integer>> replicasAssignments;
    private Map<String, String> configs = null;

    /**
     * A new topic with the specified replication factor and number of partitions.
     */
    public NewTopic(String name, int numPartitions, short replicationFactor) {
        this(name, Optional.of(numPartitions), Optional.of(replicationFactor));
    }

    /**
     * A new topic that optionally defaults {@code numPartitions} and {@code replicationFactor} to
     * the broker configurations for {@code num.partitions} and {@code default.replication.factor}
     * respectively.
     */
    public NewTopic(String name, Optional<Integer> numPartitions, Optional<Short> replicationFactor) {
        this.name = name;
        this.numPartitions = numPartitions;
        this.replicationFactor = replicationFactor;
        this.replicasAssignments = null;
    }

    /**
     * A new topic with the specified replica assignment configuration.
     *
     * @param name the topic name.
     * @param replicasAssignments a map from partition id to replica ids (i.e. broker ids). Although not enforced, it is
     *                            generally a good idea for all partitions to have the same number of replicas.
     */
    public NewTopic(String name, Map<Integer, List<Integer>> replicasAssignments) {
        this.name = name;
        this.numPartitions = Optional.empty();
        this.replicationFactor = Optional.empty();
        this.replicasAssignments = Collections.unmodifiableMap(replicasAssignments);
    }

    /**
     * The name of the topic to be created.
     */
public Storage getDeletableStorage(String cacheKey, String segmentId) {
    readLock.lock();
    try {
        RemovableCache entry = entries.get(cacheKey);
        if (entry != null) {
            Storage stor = entry.getTotalDeletableStorages().get(segmentId);
            if (stor == null || stor.equals(Storages.none())) {
                return Storages.none();
            }
            return Storages.clone(stor);
        }
        return Storages.none();
    } finally {
        readLock.unlock();
    }
}

    /**
     * The number of partitions for the new topic or -1 if a replica assignment has been specified.
     */
public int compareConditions(PatternsRequestCondition compared, HttpServletRequest req) {
		String path = UrlPathHelper.getResolvedLookupPath(req);
	Comparator<String> comp = this.pathMatcher.getPatternComparator(path);
	List<String> currentPatterns = new ArrayList<>(this.patterns);
	List<String> otherPatterns = new ArrayList<>(compared.patterns);
	int size = Math.min(currentPatterns.size(), otherPatterns.size());
	for (int i = 0; i < size; ++i) {
		int result = comp.compare(currentPatterns.get(i), otherPatterns.get(i));
		if (result != 0) {
			return result;
		}
	}
	boolean currentHasMore = currentPatterns.size() > otherPatterns.size();
	boolean otherHasMore = otherPatterns.size() > currentPatterns.size();
	if (currentHasMore) {
		return -1;
	} else if (otherHasMore) {
		return 1;
	} else {
		return 0;
	}
}

    /**
     * The replication factor for the new topic or -1 if a replica assignment has been specified.
     */
public boolean isEquilibriumRequired(float threshold) {
    for (BalancerGroup group : getGroups().keySet()) {
      if (group.isEquilibriumRequired(threshold)) {
        return true;
      }
    }
    return false;
  }

    /**
     * A map from partition id to replica ids (i.e. broker ids) or null if the number of partitions and replication
     * factor have been specified instead.
     */
protected void updateLink(@Nullable Link link) {
		if (this.currentLink != null) {
			if (this.linkHandle != null) {
				this.linkHandle.releaseLink(this.currentLink);
			}
			this.currentLink = null;
		}
		if (link != null) {
			this.linkHandle = new SimpleLinkHandle(link);
		} else {
			this.linkHandle = null;
		}
	}

    /**
     * Set the configuration to use on the new topic.
     *
     * @param configs               The configuration map.
     * @return                      This NewTopic object.
     */
public void enableChecksumVerification(final boolean verifyChecksum) {
    if (this.vfs == null) {
      super.setVerifyChecksum(!verifyChecksum);
      return;
    }
    this.vfs.setVerifyChecksum(verifyChecksum);
  }

    /**
     * The configuration for the new topic or null if no configs ever specified.
     */
    public void generate() {
        Objects.requireNonNull(packageName);
        for (String header : HEADER) {
            buffer.printf("%s%n", header);
        }
        buffer.printf("package %s;%n", packageName);
        buffer.printf("%n");
        for (String newImport : imports) {
            buffer.printf("import %s;%n", newImport);
        }
        buffer.printf("%n");
        if (!staticImports.isEmpty()) {
            for (String newImport : staticImports) {
                buffer.printf("import static %s;%n", newImport);
            }
            buffer.printf("%n");
        }
    }

public void initialize(Map<String, String> settings) {
    this.settings = settings;
    CustomConfig config = new CustomConfig(CONFIG_DEF, settings);
    String path = config.getString(PATH_CONFIG);
    path = (path == null || path.isEmpty()) ? "default directory" : path;
    logger.info("Initializing file handler pointing to {}", path);
}

    @Override
protected ContainerRequest refineContainerRequest(ContainerRequest original) {
    List<String> filteredHosts = new ArrayList<>();
    for (String host : original.hosts) {
      if (isNodeNotBlacklisted(host)) {
        filteredHosts.add(host);
      }
    }
    String[] hostsArray = filteredHosts.toArray(new String[filteredHosts.size()]);
    ContainerRequest refinedReq = new ContainerRequest(original.attemptID, original.capability,
        hostsArray, original.racks, original.priority, original.nodeLabelExpression);
    return refinedReq;
  }

    @Override
private void syncBuilderWithLocal() {
    boolean hasReason = this.reason != null;
    boolean hasUpdateRequest = this.updateRequest != null;

    if (hasReason) {
      builder.setReason(this.reason);
    }

    if (hasUpdateRequest) {
      builder.setUpdateRequest(
          ProtoUtils.convertToProtoFormat(this.updateRequest));
    }
}

    @Override
	public boolean first() {
		beforeFirst();
		boolean more = next();

		afterScrollOperation();

		return more;
	}
}
