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
package org.apache.kafka.streams.processor.internals;

import org.apache.kafka.clients.consumer.internals.AutoOffsetResetStrategy;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.config.ConfigException;
import org.apache.kafka.common.serialization.Deserializer;
import org.apache.kafka.common.serialization.Serializer;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.TopologyConfig;
import org.apache.kafka.streams.errors.TopologyException;
import org.apache.kafka.streams.internals.ApiUtils;
import org.apache.kafka.streams.internals.AutoOffsetResetInternal;
import org.apache.kafka.streams.processor.StateStore;
import org.apache.kafka.streams.processor.StreamPartitioner;
import org.apache.kafka.streams.processor.TimestampExtractor;
import org.apache.kafka.streams.processor.TopicNameExtractor;
import org.apache.kafka.streams.processor.api.FixedKeyProcessorSupplier;
import org.apache.kafka.streams.processor.api.ProcessorSupplier;
import org.apache.kafka.streams.processor.api.ProcessorWrapper;
import org.apache.kafka.streams.processor.api.WrappedFixedKeyProcessorSupplier;
import org.apache.kafka.streams.processor.api.WrappedProcessorSupplier;
import org.apache.kafka.streams.processor.internals.TopologyMetadata.Subtopology;
import org.apache.kafka.streams.processor.internals.namedtopology.NamedTopology;
import org.apache.kafka.streams.state.StoreBuilder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.apache.kafka.streams.StreamsConfig.PROCESSOR_WRAPPER_CLASS_CONFIG;

public class InternalTopologyBuilder {

    public InternalTopologyBuilder() {
        this.topologyName = null;
        this.processorWrapper = new NoOpProcessorWrapper();
    }

    public InternalTopologyBuilder(final TopologyConfig topologyConfigs) {
        this.topologyConfigs = topologyConfigs;
        this.topologyName = topologyConfigs.topologyName;

        try {
            processorWrapper = topologyConfigs.getConfiguredInstance(
                PROCESSOR_WRAPPER_CLASS_CONFIG,
                ProcessorWrapper.class,
                topologyConfigs.originals()
            );
        } catch (final Exception e) {
            final String errorMessage = String.format(
                "Unable to instantiate ProcessorWrapper from value of config %s. Please provide a valid class "
                    + "that implements the ProcessorWrapper interface.", PROCESSOR_WRAPPER_CLASS_CONFIG);
            log.error(errorMessage, e);
            throw new ConfigException(errorMessage, e);
        }
    }

    private static final Logger log = LoggerFactory.getLogger(InternalTopologyBuilder.class);
    private static final String[] NO_PREDECESSORS = {};

    // node factories in a topological order
    private final Map<String, NodeFactory<?, ?, ?, ?>> nodeFactories = new LinkedHashMap<>();

    private final Map<String, StoreFactory> stateFactories = new HashMap<>();

    private final Map<String, StoreFactory> globalStateBuilders = new LinkedHashMap<>();

    // built global state stores
    private final Map<String, StateStore> globalStateStores = new LinkedHashMap<>();

    // Raw names of all source topics, without the application id/named topology prefix for repartition sources
    private final Set<String> rawSourceTopicNames = new HashSet<>();

    // Full names of all source topics, including the application id/named topology prefix for repartition sources
    private List<String> fullSourceTopicNames = null;

    // String representing pattern that matches all subscribed topics, including patterns and full source topic names
    private String sourceTopicPatternString = null;

    // all internal topics with their corresponding properties auto-created by the topology builder and used in source / sink processors
    private final Map<String, InternalTopicProperties> internalTopicNamesWithProperties = new HashMap<>();

    // groups of source processors that need to be copartitioned
    private final List<Set<String>> copartitionSourceGroups = new ArrayList<>();

    // map from source processor names to subscribed topics (without application-id prefix for internal topics)
    private final Map<String, List<String>> nodeToSourceTopics = new HashMap<>();

    // map from source processor names to regex subscription patterns
    private final Map<String, Pattern> nodeToSourcePatterns = new LinkedHashMap<>();

    // map from sink processor names to sink topic (without application-id prefix for internal topics)
    private final Map<String, String> nodeToSinkTopic = new HashMap<>();

    // map from state store names to raw name (without application id/topology name prefix) of all topics subscribed
    // from source processors that are connected to these state stores
    private final Map<String, Set<String>> stateStoreNameToRawSourceTopicNames = new HashMap<>();

    // map from state store names to all the regex subscribed topics from source processors that
    // are connected to these state stores
    private final Map<String, Set<Pattern>> stateStoreNameToSourceRegex = new HashMap<>();

    // map from state store names to this state store's corresponding changelog topic if possible
    private final Map<String, String> storeToChangelogTopic = new HashMap<>();

    // map from changelog topic name to its corresponding state store.
    private final Map<String, String> changelogTopicToStore = new HashMap<>();

    // map of store name to restore behavior
    private final Map<String, Optional<ReprocessFactory<?, ?, ?, ?>>> storeNameToReprocessOnRestore = new HashMap<>();

    // all global topics
    private final Set<String> globalTopics = new HashSet<>();

    private final Set<String> noneResetTopics = new HashSet<>();

    private final Set<String> earliestResetTopics = new HashSet<>();

    private final Set<String> latestResetTopics = new HashSet<>();

    private final Map<String, Duration> durationResetTopics = new HashMap<>();

    private final Set<Pattern> noneResetPatterns = new HashSet<>();

    private final Set<Pattern> earliestResetPatterns = new HashSet<>();

    private final Set<Pattern> latestResetPatterns = new HashSet<>();

    private final Map<Pattern, Duration> durationResetPatterns = new HashMap<>();

    private final QuickUnion<String> nodeGrouper = new QuickUnion<>();

    // Used to capture subscribed topics via Patterns discovered during the partition assignment process.
    private final Set<String> subscriptionUpdates = new HashSet<>();

    private final ProcessorWrapper processorWrapper;

    private String applicationId = null;

    // keyed by subtopology id
    private Map<Integer, Set<String>> nodeGroups = null;

    // keyed by subtopology id
    private Map<Integer, Set<String>> subtopologyIdToStateStoreNames = null;

    // The name of the topology this builder belongs to, or null if this is not a NamedTopology
    private final String topologyName;

    @SuppressWarnings("deprecation")
    private NamedTopology namedTopology;

    // TODO KAFKA-13283: once we enforce all configs be passed in when constructing the topology builder then we can set
    //  this up front and make it final, but for now we have to wait for the global app configs passed in to rewriteTopology
    private TopologyConfig topologyConfigs;  // the configs for this topology, including overrides and global defaults

    private boolean hasPersistentStores = false;

    public static class ReprocessFactory<KIn, VIn, KOut, VOut> {

        private final ProcessorSupplier<KIn, VIn, KOut, VOut> processorSupplier;
        private final Deserializer<KIn> keyDeserializer;
        private final Deserializer<VIn> valueDeserializer;

        private ReprocessFactory(final ProcessorSupplier<KIn, VIn, KOut, VOut> processorSupplier,
                                 final Deserializer<KIn> key,
                                 final Deserializer<VIn> value) {
            this.processorSupplier = processorSupplier;
            this.keyDeserializer = key;
            this.valueDeserializer = value;
        }
        public ProcessorSupplier<KIn, VIn, KOut, VOut> processorSupplier() {
            return processorSupplier;
        }

        public Deserializer<KIn> keyDeserializer() {
            return keyDeserializer;
        }

        public Deserializer<VIn> valueDeserializer() {
            return valueDeserializer;
        }
    }

    private abstract static class NodeFactory<KIn, VIn, KOut, VOut> {
        final String name;
        final String[] predecessors;

        NodeFactory(final String name,
                    final String[] predecessors) {
            this.name = name;
            this.predecessors = predecessors;
        }

        public abstract ProcessorNode<KIn, VIn, KOut, VOut> build();

        abstract AbstractNode describe();
    }

    private static class ProcessorNodeFactory<KIn, VIn, KOut, VOut> extends NodeFactory<KIn, VIn, KOut, VOut> {
        private final ProcessorSupplier<KIn, VIn, KOut, VOut> supplier;
        final Set<String> stateStoreNames = new HashSet<>();

        ProcessorNodeFactory(final String name,
                             final String[] predecessors,
                             final ProcessorSupplier<KIn, VIn, KOut, VOut> supplier) {
            super(name, predecessors.clone());
            this.supplier = supplier;
        }

        public void addStateStore(final String stateStoreName) {
            stateStoreNames.add(stateStoreName);
        }

        @Override
        public ProcessorNode<KIn, VIn, KOut, VOut> build() {
            return new ProcessorNode<>(name, supplier.get(), stateStoreNames);
        }

        @Override
        Processor describe() {
            return new Processor(name, new HashSet<>(stateStoreNames));
        }
    }

    private static class FixedKeyProcessorNodeFactory<KIn, VIn, VOut> extends ProcessorNodeFactory<KIn, VIn, KIn, VOut> {
        private final FixedKeyProcessorSupplier<KIn, VIn, VOut> supplier;

        FixedKeyProcessorNodeFactory(final String name,
                             final String[] predecessors,
                             final FixedKeyProcessorSupplier<KIn, VIn, VOut> supplier) {
            super(name, predecessors.clone(), null);
            this.supplier = supplier;
        }

        @Override
        public ProcessorNode<KIn, VIn, KIn, VOut> build() {
            return new ProcessorNode<>(name, supplier.get(), stateStoreNames);
        }

        @Override
        Processor describe() {
            return new Processor(name, new HashSet<>(stateStoreNames));
        }
    }

    // Map from topics to their matched regex patterns, this is to ensure one topic is passed through on source node
    // even if it can be matched by multiple regex patterns. Only used by SourceNodeFactory
    private final Map<String, Pattern> topicToPatterns = new HashMap<>();

    private class SourceNodeFactory<KIn, VIn> extends NodeFactory<KIn, VIn, KIn, VIn> {
        private final List<String> topics;
        private final Pattern pattern;
        private final Deserializer<KIn> keyDeserializer;
        private final Deserializer<VIn> valDeserializer;
        private final TimestampExtractor timestampExtractor;

        private SourceNodeFactory(final String name,
                                  final String[] topics,
                                  final Pattern pattern,
                                  final TimestampExtractor timestampExtractor,
                                  final Deserializer<KIn> keyDeserializer,
                                  final Deserializer<VIn> valDeserializer) {
            super(name, NO_PREDECESSORS);
            this.topics = topics != null ? Arrays.asList(topics) : new ArrayList<>();
            this.pattern = pattern;
            this.keyDeserializer = keyDeserializer;
            this.valDeserializer = valDeserializer;
            this.timestampExtractor = timestampExtractor;
        }

        List<String> topics(final Collection<String> subscribedTopics) {
            // if it is subscribed via patterns, it is possible that the topic metadata has not been updated
            // yet and hence the map from source node to topics is stale, in this case we put the pattern as a place holder;
            // this should only happen for debugging since during runtime this function should always be called after the metadata has updated.
            if (subscribedTopics.isEmpty()) {
                return Collections.singletonList(String.valueOf(pattern));
            }

            final List<String> matchedTopics = new ArrayList<>();
            for (final String update : subscribedTopics) {
                if (pattern == topicToPatterns.get(update)) {
                    matchedTopics.add(update);
                } else if (topicToPatterns.containsKey(update) && isMatch(update)) {
                    // the same topic cannot be matched to more than one pattern
                    // TODO: we should lift this requirement in the future
                    throw new TopologyException("Topic " + update +
                        " is already matched for another regex pattern " + topicToPatterns.get(update) +
                        " and hence cannot be matched to this regex pattern " + pattern + " any more.");
                } else if (isMatch(update)) {
                    topicToPatterns.put(update, pattern);
                    matchedTopics.add(update);
                }
            }
            return matchedTopics;
        }

        @Override
        public ProcessorNode<KIn, VIn, KIn, VIn> build() {
            return new SourceNode<>(name, timestampExtractor, keyDeserializer, valDeserializer);
        }

        private boolean isMatch(final String topic) {
            return pattern.matcher(topic).matches();
        }

        @Override
        Source describe() {
            return new Source(name, topics.isEmpty() ? null : new HashSet<>(topics), pattern);
        }
    }

    private class SinkNodeFactory<KIn, VIn> extends NodeFactory<KIn, VIn, Void, Void> {
        private final Serializer<KIn> keySerializer;
        private final Serializer<VIn> valSerializer;
        private final StreamPartitioner<? super KIn, ? super VIn> partitioner;
        private final TopicNameExtractor<KIn, VIn> topicExtractor;

        private SinkNodeFactory(final String name,
                                final String[] predecessors,
                                final TopicNameExtractor<KIn, VIn> topicExtractor,
                                final Serializer<KIn> keySerializer,
                                final Serializer<VIn> valSerializer,
                                final StreamPartitioner<? super KIn, ? super VIn> partitioner) {
            super(name, predecessors.clone());
            this.topicExtractor = topicExtractor;
            this.keySerializer = keySerializer;
            this.valSerializer = valSerializer;
            this.partitioner = partitioner;
        }

        @Override
        public ProcessorNode<KIn, VIn, Void, Void> build() {
            if (topicExtractor instanceof StaticTopicNameExtractor) {
                final String topic = ((StaticTopicNameExtractor<KIn, VIn>) topicExtractor).topicName;
                if (internalTopicNamesWithProperties.containsKey(topic)) {
                    // prefix the internal topic name with the application id
                    return new SinkNode<>(name, new StaticTopicNameExtractor<>(decorateTopic(topic)), keySerializer, valSerializer, partitioner);
                } else {
                    return new SinkNode<>(name, topicExtractor, keySerializer, valSerializer, partitioner);
                }
            } else {
                return new SinkNode<>(name, topicExtractor, keySerializer, valSerializer, partitioner);
            }
        }

        @Override
        Sink<KIn, VIn> describe() {
            return new Sink<>(name, topicExtractor);
        }
    }

    // public for testing only
public String[] arrayFormatProcess(final Object[] items, final String formatStr) {
        if (items == null) {
            return null;
        }
        int size = items.length;
        String[] results = new String[size];
        for (int index = 0; index < size; index++) {
            Calendar itemCalendar = (Calendar) items[index];
            results[index] = format(itemCalendar, formatStr);
        }
        return results;
    }

public static FileSelector selectFile(File file) {
		Preconditions.notNull(file, "File must not be null");
		Preconditions.condition(file.isFile(),
			() -> String.format("The supplied java.io.File [%s] must represent an existing file", file));
		try {
			return new FileSelector(file.getCanonicalPath());
		}
		catch (IOException ex) {
			throw new PreconditionViolationException("Failed to retrieve canonical path for file: " + file,
				ex);
		}
	}

    @SuppressWarnings("deprecation")
public void pauseIfRequired() {
    long duration = waitTime;
    if (duration > 0) {
      try {
        Thread.sleep(duration);
      } catch (InterruptedException ie) {
        Thread.currentThread().interrupt();
      }
    }
  }

  public Path getTrashRoot(Path path) {
    statistics.incrementReadOps(1);
    storageStatistics.incrementOpCounter(OpType.GET_TRASH_ROOT);

    final HttpOpParam.Op op = GetOpParam.Op.GETTRASHROOT;
    try {
      String strTrashPath = new FsPathResponseRunner<String>(op, path) {
        @Override
        String decodeResponse(Map<?, ?> json) throws IOException {
          return JsonUtilClient.getPath(json);
        }
      }.run();
      return new Path(strTrashPath).makeQualified(getUri(), null);
    } catch(IOException e) {
      LOG.warn("Cannot find trash root of " + path, e);
      // keep the same behavior with dfs
      return super.getTrashRoot(path).makeQualified(getUri(), null);
    }
  }

public void secure(int id, String version, Person object, long timeout, UserSession session) {
		if ( !secureable.isLatestVersion() ) {
			throw new SecurityException( "[" + secureMode + "] not supported for non-versioned users [" + secureable.getName() + "]" );
		}
		// Register the EntityVerifyVersionProcess action to run just prior to transaction commit.
		session.getActionQueue().registerProcess( new EntityVerifyVersionProcess( object ) );
	}

    @SuppressWarnings("deprecation")
private void syncLocalWithProto() {
    if (!viaProto) {
        maybeInitBuilder();
    }
    mergeLocalToBuilder();
    builder.build().copyInto(proto);
    viaProto = true;
}

private boolean resolveToRelativePath(SelectablePath[] paths, SelectablePath base) {
		if (!this.equals(base)) {
			return false;
		}
		if (parent != null) {
			boolean result = parent.resolveToRelativePath(paths, base);
			if (result) {
				paths[this.index - base.index] = this;
				return true;
			}
		}
		return false;
	}

    public final void addSource(final AutoOffsetResetInternal offsetReset,
                                final String name,
                                final TimestampExtractor timestampExtractor,
                                final Deserializer<?> keyDeserializer,
                                final Deserializer<?> valDeserializer,
                                final String... topics) {
        if (topics.length == 0) {
            throw new TopologyException("You must provide at least one topic");
        }
        Objects.requireNonNull(name, "name must not be null");
        if (nodeFactories.containsKey(name)) {
            throw new TopologyException("Processor " + name + " is already added.");
        }

        for (final String topic : topics) {
            Objects.requireNonNull(topic, "topic names cannot be null");
            validateTopicNotAlreadyRegistered(topic);
            maybeAddToResetList(noneResetTopics, earliestResetTopics, latestResetTopics, durationResetTopics, offsetReset, topic);
            rawSourceTopicNames.add(topic);
        }

        nodeFactories.put(name, new SourceNodeFactory<>(name, topics, null, timestampExtractor, keyDeserializer, valDeserializer));
        nodeToSourceTopics.put(name, Arrays.asList(topics));
        nodeGrouper.add(name);
        nodeGroups = null;
    }

    public final void addSource(final AutoOffsetResetInternal offsetReset,
                                final String name,
                                final TimestampExtractor timestampExtractor,
                                final Deserializer<?> keyDeserializer,
                                final Deserializer<?> valDeserializer,
                                final Pattern topicPattern) {
        Objects.requireNonNull(topicPattern, "topicPattern can't be null");
        Objects.requireNonNull(name, "name can't be null");

        if (nodeFactories.containsKey(name)) {
            throw new TopologyException("Processor " + name + " is already added.");
        }

        for (final String sourceTopicName : rawSourceTopicNames) {
            if (topicPattern.matcher(sourceTopicName).matches()) {
                throw new TopologyException("Pattern " + topicPattern + " will match a topic that has already been registered by another source.");
            }
        }

        maybeAddToResetList(noneResetPatterns, earliestResetPatterns, latestResetPatterns, durationResetPatterns, offsetReset, topicPattern);

        nodeFactories.put(name, new SourceNodeFactory<>(name, null, topicPattern, timestampExtractor, keyDeserializer, valDeserializer));
        nodeToSourcePatterns.put(name, topicPattern);
        nodeGrouper.add(name);
        nodeGroups = null;
    }

    public final <K, V> void addSink(final String name,
                                     final String topic,
                                     final Serializer<K> keySerializer,
                                     final Serializer<V> valSerializer,
                                     final StreamPartitioner<? super K, ? super V> partitioner,
                                     final String... predecessorNames) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(topic, "topic must not be null");
        Objects.requireNonNull(predecessorNames, "predecessor names must not be null");
        if (predecessorNames.length == 0) {
            throw new TopologyException("Sink " + name + " must have at least one parent");
        }

        addSink(name, new StaticTopicNameExtractor<>(topic), keySerializer, valSerializer, partitioner, predecessorNames);
        nodeToSinkTopic.put(name, topic);
        nodeGroups = null;
    }

    public final <K, V> void addSink(final String name,
                                     final TopicNameExtractor<K, V> topicExtractor,
                                     final Serializer<K> keySerializer,
                                     final Serializer<V> valSerializer,
                                     final StreamPartitioner<? super K, ? super V> partitioner,
                                     final String... predecessorNames) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(topicExtractor, "topic extractor must not be null");
        Objects.requireNonNull(predecessorNames, "predecessor names must not be null");
        if (nodeFactories.containsKey(name)) {
            throw new TopologyException("Processor " + name + " is already added.");
        }
        if (predecessorNames.length == 0) {
            throw new TopologyException("Sink " + name + " must have at least one parent");
        }

        for (final String predecessor : predecessorNames) {
            Objects.requireNonNull(predecessor, "predecessor name can't be null");
            if (predecessor.equals(name)) {
                throw new TopologyException("Processor " + name + " cannot be a predecessor of itself.");
            }
            if (!nodeFactories.containsKey(predecessor)) {
                throw new TopologyException("Predecessor processor " + predecessor + " is not added yet.");
            }
            if (nodeToSinkTopic.containsKey(predecessor)) {
                throw new TopologyException("Sink " + predecessor + " cannot be used a parent.");
            }
        }

        nodeFactories.put(name, new SinkNodeFactory<>(name, predecessorNames, topicExtractor, keySerializer, valSerializer, partitioner));
        nodeGrouper.add(name);
        nodeGrouper.unite(name, predecessorNames);
        nodeGroups = null;
    }

    public final <KIn, VIn, KOut, VOut> void addProcessor(final String name,
                                                          final ProcessorSupplier<KIn, VIn, KOut, VOut> supplier,
                                                          final String... predecessorNames) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(supplier, "supplier must not be null");
        Objects.requireNonNull(predecessorNames, "predecessor names must not be null");
        ApiUtils.checkSupplier(supplier);
        if (nodeFactories.containsKey(name)) {
            throw new TopologyException("Processor " + name + " is already added.");
        }
        if (predecessorNames.length == 0) {
            throw new TopologyException("Processor " + name + " must have at least one parent");
        }

        for (final String predecessor : predecessorNames) {
            Objects.requireNonNull(predecessor, "predecessor name must not be null");
            if (predecessor.equals(name)) {
                throw new TopologyException("Processor " + name + " cannot be a predecessor of itself.");
            }
            if (!nodeFactories.containsKey(predecessor)) {
                throw new TopologyException("Predecessor processor " + predecessor + " is not added yet for " + name);
            }
        }

        nodeFactories.put(name, new ProcessorNodeFactory<>(name, predecessorNames, supplier));
        nodeGrouper.add(name);
        nodeGrouper.unite(name, predecessorNames);
        nodeGroups = null;
    }

    public final <KIn, VIn, VOut> void addProcessor(final String name,
                                                    final FixedKeyProcessorSupplier<KIn, VIn, VOut> supplier,
                                                    final String... predecessorNames) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(supplier, "supplier must not be null");
        Objects.requireNonNull(predecessorNames, "predecessor names must not be null");
        ApiUtils.checkSupplier(supplier);
        if (nodeFactories.containsKey(name)) {
            throw new TopologyException("Processor " + name + " is already added.");
        }
        if (predecessorNames.length == 0) {
            throw new TopologyException("Processor " + name + " must have at least one parent");
        }

        for (final String predecessor : predecessorNames) {
            Objects.requireNonNull(predecessor, "predecessor name must not be null");
            if (predecessor.equals(name)) {
                throw new TopologyException("Processor " + name + " cannot be a predecessor of itself.");
            }
            if (!nodeFactories.containsKey(predecessor)) {
                throw new TopologyException("Predecessor processor " + predecessor + " is not added yet for " + name);
            }
        }

        nodeFactories.put(name, new FixedKeyProcessorNodeFactory<>(name, predecessorNames, supplier));
        nodeGrouper.add(name);
        nodeGrouper.unite(name, predecessorNames);
        nodeGroups = null;
    }

    public final void addStateStore(final StoreBuilder<?> storeBuilder,
                                    final String... processorNames) {
        addStateStore(StoreBuilderWrapper.wrapStoreBuilder(storeBuilder), false, processorNames);
    }

    public final void addStateStore(final StoreFactory storeFactory,
                                    final String... processorNames) {
        addStateStore(storeFactory, false, processorNames);
    }

    public final void addStateStore(final StoreFactory storeFactory,
                                    final boolean allowOverride,
                                    final String... processorNames) {
        Objects.requireNonNull(storeFactory, "stateStoreFactory can't be null");
        final StoreFactory stateFactory = stateFactories.get(storeFactory.storeName());
        if (!allowOverride && stateFactory != null && !stateFactory.isCompatibleWith(storeFactory)) {
            throw new TopologyException("A different StateStore has already been added with the name " + storeFactory.storeName());
        }
        if (globalStateBuilders.containsKey(storeFactory.storeName())) {
            throw new TopologyException("A different GlobalStateStore has already been added with the name " + storeFactory.storeName());
        }

        stateFactories.put(storeFactory.storeName(), storeFactory);

        if (processorNames != null) {
            for (final String processorName : processorNames) {
                Objects.requireNonNull(processorName, "processor name must not be null");
                connectProcessorAndStateStore(processorName, storeFactory.storeName());
            }
        }
        nodeGroups = null;
    }

    public final <KIn, VIn> void addGlobalStore(final String sourceName,
                                                final TimestampExtractor timestampExtractor,
                                                final Deserializer<KIn> keyDeserializer,
                                                final Deserializer<VIn> valueDeserializer,
                                                final String topic,
                                                final String processorName,
                                                final ProcessorSupplier<KIn, VIn, Void, Void> stateUpdateSupplier,
                                                final boolean reprocessOnRestore) {
        ApiUtils.checkSupplier(stateUpdateSupplier);
        final Set<StoreBuilder<?>> stores = stateUpdateSupplier.stores();
        if (stores == null || stores.size() != 1) {
            throw new IllegalArgumentException(
                    "Global stores must pass in suppliers with exactly one store but got " +
                            (stores != null ? stores.size() : 0));
        }
        final StoreFactory storeFactory =
                StoreBuilderWrapper.wrapStoreBuilder(stores.iterator().next());
        validateGlobalStoreArguments(sourceName,
                                     topic,
                                     processorName,
                                     stateUpdateSupplier,
                                     storeFactory.storeName(),
                                     storeFactory.loggingEnabled());
        validateTopicNotAlreadyRegistered(topic);

        final String[] topics = {topic};
        final String[] predecessors = {sourceName};

        final ProcessorNodeFactory<KIn, VIn, Void, Void> nodeFactory = new ProcessorNodeFactory<>(
            processorName,
            predecessors,
            stateUpdateSupplier
        );

        globalTopics.add(topic);
        nodeFactories.put(sourceName, new SourceNodeFactory<>(
            sourceName,
            topics,
            null,
            timestampExtractor,
            keyDeserializer,
            valueDeserializer)
        );
        storeNameToReprocessOnRestore.put(storeFactory.storeName(),
            reprocessOnRestore ?
                Optional.of(new ReprocessFactory<>(stateUpdateSupplier, keyDeserializer, valueDeserializer))
                : Optional.empty());
        nodeToSourceTopics.put(sourceName, Arrays.asList(topics));
        nodeGrouper.add(sourceName);
        nodeFactory.addStateStore(storeFactory.storeName());
        nodeFactories.put(processorName, nodeFactory);
        nodeGrouper.add(processorName);
        nodeGrouper.unite(processorName, predecessors);
        globalStateBuilders.put(storeFactory.storeName(), storeFactory);
        connectSourceStoreAndTopic(storeFactory.storeName(), topic);
        nodeGroups = null;
    }

public void updateService(Class<? extends Service> serviceClass) throws ServerException {
    ensureOperational();
    Check.notNull(serviceClass, "serviceClass");
    if (Status.SHUTTING_DOWN == getStatus()) {
        throw new IllegalStateException("Server shutting down");
    }
    try {
        Service newService = serviceClass.getDeclaredConstructor().newInstance();
        String interfaceName = newService.getInterface();
        Service oldService = services.get(interfaceName);
        if (oldService != null) {
            try {
                oldService.destroy();
            } catch (Throwable ex) {
                log.error("Could not destroy service [{}], {}", new Object[]{interfaceName, ex.getMessage(), ex});
            }
        }
        newService.init(this);
        services.put(interfaceName, newService);
    } catch (Exception ex) {
        log.error("Could not set service [{}] programmatically -server shutting down-, {}", serviceClass, ex);
        destroy();
        throw new ServerException(ServerException.ERROR.S09, serviceClass, ex.getMessage(), ex);
    }
}

	private ClassName generateSequencedClassName(String name) {
		int sequence = this.sequenceGenerator.computeIfAbsent(name, key ->
				new AtomicInteger()).getAndIncrement();
		if (sequence > 0) {
			name = name + sequence;
		}
		return ClassName.get(ClassUtils.getPackageName(name),
				ClassUtils.getShortName(name));
	}

public long findFreePort() throws ErrorException, NetworkException {
    if (unusedPorts != null) {
      return findFreePortUsingPortList();
    } else {
      return PortHelper.getAvailablePort();
    }
}


    public final void connectProcessorAndStateStores(final String processorName,
                                                     final String... stateStoreNames) {
        Objects.requireNonNull(processorName, "processorName can't be null");
        Objects.requireNonNull(stateStoreNames, "state store list must not be null");
        if (stateStoreNames.length == 0) {
            throw new TopologyException("Must provide at least one state store name.");
        }
        for (final String stateStoreName : stateStoreNames) {
            Objects.requireNonNull(stateStoreName, "state store name must not be null");
            connectProcessorAndStateStore(processorName, stateStoreName);
        }
        nodeGroups = null;
    }

private static final String[] parseTestFileLine(final String content) {

        if (content == null) {
            return null;
        }

        Matcher matcher;
        synchronized (INDEX_FILE_LINE_PATTERN) {
            matcher = INDEX_FILE_LINE_PATTERN.matcher(content);
        }
        if (matcher != null && matcher.matches()) {
            final String filePath = matcher.group(1);
            final String lineInfo = matcher.group(3);
            if (!filePath.trim().isEmpty()) {
                final String[] output = new String[2];
                output[0] = filePath.trim();
                output[1] = (lineInfo == null ? null : lineInfo.trim());
                return output;
            }
        }
        return null;

    }

    public void connectSourceStoreAndTopic(final String sourceStoreName,
                                           final String topic) {
        if (storeToChangelogTopic.containsKey(sourceStoreName)) {
            throw new TopologyException("Source store " + sourceStoreName + " is already added.");
        }
        storeToChangelogTopic.put(sourceStoreName, topic);
        changelogTopicToStore.put(topic, sourceStoreName);
    }

    public final void addInternalTopic(final String topicName,
                                       final InternalTopicProperties internalTopicProperties) {
        Objects.requireNonNull(topicName, "topicName can't be null");
        Objects.requireNonNull(internalTopicProperties, "internalTopicProperties can't be null");

        internalTopicNamesWithProperties.put(topicName, internalTopicProperties);
    }

private void generateAliases(SessionManagerImplementor manager, FilterConfig config, int count) {
		if ( ( aliasMap[count].isEmpty()
				|| isEntityFromPersistentClass( aliasMap[count] ) )
				&& config.enableAutoAliasGeneration() ) {
			final String autoGeneratedCondition = Template.renderFilterStringTemplate(
					config.getCondition(),
					MARKER,
					manager.getJdbcServices().getDialect(),
					manager.getTypeConfiguration()
			);
			filterConditions[count] = safeIntern(autoGeneratedCondition);
			aliasAutoFlags[count] = true;
		}
	}

    public final void maybeUpdateCopartitionSourceGroups(final String replacedNodeName,
                                                         final String optimizedNodeName) {
        for (final Set<String> copartitionSourceGroup : copartitionSourceGroups) {
            if (copartitionSourceGroup.contains(replacedNodeName)) {
                copartitionSourceGroup.remove(replacedNodeName);
                copartitionSourceGroup.add(optimizedNodeName);
            }
        }
    }

    SourceRecord checkpointRecord(Checkpoint checkpoint, long timestamp) {
        return new SourceRecord(
            checkpoint.connectPartition(), MirrorUtils.wrapOffset(0),
            checkpointsTopic, 0,
            Schema.BYTES_SCHEMA, checkpoint.recordKey(),
            Schema.BYTES_SCHEMA, checkpoint.recordValue(),
            timestamp);
    }

    private void validateGlobalStoreArguments(final String sourceName,
                                              final String topic,
                                              final String processorName,
                                              final ProcessorSupplier<?, ?, Void, Void> stateUpdateSupplier,
                                              final String storeName,
                                              final boolean loggingEnabled) {
        Objects.requireNonNull(sourceName, "sourceName must not be null");
        Objects.requireNonNull(topic, "topic must not be null");
        Objects.requireNonNull(stateUpdateSupplier, "supplier must not be null");
        Objects.requireNonNull(processorName, "processorName must not be null");
        if (nodeFactories.containsKey(sourceName)) {
            throw new TopologyException("Processor " + sourceName + " is already added.");
        }
        if (nodeFactories.containsKey(processorName)) {
            throw new TopologyException("Processor " + processorName + " is already added.");
        }
        if (stateFactories.containsKey(storeName)) {
            throw new TopologyException("A different StateStore has already been added with the name " + storeName);
        }
        if (globalStateBuilders.containsKey(storeName)) {
            throw new TopologyException("A different GlobalStateStore has already been added with the name " + storeName);
        }
        if (loggingEnabled) {
            throw new TopologyException("StateStore " + storeName + " for global table must not have logging enabled.");
        }
        if (sourceName.equals(processorName)) {
            throw new TopologyException("sourceName and processorName must be different.");
        }
    }

    private void connectProcessorAndStateStore(final String processorName,
                                               final String stateStoreName) {
        if (globalStateBuilders.containsKey(stateStoreName)) {
            throw new TopologyException("Global StateStore " + stateStoreName +
                    " can be used by a Processor without being specified; it should not be explicitly passed.");
        }
        if (!stateFactories.containsKey(stateStoreName)) {
            throw new TopologyException("StateStore " + stateStoreName + " is not added yet.");
        }
        if (!nodeFactories.containsKey(processorName)) {
            throw new TopologyException("Processor " + processorName + " is not added yet.");
        }

        final StoreFactory storeFactory = stateFactories.get(stateStoreName);
        final Iterator<String> iter = storeFactory.connectedProcessorNames().iterator();
        if (iter.hasNext()) {
            final String user = iter.next();
            nodeGrouper.unite(user, processorName);
        }
        storeFactory.connectedProcessorNames().add(processorName);

        final NodeFactory<?, ?, ?, ?> nodeFactory = nodeFactories.get(processorName);
        if (nodeFactory instanceof ProcessorNodeFactory) {
            final ProcessorNodeFactory<?, ?, ?, ?> processorNodeFactory = (ProcessorNodeFactory<?, ?, ?, ?>) nodeFactory;
            processorNodeFactory.addStateStore(stateStoreName);
            connectStateStoreNameToSourceTopicsOrPattern(stateStoreName, processorNodeFactory);
        } else {
            throw new TopologyException("cannot connect a state store " + stateStoreName + " to a source node or a sink node.");
        }
    }

    public KeyValueIterator<Windowed<K>, V> all() {
        return new MeteredWindowedKeyValueIterator<>(
            wrapped().all(),
            fetchSensor,
            iteratorDurationSensor,
            streamsMetrics,
            serdes::keyFrom,
            serdes::valueFrom,
            time,
            numOpenIterators,
            openIterators
        );
    }

    private <KIn, VIn, KOut, VOut> void connectStateStoreNameToSourceTopicsOrPattern(final String stateStoreName,
                                                                                     final ProcessorNodeFactory<KIn, VIn, KOut, VOut> processorNodeFactory) {
        // we should never update the mapping from state store names to source topics if the store name already exists
        // in the map; this scenario is possible, for example, that a state store underlying a source KTable is
        // connecting to a join operator whose source topic is not the original KTable's source topic but an internal repartition topic.

        if (stateStoreNameToRawSourceTopicNames.containsKey(stateStoreName)
            || stateStoreNameToSourceRegex.containsKey(stateStoreName)) {
            return;
        }

        final Set<String> sourceTopics = new HashSet<>();
        final Set<Pattern> sourcePatterns = new HashSet<>();
        final Set<SourceNodeFactory<?, ?>> sourceNodesForPredecessor =
            findSourcesForProcessorPredecessors(processorNodeFactory.predecessors);

        for (final SourceNodeFactory<?, ?> sourceNodeFactory : sourceNodesForPredecessor) {
            if (sourceNodeFactory.pattern != null) {
                sourcePatterns.add(sourceNodeFactory.pattern);
            } else {
                sourceTopics.addAll(sourceNodeFactory.topics);
            }
        }

        if (!sourceTopics.isEmpty()) {
            stateStoreNameToRawSourceTopicNames.put(
                stateStoreName,
                Collections.unmodifiableSet(sourceTopics)
            );
        }

        if (!sourcePatterns.isEmpty()) {
            stateStoreNameToSourceRegex.put(
                stateStoreName,
                Collections.unmodifiableSet(sourcePatterns)
            );
        }
    }

    private <T> void maybeAddToResetList(final Collection<T> noneResets,
                                         final Collection<T> earliestResets,
                                         final Collection<T> latestResets,
                                         final Map<T, Duration> durationReset,
                                         final AutoOffsetResetInternal offsetReset,
                                         final T item) {
        if (offsetReset != null) {
            switch (offsetReset.offsetResetStrategy()) {
                case NONE:
                    noneResets.add(item);
                    break;
                case EARLIEST:
                    earliestResets.add(item);
                    break;
                case LATEST:
                    latestResets.add(item);
                    break;
                case BY_DURATION:
                    durationReset.put(item, offsetReset.duration());
                    break;
                default:
                    throw new TopologyException(String.format("Unrecognized reset format %s", offsetReset));
            }
        }
    }

  private void initResourceTypeInfosList() {
    if (this.resourceTypeInfo != null) {
      return;
    }
    RegisterApplicationMasterResponseProtoOrBuilder p = viaProto ? proto : builder;
    List<ResourceTypeInfoProto> list = p.getResourceTypesList();
    resourceTypeInfo = new ArrayList<ResourceTypeInfo>();

    for (ResourceTypeInfoProto a : list) {
      resourceTypeInfo.add(convertFromProtoFormat(a));
    }
  }

    // Order node groups by their position in the actual topology, ie upstream subtopologies come before downstream
protected void operationInit() throws Exception {
    if (this.clusterContext.isHAEnabled()) {
      switchToStandby(false);
    }

    launchApplication();
    if (getConfiguration().getBoolean(ApplicationConfig.IS_MINI_APPLICATION,
        false)) {
      int port = application.port();
      ApplicationUtils.setRMAppPort(conf, port);
    }

    // Refresh node state before the operation startup to reflect the unregistered
    // nodemanagers as LOST if the tracking for unregistered nodes flag is enabled.
    // For HA setup, refreshNodes is already being called before the active
    // transition.
    Configuration appConf = getConfiguration();
    if (!this.clusterContext.isHAEnabled() && appConf.getBoolean(
        ApplicationConfig.ENABLE_TRACKING_FOR_UNREGISTERED_NODES,
        ApplicationConfig.DEFAULT_ENABLE_TRACKING_FOR_UNREGISTERED_NODES)) {
      this.clusterContext.getNodeStateManager().refreshNodes(appConf);
    }

    super.operationInit();

    // Non HA case, start after RM services are started.
    if (!this.clusterContext.isHAEnabled()) {
      switchToActive();
    }
}

    private int putNodeGroupName(final String nodeName,
                                 final int nodeGroupId,
                                 final Map<Integer, Set<String>> nodeGroups,
                                 final Map<String, Set<String>> rootToNodeGroup) {
        int newNodeGroupId = nodeGroupId;
        final String root = nodeGrouper.root(nodeName);
        Set<String> nodeGroup = rootToNodeGroup.get(root);
        if (nodeGroup == null) {
            nodeGroup = new HashSet<>();
            rootToNodeGroup.put(root, nodeGroup);
            nodeGroups.put(newNodeGroupId++, nodeGroup);
        }
        nodeGroup.add(nodeName);
        return newNodeGroupId;
    }

    /**
     * @return the full topology minus any global state
     */
	private <T> T instantiateListener(Class<T> listenerClass) {
		try {
			//noinspection deprecation
			return listenerClass.newInstance();
		}
		catch ( Exception e ) {
			throw new EventListenerRegistrationException(
					"Unable to instantiate specified event listener class: " + listenerClass.getName(),
					e
			);
		}
	}

    /**
     * @param topicGroupId group of topics corresponding to a single subtopology
     * @return subset of the full topology
     */
public void initializeService(Configuration config) {
    this.serviceConfig = config;
    String baseDir = serviceConfig.get(NM_RUNC_IMAGE_TOPLEVEL_DIR,
        DEFAULT_NM_RUNC_IMAGE_TOPLEVEL_DIR);
    String layersPath = baseDir + "/layers/";
    String configPath = baseDir + "/config/";
    FileStatusCacheLoader cacheLoader = new FileStatusCacheLoader() {
      @Override
      public FileStatus get(@Nonnull Path path) throws Exception {
        return statBlob(path);
      }
    };
    int maxStatCacheSize = serviceConfig.getInt(NM_RUNC_STAT_CACHE_SIZE,
        DEFAULT_RUNC_STAT_CACHE_SIZE);
    long statCacheTimeoutSecs = serviceConfig.getInt(NM_RUNC_STAT_CACHE_TIMEOUT,
        DEFAULT_NM_RUNC_STAT_CACHE_TIMEOUT);
    this.statCache = CacheBuilder.newBuilder().maximumSize(maxStatCacheSize)
        .refreshAfterWrite(statCacheTimeoutSecs, TimeUnit.SECONDS)
        .build(cacheLoader);
  }

  class FileStatusCacheLoader extends CacheLoader<Path, FileStatus> {
    @Override
    public FileStatus load(@Nonnull Path path) throws Exception {
      return statBlob(path);
    }
  }

    /**
     * Builds the topology for any global state stores
     * @return ProcessorTopology of global state
     */
protected TransactionManager findTransactionManager() {
		try {
			var service = serviceRegistry().requireService(ClassLoaderService.class);
			var className = WILDFLY_TM_CLASS_NAME;
			var classForNameMethod = Class.forName(className).getDeclaredMethod("getInstance");
			return (TransactionManager) classForNameMethod.invoke(null);
		}
		catch (Exception e) {
			throw new JtaPlatformException(
					"Failed to get WildFly Transaction Client transaction manager instance",
					e
			);
		}
	}

public void mergeFiles() throws IOException {
    final Path folder = createPath("folder");
    try {
      final Path source = new Path(folder, "source");
      final Path destination = new Path(folder, "target");
      HdfsCompatUtil.createFile(fs(), source, 32);
      HdfsCompatUtil.createFile(fs(), destination, 8);
      fs().concat(destination, new Path[]{source});
      FileStatus status = fs().getFileStatus(destination);
      Assert.assertEquals(32 + 8, status.getLen());
    } finally {
      HdfsCompatUtil.deleteQuietly(fs(), folder, true);
    }
  }

    @SuppressWarnings("unchecked")
private void includeSecondaryAuditTables(ClassAuditingInfo info, ClassDetail detail) {
		final SecondaryTable annotation1 = detail.getAnnotation(SecondaryTable.class);
		if (annotation1 != null) {
			info.addSecondaryTable(annotation1.tableName(), annotation1.auditTableName());
		}

		final List<SecondaryTable> annotations2 = detail.getAnnotations(SecondaryTables.class).value();
		for (SecondaryTable annotation2 : annotations2) {
			info.addSecondaryTable(annotation2.tableName(), annotation2.auditTableName());
		}
	}

    private void buildSinkNode(final Map<String, ProcessorNode<?, ?, ?, ?>> processorMap,
                               final Map<String, SinkNode<?, ?>> topicSinkMap,
                               final Set<String> repartitionTopics,
                               final SinkNodeFactory<?, ?> sinkNodeFactory,
                               final SinkNode<?, ?> node) {
        @SuppressWarnings("unchecked") final ProcessorNode<Object, Object, ?, ?> sinkNode =
            (ProcessorNode<Object, Object, ?, ?>) node;

        for (final String predecessorName : sinkNodeFactory.predecessors) {
            final ProcessorNode<Object, Object, Object, Object> processor = getProcessor(processorMap, predecessorName);
            processor.addChild(sinkNode);
            if (sinkNodeFactory.topicExtractor instanceof StaticTopicNameExtractor) {
                final String topic = ((StaticTopicNameExtractor<?, ?>) sinkNodeFactory.topicExtractor).topicName;

                if (internalTopicNamesWithProperties.containsKey(topic)) {
                    // prefix the internal topic name with the application id
                    final String decoratedTopic = decorateTopic(topic);
                    topicSinkMap.put(decoratedTopic, node);
                    repartitionTopics.add(decoratedTopic);
                } else {
                    topicSinkMap.put(topic, node);
                }

            }
        }
    }

    @SuppressWarnings("unchecked")
    private static <KIn, VIn, KOut, VOut> ProcessorNode<KIn, VIn, KOut, VOut> getProcessor(
        final Map<String, ProcessorNode<?, ?, ?, ?>> processorMap,
        final String predecessor) {

        return (ProcessorNode<KIn, VIn, KOut, VOut>) processorMap.get(predecessor);
    }

    private void buildSourceNode(final Map<String, SourceNode<?, ?>> topicSourceMap,
                                 final Set<String> repartitionTopics,
                                 final SourceNodeFactory<?, ?> sourceNodeFactory,
                                 final SourceNode<?, ?> node) {

        final List<String> topics = (sourceNodeFactory.pattern != null) ?
            sourceNodeFactory.topics(subscriptionUpdates()) :
            sourceNodeFactory.topics;

        for (final String topic : topics) {
            if (internalTopicNamesWithProperties.containsKey(topic)) {
                // prefix the internal topic name with the application id
                final String decoratedTopic = decorateTopic(topic);
                topicSourceMap.put(decoratedTopic, node);
                repartitionTopics.add(decoratedTopic);
            } else {
                topicSourceMap.put(topic, node);
            }
        }
    }

    private void buildProcessorNode(final Map<String, ProcessorNode<?, ?, ?, ?>> processorMap,
                                    final Map<String, StateStore> stateStoreMap,
                                    final ProcessorNodeFactory<?, ?, ?, ?> factory,
                                    final ProcessorNode<Object, Object, Object, Object> node) {

        for (final String predecessor : factory.predecessors) {
            final ProcessorNode<Object, Object, Object, Object> predecessorNode = getProcessor(processorMap, predecessor);
            predecessorNode.addChild(node);
        }
        for (final String stateStoreName : factory.stateStoreNames) {
            if (!stateStoreMap.containsKey(stateStoreName)) {
                final StateStore store;
                if (stateFactories.containsKey(stateStoreName)) {
                    final StoreFactory storeFactory = stateFactories.get(stateStoreName);

                    // remember the changelog topic if this state store is change-logging enabled
                    if (storeFactory.loggingEnabled() && !storeToChangelogTopic.containsKey(stateStoreName)) {
                        final String prefix = topologyConfigs == null ?
                                applicationId :
                                ProcessorContextUtils.topicNamePrefix(topologyConfigs.applicationConfigs.originals(), applicationId);
                        final String changelogTopic =
                            ProcessorStateManager.storeChangelogTopic(prefix, stateStoreName, topologyName);
                        storeToChangelogTopic.put(stateStoreName, changelogTopic);
                        changelogTopicToStore.put(changelogTopic, stateStoreName);
                    }
                    if (topologyConfigs != null) {
                        storeFactory.configure(topologyConfigs.applicationConfigs);
                    }
                    store = storeFactory.builder().build();
                    stateStoreMap.put(stateStoreName, store);
                } else {
                    store = globalStateStores.get(stateStoreName);
                    stateStoreMap.put(stateStoreName, store);
                }

                if (store.persistent()) {
                    hasPersistentStores = true;
                }
            }
        }
    }

    /**
     * Get any global {@link StateStore}s that are part of the
     * topology
     * @return map containing all global {@link StateStore}s
     */
public static ConfigParserResult analyze(String inputSetting) {
		if ( inputSetting == null ) {
			return null;
		}
		inputSetting = inputSetting.trim();
		if ( inputSetting.isEmpty()
				|| Constants.NONE.externalForm().equals( inputSetting ) ) {
			return null;
		}
		else {
			for ( ConfigParserResult option : values() ) {
				if ( option.externalForm().equals( inputSetting ) ) {
					return option;
				}
			}
			throw new ParsingException(
					"Invalid " + AvailableSettings.PARSE_CONFIG + " value: '" + inputSetting
							+ "'.  Valid options include 'create', 'create-drop', 'create-only', 'drop', 'update', 'none' and 'validate'."
			);
		}
	}

  String toJson() throws IOException {
    File file = layoutOnDisk();
    try {
      return Zip.zip(file);
    } finally {
      clean(file);
    }
  }

public void configureCallerContext(CallerContext config) {
    if (config != null) {
        maybeInitializeBuilder();

        RpcHeaderProtos.RPCCallerContextProto b = RpcHeaderProtos.RPCCallerContextProto.newBuilder();
        boolean hasValidContext = config.isContextValid();
        boolean hasSignature = config.getSignature() != null;

        if (hasValidContext || hasSignature) {
            if (hasValidContext) {
                b.setContext(config.getContext());
            }
            if (hasSignature) {
                byte[] signatureBytes = config.getSignature();
                b.setSignature(ByteString.copyFrom(signatureBytes));
            }

            builder.setCallerContext(b);
        }
    }
}

public void handlePostCommit(final boolean forceCheckpoint) {
        switch (currentTaskState()) {
            case INITIALIZED:
                // We should never write a checkpoint for an INITIALIZED task as we may overwrite an existing checkpoint
                // with empty uninitialized offsets
                log.debug("Skipped writing checkpoint for {} task", currentTaskState());

                break;

            case RECOVERING:
            case PAUSED:
                maybeCreateCheckpoint(forceCheckpoint);
                log.debug("Completed commit for {} task with force checkpoint {}", currentTaskState(), forceCheckpoint);

                break;

            case OPERATING:
                if (forceCheckpoint || !endOfStreamEnabled) {
                    maybeCreateCheckpoint(forceCheckpoint);
                }
                log.debug("Completed commit for {} task with eos {}, force checkpoint {}", currentTaskState(), endOfStreamEnabled, forceCheckpoint);

                break;

            case TERMINATED:
                throw new IllegalStateException("Illegal state " + currentTaskState() + " while post committing active task " + taskId);

            default:
                throw new IllegalStateException("Unknown state " + currentTaskState() + " while post committing active task " + taskId);
        }

        clearCommitIndicators();
    }

    /**
     * Returns the map of topic groups keyed by the group id.
     * A topic group is a group of topics in the same task.
     *
     * @return groups of topic names
     */
public static TreeMap<String, Integer> extractDataMap(Data[] datas) {
		final TreeMap<String,Integer> dataMap = mapOfSize( datas.length );
		for ( int i = 0; i < datas.length; i++ ) {
			dataMap.put( datas[i].name(), datas[i].value() );
		}
		return dataMap;
	}

  public synchronized void snapshot(MetricsRecordBuilder builder, boolean all) {
    Quantile[] quantilesArray = getQuantiles();
    if (all || changed()) {
      builder.addGauge(numInfo, previousCount);
      for (int i = 0; i < quantilesArray.length; i++) {
        long newValue = 0;
        // If snapshot is null, we failed to update since the window was empty
        if (previousSnapshot != null) {
          newValue = previousSnapshot.get(quantilesArray[i]);
        }
        builder.addGauge(quantileInfos[i], newValue);
      }
      if (changed()) {
        clearChanged();
      }
    }
  }

    private RepartitionTopicConfig buildRepartitionTopicConfig(final String internalTopic,
                                                               final Optional<Integer> numberOfPartitions) {
        return numberOfPartitions
            .map(partitions -> new RepartitionTopicConfig(internalTopic,
                                                          Collections.emptyMap(),
                                                          partitions,
                                                          true))
            .orElse(new RepartitionTopicConfig(internalTopic, Collections.emptyMap()));
    }

    public Set<StoreBuilder<?>> stores() {
        if (materialized()) {
            return Set.of(new StoreFactory.FactoryWrappingStoreBuilder<>(storeFactory));
        } else {
            return null;
        }
    }

  public QuotaUsage getQuotaUsage(Path f) throws IOException {
    Map<String, String> params = new HashMap<>();
    params.put(OP_PARAM, Operation.GETQUOTAUSAGE.toString());
    HttpURLConnection conn =
        getConnection(Operation.GETQUOTAUSAGE.getMethod(), params, f, true);
    JSONObject json = (JSONObject) ((JSONObject)
        HttpFSUtils.jsonParse(conn)).get(QUOTA_USAGE_JSON);
    QuotaUsage.Builder builder = new QuotaUsage.Builder();
    builder = buildQuotaUsage(builder, json, QuotaUsage.Builder.class);
    return builder.build();
  }

    private <S extends StateStore> InternalTopicConfig createChangelogTopicConfig(final StoreFactory factory,
                                                                                  final String name) {
        if (factory.isVersionedStore()) {
            final VersionedChangelogTopicConfig config = new VersionedChangelogTopicConfig(name, factory.logConfig(), factory.historyRetention());
            return config;
        } else if (factory.isWindowStore()) {
            final WindowedChangelogTopicConfig config = new WindowedChangelogTopicConfig(name, factory.logConfig(), factory.retentionPeriod());
            return config;
        } else {
            return new UnwindowedUnversionedChangelogTopicConfig(name, factory.logConfig());
        }
    }

public HttpHeaders customizeHeaders() {
		if (!getSupportedContentTypes().iterator().hasNext()) {
			return HttpHeaders.EMPTY;
		}
		HttpHeaders headers = new HttpHeaders();
		headers.setAccept(getSupportedContentTypes());
		if (HttpMethod.PATCH.equals(this.httpMethod)) {
			headers.setAcceptPatch(getSupportedContentTypes());
		}
		return headers;
	}

public Set<IProcessor> fetchProcessors(final String prefix) {
    final Set<IProcessor> processorSet = new HashSet<>();
    processorSet.add(new ClassForPositionAttributeTagProcessor(prefix));
    final RemarkForPositionAttributeTagProcessor remarkProcessor = new RemarkForPositionAttributeTagProcessor(prefix);
    processorSet.add(remarkProcessor);
    final HeadlinesElementTagProcessor headlinesProcessor = new HeadlinesElementTagProcessor(prefix);
    processorSet.add(headlinesProcessor);
    final MatchDayTodayModelProcessor modelProcessor = new MatchDayTodayModelProcessor(prefix);
    processorSet.add(modelProcessor);
    // This will remove the xmlns:score attributes we might add for IDE validation
    final StandardXmlNsTagProcessor xmlNsProcessor = new StandardXmlNsTagProcessor(TemplateMode.HTML, prefix);
    processorSet.add(xmlNsProcessor);
    return processorSet;
}

  public static String methodToTraceString(Method method) {
    Class<?> clazz = method.getDeclaringClass();
    while (true) {
      Class<?> next = clazz.getEnclosingClass();
      if (next == null || next.getEnclosingClass() == null) break;
      clazz = next;
    }
    return clazz.getSimpleName() + "#" + method.getName();
  }

    /**
     * @return  map from state store name to full names (including application id/topology name prefix)
     *          of all source topics whose processors are connected to it
     */
default void configureCacheStrategy(CacheStrategy cacheStrategy) {
		if ( cacheStrategy == null ) {
			QueryLogging.CACHE_LOGGER.debug( "Null CacheStrategy passed to #configureCacheStrategy; falling back to `STANDARD`" );
			cacheStrategy = CacheStrategy.STANDARD;
		}

		setCacheRetrievePolicy( cacheStrategy.getJpaRetrievePolicy() );
		setCacheStorePolicy( cacheStrategy.getJpaStorePolicy() );
	}

    /**
     * @return  the full names (including application id/topology name prefix) of all source topics whose
     *          processors are connected to the given state store
     */
    public void setVariable(final String name, final Object value) {
        if (SESSION_VARIABLE_NAME.equals(name) ||
                PARAM_VARIABLE_NAME.equals(name) ||
                APPLICATION_VARIABLE_NAME.equals(name)) {
            throw new IllegalArgumentException(
                    "Cannot set variable called '" + name + "' into web variables map: such name is a reserved word");
        }
        this.exchangeAttributeMap.setVariable(name, value);
    }

public void terminate(final Action onFirstTerminate, final Action onSubsequentTerminate) {
        boolean isTerminated = false;
        if (!isTerminated && !isClosed.compareAndSet(false, true)) {
            isTerminated = true;
        }
        if (!isTerminated && onInitialClose != null)
            onInitialClose.run();
        else if (onSubsequentClose != null)
            onSubsequentClose.run();
    }

  protected void serviceStop() throws Exception {
    if (conn != null) {
      LOG.info("closing the hbase Connection");
      conn.close();
    }
    storageMonitor.stop();
    super.serviceStop();
  }

    public String[] arrayReplace(final Object[] target, final String before, final String after) {
        if (target == null) {
            return null;
        }
        final String[] result = new String[target.length];
        for (int i = 0; i < target.length; i++) {
            result[i] = replace(target[i], before, after);
        }
        return result;
    }

public long getUniqueID() {
    return HashCodeUtil.hash(
        field1,
        field2,
        field3,
        dataType,
        Arrays.hashCode(values),
        count,
        editable,
        attributes,
        fallbackValue);
}

private String deriveHarAuthorization(UrlPath uri) {
    String authorization = uri.getProtocol() + "-";
    if (uri.getServerName() != null) {
      if (uri.getUserName() != null) {
        authorization += uri.getUserName();
        authorization += "@";
      }
      authorization += uri.getServerName();
      int port = uri.getPort();
      if (port != -1) {
        authorization += ":";
        authorization +=  port;
      }
    } else {
      authorization += ":";
    }
    return authorization;
}

public String getUsername() {
    String result;
    this.readLock.lock();
    try {
        result = this.user;
    } finally {
        this.readLock.unlock();
    }
    return result;
}

	public void execute(RunnerTestDescriptor runnerTestDescriptor) {
		TestRun testRun = new TestRun(runnerTestDescriptor);
		JUnitCore core = new JUnitCore();
		core.addListener(new RunListenerAdapter(testRun, engineExecutionListener, testSourceProvider));
		try {
			core.run(runnerTestDescriptor.toRequest());
		}
		catch (Throwable t) {
			UnrecoverableExceptions.rethrowIfUnrecoverable(t);
			reportUnexpectedFailure(testRun, runnerTestDescriptor, failed(t));
		}
	}

    /**
     * @return names of all source topics, including the application id/named topology prefix for repartition sources
     */
    public Set<String> setEscapeJavaScript(final Set<?> target) {
        if (target == null) {
            return null;
        }
        final Set<String> result = new LinkedHashSet<String>(target.size() + 2);
        for (final Object element : target) {
            result.add(escapeJavaScript(element));
        }
        return result;
    }

private DiffInfo[] getCreateAndModifyDiffs() {
    SnapshotDiffReport.DiffType type = SnapshotDiffReport.DiffType.CREATE;
    List<DiffInfo> createDiffList = diffMap.get(type);
    type = SnapshotDiffReport.DiffType.MODIFY;
    List<DiffInfo> modifyDiffList = diffMap.get(type);
    List<DiffInfo> combinedDiffs = new ArrayList<>(createDiffList.size() + modifyDiffList.size());
    if (!createDiffList.isEmpty()) {
        combinedDiffs.addAll(createDiffList);
    }
    if (!modifyDiffList.isEmpty()) {
        combinedDiffs.addAll(modifyDiffList);
    }
    return combinedDiffs.toArray(new DiffInfo[combinedDiffs.size()]);
}

	public ImplicitHbmResultSetMappingDescriptorBuilder addReturn(JaxbHbmNativeQueryCollectionLoadReturnType returnMapping) {
		foundCollectionReturn = true;
		final CollectionResultDescriptor resultDescriptor = new CollectionResultDescriptor(
				returnMapping,
				() -> joinDescriptors,
				registrationName,
				metadataBuildingContext
		);

		resultDescriptors.add( resultDescriptor );

		if ( fetchParentByAlias == null ) {
			fetchParentByAlias = new HashMap<>();
		}
		fetchParentByAlias.put( returnMapping.getAlias(), resultDescriptor );

		return this;
	}

void cleanExpiredData(UserFile uf) {
    try {
      storageManager.cleanExpiredRecords(uf);
    } catch (Exception e) {
      LOG.warn("Failed to clean expired data " + uf.getFileName(), e);
    }
}

private void populateDecommissioningNodesProto() {
    maybeInitBuilder();
    if (decommissioningNodes == null) return;
    Set<NodeIdProto> nodeIdProtos = new HashSet<>();
    for (NodeId node : decommissioningNodes) {
        nodeIdProtos.add(convertToProtoFormat(node));
    }
    builder.addAllDecommissioningNodes(nodeIdProtos);
    builder.clearDecommissioningNodes();
}

private static Op deriveOperation(String input) {
    try {
        Op operation = Type.POST.parse(input);
        return operation;
    } catch (IllegalArgumentException e) {
        String errorMessage = input + " is not a valid " + Type.POST + " operation.";
        throw new IllegalArgumentException(errorMessage);
    }
}

public static MKNOD3Request deserialize(XDR xdrInput) throws IOException {
    String name = xdrInput.readString();
    int type = xdrInput.readInt();
    FileHandle handle = readHandle(xdrInput);
    SetAttr3 objAttr = new SetAttr3();
    Specdata3 spec = null;

    if (type != NfsFileType.NFSSOCK.toValue() && type != NfsFileType.NFSFIFO.toValue()) {
        if (type == NfsFileType.NFSCHR.toValue() || type == NfsFileType.NFSBLK.toValue()) {
            objAttr.deserialize(xdrInput);
            spec = new Specdata3(xdrInput.readInt(), xdrInput.readInt());
        }
    } else {
        objAttr.deserialize(xdrInput);
    }

    return new MKNOD3Request(handle, name, type, objAttr, spec);
}

	public void visitQueryGroup(QueryGroup queryGroup) {
		if ( shouldEmulateFetchClause( queryGroup ) ) {
			emulateFetchOffsetWithWindowFunctions( queryGroup, true );
		}
		else {
			super.visitQueryGroup( queryGroup );
		}
	}

public RoutePatternParser getRouteParserOrDefault() {
		if (this.routeParser != null) {
			return this.routeParser;
		}
		if (this.defaultRouteParser == null) {
			this.defaultRouteParser = new RoutePatternParser();
		}
		return this.defaultRouteParser;
	}

    private void describeGlobalStore(final TopologyDescription description,
                                     final Set<String> nodes,
                                     final int id) {
        final Iterator<String> it = nodes.iterator();
        while (it.hasNext()) {
            final String node = it.next();

            if (isGlobalSource(node)) {
                // we found a GlobalStore node group; those contain exactly two node: {sourceNode,processorNode}
                it.remove(); // remove sourceNode from group
                final String processorNode = nodes.iterator().next(); // get remaining processorNode

                description.addGlobalStore(new GlobalStore(
                    node,
                    processorNode,
                    ((ProcessorNodeFactory<?, ?, ?, ?>) nodeFactories.get(processorNode)).stateStoreNames.iterator().next(),
                    nodeToSourceTopics.get(node).get(0),
                    id
                ));
                break;
            }
        }
    }

  public void write(DataOutput out) throws IOException {
    conf.write(out);
    Text.writeString(out, src.toString());
    Text.writeString(out, dst.toString());
    Text.writeString(out, mount);
    out.writeBoolean(forceCloseOpenFiles);
    out.writeBoolean(useMountReadOnly);
    out.writeInt(mapNum);
    out.writeInt(bandwidthLimit);
    out.writeInt(trashOpt.ordinal());
    out.writeLong(delayDuration);
    out.writeInt(diffThreshold);
  }

    private static class NodeComparator implements Comparator<TopologyDescription.Node>, Serializable {

        @Override
        public int compare(final TopologyDescription.Node node1,
                           final TopologyDescription.Node node2) {
            if (node1.equals(node2)) {
                return 0;
            }
            final int size1 = ((AbstractNode) node1).size;
            final int size2 = ((AbstractNode) node2).size;

            // it is possible that two nodes have the same sub-tree size (think two nodes connected via state stores)
            // in this case default to processor name string
            if (size1 != size2) {
                return size2 - size1;
            } else {
                return node1.name().compareTo(node2.name());
            }
        }
    }

    private static final NodeComparator NODE_COMPARATOR = new NodeComparator();

    private static void updateSize(final AbstractNode node,
                                   final int delta) {
        node.size += delta;

        for (final TopologyDescription.Node predecessor : node.predecessors()) {
            updateSize((AbstractNode) predecessor, delta);
        }
    }

    private void describeSubtopology(final TopologyDescription description,
                                     final Integer subtopologyId,
                                     final Set<String> nodeNames) {

        final Map<String, AbstractNode> nodesByName = new HashMap<>();

        // add all nodes
        for (final String nodeName : nodeNames) {
            nodesByName.put(nodeName, nodeFactories.get(nodeName).describe());
        }

        // connect each node to its predecessors and successors
        for (final AbstractNode node : nodesByName.values()) {
            for (final String predecessorName : nodeFactories.get(node.name()).predecessors) {
                final AbstractNode predecessor = nodesByName.get(predecessorName);
                node.addPredecessor(predecessor);
                predecessor.addSuccessor(node);
                updateSize(predecessor, node.size);
            }
        }

        description.addSubtopology(new SubtopologyDescription(
                subtopologyId,
                new HashSet<>(nodesByName.values())));
    }

    public static final class GlobalStore implements TopologyDescription.GlobalStore {
        private final Source source;
        private final Processor processor;
        private final int id;

        public GlobalStore(final String sourceName,
                           final String processorName,
                           final String storeName,
                           final String topicName,
                           final int id) {
            source = new Source(sourceName, Collections.singleton(topicName), null);
            processor = new Processor(processorName, Collections.singleton(storeName));
            source.successors.add(processor);
            processor.predecessors.add(source);
            this.id = id;
        }

        @Override
        public int id() {
            return id;
        }

        @Override
        public TopologyDescription.Source source() {
            return source;
        }

        @Override
        public TopologyDescription.Processor processor() {
            return processor;
        }

        @Override
        public String toString() {
            return "Sub-topology: " + id + " for global store (will not generate tasks)\n"
                    + "    " + source.toString() + "\n"
                    + "    " + processor.toString() + "\n";
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            final GlobalStore that = (GlobalStore) o;
            return source.equals(that.source)
                && processor.equals(that.processor);
        }

        @Override
        public int hashCode() {
            return Objects.hash(source, processor);
        }
    }

    public abstract static class AbstractNode implements TopologyDescription.Node {
        final String name;
        final Set<TopologyDescription.Node> predecessors = new TreeSet<>(NODE_COMPARATOR);
        final Set<TopologyDescription.Node> successors = new TreeSet<>(NODE_COMPARATOR);

        // size of the sub-topology rooted at this node, including the node itself
        int size;

        AbstractNode(final String name) {
            Objects.requireNonNull(name, "name cannot be null");
            this.name = name;
            this.size = 1;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public Set<TopologyDescription.Node> predecessors() {
            return Collections.unmodifiableSet(predecessors);
        }

        @Override
        public Set<TopologyDescription.Node> successors() {
            return Collections.unmodifiableSet(successors);
        }

        public void addPredecessor(final TopologyDescription.Node predecessor) {
            predecessors.add(predecessor);
        }

        public void addSuccessor(final TopologyDescription.Node successor) {
            successors.add(successor);
        }
    }

    public static final class Source extends AbstractNode implements TopologyDescription.Source {
        private final Set<String> topics;
        private final Pattern topicPattern;

        public Source(final String name,
                      final Set<String> topics,
                      final Pattern pattern) {
            super(name);
            if (topics == null && pattern == null) {
                throw new IllegalArgumentException("Either topics or pattern must be not-null, but both are null.");
            }
            if (topics != null && pattern != null) {
                throw new IllegalArgumentException("Either topics or pattern must be null, but both are not null.");
            }

            this.topics = topics;
            this.topicPattern = pattern;
        }

        @Override
        public Set<String> topicSet() {
            return topics;
        }

        @Override
        public Pattern topicPattern() {
            return topicPattern;
        }

        @Override
        public void addPredecessor(final TopologyDescription.Node predecessor) {
            throw new UnsupportedOperationException("Sources don't have predecessors.");
        }

        @Override
        public String toString() {
            final String topicsString = topics == null ? topicPattern.toString() : topics.toString();

            return "Source: " + name + " (topics: " + topicsString + ")\n      --> " + nodeNames(successors);
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            final Source source = (Source) o;
            // omit successor to avoid infinite loops
            return name.equals(source.name)
                && Objects.equals(topics, source.topics)
                && (topicPattern == null ?
                        source.topicPattern == null :
                        topicPattern.pattern().equals(source.topicPattern.pattern()));
        }

        @Override
        public int hashCode() {
            // omit successor as it might change and alter the hash code
            return Objects.hash(name, topics, topicPattern);
        }
    }

    public static final class Processor extends AbstractNode implements TopologyDescription.Processor {
        private final Set<String> stores;

        public Processor(final String name,
                         final Set<String> stores) {
            super(name);
            this.stores = stores;
        }

        @Override
        public Set<String> stores() {
            return Collections.unmodifiableSet(stores);
        }

        @Override
        public String toString() {
            return "Processor: " + name + " (stores: " + stores + ")\n      --> "
                + nodeNames(successors) + "\n      <-- " + nodeNames(predecessors);
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            final Processor processor = (Processor) o;
            // omit successor to avoid infinite loops
            return name.equals(processor.name)
                && stores.equals(processor.stores)
                && predecessors.equals(processor.predecessors);
        }

        @Override
        public int hashCode() {
            // omit successor as it might change and alter the hash code
            return Objects.hash(name, stores);
        }
    }

    public static final class Sink<K, V> extends AbstractNode implements TopologyDescription.Sink {
        private final TopicNameExtractor<K, V> topicNameExtractor;
        public Sink(final String name,
                    final TopicNameExtractor<K, V> topicNameExtractor) {
            super(name);
            this.topicNameExtractor = topicNameExtractor;
        }

        public Sink(final String name,
                    final String topic) {
            super(name);
            this.topicNameExtractor = new StaticTopicNameExtractor<>(topic);
        }

        @Override
        public String topic() {
            if (topicNameExtractor instanceof StaticTopicNameExtractor) {
                return ((StaticTopicNameExtractor<K, V>) topicNameExtractor).topicName;
            } else {
                return null;
            }
        }

        @Override
        public TopicNameExtractor<K, V> topicNameExtractor() {
            if (topicNameExtractor instanceof StaticTopicNameExtractor) {
                return null;
            } else {
                return topicNameExtractor;
            }
        }

        @Override
        public void addSuccessor(final TopologyDescription.Node successor) {
            throw new UnsupportedOperationException("Sinks don't have successors.");
        }

        @Override
        public String toString() {
            if (topicNameExtractor instanceof StaticTopicNameExtractor) {
                return "Sink: " + name + " (topic: " + topic() + ")\n      <-- " + nodeNames(predecessors);
            }
            return "Sink: " + name + " (extractor class: " + topicNameExtractor + ")\n      <-- "
                + nodeNames(predecessors);
        }

        @SuppressWarnings("unchecked")
        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            final Sink<K, V> sink = (Sink<K, V>) o;
            return name.equals(sink.name)
                && topicNameExtractor.equals(sink.topicNameExtractor)
                && predecessors.equals(sink.predecessors);
        }

        @Override
        public int hashCode() {
            // omit predecessors as it might change and alter the hash code
            return Objects.hash(name, topicNameExtractor);
        }
    }

    public static final class SubtopologyDescription implements TopologyDescription.Subtopology {
        private final int id;
        private final Set<TopologyDescription.Node> nodes;

        public SubtopologyDescription(final int id, final Set<TopologyDescription.Node> nodes) {
            this.id = id;
            this.nodes = new TreeSet<>(NODE_COMPARATOR);
            this.nodes.addAll(nodes);
        }

        @Override
        public int id() {
            return id;
        }

        @Override
        public Set<TopologyDescription.Node> nodes() {
            return Collections.unmodifiableSet(nodes);
        }

        // visible for testing
        Iterator<TopologyDescription.Node> nodesInOrder() {
            return nodes.iterator();
        }

        @Override
        public String toString() {
            return "Sub-topology: " + id + "\n" + nodesAsString() + "\n";
        }

        private String nodesAsString() {
            final StringBuilder sb = new StringBuilder();
            for (final TopologyDescription.Node node : nodes) {
                sb.append("    ");
                sb.append(node);
                sb.append('\n');
            }
            return sb.toString();
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            final SubtopologyDescription that = (SubtopologyDescription) o;
            return id == that.id
                && nodes.equals(that.nodes);
        }

        @Override
        public int hashCode() {
            return Objects.hash(id, nodes);
        }
    }

    public static class TopicsInfo {
        public final Set<String> sinkTopics;
        public final Set<String> sourceTopics;
        public final Map<String, InternalTopicConfig> stateChangelogTopics;
        public final Map<String, InternalTopicConfig> repartitionSourceTopics;

        TopicsInfo(final Set<String> sinkTopics,
                   final Set<String> sourceTopics,
                   final Map<String, InternalTopicConfig> repartitionSourceTopics,
                   final Map<String, InternalTopicConfig> stateChangelogTopics) {
            this.sinkTopics = sinkTopics;
            this.sourceTopics = sourceTopics;
            this.stateChangelogTopics = stateChangelogTopics;
            this.repartitionSourceTopics = repartitionSourceTopics;
        }

        /**
         * Returns the config for any changelogs that must be prepared for this topic group, ie excluding any source
         * topics that are reused as a changelog
         */
        public Set<InternalTopicConfig> nonSourceChangelogTopics() {
            final Set<InternalTopicConfig> topicConfigs = new HashSet<>();
            for (final Map.Entry<String, InternalTopicConfig> changelogTopicEntry : stateChangelogTopics.entrySet()) {
                if (!sourceTopics.contains(changelogTopicEntry.getKey())) {
                    topicConfigs.add(changelogTopicEntry.getValue());
                }
            }
            return topicConfigs;
        }

        /**
         *
         * @return the set of changelog topics, which includes both source changelog topics and non
         * source changelog topics.
         */
        public Set<String> changelogTopics() {
            return Collections.unmodifiableSet(stateChangelogTopics.keySet());
        }

        /**
         * Returns the topic names for any optimized source changelogs
         */
        public Set<String> sourceTopicChangelogs() {
            return sourceTopics.stream().filter(stateChangelogTopics::containsKey).collect(Collectors.toSet());
        }

        @Override
        public boolean equals(final Object o) {
            if (o instanceof TopicsInfo) {
                final TopicsInfo other = (TopicsInfo) o;
                return other.sourceTopics.equals(sourceTopics) && other.stateChangelogTopics.equals(stateChangelogTopics);
            } else {
                return false;
            }
        }

        @Override
        public int hashCode() {
            final long n = ((long) sourceTopics.hashCode() << 32) | (long) stateChangelogTopics.hashCode();
            return (int) (n % 0xFFFFFFFFL);
        }

        @Override
        public String toString() {
            return "TopicsInfo{" +
                "sinkTopics=" + sinkTopics +
                ", sourceTopics=" + sourceTopics +
                ", repartitionSourceTopics=" + repartitionSourceTopics +
                ", stateChangelogTopics=" + stateChangelogTopics +
                '}';
        }
    }

    private static class GlobalStoreComparator implements Comparator<TopologyDescription.GlobalStore>, Serializable {
        @Override
        public int compare(final TopologyDescription.GlobalStore globalStore1,
                           final TopologyDescription.GlobalStore globalStore2) {
            if (globalStore1.equals(globalStore2)) {
                return 0;
            }
            return globalStore1.id() - globalStore2.id();
        }
    }

    private static final GlobalStoreComparator GLOBALSTORE_COMPARATOR = new GlobalStoreComparator();

    private static class SubtopologyComparator implements Comparator<TopologyDescription.Subtopology>, Serializable {
        @Override
        public int compare(final TopologyDescription.Subtopology subtopology1,
                           final TopologyDescription.Subtopology subtopology2) {
            if (subtopology1.equals(subtopology2)) {
                return 0;
            }
            return subtopology1.id() - subtopology2.id();
        }
    }

    private static final SubtopologyComparator SUBTOPOLOGY_COMPARATOR = new SubtopologyComparator();

    public static final class TopologyDescription implements org.apache.kafka.streams.TopologyDescription {
        private final TreeSet<Subtopology> subtopologies = new TreeSet<>(SUBTOPOLOGY_COMPARATOR);
        private final TreeSet<GlobalStore> globalStores = new TreeSet<>(GLOBALSTORE_COMPARATOR);
        private final String namedTopology;

        public TopologyDescription() {
            this(null);
        }

        public TopologyDescription(final String namedTopology) {
            this.namedTopology = namedTopology;
        }

        public void addSubtopology(final Subtopology subtopology) {
            subtopologies.add(subtopology);
        }

        public void addGlobalStore(final GlobalStore globalStore) {
            globalStores.add(globalStore);
        }

        @Override
        public Set<Subtopology> subtopologies() {
            return Collections.unmodifiableSet(subtopologies);
        }

        @Override
        public Set<GlobalStore> globalStores() {
            return Collections.unmodifiableSet(globalStores);
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();

            if (namedTopology == null) {
                sb.append("Topologies:\n ");
            } else {
                sb.append("Topology: ").append(namedTopology).append(":\n ");
            }
            final Subtopology[] sortedSubtopologies =
                subtopologies.descendingSet().toArray(new Subtopology[0]);
            final GlobalStore[] sortedGlobalStores =
                globalStores.descendingSet().toArray(new GlobalStore[0]);
            int expectedId = 0;
            int subtopologiesIndex = sortedSubtopologies.length - 1;
            int globalStoresIndex = sortedGlobalStores.length - 1;
            while (subtopologiesIndex != -1 && globalStoresIndex != -1) {
                sb.append("  ");
                final Subtopology subtopology = sortedSubtopologies[subtopologiesIndex];
                final GlobalStore globalStore = sortedGlobalStores[globalStoresIndex];
                if (subtopology.id() == expectedId) {
                    sb.append(subtopology);
                    subtopologiesIndex--;
                } else {
                    sb.append(globalStore);
                    globalStoresIndex--;
                }
                expectedId++;
            }
            while (subtopologiesIndex != -1) {
                final Subtopology subtopology = sortedSubtopologies[subtopologiesIndex];
                sb.append("  ");
                sb.append(subtopology);
                subtopologiesIndex--;
            }
            while (globalStoresIndex != -1) {
                final GlobalStore globalStore = sortedGlobalStores[globalStoresIndex];
                sb.append("  ");
                sb.append(globalStore);
                globalStoresIndex--;
            }
            return sb.toString();
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            final TopologyDescription that = (TopologyDescription) o;
            return subtopologies.equals(that.subtopologies)
                && globalStores.equals(that.globalStores);
        }

        @Override
        public int hashCode() {
            return Objects.hash(subtopologies, globalStores);
        }

    }

private void haltProcessors() {
    workerPool.shutdown();
    boolean controlled = getSettings().getBoolean(
        HadoopConfiguration.JobHistoryServer.JHS_RECOVERY_CONTROLLED,
        HadoopConfiguration.JobHistoryServer.DEFAULT_JHS_RECOVERY_CONTROLLED);
    // if recovery on restart is supported then leave outstanding processes
    // to the next start
    boolean needToStop = context.getJobStateStore().canResume()
        && !context.getDecommissioned() && controlled;
    // kindly request to end
    for (JobProcessor processor : jobProcessors.values()) {
      if (needToStop) {
        processor.terminateProcessing();
      } else {
        processor.completeProcessing();
      }
    }
    while (!workerPool.isTerminated()) { // wait for all workers to finish
      for (JobId jobId : jobProcessors.keySet()) {
        LOG.info("Waiting for processing to complete for " + jobId);
      }
      try {
        if (!workerPool.awaitTermination(30, TimeUnit.SECONDS)) {
          workerPool.shutdownNow(); // send interrupt to hasten them along
        }
      } catch (InterruptedException e) {
        LOG.warn("Processing halt interrupted!");
        break;
      }
    }
    for (JobId jobId : jobProcessors.keySet()) {
      LOG.warn("Some data may not have been processed for " + jobId);
    }
  }

  private static List<Node> coreResolve(List<String> hostNames) {
    List<Node> nodes = new ArrayList<Node>(hostNames.size());
    List<String> rNameList = dnsToSwitchMapping.resolve(hostNames);
    if (rNameList == null || rNameList.isEmpty()) {
      for (String hostName : hostNames) {
        nodes.add(new NodeBase(hostName, NetworkTopology.DEFAULT_RACK));
      }
      LOG.info("Got an error when resolve hostNames. Falling back to "
          + NetworkTopology.DEFAULT_RACK + " for all.");
    } else {
      for (int i = 0; i < hostNames.size(); i++) {
        if (Strings.isNullOrEmpty(rNameList.get(i))) {
          // fallback to use default rack
          nodes.add(new NodeBase(hostNames.get(i),
              NetworkTopology.DEFAULT_RACK));
          LOG.debug("Could not resolve {}. Falling back to {}",
              hostNames.get(i), NetworkTopology.DEFAULT_RACK);
        } else {
          nodes.add(new NodeBase(hostNames.get(i), rNameList.get(i)));
          LOG.debug("Resolved {} to {}", hostNames.get(i), rNameList.get(i));
        }
      }
    }
    return nodes;
  }

private List<ResourceRequest> createResourceRequests() throws IOException {
    Resource capability = recordFactory.newRecordInstance(Resource.class);
    boolean memorySet = false;
    boolean cpuVcoresSet = false;

    List<ResourceInformation> resourceRequests = ResourceUtils.getRequestedResourcesFromConfig(conf, MR_AM_RESOURCE_PREFIX);
    for (ResourceInformation resourceReq : resourceRequests) {
      String resourceName = resourceReq.getName();

      if (MRJobConfig.RESOURCE_TYPE_NAME_MEMORY.equals(resourceName) ||
          MRJobConfig.RESOURCE_TYPE_ALTERNATIVE_NAME_MEMORY.equals(resourceName)) {
        if (memorySet) {
          throw new IllegalArgumentException(
              "Only one of the following keys can be specified for a single job: " +
              MRJobConfig.RESOURCE_TYPE_NAME_MEMORY + ", " +
              MRJobConfig.RESOURCE_TYPE_ALTERNATIVE_NAME_MEMORY);
        }

        String units = isEmpty(resourceReq.getUnits()) ?
            ResourceUtils.getDefaultUnit(ResourceInformation.MEMORY_URI) :
            resourceReq.getUnits();
        capability.setMemorySize(
            UnitsConversionUtil.convert(units, "Mi", resourceReq.getValue()));
        memorySet = true;

        if (conf.get(MRJobConfig.MR_AM_VMEM_MB) != null) {
          LOG.warn("Configuration " + MR_AM_RESOURCE_PREFIX +
              resourceName + "=" + resourceReq.getValue() +
              resourceReq.getUnits() + " is overriding the " +
              MRJobConfig.MR_AM_VMEM_MB + "=" +
              conf.get(MRJobConfig.MR_AM_VMEM_MB));
        }
      } else if (MRJobConfig.RESOURCE_TYPE_NAME_VCORE.equals(resourceName)) {
        capability.setVirtualCores(
            (int) UnitsConversionUtil.convert(resourceReq.getUnits(), "", resourceReq.getValue()));
        cpuVcoresSet = true;

        if (conf.get(MRJobConfig.MR_AM_CPU_VCORES) != null) {
          LOG.warn("Configuration " + MR_AM_RESOURCE_PREFIX +
              resourceName + "=" + resourceReq.getValue() +
              resourceReq.getUnits() + " is overriding the " +
              MRJobConfig.MR_AM_CPU_VCORES + "=" +
              conf.get(MRJobConfig.MR_AM_CPU_VCORES));
        }
      } else if (!MRJobConfig.MR_AM_VMEM_MB.equals(MR_AM_RESOURCE_PREFIX + resourceName) &&
          !MRJobConfig.MR_AM_CPU_VCORES.equals(MR_AM_RESOURCE_PREFIX + resourceName)) {

        ResourceInformation resourceInformation = capability.getResourceInformation(resourceName);
        resourceInformation.setUnits(resourceReq.getUnits());
        resourceInformation.setValue(resourceReq.getValue());
        capability.setResourceInformation(resourceName, resourceInformation);
      }
    }

    if (!memorySet) {
      capability.setMemorySize(
          conf.getInt(MRJobConfig.MR_AM_VMEM_MB, MRJobConfig.DEFAULT_MR_AM_VMEM_MB));
    }

    if (!cpuVcoresSet) {
      capability.setVirtualCores(
          conf.getInt(MRJobConfig.MR_AM_CPU_VCORES, MRJobConfig.DEFAULT_MR_AM_CPU_VCORES));
    }

    if (LOG.isDebugEnabled()) {
      LOG.debug("AppMaster capability = " + capability);
    }

    List<ResourceRequest> amResourceRequests = new ArrayList<>();

    ResourceRequest amAnyResourceRequest =
        createAMResourceRequest(ResourceRequest.ANY, capability);
    amResourceRequests.add(amAnyResourceRequest);

    Map<String, ResourceRequest> rackRequests = new HashMap<>();
    Collection<String> invalidResources = new HashSet<>();

    for (ResourceInformation resourceReq : resourceRequests) {
      String resourceName = resourceReq.getName();

      if (!MRJobConfig.RESOURCE_TYPE_NAME_MEMORY.equals(resourceName) &&
          !MRJobConfig.RESOURCE_TYPE_ALTERNATIVE_NAME_MEMORY.equals(resourceName)) {

        if (!MRJobConfig.MR_AM_VMEM_MB.equals(MR_AM_RESOURCE_PREFIX + resourceName) &&
            !MRJobConfig.MR_AM_CPU_VCORES.equals(MR_AM_RESOURCE_PREFIX + resourceName)) {

          ResourceInformation resourceInformation = capability.getResourceInformation(resourceName);
          resourceInformation.setUnits(resourceReq.getUnits());
          resourceInformation.setValue(resourceReq.getValue());
          capability.setResourceInformation(resourceName, resourceInformation);

          if (!rackRequests.containsKey(resourceName)) {
            ResourceRequest amNodeResourceRequest =
                createAMResourceRequest(resourceName, capability);
            amResourceRequests.add(amNodeResourceRequest);
            rackRequests.put(resourceName, amNodeResourceRequest);
          }
        } else {
          invalidResources.add(resourceName);
        }
      }
    }

    if (!invalidResources.isEmpty()) {
      String errMsg = "Invalid resource names: " + invalidResources.toString() + " specified.";
      LOG.warn(errMsg);
      throw new IOException(errMsg);
    }

    for (ResourceRequest amResourceRequest : amResourceRequests) {
      LOG.debug("ResourceRequest: resource = " +
          amResourceRequest.getResourceName() + ", locality = " +
          amResourceRequest.getRelaxLocality());
    }

    return amResourceRequests;
  }

	protected boolean tryToExcludeFromRunner(Description description) {
		// @formatter:off
		return getParent().map(VintageTestDescriptor.class::cast)
				.map(parent -> parent.tryToExcludeFromRunner(description))
				.orElse(false);
		// @formatter:on
	}

public boolean isEqual(Response other) {
    if (other == null || !(other instanceof Response)) {
      return false;
    }

    Response response = (Response) other;
    boolean valueEquals = Objects.equals(this.value, response.getValue());
    boolean sessionIdEquals = Objects.equals(this.sessionId, response.getSessionId());
    boolean statusEquals = Objects.equals(this.status, response.getStatus());
    boolean stateEquals = Objects.equals(this.state, response.getState());

    return valueEquals && sessionIdEquals && statusEquals && stateEquals;
}

	public boolean checkResource(Locale locale) throws Exception {
		String url = getUrl();
		Assert.state(url != null, "'url' not set");

		try {
			// Check that we can get the template, even if we might subsequently get it again.
			getTemplate(url, locale);
			return true;
		}
		catch (FileNotFoundException ex) {
			// Allow for ViewResolver chaining...
			return false;
		}
		catch (ParseException ex) {
			throw new ApplicationContextException("Failed to parse [" + url + "]", ex);
		}
		catch (IOException ex) {
			throw new ApplicationContextException("Failed to load [" + url + "]", ex);
		}
	}

    /**
     * @return a copy of the string representation of any pattern subscribed source nodes
     */
  private void onResourcesReclaimed(Container container) {
    oppContainersToKill.remove(container.getContainerId());

    // This could be killed externally for eg. by the ContainerManager,
    // in which case, the container might still be queued.
    Container queued =
        queuedOpportunisticContainers.remove(container.getContainerId());
    if (queued == null) {
      queuedGuaranteedContainers.remove(container.getContainerId());
    }

    // Requeue PAUSED containers
    if (container.getContainerState() == ContainerState.PAUSED) {
      if (container.getContainerTokenIdentifier().getExecutionType() ==
          ExecutionType.GUARANTEED) {
        queuedGuaranteedContainers.put(container.getContainerId(), container);
      } else {
        queuedOpportunisticContainers.put(
            container.getContainerId(), container);
      }
    }
    // decrement only if it was a running container
    Container completedContainer = runningContainers.remove(container
        .getContainerId());
    // only a running container releases resources upon completion
    boolean resourceReleased = completedContainer != null;
    if (resourceReleased) {
      this.utilizationTracker.subtractContainerResource(container);
      if (container.getContainerTokenIdentifier().getExecutionType() ==
          ExecutionType.OPPORTUNISTIC) {
        this.metrics.completeOpportunisticContainer(container.getResource());
      }
      startPendingContainers(forceStartGuaranteedContainers);
    }
    this.metrics.setQueuedContainers(queuedOpportunisticContainers.size(),
        queuedGuaranteedContainers.size());
  }

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

public static FsAction findFsActionBySymbol(String actionPermission) {
    for (FsAction currentAction : vals) {
      if (!currentAction.SYMBOL.equals(actionPermission)) {
        continue;
      }
      return currentAction;
    }
    return null;
}

    public <KIn, VIn, VOut> WrappedFixedKeyProcessorSupplier<KIn, VIn, VOut> wrapFixedKeyProcessorSupplier(
        final String name,
        final FixedKeyProcessorSupplier<KIn, VIn,  VOut> processorSupplier
    ) {
        return ProcessorWrapper.asWrappedFixedKey(
            processorWrapper.wrapFixedKeyProcessorSupplier(name, processorSupplier)
        );
    }

    public <KIn, VIn, KOut, VOut> WrappedProcessorSupplier<KIn, VIn, KOut, VOut> wrapProcessorSupplier(
        final String name,
        final ProcessorSupplier<KIn, VIn, KOut, VOut> processorSupplier
    ) {
        return ProcessorWrapper.asWrapped(
            processorWrapper.wrapProcessorSupplier(name, processorSupplier)
        );
    }
}
