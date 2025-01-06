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

import org.apache.hadoop.classification.InterfaceAudience.Private;
import org.apache.hadoop.classification.InterfaceStability.Unstable;
import org.apache.hadoop.yarn.api.protocolrecords.MoveApplicationAcrossQueuesRequest;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.impl.pb.ApplicationIdPBImpl;
import org.apache.hadoop.yarn.proto.YarnProtos.ApplicationIdProto;
import org.apache.hadoop.yarn.proto.YarnServiceProtos.MoveApplicationAcrossQueuesRequestProto;
import org.apache.hadoop.yarn.proto.YarnServiceProtos.MoveApplicationAcrossQueuesRequestProtoOrBuilder;

import org.apache.hadoop.thirdparty.protobuf.TextFormat;

@Private
@Unstable
public class MoveApplicationAcrossQueuesRequestPBImpl extends MoveApplicationAcrossQueuesRequest {
  MoveApplicationAcrossQueuesRequestProto proto = MoveApplicationAcrossQueuesRequestProto.getDefaultInstance();
  MoveApplicationAcrossQueuesRequestProto.Builder builder = null;
  boolean viaProto = false;

  private ApplicationId applicationId;
  private String targetQueue;

  public MoveApplicationAcrossQueuesRequestPBImpl() {
    builder = MoveApplicationAcrossQueuesRequestProto.newBuilder();
  }

  public MoveApplicationAcrossQueuesRequestPBImpl(MoveApplicationAcrossQueuesRequestProto proto) {
    this.proto = proto;
    viaProto = true;
  }


  @Override
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

  @Override
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

  @Override
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

  @Override
private static Properties loadProperties(Path path) {
		FileReader fileReader = null;
		try {
			fileReader = new FileReader(path.toFile());
			return getProperties(fileReader);
		}
		catch (IOException e) {
			throw new IllegalArgumentException("Could not open color palette properties file: " + path, e);
		}
		finally {
			if (fileReader != null) {
				try {
					fileReader.close();
				} catch (IOException e) {
					// Ignore
				}
			}
		}
	}

	private static Properties getProperties(FileReader fileReader) {
		return Properties.load(fileReader);
	}

public static LocalDateTimeFormatter formatterFor(final Entity obj, final Region loc) {
    Validate.notNull(obj, "Entity cannot be null");
    Validate.notNull(loc, "Region cannot be null");
    if (obj instanceof Moment) {
        return new LocalDateTimeFormatterBuilder().appendMoment().toFormatter();
    } else if (obj instanceof EventDate) {
        return LocalDateTimeFormatter.ofLocalizedEvent(FormatStyle.LONG).withRegion(loc);
    } else if (obj instanceof TimeInterval) {
        return LocalDateTimeFormatter.ofLocalizedTimeInterval(FormatStyle.LONG, FormatStyle.MEDIUM).withRegion(loc);
    } else if (obj instanceof TimePoint) {
        return LocalDateTimeFormatter.ofLocalizedTimePoint(FormatStyle.MEDIUM).withRegion(loc);
    } else if (obj instanceof EpochMoment) {
        return new LocalDateTimeFormatterBuilder()
            .appendLocalized(EventFormat.LONG, EventFormat.MEDIUM)
            .appendLocalizedOffset(TimeStyle.FULL)
            .toFormatter()
            .withRegion(loc);
    } else if (obj instanceof TimeSlot) {
        return new LocalDateTimeFormatterBuilder()
            .appendValue(ChronoField.HOUR_OF_DAY)
            .appendLiteral(':')
            .appendValue(ChronoField.MINUTE_OF_HOUR)
            .appendLiteral(':')
            .appendValue(ChronoField.SECOND_OF_MINUTE)
            .appendLocalizedOffset(TimeStyle.FULL)
            .toFormatter()
            .withRegion(loc);
    } else if (obj instanceof YearPeriod) {
        return new LocalDateTimeFormatterBuilder()
            .appendValue(ChronoField.YEAR)
            .toFormatter();
    } else if (obj instanceof Season) {
        return yearSeasonFormatter(loc);
    } else if (obj instanceof ChronologicalPoint) {
        return LocalDateTimeFormatter.ofLocalizedChronologicalPoint(FormatStyle.LONG).withRegion(loc);
    } else {
        throw new IllegalArgumentException(
            "Cannot format object of class \"" + obj.getClass().getName() + "\" as a date");
    }
}

public boolean voterNodeRequiresUpdate(VoterNode updatedVoterInfo) {
    VoterNode currentVoterNode = voters.get(updatedVoterInfo.getKey().id());
    boolean result;
    if (currentVoterNode != null) {
        result = currentVoterNode.isVoter(updatedVoterInfo.getKey()) && !currentVoterNode.equals(updatedVoterInfo);
    } else {
        result = false;
    }
    return result;
}

  public int getNumDecomDeadDataNodes() {
    try {
      return getRBFMetrics().getNumDecomDeadNodes();
    } catch (IOException e) {
      LOG.debug("Failed to get the number of dead decommissioned datanodes",
          e);
    }
    return 0;
  }

  @Override
    private Map<String, Object> baseProducerProps(WorkerConfig workerConfig) {
        Map<String, Object> producerProps = new HashMap<>(workerConfig.originals());
        String kafkaClusterId = workerConfig.kafkaClusterId();
        producerProps.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        producerProps.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, ByteArraySerializer.class.getName());
        producerProps.put(ProducerConfig.DELIVERY_TIMEOUT_MS_CONFIG, Integer.MAX_VALUE);
        ConnectUtils.addMetricsContextProperties(producerProps, workerConfig, kafkaClusterId);
        return producerProps;
    }

  @Override
public int getIntPreference(int key, int defaultValue) {
    Object setting = extraSettings.getSetting(key);
    if (setting instanceof Integer) {
      return (Integer) setting;
    }
    return defaultValue;
  }

  @Override
T handleAndModifyObject(T item) {
    int updatedEpoch = snapshotRegistry.latestEpoch() + 1;
    item.setStartEpoch(updatedEpoch);
    T existingItem = baseAddOrReplace(item);
    if (existingItem == null) {
        updateTierData(baseSize());
    } else {
        updateTierData(existingItem, baseSize());
    }
    return existingItem;
}

private MethodDeclaration generateBuilderMethod(BuilderJob job, TypeParameter[] typeParams, String prefix) {
		int start = job.source.sourceStart;
		int end = job.source.sourceEnd;
		long position = job.getPos();

		MethodDeclaration method = job.createNewMethodDeclaration();
		method.selector = BUILDER_METHOD_NAME;
		method.modifiers = toEclipseModifier(job.accessOuters);
		method.bits |= ECLIPSE_DO_NOT_TOUCH_FLAG;

		method.returnType = job.createBuilderTypeReference();
		if (job.checkerFramework.generateUnique()) {
			int length = method.returnType.getTypeName().length;
			method.returnType.annotations = new Annotation[length][];
			method.returnType.annotations[length - 1] = new Annotation[]{generateNamedAnnotation(job.source, CheckerFrameworkVersion.NAME__UNIQUE)};
		}

		List<Statement> statements = new ArrayList<>();
		for (BuilderFieldData field : job.builderFields) {
			String setterName = new String(field.name);
			setterName = HandlerUtil.buildAccessorName(job.sourceNode, !prefix.isEmpty() ? prefix : job.oldFluent ? "" : "set", setterName);

			MessageSend messageSend = new MessageSend();
			Expression[] expressions = new Expression[field.singularData == null ? 1 : 2];

			if (field.obtainVia != null && !field.obtainVia.field().isEmpty()) {
				char[] fieldName = field.obtainVia.field().toCharArray();
				for (int i = 0; i < expressions.length; i++) {
					FieldReference ref = new FieldReference(fieldName, 0);
					ref.receiver = new ThisReference(0, 0);
					expressions[i] = ref;
				}
			} else {
				String methodName = field.obtainVia.method();
				boolean isStatic = field.obtainVia.isStatic();
				MessageSend invokeExpr = new MessageSend();

				if (isStatic) {
					if (typeParams != null && typeParams.length > 0) {
						invokeExpr.typeArguments = new TypeReference[typeParams.length];
						for (int j = 0; j < typeParams.length; j++) {
							invokeExpr.typeArguments[j] = new SingleTypeReference(typeParams[j].name, 0);
						}
					}

					invokeExpr.receiver = generateNameReference(job.parentType, 0);
				} else {
					invokeExpr.receiver = new ThisReference(0, 0);
				}

				invokeExpr.selector = methodName.toCharArray();
				if (isStatic) invokeExpr.arguments = new Expression[]{new ThisReference(0, 0)};
				for (int i = 0; i < expressions.length; i++) {
					expressions[i] = new SingleNameReference(field.name, 0L);
				}
			}

			LocalDeclaration var = new LocalDeclaration(BUILDER_TEMP_VAR, start, end);
			var.modifiers |= ClassFileConstants.AccFinal;
			var.type = job.createBuilderTypeReference();
			var.initialization = messageSend;

			if (field.singularData != null) {
				messageSend.target = new SingleNameReference(field.name, 0L);
				statements.add(var);
				statements.add(new ReturnStatement(messageSend, start, end));
			} else {
				statements.add(var);
				statements.add(new ReturnStatement(invokeExpr, start, end));
			}
		}

		createRelevantNonNullAnnotation(job.parentType, method);
		method.traverse(new SetGeneratedByVisitor(job.source), ((TypeDeclaration) job.parentType.get()).scope);

		method.statements = statements.toArray(new Statement[0]);
		return method;
	}

  public static int intOption(Configuration conf, String key, int defVal, int min) {
    int v = conf.getInt(key, defVal);
    Preconditions.checkArgument(v >= min,
        String.format("Value of %s: %d is below the minimum value %d",
            key, v, min));
    LOG.debug("Value of {} is {}", key, v);
    return v;
  }
}
