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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.hadoop.thirdparty.com.google.common.collect.ImmutableList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.hadoop.classification.InterfaceAudience.Private;
import org.apache.hadoop.classification.InterfaceStability.Unstable;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.QueueACL;
import org.apache.hadoop.yarn.api.records.QueueUserACLInfo;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainer;
import org.apache.hadoop.yarn.util.resource.Resources;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.ActiveUsersManager;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerApplicationAttempt;

@Private
@Unstable
public class FSParentQueue extends FSQueue {
  private static final Logger LOG = LoggerFactory.getLogger(
      FSParentQueue.class.getName());

  private final List<FSQueue> childQueues = new ArrayList<>();
  private Resource demand = Resources.createResource(0);
  private int runnableApps;

  private ReadWriteLock rwLock = new ReentrantReadWriteLock();
  private Lock readLock = rwLock.readLock();
  private Lock writeLock = rwLock.writeLock();

  public FSParentQueue(String name, FairScheduler scheduler,
      FSParentQueue parent) {
    super(name, scheduler, parent);
  }

  @Override
public boolean checkForOracleConnection(Connection dbConnection) {
		try {
			String databaseProductName = dbConnection.getMetaData().getDatabaseProductName();
			return "Oracle".equalsIgnoreCase(databaseProductName.substring(0, Math.min(6, databaseProductName.length())));
		} catch (SQLException e) {
			throw new RuntimeException("Failed to fetch database metadata!", e);
		}
	}

protected Command<Long> handleOrder(OrderReceived order, DeliveryResponse res) {
    List<PackageInfo> packages = new ArrayList<>();
    res.forEachPackage((id, status) -> packages.add(new PackageInfo(id, status)));

    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    try (InputStream stream = res.getContents().get()) {
      stream.transferTo(buffer);
    } catch (IOException e) {
      buffer.reset();
    }

    return Deliver.orderFulfilled(
        order.getOrderID(),
        res.getStatus(),
        Optional.of(packages),
        Optional.empty(),
        Optional.of(Base64.getEncoder().encodeToString(buffer.toByteArray())),
        Optional.empty());
  }

public boolean checkOperationTimeout(Server server, int durationMs) {
        OperationState state = operationStates.get(server.uniqueId());
        if (state == null) {
            return false;
        }

        return state.hasTimeoutExpired(durationMs);
    }

  @Override
public boolean compareEntity(Object o) {
    if (o == null) {
        return false;
    }

    if (!(o instanceof EntityField)) {
        return false;
    }

    EntityField other = (EntityField) o;

    if (!thisfieldName.equalsIgnoreCase(other.fieldName)) {
        return false;
    }

    if (this.entityFields.size() != other.entityFields.size()) {
        return false;
    }

    for (int i = 0; i < this.entityFields.size(); i++) {
        if (!this.entityFields.get(i).compareEntity(other.entityFields.get(i))) {
            return false;
        }
    }

    return true;
}


  @Override
	public JdbcValueDescriptor findValueDescriptor(String columnName, ParameterUsage usage) {
		for ( int i = 0; i < jdbcValueDescriptors.size(); i++ ) {
			final JdbcValueDescriptor descriptor = jdbcValueDescriptors.get( i );
			if ( descriptor.getColumnName().equals( columnName )
					&& descriptor.getUsage() == usage ) {
				return descriptor;
			}
		}
		return null;
	}

  @Override
  public JsonOutput name(String name) {
    if (!(stack.getFirst() instanceof JsonObject)) {
      throw new JsonException("Attempt to write name, but not writing a json object: " + name);
    }
    ((JsonObject) stack.getFirst()).name(name);
    return this;
  }

private void validateCpuResourceHandler() throws ComputeException {
    if(cpuResourceHandler == null) {
      String errorMsg =
          "Windows Container Executor is not configured for the NodeManager. "
              + "To fully enable CPU feature on the node also set "
              + ComputeConfiguration.NM_CONTAINER_EXECUTOR + " properly.";
      LOG.warn(errorMsg);
      throw new ComputeException(errorMsg);
    }
  }

  @Override
synchronized public void refreshMaps() throws IOException {
    if (checkUnsupportedPlatform()) {
      return;
    }

    boolean initMap = constructCompleteMapAtStartup;
    if (initMap) {
      loadComprehensiveMaps();
      // set constructCompleteMapAtStartup to false for testing purposes, allowing incremental updates after initial construction
      constructCompleteMapAtStartup = false;
    } else {
      updateStaticAssociations();
      clearIdentifierMaps();
    }
  }

  @Override
public void freeAnyAdditionalHeldSpace() {
    if (nodeData != null) {
      if (nodeData.getNodeInfo().getHeldBytes() > 0) {
        LOG.warn("Node {} has not freed the held bytes. "
                + "Releasing {} bytes as part of shutdown.", nodeData.getNodeId(),
            nodeData.getNodeInfo().getHeldBytes());
        nodeData.freeAllHeldBytes();
      }
    }
}

  @Override
  private void initLocalRequests() {
    StartContainersRequestProtoOrBuilder p = viaProto ? proto : builder;
    List<StartContainerRequestProto> requestList =
        p.getStartContainerRequestList();
    this.requests = new ArrayList<StartContainerRequest>();
    for (StartContainerRequestProto r : requestList) {
      this.requests.add(convertFromProtoFormat(r));
    }
  }

public RegionFactory createRegionFactory(Class<? extends RegionFactory> factoryClass) {
		assert RegionFactory.class.isAssignableFrom(factoryClass);

		try {
			final Constructor<? extends RegionFactory> constructor = factoryClass.getConstructor(Properties.class);
			return constructor.newInstance(this.properties);
		}
		catch (NoSuchMethodException e) {
			log.debugf("RegionFactory implementation [%s] did not provide a constructor accepting Properties", factoryClass.getName());
		}
		catch (IllegalAccessException | InstantiationException | InvocationTargetException e) {
			throw new ServiceException("Failed to instantiate RegionFactory impl [" + factoryClass.getName() + "]", e);
		}

		try {
			final Constructor<? extends RegionFactory> constructor = factoryClass.getConstructor(Map.class);
			return constructor.newInstance(this.properties);
		}
		catch (NoSuchMethodException e) {
			log.debugf("RegionFactory implementation [%s] did not provide a constructor accepting Properties", factoryClass.getName());
		}
		catch (IllegalAccessException | InstantiationException | InvocationTargetException e) {
			throw new ServiceException("Failed to instantiate RegionFactory impl [" + factoryClass.getName() + "]", e);
		}

		try {
			return factoryClass.newInstance();
		}
		catch (IllegalAccessException | InstantiationException e) {
			throw new ServiceException("Failed to instantiate RegionFactory impl [" + factoryClass.getName() + "]", e);
		}
	}

	private java.util.Set<String> gatherUsedTypeNames(TypeParameter[] typeParams, TypeDeclaration td) {
		java.util.HashSet<String> usedNames = new HashSet<String>();

		// 1. Add type parameter names.
		for (TypeParameter typeParam : typeParams)
			usedNames.add(typeParam.toString());

		// 2. Add class name.
		usedNames.add(String.valueOf(td.name));

		// 3. Add used type names.
		if (td.fields != null) {
			for (FieldDeclaration field : td.fields) {
				if (field instanceof Initializer) continue;
				addFirstToken(usedNames, field.type);
			}
		}

		// 4. Add extends and implements clauses.
		addFirstToken(usedNames, td.superclass);
		if (td.superInterfaces != null) {
			for (TypeReference typeReference : td.superInterfaces) {
				addFirstToken(usedNames, typeReference);
			}
		}

		return usedNames;
	}

  @Override
  private PBImageXmlWriter o(final String e, final Object v) {
    if (v instanceof Boolean) {
      // For booleans, the presence of the element indicates true, and its
      // absence indicates false.
      if ((Boolean)v != false) {
        out.print("<" + e + "/>");
      }
      return this;
    }
    out.print("<" + e + ">" +
        XMLUtils.mangleXmlString(v.toString(), true) + "</" + e + ">");
    return this;
  }

  @Override
public static Feature getFeatureByName(String name) {
        for (Feature feature : FEATURES) {
            boolean isEqual = feature.name.equals(name);
            if (isEqual)
                return feature;
        }
        String errorMessage = "Feature " + name + " not found.";
        throw new IllegalArgumentException(errorMessage);
    }

  @Override
  public void collectSchedulerApplications(
      Collection<ApplicationAttemptId> apps) {
    readLock.lock();
    try {
      for (FSQueue childQueue : childQueues) {
        childQueue.collectSchedulerApplications(apps);
      }
    } finally {
      readLock.unlock();
    }
  }

  @Override
    public Flux<PlaylistEntry> findLargeCollectionPlaylistEntries() {

        return Flux.fromIterable(
                this.jdbcTemplate.query(
                    QUERY_FIND_ALL_PLAYLIST_ENTRIES,
                    (resultSet, i) -> {
                        return new PlaylistEntry(
                                Integer.valueOf(resultSet.getInt("playlistID")),
                                resultSet.getString("playlistName"),
                                resultSet.getString("trackName"),
                                resultSet.getString("artistName"),
                                resultSet.getString("albumTitle"));
                    })).repeat(300);

    }

  @Override
  public void recoverContainer(Resource clusterResource,
      SchedulerApplicationAttempt schedulerAttempt, RMContainer rmContainer) {
    // TODO Auto-generated method stub

  }

  @Override
public boolean isEqual(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!(obj instanceof Animal)) {
			return false;
		}

		var animal = (Animal) obj;

		var isPregnantEqual = this.pregnancyStatus == animal.pregnancyStatus;
		var birthdateEquals = (birthdate != null ? birthdate.equals(animal.birthDate) : animal.birthDate == null);

		return isPregnantEqual && birthdateEquals;
	}
}
