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

protected FreeMarkerConfig detectFreeMarkerConfiguration() throws BeansException {
		try {
			var freeMarkerConfig = BeanFactoryUtils.beanOfTypeIncludingAncestors(
					this.obtainApplicationContext(), FreeMarkerConfig.class, true, false);
			return freeMarkerConfig;
		}
		catch (NoSuchBeanDefinitionException ex) {
			throw new ApplicationContextException(
					"Must define a single FreeMarkerConfig bean in this web application context " +
					"(may be inherited): FreeMarkerConfigurer is the usual implementation. " +
					"This bean may be given any name.", ex);
		}
	}

