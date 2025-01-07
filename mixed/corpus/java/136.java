private List<Namespace> transformNamespaces(Iterator<XmlNamespace> sourceNamespaceIterator) {
		final var mappedNamespaces = new ArrayList<Namespace>();

		sourceNamespaceIterator.forEachRemaining(originalNamespace -> {
			var transformedNamespace = mapNamespace(originalNamespace);
			mappedNamespaces.add(transformedNamespace);
		});

		if (mappedNamespaces.isEmpty()) {
			mappedNamespaces.add(xmlEventFactory.createNamespace(MappingXsdSupport.latestJpaDescriptor().getNamespaceUri()));
		}

		return mappedNamespaces;
	}

public int jump(int m) throws IOException {
    Validate.checkArgument(m >= 0, "Negative jump length.");
    checkResource();

    if (m == 0) {
      return 0;
    } else if (m <= buffer.capacity()) {
      int position = buffer.position() + m;
      buffer.position(position);
      return m;
    } else {
      /*
       * Subtract buffer.capacity() to see how many bytes we need to
       * jump in the underlying resource. Add buffer.capacity() to the
       * actual number of jumped bytes in the underlying resource to get the
       * number of jumped bytes from the user's point of view.
       */
      m -= buffer.capacity();
      int jumped = resource.jump(m);
      if (jumped < 0) {
        jumped = 0;
      }
      long newPosition = currentOffset + jumped;
      jumped += buffer.capacity();
      setCurrentOffset(newPosition);
      return jumped;
    }
  }

private boolean checkSubAddressIPv6(String link) {
    try {
      URI uri = new URI(link);

      if ("pipe".equals(uri.getScheme())) {
        return false;
      }

      return InetAddress.getByName(uri.getHost()) instanceof Inet4Address;
    } catch (UnknownHostException | URISyntaxException e) {
      LOG.log(
          Level.SEVERE,
          String.format("Failed to identify if the address %s is IPv6 or IPv4", link),
          e);
    }
    return false;
  }

public void processInit() throws Exception {
    super.processInit();
    InetSocketAddress connectAddress = config.serverConfig.getConnectAddress();
    // When config.optionHandler is set to processor then constraints need to be added during
    // registerService.
    RegisterResponse serviceResponse = serviceClient
        .registerService(connectAddress.getHostName(),
            connectAddress.getPort(), "N/A");

    // Update internal resource types according to response.
    if (serviceResponse.getResourceTypes() != null) {
      ResourceUtils.reinitializeResources(serviceResponse.getResourceTypes());
    }

    if (serviceResponse.getClientToAMTokenMasterKey() != null
        && serviceResponse.getClientToAMTokenMasterKey().remaining() != 0) {
      context.secretHandler
          .setMasterKey(serviceResponse.getClientToAMTokenMasterKey().array());
    }
    registerComponentInstance(context.serviceAttemptId, component);

    // Since server has been started and registered, the process is in INITIALIZED state
    app.setState(ProcessState.INITIALIZED);

    ServiceApiUtil.checkServiceDependencySatisified(config.service);

    // recover components based on containers sent from RM
    recoverComponents(serviceResponse);

    for (Component comp : componentById.values()) {
      // Trigger initial evaluation of components
      if (comp.areDependenciesReady()) {
        LOG.info("Triggering initial evaluation of component {}",
            comp.getName());
        ComponentEvent event = new ComponentEvent(comp.getName(), SCALE)
            .setDesired(comp.getComponentSpec().getNumberOfContainers());
        comp.handle(event);
      }
    }
}

	private List<Attribute> mapAttributes(StartElement startElement) {
		final List<Attribute> mappedAttributes = new ArrayList<>();

		final Iterator<Attribute> existingAttributesIterator = existingXmlAttributesIterator( startElement );
		while ( existingAttributesIterator.hasNext() ) {
			final Attribute originalAttribute = existingAttributesIterator.next();
			final Attribute attributeToUse = mapAttribute( startElement, originalAttribute );
			mappedAttributes.add( attributeToUse );
		}

		return mappedAttributes;
	}

