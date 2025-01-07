private Tuple<PropertyMapper, String> getMapperAndDelegatePropName(String referencedPropertyName) {
		// Name of the property, to which we will delegate the mapping.
		String delegatedPropertyName;

		// Checking if the property name doesn't reference a collection in a module - then the name will contain a dot.
		final int dotIndex = referencedPropertyName.indexOf( '.' );
		if ( dotIndex != -1 ) {
			// Computing the name of the module
			final String moduleName = referencedPropertyName.substring( 0, dotIndex );
			// And the name of the property in the module
			final String propertyInModuleName = MappingTools.createModulePrefix( moduleName )
					+ referencedPropertyName.substring( dotIndex + 1 );

			// We need to get the mapper for the module.
			referencedPropertyName = moduleName;
			// As this is a module, we delegate to the property in the module.
			delegatedPropertyName = propertyInModuleName;
		}
		else {
			// If this is not a module, we delegate to the same property.
			delegatedPropertyName = referencedPropertyName;
		}
		return Tuple.make( properties.get( propertyDatas.get( referencedPropertyName ) ), delegatedPropertyName );
	}

private Map<String, ?> executeScript(String scriptFileName, Object... parameters) {
    try {
      String functionContent =
          ATOM_SCRIPTS.computeIfAbsent(
              scriptFileName,
              (fileName) -> {
                String filePath = "/org/openqa/selenium/remote/" + fileName;
                String scriptCode;
                try (InputStream stream = getClass().getResourceAsStream(filePath)) {
                  scriptCode = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
                String functionName = fileName.replace(".js", "");
                return String.format(
                    "/* %s */return (%s).apply(null, arguments);", functionName, scriptCode);
              });
      return toScript(functionContent, parameters);
    } catch (UncheckedIOException e) {
      throw new WebDriverException(e.getCause());
    } catch (NullPointerException e) {
      throw new WebDriverException(e);
    }
  }

	public final MultiValueMap<HttpRequestHandler, String> getMappings() {
		MultiValueMap<HttpRequestHandler, String> mappings = new LinkedMultiValueMap<>();
		if (this.registration != null) {
			SockJsService sockJsService = this.registration.getSockJsService();
			for (String path : this.paths) {
				String pattern = (path.endsWith("/") ? path + "**" : path + "/**");
				SockJsHttpRequestHandler handler = new SockJsHttpRequestHandler(sockJsService, this.webSocketHandler);
				mappings.add(handler, pattern);
			}
		}
		else {
			for (String path : this.paths) {
				WebSocketHttpRequestHandler handler;
				if (this.handshakeHandler != null) {
					handler = new WebSocketHttpRequestHandler(this.webSocketHandler, this.handshakeHandler);
				}
				else {
					handler = new WebSocketHttpRequestHandler(this.webSocketHandler);
				}
				HandshakeInterceptor[] interceptors = getInterceptors();
				if (interceptors.length > 0) {
					handler.setHandshakeInterceptors(Arrays.asList(interceptors));
				}
				mappings.add(handler, path);
			}
		}
		return mappings;
	}

