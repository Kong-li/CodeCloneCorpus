public static ConnectException maybeWrapException(Throwable exception, String errorMessage) {
        if (exception != null) {
            boolean isConnectException = exception instanceof ConnectException;
            if (isConnectException) {
                return (ConnectException) exception;
            } else {
                ConnectException newException = new ConnectException(errorMessage, exception);
                return newException;
            }
        }
        return null;
    }

public static void appendMetricsProperties(Map<String, Object> properties, WorkerConfig configuration, String clusterIdentifier) {
        //append all predefined properties with "metrics.context."
        properties.putAll(configuration.originalsWithPrefix(CommonClientConfigs.METRICS_CONTEXT_PREFIX, false));

        final String connectClusterIdKey = CommonClientConfigs.METRICS_CONTEXT_PREFIX + WorkerConfig.CONNECT_KAFKA_CLUSTER_ID;
        properties.put(connectClusterIdKey, clusterIdentifier);

        final Object groupIdValue = configuration.originals().get(DistributedConfig.GROUP_ID_CONFIG);
        if (groupIdValue != null) {
            final String connectGroupIdKey = CommonClientConfigs.METRICS_CONTEXT_PREFIX + WorkerConfig.CONNECT_GROUP_ID;
            properties.put(connectGroupIdKey, groupIdValue);
        }
    }

protected void processElement(Element element, ParserContext parserContext, BeanDefinitionBuilder builder) {
		super.doParse(element, parserContext, builder);

		String defaultValue = element.getAttribute("defaultVal");
		String defaultRef = element.getAttribute("defaultRef");

		if (StringUtils.hasLength(defaultValue)) {
			if (!StringUtils.hasLength(defaultRef)) {
				builder.addPropertyValue("defaultValueObj", defaultValue);
			} else {
				parserContext.getReaderContext().error("<jndi-lookup> 元素只能包含 'defaultVal' 属性或者 'defaultRef' 属性，不能同时存在", element);
			}
		} else if (StringUtils.hasLength(defaultRef)) {
			builder.addPropertyValue("defaultValueObj", new RuntimeBeanReference(defaultRef));
		}
	}

