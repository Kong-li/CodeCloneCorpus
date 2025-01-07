public synchronized NodeHealthStatus getNodeState() {
    NodeStatusProtoOrBuilder p = viaProto ? proto : builder;
    if (!p.hasNodeHealthStatus()) {
      return null;
    }
    NodeHealthStatus nodeHealthStatus = convertFromProtoFormat(p.getNodeHealthStatus());
    if (nodeHealthStatus != null) {
      return nodeHealthStatus;
    }
    return null;
  }

	private @Nullable RootBeanDefinition getDefaultExecutorBeanDefinition(String channelName) {
		if (channelName.equals("brokerChannel")) {
			return null;
		}
		RootBeanDefinition executorDef = new RootBeanDefinition(ThreadPoolTaskExecutor.class);
		executorDef.getPropertyValues().add("corePoolSize", Runtime.getRuntime().availableProcessors() * 2);
		executorDef.getPropertyValues().add("maxPoolSize", Integer.MAX_VALUE);
		executorDef.getPropertyValues().add("queueCapacity", Integer.MAX_VALUE);
		executorDef.getPropertyValues().add("allowCoreThreadTimeOut", true);
		return executorDef;
	}

private void reflect(BrowserConfig that) {
    Map<String, Object> newConfig = new TreeMap<>(browserConfig);

    for (Key key : Key.values()) {
      Object value = key.reflect(browserConfig, that.browserConfig);
      if (value != null) {
        newConfig.put(key.getKey(), value);
      }
    }

    this.browserConfig = Collections.unmodifiableMap(newConfig);
  }

    protected IInliner getInliner(final ITemplateContext context, final StandardInlineMode inlineMode) {

        switch (inlineMode) {
            case NONE:
                return NoOpInliner.INSTANCE;
            case HTML:
                return new StandardHTMLInliner(context.getConfiguration());
            case TEXT:
                return new StandardTextInliner(context.getConfiguration());
            case JAVASCRIPT:
                return new StandardJavaScriptInliner(context.getConfiguration());
            case CSS:
                return new StandardCSSInliner(context.getConfiguration());
            default:
                throw new TemplateProcessingException(
                        "Invalid inline mode selected: " + inlineMode + ". Allowed inline modes in template mode " +
                       getTemplateMode() + " are: " +
                        "\"" + StandardInlineMode.HTML + "\", \"" + StandardInlineMode.TEXT + "\", " +
                        "\"" + StandardInlineMode.JAVASCRIPT + "\", \"" + StandardInlineMode.CSS + "\" and " +
                        "\"" + StandardInlineMode.NONE + "\"");
        }

    }

