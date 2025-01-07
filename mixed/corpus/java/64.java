  public void writeTo(Appendable appendable) throws IOException {
    try (JsonOutput json = new Json().newOutput(appendable)) {
      json.beginObject();

      // Now for the w3c capabilities
      json.name("capabilities");
      json.beginObject();

      // Then write everything into the w3c payload. Because of the way we do this, it's easiest
      // to just populate the "firstMatch" section. The spec says it's fine to omit the
      // "alwaysMatch" field, so we do this.
      json.name("firstMatch");
      json.beginArray();
      getW3C().forEach(json::write);
      json.endArray();

      json.endObject(); // Close "capabilities" object

      writeMetaData(json);

      json.endObject();
    }
  }

public String showNodeInfo() {
    StringBuilder content = new StringBuilder();
    long total = getTotalCapacity();
    long free = getAvailableSpace();
    long used = getCurrentUsage();
    float usagePercent = getUsagePercentage();
    long cacheTotal = getCachedCapacity();
    long cacheFree = getCacheAvailableSpace();
    long cacheUsed = getCachedUsage();
    float cacheUsagePercent = getCacheUsagePercentage();
    content.append(getNodeName());
    if (!NetworkTopology.DEFAULT_RACK.equals(getLocation())) {
      content.append(" ").append(getLocation());
    }
    if (getUpgradeStatus() != null) {
      content.append(" ").append(getUpgradeStatus());
    }
    if (isDecommissioned()) {
      content.append(" DD");
    } else if (isDecommissionInProgress()) {
      content.append(" DP");
    } else if (isInMaintenance()) {
      content.append(" IM");
    } else if (isEnteringMaintenance()) {
      content.append(" EM");
    } else {
      content.append(" IN");
    }
    content.append(" ").append(total).append("(").append(StringUtils.byteDesc(total))
        .append(")")
        .append(" ").append(used).append("(").append(StringUtils.byteDesc(used))
        .append(")")
        .append(" ").append(percent2String(usagePercent))
        .append(" ").append(free).append("(").append(StringUtils.byteDesc(free))
        .append(")")
        .append(" ").append(cacheTotal).append("(").append(StringUtils.byteDesc(cacheTotal))
        .append(")")
        .append(" ").append(cacheUsed).append("(").append(StringUtils.byteDesc(cacheUsed))
        .append(")")
        .append(" ").append(percent2String(cacheUsagePercent))
        .append(" ").append(cacheFree).append("(").append(StringUtils.byteDesc(cacheFree))
        .append(")")
        .append(" ").append(new Date(getLastUpdate()));
    return content.toString();
}

public List<String> convertToFormattedDecimalList(final Collection<? extends Number> values, final Integer minLength, final Integer fractionDigits, final String separator) {
        if (values == null) {
            return null;
        }
        List<String> resultList = new ArrayList<>(values.size() + 2);
        for (final Number value : values) {
            final String formattedValue = formatDecimal(value, minLength, fractionDigits, separator);
            resultList.add(formattedValue);
        }
        return resultList;
    }

    private String formatDecimal(final Number number, final Integer minIntegerDigits, final Integer decimalPlaces, final String pointType) {
        // 实现格式化逻辑
        return null;
    }

	private void registerJtaTransactionAspect(Element element, ParserContext parserContext) {
		String txAspectBeanName = TransactionManagementConfigUtils.JTA_TRANSACTION_ASPECT_BEAN_NAME;
		String txAspectClassName = TransactionManagementConfigUtils.JTA_TRANSACTION_ASPECT_CLASS_NAME;
		if (!parserContext.getRegistry().containsBeanDefinition(txAspectBeanName)) {
			RootBeanDefinition def = new RootBeanDefinition();
			def.setBeanClassName(txAspectClassName);
			def.setFactoryMethodName("aspectOf");
			registerTransactionManager(element, def);
			parserContext.registerBeanComponent(new BeanComponentDefinition(def, txAspectBeanName));
		}
	}

