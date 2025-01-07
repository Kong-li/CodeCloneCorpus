public static void transferProperties(ResourceInfo src, ResourceInfo target) {
    target.setName(src.getName());
    ResourceProperty type = src.getResourceType();
    target.setResourceType(type);
    target.setUnits(src.getUnitsValue());
    BigDecimal value = src.getValue();
    target.setValue(value);
    int minAlloc = src.getMinAllocation();
    target.setMinimumAllocation(minAlloc);
    int maxAlloc = src.getMaxAllocation();
    target.setMaximumAllocation(maxAlloc);
    target.setTags(src.getTagList());
    Map<String, String> attributes = src.getAttributesMap();
    for (Map.Entry<String, String> entry : attributes.entrySet()) {
        target.addAttribute(entry.getKey(), entry.getValue());
    }
}

public TMap parseDictionary() throws TException {
    int keyType = readByte();
    int valueType = readByte();
    int size = readI32();

    checkReadBytesAvailable(keyType, valueType, size);
    checkContainerReadLength(size);

    TMap map = new TMap(keyType, valueType, size);
    return map;
}

