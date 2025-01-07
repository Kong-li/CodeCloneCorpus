	public static Method getMethod(Class<?> c, String mName, Class<?>... parameterTypes) throws NoSuchMethodException {
		Method m = null;
		Class<?> oc = c;
		while (c != null) {
			try {
				m = c.getDeclaredMethod(mName, parameterTypes);
				break;
			} catch (NoSuchMethodException e) {}
			c = c.getSuperclass();
		}

		if (m == null) throw new NoSuchMethodException(oc.getName() + " :: " + mName + "(args)");
		return setAccessible(m);
	}

private GenericType locateConfigBasedParameterHandlerSuperclass(TypeToken<?> type) {
		TypeToken<?> superclass = type.getRawType();

		// Abort?
		if (superclass == null || superclass == Object.class) {
			return null;
		}

		Type genericSupertype = type.getGenericSupertype();
		if (genericSupertype instanceof GenericType) {
			Type rawType = ((GenericType) genericSupertype).getRawType();
			if (rawType == ConfigBasedParameterHandler.class) {
				return (GenericType) genericSupertype;
			}
		}
		return locateConfigBasedParameterHandlerSuperclass(superclass);
	}

public Resource determineMaxResource() {
    Resource result = Resources.none();

    if (Resources.equals(effMaxRes, Resources.none())) {
        return result;
    }

    result = Resources.clone(effMaxRes);

    return multiplyAndReturn(result, totalPartitionResource, absMaxCapacity);
}

private Resource multiplyAndReturn(Resource res, Resource partitionResource, int capacity) {
    if (!partitionResource.equals(Resources.none())) {
        return Resources.multiply(res, capacity);
    }
    return res;
}

public String fetchContent(Path filePath) throws IOException {
    FileStatus fileStatus = fileSystem.getFileStatus(filePath);
    int length = (int) fileStatus.getLen();
    byte[] buffer = new byte[length];
    FSDataInputStream inputStream = null;
    try {
        inputStream = fileSystem.open(filePath);
        int readCount = inputStream.read(buffer);
        return new String(buffer, 0, readCount, UTF_8);
    } finally {
        IOUtils.closeStream(inputStream);
    }
}

