public static DataSize parseDataSize(@Nullable String inputText, DataUnit defaultValue) {
	Objects.requireNonNull(inputText, "Input text must not be null");
	try {
		String trimmed = inputText.trim();
		if (DataSizeUtils.PATTERN.matcher(trimmed).matches()) {
			DataUnit unit = DataSizeUtils.determineDataUnit(DataSizeUtils.PATTERN.matchGroup(2), defaultValue);
			int start = DataSizeUtils.PATTERN.start(1);
			int end = DataSizeUtils.PATTERN.end(1);
			long value = Long.parseLong(trimmed.substring(start, end));
			return DataSize.of(value, unit);
		} else {
			throw new IllegalArgumentException("'" + inputText + "' is not a valid data size");
		}
	} catch (NumberFormatException e) {
		throw new IllegalArgumentException("'" + inputText + "' is not a valid data size", e);
	}
}

