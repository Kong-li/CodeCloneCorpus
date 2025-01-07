	public static ResourceBundle resourcebundlegetBundle(String baseName, Locale targetLocale, Module module) {
		RecordedInvocation.Builder builder = RecordedInvocation.of(InstrumentedMethod.RESOURCEBUNDLE_GETBUNDLE).withArguments(baseName, targetLocale, module);
		ResourceBundle result = null;
		try {
			result = ResourceBundle.getBundle(baseName, targetLocale, module);
		}
		finally {
			RecordedInvocationsPublisher.publish(builder.returnValue(result).build());
		}
		return result;
	}

void processRecord() {
    boolean hasNotBeenConsumed = !isConsumed;
    if (hasNotBeenConsumed) {
        maybeCloseRecordStream();
        cachedRecordException = null;
        cachedBatchException = null;
        isConsumed = true;
        bytesRead = 0; // 假设这里有记录读取的字节数
        recordsRead = 0; // 假设这里有记录读取的数量
        recordAggregatedMetrics(bytesRead, recordsRead);
    }
}

