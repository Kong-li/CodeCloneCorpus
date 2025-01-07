  public QuotaUsage getQuotaUsage(Path f) throws IOException {
    Map<String, String> params = new HashMap<>();
    params.put(OP_PARAM, Operation.GETQUOTAUSAGE.toString());
    HttpURLConnection conn =
        getConnection(Operation.GETQUOTAUSAGE.getMethod(), params, f, true);
    JSONObject json = (JSONObject) ((JSONObject)
        HttpFSUtils.jsonParse(conn)).get(QUOTA_USAGE_JSON);
    QuotaUsage.Builder builder = new QuotaUsage.Builder();
    builder = buildQuotaUsage(builder, json, QuotaUsage.Builder.class);
    return builder.build();
  }

public HttpHeaders customizeHeaders() {
		if (!getSupportedContentTypes().iterator().hasNext()) {
			return HttpHeaders.EMPTY;
		}
		HttpHeaders headers = new HttpHeaders();
		headers.setAccept(getSupportedContentTypes());
		if (HttpMethod.PATCH.equals(this.httpMethod)) {
			headers.setAcceptPatch(getSupportedContentTypes());
		}
		return headers;
	}

public Set<IProcessor> fetchProcessors(final String prefix) {
    final Set<IProcessor> processorSet = new HashSet<>();
    processorSet.add(new ClassForPositionAttributeTagProcessor(prefix));
    final RemarkForPositionAttributeTagProcessor remarkProcessor = new RemarkForPositionAttributeTagProcessor(prefix);
    processorSet.add(remarkProcessor);
    final HeadlinesElementTagProcessor headlinesProcessor = new HeadlinesElementTagProcessor(prefix);
    processorSet.add(headlinesProcessor);
    final MatchDayTodayModelProcessor modelProcessor = new MatchDayTodayModelProcessor(prefix);
    processorSet.add(modelProcessor);
    // This will remove the xmlns:score attributes we might add for IDE validation
    final StandardXmlNsTagProcessor xmlNsProcessor = new StandardXmlNsTagProcessor(TemplateMode.HTML, prefix);
    processorSet.add(xmlNsProcessor);
    return processorSet;
}

  public synchronized void snapshot(MetricsRecordBuilder builder, boolean all) {
    Quantile[] quantilesArray = getQuantiles();
    if (all || changed()) {
      builder.addGauge(numInfo, previousCount);
      for (int i = 0; i < quantilesArray.length; i++) {
        long newValue = 0;
        // If snapshot is null, we failed to update since the window was empty
        if (previousSnapshot != null) {
          newValue = previousSnapshot.get(quantilesArray[i]);
        }
        builder.addGauge(quantileInfos[i], newValue);
      }
      if (changed()) {
        clearChanged();
      }
    }
  }

