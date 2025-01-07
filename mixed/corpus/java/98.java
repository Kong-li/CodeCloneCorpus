protected ContainerRequest refineContainerRequest(ContainerRequest original) {
    List<String> filteredHosts = new ArrayList<>();
    for (String host : original.hosts) {
      if (isNodeNotBlacklisted(host)) {
        filteredHosts.add(host);
      }
    }
    String[] hostsArray = filteredHosts.toArray(new String[filteredHosts.size()]);
    ContainerRequest refinedReq = new ContainerRequest(original.attemptID, original.capability,
        hostsArray, original.racks, original.priority, original.nodeLabelExpression);
    return refinedReq;
  }

private void syncBuilderWithLocal() {
    boolean hasReason = this.reason != null;
    boolean hasUpdateRequest = this.updateRequest != null;

    if (hasReason) {
      builder.setReason(this.reason);
    }

    if (hasUpdateRequest) {
      builder.setUpdateRequest(
          ProtoUtils.convertToProtoFormat(this.updateRequest));
    }
}

    public void generate() {
        Objects.requireNonNull(packageName);
        for (String header : HEADER) {
            buffer.printf("%s%n", header);
        }
        buffer.printf("package %s;%n", packageName);
        buffer.printf("%n");
        for (String newImport : imports) {
            buffer.printf("import %s;%n", newImport);
        }
        buffer.printf("%n");
        if (!staticImports.isEmpty()) {
            for (String newImport : staticImports) {
                buffer.printf("import static %s;%n", newImport);
            }
            buffer.printf("%n");
        }
    }

