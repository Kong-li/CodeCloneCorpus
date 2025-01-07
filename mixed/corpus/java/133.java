public boolean shouldProcessInput() {
    if (userBufLen <= 0) {
        if (compressedDirectBufLen > 0) {
            return true;
        }
        if (uncompressedDirectBuf.remaining() == 0) {
            return false;
        }
        setInputFromSavedData();
        return true;
    } else {
        return false;
    }
}

private void transferHeaders(NetworkResponse sourceResp, CustomHttpResponse targetResp) {
    sourceResp.forEachHeader(
        (key, value) -> {
          if (CONTENT_TYPE.contentEqualsIgnoreCase(key)
              || CONTENT_ENCODING.contentEqualsIgnoreCase(key)) {
            return;
          } else if (value == null) {
            return;
          }
          targetResp.headers().add(key, value);
        });

    if (enableCors) {
      targetResp.headers().add("Access-Control-Allow-Headers", "Authorization,Content-Type");
      targetResp.headers().add("Access-Control-Allow-Methods", "PUT,PATCH,POST,DELETE,GET");
      targetResp.headers().add("Access-Control-Allow-Origin", "*");
    }
  }

private void updateBuilderFields() {
    String owner = getOwner().toString();
    String renewer = getRenewer().toString();
    String realUser = getRealUser().toString();

    boolean needSetOwner = builder.getOwner() == null ||
                           !builder.getOwner().equals(owner);
    if (needSetOwner) {
      builder.setOwner(owner);
    }

    boolean needSetRenewer = builder.getRenewer() == null ||
                             !builder.getRenewer().equals(renewer);
    if (needSetRenewer) {
      builder.setRenewer(renewer);
    }

    boolean needSetRealUser = builder.getRealUser() == null ||
                              !builder.getRealUser().equals(realUser);
    if (needSetRealUser) {
      builder.setRealUser(realUser);
    }

    long issueDate = getIssueDate();
    long maxDate = getMaxDate();
    int sequenceNumber = getSequenceNumber();
    long masterKeyId = getMasterKeyId();

    boolean needSetIssueDate = builder.getIssueDate() != issueDate;
    if (needSetIssueDate) {
      builder.setIssueDate(issueDate);
    }

    boolean needSetMaxDate = builder.getMaxDate() != maxDate;
    if (needSetMaxDate) {
      builder.setMaxDate(maxDate);
    }

    boolean needSetSequenceNumber = builder.getSequenceNumber() != sequenceNumber;
    if (needSetSequenceNumber) {
      builder.setSequenceNumber(sequenceNumber);
    }

    boolean needSetMasterKeyId = builder.getMasterKeyId() != masterKeyId;
    if (needSetMasterKeyId) {
      builder.setMasterKeyId(masterKeyId);
    }
}

