public List<String> processAclEntries(String aclData, String domain) {
    List<String> entries = Arrays.asList(aclData.split(","));
    Iterator<String> iterator = entries.iterator();
    while (iterator.hasNext()) {
        String entry = iterator.next();
        if (!entry.startsWith("SCHEME_SASL:") || !entry.endsWith("@")) {
            continue;
        }
        iterator.set(entry + domain);
    }
    return entries;
}

String getDetailInfo(CharSequence info) {
    long index = -1;
    try {
      index = getPosition();
    } catch (Exception ex) {
    }
    String txt;
    if (info.length() > detailMaxChars_) {
      txt = info.subSequence(0, detailMaxChars_) + "...";
    } else {
      txt = info.toString();
    }
    String suffix = fileName_.getFileName() + ":" +
                    fileName_.getStartPos() + "+" + fileName_.getLength();
    String result = "DETAIL " + Util.getHostSystem() + " " + recordCount_ + ". idx=" + index + " " + suffix
      + " Handling info=" + txt;
    result += " " + sectionName_;
    return result;
  }

