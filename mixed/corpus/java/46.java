public void restore(RMStateInfo state) throws Exception {
    RMStateStoreManager store = rmEnvironment.getStore();
    assert store != null;

    // recover applications
    Map<ApplicationIdentifier, ApplicationData> appStates =
        state.getApplicationMap();
    LOG.info("Recovering " + appStates.size() + " applications");

    int count = 0;

    try {
      for (ApplicationData appState : appStates.values()) {
        count += 1;
        recoverApplication(appState, state);
      }
    } finally {
      LOG.info("Successfully recovered " + count  + " out of "
          + appStates.size() + " applications");
    }
}

	public static Object calculateGuess(JCExpression expr) {
		if (expr instanceof JCLiteral) {
			JCLiteral lit = (JCLiteral) expr;
			if (lit.getKind() == com.sun.source.tree.Tree.Kind.BOOLEAN_LITERAL) {
				return ((Number) lit.value).intValue() == 0 ? false : true;
			}
			return lit.value;
		}

		if (expr instanceof JCIdent || expr instanceof JCFieldAccess) {
			String x = expr.toString();
			if (x.endsWith(".class")) return new ClassLiteral(x.substring(0, x.length() - 6));
			int idx = x.lastIndexOf('.');
			if (idx > -1) x = x.substring(idx + 1);
			return new FieldSelect(x);
		}

		return null;
	}

private FileOutputStream createFileOutputStream() throws TTransportException {
    FileOutputStream fos;
    try {
      if (outputStream_ != null) {
        ((TruncableBufferedOutputStream) outputStream_).trunc();
        fos = outputStream_;
      } else {
        fos = new TruncableBufferedOutputStream(outputFile_.getInputStream());
      }
    } catch (IOException iox) {
      throw new TTransportException(iox.getMessage(), iox);
    }
    return (fos);
  }

