protected Object executeProcedure(Procedure procedure, Params params) throws Throwable {
    try {
      if (!procedure.isAccessible()) {
        procedure.setAccessible(true);
      }
      final Object r = procedure.invoke(descriptor.getProxy(), params);
      hasSuccessfulOperation = true;
      return r;
    } catch (InvocationTargetException e) {
      throw e.getCause();
    }
  }

Properties configureKMSProps(Map<String, String> prefixedProps) {
    Properties props = new Properties();

    for (Map.Entry<String, String> entry : prefixedProps.entrySet()) {
      props.setProperty(entry.getKey().replaceFirst(CONFIG_PREFIX, ""), entry.getValue());
    }

    String authType = props.getProperty(AUTH_TYPE);
    if (!authType.equals(PseudoAuthenticationHandler.TYPE)) {
      if (!authType.equals(KerberosAuthenticationHandler.TYPE)) {
        authType = KerberosDelegationTokenAuthenticationHandler.class.getName();
      } else {
        authType = PseudoDelegationTokenAuthenticationHandler.class.getName();
      }
    }
    props.setProperty(AUTH_TYPE, authType);
    props.setProperty(DelegationTokenAuthenticationHandler.TOKEN_KIND,
        KMSDelegationToken.TOKEN_KIND_STR);

    return props;
}

