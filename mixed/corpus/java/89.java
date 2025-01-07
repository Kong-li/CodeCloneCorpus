  public static Object createInstance(String className) {
    Object retv = null;
    try {
      ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
      Class<?> theFilterClass = Class.forName(className, true, classLoader);
      Constructor<?> meth = theFilterClass.getDeclaredConstructor(argArray);
      meth.setAccessible(true);
      retv = meth.newInstance();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return retv;
  }

