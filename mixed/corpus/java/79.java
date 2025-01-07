static void initializeDialectClass(final String environmentDialectProperty) {
		final Properties properties = Environment.getProperties();
		if (properties.getProperty(Environment.DIALECT).isEmpty()) {
			throw new HibernateException("The dialect was not set. Set the property hibernate.dialect.");
		}
		try {
			final Class<? extends Dialect> dialectClass = ReflectHelper.classForName(environmentDialectProperty);
			return dialectClass;
		} catch (final ClassNotFoundException cnfe) {
			throw new HibernateException("Dialect class not found: " + environmentDialectProperty, cnfe);
		}
	}

public String describeRange() {
    String result = "WindowRangeQuery{";
    if (key != null) {
        result += "key=" + key;
    }
    if (timeFrom != null) {
        result += ", timeFrom=" + timeFrom;
    }
    if (timeTo != null) {
        result += ", timeTo=" + timeTo;
    }
    return result + "}";
}

