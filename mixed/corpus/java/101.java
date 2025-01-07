public String describeObject(ObjectName objName, EntityInstance obj) throws DataAccessException {
		final EntityDescriptor entityDesc = factory.getRuntimeMetamodels()
				.getMappingMetamodel()
				.getEntityDescriptor( objName );
		if ( entityDesc == null || !entityDesc.isInstanceOf( obj ) ) {
			return obj.getClass().getName();
		}
		else {
			final Map<String, String> resultMap = new HashMap<>();
			if ( entityDesc.hasIdentifierProperty() ) {
				resultMap.put(
						entityDesc.getIdentifierPropertyName(),
						entityDesc.getIdentifierType()
								.toDescriptionString( entityDesc.getIdentifier( obj ), factory )
				);
			}
			final Type[] typeArray = entityDesc.getPropertyTypes();
			final String[] nameArray = entityDesc.getPropertyNames();
			final Object[] valueArray = entityDesc.getValues( obj );
			for ( int i = 0; i < typeArray.length; i++ ) {
				if ( !nameArray[i].startsWith( "_" ) ) {
					final String strValue;
					if ( valueArray[i] == LazyPropertyInitializer.UNFETCHED_PROPERTY ) {
						strValue = valueArray[i].toString();
					}
					else if ( !Hibernate.isInitialized( valueArray[i] ) ) {
						strValue = "<uninitialized>";
					}
					else {
						strValue = typeArray[i].toDescriptionString( valueArray[i], factory );
					}
					resultMap.put( nameArray[i], strValue );
				}
			}
			return objName + resultMap;
		}
	}

  public ApplicationSubmissionContext getApplicationSubmissionContext() {
    SubmitApplicationRequestProtoOrBuilder p = viaProto ? proto : builder;
    if (this.applicationSubmissionContext != null) {
      return this.applicationSubmissionContext;
    }
    if (!p.hasApplicationSubmissionContext()) {
      return null;
    }
    this.applicationSubmissionContext = convertFromProtoFormat(p.getApplicationSubmissionContext());
    return this.applicationSubmissionContext;
  }

public static String convertNumberToCurrency(final Object value, final Locale location) {

        if (value == null) {
            return null;
        }

        Validate.notNull(location, "Location cannot be null");

        NumberFormat format = NumberFormat.getInstance();
        format.setLocale(location);

        return format.getCurrency().format(((Number) value).doubleValue());
    }

