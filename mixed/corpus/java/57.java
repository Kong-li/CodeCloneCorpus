  public void flush() throws TTransportException {
    // Extract request and reset buffer
    byte[] data = requestBuffer_.toByteArray();
    requestBuffer_.reset();

    try {
      // Create connection object
      connection = (HttpConnection)Connector.open(url_);

      // Make the request
      connection.setRequestMethod("POST");
      connection.setRequestProperty("Content-Type", "application/x-thrift");
      connection.setRequestProperty("Accept", "application/x-thrift");
      connection.setRequestProperty("User-Agent", "JavaME/THttpClient");

      connection.setRequestProperty("Connection", "Keep-Alive");
      connection.setRequestProperty("Keep-Alive", "5000");
      connection.setRequestProperty("Http-version", "HTTP/1.1");
      connection.setRequestProperty("Cache-Control", "no-transform");

      if (customHeaders_ != null) {
        for (Enumeration e = customHeaders_.keys() ; e.hasMoreElements() ;) {
          String key = (String)e.nextElement();
          String value = (String)customHeaders_.get(key);
          connection.setRequestProperty(key, value);
        }
      }

      OutputStream os = connection.openOutputStream();
      os.write(data);
      os.close();

      int responseCode = connection.getResponseCode();
      if (responseCode != HttpConnection.HTTP_OK) {
        throw new TTransportException("HTTP Response code: " + responseCode);
      }

      // Read the responses
      inputStream_ = connection.openInputStream();
    } catch (IOException iox) {
      System.out.println(iox.toString());
      throw new TTransportException(iox);
    }
  }

	public <X> ValueExtractor<X> getExtractor(JavaType<X> javaType) {
		return new BasicExtractor<>( javaType, this ) {
			@Override
			protected X doExtract(ResultSet rs, int paramIndex, WrapperOptions options) throws SQLException {
				final XmlAsStringArrayJdbcType jdbcType = (XmlAsStringArrayJdbcType) getJdbcType();
				final String value;
				if ( jdbcType.nationalized && options.getDialect().supportsNationalizedMethods() ) {
					value = rs.getNString( paramIndex );
				}
				else {
					value = rs.getString( paramIndex );
				}
				return jdbcType.fromString( value, getJavaType(), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, int index, WrapperOptions options)
					throws SQLException {
				final XmlAsStringArrayJdbcType jdbcType = (XmlAsStringArrayJdbcType) getJdbcType();
				final String value;
				if ( jdbcType.nationalized && options.getDialect().supportsNationalizedMethods() ) {
					value = statement.getNString( index );
				}
				else {
					value = statement.getString( index );
				}
				return jdbcType.fromString( value, getJavaType(), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, String name, WrapperOptions options)
					throws SQLException {
				final XmlAsStringArrayJdbcType jdbcType = (XmlAsStringArrayJdbcType) getJdbcType();
				final String value;
				if ( jdbcType.nationalized && options.getDialect().supportsNationalizedMethods() ) {
					value = statement.getNString( name );
				}
				else {
					value = statement.getString( name );
				}
				return jdbcType.fromString( value, getJavaType(), options );
			}

		};
	}

public static void addStandardFormatters(FormatterRegistry formatterRegistry) {
		// Standard handling of numeric values
		formatterRegistry.addFormatterForFieldAnnotation(new NumericFormatAnnotationFormatterFactory());

		// Standard handling of monetary values
		if (jsr354Present) {
			formatterRegistry.addFormatter(new CurrencyUnitFormatter());
			formatterRegistry.addFormatter(new MonetaryAmountFormatter());
			formatterRegistry.addFormatterForFieldAnnotation(new Jsr354NumericFormatAnnotationFormatterFactory());
		}

		// Standard handling of date-time values

		// just handling JSR-310 specific date and time types
		new DateTimeFormatterRegistrar().registerFormatters(formatterRegistry);

		// regular DateFormat-based Date, Calendar, Long converters
		new DateFormatterRegistrar().registerFormatters(formatterRegistry);
	}

