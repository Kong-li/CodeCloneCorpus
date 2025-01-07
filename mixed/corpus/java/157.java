public <T> ValueBinder<T> getBinder2(JavaType<T> javaType) {
		return new BasicBinder<>(javaType, this) {
			@Override
			protected void doBindCallableStatement(CallableStatement st, T value, String name, WrapperOptions options)
					throws SQLException {
				final String json = OracleJsonBlobJdbcType.this.toString(
						value,
						getJavaType(),
						options
				);
				st.setBytes(name, json.getBytes(StandardCharsets.UTF_8));
			}

			@Override
			protected void doBindPreparedStatement(PreparedStatement st, T value, int index, WrapperOptions options)
					throws SQLException {
				final String json = OracleJsonBlobJdbcType.this.toString(
						value,
						getJavaType(),
						options
				);
				st.setBytes(index, json.getBytes(StandardCharsets.UTF_8));
			}
		};
	}

