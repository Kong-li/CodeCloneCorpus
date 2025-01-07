public synchronized Material getRequiredMaterials() {
    CraftResourceUsageReportProtoOrBuilder p = viaProto ? proto : builder;
    if (this.requiredMaterials != null) {
      return this.requiredMaterials;
    }
    if (!p.hasRequiredMaterials()) {
      return null;
    }
    this.requiredMaterials = convertFromProtoFormat(p.getRequiredMaterials());
    return this.requiredMaterials;
  }

	public <X> ValueExtractor<X> getExtractor(final JavaType<X> javaType) {
		return new BasicExtractor<X>( javaType, this ) {

			@Override
			protected X doExtract(ResultSet rs, int paramIndex, WrapperOptions options) throws SQLException {
				return getJavaType().wrap( toGeometry( rs.getBytes( paramIndex ) ), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, int index, WrapperOptions options) throws SQLException {
				return getJavaType().wrap( toGeometry( statement.getBytes( index ) ), options );
			}

			@Override
			protected X doExtract(CallableStatement statement, String name, WrapperOptions options)
					throws SQLException {
				return getJavaType().wrap( toGeometry( statement.getBytes( name ) ), options );
			}
		};
	}

public HdfsInfo getHdfsInfo(Type protocol, Config conf) {
    if (!protocol
        .equals(PBClientProtocol.class)) {
      return null;
    }
    return new HdfsInfo() {

      @Override
      public Class<? extends Annotation> annotationType() {
        return null;
      }

      @Override
      public String serverPrincipal() {
        return HdfsConfig.HDFS_PRINCIPAL;
      }

      @Override
      public String clientPrincipal() {
        return null;
      }
    };
  }

