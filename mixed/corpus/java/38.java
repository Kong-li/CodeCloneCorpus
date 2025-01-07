	public SqmPathSource<?> findSubPathSource(String name) {
		final CollectionPart.Nature nature = CollectionPart.Nature.fromNameExact( name );
		if ( nature != null ) {
			switch ( nature ) {
				case INDEX:
					return indexPathSource;
				case ELEMENT:
					return getElementPathSource();
			}
		}
		return getElementPathSource().findSubPathSource( name );
	}

  public static NodeId verifyAndGetNodeId(Block html, String nodeIdStr) {
    if (nodeIdStr == null || nodeIdStr.isEmpty()) {
      html.h1().__("Cannot get container logs without a NodeId").__();
      return null;
    }
    NodeId nodeId = null;
    try {
      nodeId = NodeId.fromString(nodeIdStr);
    } catch (IllegalArgumentException e) {
      html.h1().__("Cannot get container logs. Invalid nodeId: " + nodeIdStr)
          .__();
      return null;
    }
    return nodeId;
  }

	private static boolean isValidMappedBy(AnnotatedFieldDescription persistentField, TypeDescription targetEntity, String mappedBy, ByteBuddyEnhancementContext context) {
		try {
			FieldDescription f = FieldLocator.ForClassHierarchy.Factory.INSTANCE.make( targetEntity ).locate( mappedBy ).getField();
			AnnotatedFieldDescription annotatedF = new AnnotatedFieldDescription( context, f );

			return context.isPersistentField( annotatedF ) && persistentField.getDeclaringType().asErasure().isAssignableTo( entityType( f.getType() ) );
		}
		catch ( IllegalStateException e ) {
			return false;
		}
	}

public SqlXmlValue createSqlXmlValue(final ResultType resultClass, final XmlProvider provider) {
		return new AbstractJdbc4SqlXmlValue() {
			{
				final SQLXML xmlObject = provideXml();
				if (xmlObject != null) {
					provider.provideXml(xmlObject.setResult(resultClass));
				}
			}

			private SQLXML provideXml() throws SQLException, IOException {
				return this.provideXmlResult();
			}

			protected void provideXml(SQLXML xmlObject) throws SQLException, IOException {
				provideXml(xmlObject);
			}

			private SQLXML provideXmlResult() throws SQLException, IOException {
				throw new UnsupportedOperationException("This method should be implemented by subclasses.");
			}
		};
	}

