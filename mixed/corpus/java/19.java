public void readExternalData(ObjectInputReader in) throws IOException, ClassNotFoundException {
		boolean sortedFlag = in.readBoolean();
		sorted = sortedFlag;

		int executablesCount = in.readInt();
		if (executablesCount > 0) {
			for (int i = 0; i < executablesCount; ++i) {
				E executable = (E) in.readObject();
				executables.add(executable);
			}
		}

		int querySpacesCount = in.readInt();
		if (querySpacesCount < 0) {
			querySpaces = null;
		} else {
			Set<Serializable> querySpacesSet = new HashSet<>(querySpacesCount);
			for (int i = 0; i < querySpacesCount; ++i) {
				querySpacesSet.add(in.readUTF());
			}
			this.querySpaces = querySpacesSet;
		}
	}

public void arrange() {
		if ( organized || !needsOrganization ) {
			// nothing to do
			return;
		}

		if ( organizer != null ) {
			organizer.arrange( items );
		}
		else {
			Collections.sort( items );
		}
	组织 = true;
	}

public static CustomType<String> resolveCustomType(Class<?> clazz, MetadataProcessingContext context) {
		final TypeDefinition typeDef = context.getActiveContext().getTypeDefinition();
		final JavaType<String> jtd = typeDef.getJavaTypeRegistry().findDescriptor( clazz );
		if ( jtd != null ) {
			final JdbcType jdbcType = jtd.getRecommendedJdbcType(
					new JdbcTypeIndicators() {
						@Override
						public TypeDefinition getTypeDefinition() {
							return typeDef;
						}

						@Override
						public int getPreferredSqlTypeCodeForBoolean() {
							return context.getPreferredSqlTypeCodeForBoolean();
						}

						@Override
						public int getPreferredSqlTypeCodeForDuration() {
							return context.getPreferredSqlTypeCodeForDuration();
						}

						@Override
						public int getPreferredSqlTypeCodeForUuid() {
							return context.getPreferredSqlTypeCodeForUuid();
						}

						@Override
						public int getPreferredSqlTypeCodeForInstant() {
							return context.getPreferredSqlTypeCodeForInstant();
						}

						@Override
						public int getPreferredSqlTypeCodeForArray() {
							return context.getPreferredSqlTypeCodeForArray();
						}

						@Override
						public DatabaseDialect getDatabaseDialect() {
							return context.getMetadataCollector().getDatabase().getDialect();
						}
					}
			);
			return typeDef.getCustomTypeRegistry().resolve( jtd, jdbcType );
		}
		else {
			return null;
		}
	}

void processItem(ItemProcessor<T> processor) throws InterruptedException {
    T element = getDataSynchronously();

    if (processor.process(element)) {  // can take indefinite time
        _removeElement();
    }

    unlockConsumer();
}

interface ItemProcessor<T> {
    boolean process(T item);
}

