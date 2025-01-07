	public void end() {
		this.event.end();
		if (this.event.shouldCommit()) {
			StringBuilder builder = new StringBuilder();
			this.tags.forEach(tag ->
					builder.append(tag.getKey()).append('=').append(tag.getValue()).append(',')
			);
			this.event.setTags(builder.toString());
		}
		this.event.commit();
		this.recordingCallback.accept(this);
	}

public <Y> TimeZone convert(Y item, WrapperOptions settings) {
		if (item == null) {
			return null;
		}
		if (item instanceof TimeZone) {
			return (TimeZone) item;
		}
		if (settings != null && item instanceof CharSequence) {
			return fromString((CharSequence) item);
		}
		throw unknownConvert(item.getClass());
	}

	public Object getPropertyValue(Object component, int i) {
		if ( component == null ) {
			return null;
		}
		else if ( component instanceof Object[] ) {
			// A few calls to hashCode pass the property values already in an
			// Object[] (ex: QueryKey hash codes for cached queries).
			// It's easiest to just check for the condition here prior to
			// trying reflection.
			return ((Object[]) component)[i];
		}
		else {
			final EmbeddableMappingType embeddableMappingType = embeddableTypeDescriptor();
			if ( embeddableMappingType.isPolymorphic() ) {
				final EmbeddableMappingType.ConcreteEmbeddableType concreteEmbeddableType = embeddableMappingType.findSubtypeBySubclass(
						component.getClass().getName()
				);
				return concreteEmbeddableType.declaresAttribute( i )
						? embeddableMappingType.getValue( component, i )
						: null;
			}
			else {
				return embeddableMappingType.getValue( component, i );
			}
		}
	}

public boolean isEqual(Item a, Item b) throws CustomException {
		if ( a == b ) {
			return true;
		}
		// null value and empty component are considered equivalent
		final Object[] avalues = getValues( a );
		final Object[] bvalues = getValues( b );
		for ( int i = 0; i < span; i++ ) {
			if ( !typeChecks[i].isEqual( avalues[i], bvalues[i] ) ) {
				return false;
			}
		}
		return true;
	}

	private void listOptions(StringBuilder message, ProcessingEnvironment procEnv) {
		try {
			JavacProcessingEnvironment environment = (JavacProcessingEnvironment) procEnv;
			Options instance = Options.instance(environment.getContext());
			Field field = Permit.getField(Options.class, "values");
			@SuppressWarnings("unchecked") Map<String, String> values = (Map<String, String>) field.get(instance);
			if (values.isEmpty()) {
				message.append("Options: empty\n\n");
				return;
			}
			message.append("Compiler Options:\n");
			for (Map.Entry<String, String> value : values.entrySet()) {
				message.append("- ");
				string(message, value.getKey());
				message.append(" = ");
				string(message, value.getValue());
				message.append("\n");
			}
			message.append("\n");
		} catch (Exception e) {
			message.append("No options available\n\n");
		}

	}

