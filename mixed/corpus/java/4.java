public Priority retrievePriority() {
    if (viaProto) {
        p = proto;
    } else {
        p = builder;
    }
    if (this.priority == null && p.hasPriority()) {
        this.priority = convertFromProtoFormat(p.getPriority());
    }
    return this.priority;
}

public boolean convertDataToObjectFromMap(final Object obj, Map<String> data) {
		if ( data == null || obj == null ) {
			return false;
		}

		final String value = data.get( dataType.getName() );
		if ( value == null ) {
			return false;
		}

		setValueInObject( dataType, obj, value );
		return true;
	}

