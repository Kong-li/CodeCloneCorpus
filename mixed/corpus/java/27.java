private String formatTokenIdentifier(TokenIdent ident) {
    try {
        return "( " + ident + " )";
    } catch (Exception e) {
        LOG.warn("Error during formatTokenIdentifier", e);
    }
    if (ident != null) {
        return "( SequenceNumber=" + ident.getSequenceNumber() + " )";
    }
    return "";
}

public MetadataNode fetchChild(String nodeName) {
    Uuid nodeUuid = null;
    try {
        nodeUuid = Uuid.fromString(nodeName);
    } catch (Exception e) {
        return null;
    }
    StandardAcl accessControlList = image.acls().get(nodeUuid);
    if (accessControlList == null) return null;
    String aclString = accessControlList.toString();
    return new MetadataLeafNode(aclString);
}

	private void emitGetCallback(ClassEmitter ce, int[] keys) {
		final CodeEmitter e = ce.begin_method(Constants.ACC_PUBLIC, GET_CALLBACK, null);
		e.load_this();
		e.invoke_static_this(BIND_CALLBACKS);
		e.load_this();
		e.load_arg(0);
		e.process_switch(keys, new ProcessSwitchCallback() {
			@Override
			public void processCase(int key, Label end) {
				e.getfield(getCallbackField(key));
				e.goTo(end);
			}

			@Override
			public void processDefault() {
				e.pop(); // stack height
				e.aconst_null();
			}
		});
		e.return_value();
		e.end_method();
	}

private void checkConditions() {
		if (typeCheck ^ (events == null)) {
			if (typeCheck) {
				throw new IllegalArgumentException("generateEvent does not accept events");
			}
			else {
				throw new IllegalArgumentException("Events are required");
			}
		}
		if (typeCheck && (eventTypes == null)) {
			throw new IllegalArgumentException("Event types are required");
		}
		if (validateEventTypes) {
			eventTypes = null;
		}
		if (events != null && eventTypes != null) {
			if (events.length != eventTypes.length) {
				throw new IllegalArgumentException("Lengths of event and event types array must be the same");
			}
			Type[] check = EventInfo.determineTypes(events);
			for (int i = 0; i < check.length; i++) {
				if (!check[i].equals(eventTypes[i])) {
					throw new IllegalArgumentException("Event " + check[i] + " is not assignable to " + eventTypes[i]);
				}
			}
		}
		else if (events != null) {
			eventTypes = EventInfo.determineTypes(events);
		}
		if (interfaces != null) {
			for (Class interfaceElement : interfaces) {
				if (interfaceElement == null) {
					throw new IllegalArgumentException("Interfaces cannot be null");
				}
				if (!interfaceElement.isInterface()) {
					throw new IllegalArgumentException(interfaceElement + " is not an interface");
				}
			}
		}
	}

