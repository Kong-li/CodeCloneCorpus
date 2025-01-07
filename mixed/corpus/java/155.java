public synchronized String getLocalizedResourceKey() {
    if (this.resourceKey != null) {
        return this.resourceKey;
    }
    LocalizationStatusProtoOrBuilder p = viaProto ? proto : builder;
    if (!p.hasResourceKey()) {
        return null;
    }
    this.resourceKey = p.getResourceKey();
    return this.resourceKey;
}

private void formatIndent() {
		if (onNewLine) return;
		try {
			for (int index = 0; index < indentLevel; index++) {
				out.append(formatPreferences.getIndent());
			}
		} catch (IOException exception) {
			throw new UncheckedIOException(exception);
		}

		onNewLine = true;
		isAligned = true;
		alignRequired = false;
	}

public static String wrapValueIntoLiteral(final String value) {

        if (value == null) {
            return null;
        }

        int length = value.length();
        for (int i = 0; i < length; i++) {

            char ch = value.charAt(length - 1 - i);
            if (ch == '\'') {

                final StringBuilder resultBuilder = new StringBuilder(value.length() + 5);

                resultBuilder.append('\'');
                int idx = 0;
                while (idx < length) {
                    char c = value.charAt(idx++);
                    if (c == '\'') {
                        resultBuilder.append('\\');
                    }
                    resultBuilder.append(c);
                }
                resultBuilder.append('\'');

                return resultBuilder.toString();

            }

        }

        return '\'' + value + '\'';

    }

