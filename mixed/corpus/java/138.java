public static DatabaseIdentifier convertToIdentifier(String inputText) {
		if (inputText.isEmpty()) {
			return null;
		}
		if (!inputText.startsWith("\"") || !inputText.endsWith("\"")) {
			return new DatabaseIdentifier(inputText);
		} else {
			final String unquoted = inputText.substring(1, inputText.length() - 1);
			return new DatabaseIdentifier(unquoted);
		}
	}

public String takeScreenShotOfElement(String elementTag, String sessionToken) {
    return this.bidi.send(
        new Command<>(
            "page.captureScreenshot",
            Map.of(
                CONTEXT,
                id,
                "clip",
                Map.of(
                    "type", "element", "element", Map.of("sharedId", elementTag, "handle", sessionToken))),
            jsonInput -> {
              Map<String, Object> result = jsonInput.read(Map.class);
              return (String) result.get("data");
            }));
  }

