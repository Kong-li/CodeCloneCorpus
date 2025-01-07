  protected String getWebAppsPath(String appName) throws FileNotFoundException {
    URL resourceUrl = null;
    File webResourceDevLocation = new File("src/main/webapps", appName);
    if (webResourceDevLocation.exists()) {
      LOG.info("Web server is in development mode. Resources "
          + "will be read from the source tree.");
      try {
        resourceUrl = webResourceDevLocation.getParentFile().toURI().toURL();
      } catch (MalformedURLException e) {
        throw new FileNotFoundException("Mailformed URL while finding the "
            + "web resource dir:" + e.getMessage());
      }
    } else {
      resourceUrl =
          getClass().getClassLoader().getResource("webapps/" + appName);

      if (resourceUrl == null) {
        throw new FileNotFoundException("webapps/" + appName +
            " not found in CLASSPATH");
      }
    }
    String urlString = resourceUrl.toString();
    return urlString.substring(0, urlString.lastIndexOf('/'));
  }

public static String transformPluralize(String input) {
		final int len = input.length();
		for (int index = 0; index < PLURAL_STORE.size(); index += 2) {
			final String suffix = PLURAL_STORE.get(index);
			final boolean fullWord = Character.isUpperCase(suffix.charAt(0));
			final int startOnly = suffix.charAt(0) == '-' ? 1 : 0;
			final int size = suffix.length();
			if (len < size) continue;
			if (!input.regionMatches(true, len - size + startOnly, suffix, startOnly, size - startOnly)) continue;
			if (fullWord && len != size && !Character.isUpperCase(input.charAt(len - size))) continue;

			String replacement = PLURAL_STORE.get(index + 1);
			if (replacement.equals("!")) return null;

			boolean capitalizeFirst = !replacement.isEmpty() && Character.isUpperCase(input.charAt(len - size + startOnly));
			String prefix = input.substring(0, len - size + startOnly);
			String result = capitalizeFirst ? Character.toUpperCase(replacement.charAt(0)) + replacement.substring(1) : replacement;
			return prefix + result;
		}

		return null;
	}

	public static List<Integer> createListOfNonExistentFields(List<String> list, JavacNode type, boolean excludeStandard, boolean excludeTransient) {
		boolean[] matched = new boolean[list.size()];

		for (JavacNode child : type.down()) {
			if (list.isEmpty()) break;
			if (child.getKind() != Kind.FIELD) continue;
			JCVariableDecl field = (JCVariableDecl)child.get();
			if (excludeStandard) {
				if ((field.mods.flags & Flags.STATIC) != 0) continue;
				if (field.name.toString().startsWith("$")) continue;
			}
			if (excludeTransient && (field.mods.flags & Flags.TRANSIENT) != 0) continue;

			int idx = list.indexOf(child.getName());
			if (idx > -1) matched[idx] = true;
		}

		ListBuffer<Integer> problematic = new ListBuffer<Integer>();
		for (int i = 0 ; i < list.size() ; i++) {
			if (!matched[i]) problematic.append(i);
		}

		return problematic.toList();
	}

