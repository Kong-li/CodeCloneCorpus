public static String unescapeCustomCode(final String inputText) {

        if (inputText == null) {
            return null;
        }

        StringBuilder customBuilder = null;

        final int startOffset = 0;
        final int maxLimit = inputText.length();

        int currentOffset = startOffset;
        int referencePoint = startOffset;

        for (int index = startOffset; index < maxLimit; index++) {

            final char character = inputText.charAt(index);

            /*
             * Check the need for an unescape operation at this point
             */

            if (character != CUSTOM_ESCAPE_PREFIX || (index + 1) >= maxLimit) {
                continue;
            }

            int codeValue = -1;

            if (character == CUSTOM_ESCAPE_PREFIX) {

                final char nextChar = inputText.charAt(index + 1);

                if (nextChar == CUSTOM_ESCAPE_UHEXA_PREFIX2) {
                    // This can be a uhexa escape, we need exactly four more characters

                    int startHexIndex = index + 2;
                    // First, discard any additional 'u' characters, which are allowed
                    while (startHexIndex < maxLimit) {
                        final char cf = inputText.charAt(startHexIndex);
                        if (cf != CUSTOM_ESCAPE_UHEXA_PREFIX2) {
                            break;
                        }
                        startHexIndex++;
                    }
                    int hexStart = startHexIndex;
                    // Parse the hexadecimal digits
                    while (hexStart < (index + 5) && hexStart < maxLimit) {
                        final char cf = inputText.charAt(hexStart);
                        if (!((cf >= '0' && cf <= '9') || (cf >= 'A' && cf <= 'F') || (cf >= 'a' && cf <= 'f'))) {
                            break;
                        }
                        hexStart++;
                    }

                    if ((hexStart - index) < 5) {
                        // We weren't able to consume the required four hexa chars, leave it as slash+'u', which
                        // is invalid, and let the corresponding Java parser fail.
                        index++;
                        continue;
                    }

                    codeValue = parseIntFromReference(inputText, index + 2, hexStart, 16);

                    // Fast-forward to the first char after the parsed codepoint
                    referencePoint = hexStart - 1;

                    // Don't continue here, just let the unescape code below do its job

                } else if (nextChar == CUSTOM_ESCAPE_PREFIX && index + 2 < maxLimit && inputText.charAt(index + 2) == CUSTOM_ESCAPE_UHEXA_PREFIX2){
                    // This unicode escape is actually escaped itself, so we don't need to perform the real unescaping,
                    // but we need to merge the "\\" into "\"

                    if (customBuilder == null) {
                        customBuilder = new StringBuilder(maxLimit + 5);
                    }

                    if (index - currentOffset > 0) {
                        customBuilder.append(inputText, currentOffset, index);
                    }

                    customBuilder.append('\\');

                    currentOffset = index + 3;

                    index++;
                    continue;

                } else {

                    // Other escape sequences will not be processed in this unescape step.
                    index++;
                    continue;

                }

            }


            /*
             * At this point we know for sure we will need some kind of unescape, so we
             * can increase the offset and initialize the string builder if needed, along with
             * copying to it all the contents pending up to this point.
             */

            if (customBuilder == null) {
                customBuilder = new StringBuilder(maxLimit + 5);
            }

            if (index - currentOffset > 0) {
                customBuilder.append(inputText, currentOffset, index);
            }

            index = referencePoint;
            currentOffset = index + 1;

            /*
             * --------------------------
             *
             * Peform the real unescape
             *
             * --------------------------
             */

            if (codeValue > '\uFFFF') {
                customBuilder.append(Character.toChars(codeValue));
            } else {
                customBuilder.append((char)codeValue);
            }

        }


        /*
         * -----------------------------------------------------------------------------------------------
         * Final cleaning: return the original String object if no unescape was actually needed. Otherwise
         *                 append the remaining escaped text to the string builder and return.
         * -----------------------------------------------------------------------------------------------
         */

        if (customBuilder == null) {
            return inputText;
        }

        if (maxLimit - currentOffset > 0) {
            customBuilder.append(inputText, currentOffset, maxLimit);
        }

        return customBuilder.toString();

    }

private void lazyInit() throws IOException {
    if (stream != null) {
      return;
    }

    // Load current value.
    byte[] info = null;
    try {
      info = Files.toByteArray(config);
    } catch (FileNotFoundException fnfe) {
      // Expected - this will use default value.
    }

    if (info != null && info.length != 0) {
      if (info.length != Shorts.BYTES) {
        throw new IOException("Config " + config + " had invalid length: " +
            info.length);
      }
      state = Shorts.fromByteArray(info);
    } else {
      state = initVal;
    }

    // Now open file for future writes.
    RandomAccessFile writer = new RandomAccessFile(config, "rw");
    try {
      channel = writer.getChannel();
    } finally {
      if (channel == null) {
        IOUtils.closeStream(writer);
      }
    }
  }

public static void attachPackage(ClassLoaderManager clsMgr, String moduleName, MetadataBuildingContext context) {
		final PackageInfo pack = clsMgr.findPackageByName(moduleName);
		if (pack == null) {
			return;
		}
		final ClassDetails packageMetadataClassDetails =
				context.getClassDetailsProvider().getClassDetails(pack.getFullName() + ".package-info");

		GeneratorBinder.registerGlobalGenerators(packageMetadataClassDetails, context);

		bindTypeDescriptorRegistrations(packageMetadataClassDetails, context);
		bindEmbeddableInstantiatorRegistrations(packageMetadataClassDetails, context);
		bindUserTypeRegistrations(packageMetadataClassDetails, context);
		bindCompositeUserTypeRegistrations(packageMetadataClassDetails, context);
		bindConverterRegistrations(packageMetadataClassDetails, context);

		bindQueries(packageMetadataClassDetails, context);
		bindFilterDefs(packageMetadataClassDetails, context);
	}

