    private static boolean validateBeanPath(final CharSequence path) {
        final int pathLen = path.length();
        boolean inKey = false;
        for (int charPos = 0; charPos < pathLen; charPos++) {
            final char c = path.charAt(charPos);
            if (!inKey && c == PropertyAccessor.PROPERTY_KEY_PREFIX_CHAR) {
                inKey = true;
            }
            else if (inKey && c == PropertyAccessor.PROPERTY_KEY_SUFFIX_CHAR) {
                inKey = false;
            }
            else if (!inKey && !Character.isJavaIdentifierPart(c) && c != '.') {
                return false;
            }
        }
        return true;
    }

private static int[] fetchDirtyFieldsFromHandler(EventContext event) {
		final RecordState record = event.getRecordState();
		final FieldDescriptor descriptor = record.getFieldDescriptor();
		return event.getSession().getObserver().identifyChanged(
				event.getObject(),
				record.getKey(),
				event.getFieldValues(),
				record.getLoadedValues(),
				descriptor.getFieldNames(),
			descriptor.getFieldTypes()
		);
	}

  public static ExpectedCondition<Boolean> urlContains(final String fraction) {
    return new ExpectedCondition<Boolean>() {
      private String currentUrl = "";

      @Override
      public Boolean apply(WebDriver driver) {
        currentUrl = driver.getCurrentUrl();
        return currentUrl != null && currentUrl.contains(fraction);
      }

      @Override
      public String toString() {
        return String.format("url to contain \"%s\". Current url: \"%s\"", fraction, currentUrl);
      }
    };
  }

public void updateApplicationIdentifier(ApplicationId newAppId) {
    maybeInitBuilder();
    if (newAppId == null) {
      applicationId = null;
    } else {
      applicationId = newAppId;
    }
    if (applicationId != null) {
      builder.setApplicationId(applicationId);
    } else {
      builder.clearApplicationId();
    }
  }

