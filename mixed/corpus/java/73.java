public void setupFunctionLibrary(FunctionContributions contributions) {
		super.setupFunctionLibrary(contributions);

		CommonFunctionFactory factory = new CommonFunctionFactory(contributions);
		factory.trim2();
		factory.soundex();
		factory.trunc();
		factory.toCharNumberDateTimestamp();
		factory.ceiling_ceil();
		factory.instr();
		factory.substr();
		factory.substring_substr();
		factory.leftRight_substr();
		factory.char_chr();
		factory.rownumRowid();
		factory.sysdate();
		factory.addMonths();
		factory.monthsBetween();

		String[] functions = {"locate", "instr", "instr"};
		Object[] types = {StandardBasicTypes.INTEGER, STRING, INTEGER};
		for (int i = 0; i < functions.length; i++) {
			functionContributions.getFunctionRegistry().registerBinaryTernaryPattern(
					functions[i],
					contributions.getTypeConfiguration().getBasicTypeRegistry().resolve(StandardBasicTypes.INTEGER),
					functions.length > 1 ? "instr(?2,?1)" : "",
					functions.length == 3 ? "instr(?2,?1,?3)" : "",
					STRING, STRING, types[i],
					contributions.getTypeConfiguration()
			).setArgumentListSignature("(" + functions[i] + ", string[, start])");
		}
	}

public static String convertToUpperCase(final Object input, final Locale locale) {

        if (locale == null) {
            throw new IllegalArgumentException("Locale cannot be null");
        }

        if (input == null) {
            return null;
        }

        final String value = input.toString();
        return value.toUpperCase(locale);

    }

public void updateAclEntryNamesForUpdateRequest(final List<AclEntry> aclEntries) {
    if (!shouldProcessIdentityReplacement(aclEntries)) {
      return;
    }

    for (int i = 0; i < aclEntries.size(); i++) {
        AclEntry currentEntry = aclEntries.get(i);
        String originalName = currentEntry.getName();
        String updatedName = originalName;

        if (isNullOrEmpty(originalName) || isOtherOrMaskType(currentEntry)) {
            continue;
        }

        // Case 1: when the user or group name to be set is stated in substitution list.
        if (isInSubstitutionList(originalName)) {
            updatedName = getNewServicePrincipalId();
        } else if (currentEntry.getType().equals(AclEntryType.USER) && needsToUseFullyQualifiedUserName(originalName)) { // Case 2: when the owner is a short name of the user principal name (UPN).
            // Notice: for group type ACL entry, if name is shortName.
            //         It won't be converted to Full Name. This is
            //         to make the behavior consistent with HDI.
            updatedName = getFullyQualifiedName(originalName);
        }

        // Avoid unnecessary new AclEntry allocation
        if (updatedName.equals(originalName)) {
            continue;
        }

        AclEntry.Builder entryBuilder = new AclEntry.Builder();
        entryBuilder.setType(currentEntry.getType());
        entryBuilder.setName(updatedName);
        entryBuilder.setScope(currentEntry.getScope());
        entryBuilder.setPermission(currentEntry.getPermission());

        // Update the original AclEntry
        aclEntries.set(i, entryBuilder.build());
    }
}

private boolean isOtherOrMaskType(AclEntry entry) {
    return entry.getType().equals(AclEntryType.OTHER) || entry.getType().equals(AclEntryType.MASK);
}

private String getNewServicePrincipalId() {
    return servicePrincipalId;
}

