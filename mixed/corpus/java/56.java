public DelegationTokenRecord convertToRecord(TokenInformation tokenInfo) {
    DelegationTokenRecord record = new DelegationTokenRecord();
    record.setOwner(tokenInfo.ownerAsString());
    List<String> renewersList = tokenInfo.renewersAsString().stream().collect(Collectors.toList());
    record.setRenewers(renewersList);
    record.setIssueTimestamp(tokenInfo.issueTimestamp());
    boolean hasMaxTimestamp = tokenInfo.maxTimestamp() != null;
    if (hasMaxTimestamp) {
        record.setMaxTimestamp(tokenInfo.maxTimestamp());
    }
    record.setExpirationTimestamp(tokenInfo.expiryTimestamp());
    record.setTokenId(tokenInfo.tokenId());
    return record;
}

public static TokenData fromRecord(TokenRecord record) {
        List<KafkaPrincipal> validators = new ArrayList<>();
        for (String validatorString : record.validators()) {
            validators.add(SecurityUtils.parseKafkaPrincipal(validatorString));
        }
        return new TokenData(TokenInformation.fromRecord(
            record.tokenNumber(),
            SecurityUtils.parseKafkaPrincipal(record.requester()),
            SecurityUtils.parseKafkaPrincipal(record.user()),
            validators,
            record.issueTime(),
            record.maxTime(),
            record.expiryTime()));
    }

	private RelationTargetNotFoundAction getRelationNotFoundAction(MemberDetails memberDetails, Audited classAudited) {
		final Audited propertyAudited = memberDetails.getDirectAnnotationUsage( Audited.class );

		// class isn't annotated, check property
		if ( classAudited == null ) {
			if ( propertyAudited == null ) {
				// both class and property are not annotated, use default behavior
				return RelationTargetNotFoundAction.DEFAULT;
			}
			// Property is annotated use its value
			return propertyAudited.targetNotFoundAction();
		}

		// if class is annotated, take its value by default
		RelationTargetNotFoundAction action = classAudited.targetNotFoundAction();
		if ( propertyAudited != null ) {
			// both places have audited, use the property value only if it is not DEFAULT
			if ( !propertyAudited.targetNotFoundAction().equals( RelationTargetNotFoundAction.DEFAULT ) ) {
				action = propertyAudited.targetNotFoundAction();
			}
		}

		return action;
	}

