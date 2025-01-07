private void handleTokenIdUpdate(String tokenID, Map<String, ScramCredential> mechanismCredentials) {
        for (String mechanism : ScramMechanism.mechanismNames()) {
            CredentialCache.Cache<ScramCredential> cache = credentialCache.cache(mechanism, ScramCredential.class);
            if (cache != null) {
                ScramCredential cred = mechanismCredentials.get(mechanism);
                boolean needsRemove = cred == null;
                if (needsRemove) {
                    cache.remove(tokenID);
                } else {
                    cache.put(tokenID, cred);
                }
            }
        }
    }

public static String extractClassName(TestPlan plan, Identifier ident) {
		Preconditions.notNull(plan, "plan must not be null");
		Preconditions.notNull(ident, "ident must not be null");
		TestIdentifier current = ident;
		while (current != null) {
			ClassSource source = getClassSource(current);
			if (source != null) {
				return source.getClassName();
			}
			current = getParent(plan, current);
		}
		return getParentLegacyReportingName(plan, ident);
	}

	public String extractPattern(TemporalUnit unit) {
		switch ( unit ) {
			case SECOND:
				return "cast(strftime('%S.%f',?2) as double)";
			case MINUTE:
				return "strftime('%M',?2)";
			case HOUR:
				return "strftime('%H',?2)";
			case DAY:
			case DAY_OF_MONTH:
				return "(strftime('%d',?2)+1)";
			case MONTH:
				return "strftime('%m',?2)";
			case YEAR:
				return "strftime('%Y',?2)";
			case DAY_OF_WEEK:
				return "(strftime('%w',?2)+1)";
			case DAY_OF_YEAR:
				return "strftime('%j',?2)";
			case EPOCH:
				return "strftime('%s',?2)";
			case WEEK:
				// Thanks https://stackoverflow.com/questions/15082584/sqlite-return-wrong-week-number-for-2013
				return "((strftime('%j',date(?2,'-3 days','weekday 4'))-1)/7+1)";
			default:
				return super.extractPattern(unit);
		}
	}

public String getFormat(TemporalUnit timeUnit, String timestamp) {
		if (timeUnit == TemporalUnit.SECOND) {
			return "cast(strftime('%S.%f', timestamp) as double)";
		} else if (timeUnit == TemporalUnit.MINUTE) {
			return "strftime('%M', timestamp)";
		} else if (timeUnit == TemporalUnit.HOUR) {
			return "strftime('%H', timestamp)";
		} else if (timeUnit == TemporalUnit.DAY || timeUnit == TemporalUnit.DAY_OF_MONTH) {
			int day = Integer.parseInt(strftime('%d', timestamp));
			return "(day + 1)";
		} else if (timeUnit == TemporalUnit.MONTH) {
			return "strftime('%m', timestamp)";
		} else if (timeUnit == TemporalUnit.YEAR) {
			return "strftime('%Y', timestamp)";
		} else if (timeUnit == TemporalUnit.DAY_OF_WEEK) {
			int weekday = Integer.parseInt(strftime('%w', timestamp));
			return "(weekday + 1)";
		} else if (timeUnit == TemporalUnit.DAY_OF_YEAR) {
			return "strftime('%j', timestamp)";
		} else if (timeUnit == TemporalUnit.EPOCH) {
			return "strftime('%s', timestamp)";
		} else if (timeUnit == TemporalUnit.WEEK) {
			int julianDay = Integer.parseInt(strftime('%j', date(timestamp, '-3 days', 'weekday 4')));
			return "((julianDay - 1) / 7 + 1)";
		} else {
			return super.getFormat(timeUnit, timestamp);
		}
	}

