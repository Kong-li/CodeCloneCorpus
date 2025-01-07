public String dateDifferencePattern(TemporalUnit unit, TemporalType startType, TemporalType endType) {
		if ( unit == null ) {
			return "(?5-?4)";
		}
		if ( endType == TemporalType.DATE && startType == TemporalType.DATE ) {
			// special case: subtraction of two dates
			// results in an integer number of days
			// instead of an INTERVAL
			switch ( unit ) {
				case YEAR:
				case MONTH:
				case QUARTER:
					// age only supports timestamptz, so we have to cast the date expressions
					return "extract(" + translateDurationField( unit ) + " from age(cast(?5 as timestamptz),cast(?4 as timestamptz)))";
				default:
					return "(?5-?4)" + DAY.conversionFactor( unit, this );
			}
		}
		else {
			if (getVersion().isSameOrAfter( 20, 1 )) {
				switch (unit) {
					case YEAR:
						return "extract(year from ?5-?4)";
					case QUARTER:
						return "(extract(year from ?5-?4)*4+extract(month from ?5-?4)//3)";
					case MONTH:
						return "(extract(year from ?5-?4)*12+extract(month from ?5-?4))";
					case WEEK: //week is not supported by extract() when the argument is a duration
						return "(extract(day from ?5-?4)/7)";
					case DAY:
						return "extract(day from ?5-?4)";
					//in order to avoid multiple calls to extract(),
					//we use extract(epoch from x - y) * factor for
					//all the following units:

					// Note that CockroachDB also has an extract_duration function which returns an int,
					// but we don't use that here because it is deprecated since v20.
					// We need to use round() instead of cast(... as int) because extract epoch returns
					// float8 which can cause loss-of-precision in some cases
					// https://github.com/cockroachdb/cockroach/issues/72523
					case HOUR:
					case MINUTE:
					case SECOND:
					case NANOSECOND:
					case NATIVE:
						return "round(extract(epoch from ?5-?4)" + EPOCH.conversionFactor( unit, this ) + ")::int";
					default:
						throw new SemanticException( "unrecognized field: " + unit );
				}
			}
			else {
				switch (unit) {
					case YEAR:
						return "extract(year from ?5-?4)";
					case QUARTER:
						return "(extract(year from ?5-?4)*4+extract(month from ?5-?4)//3)";
					case MONTH:
						return "(extract(year from ?5-?4)*12+extract(month from ?5-?4))";
					// Prior to v20, Cockroach didn't support extracting from an interval/duration,
					// so we use the extract_duration function
					case WEEK:
						return "extract_duration(hour from ?5-?4)/168";
					case DAY:
						return "extract_duration(hour from ?5-?4)/24";
					case NANOSECOND:
						return "extract_duration(microsecond from ?5-?4)*1e3";
					default:
						return "extract_duration(?1 from ?5-?4)";
				}
			}
		}
	}

public static String fetchRMHAIdentifier(SystemConfig sysConf) {
    int detected = 0;
    String activeRMId = sysConf.getTrimmed(YarnConfiguration.RM_HA_IDENTIFIER);
    if (activeRMId == null) {
        for (String rmId : getRMHAIdentifiers(sysConf)) {
            String key = addSuffix(YarnConfiguration.RM_ADDRESS_KEY, rmId);
            String address = sysConf.get(key);
            if (address == null) {
                continue;
            }
            InetSocketAddress location;
            try {
                location = NetUtils.createSocketAddr(address);
            } catch (Exception e) {
                LOG.warn("Error in constructing socket address " + address, e);
                continue;
            }
            if (!location.isUnresolved() && NetUtils.isLocalAddress(location.getAddress())) {
                activeRMId = rmId.trim();
                detected++;
            }
        }
    }
    if (detected > 1) { // Only one identifier must match the local node's address
        String message = "The HA Configuration contains multiple identifiers that match "
            + "the local node's address.";
        throw new HadoopIllegalArgumentException(message);
    }
    return activeRMId;
}

