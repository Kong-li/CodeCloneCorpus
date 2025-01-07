  public long valueAt(double probability) {
    int rangeFloor = floorIndex(probability);

    double segmentProbMin = getRankingAt(rangeFloor);
    double segmentProbMax = getRankingAt(rangeFloor + 1);

    long segmentMinValue = getDatumAt(rangeFloor);
    long segmentMaxValue = getDatumAt(rangeFloor + 1);

    // If this is zero, this object is based on an ill-formed cdf
    double segmentProbRange = segmentProbMax - segmentProbMin;
    long segmentDatumRange = segmentMaxValue - segmentMinValue;

    long result = (long) ((probability - segmentProbMin) / segmentProbRange * segmentDatumRange)
        + segmentMinValue;

    return result;
  }

  public void addNextValue(Object val) {
    String valCountStr = val.toString();
    int pos = valCountStr.lastIndexOf("\t");
    String valStr = valCountStr;
    String countStr = "1";
    if (pos >= 0) {
      valStr = valCountStr.substring(0, pos);
      countStr = valCountStr.substring(pos + 1);
    }

    Long count = (Long) this.items.get(valStr);
    long inc = Long.parseLong(countStr);

    if (count == null) {
      count = inc;
    } else {
      count = count.longValue() + inc;
    }
    items.put(valStr, count);
  }

private static boolean checkSubordinateTemp(EventRoot source, Object sub, String itemName) {
		if ( isProxyInstance( sub ) ) {
			// a proxy is always non-temporary
			// and ForeignKeys.isTransient()
			// is not written to expect a proxy
			// TODO: but the proxied entity might have been deleted!
			return false;
		}
		else {
			final EntityInfo info = source.getActiveContextInternal().getEntry( sub );
			if ( info != null ) {
				// if it's associated with the context
				// we are good, even if it's not yet
				// inserted, since ordering problems
				// are detected and handled elsewhere
				return info.getStatus().isDeletedOrGone();
			}
			else {
				// TODO: check if it is a merged item which has not yet been flushed
				// Currently this throws if you directly reference a new temporary
				// instance after a call to merge() that results in its managed copy
				// being scheduled for insertion, if the insert has not yet occurred.
				// This is not terrible: it's more correct to "swap" the reference to
				// point to the managed instance, but it's probably too heavy-handed.
				return checkTemp( itemName, sub, null, source );
			}
		}
	}

