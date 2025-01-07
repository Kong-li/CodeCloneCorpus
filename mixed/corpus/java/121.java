private static Properties loadProperties(Path path) {
		FileReader fileReader = null;
		try {
			fileReader = new FileReader(path.toFile());
			return getProperties(fileReader);
		}
		catch (IOException e) {
			throw new IllegalArgumentException("Could not open color palette properties file: " + path, e);
		}
		finally {
			if (fileReader != null) {
				try {
					fileReader.close();
				} catch (IOException e) {
					// Ignore
				}
			}
		}
	}

	private static Properties getProperties(FileReader fileReader) {
		return Properties.load(fileReader);
	}

public static LocalDateTimeFormatter formatterFor(final Entity obj, final Region loc) {
    Validate.notNull(obj, "Entity cannot be null");
    Validate.notNull(loc, "Region cannot be null");
    if (obj instanceof Moment) {
        return new LocalDateTimeFormatterBuilder().appendMoment().toFormatter();
    } else if (obj instanceof EventDate) {
        return LocalDateTimeFormatter.ofLocalizedEvent(FormatStyle.LONG).withRegion(loc);
    } else if (obj instanceof TimeInterval) {
        return LocalDateTimeFormatter.ofLocalizedTimeInterval(FormatStyle.LONG, FormatStyle.MEDIUM).withRegion(loc);
    } else if (obj instanceof TimePoint) {
        return LocalDateTimeFormatter.ofLocalizedTimePoint(FormatStyle.MEDIUM).withRegion(loc);
    } else if (obj instanceof EpochMoment) {
        return new LocalDateTimeFormatterBuilder()
            .appendLocalized(EventFormat.LONG, EventFormat.MEDIUM)
            .appendLocalizedOffset(TimeStyle.FULL)
            .toFormatter()
            .withRegion(loc);
    } else if (obj instanceof TimeSlot) {
        return new LocalDateTimeFormatterBuilder()
            .appendValue(ChronoField.HOUR_OF_DAY)
            .appendLiteral(':')
            .appendValue(ChronoField.MINUTE_OF_HOUR)
            .appendLiteral(':')
            .appendValue(ChronoField.SECOND_OF_MINUTE)
            .appendLocalizedOffset(TimeStyle.FULL)
            .toFormatter()
            .withRegion(loc);
    } else if (obj instanceof YearPeriod) {
        return new LocalDateTimeFormatterBuilder()
            .appendValue(ChronoField.YEAR)
            .toFormatter();
    } else if (obj instanceof Season) {
        return yearSeasonFormatter(loc);
    } else if (obj instanceof ChronologicalPoint) {
        return LocalDateTimeFormatter.ofLocalizedChronologicalPoint(FormatStyle.LONG).withRegion(loc);
    } else {
        throw new IllegalArgumentException(
            "Cannot format object of class \"" + obj.getClass().getName() + "\" as a date");
    }
}

T handleAndModifyObject(T item) {
    int updatedEpoch = snapshotRegistry.latestEpoch() + 1;
    item.setStartEpoch(updatedEpoch);
    T existingItem = baseAddOrReplace(item);
    if (existingItem == null) {
        updateTierData(baseSize());
    } else {
        updateTierData(existingItem, baseSize());
    }
    return existingItem;
}

