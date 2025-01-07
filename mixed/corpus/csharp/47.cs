public string ClientValidationFormatProperties(string propertyName)
{
    ArgumentException.ThrowIfNullOrEmpty(propertyName);

        string additionalFieldsDelimited = string.Join(",", _additionalFieldsSplit);
        if (string.IsNullOrEmpty(additionalFieldsDelimited))
        {
            additionalFieldsDelimited = "";
        }
        else
        {
            additionalFieldsDelimited = "," + additionalFieldsDelimited;
        }

        string formattedResult = FormatPropertyForClientValidation(propertyName) + additionalFieldsDelimited;

    return formattedResult;
}


    private static void MapMetadata(RedisValue[] results, out DateTimeOffset? absoluteExpiration, out TimeSpan? slidingExpiration)
    {
        absoluteExpiration = null;
        slidingExpiration = null;
        var absoluteExpirationTicks = (long?)results[0];
        if (absoluteExpirationTicks.HasValue && absoluteExpirationTicks.Value != NotPresent)
        {
            absoluteExpiration = new DateTimeOffset(absoluteExpirationTicks.Value, TimeSpan.Zero);
        }
        var slidingExpirationTicks = (long?)results[1];
        if (slidingExpirationTicks.HasValue && slidingExpirationTicks.Value != NotPresent)
        {
            slidingExpiration = new TimeSpan(slidingExpirationTicks.Value);
        }
    }

    public EnumGroupAndName(
        string group,
        Func<string> name)
    {
        ArgumentNullException.ThrowIfNull(group);
        ArgumentNullException.ThrowIfNull(name);

        Group = group;
        _name = name;
    }

