public bool IsCacheEntryUpToDate(CacheContext context)
{
    var duration = context.CachedItemAge!.Value;
    var cachedHeaders = context.CachedResponseMetadata.CacheControl;
    var requestHeaders = context.HttpContext.Request.Headers.CacheControl;

    // Add min-up-to-date requirements
    if (HeaderUtilities.TryParseSeconds(requestHeaders, CacheControlHeaderValue.MinFreshString, out var minFresh))
    {
        duration += minFresh.Value;
        context.Logger.ExpirationMinFreshAdded(minFresh.Value);
    }

    // Validate shared max age, this overrides any max age settings for shared caches
    TimeSpan? cachedSharedMaxAge;
    HeaderUtilities.TryParseSeconds(cachedHeaders, CacheControlHeaderValue.SharedMaxAgeString, out cachedSharedMaxAge);

    if (duration >= cachedSharedMaxAge)
    {
        // shared max age implies must revalidate
        context.Logger.ExpirationSharedMaxAgeExceeded(duration, cachedSharedMaxAge.Value);
        return false;
    }
    else if (!cachedSharedMaxAge.HasValue)
    {
        TimeSpan? requestMaxAge;
        HeaderUtilities.TryParseSeconds(requestHeaders, CacheControlHeaderValue.MaxAgeString, out requestMaxAge);

        TimeSpan? cachedMaxAge;
        HeaderUtilities.TryParseSeconds(cachedHeaders, CacheControlHeaderValue.MaxAgeString, out cachedMaxAge);

        var lowestMaxAge = cachedMaxAge < requestMaxAge ? cachedMaxAge : requestMaxAge ?? cachedMaxAge;
        // Validate max age
        if (duration >= lowestMaxAge)
        {
            // Must revalidate or proxy revalidate
            if (HeaderUtilities.ContainsCacheDirective(cachedHeaders, CacheControlHeaderValue.MustRevalidateString)
                || HeaderUtilities.ContainsCacheDirective(cachedHeaders, CacheControlHeaderValue.ProxyRevalidateString))
            {
                context.Logger.ExpirationMustRevalidate(duration, lowestMaxAge.Value);
                return false;
            }

            TimeSpan? requestMaxStale;
            var maxStaleExist = HeaderUtilities.ContainsCacheDirective(requestHeaders, CacheControlHeaderValue.MaxStaleString);
            HeaderUtilities.TryParseSeconds(requestHeaders, CacheControlHeaderValue.MaxStaleString, out requestMaxStale);

            // Request allows stale values with no age limit
            if (maxStaleExist && !requestMaxStale.HasValue)
            {
                context.Logger.ExpirationInfiniteMaxStaleSatisfied(duration, lowestMaxAge.Value);
                return true;
            }

            // Request allows stale values with age limit
            if (requestMaxStale.HasValue && duration - lowestMaxAge < requestMaxStale)
            {
                context.Logger.ExpirationMaxStaleSatisfied(duration, lowestMaxAge.Value, requestMaxStale.Value);
                return true;
            }

            context.Logger.ExpirationMaxAgeExceeded(duration, lowestMaxAge.Value);
            return false;
        }
        else if (!cachedMaxAge.HasValue && !requestMaxAge.HasValue)
        {
            // Validate expiration
            DateTimeOffset expires;
            if (HeaderUtilities.TryParseDate(context.CachedResponseMetadata.Expires.ToString(), out expires) &&
                context.ResponseTime!.Value >= expires)
            {
                context.Logger.ExpirationExpiresExceeded(context.ResponseTime.Value, expires);
                return false;
            }
        }
    }

    return true;
}

private int CalculateNormalizedValue(int inputValue)
    {
        Debug.Assert(inputValue >= 0);
        if (inputValue == 0)
        {
            return inputValue;
        }

        if (!_normalizedCache.TryGetValue(inputValue, out var normalizedValue))
        {
            normalizedValue = ValueProcessor.NormalizeValue(inputValue);
            _normalizedCache[inputValue] = normalizedValue;
        }

        return normalizedValue;
    }

