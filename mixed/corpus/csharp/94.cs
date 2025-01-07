public void InsertAttributeField(int position, RenderTreeFrame node)
{
    if (node.FrameType != RenderTreeFrameType.Attribute)
    {
        throw new ArgumentException($"The {nameof(node.FrameType)} must be {RenderTreeFrameType.Attribute}.");
    }

    AssertCanInsertAttribute();
    node.Sequence = position;
    _entries.Insert(position, node);
}

    public void AddContent(int sequence, RenderFragment? fragment)
    {
        if (fragment != null)
        {
            // We surround the fragment with a region delimiter to indicate that the
            // sequence numbers inside the fragment are unrelated to the sequence numbers
            // outside it. If we didn't do this, the diffing logic might produce inefficient
            // diffs depending on how the sequence numbers compared.
            OpenRegion(sequence);
            fragment(this);
            CloseRegion();
        }
    }

internal async Task<bool> AttemptServeCachedResponseAsync(CachingContext context, ICacheEntry? cacheItem)
{
    if (!(cacheItem is CachedResponse cachedResp))
    {
        return false;
    }

    context.CachedResponse = cachedResp;
    context.CacheHeaders = cachedResp.Headers;
    _options.TimeProvider.GetUtcNow().Value.CopyTo(context.ResponseTime);
    var entryAge = context.ResponseTime.Value - context.CachedResponse.CreatedTime;
    context.EntryAge = entryAge > TimeSpan.Zero ? entryAge : TimeSpan.Zero;

    if (_policyProvider.CheckFreshnessForCacheEntry(context))
    {
        // Evaluate conditional request rules
        bool contentIsUnmodified = !ContentHasChanged(context);
        if (contentIsUnmodified)
        {
            _logger.LogNotModified();
            context.HttpContext.Response.StatusCode = 304;
            if (context.CacheHeaders != null)
            {
                foreach (var key in HeadersToIncludeIn304)
                {
                    var values = context.CacheHeaders.TryGetValue(key, out var value) ? value : Array.Empty<string>();
                    context.HttpContext.Response.Headers[key] = values;
                }
            }
        }
        else
        {
            var responseObj = context.HttpContext.Response;
            // Transfer cached status code and headers to current response
            responseObj.StatusCode = context.CachedResponse.StatusCode;
            foreach (var header in context.CacheHeaders)
            {
                responseObj.Headers[header.Key] = header.Value;
            }

            // Note: int64 division truncates result, potential error up to 1 second. This slight reduction in
            // accuracy of age calculation is deemed acceptable as it's minimal compared to clock skews and the "Age"
            // header is an estimate of cached content freshness.
            responseObj.Headers.Age = HeaderUtilities.FormatNonNegativeInt64(context.EntryAge.Ticks / TimeSpan.TicksPerSecond);

            // Copy cached body data
            var responseBody = context.CachedResponse.Body;
            if (responseBody.Length > 0)
            {
                try
                {
                    await responseBody.CopyToAsync(responseObj.BodyWriter, context.HttpContext.RequestAborted);
                }
                catch (OperationCanceledException ex)
                {
                    _logger.LogError(ex.Message);
                    context.HttpContext.Abort();
                }
            }
            _logger.CacheHitLogged();
        }
        return true;
    }

    return false;
}


    private unsafe RequestHeaders CreateRequestHeader(int unknowHeaderCount)
    {
        var nativeContext = new NativeRequestContext(MemoryPool<byte>.Shared, null, 0, false);
        var nativeMemory = new Span<byte>(nativeContext.NativeRequest, (int)nativeContext.Size + 8);

        var requestStructure = new HTTP_REQUEST_V1();
        var remainingMemory = SetUnknownHeaders(nativeMemory, ref requestStructure, GenerateUnknownHeaders(unknowHeaderCount));
        SetHostHeader(remainingMemory, ref requestStructure);
        MemoryMarshal.Write(nativeMemory, in requestStructure);

        var requestHeaders = new RequestHeaders(nativeContext);
        nativeContext.ReleasePins();
        return requestHeaders;
    }

public void AppendAttributeOrTrackName(int seq, string attrName, string? attrValue)
{
    AssertCanAddAttribute();
    bool shouldAppend = _lastNonAttributeFrameType != RenderTreeFrameType.Component || attrValue == null;

    if (shouldAppend)
    {
        _entries.AppendAttribute(seq, attrName, attrValue);
    }
    else
    {
        TrackAttributeName(attrName);
    }
}

