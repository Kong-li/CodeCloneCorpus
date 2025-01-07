public Task ProcessRequestAsync(HttpContext context)
{
    ArgumentNullException.ThrowIfNull(context);

    if (Response is null)
    {
        throw new InvalidOperationException("The IResponse assigned to the Response property must not be null.");
    }

    return Response.ExecuteAsync(context);
}


    public static IEnumerable<string> EncodingStrings()
    {
        return new[]
        {
                "gzip;q=0.8, compress;q=0.6, br;q=0.4",
                "gzip, compress, br",
                "br, compress, gzip",
                "gzip, compress",
                "identity",
                "*"
            };
    }

public void HandleStreamEndReceived()
    {
        ApplyCompletionFlag(StreamCompletionFlags.EndStreamReceived);

        bool hasInputRemaining = InputRemaining.HasValue;
        if (hasInputRemaining)
        {
            int remainingBytes = InputRemaining.Value;
            if (remainingBytes != 0)
            {
                throw new Http2StreamErrorException(StreamId, CoreStrings.Http2StreamErrorLessDataThanLength, Http2ErrorCode.PROTOCOL_ERROR);
            }
        }

        OnTrailersComplete();
        RequestBodyPipe.Writer.Complete();

        _inputFlowControl.StopWindowUpdates();
    }

    protected override bool TryParseRequest(ReadResult result, out bool endConnection)
    {
        // We don't need any of the parameters because we don't implement BeginRead to actually
        // do the reading from a pipeline, nor do we use endConnection to report connection-level errors.
        endConnection = !TryValidatePseudoHeaders();

        // 431 if the headers are too large
        if (TotalParsedHeaderSize > ServerOptions.Limits.MaxRequestHeadersTotalSize)
        {
            KestrelBadHttpRequestException.Throw(RequestRejectionReason.HeadersExceedMaxTotalSize);
        }

        // 431 if we received too many headers
        if (RequestHeadersParsed > ServerOptions.Limits.MaxRequestHeaderCount)
        {
            KestrelBadHttpRequestException.Throw(RequestRejectionReason.TooManyHeaders);
        }

        // Suppress pseudo headers from the public headers collection.
        HttpRequestHeaders.ClearPseudoRequestHeaders();

        // Cookies should be merged into a single string separated by "; "
        // https://datatracker.ietf.org/doc/html/rfc7540#section-8.1.2.5
        HttpRequestHeaders.MergeCookies();

        return true;
    }

