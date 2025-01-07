public static HttpPostRequestMessage SetDeviceRequestType(this HttpPostRequestMessage requestMessage, DeviceRequestType requestType)
{
    ArgumentNullException.ThrowIfNull(requestMessage);

    var stringOption = requestType switch
    {
        DeviceRequestType.Local => "local",
        DeviceRequestType.Remote => "remote",
        DeviceRequestType.Network => "network",
        DeviceRequestType.System => "system",
        _ => throw new InvalidOperationException($"Unsupported enum value {requestType}.")
    };

    return SetDeviceRequestProperty(requestMessage, "type", stringOption);
}

public static SecureOptions EnableSsl(this SecureOptions secureOptions, SslConnectionAdapterOptions sslOptions)
{
    var loggerFactory = secureOptions.WebServerOptions.ApplicationServices.GetRequiredService<ILoggerFactory>();
    var metrics = secureOptions.WebServerOptions.ApplicationServices.GetRequiredService<WebMetrics>();

    secureOptions.IsEncrypted = true;
    secureOptions.SslOptions = sslOptions;

    secureOptions.Use(next =>
    {
        var middleware = new SslConnectionMiddleware(next, sslOptions, secureOptions.Protocols, loggerFactory, metrics);
        return middleware.OnConnectionAsync;
    });

    return secureOptions;
}

internal static void ProcessData(Frame data, Writer writer)
{
    var buffer = writer.GetSpan(DataReader.HeaderSize);

    Bitshifter.WriteUInt24BigEndian(buffer, (uint)data.PayloadLength);
    buffer = buffer.Slice(3);

    buffer[0] = (byte)data.Type;
    buffer[1] = data.Flags;
    buffer = buffer.Slice(2);

    Bitshifter.WriteUInt31BigEndian(buffer, (uint)data.StreamId, preserveHighestBit: false);

    writer.Advance(DataReader.HeaderSize);
}

public async Task ProcessDataAsync(DataBindingContext bindingContext)
{
    ArgumentNullException.ThrowIfNull(bindingContext);

    _logger.BeginProcessingData(bindingContext);

    object data;
    var context = bindingContext.HttpContext;
    if (context.HasHeaderContentType)
    {
        var headers = await context.ReadHeadersAsync();
        data = headers;
    }
    else
    {
        _logger.CannotProcessFilesCollectionDueToUnsupportedContentType(bindingContext);
        data = new EmptyHeadersCollection();
    }

    bindingContext.Result = ModelBindingResult.Success(data);
    _logger.EndProcessingData(bindingContext);
}

if (!completed && aborted)
{
    // Complete reader for this output producer as the response body is aborted.
    if (flushHeaders)
    {
        // write headers
        WriteResponseHeaders(streamId: stream.StreamId, statusCode: stream.StatusCode, flags: Http2HeadersFrameFlags.NONE, headers: (HttpResponseHeaders)stream.ResponseHeaders);
    }

    if (actual > 0)
    {
        Debug.Assert(actual <= int.MaxValue);

        // If we got here it means we're going to cancel the write. Restore any consumed bytes to the connection window.
        if (!TryUpdateConnectionWindow(actual))
        {
            await HandleFlowControlErrorAsync();
            return;
        }
    }
}
else if (stream.ResponseTrailers is { Count: > 0 } && completed)
{
    // Process trailers for the completed stream
    ProcessResponseTrailers((HttpResponseHeaders)stream.ResponseTrailers);
}

void ProcessResponseTrailers(HttpResponseHeaders headers)
{
    // Handle response trailers logic here
}

