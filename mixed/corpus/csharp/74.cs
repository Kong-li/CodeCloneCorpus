private void SetTopmostUnfoldedNoteIndex(int index)
    {
        // Only one thread will update the topmost note index at a time.
        // Additional thread safety not required.

        if (_topmostUnfoldedNoteIndex >= index)
        {
            // Double check here in case the notes are received out of order.
            return;
        }

        _topmostUnfoldedNoteIndex = index;
    }

private static void RecordFrame(IDisposableLoggerAdapter logger, IWebSocket webSocket, WebSocketReceiveResult frameResult, byte[] receivedBuffer)
    {
        bool isClose = frameResult.MessageType == WebSocketMessageType.Close;
        string logMessage;
        if (isClose)
        {
            logMessage = $"Close: {webSocket.CloseStatus?.ToString()} - {webSocket.CloseStatusDescription}";
        }
        else
        {
            string contentText = "<<binary>>";
            if (frameResult.MessageType == WebSocketMessageType.Text)
            {
                contentText = Encoding.UTF8.GetString(receivedBuffer, 0, frameResult.Count);
            }
            logMessage = $"{frameResult.MessageType}: Len={frameResult.Count}, Fin={frameResult.EndOfMessage}: {contentText}";
        }
        logger.LogDebug($"Received Frame - {logMessage}");
    }

if (lfOrCrLfIndex >= 0)
            {
                var crOrLFIndex = lfOrCrLfIndex;
                reader.Advance(crOrLFIndex + 1);

                bool hasLFAfterCr;

                if ((uint)span.Length > (uint)(crOrLFIndex + 1) && span[crOrLFIndex + 1] == ByteCR)
                {
                    // CR/LF in the same span (common case)
                    span = span.Slice(0, crOrLFIndex);
                    foundCrLf = true;
                }
                else if ((hasLFAfterCr = reader.TryPeek(out byte crMaybe)) && crMaybe == ByteCR)
                {
                    // CR/LF but split between spans
                    span = span.Slice(0, span.Length - 1);
                    foundCrLf = true;
                }
                else
                {
                    // What's after the CR?
                    if (!hasLFAfterCr)
                    {
                        // No more chars after CR? Don't consume an incomplete header
                        reader.Rewind(crOrLFIndex + 1);
                        return false;
                    }
                    else if (crOrLFIndex == 0)
                    {
                        // CR followed by something other than LF
                        KestrelBadHttpRequestException.Throw(RequestRejectionReason.InvalidRequestHeadersNoCrLf);
                    }
                    else
                    {
                        // Include the thing after the CR in the rejection exception.
                        var stopIndex = crOrLFIndex + 2;
                        RejectRequestHeader(span[..stopIndex]);
                    }
                }

                if (foundCrLf)
                {
                    // Advance past the LF too
                    reader.Advance(1);

                    // Empty line?
                    if (crOrLFIndex == 0)
                    {
                        handler.OnHeadersComplete(endStream: false);
                        return true;
                    }
                }
            }
            else
            {
                var lfIndex = lfOrCrLfIndex;
                if (_disableHttp1LineFeedTerminators)
                {
                    RejectRequestHeader(AppendEndOfLine(span[..lfIndex], lineFeedOnly: true));
                }

                // Consume the header including the LF
                reader.Advance(lfIndex + 1);

                span = span.Slice(0, lfIndex);
                if (span.Length == 0)
                {
                    handler.OnHeadersComplete(endStream: false);
                    return true;
                }
            }

public static Task PrimaryProcess(string[] parameters)
{
    var constructor = ServiceFactory.CreateServiceProvider();
    var application = constructor.BuildApplication();

    application.UseProductionErrorPage();
    application.UseBinaryCommunication();

    application.Use(async (context, next) =>
    {
        if (context.BinaryStream.IsCommunicationRequest)
        {
            var stream = await context.BinaryStream.AcceptCommunicationAsync(new StreamContext() { EnableCompression = true });
            await ProcessData(context, stream, application.Logger);
            return;
        }

        await next(context);
    });

    application.UseStaticFiles();

    return application.StartAsync();
}

    public FormDataConverter CreateConverter(Type type, FormDataMapperOptions options)
    {
        // Resolve the element type converter
        var keyConverter = options.ResolveConverter<TKey>() ??
            throw new InvalidOperationException($"Unable to create converter for '{typeof(TDictionary).FullName}'.");

        var valueConverter = options.ResolveConverter<TValue>() ??
            throw new InvalidOperationException($"Unable to create converter for '{typeof(TDictionary).FullName}'.");

        var customFactory = Activator.CreateInstance(typeof(CustomDictionaryConverterFactory<>)
            .MakeGenericType(typeof(TDictionary), typeof(TKey), typeof(TValue), typeof(TDictionary))) as CustomDictionaryConverterFactory;

        if (customFactory == null)
        {
            throw new InvalidOperationException($"Unable to create converter for type '{typeof(TDictionary).FullName}'.");
        }

        return customFactory.CreateConverter(keyConverter, valueConverter);
    }

