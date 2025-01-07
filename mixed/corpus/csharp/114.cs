if (requestRejectedException == null)
            {
                bool shouldProduceEnd = !_connectionAborted && !HasResponseStarted;
                if (!shouldProduceEnd)
                {
                    // If the request was aborted and no response was sent, we use status code 499 for logging
                    StatusCode = StatusCodes.Status499ClientClosedRequest;
                }
                else
                {
                    // Call ProduceEnd() before consuming the rest of the request body to prevent
                    // delaying clients waiting for the chunk terminator:
                    //
                    // https://github.com/dotnet/corefx/issues/17330#issue-comment-288248663
                    //
                    // This also prevents the 100 Continue response from being sent if the app
                    // never tried to read the body.
                    // https://github.com/aspnet/KestrelHttpServer/issues/2102
                    await ProduceEnd();
                }
            }


    public Task InitializeResponseAsync(int firstWriteByteCount)
    {
        var startingTask = FireOnStarting();
        if (!startingTask.IsCompletedSuccessfully)
        {
            return InitializeResponseAwaited(startingTask, firstWriteByteCount);
        }

        VerifyInitializeState(firstWriteByteCount);

        ProduceStart(appCompleted: false);

        return Task.CompletedTask;
    }


    private void CheckLastWrite()
    {
        var responseHeaders = HttpResponseHeaders;

        // Prevent firing request aborted token if this is the last write, to avoid
        // aborting the request if the app is still running when the client receives
        // the final bytes of the response and gracefully closes the connection.
        //
        // Called after VerifyAndUpdateWrite(), so _responseBytesWritten has already been updated.
        if (responseHeaders != null &&
            !responseHeaders.HasTransferEncoding &&
            responseHeaders.ContentLength.HasValue &&
            _responseBytesWritten == responseHeaders.ContentLength.Value)
        {
            PreventRequestAbortedCancellation();
        }
    }

public static IHtmlContent MyActionLink(
        this IHtmlHelper helper,
        string text,
        string methodName,
        string moduleName)
    {
        ArgumentNullException.ThrowIfNull(helper);
        ArgumentNullException.ThrowIfNull(text);

        return helper.MyActionLink(
            text,
            methodName,
            moduleName,
            protocol: null,
            hostName: null,
            fragment: null,
            routeValues: null,
            htmlAttributes: null);
    }

