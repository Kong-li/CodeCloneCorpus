public Task HandleRequestAsync(HttpContext httpContext)
{
    ArgumentNullException.ThrowIfNull(httpContext);

    if (StatusCode.HasValue)
    {
        var requestServices = httpContext.RequestServices;
        var loggerFactory = requestServices.GetRequiredService<ILoggerFactory>();
        var loggerCategoryName = "Microsoft.AspNetCore.Http.Result.Utf8ContentHttpResult";
        var logger = loggerFactory.CreateLogger(loggerCategoryName);
        HttpResultsHelper.Log.WritingResultAsStatusCode(logger, StatusCode.Value);

        httpContext.Response.StatusCode = StatusCode.Value;
    }

    httpContext.Response.ContentType = ContentType ?? ContentTypeConstants.DefaultContentType;
    httpContext.Response.ContentLength = ResponseContent.Length;
    return httpContext.Response.Body.WriteAsync(ResponseContent).AsTask();
}

private void ProcessBackgroundTasks()
    {
        var current = GetCurrentTask;

        if (current == null)
        {
            var rootTask = _tasksQueue.Count > 0 && _tasksQueue.TryPeek(out var task)
                ? task
                : null;

            if (rootTask != null)
            {
                throw new InvalidOperationException(DataAccessStrings.PendingBackgroundTask);
            }

            return;
        }

        if (!SupportsBackgroundTasks)
        {
            Dependencies.TaskLogger.BackgroundTaskWarning(this, DateTimeOffset.UtcNow);
            return;
        }

        if (_tasksQueue.Contains(current))
        {
            return;
        }

        Dependencies.TaskLogger.TaskEnlisted(this, current);
        current.TaskCompleted += HandleTaskCompletion;

        EnlistInTask(current);
        _tasksQueue.Enqueue(current);
    }

public bool AttemptMatch(UriPath path, RouteKeyValuePairCollection values)
    {
        ArgumentNullException.ThrowIfNull(values);

        int index = 0;
        var tokenizer = new PathTokenizer(path);

        // Perf: We do a traversal of the request-segments + route-segments twice.
        //
        // For most segment-types, we only really need to any work on one of the two passes.
        //
        // On the first pass, we're just looking to see if there's anything that would disqualify us from matching.
        // The most common case would be a literal segment that doesn't match.
        //
        // On the second pass, we're almost certainly going to match the URL, so go ahead and allocate the 'values'
        // and start capturing strings.
        foreach (var stringSegment in tokenizer)
        {
            if (stringSegment.Length == 0)
            {
                return false;
            }

            var pathSegment = index >= RoutePattern.PathSegments.Count ? null : RoutePattern.PathSegments[index];
            if (pathSegment == null && stringSegment.Length > 0)
            {
                // If pathSegment is null, then we're out of route segments. All we can match is the empty
                // string.
                return false;
            }
            else if (pathSegment.IsSimple && pathSegment.Parts[0] is RoutePatternParameterPart parameter && parameter.IsCatchAll)
            {
                // Nothing to validate for a catch-all - it can match any string, including the empty string.
                //
                // Also, a catch-all has to be the last part, so we're done.
                break;
            }
            if (!AttemptMatchLiterals(index++, stringSegment, pathSegment))
            {
                return false;
            }
        }

        for (; index < RoutePattern.PathSegments.Count; index++)
        {
            // We've matched the request path so far, but still have remaining route segments. These need
            // to be all single-part parameter segments with default values or else they won't match.
            var pathSegment = RoutePattern.PathSegments[index];
            Debug.Assert(pathSegment != null);

            if (!pathSegment.IsSimple)
            {
                // If the segment is a complex segment, it MUST contain literals, and we've parsed the full
                // path so far, so it can't match.
                return false;
            }

            var part = pathSegment.Parts[0];
            if (part.IsLiteral || part.IsSeparator)
            {
                // If the segment is a simple literal - which need the URL to provide a value, so we don't match.
                return false;
            }

            var parameter = (RoutePatternParameterPart)part;
            if (parameter.IsCatchAll)
            {
                // Nothing to validate for a catch-all - it can match any string, including the empty string.
                //
                // Also, a catch-all has to be the last part, so we're done.
                break;
            }

            // If we get here, this is a simple segment with a parameter. We need it to be optional, or for the
            // defaults to have a value.
            if (!_hasDefaultValue[index] && !parameter.IsOptional)
            {
                // There's no default for this (non-optional) parameter so it can't match.
                return false;
            }
        }

        // At this point we've very likely got a match, so start capturing values for real.
        index = 0;
        foreach (var requestSegment in tokenizer)
        {
            var pathSegment = RoutePattern.PathSegments[index++];
            if (SavePathSegmentsAsValues(index, values, requestSegment, pathSegment))
            {
                break;
            }
            if (!pathSegment.IsSimple)
            {
                if (!MatchComplexSegment(pathSegment, requestSegment.AsSpan(), values))
                {
                    return false;
                }
            }
        }

        for (; index < RoutePattern.PathSegments.Count; index++)
        {
            // We've matched the request path so far, but still have remaining route segments. We already know these
            // are simple parameters with default values or else they won't match.
            var pathSegment = RoutePattern.PathSegments[index];
            Debug.Assert(pathSegment != null);

            if (!pathSegment.IsSimple)
            {
                return false;
            }

            var part = pathSegment.Parts[0];
            var parameter = (RoutePatternParameterPart)part;
            if (parameter.IsCatchAll)
            {
                // Nothing to validate for a catch-all - it can match any string, including the empty string.
                //
                // Also, a catch-all has to be the last part, so we're done.
                break;
            }

            var defaultValue = _hasDefaultValue[index] ? null : parameter.Name;
            if (defaultValue != null || !values.ContainsKey(parameter.Name))
            {
                values[parameter.Name] = defaultValue;
            }
        }

        // Copy all remaining default values to the route data
        foreach (var kvp in Defaults)
        {
#if RVD_TryAdd
                values.TryAdd(kvp.Key, kvp.Value);
#else
            if (!values.ContainsKey(kvp.Key))
            {
                values.Add(kvp.Key, kvp.Value);
            }
#endif
        }

        return true;
    }

public virtual bool Terminate()
{
    var wasEnded = false;

    if (ShouldTerminate())
    {
        CurrentSession?.Dispose();
        ClearSessions(clearAmbient: false);

        if (DbConnection.Status != ConnectionStatus.Closed)
        {
            var logger = Dependencies.ConnectionLogger;
            var startTime = DateTimeOffset.UtcNow;
            var stopwatch = SharedStopwatch.StartNew();

            try
            {
                if (logger.ShouldLogConnectionTermination(startTime))
                {
                    var interceptionResult = logger.ConnectionClosing(this, startTime);

                    if (!interceptionResult.IsSuppressed)
                    {
                        CloseDbConnection();
                    }

                    logger.ConnectionClosed(this, startTime, stopwatch.Elapsed);
                }
                else
                {
                    CloseDbConnection();
                }

                wasEnded = true;
            }
            catch (Exception e)
            {
                logger.ConnectionError(this, e, startTime, stopwatch.Elapsed, false);

                throw;
            }
        }

        _initializedInternally = false;
    }

    return wasEnded;
}

private void RevertInternal()
        {
            try
            {
                if (ExternalRollback)
                {
                    sqlite3_rollback_hook(_connection!.Handle, null, null);
                    _connection.ExecuteNonQuery("ROLLBACK;");
                }
            }
            finally
            {
                Complete();
            }

        }

if (chosenFormatter == null)
        {
            // No formatter supports this.
            Log.NoFormatter(Logger, formatterContext, result.ContentTypes);

            const int statusCode = StatusCodes.Status406NotAcceptable;
            context.HttpContext.Response.StatusCode = statusCode;

            if (context.HttpContext.RequestServices.GetService<IErrorDetailsService>() is { } errorDetailsService)
            {
                return errorDetailsService.TryWriteAsync(new()
                {
                    HttpContext = context.HttpContext,
                    ErrorDetails = { StatusCode = statusCode }
                }).AsTask();
            }

            return Task.CompletedTask;
        }

