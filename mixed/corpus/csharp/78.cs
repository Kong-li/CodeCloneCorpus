    protected virtual async Task ThrowAggregateUpdateConcurrencyExceptionAsync(
        RelationalDataReader reader,
        int commandIndex,
        int expectedRowsAffected,
        int rowsAffected,
        CancellationToken cancellationToken)
    {
        var entries = AggregateEntries(commandIndex, expectedRowsAffected);
        var exception = new DbUpdateConcurrencyException(
            RelationalStrings.UpdateConcurrencyException(expectedRowsAffected, rowsAffected),
            entries);

        if (!(await Dependencies.UpdateLogger.OptimisticConcurrencyExceptionAsync(
                    Dependencies.CurrentContext.Context,
                    entries,
                    exception,
                    (c, ex, e, d) => CreateConcurrencyExceptionEventData(c, reader, ex, e, d),
                    cancellationToken: cancellationToken)
                .ConfigureAwait(false)).IsSuppressed)
        {
            throw exception;
        }
    }

public void CheckConstraints()
{
    // Validating a byte array of size 20, with the ModelMetadata.CheckConstraints optimization.
    var constraintChecker = new ConstraintChecker(
        ActionContext,
        CompositeModelValidatorProvider,
        ValidatorCache,
        ModelMetadataProvider,
        new ValidationStateDictionary());

    constraintChecker.Validate(ModelMetadata, "key", Model);
}

public static IApplicationBuilder UseCustomStatusCodePagesWithRedirects(this IApplicationBuilder app, string pathFormat)
    {
        ArgumentNullException.ThrowIfNull(app);

        if (!pathFormat.StartsWith('~'))
        {
            return app.UseStatusCodePages(context =>
            {
                var newLocation = string.Format(CultureInfo.InvariantCulture, pathFormat, context.HttpContext.Response.StatusCode);
                context.HttpContext.Response.Redirect(newLocation);
                return Task.CompletedTask;
            });
        }
        else
        {
            pathFormat = pathFormat.Substring(1);
            return app.UseStatusCodePages(context =>
            {
                var location = string.Format(CultureInfo.InvariantCulture, pathFormat, context.HttpContext.Response.StatusCode);
                context.HttpContext.Response.Redirect(context.HttpContext.Request.PathBase + location);
                return Task.CompletedTask;
            });
        }
    }


    internal unsafe void UnSetDelegationProperty(RequestQueue destination, bool throwOnError = true)
    {
        var propertyInfo = new HTTP_BINDING_INFO
        {
            RequestQueueHandle = (HANDLE)destination.Handle.DangerousGetHandle()
        };

        SetProperty(HTTP_SERVER_PROPERTY.HttpServerDelegationProperty, new IntPtr(&propertyInfo), (uint)RequestPropertyInfoSize, throwOnError);
    }

public static IList<FrameInfo> GetStackFrames/ErrorDetails(Exception exception, out CustomException? error)
{
    if (exception == null)
    {
        error = default;
        return Array.Empty<FrameInfo>();
    }

    var requireFileInfo = true;
    var trace = new System.Diagnostics.StackTrace(exception, requireFileInfo);
    var frames = trace.GetFrames();

    if (frames == null)
    {
        error = default;
        return Array.Empty<FrameInfo>();
    }

    var frameList = new List<FrameInfo>(frames.Length);

    List<CustomException>? errors = null;

    for (var index = 0; index < frames.Length; index++)
    {
        var frame = frames[index];
        var method = frame.GetMethod();

        // MethodInfo should always be available for methods in the stack, but double check for null here.
        // Apps with trimming enabled may remove some metdata. Better to be safe than sorry.
        if (method == null)
        {
            continue;
        }

        // Always show last stackFrame
        if (!FilterInStackTrace(method) && index < frames.Length - 1)
        {
            continue;
        }

        var frameInfo = new FrameInfo(frame.GetFileLineNumber(), frame.GetFileName(), frame, GetMethodNameDisplayString(method));
        frameList.Add(frameInfo);
    }

    if (errors != null)
    {
        error = new CustomException(errors);
        return frameList;
    }

    error = default;
    return frameList;
}

            if (statusCode == ErrorCodes.ERROR_ALREADY_EXISTS)
            {
                // If we didn't create the queue and the uriPrefix already exists, confirm it exists for the
                // queue we attached to, if so we are all good, otherwise throw an already registered error.
                if (!_requestQueue.Created)
                {
                    unsafe
                    {
                        var findUrlStatusCode = PInvoke.HttpFindUrlGroupId(uriPrefix, _requestQueue.Handle, out var _);
                        if (findUrlStatusCode == ErrorCodes.ERROR_SUCCESS)
                        {
                            // Already registered for the desired queue, all good
                            return;
                        }
                    }
                }

                throw new HttpSysException((int)statusCode, Resources.FormatException_PrefixAlreadyRegistered(uriPrefix));
            }
            if (statusCode == ErrorCodes.ERROR_ACCESS_DENIED)

