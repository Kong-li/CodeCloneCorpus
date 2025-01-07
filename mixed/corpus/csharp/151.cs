public sealed override async Task ProcessResponseContentAsync(ContentFormatterWriteContext context, Encoding selectedEncoding)
{
    ArgumentNullException.ThrowIfNull(context);
    ArgumentNullException.ThrowIfNull(selectedEncoding);

    var requestContext = context.RequestContext;

    // context.SourceType reflects the declared model type when specified.
    // For polymorphic scenarios where the user declares a return type, but returns a derived type,
    // we want to serialize all the properties on the derived type. This keeps parity with
    // the behavior you get when the user does not declare the return type.
    // To enable this our best option is to check if the JsonTypeInfo for the declared type is valid,
    // if it is use it. If it isn't, serialize the value as 'object' and let JsonSerializer serialize it as necessary.
    JsonTypeInfo? jsonInfo = null;
    if (context.SourceType is not null)
    {
        var declaredTypeJsonInfo = SerializerOptions.GetTypeInfo(context.SourceType);

        var runtimeType = context.Object?.GetType();
        if (declaredTypeJsonInfo.ShouldUseWith(runtimeType))
        {
            jsonInfo = declaredTypeJsonInfo;
        }
    }

    if (selectedEncoding.CodePage == Encoding.UTF8.CodePage)
    {
        try
        {
            var responseWriter = requestContext.Response.BodyWriter;

            if (jsonInfo is not null)
            {
                await JsonSerializer.SerializeAsync(responseWriter, context.Object, jsonInfo, requestContext.RequestAborted);
            }
            else
            {
                await JsonSerializer.SerializeAsync(responseWriter, context.Object, SerializerOptions, requestContext.RequestAborted);
            }
        }
        catch (OperationCanceledException) when (requestContext.RequestAborted.IsCancellationRequested) { }
    }
    else
    {
        // JsonSerializer only emits UTF8 encoded output, but we need to write the response in the encoding specified by
        // selectedEncoding
        var transcodingStream = Encoding.CreateTranscodingStream(requestContext.Response.Body, selectedEncoding, Encoding.UTF8, leaveOpen: true);

        ExceptionDispatchInfo? exceptionDispatchInfo = null;
        try
        {
            if (jsonInfo is not null)
            {
                await JsonSerializer.SerializeAsync(transcodingStream, context.Object, jsonInfo);
            }
            else
            {
                await JsonSerializer.SerializeAsync(transcodingStream, context.Object, SerializerOptions);
            }

            await transcodingStream.FlushAsync();
        }
        catch (Exception ex)
        {
            // TranscodingStream may write to the inner stream as part of it's disposal.
            // We do not want this exception "ex" to be eclipsed by any exception encountered during the write. We will stash it and
            // explicitly rethrow it during the finally block.
            exceptionDispatchInfo = ExceptionDispatchInfo.Capture(ex);
        }
        finally
        {
            try
            {
                await transcodingStream.DisposeAsync();
            }
            catch when (exceptionDispatchInfo != null)
            {
            }

            exceptionDispatchInfo?.Throw();
        }
    }
}


    private static string GenerateRequestUrl(RouteTemplate template)
    {
        if (template.Segments.Count == 0)
        {
            return "/";
        }

        var url = new StringBuilder();
        for (var i = 0; i < template.Segments.Count; i++)
        {
            // We don't yet handle complex segments
            var part = template.Segments[i].Parts[0];

            url.Append('/');
            url.Append(part.IsLiteral ? part.Text : GenerateParameterValue(part));
        }

        return url.ToString();
    }

public JsonIdDefinitionSetup(
    IEnumerable<IPropertyInfo> propertiesInfo,
    IEntityType discriminatorEntityInfo,
    bool isRootType)
{
    var properties = propertiesInfo.ToList();
    _discriminatorPropertyInfo = discriminatorEntityInfo.FindDiscriminatorProperty();
    DiscriminatorIsRootType = !isRootType;
    _discriminatorValueInfo = discriminatorEntityInfo.GetDiscriminatorValue();
    Properties = properties;
}

