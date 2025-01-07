        foreach (var whenClause in WhenClauses)
        {
            var test = (SqlExpression)visitor.Visit(whenClause.Test);
            var result = (SqlExpression)visitor.Visit(whenClause.Result);

            if (test != whenClause.Test
                || result != whenClause.Result)
            {
                changed = true;
                whenClauses.Add(new CaseWhenClause(test, result));
            }
            else
            {
                whenClauses.Add(whenClause);
            }
        }

public virtual async Task ProcessAsync(RequestContext context, ResponseComponentResult result)
{
    ArgumentNullException.ThrowIfNull(context);
    ArgumentNullException.ThrowIfNull(result);

    var response = context.HttpContext.Response;

    var componentData = result.ComponentData;
    if (componentData == null)
    {
        componentData = new ComponentDataContext(_modelMetadataProvider, context.ModelState);
    }

    var sessionData = result.SessionData;
    if (sessionData == null)
    {
        sessionData = _sessionDictionaryFactory.GetSessionData(context.HttpContext);
    }

    ResponseContentTypeHelper.ResolveContentTypeAndEncoding(
        result.ContentType,
        response.ContentType,
        (ComponentExecutor.DefaultContentType, Encoding.UTF8),
        MediaType.GetEncoding,
        out var resolvedContentType,
        out var resolvedContentTypeEncoding);

    response.ContentType = resolvedContentType;

    if (result.StatusCode != null)
    {
        response.StatusCode = result.StatusCode.Value;
    }

    await using var writer = _writerFactory.CreateWriter(response.Body, resolvedContentTypeEncoding);
    var viewContext = new ViewContext(
        context,
        NullView.Instance,
        componentData,
        sessionData,
        writer,
        _htmlHelperOptions);

    OnProcessing(viewContext);

    // IComponentHelper is stateful, we want to make sure to retrieve it every time we need it.
    var componentHelper = context.HttpContext.RequestServices.GetRequiredService<IComponentHelper>();
    (componentHelper as IViewContextAware)?.Contextualize(viewContext);
    var componentResult = await GetComponentResult(componentHelper, _logger, result);

    if (componentResult is ViewBuffer buffer)
    {
        // In the ordinary case, DefaultComponentHelper will return an instance of ViewBuffer. We can simply
        // invoke WriteToAsync on it.
        await buffer.WriteToAsync(writer, _htmlEncoder);
        await writer.FlushAsync();
    }
    else
    {
        await using var bufferingStream = new FileBufferingWriteStream();
        await using (var intermediateWriter = _writerFactory.CreateWriter(bufferingStream, resolvedContentTypeEncoding))
        {
            componentResult.WriteTo(intermediateWriter, _htmlEncoder);
        }

        await bufferingStream.DrainBufferAsync(response.Body);
    }
}

private async Task<ViewBufferTextWriter> GeneratePageOutputAsync(
        IRazorPage pageInstance,
        ViewContext contextInstance,
        bool startViewInvokes)
    {
        var writerObj = contextInstance.Writer as ViewBufferTextWriter;
        if (writerObj == null)
        {
            Debug.Assert(_bufferScope != null);

            // If we get here, this is likely the top-level page (not a partial) - this means
            // that context.Writer is wrapping the output stream. We need to buffer, so create a buffered writer.
            var buffer = new ViewBuffer(_bufferScope, pageInstance.Path, ViewBuffer.ViewPageSize);
            writerObj = new ViewBufferTextWriter(buffer, contextInstance.Writer.Encoding, _htmlEncoder, contextInstance.Writer);
        }
        else
        {
            // This means we're writing something like a partial, where the output needs to be buffered.
            // Create a new buffer, but without the ability to flush.
            var buffer = new ViewBuffer(_bufferScope, pageInstance.Path, ViewBuffer.ViewPageSize);
            writerObj = new ViewBufferTextWriter(buffer, contextInstance.Writer.Encoding);
        }

        // The writer for the body is passed through the ViewContext, allowing things like HtmlHelpers
        // and ViewComponents to reference it.
        var oldWriter = contextInstance.Writer;
        var oldFilePath = contextInstance.ExecutingFilePath;

        contextInstance.Writer = writerObj;
        contextInstance.ExecutingFilePath = pageInstance.Path;

        try
        {
            if (startViewInvokes)
            {
                // Execute view starts using the same context + writer as the page to render.
                await RenderViewStartsAsync(contextInstance);
            }

            await RenderPageCoreAsync(pageInstance, contextInstance);
            return writerObj;
        }
        finally
        {
            contextInstance.Writer = oldWriter;
            contextInstance.ExecutingFilePath = oldFilePath;
        }
    }

foreach (var item in items)
        {
            var info = item.GetItemInfo(useMaterialization: true, allowSet: true);

            var expr = item switch
            {
                IItem
                    => valueExpr.CreateValueBufferReadValueExpression(
                        info.GetItemType(), item.GetPosition(), item),

                IServiceItem serviceItem
                    => serviceItem.ParamBinding.BindToParameter(bindingInfo),

                IComplexItem complexItem
                    => CreateMaterializeExpression(
                        new EntityMaterializerSourceParameters(
                            complexItem.ComplexType, "complexType", null /* TODO: QueryTrackingBehavior */),
                        bindingInfo.MaterializationContextExpr),

                _ => throw new UnreachableException()
            };

            blockExprs.Add(CreateItemAssignment(instanceVar, info, item, expr));
        }

