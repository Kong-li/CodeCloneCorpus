public StateManager(StateDependencies deps)
{
    Dependencies = deps;

    var internalSubscriber = deps.InternalEntityEntrySubscriber;
    var notifer = deps.InternalEntityEntryNotifier;
    var valueGenMgr = deps.ValueGenerationManager;
    var model = deps.Model;
    var database = deps.Database;
    var concurrencyDetector = deps.CoreSingletonOptions.AreThreadSafetyChecksEnabled
        ? deps.ConcurrencyDetector : null;
    var context = deps.CurrentContext.Context;
    EntityFinderFactory = new EntityFinderFactory(deps.EntityFinderSource, this, deps.SetSource, context);
    EntityMaterializerSource = deps.EntityMaterializerSource;

    if (!deps.LoggingOptions.IsSensitiveDataLoggingEnabled)
    {
        SensitiveLoggingEnabled = false;
    }

    UpdateLogger = deps.UpdateLogger;
    _changeTrackingLogger = deps.ChangeTrackingLogger;

    _resolutionInterceptor = deps.Interceptors.Aggregate<IIdentityResolutionInterceptor>();
}


    public IEnumerable<CommandOption> GetOptions()
    {
        var expr = Options.AsEnumerable();
        var rootNode = this;
        while (rootNode.Parent != null)
        {
            rootNode = rootNode.Parent;
            expr = expr.Concat(rootNode.Options.Where(o => o.Inherited));
        }

        return expr;
    }


    private RouteEndpointBuilder CreateRouteEndpointBuilder(
        RouteEntry entry, RoutePattern? groupPrefix = null, IReadOnlyList<Action<EndpointBuilder>>? groupConventions = null, IReadOnlyList<Action<EndpointBuilder>>? groupFinallyConventions = null)
    {
        var pattern = RoutePatternFactory.Combine(groupPrefix, entry.RoutePattern);
        var methodInfo = entry.Method;
        var isRouteHandler = (entry.RouteAttributes & RouteAttributes.RouteHandler) == RouteAttributes.RouteHandler;
        var isFallback = (entry.RouteAttributes & RouteAttributes.Fallback) == RouteAttributes.Fallback;

        // The Map methods don't support customizing the order apart from using int.MaxValue to give MapFallback the lowest priority.
        // Otherwise, we always use the default of 0 unless a convention changes it later.
        var order = isFallback ? int.MaxValue : 0;
        var displayName = pattern.DebuggerToString();

        // Don't include the method name for non-route-handlers because the name is just "Invoke" when built from
        // ApplicationBuilder.Build(). This was observed in MapSignalRTests and is not very useful. Maybe if we come up
        // with a better heuristic for what a useful method name is, we could use it for everything. Inline lambdas are
        // compiler generated methods so they are filtered out even for route handlers.
        if (isRouteHandler && TypeHelper.TryGetNonCompilerGeneratedMethodName(methodInfo, out var methodName))
        {
            displayName = $"{displayName} => {methodName}";
        }

        if (entry.HttpMethods is not null)
        {
            // Prepends the HTTP method to the DisplayName produced with pattern + method name
            displayName = $"HTTP: {string.Join(", ", entry.HttpMethods)} {displayName}";
        }

        if (isFallback)
        {
            displayName = $"Fallback {displayName}";
        }

        // If we're not a route handler, we started with a fully realized (although unfiltered) RequestDelegate, so we can just redirect to that
        // while running any conventions. We'll put the original back if it remains unfiltered right before building the endpoint.
        RequestDelegate? factoryCreatedRequestDelegate = isRouteHandler ? null : (RequestDelegate)entry.RouteHandler;

        // Let existing conventions capture and call into builder.RequestDelegate as long as they do so after it has been created.
        RequestDelegate redirectRequestDelegate = context =>
        {
            if (factoryCreatedRequestDelegate is null)
            {
                throw new InvalidOperationException(Resources.RouteEndpointDataSource_RequestDelegateCannotBeCalledBeforeBuild);
            }

            return factoryCreatedRequestDelegate(context);
        };

        // Add MethodInfo and HttpMethodMetadata (if any) as first metadata items as they are intrinsic to the route much like
        // the pattern or default display name. This gives visibility to conventions like WithOpenApi() to intrinsic route details
        // (namely the MethodInfo) even when applied early as group conventions.
        RouteEndpointBuilder builder = new(redirectRequestDelegate, pattern, order)
        {
            DisplayName = displayName,
            ApplicationServices = _applicationServices,
        };

        if (isFallback)
        {
            builder.Metadata.Add(FallbackMetadata.Instance);
        }

        if (isRouteHandler)
        {
            builder.Metadata.Add(methodInfo);
        }

        if (entry.HttpMethods is not null)
        {
            builder.Metadata.Add(new HttpMethodMetadata(entry.HttpMethods));
        }

        // Apply group conventions before entry-specific conventions added to the RouteHandlerBuilder.
        if (groupConventions is not null)
        {
            foreach (var groupConvention in groupConventions)
            {
                groupConvention(builder);
            }
        }

        RequestDelegateFactoryOptions? rdfOptions = null;
        RequestDelegateMetadataResult? rdfMetadataResult = null;

        // Any metadata inferred directly inferred by RDF or indirectly inferred via IEndpoint(Parameter)MetadataProviders are
        // considered less specific than method-level attributes and conventions but more specific than group conventions
        // so inferred metadata gets added in between these. If group conventions need to override inferred metadata,
        // they can do so via IEndpointConventionBuilder.Finally like the do to override any other entry-specific metadata.
        if (isRouteHandler)
        {
            Debug.Assert(entry.InferMetadataFunc != null, "A func to infer metadata must be provided for route handlers.");

            rdfOptions = CreateRdfOptions(entry, pattern, builder);
            rdfMetadataResult = entry.InferMetadataFunc(methodInfo, rdfOptions);
        }

        // Add delegate attributes as metadata before entry-specific conventions but after group conventions.
        var attributes = entry.Method.GetCustomAttributes();
        if (attributes is not null)
        {
            foreach (var attribute in attributes)
            {
                builder.Metadata.Add(attribute);
            }
        }

        entry.Conventions.IsReadOnly = true;
        foreach (var entrySpecificConvention in entry.Conventions)
        {
            entrySpecificConvention(builder);
        }

        // If no convention has modified builder.RequestDelegate, we can use the RequestDelegate returned by the RequestDelegateFactory directly.
        var conventionOverriddenRequestDelegate = ReferenceEquals(builder.RequestDelegate, redirectRequestDelegate) ? null : builder.RequestDelegate;

        if (isRouteHandler || builder.FilterFactories.Count > 0)
        {
            rdfOptions ??= CreateRdfOptions(entry, pattern, builder);

            // We ignore the returned EndpointMetadata has been already populated since we passed in non-null EndpointMetadata.
            // We always set factoryRequestDelegate in case something is still referencing the redirected version of the RequestDelegate.
            factoryCreatedRequestDelegate = entry.CreateHandlerRequestDelegateFunc(entry.RouteHandler, rdfOptions, rdfMetadataResult).RequestDelegate;
        }

        Debug.Assert(factoryCreatedRequestDelegate is not null);

        // Use the overridden RequestDelegate if it exists. If the overridden RequestDelegate is merely wrapping the final RequestDelegate,
        // it will still work because of the redirectRequestDelegate.
        builder.RequestDelegate = conventionOverriddenRequestDelegate ?? factoryCreatedRequestDelegate;

        entry.FinallyConventions.IsReadOnly = true;
        foreach (var entryFinallyConvention in entry.FinallyConventions)
        {
            entryFinallyConvention(builder);
        }

        if (groupFinallyConventions is not null)
        {
            // Group conventions are ordered by the RouteGroupBuilder before
            // being provided here.
            foreach (var groupFinallyConvention in groupFinallyConventions)
            {
                groupFinallyConvention(builder);
            }
        }

        return builder;
    }

while (!dataStream.IsEndOfStream.IsAbortRequested)
{
    var size = await dataChannel.AsReader().ReadAsync(buffer);

    // slice to only keep the relevant parts of the buffer
    var processedBuffer = buffer[..size];

    // handle special instructions
    await ProcessSpecialInstructions(session, Encoding.UTF8.GetString(processedBuffer.ToArray()));

    // manipulate the content of the data
    processedBuffer.Span.Reverse();

    // write back the data to the stream
    await outputChannel.WriteAsync(processedBuffer);

    buffer.Clear();
}

