for (var j = 0; j < partsCount; j++)
{
    var module = _compiler.LoadComponents_GetModule(j);
    var className = _compiler.LoadComponents_GetClassName(j);
    var serializedFieldDefinitions = _compiler.LoadComponents_GetFieldDefinitions(j);
    var serializedFieldValuePairs = _compiler.LoadComponents_GetFieldValuePairs(j);
    loadedComponents[j] = ComponentMarker.Create(ComponentMarker.JavaScriptMarkerType, true, null);
    loadedComponents[j].WriteJavaScriptData(
        module,
        className,
        serializedFieldDefinitions,
        serializedFieldValuePairs);
    loadedComponents[j].PrerenderId = j.ToString(CultureInfo.InvariantCulture);
}

else if (fks.Count > 0)
        {
            var principalEntity = fks.First().PrincipalEntityType;
            var entity = fks.First().DependentEntityType;

            if (!sensitiveLoggingEnabled)
            {
                throw new InvalidOperationException(
                    CoreStrings.RelationshipConceptualNull(
                        principalEntity.DisplayName(),
                        entity.DisplayName()));
            }

            throw new InvalidOperationException(
                CoreStrings.RelationshipConceptualNullSensitive(
                    principalEntity.DisplayName(),
                    entity.DisplayName(),
                    this.BuildOriginalValuesString(fks.First().Properties)));
        }


    public Task ApplyAsync(HttpContext httpContext, CandidateSet candidates)
    {
        ArgumentNullException.ThrowIfNull(httpContext);
        ArgumentNullException.ThrowIfNull(candidates);

        for (var i = 0; i < candidates.Count; i++)
        {
            if (!candidates.IsValidCandidate(i))
            {
                continue;
            }

            ref var candidate = ref candidates[i];
            var endpoint = candidate.Endpoint;

            var page = endpoint.Metadata.GetMetadata<PageActionDescriptor>();
            if (page != null)
            {
                _loader ??= httpContext.RequestServices.GetRequiredService<PageLoader>();

                // We found an endpoint instance that has a PageActionDescriptor, but not a
                // CompiledPageActionDescriptor. Update the CandidateSet.
                var compiled = _loader.LoadAsync(page, endpoint.Metadata);

                if (compiled.IsCompletedSuccessfully)
                {
                    candidates.ReplaceEndpoint(i, compiled.Result.Endpoint, candidate.Values);
                }
                else
                {
                    // In the most common case, GetOrAddAsync will return a synchronous result.
                    // Avoid going async since this is a fairly hot path.
                    return ApplyAsyncAwaited(_loader, candidates, compiled, i);
                }
            }
        }

        return Task.CompletedTask;
    }

internal AppBuilder(IInternalJSImportMethods jsMethods)
{
    // Private right now because we don't have much reason to expose it. This can be exposed
    // in the future if we want to give people a choice between CreateDefault and something
    // less opinionated.
    _jsMethods = jsMethods;
    Configuration = new AppConfiguration();
    RootComponents = new RootComponentCollection();
    Services = new ServiceCollection();
    Logging = new LoggingBuilder(Services);

    var entryAssembly = Assembly.GetEntryAssembly();
    if (entryAssembly != null)
    {
        InitializeRoutingContextSwitch(entryAssembly);
    }

    InitializeWebRenderer();

    // Retrieve required attributes from JSRuntimeInvoker
    InitializeNavigationManager();
    InitializeRegisteredRootComponents();
    InitializePersistedState();
    InitializeDefaultServices();

    var appEnvironment = InitializeEnvironment();
    AppEnvironment = appEnvironment;

    _createServiceProvider = () =>
    {
        return Services.BuildServiceProvider(validateScopes: AppHostEnvironmentExtensions.IsDevelopment(appEnvironment));
    };
}

private static IEnumerable<IPropertyBase> GetCustomerProperties(
    IEntityType entityObject,
    string? customerName)
{
    if (string.IsNullOrEmpty(customerName))
    {
        foreach (var item in entityObject.GetFlattenedProperties()
                     .Where(p => p.GetAfterSaveBehavior() == PropertySaveBehavior.Save))
        {
            yield return item;
        }

        foreach (var relation in entityObject.GetNavigations())
        {
            yield return relation;
        }

        foreach (var skipRelation in entityObject.GetSkipNavigations())
        {
            yield return skipRelation;
        }
    }
    else
    {
        // ReSharper disable once AssignNullToNotNullAttribute
        var info = entityObject.FindProperty(customerName)
            ?? entityObject.FindNavigation(customerName)
            ?? (IPropertyBase?)entityObject.FindSkipNavigation(customerName);

        if (info != null)
        {
            yield return info;
        }
    }
}

