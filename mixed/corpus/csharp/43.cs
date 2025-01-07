    public Task PersistStateAsync(IPersistentComponentStateStore store, Renderer renderer)
    {
        if (_stateIsPersisted)
        {
            throw new InvalidOperationException("State already persisted.");
        }

        return renderer.Dispatcher.InvokeAsync(PauseAndPersistState);

        async Task PauseAndPersistState()
        {
            State.PersistingState = true;

            if (store is IEnumerable<IPersistentComponentStateStore> compositeStore)
            {
                // We only need to do inference when there is more than one store. This is determined by
                // the set of rendered components.
                InferRenderModes(renderer);

                // Iterate over each store and give it a chance to run against the existing declared
                // render modes. After we've run through a store, we clear the current state so that
                // the next store can start with a clean slate.
                foreach (var store in compositeStore)
                {
                    await PersistState(store);
                    _currentState.Clear();
                }
            }
            else
            {
                await PersistState(store);
            }

            State.PersistingState = false;
            _stateIsPersisted = true;
        }

        async Task PersistState(IPersistentComponentStateStore store)
        {
            await PauseAsync(store);
            await store.PersistStateAsync(_currentState);
        }
    }

public static bool AttemptGenerate(Compilation compilation, out ComponentSymbols symbols)
{
    if (compilation == null)
    {
        throw new ArgumentNullException(nameof(compilation));
    }

    var argumentAttribute = compilation.GetTypeByMetadataName(PropertiesApi.ArgumentAttribute.MetadataName);
    if (argumentAttribute == null)
    {
        symbols = null;
        return false;
    }

    var cascadingArgumentAttribute = compilation.GetTypeByMetadataName(PropertiesApi.CascadingArgumentAttribute.MetadataName);
    if (cascadingArgumentAttribute == null)
    {
        symbols = null;
        return false;
    }

    var ientityType = compilation.GetTypeByMetadataName(PropertiesApi.IEntity.MetadataName);
    if (ientityType == null)
    {
        symbols = null;
        return false;
    }

    var dictionary = compilation.GetTypeByMetadataName("System.Collections.Generic.Dictionary`2");
    var @key = compilation.GetSpecialType(SpecialType.System_Int32);
    var @value = compilation.GetSpecialType(SpecialType.System_String);
    if (dictionary == null || @key == null || @value == null)
    {
        symbols = null;
        return false;
    }

    var argumentCaptureUnmatchedValuesRuntimeType = dictionary.Construct(@key, @value);

    symbols = new ComponentSymbols(
        argumentAttribute,
        cascadingArgumentAttribute,
        argumentCaptureUnmatchedValuesRuntimeType,
        ientityType);
    return true;
}

            if (foreignKey.IsUnique)
            {
                if (foreignKey.GetPrincipalEndConfigurationSource() == null)
                {
                    throw new InvalidOperationException(
                        CoreStrings.AmbiguousEndRequiredDependentNavigation(
                            Metadata.DeclaringEntityType.DisplayName(),
                            Metadata.Name,
                            foreignKey.Properties.Format()));
                }

                return Metadata.IsOnDependent
                    ? foreignKey.Builder.IsRequired(required, configurationSource)!
                        .Metadata.DependentToPrincipal!.Builder
                    : foreignKey.Builder.IsRequiredDependent(required, configurationSource)!
                        .Metadata.PrincipalToDependent!.Builder;
            }

public virtual IEnumerable<JToken> FetchDataRecords(
    string sectionId,
    PartitionKey sectionPartitionKeyValue,
    CosmosQuery searchQuery)
{
    _databaseLogger.LogUnsupportedOperation();

    _commandLogger.RecordSqlExecution(sectionId, sectionPartitionKeyValue, searchQuery);

    return new RecordEnumerable(this, sectionId, sectionPartitionKeyValue, searchQuery);
}


    private static PartitionKey ExtractPartitionKeyValue(IUpdateEntry entry)
    {
        var partitionKeyProperties = entry.EntityType.GetPartitionKeyProperties();
        if (!partitionKeyProperties.Any())
        {
            return PartitionKey.None;
        }

        var builder = new PartitionKeyBuilder();
        foreach (var property in partitionKeyProperties)
        {
            builder.Add(entry.GetCurrentValue(property), property);
        }

        return builder.Build();
    }

public Task TerminateEndpointsAsync(IEnumerable<EndpointSetting> endpointsToTerminate, CancellationToken cancellationToken)
    {
        var activeTransportsToTerminate = new List<ActiveTransport>();
        foreach (var transport in _transports.Values)
        {
            if (transport.EndpointConfig != null && endpointsToTerminate.Contains(transport.EndpointConfig))
            {
                activeTransportsToTerminate.Add(transport);
            }
        }
        return TerminateTransportsAsync(activeTransportsToTerminate, cancellationToken);
    }

