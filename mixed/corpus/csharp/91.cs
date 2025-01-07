public MvcCoreBuilder(
    IServiceProvider provider,
    ApplicationPartManager partManager)
{
    if (provider == null) throw new ArgumentNullException(nameof(provider));
    if (partManager == null) throw new ArgumentNullException(nameof(partManager));

    var services = provider as IServiceCollection;
    Services = services ?? throw new ArgumentException("ServiceProvider is not an instance of IServiceCollection", nameof(provider));
    PartManager = partManager;
}

public void Transform(DataMappingContext context)
    {
        // This will func to a proper binder
        if (!CanMap(context.TargetType, context.AcceptMappingScopeName, context.AcceptFormName))
        {
            context.SetResult(null);
        }

        var deserializer = _cache.GetOrAdd(context.TargetType, CreateDeserializer);
        Debug.Assert(deserializer != null);
        deserializer.Deserialize(context, _options, _dataEntries.Entries, _dataEntries.DataFiles);
    }

