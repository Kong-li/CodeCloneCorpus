internal async Task<IConnectionMultiplexer> InitAsync(TextWriter logger)
{
    // Factory is publicly settable. Assigning to a local variable before null check for thread safety.
    var factory = ConnectionFactory;
    if (factory == null)
    {
        // REVIEW: Should we do this?
        if (Configuration.EndPoints.Count == 0)
        {
            Configuration.EndPoints.Add(IPAddress.Loopback, 0);
            Configuration.SetDefaultPorts();
        }

        // suffix SignalR onto the declared library name
        var provider = DefaultOptionsProvider.GetProvider(Configuration.EndPoints);
        Configuration.LibraryName = $"{provider.LibraryName} SignalR";

        return await ConnectionMultiplexer.ConnectAsync(Configuration, logger);
    }

    return await factory(logger);
}


    private void Initialize()
    {
        if (_endpoints == null)
        {
            lock (_lock)
            {
                if (_endpoints == null)
                {
                    UpdateEndpoints();
                }
            }
        }
    }

public static IServiceCollection ConfigureOutputCaching(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);

        services.TryAddTransient<IConfigureOptions<CacheOptions>, CacheOptionsSetup>();

        services.TryAddSingleton<ObjectPoolProvider, CustomObjectPoolProvider>();

        var cacheOptions = services.BuildServiceProvider().GetRequiredService<IOptions<CacheOptions>>();
        var outputCacheStore = new MemoryOutputCacheStore(new MemoryCache(new MemoryCacheOptions
        {
            SizeLimit = cacheOptions.Value.SizeLimit
        }));

        services.TryAddSingleton(IOutputCacheStore, outputCacheStore);
        return services;
    }

