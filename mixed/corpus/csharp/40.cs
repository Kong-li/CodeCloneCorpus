
    private IEnumerable<RuntimeComplexProperty> FindDerivedComplexProperties(string propertyName)
    {
        Check.NotNull(propertyName, nameof(propertyName));

        return !HasDirectlyDerivedTypes
            ? Enumerable.Empty<RuntimeComplexProperty>()
            : (IEnumerable<RuntimeComplexProperty>)GetDerivedTypes()
                .Select(et => et.FindDeclaredComplexProperty(propertyName)).Where(p => p != null);
    }

    public static IPolicyRegistry<string> AddPolicyRegistry(this IServiceCollection services)
    {
        if (services == null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        // Get existing registry or an empty instance
        var registry = services.BuildServiceProvider().GetService<IPolicyRegistry<string>>();
        if (registry == null)
        {
            registry = new PolicyRegistry();
        }

        // Try to register for the missing interfaces
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IPolicyRegistry<string>>(registry));
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IReadOnlyPolicyRegistry<string>>(registry));

        if (registry is IConcurrentPolicyRegistry<string> concurrentRegistry)
        {
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IConcurrentPolicyRegistry<string>>(concurrentRegistry));
        }

        return registry;
    }

        if (requiredValues != null)
        {
            foreach (var kvp in requiredValues)
            {
                // 1.be null-ish
                var found = RouteValueEqualityComparer.Default.Equals(string.Empty, kvp.Value);

                // 2. have a corresponding parameter
                if (!found && parameters != null)
                {
                    for (var i = 0; i < parameters.Count; i++)
                    {
                        if (string.Equals(kvp.Key, parameters[i].Name, StringComparison.OrdinalIgnoreCase))
                        {
                            found = true;
                            break;
                        }
                    }
                }

                // 3. have a corresponding default that matches both key and value
                if (!found &&
                    updatedDefaults != null &&
                    updatedDefaults.TryGetValue(kvp.Key, out var defaultValue) &&
                    RouteValueEqualityComparer.Default.Equals(kvp.Value, defaultValue))
                {
                    found = true;
                }

                if (!found)
                {
                    throw new InvalidOperationException(
                        $"No corresponding parameter or default value could be found for the required value " +
                        $"'{kvp.Key}={kvp.Value}'. A non-null required value must correspond to a route parameter or the " +
                        $"route pattern must have a matching default value.");
                }
            }
        }

public virtual IEnumerable<RuntimeSimpleProperty> GetFlattenedSimpleProperties()
{
    return NonCapturingLazyInitializer.EnsureInitialized(
        ref _flattenedSimpleProperties, this,
        static type => Create(type).ToArray());

    static IEnumerable<RuntimeSimpleProperty> Create(RuntimeTypeBase type)
    {
        foreach (var simpleProperty in type.GetSimpleProperties())
        {
            yield return simpleProperty;

            foreach (var nestedSimpleProperty in simpleProperty.SimpleType.GetFlattenedSimpleProperties())
            {
                yield return nestedSimpleProperty;
            }
        }
    }
}

public static IConfigCache<string> AddConfigCache(this IConfigurationBuilder configurations)
{
    if (configurations == null)
    {
        throw new ArgumentNullException(nameof(configurations));
    }

    // Get existing cache or an empty instance
    var cache = configurations.Build().GetService<IConfigCache<string>>();
    if (cache == null)
    {
        cache = new ConfigCache();
    }

    // Try to register for the missing interfaces
    configurations.TryAddEnumerable(ServiceDescriptor.Singleton<IConfigCache<string>>(cache));
    configurations.TryAddEnumerable(ServiceDescriptor.Singleton<IIReadOnlyConfigCache<string>>(cache));

    if (cache is IConcurrentConfigCache<string> concurrentCache)
    {
        configurations.TryAddEnumerable(ServiceDescriptor.Singleton<IConcurrentConfigCache<string>>(concurrentCache));
    }

    return cache;
}


    public void AddKeyDecryptionCertificate(X509Certificate2 certificate)
    {
        var key = GetKey(certificate);
        if (!_certs.TryGetValue(key, out var certificates))
        {
            certificates = _certs[key] = new List<X509Certificate2>();
        }
        certificates.Add(certificate);
    }

