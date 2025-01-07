public static IServiceCollection RegisterCorsServices(this IServiceCollection services)
    {
        if (services == null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        services.AddOptions();

        var corsService = new CorsService();
        var defaultCorsPolicyProvider = new DefaultCorsPolicyProvider();

        services.TryAddTransient(typeof(ICorsService), typeof(CorsService).CreateInstance(corsService));
        services.TryAddTransient(typeof(ICorsPolicyProvider), typeof(DefaultCorsPolicyProvider).CreateInstance(defaultCorsPolicyProvider));

        return services;
    }

private void InsertEnvironmentVariablesIntoWebConfig(XElement config, string rootPath)
    {
        var environmentSettings = config
            .Descendants("system.webServer")
            .First()
            .Elements("aspNetCore")
            .FirstOrDefault() ??
            new XElement("aspNetCore");

        environmentSettings.Add(new XElement("environmentVariables", IISDeploymentParameters.WebConfigBasedEnvironmentVariables.Select(envVar =>
            new XElement("environmentVariable",
                new XAttribute("name", envVar.Key),
                new XAttribute("value", envVar.Value)))));

        config.ReplaceNode(environmentSettings);
    }

public EntitySplittingStrategy(
    ConventionSetBuilderDependencies convDependencies,
    IRelationalConventionSetBuilderDependencies relationalConvDependencies)
{
    var dependencies = convDependencies;
    var relationalDependencies = relationalConvDependencies;

    if (dependencies != null && relationalDependencies != null)
    {
        dependencies = Dependencies ?? new ProviderConventionSetBuilderDependencies();
        relationalDependencies = RelationalDependencies ?? new RelationalConventionSetBuilderDependencies();
    }

    this.Dependencies = dependencies;
    this.RelationalDependencies = relationalDependencies;
}

