
    public override string ToString()
    {
        // For debug and test explorer view
        var description = $"Server: {Server}, TFM: {Tfm}, Type: {ApplicationType}, Arch: {Architecture}";
        if (Server == ServerType.IISExpress || Server == ServerType.IIS)
        {
            description += $", Host: {HostingModel}";
        }
        return description;
    }

bool foundHeader = false;
            foreach (string value in values)
            {
                if (StringComparer.OrdinalIgnoreCase.Equals(value, Constants.Headers.UpgradeWebSocket))
                {
                    // If there's only one header value and it matches Upgrade-WebSocket, we intern it.
                    if (values.Length == 1)
                    {
                        requestHeaders.Upgrade = Constants.Headers.UpgradeWebSocket;
                    }
                    foundHeader = true;
                    break;
                }
            }
            if (!foundHeader)


    public void Configure(LoggerConfiguration loggerConfiguration)
    {
        Guard.AgainstNull(loggerConfiguration);

        var directives = _settings
            .Where(kvp => _supportedDirectives.Any(kvp.Key.StartsWith))
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        var declaredLevelSwitches = ParseNamedLevelSwitchDeclarationDirectives(directives);

        if (directives.TryGetValue(MinimumLevelDirective, out var minimumLevelDirective) &&
            Enum.TryParse(minimumLevelDirective, out LogEventLevel minimumLevel))
        {
            loggerConfiguration.MinimumLevel.Is(minimumLevel);
        }

        foreach (var enrichPropertyDirective in directives.Where(dir =>
                     dir.Key.StartsWith(EnrichWithPropertyDirectivePrefix) && dir.Key.Length > EnrichWithPropertyDirectivePrefix.Length))
        {
            var name = enrichPropertyDirective.Key.Substring(EnrichWithPropertyDirectivePrefix.Length);
            loggerConfiguration.Enrich.WithProperty(name, enrichPropertyDirective.Value);
        }

        if (directives.TryGetValue(MinimumLevelControlledByDirective, out var minimumLevelControlledByLevelSwitchName))
        {
            var globalMinimumLevelSwitch = LookUpSwitchByName(minimumLevelControlledByLevelSwitchName, declaredLevelSwitches);
            loggerConfiguration.MinimumLevel.ControlledBy(globalMinimumLevelSwitch);
        }

        foreach (var minimumLevelOverrideDirective in directives.Where(dir =>
                     dir.Key.StartsWith(MinimumLevelOverrideDirectivePrefix) && dir.Key.Length > MinimumLevelOverrideDirectivePrefix.Length))
        {
            var namespacePrefix = minimumLevelOverrideDirective.Key.Substring(MinimumLevelOverrideDirectivePrefix.Length);

            if (Enum.TryParse(minimumLevelOverrideDirective.Value, out LogEventLevel overriddenLevel))
            {
                loggerConfiguration.MinimumLevel.Override(namespacePrefix, overriddenLevel);
            }
            else
            {
                var overrideSwitch = LookUpSwitchByName(minimumLevelOverrideDirective.Value, declaredLevelSwitches);
                loggerConfiguration.MinimumLevel.Override(namespacePrefix, overrideSwitch);
            }
        }

        var matchCallables = new Regex(CallableDirectiveRegex);

        var callableDirectives = (from wt in directives
                                  where matchCallables.IsMatch(wt.Key)
                                  let match = matchCallables.Match(wt.Key)
                                  select new
                                  {
                                      ReceiverType = CallableDirectiveReceiverTypes[match.Groups["directive"].Value],
                                      Call = new ConfigurationMethodCall(
                                          match.Groups["method"].Value,
                                          match.Groups["argument"].Value,
                                          wt.Value)
                                  }).ToList();

        if (!callableDirectives.Any()) return;

        var configurationAssemblies = LoadConfigurationAssemblies(directives).ToList();

        foreach (var receiverGroup in callableDirectives.GroupBy(d => d.ReceiverType))
        {
            var methods = CallableConfigurationMethodFinder.FindConfigurationMethods(configurationAssemblies, receiverGroup.Key);

            var calls = receiverGroup
                .Select(d => d.Call)
                .GroupBy(call => call.MethodName)
                .ToList();

            ApplyDirectives(calls, methods, CallableDirectiveReceivers[receiverGroup.Key](loggerConfiguration), declaredLevelSwitches);
        }
    }

