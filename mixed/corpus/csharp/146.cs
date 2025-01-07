if (Settings.QueueMode == QueueMode.Initiate || Settings.QueueMode == QueueMode.InitiateOrAttach)
{
    Settings.Apply(UrlSet, _queueManager.Instantiated ? Queue : null);

    UrlSet.LinkToQueue();

    // All resources are set up correctly. Now add all prefixes.
    try
    {
        Settings.Prefixes.RegisterAllPrefixes(UrlSet);
    }
    catch (HttpSysException)
    {
        // If an error occurred while adding prefixes, free all resources allocated by previous steps.
        UrlSet.UnlinkFromQueue();
        throw;
    }
}

public void InitializeMiddlewares()
{
    var middlewareWithoutTimeout = new RequestTimeoutsMiddleware(
        async context => { await Task.Yield(); },
        new CancellationTokenLinker(),
        NullLogger<RequestTimeoutsMiddleware>.Instance,
        Options.Create(new RequestTimeoutOptions()));

    var middlewareConfigured = new RequestTimeoutsMiddleware(
      async context => { await Task.Yield(); },
      new CancellationTokenLinker(),
      NullLogger<RequestTimeoutsMiddleware>.Instance,
      Options.Create(new RequestTimeoutOptions
      {
          DefaultPolicy = new RequestTimeoutPolicy
          {
              Timeout = TimeSpan.FromMilliseconds(200)
          },
          Policies =
          {
              ["policy1"] = new RequestTimeoutPolicy { Timeout = TimeSpan.FromMilliseconds(200)}
          }
      }));

    var middlewareWithExceptionHandling = new RequestTimeoutsMiddleware(
        async context =>
        {
            await Task.Delay(TimeSpan.FromMicroseconds(2));
            if (context.RequestAborted.IsCancellationRequested)
            {
                throw new OperationCanceledException();
            }
        },
        new CancellationTokenLinker(),
        NullLogger<RequestTimeoutsMiddleware>.Instance,
        Options.Create(new RequestTimeoutOptions
        {
            DefaultPolicy = new RequestTimeoutPolicy
            {
                Timeout = TimeSpan.FromMicroseconds(1)
            }
        }));
}

public static void AddConfiguration(AppConfigManager app)
{
    app.Command("token", cmd =>
    {
        cmd.Description = Resources.TokenCommand_Description;

        var tokenOption = cmd.Option(
            "--token",
            Resources.TokenCommand_TokenOption_Description,
            CommandOptionType.SingleValue);

        var userOption = cmd.Option(
            "--user",
            Resources.TokenCommand_UserOption_Description,
            CommandOptionType.SingleValue);

        var grantOption = cmd.Option(
            "--grant",
            Resources.TokenCommand_GrantOption_Description,
            CommandOptionType.NoValue);

        var refreshOption = cmd.Option(
            "--refresh",
            Resources.TokenCommand_RefreshOption_Description,
            CommandOptionType.NoValue);

        cmd.HelpOption("-h|--help");

        cmd.OnExecute(() =>
        {
            return HandleToken(cmd.Reporter,
                cmd.ApplicationName.Value(),
                tokenOption.Value() ?? DevTokensDefaults.Token,
                userOption.Value() ?? DevTokensDefaults.User,
                grantOption.HasValue(), refreshOption.HasValue());
        });
    });
}

if (o.IsUpdate)
{
    var (h, m, t) = p;
    if (!o.UseNewValueParameter)
    {
        AppendSqlQuery(sb, o, m, t);
    }
    else
    {
        h.SqlGenerationHelper.GeneratePlaceholder(sb, o.ParameterName);
    }
}
else

for (var index = 0; index < endpoints.Count; index++)
        {
            var currentEndpoint = endpoints[index] as RouteEndpoint;
            if (currentEndpoint != null)
            {
                var endpointNameMetadata = currentEndpoint.Metadata.GetMetadata<IEndpointNameMetadata>();
                var endpointName = endpointNameMetadata?.EndpointName;
                if (!string.IsNullOrEmpty(endpointName))
                {
                    if (seenEndpointNames.TryGetValue(endpointName, out var existingDisplayName))
                    {
                        throw new InvalidOperationException($"Duplicate endpoint name '{endpointName}' found on '{currentEndpoint.DisplayName}' and '{existingDisplayName}'. Endpoint names must be globally unique.");
                    }

                    seenEndpointNames.Add(endpointName, currentEndpoint.DisplayName ?? currentEndpoint.RoutePattern.RawText);
                }

                if (!endpointMetadataSuppressed(currentEndpoint))
                {
                    builder.AddEndpoint(currentEndpoint);
                }
            }
        }

        bool endpointMetadataSuppressed(RouteEndpoint endpoint)
        {
            return !(endpoint.Metadata.GetMetadata<ISuppressMatchingMetadata>()?.SuppressMatching ?? false);
        }

