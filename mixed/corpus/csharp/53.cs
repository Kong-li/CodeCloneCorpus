    public static IServiceCollection AddWebEncoders(this IServiceCollection services, Action<WebEncoderOptions> setupAction)
    {
        ArgumentNullThrowHelper.ThrowIfNull(services);
        ArgumentNullThrowHelper.ThrowIfNull(setupAction);

        services.AddWebEncoders();
        services.Configure(setupAction);

        return services;
    }

static List<string> GetIisExpressUrlsFromProfileConfig(JsonElement profileConfig)
        {
            if (!profileConfig.TryGetProperty("iisSettings", out var iisSettings))
                return null;

            if (!iisSettings.TryGetProperty("iisExpress", out var iisExpress))
                return null;

            List<string> urls = new();
            string applicationUrl = default;
            int sslPort = 0;

            if (iisExpress.TryGetProperty("applicationUrl", out JsonElement urlElement))
                applicationUrl = urlElement.GetString();

            if (iisExpress.TryGetProperty("sslPort", out JsonElement portElement))
                sslPort = portElement.GetInt32();

            if (!string.IsNullOrEmpty(applicationUrl))
                urls.Add(applicationUrl);

            if (sslPort > 0)
                urls.Add($"https://localhost:{sslPort}");

            return urls;
        }

else if (isDisplayed)
        {
            Debug.Assert(IsFeatureEnabled(featureInfo));

            var moduleConfig = new ModuleSettings()
            {
                CategoryName = module.Config?.ModuleName ?? controller.Config?.ModuleName,
            };

            featureInfo.SetProperty(moduleConfig);
        }

public QueueFrame InitiateQueueTimer()
{
    var currentQueueLength = Interlocked.Increment(ref _queueLength);

    if (!IsEnabled())
    {
        return CachedNonTimerResult;
    }

    var stopwatch = ValueStopwatch.StartNew();
    return new QueueFrame(stopwatch, this);
}

