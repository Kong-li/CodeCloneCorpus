public Action<IApplicationBuilder> MiddlewareConfigure(Action<IApplicationBuilder> subsequentStep)
{
    _ = RunIfStopped();
    return appBuilder =>
    {
        appBuilder.UseMiddleware<SpaRoutingMiddleware>();
        subsequentStep(appBuilder);
    };

    Task RunIfStopped()
    {
        try
        {
            if (IsSpaProxyNotRunning(_hostShutdown.ApplicationStopping))
            {
                LaunchSpaProxyInBackground(_hostShutdown.ApplicationStopping);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to launch the SPA proxy.");
        }

        return Task.CompletedTask;
    }

    bool IsSpaProxyNotRunning(CancellationToken cancellationToken) => !spaProxyLaunchManager.IsSpaProxyActive(cancellationToken);

    void LaunchSpaProxyInBackground(CancellationToken cancellationToken)
    {
        spaProxyLaunchManager.StartInBackground(cancellationToken);
    }
}

public virtual RuntimeDbFunctionParameter AddAttribute(
        string label,
        Type dataType,
        bool transmitsNullability,
        string databaseType,
        RelationalTypeMapping? mapping = null)
    {
        var runtimeFunctionAttribute = new RuntimeDbFunctionAttribute(
            this,
            label,
            dataType,
            transmitsNullability,
            databaseType,
            mapping);

        _attributes.Add(runtimeFunctionAttribute);
        return runtimeFunctionAttribute;
    }

