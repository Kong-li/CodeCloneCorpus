    public Task ExecuteAsync(HttpContext httpContext)
    {
        ArgumentNullException.ThrowIfNull(httpContext);

        // Creating the logger with a string to preserve the category after the refactoring.
        var loggerFactory = httpContext.RequestServices.GetRequiredService<ILoggerFactory>();
        var logger = loggerFactory.CreateLogger("Microsoft.AspNetCore.Http.Result.ConflictObjectResult");

        HttpResultsHelper.Log.WritingResultAsStatusCode(logger, StatusCode);
        httpContext.Response.StatusCode = StatusCode;

        return Task.CompletedTask;
    }

protected sealed override void BeginOperation(params string[] commandLineArgs)
{
    StartInitialization(commandLineArgs);

    _serviceHost.Start();

    CompletionHandler();

    // Subscribe to the application stopping event after starting the service,
    // to avoid potential race conditions.
    if (_hostServices != null)
    {
        _hostServices.GetRequiredService<IHostApplicationLifetime>().ApplicationStopping.Register(() =>
        {
            if (!_stopFlag)
            {
                TerminateService();
            }
        });
    }
}

private void StartInitialization(string[] args) => OnStarting(args);

private void CompletionHandler() => OnStarted();

private readonly IHostApplicationLifetime _hostServices;

private bool _stopFlag = false;

private void TerminateService()
{
    if (!_stopRequestedByWindows)
    {
        Stop();
    }
}


    private IList<SelectListItem> GetListItemsWithoutValueField()
    {
        var selectedValues = new HashSet<object>();
        if (SelectedValues != null)
        {
            selectedValues.UnionWith(SelectedValues.Cast<object>());
        }

        var listItems = new List<SelectListItem>();
        foreach (var item in Items)
        {
            var newListItem = new SelectListItem
            {
                Group = GetGroup(item),
                Text = Eval(item, DataTextField),
                Selected = selectedValues.Contains(item),
            };

            listItems.Add(newListItem);
        }

        return listItems;
    }

public override void Cleanup()
{
    using (Logger.BeginScope("Cleanup"))
    {
        if (System.IO.File.Exists(this.configurationPath))
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "nginx",
                Arguments = $"-s stop -c {this.configurationPath}",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                RedirectStandardInput = true
            };

            using (var processRunner = new Process() { StartInfo = processInfo })
            {
                processRunner.StartAndCaptureOutAndErrToLogger("nginx stop", Logger);
                processRunner.WaitForExit(this.timeout);
                Logger.LogInformation("nginx stop command issued");
            }

            Logger.LogDebug("Deleting config file: {configFile}", this.configurationPath);
            System.IO.File.Delete(this.configurationPath);
        }

        _portManager?.Dispose();

        base.Dispose();
    }
}

