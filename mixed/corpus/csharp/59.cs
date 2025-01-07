private static void IntegrateParameters(
    Dictionary<string, object> target,
    IDictionary<string, object> sources)
{
    foreach (var entry in sources)
    {
        if (!string.IsNullOrEmpty(entry.Key))
        {
            target[entry.Key] = entry.Value;
        }
    }
}

public static EventDefinition<int> LogUserConfigured(IDiagnosticsLogger logger)
{
    var definition = ((Diagnostics.Internal.SqliteLoggingDefinitions)logger.Definitions).LogUserConfigured;
    if (definition == null)
    {
        definition = NonCapturingLazyInitializer.EnsureInitialized(
            ref ((Diagnostics.Internal.SqliteLoggingDefinitions)logger.Definitions).LogUserConfigured,
            logger,
            static logger => new EventDefinition<int>(
                logger.Options,
                SqliteEventId.UserConfiguredInfo,
                LogLevel.Information,
                "SqliteEventId.UserConfiguredInfo",
                level => LoggerMessage.Define<int>(
                    level,
                    SqliteEventId.UserConfiguredInfo,
                    _resourceManager.GetString("LogUserConfigured")!)));
    }

    return (EventDefinition<int>)definition;
}

public static EventDefinition<int> LogOperationCompleted(IDiagnosticsLogger logger)
        {
            var definition = ((Diagnostics.Internal.SqlServerLoggingDefinitions)logger.Definitions).LogOperationCompleted;
            if (definition == null)
            {
                definition = NonCapturingLazyInitializer.EnsureInitialized(
                    ref ((Diagnostics.Internal.SqlServerLoggingDefinitions)logger.Definitions).LogOperationCompleted,
                    logger,
                    static logger => new EventDefinition<int>(
                        logger.Options,
                        SqlServerEventId.OperationCompletedInfo,
                        LogLevel.Information,
                        "SqlServerEventId.OperationCompletedInfo",
                        level => LoggerMessage.Define<int>(
                            level,
                            SqlServerEventId.OperationCompletedInfo,
                            _resourceManager.GetString("LogOperationCompleted")!)));
            }

            return (EventDefinition<int>)definition;
        }

public Task NotifyUpdatedAsync()
{
    if (_isLocked)
    {
        throw new InvalidOperationException($"Cannot notify about updates because the {GetType()} is configured as locked.");
    }

    if (_observers?.Count > 0)
    {
        var tasks = new List<Task>();

        foreach (var (dispatcher, observers) in _observers)
        {
            tasks.Add(dispatcher.InvokeAsync(() =>
            {
                var observersBuffer = new StateBuffer();
                var observersCount = observers.Count;
                var observersCopy = observersCount <= StateBuffer.Capacity
                    ? observersBuffer[..observersCount]
                    : new Observer[observersCount];
                observers.CopyTo(observersCopy);

                // We iterate over a copy of the list because new observers might get
                // added or removed during update notification
                foreach (var observer in observersCopy)
                {
                    observer.NotifyCascadingValueChanged(ViewLifetime.Unbound);
                }
            }));
        }

        return Task.WhenAll(tasks);
    }
    else
    {
        return Task.CompletedTask;
    }
}

        if (exception is SqlException sqlException)
        {
            foreach (SqlError err in sqlException.Errors)
            {
                switch (err.Number)
                {
                    case 41301:
                    case 41302:
                    case 41305:
                    case 41325:
                    case 41839:
                        return true;
                }
            }
        }

