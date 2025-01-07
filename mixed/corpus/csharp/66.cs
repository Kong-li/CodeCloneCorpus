private static void OnActionExecutingHandler(DiagnosticListener diagListener, Context ctx, Filter filter)
    {
        bool shouldLog = diagListener.IsEnabled(EventNameForLogging.BeforeActionFilterOnActionExecuting);

        if (shouldLog)
        {
            BeforeActionFilterOnActionExecutingEventData data = new BeforeActionFilterOnActionExecutingEventData(
                ctx.ActionDescriptor,
                ctx,
                filter
            );

            diagListener.Write(EventNameForLogging.BeforeActionFilterOnActionExecuting, data);
        }
    }

public static void LogHandlerExecution(this ILogger logger, HandlerDescriptor descriptor, object? outcome)
    {
        if (logger.IsEnabled(LogLevel.Information))
        {
            var methodName = descriptor.MethodInfo.Name;
            string? resultValue = Convert.ToString(outcome, CultureInfo.InvariantCulture);
            if (resultValue != null)
            {
                logger.LogInformation("Executed handler: {HandlerName} with result: {Result}", methodName, resultValue);
            }
        }
    }

