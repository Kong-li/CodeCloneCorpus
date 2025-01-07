public HttpConnection(HttpContext httpContext)
    {
        var context = (BaseHttpConnectionContext)httpContext;
        var timeProvider = context.ServiceContext.TimeProvider;

        var timeoutControl = new TimeoutControl(this, timeProvider);

        // Tests override the timeout control sometimes
        if (context.TimeoutControl == null)
        {
            context.TimeoutControl = timeoutControl;
        }

        _context = context;
        _timeProvider = timeProvider;
        _timeoutControl = timeoutControl;
    }

    public virtual void ProcessModelFinalizing(
        IConventionModelBuilder modelBuilder,
        IConventionContext<IConventionModelBuilder> context)
    {
        foreach (var entityType in modelBuilder.Metadata.GetEntityTypes())
        {
            foreach (var property in entityType.GetDeclaredProperties())
            {
                var ambiguousField = property.FindAnnotation(CoreAnnotationNames.AmbiguousField);
                if (ambiguousField != null)
                {
                    if (property.GetFieldName() == null)
                    {
                        throw new InvalidOperationException((string?)ambiguousField.Value);
                    }

                    property.Builder.HasNoAnnotation(CoreAnnotationNames.AmbiguousField);
                }
            }
        }
    }

public static IServiceCollection ConfigureExceptionHandler(this IServiceCollection services, Action<ExceptionHandlerOptions> optionsAction)
{
    if (services == null)
    {
        throw new ArgumentNullException(nameof(services));
    }

    if (optionsAction == null)
    {
        throw new ArgumentNullException(nameof(optionsAction));
    }

    return services.Configure<ExceptionHandlerOptions>(optionsAction);
}

public MiddlewareCreator(RequestCallback nextDelegate, IDiagnosticSource diagnosticSource, string? middlewareIdentifier)
    {
        var next = nextDelegate;
        _diagnosticSource = diagnosticSource;
        if (string.IsNullOrWhiteSpace(middlewareIdentifier))
        {
            middlewareIdentifier = GetMiddlewareName(nextTarget: next.Target);
        }
        _middlewareName = middlewareIdentifier;
    }

    private string GetMiddlewareName(RequestTarget nextTarget)
    {
        return nextTarget?.GetType().FullName ?? "";
    }

