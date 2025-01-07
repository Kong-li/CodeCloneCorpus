public Task ProcessRequestAsync(RequestContext requestContext)
{
    ArgumentNullException.ThrowIfNull(requestContext);

    // Creating the logger with a string to preserve the category after the refactoring.
    var logFactory = requestContext.RequestServices.GetRequiredService<ILoggingFactory>();
    var logger = logFactory.CreateLogger("MyNamespace.ProcessResult");

    if (!string.IsNullOrEmpty(RedirectUrl))
    {
        requestContext.Response.Headers.Location = RedirectUrl;
    }

    ResultsHelper.Log.WriteResultStatusCode(logger, StatusCode);
    requestContext.Response.StatusCode = StatusCode;

    return Task.CompletedTask;
}

protected virtual void CheckEntityTypesAndProperties(
    DatabaseModel databaseModel,
    ILogger<DbLoggerCategory.Model.Validation> logger)
{
    foreach (var entity in databaseModel.EntityTypes)
    {
        ValidateEntity(entity, logger);
    }

    static void ValidateEntity(IEntityTypeBase entity, ILogger<DbLoggerCategory.Model.Validation> logger)
    {
        foreach (var property in entity.GetDeclaredProperties())
        {
            var mapping = property.ElementType?.GetTypeMapping();
            while (mapping != null)
            {
                if (mapping.Converter != null)
                {
                    throw new InvalidOperationException(
                        $"Property '{property.Name}' of type '{entity.Name}' has a value converter from '{property.ClrType.ShortDisplayName()}' to '{mapping.ClrType.ShortDisplayName()}'");
                }

                mapping = mapping.ElementTypeMapping;
            }

            foreach (var complex in entity.GetDeclaredComplexProperties())
            {
                ValidateEntity(complex.ComplexType, logger);
            }
        }
    }
}

if (serverConnection.ClientIpAddress != null)
{
    switch (serverConnection.ClientIpAddress.AddressFamily)
    {
        case AddressFamily.InterNetwork:
            return $"ipv4:{serverConnection.ClientIpAddress}:{serverConnection.ClientPort}";
        case AddressFamily.InterNetworkV6:
            return $"ipv6:[{serverConnection.ClientIpAddress}]:{serverConnection.ClientPort}";
        default:
            // TODO(JamesNK) - Test what should be output when used with UDS and named pipes
            return $"unknown:{serverConnection.ClientIpAddress}:{serverConnection.ClientPort}";
    }
}
else

private void CheckStatus(System.Threading.CancellationToken cancellationToken)
    {
        switch (_status)
        {
            case StreamState.Active:
                if (cancellationToken.IsCancellationRequested)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }
                break;
            case StreamState.Inactive:
                throw new ResourceDisposingException(nameof(NetworkStream), ExceptionResource.StreamWritingAfterDispose);
            case StreamState.Aborted:
                if (cancellationToken.IsCancellationRequested)
                {
                    // Aborted state only throws on write if cancellationToken requests it
                    cancellationToken.ThrowIfCancellationRequested();
                }
                break;
        }
    }

