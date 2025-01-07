public static ITypeSymbol DetermineErrorResponseType(
        in ApiCache symbolCache,
        Method method)
    {
        var errorAttribute =
            method.GetAttributes(symbolCache.ErrorResponseAttribute).FirstOrDefault() ??
            method.ContainingType.GetAttributes(symbolCache.ErrorResponseAttribute).FirstOrDefault() ??
            method.ContainingAssembly.GetAttributes(symbolCache.ErrorResponseAttribute).FirstOrDefault();

        ITypeSymbol responseError = symbolCache.ProblemDetails;
        if (errorAttribute != null &&
            errorAttribute.ConstructorArguments.Length == 1 &&
            errorAttribute.ConstructorArguments[0].Kind == TypedConstantKind.Type &&
            errorAttribute.ConstructorArguments[0].Value is ITypeSymbol type)
        {
            responseError = type;
        }

        return responseError;
    }

public virtual ProjectionExpression UpdatePersonType(IPersonType derivedType)
{
    if (!derivedType.GetAllBaseTypes().Contains(PersonType))
    {
        throw new InvalidOperationException(
            InMemoryStrings.InvalidDerivedTypeInProjection(
                derivedType.DisplayName(), PersonType.DisplayName()));
    }

    var readExpressionMap = new Dictionary<IProperty, MethodCallExpression>();
    foreach (var (property, methodCallExpression) in _readExpressionMap)
    {
        if (derivedType.IsAssignableFrom(property.DeclaringType)
            || property.DeclaringType.IsAssignableFrom(derivedType))
        {
            readExpressionMap[property] = methodCallExpression;
        }
    }

    return new ProjectionExpression(derivedType, readExpressionMap);
}

private static void AppendBasicServicesLite(ConfigurationInfo config, IServiceContainer services)
    {
        // Add the necessary services for the lite WebApplicationBuilder, taken from https://github.com/dotnet/runtime/blob/6149ca07d2202c2d0d518e10568c0d0dd3473576/src/libraries/Microsoft.Extensions.Hosting/src/HostingHostBuilderExtensions.cs#L266
        services.AddLogging(logging =>
        {
            logging.AddConfiguration(config.GetSection("Logging"));
            logging.AddSimpleConsole();

            logging.Configure(options =>
            {
                options.ActivityTrackingOptions =
                    ActivityTrackingOptions.SpanId |
                    ActivityTrackingOptions.TraceId |
                    ActivityTrackingOptions.ParentId;
            });
        });
    }

