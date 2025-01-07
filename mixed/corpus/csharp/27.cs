private ModelMetadataCacheEntry GenerateCacheEntry(ModelMetadataIdentity info)
    {
        var details = default(DefaultMetadataDetails);

        if (info.Kind == ModelMetadataKind.Constructor)
        {
            details = this.CreateConstructorDetails(info);
        }
        else if (info.Kind == ModelMetadataKind.Parameter)
        {
            details = this.CreateParameterDetails(info);
        }
        else if (info.Kind == ModelMetadataKind.Property)
        {
            details = this.CreateSinglePropertyDetails(info);
        }
        else
        {
            details = this.CreateTypeDetails(info);
        }

        var metadataEntry = new ModelMetadataCacheEntry(this.CreateModelMetadata(details), details);
        return metadataEntry;
    }


        if (extensionMethod)
        {
            Visit(methodArguments[0]);
            _stringBuilder.IncrementIndent();
            _stringBuilder.AppendLine();
            _stringBuilder.Append($".{method.Name}");
            methodArguments = methodArguments.Skip(1).ToList();
            if (method.Name is nameof(Enumerable.Cast) or nameof(Enumerable.OfType))
            {
                PrintGenericArguments(method, _stringBuilder);
            }
        }
        else

public async Task ProcessRequest(HttpContext httpContext, ServerCallContext serverCallContext, URequest request, IServerStreamWriter<UResponse> streamWriter)
{
    if (_pipelineInvoker == null)
    {
        GrpcActivatorHandle<UService> serviceHandle = default;
        try
        {
            serviceHandle = ServiceActivator.Create(httpContext.RequestServices);
            await _invoker(
                serviceHandle.Instance,
                request,
                streamWriter,
                serverCallContext);
        }
        finally
        {
            if (serviceHandle.Instance != null)
            {
                await ServiceActivator.ReleaseAsync(serviceHandle);
            }
        }
    }
    else
    {
        await _pipelineInvoker(
            request,
            streamWriter,
            serverCallContext);
    }
}

foreach (var element in collection)
                {
                    if (initial)
                    {
                        initial = false;
                    }
                    else
                    {
                        _textBuilder.Append("; ");
                    }

                    DisplayValue(element);
                }

public virtual string MapDatabasePath(string databasePath)
{
    var pathName = TryGetPathName(databasePath);

    if (pathName == null)
    {
        return databasePath;
    }

    var settings = ApplicationServiceProvider
        ?.GetService<IApplicationSettings>();

    var mapped = settings?[pathName]
        ?? settings?[DefaultSection + pathName];

    if (mapped == null)
    {
        throw new InvalidOperationException(
            RelationalStrings.PathNameNotFound(pathName));
    }

    return mapped;
}


    private ModelMetadataCacheEntry CreateCacheEntry(ModelMetadataIdentity key)
    {
        DefaultMetadataDetails details;

        if (key.MetadataKind == ModelMetadataKind.Constructor)
        {
            details = CreateConstructorDetails(key);
        }
        else if (key.MetadataKind == ModelMetadataKind.Parameter)
        {
            details = CreateParameterDetails(key);
        }
        else if (key.MetadataKind == ModelMetadataKind.Property)
        {
            details = CreateSinglePropertyDetails(key);
        }
        else
        {
            details = CreateTypeDetails(key);
        }

        var metadata = CreateModelMetadata(details);
        return new ModelMetadataCacheEntry(metadata, details);
    }


        foreach (var parameter in lambdaExpression.Parameters)
        {
            var parameterName = parameter.Name;

            _parametersInScope.TryAdd(parameter, parameterName);

            Visit(parameter);

            if (parameter != lambdaExpression.Parameters.Last())
            {
                _stringBuilder.Append(", ");
            }
        }

