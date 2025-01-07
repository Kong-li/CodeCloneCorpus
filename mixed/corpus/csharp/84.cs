if (!object.ReferenceEquals(predicate, null))
{
    var transformedPredicate = predicate;
    var modifiedSource = TranslateWhere(source, transformedPredicate);
    if (modifiedSource == null)
    {
        return null;
    }

    source = modifiedSource;
}

public void TransferTo(Span<byte> destination)
    {
        Debug.Assert(destination.Length >= _totalBytes);

        if (_currentBuffer == null)
        {
            return;
        }

        int totalCopied = 0;

        if (_completedBuffers != null)
        {
            // Copy full buffers
            var count = _completedBuffers.Count;
            for (var i = 0; i < count; i++)
            {
                var buffer = _completedBuffers[i];
                buffer.Span.CopyTo(destination.Slice(totalCopied));
                totalCopied += buffer.Span.Length;
            }
        }

        // Copy current incomplete buffer
        _currentBuffer.AsSpan(0, _offset).CopyTo(destination.Slice(totalCopied));

        Debug.Assert(_totalBytes == totalCopied + _offset);
    }

if (compileable)
        {
            if (dataType.DeclaringType != null)
            {
                ProcessSpecificType(builder, dataType.DeclaringType, genericParams, offset, fullName, compileable);
                builder.Append('.');
            }
            else if (fullName)
            {
                builder.Append(dataType.Namespace);
                builder.Append('.');
            }
        }
        else

private static bool CheckCompatibilityForDataCreation(
        IReadOnlyField field,
        in DatabaseObjectIdentifier databaseObject,
        IDataTypeMappingSource? dataMappingSource)
    {
        if (databaseObject.DatabaseObjectType != DatabaseObjectType.View)
        {
            return false;
        }

        var valueTransformer = field.GetValueTransformer()
            ?? (field.FindRelationalTypeMapping(databaseObject)
                ?? dataMappingSource?.FindMapping((IField)field))?.Converter;

        var type = (valueTransformer?.ProviderClrType ?? field.ClrType).UnwrapNullableType();

        return (type.IsNumeric()
            || type.IsEnum
            || type == typeof(double));
    }


    private ApiDescription CreateApiDescription(
        ControllerActionDescriptor action,
        string? httpMethod,
        string? groupName)
    {
        var parsedTemplate = ParseTemplate(action);

        var apiDescription = new ApiDescription()
        {
            ActionDescriptor = action,
            GroupName = groupName,
            HttpMethod = httpMethod,
            RelativePath = GetRelativePath(parsedTemplate),
        };

        var templateParameters = parsedTemplate?.Parameters?.ToList() ?? new List<TemplatePart>();

        var parameterContext = new ApiParameterContext(_modelMetadataProvider, action, templateParameters);

        foreach (var parameter in GetParameters(parameterContext))
        {
            apiDescription.ParameterDescriptions.Add(parameter);
        }

        var apiResponseTypes = _responseTypeProvider.GetApiResponseTypes(action);
        foreach (var apiResponseType in apiResponseTypes)
        {
            apiDescription.SupportedResponseTypes.Add(apiResponseType);
        }

        // It would be possible here to configure an action with multiple body parameters, in which case you
        // could end up with duplicate data.
        if (apiDescription.ParameterDescriptions.Count > 0)
        {
            // Get the most significant accepts metadata
            var acceptsMetadata = action.EndpointMetadata.OfType<IAcceptsMetadata>().LastOrDefault();
            var requestMetadataAttributes = GetRequestMetadataAttributes(action);

            var contentTypes = GetDeclaredContentTypes(requestMetadataAttributes, acceptsMetadata);
            foreach (var parameter in apiDescription.ParameterDescriptions)
            {
                if (parameter.Source == BindingSource.Body)
                {
                    // For request body bound parameters, determine the content types supported
                    // by input formatters.
                    var requestFormats = GetSupportedFormats(contentTypes, parameter.Type);
                    foreach (var format in requestFormats)
                    {
                        apiDescription.SupportedRequestFormats.Add(format);
                    }
                }
                else if (parameter.Source == BindingSource.FormFile)
                {
                    // Add all declared media types since FormFiles do not get processed by formatters.
                    foreach (var contentType in contentTypes)
                    {
                        apiDescription.SupportedRequestFormats.Add(new ApiRequestFormat
                        {
                            MediaType = contentType,
                        });
                    }
                }
            }
        }

        return apiDescription;
    }

