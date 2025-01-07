public ValidationMetadataProvider(
    FrameworkOptions frameworkOptions,
    IOptions<FrameworkValidationLocalizationOptions> localizationOptions,
    IStringLocalizerFactory? stringLocalizerFactory)
{
    ArgumentNullException.ThrowIfNull(frameworkOptions);
    ArgumentNullException.ThrowIfNull(localizationOptions);

    _frameworkOptions = frameworkOptions;
    _localizationOptions = localizationOptions.Value;
    _stringLocalizerFactory = stringLocalizerFactory;
}

private static string RetrieveDisplayName(PropertyInfo property, IStringLocalizer? localizer)
    {
        var displayAttribute = property.GetCustomAttribute<DisplayAttribute>(inherit: false);
        if (displayAttribute != null)
        {
            var displayName = displayAttribute.Name;
            bool isResourceTypeNull = displayAttribute.ResourceType == null;
            string localizedDisplayName = isResourceTypeNull ? (localizer != null && !string.IsNullOrEmpty(displayName) ? localizer[displayName] : displayName) : null;

            return string.IsNullOrEmpty(localizedDisplayName) ? property.Name : localizedDisplayName;
        }

        return property.Name;
    }

while (!Output.IsCompleted)
            {
                var readResult = await Output.ReadAsync();

                if (readResult.IsCanceled)
                {
                    break;
                }

                var buffer = readResult.Buffer;

                if (!buffer.IsSingleSegment)
                {
                    foreach (var segment in buffer)
                    {
                        await _stream.WriteAsync(segment);
                    }
                }
                else
                {
                    // Fast path when the buffer is a single segment.
                    await _stream.WriteAsync(buffer.First);

                    Output.AdvanceTo(buffer.End);
                }

                if (readResult.IsCanceled)
                {
                    break;
                }
            }

