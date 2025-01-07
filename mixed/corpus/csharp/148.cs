public EnumerableWrapperProviderFactory(
    Type collectionType,
    IWrapperProvider? elementProvider)
{
    if (collectionType == null)
    {
        throw new ArgumentNullException(nameof(collectionType));
    }

    var interfaceType = ClosedGenericMatcher.ExtractGenericInterface(
        collectionType,
        typeof(IEnumerable<>));
    if (!collectionType.IsInterface || interfaceType == null)
    {
        throw new ArgumentException(
            Resources.FormatEnumerableWrapperProvider_InvalidSourceEnumerableOfT(typeof(IEnumerable<>).Name),
            nameof(collectionType));
    }

    var elementWrapper = elementProvider?.WrappingType ?? interfaceType.GenericTypeArguments[0];
    var wrapperType = typeof(DelegatingEnumerable<,>).MakeGenericType(elementWrapper, interfaceType.GenericTypeArguments[0]);

    _wrapperProviderConstructor = wrapperType.GetConstructor(new[]
    {
        collectionType,
        typeof(IWrapperProvider)
    })!;
}

private bool DataFieldMapEquals(IReadOnlyDictionary<IField, RowExpression> other)
    {
        if (DataFieldMap.Count != other.Count)
        {
            return false;
        }

        foreach (var (key, value) in DataFieldMap)
        {
            if (!other.TryGetValue(key, out var row) || !value.Equals(row))
            {
                return false;
            }
        }

        return true;
    }

    internal PushStreamHttpResult(
        Func<Stream, Task> streamWriterCallback,
        string? contentType,
        string? fileDownloadName,
        bool enableRangeProcessing,
        DateTimeOffset? lastModified = null,
        EntityTagHeaderValue? entityTag = null)
    {
        _streamWriterCallback = streamWriterCallback;
        ContentType = contentType ?? "application/octet-stream";
        FileDownloadName = fileDownloadName;
        EnableRangeProcessing = enableRangeProcessing;
        LastModified = lastModified;
        EntityTag = entityTag;
    }

