    public static ModelBuilder HasSequence(
        this ModelBuilder modelBuilder,
        string name,
        string? schema,
        Action<SequenceBuilder> builderAction)
    {
        Check.NotNull(builderAction, nameof(builderAction));

        builderAction(HasSequence(modelBuilder, name, schema));

        return modelBuilder;
    }

public static ModelConfigurator HasCustomFunction(
    this ModelConfigurator modelConfigurator,
    MethodInfo methodInfo,
    Action<DbFunctionBuilder> builderAction)
{
    Check.NotNull(builderAction, nameof(builderAction));

    builderAction(HasCustomFunction(modelConfigurator, methodInfo));

    return modelConfigurator;
}

if (_maxHeaderTableSize < headerLength)
        {
            int index = ResolveDynamicTableIndex(staticTableIndex, name);

            bool shouldEncodeLiteral = index != -1;
            byte[] buffer = new byte[256];
            int bytesWritten;

            return shouldEncodeLiteral
                ? HPackEncoder.EncodeLiteralHeaderFieldWithoutIndexingNewName(name, value, valueEncoding, buffer, out bytesWritten)
                : HPackEncoder.EncodeLiteralHeaderFieldWithoutIndexing(index, value, valueEncoding, buffer, out bytesWritten);
        }

public DynamicHPackEncoder(bool enableDynamicCompression = false, int maxHeadersCount = 512)
{
    _enableDynamicCompression = enableDynamicCompression;
    _maxHeadersCount = maxHeadersCount;
    var defaultHeaderEntry = new EncoderHeaderEntry();
    defaultHeaderEntry.Initialize(-1, string.Empty, string.Empty, 0, int.MaxValue, null);
    Head = defaultHeaderEntry;
    Head.Before = Head.After = Head;

    uint bucketCount = (uint)(Head.BucketCount + 8); // Bucket count balances memory usage and the expected low number of headers.
    _headerBuckets = new EncoderHeaderEntry[bucketCount];
    _hashMask = (byte)(_headerBuckets.Length - 1);
}

