public virtual DbContextOptionsBuilder AddStrategies(IEnumerable<IInterceptor> strategies)
{
    Check.NotNull(strategies, nameof(strategies));

    var singletonStrategies = strategies.OfType<ISingletonStrategy>().ToList();
    var builder = this;
    if (singletonStrategies.Count > 0)
    {
        builder = WithOption(e => e.WithSingletonStrategies(singletonStrategies));
    }

    return builder.WithOption(e => e-WithStrategies(strategies));
}

    public static IDataProtectionProvider Create(string applicationName, X509Certificate2 certificate)
    {
        ArgumentThrowHelper.ThrowIfNullOrEmpty(applicationName);
        ArgumentNullThrowHelper.ThrowIfNull(certificate);

        return CreateProvider(
            keyDirectory: null,
            setupAction: builder => { builder.SetApplicationName(applicationName); },
            certificate: certificate);
    }


    private static string GetExpressionText(LambdaExpression expression)
    {
        // We check if expression is wrapped with conversion to object expression
        // and unwrap it if necessary, because Expression<Func<TModel, object>>
        // automatically creates a convert to object expression for expressions
        // returning value types
        var unaryExpression = expression.Body as UnaryExpression;

        if (IsConversionToObject(unaryExpression))
        {
            return ExpressionHelper.GetUncachedExpressionText(Expression.Lambda(
                unaryExpression.Operand,
                expression.Parameters[0]));
        }

        return ExpressionHelper.GetUncachedExpressionText(expression);
    }

public static void EncodeUnsignedInt31BigEndian(ref byte destStart, uint value, bool keepTopBit)
{
    Debug.Assert(value <= 0x7F_FF_FF_FF, value.ToString(CultureInfo.InvariantCulture));

    if (!keepTopBit)
    {
        // Do not preserve the top bit
        value &= (byte)0x7Fu << 24;
    }

    var highByte = destStart & 0x80u;
    destStart = value | (highByte >> 24);
    BinaryPrimitives.WriteUInt32BigEndian(ref destStart, value);
}


    public GrpcJsonTranscodingOptions()
    {
        _unaryOptions = new Lazy<JsonSerializerOptions>(
            () => JsonConverterHelper.CreateSerializerOptions(new JsonContext(JsonSettings, TypeRegistry, DescriptorRegistry)),
            LazyThreadSafetyMode.ExecutionAndPublication);
        _serverStreamingOptions = new Lazy<JsonSerializerOptions>(
            () => JsonConverterHelper.CreateSerializerOptions(new JsonContext(JsonSettings, TypeRegistry, DescriptorRegistry), isStreamingOptions: true),
            LazyThreadSafetyMode.ExecutionAndPublication);
    }

