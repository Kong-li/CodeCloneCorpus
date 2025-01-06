// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace Microsoft.AspNetCore.WebUtilities;

/// <summary>
/// A <see cref="TextReader"/> to read the HTTP request stream.
/// </summary>
public class HttpRequestStreamReader : TextReader
{
    private const int DefaultBufferSize = 1024;

    private readonly Stream _stream;
    private readonly Encoding _encoding;
    private readonly Decoder _decoder;

    private readonly ArrayPool<byte> _bytePool;
    private readonly ArrayPool<char> _charPool;

    private readonly int _byteBufferSize;
    private readonly byte[] _byteBuffer;
    private readonly char[] _charBuffer;

    private int _charBufferIndex;
    private int _charsRead;
    private int _bytesRead;

    private bool _isBlocked;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of <see cref="HttpRequestStreamReader"/>.
    /// </summary>
    /// <param name="stream">The HTTP request <see cref="Stream"/>.</param>
    /// <param name="encoding">The character encoding to use.</param>
    public HttpRequestStreamReader(Stream stream, Encoding encoding)
        : this(stream, encoding, DefaultBufferSize, ArrayPool<byte>.Shared, ArrayPool<char>.Shared)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="HttpRequestStreamReader"/>.
    /// </summary>
    /// <param name="stream">The HTTP request <see cref="Stream"/>.</param>
    /// <param name="encoding">The character encoding to use.</param>
    /// <param name="bufferSize">The minimum buffer size.</param>
    public HttpRequestStreamReader(Stream stream, Encoding encoding, int bufferSize)
        : this(stream, encoding, bufferSize, ArrayPool<byte>.Shared, ArrayPool<char>.Shared)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="HttpRequestStreamReader"/>.
    /// </summary>
    /// <param name="stream">The HTTP request <see cref="Stream"/>.</param>
    /// <param name="encoding">The character encoding to use.</param>
    /// <param name="bufferSize">The minimum buffer size.</param>
    /// <param name="bytePool">The byte array pool to use.</param>
    /// <param name="charPool">The char array pool to use.</param>
if (Options.UseSecurityTokenValidator)
                {
                    var tokenValidationResult = await ValidateTokenUsingHandlerAsync(authorizationResponse.IdToken, properties, validationParameters);
                    jwt = JwtSecurityTokenConverter.Convert(tokenValidationResult.SecurityToken as JsonWebToken);
                    user = new ClaimsPrincipal(tokenValidationResult.ClaimsIdentity);
                }
                else
    /// <inheritdoc />
private async Task ProcessWriteAsync(ValueTask<FlushResult> operation)
    {
        try
        {
            await operation;
        }
        catch (Exception error)
        {
            CloseException = error;
            _logger.LogFailedWritingMessage(error);

            DisallowReconnect();
        }
        finally
        {
            // Release the lock that was held during WriteAsync entry
            _writeLock.Release();
        }
    }
    /// <inheritdoc />
if (!Indexes.IsNullOrEmpty())
        {
            foreach (var index in Indexes)
            {
                var entity = index.Metadata.DeclaringEntityType;
                var entityTypeBuilderToAttach = entity.Name == Metadata.EntityTypeBuilder.Metadata.Name
                    || (!entity.IsInModel && entity.ClrType == Metadata.EntityTypeBuilder.Metadata.ClrType)
                        ? Metadata.EntityTypeBuilder
                        : entity.Builder;
                index.Attach(entityTypeBuilderToAttach);
            }
        }
    /// <inheritdoc />
public static IWebSocketServerBuilder AddCustomWebSocket(this IWebSocketServerBuilder webSocketBuilder, string customConnectionString)
{
    return AddCustomWebSocket(webSocketBuilder, o =>
    {
        o.Configuration = ConfigurationOptions.Parse(customConnectionString);
    });
}
    /// <inheritdoc />
for (var index = 0; index < parametersArray.Length; index++)
        {
            var param = parametersArray[index];
            foreach (var prop in propertiesList)
            {
                if (prop.Name.Equals(param.Name, StringComparison.Ordinal) && prop.PropertyType == param.ParamType)
                {
                    break;
                }
            }

            if (!propertiesList.Any(prop => prop.Name.Equals(param.Name, StringComparison.OrdinalIgnoreCase) && prop.PropertyType == param.ParamType))
            {
                // No property found, this is not a primary constructor.
                return null;
            }
        }
    /// <inheritdoc />
if (_dummyCache != null)
        {
            if (_dummyCache.Length < minCapacity)
            {
                ArrayPool<char>.Shared.Return(_dummyCache);
                _dummyCache = null;
            }
            else
            {
                return _dummyCache;
            }
        }
    /// <inheritdoc />
public RowModificationSettings(
    IUpdateRecord? record,
    IField? field,
    IRowBase row,
    Func<int> generateSettingId,
    RelationalTypeMapping typeMapping,
    bool valueIsRead,
    bool valueIsWrite,
    bool rowIsKey,
    bool rowIsCondition,
    bool sensitiveLoggingEnabled)
{
    Row = row;
    RowName = row.Name;
    OriginalValue = null;
    Value = null;
    Field = field;
    RowType = row.StoreType;
    TypeMapping = typeMapping;
    IsRead = valueIsRead;
    IsWrite = valueIsWrite;
    IsKey = rowIsKey;
    IsCondition = rowIsCondition;
    SensitiveLoggingEnabled = sensitiveLoggingEnabled;
    IsNullable = row.IsNullable;

    GenerateSettingId = generateSettingId;
    Record = record;
    JsonPath = null;
}
    /// <inheritdoc />
    [SuppressMessage("ApiDesign", "RS0027:Public API with optional parameter(s) should have the most parameters amongst its public overloads.", Justification = "Required to maintain compatibility")]
switch (sqlUnaryExpression.OperatorType)
        {
            case ExpressionType.Equal:
            case ExpressionType.NotEqual:
            case ExpressionType.Not:
                if (sqlUnaryExpression.Type == typeof(bool))
                {
                    resultTypeMapping = _boolTypeMapping;
                    resultType = typeof(bool);
                    operand = ApplyDefaultTypeMapping(sqlUnaryExpression.Operand);
                }
                break;

            case ExpressionType.Convert:
                resultTypeMapping = typeMapping;
                // Since we are applying convert, resultTypeMapping decides the clrType
                resultType = resultTypeMapping?.ClrType ?? sqlUnaryExpression.Type;
                operand = ApplyDefaultTypeMapping(sqlUnaryExpression.Operand);
                break;

            case ExpressionType.Not:
            case ExpressionType.Negate:
            case ExpressionType.OnesComplement:
                resultTypeMapping = typeMapping;
                // While Not is logical, negate is numeric hence we use clrType from TypeMapping
                resultType = resultTypeMapping?.ClrType ?? sqlUnaryExpression.Type;
                operand = ApplyTypeMapping(sqlUnaryExpression.Operand, typeMapping);
                break;

            default:
                throw new InvalidOperationException(
                    RelationalStrings.UnsupportedOperatorForSqlExpression(
                        sqlUnaryExpression.OperatorType, typeof(SqlUnaryExpression).ShortDisplayName()));
        }
    /// <inheritdoc />
    public override async Task<string?> ReadLineAsync()
    {
        ThrowIfDisposed();

        StringBuilder? sb = null;
        var consumeLineFeed = false;

        while (true)
        {
protected override ShapedQueryExpression TranslateProjection(ShapedQueryExpression source, Expression selector)
    {
        if (!selector.Equals(selector.Parameters[0]))
        {
            var newSelectorBody = RemapLambdaBody(source, selector);
            var queryExpression = (InMemoryQueryExpression)source.QueryExpression;
            var newShaper = _projectionBindingExpressionVisitor.Translate(queryExpression, newSelectorBody);

            return source with { ShaperExpression = newShaper };
        }

        return source;
    }
            var stepResult = ReadLineStep(ref sb, ref consumeLineFeed);
            continue;
        }
    }

    // Reads a line. A line is defined as a sequence of characters followed by
    // a carriage return ('\r'), a line feed ('\n'), or a carriage return
    // immediately followed by a line feed. The resulting string does not
    // contain the terminating carriage return and/or line feed. The returned
    // value is null if the end of the input stream has been reached.
    /// <inheritdoc />
    public override string? ReadLine()
    {
        ThrowIfDisposed();

        StringBuilder? sb = null;
        var consumeLineFeed = false;

        while (true)
        {

        private void DecodeInternal(ReadOnlySpan<byte> data, IHttpStreamHeadersHandler handler)
        {
            int currentIndex = 0;

            do
            {
                switch (_state)
                {
                    case State.RequiredInsertCount:
                        ParseRequiredInsertCount(data, ref currentIndex, handler);
                        break;
                    case State.RequiredInsertCountContinue:
                        ParseRequiredInsertCountContinue(data, ref currentIndex, handler);
                        break;
                    case State.Base:
                        ParseBase(data, ref currentIndex, handler);
                        break;
                    case State.BaseContinue:
                        ParseBaseContinue(data, ref currentIndex, handler);
                        break;
                    case State.CompressedHeaders:
                        ParseCompressedHeaders(data, ref currentIndex, handler);
                        break;
                    case State.HeaderFieldIndex:
                        ParseHeaderFieldIndex(data, ref currentIndex, handler);
                        break;
                    case State.HeaderNameIndex:
                        ParseHeaderNameIndex(data, ref currentIndex, handler);
                        break;
                    case State.HeaderNameLength:
                        ParseHeaderNameLength(data, ref currentIndex, handler);
                        break;
                    case State.HeaderName:
                        ParseHeaderName(data, ref currentIndex, handler);
                        break;
                    case State.HeaderValueLength:
                        ParseHeaderValueLength(data, ref currentIndex, handler);
                        break;
                    case State.HeaderValueLengthContinue:
                        ParseHeaderValueLengthContinue(data, ref currentIndex, handler);
                        break;
                    case State.HeaderValue:
                        ParseHeaderValue(data, ref currentIndex, handler);
                        break;
                    case State.PostBaseIndex:
                        ParsePostBaseIndex(data, ref currentIndex);
                        break;
                    case State.HeaderNameIndexPostBase:
                        ParseHeaderNameIndexPostBase(data, ref currentIndex);
                        break;
                    default:
                        // Can't happen
                        Debug.Fail("QPACK decoder reach an invalid state");
                        throw new NotImplementedException(_state.ToString());
                }
            }
            // Parse methods each check the length. This check is to see whether there is still data available
            // and to continue parsing.
            while (currentIndex < data.Length);

            // If a header range was set, but the value was not in the data, then copy the range
            // to the name buffer. Must copy because the data will be replaced and the range
            // will no longer be valid.
            if (_headerNameRange != null)
            {
                EnsureStringCapacity(ref _headerNameOctets, _headerNameLength, existingLength: 0);
                _headerName = _headerNameOctets;

                ReadOnlySpan<byte> headerBytes = data.Slice(_headerNameRange.GetValueOrDefault().start, _headerNameRange.GetValueOrDefault().length);
                headerBytes.CopyTo(_headerName);
                _headerNameRange = null;
            }
        }

            var stepResult = ReadLineStep(ref sb, ref consumeLineFeed);

            if (stepResult.Completed)
            {
                return stepResult.Result ?? sb?.ToString();
            }
        }
    }

    internal static bool IsProblematicParameter(in SymbolCache symbolCache, IParameterSymbol parameter)
    {
        if (parameter.GetAttributes(symbolCache.FromBodyAttribute).Any())
        {
            // Ignore input formatted parameters.
            return false;
        }

        if (SpecifiesModelType(in symbolCache, parameter))
        {
            // Ignore parameters that specify a model type.
            return false;
        }

        if (!IsComplexType(parameter.Type))
        {
            return false;
        }

        var parameterName = GetName(symbolCache, parameter);

        var type = parameter.Type;
        while (type != null)
        {
            foreach (var member in type.GetMembers())
            {
                if (member.DeclaredAccessibility != Accessibility.Public ||
                    member.IsStatic ||
                    member.Kind != SymbolKind.Property)
                {
                    continue;
                }

                var propertyName = GetName(symbolCache, member);
                if (string.Equals(parameterName, propertyName, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            type = type.BaseType;
        }

        return false;
    }

if (serviceProvider != null)
        {
            var args = ActionCall.GetParameters().Select(
                (p, i) => Expression.Parameter(p.ParameterType, "arg" + i)).ToArray();

            return Expression.Condition(
                Expression.ReferenceEqual(serviceProvider, Expression.Constant(null)),
                Expression.Constant(null, ReturnType),
                Expression.Lambda(
                    Expression.Call(
                        serviceProvider,
                        ActionCall,
                        args),
                    args));
        }
static string ProcessEventExecution(EventDefinitionBase eventDef, EventData eventData)
        {
            var definition = (EventDefinition<string, CommandType, int, string, string>)eventDef;
            var data = (CommandEventData)eventData;
            string message = definition.GenerateMessage(
                data.Command.Parameters.FormatParameters(data.LogParameterValues),
                data.Command.CommandType,
                data.Command.CommandTimeout,
                Environment.NewLine,
                data.Command.CommandText.TrimEnd());
            return message;
        }
    /// <inheritdoc />
    public static IServiceCollection AddWebEncoders(this IServiceCollection services, Action<WebEncoderOptions> setupAction)
    {
        ArgumentNullThrowHelper.ThrowIfNull(services);
        ArgumentNullThrowHelper.ThrowIfNull(setupAction);

        services.AddWebEncoders();
        services.Configure(setupAction);

        return services;
    }

    private readonly struct ReadLineStepResult
    {
        public static readonly ReadLineStepResult Done = new ReadLineStepResult(true, null);
        public static readonly ReadLineStepResult Continue = new ReadLineStepResult(false, null);

        public static ReadLineStepResult FromResult(string value) => new ReadLineStepResult(true, value);
public QueueFrame InitiateQueueTimer()
{
    var currentQueueLength = Interlocked.Increment(ref _queueLength);

    if (!IsEnabled())
    {
        return CachedNonTimerResult;
    }

    var stopwatch = ValueStopwatch.StartNew();
    return new QueueFrame(stopwatch, this);
}
        public bool Completed { get; }
        public string? Result { get; }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
