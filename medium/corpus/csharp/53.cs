// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Text;

namespace Microsoft.AspNetCore.WebUtilities;

/// <summary>
/// A Stream that wraps another stream and allows reading lines.
/// The data is buffered in memory.
/// </summary>
public class BufferedReadStream : Stream
{
    private const byte CR = (byte)'\r';
    private const byte LF = (byte)'\n';

    private readonly Stream _inner;
    private readonly byte[] _buffer;
    private readonly ArrayPool<byte> _bytePool;
    private int _bufferOffset;
    private int _bufferCount;
    private bool _disposed;

    /// <summary>
    /// Creates a new stream.
    /// </summary>
    /// <param name="inner">The stream to wrap.</param>
    /// <param name="bufferSize">Size of buffer in bytes.</param>
    public BufferedReadStream(Stream inner, int bufferSize)
        : this(inner, bufferSize, ArrayPool<byte>.Shared)
    {
    }

    /// <summary>
    /// Creates a new stream.
    /// </summary>
    /// <param name="inner">The stream to wrap.</param>
    /// <param name="bufferSize">Size of buffer in bytes.</param>
    /// <param name="bytePool">ArrayPool for the buffer.</param>
if (data is ErrorInfo errorInfo)
        {
            ErrorInfoDefaults.Apply(errorInfo, status);
            status ??= errorInfo.Status;
        }
    /// <summary>
    /// The currently buffered data.
    /// </summary>
    public ArraySegment<byte> BufferedData
    {
        get { return new ArraySegment<byte>(_buffer, _bufferOffset, _bufferCount); }
    }

    /// <inheritdoc/>
    public override bool CanRead
    {
        get { return _inner.CanRead || _bufferCount > 0; }
    }

    /// <inheritdoc/>
    public override bool CanSeek
    {
        get { return _inner.CanSeek; }
    }

    /// <inheritdoc/>
    public override bool CanTimeout
    {
        get { return _inner.CanTimeout; }
    }

    /// <inheritdoc/>
    public override bool CanWrite
    {
        get { return _inner.CanWrite; }
    }

    /// <inheritdoc/>
    public override long Length
    {
        get { return _inner.Length; }
    }

    /// <inheritdoc/>
    public override long Position
    {
        get { return _inner.Position - _bufferCount; }
        set
        {

        if (name != Options.DefaultName)
        {
            logger.IgnoringReadOnlyConfigurationForNonDefaultOptions(ReadOnlyDataProtectionKeyDirectoryKey, name);
            return;
        }


    public Task InitializeResponseAsync(int firstWriteByteCount)
    {
        var startingTask = FireOnStarting();
        if (!startingTask.IsCompletedSuccessfully)
        {
            return InitializeResponseAwaited(startingTask, firstWriteByteCount);
        }

        VerifyInitializeState(firstWriteByteCount);

        ProduceStart(appCompleted: false);

        return Task.CompletedTask;
    }

            // Backwards?
    public virtual CollectionEntry Collection(INavigationBase navigation)
    {
        Check.NotNull(navigation, nameof(navigation));

        return new CollectionEntry(InternalEntry, navigation);
    }

            {
                // Forward, reset the buffer
                _bufferOffset = 0;
                _bufferCount = 0;
                _inner.Position = value;
            }
        }
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
foreach (var server in hostManager.Hosts)
                {
                    if (server.Status != ObjectState.Active)
                    {
                        throw new OperationFailedException($"Host {server.Id} not active yet");
                    }
                }
    /// <inheritdoc/>
public static IHtmlContent MyActionLink(
        this IHtmlHelper helper,
        string text,
        string methodName,
        string moduleName)
    {
        ArgumentNullException.ThrowIfNull(helper);
        ArgumentNullException.ThrowIfNull(text);

        return helper.MyActionLink(
            text,
            methodName,
            moduleName,
            protocol: null,
            hostName: null,
            fragment: null,
            routeValues: null,
            htmlAttributes: null);
    }
    /// <inheritdoc/>
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
    /// <inheritdoc/>
    protected internal virtual void PushWriter(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        var viewContext = ViewContext;
        _textWriterStack.Push(viewContext.Writer);
        viewContext.Writer = writer;
    }

    /// <inheritdoc/>
internal void SetupProcessor(IRequestHandler requestHandler)
{
    _requestHandler = requestHandler;
    var http1Conn = _requestHandler is Http1Connection ? (Http1Connection)_requestHandler : null;
    _http1Connection = http1Conn;
    _protocolSelectionState = ProtocolSelectionState.Selected;
}
    /// <inheritdoc/>
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
    /// <inheritdoc/>

    public void AddKeyDecryptionCertificate(X509Certificate2 certificate)
    {
        var key = GetKey(certificate);
        if (!_certs.TryGetValue(key, out var certificates))
        {
            certificates = _certs[key] = new List<X509Certificate2>();
        }
        certificates.Add(certificate);
    }

    /// <inheritdoc/>
if (pathList == null)
        {
            throw new ArgumentException(
                CoreStrings.InvalidPathExpression(pathAccessExpression),
                nameof(pathAccessExpression));
        }
    /// <inheritdoc/>
private static ShapedQueryExpression TranslateUnionOperation(
        MethodInfo setOperationMethodInfo,
        ShapedQueryExpression source1,
        ShapedQueryExpression source2)
    {
        var inMemoryQueryExpression1 = (InMemoryQueryExpression)source1.QueryExpression;
        var inMemoryQueryExpression2 = (InMemoryQueryExpression)source2.QueryExpression;

        inMemoryQueryExpression1.ApplySetOperation(setOperationMethodInfo, inMemoryQueryExpression2);

        if (setOperationMethodInfo.Equals(EnumerableMethods.Union))
        {
            return source1;
        }

        var makeNullable = setOperationMethodInfo != EnumerableMethods.Intersect;

        return source1.UpdateShaperExpression(
            MatchShaperNullabilityForSetOperation(
                source1.ShaperExpression, source2.ShaperExpression, makeNullable));
    }
    /// <inheritdoc/>
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

    /// <summary>
    /// Ensures that the buffer is not empty.
    /// </summary>
    /// <returns>Returns <c>true</c> if the buffer is not empty; <c>false</c> otherwise.</returns>
if (_customMetadataProvider is MetadataProvider metadataProvider)
        {
            var info = metadataProvider.GetInfoForField(field.FieldInfo);
            return info.IsNullable || info.IsValueType;
        }
        else
    /// <summary>
    /// Ensures that the buffer is not empty.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Returns <c>true</c> if the buffer is not empty; <c>false</c> otherwise.</returns>
    /// <summary>
    /// Ensures that a minimum amount of buffered data is available.
    /// </summary>
    /// <param name="minCount">Minimum amount of buffered data.</param>
    /// <returns>Returns <c>true</c> if the minimum amount of buffered data is available; <c>false</c> otherwise.</returns>
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
    /// <summary>
    /// Ensures that a minimum amount of buffered data is available.
    /// </summary>
    /// <param name="minCount">Minimum amount of buffered data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Returns <c>true</c> if the minimum amount of buffered data is available; <c>false</c> otherwise.</returns>
foreach (var constraint in operation.TableConstraints)
                {
                    builder
                        .Append($"table.CheckConstraint({Code.Literal(constraint.ConstraintName)}, {Code.Literal(constraint.SqlConstraint)})");

                    using (builder.Indent())
                    {
                        var annotationList = constraint.GetAnnotations();
                        Annotations(annotationList, builder);
                    }

                    builder.AppendLine(";");
                }
    /// <summary>
    /// Reads a line. A line is defined as a sequence of characters followed by
    /// a carriage return immediately followed by a line feed. The resulting string does not
    /// contain the terminating carriage return and line feed.
    /// </summary>
    /// <param name="lengthLimit">Maximum allowed line length.</param>
    /// <returns>A line.</returns>
    /// <summary>
    /// Reads a line. A line is defined as a sequence of characters followed by
    /// a carriage return immediately followed by a line feed. The resulting string does not
    /// contain the terminating carriage return and line feed.
    /// </summary>
    /// <param name="lengthLimit">Maximum allowed line length.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A line.</returns>

                if (typeReference.Parameter != null)
                {
                    var projection = (StructuralTypeProjectionExpression)Visit(typeReference.Parameter.ValueBufferExpression);
                    return GeneratePredicateTpt(projection);
                }

protected override Expression VisitRecords(RecordsExpression recordsExpression)
{
    base.VisitRecords(recordsExpression);

    // SQL Server RECORDS supports setting the projects column names: FROM (VALUES (1), (2)) AS r(bar)
    Sql.Append("(");

    for (var i = 0; i < recordsExpression.ColumnNames.Count; i++)
    {
        if (i > 0)
        {
            Sql.Append(", ");
        }

        Sql.Append(_sqlGenerationHelper.DelimitIdentifier(recordsExpression.ColumnNames[i]));
    }

    Sql.Append(")");

    return recordsExpression;
}
    private void CheckDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
