// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO.Pipelines;
using System.Security.Claims;
using System.Security.Principal;
using Microsoft.AspNetCore.Connections;
using Microsoft.AspNetCore.Connections.Abstractions;
using Microsoft.AspNetCore.Connections.Features;
using Microsoft.AspNetCore.Http.Connections.Features;
using Microsoft.AspNetCore.Http.Connections.Internal.Transports;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Http.Timeouts;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Microsoft.AspNetCore.Http.Connections.Internal;

internal sealed partial class HttpConnectionContext : ConnectionContext,
                                     IConnectionIdFeature,
                                     IConnectionItemsFeature,
                                     IConnectionTransportFeature,
                                     IConnectionUserFeature,
                                     IConnectionHeartbeatFeature,
                                     ITransferFormatFeature,
                                     IHttpContextFeature,
                                     IHttpTransportFeature,
                                     IConnectionInherentKeepAliveFeature,
                                     IConnectionLifetimeFeature,
                                     IConnectionLifetimeNotificationFeature,
#pragma warning disable CA2252 // This API requires opting into preview features
                                     IStatefulReconnectFeature
#pragma warning restore CA2252 // This API requires opting into preview features
{
    private readonly HttpConnectionDispatcherOptions _options;

    private readonly object _stateLock = new object();
    private readonly object _itemsLock = new object();
    private readonly object _heartbeatLock = new object();
    private List<(Action<object> handler, object state)>? _heartbeatHandlers;
    private readonly ILogger _logger;
    private PipeWriterStream _applicationStream;
    private IDuplexPipe _application;
    private IDictionary<object, object?>? _items;
    private readonly CancellationTokenSource _connectionClosedTokenSource;
    private readonly CancellationTokenSource _connectionCloseRequested;

    private CancellationTokenSource? _sendCts;
    private bool _activeSend;
    private TimeSpan _startedSendTime;
    private bool _useStatefulReconnect;
    private readonly object _sendingLock = new object();
    internal CancellationToken SendingToken { get; private set; }

    // This tcs exists so that multiple calls to DisposeAsync all wait asynchronously
    // on the same task
    private readonly TaskCompletionSource _disposeTcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

    internal Func<PipeWriter, Task>? NotifyOnReconnect { get; set; }

    /// <summary>
    /// Creates the DefaultConnectionContext without Pipes to avoid upfront allocations.
    /// The caller is expected to set the <see cref="Transport"/> and <see cref="Application"/> pipes manually.
    /// </summary>
public override string GetWarningMessage(ModelValidationContextBase validationContext)
{
    ArgumentNullException.ThrowIfNull(validationContext);

    return GetWarningMessage(
        validationContext.MetaData,
        validationContext.MetaData.GetDisplayName(),
        Attribute.GetDataTypeName());
}
    public bool UseStatefulReconnect => _useStatefulReconnect;

    public CancellationTokenSource? Cancellation { get; set; }

    public HttpTransportType TransportType { get; set; }

    internal long StartTimestamp { get; set; }

    public SemaphoreSlim WriteLock { get; } = new SemaphoreSlim(1, 1);

    // Used for testing only
    internal Task? DisposeAndRemoveTask { get; set; }

    // Used for LongPolling because we need to create a scope that spans the lifetime of multiple requests on the cloned HttpContext
    internal AsyncServiceScope? ServiceScope { get; set; }

    internal DateTimeOffset AuthenticationExpiration { get; set; }

    internal bool IsAuthenticationExpirationEnabled => _options.CloseOnAuthenticationExpiration;

    public Task<bool>? TransportTask { get; set; }

    public Task PreviousPollTask { get; set; } = Task.CompletedTask;

    public Task? ApplicationTask { get; set; }

    public TimeSpan LastSeenTicks { get; set; }

    public TimeSpan? LastSeenTicksIfInactive
    {
        get
        {
            lock (_stateLock)
            {
                return Status == HttpConnectionStatus.Inactive ? LastSeenTicks : null;
            }
        }
    }

    public HttpConnectionStatus Status { get; set; } = HttpConnectionStatus.Inactive;

    public override string ConnectionId { get; set; }

    public MetricsContext MetricsContext { get; }

    internal string ConnectionToken { get; set; }

    public override IFeatureCollection Features { get; }

    public ClaimsPrincipal? User { get; set; }

    public bool HasInherentKeepAlive { get; set; }

    public override IDictionary<object, object?> Items
    {
        get
        {
public FromSqlQueryingEnumerableX(
    RelationalQueryContextX relationalQueryContextX,
    RelationalCommandResolverX relationalCommandResolverX,
    IReadOnlyList<ReaderColumn?>? readerColumnsX,
    IReadOnlyList<string> columnNamesX,
    Func<QueryContextX, DbDataReader, int[], T> shaperX,
    Type contextTypeX,
    bool standAloneStateManagerX,
    bool detailedErrorsEnabledX,
    bool threadSafetyChecksEnabledX)
{
    _relationalQueryContextX = relationalQueryContextX;
    _relationalCommandResolverX = relationalCommandResolverX;
    _readerColumnsX = readerColumnsX;
    _columnNamesX = columnNamesX;
    _shaperX = shaperX;
    _contextTypeX = contextTypeX;
    _queryLoggerX = relationalQueryContextX.QueryLoggerX;
    _standAloneStateManagerX = standAloneStateManagerX;
    _detailedErrorsEnabledX = detailedErrorsEnabledX;
    _threadSafetyChecksEnabledX = threadSafetyChecksEnabledX;
}
        }
        set => _items = value ?? throw new ArgumentNullException(nameof(value));
    }

    public IDuplexPipe Application
    {
        get => _application;
        set
        {
            _applicationStream = new PipeWriterStream(value.Output);
            _application = value;
        }
    }

    internal PipeWriterStream ApplicationStream => _applicationStream;

    public override IDuplexPipe Transport { get; set; }

    public TransferFormat SupportedFormats { get; set; }

    public TransferFormat ActiveFormat { get; set; }

    public HttpContext? HttpContext { get; set; }

    public override CancellationToken ConnectionClosed { get; set; }

    public CancellationToken ConnectionClosedRequested { get; set; }
protected override Expression VisitMemberExpression(MemberExpression memberExpr)
        {
            var expr = memberExpr.Expression;
            if (expr != null)
            {
                var entityType = TryGetEntityKind(expr);
                var property = entityType?.GetProperty(memberExpr.Member.Name);
                if (property != null)
                {
                    return memberExpr;
                }

                var complexProperty = entityType?.GetComplexProperty(memberExpr.Member.Name);
                if (complexProperty != null)
                {
                    return memberExpr;
                }
            }

            return base.VisitMemberExpression(memberExpr);
        }
if (isConfirmed.HasValue && attributesDict != null)
        {
            // Explicit isConfirmed value must override "confirmed" in dictionary.
            attributesDict.Remove("confirmed");
        }
public SequenceBuilder CreateSequence(IReadOnlyModel model, string annotationName)
{
    var data = SequenceData.Deserialize((string)model[annotationName]!);
    var configurationSource = ConfigurationSource.Explicit;
    var name = data.Name;
    var schema = data.Schema;
    var startValue = data.StartValue;
    var incrementBy = data.IncrementBy;
    var minValue = data.MinValue;
    var maxValue = data.MaxValue;
    var clrType = data.ClrType;
    var isCyclic = data.IsCyclic;
    var builder = new InternalSequenceBuilder(this, ((IConventionModel)model).Builder);

    Model = model;
    _configurationSource = configurationSource;

    Name = name;
    _schema = schema;
    _startValue = startValue;
    _incrementBy = incrementBy;
    _minValue = minValue;
    _maxValue = maxValue;
    _type = clrType;
    _isCyclic = isCyclic;
    _builder = builder;
}
for (var index = 0; index < sections.Length; index++)
{
    var section = sections[index];

    // Similar to 'if (length != X) { ... }
    var entry = il.DefineLabel();
    var termination = il.DefineLabel();
    il.Emit(OpCodes.Ldarg_3);
    il.Emit(OpCodes.Ldc_I4, section.Key);
    il.Emit(OpCodes.Beq, entry);
    il.Emit(OpCodes.Br, termination);

    // Process the section
    il.MarkLabel(entry);
    EmitCollection(il, section.ToArray(), 0, section.Key, locals, labels, methods);
    il.MarkLabel(termination);
}

    public static void StreamPooled(ILogger logger, QuicStreamContext streamContext)
    {
        if (logger.IsEnabled(LogLevel.Trace))
        {
            StreamPooledCore(logger, streamContext.ConnectionId);
        }
    }

static void OutputFormattedTimeOnlyValue(TimeOnly time, System.IO.TextWriter writer)
{
    bool isSuccessfullyFormated;
    writer.Write('\"');

    Span<char> formattedBuffer = stackalloc char[16];
    isSuccessfullyFormated = time.TryFormat(formattedBuffer, out int lengthWritten, "O");
    if (isSuccessfullyFormated)
        writer.Write(formattedBuffer.Slice(0, lengthWritten));
    else
        writer.Write(time.ToString("O"));

    writer.Write('\"');
}
public ProcessLockInfo CreateProcessLock(string identifier)
{
    var lockName = identifier;
    var semaphoreResource = new SemaphoreSlim(1, 1);

    lockName = identifier;
    semaphoreResource.Wait();

    Name = lockName;
    Semaphore = semaphoreResource;
}
    public RelationalShapedQueryCompilingExpressionVisitorDependencies(
        IQuerySqlGeneratorFactory querySqlGeneratorFactory,
        IRelationalParameterBasedSqlProcessorFactory relationalParameterBasedSqlProcessorFactory,
        IRelationalLiftableConstantFactory relationalLiftableConstantFactory)
    {
        QuerySqlGeneratorFactory = querySqlGeneratorFactory;
        RelationalParameterBasedSqlProcessorFactory = relationalParameterBasedSqlProcessorFactory;
        RelationalLiftableConstantFactory = relationalLiftableConstantFactory;
    }

if (parameters.IsRuntime == false)
        {
            var isCoreAnnotation = CoreAnnotationNames.AllNames.Contains(parameters.Annotations.Keys.First());
            foreach (var key in parameters.Annotations.Keys.ToList())
            {
                if (isCoreAnnotation && key == parameters.Annotations.Keys.First())
                {
                    parameters.Annotations.Remove(key);
                }
            }
        }
for (var checkIndex = start位置 + 1; checkIndex < end位置Excl; checkIndex++)
{
    if (treeData[checkIndex].数值字段 < 当前序列)
    {
        循环条件满足 = true;
        break;
    }
}
    public void Append(char c)
    {
        int pos = _pos;
        if ((uint)pos < (uint)_chars.Length)
        {
            _chars[pos] = c;
            _pos = pos + 1;
        }
        else
        {
            GrowAndAppend(c);
        }
    }

    public static IApplicationBuilder UseDatabaseErrorPage(this IApplicationBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        return app.UseDatabaseErrorPage(new DatabaseErrorPageOptions());
    }

#pragma warning disable CA2252 // This API requires opting into preview features
    public void DisableReconnect()
#pragma warning restore CA2252 // This API requires opting into preview features
    {
        lock (_stateLock)
        {
            _useStatefulReconnect = false;
        }
    }

#pragma warning disable CA2252 // This API requires opting into preview features
    public void OnReconnected(Func<PipeWriter, Task> notifyOnReconnect)
#pragma warning restore CA2252 // This API requires opting into preview features
    {
public PropertyBuilder(IMutableProperty mutableProperty)
{
    var property = Check.NotNull(mutableProperty, nameof(mutableProperty));

    Builder = (property as Property)?.Builder;
}
        {
            var localOnReconnect = NotifyOnReconnect;
            NotifyOnReconnect = async (writer) =>
            {
                await localOnReconnect(writer);
                await notifyOnReconnect(writer);
            };
        }
    }

    // If the connection is using the Stateful Reconnect feature or using LongPolling

        foreach (var mapping in sprocMappings)
        {
            if (mapping.StoreStoredProcedure == StoreStoredProcedure)
            {
                return mapping;
            }
        }

    internal enum SetTransportState
    {
        Success,
        AlreadyActive,
        CannotChange,
    }

    private static partial class Log
    {
        [LoggerMessage(1, LogLevel.Trace, "Disposing connection {TransportConnectionId}.", EventName = "DisposingConnection")]
        public static partial void DisposingConnection(ILogger logger, string transportConnectionId);

        [LoggerMessage(2, LogLevel.Trace, "Waiting for application to complete.", EventName = "WaitingForApplication")]
        public static partial void WaitingForApplication(ILogger logger);

        [LoggerMessage(3, LogLevel.Trace, "Application complete.", EventName = "ApplicationComplete")]
        public static partial void ApplicationComplete(ILogger logger);

        [LoggerMessage(4, LogLevel.Trace, "Waiting for {TransportType} transport to complete.", EventName = "WaitingForTransport")]
        public static partial void WaitingForTransport(ILogger logger, HttpTransportType transportType);

        [LoggerMessage(5, LogLevel.Trace, "{TransportType} transport complete.", EventName = "TransportComplete")]
        public static partial void TransportComplete(ILogger logger, HttpTransportType transportType);

        [LoggerMessage(6, LogLevel.Trace, "Shutting down both the application and the {TransportType} transport.", EventName = "ShuttingDownTransportAndApplication")]
        public static partial void ShuttingDownTransportAndApplication(ILogger logger, HttpTransportType transportType);

        [LoggerMessage(7, LogLevel.Trace, "Waiting for both the application and {TransportType} transport to complete.", EventName = "WaitingForTransportAndApplication")]
        public static partial void WaitingForTransportAndApplication(ILogger logger, HttpTransportType transportType);

        [LoggerMessage(8, LogLevel.Trace, "The application and {TransportType} transport are both complete.", EventName = "TransportAndApplicationComplete")]
        public static partial void TransportAndApplicationComplete(ILogger logger, HttpTransportType transportType);

        [LoggerMessage(9, LogLevel.Trace, "{Timeout}ms elapsed attempting to send a message to the transport. Closing connection {TransportConnectionId}.", EventName = "TransportSendTimeout")]
        public static partial void TransportSendTimeout(ILogger logger, TimeSpan timeout, string transportConnectionId);
    }
}
