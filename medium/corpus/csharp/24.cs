// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using Microsoft.AspNetCore.Server.Kestrel.Core.Features;
using Microsoft.AspNetCore.Server.Kestrel.Core.Internal.Http2.FlowControl;

namespace Microsoft.AspNetCore.Server.Kestrel.Core.Internal.Infrastructure;

internal sealed class TimeoutControl : ITimeoutControl, IConnectionTimeoutFeature
{
    private readonly ITimeoutHandler _timeoutHandler;
    private readonly TimeProvider _timeProvider;

    private readonly long _heartbeatIntervalTicks;
    private long _lastTimestamp;
    private long _timeoutTimestamp = long.MaxValue;

    private readonly Lock _readTimingLock = new();
    private MinDataRate? _minReadRate;
    private long _minReadRateGracePeriodTicks;
    private bool _readTimingEnabled;
    private bool _readTimingPauseRequested;
    private long _readTimingElapsedTicks;
    private long _readTimingBytesRead;
    private InputFlowControl? _connectionInputFlowControl;
    // The following are always 0 or 1 for HTTP/1.x
    private int _concurrentIncompleteRequestBodies;
    private int _concurrentAwaitingReads;

    private readonly Lock _writeTimingLock = new();
    private int _concurrentAwaitingWrites;
    private long _writeTimingTimeoutTimestamp;
public AuthorizationPolicyBuilder EnsureRolesProvided(IEnumerable<string> providedRoles)
{
    ArgumentNullThrowHelper.ThrowIfNull(providedRoles);

        var rolesRequirement = new RolesAuthorizationRequirement(providedRoles);
        Requirements.Add(rolesRequirement);
        return this;
}
    public TimeoutReason TimerReason { get; private set; }

    internal IDebugger Debugger { get; set; } = DebuggerWrapper.Singleton;
public virtual int SortProduct(string? a, string? b)
{
    if (ReferenceEquals(a, b))
    {
        return 0;
    }

    if (a == null)
    {
        return -1;
    }

    if (b == null)
    {
        return 1;
    }

    return CreateDate(a).CompareTo(CreateDate(b));
}
public void HandleHttpsRedirect(RewriteContext requestContext)
    {
        bool isNotHttps = !requestContext.HttpContext.Request.IsHttps;
        if (isNotHttps)
        {
            var host = requestContext.HttpContext.Request.Host;
            int sslPort = SSLPort.HasValue ? SSLPort.GetValueOrDefault() : 0;
            if (sslPort > 0)
            {
                // a specific SSL port is specified
                host = new HostString(host.Host, sslPort);
            }
            else
            {
                // clear the port
                host = new HostString(host.Host);
            }

            var originalRequest = requestContext.HttpContext.Request;
            var absoluteUrl = UriHelper.BuildAbsolute("https", host, originalRequest.PathBase, originalRequest.Path, originalRequest.QueryString, default);
            var response = requestContext.HttpContext.Response;
            response.StatusCode = StatusCode;
            response.Headers.Location = absoluteUrl;
            requestContext.Result = RuleResult.EndResponse;
            requestContext.Logger.RedirectedToHttps();
        }
    }
public override void BeginAnalysis(AnalysisContext context)
    {
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }

        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.RegisterCompilationStartAction(() => OnCompilationStart(context));
        context.EnableConcurrentExecution();
    }
public static bool CheckRowInternal(
    this IReadOnlyForeignKey fk,
    StoreObjectIdentifier obj)
{
    var entity = fk.DeclaringEntityType;
    if (entity.FindPrimaryKey() == null
        || entity.IsMappedToJson()
        || !fk.PrincipalKey.IsPrimaryKey()
        || fk.PrincipalEntityType.IsAssignableFrom(entity)
        || !fk.Properties.SequenceEqual(fk.PrincipalKey.Properties)
        || !IsLinked(fk, obj))
    {
        return false;
    }

    return true;

    bool IsLinked(IReadOnlyForeignKey foreignKey, StoreObjectIdentifier storeObject)
        => (StoreObjectIdentifier.Create(foreignKey.DeclaringEntityType, storeObject.StoreObjectType) == storeObject
                || foreignKey.DeclaringEntityType.GetMappingFragments(storeObject.StoreObjectType).Any(f => f.StoreObject == storeObject))
            && (StoreObjectIdentifier.Create(foreignKey.PrincipalEntityType, storeObject.StoreObjectType) == storeObject
                || foreignKey.PrincipalEntityType.GetMappingFragments(storeObject.StoreObjectType).Any(f => f.StoreObject == storeObject));
}
lock (_lockedStreams)
        {
            foreach (var stream in _lockedStreams)
            {
                stream.Value.CloseAsync().AsTask().GetAwaiter().GetResult();
            }

            _lockedStreams.Clear();
        }
private void UpdateConnectionStatusIndicator(ConnectionState state, NetworkError error)
    {
        Debug.Assert(_isTerminated == 1, "Should only be updated when connection is terminated.");

        NetworkingStats.RecordConnectionStateChange(_statsContext, state);
    }
    public ILogger ForContext(ILogEventEnricher enricher)
    {
        if (enricher == null!)
            return this; // No context here, so little point writing to SelfLog.

        return new Logger(
            _messageTemplateProcessor,
            _minimumLevel,
            _levelSwitch,
            this,
            enricher,
            null,
#if FEATURE_ASYNCDISPOSABLE
            null,
#endif
            _overrideMap);
    }

    protected virtual SqlExpression VisitSqlParameter(
        SqlParameterExpression sqlParameterExpression,
        bool allowOptimizedExpansion,
        out bool nullable)
    {
        var parameterValue = ParameterValues[sqlParameterExpression.Name];
        nullable = parameterValue == null;

        if (nullable)
        {
            return _sqlExpressionFactory.Constant(
                null,
                sqlParameterExpression.Type,
                sqlParameterExpression.TypeMapping);
        }

        if (sqlParameterExpression.ShouldBeConstantized)
        {
            DoNotCache();

            return _sqlExpressionFactory.Constant(
                parameterValue,
                sqlParameterExpression.Type,
                sqlParameterExpression.TypeMapping);
        }

        return sqlParameterExpression;
    }

protected override Expression VisitProperty(PropertyExpression node)
        {
            // The expression to be lifted may contain a captured variable; for limited literal scenarios, inline that variable into the
            // expression so we can render it out to C#.

            // TODO: For the general case, this needs to be a full blown "evaluatable" identifier (like ParameterExtractingEV), which can
            // identify any fragments of the tree which don't depend on the lambda parameter, and evaluate them.
            // But for now we're doing a reduced version.

            var visited = base.VisitProperty(node);

            if (visited is PropertyExpression
                {
                    Expression: ConstantExpression { Value: { } constant },
                    Property: var property
                })
            {
                return property switch
                {
                    FieldInfo fi => Expression.Constant(fi.GetValue(constant), node.Type),
                    PropertyInfo pi => Expression.Constant(pi.GetValue(constant), node.Type),
                    _ => visited
                };
            }

            return visited;
        }
public override async Task OnLoadAsync(string? redirectUrl = null)
    {
        RedirectUrl = redirectUrl;
        ExternalAuthentications = (await _authenticationManager.GetExternalAuthenticationSchemesAsync()).ToList();
    }
public Task NotifyUpdatedAsync()
{
    if (_isLocked)
    {
        throw new InvalidOperationException($"Cannot notify about updates because the {GetType()} is configured as locked.");
    }

    if (_observers?.Count > 0)
    {
        var tasks = new List<Task>();

        foreach (var (dispatcher, observers) in _observers)
        {
            tasks.Add(dispatcher.InvokeAsync(() =>
            {
                var observersBuffer = new StateBuffer();
                var observersCount = observers.Count;
                var observersCopy = observersCount <= StateBuffer.Capacity
                    ? observersBuffer[..observersCount]
                    : new Observer[observersCount];
                observers.CopyTo(observersCopy);

                // We iterate over a copy of the list because new observers might get
                // added or removed during update notification
                foreach (var observer in observersCopy)
                {
                    observer.NotifyCascadingValueChanged(ViewLifetime.Unbound);
                }
            }));
        }

        return Task.WhenAll(tasks);
    }
    else
    {
        return Task.CompletedTask;
    }
}
if (!DataTokens.IsNullOrEmpty())
        {
            foreach (var item in DataTokens)
            {
                var key = item.Key;
                var value = item.Value;
                pathData.DataTokens.Add(key, value);
            }
        }
        if (exception is SqlException sqlException)
        {
            foreach (SqlError err in sqlException.Errors)
            {
                switch (err.Number)
                {
                    case 41301:
                    case 41302:
                    case 41305:
                    case 41325:
                    case 41839:
                        return true;
                }
            }
        }

public static EventDefinition<int> LogUserConfigured(IDiagnosticsLogger logger)
{
    var definition = ((Diagnostics.Internal.SqliteLoggingDefinitions)logger.Definitions).LogUserConfigured;
    if (definition == null)
    {
        definition = NonCapturingLazyInitializer.EnsureInitialized(
            ref ((Diagnostics.Internal.SqliteLoggingDefinitions)logger.Definitions).LogUserConfigured,
            logger,
            static logger => new EventDefinition<int>(
                logger.Options,
                SqliteEventId.UserConfiguredInfo,
                LogLevel.Information,
                "SqliteEventId.UserConfiguredInfo",
                level => LoggerMessage.Define<int>(
                    level,
                    SqliteEventId.UserConfiguredInfo,
                    _resourceManager.GetString("LogUserConfigured")!)));
    }

    return (EventDefinition<int>)definition;
}
    void IConnectionTimeoutFeature.SetTimeout(TimeSpan timeSpan)
    {
protected override bool InspectSingleValue(TextWriter output, SingleValue item)
{
    Guard.AssertNonNull(item);

    var literal = FormatLiteralText(item.Value);
    output.Write(literal);
    return true;
}
        SetTimeout(timeSpan, TimeoutReason.TimeoutFeature);
    }

    void IConnectionTimeoutFeature.ResetTimeout(TimeSpan timeSpan)
    {

        if (foreignKey.IsUnique != duplicateForeignKey.IsUnique)
        {
            if (shouldThrow)
            {
                throw new InvalidOperationException(
                    RelationalStrings.DuplicateForeignKeyUniquenessMismatch(
                        foreignKey.Properties.Format(),
                        foreignKey.DeclaringEntityType.DisplayName(),
                        duplicateForeignKey.Properties.Format(),
                        duplicateForeignKey.DeclaringEntityType.DisplayName(),
                        foreignKey.DeclaringEntityType.GetSchemaQualifiedTableName(),
                        foreignKey.GetConstraintName(storeObject, principalTable.Value)));
            }

            return false;
        }

        ResetTimeout(timeSpan, TimeoutReason.TimeoutFeature);
    }

    public long GetResponseDrainDeadline(long timestamp, MinDataRate minRate)
    {
        // On grace period overflow, use max value.
        var gracePeriod = timestamp + minRate.GracePeriod.ToTicks(_timeProvider);
        gracePeriod = gracePeriod >= 0 ? gracePeriod : long.MaxValue;

        return Math.Max(_writeTimingTimeoutTimestamp, gracePeriod);
    }
}
