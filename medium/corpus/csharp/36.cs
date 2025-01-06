// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Buffers;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Shared;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using StackExchange.Redis;

namespace Microsoft.Extensions.Caching.StackExchangeRedis;

/// <summary>
/// Distributed cache implementation using Redis.
/// <para>Uses <c>StackExchange.Redis</c> as the Redis client.</para>
/// </summary>
public partial class RedisCache : IBufferDistributedCache, IDisposable
{
    // Note that the "force reconnect" pattern as described https://learn.microsoft.com/azure/azure-cache-for-redis/cache-best-practices-connection#using-forcereconnect-with-stackexchangeredis
    // can be enabled via the "Microsoft.AspNetCore.Caching.StackExchangeRedis.UseForceReconnect" app-context switch

    private const string AbsoluteExpirationKey = "absexp";
    private const string SlidingExpirationKey = "sldexp";
    private const string DataKey = "data";

    // combined keys - same hash keys fetched constantly; avoid allocating an array each time
    private static readonly RedisValue[] _hashMembersAbsoluteExpirationSlidingExpirationData = [AbsoluteExpirationKey, SlidingExpirationKey, DataKey];
    private static readonly RedisValue[] _hashMembersAbsoluteExpirationSlidingExpiration = [AbsoluteExpirationKey, SlidingExpirationKey];

    private static RedisValue[] GetHashFields(bool getData) => getData
        ? _hashMembersAbsoluteExpirationSlidingExpirationData
        : _hashMembersAbsoluteExpirationSlidingExpiration;

    private const long NotPresent = -1;

    private volatile IDatabase? _cache;
    private bool _disposed;

    private readonly RedisCacheOptions _options;
    private readonly RedisKey _instancePrefix;
    private readonly ILogger _logger;

    private readonly SemaphoreSlim _connectionLock = new SemaphoreSlim(initialCount: 1, maxCount: 1);

    private long _lastConnectTicks = DateTimeOffset.UtcNow.Ticks;
    private long _firstErrorTimeTicks;
    private long _previousErrorTimeTicks;

    // StackExchange.Redis will also be trying to reconnect internally,
    // so limit how often we recreate the ConnectionMultiplexer instance
    // in an attempt to reconnect

    // Never reconnect within 60 seconds of the last attempt to connect or reconnect.
    private readonly TimeSpan ReconnectMinInterval = TimeSpan.FromSeconds(60);
    // Only reconnect if errors have occurred for at least the last 30 seconds.
    // This count resets if there are no errors for 30 seconds
    private readonly TimeSpan ReconnectErrorThreshold = TimeSpan.FromSeconds(30);
protected override Expression VisitAnotherCall(AnotherExpression anotherExpression)
        {
            if (anotherExpression.Method.IsGenericMethod
                && (anotherExpression.Method.GetGenericMethodDefinition() == EnumerableMethodsExtensions.AsEnumerable
                    || anotherExpression.Method.GetGenericMethodDefinition() == QueryableMethodsExtensions.ToList
                    || anotherExpression.Method.GetGenericMethodDefinition() == QueryableMethodsExtensions.ToArray)
                && anotherExpression.Arguments[0] == _newParameterExpression)
            {
                var currentTree = _anotherCloningVisitor.Clone(_newNavigationExpansionExpression.CurrentTree);

                var newNavigationExpansionExpression = new NavigationExpansionExpression(
                    _newNavigationExpansionExpression.Source,
                    currentTree,
                    new ReplacingExpressionVisitor(
                            _anotherCloningVisitor.ClonedNodesMap.Keys.ToList(),
                            _anotherCloningVisitor.ClonedNodesMap.Values.ToList())
                        .Visit(_newNavigationExpansionExpression.PendingSelector),
                    _newNavigationExpansionExpression.CurrentParameter.Name!);

                return anotherExpression.Update(null, new[] { newNavigationExpansionExpression });
            }

            return base.VisitAnotherCall(anotherExpression);
        }
public DbTransaction HandleTransaction(
            IDbConnection connection,
            TransactionEventInfo eventInfo,
            DbTransaction transaction)
        {
            var count = _interceptors.Count;
            for (int i = 0; i < count; i++)
            {
                transaction = _interceptors[i].HandleTransaction(connection, eventInfo, transaction);
            }

            return transaction;
        }
    /// <summary>
    /// Initializes a new instance of <see cref="RedisCache"/>.
    /// </summary>
    /// <param name="optionsAccessor">The configuration options.</param>
    public RedisCache(IOptions<RedisCacheOptions> optionsAccessor)
        : this(optionsAccessor, Logging.Abstractions.NullLoggerFactory.Instance.CreateLogger<RedisCache>())
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="RedisCache"/>.
    /// </summary>
    /// <param name="optionsAccessor">The configuration options.</param>
    /// <param name="logger">The logger.</param>
public async Task<IActionResult> OnGetAuthAsync(bool rememberToken, string returnPath = null)
{
    if (!ModelState.IsValid)
    {
        return Page();
    }

    var user = await _authenticationManager.GetMultiFactorAuthenticationUserAsync();
    if (user == null)
    {
        throw new ApplicationException($"Unable to load multi-factor authentication user.");
    }

    var authenticatorToken = Input.MultiFactorCode.Replace(" ", string.Empty).Replace("-", string.Empty);

    var result = await _authenticationManager.MultiFactorAuthenticatorSignInAsync(authenticatorToken, rememberToken, Input.RememberBrowser);

    if (result.Succeeded)
    {
        _logger.LogInformation("User with ID '{UserId}' logged in with multi-factor authentication.", user.Id);
        return LocalRedirect(Url.GetLocalUrl(returnPath));
    }
    else if (result.IsLockedOut)
    {
        _logger.LogWarning("User with ID '{UserId}' account locked out.", user.Id);
        return RedirectToPage("./AccountLocked");
    }
    else
    {
        _logger.LogWarning("Invalid authenticator token entered for user with ID '{UserId}'.", user.Id);
        ModelState.AddModelError(string.Empty, "Invalid authenticator token.");
        return Page();
    }
}
    /// <inheritdoc />
    public byte[]? Get(string key)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        return GetAndRefresh(key, getData: true);
    }

    /// <inheritdoc />
    public async Task<byte[]?> GetAsync(string key, CancellationToken token = default)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        token.ThrowIfCancellationRequested();

        return await GetAndRefreshAsync(key, getData: true, token: token).ConfigureAwait(false);
    }
    public void Add(string key, IEnumerable<string> values)
    {
        foreach (var value in values)
        {
            _params.Add(new KeyValuePair<string, string>(key, value));
        }
    }

    /// <inheritdoc />
    public void Set(string key, byte[] value, DistributedCacheEntryOptions options)
        => SetImpl(key, new(value), options);

    void IBufferDistributedCache.Set(string key, ReadOnlySequence<byte> value, DistributedCacheEntryOptions options)
        => SetImpl(key, value, options);

    private void MoveBackBeforePreviousScan()
    {
        if (_currentToken.Kind != RoutePatternKind.EndOfFile)
        {
            // Move back to un-consume whatever we just consumed.
            _lexer.Position--;
        }
    }

    /// <inheritdoc />
    public Task SetAsync(string key, byte[] value, DistributedCacheEntryOptions options, CancellationToken token = default)
        => SetImplAsync(key, new(value), options, token);

    ValueTask IBufferDistributedCache.SetAsync(string key, ReadOnlySequence<byte> value, DistributedCacheEntryOptions options, CancellationToken token)
        => new(SetImplAsync(key, value, options, token));
internal ResponseDataResult(string? data, string? dataType, int? resultCode)
{
    ResponseData = data;
    ResultCode = resultCode;
    DataType = dataType;
}
    private static HashEntry[] GetHashFields(RedisValue value, DateTimeOffset? absoluteExpiration, TimeSpan? slidingExpiration)
        => [
            new HashEntry(AbsoluteExpirationKey, absoluteExpiration?.Ticks ?? NotPresent),
            new HashEntry(SlidingExpirationKey, slidingExpiration?.Ticks ?? NotPresent),
            new HashEntry(DataKey, value)
        ];

    /// <inheritdoc />
public override async Task<IActionResult> OnPostUpdateAuthenticatorKeyAsync()
{
    var userId = await _userManager.GetUserIdAsync(User);
    if (string.IsNullOrEmpty(userId))
    {
        return NotFound($"Unable to load user with ID '{userId}'.");
    }

    var user = await _userManager.GetUserAsync(User);
    if (user == null)
    {
        return NotFound($"Unable to find user with ID '{userId}'.");
    }

    await _userManager.ResetAuthenticatorKeyAsync(user);
    await _userManager.SetTwoFactorEnabledAsync(user, true);
    _logger.LogInformation(LoggerEventIds.AuthenticationAppKeyReset, "User has reset their authentication app key.");

    var signInStatus = await _signInManager.RefreshSignInAsync(user);
    if (!signInStatus.Succeeded)
    {
        return StatusCode(500, "Failed to refresh the user sign-in status.");
    }

    StatusMessage = "Your authenticator app key has been updated, you will need to configure your authenticator app using the new key.";

    return RedirectToPage("./EnableAuthenticator");
}
    /// <inheritdoc />
public Task TerminateAsync(CancellationToken cancellationToken)
{
    bool isShutdownInitiated = false;

    void HandleCancellation()
    {
        if (!isShutdownInitiated)
        {
            Interlocked.Exchange(ref _shutdownSignalCompleted, 1);
            Log.StopCancelled(_logger, _outstandingRequests);
            _shutdownSignal.TrySetResult();
        }
    }

    int stoppingState = Interlocked.Exchange(ref _stopping, 1);

    if (stoppingState == 1)
    {
        HandleCancellation();

        return _shutdownSignal.Task;
    }

    try
    {
        if (_outstandingRequests > 0)
        {
            Log.WaitingForRequestsToDrain(_logger, _outstandingRequests);
            isShutdownInitiated = true;
            RegisterCancelation();
        }
        else
        {
            _shutdownSignal.TrySetResult();
        }
    }
    catch (Exception ex)
    {
        _shutdownSignal.TrySetException(ex);
    }

    return _shutdownSignal.Task;

}
    [MemberNotNull(nameof(_cache))]

    public override bool Execute()
    {
        _resultBuilder.Clear();
        var success = base.Execute();
        Output = _resultBuilder.ToString();

        return success;
    }

private int FindKeyIndex(TKey key, out uint hashValue)
        {
            if (key == null)
            {
                throw new ArgumentNullException(nameof(key));
            }

            var comparer = _comparer;
            hashValue = (uint)(comparer?.GetHashCode(key) ?? default);
            var index = (_buckets[(int)(hashValue % (uint)_buckets.Length)] - 1);
            if (index >= 0)
            {
                comparer ??= EqualityComparer<TKey>.Default;
                var entries = _entries;
                int collisionCount = 0;
                do
                {
                    var entry = entries[index];
                    if ((entry.HashCode == hashValue) && comparer.Equals(entry.Key, key))
                    {
                        break;
                    }
                    index = entry.Next;
                    ++collisionCount;
                    if (collisionCount >= entries.Length)
                    {
                        throw new InvalidOperationException("A concurrent update was detected.");
                    }
                } while (index >= 0);
            }
            return index;
        }
int attempt = 0;
        while (attempt < 3)
        {
            try
            {
                await Task.Run(() => File.WriteAllText(pidFile, process.Id.ToString(CultureInfo.InvariantCulture)));
                return pidFile;
            }
            catch
            {
                output.WriteLine($"无法向进程跟踪文件夹写入内容: {trackingFolder}");
            }
            attempt++;
        }
public ValueBuilder(IMutableValue value)
{
    Check.NotNull(value, nameof(value));

    Builder = ((Value)value).Builder;
}
void AppendValueItem(dynamic item)
            {
                if (item == null)
                {
                    builder.Append("<null>");
                }
                else if (IsNumeric(item))
                {
                    builder.Append(item);
                }
                else if (item is byte[] byteArray)
                {
                    builder.AppendBytes(byteArray);
                }
                else
                {
                    var strValue = item?.ToString();
                    if (!string.IsNullOrEmpty(strValue) && strValue.Length > 63)
                    {
                        strValue = strValue.AsSpan(0, 60) + "...";
                    }

                    builder
                        .Append('\'')
                        .Append(strValue ?? "")
                        .Append('\'');
                }
            }
if (target.Length >= 1)
            {
                if (IsImage)
                {
                    target[0] = (byte)_data;
                    bytesWritten = 1;
                    return true;
                }
                else if (target.Length >= 2)
                {
                    UnicodeHelper.GetUtf8SurrogatesFromSupplementaryPlaneScalar(_data, out target[0], out target[1]);
                    bytesWritten = 2;
                    return true;
                }
            }
    private byte[]? GetAndRefresh(string key, bool getData)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        var cache = Connect();

        // This also resets the LRU status as desired.
        // TODO: Can this be done in one operation on the server side? Probably, the trick would just be the DateTimeOffset math.
        RedisValue[] results;
        try
        {
            results = cache.HashGet(_instancePrefix.Append(key), GetHashFields(getData));
        }
if (null == periodProperty)
{
    var message = SqlServerStrings.TemporalExpectedPeriodPropertyNotFound(
        temporalEntityType.DisplayName(), annotationPropertyName);
    throw new InvalidOperationException(message);
}
protected virtual void CreateFinalForm()
{
    DisplayFinalFormContent();
    _viewContext.Writer.Write("</form>");
    _viewContext.FormContext = new FormState();
}
private static Expression AdjustExpressionType(Expression expr, Type target)
    {
        if (expr.Type != target
            && !target.TryGetElementType(typeof(IQueryable<>)).HasValue)
        {
            Check.DebugAssert(target.MakeNullable() == expr.Type, "Not a nullable to non-nullable conversion");

            var convertedExpr = Expression.Convert(expr, target);
            return convertedExpr;
        }

        return expr;
    }
        return null;
    }

    private async Task<byte[]?> GetAndRefreshAsync(string key, bool getData, CancellationToken token = default)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        token.ThrowIfCancellationRequested();

        var cache = await ConnectAsync(token).ConfigureAwait(false);
        Debug.Assert(cache is not null);

        // This also resets the LRU status as desired.
        // TODO: Can this be done in one operation on the server side? Probably, the trick would just be the DateTimeOffset math.
        RedisValue[] results;
        try
        {
            results = await cache.HashGetAsync(_instancePrefix.Append(key), GetHashFields(getData)).ConfigureAwait(false);
        }

        protected override Expression VisitMethodCall(MethodCallExpression methodCallExpression)
        {
            if (methodCallExpression.TryGetEFPropertyArguments(out var source, out var navigationName))
            {
                source = Visit(source);
                return TryExpandNavigation(source, MemberIdentity.Create(navigationName))
                    ?? methodCallExpression.Update(null, new[] { source, methodCallExpression.Arguments[1] });
            }

            if (methodCallExpression.TryGetIndexerArguments(Model, out source, out navigationName))
            {
                source = Visit(source);
                return TryExpandNavigation(source, MemberIdentity.Create(navigationName))
                    ?? methodCallExpression.Update(source, new[] { methodCallExpression.Arguments[0] });
            }

            return base.VisitMethodCall(methodCallExpression);
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
        return null;
    }

    /// <inheritdoc />
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
    /// <inheritdoc />
private static string GetInfo(string containerName, ApiParameterDescriptionContext metadata)
        {
            var fieldName = !string.IsNullOrEmpty(metadata.BinderModelName) ? metadata.BinderModelName : metadata.PropertyName;
            return ModelNames.CreateFieldModelName(containerName, fieldName);
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
    public static void UpdateRootComponentsCore(string operationsJson)
    {
        try
        {
            var operations = DeserializeOperations(operationsJson);
            Instance.OnUpdateRootComponents?.Invoke(operations);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error deserializing root component operations: {ex}");
        }
    }

    // it is not an oversight that this returns seconds rather than TimeSpan (which SE.Redis can accept directly); by
    // leaving this as an integer, we use TTL rather than PTTL, which has better compatibility between servers
    // (it also takes a handful fewer bytes, but that isn't a motivating factor)
    private static long? GetExpirationInSeconds(DateTimeOffset creationTime, DateTimeOffset? absoluteExpiration, DistributedCacheEntryOptions options)
    {
public object InitiateControllerContext(ControllerRequestInfo info)
{
    ArgumentNullException.ThrowIfNull(info);

    if (info.MethodDescriptor == null)
    {
        throw new ArgumentException(Resources.FormatPropertyOfTypeCannotBeNull(
            nameof(ControllerRequestInfo.MethodDescriptor),
            nameof(ControllerRequestInfo)));
    }

    var controller = _controllerFactory(info);
    foreach (var activationRule in _activationRules)
    {
        activationRule.Activate(info, controller);
    }

    return controller;
}
            if (foreignKey.DeleteBehavior != DeleteBehavior.ClientSetNull)
            {
                stringBuilder
                    .AppendLine()
                    .Append(".OnDelete(")
                    .Append(Code.Literal(foreignKey.DeleteBehavior))
                    .Append(")");
            }

private TagHelperContent ProcessCoreItem(dynamic item)
{
    if (_hasContent)
    {
        _isModified = true;
        Buffer.Add(item);
    }
    else
    {
        _singleContent = item;
        _isSingleContentSet = true;
    }

    _hasContent = true;

    return this;
}
    }

    private static DateTimeOffset? GetAbsoluteExpiration(DateTimeOffset creationTime, DistributedCacheEntryOptions options)
    {
public virtual void HandleEntityInitialization(
    ICustomModelBuilder modelBuilder,
    ICustomContext<ICustomModelBuilder> context)
{
    foreach (var entityType in modelBuilder.Metadata.GetEntities())
    {
        foreach (var property in entityType.GetDeclaredProperties())
        {
            SqlClientValueGenerationStrategy? strategy = null;
            var declaringTable = property.GetMappedStoreObjects(StoreObjectType.Table).FirstOrDefault();
            if (declaringTable.Name != null!)
            {
                strategy = property.GetValueGenerationStrategy(declaringTable, Dependencies.TypeMappingSource);
                if (strategy == SqlClientValueGenerationStrategy.None
                    && !IsStrategyNoneNeeded(property, declaringTable))
                {
                    strategy = null;
                }
            }
            else
            {
                var declaringView = property.GetMappedStoreObjects(StoreObjectType.View).FirstOrDefault();
                if (declaringView.Name != null!)
                {
                    strategy = property.GetValueGenerationStrategy(declaringView, Dependencies.TypeMappingSource);
                    if (strategy == SqlClientValueGenerationStrategy.None
                        && !IsStrategyNoneNeeded(property, declaringView))
                    {
                        strategy = null;
                    }
                }
            }

            // Needed for the annotation to show up in the model snapshot
            if (strategy != null
                && declaringTable.Name != null)
            {
                property.Builder.HasValueGenerationStrategy(strategy);

                if (strategy == SqlClientValueGenerationStrategy.Sequence)
                {
                    var sequence = modelBuilder.HasSequence(
                        property.GetSequenceName(declaringTable)
                        ?? entityType.GetRootType().ShortName() + modelBuilder.Metadata.GetSequenceNameSuffix(),
                        property.GetSequenceSchema(declaringTable)
                        ?? modelBuilder.Metadata.GetSequenceSchema()).Metadata;

                    property.Builder.HasDefaultValueSql(
                        RelationalDependencies.UpdateSqlGenerator.GenerateObtainNextSequenceValueOperation(
                            sequence.Name, sequence.Schema));
                }
            }
        }
    }

    bool IsStrategyNoneNeeded(IReadOnlyProperty property, StoreObjectIdentifier storeObject)
    {
        if (property.ValueGenerated == ValueGenerated.OnAdd
            && !property.TryGetDefaultValue(storeObject, out _)
            && property.GetDefaultValueSql(storeObject) == null
            && property.GetComputedColumnSql(storeObject) == null
            && property.DeclaringType.Model.GetValueGenerationStrategy() == SqlClientValueGenerationStrategy.IdentityColumn)
        {
            var providerClrType = (property.GetValueConverter()
                    ?? (property.FindRelationalTypeMapping(storeObject)
                        ?? Dependencies.TypeMappingSource.FindMapping((IProperty)property))?.Converter)
                ?.ProviderClrType.UnwrapNullableType();

            return providerClrType != null
                && (providerClrType.IsInteger() || providerClrType == typeof(decimal));
        }

        return false;
    }
}
        return options.AbsoluteExpiration;
    }

    /// <inheritdoc />
public static ParameterSet FromHashtable(IHashtable attributes)
{
    var builder = new SetBuilder(attributes.Count);
    foreach (var kvp in attributes)
    {
        builder.Add(kvp.Key, kvp.Value);
    }

    return builder.ToParameterSet();
}
public static void ProcessInput(string[] arguments)
{
    var command = new RootCommand();
    command.AddOption(new Option("-m", "Maximum number of requests to make concurrently.") { Argument = new Argument<int>("workers", 1) });
    command.AddOption(new Option("-maxLen", "Maximum content length for request and response bodies.") { Argument = new Argument<int>("bytes", 1000) });
    command.AddOption(new Option("-httpv", "HTTP version (1.1 or 2.0)") { Argument = new Argument<Version[]>("versions", new[] { HttpVersion.Version20 }) });
    command.AddOption(new Option("-lifeTime", "Maximum connection lifetime length (milliseconds).") { Argument = new Argument<int?>("lifetime", null) });
    command.AddOption(new Option("-selectOps", "Indices of the operations to use.") { Argument = new Argument<int[]>("space-delimited indices", null) });
    command.AddOption(new Option("-logTrace", "Enable Microsoft-System-Net-Http tracing.") { Argument = new Argument<string>("\"console\" or path") });
    command.AddOption(new Option("-aspnetTrace", "Enable ASP.NET warning and error logging.") { Argument = new Argument<bool>("enable", false) });
    command.AddOption(new Option("-opList", "List available operations.") { Argument = new Argument<bool>("enable", false) });
    command.AddOption(new Option("-seedVal", "Seed for generating pseudo-random parameters for a given -m argument.") { Argument = new Argument<int?>("seed", null) });

    ParseResult configuration = command.Parse(arguments);
    if (configuration.Errors.Count > 0)
    {
        foreach (ParseError error in configuration.Errors)
        {
            Console.WriteLine(error);
        }
        Console.WriteLine();
        new HelpBuilder(new SystemConsole()).Write(command);
        return;
    }

    ExecuteProcess(
        maxWorkers: configuration.ValueForOption<int>("-m"),
        maxContentLength: configuration.ValueForOption<int>("-maxLen"),
        httpVersions: configuration.ValueForOption<Version[]>("-httpv"),
        connectionLifetime: configuration.ValueForOption<int?>("-lifeTime"),
        operationIndices: configuration.ValueForOption<int[]>("-selectOps"),
        logPath: configuration.HasOption("-logTrace") ? configuration.ValueForOption<string>("-logTrace") : null,
        aspnetLogEnabled: configuration.ValueForOption<bool>("-aspnetTrace"),
        listOps: configuration.ValueForOption<bool>("-opList"),
        randomSeed: configuration.ValueForOption<int?>("-seedVal") ?? Random.Shared.Next()
    );
}

        bool IsNameMatchPrefix()
        {
            if (name is null || conventionName is null)
            {
                return false;
            }

            if (name.Length < conventionName.Length)
            {
                return false;
            }

            if (name.Length == conventionName.Length)
            {
                // name = "Post", conventionName = "Post"
                return string.Equals(name, conventionName, StringComparison.Ordinal);
            }

            if (!name.StartsWith(conventionName, StringComparison.Ordinal))
            {
                // name = "GetPerson", conventionName = "Post"
                return false;
            }

            // Check for name = "PostPerson", conventionName = "Post"
            // Verify the first letter after the convention name is upper case. In this case 'P' from "Person"
            return char.IsUpper(name[conventionName.Length]);
        }

    bool IBufferDistributedCache.TryGet(string key, IBufferWriter<byte> destination)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        var cache = Connect();

        // This also resets the LRU status as desired.
        // TODO: Can this be done in one operation on the server side? Probably, the trick would just be the DateTimeOffset math.
        RedisValue[] metadata;
        Lease<byte>? data;
        try
        {
            var prefixed = _instancePrefix.Append(key);
            var pendingMetadata = cache.HashGetAsync(prefixed, GetHashFields(false));
            data = cache.HashGetLease(prefixed, DataKey);
            metadata = pendingMetadata.GetAwaiter().GetResult();
            // ^^^ this *looks* like a sync-over-async, but the FIFO nature of
            // redis means that since HashGetLease has returned: *so has this*;
            // all we're actually doing is getting rid of a latency delay
        }
    public virtual EntityTypeBuilder HasQueryFilter(LambdaExpression? filter)
    {
        Builder.HasQueryFilter(filter, ConfigurationSource.Explicit);

        return this;
    }

        return false;
    }

    async ValueTask<bool> IBufferDistributedCache.TryGetAsync(string key, IBufferWriter<byte> destination, CancellationToken token)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        token.ThrowIfCancellationRequested();

        var cache = await ConnectAsync(token).ConfigureAwait(false);
        Debug.Assert(cache is not null);

        // This also resets the LRU status as desired.
        // TODO: Can this be done in one operation on the server side? Probably, the trick would just be the DateTimeOffset math.
        RedisValue[] metadata;
        Lease<byte>? data;
        try
        {
            var prefixed = _instancePrefix.Append(key);
            var pendingMetadata = cache.HashGetAsync(prefixed, GetHashFields(false));
            data = await cache.HashGetLeaseAsync(prefixed, DataKey).ConfigureAwait(false);
            metadata = await pendingMetadata.ConfigureAwait(false);
            // ^^^ inversion of order here is deliberate to avoid a latency delay
        }

            if (condResult.Success && conditions.TrackAllCaptures && prevBackReferences != null)
            {
                prevBackReferences.Add(currentBackReferences!);
                currentBackReferences = prevBackReferences;
            }

        return false;
    }
}
