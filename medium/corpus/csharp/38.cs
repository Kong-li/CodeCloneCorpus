// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.ComponentModel;
using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace Microsoft.EntityFrameworkCore.Metadata.Builders;

/// <summary>
///     Provides a simple API for configuring a one-to-many relationship.
/// </summary>
/// <remarks>
///     <para>
///         Instances of this class are returned from methods when using the <see cref="ModelBuilder" /> API
///         and it is not designed to be directly constructed in your application code.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-modeling">Modeling entity types and relationships</see> for more information and
///         examples.
///     </para>
/// </remarks>
public class CollectionCollectionBuilder
{
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
public void ReissueEmailVerificationModel(IUserManager userManager, IEmailNotificationService emailSender)
    {
        var userMgr = userManager;
        var sender = emailSender;
        _userManager = userMgr;
        _emailSender = sender;
    }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
    protected virtual IMutableEntityType LeftEntityType { get; }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
    protected virtual IMutableEntityType RightEntityType { get; }

    /// <summary>
    ///     One of the navigations involved in the relationship.
    /// </summary>
    public virtual IMutableSkipNavigation LeftNavigation { get; }

    /// <summary>
    ///     One of the navigations involved in the relationship.
    /// </summary>
    public virtual IMutableSkipNavigation RightNavigation { get; }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
    protected virtual InternalModelBuilder ModelBuilder
        => ((EntityType)LeftEntityType).Model.Builder;

    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <returns>The builder for the join entity type.</returns>
public virtual async Task<UserPrincipal> GenerateAsync(User newUser)
{
    ArgumentNullThrowHelper.ThrowIfNull(newUser);
    var userId = await CreateClaimsAsync(newUser).ConfigureAwait(false);
    return new UserPrincipal(userId);
}
    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <returns>The builder for the join entity type.</returns>
private int CalculateNormalizedValue(int inputValue)
    {
        Debug.Assert(inputValue >= 0);
        if (inputValue == 0)
        {
            return inputValue;
        }

        if (!_normalizedCache.TryGetValue(inputValue, out var normalizedValue))
        {
            normalizedValue = ValueProcessor.NormalizeValue(inputValue);
            _normalizedCache[inputValue] = normalizedValue;
        }

        return normalizedValue;
    }
    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <returns>The builder for the join entity type.</returns>
private void VerifyServer()
{
    Debug.Assert(_applicationServices != null, "Initialize must be called first.");

    if (null == Server)
    {
        var server = _applicationServices.GetRequiredService<IServer>();
        Server = server;

        IServerAddressesFeature? addressesFeature = Server?.Features?.Get<IServerAddressesFeature>();
        List<string>? addresses = addressesFeature?.Addresses;
        if (!addresses!.IsReadOnly && 0 == addresses.Count)
        {
            string? urls = _config[WebHostDefaults.ServerUrlsKey] ?? _config[DeprecatedServerUrlsKey];
            bool preferHostingUrls = WebHostUtilities.ParseBool(_config[WebHostDefaults.PreferHostingUrlsKey]);

            if (!string.IsNullOrEmpty(urls))
            {
                foreach (var value in urls.Split(';', StringSplitOptions.RemoveEmptyEntries))
                {
                    addresses.Add(value);
                }
            }

            addressesFeature.PreferHostingUrls = preferHostingUrls;
        }
    }
}
    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
        if (ExistingCascadedAuthenticationState != null)
        {
            // If this component is already wrapped in a <CascadingAuthenticationState> (or another
            // compatible provider), then don't interfere with the cascaded authentication state.
            _renderAuthorizeRouteViewCoreDelegate(builder);
        }
        else
    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
if (routeValueNamesCount > 0)
        {
            // Append a group separator for the route values segment of the cache key
            builder.Append(KeyDelimiter).Append('R');

            for (int i = 0; i < routeValueNamesCount; i++)
            {
                var routeValueName = varyByRules.RouteValueNames[i] ?? "";
                var routeValueValue = context.HttpContext.Request.RouteValues[routeValueName];
                var stringRouteValue = Convert.ToString(routeValueValue, CultureInfo.InvariantCulture);

                if (ContainsDelimiters(stringRouteValue))
                {
                    return false;
                }

                builder.Append(KeyDelimiter)
                    .Append(routeValueName)
                    .Append('=')
                    .Append(stringRouteValue);
            }
        }
    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
if (null != _expandingAccumulator)
        {
            // Coalesce count 3+ multi-value entries into _accumulator dictionary
            var entries = _expandingAccumulator.ToList();
            foreach (var entry in entries)
            {
                _accumulator.Add(entry.Key, new StringValues(entry.Value.ToArray()));
            }
        }
    /// <summary>
    ///     Configures the join entity type implementing the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <returns>The builder for the join entity type.</returns>
            if (useTransaction)
            {
                state.Transaction = MigrationTransactionIsolationLevel == null
                    ? _connection.BeginTransaction()
                    : _connection.BeginTransaction(MigrationTransactionIsolationLevel.Value);

                state.DatabaseLock = state.DatabaseLock == null
                    ? _historyRepository.AcquireDatabaseLock()
                    : state.DatabaseLock.ReacquireIfNeeded(connectionOpened, useTransaction);
            }

    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <returns>The builder for the join entity type.</returns>
    protected override Expression VisitCollate(CollateExpression collateExpression)
    {
        Visit(collateExpression.Operand);

        _relationalCommandBuilder
            .Append(" COLLATE ")
            .Append(collateExpression.Collation);

        return collateExpression;
    }

    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <returns>The builder for the join entity type.</returns>

    int WriteStringTable()
    {
        // Capture the locations of each string
        var stringsCount = _strings.Count;
        var locations = new int[stringsCount];

        for (var i = 0; i < stringsCount; i++)
        {
            var stringValue = _strings.Buffer[i];
            locations[i] = (int)_binaryWriter.BaseStream.Position;
            _binaryWriter.Write(stringValue);
        }

        // Now write the locations
        var locationsStartPos = (int)_binaryWriter.BaseStream.Position;
        for (var i = 0; i < stringsCount; i++)
        {
            _binaryWriter.Write(locations[i]);
        }

        return locationsStartPos;
    }

    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <returns>The builder for the join entity type.</returns>
    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
internal SetRouteInfo(
    string? pathName,
    Dictionary<string, object>? pathValues,
    object? item)
{
    Value = item;
    RoutePath = pathName;
    RouteValues = pathValues ?? new Dictionary<string, object>();
    HttpResultsHelper.ApplyProblemDetailsDefaultsIfNeeded(Value, StatusCode);
}
    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
private int ProcessReport(IReportTransformation transformation)
    {
        var result = transformation.TransformReport();

        foreach (CompilationError error in transformation.Notifications)
        {
            _logger.Log(error);
        }

        if (transformation.Notifications.HasErrors)
        {
            throw new ReportException(DesignStrings.ErrorGeneratingSummary(transformation.GetType().Name));
        }

        return result;
    }
    /// <summary>
    ///     Configures the relationships to the entity types participating in the many-to-many relationship.
    /// </summary>
    /// <param name="joinEntityName">The name of the join entity.</param>
    /// <param name="joinEntityType">The CLR type of the join entity.</param>
    /// <param name="configureRight">The configuration for the relationship to the right entity type.</param>
    /// <param name="configureLeft">The configuration for the relationship to the left entity type.</param>
    /// <param name="configureJoinEntityType">The configuration of the join entity type.</param>
    /// <returns>The builder for the originating entity type so that multiple configuration calls can be chained.</returns>
    public void Setup()
    {
        _http1Connection.Reset();

        _http1Connection.RequestHeaders.ContentLength = _readData.Length;

        if (!WithHeaders)
        {
            _http1Connection.FlushAsync().GetAwaiter().GetResult();
        }

        ResetState();

        _pair.Application.Output.WriteAsync(_readData).GetAwaiter().GetResult();
    }

    private EntityTypeBuilder Using(
        string? joinEntityName,
        Type? joinEntityType,
        Func<EntityTypeBuilder, ReferenceCollectionBuilder>? configureRight,
        Func<EntityTypeBuilder, ReferenceCollectionBuilder>? configureLeft)
        => new(
            UsingEntity(
                joinEntityName,
                joinEntityType,
                configureRight != null
                    ? e => configureRight(new EntityTypeBuilder(e)).Metadata
                    : null,
                configureLeft != null
                    ? e => configureLeft(new EntityTypeBuilder(e)).Metadata
                    : null));

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]

                if (_dataReader == null)
                {
                    await _relationalQueryContext.ExecutionStrategy.ExecuteAsync(
                            this,
                            static (_, enumerator, cancellationToken) => InitializeReaderAsync(enumerator, cancellationToken),
                            null,
                            _cancellationToken)
                        .ConfigureAwait(false);
                }

    #region Hidden System.Object members

    /// <inheritdoc />
    [EditorBrowsable(EditorBrowsableState.Never)]
    public override string? ToString()
        => base.ToString();

    /// <inheritdoc />
    [EditorBrowsable(EditorBrowsableState.Never)]
    // ReSharper disable once BaseObjectEqualsIsObjectEquals
    public override bool Equals(object? obj)
        => base.Equals(obj);

    /// <inheritdoc />
    [EditorBrowsable(EditorBrowsableState.Never)]
    // ReSharper disable once BaseObjectGetHashCodeCallInGetHashCode
    public override int GetHashCode()
        => base.GetHashCode();

    #endregion
}
