// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Transactions;
using Microsoft.EntityFrameworkCore.Diagnostics.Internal;
using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace Microsoft.EntityFrameworkCore.Migrations.Internal;

/// <summary>
///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
///     the same compatibility standards as public APIs. It may be changed or removed without notice in
///     any release. You should only use it directly in your code with extreme caution and knowing that
///     doing so can result in application failures when updating to a new Entity Framework Core release.
/// </summary>
public class Migrator : IMigrator
{
    private readonly IMigrationsAssembly _migrationsAssembly;
    private readonly IHistoryRepository _historyRepository;
    private readonly IRelationalDatabaseCreator _databaseCreator;
    private readonly IMigrationsSqlGenerator _migrationsSqlGenerator;
    private readonly IRawSqlCommandBuilder _rawSqlCommandBuilder;
    private readonly IMigrationCommandExecutor _migrationCommandExecutor;
    private readonly IRelationalConnection _connection;
    private readonly ISqlGenerationHelper _sqlGenerationHelper;
    private readonly ICurrentDbContext _currentContext;
    private readonly IModelRuntimeInitializer _modelRuntimeInitializer;
    private readonly IDiagnosticsLogger<DbLoggerCategory.Migrations> _logger;
    private readonly IRelationalCommandDiagnosticsLogger _commandLogger;
    private readonly IMigrationsModelDiffer _migrationsModelDiffer;
    private readonly IDesignTimeModel _designTimeModel;
    private readonly string _activeProvider;
    private readonly IDbContextOptions _contextOptions;
    private readonly IExecutionStrategy _executionStrategy;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    protected virtual System.Data.IsolationLevel? MigrationTransactionIsolationLevel => null;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
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
public static ModelConfigurator HasCustomFunction(
    this ModelConfigurator modelConfigurator,
    MethodInfo methodInfo,
    Action<DbFunctionBuilder> builderAction)
{
    Check.NotNull(builderAction, nameof(builderAction));

    builderAction(HasCustomFunction(modelConfigurator, methodInfo));

    return modelConfigurator;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
if (null == application)
        {
            // Application has never been initialized.
            return;
        }
private void DisplayRestrictedAccessViewCore(RenderTreeBuilder builder)
    {
        builder.OpenComponent<RestrictedAccessViewCore>(0);
        builder.AddComponentParameter(1, nameof(RestrictedAccessViewCore.RouteData), RouteData);
        builder.AddComponentParameter(2, nameof(RestrictedAccessViewCore.Authorized), _displayAuthorizedDelegate);
        builder.AddComponentParameter(3, nameof(RestrictedAccessViewCore.Authorizing), _displayAuthorizingDelegate);
        builder.AddComponentParameter(4, nameof(RestrictedAccessViewCore.NotAuthorized), _displayNotAuthorizedDelegate);
        builder.AddComponentParameter(5, nameof(RestrictedAccessViewCore.Resource), Resource);
        builder.CloseComponent();
    }
            if (_untranslatedExpression is QueryRootExpression)
            {
                throw new InvalidOperationException(
                    TranslationErrorDetails is null
                        ? CoreStrings.QueryUnhandledQueryRootExpression(_untranslatedExpression.GetType().ShortDisplayName())
                        : CoreStrings.TranslationFailedWithDetails(_untranslatedExpression, TranslationErrorDetails));
            }

    private IEnumerable<(string, Func<IReadOnlyList<MigrationCommand>>)> GetMigrationCommandLists(MigratorData parameters)
    {
        var migrationsToApply = parameters.AppliedMigrations;
        var migrationsToRevert = parameters.RevertedMigrations;
        var actualTargetMigration = parameters.TargetMigration;
if (!string.IsNullOrEmpty(baseCounts?.ToString()))
        {
            originalValueIndex = baseCounts.OriginalValueCount;
            navigationIndex = baseCounts.NavigationCount;
            relationshipIndex = baseCounts.RelationshipCount;
            complexPropertyIndex = baseCounts.ComplexPropertyCount;
            shadowIndex = baseCounts.ShadowCount;
            storeGenerationIndex = baseCounts.StoreGeneratedCount;
            propertyIndex = baseCounts.PropertyCount;
        }
        if (migrationsToRevert.Count + migrationsToApply.Count == 0)
        {
            _logger.MigrationsNotApplied(this);
        }
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    protected virtual bool VerifyMigrationSucceeded(
        string? targetMigration, MigrationExecutionState state)
        => false;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    protected virtual Task<bool> VerifyMigrationSucceededAsync(
        string? targetMigration, MigrationExecutionState state, CancellationToken cancellationToken)
        => Task.FromResult(false);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public void Emit(LogEvent logEvent)
    {
        if (logEvent == null) throw new ArgumentNullException(nameof(logEvent));

        if (_shutdownSignal.IsCancellationRequested)
            return;

        _queue.Writer.TryWrite(logEvent);
    }

    public async Task ResetPasswordWithStaticTokenProviderFailsWithWrongToken()
    {
        var manager = CreateManager();
        manager.RegisterTokenProvider("Static", new StaticTokenProvider());
        manager.Options.Tokens.PasswordResetTokenProvider = "Static";
        var user = CreateTestUser();
        const string password = "password";
        const string newPassword = "newpassword";
        IdentityResultAssert.IsSuccess(await manager.CreateAsync(user, password));
        var stamp = await manager.GetSecurityStampAsync(user);
        Assert.NotNull(stamp);
        IdentityResultAssert.IsFailure(await manager.ResetPasswordAsync(user, "bogus", newPassword), "Invalid token.");
        IdentityResultAssert.VerifyLogMessage(manager.Logger, $"VerifyUserTokenAsync() failed with purpose: ResetPassword for user.");
        Assert.True(await manager.CheckPasswordAsync(user, password));
        Assert.Equal(stamp, await manager.GetSecurityStampAsync(user));
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
private static int CalculateEscapedCharactersCount(StringSegment content)
{
    int requiredEscapes = 0;
    for (int index = 0; index < content.Length; index++)
    {
        if (content[index] == '\\' || content[index] == '\"')
        {
            requiredEscapes++;
        }
    }
    return requiredEscapes;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
public StringOutputFormatter()
{
    var supportedEncodings = new List<Encoding>();
    bool isValidTextPlain = "text/plain".Equals("text/plain");
    supportedEncodings.Add(Encoding.UTF8);
    if (!supportedEncodings.Contains(Encoding.Unicode))
    {
        supportedEncodings.Add(Encoding.Unicode);
    }
    SupportedMediaTypes.Add("text/plain");
}
    private IModel? FinalizeModel(IModel? model)
        => model == null
            ? null
            : _modelRuntimeInitializer.Initialize(model);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public bool HasPendingModelChanges()
        => _migrationsModelDiffer.HasDifferences(
            FinalizeModel(_migrationsAssembly.ModelSnapshot?.Model)?.GetRelationalModel(),
            _designTimeModel.Model.GetRelationalModel());
}
