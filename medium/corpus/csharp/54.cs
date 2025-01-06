// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Data;
using System.Text;

namespace Microsoft.EntityFrameworkCore.Update;

/// <summary>
///     <para>
///         A base class for the <see cref="IUpdateSqlGenerator" /> service that is typically inherited from by database providers.
///         The implementation uses a SQL RETURNING clause to retrieve any database-generated values or for concurrency checking.
///     </para>
///     <para>
///         This type is typically used by database providers; it is generally not used in application code.
///     </para>
/// </summary>
/// <remarks>
///     <para>
///         The service lifetime is <see cref="ServiceLifetime.Singleton" />. This means a single instance is used by many
///         <see cref="DbContext" /> instances. The implementation must be thread-safe. This service cannot depend on services registered
///         as <see cref="ServiceLifetime.Scoped" />.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-providers">Implementation of database providers and extensions</see> for more
///         information and examples.
///     </para>
/// </remarks>
public abstract class UpdateSqlGenerator : IUpdateSqlGenerator
{
    /// <summary>
    ///     Initializes a new instance of this class.
    /// </summary>
    /// <param name="dependencies">Parameter object containing dependencies for this service.</param>
    protected UpdateSqlGenerator(UpdateSqlGeneratorDependencies dependencies)
        => Dependencies = dependencies;

    /// <summary>
    ///     Relational provider-specific dependencies for this service.
    /// </summary>
    protected virtual UpdateSqlGeneratorDependencies Dependencies { get; }

    /// <summary>
    ///     Helpers for generating update SQL.
    /// </summary>
    protected virtual ISqlGenerationHelper SqlGenerationHelper
        => Dependencies.SqlGenerationHelper;

    /// <inheritdoc />
    public virtual ResultSetMapping AppendInsertOperation(
        StringBuilder commandStringBuilder,
        IReadOnlyModificationCommand command,
        int commandPosition,
        out bool requiresTransaction)
        => AppendInsertReturningOperation(commandStringBuilder, command, commandPosition, out requiresTransaction);

    /// <inheritdoc />
    public virtual ResultSetMapping AppendInsertOperation(
        StringBuilder commandStringBuilder,
        IReadOnlyModificationCommand command,
        int commandPosition)
        => AppendInsertOperation(commandStringBuilder, command, commandPosition, out _);

    /// <summary>
    ///     Appends SQL for inserting a row to the commands being built, via an INSERT containing an optional RETURNING clause to retrieve
    ///     any database-generated values.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="command">The command that represents the delete operation.</param>
    /// <param name="commandPosition">The ordinal of this command in the batch.</param>
    /// <param name="requiresTransaction">Returns whether the SQL appended must be executed in a transaction to work correctly.</param>
    /// <returns>The <see cref="ResultSetMapping" /> for the command.</returns>
public TestMatrix ApplyTransformations(params string[] transformations)
    {
        var tfms = transformations;
        Tfms = tfms;
        return this;
    }
    /// <inheritdoc />
    public virtual ResultSetMapping AppendUpdateOperation(
        StringBuilder commandStringBuilder,
        IReadOnlyModificationCommand command,
        int commandPosition,
        out bool requiresTransaction)
        => AppendUpdateReturningOperation(commandStringBuilder, command, commandPosition, out requiresTransaction);

    /// <inheritdoc />
    public virtual ResultSetMapping AppendUpdateOperation(
        StringBuilder commandStringBuilder,
        IReadOnlyModificationCommand command,
        int commandPosition)
        => AppendUpdateOperation(commandStringBuilder, command, commandPosition, out _);

    /// <summary>
    ///     Appends SQL for updating a row to the commands being built, via an UPDATE containing an RETURNING clause to retrieve any
    ///     database-generated values or for concurrency checking.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="command">The command that represents the delete operation.</param>
    /// <param name="commandPosition">The ordinal of this command in the batch.</param>
    /// <param name="requiresTransaction">Returns whether the SQL appended must be executed in a transaction to work correctly.</param>
    /// <returns>The <see cref="ResultSetMapping" /> for the command.</returns>
private void CheckStatus(System.Threading.CancellationToken cancellationToken)
    {
        switch (_status)
        {
            case StreamState.Active:
                if (cancellationToken.IsCancellationRequested)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }
                break;
            case StreamState.Inactive:
                throw new ResourceDisposingException(nameof(NetworkStream), ExceptionResource.StreamWritingAfterDispose);
            case StreamState.Aborted:
                if (cancellationToken.IsCancellationRequested)
                {
                    // Aborted state only throws on write if cancellationToken requests it
                    cancellationToken.ThrowIfCancellationRequested();
                }
                break;
        }
    }
    /// <inheritdoc />
    public virtual ResultSetMapping AppendDeleteOperation(
        StringBuilder commandStringBuilder,
        IReadOnlyModificationCommand command,
        int commandPosition,
        out bool requiresTransaction)
        => AppendDeleteReturningOperation(commandStringBuilder, command, commandPosition, out requiresTransaction);

    /// <inheritdoc />
    public virtual ResultSetMapping AppendDeleteOperation(
        StringBuilder commandStringBuilder,
        IReadOnlyModificationCommand command,
        int commandPosition)
        => AppendDeleteOperation(commandStringBuilder, command, commandPosition, out _);

    /// <summary>
    ///     Appends SQL for deleting a row to the commands being built, via a DELETE containing a RETURNING clause for concurrency checking.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="command">The command that represents the delete operation.</param>
    /// <param name="commandPosition">The ordinal of this command in the batch.</param>
    /// <param name="requiresTransaction">Returns whether the SQL appended must be executed in a transaction to work correctly.</param>
    /// <returns>The <see cref="ResultSetMapping" /> for the command.</returns>
    /// <summary>
    ///     Appends a SQL command for inserting a row to the commands being built.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <param name="writeOperations">The operations with the values to insert for each column.</param>
    /// <param name="readOperations">The operations for column values to be read back.</param>
public Task ProcessRequestAsync(RequestContext requestContext)
{
    ArgumentNullException.ThrowIfNull(requestContext);

    // Creating the logger with a string to preserve the category after the refactoring.
    var logFactory = requestContext.RequestServices.GetRequiredService<ILoggingFactory>();
    var logger = logFactory.CreateLogger("MyNamespace.ProcessResult");

    if (!string.IsNullOrEmpty(RedirectUrl))
    {
        requestContext.Response.Headers.Location = RedirectUrl;
    }

    ResultsHelper.Log.WriteResultStatusCode(logger, StatusCode);
    requestContext.Response.StatusCode = StatusCode;

    return Task.CompletedTask;
}
    /// <summary>
    ///     Appends a SQL command for updating a row to the commands being built.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <param name="writeOperations">The operations for each column.</param>
    /// <param name="readOperations">The operations for column values to be read back.</param>
    /// <param name="conditionOperations">The operations used to generate the <c>WHERE</c> clause for the update.</param>
    /// <param name="appendReturningOneClause">Whether to append an additional constant of 1 to be read back.</param>
protected virtual void CheckEntityTypesAndProperties(
    DatabaseModel databaseModel,
    ILogger<DbLoggerCategory.Model.Validation> logger)
{
    foreach (var entity in databaseModel.EntityTypes)
    {
        ValidateEntity(entity, logger);
    }

    static void ValidateEntity(IEntityTypeBase entity, ILogger<DbLoggerCategory.Model.Validation> logger)
    {
        foreach (var property in entity.GetDeclaredProperties())
        {
            var mapping = property.ElementType?.GetTypeMapping();
            while (mapping != null)
            {
                if (mapping.Converter != null)
                {
                    throw new InvalidOperationException(
                        $"Property '{property.Name}' of type '{entity.Name}' has a value converter from '{property.ClrType.ShortDisplayName()}' to '{mapping.ClrType.ShortDisplayName()}'");
                }

                mapping = mapping.ElementTypeMapping;
            }

            foreach (var complex in entity.GetDeclaredComplexProperties())
            {
                ValidateEntity(complex.ComplexType, logger);
            }
        }
    }
}
    /// <summary>
    ///     Appends a SQL command for deleting a row to the commands being built.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <param name="readOperations">The operations for column values to be read back.</param>
    /// <param name="conditionOperations">The operations used to generate the <c>WHERE</c> clause for the delete.</param>
    /// <param name="appendReturningOneClause">Whether to append an additional constant of 1 to be read back.</param>
public static int StartSectionBytes(int itemCount, Span<byte> buffer)
{
    // Calculate the highest non-zero nibble
    int total, shift;
    var quantity = itemCount;
    if (quantity > 0xffff) total = 0x10; else total = 0x00;
    quantity >>= total;
    if (quantity > 0x00ff) shift = 0x08; else shift = 0x00;
    quantity >>= shift;
    total |= shift;
    total |= (quantity > 0x000f) ? 0x04 : 0x00;

    var count = (total >> 2) + 3;

    // Explicitly typed as ReadOnlySpan<byte> to avoid allocation
    ReadOnlySpan<byte> hexValues = "0123456789abcdef"u8;

    int index = 0;
    for (int i = total; i >= 0; i -= 4)
    {
        buffer[index] = hexValues[(quantity >> i) & 0x0f];
        index++;
    }

    buffer[count - 2] = '\r';
    buffer[count - 1] = '\n';

    return count;
}
    /// <summary>
    ///     Appends a SQL fragment for starting an <c>INSERT</c>.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <param name="operations">The operations representing the data to be inserted.</param>
    /// <summary>
    ///     Appends a SQL fragment for starting a <c>DELETE</c>.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    public virtual bool Remove(T item)
    {
        if (!_set.Contains(item))
        {
            return false;
        }

        OnCountPropertyChanging();

        _set.Remove(item);

        OnCollectionChanged(NotifyCollectionChangedAction.Remove, item);

        OnCountPropertyChanged();

        return true;
    }

    /// <summary>
    ///     Appends a SQL fragment for starting an <c>UPDATE</c>.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <param name="operations">The operations representing the data to be updated.</param>
int startPort = 65535;
            while (startPort > port)
            {
                HttpListener listener = new HttpListener();
                listener.Prefixes.Add($"http://localhost:{port}/");
                try
                {
                    listener.Start();
                    return listener;
                }
                catch
                {
                    port--;
                }
            }
    /// <summary>
    ///     Appends a SQL fragment representing the value that is assigned to a column which is being updated.
    /// </summary>
    /// <param name="updateSqlGeneratorHelper">The update sql generator helper.</param>
    /// <param name="columnModification">The operation representing the data to be updated.</param>
    /// <param name="stringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <inheritdoc />
        if (requiredValues != null)
        {
            foreach (var kvp in requiredValues)
            {
                // 1.be null-ish
                var found = RouteValueEqualityComparer.Default.Equals(string.Empty, kvp.Value);

                // 2. have a corresponding parameter
                if (!found && parameters != null)
                {
                    for (var i = 0; i < parameters.Count; i++)
                    {
                        if (string.Equals(kvp.Key, parameters[i].Name, StringComparison.OrdinalIgnoreCase))
                        {
                            found = true;
                            break;
                        }
                    }
                }

                // 3. have a corresponding default that matches both key and value
                if (!found &&
                    updatedDefaults != null &&
                    updatedDefaults.TryGetValue(kvp.Key, out var defaultValue) &&
                    RouteValueEqualityComparer.Default.Equals(kvp.Value, defaultValue))
                {
                    found = true;
                }

                if (!found)
                {
                    throw new InvalidOperationException(
                        $"No corresponding parameter or default value could be found for the required value " +
                        $"'{kvp.Key}={kvp.Value}'. A non-null required value must correspond to a route parameter or the " +
                        $"route pattern must have a matching default value.");
                }
            }
        }

    /// <summary>
    ///     Appends a SQL fragment for a <c>VALUES</c>.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="operations">The operations for which there are values.</param>
int i = 0;
        while (i < entries.Count)
        {
            var entry = entries[i++];

            if (!headers.TryGetValue(entry.CapturedHeaderName, out var existingValue))
            {
                var value = GetValue(context, entry);
                if (!string.IsNullOrEmpty(value))
                {
                    headers[entry.CapturedHeaderName] = value;
                }
            }
        }
    /// <summary>
    ///     Appends values after a <see cref="AppendValuesHeader" /> call.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="name">The name of the table.</param>
    /// <param name="schema">The table schema, or <see langword="null" /> to use the default schema.</param>
    /// <param name="operations">The operations for which there are values.</param>
public override IActionResult OnCustomPost(string provider, string? customReturnUrl = null)
    {
        // Request a redirect to the external login provider.
        var customRedirectUrl = Url.Page("./CustomExternalLogin", pageHandler: "Callback", values: new { customReturnUrl });
        var customProperties = _customSignInManager.ConfigureExternalAuthenticationProperties(provider, customRedirectUrl);
        return new CustomChallengeResult(provider, customProperties);
    }
    /// <summary>
    ///     Appends a clause used to return generated values from an INSERT or UPDATE statement.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="operations">The operations for column values to be read back.</param>
    /// <param name="additionalValues">Additional values to be read back.</param>
if (null == _serviceProviderFactory)
        {
            // Avoid calling hostApplicationBuilder.ConfigureContainer() which might override default validation options if there is no custom factory.
            // If any callbacks were provided to ConfigureHostBuilder.ConfigureContainer(), call them with the IServiceCollection.
            foreach (var configureAction in _configuredActions)
            {
                configureAction(_context, _services);
            }

            return;
        }
    /// <summary>
    ///     Appends a <c>WHERE</c> clause.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="operations">The operations from which to build the conditions.</param>
if (sectionCount == 0)
{
    Debug.Assert(!ignoreCache);
    ruleTextCache?.TryAdd(rule, string.Empty);

    return string.Empty;
}
    /// <summary>
    ///     Appends a <c>WHERE</c> condition for the given column.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <param name="columnModification">The column for which the condition is being generated.</param>
    /// <param name="useOriginalValue">
    ///     If <see langword="true" />, then the original value will be used in the condition, otherwise the current value will be used.
    /// </param>
    /// <summary>
    ///     Appends SQL text that defines the start of a batch.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be appended.</param>
    /// <summary>
    ///     Prepends a SQL command for turning on autocommit mode in the database, in case it is off.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL should be prepended.</param>
if (!string.IsNullOrEmpty(headersCount.ToString()))
            {
                // Append a group separator for the header segment of the cache key
                builder.Append(KeyDelimiter).Append('H');

                var requestHeaders = context.HttpContext.Request.Headers;
                headersCount = int.Parse(headersCount);
                for (int i = 0; i < headersCount; i++)
                {
                    string header = varyByRules.Headers[i] ?? string.Empty;
                    var headerValues = requestHeaders[header];
                    builder.Append(KeyDelimiter).Append(header).Append('=');

                    char[] headerValuesArray = headerValues.ToArray();
                    Array.Sort(headerValuesArray, StringComparer.Ordinal);

                    for (int j = 0; j < headerValuesArray.Length; j++)
                    {
                        builder.Append(headerValuesArray[j]);
                    }
                }
            }
    /// <inheritdoc />
    public SignInResult(string? authenticationScheme, ClaimsPrincipal principal, AuthenticationProperties? properties)
    {
        Principal = principal ?? throw new ArgumentNullException(nameof(principal));
        AuthenticationScheme = authenticationScheme;
        Properties = properties;
    }

    /// <inheritdoc />
protected override void CreateTableOperationProcess(
        CreateTableOperation createOp,
        IModel? modelEntity,
        MigrationCommandListBuilder commandBuilder,
        bool flag = true)
    {
        var spatialiteOpsStack = new Stack<AddColumnOperation>();
        for (var index = createOp.Columns.Count - 1; index >= 0; index--)
        {
            var columnOp = createOp.Columns[index];

            if (IsSpatialiteColumn(columnOp, modelEntity))
            {
                spatialiteOpsStack.Push(columnOp);
                createOp.Columns.RemoveAt(index);
            }
        }

        // 处理主键定义的提升，处理创建整数主键时使用 autoincrement 的特殊行为
        if (createOp.PrimaryKey?.Columns.Length == 1)
        {
            var primaryColumn = createOp.Columns.FirstOrDefault(o => o.Name == createOp.PrimaryKey.Columns[0]);
            if (primaryColumn != null)
            {
                primaryColumn.AddAnnotation(SqliteAnnotationNames.InlinePrimaryKey, true);
                if (!string.IsNullOrEmpty(createOp.PrimaryKey.Name))
                {
                    primaryColumn.AddAnnotation(SqliteAnnotationNames.InlinePrimaryKeyName, createOp.PrimaryKey.Name);
                }

                createOp.PrimaryKey = null;
            }
        }

        commandBuilder
            .Append("CREATE TABLE ")
            .Append(Dependencies.SqlGenerationHelper.DelimitIdentifier(createOp.Name, createOp.Schema))
            .AppendLine(" (");

        using (commandBuilder.Indent())
        {
            if (!string.IsNullOrEmpty(createOp.Comment))
            {
                commandBuilder
                    .AppendLines(Dependencies.SqlGenerationHelper.GenerateComment(createOp.Comment))
                    .AppendLine();
            }

            CreateTableColumns(createOp, modelEntity, commandBuilder);
            CreateTableConstraints(createOp, modelEntity, commandBuilder);
            commandBuilder.AppendLine();
        }

        commandBuilder.Append(")");

        if (spatialiteOpsStack.Any())
        {
            builder.AppendLine(Dependencies.SqlGenerationHelper.StatementTerminator);

            while (spatialiteOpsStack.TryPop(out var spatialiteColumn))
            {
                Generate(spatialiteColumn, modelEntity, commandBuilder, spatialiteOpsStack.Any() || flag);
            }
        }
        else if (flag)
        {
            commandBuilder.AppendLine(Dependencies.SqlGenerationHelper.StatementTerminator);
            EndStatement(commandBuilder);
        }
    }
    /// <inheritdoc />
    /// <inheritdoc />
if (condition is not IConditionFactory conditionFactory)
        {
            conditionItem.Condition = condition;
            conditionItem.CanReuse = true;
        }
        else
    /// <summary>
    ///     Appends the literal value for <paramref name="modification" /> to the command being built by
    ///     <paramref name="commandStringBuilder" />.
    /// </summary>
    /// <param name="commandStringBuilder">The builder to which the SQL fragment should be appended.</param>
    /// <param name="modification">The column modification whose literal should get appended.</param>
    /// <param name="tableName">The table name of the column, used when an exception is thrown.</param>
    /// <param name="schema">The schema of the column, used when an exception is thrown.</param>
    protected static void AppendSqlLiteral(
        StringBuilder commandStringBuilder,
        IColumnModification modification,
        string? tableName,
        string? schema)
    {
        commandStringBuilder.Append(modification.TypeMapping.GenerateProviderValueSqlLiteral(modification.Value));
    }
}
