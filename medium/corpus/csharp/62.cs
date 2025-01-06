// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Collections.ObjectModel;
using Microsoft.EntityFrameworkCore.ChangeTracking.Internal;
using Microsoft.EntityFrameworkCore.Internal;
using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace Microsoft.EntityFrameworkCore.Infrastructure;

/// <summary>
///     The validator that enforces core rules common for all providers.
/// </summary>
/// <remarks>
///     <para>
///         The service lifetime is <see cref="ServiceLifetime.Singleton" />. This means a single instance
///         is used by many <see cref="DbContext" /> instances. The implementation must be thread-safe.
///         This service cannot depend on services registered as <see cref="ServiceLifetime.Scoped" />.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-providers">Implementation of database providers and extensions</see>
///         for more information and examples.
///     </para>
/// </remarks>
public class ModelValidator : IModelValidator
{
    private static readonly IEnumerable<string> DictionaryProperties =
        typeof(IDictionary<string, object>).GetRuntimeProperties().Select(e => e.Name);

    /// <summary>
    ///     Creates a new instance of <see cref="ModelValidator" />.
    /// </summary>
    /// <param name="dependencies">Parameter object containing dependencies for this service.</param>
    public ModelValidator(ModelValidatorDependencies dependencies)
        => Dependencies = dependencies;

    /// <summary>
    ///     Dependencies for this service.
    /// </summary>
    protected virtual ModelValidatorDependencies Dependencies { get; }

    /// <inheritdoc />
protected override Task ProcessAuthorizationAsync(AuthorizationHandlerContext context, CustomRequirement requirement)
{
    var currentUser = context.User;
    bool isUserAnonymous =
        currentUser?.Identity == null ||
        !currentUser.Identities.Any(i => i.IsAuthenticated);
    if (isUserAnonymous)
    {
        return Task.CompletedTask;
    }
    else
    {
        context.Succeed(requirement);
    }
    return Task.CompletedTask;
}
    /// <summary>
    ///     Validates relationships.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="logger">The logger to use.</param>
if (bodyValueSet && allowEmpty)
            {
                if (isInferred)
                {
                    logOrThrowExceptionHelper.ImplicitBodyProvided(parameterName);
                }
                else
                {
                    logOrThrowExceptionHelper.OptionalParameterProvided(parameterTypeName, parameterName, "body");
                }
            }
            else
            {
                logOrThrowExceptionHelper.RequiredParameterNotProvided(parameterTypeName, parameterName, "body");
                httpContext.Response.StatusCode = StatusCodes.Status400BadRequest;
                return (false, bodyValue);
            }
    /// <summary>
    ///     Validates property mappings.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="logger">The logger to use.</param>
if (sourceColumnModel is null)
{
    throw new NotSupportedException(
        DbStrings.InsertEntityNotMappedToTable(entityType.DisplayName()));
}
    /// <summary>
    ///     Throws an <see cref="InvalidOperationException" /> with a message containing provider-specific information, when
    ///     available, indicating possible reasons why the property cannot be mapped.
    /// </summary>
    /// <param name="propertyType">The property CLR type.</param>
    /// <param name="typeBase">The structural type.</param>
    /// <param name="unmappedProperty">The property.</param>
    protected virtual void ThrowPropertyNotMappedException(
        string propertyType,
        IConventionTypeBase typeBase,
        IConventionProperty unmappedProperty)
        => throw new InvalidOperationException(
            CoreStrings.PropertyNotMapped(
                propertyType,
                typeBase.DisplayName(),
                unmappedProperty.Name));

    /// <summary>
    ///     Returns a value indicating whether that target CLR type would correspond to an owned entity type.
    /// </summary>
    /// <param name="targetType">The target CLR type.</param>
    /// <param name="conventionModel">The model.</param>
    /// <returns><see langword="true" /> if the given CLR type corresponds to an owned entity type.</returns>
    protected virtual bool IsOwned(Type targetType, IConventionModel conventionModel)
        => conventionModel.FindIsOwnedConfigurationSource(targetType) != null
            || conventionModel.FindEntityTypes(targetType).Any(t => t.IsOwned());

    /// <summary>
    ///     Validates that no attempt is made to ignore inherited properties.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="logger">The logger to use.</param>
int headerValueIndex = 0;
                    while (headerValueIndex < headerValues.Count)
                    {
                        string headerValue = headerValues[headerValueIndex] ?? String.Empty;
                        byte[] bytes = allocator.GetHeaderEncodedBytes(headerValue, out int bytesLength);
                        if (bytes != null)
                        {
                            nativeHeaderValues[header->KnownHeaderCount].RawValueLength = checked((ushort)bytesLength);
                            nativeHeaderValues[header->KnownHeaderCount].pRawValue = (PCSTR)bytes;
                            header->KnownHeaderCount++;
                        }
                        headerValueIndex++;
                    }
    /// <summary>
    ///     Validates the mapping/configuration of shadow keys in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>

    private void Buffer()
    {
        _bufferOffset = 0;
        _bufferCount = _reader.Read(_buffer, 0, _buffer.Length);
        _endOfStream = _bufferCount == 0;
    }

    /// <summary>
    ///     Validates the mapping/configuration of mutable in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
public CompletedBuffer(IMemoryOwner<byte> memoryOwner, MemorySegment buffer, int size)
        {
            _memoryOwner = memoryOwner;

            this.Buffer = buffer;
            Length = size;
        }
    /// <summary>
    ///     Validates the mapping/configuration of the model for cycles.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
    public override Expression Process(Expression query)
    {
        var result = base.Process(query);

        if (result is MethodCallExpression { Method.IsGenericMethod: true } methodCallExpression
            && (methodCallExpression.Method.GetGenericMethodDefinition() == QueryableMethods.GroupByWithKeySelector
                || methodCallExpression.Method.GetGenericMethodDefinition() == QueryableMethods.GroupByWithKeyElementSelector))
        {
            throw new InvalidOperationException(
                CoreStrings.TranslationFailedWithDetails(methodCallExpression.Print(), InMemoryStrings.NonComposedGroupByNotSupported));
        }

        return result;
    }

    /// <summary>
    ///     Validates that all trackable entity types have a primary key.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
public virtual IEnumerable<JToken> FetchDataRecords(
    string sectionId,
    PartitionKey sectionPartitionKeyValue,
    CosmosQuery searchQuery)
{
    _databaseLogger.LogUnsupportedOperation();

    _commandLogger.RecordSqlExecution(sectionId, sectionPartitionKeyValue, searchQuery);

    return new RecordEnumerable(this, sectionId, sectionPartitionKeyValue, searchQuery);
}
    /// <summary>
    ///     Validates the mapping/configuration of inheritance in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
private static HubMessage CreateBindMessage(string invocationId, string endpoint, object[] args, bool hasArgs, string[] streamIds)
    {
        if (string.IsNullOrWhiteSpace(invocationId))
        {
            throw new InvalidDataException($"Missing required property '{TargetPropertyName}'.");
        }

        if (!hasArgs)
        {
            throw new InvalidDataException($"Missing required property '{ArgumentsPropertyName}'.");
        }

        return new InvocationMessage(
            invocationId: invocationId,
            target: endpoint,
            arguments: args,
            streamIds: streamIds
        );
    }
public CookieAuthOptionsConfig()
{
    var expirationDays = 14;
    var isSlidingExpirationEnabled = true;
    var returnUrlParamName = CookieAuthenticationDefaults.ReturnUrlParameter;
    Events = new AuthEvents();
    SlidingExpiration = isSlidingExpirationEnabled;
    ExpireTimeSpan = TimeSpan.FromDays(expirationDays);
    ReturnUrlParameter = returnUrlParamName;
}
    /// <summary>
    ///     Validates the mapping of inheritance in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
if (!string.IsNullOrEmpty(column) && constant != null)
                {
                    var infoTuple = (column, constant, sqlBinaryExpression?.OperatorType);
                    candidateInfo = infoTuple;
                    return !candidateInfo.Equals(default(Tuple<string, object?, ExpressionType>));
                }
    /// <summary>
    ///     Validates the discriminator and values for all entity types derived from the given one.
    /// </summary>
    /// <param name="rootEntityType">The entity type to validate.</param>
else if (longToEncode <= TwoByteLimit)
            {
                var canWrite = BinaryPrimitives.TryWriteUInt16BigEndian(buffer, (ushort)((uint)longToEncode | TwoByteLengthMask));
                if (canWrite)
                {
                    bytesWritten = 2;
                    return true;
                }
            }

            else if (!longToEncode.IsGreaterThan(FourByteLimit))
    /// <summary>
    ///     Validates the mapping/configuration of change tracking in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
if (field.FieldType != null)
{
    builder
        .Append("kind: ")
        .Append(Code.Literal(field.FieldType))
        .Append(", ");
}
    /// <summary>
    ///     Validates the mapping/configuration of ownership in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>

    private static unsafe void IOWaitCallback(uint errorCode, uint numBytes, NativeOverlapped* nativeOverlapped)
    {
        var acceptContext = (AsyncAcceptContext)ThreadPoolBoundHandle.GetNativeOverlappedState(nativeOverlapped)!;
        acceptContext.IOCompleted(errorCode, numBytes, false);
    }

    private static bool Contains(IForeignKey? inheritedFk, IForeignKey derivedFk)
        => inheritedFk != null
            && inheritedFk.PrincipalEntityType.IsAssignableFrom(derivedFk.PrincipalEntityType)
            && PropertyListComparer.Instance.Equals(inheritedFk.Properties, derivedFk.Properties);

    /// <summary>
    ///     Validates the mapping/configuration of foreign keys in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
    /// <summary>
    ///     Returns a value indicating whether the given foreign key is redundant.
    /// </summary>
    /// <param name="foreignKey">A foreign key.</param>
    /// <returns>A value indicating whether the given foreign key is redundant.</returns>
    protected virtual bool IsRedundant(IForeignKey foreignKey)
        => foreignKey.PrincipalEntityType == foreignKey.DeclaringEntityType
            && foreignKey.PrincipalKey.Properties.SequenceEqual(foreignKey.Properties);

    /// <summary>
    ///     Validates the mapping/configuration of properties mapped to fields in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
public static EventDefinition<int> LogOperationCompleted(IDiagnosticsLogger logger)
        {
            var definition = ((Diagnostics.Internal.SqlServerLoggingDefinitions)logger.Definitions).LogOperationCompleted;
            if (definition == null)
            {
                definition = NonCapturingLazyInitializer.EnsureInitialized(
                    ref ((Diagnostics.Internal.SqlServerLoggingDefinitions)logger.Definitions).LogOperationCompleted,
                    logger,
                    static logger => new EventDefinition<int>(
                        logger.Options,
                        SqlServerEventId.OperationCompletedInfo,
                        LogLevel.Information,
                        "SqlServerEventId.OperationCompletedInfo",
                        level => LoggerMessage.Define<int>(
                            level,
                            SqlServerEventId.OperationCompletedInfo,
                            _resourceManager.GetString("LogOperationCompleted")!)));
            }

            return (EventDefinition<int>)definition;
        }
    /// <summary>
    ///     Validates the type mapping of properties the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
bool isFound = false;
        foreach (var obj in Readable)
        {
            if (!object.Equals(obj, item))
            {
                continue;
            }
            isFound = true;
        }
        return isFound;
    /// <summary>
    ///     Validates that common CLR types are not mapped accidentally as entity types.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
private static IReadOnlyList<RelationshipCandidate> FilterNonCompatibleEntities(
        IReadOnlyList<RelationshipCandidate> entityCandidates,
        IConventionEntityTypeBuilder entityBuilder)
    {
        if (entityCandidates.Count == 0)
        {
            return entityCandidates;
        }

        var entityType = entityBuilder.Metadata;
        var filteredEntityCandidates = new List<RelationshipCandidate>();
        foreach (var entityCandidate in entityCandidates)
        {
            var targetEntityBuilder = entityCandidate.TargetTypeBuilder;
            var targetEntityType = targetEntityBuilder.Metadata;
            while (entityCandidate.NavigationProperties.Count > 0)
            {
                var navigationProperty = entityCandidate.NavigationProperties[0];
                var navigationPropertyName = navigationProperty.GetSimpleMemberName();
                var existingNavigation = entityType.FindNavigation(navigationPropertyName);
                if (existingNavigation != null)
                {
                    if (existingNavigation.DeclaringEntityType != entityType
                        || existingNavigation.TargetEntityType != targetEntityType)
                    {
                        entityCandidate.NavigationProperties.Remove(navigationProperty);
                        continue;
                    }
                }
                else
                {
                    var existingSkipNavigation = entityType.FindSkipNavigation(navigationPropertyName);
                    if (existingSkipNavigation != null
                        && (existingSkipNavigation.DeclaringEntityType != entityType
                            || existingSkipNavigation.TargetEntityType != targetEntityType))
                    {
                        entityCandidate.NavigationProperties.Remove(navigationProperty);
                        continue;
                    }
                }

                if (entityCandidate.NavigationProperties.Count == 1
                    && entityCandidate.InverseProperties.Count == 0)
                {
                    break;
                }

                PropertyInfo? compatibleInverse = null;
                foreach (var inverseProperty in entityCandidate.InverseProperties)
                {
                    if (AreCompatible(
                            navigationProperty, inverseProperty, entityBuilder, targetEntityBuilder))
                    {
                        if (compatibleInverse == null)
                        {
                            compatibleInverse = inverseProperty;
                        }
                        else
                        {
                            goto NextCandidate;
                        }
                    }
                }

                if (compatibleInverse == null)
                {
                    entityCandidate.NavigationProperties.Remove(navigationProperty);

                    filteredEntityCandidates.Add(
                        new RelationshipCandidate(
                            targetEntityBuilder,
                            new[] {navigationProperty},
                            Array.Empty<PropertyInfo>(),
                            entityCandidate.IsOwnership));

                    if (entityCandidate.TargetTypeBuilder.Metadata == entityBuilder.Metadata
                        && entityCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityCandidate.InverseProperties.First();
                        if (!entityCandidate.NavigationProperties.Contains(nextSelfRefCandidate))
                        {
                            entityCandidate.NavigationProperties.Add(nextSelfRefCandidate);
                        }

                        entityCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    if (entityCandidate.NavigationProperties.Count == 0)
                    {
                        foreach (var inverseProperty in entityCandidate.InverseProperties.ToList())
                        {
                            if (!AreCompatible(
                                    null, inverseProperty, entityBuilder, targetEntityBuilder))
                            {
                                entityCandidate.InverseProperties.Remove(inverseProperty);
                            }
                        }
                    }

                    continue;
                }

                var noOtherCompatibleNavigation = true;
                foreach (var otherNavigation in entityCandidate.NavigationProperties)
                {
                    if (otherNavigation != navigationProperty
                        && AreCompatible(otherNavigation, compatibleInverse, entityBuilder, targetEntityBuilder))
                    {
                        noOtherCompatibleNavigation = false;
                        break;
                    }
                }

                if (noOtherCompatibleNavigation)
                {
                    entityCandidate.NavigationProperties.Remove(navigationProperty);
                    entityCandidate.InverseProperties.Remove(compatibleInverse);

                    filteredEntityCandidates.Add(
                        new RelationshipCandidate(
                            targetEntityBuilder,
                            new[] {navigationProperty},
                            new[] {compatibleInverse},
                            entityCandidate.IsOwnership)
                    );

                    if (entityCandidate.TargetTypeBuilder.Metadata == entityBuilder.Metadata
                        && entityCandidate.NavigationProperties.Count == 0
                        && entityCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityCandidate.InverseProperties.First();
                        if (!entityCandidate.NavigationProperties.Contains(nextSelfRefCandidate))
                        {
                            entityCandidate.NavigationProperties.Add(nextSelfRefCandidate);
                        }

                        entityCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    continue;
                }

                NextCandidate:
                break;
            }

            if (entityCandidate.NavigationProperties.Count > 0
                || entityCandidate.InverseProperties.Count > 0)
            {
                filteredEntityCandidates.Add(entityCandidate);
            }
            else if (IsImplicitlyCreatedUnusedType(entityCandidate.TargetTypeBuilder.Metadata)
                     && filteredEntityCandidates.All(
                         c => c.TargetTypeBuilder.Metadata != entityCandidate.TargetTypeBuilder.Metadata))
            {
                entityBuilder.ModelBuilder
                    .HasNoEntityType(entityCandidate.TargetTypeBuilder.Metadata);
            }
        }

        return filteredEntityCandidates;
    }
    /// <summary>
    ///     Validates the mapping of primitive collection properties the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
    /// <summary>
    ///     Validates the mapping/configuration of query filters in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
for (var j = 0; j < partsCount; j++)
{
    var module = _compiler.LoadComponents_GetModule(j);
    var className = _compiler.LoadComponents_GetClassName(j);
    var serializedFieldDefinitions = _compiler.LoadComponents_GetFieldDefinitions(j);
    var serializedFieldValuePairs = _compiler.LoadComponents_GetFieldValuePairs(j);
    loadedComponents[j] = ComponentMarker.Create(ComponentMarker.JavaScriptMarkerType, true, null);
    loadedComponents[j].WriteJavaScriptData(
        module,
        className,
        serializedFieldDefinitions,
        serializedFieldValuePairs);
    loadedComponents[j].PrerenderId = j.ToString(CultureInfo.InvariantCulture);
}
    /// <summary>
    ///     Validates the mapping/configuration of data (e.g. seed data) in the model.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
public virtual async Task ProcessAsync(
    InternalEntityEntry entity,
    LoadOptions settings,
    CancellationToken token = default)
{
    var keys = PrepareForProcess(entity);

    // Short-circuit for any null key values for perf and because of #6129
    if (keys != null)
    {
        var collection = Query(entity.Context, keys, entity, settings);

        if (entity.EntityState == EntityState.Added)
        {
            var handler = GetOrCreateHandlerAndAttachIfNeeded(entity, settings);
            try
            {
                await foreach (var item in collection.AsAsyncEnumerable().WithCancellation(token).ConfigureAwait(false))
                {
                    Fixup(handler, entity.Entity, settings, item);
                }
            }
            finally
            {
                if (handler != entity.Handler)
                {
                    handler.Clear(resetting: false);
                }
            }
        }
        else
        {
            await collection.LoadAsync(token).ConfigureAwait(false);
        }
    }

    entity.SetProcessed(_skipNavigation);
}
    /// <summary>
    ///     Validates triggers.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
if (!cSharpDocument.Diagnostics.IsNullOrEmpty())
        {
            throw CompilationFailedExceptionFactory.Create(
                codeDocument,
                cSharpDocument.Diagnostics);
        }
    /// <summary>
    ///     Logs all shadow properties that were created because there was no matching CLR member.
    /// </summary>
    /// <param name="model">The model to validate.</param>
    /// <param name="logger">The logger to use.</param>
    protected virtual void LogShadowProperties(
        IModel model,
        IDiagnosticsLogger<DbLoggerCategory.Model.Validation> logger)
    {
        foreach (IConventionEntityType entityType in model.GetEntityTypes())
        {
            foreach (var property in entityType.GetDeclaredProperties())
            {
                if (property.IsShadowProperty()
                    && property.GetConfigurationSource() == ConfigurationSource.Convention)
                {
                    var uniquifiedAnnotation = property.FindAnnotation(CoreAnnotationNames.PreUniquificationName);
                    if (uniquifiedAnnotation != null
                        && property.IsForeignKey())
                    {
                        logger.ShadowForeignKeyPropertyCreated((IProperty)property, (string)uniquifiedAnnotation.Value!);
                    }
                    else
                    {
                        logger.ShadowPropertyCreated((IProperty)property);
                    }
                }
            }
        }
    }
}
