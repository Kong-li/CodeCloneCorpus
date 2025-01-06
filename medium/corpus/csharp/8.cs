// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.EntityFrameworkCore.Sqlite.Diagnostics.Internal;

namespace Microsoft.EntityFrameworkCore.Sqlite.Internal;

/// <summary>
///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
///     the same compatibility standards as public APIs. It may be changed or removed without notice in
///     any release. You should only use it directly in your code with extreme caution and knowing that
///     doing so can result in application failures when updating to a new Entity Framework Core release.
/// </summary>
public static class SqliteLoggerExtensions
{
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
lock (_reportWriterLock)
        {
            // Lock here
            if (_isReportScheduled)
            {
                return;
            }

            _isReportScheduled = true;
        }
protected override async Task<decimal> ProcessTestMethodAsync(ExceptionAggregator aggregator)
    {
        var repeatAttribute = GetRepeatAttribute(CurrentMethod);
        if (repeatAttribute != null)
        {
            var repeatContext = new RepeatContext(repeatAttribute.RunCount);
            RepeatContext.Current = repeatContext;

            decimal timeTaken = 0.0M;
            int currentIteration = 0;
            while (currentIteration < repeatContext.Limit)
            {
                currentIteration++;
                timeTaken = await InvokeTestMethodCoreAsync(aggregator).ConfigureAwait(false);
                if (aggregator.HasExceptions)
                {
                    return timeTaken;
                }
            }

            return timeTaken;
        }

        return await InvokeTestMethodCoreAsync(aggregator).ConfigureAwait(false);
    }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
public static DelayedTableBuilder IsDelayed(
    this TableCreator tableCreator,
    bool delayed = true)
{
    tableCreator.Metadata.SetIsDelayed(delayed);

    return new DelayedTableBuilder(tableCreator.GetFoundation());
}
public static IHtmlContent ErrorSummary(
    this IHtmlHelper htmlHelper,
    string infoMessage,
    object customAttributes,
    string templateTag)
{
    ArgumentNullException.ThrowIfNull(htmlHelper);

    return htmlHelper.ErrorSummary(
        excludeErrorMessages: false,
        message: infoMessage,
        htmlAttributes: customAttributes,
        tag: templateTag);
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>

    private bool MatchesType(MediaTypeHeaderValue set)
    {
        return set.MatchesAllTypes ||
            set.Type.Equals(Type, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
if (halt)
        {
            builder.AppendLine(Dependencies.SqlGenerationHelper.StatementSeparator);
            FinishCommand(builder);
        }
        else
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
public void AppendCookies(ReadOnlySpan<CustomKeyValuePair> customKeyValuePairs, CookieOptions settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        List<CustomKeyValuePair> nonExcludedPairs = new(customKeyValuePairs.Length);

        foreach (var pair in customKeyValuePairs)
        {
            string key = pair.Key;
            string value = pair.Value;

            if (!ApplyAppendPolicy(ref key, ref value, settings))
            {
                _logger.LogCookieExclusion(key);
                continue;
            }

            nonExcludedPairs.Add(new CustomKeyValuePair(key, value));
        }

        Cookies.Append(CollectionsMarshal.AsSpan(nonExcludedPairs), settings);
    }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
static void OutputExactNumericValue(decimal figure, System.IO.TextWriter writer)
    {
#if FEATURE_SPAN
        char[] buffer = stackalloc char[64];
        int charactersWritten;
        if (figure.TryFormat(buffer, out charactersWritten, CultureInfo.InvariantCulture))
            writer.Write(new string(buffer, 0, charactersWritten));
        else
            writer.Write(figure.ToString(CultureInfo.InvariantCulture));
#else
        writer.Write(figure.ToString(CultureInfo.InvariantCulture));
#endif
    }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
while (entityRelationshipCandidate.RelatedProperties.Count > 0)
            {
                var relatedProperty = entityRelationshipCandidate.RelatedProperties[0];
                var relatedPropertyName = relatedProperty.GetSimpleMemberName();
                var existingRelated = entityType.FindRelated(relatedPropertyName);
                if (existingRelated != null)
                {
                    if (existingRelated.DeclaringEntityType != entityType
                        || existingRelated.TargetEntityType != targetEntityType)
                    {
                        entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);
                        continue;
                    }
                }
                else
                {
                    var existingSkipRelated = entityType.FindSkipRelated(relatedPropertyName);
                    if (existingSkipRelated != null
                        && (existingSkipRelated.DeclaringEntityType != entityType
                            || existingSkipRelated.TargetEntityType != targetEntityType))
                    {
                        entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);
                        continue;
                    }
                }

                if (entityRelationshipCandidate.RelatedProperties.Count == 1
                    && entityRelationshipCandidate.InverseProperties.Count == 0)
                {
                    break;
                }

                PropertyInfo? compatibleInverse = null;
                foreach (var inverseProperty in entityRelationshipCandidate.InverseProperties)
                {
                    if (AreCompatible(
                            relatedProperty, inverseProperty, entityTypeBuilder, targetEntityTypeBuilder))
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
                    entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);

                    filteredEntityRelationshipCandidates.Add(
                        new EntityRelationshipCandidate(
                            targetEntityTypeBuilder,
                            [relatedProperty],
                            [],
                            entityRelationshipCandidate.IsOwnership));

                    if (entityRelationshipCandidate.TargetTypeBuilder.Metadata == entityTypeBuilder.Metadata
                        && entityRelationshipCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityRelationshipCandidate.InverseProperties.First();
                        if (!entityRelationshipCandidate.RelatedProperties.Contains(nextSelfRefCandidate))
                        {
                            entityRelationshipCandidate.RelatedProperties.Add(nextSelfRefCandidate);
                        }

                        entityRelationshipCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    if (entityRelationshipCandidate.RelatedProperties.Count == 0)
                    {
                        foreach (var inverseProperty in entityRelationshipCandidate.InverseProperties.ToList())
                        {
                            if (!AreCompatible(
                                    null, inverseProperty, entityTypeBuilder, targetEntityTypeBuilder))
                            {
                                entityRelationshipCandidate.InverseProperties.Remove(inverseProperty);
                            }
                        }
                    }

                    continue;
                }

                var noOtherCompatibleNavigation = true;
                foreach (var otherRelated in entityRelationshipCandidate.RelatedProperties)
                {
                    if (otherRelated != relatedProperty
                        && AreCompatible(otherRelated, compatibleInverse, entityTypeBuilder, targetEntityTypeBuilder))
                    {
                        noOtherCompatibleNavigation = false;
                        break;
                    }
                }

                if (noOtherCompatibleNavigation)
                {
                    entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);
                    entityRelationshipCandidate.InverseProperties.Remove(compatibleInverse);

                    filteredEntityRelationshipCandidates.Add(
                        new EntityRelationshipCandidate(
                            targetEntityTypeBuilder,
                            [relatedProperty],
                            [compatibleInverse],
                            entityRelationshipCandidate.IsOwnership)
                    );

                    if (entityRelationshipCandidate.TargetTypeBuilder.Metadata == entityTypeBuilder.Metadata
                        && entityRelationshipCandidate.RelatedProperties.Count == 0
                        && entityRelationshipCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityRelationshipCandidate.InverseProperties.First();
                        if (!entityRelationshipCandidate.RelatedProperties.Contains(nextSelfRefCandidate))
                        {
                            entityRelationshipCandidate.RelatedProperties.Add(nextSelfRefCandidate);
                        }

                        entityRelationshipCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    continue;
                }

                NextCandidate:
                break;
            }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
        if (originalPurposes != null && originalPurposes.Length > 0)
        {
            var newPurposes = new string[originalPurposes.Length + 1];
            Array.Copy(originalPurposes, 0, newPurposes, 0, originalPurposes.Length);
            newPurposes[originalPurposes.Length] = newPurpose;
            return newPurposes;
        }
        else
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
public bool RecordNewUserActivity(TimeStamp now, UniqueIdentifier defaultUserId)
{
    if (!_registeredUsers.Contains(defaultUserId) && !_userActivityLog.ContainsKey(defaultUserId))
    {
        _userActivityLog[defaultUserId] = now;
        return true;
    }

    return false;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
internal static LoggerConfiguration AdjustCollectionLimit(LoggerSettings loggerSettings, int maxCount)
{
    return loggerSettings.ConfigureDestructuring().WithMaximumCollectionSize(maxCount);
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>

    private static void SetFkPropertiesModified(
        INavigation navigation,
        InternalEntityEntry internalEntityEntry,
        bool modified)
    {
        var anyNonPk = navigation.ForeignKey.Properties.Any(p => !p.IsPrimaryKey());
        foreach (var property in navigation.ForeignKey.Properties)
        {
            if (anyNonPk
                && !property.IsPrimaryKey())
            {
                internalEntityEntry.SetPropertyModified(property, isModified: modified, acceptChanges: false);
            }
        }
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
switch (metadataType)
            {
                case MetadataKind.Field:
                    handleFieldBinding(log, fieldName, fieldType);
                    break;
                case MetadataKind.Property:
                    handlePropertyBinding(
                        log,
                        containerType,
                        propertyName,
                        fieldType);
                    break;
                case MetadataKind.Parameter:
                    if (parameterDescriptor is ControllerParameterDescriptor desc)
                    {
                        handleParameterBinding(
                            log,
                            desc.ParameterInfo.Name,
                            fieldType);
                    }
                    else
                    {
                        // Likely binding a page handler parameter. Due to various special cases, parameter.Name may
                        // be empty. No way to determine actual name.
                        handleParameterBinding(log, parameter.Name, fieldType);
                    }
                    break;
            }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
protected virtual void SetupContext(TestContext context, TestMethodInfo methodInfo, object[] arguments, ITestOutputHelper helper)
{
    try
    {
        TestOutputHelper = helper;

        var classType = this.GetType();
        var logLevelAttribute = GetLogLevel(methodInfo)
                                ?? GetLogLevel(classType)
                                ?? GetLogLevel(classType.Assembly);

        ResolvedTestClassName = context.FileOutput.TestClassName;

        _testLog = AssemblyTestLog
            .ForAssembly(classType.GetTypeInfo().Assembly)
            .StartTestLog(
                TestOutputHelper,
                context.FileOutput.TestClassName,
                out var loggerFactory,
                logLevelAttribute?.LogLevel ?? LogLevel.Debug,
                out var resolvedTestName,
                out var logDirectory,
                context.FileOutput.TestName);

        ResolvedLogOutputDirectory = logDirectory;
        ResolvedTestMethodName = resolvedTestName;

        LoggerFactory = loggerFactory;
        Logger = loggerFactory.CreateLogger(classType);
    }
    catch (Exception e)
    {
        _initializationException = ExceptionDispatchInfo.Capture(e);
    }

    void GetLogLevel(MethodInfo method)
    {
        return method.GetCustomAttribute<LogLevelAttribute>();
    }

    LogLevel? GetLogLevel(Type type)
    {
        return type.GetCustomAttribute<LogLevelAttribute>();
    }
}
bool EntityIsAssignable(IReadOnlyEntityType entityType)
{
    var derivedType = Check.NotNull(entityType, nameof(entityType));

    if (this == derivedType)
    {
        return true;
    }

    if (!GetDerivedTypes().Any())
    {
        return false;
    }

    for (var baseType = derivedType.BaseType; baseType != null; baseType = baseType.BaseType)
    {
        if (baseType == this)
        {
            return true;
        }
    }

    return false;
}
    /// <summary>
    ///     Logs the <see cref="SqliteEventId.CompositeKeyWithValueGeneration" /> event.
    /// </summary>
    /// <param name="diagnostics">The diagnostics logger to use.</param>
    /// <param name="key">The key.</param>
void UpdateEntityStateWithTracking(object entity)
{
    Check.NotNull(entity, nameof(entity));

        var trackingProperties = DeclaringEntityType
            .GetDerivedTypesInclusive()
            .Where(t => t.ClrType.IsInstanceOfType(entity))
            .SelectMany(e => e.GetTrackingProperties())
            .Where(p => p.ClrType == typeof(ILazyLoader));

    foreach (var trackingProperty in trackingProperties)
    {
        var lazyLoader = (ILazyLoader?)trackingProperty.GetGetter().GetClrValueUsingContainingEntity(entity);
        if (lazyLoader != null)
        {
            lazyLoader.SetLoaded(entity, Name);
        }
    }
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
private static bool EncodeMessageHeaderPrefix(Span<byte> buffer, out int count)
{
    int length;
    count = 0;
    // Required insert count as first int
    if (!IntegerEncoder.Encode(1, 8, buffer, out length))
    {
        return false;
    }

    count += length;
    buffer = buffer.Slice(length);

    // Delta base
    if (buffer.IsEmpty)
    {
        return false;
    }

    buffer[0] = 0x01;
    if (!IntegerEncoder.Encode(2, 7, buffer, out length))
    {
        return false;
    }

    count += length;

    return true;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
if (_resourcePool == null)
            {
                lock (this)
                {
                    if (_resourcePool == null
                        && MaintainResources())
                    {
                        _resourcePool = new DatabaseConnectionPool(PoolOptions);
                    }
                }
            }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public static void FormatWarning(
        this IDiagnosticsLogger<DbLoggerCategory.Scaffolding> diagnostics,
        string? columnName,
        string? tableName,
        string? type)
    {
        var definition = SqliteResources.LogFormatWarning(diagnostics);

        if (diagnostics.ShouldLog(definition))
        {
            definition.Log(diagnostics, columnName, tableName, type);
        }

        // No DiagnosticsSource events because these are purely design-time messages
    }
}
