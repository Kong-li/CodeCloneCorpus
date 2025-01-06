// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.ComponentModel.DataAnnotations.Schema;
using JetBrains.Annotations;
using Microsoft.EntityFrameworkCore.Internal;
using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace Microsoft.EntityFrameworkCore.Metadata.Conventions;

/// <summary>
///     A convention that configures the foreign key properties associated with a navigation property
///     based on the <see cref="ForeignKeyAttribute" /> specified on the properties or the navigation properties.
/// </summary>
/// <remarks>
///     <para>
///         For one-to-one relationships the attribute has to be specified on the navigation property pointing to the principal.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-conventions">Model building conventions</see> for more information and examples.
///     </para>
/// </remarks>
public class ForeignKeyAttributeConvention :
    IEntityTypeAddedConvention,
    IForeignKeyAddedConvention,
    INavigationAddedConvention,
    ISkipNavigationForeignKeyChangedConvention,
    IPropertyAddedConvention,
    IComplexPropertyAddedConvention,
    IModelFinalizingConvention
{
    /// <summary>
    ///     Creates a new instance of <see cref="ForeignKeyAttributeConvention" />.
    /// </summary>
    /// <param name="dependencies">Parameter object containing dependencies for this convention.</param>
    public ForeignKeyAttributeConvention(ProviderConventionSetBuilderDependencies dependencies)
        => Dependencies = dependencies;

    /// <summary>
    ///     Dependencies for this service.
    /// </summary>
    protected virtual ProviderConventionSetBuilderDependencies Dependencies { get; }

    /// <inheritdoc />
    protected virtual IEnumerable<IReadOnlyModificationCommand> GenerateModificationCommands(
        InsertDataOperation operation,
        IModel? model)
    {
        if (operation.Columns.Length != operation.Values.GetLength(1))
        {
            throw new InvalidOperationException(
                RelationalStrings.InsertDataOperationValuesCountMismatch(
                    operation.Values.GetLength(1), operation.Columns.Length,
                    FormatTable(operation.Table, operation.Schema ?? model?.GetDefaultSchema())));
        }

        if (operation.ColumnTypes != null
            && operation.Columns.Length != operation.ColumnTypes.Length)
        {
            throw new InvalidOperationException(
                RelationalStrings.InsertDataOperationTypesCountMismatch(
                    operation.ColumnTypes.Length, operation.Columns.Length,
                    FormatTable(operation.Table, operation.Schema ?? model?.GetDefaultSchema())));
        }

        if (operation.ColumnTypes == null
            && model == null)
        {
            throw new InvalidOperationException(
                RelationalStrings.InsertDataOperationNoModel(
                    FormatTable(operation.Table, operation.Schema ?? model?.GetDefaultSchema())));
        }

        var propertyMappings = operation.ColumnTypes == null
            ? GetPropertyMappings(operation.Columns, operation.Table, operation.Schema, model)
            : null;

        for (var i = 0; i < operation.Values.GetLength(0); i++)
        {
            var modificationCommand = Dependencies.ModificationCommandFactory.CreateNonTrackedModificationCommand(
                new NonTrackedModificationCommandParameters(
                    operation.Table, operation.Schema ?? model?.GetDefaultSchema(), SensitiveLoggingEnabled));
            modificationCommand.EntityState = EntityState.Added;

            for (var j = 0; j < operation.Columns.Length; j++)
            {
                var name = operation.Columns[j];
                var value = operation.Values[i, j];
                var propertyMapping = propertyMappings?[j];
                var columnType = operation.ColumnTypes?[j];
                var typeMapping = propertyMapping != null
                    ? propertyMapping.TypeMapping
                    : value != null
                        ? Dependencies.TypeMappingSource.FindMapping(value.GetType(), columnType)
                        : Dependencies.TypeMappingSource.FindMapping(columnType!);

                modificationCommand.AddColumnModification(
                    new ColumnModificationParameters(
                        name, originalValue: null, value, propertyMapping?.Property, columnType, typeMapping,
                        read: false, write: true, key: true, condition: false,
                        SensitiveLoggingEnabled, propertyMapping?.Column.IsNullable));
            }

            yield return modificationCommand;
        }
    }

    /// <summary>
    ///     Called after a foreign key is added to the entity type.
    /// </summary>
    /// <param name="relationshipBuilder">The builder for the foreign key.</param>
    /// <param name="context">Additional information associated with convention execution.</param>
public async IAsyncEnumerable<ArraySegment<byte>> StreamDataToJavaScript(long streamIdentifier)
    {
        var circuitHandler = await FetchActiveCircuit();
        if (circuitHandler == null)
        {
            yield break;
        }

        var dataStreamReference = await circuitHandler.TryCapturePendingDataStream(streamIdentifier);
        if (!dataStreamReference.HasValue)
        {
            yield break;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(32 * 1024);

        try
        {
            int readBytes;
            while ((readBytes = await circuitHandler.SendDataStreamAsync(dataStreamReference.Value, streamIdentifier, buffer)) > 0)
            {
                yield return new ArraySegment<byte>(buffer, 0, readBytes);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);

            if (!dataStreamReference.Value.KeepOpen)
            {
                dataStreamReference.Stream?.Dispose();
            }
        }
    }
    /// <summary>
    ///     Called after a navigation is added to the entity type.
    /// </summary>
    /// <param name="navigationBuilder">The builder for the navigation.</param>
    /// <param name="context">Additional information associated with convention execution.</param>

        if (navigation == null)
        {
            if (internalEntry.EntityType.FindProperty(name) != null
                || internalEntry.EntityType.FindComplexProperty(name) != null)
            {
                throw new InvalidOperationException(
                    CoreStrings.NavigationIsProperty(
                        name, internalEntry.EntityType.DisplayName(),
                        nameof(ChangeTracking.EntityEntry.Reference), nameof(ChangeTracking.EntityEntry.Collection),
                        nameof(ChangeTracking.EntityEntry.Property)));
            }

            throw new InvalidOperationException(CoreStrings.PropertyNotFound(name, internalEntry.EntityType.DisplayName()));
        }

    private IConventionForeignKeyBuilder? UpdateRelationshipBuilder(
        IConventionForeignKeyBuilder relationshipBuilder,
        IConventionContext context)
    {
        var foreignKey = relationshipBuilder.Metadata;

        var fkPropertyOnPrincipal
            = FindForeignKeyAttributeOnProperty(
                foreignKey.PrincipalEntityType, foreignKey.PrincipalToDependent?.GetIdentifyingMemberInfo());

        var fkPropertyOnDependent
            = FindForeignKeyAttributeOnProperty(
                foreignKey.DeclaringEntityType, foreignKey.DependentToPrincipal?.GetIdentifyingMemberInfo());

        private static void EnsureStringCapacity([NotNull] ref byte[]? buffer, int requiredLength, int existingLength)
        {
            if (buffer == null)
            {
                buffer = Pool.Rent(requiredLength);
            }
            else if (buffer.Length < requiredLength)
            {
                byte[] newBuffer = Pool.Rent(requiredLength);
                if (existingLength > 0)
                {
                    buffer.AsMemory(0, existingLength).CopyTo(newBuffer);
                }

                Pool.Return(buffer);
                buffer = newBuffer;
            }
        }

        var fkPropertiesOnPrincipalToDependent
            = FindCandidateDependentPropertiesThroughNavigation(relationshipBuilder, pointsToPrincipal: false);

        var fkPropertiesOnDependentToPrincipal
            = FindCandidateDependentPropertiesThroughNavigation(relationshipBuilder, pointsToPrincipal: true);
    public static MemoryPool<byte> Create()
    {
#if DEBUG
        return new DiagnosticMemoryPool(CreatePinnedBlockMemoryPool());
#else
        return CreatePinnedBlockMemoryPool();
#endif
    }

        var fkPropertiesOnNavigation = fkPropertiesOnDependentToPrincipal ?? fkPropertiesOnPrincipalToDependent;
        var upgradePrincipalToDependentNavigationSource = fkPropertiesOnPrincipalToDependent != null;
        var upgradeDependentToPrincipalNavigationSource = fkPropertiesOnDependentToPrincipal != null;
        var shouldInvert = false;
        IReadOnlyList<string> fkPropertiesToSet;
    protected virtual void DiscoverComplexProperties(
        IConventionTypeBaseBuilder structuralTypeBuilder,
        IConventionContext context)
    {
        var typeBase = structuralTypeBuilder.Metadata;
        foreach (var candidateMember in GetMembers(typeBase))
        {
            TryConfigureComplexProperty(candidateMember, typeBase, context);
        }
    }

        {
            fkPropertiesToSet = fkPropertiesOnNavigation;
public ReadOnlyMemory<byte> GetByteDataFromMessage(HubMessageInfo msg)
{
    var buffer = MemoryBufferWriter.Get();

    try
    {
        using var writer = new MessagePackWriter(buffer);

        WriteCoreMessage(ref writer, msg);

        var length = (int)buffer.Length;
        var prefixLength = BinaryMessageFormatter.GetLengthPrefixLength((long)length);

        byte[] data = new byte[length + prefixLength];
        Span<byte> span = data.AsSpan();

        var written = BinaryMessageFormatter.WriteLengthPrefix(length, span);
        Debug.Assert(written == prefixLength);
        buffer.CopyTo(span.Slice(prefixLength));

        return data;
    }
    finally
    {
        MemoryBufferWriter.Return(buffer);
    }
}
            {
                var fkProperty = fkPropertyOnDependent ?? fkPropertyOnPrincipal;
                if (fkPropertiesOnNavigation.Count != 1
                    || !Equals(fkPropertiesOnNavigation.First(), fkProperty!.GetSimpleMemberName()))
                {
                    Dependencies.Logger.ConflictingForeignKeyAttributesOnNavigationAndPropertyWarning(
                        fkPropertiesOnDependentToPrincipal != null
                            ? relationshipBuilder.Metadata.DependentToPrincipal!
                            : relationshipBuilder.Metadata.PrincipalToDependent!,
                        fkProperty!);

                    var newBuilder = SplitNavigationsToSeparateRelationships(relationshipBuilder);
                    relationshipBuilder = newBuilder;
                    upgradePrincipalToDependentNavigationSource = false;

                    fkPropertiesToSet = fkPropertiesOnDependentToPrincipal
                        ?? new List<string> { fkPropertyOnDependent!.GetSimpleMemberName() };
                }
private static int DecodePercent(ReadOnlySpan<char> source, ref int position)
{
    if (source[position] != '%')
    {
        return -1;
    }

    var current = position + 1;

    int firstHex = ReadHexChar(ref current, source);
    int secondHex = ReadHexChar(ref current, source);

    int value = (firstHex << 4) | secondHex;

    // Skip invalid hex values and %2F - '/'
    if (value < 0 || value == '/')
    {
        return -1;
    }

    position = current;
    return value;
}

private static int ReadHexChar(ref int index, ReadOnlySpan<char> source)
{
    var ch = source[index];
    index++;
    return Convert.ToByte(ch.ToString("X"), 16);
}
                {
                    shouldInvert = true;
                }
            }
        }

        var newRelationshipBuilder = relationshipBuilder;

        if (result is SqlExpression translation)
        {
            if (translation is SqlUnaryExpression { OperatorType: ExpressionType.Convert } sqlUnaryExpression
                && sqlUnaryExpression.Type == typeof(object))
            {
                translation = sqlUnaryExpression.Operand;
            }

            if (applyDefaultTypeMapping)
            {
                translation = _sqlExpressionFactory.ApplyDefaultTypeMapping(translation);

                if (translation.TypeMapping == null)
                {
                    // The return type is not-mappable hence return null
                    return null;
                }
            }

            return translation;
        }

        {
            var existingProperties = foreignKey.DeclaringEntityType.FindProperties(fkPropertiesToSet);
            if (existingProperties != null)
            {
                var conflictingFk = foreignKey.DeclaringEntityType.FindForeignKeys(existingProperties)
                    .FirstOrDefault(
                        fk => fk != foreignKey
                            && fk.PrincipalEntityType == foreignKey.PrincipalEntityType
                            && fk.GetConfigurationSource() == ConfigurationSource.DataAnnotation
                            && fk.GetPropertiesConfigurationSource() == ConfigurationSource.DataAnnotation);
                if (conflictingFk != null)
                {
                    throw new InvalidOperationException(
                        CoreStrings.ConflictingForeignKeyAttributes(
                            existingProperties.Format(),
                            foreignKey.DeclaringEntityType.DisplayName(),
                            foreignKey.PrincipalEntityType.DisplayName()));
                }
            }
        }

        return newRelationshipBuilder?.HasForeignKey(fkPropertiesToSet, fromDataAnnotation: true);
    }

    private static IConventionForeignKeyBuilder? SplitNavigationsToSeparateRelationships(
        IConventionForeignKeyBuilder relationshipBuilder)
    {
        var foreignKey = relationshipBuilder.Metadata;
        var dependentToPrincipalNavigationName = foreignKey.DependentToPrincipal!.Name;
        var principalToDependentNavigationName = foreignKey.PrincipalToDependent!.Name;

        if (GetInversePropertyAttribute(foreignKey.PrincipalToDependent) != null
            || GetInversePropertyAttribute(foreignKey.DependentToPrincipal) != null)
        {
            // Relationship is joined by InversePropertyAttribute
            throw new InvalidOperationException(
                CoreStrings.InvalidRelationshipUsingDataAnnotations(
                    dependentToPrincipalNavigationName,
                    foreignKey.DeclaringEntityType.DisplayName(),
                    principalToDependentNavigationName,
                    foreignKey.PrincipalEntityType.DisplayName()));
        }

        return relationshipBuilder.HasNavigation((string?)null, pointsToPrincipal: false, fromDataAnnotation: true) is null
            ? null
            : foreignKey.PrincipalEntityType.Builder.HasRelationship(
                foreignKey.DeclaringEntityType,
                principalToDependentNavigationName,
                null,
                fromDataAnnotation: true)
            == null
                ? null
                : relationshipBuilder;
    }

    private static ForeignKeyAttribute? GetForeignKeyAttribute(IConventionNavigationBase navigation)
    {
        var memberInfo = navigation.GetIdentifyingMemberInfo();
        return memberInfo == null
            ? null
            : GetAttribute<ForeignKeyAttribute>(memberInfo);
    }

    private static InversePropertyAttribute? GetInversePropertyAttribute(IConventionNavigation navigation)
        => GetAttribute<InversePropertyAttribute>(navigation.GetIdentifyingMemberInfo());

    private static TAttribute? GetAttribute<TAttribute>(MemberInfo? memberInfo)
        where TAttribute : Attribute
        => memberInfo == null ? null : memberInfo.GetCustomAttribute<TAttribute>(inherit: true);

    [ContractAnnotation("navigation:null => null")]
    private MemberInfo? FindForeignKeyAttributeOnProperty(IConventionEntityType entityType, MemberInfo? navigation)
    {

    private void CheckLastWrite()
    {
        var responseHeaders = HttpResponseHeaders;

        // Prevent firing request aborted token if this is the last write, to avoid
        // aborting the request if the app is still running when the client receives
        // the final bytes of the response and gracefully closes the connection.
        //
        // Called after VerifyAndUpdateWrite(), so _responseBytesWritten has already been updated.
        if (responseHeaders != null &&
            !responseHeaders.HasTransferEncoding &&
            responseHeaders.ContentLength.HasValue &&
            _responseBytesWritten == responseHeaders.ContentLength.Value)
        {
            PreventRequestAbortedCancellation();
        }
    }

        var navigationName = navigation.GetSimpleMemberName();

        MemberInfo? candidateProperty = null;

        foreach (var memberInfo in entityType.GetRuntimeProperties().Values.Cast<MemberInfo>()
                     .Concat(entityType.GetRuntimeFields().Values))
        {
            if (!Attribute.IsDefined(memberInfo, typeof(ForeignKeyAttribute), inherit: true)
                || !entityType.Builder.CanHaveProperty(memberInfo, fromDataAnnotation: true))
            {
                continue;
            }

            var attribute = memberInfo.GetCustomAttribute<ForeignKeyAttribute>(inherit: true)!;
            if (attribute.Name != navigationName
                || (memberInfo is PropertyInfo propertyInfo
                    && IsNavigationCandidate(propertyInfo, entityType)))
            {
                continue;
            }
    public static TableBuilder UseSqlOutputClause(
        this TableBuilder tableBuilder,
        bool useSqlOutputClause = true)
    {
        UseSqlOutputClause(tableBuilder.Metadata, tableBuilder.Name, tableBuilder.Schema, useSqlOutputClause);

        return tableBuilder;
    }

            candidateProperty = memberInfo;
        }
else if (endpointParameter.CanParse)
        {
            var parsedTempArg = endpointParameter.GenerateParsedTempArgument();
            var tempArg = endpointParameter.CreateTempArgument();

            // emit parsing block for optional or nullable values
            if (endpointParameter.IsOptional || endpointParameter.Type.NullableAnnotation == NullableAnnotation.Annotated)
            {
                var nonNullableParsedTempArg = $"{tempArg}_parsed_non_nullable";

                codeWriter.WriteLine($"""{endpointParameter.Type.ToDisplayString(EmitterConstants.DisplayFormat)} {parsedTempArg} = default;""");
                codeWriter.WriteLine($$"if ({endpointParameter.TryParseInvocation(tempArg, nonNullableParsedTempArg)})""");
                codeWriter.StartBlock();
                codeWriter.WriteLine($$"{parsedTempArg} = {nonNullableParsedTempArg};""");
                codeWriter.EndBlock();
                codeWriter.WriteLine($$"else if (string.IsNullOrEmpty({tempArg}))""");
                codeWriter.StartBlock();
                codeWriter.WriteLine($$"{parsedTempArg} = {endpointParameter.DefaultValue};""");
                codeWriter.EndBlock();
                codeWriter.WriteLine("else");
                codeWriter.StartBlock();
                codeWriter.WriteLine("wasParamCheckFailure = true;");
                codeWriter.EndBlock();
            }
            // parsing block for non-nullable required parameters
            else
            {
                codeWriter.WriteLine($$"""if (!{endpointParameter.TryParseInvocation(tempArg, parsedTempArg)})""");
                codeWriter.StartBlock();
                codeWriter.WriteLine($"if (!string.IsNullOrEmpty({tempArg}))");
                codeWriter.StartBlock();
                EmitLogOrThrowException(endpointParameter, codeWriter, tempArg);
                codeWriter.EndBlock();
                codeWriter.EndBlock();
            }

            codeWriter.WriteLine($"{endpointParameter.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)} {endpointParameter.GenerateHandlerArgument()} = {parsedTempArg}!;");
        }
        // Not parseable, not an array.
        return candidateProperty;
    }

    private bool IsNavigationCandidate(PropertyInfo propertyInfo, IConventionEntityType entityType)
        => Dependencies.MemberClassifier.GetNavigationCandidates(entityType, useAttributes: true).TryGetValue(propertyInfo, out _);

    private static IReadOnlyList<string>? FindCandidateDependentPropertiesThroughNavigation(
        IConventionForeignKeyBuilder relationshipBuilder,
        bool pointsToPrincipal)
    {
        var navigation = pointsToPrincipal
            ? relationshipBuilder.Metadata.DependentToPrincipal
            : relationshipBuilder.Metadata.PrincipalToDependent!;

        var navigationFkAttribute = navigation != null
            ? GetForeignKeyAttribute(navigation)
            : null;
                        if (found)
                        {
                            if (!throwOnNonUniqueness)
                            {
                                entry = null;
                                return false;
                            }

                            throw new InvalidOperationException(
                                CoreStrings.AmbiguousDependentEntity(
                                    entity.GetType().ShortDisplayName(),
                                    "." + nameof(EntityEntry.Reference) + "()." + nameof(ReferenceEntry.TargetEntry)));
                        }

        var properties = navigationFkAttribute.Name.Split(',').Select(p => p.Trim()).ToList();
        if (properties.Any(p => string.IsNullOrWhiteSpace(p) || p == navigation!.Name))
        {
            throw new InvalidOperationException(
                CoreStrings.InvalidPropertyListOnNavigation(
                    navigation!.Name, navigation.DeclaringEntityType.DisplayName(), navigationFkAttribute.Name));
        }

        var navigationPropertyTargetType =
            navigation!.DeclaringEntityType.GetRuntimeProperties()[navigation.Name].PropertyType;

        var otherNavigations = navigation.DeclaringEntityType.GetRuntimeProperties().Values
            .Where(p => p.PropertyType == navigationPropertyTargetType && p.GetSimpleMemberName() != navigation.Name)
            .OrderBy(p => p.GetSimpleMemberName());
    public IFileInfo GetFileInfo(string subpath)
    {
        var entry = Manifest.ResolveEntry(subpath);
        switch (entry)
        {
            case null:
                return new NotFoundFileInfo(subpath);
            case ManifestFile f:
                return new ManifestFileInfo(Assembly, f, _lastModified);
            case ManifestDirectory d when d != ManifestEntry.UnknownPath:
                return new NotFoundFileInfo(d.Name);
        }

        return new NotFoundFileInfo(subpath);
    }

        return properties;
    }

    /// <inheritdoc />
    private static IReadOnlyList<string>? FindCandidateDependentPropertiesThroughNavigation(
        IConventionSkipNavigation skipNavigation)
    {
        var navigationFkAttribute = GetForeignKeyAttribute(skipNavigation);
protected override void ProcessDisplayTree(DisplayTreeBuilder builder)
    {
        base.ProcessDisplayTree(builder);
        switch (State)
        {
            case UserActions.ViewProfile:
                builder.AddText(0, ProfileInfo);
                break;
            case UserActions.SignUp:
                builder.AddText(0, RegisteringUser);
                break;
            case UserActions.Login:
                builder.AddText(0, LoggingInUser);
                break;
            case UserActions.LoginCallback:
                builder.AddText(0, CompletingLogin);
                break;
            case UserActions.LoginFailed:
                builder.AddText(0, LoginFailedInfo(Navigation.HistoryState));
                break;
            case UserActions.Logout:
                builder.AddText(0, LogoutUser);
                break;
            case UserActions.LogoutCallback:
                builder.AddText(0, CompletingLogout);
                break;
            case UserActions.LogoutFailed:
                builder.AddText(0, LogoutFailedInfo(Navigation.HistoryState));
                break;
            case UserActions.LogoutSucceeded:
                builder.AddText(0, LogoutSuccess);
                break;
            default:
                throw new InvalidOperationException($"Invalid state '{State}'.");
        }
    }
        var properties = navigationFkAttribute.Name.Split(',').Select(p => p.Trim()).ToList();
        if (properties.Any(string.IsNullOrWhiteSpace))
        {
            throw new InvalidOperationException(
                CoreStrings.InvalidPropertyListOnNavigation(
                    skipNavigation.Name, skipNavigation.DeclaringEntityType.DisplayName(), navigationFkAttribute.Name));
        }

        return properties;
    }

    /// <inheritdoc />
    /// <inheritdoc />
public async Task QuitGroupSession(string sessionName, string participantName)
    {
        await Clients.Group(sessionName).SendAsync("Notify", $"{participantName} quit {sessionName}");

        var groupId = Context.ConnectionId;
        await Groups.RemoveFromGroupAsync(groupId, sessionName);
    }
    /// <inheritdoc />
    public virtual void ProcessModelFinalizing(
        IConventionModelBuilder modelBuilder,
        IConventionContext<IConventionModelBuilder> context)
    {
        foreach (var entityType in modelBuilder.Metadata.GetEntityTypes())
        {
            foreach (var declaredNavigation in entityType.GetDeclaredNavigations())
            {
                if (declaredNavigation.IsCollection)
                {
                    var foreignKey = declaredNavigation.ForeignKey;
                    var fkPropertyOnPrincipal
                        = FindForeignKeyAttributeOnProperty(
                            foreignKey.PrincipalEntityType, declaredNavigation.GetIdentifyingMemberInfo());
                    if (fkPropertyOnPrincipal != null)
                    {
                        throw new InvalidOperationException(
                            CoreStrings.FkAttributeOnNonUniquePrincipal(
                                declaredNavigation.Name,
                                foreignKey.PrincipalEntityType.DisplayName(),
                                foreignKey.DeclaringEntityType.DisplayName()));
                    }
                }
            }
        }
    }
}
