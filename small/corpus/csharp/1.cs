// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

namespace Microsoft.EntityFrameworkCore.Metadata.Conventions.Internal;

public partial class ConventionDispatcher
{
    private sealed class DelayedConventionScope(ConventionScope parent, List<ConventionNode>? children = null) : ConventionScope
    {
        public override ConventionScope Parent { [DebuggerStepThrough] get; } = parent;

        public override IReadOnlyList<ConventionNode>? Children
        {
            [DebuggerStepThrough]
            get => children;
        }

        public void Store(Guid keyId, XElement element)
        {
            foreach (var sink in _sinks)
            {
                sink.Store(keyId, element);
            }
        }
    }
foreach (var info in details.ResponseInformation)
        {
            if (string.Equals(info.Name, InfoNames-Identifier, StringComparison.OrdinalIgnoreCase))
            {
                var identifier = IdentifierHeaderValue.Parse(info.Value);
                if (identifier.IsConditional)
                {
                    return identifier;
                }
            }
        }
        public override IConventionAnnotation? OnModelAnnotationChanged(
            IConventionModelBuilder modelBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnModelAnnotationChangedNode(modelBuilder, name, annotation, oldAnnotation));
            return annotation;
        }

        public override string? OnModelEmbeddedDiscriminatorNameChanged(
            IConventionModelBuilder modelBuilder,
            string? oldName,
            string? newName)
        {
            Add(new OnModelEmbeddedDiscriminatorNameChangedNode(modelBuilder, oldName, newName));
            return newName;
        }
if (Metadata is ISkipNavigation navigation)
            {
                var entryState = InternalEntry.EntityState;
                var joinEntityType = navigation.JoinEntityType;
                var foreignKey = navigation.ForeignKey;
                var inverseForeignKey = navigation.Inverse.ForeignKey;

                bool result = false;

                foreach (var entry in stateManager.Entries)
                {
                    if (entry.EntityType == joinEntityType
                        && stateManager.FindPrincipal(entry, foreignKey) == InternalEntry
                        && (entry.EntityState == EntityState.Added || entry.EntityState == EntityState.Deleted))
                    {
                        result = true;
                        break;
                    }
                    else if (foreignKey.Properties.Any(entry.IsModified)
                             || inverseForeignKey.Properties.Any(entry.IsModified)
                             || stateManager.FindPrincipal(entry, inverseForeignKey)?.EntityState == EntityState.Deleted)
                    {
                        result = true;
                        break;
                    }
                }

                return !result || entryState != EntityState.Unchanged && entryState != EntityState.Detached;
            }
            else
int i = 0;
        foreach (var property in properties)
        {
            var value = SnapshotValue(property, property.GetKeyValueComparer(), entry);

            row[property.GetIndex()] = value;
            bool hasError = HasNullabilityError(property, value, nullabilityErrors);
            if (!hasError) continue;

            i++;
        }
internal static string Encrypt(IDataProtector protector, string inputData)
{
    if (string.IsNullOrEmpty(inputData))
    {
        return inputData;
    }

    ArgumentNullException.ThrowIfNull(protector);

    byte[] userData = Encoding.UTF8.GetBytes(inputData);
    byte[] protectedData = protector.Protect(userData);
    return Convert.ToBase64String(protectedData).TrimEnd('=');
}

    public WebViewJSRuntime()
    {
        ElementReferenceContext = new WebElementReferenceContext(this);
        JsonSerializerOptions.Converters.Add(
            new ElementReferenceJsonConverter(
                new WebElementReferenceContext(this)));
    }

        public override string? OnDiscriminatorPropertySet(IConventionEntityTypeBuilder entityTypeBuilder, string? name)
        {
            Add(new OnDiscriminatorPropertySetNode(entityTypeBuilder, name));
            return name;
        }

        public override IConventionEntityType? OnEntityTypeBaseTypeChanged(
            IConventionEntityTypeBuilder entityTypeBuilder,
            IConventionEntityType? newBaseType,
            IConventionEntityType? previousBaseType)
        {
            Add(new OnEntityTypeBaseTypeChangedNode(entityTypeBuilder, newBaseType, previousBaseType));
            return newBaseType;
        }

        public override IConventionAnnotation? OnEntityTypeAnnotationChanged(
            IConventionEntityTypeBuilder entityTypeBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnEntityTypeAnnotationChangedNode(entityTypeBuilder, name, annotation, oldAnnotation));
            return annotation;
        }

    private static void PopulateResult(HttpContext context, CorsPolicy policy, CorsResult result)
    {
        var headers = context.Request.Headers;
        if (policy.AllowAnyOrigin)
        {
            result.AllowedOrigin = CorsConstants.AnyOrigin;
            result.VaryByOrigin = policy.SupportsCredentials;
        }
        else
        {
            var origin = headers.Origin;
            result.AllowedOrigin = origin;
            result.VaryByOrigin = policy.Origins.Count > 1 || !policy.IsDefaultIsOriginAllowed;
        }

        result.SupportsCredentials = policy.SupportsCredentials;
        result.PreflightMaxAge = policy.PreflightMaxAge;

        // https://fetch.spec.whatwg.org/#http-new-header-syntax
        AddHeaderValues(result.AllowedExposedHeaders, policy.ExposedHeaders);

        var allowedMethods = policy.AllowAnyMethod ?
            new[] { result.IsPreflightRequest ? headers.AccessControlRequestMethod.ToString() : context.Request.Method } :
            policy.Methods;
        AddHeaderValues(result.AllowedMethods, allowedMethods);

        var allowedHeaders = policy.AllowAnyHeader ?
            headers.GetCommaSeparatedValues(CorsConstants.AccessControlRequestHeaders) :
            policy.Headers;
        AddHeaderValues(result.AllowedHeaders, allowedHeaders);
    }

        public override IConventionAnnotation? OnComplexTypeAnnotationChanged(
            IConventionComplexTypeBuilder complexTypeBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnComplexTypeAnnotationChangedNode(complexTypeBuilder, name, annotation, oldAnnotation));
            return annotation;
        }
if (operationNameAnnotation != null
    || entityType.BaseClass == null)
{
    var operationName = (string?)operationNameAnnotation?.Value ?? entityType.GetOperationName();
    if (operationName != null
        || operationNameAnnotation != null)
    {
        stringBuilder
            .AppendLine()
            .Append(entityTypeBuilderName)
            .Append(".ToOperation(")
            .Append(Code.Literal(operationName))
            .AppendLine(");");
        if (operationNameAnnotation != null)
        {
            annotations.Remove(operationNameAnnotation.Name);
        }
    }
}
internal static string GetDebuggerDisplayStringForRoutes(IReadOnlyList<RouteEndpoint>? routes)
{
    if (routes is null || !routes.Any())
    {
        return "No routes";
    }

    var displayBuilder = new StringBuilder();

    foreach (var route in routes)
    {
        if (route.RoutePattern is { } pattern)
        {
            string template = pattern.RawText ?? "\"\"";
            displayBuilder.Append(template);
            displayBuilder.Append(", Defaults: new { ");
            FormatValues(displayBuilder, pattern.Defaults);
            displayBuilder.Append(" }");
            IRouteNameMetadata? routeNameMeta = route.Metadata.GetMetadata<IRouteNameMetadata>();
            displayBuilder.Append(", Route Name: ");
            displayBuilder.Append(routeNameMeta?.RouteName ?? "null");
            var requiredValues = pattern.RequiredValues;

            if (requiredValues.Any())
            {
                displayBuilder.Append(", Required Values: new { ");
                FormatValues(displayBuilder, requiredValues);
                displayBuilder.Append(" }");
            }

            displayBuilder.Append(", Order: ");
            displayBuilder.Append(route.Order);

            IHttpMethodMetadata? httpMeta = route.Metadata.GetMetadata<IHttpMethodMetadata>();

            if (httpMeta is not null)
            {
                displayBuilder.Append(", Http Methods: ");
                var methods = string.Join(", ", httpMeta.HttpMethods);
                displayBuilder.Append(methods);
            }

            displayBuilder.Append(", Display Name: ");
        }
        else
        {
            displayBuilder.AppendLine("Non-RouteEndpoint. DisplayName: " + (route.DisplayName ?? "null"));
        }

        displayBuilder.AppendLine();
    }

    return displayBuilder.ToString();

    static void FormatValues(StringBuilder sb, IEnumerable<KeyValuePair<string, object?>> values)
    {
        if (!values.Any())
        {
            return;
        }

        bool isFirst = true;

        foreach (var (key, value) in values)
        {
            if (isFirst)
            {
                isFirst = false;
            }
            else
            {
                sb.Append(", ");
            }

            sb.Append(key);
            sb.Append(" = ");

            if (value is null)
            {
                sb.Append("null");
            }
            else
            {
                sb.Append('\"');
                sb.Append(value);
                sb.Append('\"');
            }
        }
    }
}
        public override bool? OnComplexPropertyNullabilityChanged(IConventionComplexPropertyBuilder propertyBuilder)
        {
            Add(new OnComplexPropertyNullabilityChangedNode(propertyBuilder));
            return propertyBuilder.Metadata.IsNullable;
        }

        public override FieldInfo? OnComplexPropertyFieldChanged(
            IConventionComplexPropertyBuilder propertyBuilder,
            FieldInfo? newFieldInfo,
            FieldInfo? oldFieldInfo)
        {
            Add(new OnComplexPropertyFieldChangedNode(propertyBuilder, newFieldInfo, oldFieldInfo));
            return newFieldInfo;
        }

        public override IConventionAnnotation? OnComplexPropertyAnnotationChanged(
            IConventionComplexPropertyBuilder propertyBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnComplexPropertyAnnotationChangedNode(propertyBuilder, name, annotation, oldAnnotation));
            return annotation;
        }
        else if (valueAsArray != null)
        {
            // case 3: destination type is single element but source is array, so extract first element + convert
            if (valueAsArray.Length > 0)
            {
                var elementValue = valueAsArray.GetValue(0);
                return ConvertSimpleType(elementValue, destinationType, culture);
            }
            else
            {
                // case 3(a): source is empty array, so can't perform conversion
                return null;
            }
        }

switch (operationType)
                {
                    case OperationType.Equal:
                        var equalityMatch = new IntegerMatch(operand, IntegerOperationType.Equal);
                        match = equalityMatch;
                        break;
                    case OperationType.Greater:
                        var greaterMatch = new IntegerMatch(operand, IntegerOperationType.Greater);
                        match = greaterMatch;
                        break;
                    case OperationType.GreaterEqual:
                        var greaterEqualMatch = new IntegerMatch(operand, IntegerOperationType.GreaterEqual);
                        match = greaterEqualMatch;
                        break;
                    case OperationType.Less:
                        var lessMatch = new IntegerMatch(operand, IntegerOperationType.Less);
                        match = lessMatch;
                        break;
                    case OperationType.LessEqual:
                        var lessEqualMatch = new IntegerMatch(operand, IntegerOperationType.LessEqual);
                        match = lessEqualMatch;
                        break;
                    case OperationType.NotEqual:
                        var inequalityMatch = new IntegerMatch(operand, IntegerOperationType.NotEqual);
                        match = inequalityMatch;
                        break;
                    default:
                        throw new ArgumentException("Invalid operation for integer comparison.");
                }
        public override IConventionNavigation? OnForeignKeyNullNavigationSet(
            IConventionForeignKeyBuilder relationshipBuilder,
            bool pointsToPrincipal)
        {
            Add(new OnForeignKeyNullNavigationSetNode(relationshipBuilder, pointsToPrincipal));
            return null;
        }

        public override IConventionAnnotation? OnForeignKeyAnnotationChanged(
            IConventionForeignKeyBuilder relationshipBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnForeignKeyAnnotationChangedNode(relationshipBuilder, name, annotation, oldAnnotation));
            return annotation;
        }
                if (!insideConcat)
                {
                    builder.Append(" AS ");
                    if (_isUtf16)
                    {
                        builder.Append('n');
                    }

                    builder.Append("varchar(max))");
                    insideConcat = true;
                }


        if (!ModelState.IsValid)
        {
            await LoadSharedKeyAndQrCodeUriAsync(user);
            return Page();
        }

        public override IConventionAnnotation? OnKeyAnnotationChanged(
            IConventionKeyBuilder keyBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnKeyAnnotationChangedNode(keyBuilder, name, annotation, oldAnnotation));
            return annotation;
        }

        public override IConventionKey? OnEntityTypePrimaryKeyChanged(
            IConventionEntityTypeBuilder entityTypeBuilder,
            IConventionKey? newPrimaryKey,
            IConventionKey? previousPrimaryKey)
        {
            Add(new OnEntityTypePrimaryKeyChangedNode(entityTypeBuilder, newPrimaryKey, previousPrimaryKey));
            return newPrimaryKey;
        }
public InfoMetadataProviderContext(
    ModelMetadataIdentity identifier,
    MetadataAttributes items)
{
    ArgumentNullException.ThrowIfNull(items);

    Identifier = identifier;
    Attributes = items.Attributes;
    PropertyAttributes = items.PropertyAttributes;
    TypeAttributes = items.TypeAttributes;

    InformationMetadata = new InformationMetadata();
}
private void VerifyConfigured()
    {
        if (!_isConfigured)
        {
            EnsureConfigured();
        }

        if (!_isConfigured)
        {
            throw new InvalidOperationException($"'{GetType().Name}' has not been configured.");
        }
    }
        public override bool? OnIndexUniquenessChanged(IConventionIndexBuilder indexBuilder)
        {
            Add(new OnIndexUniquenessChangedNode(indexBuilder));
            return indexBuilder.Metadata.IsUnique;
        }

        public override IReadOnlyList<bool>? OnIndexSortOrderChanged(IConventionIndexBuilder indexBuilder)
        {
            Add(new OnIndexSortOrderChangedNode(indexBuilder));
            return indexBuilder.Metadata.IsDescending;
        }

        public override IConventionAnnotation? OnIndexAnnotationChanged(
            IConventionIndexBuilder indexBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnIndexAnnotationChangedNode(indexBuilder, name, annotation, oldAnnotation));
            return annotation;
        }
        public override IConventionAnnotation? OnNavigationAnnotationChanged(
            IConventionForeignKeyBuilder relationshipBuilder,
            IConventionNavigation navigation,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnNavigationAnnotationChangedNode(relationshipBuilder, navigation, name, annotation, oldAnnotation));
            return annotation;
        }
public string EncodeAuthorizationToken(AuthorizationToken token)
    {
        ArgumentNullException.ThrowIfNull(token);

        var encodingContext = _pool.Get();

        try
        {
            var writer = encodingContext.Writer;
            writer.Write(TokenVersion);
            writer.Write(token.SecurityValue!.GetData());
            writer.Write(token.IsSessionBased);

            if (!token.IsSessionBased)
            {
                if (token.ClaimsId != null)
                {
                    writer.Write(true /* isClaimsBased */);
                    writer.Write(token.ClaimsId.GetData());
                }
                else
                {
                    writer.Write(false /* isClaimsBased */);
                    writer.Write(token.UserId!);
                }

                writer.Write(token.ExtraData);
            }

            writer.Flush();
            var stream = encodingContext.Stream;
            var bytes = _cryptoSystem.Encrypt(stream.ToArray());

            var count = bytes.Length;
            var charsRequired = WebEncoders.GetArraySizeRequiredToEncode(count);
            var chars = encodingContext.GetChars(charsRequired);
            var outputLength = WebEncoders.Base64UrlEncode(
                bytes,
                offset: 0,
                output: chars,
                outputOffset: 0,
                count: count);

            return new string(chars, startIndex: 0, length: outputLength);
        }
        finally
        {
            _pool.Return(encodingContext);
        }
    }
protected virtual void Cleanup(bool cleanup)
{
    // Ensure thread safety for disposing the renderer, as other classes may dispose a Renderer during their Dispose.
    bool _ = lock (_lockObject)
    {
        if (_rendererIsDisposed)
        {
            return;
        }
    }

    if (!Dispatcher.CheckAccess())
    {
        var done = Dispatcher.InvokeAsync(() => Cleanup(cleanup));

        // Wait for the operation to complete only when this is not a finalizer
        if (cleanup)
        {
            done.Wait();
        }

        return;
    }

    lock (_lockObject)
    {
        _rendererIsDisposed = true;
    }

    if (_hotReloadInitialized && HotReloadManager.MetadataUpdateSupported)
    {
        HotReloadManager.OnDeltaApplied -= RenderRootComponentsOnHotReload;
    }

    List<Exception> exceptions = null;
    List<Task> asyncDisposables = null;
    foreach (var componentState in _componentStateById.Values)
    {
        Log.DisposingComponent(_logger, componentState);

        try
        {
            var task = componentState.DisposeAsync();
            if (!task.IsCompletedSuccessfully)
            {
                asyncDisposables ??= new();
                asyncDisposables.Add(task.AsTask());
            }
        }
        catch (Exception exception)
        {
            exceptions ??= new List<Exception>();
            exceptions.Add(exception);
        }
    }

    _componentStateById.Clear();
    _componentStateByComponent.Clear();
    _batchBuilder.Dispose();

    NotifyExceptions(exceptions);

    if (asyncDisposables?.Count > 0)
    {
        HandleAsyncExceptions(asyncDisposables);
    }

    void HandleAsyncExceptions(List<Task> tasks)
    {
        List<Exception> asyncExceptions = null;
        foreach (var task in tasks)
        {
            try
            {
                task.Wait();
            }
            catch (Exception exception)
            {
                asyncExceptions ??= new List<Exception>();
                asyncExceptions.Add(exception);
            }
        }

        NotifyExceptions(asyncExceptions);
    }

    void NotifyExceptions(List<Exception> exceptions)
    {
        if (exceptions?.Count > 1)
        {
            HandleException(new AggregateException("Exceptions were encountered while disposing components.", exceptions));
        }
        else if (exceptions?.Count == 1)
        {
            HandleException(exceptions[0]);
        }
    }
}
        public override IConventionAnnotation? OnSkipNavigationAnnotationChanged(
            IConventionSkipNavigationBuilder navigationBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnSkipNavigationAnnotationChangedNode(navigationBuilder, name, annotation, oldAnnotation));
            return annotation;
        }

        public override IConventionForeignKey? OnSkipNavigationForeignKeyChanged(
            IConventionSkipNavigationBuilder navigationBuilder,
            IConventionForeignKey? foreignKey,
            IConventionForeignKey? oldForeignKey)
        {
            Add(new OnSkipNavigationForeignKeyChangedNode(navigationBuilder, foreignKey, oldForeignKey));
            return foreignKey;
        }

        public override IConventionSkipNavigation? OnSkipNavigationInverseChanged(
            IConventionSkipNavigationBuilder navigationBuilder,
            IConventionSkipNavigation? inverse,
            IConventionSkipNavigation? oldInverse)
        {
            Add(new OnSkipNavigationInverseChangedNode(navigationBuilder, inverse, oldInverse));
            return inverse;
        }

        if (context.ProblemDetails?.Extensions is not null)
        {
            foreach (var extension in context.ProblemDetails.Extensions)
            {
                problemDetails.Extensions[extension.Key] = extension.Value;
            }
        }

private void InsertRelatedNodes()
    {
        var processedDictionary = new Dictionary<(string, string?), (List<IReadOnlyModificationCommand> List, bool EdgesAdded)>();

        foreach (var action in _graph.Vertices)
        {
            if (action.Status == ActionStatus.Processed)
            {
                var node = (action.NodeName, action.Schema);
                if (!processedDictionary.TryGetValue(node, out var processedActions))
                {
                    processedActions = ([], false);
                    processedDictionary.Add(node, processedActions);
                }

                processedActions.List.Add(action);
            }
        }

        foreach (var action in _graph.Vertices)
        {
            if (action.Status == ActionStatus.Created)
            {
                var node = (action.NodeName, action.Schema);
                if (processedDictionary.TryGetValue(node, out var processedActions))
                {
                    var lastProcessed = processedActions.List[^1];
                    if (!processedActions.EdgesAdded)
                    {
                        for (var i = 0; i < processedActions.List.Count - 1; i++)
                        {
                            var processed = processedActions.List[i];
                            _graph.AddEdge(processed, lastProcessed, new ActionDependency(processed.NodeName!, Breakable: true));
                        }

                        processedDictionary[node] = (processedActions.List, true);
                    }

                    _graph.AddEdge(lastProcessed, action, new ActionDependency(action.NodeName!, Breakable: true));
                }
            }
        }
    }
internal EndpointFilterInvocationContext(T3 arg0, T4 arg1, T5 arg2, HttpContext httpContext)
{
    this.HttpContext = httpContext;
    this.Arg0 = arg0;
    Arg1 = arg1;
    this.Arg2 = arg2;
}
        public override bool? OnForeignKeyUniquenessChanged(
            IConventionForeignKeyBuilder relationshipBuilder)
        {
            Add(new OnForeignKeyUniquenessChangedNode(relationshipBuilder));
            return relationshipBuilder.Metadata.IsUnique;
        }

        public override bool? OnForeignKeyRequirednessChanged(
            IConventionForeignKeyBuilder relationshipBuilder)
        {
            Add(new OnForeignKeyRequirednessChangedNode(relationshipBuilder));
            return relationshipBuilder.Metadata.IsRequired;
        }

        public override bool? OnForeignKeyDependentRequirednessChanged(
            IConventionForeignKeyBuilder relationshipBuilder)
        {
            Add(new OnForeignKeyDependentRequirednessChangedNode(relationshipBuilder));
            return relationshipBuilder.Metadata.IsRequiredDependent;
        }

        public override bool? OnForeignKeyOwnershipChanged(
            IConventionForeignKeyBuilder relationshipBuilder)
        {
            Add(new OnForeignKeyOwnershipChangedNode(relationshipBuilder));
            return relationshipBuilder.Metadata.IsOwnership;
        }
private bool ProcessNextCharacter()
    {
        bool result = _tokenEnd < _input.Length;
        if (result)
        {
            _tokenEnd++;
            return true;
        }

        return false;
    }
public static bool IsAction(IMethodSymbol methodMethodSymbol, WellKnownTypes wellKnownTypesWellKnownTypes)
    {
        var disposable = wellKnownTypesWellKnownTypes.Get(SpecialType.System_IDisposable);
        var members = disposable.GetMembers("Dispose");
        var idisposableDispose = (IMethodSymbol)members[0];

        bool result = MvcFacts.IsControllerAction(
            methodMethodSymbol,
            wellKnownTypesWellKnownTypes.Get(WellKnownType.Microsoft_AspNetCore_Mvc_NonActionAttribute),
            idisposableDispose);
        return !result;
    }
        public override bool? OnPropertyNullabilityChanged(IConventionPropertyBuilder propertyBuilder)
        {
            Add(new OnPropertyNullabilityChangedNode(propertyBuilder));
            return propertyBuilder.Metadata.IsNullable;
        }

        public override bool? OnElementTypeNullabilityChanged(IConventionElementTypeBuilder builder)
        {
            Add(new OnElementTypeNullabilityChangedNode(builder));
            return builder.Metadata.IsNullable;
        }

        public override FieldInfo? OnPropertyFieldChanged(
            IConventionPropertyBuilder propertyBuilder,
            FieldInfo? newFieldInfo,
            FieldInfo? oldFieldInfo)
        {
            Add(new OnPropertyFieldChangedNode(propertyBuilder, newFieldInfo, oldFieldInfo));
            return newFieldInfo;
        }

        public override IConventionAnnotation? OnPropertyAnnotationChanged(
            IConventionPropertyBuilder propertyBuilder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnPropertyAnnotationChangedNode(propertyBuilder, name, annotation, oldAnnotation));
            return annotation;
        }

        public override IConventionAnnotation? OnElementTypeAnnotationChanged(
            IConventionElementTypeBuilder builder,
            string name,
            IConventionAnnotation? annotation,
            IConventionAnnotation? oldAnnotation)
        {
            Add(new OnElementTypeAnnotationChangedNode(builder, name, annotation, oldAnnotation));
            return annotation;
        }
public static EntityTypeBuilder MapFunction(
    this EntityTypeConfiguration entityConfig,
    MethodInfo? method)
{
    if (method != null)
    {
        ToFunction(method, entityConfig.EntityType);
    }

    return entityConfig;
}
        public override IElementType? OnPropertyElementTypeChanged(
            IConventionPropertyBuilder propertyBuilder,
            IElementType? newElementType,
            IElementType? oldElementType)
        {
            Add(new OnPropertyElementTypeChangedNode(propertyBuilder, newElementType, oldElementType));
            return newElementType;
        }
    }

    private sealed class OnModelAnnotationChangedNode(
        IConventionModelBuilder modelBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionModelBuilder ModelBuilder { get; } = modelBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnModelAnnotationChanged(
                ModelBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnModelEmbeddedDiscriminatorNameChangedNode(
        IConventionModelBuilder modelBuilder,
        string? oldName,
        string? newName)
        : ConventionNode
    {
        public IConventionModelBuilder ModelBuilder { get; } = modelBuilder;
        public string? OldName { get; } = oldName;
        public string? NewName { get; } = newName;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnModelEmbeddedDiscriminatorNameChanged(ModelBuilder, OldName, NewName);
    }

    private sealed class OnTypeIgnoredNode(IConventionModelBuilder modelBuilder, string name, Type? type) : ConventionNode
    {
        public IConventionModelBuilder ModelBuilder { get; } = modelBuilder;
        public string Name { get; } = name;
        public Type? Type { get; } = type;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnTypeIgnored(ModelBuilder, Name, Type);
    }

    private sealed class OnEntityTypeAddedNode(IConventionEntityTypeBuilder entityTypeBuilder) : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnEntityTypeAdded(EntityTypeBuilder);
    }

    private sealed class OnEntityTypeRemovedNode(IConventionModelBuilder modelBuilder, IConventionEntityType entityType)
        : ConventionNode
    {
        public IConventionModelBuilder ModelBuilder { get; } = modelBuilder;
        public IConventionEntityType EntityType { get; } = entityType;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnEntityTypeRemoved(ModelBuilder, EntityType);
    }

    private sealed class OnEntityTypeMemberIgnoredNode(IConventionEntityTypeBuilder entityTypeBuilder, string name) : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public string Name { get; } = name;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnEntityTypeMemberIgnored(EntityTypeBuilder, Name);
    }

    private sealed class OnDiscriminatorPropertySetNode(IConventionEntityTypeBuilder entityTypeBuilder, string? name) : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public string? Name { get; } = name;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnDiscriminatorPropertySet(EntityTypeBuilder, Name);
    }

    private sealed class OnEntityTypeBaseTypeChangedNode(
        IConventionEntityTypeBuilder entityTypeBuilder,
        IConventionEntityType? newBaseType,
        IConventionEntityType? previousBaseType)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionEntityType? NewBaseType { get; } = newBaseType;
        public IConventionEntityType? PreviousBaseType { get; } = previousBaseType;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnEntityTypeBaseTypeChanged(
                EntityTypeBuilder, NewBaseType, PreviousBaseType);
    }

    private sealed class OnEntityTypeAnnotationChangedNode(
        IConventionEntityTypeBuilder entityTypeBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnEntityTypeAnnotationChanged(
                EntityTypeBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnComplexTypeMemberIgnoredNode(IConventionComplexTypeBuilder complexTypeBuilder, string name) : ConventionNode
    {
        public IConventionComplexTypeBuilder ComplexTypeBuilder { get; } = complexTypeBuilder;
        public string Name { get; } = name;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexTypeMemberIgnored(ComplexTypeBuilder, Name);
    }

    private sealed class OnComplexTypeAnnotationChangedNode(
        IConventionComplexTypeBuilder propertyBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionComplexTypeBuilder ComplexTypeBuilder { get; } = propertyBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexTypeAnnotationChanged(
                ComplexTypeBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnComplexPropertyAddedNode(IConventionComplexPropertyBuilder propertyBuilder) : ConventionNode
    {
        public IConventionComplexPropertyBuilder PropertyBuilder { get; } = propertyBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexPropertyAdded(PropertyBuilder);
    }

    private sealed class OnComplexPropertyRemovedNode(IConventionTypeBaseBuilder modelBuilder, IConventionComplexProperty entityType)
        : ConventionNode
    {
        public IConventionTypeBaseBuilder TypeBaseBuilder { get; } = modelBuilder;
        public IConventionComplexProperty ComplexProperty { get; } = entityType;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexPropertyRemoved(TypeBaseBuilder, ComplexProperty);
    }

    private sealed class OnComplexPropertyNullabilityChangedNode(IConventionComplexPropertyBuilder propertyBuilder) : ConventionNode
    {
        public IConventionComplexPropertyBuilder PropertyBuilder { get; } = propertyBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexPropertyNullabilityChanged(PropertyBuilder);
    }

    private sealed class OnComplexPropertyFieldChangedNode(
        IConventionComplexPropertyBuilder propertyBuilder,
        FieldInfo? newFieldInfo,
        FieldInfo? oldFieldInfo)
        : ConventionNode
    {
        public IConventionComplexPropertyBuilder PropertyBuilder { get; } = propertyBuilder;
        public FieldInfo? NewFieldInfo { get; } = newFieldInfo;
        public FieldInfo? OldFieldInfo { get; } = oldFieldInfo;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexPropertyFieldChanged(PropertyBuilder, NewFieldInfo, OldFieldInfo);
    }

    private sealed class OnComplexPropertyAnnotationChangedNode(
        IConventionComplexPropertyBuilder propertyBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionComplexPropertyBuilder PropertyBuilder { get; } = propertyBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnComplexPropertyAnnotationChanged(
                PropertyBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnForeignKeyAddedNode(IConventionForeignKeyBuilder relationshipBuilder) : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyAdded(RelationshipBuilder);
    }

    private sealed class OnForeignKeyRemovedNode(IConventionEntityTypeBuilder entityTypeBuilder, IConventionForeignKey foreignKey)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionForeignKey ForeignKey { get; } = foreignKey;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyRemoved(EntityTypeBuilder, ForeignKey);
    }

    private sealed class OnForeignKeyAnnotationChangedNode(
        IConventionForeignKeyBuilder relationshipBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyAnnotationChanged(
                RelationshipBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnForeignKeyPropertiesChangedNode(
        IConventionForeignKeyBuilder relationshipBuilder,
        IReadOnlyList<IConventionProperty> oldDependentProperties,
        IConventionKey oldPrincipalKey)
        : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;
        public IReadOnlyList<IConventionProperty> OldDependentProperties { get; } = oldDependentProperties;
        public IConventionKey OldPrincipalKey { get; } = oldPrincipalKey;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyPropertiesChanged(
                RelationshipBuilder, OldDependentProperties, OldPrincipalKey);
    }

    private sealed class OnForeignKeyUniquenessChangedNode(IConventionForeignKeyBuilder relationshipBuilder) : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyUniquenessChanged(RelationshipBuilder);
    }

    private sealed class OnForeignKeyRequirednessChangedNode(IConventionForeignKeyBuilder relationshipBuilder) : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyRequirednessChanged(RelationshipBuilder);
    }

    private sealed class OnForeignKeyDependentRequirednessChangedNode(IConventionForeignKeyBuilder relationshipBuilder) : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyDependentRequirednessChanged(RelationshipBuilder);
    }

    private sealed class OnForeignKeyOwnershipChangedNode(IConventionForeignKeyBuilder relationshipBuilder) : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyOwnershipChanged(RelationshipBuilder);
    }

    private sealed class OnForeignKeyNullNavigationSetNode(IConventionForeignKeyBuilder relationshipBuilder, bool pointsToPrincipal)
        : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;
        public bool PointsToPrincipal { get; } = pointsToPrincipal;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyNullNavigationSet(RelationshipBuilder, PointsToPrincipal);
    }

    private sealed class OnForeignKeyPrincipalEndChangedNode(IConventionForeignKeyBuilder relationshipBuilder) : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnForeignKeyPrincipalEndChanged(RelationshipBuilder);
    }

    private sealed class OnNavigationAddedNode(IConventionNavigationBuilder navigationBuilder) : ConventionNode
    {
        public IConventionNavigationBuilder NavigationBuilder { get; } = navigationBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnNavigationAdded(NavigationBuilder);
    }

    private sealed class OnNavigationAnnotationChangedNode(
        IConventionForeignKeyBuilder relationshipBuilder,
        IConventionNavigation navigation,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionForeignKeyBuilder RelationshipBuilder { get; } = relationshipBuilder;
        public IConventionNavigation Navigation { get; } = navigation;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnNavigationAnnotationChanged(
                RelationshipBuilder, Navigation, Name, Annotation, OldAnnotation);
    }

    private sealed class OnNavigationRemovedNode(
        IConventionEntityTypeBuilder sourceEntityTypeBuilder,
        IConventionEntityTypeBuilder targetEntityTypeBuilder,
        string navigationName,
        MemberInfo? memberInfo)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder SourceEntityTypeBuilder { get; } = sourceEntityTypeBuilder;
        public IConventionEntityTypeBuilder TargetEntityTypeBuilder { get; } = targetEntityTypeBuilder;
        public string NavigationName { get; } = navigationName;
        public MemberInfo? MemberInfo { get; } = memberInfo;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnNavigationRemoved(
                SourceEntityTypeBuilder, TargetEntityTypeBuilder, NavigationName, MemberInfo);
    }

    private sealed class OnSkipNavigationAddedNode(IConventionSkipNavigationBuilder navigationBuilder) : ConventionNode
    {
        public IConventionSkipNavigationBuilder NavigationBuilder { get; } = navigationBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnSkipNavigationAdded(NavigationBuilder);
    }

    private sealed class OnSkipNavigationAnnotationChangedNode(
        IConventionSkipNavigationBuilder navigationBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionSkipNavigationBuilder NavigationBuilder { get; } = navigationBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnSkipNavigationAnnotationChanged(
                NavigationBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnSkipNavigationForeignKeyChangedNode(
        IConventionSkipNavigationBuilder navigationBuilder,
        IConventionForeignKey? foreignKey,
        IConventionForeignKey? oldForeignKey)
        : ConventionNode
    {
        public IConventionSkipNavigationBuilder NavigationBuilder { get; } = navigationBuilder;
        public IConventionForeignKey? ForeignKey { get; } = foreignKey;
        public IConventionForeignKey? OldForeignKey { get; } = oldForeignKey;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnSkipNavigationForeignKeyChanged(NavigationBuilder, ForeignKey, OldForeignKey);
    }

    private sealed class OnSkipNavigationInverseChangedNode(
        IConventionSkipNavigationBuilder navigationBuilder,
        IConventionSkipNavigation? inverse,
        IConventionSkipNavigation? oldInverse)
        : ConventionNode
    {
        public IConventionSkipNavigationBuilder NavigationBuilder { get; } = navigationBuilder;
        public IConventionSkipNavigation? Inverse { get; } = inverse;
        public IConventionSkipNavigation? OldInverse { get; } = oldInverse;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnSkipNavigationInverseChanged(NavigationBuilder, Inverse, OldInverse);
    }

    private sealed class OnSkipNavigationRemovedNode(
        IConventionEntityTypeBuilder entityTypeBuilder,
        IConventionSkipNavigation navigation)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionSkipNavigation Navigation { get; } = navigation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnSkipNavigationRemoved(EntityTypeBuilder, Navigation);
    }

    private sealed class OnTriggerAddedNode(IConventionTriggerBuilder triggerBuilder) : ConventionNode
    {
        public IConventionTriggerBuilder TriggerBuilder { get; } = triggerBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnTriggerAdded(TriggerBuilder);
    }

    private sealed class OnTriggerRemovedNode(
        IConventionEntityTypeBuilder entityTypeBuilder,
        IConventionTrigger trigger)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionTrigger Trigger { get; } = trigger;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnTriggerRemoved(EntityTypeBuilder, Trigger);
    }

    private sealed class OnKeyAddedNode(IConventionKeyBuilder keyBuilder) : ConventionNode
    {
        public IConventionKeyBuilder KeyBuilder { get; } = keyBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnKeyAdded(KeyBuilder);
    }

    private sealed class OnKeyRemovedNode(IConventionEntityTypeBuilder entityTypeBuilder, IConventionKey key) : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionKey Key { get; } = key;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnKeyRemoved(EntityTypeBuilder, Key);
    }

    private sealed class OnKeyAnnotationChangedNode(
        IConventionKeyBuilder keyBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionKeyBuilder KeyBuilder { get; } = keyBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnKeyAnnotationChanged(
                KeyBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnEntityTypePrimaryKeyChangedNode(
        IConventionEntityTypeBuilder entityTypeBuilder,
        IConventionKey? newPrimaryKey,
        IConventionKey? previousPrimaryKey)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionKey? NewPrimaryKey { get; } = newPrimaryKey;
        public IConventionKey? PreviousPrimaryKey { get; } = previousPrimaryKey;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnEntityTypePrimaryKeyChanged(
                EntityTypeBuilder, NewPrimaryKey, PreviousPrimaryKey);
    }

    private sealed class OnIndexAddedNode(IConventionIndexBuilder indexBuilder) : ConventionNode
    {
        public IConventionIndexBuilder IndexBuilder { get; } = indexBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnIndexAdded(IndexBuilder);
    }

    private sealed class OnIndexRemovedNode(IConventionEntityTypeBuilder entityTypeBuilder, IConventionIndex index)
        : ConventionNode
    {
        public IConventionEntityTypeBuilder EntityTypeBuilder { get; } = entityTypeBuilder;
        public IConventionIndex Index { get; } = index;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnIndexRemoved(EntityTypeBuilder, Index);
    }

    private sealed class OnIndexUniquenessChangedNode(IConventionIndexBuilder indexBuilder) : ConventionNode
    {
        public IConventionIndexBuilder IndexBuilder { get; } = indexBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnIndexUniquenessChanged(IndexBuilder);
    }

    private sealed class OnIndexSortOrderChangedNode(IConventionIndexBuilder indexBuilder) : ConventionNode
    {
        public IConventionIndexBuilder IndexBuilder { get; } = indexBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnIndexSortOrderChanged(IndexBuilder);
    }

    private sealed class OnIndexAnnotationChangedNode(
        IConventionIndexBuilder indexBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionIndexBuilder IndexBuilder { get; } = indexBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnIndexAnnotationChanged(
                IndexBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnPropertyAddedNode(IConventionPropertyBuilder propertyBuilder) : ConventionNode
    {
        public IConventionPropertyBuilder PropertyBuilder { get; } = propertyBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnPropertyAdded(PropertyBuilder);
    }

    private sealed class OnPropertyNullabilityChangedNode(IConventionPropertyBuilder propertyBuilder) : ConventionNode
    {
        public IConventionPropertyBuilder PropertyBuilder { get; } = propertyBuilder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnPropertyNullabilityChanged(PropertyBuilder);
    }

    private sealed class OnElementTypeNullabilityChangedNode(IConventionElementTypeBuilder builder) : ConventionNode
    {
        public IConventionElementTypeBuilder ElementTypeBuilder { get; } = builder;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnElementTypeNullabilityChanged(ElementTypeBuilder);
    }

    private sealed class OnPropertyFieldChangedNode(
        IConventionPropertyBuilder propertyBuilder,
        FieldInfo? newFieldInfo,
        FieldInfo? oldFieldInfo)
        : ConventionNode
    {
        public IConventionPropertyBuilder PropertyBuilder { get; } = propertyBuilder;
        public FieldInfo? NewFieldInfo { get; } = newFieldInfo;
        public FieldInfo? OldFieldInfo { get; } = oldFieldInfo;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnPropertyFieldChanged(PropertyBuilder, NewFieldInfo, OldFieldInfo);
    }

    private sealed class OnPropertyElementTypeChangedNode(
        IConventionPropertyBuilder propertyBuilder,
        IElementType? newElementType,
        IElementType? oldElementType)
        : ConventionNode
    {
        public IConventionPropertyBuilder PropertyBuilder { get; } = propertyBuilder;
        public IElementType? NewElementType { get; } = newElementType;
        public IElementType? OldElementType { get; } = oldElementType;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnPropertyElementTypeChanged(PropertyBuilder, NewElementType, OldElementType);
    }

    private sealed class OnPropertyAnnotationChangedNode(
        IConventionPropertyBuilder propertyBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionPropertyBuilder PropertyBuilder { get; } = propertyBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnPropertyAnnotationChanged(
                PropertyBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnElementTypeAnnotationChangedNode(
        IConventionElementTypeBuilder elementTypeBuilder,
        string name,
        IConventionAnnotation? annotation,
        IConventionAnnotation? oldAnnotation)
        : ConventionNode
    {
        public IConventionElementTypeBuilder ElementTypeBuilder { get; } = elementTypeBuilder;
        public string Name { get; } = name;
        public IConventionAnnotation? Annotation { get; } = annotation;
        public IConventionAnnotation? OldAnnotation { get; } = oldAnnotation;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnElementTypeAnnotationChanged(
                ElementTypeBuilder, Name, Annotation, OldAnnotation);
    }

    private sealed class OnPropertyRemovedNode(
        IConventionTypeBaseBuilder typeBaseBuilder,
        IConventionProperty property)
        : ConventionNode
    {
        public IConventionTypeBaseBuilder TypeBaseBuilder { get; } = typeBaseBuilder;
        public IConventionProperty Property { get; } = property;

        public override void Run(ConventionDispatcher dispatcher)
            => dispatcher._immediateConventionScope.OnPropertyRemoved(TypeBaseBuilder, Property);
    }
}
