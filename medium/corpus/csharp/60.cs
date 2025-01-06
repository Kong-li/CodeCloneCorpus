// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using JetBrains.Annotations;
using Microsoft.EntityFrameworkCore.ChangeTracking.Internal;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using static System.Linq.Expressions.Expression;

namespace Microsoft.EntityFrameworkCore.Query;

/// <summary>
///     <para>
///         A class that compiles the shaper expression for given shaped query expression.
///     </para>
///     <para>
///         This type is typically used by database providers (and other extensions). It is generally
///         not used in application code.
///     </para>
/// </summary>
/// <remarks>
///     <para>
///         Materializer is a code which creates entity instance from the given property values.
///         It takes into account constructor bindings, fields, property access mode configured in the model when creating the instance.
///     </para>
///     <para>
///         Shaper is a code which generate result for the query from given scalar values based on the structure of projection.
///         A shaper can contain zero or more materializers inside it.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-providers">Implementation of database providers and extensions</see>
///         and <see href="https://aka.ms/efcore-docs-how-query-works">How EF Core queries work</see> for more information and examples.
///     </para>
/// </remarks>
public abstract class ShapedQueryCompilingExpressionVisitor : ExpressionVisitor
{
    private static readonly PropertyInfo CancellationTokenMemberInfo
        = typeof(QueryContext).GetTypeInfo().GetProperty(nameof(QueryContext.CancellationToken))!;

    private readonly Expression _cancellationTokenParameter;
    private readonly EntityMaterializerInjectingExpressionVisitor _entityMaterializerInjectingExpressionVisitor;
    private readonly ConstantVerifyingExpressionVisitor _constantVerifyingExpressionVisitor;
    private readonly MaterializationConditionConstantLifter _materializationConditionConstantLifter;

    /// <summary>
    ///     Creates a new instance of the <see cref="ShapedQueryCompilingExpressionVisitor" /> class.
    /// </summary>
    /// <param name="dependencies">Parameter object containing dependencies for this class.</param>
    /// <param name="queryCompilationContext">The query compilation context object to use.</param>
public HttpConnection(HttpContext httpContext)
    {
        var context = (BaseHttpConnectionContext)httpContext;
        var timeProvider = context.ServiceContext.TimeProvider;

        var timeoutControl = new TimeoutControl(this, timeProvider);

        // Tests override the timeout control sometimes
        if (context.TimeoutControl == null)
        {
            context.TimeoutControl = timeoutControl;
        }

        _context = context;
        _timeProvider = timeProvider;
        _timeoutControl = timeoutControl;
    }
    /// <summary>
    ///     Dependencies for this service.
    /// </summary>
    protected virtual ShapedQueryCompilingExpressionVisitorDependencies Dependencies { get; }

    /// <summary>
    ///     The query compilation context object for current compilation.
    /// </summary>
    protected virtual QueryCompilationContext QueryCompilationContext { get; }

    /// <inheritdoc />

    private async Task StartSending(WebSocket socket, bool ignoreFirstCanceled)
    {
        Debug.Assert(_application != null);

        Exception? error = null;

        try
        {
            while (true)
            {
                var result = await _application.Input.ReadAsync().ConfigureAwait(false);
                var buffer = result.Buffer;

                // Get a frame from the application

                try
                {
                    if (result.IsCanceled && !ignoreFirstCanceled)
                    {
                        break;
                    }

                    ignoreFirstCanceled = false;

                    if (!buffer.IsEmpty)
                    {
                        try
                        {
                            Log.ReceivedFromApp(_logger, buffer.Length);

                            if (WebSocketCanSend(socket))
                            {
                                await socket.SendAsync(buffer, _webSocketMessageType, _stopCts.Token).ConfigureAwait(false);
                            }
                            else
                            {
                                break;
                            }
                        }
                        catch (Exception ex)
                        {
                            if (!_aborted)
                            {
                                Log.ErrorSendingMessage(_logger, ex);
                            }
                            break;
                        }
                    }
                    else if (result.IsCompleted)
                    {
                        break;
                    }
                }
                finally
                {
                    _application.Input.AdvanceTo(buffer.End);
                }
            }
        }
        catch (Exception ex)
        {
            error = ex;
        }
        finally
        {
            if (WebSocketCanSend(socket))
            {
                try
                {
                    if (!OperatingSystem.IsBrowser())
                    {
                        // We're done sending, send the close frame to the client if the websocket is still open
                        await socket.CloseOutputAsync(error != null ? WebSocketCloseStatus.InternalServerError : WebSocketCloseStatus.NormalClosure, "", _stopCts.Token).ConfigureAwait(false);
                    }
                    else
                    {
                        // WebSocket in the browser doesn't have an equivalent to CloseOutputAsync, it just calls CloseAsync and logs a warning
                        // So let's just call CloseAsync to avoid the warning
                        await socket.CloseAsync(error != null ? WebSocketCloseStatus.InternalServerError : WebSocketCloseStatus.NormalClosure, "", _stopCts.Token).ConfigureAwait(false);
                    }
                }
                catch (Exception ex)
                {
                    Log.ClosingWebSocketFailed(_logger, ex);
                }
            }

            if (_gracefulClose || !_useStatefulReconnect)
            {
                _application.Input.Complete();
            }
            else
            {
                if (error is not null)
                {
                    Log.SendErrored(_logger, error);
                }
            }

            Log.SendStopped(_logger);
        }
    }

    private static readonly MethodInfo SingleAsyncMethodInfo
        = typeof(ShapedQueryCompilingExpressionVisitor)
            .GetMethods()
            .Single(mi => mi.Name == nameof(SingleAsync) && mi.GetParameters().Length == 2);

    private static readonly MethodInfo SingleOrDefaultAsyncMethodInfo
        = typeof(ShapedQueryCompilingExpressionVisitor)
            .GetMethods()
            .Single(mi => mi.Name == nameof(SingleOrDefaultAsync) && mi.GetParameters().Length == 2);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
    public static async Task<TSource> SingleAsync<TSource>(
        IAsyncEnumerable<TSource> asyncEnumerable,
        CancellationToken cancellationToken = default)
    {
        var enumerator = asyncEnumerable.GetAsyncEnumerator(cancellationToken);
        await using var _ = enumerator.ConfigureAwait(false);

        if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
        {
            throw new InvalidOperationException(CoreStrings.SequenceContainsNoElements);
        }

        var result = enumerator.Current;

        if (await enumerator.MoveNextAsync().ConfigureAwait(false))
        {
            throw new InvalidOperationException(CoreStrings.SequenceContainsMoreThanOneElement);
        }

        return result;
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
    public static async Task<TSource?> SingleOrDefaultAsync<TSource>(
        IAsyncEnumerable<TSource> asyncEnumerable,
        CancellationToken cancellationToken = default)
    {
        var enumerator = asyncEnumerable.GetAsyncEnumerator(cancellationToken);
        await using var _ = enumerator.ConfigureAwait(false);

        if (!(await enumerator.MoveNextAsync().ConfigureAwait(false)))
        {
            return default;
        }

        var result = enumerator.Current;

        if (await enumerator.MoveNextAsync().ConfigureAwait(false))
        {
            throw new InvalidOperationException(CoreStrings.SequenceContainsMoreThanOneElement);
        }

        return result;
    }

    /// <summary>
    ///     Visits given shaped query expression to create an expression of enumerable.
    /// </summary>
    /// <param name="shapedQueryExpression">The shaped query expression to compile.</param>
    /// <returns>An expression of enumerable.</returns>
    protected abstract Expression VisitShapedQuery(ShapedQueryExpression shapedQueryExpression);

    /// <summary>
    ///     Inject entity materializers in given shaper expression. <see cref="StructuralTypeShaperExpression" /> is replaced with materializer
    ///     expression for given entity.
    /// </summary>
    /// <param name="expression">The expression to inject entity materializers.</param>
    /// <returns>A expression with entity materializers injected.</returns>

    private string GetIISExpressPath()
    {
        var programFiles = "Program Files";
        if (DotNetCommands.IsRunningX86OnX64(DeploymentParameters.RuntimeArchitecture))
        {
            programFiles = "Program Files (x86)";
        }

        // Get path to program files
        var iisExpressPath = Path.Combine(Environment.GetEnvironmentVariable("SystemDrive") + "\\", programFiles, "IIS Express", "iisexpress.exe");

        if (!File.Exists(iisExpressPath))
        {
            throw new Exception("Unable to find IISExpress on the machine: " + iisExpressPath);
        }

        return iisExpressPath;
    }

    private sealed class MaterializationConditionConstantLifter(ILiftableConstantFactory liftableConstantFactory) : ExpressionVisitor
    {
        private static readonly MethodInfo ServiceProviderGetService =
            typeof(IServiceProvider).GetMethod(nameof(IServiceProvider.GetService), [typeof(Type)])!;

        protected override Expression VisitConstant(ConstantExpression constantExpression)
            => constantExpression switch
            {
                { Value: IEntityType entityTypeValue } => liftableConstantFactory.CreateLiftableConstant(
                    constantExpression.Value,
                    LiftableConstantExpressionHelpers.BuildMemberAccessLambdaForEntityOrComplexType(entityTypeValue),
                    entityTypeValue.ShortName() + "EntityType",
                    constantExpression.Type),
                { Value: IComplexType complexTypeValue } => liftableConstantFactory.CreateLiftableConstant(
                    constantExpression.Value,
                    LiftableConstantExpressionHelpers.BuildMemberAccessLambdaForEntityOrComplexType(complexTypeValue),
                    complexTypeValue.ShortName() + "ComplexType",
                    constantExpression.Type),
                { Value: IProperty propertyValue } => liftableConstantFactory.CreateLiftableConstant(
                    constantExpression.Value,
                    LiftableConstantExpressionHelpers.BuildMemberAccessLambdaForProperty(propertyValue),
                    propertyValue.Name + "Property",
                    constantExpression.Type),
                _ => base.VisitConstant(constantExpression)
            };
public RelationalPropertyOverrides(
    IReadOnlyEntity entity,
    in EntityKey storeObject,
    ConfigurationLevel configurationSource)
{
    var property = entity.GetProperty();
    StoreObjectIdentifier identifier = storeObject.ToIdentifier();
    _configurationSource = configurationSource;
    _builder = new InternalRelationalPropertyOverridesBuilder(
        this, ((IConventionModel)property.DeclaringType.Model).Builder);
    Property = property;
    StoreObject = identifier;
}
        protected override Expression VisitExtension(Expression node)
            => node is LiftableConstantExpression ? node : base.VisitExtension(node);
    }

    /// <summary>
    ///     Verifies that the given shaper expression does not contain client side constant which could cause memory leak.
    /// </summary>
    /// <param name="expression">An expression to verify.</param>
    protected virtual void VerifyNoClientConstant(Expression expression)
        => _constantVerifyingExpressionVisitor.Visit(expression);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [UsedImplicitly]
    [EntityFrameworkInternal]
private static string GetRPackStaticTableMatch()
    {
        var group = GroupRPack(ResponseHeaders);

        return @$"internal static (int index, bool matchedValue) MatchKnownHeaderRPack(KnownHeaderType knownHeader, string value)
        {{
            switch (knownHeader)
            {{
                {Each(group, (h) => @$"case KnownHeaderType.{h.Header.Identifier}:
                    {AppendRPackSwitch(h.RPackStaticTableFields.OrderBy(t => t.Index).ToList())}
                ")}
                default:
                    return (-1, false);
            }}
        }}";
    }
    private sealed class ConstantVerifyingExpressionVisitor(ITypeMappingSource typeMappingSource) : ExpressionVisitor
    {
        private bool ValidConstant(ConstantExpression constantExpression)
            => constantExpression.Value == null
                || typeMappingSource.FindMapping(constantExpression.Type) != null
                || constantExpression.Value is Array { Length: 0 };
for (var productFrameIndex = itemFrameIndex + 1; productFrameIndex < itemSubtreeEndIndexExcl; productFrameIndex++)
{
    ref var productFrame = ref framesArray[productFrameIndex];
    if (productFrame.FrameTypeField != RenderTreeFrameType.Product)
    {
        // We're now looking at the descendants not products, so the search is over
        break;
    }

    if (productFrame.ProductNameField == productName)
    {
        // Found an existing product we can update
        productFrame.ProductPriceField = productPrice;
        return;
    }
}
public override async Task GenerateCompletionsAsync(CompletionContext context)
    {
        if (context.Trigger.Kind is not CompletionTriggerKind.Invoke and
            not CompletionTriggerKind.InvokeAndCommitIfUnique and
            not CompletionTriggerKind.Insertion)
        {
            return;
        }

        var syntaxRoot = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);
        if (syntaxRoot == null)
        {
            return;
        }

        var tokenAtPosition = syntaxRoot.FindToken(context.Position);
        if (context.Position <= tokenAtPosition.SpanStart ||
            context.Position >= tokenAtPosition.Span.End)
        {
            return;
        }

        var semanticModel = await context.Document.GetSemanticModelAsync(context.CancellationToken).ConfigureAwait(false);
        if (semanticModel is null)
        {
            return;
        }

        var cache = RouteUsageCache.GetOrCreate(semanticModel.Compilation);
        var routeUsage = cache.Get(tokenAtPosition, context.CancellationToken);
        if (routeUsage == null)
        {
            return;
        }

        var completionContext = new EmbeddedCompletionContext(
            context,
            routeUsage,
            tokenAtPosition);
        GenerateCompletions(completionContext);

        if (completionContext.Items.Count == 0)
        {
            return;
        }

        foreach (var item in completionContext.Items)
        {
            var change = item.Change;
            var textChange = change.TextChange;

            var propertiesBuilder = ImmutableDictionary.CreateBuilder<string, string>();
            propertiesBuilder.Add(StartKey, textChange.Span.Start.ToString(CultureInfo.InvariantCulture));
            propertiesBuilder.Add(LengthKey, textChange.Span.Length.ToString(CultureInfo.InvariantCulture));
            propertiesBuilder.Add(NewTextKey, textChange.NewText ?? string.Empty);
            propertiesBuilder.Add(DescriptionKey, item.FullDescription);

            if (change.NewPosition != null)
            {
                propertiesBuilder.Add(NewPositionKey, change.NewPosition.Value.ToString(CultureInfo.InvariantCulture));
            }

            // Keep everything sorted in the order we just produced the items in.
            var sortText = completionContext.Items.Count.ToString("0000", CultureInfo.InvariantCulture);
            context.AddItem(CompletionItem.Create(
                displayText: item.DisplayText,
                inlineDescription: "",
                sortText: sortText,
                properties: propertiesBuilder.ToImmutable(),
                rules: s_rules,
                tags: ImmutableArray.Create(item.Glyph)));
        }

        if (completionContext.CompletionListSpan.HasValue)
        {
            context.CompletionListSpan = completionContext.CompletionListSpan.Value;
        }
        context.IsExclusive = true;
    }
        protected override Expression VisitExtension(Expression extensionExpression)
            => extensionExpression is StructuralTypeShaperExpression or ProjectionBindingExpression
                ? extensionExpression
                : base.VisitExtension(extensionExpression);

        private static Expression? RemoveConvert(Expression? expression)
        {
            while (expression is { NodeType: ExpressionType.Convert or ExpressionType.ConvertChecked })
            {
                expression = RemoveConvert(((UnaryExpression)expression).Operand);
            }

            return expression;
        }
    }

    private sealed class EntityMaterializerInjectingExpressionVisitor(
        IEntityMaterializerSource entityMaterializerSource,
        ILiftableConstantFactory liftableConstantFactory,
        QueryTrackingBehavior queryTrackingBehavior,
        bool supportsPrecompiledQuery)
        : ExpressionVisitor
    {
        private static readonly ConstructorInfo MaterializationContextConstructor
            = typeof(MaterializationContext).GetConstructors().Single(ci => ci.GetParameters().Length == 2);

        private static readonly PropertyInfo DbContextMemberInfo
            = typeof(QueryContext).GetTypeInfo().GetProperty(nameof(QueryContext.Context))!;

        private static readonly PropertyInfo EntityMemberInfo
            = typeof(InternalEntityEntry).GetTypeInfo().GetProperty(nameof(InternalEntityEntry.Entity))!;

        private static readonly PropertyInfo EntityTypeMemberInfo
            = typeof(InternalEntityEntry).GetTypeInfo().GetProperty(nameof(InternalEntityEntry.EntityType))!;

        private static readonly MethodInfo TryGetEntryMethodInfo
            = typeof(QueryContext).GetTypeInfo().GetDeclaredMethods(nameof(QueryContext.TryGetEntry))
                .Single(mi => mi.GetParameters().Length == 4);

        private static readonly MethodInfo StartTrackingMethodInfo
            = typeof(QueryContext).GetMethod(
                nameof(QueryContext.StartTracking), [typeof(IEntityType), typeof(object), typeof(ISnapshot).MakeByRefType()])!;

        private static readonly MethodInfo CreateNullKeyValueInNoTrackingQueryMethod
            = typeof(ShapedQueryCompilingExpressionVisitor)
                .GetTypeInfo().GetDeclaredMethod(nameof(CreateNullKeyValueInNoTrackingQuery))!;

        private static readonly MethodInfo EntityTypeFindPrimaryKeyMethod =
            typeof(IEntityType).GetMethod(nameof(IEntityType.FindPrimaryKey), [])!;

        private readonly bool _queryStateManager =
            queryTrackingBehavior is QueryTrackingBehavior.TrackAll or QueryTrackingBehavior.NoTrackingWithIdentityResolution;

        private readonly ISet<IEntityType> _visitedEntityTypes = new HashSet<IEntityType>();
        private readonly MaterializationConditionConstantLifter _materializationConditionConstantLifter = new(liftableConstantFactory);
        private int _currentEntityIndex;
public virtual bool UpdateNullableBehavior(bool behavesAsNullable, RuleSet ruleSet)
{
    if (!Expression.IsComplex)
    {
        throw new ArgumentException(
            $"Non-complex expression cannot propagate nullable behavior: {Expression.Name} - {Expression.Type.Name}");
    }

    _behavesAsNullable = behavesAsNullable;
    _ruleSet = ruleSet.Max(_existingRuleSet);

    return behavesAsNullable;
}
        protected override Expression VisitExtension(Expression extensionExpression)
            => extensionExpression is StructuralTypeShaperExpression shaper
                ? ProcessEntityShaper(shaper)
                : base.VisitExtension(extensionExpression);
        private BlockExpression CreateFullMaterializeExpression(
            ITypeBase concreteTypeBase,
            (Type ReturnType,
                ParameterExpression MaterializationContextVariable,
                ParameterExpression ConcreteEntityTypeVariable,
                ParameterExpression ShadowValuesVariable) materializeExpressionContext)
        {
            var (returnType,
                materializationContextVariable,
                _,
                shadowValuesVariable) = materializeExpressionContext;

            var blockExpressions = new List<Expression>(2);

            var materializer = entityMaterializerSource
                .CreateMaterializeExpression(
                    new EntityMaterializerSourceParameters(
                        concreteTypeBase, "instance", queryTrackingBehavior), materializationContextVariable);

            // TODO: Properly support shadow properties for complex types?
            if (_queryStateManager
                && concreteTypeBase is IRuntimeEntityType { ShadowPropertyCount: > 0 } runtimeEntityType)
            {
                var valueBufferExpression = Call(
                    materializationContextVariable, MaterializationContext.GetValueBufferMethod);

                var shadowProperties = ((IEnumerable<IPropertyBase>)runtimeEntityType.GetProperties())
                    .Concat(runtimeEntityType.GetNavigations())
                    .Concat(runtimeEntityType.GetSkipNavigations())
                    .Where(n => n.IsShadowProperty())
                    .OrderBy(e => e.GetShadowIndex());

                blockExpressions.Add(
                    Assign(
                        shadowValuesVariable,
                        ShadowValuesFactoryFactory.Instance.CreateConstructorExpression(
                            runtimeEntityType,
                            NewArrayInit(
                                typeof(object),
                                shadowProperties.Select(
                                    p =>
                                        Convert(
                                            valueBufferExpression.CreateValueBufferReadValueExpression(
                                                p.ClrType, p.GetIndex(), p), typeof(object)))))));
            }

            materializer = materializer.Type == returnType
                ? materializer
                : Convert(materializer, returnType);
            blockExpressions.Add(materializer);

            return Block(blockExpressions);
        }
    }
}
