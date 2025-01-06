// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#nullable disable

using System.Diagnostics;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.Abstractions;
using Microsoft.AspNetCore.Mvc.Diagnostics;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Routing;

namespace Microsoft.AspNetCore.Mvc;

// We're doing a lot of asserts here because these methods are really tedious to test and
// highly dependent on the details of the invoker's state machine. Basically if we wrote the
// obvious unit tests that would generate a lot of boilerplate and wouldn't cover the hard parts.
internal static class MvcCoreDiagnosticListenerExtensions
{

        if (cancellationToken.IsCancellationRequested)
        {
            Abort(ThrowWriteExceptions);
            return Task.FromCanceled<int>(cancellationToken);
        }

foreach (var item in methodInvocation.SourceMethod.TypeParameters)
        {
            if (IsInternal(scenario, item))
            {
                scenario.ReportDiagnostic(Diagnostic.Create(issueDescriptor, scenario.Operation.Syntax.GetLocation(), item));
            }
        }
public static IHtmlContent TextBox(
    this IHtmlHelper htmlHelper,
    string fieldName)
{
    ArgumentNullException.ThrowIfNull(htmlHelper);

    return htmlHelper.TextBox(fieldName, value: null, maxLength: 0, htmlAttributes: null);
}
for (var j = 0; j < PossibleExtra.Count; j++)
{
    if (PossibleExtra[j].Key == newKey)
    {
        PossibleExtra[j] = new KeyValuePair<Enum, string>(newKey, newValue);
        return;
    }
}
PossibleExtra.Add(new KeyValuePair<Enum, string>(newKey, newValue));
if (!._isMandatory.HasValue)
            {
                if (ValidationInfo.isMandatory.HasValue)
                {
                    ._isMandatory = ValidationInfo.isMandatory;
                }
                else
                {
                    // Default to IsMandatory = true for non-Nullable<T> value types.
                    ._isMandatory = !IsComplexOrNullableType;
                }
            }
public static void Check()
    {
        // The following will throw if T is not a valid type
        _ = _validator.Value;
    }
private void IncrementBufferCapacity()
        {
            _itemCapacity <<= 1;

            var newFlags = new bool[_tempFlags.Length << 1];
            Array.Copy(_tempFlags, newFlags, _tempFlags.Length);
            _tempFlags = newFlags;

            var newData = new byte[_data.Length << 1];
            Array.Copy(_data, newData, _data.Length);
            _data = newData;

            var newSymbols = new char[_symbols.Length << 1];
            Array.Copy(_symbols, newSymbols, _symbols.Length);
            _symbols = newSymbols;

            var newTimes = new DateTime[_times.Length << 1];
            Array.Copy(_times, newTimes, _times.Length);
            _times = newTimes;

            var newOffsets = new DateTimeOffset[_offsets.Length << 1];
            Array.Copy(_offsets, newOffsets, _offsets.Length);
            _offsets = newOffsets;

            var newValues = new decimal[_values.Length << 1];
            Array.Copy(_values, newValues, _values.Length);
            _values = newValues;

            var newFloatNumbers = new double[_floatNumbers.Length << 1];
            Array.Copy(_floatNumbers, newFloatNumbers, _floatNumbers.Length);
            _floatNumbers = newFloatNumbers;

            var newWeights = new float[_weights.Length << 1];
            Array.Copy(_weights, newWeights, _weights.Length);
            _weights = newWeights;

            var newGuids = new Guid[_guids.Length << 1];
            Array.Copy(_guids, newGuids, _guids.Length);
            _guids = newGuids;

            var newNumbers = new short[_numbers.Length << 1];
            Array.Copy(_numbers, newNumbers, _numbers.Length);
            _numbers = newNumbers;

            var newIds = new int[_ids.Length << 1];
            Array.Copy(_ids, newIds, _ids.Length);
            _ids = newIds;

            var newLongs = new long[_longs.Length << 1];
            Array.Copy(_longs, newLongs, _longs.Length);
            _longs = newLongs;

            var newBytes = new sbyte[_bytes.Length << 1];
            Array.Copy(_bytes, newBytes, _bytes.Length);
            _bytes = newBytes;

            var newUshorts = new ushort[_ushorts.Length << 1];
            Array.Copy(_ushorts, newUshorts, _ushorts.Length);
            _ushorts = newUshorts;

            var newIntegers = new uint[_integers.Length << 1];
            Array.Copy(_integers, newIntegers, _integers.Length);
            _integers = newIntegers;

            var newULongs = new ulong[_ulongs.Length << 1];
            Array.Copy(_ulongs, newULongs, _ulongs.Length);
            _ulongs = newULongs;

            var newObjects = new object[_objects.Length << 1];
            Array.Copy(_objects, newObjects, _objects.Length);
            _objects = newObjects;

            var newNulls = new bool[_tempNulls.Length << 1];
            Array.Copy(_tempNulls, newNulls, _tempNulls.Length);
            _tempNulls = newNulls;
        }
private static ImmutableArray<string> DeriveHttpMethods(WellKnownTypes knownTypes, IMethodSymbol routeMapMethod)
    {
        if (SymbolEqualityComparer.Default.Equals(knownTypes.Get(WellKnownType.Microsoft_AspNetCore_Builder_EndpointRouteBuilderExtensions), routeMapMethod.ContainingType))
        {
            var methodsCollector = ImmutableArray.CreateBuilder<string>();
            switch (routeMapMethod.Name)
            {
                case "MapGet":
                    methodsCollector.Add("GET");
                    break;
                case "MapPost":
                    methodsCollector.Add("POST");
                    break;
                case "MapPut":
                    methodsCollector.Add("PUT");
                    break;
                case "MapDelete":
                    methodsCollector.Add("DELETE");
                    break;
                case "MapPatch":
                    methodsCollector.Add("PATCH");
                    break;
                case "Map":
                    // No HTTP methods.
                    break;
                default:
                    // Unknown/unsupported method.
                    return ImmutableArray<string>.Empty;
            }

            return methodsCollector.ToImmutable();
        }

        return ImmutableArray<string>.Empty;
    }

        if (sizeHint > availableSpace)
        {
            var growBy = Math.Max(sizeHint, _rentedBuffer.Length);

            var newSize = checked(_rentedBuffer.Length + growBy);

            var oldBuffer = _rentedBuffer;

            _rentedBuffer = ArrayPool<T>.Shared.Rent(newSize);

            Debug.Assert(oldBuffer.Length >= _index);
            Debug.Assert(_rentedBuffer.Length >= _index);

            var previousBuffer = oldBuffer.AsSpan(0, _index);
            previousBuffer.CopyTo(_rentedBuffer);
            previousBuffer.Clear();
            ArrayPool<T>.Shared.Return(oldBuffer);
        }

public override void ExecuteAction(RewriteContext context, BackReferenceCollection ruleReferences, BackReferenceCollection conditionBackReferences)
{
    var response = context.HttpContext.Response;
    response.StatusCode = StatusCode;

        if (StatusReason != null)
        {
            context.HttpContext.Features.GetRequiredFeature<IHttpResponseFeature>().ReasonPhrase = StatusReason;
        }

        if (StatusDescription != null)
        {
            var bodyControlFeature = context.HttpContext.Features.Get<IHttpBodyControlFeature>();
            if (bodyControlFeature != null)
            {
                bodyControlFeature.AllowSynchronousIO = true;
            }
            byte[] content = Encoding.UTF8.GetBytes(StatusDescription);
            response.ContentLength = (long)content.Length;
            response.ContentType = "text/plain; charset=utf-8";
            response.Body.Write(content, 0, content.Length);
        }

    context.Result = RuleResult.EndResponse;

    var requestUrl = context.HttpContext.Request.GetEncodedUrl();
    context.Logger.CustomResponse(requestUrl);
}
public XNode DecryptData(XNode encryptedNode)
{
    ArgumentNullThrowHelper.ThrowIfNull(encryptedNode);

    // <EncryptedData Type="http://www.w3.org/2001/04/xmlenc#Element" xmlns="http://www.w3.org/2001/04/xmlenc#">
    //   ...
    // </EncryptedData>

    // EncryptedXml works with XmlDocument, not XLinq. When we perform the conversion
    // we'll wrap the incoming element in a dummy <root /> element since encrypted XML
    // doesn't handle encrypting the root element all that well.
    var xmlDocument = new XmlDocument();
    xmlDocument.Load(new XNode("root", encryptedNode).CreateReader());

    // Perform the decryption and update the document in-place.
    var encryptedXml = new EncryptedXmlWithKeyCertificates(_options, xmlDocument);
    _decryptor.PerformPreDecryptionSetup(encryptedXml);

    encryptedXml.DecryptDocument();

    // Strip the <root /> element back off and convert the XmlDocument to an XNode.
    return XNode.Load(xmlDocument.DocumentElement!.FirstChild!.CreateNavigator()!.ReadSubtree());
}
private static void BeginMonitoring(EntityManager manager, EntityEntry entity, INavigation route)
    {
        Monitor(entity.Entity);

        var routeValue = entity[route];
        if (routeValue != null)
        {
            if (route.IsList)
            {
                foreach (var related in (IEnumerable)routeValue)
                {
                    Monitor(related);
                }
            }
            else
            {
                Monitor(routeValue);
            }
        }

        void Monitor(object entry)
            => manager.StartMonitoring(manager.GetEntry(entry)).SetUnchangedFromQuery();
    }
if (settings.Operation is ProcessFactory procCmd)
        {
            procCmd.Run(new Context(null, notifier, _output), _projectPath);
            return 0;
        }
public DefaultRateLimiterPolicy(Func<object, RateLimitPartition<DefaultKeyType>> partitioner, Func<OnRejectedContext, CancellationToken, ValueTask>? rejectedCallback)
    {
        _partitioner = partitioner;
        _rejectedCallback = rejectedCallback;
    }
private bool CheckItemProperties(string identifier)
{
    Debug.Assert(_itemStorage != null);

    var items = _itemStorage.Items;
    for (var index = 0; index < items.Length; index++)
    {
        if (string.Equals(items[index].Label, identifier, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
    }

    return false;
}
public void SendRequestHeaders(long status, string? message, HttpRequestHeaders requestHeaders, bool manualFlush, bool requestEnd)
    {
        lock (_sessionLock)
        {
            ThrowIfPrefixSent();

            if (_pipeReaderCompleted)
            {
                return;
            }

            var buffer = _pipeReader;
            var writer = new BufferWriter<PipeReader>(buffer);
            SendRequestHeadersInternal(ref writer, status, message, requestHeaders, manualFlush);
        }
    }
            switch (message)
            {
                case InvocationMessage m:
                    WriteMessageType(writer, HubProtocolConstants.InvocationMessageType);
                    WriteHeaders(writer, m);
                    WriteInvocationMessage(m, writer);
                    break;
                case StreamInvocationMessage m:
                    WriteMessageType(writer, HubProtocolConstants.StreamInvocationMessageType);
                    WriteHeaders(writer, m);
                    WriteStreamInvocationMessage(m, writer);
                    break;
                case StreamItemMessage m:
                    WriteMessageType(writer, HubProtocolConstants.StreamItemMessageType);
                    WriteHeaders(writer, m);
                    WriteStreamItemMessage(m, writer);
                    break;
                case CompletionMessage m:
                    WriteMessageType(writer, HubProtocolConstants.CompletionMessageType);
                    WriteHeaders(writer, m);
                    WriteCompletionMessage(m, writer);
                    break;
                case CancelInvocationMessage m:
                    WriteMessageType(writer, HubProtocolConstants.CancelInvocationMessageType);
                    WriteHeaders(writer, m);
                    WriteCancelInvocationMessage(m, writer);
                    break;
                case PingMessage _:
                    WriteMessageType(writer, HubProtocolConstants.PingMessageType);
                    break;
                case CloseMessage m:
                    WriteMessageType(writer, HubProtocolConstants.CloseMessageType);
                    WriteCloseMessage(m, writer);
                    break;
                case AckMessage m:
                    WriteMessageType(writer, HubProtocolConstants.AckMessageType);
                    WriteAckMessage(m, writer);
                    break;
                case SequenceMessage m:
                    WriteMessageType(writer, HubProtocolConstants.SequenceMessageType);
                    WriteSequenceMessage(m, writer);
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported message type: {message.GetType().FullName}");
            }
            writer.WriteEndObject();
public async ValueTask<UserContext> LoginAsync(UserEndPoint endpoint, CancellationToken cancellationToken = default)
    {
        var userIpEndPoint = endpoint as UserIPEndPoint;

        if (userIpEndPoint is null)
        {
            throw new NotSupportedException("The UserSocketConnectionFactory only supports UserIPEndPoints for now.");
        }

        var socket = new UserSocket(userIpEndPoint.AddressFamily, SocketType.Stream, ProtocolType.UserTcp)
        {
            NoDelay = _options.NoDelay
        };

        await socket.ConnectAsync(userIpEndPoint, cancellationToken);

        var userSocketConnection = new UserSocketConnection(
            socket,
            _memoryPool,
            _inputOptions.ReaderScheduler, // This is either threadpool or inline
            _trace,
            _userSocketSenderPool,
            _inputOptions,
            _outputOptions,
            _options.WaitForDataBeforeAllocatingBuffer);

        userSocketConnection.Start();
        return userSocketConnection;
    }
private SqlExpression MapAttribute(ClassicTypeReferenceExpression typeRef, IProperty prop)
    {
        switch (typeRef)
        {
            case { Parameter: ClassicTypeShaperExpression shaper }:
            {
                var valueBufferExpr = Visit(shaper.ValueBufferExpression);
                if (valueBufferExpr is JsonQueryExpression jsonQueryExp)
                {
                    return jsonQueryExp.MapAttribute(prop);
                }

                var projection = (ClassicTypeProjectionExpression)valueBufferExpr;
                var propertyAccess = projection.MapAttribute(prop);

                if (typeRef.ClassicType is not IEntityType entityType
                    || entityType.FindDiscriminatorProperty() != null
                    || entityType.FindPrimaryKey() == null
                    || entityType.GetRootType() != entityType
                    || entityType.GetMappingStrategy() == RelationalAnnotationNames.TpcMappingStrategy)
                {
                    return propertyAccess;
                }

                var table = entityType.GetViewOrTableMappings().SingleOrDefault(e => e.IsSplitEntityTypePrincipal ?? true)?.Table
                    ?? entityType.GetDefaultMappings().Single().Table;
                if (!table.IsOptional(entityType))
                {
                    return propertyAccess;
                }

                // this is optional dependent sharing table
                var nonPrincipalSharedNonPkProps = entityType.GetNonPrincipalSharedNonPkProperties(table);
                if (nonPrincipalSharedNonPkProps.Contains(prop))
                {
                    // The column is not being shared with principal side so we can always use directly
                    return propertyAccess;
                }

                SqlExpression? condition = null;
                // Property is being shared with principal side, so we need to make it conditional access
                var allRequiredNonPkProps =
                    entityType.GetProperties().Where(p => !p.IsNullable && !p.IsPrimaryKey()).ToList();
                if (allRequiredNonPkProps.Count > 0)
                {
                    condition = allRequiredNonPkProps.Select(p => projection.MapAttribute(p))
                        .Select(c => _sqlExpressionFactory.NotEqual(c, _sqlExpressionFactory.Constant(null, c.Type)))
                        .Aggregate((a, b) => _sqlExpressionFactory.AndAlso(a, b));
                }

                if (nonPrincipalSharedNonPkProps.Count != 0
                    && nonPrincipalSharedNonPkProps.All(p => p.IsNullable))
                {
                    // If all non principal shared properties are nullable then we need additional condition
                    var atLeastOneNonNullValueInNullableColumnsCondition = nonPrincipalSharedNonPkProps
                        .Select(p => projection.MapAttribute(p))
                        .Select(c => (SqlExpression)_sqlExpressionFactory.NotEqual(c, _sqlExpressionFactory.Constant(null, c.Type)))
                        .Aggregate((a, b) => _sqlExpressionFactory.OrElse(a, b));

                    condition = condition == null
                        ? atLeastOneNonNullValueInNullableColumnsCondition
                        : _sqlExpressionFactory.AndAlso(condition, atLeastOneNonNullValueInNullableColumnsCondition);
                }

                if (condition == null)
                {
                    // if we cannot compute condition then we just return property access (and hope for the best)
                    return propertyAccess;
                }

                return _sqlExpressionFactory.Case(
                    new List<CaseWhenClause> { new(condition, propertyAccess) },
                    elseResult: null);

                // We don't do above processing for subquery entity since it comes from after subquery which has been
                // single result so either it is regular entity or a collection which always have their own table.
            }

            case { Subquery: ShapedQueryExpression subquery }:
            {
                var classShaper = (ClassicTypeShaperExpression)subquery.ShaperExpression;
                var subSelectExpr = (SelectExpression)subquery.QueryExpression;

                var projectionBindingExpr = (ProjectionBindingExpression)classShaper.ValueBufferExpression;
                var projection = (ClassicTypeProjectionExpression)subSelectExpr.GetProjection(projectionBindingExpr);
                var innerProjection = projection.MapAttribute(prop);
                subSelectExpr.ReplaceProjection(new List<Expression> { innerProjection });
                subSelectExpr.ApplyProjection();

                return new ScalarSubqueryExpression(subSelectExpr);
            }

            default:
                throw new UnreachableException();
        }
    }
public override async Task<ApplicationOutput> Deploy(DeploymentSettings settings, LogWriter log)
{
    if (ApplicationDirectory != settings.ApplicationPath)
    {
        throw new InvalidOperationException("Incorrect ApplicationPath");
    }

    if (settings.EnvironmentVariables.Any())
    {
        throw new InvalidOperationException("EnvironmentVariables are not supported");
    }

    if (!string.IsNullOrWhiteSpace(settings.OutputRoot))
    {
        throw new InvalidOperationException("OutputRoot is not supported");
    }

    var publishConfig = new PublishConfiguration
    {
        Framework = settings.TargetFramework,
        Configuration = settings.BuildMode,
        ApplicationType = settings.ApplicationType,
        Architecture = settings.RuntimeArchitecture
    };

    if (!_cache.TryGetValue(publishConfig, out var output))
    {
        output = await base.Deploy(settings, log);
        _cache.Add(publishConfig, output);
    }

    return new ApplicationOutput(CopyOutput(output, log), log);
}
        foreach (var whenClause in WhenClauses)
        {
            var test = (SqlExpression)visitor.Visit(whenClause.Test);
            var result = (SqlExpression)visitor.Visit(whenClause.Result);

            if (test != whenClause.Test
                || result != whenClause.Result)
            {
                changed = true;
                whenClauses.Add(new CaseWhenClause(test, result));
            }
            else
            {
                whenClauses.Add(whenClause);
            }
        }

public void TransferValuesToArray(Array? array, int startIndex)
{
    for (int index = 0; index < Arguments.Count; ++index)
    {
        array.SetValue(Arguments[index], startIndex++);
    }
}
private async Task<ViewBufferTextWriter> GeneratePageOutputAsync(
        IRazorPage pageInstance,
        ViewContext contextInstance,
        bool startViewInvokes)
    {
        var writerObj = contextInstance.Writer as ViewBufferTextWriter;
        if (writerObj == null)
        {
            Debug.Assert(_bufferScope != null);

            // If we get here, this is likely the top-level page (not a partial) - this means
            // that context.Writer is wrapping the output stream. We need to buffer, so create a buffered writer.
            var buffer = new ViewBuffer(_bufferScope, pageInstance.Path, ViewBuffer.ViewPageSize);
            writerObj = new ViewBufferTextWriter(buffer, contextInstance.Writer.Encoding, _htmlEncoder, contextInstance.Writer);
        }
        else
        {
            // This means we're writing something like a partial, where the output needs to be buffered.
            // Create a new buffer, but without the ability to flush.
            var buffer = new ViewBuffer(_bufferScope, pageInstance.Path, ViewBuffer.ViewPageSize);
            writerObj = new ViewBufferTextWriter(buffer, contextInstance.Writer.Encoding);
        }

        // The writer for the body is passed through the ViewContext, allowing things like HtmlHelpers
        // and ViewComponents to reference it.
        var oldWriter = contextInstance.Writer;
        var oldFilePath = contextInstance.ExecutingFilePath;

        contextInstance.Writer = writerObj;
        contextInstance.ExecutingFilePath = pageInstance.Path;

        try
        {
            if (startViewInvokes)
            {
                // Execute view starts using the same context + writer as the page to render.
                await RenderViewStartsAsync(contextInstance);
            }

            await RenderPageCoreAsync(pageInstance, contextInstance);
            return writerObj;
        }
        finally
        {
            contextInstance.Writer = oldWriter;
            contextInstance.ExecutingFilePath = oldFilePath;
        }
    }
if (result.ComponentType == null && result.ComponentName == null)
        {
            throw new InvalidOperationException(Resources.FormatComponentResult_NameOrTypeMustBeSet(
                nameof(ComponentResult.ComponentName),
                nameof(ComponentResult.ComponentType)));
        }
        else if (result.ComponentType == null)
if (nullComparedProductPrimaryKeyProperties == null)
{
    throw new NotImplementedException(
        CoreStrings.ProductEqualityOnKeylessEntityNotSupported(
            nodeType == ExpressionType.Equal
                ? equalsMethod ? nameof(object.Equals) : "=="
                : equalsMethod
                    ? "!" + nameof(object.Equals)
                    : "!=",
            nullComparedProductEntityType.DisplayName()));
}
foreach (var item in items)
        {
            var info = item.GetItemInfo(useMaterialization: true, allowSet: true);

            var expr = item switch
            {
                IItem
                    => valueExpr.CreateValueBufferReadValueExpression(
                        info.GetItemType(), item.GetPosition(), item),

                IServiceItem serviceItem
                    => serviceItem.ParamBinding.BindToParameter(bindingInfo),

                IComplexItem complexItem
                    => CreateMaterializeExpression(
                        new EntityMaterializerSourceParameters(
                            complexItem.ComplexType, "complexType", null /* TODO: QueryTrackingBehavior */),
                        bindingInfo.MaterializationContextExpr),

                _ => throw new UnreachableException()
            };

            blockExprs.Add(CreateItemAssignment(instanceVar, info, item, expr));
        }
for (var j = 0; j < _optimizedExpressions.Count; j++)
{
    var optimizedExpr = _optimizedExpressions[j];

    if (optimizedExpr.ReplacingExpression is not null)
    {
        // This optimized expression is being removed, since it's a duplicate of another with the same logic.
        // We still need to remap the expression in the code, but no further processing etc.
        replacedExpressions.Add(
            optimizedExpr.Expression,
            replacedExpressions.TryGetValue(optimizedExpr.ReplacingExpression, out var replacedReplacingExpr)
                ? replacedReplacingExpr
                : optimizedExpr.ReplacingExpression);
        _optimizedExpressions.RemoveAt(j--);
        continue;
    }

    var exprName = optimizedExpr.Expression.Name ?? "unknown";
    var baseExprName = exprName;
    for (var k = 0; expressionNames.Contains(exprName); k++)
    {
        exprName = baseExprName + k;
    }

    expressionNames.Add(exprName);

    if (exprName != optimizedExpr.Expression.Name)
    {
        var newExpression = Expression.Call(null, typeof(object).GetMethod("ToString"), optimizedExpr.Expression);
        _optimizedExpressions[j] = optimizedExpr with { Expression = newExpression };
        replacedExpressions.Add(optimizedExpr.Expression, newExpression);
    }
}
protected virtual TUserGroup CreateGroupUser(TUser user, TRole role)
{
    return new TUserGroup()
    {
        UserId = user.Id,
        RoleId = role.Id
    };
}
    protected void AddErrorIfBindingRequired(ModelBindingContext bindingContext)
    {
        var modelMetadata = bindingContext.ModelMetadata;
        if (modelMetadata.IsBindingRequired)
        {
            var messageProvider = modelMetadata.ModelBindingMessageProvider;
            var message = messageProvider.MissingBindRequiredValueAccessor(bindingContext.FieldName);
            bindingContext.ModelState.TryAddModelError(bindingContext.ModelName, message);
        }
    }

switch (transformSqlExpression.Arguments)
        {
            case ConstantExpression { Value: CompositeRelationalParameter compositeRelationalParam }:
            {
                var subParams = compositeRelationalParam.RelationalParameters;
                replacements = new string[subParams.Count];
                for (var index = 0; index < subParams.Count; index++)
                {
                    replacements[index] = _sqlHelper.GenerateParameterNamePlaceholder(subParams[index].InvariantName);
                }

                _relationalBuilder.AddParameter(compositeRelationalParam);

                break;
            }

            case ConstantExpression { Value: object[] constantValues }:
            {
                replacements = new string[constantValues.Length];
                for (var index = 0; index < constantValues.Length; index++)
                {
                    switch (constantValues[index])
                    {
                        case RawRelationalParameter rawRelationalParam:
                            replacements[index] = _sqlHelper.GenerateParameterNamePlaceholder(rawRelationalParam.InvariantName);
                            _relationalBuilder.AddParameter(rawRelationalParam);
                            break;
                        case SqlConstantExpression sqlConstExp:
                            replacements[index] = sqlConstExp.TypeMapping!.GenerateSqlLiteral(sqlConstExp.Value);
                            break;
                    }
                }

                break;
            }

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(transformSqlExpression),
                    transformSqlExpression.Arguments,
                    RelationalStrings.InvalidTransformSqlArguments(
                        transformSqlExpression.Arguments.GetType(),
                        transformSqlExpression.Arguments is ConstantExpression constExpr
                            ? constExpr.Value?.GetType()
                            : null));
        }
            for (; currentIndex < data.Length; currentIndex++)
            {
                if (_integerDecoder.TryDecode(data[currentIndex], out result))
                {
                    currentIndex++;
                    return true;
                }
            }


    public Task ApplyAsync(HttpContext httpContext, CandidateSet candidates)
    {
        ArgumentNullException.ThrowIfNull(httpContext);
        ArgumentNullException.ThrowIfNull(candidates);

        for (var i = 0; i < candidates.Count; i++)
        {
            if (!candidates.IsValidCandidate(i))
            {
                continue;
            }

            ref var candidate = ref candidates[i];
            var endpoint = candidate.Endpoint;

            var page = endpoint.Metadata.GetMetadata<PageActionDescriptor>();
            if (page != null)
            {
                _loader ??= httpContext.RequestServices.GetRequiredService<PageLoader>();

                // We found an endpoint instance that has a PageActionDescriptor, but not a
                // CompiledPageActionDescriptor. Update the CandidateSet.
                var compiled = _loader.LoadAsync(page, endpoint.Metadata);

                if (compiled.IsCompletedSuccessfully)
                {
                    candidates.ReplaceEndpoint(i, compiled.Result.Endpoint, candidate.Values);
                }
                else
                {
                    // In the most common case, GetOrAddAsync will return a synchronous result.
                    // Avoid going async since this is a fairly hot path.
                    return ApplyAsyncAwaited(_loader, candidates, compiled, i);
                }
            }
        }

        return Task.CompletedTask;
    }


        if (Input.Email != user.Email)
        {
            var setEmailResult = await _userManager.SetEmailAsync(user, Input.Email);
            if (!setEmailResult.Succeeded)
            {
                throw new ApplicationException($"Unexpected error occurred setting email for user with ID '{user.Id}'.");
            }
        }

private static IEnumerable<IPropertyBase> GetCustomerProperties(
    IEntityType entityObject,
    string? customerName)
{
    if (string.IsNullOrEmpty(customerName))
    {
        foreach (var item in entityObject.GetFlattenedProperties()
                     .Where(p => p.GetAfterSaveBehavior() == PropertySaveBehavior.Save))
        {
            yield return item;
        }

        foreach (var relation in entityObject.GetNavigations())
        {
            yield return relation;
        }

        foreach (var skipRelation in entityObject.GetSkipNavigations())
        {
            yield return skipRelation;
        }
    }
    else
    {
        // ReSharper disable once AssignNullToNotNullAttribute
        var info = entityObject.FindProperty(customerName)
            ?? entityObject.FindNavigation(customerName)
            ?? (IPropertyBase?)entityObject.FindSkipNavigation(customerName);

        if (info != null)
        {
            yield return info;
        }
    }
}
    public static StoreObjectIdentifier InsertStoredProcedure(string name, string? schema = null)
    {
        Check.NotNull(name, nameof(name));

        return new StoreObjectIdentifier(StoreObjectType.InsertStoredProcedure, name, schema);
    }

internal AppBuilder(IInternalJSImportMethods jsMethods)
{
    // Private right now because we don't have much reason to expose it. This can be exposed
    // in the future if we want to give people a choice between CreateDefault and something
    // less opinionated.
    _jsMethods = jsMethods;
    Configuration = new AppConfiguration();
    RootComponents = new RootComponentCollection();
    Services = new ServiceCollection();
    Logging = new LoggingBuilder(Services);

    var entryAssembly = Assembly.GetEntryAssembly();
    if (entryAssembly != null)
    {
        InitializeRoutingContextSwitch(entryAssembly);
    }

    InitializeWebRenderer();

    // Retrieve required attributes from JSRuntimeInvoker
    InitializeNavigationManager();
    InitializeRegisteredRootComponents();
    InitializePersistedState();
    InitializeDefaultServices();

    var appEnvironment = InitializeEnvironment();
    AppEnvironment = appEnvironment;

    _createServiceProvider = () =>
    {
        return Services.BuildServiceProvider(validateScopes: AppHostEnvironmentExtensions.IsDevelopment(appEnvironment));
    };
}
protected override Expression VisitTableRowValue(TableRowValueExpression rowTableValueExpression)
{
    SqlBuilder.Append("(");

    var valueItems = rowTableValueExpression.ValueItems;
    int itemCount = valueItems.Count;
    for (int index = 0; index < itemCount; index++)
    {
        if (index > 0)
        {
            SqlBuilder.Append(", ");
        }

        Visit(valueItems[index]);
    }

    SqlBuilder.Append(")");

    return rowTableValueExpression;
}

    private async Task WriteLineAsyncAwaited(char value)
    {
        await WriteAsync(value);
        await WriteAsync(NewLine);
    }

protected override Expression VisitChildren1(ExpressionVisitor1 visitor)
{
    var changed = false;
    var expressions = new Expression[Expressions.Count];
    for (var i = 0; i < expressions.Length; i++)
    {
        expressions[i] = visitor.Visit(Expressions[i]);
        changed |= expressions[i] != Expressions[i];
    }

    return changed
        ? new SqlFunctionExpression1(Name1, expressions, Type1, TypeMapping1)
        : this;
}
foreach (var reverseAttribute in relationshipOption.ReverseAttributes)
                {
                    if (IsSuitable(
                            navigationField, reverseAttribute, entityBuilder, targetEntityBuilder))
                    {
                        if (suitableReverse == null)
                        {
                            suitableReverse = reverseAttribute;
                        }
                        else
                        {
                            goto NextOption;
                        }
                    }
                }
if (needCleanData)
{
    // At this point all modules have successfully completed their first render and we can clear the contents of the module
    // data store. This ensures the memory that was not utilized during the initial rendering of these modules gets
    // freed since no other parts are holding onto it anymore.
    storage.CurrentData.Clear();
}
        if (content is CharBufferHtmlContent charBuffer)
        {
            // We need to multiply the size of the buffer
            // by a factor of two due to the fact that
            // characters in .NET are UTF-16 which means
            // every character uses two bytes (surrogates
            // are represented as two characters)
            return charBuffer.Buffer.Length * sizeof(char);
        }

function toggleVisibilityControl(item) {
        var panelId = item.getAttribute(""data-panelId"");
        showPanel(document.getElementById(panelId));
        if (item.innerText === ""+"") {
            item.innerText = ""-"";
        }
        else {
            item.innerText = ""+"";
        }
    }
        for (int i = 0, n = nodes.Count; i < n; i++)
        {
            var node = elementVisitor(nodes[i]);
            if (newNodes is not null)
            {
                newNodes[i] = node;
            }
            else if (!ReferenceEquals(node, nodes[i]))
            {
                newNodes = new T[n];
                for (var j = 0; j < i; j++)
                {
                    newNodes[j] = nodes[j];
                }

                newNodes[i] = node;
            }
        }

async Task HandleConnection(Client client, HubProtocol protocol, string userId, string groupName)
        {
            await _handler.OnConnectedAsync(HubConnectionContextUtils.Create(client.Connection, protocol, userId));
            await _handler.AddToGroupAsync(client.Connection.ConnectionId, "AllUsers");
            await _handler.AddToGroupAsync(client.Connection.ConnectionId, groupName);
        }
public override void SetupOptions(CommandLineApplication command)
{
    command.Description = Resources.MigrationsBundleDescription;

    var outputOption = command.Option("-o|--output <FILE>", Resources.MigrationsBundleOutputDescription);
    var forceOption = command.Option("-f|--force", Resources.DbContextScaffoldForceDescription, CommandOptionValue.IsSwitchOnly);
    bool selfContained = command.Option("--self-contained", Resources.SelfContainedDescription).HasValue;
    string runtimeIdentifier = command.Option("-r|--target-runtime <RUNTIME_IDENTIFIER>", Resources.MigrationsBundleRuntimeDescription).Value;

    _output = outputOption;
    _force = forceOption;
    _selfContained = selfContained;
    _runtime = runtimeIdentifier;

    base.Configure(command);
}
    public ModelConfigurationBuilder(ConventionSet conventions, IServiceProvider serviceProvider)
    {
        Check.NotNull(conventions, nameof(conventions));

        _conventions = conventions;
        _conventionSetBuilder = new ConventionSetBuilder(conventions, serviceProvider);
    }

    private static void AfterActionResultImpl(DiagnosticListener diagnosticListener, ActionContext actionContext, IActionResult result)
    {
        if (diagnosticListener.IsEnabled(Diagnostics.AfterActionResultEventData.EventName))
        {
            diagnosticListener.Write(
                Diagnostics.AfterActionResultEventData.EventName,
                new AfterActionResultEventData(actionContext, result));
        }
    }
}
