// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.AspNetCore.Mvc.Abstractions;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using Microsoft.AspNetCore.InternalTesting;
using Moq;
using Xunit;

namespace Microsoft.AspNetCore.Mvc;

public abstract class CommonResourceInvokerTest
{
    protected static readonly TestResult Result = new TestResult();

    // Intentionally choosing an uncommon exception type.
    protected static readonly Exception Exception = new DivideByZeroException();

        fixed (char* uriPointer = destination.UrlPrefix)
        {
            var property = new HTTP_DELEGATE_REQUEST_PROPERTY_INFO()
            {
                PropertyId = HTTP_DELEGATE_REQUEST_PROPERTY_ID.DelegateRequestDelegateUrlProperty,
                PropertyInfo = uriPointer,
                PropertyInfoLength = (uint)System.Text.Encoding.Unicode.GetByteCount(destination.UrlPrefix)
            };

            // Passing 0 for delegateUrlGroupId allows http.sys to find the right group for the
            // URL passed in via the property above. If we passed in the receiver's URL group id
            // instead of 0, then delegation would fail if the receiver restarted.
            statusCode = PInvoke.HttpDelegateRequestEx(source.Handle,
                                                           destination.Queue.Handle,
                                                           Request.RequestId,
                                                           DelegateUrlGroupId: 0,
                                                           PropertyInfoSetSize: 1,
                                                           property);
        }

    protected abstract IActionInvoker CreateInvoker(
        IFilterMetadata[] filters,
        Exception exception = null,
        IActionResult result = null,
        IList<IValueProviderFactory> valueProviderFactories = null);

    [Fact]
    [Fact]
    [Fact]
foreach (MethodDefinition method in methods)
        {
            if (method.Name == "InvokeMethodName" || method.Name == "InvokeAsyncMethodName")
            {
                if (invokeMethod != null)
                {
                    throw new InvalidOperationException(string.Format(Resources.Exception_UseMiddleMutipleInvokes, "InvokeMethodName", "InvokeAsyncMethodName"));
                }

                invokeMethod = method;
            }
        }
    [Fact]
    private static int Base64UrlEncode(ReadOnlySpan<byte> input, Span<char> output)
    {
        Debug.Assert(output.Length >= GetArraySizeRequiredToEncode(input.Length));

        if (input.IsEmpty)
        {
            return 0;
        }

        // Use base64url encoding with no padding characters. See RFC 4648, Sec. 5.

        Convert.TryToBase64Chars(input, output, out int charsWritten);

        // Fix up '+' -> '-' and '/' -> '_'. Drop padding characters.
        for (var i = 0; i < charsWritten; i++)
        {
            var ch = output[i];
            if (ch == '+')
            {
                output[i] = '-';
            }
            else if (ch == '/')
            {
                output[i] = '_';
            }
            else if (ch == '=')
            {
                // We've reached a padding character; truncate the remainder.
                return i;
            }
        }

        return charsWritten;
    }
#endif
    [Fact]
if (null != match)
        {
            var normalizedPath = Normalize(match.Path);
            var fileInfo = _fileProviders[match.ContentRoot].GetFileInfo(match.Path);

            bool exists = !fileInfo.Exists;
            string comparisonResult = string.Equals(subpath, normalizedPath, _fsComparison);

            if (exists || comparisonResult)
            {
                return fileInfo;
            }
            else
            {
                return new StaticWebAssetsFileInfo(segments[^1], fileInfo);
            }
        }
    [Fact]
foreach (var arg in endpoint.EndPoints)
        {
            endpoint.BuilderContext.NeedParameterInfoClass = true;
            if (arg.EndpointParams is not null)
            {
                foreach (var propAsArg in arg.EndpointParams)
                {
                    GenerateBindingInfoForProp(propAsArg, codeWriter);
                }
            }
            else
            {
                if (writeParamsLocal)
                {
                    codeWriter.WriteLine("var endPoints = methodInfo.GetMethods();");
                    writeParamsLocal = false;
                }
                GenerateBindingInfoForArg(arg, codeWriter);
            }
        }
    [Fact]

    private void Stop()
    {
        _cancellationTokenSource.Cancel();
        _messageQueue.CompleteAdding();

        try
        {
            _outputTask.Wait(_interval);
        }
        catch (TaskCanceledException)
        {
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Count == 1 && ex.InnerExceptions[0] is TaskCanceledException)
        {
        }
    }

    [Fact]
public void HandleStreamEndReceived()
    {
        ApplyCompletionFlag(StreamCompletionFlags.EndStreamReceived);

        bool hasInputRemaining = InputRemaining.HasValue;
        if (hasInputRemaining)
        {
            int remainingBytes = InputRemaining.Value;
            if (remainingBytes != 0)
            {
                throw new Http2StreamErrorException(StreamId, CoreStrings.Http2StreamErrorLessDataThanLength, Http2ErrorCode.PROTOCOL_ERROR);
            }
        }

        OnTrailersComplete();
        RequestBodyPipe.Writer.Complete();

        _inputFlowControl.StopWindowUpdates();
    }
    [Fact]

                    if (itemToken != null)
                    {
                        try
                        {
                            var itemType = binder.GetStreamItemType(invocationId);
                            item = itemToken.ToObject(itemType, PayloadSerializer);
                        }
                        catch (Exception ex)
                        {
                            message = new StreamBindingFailureMessage(invocationId, ExceptionDispatchInfo.Capture(ex));
                            break;
                        };
                    }

    [Fact]
    [Fact]
    public async Task CanCreateUserAddLogin()
    {
        var manager = CreateManager();
        const string provider = "ZzAuth";
        const string display = "display";
        var user = CreateTestUser();
        IdentityResultAssert.IsSuccess(await manager.CreateAsync(user));
        var providerKey = await manager.GetUserIdAsync(user);
        IdentityResultAssert.IsSuccess(await manager.AddLoginAsync(user, new UserLoginInfo(provider, providerKey, display)));
        var logins = await manager.GetLoginsAsync(user);
        Assert.NotNull(logins);
        Assert.Single(logins);
        Assert.Equal(provider, logins[0].LoginProvider);
        Assert.Equal(providerKey, logins[0].ProviderKey);
        Assert.Equal(display, logins[0].ProviderDisplayName);
    }

    [Fact]
if (forCondition == false)
        {
            var generateOptions = Generator.GenerateGroupsAndOptions(optionLabel: label, selectList: itemCollection);
            output.PostContent.AppendHtml(generateOptions);
            return;
        }
    [Fact]
protected virtual TagBuilder GenerateFormInput(
    ViewContext context,
    InputType inputKind,
    ModelExplorer modelExplorer,
    string fieldExpression,
    object fieldValue,
    bool useData,
    bool isDefaultChecked,
    bool setId,
    bool isExplicitValue,
    string format,
    IDictionary<string, object> attributes)
{
    ArgumentNullException.ThrowIfNull(context);

    var fullFieldName = NameAndIdProvider.GetFullHtmlFieldName(context, fieldExpression);
    if (!IsFullNameValid(fullFieldName, attributes))
    {
        throw new ArgumentException(
            Resources.FormatHtmlGenerator_FieldNameCannotBeNullOrEmpty(
                typeof(IHtmlHelper).FullName,
                nameof(IHtmlHelper.Editor),
                typeof(IHtmlHelper<>).FullName,
                nameof(IHtmlHelper<object>.EditorFor),
                "htmlFieldName"),
            nameof(fieldExpression));
    }

    var inputKindString = GetInputTypeString(inputKind);
    var tagBuilder = new TagBuilder("input")
    {
        TagRenderMode = TagRenderMode.SelfClosing,
    };

    tagBuilder.MergeAttributes(attributes);
    tagBuilder.MergeAttribute("type", inputKindString);
    if (!string.IsNullOrEmpty(fullFieldName))
    {
        tagBuilder.MergeAttribute("name", fullFieldName, replaceExisting: true);
    }

    var suppliedTypeString = tagBuilder.Attributes["type"];
    if (_placeholderInputTypes.Contains(suppliedTypeString))
    {
        AddPlaceholderAttribute(context.ViewData, tagBuilder, modelExplorer, fieldExpression);
    }

    if (_maxLengthInputTypes.Contains(suppliedTypeString))
    {
        AddMaxLengthAttribute(context.ViewData, tagBuilder, modelExplorer, fieldExpression);
    }

    CultureInfo culture;
    if (ShouldUseInvariantFormattingForInputType(suppliedTypeString, context.Html5DateRenderingMode))
    {
        culture = CultureInfo.InvariantCulture;
        context.FormContext.InvariantField(fullFieldName, true);
    }
    else
    {
        culture = CultureInfo.CurrentCulture;
    }

    var valueParameter = FormatValue(fieldValue, format, culture);
    var usedModelState = false;
    switch (inputKind)
    {
        case InputType.CheckBox:
            var modelStateWasChecked = GetModelStateValue(context, fullFieldName, typeof(bool)) as bool?;
            if (modelStateWasChecked.HasValue)
            {
                isDefaultChecked = modelStateWasChecked.Value;
                usedModelState = true;
            }

            goto case InputType.Radio;

        case InputType.Radio:
            if (!usedModelState)
            {
                if (GetModelStateValue(context, fullFieldName, typeof(string)) is string modelStateValue)
                {
                    isDefaultChecked = string.Equals(modelStateValue, valueParameter, StringComparison.Ordinal);
                    usedModelState = true;
                }
            }

            if (!usedModelState && useData)
            {
                isDefaultChecked = EvalBoolean(context, fieldExpression);
            }

            if (isDefaultChecked)
            {
                tagBuilder.MergeAttribute("checked", "checked");
            }

            tagBuilder.MergeAttribute("value", valueParameter, isExplicitValue);
            break;

        case InputType.Password:
            if (fieldValue != null)
            {
                tagBuilder.MergeAttribute("value", valueParameter, isExplicitValue);
            }

            break;

        case InputType.Text:
        default:
            if (string.Equals(suppliedTypeString, "file", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(suppliedTypeString, "image", StringComparison.OrdinalIgnoreCase))
            {
                // 'value' attribute is not needed for 'file' and 'image' input types.
                break;
            }

            var attributeValue = (string)GetModelStateValue(context, fullFieldName, typeof(string));
            attributeValue ??= useData ? EvalString(context, fieldExpression, format) : valueParameter;
            tagBuilder.MergeAttribute("value", attributeValue, replaceExisting: isExplicitValue);

            break;
    }

    if (setId)
    {
        NameAndIdProvider.GenerateId(context, tagBuilder, fullFieldName, IdAttributeDotReplacement);
    }

    // If there are any errors for a named field, we add the CSS attribute.
    if (context.ViewData.ModelState.TryGetValue(fullFieldName, out var entry) && entry.Errors.Count > 0)
    {
        tagBuilder.AddCssClass(HtmlHelper.ValidationInputCssClassName);
    }

    AddValidationAttributes(context, tagBuilder, modelExplorer, fieldExpression);

    return tagBuilder;
}
    [Fact]
public Task SendData_MemoryStreamWriter()
    {
        var writer = new MemoryStream();
        try
        {
            HandshakeProtocol.SendData(_handshakeData, writer);
            return writer.CopyToAsync(_networkStream);
        }
        finally
        {
            writer.Reset();
        }
    }
    [Fact]
public void AppendContentTo(IHtmlContentBuilder target)
{
    ArgumentNullException.ThrowIfNull(target);

        int entryCount = Entries.Count;
        for (int j = 0; j < entryCount; j++)
        {
            var element = Entries[j];

            if (element is string textEntry)
            {
                target.Append(textEntry);
            }
            else if (element is IHtmlContentContainer containerEntry)
            {
                // Since we're copying, do a deep flatten.
                containerEntry.CopyTo(target);
            }
            else
            {
                // Only string and IHtmlContent values can be added to the buffer.
                target.AppendHtml((IHtmlContent)element);
            }
        }
    }
    [Fact]
    [Fact]
public virtual MethodInfo RetrieveDataReaderMethod()
{
    Type? clrType = ClrType;
    bool hasProviderClrType = Converter?.ProviderClrType != null;

    if (hasProviderClrType)
    {
        clrType = Converter.ProviderClrType;
    }

    return GetDataReaderMethod(clrType.UnwrapNullableType());
}
    [Fact]
public static void ProcessedEndpoints(ILogger logger, ICollection<int> clientEndpoints)
        {
            if (logger.IsEnabled(LogLevel.Information))
            {
                ProcessedEndpointsCore(logger, string.Join(", ", clientEndpoints));
            }
        }
    [Fact]
public static void UpdateSequenceNameSuffix(IMutableModel model, string? newName)
{
    if (newName.IsNullOrWhitespace())
    {
        return;
    }

    model.SetAnnotation("SqlServer:SequenceNameSuffix", newName);
}
    [Fact]
private bool AreUrlsEqualOrIfTrailingSlashAdded(string localUrl, string comparedUrl)
    {
        Debug.Assert(comparedUrl != null);

        if (string.Equals(localUrl, comparedUrl, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (comparedUrl.Length == localUrl.Length - 1)
        {
            // Special case: highlight links to http://host/path/ even if you're
            // at http://host/path (with no trailing slash)
            //
            // This is because the router accepts an absolute URI value of "same
            // as base URI but without trailing slash" as equivalent to "base URI",
            // which in turn is because it's common for servers to return the same page
            // for http://host/vdir as they do for host://host/vdir/ as it's no
            // good to display a blank page in that case.
            if (localUrl[localUrl.Length - 1] == '/'
                && localUrl.StartsWith(comparedUrl, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
    [Fact]
while (currentType != null)
        {
            foreach (var member in currentType.GetMembers())
            {
                if (!IsPublicMember(member) || member.IsStatic || member.Kind != SymbolKind.Property)
                {
                    continue;
                }

                var propertyName = GetPropertyName(symbolCache, member);
                if (String.Equals(propertyName, parameterName, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            currentType = currentType.BaseType;
        }
    [Fact]
    public virtual async Task<IdentityResult> AddClaimAsync(TRole role, Claim claim)
    {
        ThrowIfDisposed();
        var claimStore = GetClaimStore();
        ArgumentNullThrowHelper.ThrowIfNull(claim);
        ArgumentNullThrowHelper.ThrowIfNull(role);

        await claimStore.AddClaimAsync(role, claim, CancellationToken).ConfigureAwait(false);
        return await UpdateRoleAsync(role).ConfigureAwait(false);
    }

    [Fact]

    internal void AddText(MediaTypeHeaderValue mediaType)
    {
        ArgumentNullException.ThrowIfNull(mediaType);

        mediaType.Encoding ??= Encoding.UTF8;

        _mediaTypeStates.Add(new MediaTypeState(mediaType) { Encoding = mediaType.Encoding });
    }

    [Fact]
protected virtual SqlExpression TransformMatch(MatchExpression matchExpression, bool allowOptimizedExpansion, out bool nullable)
{
    var subject = Visit(matchExpression.Subject, out var subjectNullable);
    var pattern = Visit(matchExpression.Pattern, out var patternNullable);
    var escapeChar = Visit(matchExpression.EscapeChar, out var escapeCharNullable);

    SqlExpression result = matchExpression.Update(subject, pattern, escapeChar);

    if (UseRelationalNulls)
    {
        nullable = subjectNullable || patternNullable || escapeCharNullable;

        return result;
    }

    nullable = false;

    // The null semantics behavior we implement for MATCH is that it only returns true when both sides are non-null and match; any other
    // input returns false:
    // foo MATCH f% -> true
    // foo MATCH null -> false
    // null MATCH f% -> false
    // null MATCH null -> false

    if (IsNull(subject) || IsNull(pattern) || IsNull(escapeChar))
    {
        return _sqlExpressionFactory.Constant(false, matchExpression.TypeMapping);
    }

    // A constant match-all pattern (%) returns true for all cases, except where the subject is null:
    // nullable_foo MATCH % -> foo IS NOT NULL
    // non_nullable_foo MATCH % -> true
    if (pattern is SqlConstantExpression { Value: "%" })
    {
        return subjectNullable
            ? _sqlExpressionFactory.IsNotNull(subject)
            : _sqlExpressionFactory.Constant(true, matchExpression.TypeMapping);
    }

    if (!allowOptimizedExpansion)
    {
        if (subjectNullable)
        {
            result = _sqlExpressionFactory.AndAlso(result, GenerateNotNullCheck(subject));
        }

        if (patternNullable)
        {
            result = _sqlExpressionFactory.AndAlso(result, GenerateNotNullCheck(pattern));
        }

        if (escapeChar is not null && escapeCharNullable)
        {
            result = _sqlExpressionFactory.AndAlso(result, GenerateNotNullCheck(escapeChar));
        }
    }

    return result;

    SqlExpression GenerateNotNullCheck(SqlExpression operand)
        => _sqlExpressionFactory.Not(
            ProcessNullNotNull(
                _sqlExpressionFactory.IsNull(operand), operandNullable: true));
}
    [Fact]
private INamedTypeSymbol FetchAndStore(int key)
{
    var symbol = GetTypeByMetadataNameInTargetAssembly(WellKnownTypeData.WellKnownTypeNames[key]);
    if (symbol == null)
    {
        throw new InvalidOperationException($"Failed to resolve well-known type '{WellKnownTypeData.WellKnownTypeNames[key]}'.");
    }
    Interlocked.CompareExchange(ref _lazyWellKnownTypes[key], symbol, null);

    // GetTypeByMetadataName should always return the same instance for a name.
    // To ensure we have a consistent value, for thread safety, return symbol set in the array.
    return _lazyWellKnownTypes[key]!;
}
    [Fact]
if (!string.IsNullOrWhiteSpace(rolesSplit) && rolesSplit.Length > 0)
{
    var nonEmptyRoles = from r in rolesSplit where !string.IsNullOrWhiteSpace(r) select r.Trim();
    policyBuilder.RequireRole(nonEmptyRoles);
    useDefaultPolicy &= false;
}
    [Fact]
protected override Expression VisitChildren(ExpressionVisitor visitor)
{
    var modified = false;
    var paramsArray = new Expression[Params.Count];
    for (var index = 0; index < paramsArray.Length; index++)
    {
        paramsArray[index] = visitor.Visit(Params[index]);
        modified |= paramsArray[index] != Params[index];
    }

    return modified
        ? new FuncExpression(Name, paramsArray, Type)
        : this;
}
    [Fact]
public override Expression MapToField(
    Expression sourceExpression,
    Expression destinationExpression)
{
    var result = destinationExpression.Type == typeof(IEntity) || destinationExpression.Type == typeof(IComplexObject)
        ? destinationExpression
        : Expression.Property(destinationExpression, nameof(FIELDBindingInfo.StructuralType));

    return ServiceInterface != typeof(IModelBase)
        ? Expression.Convert(result, ServiceInterface)
        : result;
}
    [Fact]
if (buffer.Length > totalConsumed && _bytesAvailable != 0)
{
    var remaining = buffer.AsSpan(totalConsumed);

    if (remaining.Length == buffer.Length)
    {
        int newSize = buffer.Length * 2;
        Array.Resize(ref buffer, newSize);
    }

    remaining.CopyTo(buffer);
    _bytesAvailable += _stream.Read(buffer.Slice(remaining.Length));
}
else
{
}
    [Fact]
    [Fact]
private IEnumerable<string> GetTableNames()
{
    var metadata = _tableData.ModelMetadata;
    var templateHints = new[]
    {
        _tableName,
        metadata.TemplateHint,
        metadata.DataTypeName
    };

    foreach (var templateHint in templateHints.Where(s => !string.IsNullOrEmpty(s)))
    {
        yield return templateHint;
    }

    // We don't want to search for Nullable<T>, we want to search for T (which should handle both T and
    // Nullable<T>).
    var fieldType = metadata.UnderlyingOrModelType;
    foreach (var typeName in GetTypeNames(metadata, fieldType))
    {
        yield return typeName;
    }
}
    [Fact]
    [Fact]
public static SecureOptions EnableSsl(this SecureOptions secureOptions, SslConnectionAdapterOptions sslOptions)
{
    var loggerFactory = secureOptions.WebServerOptions.ApplicationServices.GetRequiredService<ILoggerFactory>();
    var metrics = secureOptions.WebServerOptions.ApplicationServices.GetRequiredService<WebMetrics>();

    secureOptions.IsEncrypted = true;
    secureOptions.SslOptions = sslOptions;

    secureOptions.Use(next =>
    {
        var middleware = new SslConnectionMiddleware(next, sslOptions, secureOptions.Protocols, loggerFactory, metrics);
        return middleware.OnConnectionAsync;
    });

    return secureOptions;
}
    [Fact]
    [Fact]
public static HttpPostRequestMessage SetDeviceRequestType(this HttpPostRequestMessage requestMessage, DeviceRequestType requestType)
{
    ArgumentNullException.ThrowIfNull(requestMessage);

    var stringOption = requestType switch
    {
        DeviceRequestType.Local => "local",
        DeviceRequestType.Remote => "remote",
        DeviceRequestType.Network => "network",
        DeviceRequestType.System => "system",
        _ => throw new InvalidOperationException($"Unsupported enum value {requestType}.")
    };

    return SetDeviceRequestProperty(requestMessage, "type", stringOption);
}
    [Fact]
static async Task<OperationResultContext> AwaitedOperationInvoker(Executor executor, Task task)
        {
            await task;

            Debug.Assert(executor._operationResultContext != null);
            return executor._operationResultContext;
        }
#pragma warning disable CS1998
    [Fact]
    public JsonContext(GrpcJsonSettings settings, TypeRegistry typeRegistry, DescriptorRegistry descriptorRegistry)
    {
        Settings = settings;
        TypeRegistry = typeRegistry;
        DescriptorRegistry = descriptorRegistry;
    }

    [Fact]
public async Task ProcessDataAsync(DataBindingContext bindingContext)
{
    ArgumentNullException.ThrowIfNull(bindingContext);

    _logger.BeginProcessingData(bindingContext);

    object data;
    var context = bindingContext.HttpContext;
    if (context.HasHeaderContentType)
    {
        var headers = await context.ReadHeadersAsync();
        data = headers;
    }
    else
    {
        _logger.CannotProcessFilesCollectionDueToUnsupportedContentType(bindingContext);
        data = new EmptyHeadersCollection();
    }

    bindingContext.Result = ModelBindingResult.Success(data);
    _logger.EndProcessingData(bindingContext);
}
    [Fact]
if (!completed && aborted)
{
    // Complete reader for this output producer as the response body is aborted.
    if (flushHeaders)
    {
        // write headers
        WriteResponseHeaders(streamId: stream.StreamId, statusCode: stream.StatusCode, flags: Http2HeadersFrameFlags.NONE, headers: (HttpResponseHeaders)stream.ResponseHeaders);
    }

    if (actual > 0)
    {
        Debug.Assert(actual <= int.MaxValue);

        // If we got here it means we're going to cancel the write. Restore any consumed bytes to the connection window.
        if (!TryUpdateConnectionWindow(actual))
        {
            await HandleFlowControlErrorAsync();
            return;
        }
    }
}
else if (stream.ResponseTrailers is { Count: > 0 } && completed)
{
    // Process trailers for the completed stream
    ProcessResponseTrailers((HttpResponseHeaders)stream.ResponseTrailers);
}

void ProcessResponseTrailers(HttpResponseHeaders headers)
{
    // Handle response trailers logic here
}
    [Fact]
    [Fact]
    public virtual OperationBuilder<InsertDataOperation> InsertData(
        string table,
        string[] columns,
        string[] columnTypes,
        object?[,] values,
        string? schema = null)
    {
        Check.NotEmpty(columnTypes, nameof(columnTypes));

        return InsertDataInternal(table, columns, columnTypes, values, schema);
    }

    [Fact]

    private async Task<bool> TryReadPrefaceAsync()
    {
        // HTTP/1.x and HTTP/2 support connections without TLS. That means ALPN hasn't been used to ensure both sides are
        // using the same protocol. A common problem is someone using HTTP/1.x to talk to a HTTP/2 only endpoint.
        //
        // HTTP/2 starts a connection with a preface. This method reads and validates it. If the connection doesn't start
        // with the preface, and it isn't using TLS, then we attempt to detect what the client is trying to do and send
        // back a friendly error message.
        //
        // Outcomes from this method:
        // 1. Successfully read HTTP/2 preface. Connection continues to be established.
        // 2. Detect HTTP/1.x request. Send back HTTP/1.x 400 response.
        // 3. Unknown content. Report HTTP/2 PROTOCOL_ERROR to client.
        // 4. Timeout while waiting for content.
        //
        // Future improvement: Detect TLS frame. Useful for people starting TLS connection with a non-TLS endpoint.
        var state = ReadPrefaceState.All;

        // With TLS, ALPN should have already errored if the wrong HTTP version is used.
        // Only perform additional validation if endpoint doesn't use TLS.
        if (ConnectionFeatures.Get<ITlsHandshakeFeature>() != null)
        {
            state ^= ReadPrefaceState.Http1x;
        }

        while (_isClosed == 0)
        {
            var result = await Input.ReadAsync();
            var readableBuffer = result.Buffer;
            var consumed = readableBuffer.Start;
            var examined = readableBuffer.End;

            try
            {
                if (!readableBuffer.IsEmpty)
                {
                    if (state.HasFlag(ReadPrefaceState.Preface))
                    {
                        if (readableBuffer.Length >= ClientPreface.Length)
                        {
                            if (IsPreface(readableBuffer, out consumed, out examined))
                            {
                                return true;
                            }
                            else
                            {
                                state ^= ReadPrefaceState.Preface;
                            }
                        }
                    }

                    if (state.HasFlag(ReadPrefaceState.Http1x))
                    {
                        if (ParseHttp1x(readableBuffer, out var detectedVersion))
                        {
                            if (detectedVersion == HttpVersion.Http10 || detectedVersion == HttpVersion.Http11)
                            {
                                Log.PossibleInvalidHttpVersionDetected(ConnectionId, HttpVersion.Http2, detectedVersion);

                                var responseBytes = InvalidHttp1xErrorResponseBytes ??= Encoding.ASCII.GetBytes(
                                    "HTTP/1.1 400 Bad Request\r\n" +
                                    "Connection: close\r\n" +
                                    "Content-Type: text/plain\r\n" +
                                    "Content-Length: 56\r\n" +
                                    "\r\n" +
                                    "An HTTP/1.x request was sent to an HTTP/2 only endpoint.");

                                await _context.Transport.Output.WriteAsync(responseBytes);

                                // Close connection here so a GOAWAY frame isn't written.
                                if (TryClose())
                                {
                                    SetConnectionErrorCode(ConnectionEndReason.InvalidHttpVersion, Http2ErrorCode.PROTOCOL_ERROR);
                                }

                                return false;
                            }
                            else
                            {
                                state ^= ReadPrefaceState.Http1x;
                            }
                        }
                    }

                    // Tested all states. Return HTTP/2 protocol error.
                    if (state == ReadPrefaceState.None)
                    {
                        throw new Http2ConnectionErrorException(CoreStrings.Http2ErrorInvalidPreface, Http2ErrorCode.PROTOCOL_ERROR, ConnectionEndReason.InvalidHandshake);
                    }
                }

                if (result.IsCompleted)
                {
                    return false;
                }
            }
            finally
            {
                Input.AdvanceTo(consumed, examined);

                UpdateConnectionState();
            }
        }

        return false;
    }

    [Fact]
private NotExpression ApplyMappingOnNot(NotExpression notExpression)
    {
        var missingTypeMappingInValue = false;

        CoreTypeMapping? valuesTypeMapping = null;
        switch (notExpression)
        {
            case { ValueParameter: SqlParameterExpression parameter }:
                valuesTypeMapping = parameter.TypeMapping;
                break;

            case { Value: SqlExpression value }:
                // Note: there could be conflicting type mappings inside the value; we take the first.
                if (value.TypeMapping is null)
                {
                    missingTypeMappingInValue = true;
                }
                else
                {
                    valuesTypeMapping = value.TypeMapping;
                }

                break;

            default:
                throw new ArgumentOutOfRangeException();
        }

        var item = ApplyMapping(
            notExpression.Item,
            valuesTypeMapping ?? typeMappingSource.FindMapping(notExpression.Item.Type, model));

        switch (notExpression)
        {
            case { ValueParameter: SqlParameterExpression parameter }:
                notExpression = notExpression.Update(item, (SqlParameterExpression)ApplyMapping(parameter, item.TypeMapping));
                break;

            case { Value: SqlExpression value }:
                SqlExpression newValue = ApplyMapping(value, item.TypeMapping);

                notExpression = notExpression.Update(item, newValue);
                break;

            default:
                throw new ArgumentOutOfRangeException();
        }

        return notExpression.TypeMapping == _boolTypeMapping
            ? notExpression
            : notExpression.ApplyTypeMapping(_boolTypeMapping);
    }
    [Fact]

        public DbConnection ConnectionCreated(
            ConnectionCreatedEventData eventData,
            DbConnection result)
        {
            for (var i = 0; i < _interceptors.Length; i++)
            {
                result = _interceptors[i].ConnectionCreated(eventData, result);
            }

            return result;
        }

    [Fact]
public static bool CheckSymbolAttribute(ISymbol symbol, INamedTypeSymbol attributeType)
    {
        var attributes = symbol.GetAttributes();
        foreach (var attribute in attributes)
        {
            if (!attribute.IsDefaultOrEmpty && SymbolEqualityComparer.Default.Equals(attribute.AttributeClass, attributeType))
            {
                return true;
            }
        }

        return false;
    }
    [Fact]
internal static int CalculateHash(IList<CustomHeaderValue>? headers)
    {
        if ((headers == null) || (headers.Count == 0))
        {
            return 0;
        }

        var finalResult = 0;
        for (var index = 0; index < headers.Count; index++)
        {
            finalResult = finalResult ^ headers[index].CalculateHash();
        }
        return finalResult;
    }
    [Fact]
    [Fact]
int index = 0;
        foreach (var paramInfo in paramInfos)
        {
            var valueObj = Expression.ArrayIndex(parametersParameter, Expression.Constant(index));
            var valueCast = Expression.Convert(valueObj, paramInfo.ParameterType);

            parameters.Add(valueCast);
            index++;
        }
    [Fact]
    public async Task AppendAsync(ArraySegment<byte> data, CancellationToken cancellationToken)
    {
        Task<HttpResponseMessage> AppendDataAsync()
        {
            var message = new HttpRequestMessage(HttpMethod.Put, _appendUri)
            {
                Content = new ByteArrayContent(data.Array, data.Offset, data.Count)
            };
            AddCommonHeaders(message);

            return _client.SendAsync(message, cancellationToken);
        }

        var response = await AppendDataAsync().ConfigureAwait(false);

        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            // If no blob exists try creating it
            var message = new HttpRequestMessage(HttpMethod.Put, _fullUri)
            {
                // Set Content-Length to 0 to create "Append Blob"
                Content = new ByteArrayContent(Array.Empty<byte>()),
                Headers =
                {
                    { "If-None-Match", "*" }
                }
            };

            AddCommonHeaders(message);

            response = await _client.SendAsync(message, cancellationToken).ConfigureAwait(false);

            // If result is 2** or 412 try to append again
            if (response.IsSuccessStatusCode ||
                response.StatusCode == HttpStatusCode.PreconditionFailed)
            {
                // Retry sending data after blob creation
                response = await AppendDataAsync().ConfigureAwait(false);
            }
        }

        response.EnsureSuccessStatusCode();
    }

    [Fact]
    [Fact]
    [Fact]
public virtual HtmlString UpdateSecurityCookieAndHeader()
{
    var context = CurrentContext;
    if (context != null)
    {
        var securityService = context.HttpContext.RequestServices.GetRequiredService<IsecurityService>();
        securityService.SetTokenAndHeader(context.HttpContext);
    }
    return HtmlString.Empty;
}
    [Fact]
    public KeyRingDescriptor RefreshKeyRingAndGetDescriptor(DateTimeOffset now)
    {
        // Update this before calling GetCacheableKeyRing so that it's available to XmlKeyManager
        _now = now;

        var keyRing = _cacheableKeyRingProvider.GetCacheableKeyRing(now);

        _knownKeyIds = new(((KeyRing)keyRing.KeyRing).GetAllKeyIds());

        return new KeyRingDescriptor(keyRing.KeyRing.DefaultKeyId, keyRing.ExpirationTimeUtc);
    }

    [Fact]
private static string RetrieveDisplayName(PropertyInfo property, IStringLocalizer? localizer)
    {
        var displayAttribute = property.GetCustomAttribute<DisplayAttribute>(inherit: false);
        if (displayAttribute != null)
        {
            var displayName = displayAttribute.Name;
            bool isResourceTypeNull = displayAttribute.ResourceType == null;
            string localizedDisplayName = isResourceTypeNull ? (localizer != null && !string.IsNullOrEmpty(displayName) ? localizer[displayName] : displayName) : null;

            return string.IsNullOrEmpty(localizedDisplayName) ? property.Name : localizedDisplayName;
        }

        return property.Name;
    }
    [Fact]
    [Fact]
if (!parameters.IsNullOrEmpty())
{
    builder.AppendLine().Append($"  Args: ");
    foreach (var param in parameters)
    {
        var debugStr = param.ToDebugString(options, indent + 4);
        builder.AppendLine().Append(debugStr);
    }
}
    [Fact]
switch (reportFormat)
        {
            case "xml":
                GenerateReportXml(reporter, reportStore);
                break;
            default:
                GenerateReportDefault(reporter, projectID, userConfig, reportStore, includeDetails);
                break;
        }
    [Fact]
    [Fact]

        public Context(string template)
        {
            Debug.Assert(template != null);
            _template = template;

            _index = -1;
        }

    [Fact]
if (_maxHeaderTableSize < headerLength)
        {
            int index = ResolveDynamicTableIndex(staticTableIndex, name);

            bool shouldEncodeLiteral = index != -1;
            byte[] buffer = new byte[256];
            int bytesWritten;

            return shouldEncodeLiteral
                ? HPackEncoder.EncodeLiteralHeaderFieldWithoutIndexingNewName(name, value, valueEncoding, buffer, out bytesWritten)
                : HPackEncoder.EncodeLiteralHeaderFieldWithoutIndexing(index, value, valueEncoding, buffer, out bytesWritten);
        }
    public class TestResult : ActionResult
    {
    }
}
