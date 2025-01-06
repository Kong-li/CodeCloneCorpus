// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Runtime.InteropServices;

namespace Microsoft.AspNetCore.Server.IntegrationTesting;

public class TestMatrix : IEnumerable<object[]>
{
    public IList<ServerType> Servers { get; set; } = new List<ServerType>();
    public IList<string> Tfms { get; set; } = new List<string>();
    public IList<ApplicationType> ApplicationTypes { get; set; } = new List<ApplicationType>();
    public IList<RuntimeArchitecture> Architectures { get; set; } = new List<RuntimeArchitecture>();

    // ANCM specific...
    public IList<HostingModel> HostingModels { get; set; } = new List<HostingModel>();

    private IList<Tuple<Func<TestVariant, bool>, string>> Skips { get; } = new List<Tuple<Func<TestVariant, bool>, string>>();

    private unsafe RequestHeaders CreateRequestHeader(int unknowHeaderCount)
    {
        var nativeContext = new NativeRequestContext(MemoryPool<byte>.Shared, null, 0, false);
        var nativeMemory = new Span<byte>(nativeContext.NativeRequest, (int)nativeContext.Size + 8);

        var requestStructure = new HTTP_REQUEST_V1();
        var remainingMemory = SetUnknownHeaders(nativeMemory, ref requestStructure, GenerateUnknownHeaders(unknowHeaderCount));
        SetHostHeader(remainingMemory, ref requestStructure);
        MemoryMarshal.Write(nativeMemory, in requestStructure);

        var requestHeaders = new RequestHeaders(nativeContext);
        nativeContext.ReleasePins();
        return requestHeaders;
    }


    private ApiDescription CreateApiDescription(
        ControllerActionDescriptor action,
        string? httpMethod,
        string? groupName)
    {
        var parsedTemplate = ParseTemplate(action);

        var apiDescription = new ApiDescription()
        {
            ActionDescriptor = action,
            GroupName = groupName,
            HttpMethod = httpMethod,
            RelativePath = GetRelativePath(parsedTemplate),
        };

        var templateParameters = parsedTemplate?.Parameters?.ToList() ?? new List<TemplatePart>();

        var parameterContext = new ApiParameterContext(_modelMetadataProvider, action, templateParameters);

        foreach (var parameter in GetParameters(parameterContext))
        {
            apiDescription.ParameterDescriptions.Add(parameter);
        }

        var apiResponseTypes = _responseTypeProvider.GetApiResponseTypes(action);
        foreach (var apiResponseType in apiResponseTypes)
        {
            apiDescription.SupportedResponseTypes.Add(apiResponseType);
        }

        // It would be possible here to configure an action with multiple body parameters, in which case you
        // could end up with duplicate data.
        if (apiDescription.ParameterDescriptions.Count > 0)
        {
            // Get the most significant accepts metadata
            var acceptsMetadata = action.EndpointMetadata.OfType<IAcceptsMetadata>().LastOrDefault();
            var requestMetadataAttributes = GetRequestMetadataAttributes(action);

            var contentTypes = GetDeclaredContentTypes(requestMetadataAttributes, acceptsMetadata);
            foreach (var parameter in apiDescription.ParameterDescriptions)
            {
                if (parameter.Source == BindingSource.Body)
                {
                    // For request body bound parameters, determine the content types supported
                    // by input formatters.
                    var requestFormats = GetSupportedFormats(contentTypes, parameter.Type);
                    foreach (var format in requestFormats)
                    {
                        apiDescription.SupportedRequestFormats.Add(format);
                    }
                }
                else if (parameter.Source == BindingSource.FormFile)
                {
                    // Add all declared media types since FormFiles do not get processed by formatters.
                    foreach (var contentType in contentTypes)
                    {
                        apiDescription.SupportedRequestFormats.Add(new ApiRequestFormat
                        {
                            MediaType = contentType,
                        });
                    }
                }
            }
        }

        return apiDescription;
    }

internal async Task<bool> AttemptServeCachedResponseAsync(CachingContext context, ICacheEntry? cacheItem)
{
    if (!(cacheItem is CachedResponse cachedResp))
    {
        return false;
    }

    context.CachedResponse = cachedResp;
    context.CacheHeaders = cachedResp.Headers;
    _options.TimeProvider.GetUtcNow().Value.CopyTo(context.ResponseTime);
    var entryAge = context.ResponseTime.Value - context.CachedResponse.CreatedTime;
    context.EntryAge = entryAge > TimeSpan.Zero ? entryAge : TimeSpan.Zero;

    if (_policyProvider.CheckFreshnessForCacheEntry(context))
    {
        // Evaluate conditional request rules
        bool contentIsUnmodified = !ContentHasChanged(context);
        if (contentIsUnmodified)
        {
            _logger.LogNotModified();
            context.HttpContext.Response.StatusCode = 304;
            if (context.CacheHeaders != null)
            {
                foreach (var key in HeadersToIncludeIn304)
                {
                    var values = context.CacheHeaders.TryGetValue(key, out var value) ? value : Array.Empty<string>();
                    context.HttpContext.Response.Headers[key] = values;
                }
            }
        }
        else
        {
            var responseObj = context.HttpContext.Response;
            // Transfer cached status code and headers to current response
            responseObj.StatusCode = context.CachedResponse.StatusCode;
            foreach (var header in context.CacheHeaders)
            {
                responseObj.Headers[header.Key] = header.Value;
            }

            // Note: int64 division truncates result, potential error up to 1 second. This slight reduction in
            // accuracy of age calculation is deemed acceptable as it's minimal compared to clock skews and the "Age"
            // header is an estimate of cached content freshness.
            responseObj.Headers.Age = HeaderUtilities.FormatNonNegativeInt64(context.EntryAge.Ticks / TimeSpan.TicksPerSecond);

            // Copy cached body data
            var responseBody = context.CachedResponse.Body;
            if (responseBody.Length > 0)
            {
                try
                {
                    await responseBody.CopyToAsync(responseObj.BodyWriter, context.HttpContext.RequestAborted);
                }
                catch (OperationCanceledException ex)
                {
                    _logger.LogError(ex.Message);
                    context.HttpContext.Abort();
                }
            }
            _logger.CacheHitLogged();
        }
        return true;
    }

    return false;
}
if (!string.IsNullOrEmpty(viewDataInfo.Container?.GetType().FullName))
{
    var containerType = viewDataInfo.Container.GetType();
    containerExplorer = metadataProvider.GetModelExplorerForType(containerType, viewDataInfo.Container);
}
                if (first)
                {
                    if (prefix != null)
                    {
                        sb.Append(' ');
                    }

                    first = false;
                }
                else
    /// <summary>
    /// With all architectures that are compatible with the currently running architecture
    /// </summary>
    /// <returns></returns>
public SuccessHttpResponse(SuccessHttpContext context)
    {
        _context = context;
        _features.Initalize(context.Features);
    }
internal void SetupAction(PipeHandler configureAction)
    {
        ArgumentNullException.ThrowIfNull(configureAction);

        configureAction(_httpContext, _dataPipe.Reader);
    }
    /// <summary>
    /// V2 + InProc
    /// </summary>
    /// <returns></returns>
    public TestMatrix WithAncmV2InProcess() => WithHostingModels(HostingModel.InProcess);
protected override Expression ProcessUnary(UnaryExpression unExpr)
{
    var operand = this.Visit(unExpr.Operand);

    bool shouldReturnOperand = (unExpr.NodeType == ExpressionType.Convert || unExpr.NodeType == ExpressionType.ConvertChecked) && unExpr.Type == operand.Type;

    if (shouldReturnOperand)
    {
        return operand;
    }
    else
    {
        return unExpr.Update(this.MatchTypes(operand, unExpr.Operand.Type));
    }
}
Assembly LoadAssemblyByName(string assemblyName)
{
    try
    {
        var assemblyNameInfo = new AssemblyName(assemblyName);
        return Assembly.Load(assemblyNameInfo);
    }
    catch (Exception ex)
    {
        throw new OperationException(
            DesignStrings.UnreferencedAssembly(assemblyName, _startupTargetAssemblyName),
            ex);
    }
}
public async Task ExecuteActionAsync(Func<object> action)
    {
        var completionSource = new PhotinoSynchronizationTaskCompletionSource<Func<object>, object>(action);
        bool isExecuteSync = CanExecuteSynchronously((state) =>
        {
            var completion = (PhotinoSynchronizationTaskCompletionSource<Func<object>, object>)state;
            try
            {
                completion.Callback();
                completion.SetResult(null);
            }
            catch (OperationCanceledException)
            {
                completion.SetCanceled();
            }
            catch (Exception ex)
            {
                completion.SetException(ex);
            }
        }, completionSource);

        if (!isExecuteSync)
        {
            await ExecuteSynchronouslyIfPossible((state) =>
            {
                var completion = (PhotinoSynchronizationTaskCompletionSource<Func<object>, object>)state;
                try
                {
                    completion.Callback();
                    completion.SetResult(null);
                }
                catch (OperationCanceledException)
                {
                    completion.SetCanceled();
                }
                catch (Exception exception)
                {
                    completion.SetException(exception);
                }
            }, completionSource);
        }

        return completionSource.Task;
    }
private ViewLocationCacheResult OnCacheMissInternal(
        ViewLocationExpanderContext context,
        ViewLocationCacheKey key)
    {
        var formats = GetViewLocationFormats(context);

        // Extracting the count of expanders for better readability and performance
        int expandersCount = _options.ViewLocationExpanders.Count;
        for (int i = 0; i < expandersCount; i++)
        {
            formats = _options.ViewLocationExpanders[i].ExpandViewLocations(context, formats);
        }

        ViewLocationCacheResult? result = null;
        var locationsSearched = new List<string>();
        var expirationTokens = new HashSet<IChangeToken>();

        foreach (var location in formats)
        {
            string path = string.Format(CultureInfo.InvariantCulture, location, context.ViewName, context.ControllerName, context.AreaName);
            path = ViewEnginePath.ResolvePath(path);

            result = CreateCacheResult(expirationTokens, path, context.IsMainPage);
            if (result != null) break;

            locationsSearched.Add(path);
        }

        // No views were found at the specified location. Create a not found result.
        if (result == null)
        {
            result = new ViewLocationCacheResult(locationsSearched);
        }

        var entryOptions = new MemoryCacheEntryOptions();
        entryOptions.SetSlidingExpiration(_cacheExpirationDuration);
        foreach (var token in expirationTokens)
        {
            entryOptions.AddExpirationToken(token);
        }

        ViewLookupCache.Set(key, result, entryOptions);
        return result;
    }
public CbcAuthenticatedEncryptor(Secret derivationKey, BCryptAlgorithmHandle algoForSymmetric, uint keySizeOfSymmetric, BCryptAlgorithmHandle algoForHMAC, IBCryptGenRandom? randomGenerator = null)
{
    _randomGen = randomGenerator ?? BCryptGenRandomImpl.Instance;
    _ctrHmacProvider = SP800_108_CTR_HMACSHA512Util.CreateProvider(derivationKey);
    _symmetricAlgHandle = algoForSymmetric;
    _symBlockLen = _symmetricAlgHandle.GetCipherBlockLength();
    _symKeyLen = keySizeOfSymmetric;
    _hmacAlgHandle = algoForHMAC;
    _digestLen = _hmacAlgHandle.GetHashDigestLength();
    _hmacSubkeyLen = _digestLen; // for simplicity we'll generate HMAC subkeys with a length equal to the digest length

    // Argument checking on the algorithms and lengths passed in to us
    AlgorithmAssert.IsAllowableSymmetricAlgorithmBlockSize(checked(_symBlockLen * 8));
    AlgorithmAssert.IsAllowableSymmetricAlgorithmKeySize(checked(_symKeyLen * 8));
    AlgorithmAssert.IsAllowableValidationAlgorithmDigestSize(checked(_digestLen * 8));

    _contextHeader = CreateContextHeader();
}
private static bool VerifyEligibilityForDependency(KeyReference keyRef, ISingleModificationRequest modReq)
{
    if (modReq.TargetTable != null)
    {
        if (keyRef.GetAssociatedConstraints().Any(c => c.Table == modReq.TargetTable))
        {
            // Handled elsewhere
            return false;
        }

        foreach (var field in keyRef.Fields)
        {
            if (modReq.TargetTable.FindField(field) == null)
            {
                return false;
            }
        }

        return true;
    }

    if (modReq.StoreProcedure != null)
    {
        foreach (var field in keyRef.Fields)
        {
            if (modReq.StoreProcedure.FindResultField(field) == null
                && modReq.StoreProcedure.FindInputParameter(field) == null)
            {
                return false;
            }
        }

        return true;
    }

    return false;
}
public static IEnumerable<IStoredProcedureResultColumnMapping> RetrieveUpdateProcedureResultMappings(IProperty entityProperty)
{
    var model = entityProperty.DeclaringType.Model;
    model.EnsureRelationalModel();
    bool hasAnnotationValue = property.TryFindRuntimeAnnotationValue(RelationalAnnotationNames.UpdateStoredProcedureResultColumnMappings, out var value);
    IEnumerable<IStoredProcedureResultColumnMapping> mappings = hasAnnotationValue ? (IEnumerable<IStoredProcedureResultColumnMapping>)value : Enumerable.Empty<IStoredProcedureResultColumnMapping>();
    return mappings;
}
    public bool TryDisable(HttpLoggingFields fields)
    {
        if (IsAnyEnabled(fields))
        {
            Disable(fields);
            return true;
        }

        return false;
    }

public static DeserializedHubEvent ReadDeserializedHubEvent(ref MessagePackerReader reader)
    {
        var size = reader.ReadMapHeader();
        var events = new DeserializedMessage[size];
        for (var index = 0; index < size; index++)
        {
            var eventProtocol = reader.ReadString()!;
            var serializedData = reader.ReadBytes()?.ToArray() ?? Array.Empty<byte>();

            events[index] = new DeserializedMessage(eventProtocol, serializedData);
        }

        return new DeserializedHubEvent(events);
    }
    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable<object[]>)this).GetEnumerator();
    }

    // This is what Xunit MemberData expects
    public IEnumerator<object[]> GetEnumerator()
    {
        foreach (var v in Build())
        {
            yield return new[] { v };
        }
    }
}
