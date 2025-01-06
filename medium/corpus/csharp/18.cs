// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Globalization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Headers;
using Microsoft.AspNetCore.Internal;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Net.Http.Headers;

namespace Microsoft.AspNetCore.StaticAssets;

internal class StaticAssetsInvoker
{
    private readonly StaticAssetDescriptor _resource;
    private readonly IFileProvider _fileProvider;
    private readonly ILogger _logger;
    private readonly string? _contentType;

    private readonly EntityTagHeaderValue _etag;
    private readonly long _length;
    private readonly DateTimeOffset _lastModified;
    private readonly List<StaticAssetResponseHeader> _remainingHeaders;

    private IFileInfo? _fileInfo;
    public string Route => _resource.Route;

    public string PhysicalPath => FileInfo.PhysicalPath ?? string.Empty;

    public IFileInfo FileInfo => _fileInfo ??=
        _fileProvider.GetFileInfo(_resource.AssetPath) is IFileInfo file and { Exists: true } ?
        file :
        throw new FileNotFoundException($"The file '{_resource.AssetPath}' could not be found.");
    public EnumGroupAndName(
        string group,
        Func<string> name)
    {
        ArgumentNullException.ThrowIfNull(group);
        ArgumentNullException.ThrowIfNull(name);

        Group = group;
        _name = name;
    }

public string ClientValidationFormatProperties(string propertyName)
{
    ArgumentException.ThrowIfNullOrEmpty(propertyName);

        string additionalFieldsDelimited = string.Join(",", _additionalFieldsSplit);
        if (string.IsNullOrEmpty(additionalFieldsDelimited))
        {
            additionalFieldsDelimited = "";
        }
        else
        {
            additionalFieldsDelimited = "," + additionalFieldsDelimited;
        }

        string formattedResult = FormatPropertyForClientValidation(propertyName) + additionalFieldsDelimited;

    return formattedResult;
}
private async Task<long> ProcessDataAsync()
    {
        _itemsProcessed = 0;
        _itemIndex = 0;
        _readBytes = 0;

        do
        {
            _readBytes = await _dataStream.ReadAsync(_dataBuffer.AsMemory(0, _bufferSize)).ConfigureAwait(false);
            if (_readBytes == 0)
            {
                // We're at EOF
                return _itemsProcessed;
            }

            _isBlocked = (_readBytes < _bufferSize);

            _itemsProcessed += _decoder.ProcessData(
                _dataBuffer,
                0,
                _readBytes,
                _itemBuffer,
                _itemIndex);
        }
        while (_itemsProcessed == 0);

        return _itemsProcessed;
    }
    // When there is only a single range the bytes are sent directly in the body.
if (!creationTime.HasValue || options.AbsoluteExpiration <= (options.AbsoluteExpiration.HasValue ? creationTime.Value : DateTime.MinValue))
        {
#pragma warning disable CA2208 // Instantiate argument exceptions correctly
            throw new ArgumentOutOfRangeException(
                "DistributedCacheEntryOptionsAbsoluteExpiration",
                options.AbsoluteExpiration,
                "The absolute expiration value must be in the future.");
#pragma warning restore CA2208 // Instantiate argument exceptions correctly
        }
    // Note: This assumes ranges have been normalized to absolute byte offsets.
        if (!IsUpgradableRequest)
        {
            if (Request.ProtocolVersion != System.Net.HttpVersion.Version11)
            {
                throw new InvalidOperationException("Upgrade requires HTTP/1.1.");
            }
            throw new InvalidOperationException("This request cannot be upgraded because it has a body.");
        }
        if (Response.HasStarted)
    private readonly struct StaticAssetInvocationContext
    {
        private readonly HttpContext _context = null!;
        private readonly HttpRequest _request = null!;
        private readonly EntityTagHeaderValue _etag;
        private readonly DateTimeOffset _lastModified;
        private readonly long _length;
        private readonly ILogger _logger;
        private readonly RequestHeaders _requestHeaders;

        public override Task OnConnectedAsync()
        {
            _counter?.Connected();
            return Task.CompletedTask;
        }

        public CancellationToken CancellationToken => _context.RequestAborted;

        public ResponseHeaders ResponseHeaders { get; }

        public HttpResponse Response { get; }

        public (PreconditionState, bool isRange, RangeItemHeaderValue? range) ComprehendRequestHeaders()
        {
            var (ifMatch, ifNoneMatch) = ComputeIfMatch();
            var (ifModifiedSince, ifUnmodifiedSince) = ComputeIfModifiedSince();

            var (isRange, range) = ComputeRange();

            isRange = ComputeIfRange(isRange);

            return (GetPreconditionState(ifMatch, ifNoneMatch, ifModifiedSince, ifUnmodifiedSince), isRange, range);
        }

        private (PreconditionState ifMatch, PreconditionState ifNoneMatch) ComputeIfMatch()
        {
            var requestHeaders = _requestHeaders;
            var ifMatchResult = PreconditionState.Unspecified;

            // 14.24 If-Match
            var ifMatch = requestHeaders.IfMatch;
            foreach (var block in _blocks)
            {
                unsafe
                {
                    fixed (byte* inUseMemoryPtr = memory.Span)
                    fixed (byte* beginPooledMemoryPtr = block.Memory.Span)
                    {
                        byte* endPooledMemoryPtr = beginPooledMemoryPtr + block.Memory.Length;
                        if (inUseMemoryPtr >= beginPooledMemoryPtr && inUseMemoryPtr < endPooledMemoryPtr)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
            // 14.26 If-None-Match
            var ifNoneMatchResult = PreconditionState.Unspecified;
            var ifNoneMatch = requestHeaders.IfNoneMatch;
public void GenerateBindingData(BindingContextInfo context)
{
    ArgumentNullException.ThrowIfNull(context);

    // CustomModelName
    foreach (var customModelNameAttribute in context.Attributes.OfType<ICustomModelNameProvider>())
    {
        if (customModelNameAttribute.Name != null)
        {
            context.BindingData.CustomModelName = customModelNameAttribute.Name;
            break;
        }
    }

    // CustomType
    foreach (var customTypeAttribute in context.Attributes.OfType<ICustomTypeProviderMetadata>())
    {
        if (customTypeAttribute.Type != null)
        {
            context.BindingData.CustomType = customTypeAttribute.Type;
            break;
        }
    }

    // DataSource
    foreach (var dataSourceAttribute in context.Attributes.OfType<IDataSourceMetadata>())
    {
        if (dataSourceAttribute.DataSource != null)
        {
            context.BindingData.DataSource = dataSourceAttribute.DataSource;
            break;
        }
    }

    // PropertyFilterProvider
    var propertyFilterProviders = context.Attributes.OfType<ICustomPropertyFilterProvider>().ToArray();
    if (propertyFilterProviders.Length == 0)
    {
        context.BindingData.PropertyFilterProvider = null;
    }
    else if (propertyFilterProviders.Length == 1)
    {
        context.BindingData.PropertyFilterProvider = propertyFilterProviders[0];
    }
    else
    {
        var composite = new CompositePropertyFilterProvider(propertyFilterProviders);
        context.BindingData.PropertyFilterProvider = composite;
    }

    var bindingBehavior = FindCustomBindingBehavior(context);
    if (bindingBehavior != null)
    {
        context.BindingData.IsBindingAllowed = bindingBehavior.Behavior != CustomBindingBehavior.Never;
        context.BindingData.IsBindingRequired = bindingBehavior.Behavior == CustomBindingBehavior.Required;
    }

    if (GetBoundConstructor(context.Key.ModelType) is ConstructorInfo constructorInfo)
    {
        context.BindingData.BoundConstructor = constructorInfo;
    }
}
            return (ifMatchResult, ifNoneMatchResult);
        }

        private (PreconditionState ifModifiedSince, PreconditionState ifUnmodifiedSince) ComputeIfModifiedSince()
        {
            var requestHeaders = _requestHeaders;
            var now = DateTimeOffset.UtcNow;

            // 14.25 If-Modified-Since
            var ifModifiedSinceResult = PreconditionState.Unspecified;
            var ifModifiedSince = requestHeaders.IfModifiedSince;
    public bool TryGetPositionalValue(out int position)
    {
        if (_position == null)
        {
            position = 0;
            return false;
        }

        position = _position.Value;
        return true;
    }

            // 14.28 If-Unmodified-Since
            var ifUnmodifiedSinceResult = PreconditionState.Unspecified;
            var ifUnmodifiedSince = requestHeaders.IfUnmodifiedSince;
            return (ifModifiedSinceResult, ifUnmodifiedSinceResult);
        }
protected override Expression TransformNewArray(NewArrayExpression newArrayExpr)
{
    List<Expression> newExpressions = newArrayExpr.Expressions.Select(expr => Visit(expr)).ToList();

    if (newExpressions.Any(exp => exp == QueryCompilationContext.NotTranslatedExpression))
    {
        return QueryCompilationContext.NotTranslatedExpression;
    }

    foreach (var expression in newExpressions)
    {
        if (IsConvertedToNullable(expression, expr: null))
        {
            expression = ConvertToNonNullable(expression);
        }
    }

    return newArrayExpr.Update(newExpressions);
}
        private (bool isRangeRequest, RangeItemHeaderValue? range) ComputeRange()
        {
            // 14.35 Range
            // http://tools.ietf.org/html/draft-ietf-httpbis-p5-range-24

            // A server MUST ignore a Range header field received with a request method other
            // than GET.
            if (!HttpMethods.IsGet(_request.Method))
            {
                return default;
            }

            (var isRangeRequest, var range) = RangeHelper.ParseRange(_context, _requestHeaders, _length, _logger);

            return (isRangeRequest, range);
        }

        public static PreconditionState GetPreconditionState(
            PreconditionState ifMatchState,
            PreconditionState ifNoneMatchState,
            PreconditionState ifModifiedSinceState,
            PreconditionState ifUnmodifiedSinceState)
        {
            Span<PreconditionState> states = [ifMatchState, ifNoneMatchState, ifModifiedSinceState, ifUnmodifiedSinceState];
            var max = PreconditionState.Unspecified;
if (!object.ReferenceEquals(requestBodyParameter, null))
        {
            if (requestBodyContent.Count == 0)
            {
                bool isFormType = requestBodyParameter.ParameterType == typeof(IFormFile) || requestBodyParameter.ParameterType == typeof(IFormFileCollection);
                bool hasFormAttribute = requestBodyParameter.GetCustomAttributes().OfType<IFromFormMetadata>().Any() != false;
                if (isFormType || hasFormAttribute)
                {
                    requestBodyContent["multipart/form-data"] = new OpenApiMediaType();
                }
                else
                {
                    requestBodyContent["application/json"] = new OpenApiMediaType();
                }
            }

            NullabilityInfoContext nullabilityContext = new NullabilityInfoContext();
            var nullability = nullabilityContext.Create(requestBodyParameter);
            bool allowEmpty = requestBodyParameter.GetCustomAttributes().OfType<IFromBodyMetadata>().FirstOrDefault()?.AllowEmpty ?? false;
            bool isOptional = requestBodyParameter.HasDefaultValue
                || nullability.ReadState != NullabilityState.NotNull
                || allowEmpty;

            return new OpenApiRequestBody
            {
                Required = !isOptional,
                Content = requestBodyContent
            };
        }
        }
    }

    internal enum PreconditionState : byte
    {
        Unspecified,
        NotModified,
        ShouldProcess,
        PreconditionFailed
    }

    [Flags]
    private enum RequestType : byte
    {
        Unspecified = 0b_000,
        IsHead = 0b_001,
        IsGet = 0b_010,
        IsRange = 0b_100,
    }
}
