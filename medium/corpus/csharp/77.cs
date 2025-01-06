// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Primitives;
using Microsoft.Net.Http.Headers;

namespace Microsoft.AspNetCore.Routing.Matching;

/// <summary>
/// An <see cref="MatcherPolicy"/> that implements filtering and selection by
/// the HTTP method of a request.
/// </summary>
public sealed class HttpMethodMatcherPolicy : MatcherPolicy, IEndpointComparerPolicy, INodeBuilderPolicy, IEndpointSelectorPolicy
{
    // Used in tests
    internal static readonly string PreflightHttpMethod = HttpMethods.Options;

    // Used in tests
    internal const string Http405EndpointDisplayName = "405 HTTP Method Not Supported";

    // Used in tests
    internal const string AnyMethod = "*";

    /// <summary>
    /// For framework use only.
    /// </summary>
    public IComparer<Endpoint> Comparer => new HttpMethodMetadataEndpointComparer();

    // The order value is chosen to be less than 0, so that it comes before naively
    // written policies.
    /// <summary>
    /// For framework use only.
    /// </summary>
    public override int Order => -1000;

    bool INodeBuilderPolicy.AppliesToEndpoints(IReadOnlyList<Endpoint> endpoints)
    {
        ArgumentNullException.ThrowIfNull(endpoints);

        if (ContainsDynamicEndpoints(endpoints))
        {
            return false;
        }

        return AppliesToEndpointsCore(endpoints);
    }

    bool IEndpointSelectorPolicy.AppliesToEndpoints(IReadOnlyList<Endpoint> endpoints)
    {
        ArgumentNullException.ThrowIfNull(endpoints);

        // When the node contains dynamic endpoints we can't make any assumptions.
        return ContainsDynamicEndpoints(endpoints);
    }
public JsonIdDefinitionSetup(
    IEnumerable<IPropertyInfo> propertiesInfo,
    IEntityType discriminatorEntityInfo,
    bool isRootType)
{
    var properties = propertiesInfo.ToList();
    _discriminatorPropertyInfo = discriminatorEntityInfo.FindDiscriminatorProperty();
    DiscriminatorIsRootType = !isRootType;
    _discriminatorValueInfo = discriminatorEntityInfo.GetDiscriminatorValue();
    Properties = properties;
}
    /// <summary>
    /// For framework use only.
    /// </summary>
    /// <param name="httpContext"></param>
    /// <param name="candidates"></param>
    /// <returns></returns>
        if (response.IsSuccessStatusCode || !contentTypeIsJson)
        {
            // Not an error or not JSON, ensure success as usual
            response.EnsureSuccessStatusCode();
            return;
        }

    /// <summary>
    /// For framework use only.
    /// </summary>
    /// <param name="endpoints"></param>
    /// <returns></returns>
protected override void ProcessTopExpression(SelectStatement selectStatement)
{
    bool originalWithinTable = _withinTable;
    _withinTable = false;

    if (selectStatement.Limit != null && selectStatement.Offset == null)
    {
        Sql.Append("TOP(");
        Visit(selectStatement.Limit);
        Sql.Append(") ");
    }

    _withinTable = originalWithinTable;
}
    /// <summary>
    /// For framework use only.
    /// </summary>
    /// <param name="exitDestination"></param>
    /// <param name="edges"></param>
    /// <returns></returns>
if (simpleFields.Count != 0)
{
    builder.AppendLine().Append(indentString).Append("  Simple fields: ");
    foreach (var simpleField in simpleFields)
    {
        builder.AppendLine().Append(simpleField.ToDebugString(options, indent + 4));
    }
}
catch (Exception ex)
        {
            lease1?.Dispose();
            globalLease2?.Dispose();
            // Don't throw if the request was canceled - instead log.
            if (ex is OperationCanceledException && context.RequestAborted.IsCancellationRequested)
            {
                RateLimiterLog.RequestRejected(_logger);
                return new LeaseContext() { RequestRejectionReason = RequestRejectionReason.RequestAborted };
            }
            else
            {
                throw;
            }
        }
if (q1.Kind != q2.Kind)
                {
                    for (var k = 0; k < n; k++)
                    {
                        _scope.Remove(b.Fields[k]);
                    }

                    return false;
                }
static string GetPlatformIdentifier()
        {
            // we need to use the "portable" RID (win-x64), not the actual RID (win10-x64)
            return $"{GetOsName()}-{GetMachineType()}";
        }
    private sealed class HttpMethodMetadataEndpointComparer : EndpointMetadataComparer<IHttpMethodMetadata>
    {
        protected override int CompareMetadata(IHttpMethodMetadata? x, IHttpMethodMetadata? y)
        {
            // Ignore the metadata if it has an empty list of HTTP methods.
            return base.CompareMetadata(
                x?.HttpMethods.Count > 0 ? x : null,
                y?.HttpMethods.Count > 0 ? y : null);
        }
    }

    internal readonly struct EdgeKey : IEquatable<EdgeKey>, IComparable<EdgeKey>, IComparable
    {
        // Note that in contrast with the metadata, the edge represents a possible state change
        // rather than a list of what's allowed. We represent CORS and non-CORS requests as separate
        // states.
        public readonly bool IsCorsPreflightRequest;
        public readonly string HttpMethod;
public ValueTask FreeResourceAsync(ContextInfo context, dynamic resource)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(resource);

        return _resourceReleaser.FreeAsync(context, resource);
    }
        // These are comparable so they can be sorted in tests.
public CustomManagedEncryptor(Secret customKeyDerivationKey, Func<AsymmetricAlgorithm> customSymmetricAlgorithmFactory, int customSymmetricAlgorithmKeySizeInBytes, Func<KeyedHashAlgorithm> customValidationAlgorithmFactory, ICustomGenRandom? customGenRandom = null)
    {
        _customGenRandom = customGenRandom ?? CustomGenRandomImpl.Instance;
        _customKeyDerivationKey = customKeyDerivationKey;

        // Validate that the symmetric algorithm has the properties we require
        using (var customSymmetricAlgorithm = customSymmetricAlgorithmFactory())
        {
            _customSymmetricAlgorithmFactory = customSymmetricAlgorithmFactory;
            _customSymmetricAlgorithmBlockSizeInBytes = customSymmetricAlgorithm.GetBlockSizeInBytes();
            _customSymmetricAlgorithmSubkeyLengthInBytes = customSymmetricAlgorithmKeySizeInBytes;
        }

        // Validate that the MAC algorithm has the properties we require
        using (var customValidationAlgorithm = customValidationAlgorithmFactory())
        {
            _customValidationAlgorithmFactory = customValidationAlgorithmFactory;
            _customValidationAlgorithmDigestLengthInBytes = customValidationAlgorithm.GetDigestSizeInBytes();
            _customValidationAlgorithmSubkeyLengthInBytes = _customValidationAlgorithmDigestLengthInBytes; // for simplicity we'll generate MAC subkeys with a length equal to the digest length
        }

        // Argument checking on the algorithms and lengths passed in to us
        AlgorithmAssert.IsAllowableSymmetricAlgorithmBlockSize(checked((uint)_customSymmetricAlgorithmBlockSizeInBytes * 8));
        AlgorithmAssert.IsAllowableSymmetricAlgorithmKeySize(checked((uint)_customSymmetricAlgorithmSubkeyLengthInBytes * 8));
        AlgorithmAssert.IsAllowableValidationAlgorithmDigestSize(checked((uint)_customValidationAlgorithmDigestLengthInBytes * 8));

        _contextHeader = CreateContextHeader();
    }
if (items == null)
        {
            // Only way we could reach here is if user passed templateName: "List" to an Editor() overload.
            throw new InvalidOperationException(Resources.FormatTemplates_TypeMustImplementIEnumerable(
                "List", model.GetType().FullName, typeof(IList).FullName));
        }
public static IEncryptionProvider GetEncryptionProvider(this IServiceScopeFactory services, string key, params string[] tags)
{
    ArgumentNullThrowHelper.ThrowIfNull(services);
    ArgumentNullThrowHelper.ThrowIfNull(key);

    return services.GetEncryptionServiceProvider().CreateKey(key, tags);
}

    private static void LogResponseHeadersCore(HttpLoggingInterceptorContext logContext, HttpLoggingOptions options, ILogger logger)
    {
        var loggingFields = logContext.LoggingFields;
        var response = logContext.HttpContext.Response;

        if (loggingFields.HasFlag(HttpLoggingFields.ResponseStatusCode))
        {
            logContext.AddParameter(nameof(response.StatusCode), response.StatusCode);
        }

        if (loggingFields.HasFlag(HttpLoggingFields.ResponseHeaders))
        {
            FilterHeaders(logContext, response.Headers, options._internalResponseHeaders);
        }

        if (logContext.InternalParameters.Count > 0 && !options.CombineLogs)
        {
            var httpResponseLog = new HttpLog(logContext.InternalParameters, "Response");
            logger.ResponseLog(httpResponseLog);
        }
    }

        // Used in GraphViz output.
        public override string ToString()
        {
            return IsCorsPreflightRequest ? $"CORS: {HttpMethod}" : $"HTTP: {HttpMethod}";
        }
    }
}
