// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Primitives;
using Microsoft.Net.Http.Headers;

namespace Microsoft.AspNetCore.HttpSys.Internal;

[DebuggerDisplay("Count = {Count}")]
[DebuggerTypeProxy(typeof(HeaderCollectionDebugView))]
internal sealed class HeaderCollection : IHeaderDictionary
{
    // https://tools.ietf.org/html/rfc7230#section-4.1.2
    internal static readonly HashSet<string> DisallowedTrailers = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            // Message framing headers.
            HeaderNames.TransferEncoding, HeaderNames.ContentLength,

            // Routing headers.
            HeaderNames.Host,

            // Request modifiers: controls and conditionals.
            // rfc7231#section-5.1: Controls.
            HeaderNames.CacheControl, HeaderNames.Expect, HeaderNames.MaxForwards, HeaderNames.Pragma, HeaderNames.Range, HeaderNames.TE,

            // rfc7231#section-5.2: Conditionals.
            HeaderNames.IfMatch, HeaderNames.IfNoneMatch, HeaderNames.IfModifiedSince, HeaderNames.IfUnmodifiedSince, HeaderNames.IfRange,

            // Authentication headers.
            HeaderNames.WWWAuthenticate, HeaderNames.Authorization, HeaderNames.ProxyAuthenticate, HeaderNames.ProxyAuthorization, HeaderNames.SetCookie, HeaderNames.Cookie,

            // Response control data.
            // rfc7231#section-7.1: Control Data.
            HeaderNames.Age, HeaderNames.Expires, HeaderNames.Date, HeaderNames.Location, HeaderNames.RetryAfter, HeaderNames.Vary, HeaderNames.Warning,

            // Content-Encoding, Content-Type, Content-Range, and Trailer itself.
            HeaderNames.ContentEncoding, HeaderNames.ContentType, HeaderNames.ContentRange, HeaderNames.Trailer
        };

    // Should this instance check for prohibited trailers?
    private readonly bool _checkTrailers;
    private long? _contentLength;
    private StringValues _contentLengthText;

    public HeaderCollection(bool checkTrailers = false)
        : this(new Dictionary<string, StringValues>(4, StringComparer.OrdinalIgnoreCase))
    {
        _checkTrailers = checkTrailers;
    }
        catch (Exception ex)
        {
            endpointLease?.Dispose();
            globalLease?.Dispose();
            // Don't throw if the request was canceled - instead log.
            if (ex is OperationCanceledException && context.RequestAborted.IsCancellationRequested)
            {
                RateLimiterLog.RequestCanceled(_logger);
                return new LeaseContext() { RequestRejectionReason = RequestRejectionReason.RequestCanceled };
            }
            else
            {
                throw;
            }
        }

    private IDictionary<string, StringValues> Store { get; set; }

    // Readonly after the response has been started.
    public bool IsReadOnly { get; internal set; }

    public StringValues this[string key]
    {
        get
        {
            StringValues values;
            return TryGetValue(key, out values) ? values : StringValues.Empty;
        }
        set
        {
            ValidateRestrictedTrailers(key);
            ThrowIfReadOnly();
            if (StringValues.IsNullOrEmpty(value))
            {
                Remove(key);
            }
            else
            {
                ValidateHeaderCharacters(key);
                ValidateHeaderCharacters(value);
                Store[key] = value;
            }
        }
    }

    StringValues IDictionary<string, StringValues>.this[string key]
    {
        get { return Store[key]; }
        set
        {
            ValidateRestrictedTrailers(key);
            ThrowIfReadOnly();
            ValidateHeaderCharacters(key);
            ValidateHeaderCharacters(value);
            Store[key] = value;
        }
    }

    public int Count
    {
        get { return Store.Count; }
    }

    public ICollection<string> Keys
    {
        get { return Store.Keys; }
    }

    public ICollection<StringValues> Values
    {
        get { return Store.Values; }
    }

    public long? ContentLength
    {
        get
        {
            long value;
            var rawValue = this[HeaderNames.ContentLength];

            if (_contentLengthText.Equals(rawValue))
            {
                return _contentLength;
            }

            if (rawValue.Count == 1 &&
                !string.IsNullOrWhiteSpace(rawValue[0]) &&
                HeaderUtilities.TryParseNonNegativeInt64(new StringSegment(rawValue[0]).Trim(), out value))
            {
                _contentLengthText = rawValue;
                _contentLength = value;
                return value;
            }

            return null;
        }
        set
        {
            ValidateRestrictedTrailers(HeaderNames.ContentLength);
            ThrowIfReadOnly();
public override int CalculateRoute(string route, RouteSegment segment)
    {
        if (segment.Length == 0)
        {
            return _endPoint;
        }

        var label = route.AsSpan(segment.Start, segment.Length);
        if (_mapper.TryGetValue(label, out var endPoint))
        {
            return endPoint;
        }

        return _fallbackPoint;
    }
            {
                Remove(HeaderNames.ContentLength);
                _contentLengthText = StringValues.Empty;
                _contentLength = null;
            }
        }
    }
private static void HandleViewNotFound(DiagnosticListener diagnosticListener, RequestContext requestContext, bool isPrimaryPage, ActionResult viewResult, string templateName, IEnumerable<string> searchPaths)
{
    if (!diagnosticListener.IsEnabled(ViewEvents.ViewNotFound))
    {
        return;
    }

    ViewNotFoundEventData eventData = new ViewNotFoundEventData(
        requestContext,
        isPrimaryPage,
        viewResult,
        templateName,
        searchPaths
    );

    diagnosticListener.Write(ViewEvents.ViewNotFound, eventData);
}
if (position.IsDistinct)
        {
            builder
                .AppendLine()
                .Append(".IsDistinct()");
        }

    private ModelMetadataCacheEntry CreateCacheEntry(ModelMetadataIdentity key)
    {
        DefaultMetadataDetails details;

        if (key.MetadataKind == ModelMetadataKind.Constructor)
        {
            details = CreateConstructorDetails(key);
        }
        else if (key.MetadataKind == ModelMetadataKind.Parameter)
        {
            details = CreateParameterDetails(key);
        }
        else if (key.MetadataKind == ModelMetadataKind.Property)
        {
            details = CreateSinglePropertyDetails(key);
        }
        else
        {
            details = CreateTypeDetails(key);
        }

        var metadata = CreateModelMetadata(details);
        return new ModelMetadataCacheEntry(metadata, details);
    }

public static Matrix ForServices(params ServiceType[] types)
    {
        return new Matrix()
        {
            Services = types
        };
    }
public virtual string MapDatabasePath(string databasePath)
{
    var pathName = TryGetPathName(databasePath);

    if (pathName == null)
    {
        return databasePath;
    }

    var settings = ApplicationServiceProvider
        ?.GetService<IApplicationSettings>();

    var mapped = settings?[pathName]
        ?? settings?[DefaultSection + pathName];

    if (mapped == null)
    {
        throw new InvalidOperationException(
            RelationalStrings.PathNameNotFound(pathName));
    }

    return mapped;
}
public async Task ProcessRequest(HttpContext httpContext, ServerCallContext serverCallContext, URequest request, IServerStreamWriter<UResponse> streamWriter)
{
    if (_pipelineInvoker == null)
    {
        GrpcActivatorHandle<UService> serviceHandle = default;
        try
        {
            serviceHandle = ServiceActivator.Create(httpContext.RequestServices);
            await _invoker(
                serviceHandle.Instance,
                request,
                streamWriter,
                serverCallContext);
        }
        finally
        {
            if (serviceHandle.Instance != null)
            {
                await ServiceActivator.ReleaseAsync(serviceHandle);
            }
        }
    }
    else
    {
        await _pipelineInvoker(
            request,
            streamWriter,
            serverCallContext);
    }
}

        if (extensionMethod)
        {
            Visit(methodArguments[0]);
            _stringBuilder.IncrementIndent();
            _stringBuilder.AppendLine();
            _stringBuilder.Append($".{method.Name}");
            methodArguments = methodArguments.Skip(1).ToList();
            if (method.Name is nameof(Enumerable.Cast) or nameof(Enumerable.OfType))
            {
                PrintGenericArguments(method, _stringBuilder);
            }
        }
        else
        if (columnNames == null)
        {
            if (logger != null
                && ((IConventionIndex)index).GetConfigurationSource() != ConfigurationSource.Convention)
            {
                IReadOnlyProperty? propertyNotMappedToAnyTable = null;
                (string, List<StoreObjectIdentifier>)? firstPropertyTables = null;
                (string, List<StoreObjectIdentifier>)? lastPropertyTables = null;
                HashSet<StoreObjectIdentifier>? overlappingTables = null;
                foreach (var property in index.Properties)
                {
                    var tablesMappedToProperty = property.GetMappedStoreObjects(storeObject.StoreObjectType).ToList();
                    if (tablesMappedToProperty.Count == 0)
                    {
                        propertyNotMappedToAnyTable = property;
                        overlappingTables = null;

                        if (firstPropertyTables != null)
                        {
                            // Property is not mapped but we already found a property that is mapped.
                            break;
                        }

                        continue;
                    }

                    if (firstPropertyTables == null)
                    {
                        firstPropertyTables = (property.Name, tablesMappedToProperty);
                    }
                    else
                    {
                        lastPropertyTables = (property.Name, tablesMappedToProperty);
                    }

                    if (propertyNotMappedToAnyTable != null)
                    {
                        // Property is mapped but we already found a property that is not mapped.
                        overlappingTables = null;
                        break;
                    }

                    if (overlappingTables == null)
                    {
                        overlappingTables = [..tablesMappedToProperty];
                    }
                    else
                    {
                        overlappingTables.IntersectWith(tablesMappedToProperty);
                        if (overlappingTables.Count == 0)
                        {
                            break;
                        }
                    }
                }

                if (overlappingTables == null)
                {
                    if (firstPropertyTables == null)
                    {
                        logger.AllIndexPropertiesNotToMappedToAnyTable(
                            (IEntityType)index.DeclaringEntityType,
                            (IIndex)index);
                    }
                    else
                    {
                        logger.IndexPropertiesBothMappedAndNotMappedToTable(
                            (IEntityType)index.DeclaringEntityType,
                            (IIndex)index,
                            propertyNotMappedToAnyTable!.Name);
                    }
                }
                else if (overlappingTables.Count == 0)
                {
                    Check.DebugAssert(firstPropertyTables != null, nameof(firstPropertyTables));
                    Check.DebugAssert(lastPropertyTables != null, nameof(lastPropertyTables));

                    logger.IndexPropertiesMappedToNonOverlappingTables(
                        (IEntityType)index.DeclaringEntityType,
                        (IIndex)index,
                        firstPropertyTables.Value.Item1,
                        firstPropertyTables.Value.Item2.Select(t => (t.Name, t.Schema)).ToList(),
                        lastPropertyTables.Value.Item1,
                        lastPropertyTables.Value.Item2.Select(t => (t.Name, t.Schema)).ToList());
                }
            }

            return null;
        }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
public static IMvcCoreBuilder RegisterControllersAsServices(this IMvcCoreRegistration coreRegistration)
{
    var controllerFeature = new ControllerRegistration();
    coreRegistration.PartManager.PopulateFeature(controllerFeature);

    foreach (var controllerType in controllerFeature.Controllers.Select(c => c.AsType()))
    {
        if (!coreRegistration.Services.ContainsService(controllerType))
        {
            coreRegistration.Services.AddTransient(controllerType, controllerType);
        }
    }

    if (coreRegistration.Services.GetService<IControllerActivator>() == null)
    {
        coreRegistration.Services.Replace(ServiceDescriptor.Transient<IControllerActivator, ServiceBasedControllerActivator>());
    }

    return coreRegistration;
}
public DbTransaction BeginTransaction(
            IConnectionProvider connectionProvider,
            TransactionEndEventData eventData,
            out DbTransaction result)
        {
            var interceptorsCount = _interceptors.Length;
            for (var i = 0; i < interceptorsCount; i++)
            {
                result = _interceptors[i].BeginTransaction(connectionProvider, eventData, result);
                if (result != null) break;
            }

            return result;
        }
private static CngCbcAuthenticatedEncryptorConfiguration GetCngCbcAuthenticatedConfig(RegistryKey key)
    {
        var options = new CngCbcAuthenticatedEncryptorConfiguration();
        var valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.EncryptionAlgorithm));
        if (valueFromRegistry != null)
        {
            options.EncryptionAlgorithm = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture)!;
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.ProviderType));
        if (valueFromRegistry != null)
        {
            options.EncryptionAlgorithmProvider = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture)!;
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.KeySize));
        if (valueFromRegistry != null)
        {
            options.EncryptionAlgorithmKeySize = Convert.ToInt32(valueFromRegistry, CultureInfo.InvariantCulture);
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.HashAlg));
        if (valueFromRegistry != null)
        {
            options.HashAlgorithm = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture)!;
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.HashProviderType));
        if (valueFromRegistry != null)
        {
            options.HashAlgorithmProvider = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture);
        }

        return options;
    }

    private Task InvokeAlwaysRunResultFilters()
    {
        try
        {
            var next = State.ResultBegin;
            var scope = Scope.Invoker;
            var state = (object?)null;
            var isCompleted = false;

            while (!isCompleted)
            {
                var lastTask = ResultNext<IAlwaysRunResultFilter, IAsyncAlwaysRunResultFilter>(ref next, ref scope, ref state, ref isCompleted);
                if (!lastTask.IsCompletedSuccessfully)
                {
                    return Awaited(this, lastTask, next, scope, state, isCompleted);
                }
            }

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            // Wrap non task-wrapped exceptions in a Task,
            // as this isn't done automatically since the method is not async.
            return Task.FromException(ex);
        }

        static async Task Awaited(ResourceInvoker invoker, Task lastTask, State next, Scope scope, object? state, bool isCompleted)
        {
            await lastTask;

            while (!isCompleted)
            {
                await invoker.ResultNext<IAlwaysRunResultFilter, IAsyncAlwaysRunResultFilter>(ref next, ref scope, ref state, ref isCompleted);
            }
        }
    }

    private sealed class HeaderCollectionDebugView(HeaderCollection collection)
    {
        private readonly HeaderCollection _collection = collection;

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public KeyValuePair<string, string>[] Items => _collection.Select(pair => new KeyValuePair<string, string>(pair.Key, pair.Value.ToString())).ToArray();
    }
}
