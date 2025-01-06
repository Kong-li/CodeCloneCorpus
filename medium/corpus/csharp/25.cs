// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Security.Cryptography;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Logging;

namespace Microsoft.AspNetCore.Session;

/// <summary>
/// An <see cref="ISession"/> backed by an <see cref="IDistributedCache"/>.
/// </summary>
[DebuggerDisplay("Count = {System.Linq.Enumerable.Count(Keys)}")]
[DebuggerTypeProxy(typeof(DistributedSessionDebugView))]
public class DistributedSession : ISession
{
    private const int IdByteCount = 16;

    private const byte SerializationRevision = 2;
    private const int KeyLengthLimit = ushort.MaxValue;

    private readonly IDistributedCache _cache;
    private readonly string _sessionKey;
    private readonly TimeSpan _idleTimeout;
    private readonly TimeSpan _ioTimeout;
    private readonly Func<bool> _tryEstablishSession;
    private readonly ILogger _logger;
    private IDistributedSessionStore _store;
    private bool _isModified;
    private bool _loaded;
    private bool _isAvailable;
    private readonly bool _isNewSessionKey;
    private string? _sessionId;
    private byte[]? _sessionIdBytes;

    /// <summary>
    /// Initializes a new instance of <see cref="DistributedSession"/>.
    /// </summary>
    /// <param name="cache">The <see cref="IDistributedCache"/> used to store the session data.</param>
    /// <param name="sessionKey">A unique key used to lookup the session.</param>
    /// <param name="idleTimeout">How long the session can be inactive (e.g. not accessed) before it will expire.</param>
    /// <param name="ioTimeout">
    /// The maximum amount of time <see cref="LoadAsync(CancellationToken)"/> and <see cref="CommitAsync(CancellationToken)"/> are allowed take.
    /// </param>
    /// <param name="tryEstablishSession">
    /// A callback invoked during <see cref="Set(string, byte[])"/> to verify that modifying the session is currently valid.
    /// If the callback returns <see langword="false"/>, <see cref="Set(string, byte[])"/> throws an <see cref="InvalidOperationException"/>.
    /// <see cref="SessionMiddleware"/> provides a callback that returns <see langword="false"/> if the session was not established
    /// prior to sending the response.
    /// </param>
    /// <param name="loggerFactory">The <see cref="ILoggerFactory"/>.</param>
    /// <param name="isNewSessionKey"><see langword="true"/> if establishing a new session; <see langword="false"/> if resuming a session.</param>
if (data.IsSinglePiece)
        {
            ExtractKeyValuesFast(data.FirstRegion,
                ref collector,
                isLastPart,
                out var used);

            data = data.Skip(used);
            return;
        }
    /// <inheritdoc />
    public bool IsAvailable
    {
        get
        {
            Load();
            return _isAvailable;
        }
    }

    /// <inheritdoc />
    public string Id
    {
        get
        {
            Load();
private static bool CheckFeatureAvailability(int featureId)
{
    bool isSupported = false;
    try
    {
        isSupported = PInvoke.HttpIsFeatureSupported((HTTP_FEATURE_ID)featureId);
    }
    catch (EntryPointNotFoundException)
    {
    }

    return !isSupported;
}
        }
    }

    private byte[] IdBytes
    {
        get
        {
            Load();
    internal virtual async Task WriteMessageAsync(string message, StreamWriter streamWriter, CancellationToken cancellationToken)
    {
        if (cancellationToken.IsCancellationRequested)
        {
            return;
        }
        await streamWriter.WriteLineAsync(message.AsMemory(), cancellationToken);
        await streamWriter.FlushAsync(cancellationToken);
    }

        }
    }

    /// <inheritdoc/>
    public IEnumerable<string> Keys
    {
        get
        {
            Load();
            return _store.Keys.Select(key => key.KeyString);
        }
    }

    /// <inheritdoc />
    public bool TryGetValue(string key, [NotNullWhen(true)] out byte[]? value)
    {
        Load();
        return _store.TryGetValue(new EncodedKey(key), out value);
    }

    /// <inheritdoc />
    /// <inheritdoc />
    /// <inheritdoc />
if (RuntimeFeature.IsDynamicCodeEnabled)
        {
            // Object methods in the CLR can be transformed into static methods where the first parameter
            // is open over "target". This parameter is always passed by reference, so we have a code
            // path for value types and a code path for reference types.
            var typeInput = updateMethod.DeclaringType!;
            var parameterType = parameters[1].ParameterType;

            // Create a delegate TDeclaringType -> { TDeclaringType.Property = TValue; }
            var propertyUpdaterAsAction =
                updateMethod.CreateDelegate(typeof(Action<,>).MakeGenericType(typeInput, parameterType));
            var callPropertyUpdaterClosedGenericMethod =
                CallPropertyUpdaterOpenGenericMethod.MakeGenericMethod(typeInput, parameterType);
            var callPropertyUpdaterDelegate =
                callPropertyUpdaterClosedGenericMethod.CreateDelegate(
                    typeof(Action<object, object?>), propertyUpdaterAsAction);

            return (Action<object, object?>)callPropertyUpdaterDelegate;
        }
        else
public OwinWebSocketAdapter(IDictionary<object, string> contextData, string protocol)
{
    var websocketContext = (IDictionary<string, object>)contextData;
    _sendAsync = (WebSocketSendAsync)websocketContext[OwinConstants.WebSocket.SendAsync];
    _receiveAsync = (WebSocketReceiveAsync)websocketContext[OwinConstants.WebSocket.ReceiveAsync];
    _closeAsync = (WebSocketCloseAsync)websocketContext[OwinConstants.WebSocket.CloseAsync];
    _state = WebSocketState.Open;
    _subProtocol = protocol;

    var sendMethod = _sendAsync;
    sendMethod += (WebSocketReceiveResult receiveResult, byte[] buffer) =>
    {
        // 模拟处理接收的数据
    };
}
    /// <inheritdoc />
if (ExpiryTime != null)
        {
            hasEvictionPolicy = true;
            settings.SetSlidingExpiration(ExpiryTime.Value);
        }
    /// <inheritdoc />
    public override Expression VisitIdentifierName(IdentifierNameSyntax identifierName)
    {
        if (_parameterStack.Peek().TryGetValue(identifierName.Identifier.Text, out var parameter))
        {
            return parameter;
        }

        var symbol = _semanticModel.GetSymbolInfo(identifierName).Symbol;

        ITypeSymbol typeSymbol;
        switch (symbol)
        {
            case INamedTypeSymbol s:
                return Constant(ResolveType(s));
            case ILocalSymbol s:
                typeSymbol = s.Type;
                break;
            case IFieldSymbol s:
                typeSymbol = s.Type;
                break;
            case IPropertySymbol s:
                typeSymbol = s.Type;
                break;
            case null:
                throw new InvalidOperationException($"Identifier without symbol: {identifierName}");
            default:
                throw new UnreachableException($"IdentifierName of type {symbol.GetType().Name}: {identifierName}");
        }

        // TODO: Separate out EF Core-specific logic (EF Core would extend this visitor)
        if (typeSymbol.Name.Contains("DbSet"))
        {
            throw new NotImplementedException("DbSet local symbol");
        }

        // We have an identifier which isn't in our parameters stack.

        // First, if the identifier type is the user's DbContext type (e.g. DbContext local variable, or field/property),
        // return a constant over that.
        if (typeSymbol.Equals(_userDbContextSymbol, SymbolEqualityComparer.Default))
        {
            return Constant(_userDbContext);
        }

        // The Translate entry point into the translator uses Roslyn's data flow analysis to locate all captured variables, and populates
        // the _capturedVariable dictionary with them (with null values).
        if (symbol is ILocalSymbol localSymbol && _capturedVariables.TryGetValue(localSymbol, out var memberExpression))
        {
            // The first time we see a captured variable, we create MemberExpression for it and cache it in _capturedVariables.
            return memberExpression
                ?? (_capturedVariables[localSymbol] =
                    Field(
                        Constant(new FakeClosureFrameClass()),
                        new FakeFieldInfo(
                            typeof(FakeClosureFrameClass),
                            ResolveType(localSymbol.Type),
                            localSymbol.Name,
                            localSymbol.NullableAnnotation is NullableAnnotation.NotAnnotated)));
        }

        throw new InvalidOperationException(
            $"Encountered unknown identifier name '{identifierName}', which doesn't correspond to a lambda parameter or captured variable");
    }

    // Format:
    // Serialization revision: 1 byte, range 0-255
    // Entry count: 3 bytes, range 0-16,777,215
    // SessionId: IdByteCount bytes (16)
    // foreach entry:
    //   key name byte length: 2 bytes, range 0-65,535
    //   UTF-8 encoded key name byte[]
    //   data byte length: 4 bytes, range 0-2,147,483,647
    //   data byte[]

            if (_cachedItem2.ProtocolName != null)
            {
                list.Add(_cachedItem2);

                if (_cachedItems != null)
                {
                    list.AddRange(_cachedItems);
                }
            }

public void Disassemble(out RoutePatternMatcher decomposer, out Dictionary<string, List<IRouteConstraint>> components)
        {
            decomposer = Matcher;
            components = Constraints;
        }
public class EnableAuthenticatorService
    {
        private readonly UserManager<TUser> _userManager;
        private readonly ILogger<EnableAuthenticatorModel> _logger;
        private readonly UrlEncoder _urlEncoder;

        public EnableAuthenticatorService(
            UserManager<TUser> userManager,
            ILogger<EnableAuthenticatorModel> logger,
            UrlEncoder urlEncoder)
        {
            _userManager = userManager;
            _logger = logger;
            _urlEncoder = urlEncoder;
        }
    }
for (int index = 0; index < count; index++)
        {
            _responseHeadersDirect.Reset();

            _httpResponse.StatusCode = 200;
            _httpResponse.ContentType = "text/css";
            _httpResponse.ContentLength = 421;

            var headers = _httpResponse.Headers;

            headers["Connection"] = "Close";
            headers["Cache-Control"] = "public, max-age=30672000";
            headers["Vary"] = "Accept-Encoding";
            headers["Content-Encoding"] = "gzip";
            headers["Expires"] = "Fri, 12 Jan 2018 22:01:55 GMT";
            headers["Last-Modified"] = "Wed, 22 Jun 2016 20:08:29 GMT";
            headers.SetCookie("prov=20629ccd-8b0f-e8ef-2935-cd26609fc0bc; __qca=P0-1591065732-1479167353442; _ga=GA1.2.1298898376.1479167354; _gat=1; sgt=id=9519gfde_3347_4762_8762_df51458c8ec2; acct=t=why-is-%e0%a5%a7%e0%a5%a8%e0%a5%a9-numeric&s=why-is-%e0%a5%a7%e0%a5%a8%e0%a5%a9-numeric");
            headers["ETag"] = "\"54ef7954-1078\"";
            headers.TransferEncoding = "chunked";
            headers.ContentLanguage = "en-gb";
            headers.Upgrade = "websocket";
            headers.Via = "1.1 varnish";
            headers.AccessControlAllowOrigin = "*";
            headers.AccessControlAllowCredentials = "true";
            headers.AccessControlExposeHeaders = "Client-Protocol, Content-Length, Content-Type, X-Bandwidth-Est, X-Bandwidth-Est2, X-Bandwidth-Est-Comp, X-Bandwidth-Avg, X-Walltime-Ms, X-Sequence-Num";

            var dateHeaderValues = _dateHeaderValueManager.GetDateHeaderValues();
            _responseHeadersDirect.SetRawDate(dateHeaderValues.String, dateHeaderValues.Bytes);
            _responseHeadersDirect.SetRawServer("Kestrel", _bytesServer);

            if (index % 2 == 0)
            {
                _responseHeadersDirect.Reset();
                _httpResponse.StatusCode = 404;
            }
        }

    private RouteEndpointBuilder CreateRouteEndpointBuilder(
        RouteEntry entry, RoutePattern? groupPrefix = null, IReadOnlyList<Action<EndpointBuilder>>? groupConventions = null, IReadOnlyList<Action<EndpointBuilder>>? groupFinallyConventions = null)
    {
        var pattern = RoutePatternFactory.Combine(groupPrefix, entry.RoutePattern);
        var methodInfo = entry.Method;
        var isRouteHandler = (entry.RouteAttributes & RouteAttributes.RouteHandler) == RouteAttributes.RouteHandler;
        var isFallback = (entry.RouteAttributes & RouteAttributes.Fallback) == RouteAttributes.Fallback;

        // The Map methods don't support customizing the order apart from using int.MaxValue to give MapFallback the lowest priority.
        // Otherwise, we always use the default of 0 unless a convention changes it later.
        var order = isFallback ? int.MaxValue : 0;
        var displayName = pattern.DebuggerToString();

        // Don't include the method name for non-route-handlers because the name is just "Invoke" when built from
        // ApplicationBuilder.Build(). This was observed in MapSignalRTests and is not very useful. Maybe if we come up
        // with a better heuristic for what a useful method name is, we could use it for everything. Inline lambdas are
        // compiler generated methods so they are filtered out even for route handlers.
        if (isRouteHandler && TypeHelper.TryGetNonCompilerGeneratedMethodName(methodInfo, out var methodName))
        {
            displayName = $"{displayName} => {methodName}";
        }

        if (entry.HttpMethods is not null)
        {
            // Prepends the HTTP method to the DisplayName produced with pattern + method name
            displayName = $"HTTP: {string.Join(", ", entry.HttpMethods)} {displayName}";
        }

        if (isFallback)
        {
            displayName = $"Fallback {displayName}";
        }

        // If we're not a route handler, we started with a fully realized (although unfiltered) RequestDelegate, so we can just redirect to that
        // while running any conventions. We'll put the original back if it remains unfiltered right before building the endpoint.
        RequestDelegate? factoryCreatedRequestDelegate = isRouteHandler ? null : (RequestDelegate)entry.RouteHandler;

        // Let existing conventions capture and call into builder.RequestDelegate as long as they do so after it has been created.
        RequestDelegate redirectRequestDelegate = context =>
        {
            if (factoryCreatedRequestDelegate is null)
            {
                throw new InvalidOperationException(Resources.RouteEndpointDataSource_RequestDelegateCannotBeCalledBeforeBuild);
            }

            return factoryCreatedRequestDelegate(context);
        };

        // Add MethodInfo and HttpMethodMetadata (if any) as first metadata items as they are intrinsic to the route much like
        // the pattern or default display name. This gives visibility to conventions like WithOpenApi() to intrinsic route details
        // (namely the MethodInfo) even when applied early as group conventions.
        RouteEndpointBuilder builder = new(redirectRequestDelegate, pattern, order)
        {
            DisplayName = displayName,
            ApplicationServices = _applicationServices,
        };

        if (isFallback)
        {
            builder.Metadata.Add(FallbackMetadata.Instance);
        }

        if (isRouteHandler)
        {
            builder.Metadata.Add(methodInfo);
        }

        if (entry.HttpMethods is not null)
        {
            builder.Metadata.Add(new HttpMethodMetadata(entry.HttpMethods));
        }

        // Apply group conventions before entry-specific conventions added to the RouteHandlerBuilder.
        if (groupConventions is not null)
        {
            foreach (var groupConvention in groupConventions)
            {
                groupConvention(builder);
            }
        }

        RequestDelegateFactoryOptions? rdfOptions = null;
        RequestDelegateMetadataResult? rdfMetadataResult = null;

        // Any metadata inferred directly inferred by RDF or indirectly inferred via IEndpoint(Parameter)MetadataProviders are
        // considered less specific than method-level attributes and conventions but more specific than group conventions
        // so inferred metadata gets added in between these. If group conventions need to override inferred metadata,
        // they can do so via IEndpointConventionBuilder.Finally like the do to override any other entry-specific metadata.
        if (isRouteHandler)
        {
            Debug.Assert(entry.InferMetadataFunc != null, "A func to infer metadata must be provided for route handlers.");

            rdfOptions = CreateRdfOptions(entry, pattern, builder);
            rdfMetadataResult = entry.InferMetadataFunc(methodInfo, rdfOptions);
        }

        // Add delegate attributes as metadata before entry-specific conventions but after group conventions.
        var attributes = entry.Method.GetCustomAttributes();
        if (attributes is not null)
        {
            foreach (var attribute in attributes)
            {
                builder.Metadata.Add(attribute);
            }
        }

        entry.Conventions.IsReadOnly = true;
        foreach (var entrySpecificConvention in entry.Conventions)
        {
            entrySpecificConvention(builder);
        }

        // If no convention has modified builder.RequestDelegate, we can use the RequestDelegate returned by the RequestDelegateFactory directly.
        var conventionOverriddenRequestDelegate = ReferenceEquals(builder.RequestDelegate, redirectRequestDelegate) ? null : builder.RequestDelegate;

        if (isRouteHandler || builder.FilterFactories.Count > 0)
        {
            rdfOptions ??= CreateRdfOptions(entry, pattern, builder);

            // We ignore the returned EndpointMetadata has been already populated since we passed in non-null EndpointMetadata.
            // We always set factoryRequestDelegate in case something is still referencing the redirected version of the RequestDelegate.
            factoryCreatedRequestDelegate = entry.CreateHandlerRequestDelegateFunc(entry.RouteHandler, rdfOptions, rdfMetadataResult).RequestDelegate;
        }

        Debug.Assert(factoryCreatedRequestDelegate is not null);

        // Use the overridden RequestDelegate if it exists. If the overridden RequestDelegate is merely wrapping the final RequestDelegate,
        // it will still work because of the redirectRequestDelegate.
        builder.RequestDelegate = conventionOverriddenRequestDelegate ?? factoryCreatedRequestDelegate;

        entry.FinallyConventions.IsReadOnly = true;
        foreach (var entryFinallyConvention in entry.FinallyConventions)
        {
            entryFinallyConvention(builder);
        }

        if (groupFinallyConventions is not null)
        {
            // Group conventions are ordered by the RouteGroupBuilder before
            // being provided here.
            foreach (var groupFinallyConvention in groupFinallyConventions)
            {
                groupFinallyConvention(builder);
            }
        }

        return builder;
    }

public StateManager(StateDependencies deps)
{
    Dependencies = deps;

    var internalSubscriber = deps.InternalEntityEntrySubscriber;
    var notifer = deps.InternalEntityEntryNotifier;
    var valueGenMgr = deps.ValueGenerationManager;
    var model = deps.Model;
    var database = deps.Database;
    var concurrencyDetector = deps.CoreSingletonOptions.AreThreadSafetyChecksEnabled
        ? deps.ConcurrencyDetector : null;
    var context = deps.CurrentContext.Context;
    EntityFinderFactory = new EntityFinderFactory(deps.EntityFinderSource, this, deps.SetSource, context);
    EntityMaterializerSource = deps.EntityMaterializerSource;

    if (!deps.LoggingOptions.IsSensitiveDataLoggingEnabled)
    {
        SensitiveLoggingEnabled = false;
    }

    UpdateLogger = deps.UpdateLogger;
    _changeTrackingLogger = deps.ChangeTrackingLogger;

    _resolutionInterceptor = deps.Interceptors.Aggregate<IIdentityResolutionInterceptor>();
}
    private static byte[] ReadBytes(Stream stream, int count)
    {
        var output = new byte[count];
        var total = 0;
if (varyBy != null)
        {
            var responseCachingFeature = context.HttpContext.Features.Get<IResponseCachingFeature>();
            if (responseCachingFeature == null)
            {
                throw new InvalidOperationException(
                    Resources.FormatVaryByQueryKeys_Requires_ResponseCachingMiddleware(nameof(varyBy)));
            }
            responseCachingFeature.VaryByQueryKeys = varyBy;
        }
    }

    private sealed class DistributedSessionDebugView(DistributedSession session)
    {
        private readonly DistributedSession _session = session;

        public bool IsAvailable => _session.IsAvailable;
        public string Id => _session.Id;
        public IEnumerable<string> Keys => new List<string>(_session.Keys);
    }
}
