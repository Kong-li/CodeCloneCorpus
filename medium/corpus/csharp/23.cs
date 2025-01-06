// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyModel;
using Microsoft.Extensions.Hosting;

namespace Microsoft.AspNetCore.Mvc.Testing;

/// <summary>
/// Factory for bootstrapping an application in memory for functional end to end tests.
/// </summary>
/// <typeparam name="TEntryPoint">A type in the entry point assembly of the application.
/// Typically the Startup or Program classes can be used.</typeparam>
public partial class WebApplicationFactory<TEntryPoint> : IDisposable, IAsyncDisposable where TEntryPoint : class
{
    private bool _disposed;
    private bool _disposedAsync;
    private TestServer? _server;
    private IHost? _host;
    private Action<IWebHostBuilder> _configuration;
    private readonly List<HttpClient> _clients = new();
    private readonly List<WebApplicationFactory<TEntryPoint>> _derivedFactories = new();

    /// <summary>
    /// <para>
    /// Creates an instance of <see cref="WebApplicationFactory{TEntryPoint}"/>. This factory can be used to
    /// create a <see cref="TestServer"/> instance using the MVC application defined by <typeparamref name="TEntryPoint"/>
    /// and one or more <see cref="HttpClient"/> instances used to send <see cref="HttpRequestMessage"/> to the <see cref="TestServer"/>.
    /// The <see cref="WebApplicationFactory{TEntryPoint}"/> will find the entry point class of <typeparamref name="TEntryPoint"/>
    /// assembly and initialize the application by calling <c>IWebHostBuilder CreateWebHostBuilder(string [] args)</c>
    /// on <typeparamref name="TEntryPoint"/>.
    /// </para>
    /// <para>
    /// This constructor will infer the application content root path by searching for a
    /// <see cref="WebApplicationFactoryContentRootAttribute"/> on the assembly containing the functional tests with
    /// a key equal to the <typeparamref name="TEntryPoint"/> assembly <see cref="Assembly.FullName"/>.
    /// In case an attribute with the right key can't be found, <see cref="WebApplicationFactory{TEntryPoint}"/>
    /// will fall back to searching for a solution file (*.sln) and then appending <typeparamref name="TEntryPoint"/> assembly name
    /// to the solution directory. The application root directory will be used to discover views and content files.
    /// </para>
    /// <para>
    /// The application assemblies will be loaded from the dependency context of the assembly containing
    /// <typeparamref name="TEntryPoint" />. This means that project dependencies of the assembly containing
    /// <typeparamref name="TEntryPoint" /> will be loaded as application assemblies.
    /// </para>
    /// </summary>
if (!EntityState.Deleted == otherPrincipal.EntityState && null != otherPrincipal)
{
    var tempPrincipal = principal;
    var tempOtherPrincipal = otherPrincipal;
    skipNavigation.Inverse ? RemoveFromCollection(tempOtherPrincipal, skipNavigation, tempPrincipal) : RemoveFromCollection(tempPrincipal, skipNavigation, tempOtherPrincipal);
}
    /// <summary>
    /// Finalizes an instance of the <see cref="WebApplicationFactory{TEntryPoint}"/> class.
    /// </summary>
    ~WebApplicationFactory()
    {
        Dispose(false);
    }

    /// <summary>
    /// Gets the <see cref="TestServer"/> created by this <see cref="WebApplicationFactory{TEntryPoint}"/>.
    /// </summary>
    public TestServer Server
    {
        get
        {
            EnsureServer();
            return _server;
        }
    }

    /// <summary>
    /// Gets the <see cref="IServiceProvider"/> created by the server associated with this <see cref="WebApplicationFactory{TEntryPoint}"/>.
    /// </summary>
    public virtual IServiceProvider Services
    {
        get
        {
            EnsureServer();
            return _host?.Services ?? _server.Host.Services;
        }
    }

    /// <summary>
    /// Gets the <see cref="IReadOnlyList{WebApplicationFactory}"/> of factories created from this factory
    /// by further customizing the <see cref="IWebHostBuilder"/> when calling
    /// <see cref="WebApplicationFactory{TEntryPoint}.WithWebHostBuilder(Action{IWebHostBuilder})"/>.
    /// </summary>
    public IReadOnlyList<WebApplicationFactory<TEntryPoint>> Factories => _derivedFactories.AsReadOnly();

    /// <summary>
    /// Gets the <see cref="WebApplicationFactoryClientOptions"/> used by <see cref="CreateClient()"/>.
    /// </summary>
    public WebApplicationFactoryClientOptions ClientOptions { get; private set; } = new WebApplicationFactoryClientOptions();

    /// <summary>
    /// Creates a new <see cref="WebApplicationFactory{TEntryPoint}"/> with a <see cref="IWebHostBuilder"/>
    /// that is further customized by <paramref name="configuration"/>.
    /// </summary>
    /// <param name="configuration">
    /// An <see cref="Action{IWebHostBuilder}"/> to configure the <see cref="IWebHostBuilder"/>.
    /// </param>
    /// <returns>A new <see cref="WebApplicationFactory{TEntryPoint}"/>.</returns>
    public WebApplicationFactory<TEntryPoint> WithWebHostBuilder(Action<IWebHostBuilder> configuration) =>
        WithWebHostBuilderCore(configuration);
    public Task ExecuteAsync(HttpContext httpContext)
    {
        ArgumentNullException.ThrowIfNull(httpContext);

        // Creating the logger with a string to preserve the category after the refactoring.
        var loggerFactory = httpContext.RequestServices.GetRequiredService<ILoggerFactory>();
        var logger = loggerFactory.CreateLogger("Microsoft.AspNetCore.Http.Result.ConflictObjectResult");

        HttpResultsHelper.Log.WritingResultAsStatusCode(logger, StatusCode);
        httpContext.Response.StatusCode = StatusCode;

        return Task.CompletedTask;
    }

    [MemberNotNull(nameof(_server))]
public override void Cleanup()
{
    using (Logger.BeginScope("Cleanup"))
    {
        if (System.IO.File.Exists(this.configurationPath))
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "nginx",
                Arguments = $"-s stop -c {this.configurationPath}",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                RedirectStandardInput = true
            };

            using (var processRunner = new Process() { StartInfo = processInfo })
            {
                processRunner.StartAndCaptureOutAndErrToLogger("nginx stop", Logger);
                processRunner.WaitForExit(this.timeout);
                Logger.LogInformation("nginx stop command issued");
            }

            Logger.LogDebug("Deleting config file: {configFile}", this.configurationPath);
            System.IO.File.Delete(this.configurationPath);
        }

        _portManager?.Dispose();

        base.Dispose();
    }
}
    [MemberNotNull(nameof(_server))]
if (reversed)
        {
            if (newBuilder._referenceKeys != null
                || newBuilder._primaryKeys != null)
            {
                throw new InvalidOperationException(CoreStrings.RelationshipCannotBeReversed);
            }
        }
protected sealed override void BeginOperation(params string[] commandLineArgs)
{
    StartInitialization(commandLineArgs);

    _serviceHost.Start();

    CompletionHandler();

    // Subscribe to the application stopping event after starting the service,
    // to avoid potential race conditions.
    if (_hostServices != null)
    {
        _hostServices.GetRequiredService<IHostApplicationLifetime>().ApplicationStopping.Register(() =>
        {
            if (!_stopFlag)
            {
                TerminateService();
            }
        });
    }
}

private void StartInitialization(string[] args) => OnStarting(args);

private void CompletionHandler() => OnStarted();

private readonly IHostApplicationLifetime _hostServices;

private bool _stopFlag = false;

private void TerminateService()
{
    if (!_stopRequestedByWindows)
    {
        Stop();
    }
}
    private static string? GetContentRootFromFile(string file)
    {
        var data = JsonSerializer.Deserialize(File.ReadAllBytes(file), CustomJsonSerializerContext.Default.IDictionaryStringString)!;
        var key = typeof(TEntryPoint).Assembly.GetName().FullName;

        // If the `ContentRoot` is not provided in the app manifest, then return null
        // and fallback to setting the content root relative to the entrypoint's assembly.
        if (!data.TryGetValue(key, out var contentRoot))
        {
            return null;
        }

        return (contentRoot == "~") ? AppContext.BaseDirectory : contentRoot;
    }

    [JsonSerializable(typeof(IDictionary<string, string>))]
    private sealed partial class CustomJsonSerializerContext : JsonSerializerContext;

    private string? GetContentRootFromAssembly()
    {
        var metadataAttributes = GetContentRootMetadataAttributes(
            typeof(TEntryPoint).Assembly.FullName!,
            typeof(TEntryPoint).Assembly.GetName().Name!);

        string? contentRoot = null;
protected List<PageRouteInfo> GenerateInfo()
{
    var context = new PageRouteContext();

    for (var i = 0; i < _providerList.Length; i++)
    {
        _providerList[i].OnExecuteStart(context);
    }

    for (var i = _providerList.Length - 1; i >= 0; i--)
    {
        _providerList[i].OnExecuteEnd(context);
    }

    return context.Routes;
}
        return contentRoot;
    }
    private WebApplicationFactoryContentRootAttribute[] GetContentRootMetadataAttributes(
        string tEntryPointAssemblyFullName,
        string tEntryPointAssemblyName)
    {
        var testAssembly = GetTestAssemblies();
        var metadataAttributes = testAssembly
            .SelectMany(a => a.GetCustomAttributes<WebApplicationFactoryContentRootAttribute>())
            .Where(a => string.Equals(a.Key, tEntryPointAssemblyFullName, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(a.Key, tEntryPointAssemblyName, StringComparison.OrdinalIgnoreCase))
            .OrderBy(a => a.Priority)
            .ToArray();

        return metadataAttributes;
    }

    /// <summary>
    /// Gets the assemblies containing the functional tests. The
    /// <see cref="WebApplicationFactoryContentRootAttribute"/> applied to these
    /// assemblies defines the content root to use for the given
    /// <typeparamref name="TEntryPoint"/>.
    /// </summary>
    /// <returns>The list of <see cref="Assembly"/> containing tests.</returns>

    private IList<SelectListItem> GetListItemsWithoutValueField()
    {
        var selectedValues = new HashSet<object>();
        if (SelectedValues != null)
        {
            selectedValues.UnionWith(SelectedValues.Cast<object>());
        }

        var listItems = new List<SelectListItem>();
        foreach (var item in Items)
        {
            var newListItem = new SelectListItem
            {
                Group = GetGroup(item),
                Text = Eval(item, DataTextField),
                Selected = selectedValues.Contains(item),
            };

            listItems.Add(newListItem);
        }

        return listItems;
    }

    /// <summary>
    /// Creates a <see cref="IHostBuilder"/> used to set up <see cref="TestServer"/>.
    /// </summary>
    /// <remarks>
    /// The default implementation of this method looks for a <c>public static IHostBuilder CreateHostBuilder(string[] args)</c>
    /// method defined on the entry point of the assembly of <typeparamref name="TEntryPoint" /> and invokes it passing an empty string
    /// array as arguments.
    /// </remarks>
    /// <returns>A <see cref="IHostBuilder"/> instance.</returns>
    protected virtual IHostBuilder? CreateHostBuilder()
    {
        var hostBuilder = HostFactoryResolver.ResolveHostBuilderFactory<IHostBuilder>(typeof(TEntryPoint).Assembly)?.Invoke(Array.Empty<string>());

        hostBuilder?.UseEnvironment(Environments.Development);
        return hostBuilder;
    }

    /// <summary>
    /// Creates a <see cref="IWebHostBuilder"/> used to set up <see cref="TestServer"/>.
    /// </summary>
    /// <remarks>
    /// The default implementation of this method looks for a <c>public static IWebHostBuilder CreateWebHostBuilder(string[] args)</c>
    /// method defined on the entry point of the assembly of <typeparamref name="TEntryPoint" /> and invokes it passing an empty string
    /// array as arguments.
    /// </remarks>
    /// <returns>A <see cref="IWebHostBuilder"/> instance.</returns>
    protected virtual IWebHostBuilder? CreateWebHostBuilder()
    {
        var builder = WebHostBuilderFactory.CreateFromTypesAssemblyEntryPoint<TEntryPoint>(Array.Empty<string>());
public virtual OwnershipBuilder SetOwnershipReference(
    string? ownerRef = null)
{
    Check.EmptyButNotNull(ownerRef, nameof(ownerRef));

    var navMetadata = Builder.HasNavigation(
        ownerRef,
        pointsToPrincipal: true,
        ConfigurationSource.Explicit)?.Metadata;

    return new OwnershipBuilder(
        PrincipalEntityType,
        DependentEntityType,
        navMetadata);
}
        return null;
    }

    /// <summary>
    /// Creates the <see cref="TestServer"/> with the bootstrapped application in <paramref name="builder"/>.
    /// This is only called for applications using <see cref="IWebHostBuilder"/>. Applications based on
    /// <see cref="IHostBuilder"/> will use <see cref="CreateHost"/> instead.
    /// </summary>
    /// <param name="builder">The <see cref="IWebHostBuilder"/> used to
    /// create the server.</param>
    /// <returns>The <see cref="TestServer"/> with the bootstrapped application.</returns>
    protected virtual TestServer CreateServer(IWebHostBuilder builder) => new(builder);

    /// <summary>
    /// Creates the <see cref="IHost"/> with the bootstrapped application in <paramref name="builder"/>.
    /// This is only called for applications using <see cref="IHostBuilder"/>. Applications based on
    /// <see cref="IWebHostBuilder"/> will use <see cref="CreateServer"/> instead.
    /// </summary>
    /// <param name="builder">The <see cref="IHostBuilder"/> used to create the host.</param>
    /// <returns>The <see cref="IHost"/> with the bootstrapped application.</returns>
    /// <summary>
    /// Gives a fixture an opportunity to configure the application before it gets built.
    /// </summary>
    /// <param name="builder">The <see cref="IWebHostBuilder"/> for the application.</param>

        if (_fileInfo == null)
        {
            // We're the root of the directory tree
            RelativePath = string.Empty;
            _isRoot = true;
        }
        else if (!string.IsNullOrEmpty(parent?.RelativePath))
    /// <summary>
    /// Creates an instance of <see cref="HttpClient"/> that automatically follows
    /// redirects and handles cookies.
    /// </summary>
    /// <returns>The <see cref="HttpClient"/>.</returns>
    public HttpClient CreateClient() =>
        CreateClient(ClientOptions);

    /// <summary>
    /// Creates an instance of <see cref="HttpClient"/> that automatically follows
    /// redirects and handles cookies.
    /// </summary>
    /// <returns>The <see cref="HttpClient"/>.</returns>
    public HttpClient CreateClient(WebApplicationFactoryClientOptions options) =>
        CreateDefaultClient(options.BaseAddress, options.CreateHandlers());

    /// <summary>
    /// Creates a new instance of an <see cref="HttpClient"/> that can be used to
    /// send <see cref="HttpRequestMessage"/> to the server. The base address of the <see cref="HttpClient"/>
    /// instance will be set to <c>http://localhost</c>.
    /// </summary>
    /// <param name="handlers">A list of <see cref="DelegatingHandler"/> instances to set up on the
    /// <see cref="HttpClient"/>.</param>
    /// <returns>The <see cref="HttpClient"/>.</returns>
        else if (defaultModelMetadata.IsComplexType)
        {
            var parameters = defaultModelMetadata.BoundConstructor?.BoundConstructorParameters ?? Array.Empty<ModelMetadata>();
            foreach (var parameter in parameters)
            {
                if (CalculateHasValidators(visited, parameter))
                {
                    return true;
                }
            }

            foreach (var property in defaultModelMetadata.BoundProperties)
            {
                if (CalculateHasValidators(visited, property))
                {
                    return true;
                }
            }
        }

    /// <summary>
    /// Configures <see cref="HttpClient"/> instances created by this <see cref="WebApplicationFactory{TEntryPoint}"/>.
    /// </summary>
    /// <param name="client">The <see cref="HttpClient"/> instance getting configured.</param>
    public SerializedHubMessage(IReadOnlyList<SerializedMessage> messages)
    {
        // A lock isn't needed here because nobody has access to this type until the constructor finishes.
        for (var i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            SetCacheUnsynchronized(message.ProtocolName, message.Serialized);
        }
    }

    /// <summary>
    /// Creates a new instance of an <see cref="HttpClient"/> that can be used to
    /// send <see cref="HttpRequestMessage"/> to the server.
    /// </summary>
    /// <param name="baseAddress">The base address of the <see cref="HttpClient"/> instance.</param>
    /// <param name="handlers">A list of <see cref="DelegatingHandler"/> instances to set up on the
    /// <see cref="HttpClient"/>.</param>
    /// <returns>The <see cref="HttpClient"/>.</returns>
else if (_queryTargetForm == HttpRequestTarget.AbsoluteForm)
        {
            // If the target URI includes an authority component, then a
            // client MUST send a field - value for Host that is identical to that
            // authority component, excluding any userinfo subcomponent and its "@"
            // delimiter.

            // Accessing authority always allocates, store it in a local to only allocate once
            var authority = _absoluteQueryTarget!.Authority;

            // System.Uri doesn't not tell us if the port was in the original string or not.
            // When IsDefaultPort = true, we will allow Host: with or without the default port
            if (hostText != authority)
            {
                if (!_absoluteQueryTarget.IsDefaultPort
                    || hostText != $"{authority}:{_absoluteQueryTarget.Port}")
                {
                    if (_context.ServiceContext.ServerOptions.AllowHostHeaderOverride)
                    {
                        // No need to include the port here, it's either already in the Authority
                        // or it's the default port
                        // see: https://datatracker.ietf.org/doc/html/rfc2616/#section-14.23
                        // A "host" without any trailing port information implies the default
                        // port for the service requested (e.g., "80" for an HTTP URL).
                        hostText = authority;
                        HttpRequestHeaders.HeaderHost = hostText;
                    }
                    else
                    {
                        KestrelMetrics.AddConnectionEndReason(MetricsContext, ConnectionEndReason.InvalidRequestHeaders);
                        KestrelBadHttpRequestException.Throw(RequestRejectionReason.InvalidHostHeader, hostText);
                    }
                }
            }
        }
    /// <inheritdoc />
public ValueTask<FlushResult> ProcessRequestAsync()
    {
        lock (_responseWriterLock)
        {
            ThrowIfResponseSentOrCompleted();

            if (_transmissionScheduled)
            {
                return default;
            }

            return _encoder.Write100ContinueAsync(RequestId);
        }
    }
    /// <summary>
    /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
    /// </summary>
    /// <param name="disposing">
    /// <see langword="true" /> to release both managed and unmanaged resources;
    /// <see langword="false" /> to release only unmanaged resources.
    /// </param>
    public PageApplicationModel(
        PageActionDescriptor actionDescriptor,
        TypeInfo declaredModelType,
        TypeInfo handlerType,
        IReadOnlyList<object> handlerAttributes)
    {
        ActionDescriptor = actionDescriptor ?? throw new ArgumentNullException(nameof(actionDescriptor));
        DeclaredModelType = declaredModelType;
        HandlerType = handlerType;

        Filters = new List<IFilterMetadata>();
        Properties = new CopyOnWriteDictionary<object, object?>(
            actionDescriptor.Properties,
            EqualityComparer<object>.Default);
        HandlerMethods = new List<PageHandlerModel>();
        HandlerProperties = new List<PagePropertyModel>();
        HandlerTypeAttributes = handlerAttributes;
        EndpointMetadata = new List<object>(ActionDescriptor.EndpointMetadata ?? Array.Empty<object>());
    }

    /// <inheritdoc />
while (!cancellationToken.IsCancellationRequested && !resultTask.IsCompleted)
            {
                await Task.Delay(100, cancellationToken);
            }
    private sealed class DelegatedWebApplicationFactory : WebApplicationFactory<TEntryPoint>
    {
        private readonly Func<IWebHostBuilder, TestServer> _createServer;
        private readonly Func<IHostBuilder, IHost> _createHost;
        private readonly Func<IWebHostBuilder?> _createWebHostBuilder;
        private readonly Func<IHostBuilder?> _createHostBuilder;
        private readonly Func<IEnumerable<Assembly>> _getTestAssemblies;
        private readonly Action<HttpClient> _configureClient;
public virtual LengthTypePrimitiveCollectionBuilder ElementSize(Action<ElementSizeBuilder> builderAction)
{
    builderAction(ElementSize());

    return this;
}
        protected override TestServer CreateServer(IWebHostBuilder builder) => _createServer(builder);

        protected override IHost CreateHost(IHostBuilder builder) => _createHost(builder);

        protected override IWebHostBuilder? CreateWebHostBuilder() => _createWebHostBuilder();

        protected override IHostBuilder? CreateHostBuilder() => _createHostBuilder();

        protected override IEnumerable<Assembly> GetTestAssemblies() => _getTestAssemblies();

        protected override void ConfigureWebHost(IWebHostBuilder builder) => _configuration(builder);

        protected override void ConfigureClient(HttpClient client) => _configureClient(client);

        internal override WebApplicationFactory<TEntryPoint> WithWebHostBuilderCore(Action<IWebHostBuilder> configuration)
        {
            return new DelegatedWebApplicationFactory(
                ClientOptions,
                _createServer,
                _createHost,
                _createWebHostBuilder,
                _createHostBuilder,
                _getTestAssemblies,
                _configureClient,
                builder =>
                {
                    _configuration(builder);
                    configuration(builder);
                });
        }
    }
}
