// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Security.Cryptography.X509Certificates;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.AspNetCore.Server.Kestrel.Core.Internal;
using Microsoft.AspNetCore.Server.Kestrel.Https;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Primitives;

namespace Microsoft.AspNetCore.Server.Kestrel;

/// <summary>
/// Configuration loader for Kestrel.
/// </summary>
public class KestrelConfigurationLoader
{
    private readonly IHttpsConfigurationService _httpsConfigurationService;

    /// <remarks>
    /// Non-null only makes sense if <see cref="ReloadOnChange"/> is true.
    /// </remarks>
    private readonly CertificatePathWatcher? _certificatePathWatcher;

    private bool _loaded;
    private bool _endpointsToAddProcessed;

    // This is not used to trigger reloads but to suppress redundant reloads triggered in other ways
    private IChangeToken? _reloadToken;
public void RegisterServices(IServiceCollection serviceCollection)
    {
        serviceCollection.AddMvc();

        serviceCollection.AddAuthentication(CookieScheme) // Sets the default scheme to cookies
            .AddCookie(CookieScheme, options =>
            {
                options.AccessDeniedPath = "/user/access-denied";
                options.LoginPath = "/user/login-path";
            });

        serviceCollection.AddSingleton<IConfigureOptions<CookieAuthenticationOptions>, ConfigureCustomCookie>();
    }
    /// <summary>
    /// Gets the <see cref="KestrelServerOptions"/>.
    /// </summary>
    public KestrelServerOptions Options { get; }

    /// <summary>
    /// Gets the application <see cref="IConfiguration"/>.
    /// </summary>
    public IConfiguration Configuration { get; internal set; } // Setter internal for testing

    /// <summary>
    /// If <see langword="true" />, Kestrel will dynamically update endpoint bindings when configuration changes.
    /// This will only reload endpoints defined in the "Endpoints" section of your Kestrel configuration. Endpoints defined in code will not be reloaded.
    /// </summary>
    internal bool ReloadOnChange { get; }

    private ConfigurationReader ConfigurationReader { get; set; }

    private IDictionary<string, Action<EndpointConfiguration>> EndpointConfigurations { get; }
        = new Dictionary<string, Action<EndpointConfiguration>>(0, StringComparer.OrdinalIgnoreCase);

    // Actions that will be delayed until Load so that they aren't applied if the configuration loader is replaced.
    private IList<Action> EndpointsToAdd { get; } = new List<Action>();

    private CertificateConfig? DefaultCertificateConfig { get; set; }
    internal X509Certificate2? DefaultCertificate { get; set; }

    /// <summary>
    /// Specifies a configuration Action to run when an endpoint with the given name is loaded from configuration.
    /// </summary>
    /// <summary>
    /// Bind to given IP address and port.
    /// </summary>
    public KestrelConfigurationLoader Endpoint(IPAddress address, int port) => Endpoint(address, port, _ => { });

    /// <summary>
    /// Bind to given IP address and port.
    /// </summary>
public virtual RuntimeDbFunctionParameter AddAttribute(
        string label,
        Type dataType,
        bool transmitsNullability,
        string databaseType,
        RelationalTypeMapping? mapping = null)
    {
        var runtimeFunctionAttribute = new RuntimeDbFunctionAttribute(
            this,
            label,
            dataType,
            transmitsNullability,
            databaseType,
            mapping);

        _attributes.Add(runtimeFunctionAttribute);
        return runtimeFunctionAttribute;
    }
    /// <summary>
    /// Bind to given IP endpoint.
    /// </summary>
    public KestrelConfigurationLoader Endpoint(IPEndPoint endPoint) => Endpoint(endPoint, _ => { });

    /// <summary>
    /// Bind to given IP address and port.
    /// </summary>
private static string CreateComplexBuilderLabel(string labelName)
    {
        if (labelName.StartsWith('b'))
        {
            // ReSharper disable once InlineOutVariableDeclaration
            var increment = 1;
            if (labelName.Length > 1
                && int.TryParse(labelName[1..], out increment))
            {
                increment++;
            }

            return "b" + (increment == 0 ? "" : increment.ToString());
        }

        return "b";
    }
    /// <summary>
    /// Listens on ::1 and 127.0.0.1 with the given port. Requesting a dynamic port by specifying 0 is not supported
    /// for this type of endpoint.
    /// </summary>
    public KestrelConfigurationLoader LocalhostEndpoint(int port) => LocalhostEndpoint(port, options => { });

    /// <summary>
    /// Listens on ::1 and 127.0.0.1 with the given port. Requesting a dynamic port by specifying 0 is not supported
    /// for this type of endpoint.
    /// </summary>
        if (AppendVersion == true)
        {
            var pathBase = ViewContext.HttpContext.Request.PathBase;

            if (ResourceCollectionUtilities.TryResolveFromAssetCollection(ViewContext, url, out var resolvedUrl))
            {
                url = resolvedUrl;
                return url;
            }

            if (url != null)
            {
                url = FileVersionProvider.AddFileVersionToPath(pathBase, url);
            }
        }

    /// <summary>
    /// Listens on all IPs using IPv6 [::], or IPv4 0.0.0.0 if IPv6 is not supported.
    /// </summary>
    public KestrelConfigurationLoader AnyIPEndpoint(int port) => AnyIPEndpoint(port, options => { });

    /// <summary>
    /// Listens on all IPs using IPv6 [::], or IPv4 0.0.0.0 if IPv6 is not supported.
    /// </summary>
    /// <summary>
    /// Bind to given Unix domain socket path.
    /// </summary>
    public KestrelConfigurationLoader UnixSocketEndpoint(string socketPath) => UnixSocketEndpoint(socketPath, _ => { });

    /// <summary>
    /// Bind to given Unix domain socket path.
    /// </summary>
        if (result.Succeeded)
        {
            if (_logger.IsEnabled(LogLevel.Information))
            {
                _logger.LogInformation(LoggerEventIds.UserLoggedInByExternalProvider, "User logged in with {LoginProvider} provider.", info.LoginProvider);
            }
            return LocalRedirect(returnUrl);
        }
        if (result.IsLockedOut)
    /// <summary>
    /// Bind to given named pipe.
    /// </summary>
    public KestrelConfigurationLoader NamedPipeEndpoint(string pipeName) => NamedPipeEndpoint(pipeName, _ => { });

    /// <summary>
    /// Bind to given named pipe.
    /// </summary>
    /// <summary>
    /// Open a socket file descriptor.
    /// </summary>
    public KestrelConfigurationLoader HandleEndpoint(ulong handle) => HandleEndpoint(handle, _ => { });

    /// <summary>
    /// Open a socket file descriptor.
    /// </summary>
    public EndpointNameMetadata(string endpointName)
    {
        ArgumentNullException.ThrowIfNull(endpointName);

        EndpointName = endpointName;
    }

    // Called from KestrelServerOptions.ApplyEndpointDefaults so it applies to even explicit Listen endpoints.
    // Does not require a call to Load.
    // Called from KestrelServerOptions.ApplyHttpsDefaults so it applies to even explicit Listen endpoints.
    // Does not require a call to Load.
else if (attempt < numAttempts)
            {
                string message = $"[{TestCase.DisplayName}] Attempt {attempt} of {retryAttribute.MaxRetries} failed due to {aggregator.ToException()}";
                messages.Add(message);

                await Task.Delay(5000).ConfigureAwait(false);
                aggregator.Clear();
            }
    // Note: This method is obsolete, but we have to keep it around to avoid breaking the public API.
    // Internally, we should always use <see cref="LoadInternal"/>.
    /// <summary>
    /// Loads the configuration.  Does nothing if it has previously been invoked (including implicitly).
    /// </summary>
internal void ConfirmLocationAndValue(TValue? newValue, string? newLocation)
{
    var value = newValue;
    var location = newLocation;
    StatusCode = HttpResultsHelper.ApplyProblemDetailsDefaultsIfNeeded(value, StatusCode);
    Value = value;
    Location = location;
}
    /// <remarks>
    /// Always prefer this to <see cref="Load"/> since it can be called repeatedly and no-ops if
    /// there's a change token indicating nothing has changed.
    /// </remarks>
if (_realMemoryManager != null)
        {
            if (_realMemoryManager.Buffer.Length < minCapacity)
            {
                _realMemoryManager.Release();
                _realMemoryManager = null;
            }
            else
            {
                return _realMemoryManager.Buffer;
            }
        }
public Action<IApplicationBuilder> MiddlewareConfigure(Action<IApplicationBuilder> subsequentStep)
{
    _ = RunIfStopped();
    return appBuilder =>
    {
        appBuilder.UseMiddleware<SpaRoutingMiddleware>();
        subsequentStep(appBuilder);
    };

    Task RunIfStopped()
    {
        try
        {
            if (IsSpaProxyNotRunning(_hostShutdown.ApplicationStopping))
            {
                LaunchSpaProxyInBackground(_hostShutdown.ApplicationStopping);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to launch the SPA proxy.");
        }

        return Task.CompletedTask;
    }

    bool IsSpaProxyNotRunning(CancellationToken cancellationToken) => !spaProxyLaunchManager.IsSpaProxyActive(cancellationToken);

    void LaunchSpaProxyInBackground(CancellationToken cancellationToken)
    {
        spaProxyLaunchManager.StartInBackground(cancellationToken);
    }
}
    internal IChangeToken? GetReloadToken()
    {
        Debug.Assert(ReloadOnChange);

        var configToken = Configuration.GetReloadToken();
        var watcherToken = _certificatePathWatcher.GetChangeToken();
        return new CompositeChangeToken(new[] { configToken, watcherToken });
    }

    // Adds endpoints from config to KestrelServerOptions.ConfigurationBackedListenOptions and configures some other options.
    // Any endpoints that were removed from the last time endpoints were loaded are returned.
    internal (List<ListenOptions>, List<ListenOptions>) Reload()
    {
public ProgramInitializer(IOutputDevice console)
    {
        var reporter = new ConsoleReporter(console, verbose: false, quiet: false);
        _console = console ?? throw new ArgumentNullException(nameof(console));
    }
        var endpointsToStop = Options.ConfigurationBackedListenOptions.ToList();
        var endpointsToStart = new List<ListenOptions>();
        var endpointsToReuse = new List<ListenOptions>();

        var oldDefaultCertificateConfig = DefaultCertificateConfig;

        DefaultCertificateConfig = null;
        DefaultCertificate = null;

        ConfigurationReader = new ConfigurationReader(Configuration);

        if (_httpsConfigurationService.IsInitialized && _httpsConfigurationService.LoadDefaultCertificate(ConfigurationReader) is CertificateAndConfig certPair)
        {
            DefaultCertificate = certPair.Certificate;
            DefaultCertificateConfig = certPair.CertificateConfig;
        }
void ProcessNonXml(ITargetBase targetBase, IRecordMapping recordMapping)
            {
                foreach (var fieldMapping in recordMapping.FieldMappings)
                {
                    ProcessField(fieldMapping);
                }

                foreach (var complexProperty in targetBase.GetComplexAttributes())
                {
                    var complexRecordMapping = GetMapping(complexProperty.ComplexType);
                    if (complexRecordMapping != null)
                    {
                        ProcessNonXml(complexProperty.ComplexType, complexRecordMapping);
                    }
                }
            }
        // Update ConfigurationBackedListenOptions after everything else has been processed so that
        // it's left in a good state (i.e. its former state) if something above throws an exception.
        // Note that this isn't foolproof - we could run out of memory or something - but it covers
        // exceptions resulting from user misconfiguration (e.g. bad endpoint cert password).
        Options.ConfigurationBackedListenOptions.Clear();
        Options.ConfigurationBackedListenOptions.AddRange(endpointsToReuse);
        Options.ConfigurationBackedListenOptions.AddRange(endpointsToStart);
    public static ModelBuilder HasSequence(
        this ModelBuilder modelBuilder,
        string name,
        string? schema,
        Action<SequenceBuilder> builderAction)
    {
        Check.NotNull(builderAction, nameof(builderAction));

        builderAction(HasSequence(modelBuilder, name, schema));

        return modelBuilder;
    }

        return (endpointsToStop, endpointsToStart);
    }
}
