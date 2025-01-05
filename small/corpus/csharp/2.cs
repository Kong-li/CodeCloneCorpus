// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.AspNetCore.Hosting.Internal;
using Microsoft.Extensions.DependencyInjection;

namespace Microsoft.AspNetCore.Hosting;

internal sealed class StartupLoader
{
    // Creates an <see cref="StartupMethods"/> instance with the actions to run for configuring the application services and the
    // request pipeline of the application.
    // When using convention based startup, the process for initializing the services is as follows:
    // The host looks for a method with the signature <see cref="IServiceProvider"/> ConfigureServices(<see cref="IServiceCollection"/> services).
    // If it can't find one, it looks for a method with the signature <see cref="void"/> ConfigureServices(<see cref="IServiceCollection"/> services).
    // When the configure services method is void returning, the host builds a services configuration function that runs all the <see cref="IStartupConfigureServicesFilter"/>
    // instances registered on the host, along with the ConfigureServices method following a decorator pattern.
    // Additionally to the ConfigureServices method, the Startup class can define a <see cref="void"/> ConfigureContainer&lt;TContainerBuilder&gt;(TContainerBuilder builder)
    // method that further configures services into the container. If the ConfigureContainer method is defined, the services configuration function
    // creates a TContainerBuilder <see cref="IServiceProviderFactory{TContainerBuilder}"/> and runs all the <see cref="IStartupConfigureContainerFilter{TContainerBuilder}"/>
    // instances registered on the host, along with the ConfigureContainer method following a decorator pattern.
    // For example:
    // StartupFilter1
    //   StartupFilter2
    //     ConfigureServices
    //   StartupFilter2
    // StartupFilter1
    // ConfigureContainerFilter1
    //   ConfigureContainerFilter2
    //     ConfigureContainer
    //   ConfigureContainerFilter2
    // ConfigureContainerFilter1
    //
    // If the Startup class ConfigureServices returns an <see cref="IServiceProvider"/> and there is at least an <see cref="IStartupConfigureServicesFilter"/> registered we
    // throw as the filters can't be applied.
    public static StartupMethods LoadMethods(IServiceProvider hostingServiceProvider, [DynamicallyAccessedMembers(StartupLinkerOptions.Accessibility)] Type startupType, string environmentName, object? instance = null)
    {
        var configureMethod = FindConfigureDelegate(startupType, environmentName);

        var servicesMethod = FindConfigureServicesDelegate(startupType, environmentName);
        var configureContainerMethod = FindConfigureContainerDelegate(startupType, environmentName);

        if (instance == null && (!configureMethod.MethodInfo.IsStatic || (servicesMethod?.MethodInfo != null && !servicesMethod.MethodInfo.IsStatic)))
        {
            instance = ActivatorUtilities.GetServiceOrCreateInstance(hostingServiceProvider, startupType);
        }

        // The type of the TContainerBuilder. If there is no ConfigureContainer method we can just use object as it's not
        // going to be used for anything.
        var type = configureContainerMethod.MethodInfo != null ? configureContainerMethod.GetContainerType() : typeof(object);

        var builder = (ConfigureServicesDelegateBuilder)Activator.CreateInstance(
            CreateConfigureServicesDelegateBuilder(type),
            hostingServiceProvider,
            servicesMethod,
            configureContainerMethod,
            instance)!;

        return new StartupMethods(instance, configureMethod.Build(instance), builder.Build());

        [return: DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
        [UnconditionalSuppressMessage("AOT", "IL3050:RequiresDynamicCode",
            Justification = "There is a runtime check for ValueType startup container. It's unlikely anyone will use a ValueType here.")]

    public static void ViewFound(
        this DiagnosticListener diagnosticListener,
        ActionContext actionContext,
        bool isMainPage,
        ActionResult viewResult,
        string viewName,
        IView view)
    {
        // Inlinable fast-path check if Diagnositcs is enabled
        if (diagnosticListener.IsEnabled())
        {
            ViewFoundImpl(diagnosticListener, actionContext, isMainPage, viewResult, viewName, view);
        }
    }


    private abstract class ConfigureServicesDelegateBuilder
    {
        public abstract Func<IServiceCollection, IServiceProvider> Build();
    }

    private sealed class ConfigureServicesDelegateBuilder<TContainerBuilder> : ConfigureServicesDelegateBuilder where TContainerBuilder : notnull
    {
            fixed (byte* pNewTemplate = newTemplate)
            {
                WriteGuid(&pNewTemplate[sizeof(uint)], keyId);
                if (isProtecting)
                {
                    Volatile.Write(ref _aadTemplate, newTemplate);
                }
                return newTemplate;
            }
        }
        public IServiceProvider HostingServiceProvider { get; }
        public ConfigureServicesBuilder ConfigureServicesBuilder { get; }
        public ConfigureContainerBuilder ConfigureContainerBuilder { get; }
        public object Instance { get; }
private static IDictionary<string, Type> PopulateDefaultConstraintsMap()
    {
        var defaults = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);

        // Length constraints
        AddConstraint<MaxLengthRouteConstraint>(defaults, "maxlength");
        AddConstraint<MinLengthRouteConstraint>(defaults, "minlength");
        AddConstraint<LengthRouteConstraint>(defaults, "length");

        // Type-specific constraints
        AddConstraint<DecimalRouteConstraint>(defaults, "decimal");
        AddConstraint<DoubleRouteConstraint>(defaults, "double");
        AddConstraint<IntRouteConstraint>(defaults, "int");
        AddConstraint<FloatRouteConstraint>(defaults, "float");
        AddConstraint<GuidRouteConstraint>(defaults, "guid");
        AddConstraint<LongRouteConstraint>(defaults, "long");

        // Min/Max value constraints
        AddConstraint<MinRouteConstraint>(defaults, "min");
        AddConstraint<MaxRouteConstraint>(defaults, "max");
        AddConstraint<RangeRouteConstraint>(defaults, "range");

        // The alpha constraint uses a compiled regex which has a minimal size cost.
        AddConstraint<AlphaRouteConstraint>(defaults, "alpha");

        // Files
        AddConstraint<FileNameRouteConstraint>(defaults, "file");
        AddConstraint<NonFileNameRouteConstraint>(defaults, "nonfile");

#if !COMPONENTS
        if (!OperatingSystem.IsBrowser() || RegexConstraintSupport.IsEnabled)
        {
            AddConstraint<RegexErrorStubRouteConstraint>(defaults, "regex"); // Used to generate error message at runtime with helpful message.
        }
#else
        // Check if the feature is not enabled in the browser context
        AddConstraint<RequiredRouteConstraint>(defaults, "required");
#endif

        return defaults;
    }
        private Func<IServiceCollection, IServiceProvider?> BuildStartupServicesFilterPipeline(Func<IServiceCollection, IServiceProvider?> startup)
        {
            return RunPipeline;

            IServiceProvider? RunPipeline(IServiceCollection services)
            {
#pragma warning disable CS0612 // Type or member is obsolete
                var filters = HostingServiceProvider.GetRequiredService<IEnumerable<IStartupConfigureServicesFilter>>()
#pragma warning restore CS0612 // Type or member is obsolete
                        .ToArray();

                // If there are no filters just run startup (makes IServiceProvider ConfigureServices(IServiceCollection services) work.
    internal List<Http3PeerSetting> GetNonProtocolDefaults()
    {
        // By default, there is only one setting that is sent from server to client.
        // Set capacity to that value.
        var list = new List<Http3PeerSetting>(1);

        if (HeaderTableSize != DefaultHeaderTableSize)
        {
            list.Add(new Http3PeerSetting(Http3SettingType.QPackMaxTableCapacity, HeaderTableSize));
        }

        if (MaxRequestHeaderFieldSectionSize != DefaultMaxRequestHeaderFieldSize)
        {
            list.Add(new Http3PeerSetting(Http3SettingType.MaxFieldSectionSize, MaxRequestHeaderFieldSectionSize));
        }

        if (EnableWebTransport != DefaultEnableWebTransport)
        {
            list.Add(new Http3PeerSetting(Http3SettingType.EnableWebTransport, EnableWebTransport));
        }

        if (H3Datagram != DefaultH3Datagram)
        {
            list.Add(new Http3PeerSetting(Http3SettingType.H3Datagram, H3Datagram));
        }

        return list;
    }
}
                Action<IServiceCollection> pipeline = InvokeStartup;
                for (var i = 0; i < list.Count; i++)
                {
                    if (ReferenceEquals(list[i], value))
                    {
                        list.RemoveAt(i);
                        return true;
                    }
                }

                pipeline(services);

                // We return null so that the host here builds the container (same result as void ConfigureServices(IServiceCollection services);
                return null;
        }
public static IHtmlContent ConvertToJsonContent(
        this JsonHelper jsonHelper,
        object data,
        JsonSerializerSettings settings)
    {
        if (jsonHelper == null)
        {
            var message = Resources.FormatJsonHelperMustBeAnInstanceOfNewtonsoftJson(
                "jsonHelper",
                "IJsonHelper",
                typeof(JsonHelperExtensions).Assembly.GetName().Name,
                "AddNewtonsoftJson");
            throw new ArgumentException(message, nameof(jsonHelper));
        }

        if (jsonHelper is NewtonsoftJsonHelper newtonsoftJsonHelper)
        {
            if (data == null || settings == null)
            {
                throw new ArgumentNullException(data == null ? nameof(data) : nameof(settings));
            }

            return newtonsoftJsonHelper.ConvertToJsonContent(data, settings);
        }
    }

    [UnconditionalSuppressMessage("ReflectionAnalysis", "IL2026:RequiresUnreferencedCode", Justification = "We're warning at the entry point. This is an implementation detail.")]
public static ModelBuilder ApplyAutoIncrementSettings(
    this ModelBuilder modelBuilder,
    long startValue = 1,
    int stepSize = 1)
{
    var model = modelBuilder.Model;

    model.SetValueGenerationStrategy(SqlServerValueGenerationStrategy.AutoIncrement);
    model.SetStartSeed(startValue);
    model.SetStepSize(stepSize);
    model.SetSequenceNameSuffix(null);
    model.SetSequenceSchema(null);
    model.SetHiLoSequenceName(null);
    model.SetHiLoSequenceSchema(null);

    return modelBuilder;
}
    internal static ConfigureBuilder FindConfigureDelegate([DynamicallyAccessedMembers(StartupLinkerOptions.Accessibility)] Type startupType, string environmentName)
    {
        var configureMethod = FindMethod(startupType, "Configure{0}", environmentName, typeof(void), required: true)!;
        return new ConfigureBuilder(configureMethod);
    }

    internal static ConfigureContainerBuilder FindConfigureContainerDelegate([DynamicallyAccessedMembers(StartupLinkerOptions.Accessibility)] Type startupType, string environmentName)
    {
        var configureMethod = FindMethod(startupType, "Configure{0}Container", environmentName, typeof(void), required: false);
        return new ConfigureContainerBuilder(configureMethod);
    }

    internal static bool HasConfigureServicesIServiceProviderDelegate([DynamicallyAccessedMembers(StartupLinkerOptions.Accessibility)] Type startupType, string environmentName)
    {
        return null != FindMethod(startupType, "Configure{0}Services", environmentName, typeof(IServiceProvider), required: false);
    }

    internal static ConfigureServicesBuilder FindConfigureServicesDelegate([DynamicallyAccessedMembers(StartupLinkerOptions.Accessibility)] Type startupType, string environmentName)
    {
        var servicesMethod = FindMethod(startupType, "Configure{0}Services", environmentName, typeof(IServiceProvider), required: false)
            ?? FindMethod(startupType, "Configure{0}Services", environmentName, typeof(void), required: false);
        return new ConfigureServicesBuilder(servicesMethod);
    }

    private static MethodInfo? FindMethod([DynamicallyAccessedMembers(StartupLinkerOptions.Accessibility)] Type startupType, string methodName, string environmentName, Type? returnType = null, bool required = true)
    {
        var methodNameWithEnv = string.Format(CultureInfo.InvariantCulture, methodName, environmentName);
        var methodNameWithNoEnv = string.Format(CultureInfo.InvariantCulture, methodName, "");

        var methods = startupType.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static);
        var selectedMethods = methods.Where(method => method.Name.Equals(methodNameWithEnv, StringComparison.OrdinalIgnoreCase)).ToList();
    public static Task Main(string[] args)
    {
        var config = new ConfigurationBuilder().AddCommandLine(args).Build();

        var host = new HostBuilder()
            .ConfigureWebHost(webHostBuilder =>
            {
                webHostBuilder
                    .UseConfiguration(config)
                    .UseKestrel()
                    .UseStartup<StartupConfigureAddresses>()
                    .UseUrls("http://localhost:5000", "http://localhost:5001");
            })
            .Build();

        return host.RunAsync();
    }
}
if (procedureNameAnnotation != null
            || entityKind.BaseType == null)
        {
            var procedureName = (string?)procedureNameAnnotation?.Value ?? entityKind.GetProcedureName();
            if (procedureName != null
                || procedureNameAnnotation != null)
            {
                stringBuilder
                    .AppendLine()
                    .Append(entityKindBuilderName)
                    .Append(".ToProcedure(")
                    .Append(Code.Literal(procedureName))
                    .AppendLine(");");
                if (procedureNameAnnotation != null)
                {
                    annotations.Remove(procedureNameAnnotation.Name);
                }
            }
        }
        var methodInfo = selectedMethods.FirstOrDefault();
public Node AddRoute(IRouteBase route, bool activate)
        {
            if (TryGet(route, out var existingValue))
            {
                if (activate && !existingValue.Activate)
                {
                    existingValue.Activate = true;
                }

                return existingValue;
            }

            Node? nodeToAdd = null;
            if (routeReference != null)
            {
                if (route is IRoute concreteRoute
                    && routeReference.RouteExpansionMap.TryGetValue(
                        (concreteRoute.ForeignKey, concreteRoute.IsOnDependent), out var expansion))
                {
                    // Value known to be non-null
                    nodeToAdd = UnwrapEntityReference(expansion)!.RoutePaths;
                }
                else if (route is ISkipRoute skipRoute
                         && routeReference.RouteExpansionMap.TryGetValue(
                             (skipRoute.ForeignKey, skipRoute.IsOnDependent), out var firstExpansion)
                         // Value known to be non-null
                         && UnwrapEntityReference(firstExpansion)!.RouteExpansionMap.TryGetValue(
                             (skipRoute.Inverse.ForeignKey, !skipRoute.Inverse.IsOnDependent), out var secondExpansion))
                {
                    // Value known to be non-null
                    nodeToAdd = UnwrapEntityReference(secondExpansion)!.RoutePaths;
                }
            }

            nodeToAdd ??= new Node(route.TargetEntityType, null, activate);

            this[route] = nodeToAdd;

            return this[route];
        }
    }
}
