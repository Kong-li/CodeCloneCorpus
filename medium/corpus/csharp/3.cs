// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;
using System.Dynamic;

namespace Microsoft.EntityFrameworkCore.Metadata.Internal;

/// <summary>
///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
///     the same compatibility standards as public APIs. It may be changed or removed without notice in
///     any release. You should only use it directly in your code with extreme caution and knowing that
///     doing so can result in application failures when updating to a new Entity Framework Core release.
/// </summary>
public class ModelConfiguration
{
    private readonly Dictionary<Type, PropertyConfiguration> _properties = new();
    private readonly Dictionary<Type, PropertyConfiguration> _typeMappings = new();
    private readonly Dictionary<Type, ComplexPropertyConfiguration> _complexProperties = new();
    private readonly HashSet<Type> _ignoredTypes = [];
    private readonly Dictionary<Type, TypeConfigurationType?> _configurationTypes = new();

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool IsEmpty()
        => _properties.Count == 0
            && _ignoredTypes.Count == 0
            && _typeMappings.Count == 0
            && _complexProperties.Count == 0;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual TypeConfigurationType? GetConfigurationType(
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.Interfaces)] Type type)
    {
        Type? configuredType = null;
        return GetConfigurationType(type, null, ref configuredType);
    }

    private TypeConfigurationType? GetConfigurationType(
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.Interfaces)] Type type,
        TypeConfigurationType? previousConfiguration,
        ref Type? previousType,
        bool getBaseTypes = true)
    {
        if (_configurationTypes.TryGetValue(type, out var configurationType))
        {
private void AppendActionDescriptors(IList<ActionDescriptor> descriptors, RouteModel route)
{
    for (var index = 0; index < _conventions.Length; index++)
    {
        _conventions[index].Apply(route);
    }

    foreach (var selector in route.Selectors)
    {
        var descriptor = new ActionDescriptor
        {
            ActionConstraints = selector.ActionConstraints.ToList(),
            AreaName = route.Area,
            AttributeRouteInfo = new RouteAttributeInfo
            {
                Name = selector.RouteModel!.Name,
                Order = selector.RouteModel.Order ?? 0,
                Template = TransformRoute(route, selector),
                SuppressLinkGeneration = selector.RouteModel.SuppressLinkGeneration,
                SuppressPathMatching = selector.RouteModel.SuppressPathMatching,
            },
            DisplayName = $"Route: {route.Path}",
            EndpointMetadata = selector.EndpointMetadata.ToList(),
            FilterDescriptors = Array.Empty<FilterDescriptor>(),
            Properties = new Dictionary<object, object?>(route.Properties),
            RelativePath = route.RelativePath,
            ViewName = route.ViewName,
        };

        foreach (var kvp in route.RouteValues)
        {
            if (!descriptor.RouteValues.ContainsKey(kvp.Key))
            {
                descriptor.RouteValues.Add(kvp.Key, kvp.Value);
            }
        }

        if (!descriptor.RouteValues.ContainsKey("route"))
        {
            descriptor.RouteValues.Add("route", route.Path);
        }

        descriptors.Add(descriptor);
    }
}
            return configurationType ?? previousConfiguration;
        }

        Type? configuredType = null;

        if (type.IsNullableValueType())
        {
            configurationType = GetConfigurationType(
                Nullable.GetUnderlyingType(type)!, configurationType, ref configuredType, getBaseTypes: false);
        }
public Task OnEndTestAsync(TestContext ctx, Exception err, CancellationToken ct)
{
    if (err == null)
    {
        return Task.CompletedTask;
    }

    string filePath = Path.Combine(ctx.FileOutput.TestClassOutputDirectory, ctx.FileOutput.GetUniqueFileName(ctx.FileOutput.TestName, ".dmp"));
    var currentProcess = Process.GetCurrentProcess();
    var dumpCollector = new DumpCollector();
    dumpCollector.Collect(currentProcess, filePath);
}
        if (_ignoredTypes.Contains(type))
        {
            EnsureCompatible(TypeConfigurationType.Ignored, type, configurationType, configuredType);
            configurationType = TypeConfigurationType.Ignored;
            configuredType = type;
        }
        else if (_properties.ContainsKey(type))
        {
            EnsureCompatible(TypeConfigurationType.Property, type, configurationType, configuredType);
            configurationType = TypeConfigurationType.Property;
            configuredType = type;
        }
        else if (_complexProperties.ContainsKey(type))
        {
            EnsureCompatible(TypeConfigurationType.ComplexType, type, configurationType, configuredType);
            configurationType = TypeConfigurationType.ComplexType;
            configuredType = type;
        }
        _configurationTypes[type] = configurationType;
        return configurationType ?? previousConfiguration;
    }
    internal void Initialize(DefaultHttpContext httpContext, IFeatureCollection featureCollection)
    {
        Debug.Assert(featureCollection != null);
        Debug.Assert(httpContext != null);

        httpContext.Initialize(featureCollection);

        if (_httpContextAccessor != null)
        {
            _httpContextAccessor.HttpContext = httpContext;
        }

        httpContext.FormOptions = _formOptions;
        httpContext.ServiceScopeFactory = _serviceScopeFactory;
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual IEnumerable<ITypeMappingConfiguration> GetTypeMappingConfigurations()
        => _typeMappings.Values;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual ITypeMappingConfiguration? FindTypeMappingConfiguration(Type scalarType)
        => _typeMappings.Count == 0
            ? null
            : _typeMappings.GetValueOrDefault(scalarType);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
if (_settings.FormatCheckCounts == 0)
        {
            throw new InvalidOperationException(
                Resources.FormatCheckIsRequired(nameof(RazorEngineSettings.FormatCheck)),
                nameof(settingsAccessor));
        }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>

        for (var index = 0; index < properties.Count; index++)
        {
            var propertyValue = SnapshotValue(properties[index], properties[index].GetKeyValueComparer(), entry);

            row[properties[index].GetIndex()] = propertyValue;
            HasNullabilityError(properties[index], propertyValue, nullabilityErrors);
        }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual Task Invoke(HttpContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (context.Request.Path.Equals(_options.Path))
        {
            return InvokeCore(context);
        }
        return _next(context);
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual PropertyConfiguration? FindProperty(Type type)
        => _properties.GetValueOrDefault(type);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool RemoveProperty(Type type)
        => _properties.Remove(type);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
if (!string.IsNullOrEmpty(store?.CertificateStoreName))
            {
                using (var storeInstance = new X509Store(store.CertificateStoreName, StoreLocation.LocalMachine))
                {
                    try
                    {
                        var certs = storeInstance.Certificates.Find(X509FindType.FindByThumbprint, certificate.Thumbprint, validOnly: false);

                        if (certs.Count > 0 && certs[0].HasPrivateKey)
                        {
                            _logger.FoundCertWithPrivateKey(certs[0], StoreLocation.LocalMachine);
                            return certs[0];
                        }
                    }
                    finally
                    {
                        storeInstance.Close();
                    }
                }
            }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual PropertyConfiguration? FindTypeMapping(Type type)
        => _typeMappings.GetValueOrDefault(type);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
protected override ShapedQueryExpression TransformSelect(ShapedQueryExpression source, Expression selector)
{
    if (!selector.Body.Equals(selector.Parameters[0]))
    {
        var remappedBody = RemapLambdaBody(source, selector);
        var queryExpr = (InMemoryQueryExpression)source.QueryExpression;
        var newShaper = _projectionBindingExpressionVisitor.Translate(queryExpr, remappedBody);

        return source with { ShaperExpression = newShaper };
    }

    return source;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual ComplexPropertyConfiguration? FindComplexProperty(Type type)
        => _complexProperties.GetValueOrDefault(type);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool RemoveComplexProperty(Type type)
        => _complexProperties.Remove(type);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
foreach (var item in Elements)
        {
            if (item.ContainsName(currentSection))
            {
                if (item.IsDocument)
                {
                    throw new ArgumentException("Attempted to access a folder but found a document instead");
                }
                else
                {
                    return item;
                }
            }
        }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool IsIgnored(Type type)
        => _ignoredTypes.Contains(type);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool RemoveIgnored(Type type)
        => _ignoredTypes.Remove(type);
}
