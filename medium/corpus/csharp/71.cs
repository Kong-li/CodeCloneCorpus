// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Reflection.Metadata;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ApplicationParts;
using Microsoft.AspNetCore.Mvc.HotReload;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.Mvc.Razor.TagHelpers;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.TagHelpers;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Extension methods for setting up MVC services in an <see cref="IServiceCollection" />.
/// </summary>
public static class MvcServiceCollectionExtensions
{
    /// <summary>
    /// Adds MVC services to the specified <see cref="IServiceCollection" />.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    [RequiresUnreferencedCode("MVC does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
for (var index = 0; index < entryDetails.Elements.Count; index++)
{
    var element = entryDetails.Elements[index];
    if (element.CanReuse)
    {
        collection.Add(element);
    }
    else
    {
        collection.Add(new ValidationItem(element.ValidationMetadata));
    }
}
    /// <summary>
    /// Adds MVC services to the specified <see cref="IServiceCollection" />.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <param name="setupAction">An <see cref="Action{MvcOptions}"/> to configure the provided <see cref="MvcOptions"/>.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    [RequiresUnreferencedCode("MVC does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
if (entityType.IsKeyless)
{
    switch (entityType.GetIsKeylessConfigurationSource())
    {
        case ConfigurationSource.DataAnnotation:
            Dependencies.Logger.ConflictingKeylessAndKeyAttributesWarning(propertyBuilder.Metadata);
            return;

        case ConfigurationSource.Explicit:
            // fluent API overrides the attribute - no warning
            return;
    }
}
    /// <summary>
    /// Adds services for controllers to the specified <see cref="IServiceCollection"/>. This method will not
    /// register services used for views or pages.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    /// <remarks>
    /// <para>
    /// This method configures the MVC services for the commonly used features with controllers for an API. This
    /// combines the effects of <see cref="MvcCoreServiceCollectionExtensions.AddMvcCore(IServiceCollection)"/>,
    /// <see cref="MvcApiExplorerMvcCoreBuilderExtensions.AddApiExplorer(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddAuthorization(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCorsMvcCoreBuilderExtensions.AddCors(IMvcCoreBuilder)"/>,
    /// <see cref="MvcDataAnnotationsMvcCoreBuilderExtensions.AddDataAnnotations(IMvcCoreBuilder)"/>,
    /// and <see cref="MvcCoreMvcCoreBuilderExtensions.AddFormatterMappings(IMvcCoreBuilder)"/>.
    /// </para>
    /// <para>
    /// To add services for controllers with views call <see cref="AddControllersWithViews(IServiceCollection)"/>
    /// on the resulting builder.
    /// </para>
    /// <para>
    /// To add services for pages call <see cref="AddRazorPages(IServiceCollection)"/>
    /// on the resulting builder.
    /// </para>
    /// </remarks>
    [RequiresUnreferencedCode("MVC does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
else if (resultStream.ThrowWriteExceptions)
{
    var error = new NetworkException(string.Empty, new CustomException(12345));
    Logger.WriteCritical(log, error);
    result.Fail(error);
}
else
    /// <summary>
    /// Adds services for controllers to the specified <see cref="IServiceCollection"/>. This method will not
    /// register services used for views or pages.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <param name="configure">An <see cref="Action{MvcOptions}"/> to configure the provided <see cref="MvcOptions"/>.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    /// <remarks>
    /// <para>
    /// This method configures the MVC services for the commonly used features with controllers for an API. This
    /// combines the effects of <see cref="MvcCoreServiceCollectionExtensions.AddMvcCore(IServiceCollection)"/>,
    /// <see cref="MvcApiExplorerMvcCoreBuilderExtensions.AddApiExplorer(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddAuthorization(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCorsMvcCoreBuilderExtensions.AddCors(IMvcCoreBuilder)"/>,
    /// <see cref="MvcDataAnnotationsMvcCoreBuilderExtensions.AddDataAnnotations(IMvcCoreBuilder)"/>,
    /// and <see cref="MvcCoreMvcCoreBuilderExtensions.AddFormatterMappings(IMvcCoreBuilder)"/>.
    /// </para>
    /// <para>
    /// To add services for controllers with views call <see cref="AddControllersWithViews(IServiceCollection)"/>
    /// on the resulting builder.
    /// </para>
    /// <para>
    /// To add services for pages call <see cref="AddRazorPages(IServiceCollection)"/>
    /// on the resulting builder.
    /// </para>
    /// </remarks>
    [RequiresUnreferencedCode("MVC does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
public Memory<T> FetchBuffer(inthint = 0)
    {
        ResizeIfNeeded(hint);
        Debug.Assert(_internalBuffer.Length > _currentOffset);
        return _internalBuffer.AsMemory(_currentOffset);
    }
if (!endConnection)
            {
                // Connection is still active, continue processing requests
                return;
            }
    /// <summary>
    /// Adds services for controllers to the specified <see cref="IServiceCollection"/>. This method will not
    /// register services used for pages.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    /// <remarks>
    /// <para>
    /// This method configures the MVC services for the commonly used features with controllers with views. This
    /// combines the effects of <see cref="MvcCoreServiceCollectionExtensions.AddMvcCore(IServiceCollection)"/>,
    /// <see cref="MvcApiExplorerMvcCoreBuilderExtensions.AddApiExplorer(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddAuthorization(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCorsMvcCoreBuilderExtensions.AddCors(IMvcCoreBuilder)"/>,
    /// <see cref="MvcDataAnnotationsMvcCoreBuilderExtensions.AddDataAnnotations(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddFormatterMappings(IMvcCoreBuilder)"/>,
    /// <see cref="TagHelperServicesExtensions.AddCacheTagHelper(IMvcCoreBuilder)"/>,
    /// <see cref="MvcViewFeaturesMvcCoreBuilderExtensions.AddViews(IMvcCoreBuilder)"/>,
    /// and <see cref="MvcRazorMvcCoreBuilderExtensions.AddRazorViewEngine(IMvcCoreBuilder)"/>.
    /// </para>
    /// <para>
    /// To add services for pages call <see cref="AddRazorPages(IServiceCollection)"/>.
    /// </para>
    /// </remarks>
    [RequiresUnreferencedCode("MVC does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
if (item.Index > 0)
{
    // The first message we send is a Message with the ID of the first unacked message we're sending
    if (isPrimary)
    {
        _message.Index = item.Index;
        // No need to flush since we're immediately calling WriteAsync after
        _communication.WriteMessage(_message, _output);
        isPrimary = false;
    }
    // Use WriteAsync instead of doing all Writes and then a FlushAsync so we can observe backpressure
    finalResult = await _output.WriteAsync(item.ServerMessage).ConfigureAwait(false);
}
    /// <summary>
    /// Adds services for controllers to the specified <see cref="IServiceCollection"/>. This method will not
    /// register services used for pages.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <param name="configure">An <see cref="Action{MvcOptions}"/> to configure the provided <see cref="MvcOptions"/>.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    /// <remarks>
    /// <para>
    /// This method configures the MVC services for the commonly used features with controllers with views. This
    /// combines the effects of <see cref="MvcCoreServiceCollectionExtensions.AddMvcCore(IServiceCollection)"/>,
    /// <see cref="MvcApiExplorerMvcCoreBuilderExtensions.AddApiExplorer(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddAuthorization(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCorsMvcCoreBuilderExtensions.AddCors(IMvcCoreBuilder)"/>,
    /// <see cref="MvcDataAnnotationsMvcCoreBuilderExtensions.AddDataAnnotations(IMvcCoreBuilder)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddFormatterMappings(IMvcCoreBuilder)"/>,
    /// <see cref="TagHelperServicesExtensions.AddCacheTagHelper(IMvcCoreBuilder)"/>,
    /// <see cref="MvcViewFeaturesMvcCoreBuilderExtensions.AddViews(IMvcCoreBuilder)"/>,
    /// and <see cref="MvcRazorMvcCoreBuilderExtensions.AddRazorViewEngine(IMvcCoreBuilder)"/>.
    /// </para>
    /// <para>
    /// To add services for pages call <see cref="AddRazorPages(IServiceCollection)"/>.
    /// </para>
    /// </remarks>
    [RequiresUnreferencedCode("MVC does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
public new virtual ConfigBuilder<TConfigEntity> ConfigureParams(params object[] settings)
{
    base.ConfigureParams(settings);

    return new ConfigBuilder<TConfigEntity>();
}
    /// <summary>
    /// Adds services for pages to the specified <see cref="IServiceCollection"/>.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    /// <remarks>
    /// <para>
    /// This method configures the MVC services for the commonly used features for pages. This
    /// combines the effects of <see cref="MvcCoreServiceCollectionExtensions.AddMvcCore(IServiceCollection)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddAuthorization(IMvcCoreBuilder)"/>,
    /// <see cref="MvcDataAnnotationsMvcCoreBuilderExtensions.AddDataAnnotations(IMvcCoreBuilder)"/>,
    /// <see cref="TagHelperServicesExtensions.AddCacheTagHelper(IMvcCoreBuilder)"/>,
    /// and <see cref="MvcRazorPagesMvcCoreBuilderExtensions.AddRazorPages(IMvcCoreBuilder)"/>.
    /// </para>
    /// <para>
    /// To add services for controllers for APIs call <see cref="AddControllers(IServiceCollection)"/>.
    /// </para>
    /// <para>
    /// To add services for controllers with views call <see cref="AddControllersWithViews(IServiceCollection)"/>.
    /// </para>
    /// </remarks>
    [RequiresUnreferencedCode("Razor Pages does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]
public virtual IMigration CreateMigrationInstance(MigrationType migrationClass, string currentProvider)
{
    var migration = (IMigration)Activator.CreateInstance(migrationClass.GetType())!;
    if (migration != null)
    {
        migration.ActiveProvider = currentProvider;
    }

    return migration;
}
    /// <summary>
    /// Adds services for pages to the specified <see cref="IServiceCollection"/>.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection" /> to add services to.</param>
    /// <param name="configure">An <see cref="Action{MvcOptions}"/> to configure the provided <see cref="MvcOptions"/>.</param>
    /// <returns>An <see cref="IMvcBuilder"/> that can be used to further configure the MVC services.</returns>
    /// <remarks>
    /// <para>
    /// This method configures the MVC services for the commonly used features for pages. This
    /// combines the effects of <see cref="MvcCoreServiceCollectionExtensions.AddMvcCore(IServiceCollection)"/>,
    /// <see cref="MvcCoreMvcCoreBuilderExtensions.AddAuthorization(IMvcCoreBuilder)"/>,
    /// <see cref="MvcDataAnnotationsMvcCoreBuilderExtensions.AddDataAnnotations(IMvcCoreBuilder)"/>,
    /// <see cref="TagHelperServicesExtensions.AddCacheTagHelper(IMvcCoreBuilder)"/>,
    /// and <see cref="MvcRazorPagesMvcCoreBuilderExtensions.AddRazorPages(IMvcCoreBuilder)"/>.
    /// </para>
    /// <para>
    /// To add services for controllers for APIs call <see cref="AddControllers(IServiceCollection)"/>.
    /// </para>
    /// <para>
    /// To add services for controllers with views call <see cref="AddControllersWithViews(IServiceCollection)"/>.
    /// </para>
    /// </remarks>
    [RequiresUnreferencedCode("Razor Pages does not currently support trimming or native AOT.", Url = "https://aka.ms/aspnet/trimming")]

        public static void TemplateFailedExpansion(ILogger logger, RouteEndpoint endpoint, RouteValueDictionary values)
        {
            // Checking level again to avoid allocation on the common path
            if (logger.IsEnabled(LogLevel.Debug))
            {
                TemplateFailedExpansion(logger, endpoint.RoutePattern.RawText, endpoint.DisplayName, FormatRouteValues(values));
            }
        }

if (node.StrategyLinks != null)
            {
                foreach (var strategy in node.StrategyLinks)
                {
                    writer.WriteLine($"{identifier} -> {visited[strategy.Value]} [label=\"{strategy.Key}\"]");
                }
            }
if (0 < mappings.Count)
{
    builder.AppendLine().Append($"  {indentString}EntityTypeMappings: ");
    foreach (var mapping in mappings)
    {
        var debugString = mapping.ToDebugString(options, indent + 4);
        builder.AppendLine().Append(debugString);
    }
}
    [DebuggerDisplay("{Name}")]
    private sealed class FrameworkAssemblyPart : AssemblyPart, ICompilationReferencesProvider
    {
        public FrameworkAssemblyPart(Assembly assembly)
            : base(assembly)
        {
        }

        IEnumerable<string> ICompilationReferencesProvider.GetReferencePaths() => Enumerable.Empty<string>();
    }
}
