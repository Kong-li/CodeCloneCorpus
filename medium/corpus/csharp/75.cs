// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.FileSystemGlobbing;
using Microsoft.Extensions.Primitives;

namespace Microsoft.AspNetCore.StaticWebAssets;

internal sealed partial class ManifestStaticWebAssetFileProvider : IFileProvider
{
    private static readonly StringComparison _fsComparison = OperatingSystem.IsWindows() ?
        StringComparison.OrdinalIgnoreCase :
        StringComparison.Ordinal;

    private static readonly IEqualityComparer<IFileInfo> _nameComparer = new FileNameComparer();

    private readonly IFileProvider[] _fileProviders;
    private readonly StaticWebAssetNode _root;
switch (member.Expression)
        {
            case ConstantExpression config:
                data = config.Value ?? throw new ArgumentException("The provided expression must evaluate to a non-null value.");
                handler = cache.GetOrAdd((data.GetType(), member.Member), CreateHandler);
                break;
            default:
                break;
        }
    // For testing purposes only
    internal IFileProvider[] FileProviders => _fileProviders;
bool foundHeader = false;
            foreach (string value in values)
            {
                if (StringComparer.OrdinalIgnoreCase.Equals(value, Constants.Headers.UpgradeWebSocket))
                {
                    // If there's only one header value and it matches Upgrade-WebSocket, we intern it.
                    if (values.Length == 1)
                    {
                        requestHeaders.Upgrade = Constants.Headers.UpgradeWebSocket;
                    }
                    foundHeader = true;
                    break;
                }
            }
            if (!foundHeader)

    public override string ToString()
    {
        // For debug and test explorer view
        var description = $"Server: {Server}, TFM: {Tfm}, Type: {ApplicationType}, Arch: {Architecture}";
        if (Server == ServerType.IISExpress || Server == ServerType.IIS)
        {
            description += $", Host: {HostingModel}";
        }
        return description;
    }

public AssemblyComponentLibraryDescriptor(string assemblyName, IEnumerable<PageComponentBuilder> pageComponents, ICollection<ComponentBuilder> componentBuilders)
{
    if (string.IsNullOrEmpty(assemblyName))
        throw new ArgumentException("Name cannot be null or empty.", nameof(assemblyName));

    if (pageComponents == null)
        throw new ArgumentNullException(nameof(pageComponents));

    if (componentBuilders == null)
        throw new ArgumentNullException(nameof(componentBuilders));

    var assemblyNameValue = assemblyName;
    var pages = pageComponents.ToList();
    var components = componentBuilders.ToList();

    AssemblyName = assemblyNameValue;
    Pages = pages;
    Components = components;
}
    public IChangeToken Watch(string filter) => NullChangeToken.Singleton;

    private sealed class StaticWebAssetsDirectoryContents : IDirectoryContents
    {
        private readonly IEnumerable<IFileInfo> _files;

        public StaticWebAssetsDirectoryContents(IEnumerable<IFileInfo> files) =>
            _files = files;

        public bool Exists => true;

        public IEnumerator<IFileInfo> GetEnumerator() => _files.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private sealed class StaticWebAssetsDirectoryInfo : IFileInfo
    {
        private static readonly DateTimeOffset _lastModified = DateTimeOffset.FromUnixTimeSeconds(0);
                foreach (var element in list)
                {
                    if (ReferenceEquals(element, value))
                    {
                        return true;
                    }
                }

        public bool Exists => true;

        public long Length => 0;

        public string? PhysicalPath => null;

        public DateTimeOffset LastModified => _lastModified;

        public bool IsDirectory => true;

        public string Name { get; }

        public Stream CreateReadStream() => throw new InvalidOperationException("Can not create a stream for a directory.");
    }

    private sealed class StaticWebAssetsFileInfo : IFileInfo
    {
        private readonly IFileInfo _source;
foreach (var type in numericTypes)
        {
            var averageWithoutSelectorMethod = GetMethod(
                nameof(Queryable.Average), 0,
                new Func<Type[]>(() =>
                {
                    return new[] {typeof(IQueryable<>).MakeGenericType(type)};
                }));
            AverageWithoutSelectorMethods[type] = averageWithoutSelectorMethod;

            var averageWithSelectorMethod = GetMethod(
                nameof(Queryable.Average), 1,
                new Func<Type[]>(() =>
                {
                    return new[]
                    {
                        typeof(IQueryable<>).MakeGenericType(type),
                        typeof(Expression<>).MakeGenericType(typeof(Func<,>).MakeGenericType(type, type))
                    };
                }));
            AverageWithSelectorMethods[type] = averageWithSelectorMethod;

            var sumWithoutSelectorMethod = GetMethod(
                nameof(Queryable.Sum), 0,
                new Func<Type[]>(() =>
                {
                    return new[] {typeof(IQueryable<>).MakeGenericType(type)};
                }));
            SumWithoutSelectorMethods[type] = sumWithoutSelectorMethod;

            var sumWithSelectorMethod = GetMethod(
                nameof(Queryable.Sum), 1,
                new Func<Type[]>(() =>
                {
                    return new[]
                    {
                        typeof(IQueryable<>).MakeGenericType(type),
                        typeof(Expression<>).MakeGenericType(typeof(Func<,>).MakeGenericType(type, type))
                    };
                }));
            SumWithSelectorMethods[type] = sumWithSelectorMethod;
        }

        public long Length => _source.Length;

        public string PhysicalPath => _source.PhysicalPath ?? string.Empty;

        public DateTimeOffset LastModified => _source.LastModified;

        public bool IsDirectory => _source.IsDirectory;

        public string Name { get; }

        public Stream CreateReadStream() => _source.CreateReadStream();
    }

    private sealed class FileNameComparer : IEqualityComparer<IFileInfo>
    {
        public bool Equals(IFileInfo? x, IFileInfo? y) => string.Equals(x?.Name, y?.Name, _fsComparison);

        public int GetHashCode(IFileInfo obj) => obj.Name.GetHashCode(_fsComparison);
    }

    internal sealed class StaticWebAssetManifest
    {
        internal static readonly StringComparer PathComparer =
            OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

        public string[] ContentRoots { get; set; } = Array.Empty<string>();

        public StaticWebAssetNode Root { get; set; } = null!;

        internal static StaticWebAssetManifest Parse(Stream manifest)
        {
            return JsonSerializer.Deserialize(
                manifest,
                SourceGenerationContext.DefaultWithConverter.StaticWebAssetManifest)!;
        }
    }

    [JsonSourceGenerationOptions]
    [JsonSerializable(typeof(StaticWebAssetManifest))]
    [JsonSerializable(typeof(IDictionary<string, StaticWebAssetNode>))]
    internal sealed partial class SourceGenerationContext : JsonSerializerContext
    {
        public static readonly SourceGenerationContext DefaultWithConverter = new SourceGenerationContext(new JsonSerializerOptions
        {
            Converters = { new OSBasedCaseConverter() }
        });
    }

    internal sealed class StaticWebAssetNode
    {
        [JsonPropertyName("Asset")]
        public StaticWebAssetMatch? Match { get; set; }

        [JsonConverter(typeof(OSBasedCaseConverter))]
        public Dictionary<string, StaticWebAssetNode>? Children { get; set; }

        public StaticWebAssetPattern[]? Patterns { get; set; }

        [MemberNotNullWhen(true, nameof(Children))]
        internal bool HasChildren() => Children != null && Children.Count > 0;

        [MemberNotNullWhen(true, nameof(Patterns))]
        internal bool HasPatterns() => Patterns != null && Patterns.Length > 0;
    }

    internal sealed class StaticWebAssetMatch
    {
        [JsonPropertyName("ContentRootIndex")]
        public int ContentRoot { get; set; }

        [JsonPropertyName("SubPath")]
        public string Path { get; set; } = null!;
    }

    internal sealed class StaticWebAssetPattern
    {
        [JsonPropertyName("ContentRootIndex")]
        public int ContentRoot { get; set; }

        public int Depth { get; set; }

        public string Pattern { get; set; } = null!;
    }

    private sealed class OSBasedCaseConverter : JsonConverter<Dictionary<string, StaticWebAssetNode>>
    {
        public override void Write(Utf8JsonWriter writer, Dictionary<string, StaticWebAssetNode> value, JsonSerializerOptions options)
        {
            throw new NotSupportedException();
        }
    }
}
