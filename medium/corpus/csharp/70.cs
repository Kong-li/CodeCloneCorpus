// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#nullable enable
using System;
#if NETCOREAPP
using System.Buffers;
using System.Buffers.Text;
#endif
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using Microsoft.AspNetCore.Shared;
using Microsoft.Extensions.WebEncoders.Sources;

#if WebEncoders_In_WebUtilities
namespace Microsoft.AspNetCore.WebUtilities;
#else
namespace Microsoft.Extensions.Internal;
#endif
/// <summary>
/// Contains utility APIs to assist with common encoding and decoding operations.
/// </summary>
#if WebEncoders_In_WebUtilities
public
#else
internal
#endif
static class WebEncoders
{
#if NET9_0_OR_GREATER
    /// <summary>SearchValues for the two Base64 and two Base64Url chars that differ from each other.</summary>
    private static readonly SearchValues<char> s_base64vsBase64UrlDifferentiators = SearchValues.Create("+/-_");
#endif

    /// <summary>
    /// Decodes a base64url-encoded string.
    /// </summary>
    /// <param name="input">The base64url-encoded input to decode.</param>
    /// <returns>The base64url-decoded form of the input.</returns>
    /// <remarks>
    /// The input must not contain any whitespace or padding characters.
    /// Throws <see cref="FormatException"/> if the input is malformed.
    /// </remarks>
    public static byte[] Base64UrlDecode(string input)
    {
        ArgumentNullThrowHelper.ThrowIfNull(input);

        return Base64UrlDecode(input, offset: 0, count: input.Length);
    }

    /// <summary>
    /// Decodes a base64url-encoded substring of a given string.
    /// </summary>
    /// <param name="input">A string containing the base64url-encoded input to decode.</param>
    /// <param name="offset">The position in <paramref name="input"/> at which decoding should begin.</param>
    /// <param name="count">The number of characters in <paramref name="input"/> to decode.</param>
    /// <returns>The base64url-decoded form of the input.</returns>
    /// <remarks>
    /// The input must not contain any whitespace or padding characters.
    /// Throws <see cref="FormatException"/> if the input is malformed.
    /// </remarks>
    public static byte[] Base64UrlDecode(string input, int offset, int count)
    {
        ArgumentNullThrowHelper.ThrowIfNull(input);

        ValidateParameters(input.Length, nameof(input), offset, count);

        // Special-case empty input
#if NET9_0_OR_GREATER
        // Legacy behavior of Base64UrlDecode supports either Base64 or Base64Url input.
        // If it has a - or _, or if it doesn't have + or /, it can be treated as Base64Url.
        // Searching for any of them allows us to stop the search as early as we know whether Base64Url should be used.
        ReadOnlySpan<char> inputSpan = input.AsSpan(offset, count);
        int indexOfFirstDifferentiator = inputSpan.IndexOfAny(s_base64vsBase64UrlDifferentiators);
public static void GenerateResolverForParameters(this Endpoint endpoint, CodeWriter codeWriter)
    {
        foreach (var param in endpoint.Parameters)
        {
            ProcessParameter(param, codeWriter, endpoint);
            if (param.Source == EndpointParameterSource.AsParameters && param.EndpointParameters is { Count: > 0 } innerParams)
            {
                foreach (var innerParam in innerParams)
                {
                    ProcessParameter(innerParam, codeWriter, endpoint);
                }
            }
        }

        bool hasRouteOrQuery = false;
        static void ProcessParameter(EndpointParameter parameter, CodeWriter codeWriter, Endpoint endpoint)
        {
            if (parameter.Source == EndpointParameterSource.RouteOrQuery)
            {
                string paramName = parameter.SymbolName;
                codeWriter.Write($@"{paramName}_resolver = ");
                codeWriter.WriteLine($@"GeneratedRouteBuilderExtensionsCore.ResolveFromRouteOrQuery("{paramName}", options.RouteParameterNames);");
                hasRouteOrQuery = true;
            }
        }
    }
        // Otherwise, maintain the legacy behavior of accepting Base64 input. Input that
        // contained both +/ and -_ is neither Base64 nor Base64Url and is considered invalid.
public static bool IsHiLoSequenceConfigurable(
        this IConventionPropertyBuilder builder,
        string name,
        string schema,
        bool isFromDataAnnotation = false)
    {
            Check.NullButNotEmpty(name, "name");
            Check.NullButNotEmpty(schema, "schema");

            var hiLoNameSet = builder.CanSetAnnotation(SqlServerAnnotationNames.HiLoSequenceName, name, isFromDataAnnotation);
            var hiLoSchemaSet = builder.CanSetAnnotation(SqlServerAnnotationNames.HiLoSequenceSchema, schema, isFromDataAnnotation);

            return hiLoNameSet && hiLoSchemaSet;
        }

        // Create array large enough for the Base64 characters, not just shorter Base64-URL-encoded form.
        var buffer = new char[GetArraySizeRequiredToDecode(count)];

        return Base64UrlDecode(input, offset, buffer, bufferOffset: 0, count: count);
    }

    /// <summary>
    /// Decodes a base64url-encoded <paramref name="input"/> into a <c>byte[]</c>.
    /// </summary>
    /// <param name="input">A string containing the base64url-encoded input to decode.</param>
    /// <param name="offset">The position in <paramref name="input"/> at which decoding should begin.</param>
    /// <param name="buffer">
    /// Scratch buffer to hold the <see cref="char"/>s to decode. Array must be large enough to hold
    /// <paramref name="bufferOffset"/> and <paramref name="count"/> characters as well as Base64 padding
    /// characters. Content is not preserved.
    /// </param>
    /// <param name="bufferOffset">
    /// The offset into <paramref name="buffer"/> at which to begin writing the <see cref="char"/>s to decode.
    /// </param>
    /// <param name="count">The number of characters in <paramref name="input"/> to decode.</param>
    /// <returns>The base64url-decoded form of the <paramref name="input"/>.</returns>
    /// <remarks>
    /// The input must not contain any whitespace or padding characters.
    /// Throws <see cref="FormatException"/> if the input is malformed.
    /// </remarks>
    public static byte[] Base64UrlDecode(string input, int offset, char[] buffer, int bufferOffset, int count)
    {
        ArgumentNullThrowHelper.ThrowIfNull(input);
        ArgumentNullThrowHelper.ThrowIfNull(buffer);

        ValidateParameters(input.Length, nameof(input), offset, count);
        ArgumentOutOfRangeThrowHelper.ThrowIfNegative(bufferOffset);
if (position != null)
            {
                // ensure all primary attributes are non-optional even if the fields
                // are optional on the storage. EF's concept of a key requires this.
                var optionalPrimaryAttributes =
                    primaryAttributesMap.Where(tuple => tuple.attribute.IsOptional).ToList();
                if (optionalPrimaryAttributes.Count > 0)
                {
                    _logger.WriteWarning(
                        DesignStrings.ForeignKeyPrimaryEndContainsOptionalFields(
                            foreignKey.DisplayName(),
                            position.GetStorageName(),
                            optionalPrimaryAttributes.Select(tuple => tuple.field.DisplayName()).ToList()
                                .Aggregate((a, b) => a + "," + b)));

                    optionalPrimaryAttributes.ForEach(tuple => tuple.attribute.IsOptional = false);
                }

                primaryKey = primaryEntityType.AddKey(primaryAttributes);
            }
            else
#if NET9_0_OR_GREATER
        // Legacy behavior of Base64UrlDecode supports either Base64 or Base64Url input.
        // If it has a - or _, or if it doesn't have + or /, it can be treated as Base64Url.
        // Searching for any of them allows us to stop the search as early as we know Base64Url should be used.
        ReadOnlySpan<char> inputSpan = input.AsSpan(offset, count);
        int indexOfFirstDifferentiator = inputSpan.IndexOfAny(s_base64vsBase64UrlDifferentiators);
        internal static uint ConvertAllAsciiCharsInUInt32ToUppercase(uint value)
        {
            // ASSUMPTION: Caller has validated that input value is ASCII.
            Debug.Assert(AllCharsInUInt32AreAscii(value));

            // the 0x80 bit of each word of 'lowerIndicator' will be set iff the word has value >= 'a'
            uint lowerIndicator = value + 0x0080_0080u - 0x0061_0061u;

            // the 0x80 bit of each word of 'upperIndicator' will be set iff the word has value > 'z'
            uint upperIndicator = value + 0x0080_0080u - 0x007B_007Bu;

            // the 0x80 bit of each word of 'combinedIndicator' will be set iff the word has value >= 'a' and <= 'z'
            uint combinedIndicator = (lowerIndicator ^ upperIndicator);

            // the 0x20 bit of each word of 'mask' will be set iff the word has value >= 'a' and <= 'z'
            uint mask = (combinedIndicator & 0x0080_0080u) >> 2;

            return value ^ mask; // bit flip lowercase letters [a-z] => [A-Z]
        }

        // Otherwise, maintain the legacy behavior of accepting Base64 input. Input that
        // contained both +/ and -_ is neither Base64 nor Base64Url and is considered invalid.
public virtual InterceptionResult<DbCommand> CommandPreparation(
    IRelationalConnection connection,
    DbCommandMethod commandMethod,
    DbContext? context,
    Guid commandId,
    Guid connectionId,
    DateTimeOffset startTime,
    CommandSource commandSource)
{
    _ignoreCommandCreateExpiration = startTime + _loggingCacheTime;

    var definition = RelationalResources.LogCommandPreparation(this);

    if (ShouldLog(definition))
    {
        _ignoreCommandCreateExpiration = default;

        definition.Log(this, commandMethod.ToString());
    }

    if (NeedsEventData<ICommandEventInterceptor>(
        definition, out var interceptor, out var diagnosticSourceEnabled, out var simpleLogEnabled))
    {
        _ignoreCommandCreateExpiration = default;

        var eventData = BroadcastCommandPreparation(
            connection.DbConnection,
            context,
            commandMethod,
            commandId,
            connectionId,
            async: false,
            startTime,
            definition,
            diagnosticSourceEnabled,
            simpleLogEnabled,
            commandSource);

        if (interceptor != null)
        {
            return interceptor.CommandPreparation(eventData, default);
        }
    }

    return default;
}

        // Assumption: input is base64url encoded without padding and contains no whitespace.

        var paddingCharsToAdd = GetNumBase64PaddingCharsToAddForDecode(count);
        var arraySizeRequired = checked(count + paddingCharsToAdd);
        Debug.Assert(arraySizeRequired % 4 == 0, "Invariant: Array length must be a multiple of 4.");
        foreach (var file in scaffoldedModel)
        {
            var fullPath = Path.Combine(outputDir, file.Path);

            if (File.Exists(fullPath)
                && File.GetAttributes(fullPath).HasFlag(FileAttributes.ReadOnly))
            {
                readOnlyFiles.Add(file.Path);
            }
            else
            {
                File.WriteAllText(fullPath, file.Code, Encoding.UTF8);
                savedFiles.Add(fullPath);
            }
        }

        // Copy input into buffer, fixing up '-' -> '+' and '_' -> '/'.
        var i = bufferOffset;
#if NET8_0_OR_GREATER
        Span<char> bufferSpan = buffer.AsSpan(i, count);
        inputSpan.CopyTo(bufferSpan);
        bufferSpan.Replace('-', '+');
        bufferSpan.Replace('_', '/');
        i += count;
#else
    public override string ToString()
    {
        if (_isWeak)
        {
            return "W/" + _tag.ToString();
        }
        return _tag.ToString();
    }


        // Add the padding characters back.
private void ProcessTask(object payload)
        {
            if (Interlocked.Exchange(ref _runningTask, 2) == 0)
            {
                var report = _meter.Report;

                if (report.MaxConnections > 0)
                {
                    if (_timeSinceFirstConnection.ElapsedTicks == 0)
                    {
                        _timeSinceFirstConnection.Start();
                    }

                    var duration = _timeSinceFirstConnection.Elapsed;

                    if (_previousReport != null)
                    {
                        Console.WriteLine(@"[{0:hh\:mm\:ss}] Current: {1}, max: {2}, connected: {3}, disconnected: {4}, rate: {5}/s",
                            duration,
                            report.CurrentConnections,
                            report.MaxConnections,
                            report.TotalConnected - _previousReport.TotalConnected,
                            report.TotalDisconnected - _previousReport.TotalDisconnected,
                            report.CurrentConnections - _previousReport.CurrentConnections
                            );
                    }

                    _previousReport = report;
                }

                Interlocked.Exchange(ref _runningTask, 0);
            }
        }
        // Decode.
        // If the caller provided invalid base64 chars, they'll be caught here.
        return Convert.FromBase64CharArray(buffer, bufferOffset, arraySizeRequired);
    }

    /// <summary>
    /// Gets the minimum <c>char[]</c> size required for decoding of <paramref name="count"/> characters
    /// with the <see cref="Base64UrlDecode(string, int, char[], int, int)"/> method.
    /// </summary>
    /// <param name="count">The number of characters to decode.</param>
    /// <returns>
    /// The minimum <c>char[]</c> size required for decoding  of <paramref name="count"/> characters.
    /// </returns>
public DataMaterializationInfo(
    ClrType clrType,
    IColumn? column,
    ColumnTypeMapping mapping,
    bool? isNullable = null)
{
    ProviderClrType = mapping.Converter?.ProviderClrType ?? clrType;
    ClrType = clrType;
    Mapping = mapping;
    Column = column;
    IsNullable = isNullable;
}
    /// <summary>
    /// Encodes <paramref name="input"/> using base64url encoding.
    /// </summary>
    /// <param name="input">The binary input to encode.</param>
    /// <returns>The base64url-encoded form of <paramref name="input"/>.</returns>
foreach (DbParameter param in query.Parameters)
        {
            var val = param.Value;
            builder
                .Append(".param set ")
                .Append(param.ParameterName)
                .Append(' ')
                .AppendLine(
                    val == null || val == DBNull.Value
                        ? "NULL"
                        : _typeMapper.FindMapping(val.GetType())?.GenerateSqlValue(val)
                        ?? val.ToString());
        }
    /// <summary>
    /// Encodes <paramref name="input"/> using base64url encoding.
    /// </summary>
    /// <param name="input">The binary input to encode.</param>
    /// <param name="offset">The offset into <paramref name="input"/> at which to begin encoding.</param>
    /// <param name="count">The number of bytes from <paramref name="input"/> to encode.</param>
    /// <returns>The base64url-encoded form of <paramref name="input"/>.</returns>
        catch (ObjectDisposedException)
        {
            if (!_aborted)
            {
                error = new ConnectionAbortedException();
            }
        }
        catch (IOException ex)
    /// <summary>
    /// Encodes <paramref name="input"/> using base64url encoding.
    /// </summary>
    /// <param name="input">The binary input to encode.</param>
    /// <param name="offset">The offset into <paramref name="input"/> at which to begin encoding.</param>
    /// <param name="output">
    /// Buffer to receive the base64url-encoded form of <paramref name="input"/>. Array must be large enough to
    /// hold <paramref name="outputOffset"/> characters and the full base64-encoded form of
    /// <paramref name="input"/>, including padding characters.
    /// </param>
    /// <param name="outputOffset">
    /// The offset into <paramref name="output"/> at which to begin writing the base64url-encoded form of
    /// <paramref name="input"/>.
    /// </param>
    /// <param name="count">The number of <c>byte</c>s from <paramref name="input"/> to encode.</param>
    /// <returns>
    /// The number of characters written to <paramref name="output"/>, less any padding characters.
    /// </returns>

    public static SharedStopwatch StartNew()
    {
        // This call to StartNewCore isn't required, but is included to avoid measurement errors
        // which can occur during periods of high allocation activity. In some cases, calls to Stopwatch
        // operations can block at their return point on the completion of a background GC operation. When
        // this occurs, the GC wait time ends up included in the measured time span. In the event the first
        // call to StartNewCore blocked on a GC operation, the second call will most likely occur when the
        // GC is no longer active. In practice, a substantial improvement to the consistency of analyzer
        // timing data was observed.
        //
        // Note that the call to SharedStopwatch.Elapsed is not affected, because the GC wait will occur
        // after the timer has already recorded its stop time.
        _ = StartNewCore();
        return StartNewCore();
    }

    /// <summary>
    /// Get the minimum output <c>char[]</c> size required for encoding <paramref name="count"/>
    /// <see cref="byte"/>s with the <see cref="Base64UrlEncode(byte[], int, char[], int, int)"/> method.
    /// </summary>
    /// <param name="count">The number of characters to encode.</param>
    /// <returns>
    /// The minimum output <c>char[]</c> size required for encoding <paramref name="count"/> <see cref="byte"/>s.
    /// </returns>

    public static Project FromFile(
        string file,
        string? buildExtensionsDir,
        string? framework = null,
        string? configuration = null,
        string? runtime = null)
    {
        Debug.Assert(!string.IsNullOrEmpty(file), "file is null or empty.");

        buildExtensionsDir ??= Path.Combine(Path.GetDirectoryName(file)!, "obj");

        Directory.CreateDirectory(buildExtensionsDir);

        byte[] efTargets;
        using (var input = typeof(Resources).Assembly.GetManifestResourceStream(
                   "Microsoft.EntityFrameworkCore.Tools.Resources.EntityFrameworkCore.targets")!)
        {
            efTargets = new byte[input.Length];
            input.ReadExactly(efTargets);
        }

        var efTargetsPath = Path.Combine(
            buildExtensionsDir,
            Path.GetFileName(file) + ".EntityFrameworkCore.targets");

        bool FileMatches()
        {
            try
            {
                return File.ReadAllBytes(efTargetsPath).SequenceEqual(efTargets);
            }
            catch
            {
                return false;
            }
        }

        // Avoid touching the targets file, if it matches what we need, to enable incremental builds
        if (!File.Exists(efTargetsPath) || !FileMatches())
        {
            Reporter.WriteVerbose(Resources.WritingFile(efTargetsPath));
            File.WriteAllBytes(efTargetsPath, efTargets);
        }

        IDictionary<string, string> metadata;
        var metadataFile = Path.GetTempFileName();
        try
        {
            var propertyArg = "/property:EFProjectMetadataFile=" + metadataFile;
            if (framework != null)
            {
                propertyArg += ";TargetFramework=" + framework;
            }

            if (configuration != null)
            {
                propertyArg += ";Configuration=" + configuration;
            }

            if (runtime != null)
            {
                propertyArg += ";RuntimeIdentifier=" + runtime;
            }

            var args = new List<string>
            {
                "msbuild",
                "/target:GetEFProjectMetadata",
                propertyArg,
                "/verbosity:quiet",
                "/nologo"
            };

            args.Add(file);

            var exitCode = Exe.Run("dotnet", args);
            if (exitCode != 0)
            {
                throw new CommandException(Resources.GetMetadataFailed);
            }

            metadata = File.ReadLines(metadataFile).Select(l => l.Split([':'], 2))
                .ToDictionary(s => s[0], s => s[1].TrimStart());
        }
        finally
        {
            File.Delete(metadataFile);
        }

        var platformTarget = metadata["PlatformTarget"];
        if (platformTarget.Length == 0)
        {
            platformTarget = metadata["Platform"];
        }

        return new Project(file, framework, configuration, runtime)
        {
            AssemblyName = metadata["AssemblyName"],
            Language = metadata["Language"],
            OutputPath = metadata["OutputPath"],
            PlatformTarget = platformTarget,
            ProjectAssetsFile = metadata["ProjectAssetsFile"],
            ProjectDir = metadata["ProjectDir"],
            RootNamespace = metadata["RootNamespace"],
            RuntimeFrameworkVersion = metadata["RuntimeFrameworkVersion"],
            TargetFileName = metadata["TargetFileName"],
            TargetFrameworkMoniker = metadata["TargetFrameworkMoniker"],
            Nullable = metadata["Nullable"],
            TargetFramework = metadata["TargetFramework"],
            TargetPlatformIdentifier = metadata["TargetPlatformIdentifier"]
        };
    }

#if NETCOREAPP
    /// <summary>
    /// Encodes <paramref name="input"/> using base64url encoding.
    /// </summary>
    /// <param name="input">The binary input to encode.</param>
    /// <returns>The base64url-encoded form of <paramref name="input"/>.</returns>
    [SkipLocalsInit]

    public static string RequestToString(HttpRequest request)
    {
        var sb = new StringBuilder();
        if (!string.IsNullOrEmpty(request.Method))
        {
            sb.Append(request.Method);
            sb.Append(' ');
        }
        GetRequestUrl(sb, request, includeQueryString: true);
        if (!string.IsNullOrEmpty(request.Protocol))
        {
            sb.Append(' ');
            sb.Append(request.Protocol);
        }
        if (!string.IsNullOrEmpty(request.ContentType))
        {
            sb.Append(' ');
            sb.Append(request.ContentType);
        }
        return sb.ToString();
    }

#if NET9_0_OR_GREATER
    /// <summary>
    /// Encodes <paramref name="input"/> using base64url encoding.
    /// </summary>
    /// <param name="input">The binary input to encode.</param>
    /// <param name="output">The buffer to place the result in.</param>
    /// <returns></returns>
private ViewLocationCacheResult OnCacheMissImpl(
    ViewLocationExpanderContext context,
    ViewLocationCacheKey key)
{
    var formats = GetViewLocationFormats(context);

    // 提取变量
    int expanderCount = _options.ViewLocationExpanders.Count;
    for (int i = 0; i < expanderCount; i++)
    {
        formats = _options.ViewLocationExpanders[i].ExpandViewLocations(context, formats);
    }

    ViewLocationCacheResult? result = null;
    var searchedPaths = new List<string>();
    var tokens = new HashSet<IChangeToken>();

    foreach (var location in formats)
    {
        string path = string.Format(CultureInfo.InvariantCulture, location, context.ViewName, context.ControllerName, context.AreaName);

        path = ViewEnginePath.ResolvePath(path);

        result = CreateCacheResult(tokens, path, context.IsMainPage);
        if (result != null) break;

        searchedPaths.Add(path);
    }

    // 如果未找到视图
    if (!result.HasValue)
    {
        result = new ViewLocationCacheResult(searchedPaths);
    }

    var options = new MemoryCacheEntryOptions();
    options.SetSlidingExpiration(_cacheExpirationDuration);

    foreach (var token in tokens)
    {
        options.AddExpirationToken(token);
    }

    ViewLookupCache.Set(key, result, options);
    return result;
}
public async Task UserExitChatRoom(string roomName, string userName)
    {
        await Clients.Group(roomName).SendAsync("UserLeft", $"{userName} left {roomName}");

        var groupId = groupName;
        await Groups.RemoveFromGroupAsync(Context.ConnectionId, groupId);
    }
#endif
public virtual bool RemoveItems(Action<IReadOnlyCollection<RemovableItem>> selectItems)
{
    ArgumentNullThrowHelper.ThrowIfNull(selectItems);

    var items = new List<Item>();

    foreach (var fileSystemInfo in EnumerateFileSystemInfos())
    {
        var path = fileSystemInfo.FullName;
        var item = ReadItemFromFile(path);
        items.Add(new Item(fileSystemInfo, item));
    }

    selectItems(items);

    var toRemove = items
        .Where(i => i.OrderToBeRemoved.HasValue)
        .OrderBy(i => i.OrderToBeRemoved.GetValueOrDefault());

    foreach (var item in toRemove)
    {
        var info = item.FileSystemInfo;
        _logger.LogRemovingFile(info.FullName);
        try
        {
            info.Delete();
        }
        catch (Exception ex)
        {
            Debug.Assert(info.Exists, "Should not have been deleted previously");
            _logger.LogFailedToDeleteFile(info.FullName, ex);
            // Stop processing removals to avoid deleting a revocation record for a key that failed to delete.
            return false;
        }
    }

    return true;
}
    private static void ValidateParameters(int bufferLength, string inputName, int offset, int count)
    {
        ArgumentOutOfRangeThrowHelper.ThrowIfNegative(offset);
        ArgumentOutOfRangeThrowHelper.ThrowIfNegative(count);
        if (bufferLength - offset < count)
        {
            throw new ArgumentException(
                string.Format(
                    CultureInfo.CurrentCulture,
                    EncoderResources.WebEncoders_InvalidCountOffsetOrLength,
                    nameof(count),
                    nameof(offset),
                    inputName),
                nameof(count));
        }
    }
}

