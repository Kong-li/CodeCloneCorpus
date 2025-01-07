
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

