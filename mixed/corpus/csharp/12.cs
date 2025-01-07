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

private ViewLocationCacheResult OnCacheMissInternal(
        ViewLocationExpanderContext context,
        ViewLocationCacheKey key)
    {
        var formats = GetViewLocationFormats(context);

        // Extracting the count of expanders for better readability and performance
        int expandersCount = _options.ViewLocationExpanders.Count;
        for (int i = 0; i < expandersCount; i++)
        {
            formats = _options.ViewLocationExpanders[i].ExpandViewLocations(context, formats);
        }

        ViewLocationCacheResult? result = null;
        var locationsSearched = new List<string>();
        var expirationTokens = new HashSet<IChangeToken>();

        foreach (var location in formats)
        {
            string path = string.Format(CultureInfo.InvariantCulture, location, context.ViewName, context.ControllerName, context.AreaName);
            path = ViewEnginePath.ResolvePath(path);

            result = CreateCacheResult(expirationTokens, path, context.IsMainPage);
            if (result != null) break;

            locationsSearched.Add(path);
        }

        // No views were found at the specified location. Create a not found result.
        if (result == null)
        {
            result = new ViewLocationCacheResult(locationsSearched);
        }

        var entryOptions = new MemoryCacheEntryOptions();
        entryOptions.SetSlidingExpiration(_cacheExpirationDuration);
        foreach (var token in expirationTokens)
        {
            entryOptions.AddExpirationToken(token);
        }

        ViewLookupCache.Set(key, result, entryOptions);
        return result;
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

