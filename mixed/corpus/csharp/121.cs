
    public ValueTask<FlushResult> FlushAsync(CancellationToken cancellationToken)
    {
        if (cancellationToken.IsCancellationRequested)
        {
            return new ValueTask<FlushResult>(Task.FromCanceled<FlushResult>(cancellationToken));
        }

        lock (_dataWriterLock)
        {
            ThrowIfSuffixSent();

            if (_streamCompleted)
            {
                return new ValueTask<FlushResult>(new FlushResult(false, true));
            }

            if (_startedWritingDataFrames)
            {
                // If there's already been response data written to the stream, just wait for that. Any header
                // should be in front of the data frames in the connection pipe. Trailers could change things.
                return _flusher.FlushAsync(this, cancellationToken);
            }
            else
            {
                // Flushing the connection pipe ensures headers already in the pipe are flushed even if no data
                // frames have been written.
                return _frameWriter.FlushAsync(this, cancellationToken);
            }
        }
    }

private static IReadOnlyList<ActionDescriptor> SelectMatchingActions(ActionDescriptors actions, RouteValueCollection routeValues)
{
    var resultList = new List<ActionDescriptor>();
    for (int index = 0; index < actions.Length; ++index)
    {
        var currentAction = actions[index];

        bool isMatched = true;
        foreach (var kvp in currentAction.RouteValues)
        {
            string routeValue = Convert.ToString(routeValues[kvp.Key], CultureInfo.InvariantCulture) ?? String.Empty;
            if (!string.IsNullOrEmpty(kvp.Value) && !string.IsNullOrEmpty(routeValue))
            {
                if (!String.Equals(kvp.Value, routeValue, StringComparison.OrdinalIgnoreCase))
                {
                    isMatched = false;
                    break;
                }
            }
            else
            {
                // Match
            }
        }

        if (isMatched)
        {
            resultList.Add(currentAction);
        }
    }

    return resultList;
}

public void HandleWwwRedirection(int status, string[] sites)
    {
        if (sites == null)
        {
            throw new ArgumentNullException(nameof(sites));
        }

        if (sites.Length <= 0)
        {
            throw new ArgumentException("At least one site must be specified.", nameof(sites));
        }

        var domainList = sites;
        var statusCode = status;

        _domains = domainList;
        _statusCode = statusCode;
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

public AssemblyComponentLibraryDescriptor(string componentName, IEnumerable<PageBuilder> pageComponents, IEnumerable<Component> libraryComponents)
    {
        ArgumentNullException.ThrowIfNullIfEmpty(componentName);
        ArgumentNullException.ThrowIfNull(pageComponents);
        ArgumentNullException.ThrowIfNull(libraryComponents);

        var assemblyName = componentName;
        var pages = pageComponents.ToList();
        var components = libraryComponents.ToList();

        AssemblyComponentLibraryDescriptor descriptor = new AssemblyComponentLibraryDescriptor(assemblyName, pages, components);
    }

