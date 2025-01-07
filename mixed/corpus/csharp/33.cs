public void InvertSequence()
{
    if (Bound != null
        || Skip != null)
    {
        throw new InvalidOperationException(DataStrings.ReverseAfterSkipTakeNotSupported);
    }

    var currentOrderings = _sequenceOrders.ToArray();

    _sequenceOrders.Clear();

    foreach (var currentOrdering in currentOrderings)
    {
        _sequenceOrders.Add(
            new SequenceOrdering(
                currentOrdering.Expression,
                !currentOrdering.IsAscending));
    }
}

public static ModelBuilder ConfigureHiLoSequence(
    this ModelBuilder modelBuilder,
    string? sequenceName = null,
    string? schemaName = null)
{
    Check.NullButNotEmpty(sequenceName, nameof(sequenceName));
    Check NullButNotEmpty(schemaName, nameof(schemaName));

    var model = modelBuilder.Model;

    if (string.IsNullOrEmpty(sequenceName))
    {
        sequenceName = SqlServerModelExtensions.DefaultHiLoSequenceName;
    }

    if (model.FindSequence(sequenceName, schemaName) == null)
    {
        modelBuilder.HasSequence(sequenceName, schemaName).IncrementsBy(10);
    }

    model.SetValueGenerationStrategy(SqlServerValueGenerationStrategy.SequenceHiLo);
    model.SetHiLoSequenceName(sequenceName);
    model.SetHiLoSequenceSchema(schemaName);
    model.SetSequenceNameSuffix(null);
    model.SetSequenceSchema(null);
    model.SetIdentitySeed(null);
    model.SetIdentityIncrement(null);

    return modelBuilder;
}

    public override void Process(TagHelperContext context, TagHelperOutput output)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(output);

        // Pass through attribute that is also a well-known HTML attribute.
        if (Href != null)
        {
            output.CopyHtmlAttribute(HrefAttributeName, context);
        }

        // If there's no "href" attribute in output.Attributes this will noop.
        ProcessUrlAttribute(HrefAttributeName, output);

        // Retrieve the TagHelperOutput variation of the "href" attribute in case other TagHelpers in the
        // pipeline have touched the value. If the value is already encoded this LinkTagHelper may
        // not function properly.
        Href = output.Attributes[HrefAttributeName]?.Value as string;

        if (!AttributeMatcher.TryDetermineMode(context, ModeDetails, Compare, out var mode))
        {
            // No attributes matched so we have nothing to do
            return;
        }

        if (AppendVersion == true)
        {
            EnsureFileVersionProvider();

            if (Href != null)
            {
                var href = GetVersionedResourceUrl(Href);
                var index = output.Attributes.IndexOfName(HrefAttributeName);
                var existingAttribute = output.Attributes[index];
                output.Attributes[index] = new TagHelperAttribute(
                    existingAttribute.Name,
                    href,
                    existingAttribute.ValueStyle);
            }
        }

        var builder = output.PostElement;
        builder.Clear();

        if (mode == Mode.GlobbedHref || mode == Mode.Fallback && !string.IsNullOrEmpty(HrefInclude))
        {
            BuildGlobbedLinkTags(output.Attributes, builder);
            if (string.IsNullOrEmpty(Href))
            {
                // Only HrefInclude is specified. Don't render the original tag.
                output.TagName = null;
                output.Content.SetHtmlContent(HtmlString.Empty);
            }
        }

        if (mode == Mode.Fallback && HasStyleSheetLinkType(output.Attributes))
        {
            if (TryResolveUrl(FallbackHref, resolvedUrl: out string resolvedUrl))
            {
                FallbackHref = resolvedUrl;
            }

            BuildFallbackBlock(output.Attributes, builder);
        }
    }

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

