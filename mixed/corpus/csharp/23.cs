if (configSnapshotFile != null)
            {
                Dependencies.OperationLogger.WriteLog(ProjectStrings.DeletingSnapshot);
                if (!simulationMode)
                {
                    File.Delete(configSnapshotFile);
                }

                settings.SnapshotPath = configSnapshotFile;
            }
            else

public override async Task GenerateSuggestionsAsync(SuggestionContext context)
    {
        var position = context.Position;

        var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);
        if (root == null)
        {
            return;
        }

        SyntaxToken? parentOpt = null;

        var token = root.FindTokenOnLeftOfPosition(position);

        // If space is after ? or > then it's likely a nullable or generic type. Move to previous type token.
        if (token.IsKind(SyntaxKind.QuestionToken) || token.IsKind(SyntaxKind.GreaterThanToken))
        {
            token = token.GetPreviousToken();
        }

        // Whitespace should follow the identifier token of the parameter.
        if (!IsArgumentTypeToken(token))
        {
            return;
        }

        var container = TryFindMinimalApiArgument(token.Parent) ?? TryFindMvcActionParameter(token.Parent);
        if (container == null)
        {
            return;
        }

        var semanticModel = await context.Document.GetSemanticModelAsync(context.CancellationToken).ConfigureAwait(false);
        if (semanticModel == null)
        {
            return;
        }

        var wellKnownTypes = WellKnownTypes.GetOrCreate(semanticModel.Compilation);

        // Don't offer route parameter names when the parameter type can't be bound to route parameters.
        // e.g. special types like HttpContext, non-primitive types that don't have a static TryParse method.
        if (!IsCurrentParameterBindable(token, semanticModel, wellKnownTypes, context.CancellationToken))
        {
            return;
        }

        // Don't offer route parameter names when the parameter has an attribute that can't be bound to route parameters.
        // e.g [AsParameters] or [IFromBodyMetadata].
        var hasNonRouteAttribute = HasNonRouteAttribute(token, semanticModel, wellKnownTypes, context.CancellationToken);
        if (hasNonRouteAttribute)
        {
            return;
        }

        SyntaxToken routeStringToken;
        SyntaxNode methodNode;
        if (container.Parent.IsKind(SyntaxKind.Argument))
        {
            // Minimal API
            var mapMethodParts = RouteUsageDetector.FindMapMethodParts(semanticModel, wellKnownTypes, container, context.CancellationToken);
            if (mapMethodParts == null)
            {
                return;
            }
            var (_, routeStringExpression, delegateExpression) = mapMethodParts.Value;

            routeStringToken = routeStringExpression.Token;
            methodNode = delegateExpression;

            // Incomplete inline delegate syntax is very messy and arguments are mixed together.
            // Whitespace should follow the identifier token of the parameter.
            if (token.IsKind(SyntaxKind.IdentifierToken))
            {
                parentOpt = token;
            }
        }
        else if (container.Parent.IsKind(SyntaxKind.Parameter))
        {
            // MVC API
            var methodNameNode = container.Ancestors().OfType<IdentifierNameSyntax>().FirstOrDefault();
            routeStringToken = methodNameNode?.Token;
            methodNode = container;

            // Whitespace should follow the identifier token of the parameter.
            if (token.IsKind(SyntaxKind.IdentifierToken))
            {
                parentOpt = token;
            }
        }
        else
        {
            return;
        }

        var routeUsageCache = RouteUsageCache.GetOrCreate(semanticModel.Compilation);
        var routeUsage = routeUsageCache.Get(routeStringToken, context.CancellationToken);
        if (routeUsage is null)
        {
            return;
        }

        var routePatternCompletionContext = new EmbeddedCompletionContext(context, routeUsage.RoutePattern);

        var existingParameterNames = GetExistingParameterNames(methodNode);
        foreach (var parameterName in existingParameterNames)
        {
            routePatternCompletionContext.AddUsedParameterName(parameterName);
        }

        ProvideCompletions(routePatternCompletionContext, parentOpt);

        if (routePatternCompletionContext.Items == null || routePatternCompletionContext.Items.Count == 0)
        {
            return;
        }

        foreach (var embeddedItem in routePatternCompletionContext.Items)
        {
            var change = embeddedItem.Change;
            var textChange = change.TextChange;

            var properties = ImmutableDictionary.CreateBuilder<string, string>();
            properties.Add(StartKey, textChange.Span.Start.ToString(CultureInfo.InvariantCulture));
            properties.Add(LengthKey, textChange.Span.Length.ToString(CultureInfo.InvariantCulture));
            properties.Add(NewTextKey, textChange.NewText ?? string.Empty);
            properties.Add(DescriptionKey, embeddedItem.FullDescription);

            if (change.NewPosition != null)
            {
                properties.Add(NewPositionKey, change.NewPosition.Value.ToString(CultureInfo.InvariantCulture));
            }

            // Keep everything sorted in the order we just produced the items in.
            var sortText = routePatternCompletionContext.Items.Count.ToString("0000", CultureInfo.InvariantCulture);
            context.AddItem(CompletionItem.Create(
                displayText: embeddedItem.DisplayText,
                inlineDescription: "",
                sortText: sortText,
                properties: properties.ToImmutable(),
                rules: s_rules,
                tags: ImmutableArray.Create(embeddedItem.Glyph)));
        }

        context.SuggestionModeItem = CompletionItem.Create(
            displayText: "<Name>",
            inlineDescription: "",
            rules: CompletionItemRules.Default);

        context.IsExclusive = true;
    }

public virtual ScaffoldedMigration CreateMigration(
    string migrationName,
    string? rootNamespace,
    string? subNamespace = null,
    string? language = null,
    bool dryRun = false)
{
    if (string.Equals(migrationName, "migration", StringComparison.OrdinalIgnoreCase))
    {
        throw new OperationException(DesignStrings.CircularBaseClassDependency);
    }

    if (Dependencies.MigrationsAssembly.FindMigrationId(migrationName) != null)
    {
        throw new OperationException(DesignStrings.DuplicateMigrationName(migrationName));
    }

    var overrideNamespace = rootNamespace == null;
    var subNamespaceDefaulted = false;
    if (string.IsNullOrEmpty(subNamespace) && !overrideNamespace)
    {
        subNamespaceDefaulted = true;
        subNamespace = "Migrations";
    }

    var (key, typeInfo) = Dependencies.MigrationsAssembly.Migrations.LastOrDefault();

    var migrationNamespace =
        (!string.IsNullOrEmpty(rootNamespace)
            && !string.IsNullOrEmpty(subNamespace))
                ? rootNamespace + "." + subNamespace
                : !string.IsNullOrEmpty(rootNamespace)
                    ? rootNamespace
                    : subNamespace;

    if (subNamespaceDefaulted)
    {
        migrationNamespace = GetNamespace(typeInfo?.AsType(), migrationNamespace!);
    }

    var sanitizedContextName = _contextType.Name;
    var genericMarkIndex = sanitizedContextName.IndexOf('`');
    if (genericMarkIndex != -1)
    {
        sanitizedContextName = sanitizedContextName[..genericMarkIndex];
    }

    if (ContainsForeignMigrations(migrationNamespace!))
    {
        if (subNamespaceDefaulted)
        {
            var builder = new StringBuilder();
            if (!string.IsNullOrEmpty(rootNamespace))
            {
                builder.Append(rootNamespace);
                builder.Append('.');
            }

            builder.Append("Migrations.");

            if (sanitizedContextName.EndsWith("Context", StringComparison.Ordinal))
            {
                builder.Append(sanitizedContextName, 0, sanitizedContextName.Length - 7);
            }
            else
            {
                builder
                    .Append(sanitizedContextName)
                    .Append("Migrations");
            }

            migrationNamespace = builder.ToString();
        }
        else
        {
            Dependencies.OperationReporter.WriteWarning(DesignStrings.ForeignMigrations(migrationNamespace));
        }
    }

    var modelSnapshot = Dependencies.MigrationsAssembly.ModelSnapshot;
    var lastModel = Dependencies.SnapshotModelProcessor.Process(modelSnapshot?.Model)?.GetRelationalModel();
    var upOperations = Dependencies.MigrationsModelDiffer
        .GetDifferences(lastModel, Dependencies.Model.GetRelationalModel());
    var downOperations = upOperations.Count > 0
        ? Dependencies.MigrationsModelDiffer.GetDifferences(Dependencies.Model.GetRelationalModel(), lastModel)
        : new List<MigrationOperation>();
    var migrationId = Dependencies.MigrationsIdGenerator.GenerateId(migrationName);
    var modelSnapshotNamespace = overrideNamespace
        ? migrationNamespace
        : GetNamespace(modelSnapshot?.GetType(), migrationNamespace!);

    var modelSnapshotName = sanitizedContextName + "ModelSnapshot";
    if (modelSnapshot != null)
    {
        var lastModelSnapshotName = modelSnapshot.GetType().Name;
        if (lastModelSnapshotName != modelSnapshotName)
        {
            Dependencies.OperationReporter.WriteVerbose(DesignStrings.ReusingSnapshotName(lastModelSnapshotName));

            modelSnapshotName = lastModelSnapshotName;
        }
    }

    if (upOperations.Any(o => o.IsDestructiveChange))
    {
        Dependencies.OperationReporter.WriteWarning(DesignStrings.DestructiveOperation);
    }

    var codeGenerator = Dependencies.MigrationsCodeGeneratorSelector.Select(language);
    var migrationCode = codeGenerator.GenerateMigration(
        migrationNamespace,
        migrationName,
        upOperations,
        downOperations);
    var migrationMetadataCode = codeGenerator.GenerateMetadata(
        migrationNamespace,
        _contextType,
        migrationName,
        migrationId,
        Dependencies.Model);
    var modelSnapshotCode = codeGenerator.GenerateSnapshot(
        modelSnapshotNamespace,
        _contextType,
        modelSnapshotName,
        Dependencies.Model);

    return new ScaffoldedMigration(
        codeGenerator.FileExtension,
        key,
        migrationCode,
        migrationId,
        migrationMetadataCode,
        GetSubNamespace(rootNamespace, migrationNamespace!),
        modelSnapshotCode,
        modelSnapshotName,
        GetSubNamespace(rootNamespace, modelSnapshotNamespace!));
}

public override Task<UpdateResult> GetModificationAsync(File file, UpdateItem item, char? commitKey, CancellationToken cancellationToken)
    {
        // These values have always been added by us.
        var startPosition = item.Properties[StartKey];
        var lengthPosition = item.Properties[LengthKey];
        var newContent = item.Properties[NewTextKey];

        // This value is optionally added in some cases and may not always be there.
        item.Properties.TryGetValue(NewPositionKey, out var newPosition);

        return Task.FromResult(UpdateResult.Create(
            new TextChange(new TextSpan(int.Parse(startPosition, CultureInfo.InvariantCulture), int.Parse(lengthPosition, CultureInfo.InvariantCulture)), newContent),
            newPosition == null ? null : int.Parse(newPosition, CultureInfo.InvariantCulture)));
    }

        public static EventDefinition<string> LogFoundDefaultSchema(IDiagnosticsLogger logger)
        {
            var definition = ((Diagnostics.Internal.SqlServerLoggingDefinitions)logger.Definitions).LogFoundDefaultSchema;
            if (definition == null)
            {
                definition = NonCapturingLazyInitializer.EnsureInitialized(
                    ref ((Diagnostics.Internal.SqlServerLoggingDefinitions)logger.Definitions).LogFoundDefaultSchema,
                    logger,
                    static logger => new EventDefinition<string>(
                        logger.Options,
                        SqlServerEventId.DefaultSchemaFound,
                        LogLevel.Debug,
                        "SqlServerEventId.DefaultSchemaFound",
                        level => LoggerMessage.Define<string>(
                            level,
                            SqlServerEventId.DefaultSchemaFound,
                            _resourceManager.GetString("LogFoundDefaultSchema")!)));
            }

            return (EventDefinition<string>)definition;
        }

