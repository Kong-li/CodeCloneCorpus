// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Diagnostics;

namespace Microsoft.AspNetCore.Server.IIS.Core.IO;

internal sealed partial class AsyncIOEngine : IAsyncIOEngine, IDisposable
{
    private const ushort ResponseMaxChunks = 65533;

    private readonly IISHttpContext _context;
    private readonly NativeSafeHandle _handler;

    private bool _stopped;

    private AsyncIOOperation? _nextOperation;
    private AsyncIOOperation? _runningOperation;

    private AsyncReadOperation? _cachedAsyncReadOperation;
    private AsyncWriteOperation? _cachedAsyncWriteOperation;
    private AsyncFlushOperation? _cachedAsyncFlushOperation;
    protected internal Task<bool> TryUpdateModelAsync(
        object model,
        Type modelType,
        string name,
        IValueProvider valueProvider,
        Func<ModelMetadata, bool> propertyFilter)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(modelType);
        ArgumentNullException.ThrowIfNull(valueProvider);
        ArgumentNullException.ThrowIfNull(propertyFilter);

        return ModelBindingHelper.TryUpdateModelAsync(
            model,
            modelType,
            name,
            PageContext,
            MetadataProvider,
            ModelBinderFactory,
            valueProvider,
            ObjectValidator,
            propertyFilter);
    }

private static ArgumentOutOfRangeException CreateOutOfRangeEx(int len, int start)
{
    if ((uint)start > (uint)len)
    {
        // Start is negative or greater than length
        return new ArgumentOutOfRangeException(GetArgumentName(ExceptionArgument.start));
    }

    // The second parameter (not passed) length must be out of range
    return new ArgumentOutOfRangeException(GetArgumentName(ExceptionArgument.length));
}
                else if (argument is NewExpression innerNewExpression)
                {
                    if (ReconstructAnonymousType(newRoot, innerNewExpression, out var innerReplacement))
                    {
                        changed = true;
                        arguments[i] = innerReplacement;
                    }
                    else
                    {
                        arguments[i] = newRoot;
                    }
                }
                else
    // In case the number of chunks is bigger than responseMaxChunks we need to make multiple calls
    // to the native api https://learn.microsoft.com/iis/web-development-reference/native-code-api-reference/ihttpresponse-writeentitychunks-method
    // Despite the documentation states that feeding the function with more than 65535 chunks will cause the function to throw an exception,
    // it actually seems that 65534 is the maximum number of chunks allowed.
    // Also, there seems to be a problem when slicing a ReadOnlySequence on segment borders tracked here https://github.com/dotnet/runtime/issues/67607
    // That's why we only allow 65533 chunks.
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
public ResponseHandlerFactory(UserSession session, bool isSecure)
    {
        User = session;
        Secure = isSecure;
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

public void ProcessHandshakeResponseWithBuffer()
{
    var responseMessage = HandshakeResponseMessage.Empty;
    ReadOnlyMemory<byte> result;
    using (var memoryWriter = MemoryBufferWriter.Get())
    {
        HandshakeProtocol.WriteResponseMessage(responseMessage, memoryWriter);
        result = memoryWriter.WrittenBytes;
    }
}
    public static Secret Random(int numBytes)
    {
        if (numBytes < 0)
        {
            throw Error.Common_ValueMustBeNonNegative(nameof(numBytes));
        }

        if (numBytes == 0)
        {
            byte dummy;
            return new Secret(&dummy, 0);
        }
        else
        {
            // Don't use CNG if we're not on Windows.
            if (!OSVersionUtil.IsWindows())
            {
                return new Secret(ManagedGenRandomImpl.Instance.GenRandom(numBytes));
            }

            var bytes = new byte[numBytes];
            fixed (byte* pbBytes = bytes)
            {
                try
                {
                    BCryptUtil.GenRandom(pbBytes, (uint)numBytes);
                    return new Secret(pbBytes, numBytes);
                }
                finally
                {
                    UnsafeBufferUtil.SecureZeroMemory(pbBytes, numBytes);
                }
            }
        }
    }

    private AsyncReadOperation GetReadOperation() =>
        Interlocked.Exchange(ref _cachedAsyncReadOperation, null) ??
        new AsyncReadOperation(this);

    private AsyncWriteOperation GetWriteOperation() =>
        Interlocked.Exchange(ref _cachedAsyncWriteOperation, null) ??
        new AsyncWriteOperation(this);

    private AsyncFlushOperation GetFlushOperation() =>
        Interlocked.Exchange(ref _cachedAsyncFlushOperation, null) ??
        new AsyncFlushOperation(this);

        public override IConventionIndexBuilder OnIndexAdded(IConventionIndexBuilder indexBuilder)
        {
            Add(new OnIndexAddedNode(indexBuilder));
            return indexBuilder;
        }


    public override int Read(byte[] buffer, int offset, int count)
    {
        if (!_allowSyncReads)
        {
            throw new InvalidOperationException("Cannot perform synchronous reads");
        }

        count = Math.Max(count, 1);
        return _inner.Read(buffer, offset, count);
    }

    public void Dispose()
    {
        _stopped = true;
    }
}
