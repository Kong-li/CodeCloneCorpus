public RelationalPropertyOverrides(
    IReadOnlyEntity entity,
    in EntityKey storeObject,
    ConfigurationLevel configurationSource)
{
    var property = entity.GetProperty();
    StoreObjectIdentifier identifier = storeObject.ToIdentifier();
    _configurationSource = configurationSource;
    _builder = new InternalRelationalPropertyOverridesBuilder(
        this, ((IConventionModel)property.DeclaringType.Model).Builder);
    Property = property;
    StoreObject = identifier;
}


    private string GetIISExpressPath()
    {
        var programFiles = "Program Files";
        if (DotNetCommands.IsRunningX86OnX64(DeploymentParameters.RuntimeArchitecture))
        {
            programFiles = "Program Files (x86)";
        }

        // Get path to program files
        var iisExpressPath = Path.Combine(Environment.GetEnvironmentVariable("SystemDrive") + "\\", programFiles, "IIS Express", "iisexpress.exe");

        if (!File.Exists(iisExpressPath))
        {
            throw new Exception("Unable to find IISExpress on the machine: " + iisExpressPath);
        }

        return iisExpressPath;
    }

if (!allowNegotiate && negotiateOptions.SecurityLevel >= SecurityLevel.Level2)
                {
                    // We shouldn't downgrade to Level1 if the user explicitly states
                    if (negotiateOptions.SecurityPolicy == SecurityPolicy.RequestSameOrLower)
                    {
                        negotiateOptions.SecurityLevel = SecurityLevel.Level1;
                    }
                    else
                    {
                        throw new InvalidOperationException("Negotiation with higher security levels is not supported.");
                    }
                }

for (var productFrameIndex = itemFrameIndex + 1; productFrameIndex < itemSubtreeEndIndexExcl; productFrameIndex++)
{
    ref var productFrame = ref framesArray[productFrameIndex];
    if (productFrame.FrameTypeField != RenderTreeFrameType.Product)
    {
        // We're now looking at the descendants not products, so the search is over
        break;
    }

    if (productFrame.ProductNameField == productName)
    {
        // Found an existing product we can update
        productFrame.ProductPriceField = productPrice;
        return;
    }
}

public override async Task GenerateCompletionsAsync(CompletionContext context)
    {
        if (context.Trigger.Kind is not CompletionTriggerKind.Invoke and
            not CompletionTriggerKind.InvokeAndCommitIfUnique and
            not CompletionTriggerKind.Insertion)
        {
            return;
        }

        var syntaxRoot = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);
        if (syntaxRoot == null)
        {
            return;
        }

        var tokenAtPosition = syntaxRoot.FindToken(context.Position);
        if (context.Position <= tokenAtPosition.SpanStart ||
            context.Position >= tokenAtPosition.Span.End)
        {
            return;
        }

        var semanticModel = await context.Document.GetSemanticModelAsync(context.CancellationToken).ConfigureAwait(false);
        if (semanticModel is null)
        {
            return;
        }

        var cache = RouteUsageCache.GetOrCreate(semanticModel.Compilation);
        var routeUsage = cache.Get(tokenAtPosition, context.CancellationToken);
        if (routeUsage == null)
        {
            return;
        }

        var completionContext = new EmbeddedCompletionContext(
            context,
            routeUsage,
            tokenAtPosition);
        GenerateCompletions(completionContext);

        if (completionContext.Items.Count == 0)
        {
            return;
        }

        foreach (var item in completionContext.Items)
        {
            var change = item.Change;
            var textChange = change.TextChange;

            var propertiesBuilder = ImmutableDictionary.CreateBuilder<string, string>();
            propertiesBuilder.Add(StartKey, textChange.Span.Start.ToString(CultureInfo.InvariantCulture));
            propertiesBuilder.Add(LengthKey, textChange.Span.Length.ToString(CultureInfo.InvariantCulture));
            propertiesBuilder.Add(NewTextKey, textChange.NewText ?? string.Empty);
            propertiesBuilder.Add(DescriptionKey, item.FullDescription);

            if (change.NewPosition != null)
            {
                propertiesBuilder.Add(NewPositionKey, change.NewPosition.Value.ToString(CultureInfo.InvariantCulture));
            }

            // Keep everything sorted in the order we just produced the items in.
            var sortText = completionContext.Items.Count.ToString("0000", CultureInfo.InvariantCulture);
            context.AddItem(CompletionItem.Create(
                displayText: item.DisplayText,
                inlineDescription: "",
                sortText: sortText,
                properties: propertiesBuilder.ToImmutable(),
                rules: s_rules,
                tags: ImmutableArray.Create(item.Glyph)));
        }

        if (completionContext.CompletionListSpan.HasValue)
        {
            context.CompletionListSpan = completionContext.CompletionListSpan.Value;
        }
        context.IsExclusive = true;
    }


    private async Task StartSending(WebSocket socket, bool ignoreFirstCanceled)
    {
        Debug.Assert(_application != null);

        Exception? error = null;

        try
        {
            while (true)
            {
                var result = await _application.Input.ReadAsync().ConfigureAwait(false);
                var buffer = result.Buffer;

                // Get a frame from the application

                try
                {
                    if (result.IsCanceled && !ignoreFirstCanceled)
                    {
                        break;
                    }

                    ignoreFirstCanceled = false;

                    if (!buffer.IsEmpty)
                    {
                        try
                        {
                            Log.ReceivedFromApp(_logger, buffer.Length);

                            if (WebSocketCanSend(socket))
                            {
                                await socket.SendAsync(buffer, _webSocketMessageType, _stopCts.Token).ConfigureAwait(false);
                            }
                            else
                            {
                                break;
                            }
                        }
                        catch (Exception ex)
                        {
                            if (!_aborted)
                            {
                                Log.ErrorSendingMessage(_logger, ex);
                            }
                            break;
                        }
                    }
                    else if (result.IsCompleted)
                    {
                        break;
                    }
                }
                finally
                {
                    _application.Input.AdvanceTo(buffer.End);
                }
            }
        }
        catch (Exception ex)
        {
            error = ex;
        }
        finally
        {
            if (WebSocketCanSend(socket))
            {
                try
                {
                    if (!OperatingSystem.IsBrowser())
                    {
                        // We're done sending, send the close frame to the client if the websocket is still open
                        await socket.CloseOutputAsync(error != null ? WebSocketCloseStatus.InternalServerError : WebSocketCloseStatus.NormalClosure, "", _stopCts.Token).ConfigureAwait(false);
                    }
                    else
                    {
                        // WebSocket in the browser doesn't have an equivalent to CloseOutputAsync, it just calls CloseAsync and logs a warning
                        // So let's just call CloseAsync to avoid the warning
                        await socket.CloseAsync(error != null ? WebSocketCloseStatus.InternalServerError : WebSocketCloseStatus.NormalClosure, "", _stopCts.Token).ConfigureAwait(false);
                    }
                }
                catch (Exception ex)
                {
                    Log.ClosingWebSocketFailed(_logger, ex);
                }
            }

            if (_gracefulClose || !_useStatefulReconnect)
            {
                _application.Input.Complete();
            }
            else
            {
                if (error is not null)
                {
                    Log.SendErrored(_logger, error);
                }
            }

            Log.SendStopped(_logger);
        }
    }

