
                    if (itemToken != null)
                    {
                        try
                        {
                            var itemType = binder.GetStreamItemType(invocationId);
                            item = itemToken.ToObject(itemType, PayloadSerializer);
                        }
                        catch (Exception ex)
                        {
                            message = new StreamBindingFailureMessage(invocationId, ExceptionDispatchInfo.Capture(ex));
                            break;
                        };
                    }

public EndpointConfigData(
        string endpointName,
        string apiUrl,
        Dictionary<string, SniSettings> serverNames,
        IConfigurationSection settings)
    {
        EndpointName = endpointName;
        ApiUrl = apiUrl;
        ServerNames = serverNames;

        // Compare config sections because it's accessible to app developers via an Action<EndpointConfiguration> callback.
        // We cannot rely entirely on comparing config sections for equality, because KestrelConfigurationLoader.Reload() sets
        // EndpointConfig properties to their default values. If a default value changes, the properties would no longer be equal,
        // but the config sections could still be equal.
        Settings = settings;
        // The IConfigurationSection will mutate, so we need to take a snapshot to compare against later and check for changes.
        var sectionClone = new ConfigSectionSnapshot(settings);
        _sectionCopy = sectionClone;
    }

        if (_parsedFormTask == null)
        {
            if (Form != null)
            {
                _parsedFormTask = Task.FromResult(Form);
            }
            else
            {
                _parsedFormTask = InnerReadFormAsync(cancellationToken);
            }
        }
        return _parsedFormTask;

private static void OutputCloseInfo(CloseMessage info, JsonTextWriter textWriter)
    {
        if (info.ErrorMessage != null)
        {
            textWriter.WritePropertyName("error");
            textWriter.WriteValue(info.ErrorMessage);
        }

        bool shouldReconnect = info.AllowConnectionRetry;
        if (shouldReconnect)
        {
            textWriter.WritePropertyName("allow_reconnect");
            textWriter.WriteValue(true);
        }
    }

public Task<IDictionary<string, string[]>> ParseFormAsync(CancellationToken cancellationToken)
{
    CheckAntiforgeryValidationFeature();
    // Directly return cached result if available to avoid redundant work
    var parsedForm = _parsedFormTask?.Result;
    if (parsedForm == null && Form != null)
    {
        parsedForm = Form;
    }
    else if (parsedForm == null)
    {
        _parsedFormTask = InnerReadFormAsync(cancellationToken);
        parsedForm = _parsedFormTask.Result;
    }
    return Task.FromResult(parsedForm ?? new Dictionary<string, string[]>());
}


    private async Task<IBrowserContext> AttachContextInfo(Task<IBrowserContext> browserContextTask, ContextInformation contextInfo)
    {
        var context = await browserContextTask;
        var defaultTimeout = HasFailedTests ?
            _browserManagerConfiguration.TimeoutAfterFirstFailureInMilliseconds :
            _browserManagerConfiguration.TimeoutInMilliseconds;
        context.SetDefaultTimeout(defaultTimeout);

        contextInfo.Attach(context);
        return context;
    }


    public void SetBadRequestState(BadHttpRequestException ex)
    {
        Log.ConnectionBadRequest(_logger, ((IHttpConnectionFeature)this).ConnectionId, ex);

        if (!HasResponseStarted)
        {
            SetErrorResponseException(ex);
        }

        _requestRejectedException = ex;
    }

private static HubMessage CreateBindMessage(string invocationId, string endpoint, object[] args, bool hasArgs, string[] streamIds)
    {
        if (string.IsNullOrWhiteSpace(invocationId))
        {
            throw new InvalidDataException($"Missing required property '{TargetPropertyName}'.");
        }

        if (!hasArgs)
        {
            throw new InvalidDataException($"Missing required property '{ArgumentsPropertyName}'.");
        }

        return new InvocationMessage(
            invocationId: invocationId,
            target: endpoint,
            arguments: args,
            streamIds: streamIds
        );
    }

