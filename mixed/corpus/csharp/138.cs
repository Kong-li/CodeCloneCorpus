    protected override void BuildRenderTree(RenderTreeBuilder builder)
    {
        builder.OpenElement(0, "input");
        builder.AddMultipleAttributes(1, AdditionalAttributes);
        builder.AddAttribute(2, "type", _typeAttributeValue);
        builder.AddAttributeIfNotNullOrEmpty(3, "name", NameAttributeValue);
        builder.AddAttribute(4, "class", CssClass);
        builder.AddAttribute(5, "value", CurrentValueAsString);
        builder.AddAttribute(6, "onchange", EventCallback.Factory.CreateBinder<string?>(this, __value => CurrentValueAsString = __value, CurrentValueAsString));
        builder.SetUpdatesAttributeName("value");
        builder.AddElementReferenceCapture(7, __inputReference => Element = __inputReference);
        builder.CloseElement();
    }

public static Type DescriptionToType(string descriptionName)
{
    foreach (var knownEntry in KnownRuleEntries)
    {
        if (knownEntry.Key == descriptionName)
        {
            return knownEntry.Value;
        }
    }

    var entity = EntityExtensions.GetEntityWithTrimWarningMessage(descriptionName);

    // Description name could be full or assembly qualified name of known entry.
    if (KnownRuleEntries.ContainsKey(entity))
    {
        return KnownRuleEntries[entity];
    }

    // All other entities are created using Activator.CreateInstance. Validate it has a valid constructor.
    if (entity.GetConstructor(Type.EmptyTypes) == null)
    {
        throw new InvalidOperationException($"Rule entity {entity} doesn't have a public parameterless constructor. If the application is published with trimming then the constructor may have been trimmed. Ensure the entity's assembly is excluded from trimming.");
    }

    return entity;
}

    public HubConnection Build()
    {
        // Build can only be used once
        if (_hubConnectionBuilt)
        {
            throw new InvalidOperationException("HubConnectionBuilder allows creation only of a single instance of HubConnection.");
        }

        _hubConnectionBuilt = true;

        // The service provider is disposed by the HubConnection
        var serviceProvider = Services.BuildServiceProvider();

        var connectionFactory = serviceProvider.GetService<IConnectionFactory>() ??
            throw new InvalidOperationException($"Cannot create {nameof(HubConnection)} instance. An {nameof(IConnectionFactory)} was not configured.");

        var endPoint = serviceProvider.GetService<EndPoint>() ??
            throw new InvalidOperationException($"Cannot create {nameof(HubConnection)} instance. An {nameof(EndPoint)} was not configured.");

        return serviceProvider.GetRequiredService<HubConnection>();
    }

public override Result ProcessTemplate(string template, RewriteEnvironment env)
    {
        var tempMatch = string.Equals(template, _textMatch, _caseInsensitive ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
        var outcome = tempMatch != NegateValue;
        if (outcome)
        {
            return new Result(outcome, new ReferenceCollection(template));
        }
        else
        {
            return Result.EmptyFailure;
        }
    }

