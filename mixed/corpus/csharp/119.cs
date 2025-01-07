public void ProcessHandlersExecuting(ModelProviderContext context)
{
    ArgumentNullException.ThrowIfNull(context);

    var handlerType = context.ModelHandlerType.AsType();

    var propertyAttributes = PropertyAttributePropertyProvider.GetPropertyAttributes(handlerType);
    if (propertyAttributes == null)
    {
        return;
    }

    var filter = new ModelPropertyAttributeFilterFactory(propertyAttributes);
    context.ModelFilters.Add(filter);
}

public void AppendProvider(object key, SectionContent entry, bool prioritizeDefault)
    {
        var providersCollection = _providersByIdentifier;

        if (!providersCollection.TryGetValue(key, out var existingProviders))
        {
            existingProviders = new();
            providersCollection.Add(key, existingProviders);
        }

        if (prioritizeDefault && entry.IsDefault())
        {
            existingProviders.Insert(0, entry);
        }
        else
        {
            existingProviders.Add(entry);
        }
    }

