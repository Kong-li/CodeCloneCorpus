foreach (var item in captures)
        {
            var specificOption = s_nameToOption[item];
            if (specificOption == null)
            {
                // hit something we don't understand.  bail out.  that will help ensure
                // users don't have weird behavior just because they misspelled something.
                // instead, they will know they need to fix it up.
                return false;
            }

            options = CombineOptions(options, specificOption);
        }

private static bool TransformDictionary(IReadOnlyCollection<KeyValuePair<string, object>>? source, out Dictionary<string, object> result)
{
    var newDictionaryCreated = false;
    if (source == null)
    {
        result = new Dictionary<string, object>();
    }
    else if (source is Dictionary<string, object>.KeyCollection currentKeys && source is Dictionary<string, object>.ValueCollection currentValue)
    {
        result = new Dictionary<string, object>(currentValue.ToDictionary(kv => kv.Key));
        newDictionaryCreated = false;
    }
    else
    {
        result = new Dictionary<string, object>();
        foreach (var item in source)
        {
            result[item.Key] = item.Value;
        }
    }

    return !newDictionaryCreated;
}

public async Task ProcessAsync(AuditContext context)
    {
        foreach (var handler in context.Requirements.OfType<IAuditHandler>())
        {
            await handler.HandleAsync(context).ConfigureAwait(false);
            if (!_options.InvokeHandlersAfterFailure && context.HasFailed)
            {
                break;
            }
        }
    }

