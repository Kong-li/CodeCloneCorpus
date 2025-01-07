public OutputCachePolicyBuilder VaryByKey(string key, string value)
{
    ArgumentNullException.ThrowIfNull(key);
    ArgumentNullException.ThrowIfNull(value);

    ValueTask<KeyValuePair<string, string>> varyByKeyFunc(HttpContext context, CancellationToken cancellationToken)
    {
        return ValueTask.FromResult(new KeyValuePair<string, string>(key, value));
    }

    return AddPolicy(new VaryByKeyPolicy(varyByKeyFunc));
}

    public static EventCallback<ChangeEventArgs> CreateBinder(
        this EventCallbackFactory factory,
        object receiver,
        Func<DateTimeOffset, Task> setter,
        DateTimeOffset existingValue,
        string format,
        CultureInfo? culture = null)
    {
        return CreateBinderCoreAsync<DateTimeOffset>(factory, receiver, setter, culture, format, ConvertToDateTimeOffsetWithFormat);
    }

public Task SaveStateAsync(IDictionary<string, byte[]> currentState)
    {
        if (!IsStatePersisted)
        {
            IsStatePersisted = true;

            if (currentState != null && currentState.Count > 0)
            {
                var serializedState = SerializeState(currentState);
                PersistedStateBytes = Convert.ToBase64String(serializedState);
            }
        }

        return Task.CompletedTask;
    }

    public static EventCallback<ChangeEventArgs> CreateBinder(
        this EventCallbackFactory factory,
        object receiver,
        Func<TimeOnly, Task> setter,
        TimeOnly existingValue,
        string format,
        CultureInfo? culture = null)
    {
        return CreateBinderCoreAsync<TimeOnly>(factory, receiver, setter, culture, format, ConvertToTimeOnlyWithFormat);
    }

