
        if (navigation == null)
        {
            if (internalEntry.EntityType.FindProperty(name) != null
                || internalEntry.EntityType.FindComplexProperty(name) != null)
            {
                throw new InvalidOperationException(
                    CoreStrings.NavigationIsProperty(
                        name, internalEntry.EntityType.DisplayName(),
                        nameof(ChangeTracking.EntityEntry.Reference), nameof(ChangeTracking.EntityEntry.Collection),
                        nameof(ChangeTracking.EntityEntry.Property)));
            }

            throw new InvalidOperationException(CoreStrings.PropertyNotFound(name, internalEntry.EntityType.DisplayName()));
        }

public virtual bool UpdateNullableBehavior(bool behavesAsNullable, RuleSet ruleSet)
{
    if (!Expression.IsComplex)
    {
        throw new ArgumentException(
            $"Non-complex expression cannot propagate nullable behavior: {Expression.Name} - {Expression.Type.Name}");
    }

    _behavesAsNullable = behavesAsNullable;
    _ruleSet = ruleSet.Max(_existingRuleSet);

    return behavesAsNullable;
}

public async IAsyncEnumerable<ArraySegment<byte>> StreamDataToJavaScript(long streamIdentifier)
    {
        var circuitHandler = await FetchActiveCircuit();
        if (circuitHandler == null)
        {
            yield break;
        }

        var dataStreamReference = await circuitHandler.TryCapturePendingDataStream(streamIdentifier);
        if (!dataStreamReference.HasValue)
        {
            yield break;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(32 * 1024);

        try
        {
            int readBytes;
            while ((readBytes = await circuitHandler.SendDataStreamAsync(dataStreamReference.Value, streamIdentifier, buffer)) > 0)
            {
                yield return new ArraySegment<byte>(buffer, 0, readBytes);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);

            if (!dataStreamReference.Value.KeepOpen)
            {
                dataStreamReference.Stream?.Dispose();
            }
        }
    }

    protected virtual void DiscoverComplexProperties(
        IConventionTypeBaseBuilder structuralTypeBuilder,
        IConventionContext context)
    {
        var typeBase = structuralTypeBuilder.Metadata;
        foreach (var candidateMember in GetMembers(typeBase))
        {
            TryConfigureComplexProperty(candidateMember, typeBase, context);
        }
    }

