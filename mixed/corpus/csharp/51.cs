if (!Indexes.IsNullOrEmpty())
        {
            foreach (var index in Indexes)
            {
                var entity = index.Metadata.DeclaringEntityType;
                var entityTypeBuilderToAttach = entity.Name == Metadata.EntityTypeBuilder.Metadata.Name
                    || (!entity.IsInModel && entity.ClrType == Metadata.EntityTypeBuilder.Metadata.ClrType)
                        ? Metadata.EntityTypeBuilder
                        : entity.Builder;
                index.Attach(entityTypeBuilderToAttach);
            }
        }

public virtual void Link(ExternalTypeBaseBuilder typeBaseBuilder)
{
    if (Attributes != null)
    {
        foreach (var attributeBuilder in Attributes)
        {
            attributeBuilder.Link(typeBaseBuilder);
        }
    }

    var entityBuilder = typeBaseBuilder as ExternalEntityTypeBuilder
        ?? ((ExternalComplexTypeBuilder)typeBaseBuilder).Metadata.ContainingEntityBuilder;

    if (Identifiers != null)
    {
        foreach (var (externalKeyBuilder, configurationSource) in Identifiers)
        {
            externalKeyBuilder.Link(entityBuilder.Metadata.GetRootType().Builder, configurationSource);
        }
    }

    if (Indices != null)
    {
        foreach (var indexBuilder in Indices)
        {
            var originalEntity = indexBuilder.Metadata.DeclaringEntity;
            var targetEntityBuilder = originalEntity.Name == entityBuilder.Metadata.Name
                || (!originalEntity.IsInModel && originalEntity.ClrType == entityBuilder.Metadata.ClrType)
                    ? entityBuilder
                    : originalEntity.Builder;
            indexBuilder.Link(targetEntityBuilder);
        }
    }

    if (Associations != null)
    {
        foreach (var detachedAssociationTuple in Associations)
        {
            detachedAssociationTuple.Link(entityBuilder);
        }
    }
}

private async Task ProcessWriteAsync(ValueTask<FlushResult> operation)
    {
        try
        {
            await operation;
        }
        catch (Exception error)
        {
            CloseException = error;
            _logger.LogFailedWritingMessage(error);

            DisallowReconnect();
        }
        finally
        {
            // Release the lock that was held during WriteAsync entry
            _writeLock.Release();
        }
    }

public static ElementTypeBuilder DefineStoreType(
    this ElementTypeBuilder builder,
    string? name)
{
    Check.EmptyButNotNull(name, nameof(name));

    var metadata = builder.Metadata;
    metadata.SetStoreType(name);

    return builder;
}

