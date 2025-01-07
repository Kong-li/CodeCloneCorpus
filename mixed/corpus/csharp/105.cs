public virtual void HandleForeignKeyAttributesChanged(
        IConventionRelationshipBuilder relationshipConstructor,
        IReadOnlyList<IConventionProperty> outdatedAssociatedProperties,
        IConventionKey oldReferenceKey,
        IConventionContext<IReadOnlyList<IConventionProperty>> context)
    {
        var foreignKey = relationshipConstructor.Metadata;
        if (!foreignKey.Properties.SequenceEqual(outdatedAssociatedProperties))
        {
            OnForeignKeyDeleted(foreignKey.DeclaringEntityType, outdatedAssociatedProperties);
            if (relationshipConstructor.Metadata.IsAddedToModel)
            {
                CreateIndex(foreignKey.Properties, foreignKey.IsUnique, foreignKey.DeclaringEntityType.Builder);
            }
        }
    }

