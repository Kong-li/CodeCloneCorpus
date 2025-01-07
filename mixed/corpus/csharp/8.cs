if (needInitialize)
        {
            var entityConfigBuilder = configBuilder
                .GetEntityConfigBuilder(
                    targetType,
                    navigationPropertyInfo,
                    createIfRequired: true,
                    isOwnedNavigation ?? ShouldBeOwned(targetType, configBuilder.Metadata.Model));
            if (entityConfigBuilder != null)
            {
                return entityConfigBuilder;
            }
        }

protected override Expression VisitChildren1(ExpressionVisitor1 visitor)
{
    var changed = false;
    var expressions = new Expression[Expressions.Count];
    for (var i = 0; i < expressions.Length; i++)
    {
        expressions[i] = visitor.Visit(Expressions[i]);
        changed |= expressions[i] != Expressions[i];
    }

    return changed
        ? new SqlFunctionExpression1(Name1, expressions, Type1, TypeMapping1)
        : this;
}

        if (shouldCreate)
        {
            var targetEntityTypeBuilder = entityTypeBuilder
                .GetTargetEntityTypeBuilder(
                    targetClrType,
                    navigationMemberInfo,
                    createIfMissing: true,
                    shouldBeOwned ?? ShouldBeOwned(targetClrType, entityTypeBuilder.Metadata.Model));
            if (targetEntityTypeBuilder != null)
            {
                return targetEntityTypeBuilder;
            }
        }

while (entityRelationshipCandidate.RelatedProperties.Count > 0)
            {
                var relatedProperty = entityRelationshipCandidate.RelatedProperties[0];
                var relatedPropertyName = relatedProperty.GetSimpleMemberName();
                var existingRelated = entityType.FindRelated(relatedPropertyName);
                if (existingRelated != null)
                {
                    if (existingRelated.DeclaringEntityType != entityType
                        || existingRelated.TargetEntityType != targetEntityType)
                    {
                        entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);
                        continue;
                    }
                }
                else
                {
                    var existingSkipRelated = entityType.FindSkipRelated(relatedPropertyName);
                    if (existingSkipRelated != null
                        && (existingSkipRelated.DeclaringEntityType != entityType
                            || existingSkipRelated.TargetEntityType != targetEntityType))
                    {
                        entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);
                        continue;
                    }
                }

                if (entityRelationshipCandidate.RelatedProperties.Count == 1
                    && entityRelationshipCandidate.InverseProperties.Count == 0)
                {
                    break;
                }

                PropertyInfo? compatibleInverse = null;
                foreach (var inverseProperty in entityRelationshipCandidate.InverseProperties)
                {
                    if (AreCompatible(
                            relatedProperty, inverseProperty, entityTypeBuilder, targetEntityTypeBuilder))
                    {
                        if (compatibleInverse == null)
                        {
                            compatibleInverse = inverseProperty;
                        }
                        else
                        {
                            goto NextCandidate;
                        }
                    }
                }

                if (compatibleInverse == null)
                {
                    entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);

                    filteredEntityRelationshipCandidates.Add(
                        new EntityRelationshipCandidate(
                            targetEntityTypeBuilder,
                            [relatedProperty],
                            [],
                            entityRelationshipCandidate.IsOwnership));

                    if (entityRelationshipCandidate.TargetTypeBuilder.Metadata == entityTypeBuilder.Metadata
                        && entityRelationshipCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityRelationshipCandidate.InverseProperties.First();
                        if (!entityRelationshipCandidate.RelatedProperties.Contains(nextSelfRefCandidate))
                        {
                            entityRelationshipCandidate.RelatedProperties.Add(nextSelfRefCandidate);
                        }

                        entityRelationshipCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    if (entityRelationshipCandidate.RelatedProperties.Count == 0)
                    {
                        foreach (var inverseProperty in entityRelationshipCandidate.InverseProperties.ToList())
                        {
                            if (!AreCompatible(
                                    null, inverseProperty, entityTypeBuilder, targetEntityTypeBuilder))
                            {
                                entityRelationshipCandidate.InverseProperties.Remove(inverseProperty);
                            }
                        }
                    }

                    continue;
                }

                var noOtherCompatibleNavigation = true;
                foreach (var otherRelated in entityRelationshipCandidate.RelatedProperties)
                {
                    if (otherRelated != relatedProperty
                        && AreCompatible(otherRelated, compatibleInverse, entityTypeBuilder, targetEntityTypeBuilder))
                    {
                        noOtherCompatibleNavigation = false;
                        break;
                    }
                }

                if (noOtherCompatibleNavigation)
                {
                    entityRelationshipCandidate.RelatedProperties.Remove(relatedProperty);
                    entityRelationshipCandidate.InverseProperties.Remove(compatibleInverse);

                    filteredEntityRelationshipCandidates.Add(
                        new EntityRelationshipCandidate(
                            targetEntityTypeBuilder,
                            [relatedProperty],
                            [compatibleInverse],
                            entityRelationshipCandidate.IsOwnership)
                    );

                    if (entityRelationshipCandidate.TargetTypeBuilder.Metadata == entityTypeBuilder.Metadata
                        && entityRelationshipCandidate.RelatedProperties.Count == 0
                        && entityRelationshipCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityRelationshipCandidate.InverseProperties.First();
                        if (!entityRelationshipCandidate.RelatedProperties.Contains(nextSelfRefCandidate))
                        {
                            entityRelationshipCandidate.RelatedProperties.Add(nextSelfRefCandidate);
                        }

                        entityRelationshipCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    continue;
                }

                NextCandidate:
                break;
            }

foreach (var reverseAttribute in relationshipOption.ReverseAttributes)
                {
                    if (IsSuitable(
                            navigationField, reverseAttribute, entityBuilder, targetEntityBuilder))
                    {
                        if (suitableReverse == null)
                        {
                            suitableReverse = reverseAttribute;
                        }
                        else
                        {
                            goto NextOption;
                        }
                    }
                }

private async Task HandleCircuitEventAsync(CancellationToken cancellationToken)
{
    Log.CircuitClosed(_logger, CircuitId);

    List<Exception> errorList = null;

    for (int j = 0; j < _circuitHandlers.Length; j++)
    {
        var handler = _circuitHandlers[j];
        try
        {
            await handler.OnCircuitDownAsync(Circuit, cancellationToken);
        }
        catch (Exception ex)
        {
            errorList ??= new List<Exception>();
            errorList.Add(ex);
            Log.CircuitHandlerFailed(_logger, handler, nameof(CircuitHandler.OnCircuitClosedAsync), ex);
        }
    }

    if (errorList != null && errorList.Count > 0)
    {
        throw new AggregateException("Encountered exceptions while executing circuit handlers.", errorList);
    }
}

private static IReadOnlyList<RelationshipCandidate> FilterNonCompatibleEntities(
        IReadOnlyList<RelationshipCandidate> entityCandidates,
        IConventionEntityTypeBuilder entityBuilder)
    {
        if (entityCandidates.Count == 0)
        {
            return entityCandidates;
        }

        var entityType = entityBuilder.Metadata;
        var filteredEntityCandidates = new List<RelationshipCandidate>();
        foreach (var entityCandidate in entityCandidates)
        {
            var targetEntityBuilder = entityCandidate.TargetTypeBuilder;
            var targetEntityType = targetEntityBuilder.Metadata;
            while (entityCandidate.NavigationProperties.Count > 0)
            {
                var navigationProperty = entityCandidate.NavigationProperties[0];
                var navigationPropertyName = navigationProperty.GetSimpleMemberName();
                var existingNavigation = entityType.FindNavigation(navigationPropertyName);
                if (existingNavigation != null)
                {
                    if (existingNavigation.DeclaringEntityType != entityType
                        || existingNavigation.TargetEntityType != targetEntityType)
                    {
                        entityCandidate.NavigationProperties.Remove(navigationProperty);
                        continue;
                    }
                }
                else
                {
                    var existingSkipNavigation = entityType.FindSkipNavigation(navigationPropertyName);
                    if (existingSkipNavigation != null
                        && (existingSkipNavigation.DeclaringEntityType != entityType
                            || existingSkipNavigation.TargetEntityType != targetEntityType))
                    {
                        entityCandidate.NavigationProperties.Remove(navigationProperty);
                        continue;
                    }
                }

                if (entityCandidate.NavigationProperties.Count == 1
                    && entityCandidate.InverseProperties.Count == 0)
                {
                    break;
                }

                PropertyInfo? compatibleInverse = null;
                foreach (var inverseProperty in entityCandidate.InverseProperties)
                {
                    if (AreCompatible(
                            navigationProperty, inverseProperty, entityBuilder, targetEntityBuilder))
                    {
                        if (compatibleInverse == null)
                        {
                            compatibleInverse = inverseProperty;
                        }
                        else
                        {
                            goto NextCandidate;
                        }
                    }
                }

                if (compatibleInverse == null)
                {
                    entityCandidate.NavigationProperties.Remove(navigationProperty);

                    filteredEntityCandidates.Add(
                        new RelationshipCandidate(
                            targetEntityBuilder,
                            new[] {navigationProperty},
                            Array.Empty<PropertyInfo>(),
                            entityCandidate.IsOwnership));

                    if (entityCandidate.TargetTypeBuilder.Metadata == entityBuilder.Metadata
                        && entityCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityCandidate.InverseProperties.First();
                        if (!entityCandidate.NavigationProperties.Contains(nextSelfRefCandidate))
                        {
                            entityCandidate.NavigationProperties.Add(nextSelfRefCandidate);
                        }

                        entityCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    if (entityCandidate.NavigationProperties.Count == 0)
                    {
                        foreach (var inverseProperty in entityCandidate.InverseProperties.ToList())
                        {
                            if (!AreCompatible(
                                    null, inverseProperty, entityBuilder, targetEntityBuilder))
                            {
                                entityCandidate.InverseProperties.Remove(inverseProperty);
                            }
                        }
                    }

                    continue;
                }

                var noOtherCompatibleNavigation = true;
                foreach (var otherNavigation in entityCandidate.NavigationProperties)
                {
                    if (otherNavigation != navigationProperty
                        && AreCompatible(otherNavigation, compatibleInverse, entityBuilder, targetEntityBuilder))
                    {
                        noOtherCompatibleNavigation = false;
                        break;
                    }
                }

                if (noOtherCompatibleNavigation)
                {
                    entityCandidate.NavigationProperties.Remove(navigationProperty);
                    entityCandidate.InverseProperties.Remove(compatibleInverse);

                    filteredEntityCandidates.Add(
                        new RelationshipCandidate(
                            targetEntityBuilder,
                            new[] {navigationProperty},
                            new[] {compatibleInverse},
                            entityCandidate.IsOwnership)
                    );

                    if (entityCandidate.TargetTypeBuilder.Metadata == entityBuilder.Metadata
                        && entityCandidate.NavigationProperties.Count == 0
                        && entityCandidate.InverseProperties.Count > 0)
                    {
                        var nextSelfRefCandidate = entityCandidate.InverseProperties.First();
                        if (!entityCandidate.NavigationProperties.Contains(nextSelfRefCandidate))
                        {
                            entityCandidate.NavigationProperties.Add(nextSelfRefCandidate);
                        }

                        entityCandidate.InverseProperties.Remove(nextSelfRefCandidate);
                    }

                    continue;
                }

                NextCandidate:
                break;
            }

            if (entityCandidate.NavigationProperties.Count > 0
                || entityCandidate.InverseProperties.Count > 0)
            {
                filteredEntityCandidates.Add(entityCandidate);
            }
            else if (IsImplicitlyCreatedUnusedType(entityCandidate.TargetTypeBuilder.Metadata)
                     && filteredEntityCandidates.All(
                         c => c.TargetTypeBuilder.Metadata != entityCandidate.TargetTypeBuilder.Metadata))
            {
                entityBuilder.ModelBuilder
                    .HasNoEntityType(entityCandidate.TargetTypeBuilder.Metadata);
            }
        }

        return filteredEntityCandidates;
    }

