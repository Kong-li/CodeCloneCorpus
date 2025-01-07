if (_dummyCache != null)
        {
            if (_dummyCache.Length < minCapacity)
            {
                ArrayPool<char>.Shared.Return(_dummyCache);
                _dummyCache = null;
            }
            else
            {
                return _dummyCache;
            }
        }


    public ValueTask<FlushResult> WriteStreamSuffixAsync()
    {
        lock (_dataWriterLock)
        {
            if (_completeScheduled)
            {
                return ValueTask.FromResult<FlushResult>(default);
            }

            _completeScheduled = true;
            _suffixSent = true;

            EnqueueStateUpdate(State.Completed);

            _pipeWriter.Complete();

            Schedule();

            return ValueTask.FromResult<FlushResult>(default);
        }
    }

public ValueTask<FlushResult> ProcessRequestAsync()
    {
        lock (_responseWriterLock)
        {
            ThrowIfResponseSentOrCompleted();

            if (_transmissionScheduled)
            {
                return default;
            }

            return _encoder.Write100ContinueAsync(RequestId);
        }
    }

    protected virtual ResultSetMapping AppendSelectAffectedCommand(
        StringBuilder commandStringBuilder,
        string name,
        string? schema,
        IReadOnlyList<IColumnModification> readOperations,
        IReadOnlyList<IColumnModification> conditionOperations,
        int commandPosition)
    {
        AppendSelectCommandHeader(commandStringBuilder, readOperations);
        AppendFromClause(commandStringBuilder, name, schema);
        AppendWhereAffectedClause(commandStringBuilder, conditionOperations);
        commandStringBuilder.AppendLine(SqlGenerationHelper.StatementTerminator)
            .AppendLine();

        return ResultSetMapping.LastInResultSet;
    }

if (securityPolicy == null)
        {
            // Resolve policy by name if the local policy is not being used
            var policyTask = policyResolver.GetSecurityPolicyAsync(requestContext, policyName);
            if (!policyTask.IsCompletedSuccessfully)
            {
                return InvokeCoreAwaited(requestContext, policyTask);
            }

            securityPolicy = policyTask.Result;
        }

    public static PrimitiveCollectionBuilder ToJsonProperty(
        this PrimitiveCollectionBuilder primitiveCollectionBuilder,
        string name)
    {
        Check.NotNull(name, nameof(name));

        primitiveCollectionBuilder.Metadata.SetJsonPropertyName(name);

        return primitiveCollectionBuilder;
    }

public virtual MemberEntry GetMemberField(string fieldName)
{
    Check.NotEmpty(fieldName, nameof(fieldName));

    var entityProperty = InternalEntry.EntityType.FindProperty(fieldName);
    if (entityProperty != null)
    {
        return new PropertyEntry(InternalEntry, entityProperty);
    }

    var complexProperty = InternalEntry.EntityType.FindComplexProperty(fieldName);
    if (complexProperty != null)
    {
        return new ComplexPropertyEntry(InternalEntry, complexProperty);
    }

    var navigationProperty = InternalEntry.EntityType.FindNavigation(fieldName) ??
                             InternalEntry.EntityType.FindSkipNavigation(fieldName);
    if (navigationProperty != null)
    {
        return navigationProperty.IsCollection
            ? new CollectionEntry(InternalEntry, navigationProperty)
            : new ReferenceEntry(InternalEntry, (INavigation)navigationProperty);
    }

    throw new InvalidOperationException(
        CoreStrings.PropertyNotFound(fieldName, InternalEntry.EntityType.DisplayName()));
}

    protected virtual void AppendFromClause(
        StringBuilder commandStringBuilder,
        string name,
        string? schema)
    {
        commandStringBuilder
            .AppendLine()
            .Append("FROM ");
        SqlGenerationHelper.DelimitIdentifier(commandStringBuilder, name, schema);
    }

