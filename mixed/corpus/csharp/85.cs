public virtual void HandleEntityInitialization(
    ICustomModelBuilder modelBuilder,
    ICustomContext<ICustomModelBuilder> context)
{
    foreach (var entityType in modelBuilder.Metadata.GetEntities())
    {
        foreach (var property in entityType.GetDeclaredProperties())
        {
            SqlClientValueGenerationStrategy? strategy = null;
            var declaringTable = property.GetMappedStoreObjects(StoreObjectType.Table).FirstOrDefault();
            if (declaringTable.Name != null!)
            {
                strategy = property.GetValueGenerationStrategy(declaringTable, Dependencies.TypeMappingSource);
                if (strategy == SqlClientValueGenerationStrategy.None
                    && !IsStrategyNoneNeeded(property, declaringTable))
                {
                    strategy = null;
                }
            }
            else
            {
                var declaringView = property.GetMappedStoreObjects(StoreObjectType.View).FirstOrDefault();
                if (declaringView.Name != null!)
                {
                    strategy = property.GetValueGenerationStrategy(declaringView, Dependencies.TypeMappingSource);
                    if (strategy == SqlClientValueGenerationStrategy.None
                        && !IsStrategyNoneNeeded(property, declaringView))
                    {
                        strategy = null;
                    }
                }
            }

            // Needed for the annotation to show up in the model snapshot
            if (strategy != null
                && declaringTable.Name != null)
            {
                property.Builder.HasValueGenerationStrategy(strategy);

                if (strategy == SqlClientValueGenerationStrategy.Sequence)
                {
                    var sequence = modelBuilder.HasSequence(
                        property.GetSequenceName(declaringTable)
                        ?? entityType.GetRootType().ShortName() + modelBuilder.Metadata.GetSequenceNameSuffix(),
                        property.GetSequenceSchema(declaringTable)
                        ?? modelBuilder.Metadata.GetSequenceSchema()).Metadata;

                    property.Builder.HasDefaultValueSql(
                        RelationalDependencies.UpdateSqlGenerator.GenerateObtainNextSequenceValueOperation(
                            sequence.Name, sequence.Schema));
                }
            }
        }
    }

    bool IsStrategyNoneNeeded(IReadOnlyProperty property, StoreObjectIdentifier storeObject)
    {
        if (property.ValueGenerated == ValueGenerated.OnAdd
            && !property.TryGetDefaultValue(storeObject, out _)
            && property.GetDefaultValueSql(storeObject) == null
            && property.GetComputedColumnSql(storeObject) == null
            && property.DeclaringType.Model.GetValueGenerationStrategy() == SqlClientValueGenerationStrategy.IdentityColumn)
        {
            var providerClrType = (property.GetValueConverter()
                    ?? (property.FindRelationalTypeMapping(storeObject)
                        ?? Dependencies.TypeMappingSource.FindMapping((IProperty)property))?.Converter)
                ?.ProviderClrType.UnwrapNullableType();

            return providerClrType != null
                && (providerClrType.IsInteger() || providerClrType == typeof(decimal));
        }

        return false;
    }
}

    public static void UpdateRootComponentsCore(string operationsJson)
    {
        try
        {
            var operations = DeserializeOperations(operationsJson);
            Instance.OnUpdateRootComponents?.Invoke(operations);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error deserializing root component operations: {ex}");
        }
    }

public object InitiateControllerContext(ControllerRequestInfo info)
{
    ArgumentNullException.ThrowIfNull(info);

    if (info.MethodDescriptor == null)
    {
        throw new ArgumentException(Resources.FormatPropertyOfTypeCannotBeNull(
            nameof(ControllerRequestInfo.MethodDescriptor),
            nameof(ControllerRequestInfo)));
    }

    var controller = _controllerFactory(info);
    foreach (var activationRule in _activationRules)
    {
        activationRule.Activate(info, controller);
    }

    return controller;
}

private TagHelperContent ProcessCoreItem(dynamic item)
{
    if (_hasContent)
    {
        _isModified = true;
        Buffer.Add(item);
    }
    else
    {
        _singleContent = item;
        _isSingleContentSet = true;
    }

    _hasContent = true;

    return this;
}

