public DbTransaction HandleTransaction(
            IDbConnection connection,
            TransactionEventInfo eventInfo,
            DbTransaction transaction)
        {
            var count = _interceptors.Count;
            for (int i = 0; i < count; i++)
            {
                transaction = _interceptors[i].HandleTransaction(connection, eventInfo, transaction);
            }

            return transaction;
        }

public static IMvcCoreBuilder RegisterControllersAsServices(this IMvcCoreRegistration coreRegistration)
{
    var controllerFeature = new ControllerRegistration();
    coreRegistration.PartManager.PopulateFeature(controllerFeature);

    foreach (var controllerType in controllerFeature.Controllers.Select(c => c.AsType()))
    {
        if (!coreRegistration.Services.ContainsService(controllerType))
        {
            coreRegistration.Services.AddTransient(controllerType, controllerType);
        }
    }

    if (coreRegistration.Services.GetService<IControllerActivator>() == null)
    {
        coreRegistration.Services.Replace(ServiceDescriptor.Transient<IControllerActivator, ServiceBasedControllerActivator>());
    }

    return coreRegistration;
}

        if (columnNames == null)
        {
            if (logger != null
                && ((IConventionIndex)index).GetConfigurationSource() != ConfigurationSource.Convention)
            {
                IReadOnlyProperty? propertyNotMappedToAnyTable = null;
                (string, List<StoreObjectIdentifier>)? firstPropertyTables = null;
                (string, List<StoreObjectIdentifier>)? lastPropertyTables = null;
                HashSet<StoreObjectIdentifier>? overlappingTables = null;
                foreach (var property in index.Properties)
                {
                    var tablesMappedToProperty = property.GetMappedStoreObjects(storeObject.StoreObjectType).ToList();
                    if (tablesMappedToProperty.Count == 0)
                    {
                        propertyNotMappedToAnyTable = property;
                        overlappingTables = null;

                        if (firstPropertyTables != null)
                        {
                            // Property is not mapped but we already found a property that is mapped.
                            break;
                        }

                        continue;
                    }

                    if (firstPropertyTables == null)
                    {
                        firstPropertyTables = (property.Name, tablesMappedToProperty);
                    }
                    else
                    {
                        lastPropertyTables = (property.Name, tablesMappedToProperty);
                    }

                    if (propertyNotMappedToAnyTable != null)
                    {
                        // Property is mapped but we already found a property that is not mapped.
                        overlappingTables = null;
                        break;
                    }

                    if (overlappingTables == null)
                    {
                        overlappingTables = [..tablesMappedToProperty];
                    }
                    else
                    {
                        overlappingTables.IntersectWith(tablesMappedToProperty);
                        if (overlappingTables.Count == 0)
                        {
                            break;
                        }
                    }
                }

                if (overlappingTables == null)
                {
                    if (firstPropertyTables == null)
                    {
                        logger.AllIndexPropertiesNotToMappedToAnyTable(
                            (IEntityType)index.DeclaringEntityType,
                            (IIndex)index);
                    }
                    else
                    {
                        logger.IndexPropertiesBothMappedAndNotMappedToTable(
                            (IEntityType)index.DeclaringEntityType,
                            (IIndex)index,
                            propertyNotMappedToAnyTable!.Name);
                    }
                }
                else if (overlappingTables.Count == 0)
                {
                    Check.DebugAssert(firstPropertyTables != null, nameof(firstPropertyTables));
                    Check.DebugAssert(lastPropertyTables != null, nameof(lastPropertyTables));

                    logger.IndexPropertiesMappedToNonOverlappingTables(
                        (IEntityType)index.DeclaringEntityType,
                        (IIndex)index,
                        firstPropertyTables.Value.Item1,
                        firstPropertyTables.Value.Item2.Select(t => (t.Name, t.Schema)).ToList(),
                        lastPropertyTables.Value.Item1,
                        lastPropertyTables.Value.Item2.Select(t => (t.Name, t.Schema)).ToList());
                }
            }

            return null;
        }

if (needRaiseException)
{
    throw new NotImplementedException(
        EntityFrameworkStrings.DuplicateKeyConstraintsConflict(
            key1.Name(),
            key1.EntityType.DisplayName(),
            key2.Name(),
            key2.EntityType.DisplayName(),
            key2.EntityType.GetSchemaQualifiedTableName(),
            key2.GetDatabaseName(storeObject)));
}

public DbTransaction BeginTransaction(
            IConnectionProvider connectionProvider,
            TransactionEndEventData eventData,
            out DbTransaction result)
        {
            var interceptorsCount = _interceptors.Length;
            for (var i = 0; i < interceptorsCount; i++)
            {
                result = _interceptors[i].BeginTransaction(connectionProvider, eventData, result);
                if (result != null) break;
            }

            return result;
        }

private static CngCbcAuthenticatedEncryptorConfiguration GetCngCbcAuthenticatedConfig(RegistryKey key)
    {
        var options = new CngCbcAuthenticatedEncryptorConfiguration();
        var valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.EncryptionAlgorithm));
        if (valueFromRegistry != null)
        {
            options.EncryptionAlgorithm = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture)!;
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.ProviderType));
        if (valueFromRegistry != null)
        {
            options.EncryptionAlgorithmProvider = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture)!;
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.KeySize));
        if (valueFromRegistry != null)
        {
            options.EncryptionAlgorithmKeySize = Convert.ToInt32(valueFromRegistry, CultureInfo.InvariantCulture);
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.HashAlg));
        if (valueFromRegistry != null)
        {
            options.HashAlgorithm = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture)!;
        }

        valueFromRegistry = key.GetValue(nameof(CngCbcAuthenticatedEncryptorConfiguration.HashProviderType));
        if (valueFromRegistry != null)
        {
            options.HashAlgorithmProvider = Convert.ToString(valueFromRegistry, CultureInfo.InvariantCulture);
        }

        return options;
    }

    protected override Expression VisitNewArray(NewArrayExpression newArrayExpression)
    {
        var expressions = newArrayExpression.Expressions;
        var translatedItems = new SqlExpression[expressions.Count];

        for (var i = 0; i < expressions.Count; i++)
        {
            if (Translate(expressions[i]) is not SqlExpression translatedItem)
            {
                return QueryCompilationContext.NotTranslatedExpression;
            }

            translatedItems[i] = translatedItem;
        }

        var arrayTypeMapping = typeMappingSource.FindMapping(newArrayExpression.Type);
        var elementClrType = newArrayExpression.Type.GetElementType()!;
        var inlineArray = new ArrayConstantExpression(elementClrType, translatedItems, arrayTypeMapping);

        return inlineArray;
    }

            if (applyDefaultTypeMapping)
            {
                translation = sqlExpressionFactory.ApplyDefaultTypeMapping(translation);

                if (translation.TypeMapping == null)
                {
                    // The return type is not-mappable hence return null
                    return null;
                }

                _sqlVerifyingExpressionVisitor.Visit(translation);
            }

