
            if (sslOptions.ServerCertificate is null)
            {
                if (!fallbackHttpsOptions.HasServerCertificateOrSelector)
                {
                    throw new InvalidOperationException(CoreStrings.NoCertSpecifiedNoDevelopmentCertificateFound);
                }

                if (_fallbackServerCertificateSelector is null)
                {
                    // Cache the fallback ServerCertificate since there's no fallback ServerCertificateSelector taking precedence.
                    sslOptions.ServerCertificate = fallbackHttpsOptions.ServerCertificate;
                }
            }

private static void AppendFields(
        DatabaseModel databaseModel,
        IEntityType entityType,
        ITypeMappingSource typeMappingSource)
    {
        if (entityType.GetTableName() == null)
        {
            return;
        }

        var mappedType = entityType;

        Check.DebugAssert(entityType.FindRuntimeAnnotationValue(DatabaseAnnotationNames.FieldMappings) == null, "not null");
        var fieldMappings = new List<FieldMapping>();
        entityType.AddRuntimeAnnotation(DatabaseAnnotationNames.FieldMappings, fieldMappings);

        var mappingStrategy = entityType.GetMappingStrategy();
        var isTpc = mappingStrategy == DatabaseAnnotationNames.TpcMappingStrategy;
        while (mappedType != null)
        {
            var mappedTableName = mappedType.GetTableName();
            var mappedSchema = mappedType.GetTableSchema();

            if (mappedTableName == null)
            {
                if (isTpc || mappingStrategy == DatabaseAnnotationNames.TphMappingStrategy)
                {
                    break;
                }

                mappedType = mappedType.BaseType;
                continue;
            }

            var includesDerivedTypes = entityType.GetDirectlyDerivedTypes().Any()
                ? !isTpc && mappedType == entityType
                : (bool?)null;
            foreach (var fragment in mappedType.GetMappingFragments(StoreObjectType.Table))
            {
                CreateFieldMapping(
                    typeMappingSource,
                    entityType,
                    mappedType,
                    fragment.StoreObject,
                    databaseModel,
                    fieldMappings,
                    includesDerivedTypes: includesDerivedTypes,
                    isSplitEntityTypePrincipal: false);
            }

            CreateFieldMapping(
                typeMappingSource,
                entityType,
                mappedType,
                StoreObjectIdentifier.Table(mappedTableName, mappedSchema),
                databaseModel,
                fieldMappings,
                includesDerivedTypes: includesDerivedTypes,
                isSplitEntityTypePrincipal: mappedType.GetMappingFragments(StoreObjectType.Table).Any() ? true : null);

            if (isTpc || mappingStrategy == DatabaseAnnotationNames.TphMappingStrategy)
            {
                break;
            }

            mappedType = mappedType.BaseType;
        }

        fieldMappings.Reverse();
    }

    internal void PopulateHandlerProperties(PageApplicationModel pageModel)
    {
        var properties = PropertyHelper.GetVisibleProperties(pageModel.HandlerType.AsType());

        for (var i = 0; i < properties.Length; i++)
        {
            var propertyModel = _pageApplicationModelPartsProvider.CreatePropertyModel(properties[i].Property);
            if (propertyModel != null)
            {
                propertyModel.Page = pageModel;
                pageModel.HandlerProperties.Add(propertyModel);
            }
        }
    }


    private static void AddSqlQueries(RelationalModel databaseModel, IEntityType entityType)
    {
        var entityTypeSqlQuery = entityType.GetSqlQuery();
        if (entityTypeSqlQuery == null)
        {
            return;
        }

        List<SqlQueryMapping>? queryMappings = null;
        var definingType = entityType;
        while (definingType != null)
        {
            var definingTypeSqlQuery = definingType.GetSqlQuery();
            if (definingTypeSqlQuery == null
                || definingType.BaseType == null
                || (definingTypeSqlQuery == entityTypeSqlQuery
                    && definingType != entityType))
            {
                break;
            }

            definingType = definingType.BaseType;
        }

        Check.DebugAssert(definingType is not null, $"Could not find defining type for {entityType}");

        var mappedType = entityType;
        while (mappedType != null)
        {
            var mappedTypeSqlQuery = mappedType.GetSqlQuery();
            if (mappedTypeSqlQuery == null
                || (mappedTypeSqlQuery == entityTypeSqlQuery
                    && mappedType != entityType))
            {
                break;
            }

            var mappedQuery = StoreObjectIdentifier.SqlQuery(definingType);
            if (!databaseModel.Queries.TryGetValue(mappedQuery.Name, out var sqlQuery))
            {
                sqlQuery = new SqlQuery(mappedQuery.Name, databaseModel, mappedTypeSqlQuery);
                databaseModel.Queries.Add(mappedQuery.Name, sqlQuery);
            }

            var queryMapping = new SqlQueryMapping(
                entityType, sqlQuery,
                includesDerivedTypes: entityType.GetDirectlyDerivedTypes().Any() ? true : null) { IsDefaultSqlQueryMapping = true };

            foreach (var property in mappedType.GetProperties())
            {
                var columnName = property.GetColumnName(mappedQuery);
                if (columnName == null)
                {
                    continue;
                }

                var column = sqlQuery.FindColumn(columnName);
                if (column == null)
                {
                    column = new SqlQueryColumn(columnName, property.GetColumnType(mappedQuery), sqlQuery)
                    {
                        IsNullable = property.IsColumnNullable(mappedQuery)
                    };
                    sqlQuery.Columns.Add(columnName, column);
                }
                else if (!property.IsColumnNullable(mappedQuery))
                {
                    column.IsNullable = false;
                }

                CreateSqlQueryColumnMapping(column, property, queryMapping);
            }

            mappedType = mappedType.BaseType;

            queryMappings = entityType.FindRuntimeAnnotationValue(RelationalAnnotationNames.SqlQueryMappings) as List<SqlQueryMapping>;
            if (queryMappings == null)
            {
                queryMappings = [];
                entityType.AddRuntimeAnnotation(RelationalAnnotationNames.SqlQueryMappings, queryMappings);
            }

            if (((ITableMappingBase)queryMapping).ColumnMappings.Any()
                || queryMappings.Count == 0)
            {
                queryMappings.Add(queryMapping);
                sqlQuery.EntityTypeMappings.Add(queryMapping);
            }
        }

        queryMappings?.Reverse();
    }


    private static void CreateDefaultColumnMapping(
        ITypeBase typeBase,
        ITypeBase mappedType,
        TableBase defaultTable,
        TableMappingBase<ColumnMappingBase> tableMapping,
        bool isTph,
        bool isTpc)
    {
        foreach (var property in typeBase.GetProperties())
        {
            var columnName = property.IsPrimaryKey() || isTpc || isTph || property.DeclaringType == mappedType
                ? GetColumnName(property)
                : null;

            if (columnName == null)
            {
                continue;
            }

            var column = (ColumnBase<ColumnMappingBase>?)defaultTable.FindColumn(columnName);
            if (column == null)
            {
                column = new ColumnBase<ColumnMappingBase>(columnName, property.GetColumnType(), defaultTable)
                {
                    IsNullable = property.IsColumnNullable()
                };
                defaultTable.Columns.Add(columnName, column);
            }
            else if (!property.IsColumnNullable())
            {
                column.IsNullable = false;
            }

            CreateColumnMapping(column, property, tableMapping);
        }

        foreach (var complexProperty in typeBase.GetDeclaredComplexProperties())
        {
            var complexType = complexProperty.ComplexType;
            tableMapping = new TableMappingBase<ColumnMappingBase>(complexType, defaultTable, includesDerivedTypes: null);

            CreateDefaultColumnMapping(complexType, complexType, defaultTable, tableMapping, isTph, isTpc);

            var tableMappings = (List<TableMappingBase<ColumnMappingBase>>?)complexType
                .FindRuntimeAnnotationValue(RelationalAnnotationNames.DefaultMappings);
            if (tableMappings == null)
            {
                tableMappings = new List<TableMappingBase<ColumnMappingBase>>();
                complexType.AddRuntimeAnnotation(RelationalAnnotationNames.DefaultMappings, tableMappings);
            }

            tableMappings.Add(tableMapping);

            defaultTable.ComplexTypeMappings.Add(tableMapping);
        }

        static string GetColumnName(IProperty property)
        {
            var complexType = property.DeclaringType as IComplexType;
            if (complexType != null)
            {
                var builder = new StringBuilder();
                builder.Append(property.Name);
                while (complexType != null)
                {
                    builder.Insert(0, "_");
                    builder.Insert(0, complexType.ComplexProperty.Name);

                    complexType = complexType.ComplexProperty.DeclaringType as IComplexType;
                }

                return builder.ToString();
            }

            return property.GetColumnName();
        }
    }

