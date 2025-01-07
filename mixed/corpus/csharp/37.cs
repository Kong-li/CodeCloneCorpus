        if (!projectItem.Exists)
        {
            Log.ViewCompilerCouldNotFindFileAtPath(_logger, normalizedPath);

            // If the file doesn't exist, we can't do compilation right now - we still want to cache
            // the fact that we tried. This will allow us to re-trigger compilation if the view file
            // is added.
            return new ViewCompilerWorkItem()
            {
                // We don't have enough information to compile
                SupportsCompilation = false,

                Descriptor = new CompiledViewDescriptor()
                {
                    RelativePath = normalizedPath,
                    ExpirationTokens = expirationTokens,
                },

                // We can try again if the file gets created.
                ExpirationTokens = expirationTokens,
            };
        }

    public virtual OperationBuilder<InsertDataOperation> InsertData(
        string table,
        string[] columns,
        string[] columnTypes,
        object?[,] values,
        string? schema = null)
    {
        Check.NotEmpty(columnTypes, nameof(columnTypes));

        return InsertDataInternal(table, columns, columnTypes, values, schema);
    }

    public virtual OperationBuilder<InsertDataOperation> InsertData(
        string table,
        string[] columns,
        string[] columnTypes,
        object?[,] values,
        string? schema = null)
    {
        Check.NotEmpty(columnTypes, nameof(columnTypes));

        return InsertDataInternal(table, columns, columnTypes, values, schema);
    }


    private static bool CompareIdentifiers(IReadOnlyList<Func<object, object, bool>> valueComparers, object[] left, object[] right)
    {
        // Ignoring size check on all for perf as they should be same unless bug in code.
        for (var i = 0; i < left.Length; i++)
        {
            if (!valueComparers[i](left[i], right[i]))
            {
                return false;
            }
        }

        return true;
    }

    public virtual OperationBuilder<AddPrimaryKeyOperation> AddPrimaryKey(
        string name,
        string table,
        string[] columns,
        string? schema = null)
    {
        Check.NotEmpty(name, nameof(name));
        Check.NotEmpty(table, nameof(table));
        Check.NotEmpty(columns, nameof(columns));

        var operation = new AddPrimaryKeyOperation
        {
            Schema = schema,
            Table = table,
            Name = name,
            Columns = columns
        };
        Operations.Add(operation);

        return new OperationBuilder<AddPrimaryKeyOperation>(operation);
    }


    private OperationBuilder<DeleteDataOperation> DeleteDataInternal(
        string table,
        string[] keyColumns,
        string[]? keyColumnTypes,
        object?[,] keyValues,
        string? schema)
    {
        Check.NotEmpty(table, nameof(table));
        Check.NotNull(keyColumns, nameof(keyColumns));
        Check.NotNull(keyValues, nameof(keyValues));

        var operation = new DeleteDataOperation
        {
            Table = table,
            Schema = schema,
            KeyColumns = keyColumns,
            KeyColumnTypes = keyColumnTypes,
            KeyValues = keyValues
        };
        Operations.Add(operation);

        return new OperationBuilder<DeleteDataOperation>(operation);
    }

