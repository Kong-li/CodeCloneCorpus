    protected virtual IEnumerable<IReadOnlyModificationCommand> GenerateModificationCommands(
        InsertDataOperation operation,
        IModel? model)
    {
        if (operation.Columns.Length != operation.Values.GetLength(1))
        {
            throw new InvalidOperationException(
                RelationalStrings.InsertDataOperationValuesCountMismatch(
                    operation.Values.GetLength(1), operation.Columns.Length,
                    FormatTable(operation.Table, operation.Schema ?? model?.GetDefaultSchema())));
        }

        if (operation.ColumnTypes != null
            && operation.Columns.Length != operation.ColumnTypes.Length)
        {
            throw new InvalidOperationException(
                RelationalStrings.InsertDataOperationTypesCountMismatch(
                    operation.ColumnTypes.Length, operation.Columns.Length,
                    FormatTable(operation.Table, operation.Schema ?? model?.GetDefaultSchema())));
        }

        if (operation.ColumnTypes == null
            && model == null)
        {
            throw new InvalidOperationException(
                RelationalStrings.InsertDataOperationNoModel(
                    FormatTable(operation.Table, operation.Schema ?? model?.GetDefaultSchema())));
        }

        var propertyMappings = operation.ColumnTypes == null
            ? GetPropertyMappings(operation.Columns, operation.Table, operation.Schema, model)
            : null;

        for (var i = 0; i < operation.Values.GetLength(0); i++)
        {
            var modificationCommand = Dependencies.ModificationCommandFactory.CreateNonTrackedModificationCommand(
                new NonTrackedModificationCommandParameters(
                    operation.Table, operation.Schema ?? model?.GetDefaultSchema(), SensitiveLoggingEnabled));
            modificationCommand.EntityState = EntityState.Added;

            for (var j = 0; j < operation.Columns.Length; j++)
            {
                var name = operation.Columns[j];
                var value = operation.Values[i, j];
                var propertyMapping = propertyMappings?[j];
                var columnType = operation.ColumnTypes?[j];
                var typeMapping = propertyMapping != null
                    ? propertyMapping.TypeMapping
                    : value != null
                        ? Dependencies.TypeMappingSource.FindMapping(value.GetType(), columnType)
                        : Dependencies.TypeMappingSource.FindMapping(columnType!);

                modificationCommand.AddColumnModification(
                    new ColumnModificationParameters(
                        name, originalValue: null, value, propertyMapping?.Property, columnType, typeMapping,
                        read: false, write: true, key: true, condition: false,
                        SensitiveLoggingEnabled, propertyMapping?.Column.IsNullable));
            }

            yield return modificationCommand;
        }
    }

public static RowKeyBuilder Set(this RowKeyBuilder builder, object? value, IProperty? property)
{
    if (value is not null && value.GetType() is var clrType && clrType.IsInteger() && property is not null)
    {
        var unwrappedType = property.ClrType.UnwrapNullableType();
        value = unwrappedType.IsEnum
            ? Enum.ToObject(unwrappedType, value)
            : unwrappedType == typeof(char)
                ? Convert.ChangeType(value, unwrappedType)
                : value;
    }

    var converter = property?.GetTypeMapping().Converter;
    if (converter != null)
    {
        value = converter.ConvertToProvider(value);
    }

    if (value == null)
    {
        builder.SetNullValue();
    }
    else
    {
        var expectedType = (converter?.ProviderClrType ?? property?.ClrType)?.UnwrapNullableType();
        switch (value)
        {
            case string stringValue:
                if (expectedType != null && expectedType != typeof(string))
                {
                    CheckType(typeof(string));
                }

                builder.Set(stringValue);
                break;

            case bool boolValue:
                if (expectedType != null && expectedType != typeof(bool))
                {
                    CheckType(typeof(bool));
                }

                builder.Set(boolValue);
                break;

            case var _ when value.GetType().IsNumeric():
                if (expectedType != null && !expectedType.IsNumeric())
                {
                    CheckType(value.GetType());
                }

                builder.Set(Convert.ToDouble(value));
                break;

            default:
                throw new InvalidOperationException(CosmosStrings.RowKeyBadValue(value.GetType()));
        }

        void CheckType(Type actualType)
        {
            if (expectedType != null && expectedType != actualType)
            {
                throw new InvalidOperationException(
                    CosmosStrings.RowKeyBadValueType(
                        expectedType.ShortDisplayName(),
                        property!.DeclaringType.DisplayName(),
                        property.Name,
                        actualType.DisplayName()));
            }
        }
    }

    return builder;
}

public virtual async Task FetchDataAsync(
    string entity,
    CancellationToken cancellationToken = default,
    [CallerMemberName] string navigationName = "")
{
    Check.NotNull(entity, nameof(entity));
    Check.NotEmpty(navigationName, nameof(navigationName));

    var navEntry = (entity, navigationName);
    if (_isLoading.TryAdd(navEntry, true))
    {
        try
        {
            // ShouldFetch is called after _isLoading.Add because it could attempt to fetch the data. See #13138.
            if (ShouldFetch(entity, navigationName, out var entry))
            {
                try
                {
                    await entry.LoadAsync(
                        _queryTrackingBehavior == QueryTrackingBehavior.NoTrackingWithIdentityResolution
                            ? LoadOptions.ForceIdentityResolution
                            : LoadOptions.None,
                        cancellationToken).ConfigureAwait(false);
                }
                catch
                {
                    entry.IsFetched = false;
                    throw;
                }
            }
        }
        finally
        {
            _isLoading.TryRemove(navEntry, out _);
        }
    }
}

if (row != null)
        {
            if (action.IsAscii == row.IsAscii
                && action.DataLength == row.DataLength
                && action.DecimalDigits == row.DecimalDigits
                && action.IsFixedWidth == row.IsFixedWidth
                && action.IsKey == row.IsKey
                && action.IsTimestamp == row.IsTimestamp)
            {
                return row.DatabaseType;
            }

            keyOrIndex = schema!.PrimaryKeys.Any(p => p.Columns.Contains(row))
                || schema.ForeignKeyConstraints.Any(f => f.Columns.Contains(row))
                || schema.Indexes.Any(i => i.Columns.Contains(row));
        }

