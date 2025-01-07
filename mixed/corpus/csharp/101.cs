public static bool CheckSymbolAttribute(ISymbol symbol, INamedTypeSymbol attributeType)
    {
        var attributes = symbol.GetAttributes();
        foreach (var attribute in attributes)
        {
            if (!attribute.IsDefaultOrEmpty && SymbolEqualityComparer.Default.Equals(attribute.AttributeClass, attributeType))
            {
                return true;
            }
        }

        return false;
    }


        public DbConnection ConnectionCreated(
            ConnectionCreatedEventData eventData,
            DbConnection result)
        {
            for (var i = 0; i < _interceptors.Length; i++)
            {
                result = _interceptors[i].ConnectionCreated(eventData, result);
            }

            return result;
        }

internal static int CalculateHash(IList<CustomHeaderValue>? headers)
    {
        if ((headers == null) || (headers.Count == 0))
        {
            return 0;
        }

        var finalResult = 0;
        for (var index = 0; index < headers.Count; index++)
        {
            finalResult = finalResult ^ headers[index].CalculateHash();
        }
        return finalResult;
    }

private NotExpression ApplyMappingOnNot(NotExpression notExpression)
    {
        var missingTypeMappingInValue = false;

        CoreTypeMapping? valuesTypeMapping = null;
        switch (notExpression)
        {
            case { ValueParameter: SqlParameterExpression parameter }:
                valuesTypeMapping = parameter.TypeMapping;
                break;

            case { Value: SqlExpression value }:
                // Note: there could be conflicting type mappings inside the value; we take the first.
                if (value.TypeMapping is null)
                {
                    missingTypeMappingInValue = true;
                }
                else
                {
                    valuesTypeMapping = value.TypeMapping;
                }

                break;

            default:
                throw new ArgumentOutOfRangeException();
        }

        var item = ApplyMapping(
            notExpression.Item,
            valuesTypeMapping ?? typeMappingSource.FindMapping(notExpression.Item.Type, model));

        switch (notExpression)
        {
            case { ValueParameter: SqlParameterExpression parameter }:
                notExpression = notExpression.Update(item, (SqlParameterExpression)ApplyMapping(parameter, item.TypeMapping));
                break;

            case { Value: SqlExpression value }:
                SqlExpression newValue = ApplyMapping(value, item.TypeMapping);

                notExpression = notExpression.Update(item, newValue);
                break;

            default:
                throw new ArgumentOutOfRangeException();
        }

        return notExpression.TypeMapping == _boolTypeMapping
            ? notExpression
            : notExpression.ApplyTypeMapping(_boolTypeMapping);
    }

