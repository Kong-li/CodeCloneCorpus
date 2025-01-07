public virtual InterceptionResult<DbCommand> CommandPreparation(
    IRelationalConnection connection,
    DbCommandMethod commandMethod,
    DbContext? context,
    Guid commandId,
    Guid connectionId,
    DateTimeOffset startTime,
    CommandSource commandSource)
{
    _ignoreCommandCreateExpiration = startTime + _loggingCacheTime;

    var definition = RelationalResources.LogCommandPreparation(this);

    if (ShouldLog(definition))
    {
        _ignoreCommandCreateExpiration = default;

        definition.Log(this, commandMethod.ToString());
    }

    if (NeedsEventData<ICommandEventInterceptor>(
        definition, out var interceptor, out var diagnosticSourceEnabled, out var simpleLogEnabled))
    {
        _ignoreCommandCreateExpiration = default;

        var eventData = BroadcastCommandPreparation(
            connection.DbConnection,
            context,
            commandMethod,
            commandId,
            connectionId,
            async: false,
            startTime,
            definition,
            diagnosticSourceEnabled,
            simpleLogEnabled,
            commandSource);

        if (interceptor != null)
        {
            return interceptor.CommandPreparation(eventData, default);
        }
    }

    return default;
}

if (serviceProvider != null)
        {
            var args = ActionCall.GetParameters().Select(
                (p, i) => Expression.Parameter(p.ParameterType, "arg" + i)).ToArray();

            return Expression.Condition(
                Expression.ReferenceEqual(serviceProvider, Expression.Constant(null)),
                Expression.Constant(null, ReturnType),
                Expression.Lambda(
                    Expression.Call(
                        serviceProvider,
                        ActionCall,
                        args),
                    args));
        }


    internal static bool IsProblematicParameter(in SymbolCache symbolCache, IParameterSymbol parameter)
    {
        if (parameter.GetAttributes(symbolCache.FromBodyAttribute).Any())
        {
            // Ignore input formatted parameters.
            return false;
        }

        if (SpecifiesModelType(in symbolCache, parameter))
        {
            // Ignore parameters that specify a model type.
            return false;
        }

        if (!IsComplexType(parameter.Type))
        {
            return false;
        }

        var parameterName = GetName(symbolCache, parameter);

        var type = parameter.Type;
        while (type != null)
        {
            foreach (var member in type.GetMembers())
            {
                if (member.DeclaredAccessibility != Accessibility.Public ||
                    member.IsStatic ||
                    member.Kind != SymbolKind.Property)
                {
                    continue;
                }

                var propertyName = GetName(symbolCache, member);
                if (string.Equals(parameterName, propertyName, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            type = type.BaseType;
        }

        return false;
    }

static string ProcessEventExecution(EventDefinitionBase eventDef, EventData eventData)
        {
            var definition = (EventDefinition<string, CommandType, int, string, string>)eventDef;
            var data = (CommandEventData)eventData;
            string message = definition.GenerateMessage(
                data.Command.Parameters.FormatParameters(data.LogParameterValues),
                data.Command.CommandType,
                data.Command.CommandTimeout,
                Environment.NewLine,
                data.Command.CommandText.TrimEnd());
            return message;
        }

public RowModificationSettings(
    IUpdateRecord? record,
    IField? field,
    IRowBase row,
    Func<int> generateSettingId,
    RelationalTypeMapping typeMapping,
    bool valueIsRead,
    bool valueIsWrite,
    bool rowIsKey,
    bool rowIsCondition,
    bool sensitiveLoggingEnabled)
{
    Row = row;
    RowName = row.Name;
    OriginalValue = null;
    Value = null;
    Field = field;
    RowType = row.StoreType;
    TypeMapping = typeMapping;
    IsRead = valueIsRead;
    IsWrite = valueIsWrite;
    IsKey = rowIsKey;
    IsCondition = rowIsCondition;
    SensitiveLoggingEnabled = sensitiveLoggingEnabled;
    IsNullable = row.IsNullable;

    GenerateSettingId = generateSettingId;
    Record = record;
    JsonPath = null;
}

while (currentType != null)
        {
            foreach (var member in currentType.GetMembers())
            {
                if (!IsPublicMember(member) || member.IsStatic || member.Kind != SymbolKind.Property)
                {
                    continue;
                }

                var propertyName = GetPropertyName(symbolCache, member);
                if (String.Equals(propertyName, parameterName, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            currentType = currentType.BaseType;
        }

switch (sqlUnaryExpression.OperatorType)
        {
            case ExpressionType.Equal:
            case ExpressionType.NotEqual:
            case ExpressionType.Not:
                if (sqlUnaryExpression.Type == typeof(bool))
                {
                    resultTypeMapping = _boolTypeMapping;
                    resultType = typeof(bool);
                    operand = ApplyDefaultTypeMapping(sqlUnaryExpression.Operand);
                }
                break;

            case ExpressionType.Convert:
                resultTypeMapping = typeMapping;
                // Since we are applying convert, resultTypeMapping decides the clrType
                resultType = resultTypeMapping?.ClrType ?? sqlUnaryExpression.Type;
                operand = ApplyDefaultTypeMapping(sqlUnaryExpression.Operand);
                break;

            case ExpressionType.Not:
            case ExpressionType.Negate:
            case ExpressionType.OnesComplement:
                resultTypeMapping = typeMapping;
                // While Not is logical, negate is numeric hence we use clrType from TypeMapping
                resultType = resultTypeMapping?.ClrType ?? sqlUnaryExpression.Type;
                operand = ApplyTypeMapping(sqlUnaryExpression.Operand, typeMapping);
                break;

            default:
                throw new InvalidOperationException(
                    RelationalStrings.UnsupportedOperatorForSqlExpression(
                        sqlUnaryExpression.OperatorType, typeof(SqlUnaryExpression).ShortDisplayName()));
        }

