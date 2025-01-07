protected virtual SqlExpression TransformMatch(MatchExpression matchExpression, bool allowOptimizedExpansion, out bool nullable)
{
    var subject = Visit(matchExpression.Subject, out var subjectNullable);
    var pattern = Visit(matchExpression.Pattern, out var patternNullable);
    var escapeChar = Visit(matchExpression.EscapeChar, out var escapeCharNullable);

    SqlExpression result = matchExpression.Update(subject, pattern, escapeChar);

    if (UseRelationalNulls)
    {
        nullable = subjectNullable || patternNullable || escapeCharNullable;

        return result;
    }

    nullable = false;

    // The null semantics behavior we implement for MATCH is that it only returns true when both sides are non-null and match; any other
    // input returns false:
    // foo MATCH f% -> true
    // foo MATCH null -> false
    // null MATCH f% -> false
    // null MATCH null -> false

    if (IsNull(subject) || IsNull(pattern) || IsNull(escapeChar))
    {
        return _sqlExpressionFactory.Constant(false, matchExpression.TypeMapping);
    }

    // A constant match-all pattern (%) returns true for all cases, except where the subject is null:
    // nullable_foo MATCH % -> foo IS NOT NULL
    // non_nullable_foo MATCH % -> true
    if (pattern is SqlConstantExpression { Value: "%" })
    {
        return subjectNullable
            ? _sqlExpressionFactory.IsNotNull(subject)
            : _sqlExpressionFactory.Constant(true, matchExpression.TypeMapping);
    }

    if (!allowOptimizedExpansion)
    {
        if (subjectNullable)
        {
            result = _sqlExpressionFactory.AndAlso(result, GenerateNotNullCheck(subject));
        }

        if (patternNullable)
        {
            result = _sqlExpressionFactory.AndAlso(result, GenerateNotNullCheck(pattern));
        }

        if (escapeChar is not null && escapeCharNullable)
        {
            result = _sqlExpressionFactory.AndAlso(result, GenerateNotNullCheck(escapeChar));
        }
    }

    return result;

    SqlExpression GenerateNotNullCheck(SqlExpression operand)
        => _sqlExpressionFactory.Not(
            ProcessNullNotNull(
                _sqlExpressionFactory.IsNull(operand), operandNullable: true));
}

    public ILogger ForContext(ILogEventEnricher enricher)
    {
        if (enricher == null!)
            return this; // No context here, so little point writing to SelfLog.

        return new Logger(
            _messageTemplateProcessor,
            _minimumLevel,
            _levelSwitch,
            this,
            enricher,
            null,
#if FEATURE_ASYNCDISPOSABLE
            null,
#endif
            _overrideMap);
    }

public static bool CheckRowInternal(
    this IReadOnlyForeignKey fk,
    StoreObjectIdentifier obj)
{
    var entity = fk.DeclaringEntityType;
    if (entity.FindPrimaryKey() == null
        || entity.IsMappedToJson()
        || !fk.PrincipalKey.IsPrimaryKey()
        || fk.PrincipalEntityType.IsAssignableFrom(entity)
        || !fk.Properties.SequenceEqual(fk.PrincipalKey.Properties)
        || !IsLinked(fk, obj))
    {
        return false;
    }

    return true;

    bool IsLinked(IReadOnlyForeignKey foreignKey, StoreObjectIdentifier storeObject)
        => (StoreObjectIdentifier.Create(foreignKey.DeclaringEntityType, storeObject.StoreObjectType) == storeObject
                || foreignKey.DeclaringEntityType.GetMappingFragments(storeObject.StoreObjectType).Any(f => f.StoreObject == storeObject))
            && (StoreObjectIdentifier.Create(foreignKey.PrincipalEntityType, storeObject.StoreObjectType) == storeObject
                || foreignKey.PrincipalEntityType.GetMappingFragments(storeObject.StoreObjectType).Any(f => f.StoreObject == storeObject));
}

public override void BeginAnalysis(AnalysisContext context)
    {
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }

        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.RegisterCompilationStartAction(() => OnCompilationStart(context));
        context.EnableConcurrentExecution();
    }

    protected virtual SqlExpression VisitSqlParameter(
        SqlParameterExpression sqlParameterExpression,
        bool allowOptimizedExpansion,
        out bool nullable)
    {
        var parameterValue = ParameterValues[sqlParameterExpression.Name];
        nullable = parameterValue == null;

        if (nullable)
        {
            return _sqlExpressionFactory.Constant(
                null,
                sqlParameterExpression.Type,
                sqlParameterExpression.TypeMapping);
        }

        if (sqlParameterExpression.ShouldBeConstantized)
        {
            DoNotCache();

            return _sqlExpressionFactory.Constant(
                parameterValue,
                sqlParameterExpression.Type,
                sqlParameterExpression.TypeMapping);
        }

        return sqlParameterExpression;
    }

