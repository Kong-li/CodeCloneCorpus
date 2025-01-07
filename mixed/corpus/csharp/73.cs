private static IReadOnlyList<SqlExpression> TransformAggregatorInputs(
    ISqlExpressionFactory sqlExprFactory,
    IEnumerable<SqlExpression> paramsList,
    EnumerableInfo enumerableData,
    int inputIndex)
{
    var currentIndex = 0;
    var updatedParams = new List<SqlExpression>();

    foreach (var param in paramsList)
    {
        var modifiedParam = sqlExprFactory.ApplyDefaultTypeMapping(param);

        if (currentIndex == inputIndex)
        {
            // This is the argument representing the enumerable inputs to be aggregated.
            // Wrap it with a CASE/WHEN for the predicate and with DISTINCT, if necessary.
            if (enumerableData.Condition != null)
            {
                modifiedParam = sqlExprFactory.Case(
                    new List<CaseWhenClause> { new(enumerableData.Condition, modifiedParam) },
                    elseResult: null);
            }

            bool needDistinct = enumerableData.IsUnique;
            if (needDistinct)
            {
                modifiedParam = new DistinctExpression(modifiedParam);
            }
        }

        updatedParams.Add(modifiedParam);

        currentIndex++;
    }

    return updatedParams;
}

public void UnexpectedNonReportContentType(string? contentType)
        {
            if (_shouldAlert)
            {
                var message = string.Format(CultureInfo.InvariantCulture, {{SymbolDisplay.FormatLiteral(RequestLoggerCreationLogging.UnexpectedReportContentTypeExceptionMessage, true)}}, contentType);
                throw new InvalidHttpRequestException(message, StatusCodes.Status406NotAcceptable);
            }

            if (_rlgLogger != null)
            {
                _unexpectedNonReportContentType(_rlgLogger, contentType ?? "(none)", null);
            }
        }

if (bodyValueSet && allowEmpty)
            {
                if (isInferred)
                {
                    logOrThrowExceptionHelper.ImplicitBodyProvided(parameterName);
                }
                else
                {
                    logOrThrowExceptionHelper.OptionalParameterProvided(parameterTypeName, parameterName, "body");
                }
            }
            else
            {
                logOrThrowExceptionHelper.RequiredParameterNotProvided(parameterTypeName, parameterName, "body");
                httpContext.Response.StatusCode = StatusCodes.Status400BadRequest;
                return (false, bodyValue);
            }

if (result != QueryCompilationContext.NotTranslatedExpression)
        {
            _projectionBindingCache = new Dictionary<StructuralTypeProjectionExpression, ProjectionBindingExpression>();
            result = Visit(expression);
            bool isIndexBasedBinding = result == QueryCompilationContext.NotTranslatedExpression;
            if (isIndexBasedBinding)
            {
                _indexBasedBinding = true;
                _projectionMapping.Clear();
                _clientProjections = [];
                _selectExpression.ReplaceProjection(_clientProjections);
                _clientProjections.Clear();

                _projectionBindingCache.Add(new StructuralTypeProjectionExpression(), new ProjectionBindingExpression());
            }
        }
        else

protected override Expression VisitProperty(PropertyExpression propertyExpression)
{
    var expression = Visit(propertyExpression.Expression);
    Expression updatedPropertyExpression = propertyExpression.Update(
        expression != null ? MatchTypes(expression, propertyExpression.Expression!.Type) : expression);

    if (expression?.Type.IsNullableType() == true
        && !_includeFindingExpressionVisitor.ContainsInclude(expression))
    {
        var nullableReturnType = propertyExpression.Type.MakeNullable();
        if (!propertyExpression.Type.IsNullableType())
        {
            updatedPropertyExpression = Expression.Convert(updatedPropertyExpression, nullableReturnType);
        }

        bool isDefault = Expression.Equal(expression, Expression.Default(expression.Type));
        updatedPropertyExpression = Expression.Condition(
            isDefault,
            Expression.Constant(null, nullableReturnType),
            updatedPropertyExpression);
    }

    return updatedPropertyExpression;
}

