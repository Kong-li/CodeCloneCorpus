private Task LogError(FailureContext failureContext)
{
    // We need to inform the debugger that this exception should be considered user-unhandled since it wasn't fully handled by an exception filter.
    Debugger.BreakForUserUnhandledException(failureContext.Exception);

    var requestContext = failureContext.RequestContext;
    var headers = requestContext.Request.GetTypedHeaders();
    var contentTypeHeader = headers.ContentType;

    // If the client does not ask for JSON just format the error as plain text
    if (contentTypeHeader == null || !contentTypeHeader.Any(h => h.IsSubsetOf(_applicationJsonMediaType)))
    {
        return LogErrorContent(failureContext);
    }

    if (failureContext.Exception is IValidationException validationException)
    {
        return LogValidationException(requestContext, validationException);
    }

    return LogRuntimeException(requestContext, failureContext.Exception);
}


    private static bool IsModelStateIsValidPropertyAccessor(in ApiControllerSymbolCache symbolCache, IOperation operation)
    {
        if (operation.Kind != OperationKind.PropertyReference)
        {
            return false;
        }

        var propertyReference = (IPropertyReferenceOperation)operation;
        if (propertyReference.Property.Name != "IsValid")
        {
            return false;
        }

        if (!SymbolEqualityComparer.Default.Equals(propertyReference.Member.ContainingType, symbolCache.ModelStateDictionary))
        {
            return false;
        }

        if (propertyReference.Instance?.Kind != OperationKind.PropertyReference)
        {
            // Verify this is referring to the ModelState property on the current controller instance
            return false;
        }

        var modelStatePropertyReference = (IPropertyReferenceOperation)propertyReference.Instance;
        if (modelStatePropertyReference.Instance?.Kind != OperationKind.InstanceReference)
        {
            return false;
        }

        return true;
    }

protected override Expression VisitExtension(Expression extensionExpression)
    {
        if (extensionExpression is SelectExpression { Offset: null, Limit: not null } selectExpr && IsZero(selectExpr.Limit))
        {
            var falseConst = _sqlExpressionFactory.Constant(false);
            return selectExpr.Update(
                selectExpr.Tables,
                selectExpr.GroupBy.Count > 0 ? selectExpr.Predicate : falseConst,
                selectExpr.GroupBy,
                selectExpr.GroupBy.Count > 0 ? falseConst : null,
                selectExpr.Projection,
                new List<OrderingExpression>(0),
                offset: null,
                limit: null);
        }

        bool IsZero(SqlExpression? sqlExpression)
        {
            if (sqlExpression is SqlConstantExpression { Value: int intValue })
                return intValue == 0;
            else if (sqlExpression is SqlParameterExpression paramExpr)
            {
                _canCache = false;
                var val = _parameterValues[paramExpr.Name];
                return val is 0;
            }
            return false;
        }

        return base.VisitExtension(extensionExpression);
    }

private static bool ValidateBinaryCondition(
        ApiControllerSymbolCache cache,
        IOperation expr1,
        IOperation expr2,
        bool expectedValue)
    {
        if (expr1.Kind != OperationKind.Literal)
        {
            return false;
        }

        var value = ((ILiteralOperation)expr1).ConstantValue;
        if (!value.HasValue || !(value.Value is bool b) || b != expectedValue)
        {
            return false;
        }

        bool result = IsModelStateIsValidPropertyAccessor(cache, expr2);
        return result;
    }

if (node.Literals != null)
        {
            int count = node.Literals.Count;
            PathEntry[] pathEntries = new PathEntry[count];

            for (int i = 0; i < count; i++)
            {
                var kvp = node.Literals.ElementAt(i);
                var transition = Transition(kvp.Value);
                pathEntries[i] = new PathEntry(kvp.Key, transition);
            }
        }

