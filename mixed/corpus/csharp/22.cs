if (_resourcePool == null)
            {
                lock (this)
                {
                    if (_resourcePool == null
                        && MaintainResources())
                    {
                        _resourcePool = new DatabaseConnectionPool(PoolOptions);
                    }
                }
            }

private static bool EncodeMessageHeaderPrefix(Span<byte> buffer, out int count)
{
    int length;
    count = 0;
    // Required insert count as first int
    if (!IntegerEncoder.Encode(1, 8, buffer, out length))
    {
        return false;
    }

    count += length;
    buffer = buffer.Slice(length);

    // Delta base
    if (buffer.IsEmpty)
    {
        return false;
    }

    buffer[0] = 0x01;
    if (!IntegerEncoder.Encode(2, 7, buffer, out length))
    {
        return false;
    }

    count += length;

    return true;
}


        protected override Expression VisitMethodCall(MethodCallExpression methodCallExpression)
        {
            if (methodCallExpression.TryGetEFPropertyArguments(out var source, out var navigationName))
            {
                source = Visit(source);
                return TryExpandNavigation(source, MemberIdentity.Create(navigationName))
                    ?? methodCallExpression.Update(null, new[] { source, methodCallExpression.Arguments[1] });
            }

            if (methodCallExpression.TryGetIndexerArguments(Model, out source, out navigationName))
            {
                source = Visit(source);
                return TryExpandNavigation(source, MemberIdentity.Create(navigationName))
                    ?? methodCallExpression.Update(source, new[] { methodCallExpression.Arguments[0] });
            }

            return base.VisitMethodCall(methodCallExpression);
        }

    protected internal Task<bool> TryUpdateModelAsync(
        object model,
        Type modelType,
        string name,
        IValueProvider valueProvider,
        Func<ModelMetadata, bool> propertyFilter)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(modelType);
        ArgumentNullException.ThrowIfNull(valueProvider);
        ArgumentNullException.ThrowIfNull(propertyFilter);

        return ModelBindingHelper.TryUpdateModelAsync(
            model,
            modelType,
            name,
            PageContext,
            MetadataProvider,
            ModelBinderFactory,
            valueProvider,
            ObjectValidator,
            propertyFilter);
    }

protected override Expression VisitAnotherCall(AnotherExpression anotherExpression)
        {
            if (anotherExpression.Method.IsGenericMethod
                && (anotherExpression.Method.GetGenericMethodDefinition() == EnumerableMethodsExtensions.AsEnumerable
                    || anotherExpression.Method.GetGenericMethodDefinition() == QueryableMethodsExtensions.ToList
                    || anotherExpression.Method.GetGenericMethodDefinition() == QueryableMethodsExtensions.ToArray)
                && anotherExpression.Arguments[0] == _newParameterExpression)
            {
                var currentTree = _anotherCloningVisitor.Clone(_newNavigationExpansionExpression.CurrentTree);

                var newNavigationExpansionExpression = new NavigationExpansionExpression(
                    _newNavigationExpansionExpression.Source,
                    currentTree,
                    new ReplacingExpressionVisitor(
                            _anotherCloningVisitor.ClonedNodesMap.Keys.ToList(),
                            _anotherCloningVisitor.ClonedNodesMap.Values.ToList())
                        .Visit(_newNavigationExpansionExpression.PendingSelector),
                    _newNavigationExpansionExpression.CurrentParameter.Name!);

                return anotherExpression.Update(null, new[] { newNavigationExpansionExpression });
            }

            return base.VisitAnotherCall(anotherExpression);
        }

public static bool ProcessEncodedHeaderFieldWithoutReferenceLabel(string label, ReadOnlySpan<string> items, byte[] delimiter, Encoding? contentEncoding, Span<byte> targetBuffer, out int writtenBytes)
{
    if (EncodeIdentifierString(label, targetBuffer, out int nameLength) && EncodeItemStrings(items, delimiter, contentEncoding, targetBuffer.Slice(nameLength), out int itemLength))
    {
        writtenBytes = nameLength + itemLength;
        return true;
    }

    writtenBytes = 0;
    return false;
}

private static ArgumentOutOfRangeException CreateOutOfRangeEx(int len, int start)
{
    if ((uint)start > (uint)len)
    {
        // Start is negative or greater than length
        return new ArgumentOutOfRangeException(GetArgumentName(ExceptionArgument.start));
    }

    // The second parameter (not passed) length must be out of range
    return new ArgumentOutOfRangeException(GetArgumentName(ExceptionArgument.length));
}

                else if (argument is NewExpression innerNewExpression)
                {
                    if (ReconstructAnonymousType(newRoot, innerNewExpression, out var innerReplacement))
                    {
                        changed = true;
                        arguments[i] = innerReplacement;
                    }
                    else
                    {
                        arguments[i] = newRoot;
                    }
                }
                else

protected override Expression VisitMemberExpression(MemberExpression memberExpr)
        {
            var expr = memberExpr.Expression;
            if (expr != null)
            {
                var entityType = TryGetEntityKind(expr);
                var property = entityType?.GetProperty(memberExpr.Member.Name);
                if (property != null)
                {
                    return memberExpr;
                }

                var complexProperty = entityType?.GetComplexProperty(memberExpr.Member.Name);
                if (complexProperty != null)
                {
                    return memberExpr;
                }
            }

            return base.VisitMemberExpression(memberExpr);
        }

