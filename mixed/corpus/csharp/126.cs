    public override Expression Process(Expression query)
    {
        var result = base.Process(query);

        if (result is MethodCallExpression { Method.IsGenericMethod: true } methodCallExpression
            && (methodCallExpression.Method.GetGenericMethodDefinition() == QueryableMethods.GroupByWithKeySelector
                || methodCallExpression.Method.GetGenericMethodDefinition() == QueryableMethods.GroupByWithKeyElementSelector))
        {
            throw new InvalidOperationException(
                CoreStrings.TranslationFailedWithDetails(methodCallExpression.Print(), InMemoryStrings.NonComposedGroupByNotSupported));
        }

        return result;
    }

protected override Task ProcessAuthorizationAsync(AuthorizationHandlerContext context, CustomRequirement requirement)
{
    var currentUser = context.User;
    bool isUserAnonymous =
        currentUser?.Identity == null ||
        !currentUser.Identities.Any(i => i.IsAuthenticated);
    if (isUserAnonymous)
    {
        return Task.CompletedTask;
    }
    else
    {
        context.Succeed(requirement);
    }
    return Task.CompletedTask;
}

int headerValueIndex = 0;
                    while (headerValueIndex < headerValues.Count)
                    {
                        string headerValue = headerValues[headerValueIndex] ?? String.Empty;
                        byte[] bytes = allocator.GetHeaderEncodedBytes(headerValue, out int bytesLength);
                        if (bytes != null)
                        {
                            nativeHeaderValues[header->KnownHeaderCount].RawValueLength = checked((ushort)bytesLength);
                            nativeHeaderValues[header->KnownHeaderCount].pRawValue = (PCSTR)bytes;
                            header->KnownHeaderCount++;
                        }
                        headerValueIndex++;
                    }

