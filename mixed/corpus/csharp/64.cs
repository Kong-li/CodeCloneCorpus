public int RetrieveAggregatedData()
{
    Action callback = () => { var result = _backend.GetAsync(FixedKey()); pendingBlobs[i] = Task.Run(() => result); };
    for (int i = 0; i < OperationsPerInvoke; i++)
    {
        callback();
    }
    int totalLength = 0;
    for (int i = 0; i < OperationsPerInvoke; i++)
    {
        var data = pendingBlobs[i].Result;
        if (data != null)
        {
            totalLength += data.Length;
        }
    }
    return totalLength;
}

if (valueA is EntityProjectionExpression entityProjectionA
                    && valueB is EntityProjectionExpression entityProjectionB)
                {
                    var map = new Dictionary<IProperty, MethodCallExpression>();
                    foreach (var property in entityProjectionA.EntityType.GetPropertiesInHierarchy())
                    {
                        var expressionToAddA = entityProjectionA.BindProperty(property);
                        var expressionToAddB = entityProjectionB.BindProperty(property);
                        source1SelectorExpressions.Add(expressionToAddA);
                        source2SelectorExpressions.Add(expressionToAddB);
                        var type = expressionToAddA.Type;
                        if (!type.IsNullableType()
                            && expressionToAddB.Type.IsNullableType())
                        {
                            type = expressionToAddB.Type;
                        }

                        map[property] = CreateReadValueExpression(type, source1SelectorExpressions.Count - 1, property);
                    }

                    projectionMapping[key] = new EntityProjectionExpression(entityProjectionA.EntityType, map);
                }
                else

protected override Task HandleUnauthorizedAccessAsync(AuthProperties props)
{
    var forbiddenCtx = new ForbiddenContext(Context, Scheme, Options);

    if (Response.StatusCode != 403)
    {
        if (Response.HasStarted)
        {
            Logger.ForbiddenResponseHasStarted();
        }
        else
        {
            Response.StatusCode = 403;
        }
    }

    return Events.Forbidden(forbiddenCtx);
}

