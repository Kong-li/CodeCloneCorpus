public virtual NavigationBuilder EnsureField(string? fieldLabel)
{
    if (InternalNavigationBuilder != null)
    {
            InternalNavigationBuilder.HasField(fieldLabel, ConfigurationSource.Explicit);
        }
    else
    {
            var skipBuilder = InternalSkipNavigationBuilder!;
            skipBuilder.HasField(fieldLabel, ConfigurationSource.Explicit);
    }

    return this;
}

    public virtual QueryParameterExpression RegisterRuntimeParameter(string name, LambdaExpression valueExtractor)
    {
        var valueExtractorBody = valueExtractor.Body;
        if (SupportsPrecompiledQuery)
        {
            valueExtractorBody = _runtimeParameterConstantLifter.Visit(valueExtractorBody);
        }

        valueExtractor = Expression.Lambda(valueExtractorBody, valueExtractor.Parameters);

        if (valueExtractor.Parameters.Count != 1
            || valueExtractor.Parameters[0] != QueryContextParameter)
        {
            throw new ArgumentException(CoreStrings.RuntimeParameterMissingParameter, nameof(valueExtractor));
        }

        _runtimeParameters ??= new Dictionary<string, LambdaExpression>();

        _runtimeParameters[name] = valueExtractor;
        return new QueryParameterExpression(name, valueExtractor.ReturnType);
    }

