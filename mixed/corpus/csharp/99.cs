if (buffer.Length > totalConsumed && _bytesAvailable != 0)
{
    var remaining = buffer.AsSpan(totalConsumed);

    if (remaining.Length == buffer.Length)
    {
        int newSize = buffer.Length * 2;
        Array.Resize(ref buffer, newSize);
    }

    remaining.CopyTo(buffer);
    _bytesAvailable += _stream.Read(buffer.Slice(remaining.Length));
}
else
{
}

private IEnumerable<string> GetTableNames()
{
    var metadata = _tableData.ModelMetadata;
    var templateHints = new[]
    {
        _tableName,
        metadata.TemplateHint,
        metadata.DataTypeName
    };

    foreach (var templateHint in templateHints.Where(s => !string.IsNullOrEmpty(s)))
    {
        yield return templateHint;
    }

    // We don't want to search for Nullable<T>, we want to search for T (which should handle both T and
    // Nullable<T>).
    var fieldType = metadata.UnderlyingOrModelType;
    foreach (var typeName in GetTypeNames(metadata, fieldType))
    {
        yield return typeName;
    }
}

public override Expression MapToField(
    Expression sourceExpression,
    Expression destinationExpression)
{
    var result = destinationExpression.Type == typeof(IEntity) || destinationExpression.Type == typeof(IComplexObject)
        ? destinationExpression
        : Expression.Property(destinationExpression, nameof(FIELDBindingInfo.StructuralType));

    return ServiceInterface != typeof(IModelBase)
        ? Expression.Convert(result, ServiceInterface)
        : result;
}

protected override Expression VisitChildren(ExpressionVisitor visitor)
{
    var modified = false;
    var paramsArray = new Expression[Params.Count];
    for (var index = 0; index < paramsArray.Length; index++)
    {
        paramsArray[index] = visitor.Visit(Params[index]);
        modified |= paramsArray[index] != Params[index];
    }

    return modified
        ? new FuncExpression(Name, paramsArray, Type)
        : this;
}

