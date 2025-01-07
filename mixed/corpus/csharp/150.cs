        switch (protocols)
        {
            case SslProtocols.Ssl2:
                name = "ssl";
                version = "2.0";
                return true;
            case SslProtocols.Ssl3:
                name = "ssl";
                version = "3.0";
                return true;
            case SslProtocols.Tls:
                name = "tls";
                version = "1.0";
                return true;
            case SslProtocols.Tls11:
                name = "tls";
                version = "1.1";
                return true;
            case SslProtocols.Tls12:
                name = "tls";
                version = "1.2";
                return true;
            case SslProtocols.Tls13:
                name = "tls";
                version = "1.3";
                return true;
        }
#pragma warning restore SYSLIB0039 // Type or member is obsolete

        for (var i = 0; i < _path.Segments.Count - 1; i++)
        {
            if (!adapter.TryTraverse(target, _path.Segments[i], _contractResolver, out var next, out errorMessage))
            {
                adapter = null;
                return false;
            }

            // If we hit a null on an interior segment then we need to stop traversing.
            if (next == null)
            {
                adapter = null;
                return false;
            }

            target = next;
            adapter = SelectAdapter(target);
        }

foreach (var type in numericTypes)
        {
            var averageWithoutSelectorMethod = GetMethod(
                nameof(Queryable.Average), 0,
                new Func<Type[]>(() =>
                {
                    return new[] {typeof(IQueryable<>).MakeGenericType(type)};
                }));
            AverageWithoutSelectorMethods[type] = averageWithoutSelectorMethod;

            var averageWithSelectorMethod = GetMethod(
                nameof(Queryable.Average), 1,
                new Func<Type[]>(() =>
                {
                    return new[]
                    {
                        typeof(IQueryable<>).MakeGenericType(type),
                        typeof(Expression<>).MakeGenericType(typeof(Func<,>).MakeGenericType(type, type))
                    };
                }));
            AverageWithSelectorMethods[type] = averageWithSelectorMethod;

            var sumWithoutSelectorMethod = GetMethod(
                nameof(Queryable.Sum), 0,
                new Func<Type[]>(() =>
                {
                    return new[] {typeof(IQueryable<>).MakeGenericType(type)};
                }));
            SumWithoutSelectorMethods[type] = sumWithoutSelectorMethod;

            var sumWithSelectorMethod = GetMethod(
                nameof(Queryable.Sum), 1,
                new Func<Type[]>(() =>
                {
                    return new[]
                    {
                        typeof(IQueryable<>).MakeGenericType(type),
                        typeof(Expression<>).MakeGenericType(typeof(Func<,>).MakeGenericType(type, type))
                    };
                }));
            SumWithSelectorMethods[type] = sumWithSelectorMethod;
        }

public void ProcessConfigureServicesContext(BlockStartAnalysisContext context)
{
    var methodSymbol = (IMethodSymbol)context.Method;
    var optionsBuilder = ImmutableArray.CreateBuilder<OptionsItem>();
    context.OperationBlock.Operations.ToList().ForEach(operation =>
    {
        if (operation is ISimpleAssignmentOperation assignOp && assignOp.Value.ConstantValue.HasValue &&
            operation.Target?.Target as IPropertyReferenceOperation property != null &&
            property.Property?.ContainingType?.Name != null &&
            property.Property.ContainingType.Name.EndsWith("Options", StringComparison.Ordinal))
        {
            optionsBuilder.Add(new OptionsItem(property.Property, assignOp.Value.ConstantValue.Value));
        }
    });

    if (optionsBuilder.Count > 0)
    {
        _context.ReportAnalysis(new OptionsAnalysis(methodSymbol, optionsBuilder.ToImmutable()));
    }
}

