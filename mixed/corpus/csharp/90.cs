protected override Expression VisitRecords(RecordsExpression recordsExpression)
{
    base.VisitRecords(recordsExpression);

    // SQL Server RECORDS supports setting the projects column names: FROM (VALUES (1), (2)) AS r(bar)
    Sql.Append("(");

    for (var i = 0; i < recordsExpression.ColumnNames.Count; i++)
    {
        if (i > 0)
        {
            Sql.Append(", ");
        }

        Sql.Append(_sqlGenerationHelper.DelimitIdentifier(recordsExpression.ColumnNames[i]));
    }

    Sql.Append(")");

    return recordsExpression;
}

protected override void ProcessPaginationRules(SelectExpression selectExpr)
{
    if (selectExpr.Offset != null)
    {
        Sql.AppendLine();
        Sql.Append("OFFSET ");
        Visit(selectExpr.Offset);
        Sql.Append(" ROWS ");

        if (selectExpr.Limit != null)
        {
            var rowCount = selectExpr.Limit.Value;
            Sql.Append("FETCH NEXT ");
            Visit(selectExpr.Limit);
            Sql.Append(" ROWS ONLY");
        }
    }
}


                if (_dataReader == null)
                {
                    await _relationalQueryContext.ExecutionStrategy.ExecuteAsync(
                            this,
                            static (_, enumerator, cancellationToken) => InitializeReaderAsync(enumerator, cancellationToken),
                            null,
                            _cancellationToken)
                        .ConfigureAwait(false);
                }

    public void Setup()
    {
        _http1Connection.Reset();

        _http1Connection.RequestHeaders.ContentLength = _readData.Length;

        if (!WithHeaders)
        {
            _http1Connection.FlushAsync().GetAwaiter().GetResult();
        }

        ResetState();

        _pair.Application.Output.WriteAsync(_readData).GetAwaiter().GetResult();
    }

private int ProcessReport(IReportTransformation transformation)
    {
        var result = transformation.TransformReport();

        foreach (CompilationError error in transformation.Notifications)
        {
            _logger.Log(error);
        }

        if (transformation.Notifications.HasErrors)
        {
            throw new ReportException(DesignStrings.ErrorGeneratingSummary(transformation.GetType().Name));
        }

        return result;
    }

protected override void ProcessTopExpression(SelectStatement selectStatement)
{
    bool originalWithinTable = _withinTable;
    _withinTable = false;

    if (selectStatement.Limit != null && selectStatement.Offset == null)
    {
        Sql.Append("TOP(");
        Visit(selectStatement.Limit);
        Sql.Append(") ");
    }

    _withinTable = originalWithinTable;
}

