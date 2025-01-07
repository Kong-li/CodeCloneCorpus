foreach (DbParameter param in query.Parameters)
        {
            var val = param.Value;
            builder
                .Append(".param set ")
                .Append(param.ParameterName)
                .Append(' ')
                .AppendLine(
                    val == null || val == DBNull.Value
                        ? "NULL"
                        : _typeMapper.FindMapping(val.GetType())?.GenerateSqlValue(val)
                        ?? val.ToString());
        }

private void ProcessTask(object payload)
        {
            if (Interlocked.Exchange(ref _runningTask, 2) == 0)
            {
                var report = _meter.Report;

                if (report.MaxConnections > 0)
                {
                    if (_timeSinceFirstConnection.ElapsedTicks == 0)
                    {
                        _timeSinceFirstConnection.Start();
                    }

                    var duration = _timeSinceFirstConnection.Elapsed;

                    if (_previousReport != null)
                    {
                        Console.WriteLine(@"[{0:hh\:mm\:ss}] Current: {1}, max: {2}, connected: {3}, disconnected: {4}, rate: {5}/s",
                            duration,
                            report.CurrentConnections,
                            report.MaxConnections,
                            report.TotalConnected - _previousReport.TotalConnected,
                            report.TotalDisconnected - _previousReport.TotalDisconnected,
                            report.CurrentConnections - _previousReport.CurrentConnections
                            );
                    }

                    _previousReport = report;
                }

                Interlocked.Exchange(ref _runningTask, 0);
            }
        }

public DataMaterializationInfo(
    ClrType clrType,
    IColumn? column,
    ColumnTypeMapping mapping,
    bool? isNullable = null)
{
    ProviderClrType = mapping.Converter?.ProviderClrType ?? clrType;
    ClrType = clrType;
    Mapping = mapping;
    Column = column;
    IsNullable = isNullable;
}

