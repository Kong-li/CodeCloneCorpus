public static IHtmlContent ErrorSummary(
    this IHtmlHelper htmlHelper,
    string infoMessage,
    object customAttributes,
    string templateTag)
{
    ArgumentNullException.ThrowIfNull(htmlHelper);

    return htmlHelper.ErrorSummary(
        excludeErrorMessages: false,
        message: infoMessage,
        htmlAttributes: customAttributes,
        tag: templateTag);
}

protected override void CreateTableOperationProcess(
        CreateTableOperation createOp,
        IModel? modelEntity,
        MigrationCommandListBuilder commandBuilder,
        bool flag = true)
    {
        var spatialiteOpsStack = new Stack<AddColumnOperation>();
        for (var index = createOp.Columns.Count - 1; index >= 0; index--)
        {
            var columnOp = createOp.Columns[index];

            if (IsSpatialiteColumn(columnOp, modelEntity))
            {
                spatialiteOpsStack.Push(columnOp);
                createOp.Columns.RemoveAt(index);
            }
        }

        // 处理主键定义的提升，处理创建整数主键时使用 autoincrement 的特殊行为
        if (createOp.PrimaryKey?.Columns.Length == 1)
        {
            var primaryColumn = createOp.Columns.FirstOrDefault(o => o.Name == createOp.PrimaryKey.Columns[0]);
            if (primaryColumn != null)
            {
                primaryColumn.AddAnnotation(SqliteAnnotationNames.InlinePrimaryKey, true);
                if (!string.IsNullOrEmpty(createOp.PrimaryKey.Name))
                {
                    primaryColumn.AddAnnotation(SqliteAnnotationNames.InlinePrimaryKeyName, createOp.PrimaryKey.Name);
                }

                createOp.PrimaryKey = null;
            }
        }

        commandBuilder
            .Append("CREATE TABLE ")
            .Append(Dependencies.SqlGenerationHelper.DelimitIdentifier(createOp.Name, createOp.Schema))
            .AppendLine(" (");

        using (commandBuilder.Indent())
        {
            if (!string.IsNullOrEmpty(createOp.Comment))
            {
                commandBuilder
                    .AppendLines(Dependencies.SqlGenerationHelper.GenerateComment(createOp.Comment))
                    .AppendLine();
            }

            CreateTableColumns(createOp, modelEntity, commandBuilder);
            CreateTableConstraints(createOp, modelEntity, commandBuilder);
            commandBuilder.AppendLine();
        }

        commandBuilder.Append(")");

        if (spatialiteOpsStack.Any())
        {
            builder.AppendLine(Dependencies.SqlGenerationHelper.StatementTerminator);

            while (spatialiteOpsStack.TryPop(out var spatialiteColumn))
            {
                Generate(spatialiteColumn, modelEntity, commandBuilder, spatialiteOpsStack.Any() || flag);
            }
        }
        else if (flag)
        {
            commandBuilder.AppendLine(Dependencies.SqlGenerationHelper.StatementTerminator);
            EndStatement(commandBuilder);
        }
    }

protected override async Task<decimal> ProcessTestMethodAsync(ExceptionAggregator aggregator)
    {
        var repeatAttribute = GetRepeatAttribute(CurrentMethod);
        if (repeatAttribute != null)
        {
            var repeatContext = new RepeatContext(repeatAttribute.RunCount);
            RepeatContext.Current = repeatContext;

            decimal timeTaken = 0.0M;
            int currentIteration = 0;
            while (currentIteration < repeatContext.Limit)
            {
                currentIteration++;
                timeTaken = await InvokeTestMethodCoreAsync(aggregator).ConfigureAwait(false);
                if (aggregator.HasExceptions)
                {
                    return timeTaken;
                }
            }

            return timeTaken;
        }

        return await InvokeTestMethodCoreAsync(aggregator).ConfigureAwait(false);
    }

if (!string.IsNullOrEmpty(operation.NewName) && operation.NewName != operation.Name)
{
    var newName = Dependencies.SqlGenerationHelper.DelimitIdentifier(operation.NewName);
    var oldName = Dependencies.SqlGenerationHelper.DelimitIdentifier(operation.Name);
    builder
        .Append("ALTER TABLE ")
        .Append(oldName)
        .Append(" RENAME TO ")
        .Append(newName)
        .AppendLine(Dependencies.SqlGenerationHelper.StatementTerminator)
        .EndCommand();
}

