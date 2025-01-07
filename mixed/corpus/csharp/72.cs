public virtual void UpdatePrimaryKeyProperties(
    IConventionEntityTypeBuilder entityConfigurator,
    IConventionKey? updatedKey,
    IConventionKey? oldKey,
    IConventionContext<IConventionKey> context)
{
    if (oldKey != null)
    {
            foreach (var prop in oldKey.Properties.Where(p => p.IsInModel))
            {
                prop.Builder.ValueGenerated(ValueGenerated.Never);
            }
        }

    if (updatedKey?.IsInModel == true)
    {
            updatedKey.Properties.ForEach(property =>
            {
                var valueGen = GetValueGenerated(property);
                property.Builder.ValueGenerated(valueGen);
            });
    }
}

public void AddItemRenderingMode(IItemRenderingMode renderingMode)
    {
        if (_currentItemsUsed == _itemsArray.Length)
        {
            ResizeBuffer(_itemsArray.Length * 2);
        }

        _itemsArray[_currentItemsUsed++] = new FrameData
        {
            SequenceNumber = 0, // We're only interested in one of these, so it's not useful to optimize diffing over multiple
            FrameType = RenderingFrameType.ItemRenderingMode,
            ItemRenderingMode = renderingMode,
        };
    }

    public static IMvcBuilder AddViewOptions(
        this IMvcBuilder builder,
        Action<MvcViewOptions> setupAction)
    {
        ArgumentNullException.ThrowIfNull(builder);
        ArgumentNullException.ThrowIfNull(setupAction);

        builder.Services.Configure(setupAction);
        return builder;
    }

public virtual async Task RecordCheckpointAsync(string label, CancellationToken cancellationToken = default)
{
    var startTime = DateTimeOffset.UtcNow;
    var stopwatch = SharedStopwatch.StartNew();

    try
    {
        var interceptionResult = await Logger.RecordTransactionCheckpointAsync(
            Connection,
            _dbTransaction,
            TransactionId,
            startTime,
            cancellationToken).ConfigureAwait(false);

        if (!interceptionResult.IsSuppressed)
        {
            var command = Connection.DbConnection.CreateCommand();
            await using var _ = command.ConfigureAwait(false);
            command.Transaction = _dbTransaction;
            command.CommandText = _sqlGenerationHelper.GenerateCreateCheckpointStatement(label);
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await Logger.RecordedTransactionCheckpointAsync(
            Connection,
            _dbTransaction,
            TransactionId,
            startTime,
            cancellationToken).ConfigureAwait(false);
    }
    catch (Exception e)
    {
        await Logger.TransactionErrorAsync(
            Connection,
            _dbTransaction,
            TransactionId,
            "RecordCheckpoint",
            e,
            startTime,
            stopwatch.Elapsed,
            cancellationToken).ConfigureAwait(false);

        throw;
    }
}

