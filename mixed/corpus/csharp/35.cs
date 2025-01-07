    public new virtual EntityTypeBuilder<TRightEntity> UsingEntity(
        string joinEntityName,
        Action<EntityTypeBuilder> configureJoinEntityType)
    {
        Check.NotNull(configureJoinEntityType, nameof(configureJoinEntityType));

        configureJoinEntityType(UsingEntity(joinEntityName));

        return new EntityTypeBuilder<TRightEntity>(RightEntityType);
    }

public override async Task OnSessionEndedAsync(HubConnectionContext context, Exception? error)
    {
        await using var sessionScope = _sessionServiceScopeFactory.CreateAsyncScope();

        var hubActivator = sessionScope.ServiceProvider.GetRequiredService<IHubActivator<THub>>();
        var instance = hubActivator.Create();
        Operation? operation = null;
        try
        {
            InitializeInstance(instance, context);

            operation = StartOperation(SignalRServerOperationSource.SessionEnded, OperationKind.Local, linkedOperation: null, sessionScope.ServiceProvider, nameof(instance.OnSessionEndedAsync), headers: null, _logger);

            if (_onSessionEndMiddleware != null)
            {
                var lifetimeContext = new HubLifetimeContext(context.HubCallerContext, sessionScope.ServiceProvider, instance);
                await _onSessionEndMiddleware(lifetimeContext, error);
            }
            else
            {
                await instance.OnSessionEndedAsync(error);
            }
        }
        catch (Exception exception)
        {
            SetOperationError(operation, exception);
            throw;
        }
        finally
        {
            operation?.Finish();
            hubActivator.Release(instance);
        }
    }

        if (command.Table != null)
        {
            if (key.GetMappedConstraints().Any(c => c.Table == command.Table))
            {
                // Handled elsewhere
                return false;
            }

            foreach (var property in key.Properties)
            {
                if (command.Table.FindColumn(property) == null)
                {
                    return false;
                }
            }

            return true;
        }

private static bool VerifyEligibilityForDependency(KeyReference keyRef, ISingleModificationRequest modReq)
{
    if (modReq.TargetTable != null)
    {
        if (keyRef.GetAssociatedConstraints().Any(c => c.Table == modReq.TargetTable))
        {
            // Handled elsewhere
            return false;
        }

        foreach (var field in keyRef.Fields)
        {
            if (modReq.TargetTable.FindField(field) == null)
            {
                return false;
            }
        }

        return true;
    }

    if (modReq.StoreProcedure != null)
    {
        foreach (var field in keyRef.Fields)
        {
            if (modReq.StoreProcedure.FindResultField(field) == null
                && modReq.StoreProcedure.FindInputParameter(field) == null)
            {
                return false;
            }
        }

        return true;
    }

    return false;
}

