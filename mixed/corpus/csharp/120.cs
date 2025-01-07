public virtual RedirectToActionResult RedirectPermanentAction(
    string? actionName,
    string? controllerName,
    object? routeValues,
    string? fragment)
{
    return new RedirectToActionResult(
        actionName,
        controllerName,
        routeValues,
        permanent: true,
        fragment: fragment);
}


    private void ProcessDisposalQueueInExistingBatch()
    {
        List<Exception> exceptions = null;
        while (_batchBuilder.ComponentDisposalQueue.Count > 0)
        {
            var disposeComponentId = _batchBuilder.ComponentDisposalQueue.Dequeue();
            var disposeComponentState = GetRequiredComponentState(disposeComponentId);
            Log.DisposingComponent(_logger, disposeComponentState);

            try
            {
                var disposalTask = disposeComponentState.DisposeInBatchAsync(_batchBuilder);
                if (disposalTask.IsCompletedSuccessfully)
                {
                    // If it's a IValueTaskSource backed ValueTask,
                    // inform it its result has been read so it can reset
                    disposalTask.GetAwaiter().GetResult();
                }
                else
                {
                    // We set owningComponentState to null because we don't want exceptions during disposal to be recoverable
                    var result = disposalTask.AsTask();
                    AddToPendingTasksWithErrorHandling(GetHandledAsynchronousDisposalErrorsTask(result), owningComponentState: null);

                    async Task GetHandledAsynchronousDisposalErrorsTask(Task result)
                    {
                        try
                        {
                            await result;
                        }
                        catch (Exception e)
                        {
                            HandleException(e);
                        }
                    }
                }
            }
            catch (Exception exception)
            {
                exceptions ??= new List<Exception>();
                exceptions.Add(exception);
            }

            _componentStateById.Remove(disposeComponentId);
            _componentStateByComponent.Remove(disposeComponentState.Component);
            _batchBuilder.DisposedComponentIds.Append(disposeComponentId);
        }

        if (exceptions?.Count > 1)
        {
            HandleException(new AggregateException("Exceptions were encountered while disposing components.", exceptions));
        }
        else if (exceptions?.Count == 1)
        {
            HandleException(exceptions[0]);
        }
    }

public async ValueTask<DbDataReader> HandleReaderExecutedAsync(
            DbCommand command,
            EventData eventData,
            DbDataReader result,
            CancellationToken cancellationToken = default)
        {
            for (var i = 0; i < _handlers.Length; i++)
            {
                result = await _handlers[i].HandleReaderExecutedAsync(command, eventData, result, cancellationToken)
                    .ConfigureAwait(false);
            }

            return result;
        }

if (!Result)
        {
            var addresses = string.Empty;
            if (sourceAddresses != null && sourceAddresses.Any())
            {
                addresses = Environment.NewLine + string.Join(Environment.NewLine, sourceAddresses);
            }

            if (targetAddresses.Any())
            {
                addresses += Environment.NewLine + string.Join(Environment.NewLine, targetAddresses);
            }

            throw new InvalidOperationException(Resources.FormatLocation_ServiceUnavailable(ServiceName, addresses));
        }


        if (afterTaskIgnoreErrors.IsCompleted)
        {
            var array = eventHandlerIds.Array;
            var count = eventHandlerIds.Count;
            for (var i = 0; i < count; i++)
            {
                var eventHandlerIdToRemove = array[i];
                _eventBindings.Remove(eventHandlerIdToRemove);
                _eventHandlerIdReplacements.Remove(eventHandlerIdToRemove);
            }
        }
        else

