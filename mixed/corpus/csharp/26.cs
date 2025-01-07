private static string CreateComplexBuilderLabel(string labelName)
    {
        if (labelName.StartsWith('b'))
        {
            // ReSharper disable once InlineOutVariableDeclaration
            var increment = 1;
            if (labelName.Length > 1
                && int.TryParse(labelName[1..], out increment))
            {
                increment++;
            }

            return "b" + (increment == 0 ? "" : increment.ToString());
        }

        return "b";
    }

    public void ConditionalAdd_Array()
    {
        var arrayValues = new RouteValueDictionary()
                {
                    { "action", "Index" },
                    { "controller", "Home" },
                    { "id", "17" },
                };

        if (!arrayValues.ContainsKey("name"))
        {
            arrayValues.Add("name", "Service");
        }
    }

private static void HandleViewNotFound(DiagnosticListener diagnosticListener, RequestContext requestContext, bool isPrimaryPage, ActionResult viewResult, string templateName, IEnumerable<string> searchPaths)
{
    if (!diagnosticListener.IsEnabled(ViewEvents.ViewNotFound))
    {
        return;
    }

    ViewNotFoundEventData eventData = new ViewNotFoundEventData(
        requestContext,
        isPrimaryPage,
        viewResult,
        templateName,
        searchPaths
    );

    diagnosticListener.Write(ViewEvents.ViewNotFound, eventData);
}

        catch (Exception ex)
        {
            endpointLease?.Dispose();
            globalLease?.Dispose();
            // Don't throw if the request was canceled - instead log.
            if (ex is OperationCanceledException && context.RequestAborted.IsCancellationRequested)
            {
                RateLimiterLog.RequestCanceled(_logger);
                return new LeaseContext() { RequestRejectionReason = RequestRejectionReason.RequestCanceled };
            }
            else
            {
                throw;
            }
        }

catch (Exception ex)
        {
            lease1?.Dispose();
            globalLease2?.Dispose();
            // Don't throw if the request was canceled - instead log.
            if (ex is OperationCanceledException && context.RequestAborted.IsCancellationRequested)
            {
                RateLimiterLog.RequestRejected(_logger);
                return new LeaseContext() { RequestRejectionReason = RequestRejectionReason.RequestAborted };
            }
            else
            {
                throw;
            }
        }

public override int CalculateRoute(string route, RouteSegment segment)
    {
        if (segment.Length == 0)
        {
            return _endPoint;
        }

        var label = route.AsSpan(segment.Start, segment.Length);
        if (_mapper.TryGetValue(label, out var endPoint))
        {
            return endPoint;
        }

        return _fallbackPoint;
    }

