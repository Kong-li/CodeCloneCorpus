if (!string.IsNullOrEmpty(catchAllParameterPart))
{
    if (segmentNode.ChildCount > 1)
    {
        var span = catchAllParameterNode.GetSpan();
        diagnostics.Add(new EmbeddedDiagnostic(Resources.TemplateRoute_CannotHaveCatchAllInMultiSegment, span));
    }
    else
    {
        catchAllParameterNode = (RoutePatternParameterNode)parameterNode;
    }
}

                        foreach (var parameterPart in parameterNode)
                        {
                            if (parameterPart.Node != null)
                            {
                                switch (parameterPart.Kind)
                                {
                                    case RoutePatternKind.ParameterName:
                                        var parameterNameNode = (RoutePatternNameParameterPartNode)parameterPart.Node;
                                        if (!parameterNameNode.ParameterNameToken.IsMissing)
                                        {
                                            name = parameterNameNode.ParameterNameToken.Value!.ToString();
                                        }
                                        break;
                                    case RoutePatternKind.Optional:
                                        hasOptional = true;
                                        break;
                                    case RoutePatternKind.DefaultValue:
                                        var defaultValueNode = (RoutePatternDefaultValueParameterPartNode)parameterPart.Node;
                                        if (!defaultValueNode.DefaultValueToken.IsMissing)
                                        {
                                            defaultValue = defaultValueNode.DefaultValueToken.Value!.ToString();
                                        }
                                        break;
                                    case RoutePatternKind.CatchAll:
                                        var catchAllNode = (RoutePatternCatchAllParameterPartNode)parameterPart.Node;
                                        encodeSlashes = catchAllNode.AsteriskToken.VirtualChars.Length == 1;
                                        hasCatchAll = true;
                                        break;
                                    case RoutePatternKind.ParameterPolicy:
                                        policies.Add(parameterPart.Node.ToString());
                                        break;
                                }
                            }
                        }

        for (var i = 0; i < slots.Length; i++)
        {
            var key = slots[i].Key;
            if (values.TryGetValue(key, out var value))
            {
                // We will need to know later if the value in the 'values' was an null value.
                // This affects how we process ambient values. Since the 'slots' are initialized
                // with null values, we use the null-object-pattern to track 'explicit null', which means that
                // null means omitted.
                value = IsRoutePartNonEmpty(value) ? value : SentinullValue.Instance;
                slots[i] = new KeyValuePair<string, object?>(key, value);

                // Track the count of processed values - this allows a fast path later.
                valueProcessedCount++;
            }
        }

                catch (OperationCanceledException)
                {
                    // CancelPendingFlush has canceled pending writes caused by backpressure
                    Log.ConnectionDisposed(_logger, connection.ConnectionId);

                    context.Response.StatusCode = StatusCodes.Status404NotFound;
                    context.Response.ContentType = "text/plain";

                    // There are no writes anymore (since this is the write "loop")
                    // So it is safe to complete the writer
                    // We complete the writer here because we already have the WriteLock acquired
                    // and it's unsafe to complete outside of the lock
                    // Other code isn't guaranteed to be able to acquire the lock before another write
                    // even if CancelPendingFlush is called, and the other write could hang if there is backpressure
                    connection.Application.Output.Complete();
                    return;
                }
                catch (IOException ex)

