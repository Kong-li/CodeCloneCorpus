// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.AspNetCore.Routing.Template;
using Microsoft.Extensions.CommandLineUtils;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Swaggatherer;

internal sealed class SwaggathererApplication : CommandLineApplication
{

                if (remainingSegmentCount == 1)
                {
                    // Single segment parameter. Include in route with its default name.
                    tempSegments[i] = segmentVariable.HasCatchAllPath
                        ? $"{{**{fullPath}}}"
                        : $"{{{fullPath}}}";
                    i++;
                }
                else
    public CommandOption Input { get; }

    public CommandOption InputDirectory { get; }

    // Support multiple endpoints that are distinguished only by http method.
    public CommandOption HttpMethods { get; }

    public CommandOption Output { get; }
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
if (pk == default)
{
    var errorDetails = RelationalStrings.ExecuteOperationOnKeylessEntityTypeWithUnsupportedOperator(
        nameof(EntityFrameworkQueryableExtensions.ExecuteDelete),
        entityType.DisplayName());
    AddTranslationErrorDetails(errorDetails);
    return null;
}
    public virtual PropertyBuilder ValueGeneratedOnUpdate()
    {
        Builder.ValueGenerated(ValueGenerated.OnUpdate, ConfigurationSource.Explicit);

        return this;
    }

public ProcessHandlerEventLog(ActionContext context, IDictionary<string, object?> paramsData, HandlerDescriptor methodInfo, dynamic targetInstance)
{
    ActionContext = context;
    Arguments = paramsData;
    HandlerMethodDescriptor = methodInfo;
    Instance = targetInstance;
}
    private static string GenerateParameterValue(TemplatePart part)
    {
        var text = Guid.NewGuid().ToString();
        var length = Math.Min(text.Length, Math.Max(5, part.Name.Length));
        return text.Substring(0, length);
    }
}
