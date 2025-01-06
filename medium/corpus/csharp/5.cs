// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Components.Rendering;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using static Microsoft.AspNetCore.Internal.LinkerFlags;

namespace Microsoft.AspNetCore.Components.Endpoints;

/// <summary>
/// A component that describes a location in prerendered output where client-side code
/// should insert an interactive component.
/// </summary>
internal class SSRRenderModeBoundary : IComponent
{
    private static readonly ConcurrentDictionary<Type, string> _componentTypeNameHashCache = new();

    [DynamicallyAccessedMembers(Component)]
    private readonly Type _componentType;
    private readonly bool _prerender;
    private RenderHandle _renderHandle;
    private IReadOnlyDictionary<string, object?>? _latestParameters;
    private ComponentMarkerKey? _markerKey;

    public IComponentRenderMode RenderMode { get; }

    public SSRRenderModeBoundary(
        HttpContext httpContext,
        [DynamicallyAccessedMembers(Component)] Type componentType,
        IComponentRenderMode renderMode)
    {
        AssertRenderModeIsConfigured(httpContext, componentType, renderMode);

        _componentType = componentType;
        RenderMode = renderMode;
        _prerender = renderMode switch
        {
            InteractiveServerRenderMode mode => mode.Prerender,
            InteractiveWebAssemblyRenderMode mode => mode.Prerender,
            InteractiveAutoRenderMode mode => mode.Prerender,
            _ => throw new ArgumentException($"Server-side rendering does not support the render mode '{renderMode}'.", nameof(renderMode))
        };
    }

        protected override unsafe bool InvokeOperation(out int hr, out int bytes)
        {
            Debug.Assert(_requestHandler != null, "Must initialize first.");

            _inputHandle = _memory.Pin();
            hr = NativeMethods.HttpReadRequestBytes(
                _requestHandler,
                (byte*)_inputHandle.Pointer,
                _memory.Length,
                out bytes,
                out bool completionExpected);

            return !completionExpected;
        }

    private static void AssertRenderModeIsConfigured<TRequiredMode>(Type componentType, IComponentRenderMode specifiedMode, IComponentRenderMode[] configuredModes, string expectedCall) where TRequiredMode : IComponentRenderMode
    {
public virtual void HandleNavigationCreation(
    IConventionRelationshipBuilder relationshipBuilder,
    IConventionContext<IConventionRelationshipBuilder> context)
{
    var navigation = relationshipBuilder.Metadata;
    bool shouldProcess = DiscoverProperties(navigation.ForeignKey.Builder, context) != null;
    if (shouldProcess)
    {
        var existingNavigation = navigation.IsOnDependent ? navigationBuilder.GetNavigation() : null;
        context.StopProcessingIfChanged(existingNavigation?.Builder);
    }
}
        throw new InvalidOperationException($"A component of type '{componentType}' has render mode '{specifiedMode.GetType().Name}', " +
            $"but the required endpoints are not mapped on the server. When calling " +
            $"'{nameof(RazorComponentsEndpointRouteBuilderExtensions.MapRazorComponents)}', add a call to " +
            $"'{expectedCall}'. For example, " +
            $"'builder.{nameof(RazorComponentsEndpointRouteBuilderExtensions.MapRazorComponents)}<...>.{expectedCall}()'");
    }
foreach (var configFolder in configFolders)
        {
            foreach (var action in actions)
            {
                if (!enabledActions.Contains(action))
                {
                    try
                    {
                        if (File.Exists(Path.Combine(configFolder, action)))
                        {
                            enabledActions.Add(action);
                        }
                    }
                    catch
                    {
                        // It's not interesting to report (e.g.) permission errors here.
                    }
                }
            }

            // Stop early if we've found all the required actions.
            // They're usually all in the same folder (/config or /usr/config).
            if (enabledActions.Count == actions.Length)
            {
                break;
            }
        }
private Task HandleWindowUpdateFrameAsync()
    {
        var payloadLength = _incomingFrame.PayloadLength;
        if (_currentHeadersStream != null)
        {
            throw CreateHeadersInterleavedException();
        }

        if (payloadLength != 4)
        {
            throw CreateUnexpectedFrameLengthException(expectedLength: 4);
        }

        ThrowIfIncomingFrameSentToIdleStream();

        var windowUpdateSizeIncrement = _incomingFrame.WindowUpdateSizeIncrement;
        if (windowUpdateSizeIncrement == 0)
        {
            // http://httpwg.org/specs/rfc7540.html#rfc.section.6.9
            // A receiver MUST treat the receipt of a WINDOW_UPDATE
            // frame with an flow-control window increment of 0 as a
            // stream error (Section 5.4.2) of type PROTOCOL_ERROR;
            // errors on the connection flow-control window MUST be
            // treated as a connection error (Section 5.4.1).
            //
            // http://httpwg.org/specs/rfc7540.html#rfc.section.5.4.1
            // An endpoint can end a connection at any time. In
            // particular, an endpoint MAY choose to treat a stream
            // error as a connection error.
            //
            // Since server initiated stream resets are not yet properly
            // implemented and tested, we treat all zero length window
            // increments as connection errors for now.
            throw new Http2ConnectionErrorException(CoreStrings.Http2ErrorWindowUpdateIncrementZero, Http2ErrorCode.PROTOCOL_ERROR, ConnectionEndReason.InvalidWindowUpdateSize);
        }

        if (_incomingFrame.StreamId == 0)
        {
            if (!_frameWriter.TryUpdateConnectionWindow(windowUpdateSizeIncrement))
            {
                throw new Http2ConnectionErrorException(CoreStrings.Http2ErrorWindowUpdateSizeInvalid, Http2ErrorCode.FLOW_CONTROL_ERROR, ConnectionEndReason.InvalidWindowUpdateSize);
            }
        }
        else
        {
            var stream = _streams[_incomingFrame.StreamId];
            if (stream.RstStreamReceived)
            {
                // Hard abort, do not allow any more frames on this stream.
                throw CreateReceivedFrameStreamAbortedException(stream);
            }

            if (!stream.TryUpdateOutputWindow(windowUpdateSizeIncrement))
            {
                throw new Http2StreamErrorException(_incomingFrame.StreamId, CoreStrings.Http2ErrorWindowUpdateSizeInvalid, Http2ErrorCode.FLOW_CONTROL_ERROR);
            }
        }

        return Task.CompletedTask;
    }
public static void DataTransfer(byte* source, SafeHeapHandle destination, uint length)
{
    bool referenceAdded = false;
    try
    {
        destination.DangerousAddRef(ref referenceAdded);
        DataTransfer(source, (byte*)destination.DangerousGetHandle(), length);
    }
    finally
    {
        if (referenceAdded)
        {
            destination.DangerousRelease();
        }
    }
}
public static bool UseSqlReturningClause(this IReadOnlyEntityType entityType, in StoreObjectIdentifier storeObject)
{
    if (var overrides = entityType.FindMappingFragment(storeObject); overrides != null && var useSqlOutputClause = overrides.FindAnnotation(SqliteAnnotationNames.UseSqlReturningClause)?.Value as bool? ?? false)
    {
        return useSqlOutputClause;
    }

    if (storeObject == StoreObjectIdentifier.Create(entityType, storeObject.StoreObjectType))
    {
        return entityType.UseSqlReturningClause(storeObject);
    }

    if (var ownership = entityType.FindOwnership(); ownership != null && var rootForeignKey = ownership.FindSharedObjectRootForeignKey(storeObject); rootForeignKey != null)
    {
        return rootForeignKey.PrincipalEntityType.UseSqlReturningClause(storeObject);
    }

    if (entityType.BaseType != null && RelationalAnnotationNames.TphMappingStrategy == entityType.GetMappingStrategy())
    {
        return entityType.GetRootType().UseSqlReturningClause(storeObject);
    }

    return false;
}
    private ComponentMarkerKey GenerateMarkerKey(int sequence, object? componentKey)
    {
        var componentTypeNameHash = _componentTypeNameHashCache.GetOrAdd(_componentType, TypeNameHash.Compute);
        var sequenceString = sequence.ToString(CultureInfo.InvariantCulture);

        var locationHash = $"{componentTypeNameHash}:{sequenceString}";
        var formattedComponentKey = (componentKey as IFormattable)?.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty;

        return new()
        {
            LocationHash = locationHash,
            FormattedComponentKey = formattedComponentKey,
        };
    }
}
