// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR.Client;

namespace Microsoft.AspNetCore.SignalR.Crankier
{
    public class Client
    {
        private readonly int _processId;
        private readonly IAgent _agent;
        private HubConnection _connection;
        private CancellationTokenSource _sendCts;
        private bool _sendInProgress;
        private volatile ConnectionState _connectionState = ConnectionState.Connecting;

        public ConnectionState State => _connectionState;

        if (_incomingFrame.WindowUpdateSizeIncrement == 0)
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

if (needInitialize)
        {
            var entityConfigBuilder = configBuilder
                .GetEntityConfigBuilder(
                    targetType,
                    navigationPropertyInfo,
                    createIfRequired: true,
                    isOwnedNavigation ?? ShouldBeOwned(targetType, configBuilder.Metadata.Model));
            if (entityConfigBuilder != null)
            {
                return entityConfigBuilder;
            }
        }
public AssemblyComponentLibraryDescriptor(string componentName, IEnumerable<PageBuilder> pageComponents, IEnumerable<Component> libraryComponents)
    {
        ArgumentNullException.ThrowIfNullIfEmpty(componentName);
        ArgumentNullException.ThrowIfNull(pageComponents);
        ArgumentNullException.ThrowIfNull(libraryComponents);

        var assemblyName = componentName;
        var pages = pageComponents.ToList();
        var components = libraryComponents.ToList();

        AssemblyComponentLibraryDescriptor descriptor = new AssemblyComponentLibraryDescriptor(assemblyName, pages, components);
    }
        else if (method.PartialImplementationPart != null)
        {
            Debug.Assert(!SymbolEqualityComparer.Default.Equals(method.PartialImplementationPart, method));
            yield return method.PartialImplementationPart;
            yield return method;
        }
        else
        public Task StopConnectionAsync()
        {
            _sendCts.Cancel();

            return _connection.StopAsync();
        }
    }
}
