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


    private async Task<bool> TryReadPrefaceAsync()
    {
        // HTTP/1.x and HTTP/2 support connections without TLS. That means ALPN hasn't been used to ensure both sides are
        // using the same protocol. A common problem is someone using HTTP/1.x to talk to a HTTP/2 only endpoint.
        //
        // HTTP/2 starts a connection with a preface. This method reads and validates it. If the connection doesn't start
        // with the preface, and it isn't using TLS, then we attempt to detect what the client is trying to do and send
        // back a friendly error message.
        //
        // Outcomes from this method:
        // 1. Successfully read HTTP/2 preface. Connection continues to be established.
        // 2. Detect HTTP/1.x request. Send back HTTP/1.x 400 response.
        // 3. Unknown content. Report HTTP/2 PROTOCOL_ERROR to client.
        // 4. Timeout while waiting for content.
        //
        // Future improvement: Detect TLS frame. Useful for people starting TLS connection with a non-TLS endpoint.
        var state = ReadPrefaceState.All;

        // With TLS, ALPN should have already errored if the wrong HTTP version is used.
        // Only perform additional validation if endpoint doesn't use TLS.
        if (ConnectionFeatures.Get<ITlsHandshakeFeature>() != null)
        {
            state ^= ReadPrefaceState.Http1x;
        }

        while (_isClosed == 0)
        {
            var result = await Input.ReadAsync();
            var readableBuffer = result.Buffer;
            var consumed = readableBuffer.Start;
            var examined = readableBuffer.End;

            try
            {
                if (!readableBuffer.IsEmpty)
                {
                    if (state.HasFlag(ReadPrefaceState.Preface))
                    {
                        if (readableBuffer.Length >= ClientPreface.Length)
                        {
                            if (IsPreface(readableBuffer, out consumed, out examined))
                            {
                                return true;
                            }
                            else
                            {
                                state ^= ReadPrefaceState.Preface;
                            }
                        }
                    }

                    if (state.HasFlag(ReadPrefaceState.Http1x))
                    {
                        if (ParseHttp1x(readableBuffer, out var detectedVersion))
                        {
                            if (detectedVersion == HttpVersion.Http10 || detectedVersion == HttpVersion.Http11)
                            {
                                Log.PossibleInvalidHttpVersionDetected(ConnectionId, HttpVersion.Http2, detectedVersion);

                                var responseBytes = InvalidHttp1xErrorResponseBytes ??= Encoding.ASCII.GetBytes(
                                    "HTTP/1.1 400 Bad Request\r\n" +
                                    "Connection: close\r\n" +
                                    "Content-Type: text/plain\r\n" +
                                    "Content-Length: 56\r\n" +
                                    "\r\n" +
                                    "An HTTP/1.x request was sent to an HTTP/2 only endpoint.");

                                await _context.Transport.Output.WriteAsync(responseBytes);

                                // Close connection here so a GOAWAY frame isn't written.
                                if (TryClose())
                                {
                                    SetConnectionErrorCode(ConnectionEndReason.InvalidHttpVersion, Http2ErrorCode.PROTOCOL_ERROR);
                                }

                                return false;
                            }
                            else
                            {
                                state ^= ReadPrefaceState.Http1x;
                            }
                        }
                    }

                    // Tested all states. Return HTTP/2 protocol error.
                    if (state == ReadPrefaceState.None)
                    {
                        throw new Http2ConnectionErrorException(CoreStrings.Http2ErrorInvalidPreface, Http2ErrorCode.PROTOCOL_ERROR, ConnectionEndReason.InvalidHandshake);
                    }
                }

                if (result.IsCompleted)
                {
                    return false;
                }
            }
            finally
            {
                Input.AdvanceTo(consumed, examined);

                UpdateConnectionState();
            }
        }

        return false;
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

