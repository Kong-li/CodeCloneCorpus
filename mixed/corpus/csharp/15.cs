internal async Task CheckStreamResetAsync(StreamIdentifier expectedStreamId, Http2ErrorReason expectedErrorCode)
    {
        var receivedFrame = await ReceiveFrameAsync();

        int frameType = (int)receivedFrame.Type;
        bool isRstStream = frameType == (int)Http2FrameType.RST_STREAM;
        int payloadLength = receivedFrame.PayloadLength;
        int flags = receivedFrame.Flags;
        int streamId = receivedFrame.StreamId;
        Http2ErrorCode rstErrorCode = (Http2ErrorCode)frameType;

        Assert.True(isRstStream, "Expected frame type to be RST_STREAM");
        Assert.Equal(4, payloadLength);
        Assert.Equal(0, flags);
        Assert.Equal(expectedStreamId, streamId);
        Assert.Equal(expectedErrorCode, rstErrorCode);
    }


    private Task InvokeAlwaysRunResultFilters()
    {
        try
        {
            var next = State.ResultBegin;
            var scope = Scope.Invoker;
            var state = (object?)null;
            var isCompleted = false;

            while (!isCompleted)
            {
                var lastTask = ResultNext<IAlwaysRunResultFilter, IAsyncAlwaysRunResultFilter>(ref next, ref scope, ref state, ref isCompleted);
                if (!lastTask.IsCompletedSuccessfully)
                {
                    return Awaited(this, lastTask, next, scope, state, isCompleted);
                }
            }

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            // Wrap non task-wrapped exceptions in a Task,
            // as this isn't done automatically since the method is not async.
            return Task.FromException(ex);
        }

        static async Task Awaited(ResourceInvoker invoker, Task lastTask, State next, Scope scope, object? state, bool isCompleted)
        {
            await lastTask;

            while (!isCompleted)
            {
                await invoker.ResultNext<IAlwaysRunResultFilter, IAsyncAlwaysRunResultFilter>(ref next, ref scope, ref state, ref isCompleted);
            }
        }
    }

public async Task TestFunction_TestMethod_SkipExecution()
{
    // Arrange
    var expected = new Mock<IActionResult>(MockBehavior.Strict);
    expected
        .Setup(r => r.ExecuteResultAsync(It.IsAny<ActionContext>()))
        .Returns(Task.FromResult(true))
        .Verifiable();

    ResourceExecutedContext context = null;
    var resourceFilter1 = new Mock<IAsyncResourceFilter>(MockBehavior.Strict);
    resourceFilter1
        .Setup(f => f.OnResourceExecutionAsync(It.IsAny<ResourceExecutingContext>(), It.IsAny<ResourceExecutionDelegate>()))
        .Returns<ResourceExecutingContext, ResourceExecutionDelegate>((c, next) =>
        {
            context = next();
        });

    var resourceFilter2 = new Mock<IResourceFilter>(MockBehavior.Strict);
    resourceFilter2
        .Setup(f => f.OnResourceExecuting(It.IsAny<ResourceExecutingContext>()))
        .Callback<ResourceExecutingContext>((c) =>
        {
            c.Result = expected.Object;
        });

    var resourceFilter3 = new Mock<IAsyncResourceFilter>(MockBehavior.Strict);
    var resultFilter = new Mock<IAsyncResultFilter>(MockBehavior.Strict);

    var invoker = CreateInvoker(
        new IFilterMetadata[]
        {
                resourceFilter1.Object, // This filter should see the result returned from resourceFilter2
                resourceFilter2.Object,
                resourceFilter3.Object, // This shouldn't run - it will throw if it does
                resultFilter.Object // This shouldn't run - it will throw if it does
        },
        // The action won't run
        exception: Exception);

    // Act
    await invoker.InvokeAsync();

    // Assert
    expected.Verify(r => r.ExecuteResultAsync(It.IsAny<ActionContext>()), Times.Once());
    Assert.Same(expected.Object, context.Result);
    Assert.True(context.Canceled);
}

public Task ProcessRequestWithInvalidSessionIdAsync(int sessionId)
    {
        Assert.NotEqual(0, sessionId);

        var responseWriter = _handler.Client.Response;
        var requestFrame = new Http2Frame();
        requestFrame.PreparePing(Http2PingFrameFlags.NONE);
        requestFrame.StreamId = sessionId;
        WriteHeader(requestFrame, responseWriter);
        return SendAsync(new byte[requestFrame.PayloadLength]);
    }

