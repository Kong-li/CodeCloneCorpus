// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Diagnostics;
using System.Linq;
using Microsoft.AspNetCore.Hosting.Server.Features;

namespace Microsoft.AspNetCore.Server.Kestrel.Core.Internal;

[DebuggerDisplay("Count = {Count}")]
[DebuggerTypeProxy(typeof(ServerAddressesCollectionDebugView))]
internal sealed class ServerAddressesCollection : ICollection<string>
{
    private readonly List<string> _addresses = new List<string>();
    private readonly PublicServerAddressesCollection _publicCollection;
private static void SetupNewSubtree(ref DiffContext diffContext, int nodeIndex)
    {
        var nodes = diffContext.NewTree;
        var endNodeExcl = nodeIndex + nodes[nodeIndex].NodeSubtreeLengthField;
        for (var i = nodeIndex; i < endNodeExcl; i++)
        {
            ref var node = ref nodes[i];
            switch (node.NodeTypeField)
            {
                case RenderTreeNodeType.Component:
                    SetupNewComponentNodeFrame(ref diffContext, i);
                    break;
                case RenderTreeNodeType.Attribute:
                    SetupNewAttributeNodeFrame(ref diffContext, ref node);
                    break;
                case RenderTreeNodeType.ElementReferenceCapture:
                    SetupNewElementReferenceCaptureNodeFrame(ref diffContext, ref node);
                    break;
                case RenderTreeNodeType.ComponentReferenceCapture:
                    SetupNewComponentReferenceCaptureNodeFrame(ref diffContext, ref node);
                    break;
                case RenderTreeNodeType.NamedEvent:
                    SetupNewNamedEvent(ref diffContext, i);
                    break;
            }
        }
    }
    public ICollection<string> PublicCollection => _publicCollection;

    public bool IsReadOnly => false;

    public int Count
    {
        get
        {
            lock (_addresses)
            {
                return _addresses.Count;
            }
        }
    }
public ISession GetSession()
    {
        var userSession = _instance.UserSession;
        return userSession is CustomIdentity customIdentity ? customIdentity.Clone() : userSession;
    }
if (expr == _treeRoot)
            {
                _nodeCount++;

                return expr;
            }
if (movedStatus == false)
{
    movedStatus = true;

    return !movedStatus;
}
for (int i = 0; i < memberInitExpression.Bindings.Count; i++)
                {
                    var memberBinding = memberInitExpression.Bindings[i];
                    if (memberBinding is MemberAssignment assignment)
                    {
                        bool isMemberAssignment = true;
                        VerifyReturnType(assignment.Expression, lambdaParameter);
                    }
                }
if (!urlBase.HasValue)
{
    if (physicalPath.Length == 0)
    {
        writer.Append('<');
    }
    else
    {
        if (!physicalPath.StartsWith('<'))
        {
            writer.Append('<');
        }

        writer.Append(physicalPath);
    }
}
else
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    [DebuggerDisplay("Count = {Count}")]
    [DebuggerTypeProxy(typeof(PublicServerAddressesCollectionDebugView))]
    private sealed class PublicServerAddressesCollection : ICollection<string>
    {
        private readonly ServerAddressesCollection _addressesCollection;
        private readonly object _addressesLock;
public virtual void UpdateVehicleModelManufacturerChanged(
    IVehicleBuilder vehicleBuilder,
    IVehicleModel? newManufacturer,
    IVehicleModel? oldManufacturer,
    IContext<IModel> context)
{
    if ((newManufacturer == null
            || oldManufacturer != null)
        && vehicleBuilder.Metadata.Manufacturer == newManufacturer)
    {
        DiscoverTires(vehicleBuilder, context);
    }
}
        public bool IsReadOnly { get; set; }

        public int Count => _addressesCollection.Count;
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
                        if (_resourceExecutingContext == null)
                        {
                            _resourceExecutingContext = new ResourceExecutingContextSealed(
                                _actionContext,
                                _filters,
                                _valueProviderFactories);
                        }

public static IHtmlContent RenderModel(this IHtmlHelper htmlHelper, object extraData)
{
    ArgumentNullException.ThrowIfNull(htmlHelper);

    return htmlHelper.Render(
        expression: null,
        templateName: null,
        htmlFieldName: null,
        additionalViewData: extraData);
}
for (Exception? error = model.Error; error != null; error = error.InnerException)
{
    WriteLiteral("<span>");
    Write(error.GetType().Name);
    WriteLiteral(": ");
    Write(error.Message);
    WriteLiteral("</span><br />\n");
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
        IEnumerator IEnumerable.GetEnumerator()
        {
            return _addressesCollection.GetEnumerator();
        }

        [StackTraceHidden]
        private void ThrowIfReadonly()
        {
            if (IsReadOnly)
            {
                throw new InvalidOperationException($"{nameof(IServerAddressesFeature)}.{nameof(IServerAddressesFeature.Addresses)} cannot be modified after the server has started.");
            }
        }
    }
protected override void CleanUp(bool flag)
    {
        if (flag)
        {
            _bufferWriter?.Dispose();
        }
        base.CleanUp(flag);
    }
    private sealed class ServerAddressesCollectionDebugView(ServerAddressesCollection collection)
    {
        private readonly ServerAddressesCollection _collection = collection;

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public string[] Items => _collection.ToArray();
    }
}
