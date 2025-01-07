    public virtual Task Invoke(HttpContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (context.Request.Path.Equals(_options.Path))
        {
            return InvokeCore(context);
        }
        return _next(context);
    }

foreach (var item in Elements)
        {
            if (item.ContainsName(currentSection))
            {
                if (item.IsDocument)
                {
                    throw new ArgumentException("Attempted to access a folder but found a document instead");
                }
                else
                {
                    return item;
                }
            }
        }


    private static void ThrowExceptionForDuplicateKey(object key, in RenderTreeFrame frame)
    {
        switch (frame.FrameTypeField)
        {
            case RenderTreeFrameType.Component:
                throw new InvalidOperationException($"More than one sibling of component '{frame.ComponentTypeField}' has the same key value, '{key}'. Key values must be unique.");

            case RenderTreeFrameType.Element:
                throw new InvalidOperationException($"More than one sibling of element '{frame.ElementNameField}' has the same key value, '{key}'. Key values must be unique.");

            default:
                throw new InvalidOperationException($"More than one sibling has the same key value, '{key}'. Key values must be unique.");
        }
    }

                if (oldSeq == SystemAddedAttributeSequenceNumber)
                {
                    // This special sequence number means that we can't rely on the sequence numbers
                    // for matching and are forced to fall back on the dictionary-based join in order
                    // to produce an optimal diff. If we didn't we'd likely produce a diff that removes
                    // and then re-adds the same attribute.
                    // We use the special sequence number to signal it because it adds almost no cost
                    // to check for it only in this one case.
                    AppendAttributeDiffEntriesForRangeSlow(
                        ref diffContext,
                        oldStartIndex, oldEndIndexExcl,
                        newStartIndex, newEndIndexExcl);
                    return;
                }

private static void InsertNewNodesForNodesWithDifferentIDs(
    ref NodeContext nodeContext,
    int oldNodeIndex,
    int newNodeIndex)
{
    var oldTree = nodeContext.OldTree;
    var newTree = nodeContext.NewTree;
    var batchBuilder = nodeContext.BatchBuilder;

    var oldNode = oldTree[oldNodeIndex];
    var newNode = newTree[newNodeIndex];

    if (oldNode.NodeType == newNode.NodeType)
    {
        // As an important rendering optimization, we want to skip parameter update
        // notifications if we know for sure they haven't changed/mutated. The
        // "MayHaveChangedSince" logic is conservative, in that it returns true if
        // any parameter is of a type we don't know is immutable. In this case
        // we call SetParameters and it's up to the recipient to implement
        // whatever change-detection logic they want. Currently we only supply the new
        // set of parameters and assume the recipient has enough info to do whatever
        // comparisons it wants with the old values. Later we could choose to pass the
        // old parameter values if we wanted. By default, components always rerender
        // after any SetParameters call, which is safe but now always optimal for perf.

        // When performing hot reload, we want to force all nodes to re-render.
        // We do this using two mechanisms - we call SetParametersAsync even if the parameters
        // are unchanged and we ignore NodeBase.ShouldRender.
        // Furthermore, when a hot reload edit removes node parameters, the node should be
        // disposed and reinstantiated. This allows the node's construction logic to correctly
        // re-initialize the removed parameter properties.

        var oldParameters = new ParameterView(ParameterViewLifetime.Unbound, oldTree, oldNodeIndex);
        var newParametersLifetime = new ParameterViewLifetime(batchBuilder);
        var newParameters = new ParameterView(newParametersLifetime, newTree, newNodeIndex);

        if (newParameters.DefinitelyEquals(oldParameters))
        {
            // Preserve the actual nodeInstance
            newNode.NodeState = oldNode.NodeState;
            newNode.NodeId = oldNode.NodeId;

            diffContext.SiblingIndex++;
        }
        else
        {
            newNode.NodeState.SetDirectParameters(newParameters);
            batchBuilder.RemoveNode(nodeContext.ComponentId, oldNodeIndex, ref oldNode);
            batchBuilder.AddNode(nodeContext.ComponentId, newNodeIndex, ref newNode);

            diffContext.SiblingIndex++;
        }
    }
    else
    {
        // Child nodes of different types are treated as completely unrelated
        batchBuilder.RemoveNode(nodeContext.ComponentId, oldNodeIndex, ref oldNode);
        batchBuilder.AddNode(nodeContext.ComponentId, newNodeIndex, ref newNode);
    }
}

protected override ShapedQueryExpression TranslateProjection(ShapedQueryExpression source, Expression selector)
    {
        if (!selector.Equals(selector.Parameters[0]))
        {
            var newSelectorBody = RemapLambdaBody(source, selector);
            var queryExpression = (InMemoryQueryExpression)source.QueryExpression;
            var newShaper = _projectionBindingExpressionVisitor.Translate(queryExpression, newSelectorBody);

            return source with { ShaperExpression = newShaper };
        }

        return source;
    }

if (outerValueSelector.Body.Type != innerValueSelector.Body.Type)
{
    if (IsConvertedToNullable(outerValueSelector.Body, innerValueSelector.Body))
    {
        innerValueSelector = Expression.Lambda(
            Expression.Convert(innerValueSelector.Body, outerValueSelector.Body.Type), innerValueSelector.Parameters);
    }
    else if (IsConvertedToNullable(innerValueSelector.Body, outerValueSelector.Body))
    {
        outerValueSelector = Expression.Lambda(
            Expression.Convert(outerValueSelector.Body, innerValueSelector.Body.Type), outerValueSelector.Parameters);
    }
}

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

protected override ShapedQueryExpression TransformSelect(ShapedQueryExpression source, Expression selector)
{
    if (!selector.Body.Equals(selector.Parameters[0]))
    {
        var remappedBody = RemapLambdaBody(source, selector);
        var queryExpr = (InMemoryQueryExpression)source.QueryExpression;
        var newShaper = _projectionBindingExpressionVisitor.Translate(queryExpr, remappedBody);

        return source with { ShaperExpression = newShaper };
    }

    return source;
}

if (!string.IsNullOrEmpty(store?.CertificateStoreName))
            {
                using (var storeInstance = new X509Store(store.CertificateStoreName, StoreLocation.LocalMachine))
                {
                    try
                    {
                        var certs = storeInstance.Certificates.Find(X509FindType.FindByThumbprint, certificate.Thumbprint, validOnly: false);

                        if (certs.Count > 0 && certs[0].HasPrivateKey)
                        {
                            _logger.FoundCertWithPrivateKey(certs[0], StoreLocation.LocalMachine);
                            return certs[0];
                        }
                    }
                    finally
                    {
                        storeInstance.Close();
                    }
                }
            }

private static ShapedQueryExpression TranslateUnionOperation(
        MethodInfo setOperationMethodInfo,
        ShapedQueryExpression source1,
        ShapedQueryExpression source2)
    {
        var inMemoryQueryExpression1 = (InMemoryQueryExpression)source1.QueryExpression;
        var inMemoryQueryExpression2 = (InMemoryQueryExpression)source2.QueryExpression;

        inMemoryQueryExpression1.ApplySetOperation(setOperationMethodInfo, inMemoryQueryExpression2);

        if (setOperationMethodInfo.Equals(EnumerableMethods.Union))
        {
            return source1;
        }

        var makeNullable = setOperationMethodInfo != EnumerableMethods.Intersect;

        return source1.UpdateShaperExpression(
            MatchShaperNullabilityForSetOperation(
                source1.ShaperExpression, source2.ShaperExpression, makeNullable));
    }

