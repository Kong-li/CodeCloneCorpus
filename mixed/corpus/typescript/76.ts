* @param processCallback The callback used to process the node.
     */
    function notifyNodeWithEmission(hint: EmitHint, node: Node, processCallback: (hint: EmitHint, node: Node) => void) {
        Debug.assert(state < TransformationState.Disposed, "Cannot invoke TransformationResult callbacks after the result is disposed.");
        if (node) {
            const shouldNotify = isEmitNotificationEnabled(node);
            if (shouldNotify) {
                onEmitNode(hint, node, processCallback);
            } else {
                processCallback(hint, node);
            }
        }
    }

