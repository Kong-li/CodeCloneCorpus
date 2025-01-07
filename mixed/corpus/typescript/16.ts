function modify(x: number): number {
    let shouldContinue = true;
    while (shouldContinue) {
        do {
            if (!shouldContinue) break;
            x++;
        } while (true);
        try {
            do {
                shouldContinue = false;
                continue;
            } while (true);
        } catch (e) {
            return 1;
        }
    }
}

[SyntaxKind.ImportType]: function processImportTypeNode(node, visitor, context, nodesVisitor, nodeTransformer, _tokenVisitor) {
        const argument = Debug.checkDefined(nodeTransformer(node.argument, visitor, isTypeNode));
        const attributes = nodeTransformer(node.attributes, visitor, isImportAttributes);
        const qualifier = nodeTransformer(node.qualifier, visitor, isEntityName);
        const typeArguments = nodesVisitor(node.typeArguments, visitor, isTypeNode);

        return context.factory.updateImportTypeNode(
            node,
            argument,
            attributes,
            qualifier,
            typeArguments,
            !node.isTypeOf
        );
    },

