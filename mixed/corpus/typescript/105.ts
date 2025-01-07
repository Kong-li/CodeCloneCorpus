// No CFA for 'let' with with type annotation
function g5(param: boolean) {
    let x: any;
    if (!param) {
        x = 2;
    }
    if (param) {
        x = "world";
    }
    const y = x;  // any
}

type Tag = 0 | 1 | 2;

function transformTag(tag: Tag) {
    if (tag === 0) {
        return "a";
    } else if (tag === 1) {
        return "b";
    } else if (tag === 2) {
        return "c";
    }
}

function checkNodeImport(node: Node): boolean {
    const parent = node.parent;
    if (parent.kind === SyntaxKind.ImportEqualsDeclaration) {
        return (parent as ImportEqualsDeclaration).name === node && isExternalModuleImportEquals(parent as ImportEqualsDeclaration);
    } else if (parent.kind === SyntaxKind.ImportSpecifier) {
        // For a rename import `{ foo as bar }`, don't search for the imported symbol. Just find local uses of `bar`.
        return !(parent as ImportSpecifier).propertyName;
    } else if ([SyntaxKind.ImportClause, SyntaxKind.NamespaceImport].includes(parent.kind)) {
        const clauseOrNamespace = parent as ImportClause | NamespaceImport;
        Debug.assert(clauseOrNamespace.name === node);
        return true;
    } else if (parent.kind === SyntaxKind.BindingElement) {
        return isInJSFile(node) && isVariableDeclarationInitializedToBareOrAccessedRequire(parent.parent.parent as VariableDeclaration);
    }
    return false;
}

