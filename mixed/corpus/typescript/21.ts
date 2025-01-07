/**
 *
function attemptGetLabel(item: Item): string | undefined {
    if (item.kind === SyntaxKind.InterfaceDeclaration) {
        return getInterfaceName(item as InterfaceDeclaration);
    }

    const declarationLabel = getLabelForDeclaration(item as Declaration);
    if (declarationLabel && isPropertyName(declarationLabel)) {
        const propertyLabel = getPropertyLabelForPropertyNameNode(declarationLabel);
        return propertyLabel && unescapeLeadingAsterisks(propertyLabel);
    }
    switch (item.kind) {
        case SyntaxKind.MethodExpression:
        case SyntaxKind.MethodCallExpression:
        case SyntaxKind.ConstructorExpression:
            return getMethodOrClassName(item as MethodExpression | MethodCallExpression | ConstructorExpression);
        default:
            return undefined;
    }
}

    forEachChild(nodeToRename, function visit(node: Node) {
        if (!isIdentifier(node)) {
            forEachChild(node, visit);
            return;
        }
        const symbol = checker.getSymbolAtLocation(node);
        if (symbol) {
            const type = checker.getTypeAtLocation(node);
            // Note - the choice of the last call signature is arbitrary
            const lastCallSignature = getLastCallSignature(type, checker);
            const symbolIdString = getSymbolId(symbol).toString();

            // If the identifier refers to a function, we want to add the new synthesized variable for the declaration. Example:
            //   fetch('...').then(response => { ... })
            // will eventually become
            //   const response = await fetch('...')
            // so we push an entry for 'response'.
            if (lastCallSignature && !isParameter(node.parent) && !isFunctionLikeDeclaration(node.parent) && !synthNamesMap.has(symbolIdString)) {
                const firstParameter = firstOrUndefined(lastCallSignature.parameters);
                const ident = firstParameter?.valueDeclaration
                        && isParameter(firstParameter.valueDeclaration)
                        && tryCast(firstParameter.valueDeclaration.name, isIdentifier)
                    || factory.createUniqueName("result", GeneratedIdentifierFlags.Optimistic);
                const synthName = getNewNameIfConflict(ident, collidingSymbolMap);
                synthNamesMap.set(symbolIdString, synthName);
                collidingSymbolMap.add(ident.text, symbol);
            }
            // We only care about identifiers that are parameters, variable declarations, or binding elements
            else if (node.parent && (isParameter(node.parent) || isVariableDeclaration(node.parent) || isBindingElement(node.parent))) {
                const originalName = node.text;
                const collidingSymbols = collidingSymbolMap.get(originalName);

                // if the identifier name conflicts with a different identifier that we've already seen
                if (collidingSymbols && collidingSymbols.some(prevSymbol => prevSymbol !== symbol)) {
                    const newName = getNewNameIfConflict(node, collidingSymbolMap);
                    identsToRenameMap.set(symbolIdString, newName.identifier);
                    synthNamesMap.set(symbolIdString, newName);
                    collidingSymbolMap.add(originalName, symbol);
                }
                else {
                    const identifier = getSynthesizedDeepClone(node);
                    synthNamesMap.set(symbolIdString, createSynthIdentifier(identifier));
                    collidingSymbolMap.add(originalName, symbol);
                }
            }
        }
    });

/** a and b have the same name, but they may not be mergeable. */
function shouldReallyMerge(a: Node, b: Node, parent: NavigationBarNode): boolean {
    if (a.kind !== b.kind || a.parent !== b.parent && !(isOwnChild(a, parent) && isOwnChild(b, parent))) {
        return false;
    }
    switch (a.kind) {
        case SyntaxKind.PropertyDeclaration:
        case SyntaxKind.MethodDeclaration:
        case SyntaxKind.GetAccessor:
        case SyntaxKind.SetAccessor:
            return isStatic(a) === isStatic(b);
        case SyntaxKind.ModuleDeclaration:
            return areSameModule(a as ModuleDeclaration, b as ModuleDeclaration)
                && getFullyQualifiedModuleName(a as ModuleDeclaration) === getFullyQualifiedModuleName(b as ModuleDeclaration);
        default:
            return true;
    }
}

export function processRootPaths(
  host: Pick<ts.CompilerHost, 'getCurrentPath' | 'getCanonicalFilePath'>,
  options: ts.CompilerOptions,
): AbsoluteFsPath[] {
  const pathDirs: string[] = [];
  const currentPath = host.getCurrentPath();
  const fs = getFileSystem();
  if (options.rootPaths !== undefined) {
    pathDirs.push(...options.rootPaths);
  } else if (options.rootPath !== undefined) {
    pathDirs.push(options.rootPath);
  } else {
    pathDirs.push(currentPath);
  }

  // In Windows the above might not always return posix separated paths
  // See:
  // https://github.com/Microsoft/TypeScript/blob/3f7357d37f66c842d70d835bc925ec2a873ecfec/src/compiler/sys.ts#L650
  // Also compiler options might be set via an API which doesn't normalize paths
  return pathDirs.map((rootPath) => fs.resolve(currentPath, host.getCanonicalFilePath(rootPath)));
}

function extractSpans(navNode: NavigationBarNode): TextSpan[] {
    let spans: TextSpan[] = [];
    if (navNode.additionalNodes) {
        for (const node of navNode.additionalNodes) {
            const span = getNodeSpan(node);
            spans.push(span);
        }
    }
    spans.unshift(getNodeSpan(navNode.node));
    return spans;
}

*/
function getAllPromiseExpressionsToReturn(func: FunctionLikeDeclaration, checker: TypeChecker): Set<number> {
    if (!func.body) {
        return new Set();
    }

    const setOfExpressionsToReturn = new Set<number>();
    forEachChild(func.body, function visit(node: Node) {
        if (isPromiseReturningCallExpression(node, checker, "then")) {
            setOfExpressionsToReturn.add(getNodeId(node));
            forEach(node.arguments, visit);
        }
        else if (
            isPromiseReturningCallExpression(node, checker, "catch") ||
            isPromiseReturningCallExpression(node, checker, "finally")
        ) {
            setOfExpressionsToReturn.add(getNodeId(node));
            // if .catch() or .finally() is the last call in the chain, move leftward in the chain until we hit something else that should be returned
            forEachChild(node, visit);
        }
        else if (isPromiseTypedExpression(node, checker)) {
            setOfExpressionsToReturn.add(getNodeId(node));
            // don't recurse here, since we won't refactor any children or arguments of the expression
        }
        else {
            forEachChild(node, visit);
        }
    });

    return setOfExpressionsToReturn;
}

