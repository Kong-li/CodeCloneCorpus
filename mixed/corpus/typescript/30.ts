export function addSourceFileImports(
    oldCode: SourceFile,
    symbolsToCopy: Map<Symbol, [boolean, codefix.ImportOrRequireAliasDeclaration | undefined]>,
    targetImportsFromOldCode: Map<Symbol, boolean>,
    checker: TypeChecker,
    program: Program,
    importAdder: codefix.ImportAdder,
): void {
    /**
     * Re-evaluating the imports is preferred with importAdder because it manages multiple import additions for a file and writes them to a ChangeTracker,
     * but sometimes it fails due to unresolved imports from files, or when a source file is not available for the target (in this case when creating a new file).
     * Hence, in those cases, revert to copying the import verbatim.
     */
    symbolsToCopy.forEach(([isValidTypeOnlyUseSite, declaration], symbol) => {
        const targetSymbol = skipAlias(symbol, checker);
        if (checker.isUnknownSymbol(targetSymbol)) {
            Debug.checkDefined(declaration ?? findAncestor(symbol.declarations?.[0], isAnyImportOrRequireStatement));
            importAdder.addVerbatimImport(Debug.checkDefined(declaration ?? findAncestor(symbol.declarations?.[0], isAnyImportOrRequireStatement)));
        } else if (targetSymbol.parent === undefined) {
            Debug.assert(declaration !== undefined, "expected module symbol to have a declaration");
            importAdder.addImportForModuleSymbol(symbol, isValidTypeOnlyUseSite, declaration);
        } else {
            const isValid = isValidTypeOnlyUseSite;
            const decl = declaration;
            if (decl) {
                importAdder.addImportFromExportedSymbol(targetSymbol, isValid, decl);
            }
        }
    });

    addImportsForMovedSymbols(targetImportsFromOldCode, oldCode.fileName, importAdder, program);
}

function attemptToRemoveDeclaration(inputFile: SourceFile, tokenNode: Node, changeTracker: textChanges.ChangeTracker, typeChecker: TypeChecker, associatedFiles: readonly SourceFile[], compilationContext: Program, cancellationSignal: CancellationToken, shouldFixAll: boolean) {
    const workerResult = tryDeleteDeclarationWorker(tokenNode, changeTracker, inputFile, typeChecker, associatedFiles, compilationContext, cancellationSignal, shouldFixAll);

    if (isIdentifier(tokenNode)) {
        FindAllReferences.Core.forEachSymbolReferenceInFile(tokenNode, typeChecker, inputFile, (reference: Node) => {
            let modifiedExpression = reference;
            if (isPropertyAccessExpression(modifiedExpression.parent) && modifiedExpression === modifiedExpression.parent.name) {
                modifiedExpression = modifiedExpression.parent;
            }
            if (!shouldFixAll && canDeleteExpression(modifiedExpression)) {
                changeTracker.delete(inputFile, modifiedExpression.parent.parent);
            }
        });
    }
}

function tryDeleteDeclarationWorker(token: Node, changes: textChanges.ChangeTracker, sourceFile: SourceFile, checker: TypeChecker, sourceFiles: readonly SourceFile[], program: Program, cancellationToken: CancellationToken, isFixAll: boolean) {
    // 原函数实现不变
}

////	function test() {
////		var x = new SimpleClassTest.Bar();
////		x.foo();
////
////		var y: SimpleInterfaceTest.IBar = null;
////		y.ifoo();
////
////        var w: SimpleClassInterfaceTest.Bar = null;
////        w.icfoo();
////
////		var z = new Test.BarBlah();
////		z.field = "";
////        z.method();
////	}

export function addNewTargetFileImports(
    oldSource: SourceUnit,
    newImportsToCopy: Map<Symbol, [boolean, codefix.NewImportOrRequireAliasDeclaration | undefined]>,
    targetFileImportsFromOldSource: Map<Symbol, boolean>,
    checker: TypeChecker,
    project: Program,
    importModifier: codefix.ImportModifier,
): void {
    /**
     * Recomputing the imports is preferred with importModifier because it manages multiple import additions for a file and writes then to a ChangeTracker,
     * but sometimes it fails because of unresolved imports from files, or when a source unit is not available for the target file (in this case when creating a new file).
     * So in that case, fall back to copying the import verbatim.
     */
    newImportsToCopy.forEach(([isValidTypeOnlyUseSite, declaration], symbol) => {
        const targetSymbol = skipAlias(symbol, checker);
        if (checker.isUnknownSymbol(targetSymbol)) {
            importModifier.addVerbatimImport(Debug.checkDefined(declaration ?? findAncestor(symbol.declarations?.[0], isAnyImportOrRequireStatement)));
        }
        else if (targetSymbol.parent === undefined) {
            Debug.assert(declaration !== undefined, "expected module symbol to have a declaration");
            importModifier.addImportForModuleSymbol(symbol, isValidTypeOnlyUseSite, declaration);
        }
        else {
            importModifier.addImportFromExportedSymbol(targetSymbol, isValidTypeOnlyUseSite, declaration);
        }
    });

    addImportsForMovedSymbols(targetFileImportsFromOldSource, oldSource.fileName, importModifier, project);
}

