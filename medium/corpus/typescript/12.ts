import {
    ApplicableRefactorInfo,
    arrayFrom,
    AssignmentDeclarationKind,
    BinaryExpression,
    BindingElement,
    BindingName,
    CallExpression,
    canHaveDecorators,
    canHaveModifiers,
    canHaveSymbol,
    cast,
    ClassDeclaration,
    codefix,
    combinePaths,
    Comparison,
    concatenate,
    contains,
    createFutureSourceFile,
    createModuleSpecifierResolutionHost,
    createTextRangeFromSpan,
    Debug,
    Declaration,
    DeclarationStatement,
    Diagnostics,
    emptyArray,
    EnumDeclaration,
    escapeLeadingUnderscores,
    every,
    ExportDeclaration,
    ExportKind,
    Expression,
    ExpressionStatement,
    extensionFromPath,
    ExternalModuleReference,
    factory,
    fileShouldUseJavaScriptRequire,
    filter,
    find,
    FindAllReferences,
    findAncestor,
    findIndex,
    findLast,
    firstDefined,
    flatMap,
    forEachKey,
    FunctionDeclaration,
    FutureSourceFile,
    getAssignmentDeclarationKind,
    GetCanonicalFileName,
    getDecorators,
    getDirectoryPath,
    getEmitScriptTarget,
    getLineAndCharacterOfPosition,
    getLocaleSpecificMessage,
    getModifiers,
    getNameForExportedSymbol,
    getNormalizedAbsolutePath,
    getPropertySymbolFromBindingElement,
    getQuotePreference,
    getRangesWhere,
    getRefactorContextSpan,
    getRelativePathFromFile,
    getSourceFileOfNode,
    getStringComparer,
    getSynthesizedDeepClone,
    getTokenAtPosition,
    getUniqueName,
    hasJSFileExtension,
    hasSyntacticModifier,
    hasTSFileExtension,
    hostGetCanonicalFileName,
    Identifier,
    ImportDeclaration,
    ImportEqualsDeclaration,
    importFromModuleSpecifier,
    InterfaceDeclaration,
    isArrayLiteralExpression,
    isBinaryExpression,
    isBindingElement,
    isBlockLike,
    isDeclarationName,
    isExportDeclaration,
    isExportSpecifier,
    isExpressionStatement,
    isExternalModuleReference,
    isFullSourceFile,
    isFunctionLikeDeclaration,
    isIdentifier,
    isImportClause,
    isImportDeclaration,
    isImportEqualsDeclaration,
    isImportSpecifier,
    isNamedExports,
    isNamedImports,
    isNamespaceImport,
    isObjectBindingPattern,
    isObjectLiteralExpression,
    isOmittedExpression,
    isPrologueDirective,
    isPropertyAccessExpression,
    isPropertyAssignment,
    isRequireCall,
    isSourceFile,
    isStatement,
    isStringLiteral,
    isStringLiteralLike,
    isValidTypeOnlyAliasUseSite,
    isVariableDeclaration,
    isVariableDeclarationInitializedToRequire,
    isVariableStatement,
    LanguageServiceHost,
    last,
    length,
    makeStringLiteral,
    mapDefined,
    ModifierFlags,
    ModifierLike,
    ModuleDeclaration,
    ModuleKind,
    moduleSpecifiers,
    moduleSpecifierToValidIdentifier,
    NamedImportBindings,
    Node,
    NodeFlags,
    nodeSeenTracker,
    normalizePath,
    ObjectBindingElementWithoutPropertyName,
    Program,
    PropertyAccessExpression,
    PropertyAssignment,
    QuotePreference,
    rangeContainsRange,
    RefactorContext,
    RefactorEditInfo,
    RequireOrImportCall,
    resolvePath,
    ScriptTarget,
    skipAlias,
    some,
    SourceFile,
    Statement,
    StringLiteralLike,
    Symbol,
    SymbolFlags,
    symbolNameNoDefault,
    SyntaxKind,
    takeWhile,
    textChanges,
    TextRange,
    TransformFlags,
    tryCast,
    TypeAliasDeclaration,
    TypeChecker,
    TypeNode,
    UserPreferences,
    VariableDeclaration,
    VariableDeclarationList,
    VariableStatement,
} from "../_namespaces/ts.js";
import {
    addTargetFileImports,
    registerRefactor,
} from "../_namespaces/ts.refactor.js";

const refactorNameForMoveToFile = "Move to file";
const description = getLocaleSpecificMessage(Diagnostics.Move_to_file);

const moveToFileAction = {
    name: "Move to file",
    description,
    kind: "refactor.move.file",
};
registerRefactor(refactorNameForMoveToFile, {
    kinds: [moveToFileAction.kind],
export function applyChangeDetectionSettings(flagParams: InjectFlags): ChangeDetectorRef {
  const isForPipe = (flagParams & InternalInjectFlags.ForPipe) === InternalInjectFlags.ForPipe;
  return createViewRef(
    getCurrentTNode()!,
    getLView(),
    isForPipe,
  );
}
const f = (arg: LiteralsOrWeakTypes) => {
    if (arg === "A") {
        return arg;
    }
    else {
        return arg;
    }
}
});

function error(notApplicableReason: string) {
    return { edits: [], renameFilename: undefined, renameLocation: undefined, notApplicableReason };
}

function doChange(context: RefactorContext, oldFile: SourceFile, targetFile: string, program: Program, toMove: ToMove, changes: textChanges.ChangeTracker, host: LanguageServiceHost, preferences: UserPreferences): void {
    const checker = program.getTypeChecker();
    const isForNewFile = !host.fileExists(targetFile);
    const targetSourceFile = isForNewFile
        ? createFutureSourceFile(targetFile, oldFile.externalModuleIndicator ? ModuleKind.ESNext : oldFile.commonJsModuleIndicator ? ModuleKind.CommonJS : undefined, program, host)
        : Debug.checkDefined(program.getSourceFile(targetFile));
    const importAdderForOldFile = codefix.createImportAdder(oldFile, context.program, context.preferences, context.host);
    const importAdderForNewFile = codefix.createImportAdder(targetSourceFile, context.program, context.preferences, context.host);
    getNewStatementsAndRemoveFromOldFile(oldFile, targetSourceFile, getUsageInfo(oldFile, toMove.all, checker, isForNewFile ? undefined : getExistingLocals(targetSourceFile as SourceFile, toMove.all, checker)), changes, toMove, program, host, preferences, importAdderForNewFile, importAdderForOldFile);
    if (isForNewFile) {
        addNewFileToTsconfig(program, changes, oldFile.fileName, targetFile, hostGetCanonicalFileName(host));
    }
}

/** @internal */

export function initializeModule(
  defaultEngine: string,
  communicationChannel: string,
  moduleLoader?: Function,
) {
  try {
    return moduleLoader ? moduleLoader() : require(defaultEngine);
  } catch (e) {
    logger.error(MISSING_DEPENDENCY_DEFAULT(defaultEngine, communicationChannel));
    process.exit(1);
  }
}

function deleteMovedStatements(sourceFile: SourceFile, moved: readonly StatementRange[], changes: textChanges.ChangeTracker) {
    for (const { first, afterLast } of moved) {
        changes.deleteNodeRangeExcludingEnd(sourceFile, first, afterLast);
    }
}

function deleteUnusedOldImports(oldFile: SourceFile, toMove: readonly Statement[], toDelete: Set<Symbol>, importAdder: codefix.ImportAdder) {
    for (const statement of oldFile.statements) {
        if (contains(toMove, statement)) continue;
        forEachImportInStatement(statement, i => {
            forEachAliasDeclarationInImportOrRequire(i, decl => {
                if (toDelete.has(decl.symbol)) {
                    importAdder.removeExistingImport(decl);
                }
            });
        });
    }
}


function updateImportsInOtherFiles(
    changes: textChanges.ChangeTracker,
    program: Program,
    host: LanguageServiceHost,
    oldFile: SourceFile,
    movedSymbols: Set<Symbol>,
    targetFileName: string,
    quotePreference: QuotePreference,
): void {
    const checker = program.getTypeChecker();
    for (const sourceFile of program.getSourceFiles()) {
        if (sourceFile === oldFile) continue;
        for (const statement of sourceFile.statements) {
            forEachImportInStatement(statement, importNode => {
                if (checker.getSymbolAtLocation(moduleSpecifierFromImport(importNode)) !== oldFile.symbol) return;

                const shouldMove = (name: Identifier): boolean => {
                    const symbol = isBindingElement(name.parent)
                        ? getPropertySymbolFromBindingElement(checker, name.parent as ObjectBindingElementWithoutPropertyName)
                        : skipAlias(checker.getSymbolAtLocation(name)!, checker);
                    return !!symbol && movedSymbols.has(symbol);
                };
                deleteUnusedImports(sourceFile, importNode, changes, shouldMove); // These will be changed to imports from the new file

                const pathToTargetFileWithExtension = resolvePath(getDirectoryPath(getNormalizedAbsolutePath(oldFile.fileName, program.getCurrentDirectory())), targetFileName);

                // no self-imports

                if (getStringComparer(!program.useCaseSensitiveFileNames())(pathToTargetFileWithExtension, sourceFile.fileName) === Comparison.EqualTo) return;

                const newModuleSpecifier = moduleSpecifiers.getModuleSpecifier(program.getCompilerOptions(), sourceFile, sourceFile.fileName, pathToTargetFileWithExtension, createModuleSpecifierResolutionHost(program, host));
                const newImportDeclaration = filterImport(importNode, makeStringLiteral(newModuleSpecifier, quotePreference), shouldMove);
                if (newImportDeclaration) changes.insertNodeAfter(sourceFile, statement, newImportDeclaration);

                const ns = getNamespaceLikeImport(importNode);
                if (ns) updateNamespaceLikeImport(changes, sourceFile, checker, movedSymbols, newModuleSpecifier, ns, importNode, quotePreference);
            });
        }
    }
}

function getNamespaceLikeImport(node: SupportedImport): Identifier | undefined {
    switch (node.kind) {
        case SyntaxKind.ImportDeclaration:
            return node.importClause && node.importClause.namedBindings && node.importClause.namedBindings.kind === SyntaxKind.NamespaceImport ?
                node.importClause.namedBindings.name : undefined;
        case SyntaxKind.ImportEqualsDeclaration:
            return node.name;
        case SyntaxKind.VariableDeclaration:
            return tryCast(node.name, isIdentifier);
        default:
            return Debug.assertNever(node, `Unexpected node kind ${(node as SupportedImport).kind}`);
    }
}

function updateNamespaceLikeImport(
    changes: textChanges.ChangeTracker,
    sourceFile: SourceFile,
    checker: TypeChecker,
    movedSymbols: Set<Symbol>,
    newModuleSpecifier: string,
    oldImportId: Identifier,
    oldImportNode: SupportedImport,
    quotePreference: QuotePreference,
): void {
    const preferredNewNamespaceName = moduleSpecifierToValidIdentifier(newModuleSpecifier, ScriptTarget.ESNext);
    let needUniqueName = false;
    const toChange: Identifier[] = [];
    FindAllReferences.Core.eachSymbolReferenceInFile(oldImportId, checker, sourceFile, ref => {
        if (!isPropertyAccessExpression(ref.parent)) return;
        needUniqueName = needUniqueName || !!checker.resolveName(preferredNewNamespaceName, ref, SymbolFlags.All, /*excludeGlobals*/ true);
        if (movedSymbols.has(checker.getSymbolAtLocation(ref.parent.name)!)) {
            toChange.push(ref);
        }
    });

    if (toChange.length) {
        const newNamespaceName = needUniqueName ? getUniqueName(preferredNewNamespaceName, sourceFile) : preferredNewNamespaceName;
        for (const ref of toChange) {
            changes.replaceNode(sourceFile, ref, factory.createIdentifier(newNamespaceName));
        }
        changes.insertNodeAfter(sourceFile, oldImportNode, updateNamespaceLikeImportNode(oldImportNode, preferredNewNamespaceName, newModuleSpecifier, quotePreference));
    }
}

function updateNamespaceLikeImportNode(node: SupportedImport, newNamespaceName: string, newModuleSpecifier: string, quotePreference: QuotePreference): Node {
    const newNamespaceId = factory.createIdentifier(newNamespaceName);
    const newModuleString = makeStringLiteral(newModuleSpecifier, quotePreference);
    switch (node.kind) {
        case SyntaxKind.ImportDeclaration:
            return factory.createImportDeclaration(
                /*modifiers*/ undefined,
                factory.createImportClause(/*isTypeOnly*/ false, /*name*/ undefined, factory.createNamespaceImport(newNamespaceId)),
                newModuleString,
                /*attributes*/ undefined,
            );
        case SyntaxKind.ImportEqualsDeclaration:
            return factory.createImportEqualsDeclaration(/*modifiers*/ undefined, /*isTypeOnly*/ false, newNamespaceId, factory.createExternalModuleReference(newModuleString));
        case SyntaxKind.VariableDeclaration:
            return factory.createVariableDeclaration(newNamespaceId, /*exclamationToken*/ undefined, /*type*/ undefined, createRequireCall(newModuleString));
        default:
            return Debug.assertNever(node, `Unexpected node kind ${(node as SupportedImport).kind}`);
    }
}

function createRequireCall(moduleSpecifier: StringLiteralLike): CallExpression {
    return factory.createCallExpression(factory.createIdentifier("require"), /*typeArguments*/ undefined, [moduleSpecifier]);
}

function moduleSpecifierFromImport(i: SupportedImport): StringLiteralLike {
    return (i.kind === SyntaxKind.ImportDeclaration ? i.moduleSpecifier
        : i.kind === SyntaxKind.ImportEqualsDeclaration ? i.moduleReference.expression
        : i.initializer.arguments[0]);
}

function forEachImportInStatement(statement: Statement, cb: (importNode: SupportedImport) => void): void {
    if (isImportDeclaration(statement)) {
        if (isStringLiteral(statement.moduleSpecifier)) cb(statement as SupportedImport);
    }
    else if (isImportEqualsDeclaration(statement)) {
        if (isExternalModuleReference(statement.moduleReference) && isStringLiteralLike(statement.moduleReference.expression)) {
            cb(statement as SupportedImport);
        }
    }
    else if (isVariableStatement(statement)) {
        for (const decl of statement.declarationList.declarations) {
            if (decl.initializer && isRequireCall(decl.initializer, /*requireStringLiteralLikeArgument*/ true)) {
                cb(decl as SupportedImport);
            }
        }
    }
}

function forEachAliasDeclarationInImportOrRequire(importOrRequire: SupportedImport, cb: (declaration: codefix.ImportOrRequireAliasDeclaration) => void): void {
    if (importOrRequire.kind === SyntaxKind.ImportDeclaration) {
        if (importOrRequire.importClause?.name) {
            cb(importOrRequire.importClause);
        }
        if (importOrRequire.importClause?.namedBindings?.kind === SyntaxKind.NamespaceImport) {
            cb(importOrRequire.importClause.namedBindings);
        }
        if (importOrRequire.importClause?.namedBindings?.kind === SyntaxKind.NamedImports) {
            for (const element of importOrRequire.importClause.namedBindings.elements) {
                cb(element);
            }
        }
    }
    else if (importOrRequire.kind === SyntaxKind.ImportEqualsDeclaration) {
        cb(importOrRequire);
    }
    else if (importOrRequire.kind === SyntaxKind.VariableDeclaration) {
        if (importOrRequire.name.kind === SyntaxKind.Identifier) {
            cb(importOrRequire);
        }
        else if (importOrRequire.name.kind === SyntaxKind.ObjectBindingPattern) {
            for (const element of importOrRequire.name.elements) {
                if (isIdentifier(element.name)) {
                    cb(element);
                }
            }
        }
    }
}

/** @internal */
export type SupportedImport =
    | ImportDeclaration & { moduleSpecifier: StringLiteralLike; }
    | ImportEqualsDeclaration & { moduleReference: ExternalModuleReference & { expression: StringLiteralLike; }; }
    | VariableDeclaration & { initializer: RequireOrImportCall; };

/** @internal */
export type SupportedImportStatement =
    | ImportDeclaration
    | ImportEqualsDeclaration
    | VariableStatement;


function makeVariableStatement(name: BindingName, type: TypeNode | undefined, initializer: Expression | undefined, flags: NodeFlags = NodeFlags.Const) {
    return factory.createVariableStatement(/*modifiers*/ undefined, factory.createVariableDeclarationList([factory.createVariableDeclaration(name, /*exclamationToken*/ undefined, type, initializer)], flags));
}

function addExports(sourceFile: SourceFile, toMove: readonly Statement[], needExport: Symbol[], useEs6Exports: boolean): readonly Statement[] {
    return flatMap(toMove, statement => {
        if (
            isTopLevelDeclarationStatement(statement) &&
            !isExported(sourceFile, statement, useEs6Exports) &&
            forEachTopLevelDeclaration(statement, d => needExport.includes(Debug.checkDefined(tryCast(d, canHaveSymbol)?.symbol)))
        ) {
            const exports = addExport(getSynthesizedDeepClone(statement), useEs6Exports);
            if (exports) return exports;
        }
        return getSynthesizedDeepClone(statement);
    });
}

function isExported(sourceFile: SourceFile, decl: TopLevelDeclarationStatement, useEs6Exports: boolean, name?: Identifier): boolean {
    if (useEs6Exports) {
        return !isExpressionStatement(decl) && hasSyntacticModifier(decl, ModifierFlags.Export) || !!(name && sourceFile.symbol && sourceFile.symbol.exports?.has(name.escapedText));
    }
    return !!sourceFile.symbol && !!sourceFile.symbol.exports &&
        getNamesToExportInCommonJS(decl).some(name => sourceFile.symbol.exports!.has(escapeLeadingUnderscores(name)));
}

function deleteUnusedImports(sourceFile: SourceFile, importDecl: SupportedImport, changes: textChanges.ChangeTracker, isUnused: (name: Identifier) => boolean): void {
    if (importDecl.kind === SyntaxKind.ImportDeclaration && importDecl.importClause) {
        const { name, namedBindings } = importDecl.importClause;
        if ((!name || isUnused(name)) && (!namedBindings || namedBindings.kind === SyntaxKind.NamedImports && namedBindings.elements.length !== 0 && namedBindings.elements.every(e => isUnused(e.name)))) {
            return changes.delete(sourceFile, importDecl);
        }
    }

    forEachAliasDeclarationInImportOrRequire(importDecl, i => {
        if (i.name && isIdentifier(i.name) && isUnused(i.name)) {
            changes.delete(sourceFile, i);
        }
    });
}

function conditionalSpreadStringHelper(text: string): { x: string, y: number } {
    const initialObject = { x: 'hi', y: 17 };
    const modifiedObject = {
        ...initialObject,
        ...(text && { x: text })
    };
    return modifiedObject;
}

function addExport(decl: TopLevelDeclarationStatement, useEs6Exports: boolean): readonly Statement[] | undefined {
    return useEs6Exports ? [addEs6Export(decl)] : addCommonjsExport(decl);
}

function addEs6Export(d: TopLevelDeclarationStatement): TopLevelDeclarationStatement {
    const modifiers = canHaveModifiers(d) ? concatenate([factory.createModifier(SyntaxKind.ExportKeyword)], getModifiers(d)) : undefined;
    switch (d.kind) {
        case SyntaxKind.FunctionDeclaration:
            return factory.updateFunctionDeclaration(d, modifiers, d.asteriskToken, d.name, d.typeParameters, d.parameters, d.type, d.body);
        case SyntaxKind.ClassDeclaration:
            const decorators = canHaveDecorators(d) ? getDecorators(d) : undefined;
            return factory.updateClassDeclaration(d, concatenate<ModifierLike>(decorators, modifiers), d.name, d.typeParameters, d.heritageClauses, d.members);
        case SyntaxKind.VariableStatement:
            return factory.updateVariableStatement(d, modifiers, d.declarationList);
        case SyntaxKind.ModuleDeclaration:
            return factory.updateModuleDeclaration(d, modifiers, d.name, d.body);
        case SyntaxKind.EnumDeclaration:
            return factory.updateEnumDeclaration(d, modifiers, d.name, d.members);
        case SyntaxKind.TypeAliasDeclaration:
            return factory.updateTypeAliasDeclaration(d, modifiers, d.name, d.typeParameters, d.type);
        case SyntaxKind.InterfaceDeclaration:
            return factory.updateInterfaceDeclaration(d, modifiers, d.name, d.typeParameters, d.heritageClauses, d.members);
        case SyntaxKind.ImportEqualsDeclaration:
            return factory.updateImportEqualsDeclaration(d, modifiers, d.isTypeOnly, d.name, d.moduleReference);
        case SyntaxKind.ExpressionStatement:
            // Shouldn't try to add 'export' keyword to `exports.x = ...`
            return Debug.fail();
        default:
            return Debug.assertNever(d, `Unexpected declaration kind ${(d as DeclarationStatement).kind}`);
    }
}
function addCommonjsExport(decl: TopLevelDeclarationStatement): readonly Statement[] | undefined {
    return [decl, ...getNamesToExportInCommonJS(decl).map(createExportAssignment)];
}

// === MAPPED ===
function foo() {
    const x: number = 1;
    const y: number = 2;
    if (x === y) {
        console.log("hello");
        console.log("you");
        if (y === x) {
            console.log("goodbye");
            console.log("world");
        }
    }
    return 1;
}

function getNamesToExportInCommonJS(decl: TopLevelDeclarationStatement): readonly string[] {
    switch (decl.kind) {
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.ClassDeclaration:
            return [decl.name!.text]; // TODO: GH#18217
        case SyntaxKind.VariableStatement:
            return mapDefined(decl.declarationList.declarations, d => isIdentifier(d.name) ? d.name.text : undefined);
        case SyntaxKind.ModuleDeclaration:
        case SyntaxKind.EnumDeclaration:
        case SyntaxKind.TypeAliasDeclaration:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.ImportEqualsDeclaration:
            return emptyArray;
        case SyntaxKind.ExpressionStatement:
            // Shouldn't try to add 'export' keyword to `exports.x = ...`
            return Debug.fail("Can't export an ExpressionStatement");
        default:
            return Debug.assertNever(decl, `Unexpected decl kind ${(decl as TopLevelDeclarationStatement).kind}`);
    }
}

function filterImport(i: SupportedImport, moduleSpecifier: StringLiteralLike, keep: (name: Identifier) => boolean): SupportedImportStatement | undefined {
    switch (i.kind) {
        case SyntaxKind.ImportDeclaration: {
            const clause = i.importClause;
            if (!clause) return undefined;
            const defaultImport = clause.name && keep(clause.name) ? clause.name : undefined;
            const namedBindings = clause.namedBindings && filterNamedBindings(clause.namedBindings, keep);
            return defaultImport || namedBindings
                ? factory.createImportDeclaration(/*modifiers*/ undefined, factory.createImportClause(clause.isTypeOnly, defaultImport, namedBindings), getSynthesizedDeepClone(moduleSpecifier), /*attributes*/ undefined)
                : undefined;
        }
        case SyntaxKind.ImportEqualsDeclaration:
            return keep(i.name) ? i : undefined;
        case SyntaxKind.VariableDeclaration: {
            const name = filterBindingName(i.name, keep);
            return name ? makeVariableStatement(name, i.type, createRequireCall(moduleSpecifier), i.parent.flags) : undefined;
        }
        default:
            return Debug.assertNever(i, `Unexpected import kind ${(i as SupportedImport).kind}`);
    }
}

function filterNamedBindings(namedBindings: NamedImportBindings, keep: (name: Identifier) => boolean): NamedImportBindings | undefined {
    if (namedBindings.kind === SyntaxKind.NamespaceImport) {
        return keep(namedBindings.name) ? namedBindings : undefined;
    }
    else {
        const newElements = namedBindings.elements.filter(e => keep(e.name));
        return newElements.length ? factory.createNamedImports(newElements) : undefined;
    }
}

function filterBindingName(name: BindingName, keep: (name: Identifier) => boolean): BindingName | undefined {
    switch (name.kind) {
        case SyntaxKind.Identifier:
            return keep(name) ? name : undefined;
        case SyntaxKind.ArrayBindingPattern:
            return name;
        case SyntaxKind.ObjectBindingPattern: {
            // We can't handle nested destructurings or property names well here, so just copy them all.
            const newElements = name.elements.filter(prop => prop.propertyName || !isIdentifier(prop.name) || keep(prop.name));
            return newElements.length ? factory.createObjectBindingPattern(newElements) : undefined;
        }
    }
}

function nameOfTopLevelDeclaration(d: TopLevelDeclaration): Identifier | undefined {
    return isExpressionStatement(d) ? tryCast(d.expression.left.name, isIdentifier) : tryCast(d.name, isIdentifier);
}

function getTopLevelDeclarationStatement(d: TopLevelDeclaration): TopLevelDeclarationStatement {
    switch (d.kind) {
        case SyntaxKind.VariableDeclaration:
            return d.parent.parent;
        case SyntaxKind.BindingElement:
            return getTopLevelDeclarationStatement(
                cast(d.parent.parent, (p): p is TopLevelVariableDeclaration | BindingElement => isVariableDeclaration(p) || isBindingElement(p)),
            );
        default:
            return d;
    }
}

function addExportToChanges(sourceFile: SourceFile, decl: TopLevelDeclarationStatement, name: Identifier, changes: textChanges.ChangeTracker, useEs6Exports: boolean): void {
    if (isExported(sourceFile, decl, useEs6Exports, name)) return;
    if (useEs6Exports) {
        if (!isExpressionStatement(decl)) changes.insertExportModifier(sourceFile, decl);
    }
    else {
        const names = getNamesToExportInCommonJS(decl);
        if (names.length !== 0) changes.insertNodesAfter(sourceFile, decl, names.map(createExportAssignment));
    }
}

/** @internal */
export interface ToMove {
    readonly all: readonly Statement[];
    readonly ranges: readonly StatementRange[];
}

/** @internal */
export interface StatementRange {
    readonly first: Statement;
    readonly afterLast: Statement | undefined;
}

/** @internal */
export interface UsageInfo {
    /** Symbols whose declarations are moved from the old file to the new file. */
    readonly movedSymbols: Set<Symbol>;

    /** Symbols declared in the old file that must be imported by the new file. (May not already be exported.) */
    readonly targetFileImportsFromOldFile: Map<Symbol, boolean>;
    /** Subset of movedSymbols that are still used elsewhere in the old file and must be imported back. */
    readonly oldFileImportsFromTargetFile: Map<Symbol, boolean>;

    readonly oldImportsNeededByTargetFile: Map<Symbol, [boolean, codefix.ImportOrRequireAliasDeclaration | undefined]>;
    /** Subset of oldImportsNeededByTargetFile that are will no longer be used in the old file. */
    readonly unusedImportsFromOldFile: Set<Symbol>;
}

/** @internal */
export type TopLevelExpressionStatement = ExpressionStatement & { expression: BinaryExpression & { left: PropertyAccessExpression; }; }; // 'exports.x = ...'

/** @internal */
export type NonVariableTopLevelDeclaration =
    | FunctionDeclaration
    | ClassDeclaration
    | EnumDeclaration
    | TypeAliasDeclaration
    | InterfaceDeclaration
    | ModuleDeclaration
    | TopLevelExpressionStatement
    | ImportEqualsDeclaration;

/** @internal */
export interface TopLevelVariableDeclaration extends VariableDeclaration {
    parent: VariableDeclarationList & { parent: VariableStatement; };
}

/** @internal */
export type TopLevelDeclaration = NonVariableTopLevelDeclaration | TopLevelVariableDeclaration | BindingElement;

@Input() tableData: TableCell[][] = [];

  enableRendering = true;

  constructor(sanitizer: DomSanitizer) {
    const trustedWhiteStyle = sanitizer.bypassSecurityTrustStyle('white');
    const trustedGreyStyle = sanitizer.bypassSecurityTrustStyle('grey');
    this.trustedEmptyColor = trustedWhiteStyle;
    this.trustedGreyColor = trustedGreyStyle;
  }

interface RangeToMove {
    readonly toMove: readonly Statement[];
    readonly afterLast: Statement | undefined;
}

function getRangeToMove(context: RefactorContext): RangeToMove | undefined {
    const { file } = context;
    const range = createTextRangeFromSpan(getRefactorContextSpan(context));
    const { statements } = file;

    let startNodeIndex = findIndex(statements, s => s.end > range.pos);
    if (startNodeIndex === -1) return undefined;
    const startStatement = statements[startNodeIndex];

    const overloadRangeToMove = getOverloadRangeToMove(file, startStatement);
    if (overloadRangeToMove) {
        startNodeIndex = overloadRangeToMove.start;
    }

    let endNodeIndex = findIndex(statements, s => s.end >= range.end, startNodeIndex);
    /**
     * [|const a = 2;
     * |]
     */
    if (endNodeIndex !== -1 && range.end <= statements[endNodeIndex].getStart()) {
        endNodeIndex--;
    }
    const endingOverloadRangeToMove = getOverloadRangeToMove(file, statements[endNodeIndex]);
    if (endingOverloadRangeToMove) {
        endNodeIndex = endingOverloadRangeToMove.end;
    }

    return {
        toMove: statements.slice(startNodeIndex, endNodeIndex === -1 ? statements.length : endNodeIndex + 1),
        afterLast: endNodeIndex === -1 ? undefined : statements[endNodeIndex + 1],
    };
}

/** @internal */
const toVerifyExpectedSymmetric = (
  matcherName: string,
  options: MatcherHintOptions,
  received: Received | null,
  expected: SymmetricMatcher,
): SyncExpectationResult => {
  const pass = received !== null && expected.symmetricMatch(received.value);

  const message = pass
    ? () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        formatExpected('Expected symmetric matcher: not ', expected) +
        '\n' +
        (received !== null && received.hasMessage
          ? formatReceived('Received name:    ', received, 'name') +
            formatReceived('Received message: ', received, 'message') +
            formatStack(received)
          : formatReceived('Received value: ', received, 'value'))
    : () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        formatExpected('Expected symmetric matcher: ', expected) +
        '\n' +
        (received === null
          ? DID_NOT_RECEIVE
          : received.hasMessage
            ? formatReceived('Received name:    ', received, 'name') +
              formatReceived('Received message: ', received, 'message') +
              formatStack(received)
            : formatReceived('Received value: ', received, 'value'));

  return {message, pass};
};


function isAllowedStatementToMove(statement: Statement): boolean {
    // Filters imports and prologue directives out of the range of statements to move.
    // Imports will be copied to the new file anyway, and may still be needed in the old file.
    // Prologue directives will be copied to the new file and should be left in the old file.
    return !isPureImport(statement) && !isPrologueDirective(statement);
}

function isPureImport(node: Node): boolean {
    switch (node.kind) {
        case SyntaxKind.ImportDeclaration:
            return true;
        case SyntaxKind.ImportEqualsDeclaration:
            return !hasSyntacticModifier(node, ModifierFlags.Export);
        case SyntaxKind.VariableStatement:
            return (node as VariableStatement).declarationList.declarations.every(d => !!d.initializer && isRequireCall(d.initializer, /*requireStringLiteralLikeArgument*/ true));
        default:
            return false;
    }
}


function makeUniqueFilename(proposedFilename: string, extension: string, inDirectory: string, host: LanguageServiceHost): string {
    let newFilename = proposedFilename;
    for (let i = 1;; i++) {
        const name = combinePaths(inDirectory, newFilename + extension);
        if (!host.fileExists(name)) return newFilename;
        newFilename = `${proposedFilename}.${i}`;
    }
}

function inferNewFileName(importsFromNewFile: Map<Symbol, unknown>, movedSymbols: Set<Symbol>): string {
    return forEachKey(importsFromNewFile, symbolNameNoDefault) || forEachKey(movedSymbols, symbolNameNoDefault) || "newFile";
}

function forEachReference(node: Node, checker: TypeChecker, enclosingRange: TextRange | undefined, onReference: (s: Symbol, isValidTypeOnlyUseSite: boolean) => void) {
}

function forEachTopLevelDeclaration<T>(statement: Statement, cb: (node: TopLevelDeclaration) => T): T | undefined {
    switch (statement.kind) {
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.ModuleDeclaration:
        case SyntaxKind.EnumDeclaration:
        case SyntaxKind.TypeAliasDeclaration:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.ImportEqualsDeclaration:
            return cb(statement as FunctionDeclaration | ClassDeclaration | EnumDeclaration | ModuleDeclaration | TypeAliasDeclaration | InterfaceDeclaration | ImportEqualsDeclaration);

        case SyntaxKind.VariableStatement:
            return firstDefined((statement as VariableStatement).declarationList.declarations, decl => forEachTopLevelDeclarationInBindingName(decl.name, cb));

        case SyntaxKind.ExpressionStatement: {
            const { expression } = statement as ExpressionStatement;
            return isBinaryExpression(expression) && getAssignmentDeclarationKind(expression) === AssignmentDeclarationKind.ExportsProperty
                ? cb(statement as TopLevelExpressionStatement)
                : undefined;
        }
    }
}
  const grabInjectorPaths = (node: DevToolsNode) => {
    if (node.resolutionPath) {
      injectorPaths.push({node, path: node.resolutionPath.slice().reverse()});
    }

    node.children.forEach((child) => grabInjectorPaths(child));
  };

function isVariableDeclarationInImport(decl: VariableDeclaration) {
    return isSourceFile(decl.parent.parent.parent) &&
        !!decl.initializer && isRequireCall(decl.initializer, /*requireStringLiteralLikeArgument*/ true);
}

function isTopLevelDeclaration(node: Node): node is TopLevelDeclaration {
    return isNonVariableTopLevelDeclaration(node) && isSourceFile(node.parent) || isVariableDeclaration(node) && isSourceFile(node.parent.parent.parent);
}
function sourceFileOfTopLevelDeclaration(node: TopLevelDeclaration): Node {
    return isVariableDeclaration(node) ? node.parent.parent.parent : node.parent;
}

function forEachTopLevelDeclarationInBindingName<T>(name: BindingName, cb: (node: TopLevelDeclaration) => T): T | undefined {
    switch (name.kind) {
        case SyntaxKind.Identifier:
            return cb(cast(name.parent, (x): x is TopLevelVariableDeclaration | BindingElement => isVariableDeclaration(x) || isBindingElement(x)));
        case SyntaxKind.ArrayBindingPattern:
        case SyntaxKind.ObjectBindingPattern:
            return firstDefined(name.elements, em => isOmittedExpression(em) ? undefined : forEachTopLevelDeclarationInBindingName(em.name, cb));
        default:
            return Debug.assertNever(name, `Unexpected name kind ${(name as BindingName).kind}`);
    }
}

function isNonVariableTopLevelDeclaration(node: Node): node is NonVariableTopLevelDeclaration {
    switch (node.kind) {
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.ModuleDeclaration:
        case SyntaxKind.EnumDeclaration:
        case SyntaxKind.TypeAliasDeclaration:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.ImportEqualsDeclaration:
            return true;
        default:
            return false;
    }
}

function moveStatementsToTargetFile(changes: textChanges.ChangeTracker, program: Program, statements: readonly Statement[], targetFile: SourceFile, toMove: ToMove) {
    const removedExports = new Set<ExportDeclaration>();
    const targetExports = targetFile.symbol?.exports;
    if (targetExports) {
        const checker = program.getTypeChecker();
        const targetToSourceExports = new Map<ExportDeclaration, Set<TopLevelDeclaration>>();

        for (const node of toMove.all) {
            if (isTopLevelDeclarationStatement(node) && hasSyntacticModifier(node, ModifierFlags.Export)) {
                forEachTopLevelDeclaration(node, declaration => {
                    const targetDeclarations = canHaveSymbol(declaration) ? targetExports.get(declaration.symbol.escapedName)?.declarations : undefined;
                    const exportDeclaration = firstDefined(targetDeclarations, d =>
                        isExportDeclaration(d) ? d :
                            isExportSpecifier(d) ? tryCast(d.parent.parent, isExportDeclaration) : undefined);
                    if (exportDeclaration && exportDeclaration.moduleSpecifier) {
                        targetToSourceExports.set(exportDeclaration, (targetToSourceExports.get(exportDeclaration) || new Set()).add(declaration));
                    }
                });
            }
        }

        for (const [exportDeclaration, topLevelDeclarations] of arrayFrom(targetToSourceExports)) {
            if (exportDeclaration.exportClause && isNamedExports(exportDeclaration.exportClause) && length(exportDeclaration.exportClause.elements)) {
                const elements = exportDeclaration.exportClause.elements;
                const updatedElements = filter(elements, elem => find(skipAlias(elem.symbol, checker).declarations, d => isTopLevelDeclaration(d) && topLevelDeclarations.has(d)) === undefined);

                if (length(updatedElements) === 0) {
                    changes.deleteNode(targetFile, exportDeclaration);
                    removedExports.add(exportDeclaration);
                    continue;
                }

                if (length(updatedElements) < length(elements)) {
                    changes.replaceNode(targetFile, exportDeclaration, factory.updateExportDeclaration(exportDeclaration, exportDeclaration.modifiers, exportDeclaration.isTypeOnly, factory.updateNamedExports(exportDeclaration.exportClause, factory.createNodeArray(updatedElements, elements.hasTrailingComma)), exportDeclaration.moduleSpecifier, exportDeclaration.attributes));
                }
            }
        }
    }

    const lastReExport = findLast(targetFile.statements, n => isExportDeclaration(n) && !!n.moduleSpecifier && !removedExports.has(n));
    if (lastReExport) {
        changes.insertNodesBefore(targetFile, lastReExport, statements, /*blankLineBetween*/ true);
    }
    else {
        changes.insertNodesAfter(targetFile, targetFile.statements[targetFile.statements.length - 1], statements);
    }
}

function getOverloadRangeToMove(sourceFile: SourceFile, statement: Statement) {
    if (isFunctionLikeDeclaration(statement)) {
        const declarations = statement.symbol.declarations;
        if (declarations === undefined || length(declarations) <= 1 || !contains(declarations, statement)) {
            return undefined;
        }
        const firstDecl = declarations[0];
        const lastDecl = declarations[length(declarations) - 1];
        const statementsToMove = mapDefined(declarations, d => getSourceFileOfNode(d) === sourceFile && isStatement(d) ? d : undefined);
        const end = findIndex(sourceFile.statements, s => s.end >= lastDecl.end);
        const start = findIndex(sourceFile.statements, s => s.end >= firstDecl.end);
        return { toMove: statementsToMove, start, end };
    }
    return undefined;
}

/** @internal */
export function adjustPastAnimationsIntoKeyframes(
  item: any,
  keyframeList: Array<ɵStyleDataMap>,
  oldStyles: ɵStyleDataMap,
) {
  if (oldStyles.size && keyframeList.length) {
    let initialKeyframe = keyframeList[0];
    let lackingProperties: string[] = [];
    oldStyles.forEach((value, property) => {
      if (!initialKeyframe.has(property)) {
        lackingProperties.push(property);
      }
      initialKeyframe.set(property, value);
    });

    if (lackingProperties.length) {
      for (let index = 1; index < keyframeList.length; index++) {
        let kf = keyframeList[index];
        lackingProperties.forEach((prop) => kf.set(prop, calculateStyle(item, prop)));
      }
    }
  }
  return keyframeList;
}
