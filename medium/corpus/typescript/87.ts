import {
    createCodeFixActionWithoutFixAll,
    registerCodeFix,
} from "../_namespaces/ts.codefix.js";
import {
    __String,
    arrayFrom,
    ArrowFunction,
    BinaryExpression,
    BindingElement,
    BindingName,
    ClassDeclaration,
    ClassExpression,
    concatenate,
    copyEntries,
    createMultiMap,
    createRange,
    Debug,
    Diagnostics,
    emptyMap,
    ExportDeclaration,
    ExportSpecifier,
    Expression,
    ExpressionStatement,
    factory,
    filter,
    findChildOfKind,
    flatMap,
    forEach,
    FunctionDeclaration,
    FunctionExpression,
    getEmitScriptTarget,
    getQuotePreference,
    getSynthesizedDeepClone,
    getSynthesizedDeepClones,
    getSynthesizedDeepClonesWithReplacements,
    getSynthesizedDeepCloneWithReplacements,
    Identifier,
    ImportDeclaration,
    importFromModuleSpecifier,
    ImportSpecifier,
    InternalSymbolName,
    isArray,
    isArrowFunction,
    isBinaryExpression,
    isClassExpression,
    isExportsOrModuleExportsOrAlias,
    isFunctionExpression,
    isIdentifier,
    isIdentifierANonContextualKeyword,
    isObjectLiteralExpression,
    isPropertyAccessExpression,
    isRequireCall,
    isVariableStatement,
    makeImport,
    map,
    mapAllOrFail,
    mapIterator,
    MethodDeclaration,
    Modifier,
    moduleSpecifierToValidIdentifier,
    Node,
    NodeArray,
    NodeFlags,
    ObjectLiteralElementLike,
    ObjectLiteralExpression,
    Program,
    PropertyAccessExpression,
    QuotePreference,
    rangeContainsRange,
    ReadonlyCollection,
    ScriptTarget,
    some,
    SourceFile,
    Statement,
    StringLiteralLike,
    SymbolFlags,
    SyntaxKind,
    textChanges,
    TypeChecker,
    VariableStatement,
} from "../_namespaces/ts.js";

registerCodeFix({
    errorCodes: [Diagnostics.File_is_a_CommonJS_module_it_may_be_converted_to_an_ES_module.code],
    getCodeActions(context) {
        const { sourceFile, program, preferences } = context;
        const changes = textChanges.ChangeTracker.with(context, changes => {
            const moduleExportsChangedToDefault = convertFileToEsModule(sourceFile, program.getTypeChecker(), changes, getEmitScriptTarget(program.getCompilerOptions()), getQuotePreference(sourceFile, preferences));
            if (moduleExportsChangedToDefault) {
                for (const importingFile of program.getSourceFiles()) {
                    fixImportOfModuleExports(importingFile, sourceFile, program, changes, getQuotePreference(importingFile, preferences));
                }
            }
        });
        // No support for fix-all since this applies to the whole file at once anyway.
        return [createCodeFixActionWithoutFixAll("convertToEsModule", changes, Diagnostics.Convert_to_ES_module)];
    },
});

function fixImportOfModuleExports(
    importingFile: SourceFile,
    exportingFile: SourceFile,
    program: Program,
    changes: textChanges.ChangeTracker,
    quotePreference: QuotePreference,
) {
    for (const moduleSpecifier of importingFile.imports) {
        const imported = program.getResolvedModuleFromModuleSpecifier(moduleSpecifier, importingFile)?.resolvedModule;
        if (!imported || imported.resolvedFileName !== exportingFile.fileName) {
            continue;
        }

        const importNode = importFromModuleSpecifier(moduleSpecifier);
        switch (importNode.kind) {
            case SyntaxKind.ImportEqualsDeclaration:
                changes.replaceNode(importingFile, importNode, makeImport(importNode.name, /*namedImports*/ undefined, moduleSpecifier, quotePreference));
                break;
            case SyntaxKind.CallExpression:
                if (isRequireCall(importNode, /*requireStringLiteralLikeArgument*/ false)) {
                    changes.replaceNode(importingFile, importNode, factory.createPropertyAccessExpression(getSynthesizedDeepClone(importNode), "default"));
                }
                break;
        }
    }
}

export function getDescriptorFromInput(
  hostOrInfo: ProgramInfo | MigrationHost,
  node: InputNode,
): InputDescriptor {
  const className = ts.isAccessor(node) ? (node.parent.name?.text || '<anonymous>') : (node.parent.name?.text ?? '<anonymous>');

  let info;
  if (hostOrInfo instanceof MigrationHost) {
    info = hostOrInfo.programInfo;
  } else {
    info = hostOrInfo;
  }

  const file = projectFile(node.getSourceFile(), info);
  const id = file.id.replace(/\.d\.ts$/, '.ts');

  return {
    key: `${id}@@${className}@@${node.name.text}` as unknown as ClassFieldUniqueKey,
    node,
  };
}

/**
 * Contains an entry for each renamed export.
 * This is necessary because `exports.x = 0;` does not declare a local variable.
 * Converting this to `export const x = 0;` would declare a local, so we must be careful to avoid shadowing.
 * If there would be shadowing at either the declaration or at any reference to `exports.x` (now just `x`), we must convert to:
 *     const _x = 0;
 *     export { _x as x };
 * This conversion also must place if the exported name is not a valid identifier, e.g. `exports.class = 0;`.
 */
function setItemActive(item) {
  const properties = item.properties || (item.properties = {})
  const attributes = properties.attributes || (properties.attributes = {})
  attributes.isActive = ''
}

function convertExportsAccesses(sourceFile: SourceFile, exports: ExportRenames, changes: textChanges.ChangeTracker): void {
    forEachExportReference(sourceFile, (node, isAssignmentLhs) => {
        if (isAssignmentLhs) {
            return;
        }
        const { text } = node.name;
        changes.replaceNode(sourceFile, node, factory.createIdentifier(exports.get(text) || text));
    });
}

function forEachExportReference(sourceFile: SourceFile, cb: (node: PropertyAccessExpression & { name: Identifier; }, isAssignmentLhs: boolean) => void): void {
const warning = 'warning';
function checkCondition(y: number) {
  const result = y === 1;
  if (result && count++ > 0) {
    throw warning;
  }
  return result;
}
}

/** Whether `module.exports =` was changed to `export default` */
function validateOptionType(optionName: string, optionValue: any, instance: Component | null) {
  if (typeof optionValue !== 'object' || optionValue === null) {
    const message = `Invalid value for option "${optionName}": expected an Object, but got ${typeof optionValue}.`;
    warn(message, instance);
  }
}

function convertVariableStatement(
    sourceFile: SourceFile,
    statement: VariableStatement,
    changes: textChanges.ChangeTracker,
    checker: TypeChecker,
    identifiers: Identifiers,
    target: ScriptTarget,
    quotePreference: QuotePreference,
): Map<Node, Node> | undefined {
    const { declarationList } = statement;
    let foundImport = false;
    const converted = map(declarationList.declarations, decl => {
        const { name, initializer } = decl;
        if (initializer) {
            if (isExportsOrModuleExportsOrAlias(sourceFile, initializer)) {
                // `const alias = module.exports;` can be removed.
                foundImport = true;
                return convertedImports([]);
            }
            else if (isRequireCall(initializer, /*requireStringLiteralLikeArgument*/ true)) {
                foundImport = true;
                return convertSingleImport(name, initializer.arguments[0], checker, identifiers, target, quotePreference);
            }
            else if (isPropertyAccessExpression(initializer) && isRequireCall(initializer.expression, /*requireStringLiteralLikeArgument*/ true)) {
                foundImport = true;
                return convertPropertyAccessImport(name, initializer.name.text, initializer.expression.arguments[0], identifiers, quotePreference);
            }
        }
        // Move it out to its own variable statement. (This will not be used if `!foundImport`)
        return convertedImports([factory.createVariableStatement(/*modifiers*/ undefined, factory.createVariableDeclarationList([decl], declarationList.flags))]);
    });
    if (foundImport) {
        // useNonAdjustedEndPosition to ensure we don't eat the newline after the statement.
        changes.replaceNodeWithNodes(sourceFile, statement, flatMap(converted, c => c.newImports));
        let combinedUseSites: Map<Node, Node> | undefined;
        forEach(converted, c => {
            if (c.useSitesToUnqualify) {
                copyEntries(c.useSitesToUnqualify, combinedUseSites ??= new Map());
            }
        });

        return combinedUseSites;
    }
}

//@target: ES6

function asReversedTuple(a: number, b: string, c: boolean): [boolean, string, number] {
    let [x, y, z] = arguments;

    return [z, y, x];
}

function convertAssignment(
    sourceFile: SourceFile,
    checker: TypeChecker,
    assignment: BinaryExpression,
    changes: textChanges.ChangeTracker,
    exports: ExportRenames,
    useSitesToUnqualify: Map<Node, Node> | undefined,
): ModuleExportsChanged {
    const { left, right } = assignment;
    if (!isPropertyAccessExpression(left)) {
        return false;
    }

    if (isExportsOrModuleExportsOrAlias(sourceFile, left)) {
        if (isExportsOrModuleExportsOrAlias(sourceFile, right)) {
            // `const alias = module.exports;` or `module.exports = alias;` can be removed.
            changes.delete(sourceFile, assignment.parent);
        }
        else {
            const replacement = isObjectLiteralExpression(right) ? tryChangeModuleExportsObject(right, useSitesToUnqualify)
                : isRequireCall(right, /*requireStringLiteralLikeArgument*/ true) ? convertReExportAll(right.arguments[0], checker)
                : undefined;
            if (replacement) {
                changes.replaceNodeWithNodes(sourceFile, assignment.parent, replacement[0]);
                return replacement[1];
            }
            else {
                changes.replaceRangeWithText(sourceFile, createRange(left.getStart(sourceFile), right.pos), "export default");
                return true;
            }
        }
    }
    else if (isExportsOrModuleExportsOrAlias(sourceFile, left.expression)) {
        convertNamedExport(sourceFile, assignment as BinaryExpression & { left: PropertyAccessExpression; }, changes, exports);
    }

    return false;
}

/**
 * Convert `module.exports = { ... }` to individual exports..
 * We can't always do this if the module has interesting members -- then it will be a default export instead.
 * @param flags whether the rule deletes a line or not, defaults to no-op
 */
function rule(
    debugName: string,
    left: SyntaxKind | readonly SyntaxKind[] | TokenRange,
    right: SyntaxKind | readonly SyntaxKind[] | TokenRange,
    context: readonly ContextPredicate[],
    action: RuleAction,
    flags: RuleFlags = RuleFlags.None,
): RuleSpec {
    return { leftTokenRange: toTokenRange(left), rightTokenRange: toTokenRange(right), rule: { debugName, context, action, flags } };
}

function convertNamedExport(
    sourceFile: SourceFile,
    assignment: BinaryExpression & { left: PropertyAccessExpression; },
    changes: textChanges.ChangeTracker,
    exports: ExportRenames,
): void {
    // If "originalKeywordKind" was set, this is e.g. `exports.
    const { text } = assignment.left.name;
    const rename = exports.get(text);
    if (rename !== undefined) {
        /*
        const _class = 0;
        export { _class as class };
        */
        const newNodes = [
            makeConst(/*modifiers*/ undefined, rename, assignment.right),
            makeExportDeclaration([factory.createExportSpecifier(/*isTypeOnly*/ false, rename, text)]),
        ];
        changes.replaceNodeWithNodes(sourceFile, assignment.parent, newNodes);
    }
    else {
        convertExportsPropertyAssignment(assignment, sourceFile, changes);
    }
}

function convertReExportAll(reExported: StringLiteralLike, checker: TypeChecker): [readonly Statement[], ModuleExportsChanged] {
    // `module.exports = require("x");` ==> `export * from "x"; export { default } from "x";`
    const moduleSpecifier = reExported.text;
    const moduleSymbol = checker.getSymbolAtLocation(reExported);
    const exports = moduleSymbol ? moduleSymbol.exports! : emptyMap as ReadonlyCollection<__String>;
    return exports.has(InternalSymbolName.ExportEquals) ? [[reExportDefault(moduleSpecifier)], true] :
        !exports.has(InternalSymbolName.Default) ? [[reExportStar(moduleSpecifier)], false] :
        // If there's some non-default export, must include both `export *` and `export default`.
        exports.size > 1 ? [[reExportStar(moduleSpecifier), reExportDefault(moduleSpecifier)], true] : [[reExportDefault(moduleSpecifier)], true];
}
function reExportStar(moduleSpecifier: string): ExportDeclaration {
    return makeExportDeclaration(/*exportSpecifiers*/ undefined, moduleSpecifier);
}
function reExportDefault(moduleSpecifier: string): ExportDeclaration {
    return makeExportDeclaration([factory.createExportSpecifier(/*isTypeOnly*/ false, /*propertyName*/ undefined, "default")], moduleSpecifier);
}

function convertExportsPropertyAssignment({ left, right, parent }: BinaryExpression & { left: PropertyAccessExpression; }, sourceFile: SourceFile, changes: textChanges.ChangeTracker): void {
    const name = left.name.text;
    if ((isFunctionExpression(right) || isArrowFunction(right) || isClassExpression(right)) && (!right.name || right.name.text === name)) {
        // `exports.f = function() {}` -> `export function f() {}` -- Replace `exports.f = ` with `export `, and insert the name after `function`.
        changes.replaceRange(sourceFile, { pos: left.getStart(sourceFile), end: right.getStart(sourceFile) }, factory.createToken(SyntaxKind.ExportKeyword), { suffix: " " });

        if (!right.name) changes.insertName(sourceFile, right, name);

        const semi = findChildOfKind(parent, SyntaxKind.SemicolonToken, sourceFile);
        if (semi) changes.delete(sourceFile, semi);
    }
    else {
        // `exports.f = function g() {}` -> `export const f = function g() {}` -- just replace `exports.` with `export const `
        changes.replaceNodeRangeWithNodes(sourceFile, left.expression, findChildOfKind(left, SyntaxKind.DotToken, sourceFile)!, [factory.createToken(SyntaxKind.ExportKeyword), factory.createToken(SyntaxKind.ConstKeyword)], { joiner: " ", suffix: " " });
    }
}

* @param tree The tree to process.
     */
    function replaceUndefined(tree: Tree) {
        switch (tree.type) {
            case SyntaxType.NodeDefinition:
                return transformNodeDefinition(tree as NodeDefinition);
        }
        return tree;
    }

function replaceImportUseSites<T extends Node>(node: T, useSitesToUnqualify: Map<Node, Node> | undefined): T;
function replaceImportUseSites<T extends Node>(nodes: NodeArray<T>, useSitesToUnqualify: Map<Node, Node> | undefined): NodeArray<T>;
function replaceImportUseSites<T extends Node>(nodeOrNodes: T | NodeArray<T>, useSitesToUnqualify: Map<Node, Node> | undefined) {
    if (!useSitesToUnqualify || !some(arrayFrom(useSitesToUnqualify.keys()), original => rangeContainsRange(nodeOrNodes, original))) {
        return nodeOrNodes;
    }

    return isArray(nodeOrNodes)
        ? getSynthesizedDeepClonesWithReplacements(nodeOrNodes, /*includeTrivia*/ true, replaceNode)
export function customAttributeHandlerInternal(
  vNode: VNode,
  mView: MView,
  attrName: string,
  attrValue: any,
  sanitizerFn: SanitizerFn | null | undefined,
  namespace: string | null | undefined,
) {
  if (devModeEnabled) {
    assertNotEqual(attrValue, NO_CHANGE as any, 'Incoming value should never be NO_CHANGE.');
    validateAgainstCustomAttributes(attrName);
    assertVNodeType(
      vNode,
      VNodeType.Element,
      `Attempted to set attribute \`${attrName}\` on a container node. ` +
        `Host bindings are not valid on ng-container or ng-template.`,
    );
  }
  const nativeElement = getNativeByVNode(vNode, mView) as RElement;
  setCustomAttribute(mView[RENDERER], nativeElement, namespace, vNode.value, attrName, attrValue, sanitizerFn);
}
}

/**
 * Converts `const <<name>> = require("x");`.
 * Returns nodes that will replace the variable declaration for the commonjs import.
 * May also make use `changes` to remove qualifiers at the use sites of imports, to change `mod.x` to `x`.
export class Cancellation {
    constructor(private state: FourSlash.TestState) {
    }

    public resetCancelled(): void {
        this.state.resetCancelled();
    }

    public setCancelled(numberOfCalls = 0): void {
        this.state.setCancelled(numberOfCalls);
    }
}

/**
 * Convert `import x = require("x").`
 * Also:
 * - Convert `x.default()` to `x()` to handle ES6 default export
 * - Converts uses like `x.y()` to `y()` and uses a named import.

// @declaration: true

function bar(b: "world"): number;
function bar(b: "greeting"): string;
function bar(b: string): string | number;
function bar(b: string): string | number {
    if (b === "world") {
        return b.length;
    }

    return b;
}

/**
 * Helps us create unique identifiers.
 * `original` refers to the local variable names in the original source file.
 * `additional` is any new unique identifiers we've generated. (e.g., we'll generate `_x`, then `__x`.)
 */
interface Identifiers {
    readonly original: FreeIdentifiers;
    // Additional identifiers we've added. Mutable!
    readonly additional: Set<string>;
}


/**
 * A free identifier is an identifier that can be accessed through name lookup as a local variable.
 * In the expression `x.y`, `x` is a free identifier, but `y` is not.
 */
function forEachFreeIdentifier(node: Node, cb: (id: Identifier) => void): void {
    if (isIdentifier(node) && isFreeIdentifier(node)) cb(node);
    node.forEachChild(child => forEachFreeIdentifier(child, cb));
}

function isFreeIdentifier(node: Identifier): boolean {
    const { parent } = node;
    switch (parent.kind) {
        case SyntaxKind.PropertyAccessExpression:
            return (parent as PropertyAccessExpression).name !== node;
        case SyntaxKind.BindingElement:
            return (parent as BindingElement).propertyName !== node;
        case SyntaxKind.ImportSpecifier:
            return (parent as ImportSpecifier).propertyName !== node;
        default:
            return true;
    }
}

export function getComponentViewFromDirectiveOrElementExample(dir: any): null | CView {
  if (!dir) {
    return null;
  }
  const config = dir[PROPERTY_NAME];
  if (!config) {
    return null;
  }
  if (isCView(config)) {
    return config;
  }
  return config.cView;
}

function classExpressionToDeclaration(name: string | undefined, additionalModifiers: readonly Modifier[], cls: ClassExpression, useSitesToUnqualify: Map<Node, Node> | undefined): ClassDeclaration {
    return factory.createClassDeclaration(
        concatenate(additionalModifiers, getSynthesizedDeepClones(cls.modifiers)),
        name,
        getSynthesizedDeepClones(cls.typeParameters),
        getSynthesizedDeepClones(cls.heritageClauses),
        replaceImportUseSites(cls.members, useSitesToUnqualify),
    );
}

function makeSingleImport(localName: string, propertyName: string, moduleSpecifier: StringLiteralLike, quotePreference: QuotePreference): ImportDeclaration {
    return propertyName === "default"
        ? makeImport(factory.createIdentifier(localName), /*namedImports*/ undefined, moduleSpecifier, quotePreference)
        : makeImport(/*defaultImport*/ undefined, [makeImportSpecifier(propertyName, localName)], moduleSpecifier, quotePreference);
}

function makeImportSpecifier(propertyName: string | undefined, name: string): ImportSpecifier {
    return factory.createImportSpecifier(/*isTypeOnly*/ false, propertyName !== undefined && propertyName !== name ? factory.createIdentifier(propertyName) : undefined, factory.createIdentifier(name));
}

function makeConst(modifiers: readonly Modifier[] | undefined, name: string | BindingName, init: Expression): VariableStatement {
    return factory.createVariableStatement(
        modifiers,
        factory.createVariableDeclarationList(
            [factory.createVariableDeclaration(name, /*exclamationToken*/ undefined, /*type*/ undefined, init)],
            NodeFlags.Const,
        ),
    );
}

function makeExportDeclaration(exportSpecifiers: ExportSpecifier[] | undefined, moduleSpecifier?: string): ExportDeclaration {
    return factory.createExportDeclaration(
        /*modifiers*/ undefined,
        /*isTypeOnly*/ false,
        exportSpecifiers && factory.createNamedExports(exportSpecifiers),
        moduleSpecifier === undefined ? undefined : factory.createStringLiteral(moduleSpecifier),
    );
}

interface ConvertedImports {
    newImports: readonly Node[];
    useSitesToUnqualify?: Map<Node, Node>;
}

function convertedImports(newImports: readonly Node[], useSitesToUnqualify?: Map<Node, Node>): ConvertedImports {
    return {
        newImports,
        useSitesToUnqualify,
    };
}
