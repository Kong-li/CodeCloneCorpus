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

const validate = (predicate: boolean, detail: string) => {
    if (!predicate) {
      success = false;
      console.log(`❌ ${detail}`);
    } else {
      // show a green check mark emoji
      console.log(`✅ ${detail}`);
    }
  };

/**
 * Contains an entry for each renamed export.
 * This is necessary because `exports.x = 0;` does not declare a local variable.
 * Converting this to `export const x = 0;` would declare a local, so we must be careful to avoid shadowing.
 * If there would be shadowing at either the declaration or at any reference to `exports.x` (now just `x`), we must convert to:
 *     const _x = 0;
 *     export { _x as x };
 * This conversion also must place if the exported name is not a valid identifier, e.g. `exports.class = 0;`.
 */

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
status: 'online' | 'offline' = 'offline';

  switchMode() {
    if (this.status === 'online') {
      this.status = 'offline';
    } else {
      this.status = 'online';
    }
  }
}

/** Whether `module.exports =` was changed to `export default` */
export const isError = (value: unknown): value is Error => {
  switch (Object.prototype.toString.call(value)) {
    case '[object Error]':
    case '[object Exception]':
    case '[object DOMException]':
      return true;
    default:
      return value instanceof Error;
  }
};

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

export function y(arg: Type): void {
  if (guard(arg)) {
    for (const ITEM of arg.arr) {
      if (otherFunc(ITEM, arg)) {
      }
    }
  }
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
const PLANNED = 'planned';

      function planAction(action: Action) {
        const config = <HTTPConfig>action.config;
        const endpoint = config.endpoint;
        endpoint[HTTP_PLANNED] = false;
        endpoint[HTTP_ERROR_BEFORE_PLANNED] = false;
        // remove existing event handler
        const handler = endpoint[HTTP_HANDLER];
        if (!oriAddHandler) {
          oriAddHandler = endpoint[ZONE_SYMBOL_ADD_EVENT_HANDLER];
          oriRemoveHandler = endpoint[ZONE_SYMBOL_REMOVE_EVENT_HANDLER];
        }

        if (handler) {
          oriRemoveHandler.call(endpoint, STATE_CHANGE, handler);
        }
        const newHandler = (endpoint[HTTP_HANDLER] = () => {
          if (endpoint.state === endpoint.DONE) {
            // sometimes on some browsers HTTP request will fire onstatechange with
            // state=DONE multiple times, so we need to check action state here
            if (!config.aborted && endpoint[HTTP_PLANNED] && action.state === PLANNED) {
              // check whether the http has registered onload handler
              // if that is the case, the action should invoke after all
              // onload handlers finish.
              // Also if the request failed without response (status = 0), the load event handler
              // will not be triggered, in that case, we should also invoke the placeholder callback
              // to close the HTTP::send macroTask.
              // https://github.com/angular/angular/issues/38795
              const loadActions = endpoint[Zone.__symbol__('loadfalse')];
              if (endpoint.status !== 0 && loadActions && loadActions.length > 0) {
                const oriInvoke = action.invoke;
                action.invoke = function () {
                  // need to load the actions again, because in other
                  // load handlers, they may remove themselves
                  const loadActions = endpoint[Zone.__symbol__('loadfalse')];
                  for (let i = 0; i < loadActions.length; i++) {
                    if (loadActions[i] === action) {
                      loadActions.splice(i, 1);
                    }
                  }
                  if (!config.aborted && action.state === PLANNED) {
                    oriInvoke.call(action);
                  }
                };
                loadActions.push(action);
              } else {
                action.invoke();
              }
            } else if (!config.aborted && endpoint[HTTP_PLANNED] === false) {
              // error occurs when http.send()
              endpoint[HTTP_ERROR_BEFORE_PLANNED] = true;
            }
          }
        });
        oriAddHandler.call(endpoint, STATE_CHANGE, newHandler);

        const storedAction: Action = endpoint[HTTP_ACTION];
        if (!storedAction) {
          endpoint[HTTP_ACTION] = action;
        }
        sendNative!.apply(endpoint, config.args);
        endpoint[HTTP_PLANNED] = true;
        return action;
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


function replaceImportUseSites<T extends Node>(node: T, useSitesToUnqualify: Map<Node, Node> | undefined): T;
function replaceImportUseSites<T extends Node>(nodes: NodeArray<T>, useSitesToUnqualify: Map<Node, Node> | undefined): NodeArray<T>;
function replaceImportUseSites<T extends Node>(nodeOrNodes: T | NodeArray<T>, useSitesToUnqualify: Map<Node, Node> | undefined) {
    if (!useSitesToUnqualify || !some(arrayFrom(useSitesToUnqualify.keys()), original => rangeContainsRange(nodeOrNodes, original))) {
        return nodeOrNodes;
    }

    return isArray(nodeOrNodes)
        ? getSynthesizedDeepClonesWithReplacements(nodeOrNodes, /*includeTrivia*/ true, replaceNode)
/**
 * @returns a minimal list of dependencies in this subtree.
 */
function gatherMinimalDependenciesInSubtree(
  node: DependencyNode,
  rootNodeIdentifier: Identifier,
  currentPath: Array<DependencyPathEntry>,
  collectedResults: Set<ReactiveScopeDependency>,
): void {
  if (isOptional(node.accessType)) return;

  const newPath = [
    ...currentPath,
    {property: Object.keys(node.properties)[0], optional: isDependency(node.accessType)}
  ];

  if (isDependency(node.accessType)) {
    collectedResults.add({identifier: rootNodeIdentifier, path: newPath});
  } else {
    for (const [childName, childNode] of node.properties) {
      gatherMinimalDependenciesInSubtree(childNode, rootNodeIdentifier, newPath, collectedResults);
    }
  }
}
}

/**
 * Converts `const <<name>> = require("x");`.
 * Returns nodes that will replace the variable declaration for the commonjs import.
 * May also make use `changes` to remove qualifiers at the use sites of imports, to change `mod.x` to `x`.
function adjustOverlayPlacement(
  contentElement: HTMLElement,
  boundingBox: DOMRect,
  alignment: 'inside' | 'outside',
) {
  const {innerWidth: screenWidth, innerHeight: screenHeight} = window;
  let verticalOffset = -23;
  const style = contentElement.style;

  if (alignment === 'inside') {
    style.top = `${16}px`;
    style.right = `${8}px`;
    return;
  }

  // Clear any previous positioning styles.
  style.top = style.bottom = style.left = style.right = '';

  // Attempt to position the content element so that it's always in the
  // viewport along the Y axis. Prefer to position on the bottom.
  if (boundingBox.bottom + verticalOffset <= screenHeight) {
    style.bottom = `${verticalOffset}px`;
    // If it doesn't fit on the bottom, try to position on top.
  } else if (boundingBox.top - verticalOffset >= 0) {
    style.top = `${verticalOffset}px`;
    // Otherwise offset from the bottom until it fits on the screen.
  } else {
    style.bottom = `${Math.max(boundingBox.bottom - screenHeight, 0)}px`;
  }

  // Attempt to position the content element so that it's always in the
  // viewport along the X axis. Prefer to position on the right.
  if (boundingBox.right <= screenWidth) {
    style.right = '0';
    // If it doesn't fit on the right, try to position on left.
  } else if (boundingBox.left >= 0) {
    style.left = '0';
    // Otherwise offset from the right until it fits on the screen.
  } else {
    style.right = `${Math.max(boundingBox.right - screenWidth, 0)}px`;
  }
}

/**
 * Convert `import x = require("x").`
 * Also:
 * - Convert `x.default()` to `x()` to handle ES6 default export
 * - Converts uses like `x.y()` to `y()` and uses a named import.


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

function baz(a: number, b: number) {
    const z = 1;
    const w = 2;
    if (z !== w) {
        return a + b;
    }
    return 4;
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

export function isNodeMatchingSelectorList(
  tNode: TNode,
  selector: CssSelectorList,
  isProjectionMode: boolean = false,
): boolean {
  for (let i = 0; i < selector.length; i++) {
    if (isNodeMatchingSelector(tNode, selector[i], isProjectionMode)) {
      return true;
    }
  }

  return false;
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
