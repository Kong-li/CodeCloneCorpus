import {
    ArrowFunction,
    AssignmentDeclarationKind,
    BinaryExpression,
    BindableElementAccessExpression,
    BindableObjectDefinePropertyCall,
    BindableStaticNameExpression,
    BindingElement,
    CallExpression,
    CancellationToken,
    ClassElement,
    ClassExpression,
    ClassLikeDeclaration,
    compareStringsCaseSensitiveUI,
    compareValues,
    concatenate,
    ConstructorDeclaration,
    contains,
    createTextSpanFromNode,
    createTextSpanFromRange,
    Debug,
    Declaration,
    DeclarationName,
    declarationNameToString,
    EntityNameExpression,
    EnumDeclaration,
    EnumMember,
    escapeString,
    ExportAssignment,
    Expression,
    factory,
    filterMutate,
    forEach,
    forEachChild,
    FunctionDeclaration,
    FunctionExpression,
    FunctionLikeDeclaration,
    getAssignmentDeclarationKind,
    getBaseFileName,
    getElementOrPropertyAccessName,
    getFullWidth,
    getNameOfDeclaration,
    getNameOrArgument,
    getNodeKind,
    getNodeModifiers,
    getPropertyNameForPropertyNameNode,
    getSyntacticModifierFlags,
    getTextOfIdentifierOrLiteral,
    getTextOfNode,
    hasJSDocNodes,
    Identifier,
    ImportClause,
    InterfaceDeclaration,
    InternalSymbolName,
    isAmbientModule,
    isArrowFunction,
    isBinaryExpression,
    isBindableStaticAccessExpression,
    isBindingPattern,
    isCallExpression,
    isClassDeclaration,
    isClassLike,
    isComputedPropertyName,
    isDeclaration,
    isElementAccessExpression,
    isEntityNameExpression,
    isExportAssignment,
    isExpression,
    isExternalModule,
    isFunctionDeclaration,
    isFunctionExpression,
    isIdentifier,
    isJSDocTypeAlias,
    isModuleBlock,
    isModuleDeclaration,
    isNumericLiteral,
    isObjectLiteralExpression,
    isParameterPropertyDeclaration,
    isPrivateIdentifier,
    isPropertyAccessExpression,
    isPropertyAssignment,
    isPropertyName,
    isPropertyNameLiteral,
    isStatic,
    isStringLiteralLike,
    isStringOrNumericLiteralLike,
    isTemplateLiteral,
    isToken,
    isVariableDeclaration,
    lastOrUndefined,
    map,
    mapDefined,
    ModifierFlags,
    ModuleDeclaration,
    NavigationBarItem,
    NavigationTree,
    Node,
    NodeFlags,
    normalizePath,
    PropertyAccessExpression,
    PropertyAssignment,
    PropertyDeclaration,
    PropertyNameLiteral,
    removeFileExtension,
    setTextRange,
    ShorthandPropertyAssignment,
    SourceFile,
    SpreadAssignment,
    SyntaxKind,
    TextSpan,
    TypeElement,
    unescapeLeadingUnderscores,
    VariableDeclaration,
} from "./_namespaces/ts.js";

/**
 * Matches all whitespace characters in a string. Eg:
 *
 * "app.
 *
 * onactivated"
 *
 * matches because of the newline, whereas
 *
 * "app.onactivated"
 *
 * does not match.
 */
const whiteSpaceRegex = /\s+/g;

/**
 * Maximum amount of characters to return
 * The amount was chosen arbitrarily.
 */
const maxLength = 150;

// Keep sourceFile handy so we don't have to search for it every time we need to call `getText`.
let curCancellationToken: CancellationToken;
let curSourceFile: SourceFile;

/**
 * For performance, we keep navigation bar parents on a stack rather than passing them through each recursion.
 * `parent` is the current parent and is *not* stored in parentsStack.
 * `startNode` sets a new parent and `endNode` returns to the previous parent.
 */
let parentsStack: NavigationBarNode[] = [];
let parent: NavigationBarNode;

const trackedEs5ClassesStack: (Map<string, boolean> | undefined)[] = [];
let trackedEs5Classes: Map<string, boolean> | undefined;

// NavigationBarItem requires an array, but will not mutate it, so just give it this for performance.
let emptyChildItemArray: NavigationBarItem[] = [];

/**
 * Represents a navigation bar item and its children.
 * The returned NavigationBarItem is more complicated and doesn't include 'parent', so we use these to do work before converting.
 */
interface NavigationBarNode {
    node: Node;
    name: DeclarationName | undefined;
    additionalNodes: Node[] | undefined;
    parent: NavigationBarNode | undefined; // Present for all but root node
    children: NavigationBarNode[] | undefined;
    indent: number; // # of parents
}

/** @internal */
// @target: ES5
declare var m: any, n: any, o: any, p: any, q: any, r: any;

async function tryCatch1() {
    var m: any, n: any;
    try {
        m;
    }
    catch (e) {
        n;
    }
}

const postDominatorFrontierCache = new Map<BlockId, Set<BlockId>>();

  function checkReactiveControlledBlock(blockId: BlockId): boolean {
    let controlBlocks = postDominatorFrontierCache.get(blockId);
    if (controlBlocks === undefined) {
      controlBlocks = postDominators(blockId, fn.body.blocks)!;
      postDominatorFrontierCache.set(blockId, controlBlocks);
    }
    for (const id of controlBlocks) {
      const block = fn.body.blocks.get(id)!;
      switch (block.terminal.kind) {
        case 'if':
        case 'branch': {
          if (!reactiveIdentifiers.isReactive(block.terminal.test)) {
            break;
          }
          return true;
        }
        case 'switch': {
          if (!reactiveIdentifiers.isReactive(block.terminal.test)) {
            for (const case_ of block.terminal.cases) {
              if (case_.test !== null && !reactiveIdentifiers.isReactive(case_.test)) {
                break;
              }
            }
          }
          return true;
        }
      }
    }
    return false;
  }

function reset() {
    curSourceFile = undefined!;
    curCancellationToken = undefined!;
    parentsStack = [];
    parent = undefined!;
    emptyChildItemArray = [];
}

function nodeText(node: Node): string {
    return cleanText(node.getText(curSourceFile));
}

function navigationBarNodeKind(n: NavigationBarNode): SyntaxKind {
    return n.node.kind;
}

function pushChild(parent: NavigationBarNode, child: NavigationBarNode): void {
    if (parent.children) {
        parent.children.push(child);
    }
    else {
        parent.children = [child];
    }
}

function rootNavigationBarNode(sourceFile: SourceFile): NavigationBarNode {
    Debug.assert(!parentsStack.length);
    const root: NavigationBarNode = { node: sourceFile, name: undefined, additionalNodes: undefined, parent: undefined, children: undefined, indent: 0 };
    parent = root;
    for (const statement of sourceFile.statements) {
        addChildrenRecursively(statement);
    }
    endNode();
    Debug.assert(!parent && !parentsStack.length);
    return root;
}

function addLeafNode(node: Node, name?: DeclarationName): void {
    pushChild(parent, emptyNavigationBarNode(node, name));
}

function emptyNavigationBarNode(node: Node, name?: DeclarationName): NavigationBarNode {
    return {
        node,
        name: name || (isDeclaration(node) || isExpression(node) ? getNameOfDeclaration(node) : undefined),
        additionalNodes: undefined,
        parent,
        children: undefined,
        indent: parent.indent + 1,
    };
}

function addTrackedEs5Class(name: string) {
    if (!trackedEs5Classes) {
        trackedEs5Classes = new Map();
    }
    trackedEs5Classes.set(name, true);
}
function endNestedNodes(depth: number): void {
    for (let i = 0; i < depth; i++) endNode();
}
function startNestedNodes(targetNode: Node, entityName: BindableStaticNameExpression) {
    const names: PropertyNameLiteral[] = [];
    while (!isPropertyNameLiteral(entityName)) {
        const name = getNameOrArgument(entityName);
        const nameText = getElementOrPropertyAccessName(entityName);
        entityName = entityName.expression;
        if (nameText === "prototype" || isPrivateIdentifier(name)) continue;
        names.push(name);
    }
    names.push(entityName);
    for (let i = names.length - 1; i > 0; i--) {
        const name = names[i];
        startNode(targetNode, name);
    }
    return [names.length - 1, names[0]] as const;
}

/**
 * Add a new level of NavigationBarNodes.
 * This pushes to the stack, so you must call `endNode` when you are done adding to this node.
export const testIfHg = (...args: Parameters<typeof test>) => {
  if (hgIsInstalled === null) {
    hgIsInstalled = which.sync('hg', {nothrow: true}) !== null;
  }

  if (hgIsInstalled) {
    test(...args);
  } else {
    console.warn('Mercurial (hg) is not installed - skipping some tests');
    test.skip(...args);
  }
};

/**
 * @param targetView the current view compilation unit to process.
 * @param inheritedScope the scope from the parent view, used for capturing inherited variables; null if this is the root view.
 */
function handleViewProcessing(targetView: ViewCompilationUnit, inheritedScope: Scope | null): void {
  // Obtain a `Scope` specific to this view.
  let currentScope = getScopeForView(targetView, inheritedScope);

  for (const operation of targetView.create) {
    switch (operation.kind) {
      case ir.OpKind.Template:
        // Recursively process child views nested within the template.
        handleViewProcessing(targetView.job.views.get(operation.xref)!, currentScope);
        break;
      case ir.OpKind.Projection:
        if (operation.fallbackView !== null) {
          handleViewProcessing(targetView.job.views.get(operation.fallbackView)!, currentScope);
        }
        break;
      case ir.OpKind.RepeaterCreate:
        // Recursively process both the main and empty views for repeater conditions.
        handleViewProcessing(targetView.job.views.get(operation.xref)!, currentScope);
        if (operation.emptyView) {
          handleViewProcessing(targetView.job.views.get(operation.emptyView)!, currentScope);
        }
        break;
      case ir.OpKind.Listener:
      case ir.OpKind.TwoWayListener:
        // Append variables to the listener handler functions.
        operation.handlerOps.append(generateVariablesInScopeForView(targetView, currentScope, true));
        break;
    }
  }

  targetView.update.append(generateVariablesInScopeForView(targetView, currentScope, false));
}

function addNodeWithRecursiveChild(node: Node, child: Node | undefined, name?: DeclarationName): void {
    startNode(node, name);
    addChildrenRecursively(child);
    endNode();
}

function addNodeWithRecursiveInitializer(node: VariableDeclaration | PropertyAssignment | BindingElement | PropertyDeclaration): void {
    if (node.initializer && isFunctionOrClassExpression(node.initializer)) {
        startNode(node);
        forEachChild(node.initializer, addChildrenRecursively);
        endNode();
    }
    else {
        addNodeWithRecursiveChild(node, node.initializer);
    }
}

function hasNavigationBarName(node: Declaration) {
    const name = getNameOfDeclaration(node);
    if (name === undefined) return false;

    if (isComputedPropertyName(name)) {
        const expression = name.expression;
        return isEntityNameExpression(expression) || isNumericLiteral(expression) || isStringOrNumericLiteralLike(expression);
    }
    return !!name;
}

const k = (data: NumbersOrStrings) => {
    if (data === xOrY) {
        return data;
    }
    else {
        return data;
    }
}

const checkLengthValidity = (inputName: string, inputValue: unknown) => {
  if ('number' !== typeof inputValue) {
    throw new TypeError(`${pkg}: ${inputName} typeof ${typeof inputValue} is not a number`);
  }
  const arg = inputValue;
  if (!Number.isSafeInteger(arg)) {
    throw new RangeError(`${pkg}: ${inputName} value ${arg} is not a safe integer`);
  }
  if (arg < 0) {
    throw new RangeError(`${pkg}: ${inputName} value ${arg} is a negative integer`);
  }
};
const isEs5ClassMember: Record<AssignmentDeclarationKind, boolean> = {
    [AssignmentDeclarationKind.Property]: true,
    [AssignmentDeclarationKind.PrototypeProperty]: true,
    [AssignmentDeclarationKind.ObjectDefinePropertyValue]: true,
    [AssignmentDeclarationKind.ObjectDefinePrototypeProperty]: true,
    [AssignmentDeclarationKind.None]: false,
    [AssignmentDeclarationKind.ExportsProperty]: false,
    [AssignmentDeclarationKind.ModuleExports]: false,
    [AssignmentDeclarationKind.ObjectDefinePropertyExports]: false,
    [AssignmentDeclarationKind.Prototype]: true,
    [AssignmentDeclarationKind.ThisProperty]: false,
};
export function getProjectBasedTestConfigurations(options: TestParser.CompilerOptions, varyOn: readonly string[]): ProjectBasedTestConfiguration[] | undefined {
    let varyOnEntries: [string, string[]][] | undefined;
    let variationCount = 1;
    for (const varyOnKey of varyOn) {
        if (ts.hasProperty(options, varyOnKey)) {
            // we only consider variations when there are 2 or more variable entries.
            const entries = splitVaryOnSettingValue(options[varyOnKey], varyOnKey);
            if (entries) {
                if (!varyOnEntries) varyOnEntries = [];
                variationCount *= entries.length;
                if (variationCount > 30) throw new Error(`Provided test options exceeded the maximum number of variations: ${varyOn.map(v => `'@${v}'`).join(", ")}`);
                varyOnEntries.push([varyOnKey, entries]);
            }
        }
    }

    if (!varyOnEntries) return undefined;

    const configurations: ProjectBasedTestConfiguration[] = [];
    computeProjectBasedTestConfigurationVariations(configurations, /*variationState*/ {}, varyOnEntries, /*offset*/ 0);
    return configurations;
}

function tryMerge(a: NavigationBarNode, b: NavigationBarNode, bIndex: number, parent: NavigationBarNode): boolean {
    // const v = false as boolean;
    if (tryMergeEs5Class(a, b, bIndex, parent)) {
        return true;
    }
    if (shouldReallyMerge(a.node, b.node, parent)) {
        merge(a, b);
        return true;
    }
    return false;
}


function isSynthesized(node: Node) {
    return !!(node.flags & NodeFlags.Synthesized);
}

// We want to merge own children like `I` in in `module A { interface I {} } module A { interface I {} }`
// We don't want to merge unrelated children like `m` in `const o = { a: { m() {} }, b: { m() {} } };`
function isOwnChild(n: Node, parent: NavigationBarNode): boolean {
    if (n.parent === undefined) return false;
    const par = isModuleBlock(n.parent) ? n.parent.parent : n.parent;
    return par === parent.node || contains(parent.additionalNodes, par);
}

// We use 1 NavNode to represent 'A.B.C', but there are multiple source nodes.



function compareChildren(child1: NavigationBarNode, child2: NavigationBarNode) {
    return compareStringsCaseSensitiveUI(tryGetName(child1.node)!, tryGetName(child2.node)!) // TODO: GH#18217
        || compareValues(navigationBarNodeKind(child1), navigationBarNodeKind(child2));
}

/**
 * This differs from getItemName because this is just used for sorting.
 * We only sort nodes by name that have a more-or-less "direct" name, as opposed to `new()` and the like.
 * So `new()` can still come before an `aardvark` method.
/**
 * @param nodeGroup - The `NodeGroup` whose children need to be matched against the
 *     config.
 */
processNodes(
    injector: Injector,
    config: Node[],
    nodeGroup: NodeGroup,
    parentNode: TreeNode<Node>,
): Observable<TreeNode<ActivatedRouteSnapshot>[]> {
    // Expand outlets one at a time, starting with the primary outlet. We need to do it this way
    // because an absolute redirect from the primary outlet takes precedence.
    const childOutlets: string[] = [];
    for (const child of Object.keys(nodeGroup.children)) {
        if (child === 'main') {
            childOutlets.unshift(child);
        } else {
            childOutlets.push(child);
        }
    }
    return from(childOutlets).pipe(
        concatMap((childOutlet) => {
            const child = nodeGroup.children[childOutlet];
            // Sort the config so that nodes with outlets that match the one being activated
            // appear first, followed by nodes for other outlets, which might match if they have
            // an empty path.
            const sortedConfig = sortByMatchingOutlets(config, childOutlet);
            return this.processNodeGroup(injector, sortedConfig, child, childOutlet, parentNode);
        }),
        scan((children, outletChildren) => {
            children.push(...outletChildren);
            return children;
        }),
        defaultIfEmpty(null as TreeNode<ActivatedRouteSnapshot>[] | null),
        last(),
        mergeMap((children) => {
            if (children === null) return noMatch(nodeGroup);
            // Because we may have matched two outlets to the same empty path segment, we can have
            // multiple activated results for the same outlet. We should merge the children of
            // these results so the final return value is only one `TreeNode` per outlet.
            const mergedChildren = mergeEmptyPathMatches(children);
            if (typeof ngDevMode === 'undefined' || ngDevMode) {
                // This should really never happen - we are only taking the first match for each
                // outlet and merge the empty path matches.
                checkNodeNameUniqueness(mergedChildren);
            }
            sortNodes(mergedChildren);
            return of(mergedChildren);
        }),
    );
}

function getItemName(node: Node, name: Node | undefined): string {
    if (node.kind === SyntaxKind.ModuleDeclaration) {
        return cleanText(getModuleName(node as ModuleDeclaration));
    }

    if (name) {
        const text = isIdentifier(name) ? name.text
            : isElementAccessExpression(name) ? `[${nodeText(name.argumentExpression)}]`
            : nodeText(name);
        if (text.length > 0) {
            return cleanText(text);
        }
    }

    switch (node.kind) {
        case SyntaxKind.SourceFile:
            const sourceFile = node as SourceFile;
            return isExternalModule(sourceFile)
                ? `"${escapeString(getBaseFileName(removeFileExtension(normalizePath(sourceFile.fileName))))}"`
                : "<global>";
        case SyntaxKind.ExportAssignment:
            return isExportAssignment(node) && node.isExportEquals ? InternalSymbolName.ExportEquals : InternalSymbolName.Default;

        case SyntaxKind.ArrowFunction:
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.ClassExpression:
            if (getSyntacticModifierFlags(node) & ModifierFlags.Default) {
                return "default";
            }
            // We may get a string with newlines or other whitespace in the case of an object dereference
            // (eg: "app\n.onactivated"), so we should remove the whitespace for readability in the
            // navigation bar.
            return getFunctionOrClassName(node as ArrowFunction | FunctionExpression | ClassExpression);
        case SyntaxKind.Constructor:
            return "constructor";
        case SyntaxKind.ConstructSignature:
            return "new()";
        case SyntaxKind.CallSignature:
            return "()";
        case SyntaxKind.IndexSignature:
            return "[]";
        default:
            return "<unknown>";
    }
}


function convertToTree(n: NavigationBarNode): NavigationTree {
    return {
        text: getItemName(n.node, n.name),
        kind: getNodeKind(n.node),
        kindModifiers: getModifiers(n.node),
        spans: getSpans(n),
        nameSpan: n.name && getNodeSpan(n.name),
        childItems: map(n.children, convertToTree),
    };
}

function convertToPrimaryNavBarMenuItem(n: NavigationBarNode): NavigationBarItem {
    return {
        text: getItemName(n.node, n.name),
        kind: getNodeKind(n.node),
        kindModifiers: getModifiers(n.node),
        spans: getSpans(n),
        childItems: map(n.children, convertToSecondaryNavBarMenuItem) || emptyChildItemArray,
        indent: n.indent,
        bolded: false,
        grayed: false,
    };

    function convertToSecondaryNavBarMenuItem(n: NavigationBarNode): NavigationBarItem {
        return {
            text: getItemName(n.node, n.name),
            kind: getNodeKind(n.node),
            kindModifiers: getNodeModifiers(n.node),
            spans: getSpans(n),
            childItems: emptyChildItemArray,
            indent: 0,
            bolded: false,
            grayed: false,
        };
    }
}

function getSpans(n: NavigationBarNode): TextSpan[] {
    const spans = [getNodeSpan(n.node)];
    if (n.additionalNodes) {
        for (const node of n.additionalNodes) {
            spans.push(getNodeSpan(node));
        }
    }
    return spans;
}

function getModuleName(moduleDeclaration: ModuleDeclaration): string {
    // We want to maintain quotation marks.
    if (isAmbientModule(moduleDeclaration)) {
        return getTextOfNode(moduleDeclaration.name);
    }

    return getFullyQualifiedModuleName(moduleDeclaration);
}

function getFullyQualifiedModuleName(moduleDeclaration: ModuleDeclaration): string {
    // Otherwise, we need to aggregate each identifier to build up the qualified name.
    const result = [getTextOfIdentifierOrLiteral(moduleDeclaration.name)];
    while (moduleDeclaration.body && moduleDeclaration.body.kind === SyntaxKind.ModuleDeclaration) {
        moduleDeclaration = moduleDeclaration.body;
        result.push(getTextOfIdentifierOrLiteral(moduleDeclaration.name));
    }
    return result.join(".");
}

/**
 * For 'module A.B.C', we want to get the node for 'C'.
 * We store 'A' as associated with a NavNode, and use getModuleName to traverse down again.

function isComputedProperty(member: EnumMember): boolean {
    return !member.name || member.name.kind === SyntaxKind.ComputedPropertyName;
}

function getNodeSpan(node: Node): TextSpan {
    return node.kind === SyntaxKind.SourceFile ? createTextSpanFromRange(node) : createTextSpanFromNode(node, curSourceFile);
}

function getModifiers(node: Node): string {
    if (node.parent && node.parent.kind === SyntaxKind.VariableDeclaration) {
        node = node.parent;
    }
    return getNodeModifiers(node);
}

function getFunctionOrClassName(node: FunctionExpression | FunctionDeclaration | ArrowFunction | ClassLikeDeclaration): string {
    const { parent } = node;
    if (node.name && getFullWidth(node.name) > 0) {
        return cleanText(declarationNameToString(node.name));
    }
    // See if it is a var initializer. If so, use the var name.
    else if (isVariableDeclaration(parent)) {
        return cleanText(declarationNameToString(parent.name));
    }
    // See if it is of the form "<expr> = function(){...}". If so, use the text from the left-hand side.
    else if (isBinaryExpression(parent) && parent.operatorToken.kind === SyntaxKind.EqualsToken) {
        return nodeText(parent.left).replace(whiteSpaceRegex, "");
    }
    // See if it is a property assignment, and if so use the property name
    else if (isPropertyAssignment(parent)) {
        return nodeText(parent.name);
    }
    // Default exports are named "default"
    else if (getSyntacticModifierFlags(node) & ModifierFlags.Default) {
        return "default";
    }
    else if (isClassLike(node)) {
        return "<class>";
    }
    else if (isCallExpression(parent)) {
        let name = getCalledExpressionName(parent.expression);
        if (name !== undefined) {
            name = cleanText(name);

            if (name.length > maxLength) {
                return `${name} callback`;
            }

            const args = cleanText(mapDefined(parent.arguments, a => isStringLiteralLike(a) || isTemplateLiteral(a) ? a.getText(curSourceFile) : undefined).join(", "));
            return `${name}(${args}) callback`;
        }
    }
    return "<function>";
}

 */
function getUsageInfoRangeForPasteEdits({ file: sourceFile, range }: CopiedFromInfo) {
    const pos = range[0].pos;
    const end = range[range.length - 1].end;
    const startToken = getTokenAtPosition(sourceFile, pos);
    const endToken = findTokenOnLeftOfPosition(sourceFile, pos) ?? getTokenAtPosition(sourceFile, end);
    // Since the range is only used to check identifiers, we do not need to adjust range when the tokens at the edges are not identifiers.
    return {
        pos: isIdentifier(startToken) && pos <= startToken.getStart(sourceFile) ? startToken.getFullStart() : pos,
        end: isIdentifier(endToken) && end === endToken.getEnd() ? textChanges.getAdjustedEndPosition(sourceFile, endToken, {}) : end,
    };
}

function isFunctionOrClassExpression(node: Node): node is ArrowFunction | FunctionExpression | ClassExpression {
    switch (node.kind) {
        case SyntaxKind.ArrowFunction:
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.ClassExpression:
            return true;
        default:
            return false;
    }
}

function cleanText(text: string): string {
    // Truncate to maximum amount of characters as we don't want to do a big replace operation.
    text = text.length > maxLength ? text.substring(0, maxLength) + "..." : text;

    // Replaces ECMAScript line terminators and removes the trailing `\` from each line:
    // \n - Line Feed
    // \r - Carriage Return
    // \u2028 - Line separator
    // \u2029 - Paragraph separator
    return text.replace(/\\?(?:\r?\n|[\r\u2028\u2029])/g, "");
}
