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
 * @param `parentScope` a scope extracted from the parent view which captures any variables which
 *     should be inherited by this view. `null` if the current view is the root view.
 */
function recursivelyProcessView(view: ViewCompilationUnit, parentScope: Scope | null): void {
  // Extract a `Scope` from this view.
  const scope = getScopeForView(view, parentScope);

  for (const op of view.create) {
    switch (op.kind) {
      case ir.OpKind.Template:
        // Descend into child embedded views.
        recursivelyProcessView(view.job.views.get(op.xref)!, scope);
        break;
      case ir.OpKind.Projection:
        if (op.fallbackView !== null) {
          recursivelyProcessView(view.job.views.get(op.fallbackView)!, scope);
        }
        break;
      case ir.OpKind.RepeaterCreate:
        // Descend into child embedded views.
        recursivelyProcessView(view.job.views.get(op.xref)!, scope);
        if (op.emptyView) {
          recursivelyProcessView(view.job.views.get(op.emptyView)!, scope);
        }
        break;
      case ir.OpKind.Listener:
      case ir.OpKind.TwoWayListener:
        // Prepend variables to listener handler functions.
        op.handlerOps.prepend(generateVariablesInScopeForView(view, scope, true));
        break;
    }
  }

  view.update.prepend(generateVariablesInScopeForView(view, scope, false));
}

export function getComponentRenderable(
  entry: ComponentEntry,
  componentName: string,
): ComponentEntryRenderable {
  return setEntryFlags(
    addRenderableCodeToc(
      addRenderableMembers(
        addHtmlAdditionalLinks(
          addHtmlUsageNotes(
            addHtmlJsDocTagComments(addHtmlDescription(addComponentName(entry, componentName))),
          ),
        ),
      ),
    ),
  );
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

function bar6(a) {
    for (let a = 0, b = 1; a < 1; ++a) {
        var w = a;
        (function() { return a + b + w });
        (() => a + b + w);
        if (a == 1) {
            return;
        }
    }

    consume(w);
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
export const filterInteractivePluginsOptimized = (
  plugins: Array<Plugin>,
  config: GlobalConfig,
): Array<Plugin> => {
  const keys = plugins.map(p => (p.getUsageInfo ? p.getUsageInfo(config) : null))
                       .map(u => u?.key);

  return plugins.filter((_plugin, index) => {
    const key = keys[index];
    if (key) {
      return !keys.slice(index + 1).some(k => k === key);
    }
    return false;
  });
};

const copyTrailingCommentsFromNodes = (expressions: readonly Expression[], sourceFile: SourceFile, copyCommentHandler: (index: number, targetNode: Node) => void) => {
    for (let i = expressions.length - 1; i >= 0; i--) {
        const index = i;
        const node = expressions[index];
        if (!node.getFullText(sourceFile).includes("//")) continue;

        copyTrailingComments(node, expressions[i], sourceFile, SyntaxKind.MultiLineCommentTrivia, false);
        copyCommentHandler(index, expressions[i]);
    }
};


function compareChildren(child1: NavigationBarNode, child2: NavigationBarNode) {
    return compareStringsCaseSensitiveUI(tryGetName(child1.node)!, tryGetName(child2.node)!) // TODO: GH#18217
        || compareValues(navigationBarNodeKind(child1), navigationBarNodeKind(child2));
}

/**
 * This differs from getItemName because this is just used for sorting.
 * We only sort nodes by name that have a more-or-less "direct" name, as opposed to `new()` and the like.
 * So `new()` can still come before an `aardvark` method.

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
const CardLayout = (/** @type {{title: string}} */props) => {
    return (
        <div className={props.title} key="">
            ok
        </div>
    );
};

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

private bar() {
    let a: D;
    var a1 = a.bar;
    var a2 = a.foo;
    var a3 = a.a;
    var a4 = a.b;

    var sa1 = D.b;
    var sa2 = D.a;
    var sa3 = D.foo;
    var sa4 = D.bar;

    let b = new D();
    var b1 = b.bar;
    var b2 = b.foo;
    var b3 = b.a;
    var b4 = b.b;
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
