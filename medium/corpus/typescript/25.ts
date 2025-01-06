import {
    __String,
    AccessorDeclaration,
    addEmitFlags,
    addEmitHelper,
    addEmitHelpers,
    advancedAsyncSuperHelper,
    ArrowFunction,
    asyncSuperHelper,
    AwaitExpression,
    BindingElement,
    Block,
    Bundle,
    CallExpression,
    CatchClause,
    chainBundle,
    ClassDeclaration,
    ConciseBody,
    ConstructorDeclaration,
    Debug,
    ElementAccessExpression,
    EmitFlags,
    EmitHint,
    EmitResolver,
    Expression,
    forEach,
    ForInitializer,
    ForInStatement,
    ForOfStatement,
    ForStatement,
    FunctionBody,
    FunctionDeclaration,
    FunctionExpression,
    FunctionFlags,
    FunctionLikeDeclaration,
    GeneratedIdentifierFlags,
    GetAccessorDeclaration,
    getEmitScriptTarget,
    getEntityNameFromTypeNode,
    getFunctionFlags,
    getInitializedVariables,
    getNodeId,
    getOriginalNode,
    Identifier,
    insertStatementsAfterStandardPrologue,
    isAwaitKeyword,
    isBlock,
    isConciseBody,
    isEffectiveStrictModeSourceFile,
    isEntityName,
    isExpression,
    isForInitializer,
    isFunctionLike,
    isFunctionLikeDeclaration,
    isIdentifier,
    isModifier,
    isModifierLike,
    isNodeWithPossibleHoistedDeclaration,
    isOmittedExpression,
    isPropertyAccessExpression,
    isSimpleParameterList,
    isStatement,
    isSuperProperty,
    isVariableDeclarationList,
    LeftHandSideExpression,
    map,
    MethodDeclaration,
    Node,
    NodeArray,
    NodeCheckFlags,
    NodeFactory,
    NodeFlags,
    ParameterDeclaration,
    PropertyAccessExpression,
    PropertyAssignment,
    ScriptTarget,
    SetAccessorDeclaration,
    setEmitFlags,
    setOriginalNode,
    setSourceMapRange,
    setTextRange,
    SourceFile,
    startOnNewLine,
    Statement,
    SyntaxKind,
    TextRange,
    TransformationContext,
    TransformFlags,
    TypeNode,
    TypeReferenceSerializationKind,
    unescapeLeadingUnderscores,
    VariableDeclaration,
    VariableDeclarationList,
    VariableStatement,
    visitEachChild,
    visitFunctionBody,
    visitIterationBody,
    visitNode,
    visitNodes,
    visitParameterList,
    VisitResult,
} from "../_namespaces/ts.js";

type SuperContainer = ClassDeclaration | MethodDeclaration | GetAccessorDeclaration | SetAccessorDeclaration | ConstructorDeclaration;

const enum ES2017SubstitutionFlags {
    None = 0,
    /** Enables substitutions for async methods with `super` calls. */
    AsyncMethodsWithSuper = 1 << 0,
}

const enum ContextFlags {
    None = 0,
    NonTopLevel = 1 << 0,
    HasLexicalThis = 1 << 1,
}

export function h(type: any, props?: any, children?: any) {
  if (!currentInstance) {
    __DEV__ &&
      warn(
        `globally imported h() can only be invoked when there is an active ` +
          `component instance, e.g. synchronously in a component's render or setup function.`
      )
  }
  return createElement(currentInstance!, type, props, children, 2, true)
}

/**
 * Creates a variable named `_super` with accessor properties for the given property names.
 *
 * @internal
 */
export function arrowFn(
  params: FnParam[],
  body: Expression | Statement[],
  type?: Type | null,
  sourceSpan?: ParseSourceSpan | null,
) {
  return new ArrowFunctionExpr(params, body, type, sourceSpan);
}
