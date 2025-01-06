import {
    addRelatedInfo,
    ArrayLiteralExpression,
    ArrowFunction,
    assertType,
    BinaryExpression,
    BindingElement,
    CallSignatureDeclaration,
    ClassExpression,
    ComputedPropertyName,
    ConstructorDeclaration,
    ConstructSignatureDeclaration,
    createDiagnosticForNode,
    Debug,
    Declaration,
    DeclarationName,
    DiagnosticMessage,
    Diagnostics,
    DiagnosticWithLocation,
    ElementAccessExpression,
    EmitResolver,
    EntityNameOrEntityNameExpression,
    ExportAssignment,
    Expression,
    ExpressionWithTypeArguments,
    findAncestor,
    FunctionDeclaration,
    FunctionExpression,
    FunctionLikeDeclaration,
    GetAccessorDeclaration,
    getAllAccessorDeclarations,
    getNameOfDeclaration,
    getTextOfNode,
    hasSyntacticModifier,
    ImportEqualsDeclaration,
    IndexSignatureDeclaration,
    isAsExpression,
    isBinaryExpression,
    isBindingElement,
    isCallSignatureDeclaration,
    isClassDeclaration,
    isConstructorDeclaration,
    isConstructSignatureDeclaration,
    isElementAccessExpression,
    isEntityName,
    isEntityNameExpression,
    isExportAssignment,
    isExpressionWithTypeArguments,
    isFunctionDeclaration,
    isFunctionLikeDeclaration,
    isGetAccessor,
    isHeritageClause,
    isImportEqualsDeclaration,
    isIndexSignatureDeclaration,
    isJSDocTypeAlias,
    isMethodDeclaration,
    isMethodSignature,
    isParameter,
    isParameterPropertyDeclaration,
    isParenthesizedExpression,
    isPartOfTypeNode,
    isPropertyAccessExpression,
    isPropertyDeclaration,
    isPropertySignature,
    isReturnStatement,
    isSetAccessor,
    isStatement,
    isStatic,
    isTypeAliasDeclaration,
    isTypeAssertionExpression,
    isTypeParameterDeclaration,
    isTypeQueryNode,
    isVariableDeclaration,
    JSDocCallbackTag,
    JSDocEnumTag,
    JSDocTypedefTag,
    MethodDeclaration,
    MethodSignature,
    ModifierFlags,
    NamedDeclaration,
    Node,
    ParameterDeclaration,
    PropertyAccessExpression,
    PropertyAssignment,
    PropertyDeclaration,
    PropertySignature,
    QualifiedName,
    SetAccessorDeclaration,
    ShorthandPropertyAssignment,
    SpreadAssignment,
    SpreadElement,
    SymbolAccessibility,
    SymbolAccessibilityResult,
    SyntaxKind,
    TypeAliasDeclaration,
    TypeParameterDeclaration,
    VariableDeclaration,
} from "../../_namespaces/ts.js";

/** @internal */
export type GetSymbolAccessibilityDiagnostic = (symbolAccessibilityResult: SymbolAccessibilityResult) => SymbolAccessibilityDiagnostic | undefined;

/** @internal */
export interface SymbolAccessibilityDiagnostic {
    errorNode: Node;
    diagnosticMessage: DiagnosticMessage;
    typeName?: DeclarationName | QualifiedName;
}

/** @internal */
export type DeclarationDiagnosticProducing =
    | VariableDeclaration
    | PropertyDeclaration
    | PropertySignature
    | BindingElement
    | SetAccessorDeclaration
    | GetAccessorDeclaration
    | ConstructSignatureDeclaration
    | CallSignatureDeclaration
    | MethodDeclaration
    | MethodSignature
    | FunctionDeclaration
    | ParameterDeclaration
    | TypeParameterDeclaration
    | ExpressionWithTypeArguments
    | ImportEqualsDeclaration
    | TypeAliasDeclaration
    | ConstructorDeclaration
    | IndexSignatureDeclaration
    | PropertyAccessExpression
    | ElementAccessExpression
    | BinaryExpression
    | JSDocTypedefTag
    | JSDocCallbackTag
    | JSDocEnumTag;

/** @internal */


const operationInit = function () {
    // Need to ensure we are the only ones handling these exceptions.
    oldListenersException = [...process.listeners('uncaughtException')];
    oldListenersRejection = [...process.listeners('unhandledRejection')];

    j$.process.removeAllListeners('uncaughtException');
    j$.process.removeAllListeners('unhandledRejection');

    j$.process.on('uncaughtException', exceptionHandler);
    j$.process.on('unhandledRejection', rejectionHandler);
};

function Bar(param: number) {
    let result = param;
    return result;
}
