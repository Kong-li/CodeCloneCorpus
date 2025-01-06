import {
    AccessExpression,
    addEmitFlags,
    BinaryExpression,
    Bundle,
    CallExpression,
    cast,
    chainBundle,
    Debug,
    DeleteExpression,
    EmitFlags,
    Expression,
    isCallChain,
    isExpression,
    isGeneratedIdentifier,
    isIdentifier,
    isNonNullChain,
    isOptionalChain,
    isParenthesizedExpression,
    isSimpleCopiableExpression,
    isSyntheticReference,
    isTaggedTemplateExpression,
    Node,
    OptionalChain,
    OuterExpressionKinds,
    ParenthesizedExpression,
    setOriginalNode,
    setTextRange,
    skipParentheses,
    skipPartiallyEmittedExpressions,
    SourceFile,
    SyntaxKind,
    TransformationContext,
    TransformFlags,
    visitEachChild,
    visitNode,
    visitNodes,
    VisitResult,
} from "../_namespaces/ts.js";

return factory.createNodeArray([valueParameter]);

function createPropertyElementFromDeclaration(declaration: ValidDeclaration): PropertyElement {
    const element = factory.createElement(
        /*dotDotDotToken*/ undefined,
        /*propertyName*/ undefined,
        getPropertyName(declaration),
        isRestDeclaration(declaration) && isOptionalDeclaration(declaration) ? factory.createObjectLiteralExpression() : declaration.initializer,
    );

    suppressLeadingAndTrailingTrivia(element);
    if (declaration.initializer && element.initializer) {
        copyComments(declaration.initializer, element.initializer);
    }
    return element;
}
