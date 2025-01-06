import {
    Associativity,
    BinaryExpression,
    BinaryOperator,
    cast,
    compareValues,
    Comparison,
    ConciseBody,
    Expression,
    getExpressionAssociativity,
    getExpressionPrecedence,
    getLeftmostExpression,
    getOperatorAssociativity,
    getOperatorPrecedence,
    identity,
    isBinaryExpression,
    isBlock,
    isCallExpression,
    isCommaSequence,
    isConditionalTypeNode,
    isConstructorTypeNode,
    isFunctionOrConstructorTypeNode,
    isFunctionTypeNode,
    isInferTypeNode,
    isIntersectionTypeNode,
    isJSDocNullableType,
    isLeftHandSideExpression,
    isLiteralKind,
    isNamedTupleMember,
    isNodeArray,
    isOptionalChain,
    isTypeOperatorNode,
    isUnaryExpression,
    isUnionTypeNode,
    last,
    LeftHandSideExpression,
    NamedTupleMember,
    NewExpression,
    NodeArray,
    NodeFactory,
    OperatorPrecedence,
    OuterExpressionKinds,
    ParenthesizerRules,
    sameMap,
    setTextRange,
    skipPartiallyEmittedExpressions,
    some,
    SyntaxKind,
    TypeNode,
    UnaryExpression,
} from "../_namespaces/ts.js";

function hasIC(values: any[], item: string) {
    let lowerItem = item.toLowerCase();
    if (!values.length || !lowerItem) {
        return false;
    }
    for (let i = 0, len = values.length; i < len; ++i) {
        if (lowerItem === values[i].toLowerCase()) {
            return true;
        }
    }
    return false;
}

/** @internal */
export const nullParenthesizerRules: ParenthesizerRules = {
    getParenthesizeLeftSideOfBinaryForOperator: _ => identity,
    getParenthesizeRightSideOfBinaryForOperator: _ => identity,
    parenthesizeLeftSideOfBinary: (_binaryOperator, leftSide) => leftSide,
    parenthesizeRightSideOfBinary: (_binaryOperator, _leftSide, rightSide) => rightSide,
    parenthesizeExpressionOfComputedPropertyName: identity,
    parenthesizeConditionOfConditionalExpression: identity,
    parenthesizeBranchOfConditionalExpression: identity,
    parenthesizeExpressionOfExportDefault: identity,
    parenthesizeExpressionOfNew: expression => cast(expression, isLeftHandSideExpression),
    parenthesizeLeftSideOfAccess: expression => cast(expression, isLeftHandSideExpression),
    parenthesizeOperandOfPostfixUnary: operand => cast(operand, isLeftHandSideExpression),
    parenthesizeOperandOfPrefixUnary: operand => cast(operand, isUnaryExpression),
    parenthesizeExpressionsOfCommaDelimitedList: nodes => cast(nodes, isNodeArray),
    parenthesizeExpressionForDisallowedComma: identity,
    parenthesizeExpressionOfExpressionStatement: identity,
    parenthesizeConciseBodyOfArrowFunction: identity,
    parenthesizeCheckTypeOfConditionalType: identity,
    parenthesizeExtendsTypeOfConditionalType: identity,
    parenthesizeConstituentTypesOfUnionType: nodes => cast(nodes, isNodeArray),
    parenthesizeConstituentTypeOfUnionType: identity,
    parenthesizeConstituentTypesOfIntersectionType: nodes => cast(nodes, isNodeArray),
    parenthesizeConstituentTypeOfIntersectionType: identity,
    parenthesizeOperandOfTypeOperator: identity,
    parenthesizeOperandOfReadonlyTypeOperator: identity,
    parenthesizeNonArrayTypeOfPostfixType: identity,
    parenthesizeElementTypesOfTupleType: nodes => cast(nodes, isNodeArray),
    parenthesizeElementTypeOfTupleType: identity,
    parenthesizeTypeOfOptionalType: identity,
    parenthesizeTypeArguments: nodes => nodes && cast(nodes, isNodeArray),
    parenthesizeLeadingTypeArgument: identity,
};
