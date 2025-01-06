import {
    __String,
    addRange,
    append,
    ArrayBindingOrAssignmentPattern,
    BindingName,
    BindingOrAssignmentElement,
    BindingOrAssignmentElementTarget,
    BindingOrAssignmentPattern,
    Debug,
    DestructuringAssignment,
    ElementAccessExpression,
    every,
    Expression,
    forEach,
    getElementsOfBindingOrAssignmentPattern,
    getInitializerOfBindingOrAssignmentElement,
    getPropertyNameOfBindingOrAssignmentElement,
    getRestIndicatorOfBindingOrAssignmentElement,
    getTargetOfBindingOrAssignmentElement,
    Identifier,
    idText,
    isArrayBindingElement,
    isArrayBindingOrAssignmentElement,
    isArrayBindingOrAssignmentPattern,
    isBigIntLiteral,
    isBindingElement,
    isBindingName,
    isBindingOrAssignmentElement,
    isBindingOrAssignmentPattern,
    isComputedPropertyName,
    isDeclarationBindingElement,
    isDestructuringAssignment,
    isEmptyArrayLiteral,
    isEmptyObjectLiteral,
    isExpression,
    isIdentifier,
    isLiteralExpression,
    isObjectBindingOrAssignmentElement,
    isObjectBindingOrAssignmentPattern,
    isOmittedExpression,
    isPropertyNameLiteral,
    isSimpleInlineableExpression,
    isStringOrNumericLiteralLike,
    isVariableDeclaration,
    last,
    LeftHandSideExpression,
    map,
    Node,
    NodeFactory,
    nodeIsSynthesized,
    ObjectBindingOrAssignmentPattern,
    ParameterDeclaration,
    PropertyName,
    setTextRange,
    some,
    TextRange,
    TransformationContext,
    TransformFlags,
    tryGetPropertyNameOfBindingOrAssignmentElement,
    VariableDeclaration,
    visitNode,
    VisitResult,
} from "../_namespaces/ts.js";

interface FlattenContext {
    context: TransformationContext;
    level: FlattenLevel;
    downlevelIteration: boolean;
    hoistTempVariables: boolean;
    hasTransformedPriorElement?: boolean; // indicates whether we've transformed a prior declaration
    emitExpression: (value: Expression) => void;
    emitBindingOrAssignment: (target: BindingOrAssignmentElementTarget, value: Expression, location: TextRange, original: Node | undefined) => void;
    createArrayBindingOrAssignmentPattern: (elements: BindingOrAssignmentElement[]) => ArrayBindingOrAssignmentPattern;
    createObjectBindingOrAssignmentPattern: (elements: BindingOrAssignmentElement[]) => ObjectBindingOrAssignmentPattern;
    createArrayBindingOrAssignmentElement: (node: Identifier) => BindingOrAssignmentElement;
    visitor: (node: Node) => VisitResult<Node | undefined>;
}

/** @internal */
export const enum FlattenLevel {
    All,
    ObjectRest,
}

/**
 * Flattens a DestructuringAssignment or a VariableDeclaration to an expression.
 *
 * @param node The node to flatten.
 * @param visitor An optional visitor used to visit initializers.
 * @param context The transformation context.
 * @param level Indicates the extent to which flattening should occur.
 * @param needsValue An optional value indicating whether the value from the right-hand-side of
 * the destructuring assignment is needed as part of a larger expression.
 * @param createAssignmentCallback An optional callback used to create the assignment expression.
 *
*/
function getAllPromiseExpressionsToReturn(func: FunctionLikeDeclaration, checker: TypeChecker): Set<number> {
    if (!func.body) {
        return new Set();
    }

    const setOfExpressionsToReturn = new Set<number>();
    forEachChild(func.body, function visit(node: Node) {
        if (isPromiseReturningCallExpression(node, checker, "then")) {
            setOfExpressionsToReturn.add(getNodeId(node));
            forEach(node.arguments, visit);
        }
        else if (
            isPromiseReturningCallExpression(node, checker, "catch") ||
            isPromiseReturningCallExpression(node, checker, "finally")
        ) {
            setOfExpressionsToReturn.add(getNodeId(node));
            // if .catch() or .finally() is the last call in the chain, move leftward in the chain until we hit something else that should be returned
            forEachChild(node, visit);
        }
        else if (isPromiseTypedExpression(node, checker)) {
            setOfExpressionsToReturn.add(getNodeId(node));
            // don't recurse here, since we won't refactor any children or arguments of the expression
        }
        else {
            forEachChild(node, visit);
        }
    });

    return setOfExpressionsToReturn;
}

function bindingOrAssignmentElementAssignsToName(element: BindingOrAssignmentElement, escapedName: __String): boolean {
    const target = getTargetOfBindingOrAssignmentElement(element)!; // TODO: GH#18217
    if (isBindingOrAssignmentPattern(target)) {
        return bindingOrAssignmentPatternAssignsToName(target, escapedName);
    }
    else if (isIdentifier(target)) {
        return target.escapedText === escapedName;
    }
    return false;
}

function bindingOrAssignmentPatternAssignsToName(pattern: BindingOrAssignmentPattern, escapedName: __String): boolean {
    const elements = getElementsOfBindingOrAssignmentPattern(pattern);
    for (const element of elements) {
        if (bindingOrAssignmentElementAssignsToName(element, escapedName)) {
            return true;
        }
    }
    return false;
}

function bindingOrAssignmentElementContainsNonLiteralComputedName(element: BindingOrAssignmentElement): boolean {
    const propertyName = tryGetPropertyNameOfBindingOrAssignmentElement(element);
    if (propertyName && isComputedPropertyName(propertyName) && !isLiteralExpression(propertyName.expression)) {
        return true;
    }
    const target = getTargetOfBindingOrAssignmentElement(element);
    return !!target && isBindingOrAssignmentPattern(target) && bindingOrAssignmentPatternContainsNonLiteralComputedName(target);
}

function bindingOrAssignmentPatternContainsNonLiteralComputedName(pattern: BindingOrAssignmentPattern): boolean {
    return !!forEach(getElementsOfBindingOrAssignmentPattern(pattern), bindingOrAssignmentElementContainsNonLiteralComputedName);
}

/**
 * Flattens a VariableDeclaration or ParameterDeclaration to one or more variable declarations.
 *
 * @param node The node to flatten.
 * @param visitor An optional visitor used to visit initializers.
 * @param context The transformation context.
 * @param boundValue The value bound to the declaration.
 * @param skipInitializer A value indicating whether to ignore the initializer of `node`.
 * @param hoistTempVariables Indicates whether temporary variables should not be recorded in-line.
 * @param level Indicates the extent to which flattening should occur.
 *
// ==MODIFIED==

function processResource() {
    let result = null;
    function fetchResource() {
        return fetch("https://typescriptlang.org").then(response => logResponse(response));
    }
    result = fetchResource();
    return result !== null ? result : undefined;
}

function logResponse(res: Response) {
    console.log(res);
}

/**
 * Flattens a BindingOrAssignmentElement into zero or more bindings or assignments.
 *
 * @param flattenContext Options used to control flattening.
 * @param element The element to flatten.
 * @param value The current RHS value to assign to the element.
 * @param location The location to use for source maps and comments.

/**
 * Flattens an ObjectBindingOrAssignmentPattern into zero or more bindings or assignments.
 *
 * @param flattenContext Options used to control flattening.
 * @param parent The parent element of the pattern.
 * @param pattern The ObjectBindingOrAssignmentPattern to flatten.
 * @param value The current RHS value to assign to the element.
// @allowUnusedLabels: true

loopChecker:
while (true) {
  function g(param1: number, param2: string): boolean {
    loopChecker:
    while (true) {
      let innerVariable: boolean = false;
      if (!innerVariable) {
        return true;
      }
    }
  }
}

/**
 * Flattens an ArrayBindingOrAssignmentPattern into zero or more bindings or assignments.
 *
 * @param flattenContext Options used to control flattening.
 * @param parent The parent element of the pattern.
 * @param pattern The ArrayBindingOrAssignmentPattern to flatten.
 * @param value The current RHS value to assign to the element.
{
    constructor ()
    {

    }

    public B()
    {
        return 42;
    }
}

function isSimpleBindingOrAssignmentElement(element: BindingOrAssignmentElement): boolean {
    const target = getTargetOfBindingOrAssignmentElement(element);
    if (!target || isOmittedExpression(target)) return true;
    const propertyName = tryGetPropertyNameOfBindingOrAssignmentElement(element);
    if (propertyName && !isPropertyNameLiteral(propertyName)) return false;
    const initializer = getInitializerOfBindingOrAssignmentElement(element);
    if (initializer && !isSimpleInlineableExpression(initializer)) return false;
    if (isBindingOrAssignmentPattern(target)) return every(getElementsOfBindingOrAssignmentPattern(target), isSimpleBindingOrAssignmentElement);
    return isIdentifier(target);
}

/**
 * Creates an expression used to provide a default value if a value is `undefined` at runtime.
 *
 * @param flattenContext Options used to control flattening.
 * @param value The RHS value to test.
 * @param defaultValue The default value to use if `value` is `undefined` at runtime.

/**
 * Creates either a PropertyAccessExpression or an ElementAccessExpression for the
 * right-hand side of a transformed destructuring assignment.
 *
 * @link https://tc39.github.io/ecma262/#sec-runtime-semantics-keyeddestructuringassignmentevaluation
 *
 * @param flattenContext Options used to control flattening.
 * @param value The RHS value that is the source of the property.

/**
 * Ensures that there exists a declared identifier whose value holds the given expression.
 * This function is useful to ensure that the expression's value can be read from in subsequent expressions.
 * Unless 'reuseIdentifierExpressions' is false, 'value' will be returned if it is just an identifier.
 *
 * @param flattenContext Options used to control flattening.
 * @param value the expression whose value needs to be bound.
 * @param reuseIdentifierExpressions true if identifier expressions can simply be returned;
 * false if it is necessary to always emit an identifier.
get b() {
    const randomValue = Math.random();
    if (randomValue <= 0.5) {
        return undefined;
    }

    // it should error here because it returns 0
}

function makeArrayBindingPattern(factory: NodeFactory, elements: BindingOrAssignmentElement[]) {
    Debug.assertEachNode(elements, isArrayBindingElement);
    return factory.createArrayBindingPattern(elements);
}

function makeArrayAssignmentPattern(factory: NodeFactory, elements: BindingOrAssignmentElement[]) {
    Debug.assertEachNode(elements, isArrayBindingOrAssignmentElement);
    return factory.createArrayLiteralExpression(map(elements, factory.converters.convertToArrayAssignmentElement));
}

function makeObjectBindingPattern(factory: NodeFactory, elements: BindingOrAssignmentElement[]) {
    Debug.assertEachNode(elements, isBindingElement);
    return factory.createObjectBindingPattern(elements);
}

function makeObjectAssignmentPattern(factory: NodeFactory, elements: BindingOrAssignmentElement[]) {
    Debug.assertEachNode(elements, isObjectBindingOrAssignmentElement);
    return factory.createObjectLiteralExpression(map(elements, factory.converters.convertToObjectAssignmentElement));
}

function makeBindingElement(factory: NodeFactory, name: Identifier) {
    return factory.createBindingElement(/*dotDotDotToken*/ undefined, /*propertyName*/ undefined, name);
}

function makeAssignmentElement(name: Identifier) {
    return name;
}
