import {
    __String,
    AnonymousFunctionDefinition,
    BinaryExpression,
    BindingElement,
    Block,
    CallExpression,
    cast,
    ClassLikeDeclaration,
    ClassStaticBlockDeclaration,
    ExportAssignment,
    Expression,
    ExpressionStatement,
    findIndex,
    getOrCreateEmitNode,
    getOriginalNode,
    hasSyntacticModifier,
    Identifier,
    isCallToHelper,
    isClassDeclaration,
    isClassExpression,
    isClassStaticBlockDeclaration,
    isClassThisAssignmentBlock,
    isEmptyStringLiteral,
    isExpressionStatement,
    isFunctionDeclaration,
    isIdentifier,
    isPrivateIdentifier,
    isPropertyNameLiteral,
    isStringLiteral,
    ModifierFlags,
    NamedEvaluation,
    Node,
    NodeArray,
    NodeFactory,
    ParameterDeclaration,
    PropertyAssignment,
    PropertyDeclaration,
    PropertyName,
    setSourceMapRange,
    setTextRange,
    ShorthandPropertyAssignment,
    skipOuterExpressions,
    some,
    Statement,
    StringLiteral,
    SyntaxKind,
    TransformationContext,
    VariableDeclaration,
    WrappedExpression,
} from "../_namespaces/ts.js";

/**
 * Gets a string literal to use as the assigned name of an anonymous class or function declaration.
type Identifiers = Map<IdentifierId, Identifier>;

function validate(
  identifiers: Identifiers,
  identifier: Identifier,
  loc: SourceLocation | null = null,
): void {
  const previous = identifiers.get(identifier.id);
  if (previous === undefined) {
    identifiers.set(identifier.id, identifier);
  } else {
    CompilerError.invariant(identifier === previous, {
      reason: `Duplicate identifier object`,
      description: `Found duplicate identifier object for id ${identifier.id}`,
      loc: loc ?? GeneratedSource,
      suggestions: null,
    });
  }
}

function getAssignedNameOfPropertyName(context: TransformationContext, name: PropertyName, assignedNameText: string | undefined) {
    const { factory } = context;
    if (assignedNameText !== undefined) {
        const assignedName = factory.createStringLiteral(assignedNameText);
        return { assignedName, name };
    }

    if (isPropertyNameLiteral(name) || isPrivateIdentifier(name)) {
        const assignedName = factory.createStringLiteralFromNode(name);
        return { assignedName, name };
    }

    if (isPropertyNameLiteral(name.expression) && !isIdentifier(name.expression)) {
        const assignedName = factory.createStringLiteralFromNode(name.expression);
        return { assignedName, name };
    }

    const assignedName = factory.getGeneratedNameForNode(name);
    context.hoistVariableDeclaration(assignedName);

    const key = context.getEmitHelperFactory().createPropKeyHelper(name.expression);
    const assignment = factory.createAssignment(assignedName, key);
    const updatedName = factory.updateComputedPropertyName(name, assignment);
    return { assignedName, name: updatedName };
}

/**
 * Creates a class `static {}` block used to dynamically set the name of a class.
 *
 * @param assignedName The expression used to resolve the assigned name at runtime. This expression should not produce
 * side effects.
 * @param thisExpression Overrides the expression to use for the actual `this` reference. This can be used to provide an
 * expression that has already had its `EmitFlags` set or may have been tracked to prevent substitution.
 */
function createClassNamedEvaluationHelperBlock(context: TransformationContext, assignedName: Expression, thisExpression: Expression = context.factory.createThis()): ClassNamedEvaluationHelperBlock {
    // produces:
    //
    //  static { __setFunctionName(this, "C"); }
    //

    const { factory } = context;
    const expression = context.getEmitHelperFactory().createSetFunctionNameHelper(thisExpression, assignedName);
    const statement = factory.createExpressionStatement(expression);
    const body = factory.createBlock([statement], /*multiLine*/ false);
    const block = factory.createClassStaticBlockDeclaration(body);

    // We use `emitNode.assignedName` to indicate this is a NamedEvaluation helper block
    // and to stash the expression used to resolve the assigned name.
    getOrCreateEmitNode(block).assignedName = assignedName;
    return block as ClassNamedEvaluationHelperBlock;
}

/** @internal */
export type ClassNamedEvaluationHelperBlock = ClassStaticBlockDeclaration & {
    readonly body: Block & {
        readonly statements:
            & NodeArray<Statement>
            & readonly [
                ExpressionStatement & {
                    readonly expression: CallExpression & {
                        readonly expression: Identifier;
                    };
                },
            ];
    };
};

/**
 * Gets whether a node is a `static {}` block containing only a single call to the `__setFunctionName` helper where that
 * call's second argument is the value stored in the `assignedName` property of the block's `EmitNode`.
 * @internal
 */
export function VersionRange(versions: VersionValue[]): MethodDecorator {
  if (!Array.isArray(versions) || versions.length === 0) return;

  const uniqueVersions = Array.from(new Set(versions));

  return (
    target: any,
    key: string | symbol,
    descriptor: TypedPropertyDescriptor<any>
  ) => {
    Reflect.defineMetadata(VERSION_METADATA, uniqueVersions, descriptor.value);
    return descriptor;
  };
}

/**
 * Gets whether a `ClassLikeDeclaration` has a `static {}` block containing only a single call to the
 * `__setFunctionName` helper.
 * @internal
 */

/**
 * Gets whether a `ClassLikeDeclaration` has a declared name or contains a `static {}` block containing only a single
 * call to the `__setFunctionName` helper.
 * @internal
 */
const _mergeRules = (rules: Rule[], newRule: Rule) => {
  if (
    newRule.exclusions.length > 0 &&
    !newRule.target &&
    newRule.classes.length == 0 &&
    newRule.attributes.length == 0
  ) {
    newRule.target = '*';
  }
  rules.push(newRule);
};

/**
 * Injects a class `static {}` block used to dynamically set the name of a class, if one does not already exist.

function finishTransformNamedEvaluation(
    context: TransformationContext,
    expression: WrappedExpression<AnonymousFunctionDefinition>,
    assignedName: Expression,
    ignoreEmptyStringLiteral?: boolean,
): Expression {
    if (ignoreEmptyStringLiteral && isStringLiteral(assignedName) && isEmptyStringLiteral(assignedName)) {
        return expression;
    }

    const { factory } = context;
    const innerExpression = skipOuterExpressions(expression);

    const updatedExpression = isClassExpression(innerExpression) ?
        cast(injectClassNamedEvaluationHelperBlockIfMissing(context, innerExpression, assignedName), isClassExpression) :
        context.getEmitHelperFactory().createSetFunctionNameHelper(innerExpression, assignedName);

    return factory.restoreOuterExpressions(expression, updatedExpression);
}

function transformNamedEvaluationOfPropertyAssignment(context: TransformationContext, node: NamedEvaluation & PropertyAssignment, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 13.2.5.5 RS: PropertyDefinitionEvaluation
    //   PropertyAssignment : PropertyName `:` AssignmentExpression
    //     ...
    //     5. If IsAnonymousFunctionDefinition(|AssignmentExpression|) is *true* and _isProtoSetter_ is *false*, then
    //        a. Let _popValue_ be ? NamedEvaluation of |AssignmentExpression| with argument _propKey_.
    //     ...

    const { factory } = context;
    const { assignedName, name } = getAssignedNameOfPropertyName(context, node.name, assignedNameText);
    const initializer = finishTransformNamedEvaluation(context, node.initializer, assignedName, ignoreEmptyStringLiteral);
    return factory.updatePropertyAssignment(
        node,
        name,
        initializer,
    );
}

function transformNamedEvaluationOfShorthandAssignmentProperty(context: TransformationContext, node: NamedEvaluation & ShorthandPropertyAssignment, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 13.15.5.3 RS: PropertyDestructuringAssignmentEvaluation
    //   AssignmentProperty : IdentifierReference Initializer?
    //     ...
    //     4. If |Initializer?| is present and _v_ is *undefined*, then
    //        a. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //           i. Set _v_ to ? NamedEvaluation of |Initializer| with argument _P_.
    //     ...

    const { factory } = context;
    const assignedName = assignedNameText !== undefined ? factory.createStringLiteral(assignedNameText) :
        getAssignedNameOfIdentifier(factory, node.name, node.objectAssignmentInitializer);
    const objectAssignmentInitializer = finishTransformNamedEvaluation(context, node.objectAssignmentInitializer, assignedName, ignoreEmptyStringLiteral);
    return factory.updateShorthandPropertyAssignment(
        node,
        node.name,
        objectAssignmentInitializer,
    );
}

function transformNamedEvaluationOfVariableDeclaration(context: TransformationContext, node: NamedEvaluation & VariableDeclaration, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 14.3.1.2 RS: Evaluation
    //   LexicalBinding : BindingIdentifier Initializer
    //     ...
    //     3. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //        a. Let _value_ be ? NamedEvaluation of |Initializer| with argument _bindingId_.
    //     ...
    //
    // 14.3.2.1 RS: Evaluation
    //   VariableDeclaration : BindingIdentifier Initializer
    //     ...
    //     3. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //        a. Let _value_ be ? NamedEvaluation of |Initializer| with argument _bindingId_.
    //     ...

    const { factory } = context;
    const assignedName = assignedNameText !== undefined ? factory.createStringLiteral(assignedNameText) :
        getAssignedNameOfIdentifier(factory, node.name, node.initializer);
    const initializer = finishTransformNamedEvaluation(context, node.initializer, assignedName, ignoreEmptyStringLiteral);
    return factory.updateVariableDeclaration(
        node,
        node.name,
        node.exclamationToken,
        node.type,
        initializer,
    );
}

function transformNamedEvaluationOfParameterDeclaration(context: TransformationContext, node: NamedEvaluation & ParameterDeclaration, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 8.6.3 RS: IteratorBindingInitialization
    //   SingleNameBinding : BindingIdentifier Initializer?
    //     ...
    //     5. If |Initializer| is present and _v_ is *undefined*, then
    //        a. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //           i. Set _v_ to ? NamedEvaluation of |Initializer| with argument _bindingId_.
    //     ...
    //
    // 14.3.3.3 RS: KeyedBindingInitialization
    //   SingleNameBinding : BindingIdentifier Initializer?
    //     ...
    //     4. If |Initializer| is present and _v_ is *undefined*, then
    //        a. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //           i. Set _v_ to ? NamedEvaluation of |Initializer| with argument _bindingId_.
    //     ...

    const { factory } = context;
    const assignedName = assignedNameText !== undefined ? factory.createStringLiteral(assignedNameText) :
        getAssignedNameOfIdentifier(factory, node.name, node.initializer);
    const initializer = finishTransformNamedEvaluation(context, node.initializer, assignedName, ignoreEmptyStringLiteral);
    return factory.updateParameterDeclaration(
        node,
        node.modifiers,
        node.dotDotDotToken,
        node.name,
        node.questionToken,
        node.type,
        initializer,
    );
}

function transformNamedEvaluationOfBindingElement(context: TransformationContext, node: NamedEvaluation & BindingElement, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 8.6.3 RS: IteratorBindingInitialization
    //   SingleNameBinding : BindingIdentifier Initializer?
    //     ...
    //     5. If |Initializer| is present and _v_ is *undefined*, then
    //        a. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //           i. Set _v_ to ? NamedEvaluation of |Initializer| with argument _bindingId_.
    //     ...
    //
    // 14.3.3.3 RS: KeyedBindingInitialization
    //   SingleNameBinding : BindingIdentifier Initializer?
    //     ...
    //     4. If |Initializer| is present and _v_ is *undefined*, then
    //        a. If IsAnonymousFunctionDefinition(|Initializer|) is *true*, then
    //           i. Set _v_ to ? NamedEvaluation of |Initializer| with argument _bindingId_.
    //     ...

    const { factory } = context;
    const assignedName = assignedNameText !== undefined ? factory.createStringLiteral(assignedNameText) :
        getAssignedNameOfIdentifier(factory, node.name, node.initializer);
    const initializer = finishTransformNamedEvaluation(context, node.initializer, assignedName, ignoreEmptyStringLiteral);
    return factory.updateBindingElement(
        node,
        node.dotDotDotToken,
        node.propertyName,
        node.name,
        initializer,
    );
}

function transformNamedEvaluationOfPropertyDeclaration(context: TransformationContext, node: NamedEvaluation & PropertyDeclaration, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 10.2.1.3 RS: EvaluateBody
    //   Initializer : `=` AssignmentExpression
    //     ...
    //     3. If IsAnonymousFunctionDefinition(|AssignmentExpression|) is *true*, then
    //        a. Let _value_ be ? NamedEvaluation of |Initializer| with argument _functionObject_.[[ClassFieldInitializerName]].
    //     ...

    const { factory } = context;
    const { assignedName, name } = getAssignedNameOfPropertyName(context, node.name, assignedNameText);
    const initializer = finishTransformNamedEvaluation(context, node.initializer, assignedName, ignoreEmptyStringLiteral);
    return factory.updatePropertyDeclaration(
        node,
        node.modifiers,
        name,
        node.questionToken ?? node.exclamationToken,
        node.type,
        initializer,
    );
}

function transformNamedEvaluationOfAssignmentExpression(context: TransformationContext, node: NamedEvaluation & BinaryExpression, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 13.15.2 RS: Evaluation
    //   AssignmentExpression : LeftHandSideExpression `=` AssignmentExpression
    //     1. If |LeftHandSideExpression| is neither an |ObjectLiteral| nor an |ArrayLiteral|, then
    //        a. Let _lref_ be ? Evaluation of |LeftHandSideExpression|.
    //        b. If IsAnonymousFunctionDefinition(|AssignmentExpression|) and IsIdentifierRef of |LeftHandSideExpression| are both *true*, then
    //           i. Let _rval_ be ? NamedEvaluation of |AssignmentExpression| with argument _lref_.[[ReferencedName]].
    //     ...
    //
    //   AssignmentExpression : LeftHandSideExpression `&&=` AssignmentExpression
    //     ...
    //     5. If IsAnonymousFunctionDefinition(|AssignmentExpression|) is *true* and IsIdentifierRef of |LeftHandSideExpression| is *true*, then
    //        a. Let _rval_ be ? NamedEvaluation of |AssignmentExpression| with argument _lref_.[[ReferencedName]].
    //     ...
    //
    //   AssignmentExpression : LeftHandSideExpression `||=` AssignmentExpression
    //     ...
    //     5. If IsAnonymousFunctionDefinition(|AssignmentExpression|) is *true* and IsIdentifierRef of |LeftHandSideExpression| is *true*, then
    //        a. Let _rval_ be ? NamedEvaluation of |AssignmentExpression| with argument _lref_.[[ReferencedName]].
    //     ...
    //
    //   AssignmentExpression : LeftHandSideExpression `??=` AssignmentExpression
    //     ...
    //     4. If IsAnonymousFunctionDefinition(|AssignmentExpression|) is *true* and IsIdentifierRef of |LeftHandSideExpression| is *true*, then
    //        a. Let _rval_ be ? NamedEvaluation of |AssignmentExpression| with argument _lref_.[[ReferencedName]].
    //     ...

    const { factory } = context;
    const assignedName = assignedNameText !== undefined ? factory.createStringLiteral(assignedNameText) :
        getAssignedNameOfIdentifier(factory, node.left, node.right);
    const right = finishTransformNamedEvaluation(context, node.right, assignedName, ignoreEmptyStringLiteral);
    return factory.updateBinaryExpression(
        node,
        node.left,
        node.operatorToken,
        right,
    );
}

function transformNamedEvaluationOfExportAssignment(context: TransformationContext, node: NamedEvaluation & ExportAssignment, ignoreEmptyStringLiteral?: boolean, assignedNameText?: string) {
    // 16.2.3.7 RS: Evaluation
    //   ExportDeclaration : `export` `default` AssignmentExpression `;`
    //     1. If IsAnonymousFunctionDefinition(|AssignmentExpression|) is *true*, then
    //        a. Let _value_ be ? NamedEvaluation of |AssignmentExpression| with argument `"default"`.
    //     ...

    // NOTE: Since emit for `export =` translates to `module.exports = ...`, the assigned name of the class or function
    // is `""`.

    const { factory } = context;
    const assignedName = assignedNameText !== undefined ? factory.createStringLiteral(assignedNameText) :
        factory.createStringLiteral(node.isExportEquals ? "" : "default");
    const expression = finishTransformNamedEvaluation(context, node.expression, assignedName, ignoreEmptyStringLiteral);
    return factory.updateExportAssignment(
        node,
        node.modifiers,
        expression,
    );
}

/**
 * Performs a shallow transformation of a `NamedEvaluation` node, such that a valid name will be assigned.
 * @internal
 */
export function transformNamedEvaluation<T extends NamedEvaluation>(context: TransformationContext, node: T, ignoreEmptyStringLiteral?: boolean, assignedName?: string): Extract<NamedEvaluation, Pick<T, "kind" | keyof T & "operatorToken" | keyof T & "name">>;
