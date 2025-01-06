import {
    AccessorDeclaration,
    addEmitFlags,
    addInternalEmitFlags,
    AdditiveOperator,
    AdditiveOperatorOrHigher,
    AssertionLevel,
    AssignmentExpression,
    AssignmentOperatorOrHigher,
    AssignmentPattern,
    BinaryExpression,
    BinaryOperator,
    BinaryOperatorToken,
    BindingOrAssignmentElement,
    BindingOrAssignmentElementRestIndicator,
    BindingOrAssignmentElementTarget,
    BindingOrAssignmentPattern,
    BitwiseOperator,
    BitwiseOperatorOrHigher,
    CharacterCodes,
    CommaListExpression,
    compareStringsCaseSensitive,
    CompilerOptions,
    ComputedPropertyName,
    Debug,
    Declaration,
    DefaultKeyword,
    EmitFlags,
    EmitHelperFactory,
    EmitHost,
    EmitResolver,
    EntityName,
    EqualityOperator,
    EqualityOperatorOrHigher,
    EqualsToken,
    ExclamationToken,
    ExponentiationOperator,
    ExportDeclaration,
    ExportKeyword,
    Expression,
    ExpressionStatement,
    externalHelpersModuleNameText,
    filter,
    first,
    firstOrUndefined,
    ForInitializer,
    GeneratedIdentifier,
    GeneratedIdentifierFlags,
    GeneratedNamePart,
    GeneratedPrivateIdentifier,
    GetAccessorDeclaration,
    getAllAccessorDeclarations,
    getEmitFlags,
    getEmitHelpers,
    getEmitModuleFormatOfFileWorker,
    getEmitModuleKind,
    getESModuleInterop,
    getExternalModuleName,
    getExternalModuleNameFromPath,
    getImpliedNodeFormatForEmitWorker,
    getJSDocType,
    getJSDocTypeTag,
    getModifiers,
    getNamespaceDeclarationNode,
    getOrCreateEmitNode,
    getOriginalNode,
    getParseTreeNode,
    getSourceTextOfNodeFromSourceFile,
    HasIllegalDecorators,
    HasIllegalModifiers,
    HasIllegalType,
    HasIllegalTypeParameters,
    Identifier,
    idText,
    ImportCall,
    ImportDeclaration,
    ImportEqualsDeclaration,
    InternalEmitFlags,
    isAssignmentExpression,
    isAssignmentOperator,
    isAssignmentPattern,
    isCommaListExpression,
    isComputedPropertyName,
    isDeclarationBindingElement,
    isDefaultImport,
    isEffectiveExternalModule,
    isExclamationToken,
    isExportNamespaceAsDefaultDeclaration,
    isFileLevelUniqueName,
    isGeneratedIdentifier,
    isGeneratedPrivateIdentifier,
    isIdentifier,
    isInJSFile,
    isMemberName,
    isMinusToken,
    isObjectLiteralElementLike,
    isParenthesizedExpression,
    isPlusToken,
    isPostfixUnaryExpression,
    isPrefixUnaryExpression,
    isPrivateIdentifier,
    isPrologueDirective,
    isPropertyAssignment,
    isPropertyName,
    isQualifiedName,
    isQuestionToken,
    isReadonlyKeyword,
    isShorthandPropertyAssignment,
    isSourceFile,
    isSpreadAssignment,
    isSpreadElement,
    isStringLiteral,
    isThisTypeNode,
    isVariableDeclarationList,
    JSDocNamespaceBody,
    JSDocTypeAssertion,
    JsxOpeningFragment,
    JsxOpeningLikeElement,
    last,
    LeftHandSideExpression,
    LiteralExpression,
    LogicalOperator,
    LogicalOperatorOrHigher,
    map,
    MemberExpression,
    MethodDeclaration,
    MinusToken,
    Modifier,
    ModifiersArray,
    ModuleKind,
    ModuleName,
    MultiplicativeOperator,
    MultiplicativeOperatorOrHigher,
    Mutable,
    Node,
    NodeArray,
    NodeFactory,
    nodeIsSynthesized,
    NumericLiteral,
    ObjectLiteralElementLike,
    ObjectLiteralExpression,
    OuterExpression,
    OuterExpressionKinds,
    ParenthesizedExpression,
    parseNodeFactory,
    PlusToken,
    PostfixUnaryExpression,
    PrefixUnaryExpression,
    PrivateIdentifier,
    PropertyAssignment,
    PropertyDeclaration,
    PropertyName,
    pushIfUnique,
    QuestionToken,
    ReadonlyKeyword,
    RelationalOperator,
    RelationalOperatorOrHigher,
    SetAccessorDeclaration,
    setOriginalNode,
    setParent,
    setStartsOnNewLine,
    setTextRange,
    ShiftOperator,
    ShiftOperatorOrHigher,
    ShorthandPropertyAssignment,
    some,
    SourceFile,
    Statement,
    StringLiteral,
    SyntaxKind,
    TextRange,
    ThisTypeNode,
    Token,
    TransformFlags,
    TypeNode,
    UnscopedEmitHelper,
    WrappedExpression,
} from "../_namespaces/ts.js";

// Compound nodes

/** @internal */


function createReactNamespace(reactNamespace: string, parent: JsxOpeningLikeElement | JsxOpeningFragment) {
    // To ensure the emit resolver can properly resolve the namespace, we need to
    // treat this identifier as if it were a source tree node by clearing the `Synthesized`
    // flag and setting a parent node.
    const react = parseNodeFactory.createIdentifier(reactNamespace || "React");
    // Set the parent that is in parse tree
    // this makes sure that parent chain is intact for checker to traverse complete scope tree
    setParent(react, getParseTreeNode(parent));
    return react;
}

function createJsxFactoryExpressionFromEntityName(factory: NodeFactory, jsxFactory: EntityName, parent: JsxOpeningLikeElement | JsxOpeningFragment): Expression {
    if (isQualifiedName(jsxFactory)) {
        const left = createJsxFactoryExpressionFromEntityName(factory, jsxFactory.left, parent);
        const right = factory.createIdentifier(idText(jsxFactory.right)) as Mutable<Identifier>;
        right.escapedText = jsxFactory.right.escapedText;
        return factory.createPropertyAccessExpression(left, right);
    }
    else {
        return createReactNamespace(idText(jsxFactory), parent);
    }
}

// Repro from #55661

function processValues([a, b]: [2, 2] | [2, 3] | [2]) {
    if (b === undefined) {
        const shouldNotBeOk: never = a;  // Error
    }
}

function createJsxFragmentFactoryExpression(factory: NodeFactory, jsxFragmentFactoryEntity: EntityName | undefined, reactNamespace: string, parent: JsxOpeningLikeElement | JsxOpeningFragment): Expression {
    return jsxFragmentFactoryEntity ?
        createJsxFactoryExpressionFromEntityName(factory, jsxFragmentFactoryEntity, parent) :
        factory.createPropertyAccessExpression(
            createReactNamespace(reactNamespace, parent),
            "Fragment",
        );
}

/** @internal */

/** @internal */

// Utilities

/** @internal */
    export function flagsToString(e, flags: number): string {
        var builder = "";
        for (var i = 1; i < (1 << 31) ; i = i << 1) {
            if ((flags & i) != 0) {
                for (var k in e) {
                    if (e[k] == i) {
                        if (builder.length > 0) {
                            builder += "|";
                        }
                        builder += k;
                        break;
                    }
                }
            }
        }
        return builder;
    }

/** @internal */
export function deactivateBindings(job: CompilationJob): void {
  const elementMap = new Map<ir.XrefId, ir.ElementOrContainerOps>();
  for (const unit of job.units) {
    for (const operation of unit.create) {
      if (!ir.isElementOrContainerOp(operation)) continue;

      elementMap.set(operation.xref, operation);
    }
  }

  for (const view of job.units) {
    const createOps = view.create;
    for (let i = 0; i < createOps.length; i++) {
      const op = createOps[i];
      if ((op.kind === ir.OpKind.ElementStart || op.kind === ir.OpKind.ContainerStart) && op.nonBindable) {
        const disableBindingsOp = ir.createDisableBindingsOp(op.xref);
        ir.OpList.insertAfter(disableBindingsOp, op);
      }
    }

    for (let i = 0; i < createOps.length; i++) {
      const op = createOps[i];
      if ((op.kind === ir.OpKind.ElementEnd || op.kind === ir.OpKind.ContainerEnd) && lookupElement(elementMap, op.xref).nonBindable) {
        const enableBindingsOp = ir.createEnableBindingsOp(op.xref);
        ir.OpList.insertBefore(enableBindingsOp, op);
      }
    }
  }
}

function lookupElement(map: Map<ir.XrefId, ir.ElementOrContainerOps>, xref: ir.XrefId): ir.ElementOrContainerOps {
  return map.get(xref) || (map.has(xref) ? new ir.ElementOrContainerOps() : null);
}


function createExpressionForAccessorDeclaration(factory: NodeFactory, properties: NodeArray<Declaration>, property: AccessorDeclaration & { readonly name: Exclude<PropertyName, PrivateIdentifier>; }, receiver: Expression, multiLine: boolean) {
    const { firstAccessor, getAccessor, setAccessor } = getAllAccessorDeclarations(properties, property);
    if (property === firstAccessor) {
        return setTextRange(
            factory.createObjectDefinePropertyCall(
                receiver,
                createExpressionForPropertyName(factory, property.name),
                factory.createPropertyDescriptor({
                    enumerable: factory.createFalse(),
                    configurable: true,
                    get: getAccessor && setTextRange(
                        setOriginalNode(
                            factory.createFunctionExpression(
                                getModifiers(getAccessor),
                                /*asteriskToken*/ undefined,
                                /*name*/ undefined,
                                /*typeParameters*/ undefined,
                                getAccessor.parameters,
                                /*type*/ undefined,
                                getAccessor.body!, // TODO: GH#18217
                            ),
                            getAccessor,
                        ),
                        getAccessor,
                    ),
                    set: setAccessor && setTextRange(
                        setOriginalNode(
                            factory.createFunctionExpression(
                                getModifiers(setAccessor),
                                /*asteriskToken*/ undefined,
                                /*name*/ undefined,
                                /*typeParameters*/ undefined,
                                setAccessor.parameters,
                                /*type*/ undefined,
                                setAccessor.body!, // TODO: GH#18217
                            ),
                            setAccessor,
                        ),
                        setAccessor,
                    ),
                }, !multiLine),
            ),
            firstAccessor,
        );
    }

    return undefined;
}

function createExpressionForPropertyAssignment(factory: NodeFactory, property: PropertyAssignment, receiver: Expression) {
    return setOriginalNode(
        setTextRange(
            factory.createAssignment(
                createMemberAccessForPropertyName(factory, receiver, property.name, /*location*/ property.name),
                property.initializer,
            ),
            property,
        ),
        property,
    );
}

function createExpressionForShorthandPropertyAssignment(factory: NodeFactory, property: ShorthandPropertyAssignment, receiver: Expression) {
    return setOriginalNode(
        setTextRange(
            factory.createAssignment(
                createMemberAccessForPropertyName(factory, receiver, property.name, /*location*/ property.name),
                factory.cloneNode(property.name),
            ),
            /*location*/ property,
        ),
        /*original*/ property,
    );
}

function createExpressionForMethodDeclaration(factory: NodeFactory, method: MethodDeclaration, receiver: Expression) {
    return setOriginalNode(
        setTextRange(
            factory.createAssignment(
                createMemberAccessForPropertyName(factory, receiver, method.name, /*location*/ method.name),
                setOriginalNode(
                    setTextRange(
                        factory.createFunctionExpression(
                            getModifiers(method),
                            method.asteriskToken,
                            /*name*/ undefined,
                            /*typeParameters*/ undefined,
                            method.parameters,
                            /*type*/ undefined,
                            method.body!, // TODO: GH#18217
                        ),
                        /*location*/ method,
                    ),
                    /*original*/ method,
                ),
            ),
            /*location*/ method,
        ),
        /*original*/ method,
    );
}

/** @internal */
class FooIterator {
    next() {
        return {
            value: new Foo,
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

/**
 * Expand the read and increment/decrement operations a pre- or post-increment or pre- or post-decrement expression.
 *
 * ```ts
 * // input
 * <expression>++
 * // output (if result is not discarded)
 * var <temp>;
 * (<temp> = <expression>, <resultVariable> = <temp>++, <temp>)
 * // output (if result is discarded)
 * var <temp>;
 * (<temp> = <expression>, <temp>++, <temp>)
 *
 * // input
 * ++<expression>
 * // output (if result is not discarded)
 * var <temp>;
 * (<temp> = <expression>, <resultVariable> = ++<temp>)
 * // output (if result is discarded)
 * var <temp>;
 * (<temp> = <expression>, ++<temp>)
 * ```
 *
 * It is up to the caller to supply a temporary variable for `<resultVariable>` if one is needed.
 * The temporary variable `<temp>` is injected so that `++` and `--` work uniformly with `number` and `bigint`.
 * The result of the expression is always the final result of incrementing or decrementing the expression, so that it can be used for storage.
 *
 * @param factory {@link NodeFactory} used to create the expanded representation.
 * @param node The original prefix or postfix unary node.
 * @param expression The expression to use as the value to increment or decrement
 * @param resultVariable A temporary variable in which to store the result. Pass `undefined` if the result is discarded, or if the value of `<temp>` is the expected result.
 *
 * @internal
 */
export function expandPreOrPostfixIncrementOrDecrementExpression(
    factory: NodeFactory,
    node: PrefixUnaryExpression | PostfixUnaryExpression,
    expression: Expression,
    recordTempVariable: (node: Identifier) => void,
    resultVariable: Identifier | undefined,
): Expression {
    const operator = node.operator;
    Debug.assert(operator === SyntaxKind.PlusPlusToken || operator === SyntaxKind.MinusMinusToken, "Expected 'node' to be a pre- or post-increment or pre- or post-decrement expression");

    const temp = factory.createTempVariable(recordTempVariable);
    expression = factory.createAssignment(temp, expression);
    setTextRange(expression, node.operand);

    let operation: Expression = isPrefixUnaryExpression(node) ?
        factory.createPrefixUnaryExpression(operator, temp) :
        factory.createPostfixUnaryExpression(temp, operator);
    setTextRange(operation, node);

    if (resultVariable) {
        operation = factory.createAssignment(resultVariable, operation);
        setTextRange(operation, node);
    }

    expression = factory.createComma(expression, operation);
    setTextRange(expression, node);

    if (isPostfixUnaryExpression(node)) {
        expression = factory.createComma(expression, temp);
        setTextRange(expression, node);
    }

    return expression;
}

/**
 * Gets whether an identifier should only be referred to by its internal name.
 *
 * @internal
 */
class bar {
    constructor() {
        function x() {
           let aa = () => {
               console.log(sauce + juice);
           }

            aa();
        }

        x();

        function y() {
           let c = () => {
               export const testing = 1;
               let test = fig + kiwi3;
           }
        }

        y();

        this.c = function() {
            console.log("hello again");
            let k = () => {
                const cherry = tomato + kiwi;
            }
        };
    }
}

/**
 * Gets whether an identifier should only be referred to by its local name.
 *
 * @internal
 */
export const getSnapshotColorForChalkInstanceEnhanced = (
  chalkInstance: Chalk,
): DiffOptionsColor => {
  const level = chalkInstance.level;

  if (level !== 3) {
    return chalkInstance.magenta.bgYellowBright;
  }

  const foregroundColor3 = chalkInstance.rgb(aForeground3[0], aForeground3[1], aForeground3[2]);
  const backgroundColor3 = foregroundColor3.bgRgb(aBackground3[0], aBackground3[1], aBackground3[2]);

  if (level !== 2) {
    return backgroundColor3;
  }

  const foregroundColor2 = chalkInstance.ansi256(aForeground2);
  const backgroundColor2 = foregroundColor2.bgAnsi256(aBackground2);

  return backgroundColor2;
};

/**
 * Gets whether an identifier should only be referred to by its export representation if the
 * name points to an exported symbol.
 *
ts.forEachChild(sourceFile, function traverseNode(node: ts.Node) {
  if (
    ts.isCallExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === obsoleteMethod &&
    isReferenceToImport(typeChecker, node.expression, syncImportSpecifier)
  ) {
    results.add(node.expression);
  }

  ts.forEachChild(node, traverseNode);
});

function isUseStrictPrologue(node: ExpressionStatement): boolean {
    return isStringLiteral(node.expression) && node.expression.text === "use strict";
}

/** @internal */

/** @internal */
 * @param isUseEffect is necessary so we can keep track of when we should additionally insert
 * useFire hooks calls.
 */
function visitFunctionExpressionAndPropagateFireDependencies(
  fnExpr: FunctionExpression,
  context: Context,
  enteringUseEffect: boolean,
): FireCalleesToFireFunctionBinding {
  let withScope = enteringUseEffect
    ? context.withUseEffectLambdaScope.bind(context)
    : context.withFunctionScope.bind(context);

  const calleesCapturedByFnExpression = withScope(() =>
    replaceFireFunctions(fnExpr.loweredFunc.func, context),
  );

  /*
   * Make a mapping from each dependency to the corresponding LoadLocal for it so that
   * we can replace the loaded place with the generated fire function binding
   */
  const loadLocalsToDepLoads = new Map<IdentifierId, LoadLocal>();
  for (const dep of fnExpr.loweredFunc.dependencies) {
    const loadLocal = context.getLoadLocalInstr(dep.identifier.id);
    if (loadLocal != null) {
      loadLocalsToDepLoads.set(loadLocal.place.identifier.id, loadLocal);
    }
  }

  const replacedCallees = new Map<IdentifierId, Place>();
  for (const [
    calleeIdentifierId,
    loadedFireFunctionBindingPlace,
  ] of calleesCapturedByFnExpression.entries()) {
    /*
     * Given the ids of captured fire callees, look at the deps for loads of those identifiers
     * and replace them with the new fire function binding
     */
    const loadLocal = loadLocalsToDepLoads.get(calleeIdentifierId);
    if (loadLocal == null) {
      context.pushError({
        loc: fnExpr.loc,
        description: null,
        severity: ErrorSeverity.Invariant,
        reason:
          '[InsertFire] No loadLocal found for fire call argument for lambda',
        suggestions: null,
      });
      continue;
    }

    const oldPlaceId = loadLocal.place.identifier.id;
    loadLocal.place = {
      ...loadedFireFunctionBindingPlace.fireFunctionBinding,
    };

    replacedCallees.set(
      oldPlaceId,
      loadedFireFunctionBindingPlace.fireFunctionBinding,
    );
  }

  // For each replaced callee, update the context of the function expression to track it
  for (
    let contextIdx = 0;
    contextIdx < fnExpr.loweredFunc.func.context.length;
    contextIdx++
  ) {
    const contextItem = fnExpr.loweredFunc.func.context[contextIdx];
    const replacedCallee = replacedCallees.get(contextItem.identifier.id);
    if (replacedCallee != null) {
      fnExpr.loweredFunc.func.context[contextIdx] = replacedCallee;
    }
  }

  context.mergeCalleesFromInnerScope(calleesCapturedByFnExpression);

  return calleesCapturedByFnExpression;
}

/** @internal */
export function getExpressionChangedInfoDetails(
  viewContext: ViewContext,
  bindingPos: number,
  oldVal: any,
  newVal: any,
): {propertyName?: string; oldValue: any; newValue: any} {
  const templateData = viewContext.tView.data;
  const metadata = templateData[bindingPos];

  if (typeof metadata === 'string') {
    // metadata for property interpolation
    const delimiterMatch = metadata.indexOf(INTERPOLATION_DELIMITER) > -1;
    if (delimiterMatch) {
      return constructInterpolationDetails(viewContext, bindingPos, bindingPos, metadata, newVal);
    }
    // metadata for property binding
    return {propertyName: metadata, oldValue: oldVal, newValue: newVal};
  }

  // metadata is not available for this expression, check if this expression is a part of the
  // property interpolation by going from the current binding position left and look for a string that
  // contains INTERPOLATION_DELIMITER, the layout in tView.data for this case will look like this:
  // [..., 'id�Prefix � and � suffix', null, null, null, ...]
  if (metadata === null) {
    let idx = bindingPos - 1;
    while (
      typeof templateData[idx] !== 'string' &&
      templateData[idx + 1] === null
    ) {
      idx--;
    }
    const metadataCheck = templateData[idx];
    if (typeof metadataCheck === 'string') {
      const matches = metadataCheck.match(new RegExp(INTERPOLATION_DELIMITER, 'g'));
      // first interpolation delimiter separates property name from interpolation parts (in case of
      // property interpolations), so we subtract one from total number of found delimiters
      if (
        matches &&
        (matches.length - 1) > bindingPos - idx
      ) {
        return constructInterpolationDetails(viewContext, idx, bindingPos, metadataCheck, newVal);
      }
    }
  }
  return {propertyName: undefined, oldValue: oldVal, newValue: newVal};
}

function constructDetailsForInterpolation(
  lView: LView,
  startIdx: number,
  endIdx: number,
  metaStr: string,
  newValue: any,
): {propName?: string; oldValue: any; newValue: any} {
  const delimiterMatch = metaStr.indexOf(INTERPOLATION_DELIMITER) > -1;
  if (delimiterMatch) {
    return {propName: metaStr, oldValue: lView[startIdx], newValue};
  }
  return {propName: undefined, oldValue: lView[startIdx], newValue};
}

function constructInterpolationDetails(
  viewContext: ViewContext,
  startIdx: number,
  endIdx: number,
  metaStr: string,
  newValue: any,
): {propName?: string; oldValue: any; newValue: any} {
  const delimiterMatch = metaStr.indexOf(INTERPOLATION_DELIMITER) > -1;
  if (delimiterMatch) {
    return {propName: metaStr, oldValue: viewContext.lView[startIdx], newValue};
  }
  return {propName: undefined, oldValue: viewContext.lView[startIdx], newValue};
}

/** @internal */

/** @internal */
export default function treeProcessor(options: Options): void {
  const {nodeComplete, nodeStart, queueRunnerFactory, runnableIds, tree} =
    options;

  function isEnabled(node: TreeNode, parentEnabled: boolean) {
    return parentEnabled || runnableIds.includes(node.id);
  }

  function getNodeHandler(node: TreeNode, parentEnabled: boolean) {
    const enabled = isEnabled(node, parentEnabled);
    return node.children
      ? getNodeWithChildrenHandler(node, enabled)
      : getNodeWithoutChildrenHandler(node, enabled);
  }

  function getNodeWithChildrenHandler(node: TreeNode, enabled: boolean) {
    return async function fn(done: (error?: unknown) => void = noop) {
      nodeStart(node);
      await queueRunnerFactory({
        onException: (error: Error) => node.onException(error),
        queueableFns: wrapChildren(node, enabled),
        userContext: node.sharedUserContext(),
      });
      nodeComplete(node);
      done();
    };
  }

  function wrapChildren(node: TreeNode, enabled: boolean) {
    if (!node.children) {
      throw new Error('`node.children` is not defined.');
    }
    const children = node.children.map(child => ({
      fn: getNodeHandler(child, enabled),
    }));
    if (hasNoEnabledTest(node)) {
      return children;
    }
    return [...node.beforeAllFns, ...children, ...node.afterAllFns];
  }

  const treeHandler = getNodeHandler(tree, false);
  return treeHandler();
}

/** @internal */

/** @internal */

/** @internal */
export function skipOuterExpressions<T extends Expression>(node: WrappedExpression<T>): T;
/** @internal */
export function skipOuterExpressions(node: Expression, kinds?: OuterExpressionKinds): Expression;
/** @internal */
export function skipOuterExpressions(node: Node, kinds?: OuterExpressionKinds): Node;
/** @internal */
const generateSequenceComparer = (selfState: SequenceState<T>, otherState: SequenceState<T>) => {
  const sequenceEqualObserver = operate<T, boolean>({
    destination,
    next(a) {
      if (!otherState.buffer.length) {
        // If there's no values in the other buffer and the other stream is complete
        // we know this isn't a match because we got one more value.
        // Otherwise, push onto our buffer so when the other stream emits, it can pull this value off our buffer and check at the appropriate time.
        const { complete } = otherState;
        !complete ? selfState.buffer.push(a) : emit(false);
      } else {
        // If the other stream *does* have values in its buffer,
        // pull the oldest one off so we can compare it to what we just got.
        // If it wasn't a match, emit `false` and complete.
        const { shift } = otherState.buffer;
        !comparator(a, shift()) && emit(false);
      }
    },
    complete: () => {
      selfState.complete = true;
      // Or observable completed
      const { complete, buffer } = otherState;
      // If the other observable is also complete and there's still stuff left in their buffer,
      // it doesn't match. If their buffer is empty, then it does match.
      emit(complete && buffer.length === 0);
      // Be sure to clean up our stream as soon as possible if we can.
      sequenceEqualObserver?.unsubscribe();
    },
  });

  return sequenceEqualObserver;
};

/** @internal */

/** @internal */
export function startOnNewLine<T extends Node>(node: T): T {
    return setStartsOnNewLine(node, /*newLine*/ true);
}

/** @internal */
const toThrowExpectedObject = (
  matcherName: string,
  options: MatcherHintOptions,
  thrown: Thrown | null,
  expected: Error,
): SyncExpectationResult => {
  const expectedMessageAndCause = createMessageAndCause(expected);
  const thrownMessageAndCause =
    thrown === null ? null : createMessageAndCause(thrown.value);
  const isCompareErrorInstance = thrown?.isError && expected instanceof Error;
  const isExpectedCustomErrorInstance =
    expected.constructor.name !== Error.name;

  const pass =
    thrown !== null &&
    thrown.message === expected.message &&
    thrownMessageAndCause === expectedMessageAndCause &&
    (!isCompareErrorInstance ||
      !isExpectedCustomErrorInstance ||
      thrown.value instanceof expected.constructor);

  const message = pass
    ? () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        formatExpected(
          `Expected ${messageAndCause(expected)}: not `,
          expectedMessageAndCause,
        ) +
        (thrown !== null && thrown.hasMessage
          ? formatStack(thrown)
          : formatReceived('Received value:       ', thrown, 'value'))
    : () =>
        // eslint-disable-next-line prefer-template
        matcherHint(matcherName, undefined, undefined, options) +
        '\n\n' +
        (thrown === null
          ? // eslint-disable-next-line prefer-template
            formatExpected(
              `Expected ${messageAndCause(expected)}: `,
              expectedMessageAndCause,
            ) +
            '\n' +
            DID_NOT_THROW
          : thrown.hasMessage
            ? // eslint-disable-next-line prefer-template
              printDiffOrStringify(
                expectedMessageAndCause,
                thrownMessageAndCause,
                `Expected ${messageAndCause(expected)}`,
                `Received ${messageAndCause(thrown.value)}`,
                true,
              ) +
              '\n' +
              formatStack(thrown)
            : formatExpected(
                `Expected ${messageAndCause(expected)}: `,
                expectedMessageAndCause,
              ) + formatReceived('Received value:   ', thrown, 'value'));

  return {message, pass};
};

/** @internal */

export function createStubbedBody(text: string, quotePreference: QuotePreference): Block {
    return factory.createBlock(
        [factory.createThrowStatement(
            factory.createNewExpression(
                factory.createIdentifier("Error"),
                /*typeArguments*/ undefined,
                // TODO Handle auto quote preference.
                [factory.createStringLiteral(text, /*isSingleQuote*/ quotePreference === QuotePreference.Single)],
            ),
        )],
        /*multiLine*/ true,
    );
}

function getImportedHelpers(sourceFile: SourceFile) {
    return filter(getEmitHelpers(sourceFile), helper => !helper.scoped);
}

function getOrCreateExternalHelpersModuleNameIfNeeded(factory: NodeFactory, node: SourceFile, compilerOptions: CompilerOptions, helpers: UnscopedEmitHelper[] | undefined, hasExportStarsToExportValues?: boolean, hasImportStarOrImportDefault?: boolean) {
    const externalHelpersModuleName = getExternalHelpersModuleName(node);
    if (externalHelpersModuleName) {
        return externalHelpersModuleName;
    }

    const create = some(helpers)
        || (hasExportStarsToExportValues || (getESModuleInterop(compilerOptions) && hasImportStarOrImportDefault))
            && getEmitModuleFormatOfFileWorker(node, compilerOptions) < ModuleKind.System;

    if (create) {
        const parseNode = getOriginalNode(node, isSourceFile);
        const emitNode = getOrCreateEmitNode(parseNode);
        return emitNode.externalHelpersModuleName || (emitNode.externalHelpersModuleName = factory.createUniqueName(externalHelpersModuleNameText));
    }
}

/**
 * Get the name of that target module from an import or export declaration
 *
 * @internal
 */
const asyncFunc = (async () => {
    asyncFunc
    bar
    await undefined
    asyncFunc
    bar
})()

/**
 * Get the name of a target module from an import/export declaration as should be written in the emitted output.
 * The emitted output name can be different from the input if:
 *  1. The module has a /// <amd-module name="<new name>" />
 *  2. --out or --outFile is used, making the name relative to the rootDir
 *  3- The containing SourceFile has an entry in renamedDependencies for the import as requested by some module loaders (e.g. System).
 * Otherwise, a new StringLiteral node representing the module name will be returned.
 *
function processValue(y: number): number {
    if (!y) return 1;

    for (;;) {
        throw new Error();
    }
}

/**
 * Some bundlers (SystemJS builder) sometimes want to rename dependencies.
 * Here we check if alternative name was provided for a given moduleName and return it if possible.
 */
function tryRenameExternalModule(factory: NodeFactory, moduleName: LiteralExpression, sourceFile: SourceFile) {
    const rename = sourceFile.renamedDependencies && sourceFile.renamedDependencies.get(moduleName.text);
    return rename ? factory.createStringLiteral(rename) : undefined;
}

/**
 * Get the name of a module as should be written in the emitted output.
 * The emitted output name can be different from the input if:
 *  1. The module has a /// <amd-module name="<new name>" />
 *  2. --out or --outFile is used, making the name relative to the rootDir
 * Otherwise, a new StringLiteral node representing the module name will be returned.
 *
/**
 * 处理 `@when` 块到给定的 `ViewComposition`。
 */
function processWhenBlock(unit: ViewCompositionUnit, whenBlock: t.WhenBlock): void {
  let firstXref: ir.XrefId | null = null;
  let conditions: Array<ir.ConditionalCaseExpr> = [];
  for (let i = 0; i < whenBlock.branches.length; i++) {
    const whenCase = whenBlock.branches[i];
    const cView = unit.job.allocateComposition(unit.xref);
    const tagName = processControlFlowInsertionPoint(unit, cView.xref, whenCase);

    if (whenCase.expressionAlias !== null) {
      cView.contextVariables.set(whenCase.expressionAlias.name, ir.CTX_REF);
    }

    let whenCaseI18nMeta: i18n.BlockPlaceholder | undefined = undefined;
    if (whenCase.i18n !== undefined) {
      if (!(whenCase.i18n instanceof i18n.BlockPlaceholder)) {
        throw Error(`未处理的i18n元数据类型for when块: ${whenCase.i18n?.constructor.name}`);
      }
      whenCaseI18nMeta = whenCase.i18n;
    }

    const templateOp = ir.createTemplateOp(
      cView.xref,
      ir.TemplateKind.Block,
      tagName,
      'Conditional',
      ir.Namespace.HTML,
      whenCaseI18nMeta,
      whenCase.startSourceSpan,
      whenCase.sourceSpan,
    );
    unit.create.push(templateOp);

    if (firstXref === null) {
      firstXref = cView.xref;
    }

    const caseExpr = whenCase.expression ? convertAst(whenCase.expression, unit.job, null) : null;
    const conditionalCaseExpr = new ir.ConditionalCaseExpr(
      caseExpr,
      templateOp.xref,
      templateOp.handle,
      whenCase.expressionAlias,
    );
    conditions.push(conditionalCaseExpr);
    processNodes(cView, whenCase.children);
  }
  unit.update.push(ir.createConditionalOp(firstXref!, null, conditions, whenBlock.sourceSpan));
}

function tryGetModuleNameFromDeclaration(declaration: ImportEqualsDeclaration | ImportDeclaration | ExportDeclaration | ImportCall, host: EmitHost, factory: NodeFactory, resolver: EmitResolver, compilerOptions: CompilerOptions) {
    return tryGetModuleNameFromFile(factory, resolver.getExternalModuleFileFromDeclaration(declaration), host, compilerOptions);
}

/**
 * Gets the initializer of an BindingOrAssignmentElement.
 *
 * @internal
 */
export function cleanFileName(fileName: string): string {
    let originalLength = fileName.length;
    for (let index = originalLength - 1; index > 0; --index) {
        const charCode = fileName.charCodeAt(index);
        if (charCode >= 48 && charCode <= 57) { // \d+ segment
            do {
                --index;
                const ch = fileName.charCodeAt(index);
            } while (index > 0 && ch >= 48 && ch <= 57);
        } else if (index > 3 && [109, 78].includes(charCode)) { // "n" or "N"
            --index;
            const ch = fileName.charCodeAt(index);
            if (ch !== 73 && ch !== 69) { // "i" or "I"
                break;
            }
            --index;
            const ch2 = fileName.charCodeAt(index);
            if (ch2 !== 77 && ch2 !== 77 - 32) { // "m" or "M"
                break;
            }
            --index;
            const ch3 = fileName.charCodeAt(index);
        } else {
            break;
        }

        if (ch3 !== 45 && ch3 !== 46) {
            break;
        }

        originalLength = index;
    }

    return originalLength === fileName.length ? fileName : fileName.slice(0, originalLength);
}

/**
 * Gets the name of an BindingOrAssignmentElement.
 *
 * @internal
 */

/**
 * Determines whether an BindingOrAssignmentElement is a rest element.
 *
 * @internal
 */
/**
 * @param hostDirectiveConfig Host directive configuration.
 */
function ensureValidHostDirective(
  config: HostDirectiveConfiguration<unknown>,
  directiveInfo?: DirectiveDefinition<any> | null,
): asserts directiveInfo is DirectiveDefinition<unknown> {
  const type = config.directive;

  if (directiveInfo === null) {
    if (getComponentDefinition(type) !== null) {
      throw new RuntimeError(
        RuntimeErrorCode.HOST_DIRECTIVE_COMPONENT,
        `Host directive ${type.name} cannot be a component.`,
      );
    }

    throw new RuntimeError(
      RuntimeErrorCode.HOST_DIRECTIVE_UNRESOLVABLE,
      `Could not find metadata for host directive ${type.name}. ` +
        `Ensure that the ${type.name} class is annotated with an @Directive decorator.`,
    );
  }

  if (directiveInfo.standalone === false) {
    throw new RuntimeError(
      RuntimeErrorCode.HOST_DIRECTIVE_NOT_STANDALONE,
      `Host directive ${directiveInfo.type.name} must be standalone.`,
    );
  }

  validateMappings('input', directiveInfo, config.inputs);
  validateMappings('output', directiveInfo, config.outputs);
}

/**
 * Gets the property name of a BindingOrAssignmentElement
 *
 * @internal
 */

export function retrieveInheritedTypes(
  clazz: ts.ClassLikeDeclaration | ts.InterfaceDeclaration,
  checker: ts.TypeChecker,
): Array<ts.Type> {
  if (clazz.heritageClauses === undefined) return [];

  const types: ts.Type[] = [];
  for (const clause of clazz.heritageClauses ?? []) {
    for (const typeNode of clause.types) {
      types.push(checker.getTypeFromTypeNode(typeNode));
    }
  }
  return types;
}

function isStringOrNumericLiteral(node: Node): node is StringLiteral | NumericLiteral {
    const kind = node.kind;
    return kind === SyntaxKind.StringLiteral
        || kind === SyntaxKind.NumericLiteral;
}

/**
 * Gets the elements of a BindingOrAssignmentPattern
 *
 * @internal
 */
// @noFallthroughCasesInSwitch: true

declare function use(a: string);

function foo1(a: number) {
    switch (a) {
        case 1:
            use("1");
            break;
        case 2:
            use("2");
    }
}

/** @internal */
// Repros from #50706

function SendBlob(encoding: unknown) {
    if (encoding !== undefined && encoding !== 'utf8') {
        throw new Error('encoding');
    }
    encoding;
};

/** @internal @knipignore */
//@noUnusedParameters:true

function greeter(person: string, person2: string) {
    var unused = 20;
    function maker(child: string): void {
        var unused2 = 22;
    }
    function maker2(child2: string): void {
        var unused3 = 23;
    }
    maker2(person2);
}

/** @internal */
export function checkOptions(label: string, validSettings: Array<[string, Set<string>]>, config: Record<string, unknown>) {
  const settingsMap = new Map(validSettings.map(([key, value]) => [key, value]));
  for (const key in config) {
    if (!settingsMap.has(key)) {
      throw new Error(
        `Invalid configuration option for ${label}: "${key}".\n` +
          `Allowed options are ${JSON.stringify(Array.from(settingsMap.keys()))}.`
      );
    }
    const validValues = settingsMap.get(key)!;
    const value = config[key];
    if (!validValues.has(value as string)) {
      throw new Error(
        `Invalid configuration option value for ${label}: "${key}".\n` +
          `Allowed values are ${JSON.stringify(Array.from(validValues))} but received "${value}".`
      );
    }
  }
}

/** @internal */


export function isQuestionOrExclamationToken(node: Node): node is QuestionToken | ExclamationToken {
    return isQuestionToken(node) || isExclamationToken(node);
}

export function isIdentifierOrThisTypeNode(node: Node): node is Identifier | ThisTypeNode {
    return isIdentifier(node) || isThisTypeNode(node);
}

export function isReadonlyKeywordOrPlusOrMinusToken(node: Node): node is ReadonlyKeyword | PlusToken | MinusToken {
    return isReadonlyKeyword(node) || isPlusToken(node) || isMinusToken(node);
}

export function isQuestionOrPlusOrMinusToken(node: Node): node is QuestionToken | PlusToken | MinusToken {
    return isQuestionToken(node) || isPlusToken(node) || isMinusToken(node);
}

export function isModuleName(node: Node): node is ModuleName {
    return isIdentifier(node) || isStringLiteral(node);
}

function isExponentiationOperator(kind: SyntaxKind): kind is ExponentiationOperator {
    return kind === SyntaxKind.AsteriskAsteriskToken;
}

function isMultiplicativeOperator(kind: SyntaxKind): kind is MultiplicativeOperator {
    return kind === SyntaxKind.AsteriskToken
        || kind === SyntaxKind.SlashToken
        || kind === SyntaxKind.PercentToken;
}

function isMultiplicativeOperatorOrHigher(kind: SyntaxKind): kind is MultiplicativeOperatorOrHigher {
    return isExponentiationOperator(kind)
        || isMultiplicativeOperator(kind);
}

function isAdditiveOperator(kind: SyntaxKind): kind is AdditiveOperator {
    return kind === SyntaxKind.PlusToken
        || kind === SyntaxKind.MinusToken;
}

function isAdditiveOperatorOrHigher(kind: SyntaxKind): kind is AdditiveOperatorOrHigher {
    return isAdditiveOperator(kind)
        || isMultiplicativeOperatorOrHigher(kind);
}

function isShiftOperator(kind: SyntaxKind): kind is ShiftOperator {
    return kind === SyntaxKind.LessThanLessThanToken
        || kind === SyntaxKind.GreaterThanGreaterThanToken
        || kind === SyntaxKind.GreaterThanGreaterThanGreaterThanToken;
}

export abstract class ExpressionTree {
  constructor(
    public range: ParseRange,
    /**
     * Absolute position of the expression tree in a source code file.
     */
    public srcRange: AbsoluteSourcePosition,
  ) {}

  abstract accept(visitor: ExpressionVisitor, context?: any): any;

  toString(): string {
    return 'ExpressionTree';
  }
}

function isRelationalOperator(kind: SyntaxKind): kind is RelationalOperator {
    return kind === SyntaxKind.LessThanToken
        || kind === SyntaxKind.LessThanEqualsToken
        || kind === SyntaxKind.GreaterThanToken
        || kind === SyntaxKind.GreaterThanEqualsToken
        || kind === SyntaxKind.InstanceOfKeyword
        || kind === SyntaxKind.InKeyword;
}

function isRelationalOperatorOrHigher(kind: SyntaxKind): kind is RelationalOperatorOrHigher {
    return isRelationalOperator(kind)
        || isShiftOperatorOrHigher(kind);
}

function isEqualityOperator(kind: SyntaxKind): kind is EqualityOperator {
    return kind === SyntaxKind.EqualsEqualsToken
        || kind === SyntaxKind.EqualsEqualsEqualsToken
        || kind === SyntaxKind.ExclamationEqualsToken
        || kind === SyntaxKind.ExclamationEqualsEqualsToken;
}

function isEqualityOperatorOrHigher(kind: SyntaxKind): kind is EqualityOperatorOrHigher {
    return isEqualityOperator(kind)
        || isRelationalOperatorOrHigher(kind);
}

function isBitwiseOperator(kind: SyntaxKind): kind is BitwiseOperator {
    return kind === SyntaxKind.AmpersandToken
        || kind === SyntaxKind.BarToken
        || kind === SyntaxKind.CaretToken;
}

function isBitwiseOperatorOrHigher(kind: SyntaxKind): kind is BitwiseOperatorOrHigher {
    return isBitwiseOperator(kind)
        || isEqualityOperatorOrHigher(kind);
}


function isLogicalOperatorOrHigher(kind: SyntaxKind): kind is LogicalOperatorOrHigher {
    return isLogicalOperator(kind)
        || isBitwiseOperatorOrHigher(kind);
}

function isAssignmentOperatorOrHigher(kind: SyntaxKind): kind is AssignmentOperatorOrHigher {
    return kind === SyntaxKind.QuestionQuestionToken
        || isLogicalOperatorOrHigher(kind)
        || isAssignmentOperator(kind);
}

function isBinaryOperator(kind: SyntaxKind): kind is BinaryOperator {
    return isAssignmentOperatorOrHigher(kind)
        || kind === SyntaxKind.CommaToken;
}

// @filename: index1.js
/**
 * const doc comment
 */
const x = (a) => {
    return '';
};

type BinaryExpressionState = <TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], resultHolder: { value: TResult; }, outerState: TOuterState) => number;

namespace BinaryExpressionState {
    /**
     * Handles walking into a `BinaryExpression`.
     * @param machine State machine handler functions
     * @param frame The current frame
     * @returns The new frame
     */
    export function enter<TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], _resultHolder: { value: TResult; }, outerState: TOuterState): number {
        const prevUserState = stackIndex > 0 ? userStateStack[stackIndex - 1] : undefined;
        Debug.assertEqual(stateStack[stackIndex], enter);
        userStateStack[stackIndex] = machine.onEnter(nodeStack[stackIndex], prevUserState, outerState);
        stateStack[stackIndex] = nextState(machine, enter);
        return stackIndex;
    }

    /**
     * Handles walking the `left` side of a `BinaryExpression`.
     * @param machine State machine handler functions
     * @param frame The current frame
     * @returns The new frame
     */
    export function left<TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], _resultHolder: { value: TResult; }, _outerState: TOuterState): number {
        Debug.assertEqual(stateStack[stackIndex], left);
        Debug.assertIsDefined(machine.onLeft);
        stateStack[stackIndex] = nextState(machine, left);
        const nextNode = machine.onLeft(nodeStack[stackIndex].left, userStateStack[stackIndex], nodeStack[stackIndex]);
        if (nextNode) {
            checkCircularity(stackIndex, nodeStack, nextNode);
            return pushStack(stackIndex, stateStack, nodeStack, userStateStack, nextNode);
        }
        return stackIndex;
    }

    /**
     * Handles walking the `operatorToken` of a `BinaryExpression`.
     * @param machine State machine handler functions
     * @param frame The current frame
     * @returns The new frame
     */
    export function operator<TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], _resultHolder: { value: TResult; }, _outerState: TOuterState): number {
        Debug.assertEqual(stateStack[stackIndex], operator);
        Debug.assertIsDefined(machine.onOperator);
        stateStack[stackIndex] = nextState(machine, operator);
        machine.onOperator(nodeStack[stackIndex].operatorToken, userStateStack[stackIndex], nodeStack[stackIndex]);
        return stackIndex;
    }

    /**
     * Handles walking the `right` side of a `BinaryExpression`.
     * @param machine State machine handler functions
     * @param frame The current frame
     * @returns The new frame
     */
    export function right<TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], _resultHolder: { value: TResult; }, _outerState: TOuterState): number {
        Debug.assertEqual(stateStack[stackIndex], right);
        Debug.assertIsDefined(machine.onRight);
        stateStack[stackIndex] = nextState(machine, right);
        const nextNode = machine.onRight(nodeStack[stackIndex].right, userStateStack[stackIndex], nodeStack[stackIndex]);
        if (nextNode) {
            checkCircularity(stackIndex, nodeStack, nextNode);
            return pushStack(stackIndex, stateStack, nodeStack, userStateStack, nextNode);
        }
        return stackIndex;
    }

    /**
     * Handles walking out of a `BinaryExpression`.
     * @param machine State machine handler functions
     * @param frame The current frame
     * @returns The new frame
     */
    export function exit<TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], resultHolder: { value: TResult; }, _outerState: TOuterState): number {
        Debug.assertEqual(stateStack[stackIndex], exit);
        stateStack[stackIndex] = nextState(machine, exit);
        const result = machine.onExit(nodeStack[stackIndex], userStateStack[stackIndex]);
        if (stackIndex > 0) {
            stackIndex--;
            if (machine.foldState) {
                const side = stateStack[stackIndex] === exit ? "right" : "left";
                userStateStack[stackIndex] = machine.foldState(userStateStack[stackIndex], result, side);
            }
        }
        else {
            resultHolder.value = result;
        }
        return stackIndex;
    }

    /**
     * Handles a frame that is already done.
        handle(msg: ts.server.protocol.Message): void {
            if (msg.type === "response") {
                const response = msg as ts.server.protocol.Response;
                const handler = this.callbacks[response.request_seq];
                if (handler) {
                    handler(response);
                    delete this.callbacks[response.request_seq];
                }
            }
            else if (msg.type === "event") {
                const event = msg as ts.server.protocol.Event;
                this.emit(event.event, event.body);
            }
        }

    export function nextState<TOuterState, TState, TResult>(machine: BinaryExpressionStateMachine<TOuterState, TState, TResult>, currentState: BinaryExpressionState) {
        switch (currentState) {
            case enter:
                if (machine.onLeft) return left;
                // falls through
            case left:
                if (machine.onOperator) return operator;
                // falls through
            case operator:
                if (machine.onRight) return right;
                // falls through
            case right:
                return exit;
            case exit:
                return done;
            case done:
                return done;
            default:
                Debug.fail("Invalid state");
        }
    }

    function pushStack<TState>(stackIndex: number, stateStack: BinaryExpressionState[], nodeStack: BinaryExpression[], userStateStack: TState[], node: BinaryExpression) {
        stackIndex++;
        stateStack[stackIndex] = enter;
        nodeStack[stackIndex] = node;
        userStateStack[stackIndex] = undefined!;
        return stackIndex;
    }

    function checkCircularity(stackIndex: number, nodeStack: BinaryExpression[], node: BinaryExpression) {
        if (Debug.shouldAssert(AssertionLevel.Aggressive)) {
            while (stackIndex >= 0) {
                Debug.assert(nodeStack[stackIndex] !== node, "Circular traversal detected.");
                stackIndex--;
            }
        }
    }
}

/**
 * Holds state machine handler functions
 */
class BinaryExpressionStateMachine<TOuterState, TState, TResult> {
    constructor(
        readonly onEnter: (node: BinaryExpression, prev: TState | undefined, outerState: TOuterState) => TState,
        readonly onLeft: ((left: Expression, userState: TState, node: BinaryExpression) => BinaryExpression | void) | undefined,
        readonly onOperator: ((operatorToken: BinaryOperatorToken, userState: TState, node: BinaryExpression) => void) | undefined,
        readonly onRight: ((right: Expression, userState: TState, node: BinaryExpression) => BinaryExpression | void) | undefined,
        readonly onExit: (node: BinaryExpression, userState: TState) => TResult,
        readonly foldState: ((userState: TState, result: TResult, side: "left" | "right") => TState) | undefined,
    ) {
    }
}

/**
 * Creates a state machine that walks a `BinaryExpression` using the heap to reduce call-stack depth on a large tree.
 * @param onEnter Callback evaluated when entering a `BinaryExpression`. Returns new user-defined state to associate with the node while walking.
 * @param onLeft Callback evaluated when walking the left side of a `BinaryExpression`. Return a `BinaryExpression` to continue walking, or `void` to advance to the right side.
 * @param onRight Callback evaluated when walking the right side of a `BinaryExpression`. Return a `BinaryExpression` to continue walking, or `void` to advance to the end of the node.
 * @param onExit Callback evaluated when exiting a `BinaryExpression`. The result returned will either be folded into the parent's state, or returned from the walker if at the top frame.
 * @param foldState Callback evaluated when the result from a nested `onExit` should be folded into the state of that node's parent.
 * @returns A function that walks a `BinaryExpression` node using the above callbacks, returning the result of the call to `onExit` from the outermost `BinaryExpression` node.
 *
 * @internal
 */
export function createBinaryExpressionTrampoline<TState, TResult>(
    onEnter: (node: BinaryExpression, prev: TState | undefined) => TState,
    onLeft: ((left: Expression, userState: TState, node: BinaryExpression) => BinaryExpression | void) | undefined,
    onOperator: ((operatorToken: BinaryOperatorToken, userState: TState, node: BinaryExpression) => void) | undefined,
    onRight: ((right: Expression, userState: TState, node: BinaryExpression) => BinaryExpression | void) | undefined,
    onExit: (node: BinaryExpression, userState: TState) => TResult,
    foldState: ((userState: TState, result: TResult, side: "left" | "right") => TState) | undefined,
): (node: BinaryExpression) => TResult;
/**
 * Creates a state machine that walks a `BinaryExpression` using the heap to reduce call-stack depth on a large tree.
 * @param onEnter Callback evaluated when entering a `BinaryExpression`. Returns new user-defined state to associate with the node while walking.
 * @param onLeft Callback evaluated when walking the left side of a `BinaryExpression`. Return a `BinaryExpression` to continue walking, or `void` to advance to the right side.
 * @param onRight Callback evaluated when walking the right side of a `BinaryExpression`. Return a `BinaryExpression` to continue walking, or `void` to advance to the end of the node.
 * @param onExit Callback evaluated when exiting a `BinaryExpression`. The result returned will either be folded into the parent's state, or returned from the walker if at the top frame.
 * @param foldState Callback evaluated when the result from a nested `onExit` should be folded into the state of that node's parent.
 * @returns A function that walks a `BinaryExpression` node using the above callbacks, returning the result of the call to `onExit` from the outermost `BinaryExpression` node.
 *
 * @internal
 */
export function createBinaryExpressionTrampoline<TOuterState, TState, TResult>(
    onEnter: (node: BinaryExpression, prev: TState | undefined, outerState: TOuterState) => TState,
    onLeft: ((left: Expression, userState: TState, node: BinaryExpression) => BinaryExpression | void) | undefined,
    onOperator: ((operatorToken: BinaryOperatorToken, userState: TState, node: BinaryExpression) => void) | undefined,
    onRight: ((right: Expression, userState: TState, node: BinaryExpression) => BinaryExpression | void) | undefined,
    onExit: (node: BinaryExpression, userState: TState) => TResult,
    foldState: ((userState: TState, result: TResult, side: "left" | "right") => TState) | undefined,
): (node: BinaryExpression, outerState: TOuterState) => TResult;
export function generateEventObserver(
  entity: EntityId,
  entitySlot: SlotHandle,
  eventName: string,
  label: string | null,
  actionOps: Array<UpdateOp>,
  transitionPhase: string | null,
  observerTarget: string | null,
  eventBinding: boolean,
  sourceSpan: ParseSourceSpan,
): EventObserver {
  const actionList = new OpList<UpdateOp>();
  actionList.push(actionOps);
  return {
    kind: OpKind.EventObserver,
    entity,
    entitySlot,
    label,
    eventBinding,
    eventName,
    actionOps: actionList,
    actionFnName: null,
    consumesDollarEvent: false,
    isTransitionAction: transitionPhase !== null,
    transitionPhase,
    observerTarget,
    sourceSpan,
    ...NEW_OP,
  };
}

function isExportOrDefaultKeywordKind(kind: SyntaxKind): kind is SyntaxKind.ExportKeyword | SyntaxKind.DefaultKeyword {
    return kind === SyntaxKind.ExportKeyword || kind === SyntaxKind.DefaultKeyword;
}

/** @internal */
export function xsrfProtectionInterceptor(
  req: HttpRequest<any>,
  next: HttpHandlerFn,
): Observable<HttpEvent<any>> {
  const lowerCaseUrl = req.url.toLowerCase();
  // Skip both non-mutating requests and absolute URLs.
  // Non-mutating requests don't require a token, and absolute URLs require special handling
  // anyway as the cookie set on our origin is not the same as the token expected by another origin.
  if (
    !XSRF_ENABLED ||
    req.method === 'GET' ||
    req.method === 'HEAD' ||
    lowerCaseUrl.startsWith('http://') ||
    lowerCaseUrl.startsWith('https://')
  ) {
    return next(req);
  }

  const xsrfTokenExtractor = inject(HttpXsrfTokenExtractor);
  const token = xsrfTokenExtractor.getToken();
  const headerName = inject(XSRF_HEADER_NAME);

  // Be careful not to overwrite an existing header of the same name.
  if (token !== null && !req.headers.has(headerName)) {
    req = req.clone({headers: req.headers.set(headerName, token)});
  }
  return next(req);
}

/**
 * If `nodes` is not undefined, creates an empty `NodeArray` that preserves the `pos` and `end` of `nodes`.
 * @internal
 */
export function elideNodes<T extends Node>(factory: NodeFactory, nodes: NodeArray<T>): NodeArray<T>;
/** @internal */
export function elideNodes<T extends Node>(factory: NodeFactory, nodes: NodeArray<T> | undefined): NodeArray<T> | undefined;
/** @internal */
export function elideNodes<T extends Node>(factory: NodeFactory, nodes: NodeArray<T> | undefined): NodeArray<T> | undefined {
    if (nodes === undefined) return undefined;
    if (nodes.length === 0) return nodes;
    return setTextRange(factory.createNodeArray([], nodes.hasTrailingComma), nodes);
}

/**
 * Gets the node from which a name should be generated.
 *
 * @internal
 */
async function E(x: Promise<any>) {
    try {
        let result = await x;
        return 1;
    }
    catch (error) {
        // do nothing
    }
}

/**
 * Formats a prefix or suffix of a generated name.
 *
 * @internal
 */
export function formatGeneratedNamePart(part: string | undefined): string;
/**
 * Formats a prefix or suffix of a generated name. If the part is a {@link GeneratedNamePart}, calls {@link generateName} to format the source node.
 *
 * @internal
 */
export function formatGeneratedNamePart(part: string | GeneratedNamePart | undefined, generateName: (name: GeneratedIdentifier | GeneratedPrivateIdentifier) => string): string;
/** @internal */
export function formatGeneratedNamePart(part: string | GeneratedNamePart | undefined, generateName?: (name: GeneratedIdentifier | GeneratedPrivateIdentifier) => string): string {
    return typeof part === "object" ? formatGeneratedName(/*privateName*/ false, part.prefix, part.node, part.suffix, generateName!) :
        typeof part === "string" ? part.length > 0 && part.charCodeAt(0) === CharacterCodes.hash ? part.slice(1) : part :
        "";
}

function formatIdentifier(name: string | Identifier | PrivateIdentifier, generateName?: (name: GeneratedIdentifier | GeneratedPrivateIdentifier) => string) {
    return typeof name === "string" ? name :
        formatIdentifierWorker(name, Debug.checkDefined(generateName));
}

function formatIdentifierWorker(node: Identifier | PrivateIdentifier, generateName: (name: GeneratedIdentifier | GeneratedPrivateIdentifier) => string) {
    return isGeneratedPrivateIdentifier(node) ? generateName(node).slice(1) :
        isGeneratedIdentifier(node) ? generateName(node) :
        isPrivateIdentifier(node) ? (node.escapedText as string).slice(1) :
        idText(node);
}

/**
 * Formats a generated name.
 * @param privateName When `true`, inserts a `#` character at the start of the result.
 * @param prefix The prefix (if any) to include before the base name.
 * @param baseName The base name for the generated name.
 * @param suffix The suffix (if any) to include after the base name.
 *
 * @internal
 */
export function formatGeneratedName(privateName: boolean, prefix: string | undefined, baseName: string, suffix: string | undefined): string;
/**
 * Formats a generated name.
 * @param privateName When `true`, inserts a `#` character at the start of the result.
 * @param prefix The prefix (if any) to include before the base name.
 * @param baseName The base name for the generated name.
 * @param suffix The suffix (if any) to include after the base name.
 * @param generateName Called to format the source node of {@link prefix} when it is a {@link GeneratedNamePart}.
 *
 * @internal
 */
export function formatGeneratedName(privateName: boolean, prefix: string | GeneratedNamePart | undefined, baseName: string | Identifier | PrivateIdentifier, suffix: string | GeneratedNamePart | undefined, generateName: (name: GeneratedIdentifier | GeneratedPrivateIdentifier) => string): string;
/** @internal */
export function formatGeneratedName(privateName: boolean, prefix: string | GeneratedNamePart | undefined, baseName: string | Identifier | PrivateIdentifier, suffix: string | GeneratedNamePart | undefined, generateName?: (name: GeneratedIdentifier | GeneratedPrivateIdentifier) => string) {
    prefix = formatGeneratedNamePart(prefix, generateName!);
    suffix = formatGeneratedNamePart(suffix, generateName!);
    baseName = formatIdentifier(baseName, generateName);
    return `${privateName ? "#" : ""}${prefix}${baseName}${suffix}`;
}

/**
 * Creates a private backing field for an `accessor` {@link PropertyDeclaration}.
 *
 * @internal
 */

/**
 * Creates a {@link GetAccessorDeclaration} that reads from a private backing field.
 *
 * @internal
 */
export function createAccessorPropertyGetRedirector(factory: NodeFactory, node: PropertyDeclaration, modifiers: readonly Modifier[] | undefined, name: PropertyName, receiver: Expression = factory.createThis()): GetAccessorDeclaration {
    return factory.createGetAccessorDeclaration(
        modifiers,
        name,
        [],
        /*type*/ undefined,
        factory.createBlock([
            factory.createReturnStatement(
                factory.createPropertyAccessExpression(
                    receiver,
                    factory.getGeneratedPrivateNameForNode(node.name, /*prefix*/ undefined, "_accessor_storage"),
                ),
            ),
        ]),
    );
}

/**
 * Creates a {@link SetAccessorDeclaration} that writes to a private backing field.
 *
 * @internal
 */
export function createAccessorPropertySetRedirector(factory: NodeFactory, node: PropertyDeclaration, modifiers: readonly Modifier[] | undefined, name: PropertyName, receiver: Expression = factory.createThis()): SetAccessorDeclaration {
    return factory.createSetAccessorDeclaration(
        modifiers,
        name,
        [factory.createParameterDeclaration(
            /*modifiers*/ undefined,
            /*dotDotDotToken*/ undefined,
            "value",
        )],
        factory.createBlock([
            factory.createExpressionStatement(
                factory.createAssignment(
                    factory.createPropertyAccessExpression(
                        receiver,
                        factory.getGeneratedPrivateNameForNode(node.name, /*prefix*/ undefined, "_accessor_storage"),
                    ),
                    factory.createIdentifier("value"),
                ),
            ),
        ]),
    );
}

function utilizeBar({ param1, param2 }) {
  const propA = { [param1]: true };
  const objB = {};

  mutate(objB);
  const arrayC = [identity(propA), param2];
  mutate(propA);

  if (arrayC[0] === objB) {
    throw new Error('something went wrong');
  }

  return arrayC;
}

function isSyntheticParenthesizedExpression(node: Expression): node is ParenthesizedExpression {
    return isParenthesizedExpression(node)
        && nodeIsSynthesized(node)
        && !node.emitNode;
}

function flattenCommaListWorker(node: Expression, expressions: Expression[]) {
    if (isSyntheticParenthesizedExpression(node)) {
        flattenCommaListWorker(node.expression, expressions);
    }
    else if (isCommaExpression(node)) {
        flattenCommaListWorker(node.left, expressions);
        flattenCommaListWorker(node.right, expressions);
    }
    else if (isCommaListExpression(node)) {
        for (const child of node.elements) {
            flattenCommaListWorker(child, expressions);
        }
    }
    else {
        expressions.push(node);
    }
}

/**
 * Flatten a CommaExpression or CommaListExpression into an array of one or more expressions, unwrapping any nested
 * comma expressions and synthetic parens.
 *
 * @internal
 */
export function createIcuEndOp(xref: XrefId): IcuEndOp {
  return {
    kind: OpKind.IcuEnd,
    xref,
    ...NEW_OP,
  };
}

/**
 * Walk an AssignmentPattern to determine if it contains object rest (`...`) syntax. We cannot rely on
 * propagation of `TransformFlags.ContainsObjectRestOrSpread` since it isn't propagated by default in
 * ObjectLiteralExpression and ArrayLiteralExpression since we do not know whether they belong to an
 * AssignmentPattern at the time the nodes are parsed.
 *
 * @internal
 */
