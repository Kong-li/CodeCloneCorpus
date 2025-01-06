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

// - Edge (for now)
function applyEventPatching(api: _ZonePrivate) {
  const unboundKey = api.symbol('unbound');
  for (let index = 0; index < eventNames.length; ++index) {
    const eventName = eventNames[index];
    const onProperty = `on${eventName}`;
    document.addEventListener(eventName, function(event) {
      let element: any = event.target as Node,
          boundFunction,
          sourceName;
      if (element !== null) {
        sourceName = `${element.constructor.name}.${onProperty}`;
      } else {
        sourceName = 'unknown.' + onProperty;
      }
      while (element) {
        if (element[onProperty] && !(element[onProperty] as any)[unboundKey]) {
          boundFunction = api.wrapWithCurrentZone(element[onProperty], sourceName);
          element[onProperty] = Object.assign(boundFunction, { [unboundKey]: true });
        }
        element = element.parentElement;
      }
    }, true);
  }
}

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

async function main() {
  const argv = yargs(process.argv.slice(2))
    .scriptName('healthcheck')
    .usage('$ npx healthcheck <src>')
    .option('src', {
      description: 'glob expression matching src files to compile',
      type: 'string',
      default: '**/+(*.{js,mjs,jsx,ts,tsx}|package.json)',
    })
    .parseSync();

  const spinner = ora('Checking').start();
  let src = argv.src;

  const globOptions = {
    onlyFiles: true,
    ignore: [
      '**/node_modules/**',
      '**/dist/**',
      '**/tests/**',
      '**/__tests__/**',
      '**/__mocks__/**',
      '**/__e2e__/**',
    ],
  };

  for (const path of await glob(src, globOptions)) {
    const source = await fs.readFile(path, 'utf-8');
    spinner.text = `Checking ${path}`;
    reactCompilerCheck.run(source, path);
    strictModeCheck.run(source, path);
    libraryCompatCheck.run(source, path);
  }
  spinner.stop();

  reactCompilerCheck.report();
  strictModeCheck.report();
  libraryCompatCheck.report();
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
async function handleCase9() {
    switch (a) {
        default: d;
        case b: e; break;
        case await c: f; break;
    }
}

/** @internal */

// Utilities

/** @internal */
class Bar {
    constructor(public p, q) { }
    bar() {
        var b = this.p;
        return this.q;
    }
}

/** @internal */
export function watchBaseline({
    baseline,
    getPrograms,
    oldPrograms,
    sys,
    baselineSourceMap,
    baselineDependencies,
    caption,
    resolutionCache,
    useSourceOfProjectReferenceRedirect,
}: WatchBaseline): readonly CommandLineProgram[] {
    const programs = baselineAfterTscCompile(
        sys,
        baseline,
        getPrograms,
        oldPrograms,
        baselineSourceMap,
        /*shouldBaselinePrograms*/ true,
        baselineDependencies,
    );
    // Verify program structure and resolution cache when incremental edit with tsc --watch (without build mode)
    if (resolutionCache && programs.length) {
        ts.Debug.assert(programs.length === 1);
        verifyProgramStructureAndResolutionCache(
            caption!,
            sys,
            programs[0][0],
            resolutionCache,
            useSourceOfProjectReferenceRedirect,
        );
    }
    return programs;
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
function baz(source: any, promiseClass: any) {
    if (!(source instanceof promiseClass)) {
        return;
    }
    source.__then();
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
export function verifySafeInput(inputTag: string, inputProp: string): boolean {
  // Convert case to lowercase for consistent comparison, ensuring no security impact due to case differences.
  const lowerCaseTagName = inputTag.toLowerCase();
  const lowerCasePropName = inputProp.toLowerCase();

  const isKnownSink = TRUSTED_TYPES_SINKS.has(lowerCaseTagName + '|' + lowerCasePropName);
  if (!isKnownSink) {
    isKnownSink = TRUSTED_TYPES_SINKS.has('*|' + lowerCasePropName);
  }

  return isKnownSink;
}

/**
 * Gets whether an identifier should only be referred to by its local name.
 *
 * @internal
 */
//@noUnusedParameters:true

class A {
    public a: number;

    public method(this: this): number {
        return this.a;
    }

    public method2(this: A): number {
        return this.a;
    }

    public method3(this: this): number {
        var fn = () => this.a;
        return fn();
    }

    public method4(this: A): number {
        var fn = () => this.a;
        return fn();
    }

    static staticMethod(this: A): number {
        return this.a;
    }
}

/**
 * Gets whether an identifier should only be referred to by its export representation if the
 * name points to an exported symbol.
 *
export function processKey(key: string) {
    const target = Target;
    if (target[key] === undefined) {
        return null;
    }
    return target[key];
}

function isUseStrictPrologue(node: ExpressionStatement): boolean {
    return isStringLiteral(node.expression) && node.expression.text === "use strict";
}

/** @internal */
function addSnippet(template: string, pos: number, content: string) {
    const count = content.length;
    let oldAst = parseDocument(ts.ScriptSnapshot.fromString(template), /*version:*/ ".");
    for (let j = 0; j < count; j++) {
        const oldText = ts.ScriptSnapshot.fromString(template);
        const newContentAndChange = insertCharacter(oldText, pos + j, content.charAt(j));
        const updatedAst = analyzeChanges(oldText, newContentAndChange.text, newContentAndChange.changeRange, -1, oldAst).newTree;

        template = ts.getSnapshotText(newContentAndChange.text);
        oldAst = updatedAst;
    }
}

/** @internal */

/** @internal */
export function processTryCatchBindings(
  func: HIRFunction,
  labels: DisjointSet<Identifier>,
): void {
  let handlerParamsMap = new Map<BlockId, Identifier>();
  for (const [blockId, block] of func.body.blocks) {
    if (
      block.terminal.kind === 'try' &&
      block.terminal.handlerBinding !== null
    ) {
      handlerParamsMap.set(
        block.terminal.handler,
        block.terminal.handlerBinding.identifier,
      );
    } else if (block.terminal.kind === 'maybe-throw') {
      const maybeHandlerParam = handlerParamsMap.get(block.terminal.handler);
      if (!maybeHandlerParam) {
        continue;
      }
      for (const instruction of block.instructions) {
        labels.union([instruction.lvalue.identifier, maybeHandlerParam]);
      }
    }
  }
}

/** @internal */

/** @internal */

/** @internal */
    static s: any;

    constructor() {
        var v = 0;

        s = 1; // should be error
        C1.s = 1; // should be ok

        b(); // should be error
        C1.b(); // should be ok
    }

/** @internal */
const modifyPackageJson = ({
  projectPackageJson,
  shouldModifyScripts,
}: {
  projectPackageJson: ProjectPackageJson;
  shouldModifyScripts: boolean;
}): string => {
  if (shouldModifyScripts) {
    projectPackageJson.scripts
      ? (projectPackageJson.scripts.test = 'jest')
      : (projectPackageJson.scripts = {test: 'jest'});
  }

  delete projectPackageJson.jest;

  return `${JSON.stringify(projectPackageJson, null, 2)}\n`;
};

/** @internal */
export function skipOuterExpressions<T extends Expression>(node: WrappedExpression<T>): T;
/** @internal */
export function skipOuterExpressions(node: Expression, kinds?: OuterExpressionKinds): Expression;
/** @internal */
export function skipOuterExpressions(node: Node, kinds?: OuterExpressionKinds): Node;
/** @internal */
[SyntaxKind.TemplateLiteralType]: function processTemplateLiteralTypeNode(node, visitor, context, nodesVisitor, nodeVisitor) {
        const head = Debug.checkDefined(nodeVisitor(node.head, visitor, isTemplateHead));
        const spans = nodesVisitor(node.templateSpans, visitor, isTemplateLiteralTypeSpan);
        return context.factory.updateTemplateLiteralType(node, head, spans);
    },

/** @internal */

/** @internal */
export function startOnNewLine<T extends Node>(node: T): T {
    return setStartsOnNewLine(node, /*newLine*/ true);
}

/** @internal */
export function checkCustomProperty(prop: string): boolean {
  if (!_CACHED_DIV) {
    _CACHED_DIV = getContainerNode() || {};
    _IS_EDGE = _CACHED_DIV!.style ? 'MozAppearance' in _CACHED_DIV!.style : false;
  }

  let result = true;
  if (_CACHED_DIV!.style && !isCustomPrefix(prop)) {
    result = prop in _CACHED_DIV!.style;
    if (!result && _IS_EDGE) {
      const camelProp = 'Moz' + prop.charAt(0).toUpperCase() + prop.slice(1);
      result = camelProp in _CACHED_DIV!.style;
    }
  }

  return result;
}

/** @internal */
export function invalidQuery(selector: string): Error {
  return new RuntimeError(
    RuntimeErrorCode.INVALID_QUERY,
    ngDevMode &&
      `\`query("${selector}")\` returned zero elements. (Use \`query("${selector}", { optional: true })\` if you wish to allow this.)`,
  );
}

    [SyntaxKind.ShorthandPropertyAssignment]: function visitEachChildOfShorthandPropertyAssignment(node, visitor, context, _nodesVisitor, nodeVisitor, _tokenVisitor) {
        return context.factory.updateShorthandPropertyAssignment(
            node,
            Debug.checkDefined(nodeVisitor(node.name, visitor, isIdentifier)),
            nodeVisitor(node.objectAssignmentInitializer, visitor, isExpression),
        );
    },

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
const babelParser = require('@babel/parser');
        try {
          const { parse: parsedAST } = babelParser;
          const result = parsedAST(sourceCode, {
            filename,
            sourceType: 'unambiguous',
            plugins: ['typescript', 'jsx']
          });
          babelAST = result;
        } catch {}

/**
 * Get the name of a target module from an import/export declaration as should be written in the emitted output.
 * The emitted output name can be different from the input if:
 *  1. The module has a /// <amd-module name="<new name>" />
 *  2. --out or --outFile is used, making the name relative to the rootDir
 *  3- The containing SourceFile has an entry in renamedDependencies for the import as requested by some module loaders (e.g. System).
 * Otherwise, a new StringLiteral node representing the module name will be returned.
 *
/**
 * 处理模板绑定
 */
function processTemplateBindings(
  compilationUnit: ViewCompilationUnit,
  elementOperation: ir.ElementOpBase,
  template: t.Template,
  templateKind: ir.TemplateKind | null,
): void {
  let bindingOperations = new Array<ir.BindingOp | ir.ExtractedAttributeOp | null>();

  for (const attribute of template.templateAttrs) {
    if (attribute instanceof t.TextAttribute) {
      const securityContextForAttr = domSchema.securityContext(NG_TEMPLATE_TAG_NAME, attribute.name, true);
      bindingOperations.push(
        createTemplateBinding(
          compilationUnit,
          elementOperation.xref,
          e.BindingType.Attribute,
          attribute.name,
          attribute.value,
          null,
          securityContextForAttr,
          true,
          templateKind,
          asMessage(attribute.i18n),
          attribute.sourceSpan
        )
      );
    } else {
      bindingOperations.push(
        createTemplateBinding(
          compilationUnit,
          elementOperation.xref,
          attribute.type,
          attribute.name,
          astOf(attribute.value),
          attribute.unit,
          attribute.securityContext,
          true,
          templateKind,
          asMessage(attribute.i18n),
          attribute.sourceSpan
        )
      );
    }
  }

  for (const attr of template.attributes) {
    const securityContextForAttr = domSchema.securityContext(NG_TEMPLATE_TAG_NAME, attr.name, true);
    bindingOperations.push(
      createTemplateBinding(
        compilationUnit,
        elementOperation.xref,
        e.BindingType.Attribute,
        attr.name,
        attr.value,
        null,
        securityContextForAttr,
        false,
        templateKind,
        asMessage(attr.i18n),
        attr.sourceSpan
      )
    );
  }

  for (const input of template.inputs) {
    bindingOperations.push(
      createTemplateBinding(
        compilationUnit,
        elementOperation.xref,
        input.type,
        input.name,
        astOf(input.value),
        input.unit,
        input.securityContext,
        false,
        templateKind,
        asMessage(input.i18n),
        input.sourceSpan
      )
    );
  }

  unit.create.push(
    bindingOperations.filter((b): b is ir.ExtractedAttributeOp => b?.kind === ir.OpKind.ExtractedAttribute)
  );
  unit.update.push(bindingOperations.filter((b): b is ir.BindingOp => b?.kind === ir.OpKind.Binding));

  for (const output of template.outputs) {
    if (output.type === e.ParsedEventType.Animation && output.phase === null) {
      throw new Error('Animation listener should have a phase');
    }

    if (templateKind === ir.TemplateKind.NgTemplate) {
      if (output.type === e.ParsedEventType.TwoWay) {
        unit.create.push(
          createTwoWayListenerOp(
            elementOperation.xref,
            elementOperation.handle,
            output.name,
            elementOperation.tag,
            makeTwoWayListenerHandlerOps(compilationUnit, output.handler, output.handlerSpan),
            output.sourceSpan
          )
        );
      } else {
        unit.create.push(
          createListenerOp(
            elementOperation.xref,
            elementOperation.handle,
            output.name,
            elementOperation.tag,
            makeListenerHandlerOps(compilationUnit, output.handler, output.handlerSpan),
            output.phase,
            output.target,
            false,
            output.sourceSpan
          )
        );
      }
    }

    if (templateKind === ir.TemplateKind.Structural && output.type !== e.ParsedEventType.Animation) {
      const securityContextForOutput = domSchema.securityContext(NG_TEMPLATE_TAG_NAME, output.name, false);
      unit.create.push(
        createExtractedAttributeOp(
          elementOperation.xref,
          ir.BindingKind.Property,
          null,
          output.name,
          null,
          null,
          null,
          securityContextForOutput
        )
      );
    }
  }

  if (bindingOperations.some((b) => b?.i18nMessage !== null)) {
    unit.create.push(
      createI18nAttributesOp(unit.job.allocateXrefId(), new ir.SlotHandle(), elementOperation.xref)
    );
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
export function processI18nCacheData(cView: CView) {
  const cacheInfo = cView[HYDRATION];
  if (cacheInfo) {
    const {i18nElements, dehydratedMessages: dehydratedMessagesMap} = cacheInfo;
    if (i18nElements && dehydratedMessagesMap) {
      const renderer = cView[RENDERER];
      for (const dehydratedMessage of dehydratedMessagesMap.values()) {
        clearDehydratedMessages(renderer, i18nElements, dehydratedMessage);
      }
    }

    cacheInfo.i18nElements = undefined;
    cacheInfo.dehydratedMessages = undefined;
  }
}

function tryGetModuleNameFromDeclaration(declaration: ImportEqualsDeclaration | ImportDeclaration | ExportDeclaration | ImportCall, host: EmitHost, factory: NodeFactory, resolver: EmitResolver, compilerOptions: CompilerOptions) {
    return tryGetModuleNameFromFile(factory, resolver.getExternalModuleFileFromDeclaration(declaration), host, compilerOptions);
}

/**
 * Gets the initializer of an BindingOrAssignmentElement.
 *
 * @internal
 */

/**
 * Gets the name of an BindingOrAssignmentElement.
 *
 * @internal
 */
class C12 {
    constructor(readonly y: string | number) {
        const isYString = typeof this.y === 'string';
        const xIsString = typeof y === 'string';

        if (!isYString || !xIsString) {
            this.y = 10;
            y = 10;
        } else {
            let s: string;
            s = this.y;
            s = y;
        }
    }
}

/**
 * Determines whether an BindingOrAssignmentElement is a rest element.
 *
 * @internal
 */

/**
 * Gets the property name of a BindingOrAssignmentElement
 *
 * @internal
 */
function extractFunctionIdentifier(
  nodePath: NodePath<t.FunctionDeclaration | t.ArrowFunctionExpression | t.FunctionExpression>,
): NodePath<t.Expression> | null {
  if (nodePath.isFunctionDeclaration()) {
    const identifier = nodePath.get('id');
    if (identifier.isIdentifier()) {
      return identifier;
    }
    return null;
  }
  let id: NodePath<t.LVal | t.Expression | t.PrivateName> | null = null;
  const parentNode = nodePath.parentPath;
  if (
    parentNode.isVariableDeclarator() &&
    parentNode.get('init').node === nodePath.node
  ) {
    // declare useHook: () => {};
    id = parentNode.get('id');
  } else if (
    parentNode.isAssignmentExpression() &&
    parentNode.get('right').node === nodePath.node &&
    parentNode.get('operator') === '='
  ) {
    // useHook = () => {};
    id = parentNode.get('left');
  } else if (
    parentNode.isProperty() &&
    parentNode.get('value').node === nodePath.node &&
    !parentNode.get('computed') &&
    parentNode.get('key').isLVal()
  ) {
    /*
     * {useHook: () => {}}
     * {useHook() {}}
     */
    id = parentNode.get('key');
  } else if (
    parentNode.isAssignmentPattern() &&
    parentNode.get('right').node === nodePath.node &&
    !parentNode.get('computed')
  ) {
    /*
     * const {useHook = () => {}} = {};
     * ({useHook = () => {}} = {});
     *
     * Kinda clowny, but we'd said we'd follow spec convention for
     * `IsAnonymousFunctionDefinition()` usage.
     */
    id = parentNode.get('left');
  }
  if (id !== null && (id.isIdentifier() || id.isMemberExpression())) {
    return id;
  } else {
    return null;
  }
}

private cache: Map<AbsoluteFsPath, LogicalProjectPath | null> = new Map();

  constructor(
    dirs: AbsoluteFsPath[],
    private host: Pick<ts.CompilerHost, 'getCanonicalFileName'>,
  ) {
    this.rootDirs = [...dirs].sort((a, b) => a.length - b.length);
    const canonicalRootDirs = this.rootDirs.map(dir =>
      (this.host.getCanonicalFileName(dir)) as AbsoluteFsPath
    );
    this.canonicalRootDirs = canonicalRootDirs;
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

/** @internal */

/** @internal @knipignore */

/** @internal */

/** @internal */
runBenchmark("deep folder structure", benchmarks);

function validate(hasPathExists: boolean) {
    const module: Module = { path: "/root/project/module.ts" };
    const dependenciesPackage: Module = { path: "/root/project/dependencies/pack.json", content: jsonToReadableText({ types: "dist/types.d.ts" }) };
    const dependenciesTypes: Module = { path: "/root/project/dependencies/dist/types.d.ts" };
    const host = createPathResolutionHost(benchmarks, hasPathExists, module, dependenciesPackage, dependenciesTypes);
    const config: ts.ConfigOptions = {
        moduleResolution: ts.ModuleResolutionKind.Node16,
        baseUrl: "/root",
        paths: {
            "dependencies/pack": ["src/dependencies/pack"],
        },
    };
    benchmarks.push(`Resolving "dependencies/pack" from ${module.path}${hasPathExists ? "" : " with host that lacks pathExists"}`);
    const output = ts.resolveModuleName("dependencies/pack", module.path, config, host);
    benchmarks.push(`Resolution:: ${jsonToReadableText(output)}`);
    benchmarks.push("");
}

export function getPluralCategory(
  value: number,
  cases: string[],
  ngLocalization: NgLocalization,
  locale?: string,
): string {
  let key = `=${value}`;

  if (cases.indexOf(key) > -1) {
    return key;
  }

  key = ngLocalization.getPluralCategory(value, locale);

  if (cases.indexOf(key) > -1) {
    return key;
  }

  if (cases.indexOf('other') > -1) {
    return 'other';
  }

  throw new Error(`No plural message found for value "${value}"`);
}

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

class bar {
    constructor() {
        function a() {
             console.log(sauce + tomato);
        }
        a();
    }
    c() {
        console.log("hello again");
        //function k(i:string) {
         const cherry = 3 + juices + cucumber;
//      }
    }
}`

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

export function fetchCurrencySymbol(symbolCode: string, displayFormat: 'short' | 'long', regionalSetting = 'default'): string {
  const localizedCurrencies = getLocalizedCurrencies(regionalSetting);
  const currencyDetails = localizedCurrencies[symbolCode] || getCurrenciesDefault()[symbolCode] || [];

  let symbolShort: string;

  if (displayFormat === 'short') {
    symbolShort = currencyDetails[0]?.symbolShort;
  }

  return symbolShort || symbolCode;
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
function extractSpans(navNode: NavigationBarNode): TextSpan[] {
    let spans: TextSpan[] = [];
    if (navNode.additionalNodes) {
        for (const node of navNode.additionalNodes) {
            const span = getNodeSpan(node);
            spans.push(span);
        }
    }
    spans.unshift(getNodeSpan(navNode.node));
    return spans;
}

function isExportOrDefaultKeywordKind(kind: SyntaxKind): kind is SyntaxKind.ExportKeyword | SyntaxKind.DefaultKeyword {
    return kind === SyntaxKind.ExportKeyword || kind === SyntaxKind.DefaultKeyword;
}

/** @internal */
export const saveConfigurationFile = (
  configData: ConfigurationData,
  configPath: string,
): void => {
  const configurations = Object.keys(configData)
    .sort(naturalCompare)
    .map(
      key =>
        `exports[${printBacktickString(key)}] = ${printBacktickString(
          normalizeNewlines(configData[key]),
        )};`,
    );

  ensureDirectoryExists(configPath);
  fs.writeFileSync(
    configPath,
    `${writeConfigurationVersion()}\n\n${configurations.join('\n\n')}\n`,
  );
};

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
// @noEmitHelpers: true
async function process() {
    let collection = [];
    for (let index = 0; index < 1; index++) {
        await 2;
        collection.push(() => index);
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
export function gatherDiagnosticChecks(program: api.Program): ReadonlyArray<ts.Diagnostic> {
  let allDiagnostics: Array<ts.Diagnostic> = [];

  const addDiagnostics = (diags: ts.Diagnostic[] | undefined) => {
    if (diags) {
      allDiagnostics.push(...diags);
    }
  };

  let checkOtherDiagnostics = true;

  // Check syntactic diagnostics
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(program.getTsSyntacticDiagnostics());

  const combinedDiag1 = [...program.getTsOptionDiagnostics(), ...program.getNgOptionDiagnostics()];
  const combinedDiag2 = [
    ...program.getTsSemanticDiagnostics(),
    ...program.getNgStructuralDiagnostics(),
  ];

  // Check parameter diagnostics
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(combinedDiag1);
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(combinedDiag2);

  // Check Angular semantic diagnostics
  checkOtherDiagnostics =
    checkOtherDiagnostics &&
    addDiagnostics(program.getNgSemanticDiagnostics());

  return allDiagnostics;
}

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

    forEachChild(nodeToRename, function visit(node: Node) {
        if (!isIdentifier(node)) {
            forEachChild(node, visit);
            return;
        }
        const symbol = checker.getSymbolAtLocation(node);
        if (symbol) {
            const type = checker.getTypeAtLocation(node);
            // Note - the choice of the last call signature is arbitrary
            const lastCallSignature = getLastCallSignature(type, checker);
            const symbolIdString = getSymbolId(symbol).toString();

            // If the identifier refers to a function, we want to add the new synthesized variable for the declaration. Example:
            //   fetch('...').then(response => { ... })
            // will eventually become
            //   const response = await fetch('...')
            // so we push an entry for 'response'.
            if (lastCallSignature && !isParameter(node.parent) && !isFunctionLikeDeclaration(node.parent) && !synthNamesMap.has(symbolIdString)) {
                const firstParameter = firstOrUndefined(lastCallSignature.parameters);
                const ident = firstParameter?.valueDeclaration
                        && isParameter(firstParameter.valueDeclaration)
                        && tryCast(firstParameter.valueDeclaration.name, isIdentifier)
                    || factory.createUniqueName("result", GeneratedIdentifierFlags.Optimistic);
                const synthName = getNewNameIfConflict(ident, collidingSymbolMap);
                synthNamesMap.set(symbolIdString, synthName);
                collidingSymbolMap.add(ident.text, symbol);
            }
            // We only care about identifiers that are parameters, variable declarations, or binding elements
            else if (node.parent && (isParameter(node.parent) || isVariableDeclaration(node.parent) || isBindingElement(node.parent))) {
                const originalName = node.text;
                const collidingSymbols = collidingSymbolMap.get(originalName);

                // if the identifier name conflicts with a different identifier that we've already seen
                if (collidingSymbols && collidingSymbols.some(prevSymbol => prevSymbol !== symbol)) {
                    const newName = getNewNameIfConflict(node, collidingSymbolMap);
                    identsToRenameMap.set(symbolIdString, newName.identifier);
                    synthNamesMap.set(symbolIdString, newName);
                    collidingSymbolMap.add(originalName, symbol);
                }
                else {
                    const identifier = getSynthesizedDeepClone(node);
                    synthNamesMap.set(symbolIdString, createSynthIdentifier(identifier));
                    collidingSymbolMap.add(originalName, symbol);
                }
            }
        }
    });

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
const processedRules = new WeakSet<Rule>()

function processRule(id: string, rule: Rule) {
  if (
    processedRules.has(rule) ||
    (rule.parent &&
      rule.parent.type === 'atrule' &&
      /-?keyframes$/.test((rule.parent as AtRule).name))
  ) {
    return
  }
  processedRules.add(rule)
  rule.selector = selectorParser(selectorRoot => {
    selectorRoot.each(selector => {
      rewriteSelector(id, selector, selectorRoot)
    })
  }).processSync(rule.selector)
}

/**
 * Walk an AssignmentPattern to determine if it contains object rest (`...`) syntax. We cannot rely on
 * propagation of `TransformFlags.ContainsObjectRestOrSpread` since it isn't propagated by default in
 * ObjectLiteralExpression and ArrayLiteralExpression since we do not know whether they belong to an
 * AssignmentPattern at the time the nodes are parsed.
 *
 * @internal
 */
private _statusIndicators: StatusIndicator[];
        constructor(config?: Partial<StatusConfig>) {
            if (!config) config = {};
            const start = config.start || "(";
            const end = config.end || ")";
            const full = config.full || "█";
            const empty = config.empty || Base.symbols.dot;
            const maxLen = Base.window.width - start.length - end.length - 30;
            const length = minMax(config.length || maxLen, 5, maxLen);
            this._config = {
                start,
                full,
                empty,
                end,
                length,
                noColors: config.noColors || false,
            };

            this._statusIndicators = [];
            this._lineCount = 0;
            this._active = false;
        }
