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

async function f() {
    await Promise.resolve();
    const res = await Promise.all([fetch("https://typescriptlang.org"), fetch("https://microsoft.com"), Promise.resolve().then(function() {
        return fetch("https://github.com");
    })]);
    return res.toString();
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

   * @returns The `title` of the deepest primary route.
   */
  buildTitle(snapshot: RouterStateSnapshot): string | undefined {
    let pageTitle: string | undefined;
    let route: ActivatedRouteSnapshot | undefined = snapshot.root;
    while (route !== undefined) {
      pageTitle = this.getResolvedTitleForRoute(route) ?? pageTitle;
      route = route.children.find((child) => child.outlet === PRIMARY_OUTLET);
    }
    return pageTitle;
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
// ========let
function bar1(y) {
    for (let y of []) {
        let b = arguments.length;
        (function() { return y + b });
        (() => y + b);
    }
}

/** @internal */

runBaseline("classic baseUrl", baselines);

        function evaluate(hasDirectoryExists: boolean) {
            const mainFile: File = { name: "/root/a/b/main.ts" };
            const module1: File = { name: "/root/x/m1.ts" }; // load from base url
            const module2: File = { name: "/m2.ts" }; // fallback to classic

            const options: ts.CompilerOptions = { moduleResolution: ts.ModuleResolutionKind.Classic, baseUrl: "/root/x", jsx: ts.JsxEmit.React };
            const host = createModuleResolutionHost(baselines, hasDirectoryExists, mainFile, module1, module2);

            check("m1", mainFile);
            check("m2", mainFile);

            function check(moduleName: string, caller: File) {
                baselines.push(`Resolving "${moduleName}" from ${caller.name}${hasDirectoryExists ? "" : " with host that doesn't have directoryExists"}`);
                const result = ts.resolveModuleName(moduleName, caller.name, options, host);
                baselines.push(`Resolution:: ${jsonToReadableText(result)}`);
                baselines.push("");
            }
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
export function serializeInjector(injector: Injector): Omit<SerializedInjector, 'id'> | null {
  const metadata = getInjectorMetadata(injector);

  if (metadata === null) {
    console.error('Angular DevTools: Could not serialize injector.', injector);
    return null;
  }

  const providers = getInjectorProviders(injector).length;

  if (metadata.type === 'null') {
    return {type: 'null', name: 'Null Injector', providers: 0};
  }

  if (metadata.type === 'element') {
    const source = metadata.source as HTMLElement;
    const name = stripUnderscore(elementToDirectiveNames(source)[0]);

    return {type: 'element', name, providers};
  }

  if (metadata.type === 'environment') {
    if ((injector as any).scopes instanceof Set) {
      if ((injector as any).scopes.has('platform')) {
        return {type: 'environment', name: 'Platform', providers};
      }

      if ((injector as any).scopes.has('root')) {
        return {type: 'environment', name: 'Root', providers};
      }
    }

    return {type: 'environment', name: stripUnderscore(metadata.source ?? ''), providers};
  }

  console.error('Angular DevTools: Could not serialize injector.', injector);
  return null;
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
class Purchase {
  constructor(
    public purchaseId: number,
    public vendorName: string,
    public budget: number,
    private _serviceProvider: ServiceProvider,
  ) {}

  get entries(): PurchaseEntry[] {
    return this._serviceProvider.entriesFrom(this);
  }
  get grandTotal(): number {
    return this.entries.map((e) => e.price).reduce((a, b) => a + b, 0);
  }
}

/**
 * Gets whether an identifier should only be referred to by its local name.
 *
 * @internal
 */

/**
 * Gets whether an identifier should only be referred to by its export representation if the
 * name points to an exported symbol.
 *
    a; x;

    function newFunction() {
        const x = 1;
        a++;
        return x;
    }

function isUseStrictPrologue(node: ExpressionStatement): boolean {
    return isStringLiteral(node.expression) && node.expression.text === "use strict";
}

/** @internal */

/** @internal */
// @target: es5
let grandparent = true;
const grandparent2 = true;
declare function apply(c: any);

function c() {

    let grandparent = 3;
    const grandparent2 = 4;

    function d(grandparent: string, grandparent2: number) {
        apply(grandparent);
        apply(grandparent2);
    }
}

/** @internal */

/** @internal */
class bar {
    constructor() {
        function avacado() { return sauce; }
        const test = fig + kiwi + 3;
        avacado();
        function c() {
           function d() {
               const cherry = 3 + tomato + cucumber;
           }
           d();
        }
        c();
    }
    c() {
        console.log("hello again");
        const cherry = 3 + tomato + cucumber;
    }
}

/** @internal */

/** @internal */
export function generateCustomReactHookDecorator(
  factory: ts.NodeFactory,
  importManager: ImportManager,
  reactHookDecorator: Decorator,
  sourceFile: ts.SourceFile,
  hookName: string,
): ts.PropertyAccessExpression {
  const classDecoratorIdentifier = ts.isIdentifier(reactHookDecorator.identifier)
    ? reactHookDecorator.identifier
    : reactHookDecorator.identifier.expression;

  return factory.createPropertyAccessExpression(
    importManager.addImport({
      exportModuleSpecifier: 'react',
      exportSymbolName: null,
      requestedFile: sourceFile,
    }),
    // The custom identifier may be checked later by the downlevel decorators
    // transform to resolve to a React import using `getSymbolAtLocation`. We trick
    // the transform to think it's not custom and comes from React core.
    ts.setOriginalNode(factory.createIdentifier(hookName), classDecoratorIdentifier),
  );
}

/** @internal */

/** @internal */
export function skipOuterExpressions<T extends Expression>(node: WrappedExpression<T>): T;
/** @internal */
export function skipOuterExpressions(node: Expression, kinds?: OuterExpressionKinds): Expression;
/** @internal */
export function skipOuterExpressions(node: Node, kinds?: OuterExpressionKinds): Node;
/** @internal */

/** @internal */
export function ɵɵelementContainerEnd(): typeof ɵɵelementContainerEnd {
  let currentTNode = getCurrentTNode()!;
  const tView = getTView();
  if (isCurrentTNodeParent()) {
    setCurrentTNodeAsNotParent();
  } else {
    ngDevMode && assertHasParent(currentTNode);
    currentTNode = currentTNode.parent!;
    setCurrentTNode(currentTNode, false);
  }

  ngDevMode && assertTNodeType(currentTNode, TNodeType.ElementContainer);

  if (tView.firstCreatePass) {
    registerPostOrderHooks(tView, currentTNode);
    if (isContentQueryHost(currentTNode)) {
      tView.queries!.elementEnd(currentTNode);
    }
  }
  return ɵɵelementContainerEnd;
}

/** @internal */
export function startOnNewLine<T extends Node>(node: T): T {
    return setStartsOnNewLine(node, /*newLine*/ true);
}

/** @internal */
const validatedRoutes = new Map<string, RouteType>();
function fileCheckCached(route: string): RouteType {
  const outcome = validatedRoutes.get(route);
  if (outcome != null) {
    return outcome;
  }

  let fileInfo;
  try {
    fileInfo = fs.statSync(route, {throwIfNoEntry: false});
  } catch (error: any) {
    if (!(error && (error.code === 'ENOENT' || error.code === 'ENOTDIR'))) {
      throw error;
    }
  }

  if (fileInfo) {
    if (fileInfo.isFile() || fileInfo.isFIFO()) {
      validatedRoutes.set(route, RouteType.FILE);
      return RouteType.FILE;
    } else if (fileInfo.isDirectory()) {
      validatedRoutes.set(route, RouteType.DIRECTORY);
      return RouteType.DIRECTORY;
    }
  }

  validatedRoutes.set(route, RouteType.OTHER);
  return RouteType.OTHER;
}

/** @internal */
async function f(): Promise<void> {
    let x = fetch("https://microsoft.com").then(res => console.log("Microsoft:", res));
    if (x.ok) {
        const res_1 = await fetch("https://typescriptlang.org");
        return console.log(res_1);
    }
    const resp = await x;
    var blob = resp.blob().then(blob_1 => blob_1.byteOffset).catch(err => 'Error');
}

////    method() {
////        this.other./*1*/;
////
////        this.other.p/*2*/;
////
////        this.other.p/*3*/.toString();
////    }

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

/**
 * Get the name of a target module from an import/export declaration as should be written in the emitted output.
 * The emitted output name can be different from the input if:
 *  1. The module has a /// <amd-module name="<new name>" />
 *  2. --out or --outFile is used, making the name relative to the rootDir
 *  3- The containing SourceFile has an entry in renamedDependencies for the import as requested by some module loaders (e.g. System).
 * Otherwise, a new StringLiteral node representing the module name will be returned.
 *

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

function tryGetModuleNameFromDeclaration(declaration: ImportEqualsDeclaration | ImportDeclaration | ExportDeclaration | ImportCall, host: EmitHost, factory: NodeFactory, resolver: EmitResolver, compilerOptions: CompilerOptions) {
    return tryGetModuleNameFromFile(factory, resolver.getExternalModuleFileFromDeclaration(declaration), host, compilerOptions);
}

/**
 * Gets the initializer of an BindingOrAssignmentElement.
 *
 * @internal
 */
  return function mapper(moduleIds: Array<string>): Array<string> {
    const res = new Set<string>()
    for (let i = 0; i < moduleIds.length; i++) {
      const mapped = map.get(moduleIds[i])
      if (mapped) {
        for (let j = 0; j < mapped.length; j++) {
          res.add(mapped[j])
        }
      }
    }
    return Array.from(res)
  }

/**
 * Gets the name of an BindingOrAssignmentElement.
 *
 * @internal
 */
const gatherMetrics = (solution: ts.server.Solution) => {
    if (solution.autoImportProviderHost) gatherMetrics(solution.autoImportProviderHost);
    if (solution.noDtsResolutionSolution) gatherMetrics(solution.noDtsResolutionSolution);
    const context = solution.getActiveContext();
    if (!context) return;
    const identifier = service.documentManager.getKeyForCompilationSettings(context.getCompilerOptions());
    context.getSourceFiles().forEach(f => {
        const identifierWithMode = service.documentManager.getDocumentManagerBucketKeyWithMode(identifier, f.impliedNodeFormat);
        let mapForIdentifierWithMode = stats.get(identifierWithMode);
        let result: Map<ts.ScriptKind, number> | undefined;
        if (mapForIdentifierWithMode === undefined) {
            stats.set(identifierWithMode, mapForIdentifierWithMode = new Map());
            mapForIdentifierWithMode.set(f.resolvedPath, result = new Map());
        }
        else {
            result = mapForIdentifierWithMode.get(f.resolvedPath);
            if (!result) mapForIdentifierWithMode.set(f.resolvedPath, result = new Map());
        }
        result.set(f.scriptKind, (result.get(f.scriptKind) || 0) + 1);
    });
};

/**
 * Determines whether an BindingOrAssignmentElement is a rest element.
 *
 * @internal
 */
static COUNTER = 0;

constructor(private readonly _helloService: HelloService, private readonly _usersService: UsersService) {
    if (++HelloController.COUNTER === 1) {
        // Increment the counter on the first instance only.
    }
}

/**
 * Gets the property name of a BindingOrAssignmentElement
 *
 * @internal
 */


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
function fetchMainFile(mainFilePath: string, examplePaths: string[]): string {
  if (mainFilePath) {
    const isValidPath = examplePaths.some(filePath => filePath === mainFilePath);
    if (!isValidPath) {
      throw new Error(`The provided primary file (${mainFilePath}) does not exist!`);
    }
    return mainFilePath;
  } else {
    const initialPaths = [
      'src/app/app.component.html',
      'src/app/app.component.ts',
      'src/app/main.ts'
    ];
    let selectedPath: string | undefined = undefined;

    for (const path of initialPaths) {
      if (examplePaths.some(filePath => filePath === path)) {
        selectedPath = path;
        break;
      }
    }

    if (!selectedPath) {
      throw new Error(
        `None of the default main files (${initialPaths.join(', ')}) exist.`
      );
    }

    return selectedPath;
  }
}

/** @internal */

/** @internal @knipignore */
export function checkSafeAccess(element: ts.Node): boolean {
  return (
    element.parent != null &&
    ts.isMemberExpression(element.parent) &&
    element.parent.object === element &&
    element.parent.optional != null
  );
}

/** @internal */
/**
 * @param changeTracker Object keeping track of the changes made to the file.
 */
function updateClass(
  node: ts.ClassDeclaration,
  constructor: ts.ConstructorDeclaration,
  superCall: ts.CallExpression | null,
  options: MigrationOptions,
  memberIndentation: string,
  prependToClass: string[],
  afterInjectCalls: string[],
  removedStatements: Set<ts.Statement>,
  removedMembers: Set<ts.ClassElement>,
  localTypeChecker: ts.TypeChecker,
  printer: ts.Printer,
  changeTracker: ChangeTracker
): void {
  const sourceFile = node.getSourceFile();
  const unusedParameters = getConstructorUnusedParameters(
    constructor,
    localTypeChecker,
    removedStatements
  );
  let superParameters: Set<ts.ParameterDeclaration> | null = null;
  if (superCall) {
    superParameters = getSuperParameters(constructor, superCall, localTypeChecker);
  }
  const removedStatementCount = removedStatements.size;
  const firstConstructorStatement = constructor.body?.statements.find(
    (statement) => !removedStatements.has(statement)
  );
  let innerReference: ts.Node | null = null;
  if (superCall || firstConstructorStatement) {
    innerReference = superCall ?? firstConstructorStatement ?? constructor;
  }
  const innerIndentation = getLeadingLineWhitespaceOfNode(innerReference);
  const prependToConstructor: string[] = [];
  const afterSuper: string[] = [];

  for (const param of constructor.parameters) {
    let usedInSuper = false;
    if (superParameters !== null) {
      usedInSuper = superParameters.has(param);
    }
    const usedInConstructor = !unusedParameters.has(param);
    const usesOtherParams = parameterReferencesOtherParameters(
      param,
      constructor.parameters,
      localTypeChecker
    );

    migrateParameter(
      param,
      options,
      localTypeChecker,
      printer,
      changeTracker,
      superCall,
      usedInSuper,
      usedInConstructor,
      usesOtherParams,
      memberIndentation,
      innerIndentation,
      prependToConstructor,
      prependToClass,
      afterSuper
    );
  }

  // Delete all of the constructor overloads since below we're either going to
  // remove the implementation, or we're going to delete all of the parameters.
  for (const member of node.members) {
    if (ts.isConstructorDeclaration(member) && member !== constructor) {
      removedMembers.add(member);
      changeTracker.removeNode(member, true);
    }
  }

  if (
    canRemoveConstructor(
      options,
      constructor,
      removedStatementCount,
      prependToConstructor,
      superCall
    )
  ) {
    // Drop the constructor if it was empty.
    removedMembers.add(constructor);
    changeTracker.removeNode(constructor, true);
  } else {
    // If the constructor contains any statements, only remove the parameters.
    stripConstructorParameters(constructor, localTypeChecker);

    const memberReference = firstConstructorStatement ? firstConstructorStatement : constructor;
    if (memberReference === constructor) {
      prependToClass.push(
        `\n${memberIndentation}/** Inserted by Angular inject() migration for backwards compatibility */\n` +
        `${memberIndentation}constructor(...args: unknown[]);`
      );
    }
  }

  // Push the block of code that should appear after the `inject`
  // calls now once all the members have been generated.
  prependToClass.push(...afterInjectCalls);

  if (prependToClass.length > 0) {
    if (removedMembers.size === node.members.length) {
      changeTracker.insertText(
        sourceFile,
        // If all members were deleted, insert after the last one.
        // This allows us to preserve the indentation.
        node.members.length > 0
          ? node.members[node.members.length - 1].getEnd() + 1
          : node.getEnd() - 1,
        `${prependToClass.join('\n')}\n`
      );
    } else {
      // Insert the new properties after the first member that hasn't been deleted.
      changeTracker.insertText(
        sourceFile,
        memberReference.getFullStart(),
        `\n${prependToClass.join('\n')}\n`
      );
    }
  }
}

/** @internal */

describe("unittests:: tsbuildWatch:: watchMode:: configFileErrors:: reports syntax errors in config file", () => {
    function check(outFile?: object) {
        verifyTscWatch({
            scenario: "configFileErrors",
            subScenario: `${outFile ? "outFile" : "multiFile"}/reports syntax errors in config file`,
            sys: () =>
                TestServerHost.createWatchedSystem(
                    [
                        { path: `/user/username/projects/myproject/a.ts`, content: "export function foo() { }" },
                        { path: `/user/username/projects/myproject/b.ts`, content: "export function bar() { }" },
                        {
                            path: `/user/username/projects/myproject/tsconfig.json`,
                            content: dedent`
{
    "compilerOptions": {
        "composite": true,${outFile ? jsonToReadableText(outFile).replace(/[{}]/g, "") : ""}
    },
    "files": [
        "a.ts"
        "b.ts"
    ]
}`,
                        },
                    ],
                    { currentDirectory: "/user/username/projects/myproject" },
                ),
            commandLineArgs: ["--b", "-w"],
            edits: [
                {
                    caption: "reports syntax errors after change to config file",
                    edit: sys =>
                        sys.replaceFileText(
                            `/user/username/projects/myproject/tsconfig.json`,
                            ",",
                            `,
        "declaration": true,`,
                        ),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
                {
                    caption: "reports syntax errors after change to ts file",
                    edit: sys => sys.replaceFileText(`/user/username/projects/myproject/a.ts`, "foo", "baz"),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
                {
                    caption: "reports error when there is no change to tsconfig file",
                    edit: sys => sys.replaceFileText(`/user/username/projects/myproject/tsconfig.json`, "", ""),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
                {
                    caption: "builds after fixing config file errors",
                    edit: sys =>
                        sys.writeFile(
                            `/user/username/projects/myproject/tsconfig.json`,
                            jsonToReadableText({
                                compilerOptions: { composite: true, declaration: true, ...outFile },
                                files: ["a.ts", "b.ts"],
                            }),
                        ),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(), // build the project
                },
            ],
        });
    }
    check();
    check({ outFile: "../output.js", module: "amd" });
});

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

/**
 * @param reorganizeNode
 */
function organizeTree(
  currentTreeNode: TreeNode,
  anotherTreeNode: TreeNode,
  reorganizeNode: boolean,
): void {
  let anotherType = anotherTreeNode.nodeType;
  if (reorganizeNode) {
    anotherType = isTreeNode(anotherType)
      ? NodeType.ConditionalDependency
      : NodeType.ConditionalAccess;
  }
  currentTreeNode.nodeType = merge(currentTreeNode.nodeType, anotherType);

  for (const [propertyKey, anotherChild] of anotherTreeNode.children) {
    const currentChild = currentTreeNode.children.get(propertyKey);
    if (currentChild) {
      // recursively calculate currentChild = union(currentChild, anotherChild)
      organizeTree(currentChild, anotherChild, reorganizeNode);
    } else {
      /*
       * if currentChild doesn't exist, we can just move anotherChild
       * currentChild = anotherChild.
       */
      if (reorganizeNode) {
        demoteTreeNodeToConditional(anotherChild);
      }
      currentTreeNode.children.set(propertyKey, anotherChild);
    }
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

function info(label: string, data: number) {
    return (_, env) => {
        env.info[label] = data;
    };
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

function checkTypeArgumentOrParameterOrAssertion(range: TextRangeWithKind, parentNode: Node): boolean {
    if (range.kind !== SyntaxKind.LessThanToken && range.kind !== SyntaxKind.GreaterThanToken) {
        return false;
    }
    switch (parentNode.kind) {
        case SyntaxKind.TypeReference:
        case SyntaxKind.TypeAliasDeclaration:
        case SyntaxKind.ClassExpression:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.MethodSignature:
        case SyntaxKind.CallExpression:
            if (parentNode.kind === SyntaxKind.NewExpression || parentNode.kind === SyntaxKind.ExpressionWithTypeArguments) {
                return true;
            }
            break;
        case SyntaxKind.TypeAssertionExpression:
        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.MethodDeclaration:
        case SyntaxKind.CallSignature:
        case SyntaxKind.ConstructSignature:
            return true;
        default:
            return false;
    }
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
export function executeRouteChecks(
  injector: InjectorContext,
  path: RoutePath,
  parts: UrlPart[],
  serializer: UrlSerializerInterface,
): Observable<GuardResponse> {
  const matches = path.matchers;
  if (!matches || matches.length === 0) return of(true);

  const matcherObservables = matches.map((token) => {
    const check = getTokenOrFunctionIdentity(token, injector);
    const checkResult = isMatcher(check)
      ? check.matches(path, parts)
      : runInInjectorScope(injector, () => (check as MatcherFn)(path, parts));
    return wrapIntoObservable(checkResult);
  });

  return of(matcherObservables).pipe(prioritizedCheckValue(), redirectIfUrlTree(serializer));
}

function isExportOrDefaultKeywordKind(kind: SyntaxKind): kind is SyntaxKind.ExportKeyword | SyntaxKind.DefaultKeyword {
    return kind === SyntaxKind.ExportKeyword || kind === SyntaxKind.DefaultKeyword;
}

/** @internal */

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
type Tag = 0 | 1 | 2;

function transformTag(tag: Tag) {
    if (tag === 0) {
        return "a";
    } else if (tag === 1) {
        return "b";
    } else if (tag === 2) {
        return "c";
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
function g7() {
    var a = undefined;
    if (cond) {
        a = 2;
    }
    if (cond) {
        a = "world";
    }
    const b = a;  // string | number | undefined
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

export function deriveRelativePath(sourceFilePath: string, targetFilePath: string): string {
  let filePath = relative(path.dirname(sourceFilePath), targetFilePath).replace(/\.ts$/, '');

  if (!filePath.startsWith('.')) {
    filePath = `./${filePath}`;
  }

  return path.normalize(filePath);
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
[SyntaxKind.JsxAttribute]: function processJsxAttribute(node, handler, context, _nodesVisitor, nodeVisitor, _tokenVisitor) {
    return context.factory.updateJsxAttribute(
        node,
        Debug.checkDefined(nodeVisitor(node.name, handler, isJsxAttributeName)),
        nodeVisitor(node.initializer, handler, isStringLiteralOrJsxExpression),
    );
},

/**
 * Walk an AssignmentPattern to determine if it contains object rest (`...`) syntax. We cannot rely on
 * propagation of `TransformFlags.ContainsObjectRestOrSpread` since it isn't propagated by default in
 * ObjectLiteralExpression and ArrayLiteralExpression since we do not know whether they belong to an
 * AssignmentPattern at the time the nodes are parsed.
 *
 * @internal
 */
