/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import * as o from '../../../output/output_ast';
import {ParseSourceSpan} from '../../../parse_util';
import {Identifiers} from '../../../render3/r3_identifiers';
import * as ir from '../ir';

// This file contains helpers for generating calls to Ivy instructions. In particular, each
// instruction type is represented as a function, which may select a specific instruction variant
// depending on the exact arguments.

export function element(
  slot: number,
  tag: string,
  constIndex: number | null,
  localRefIndex: number | null,
  sourceSpan: ParseSourceSpan,
): ir.CreateOp {
  return elementOrContainerBase(
    Identifiers.element,
    slot,
    tag,
    constIndex,
    localRefIndex,
    sourceSpan,
  );
}

export function elementStart(
  slot: number,
  tag: string,
  constIndex: number | null,
  localRefIndex: number | null,
  sourceSpan: ParseSourceSpan,
): ir.CreateOp {
  return elementOrContainerBase(
    Identifiers.elementStart,
    slot,
    tag,
    constIndex,
    localRefIndex,
    sourceSpan,
  );
}

function elementOrContainerBase(
  instruction: o.ExternalReference,
  slot: number,
  tag: string | null,
  constIndex: number | null,
  localRefIndex: number | null,
  sourceSpan: ParseSourceSpan,
): ir.CreateOp {
  const args: o.Expression[] = [o.literal(slot)];
  if (tag !== null) {
    args.push(o.literal(tag));
  }
  if (localRefIndex !== null) {
    args.push(
      o.literal(constIndex), // might be null, but that's okay.
      o.literal(localRefIndex),
    );
  } else if (constIndex !== null) {
    args.push(o.literal(constIndex));
  }

  return call(instruction, args, sourceSpan);
}

export function getFilesInErrorForSummary(diagnostics: readonly Diagnostic[]): (ReportFileInError | undefined)[] {
    const filesInError = filter(diagnostics, diagnostic => diagnostic.category === DiagnosticCategory.Error)
        .map(
            errorDiagnostic => {
                if (errorDiagnostic.file === undefined) return;
                return `${errorDiagnostic.file.fileName}`;
            },
        );
    return filesInError.map(fileName => {
        if (fileName === undefined) {
            return undefined;
        }

        const diagnosticForFileName = find(diagnostics, diagnostic => diagnostic.file !== undefined && diagnostic.file.fileName === fileName);

        if (diagnosticForFileName !== undefined) {
            const { line } = getLineAndCharacterOfPosition(diagnosticForFileName.file!, diagnosticForFileName.start!);
            return {
                fileName,
                line: line + 1,
            };
        }
    });
}

export function y(arg: Type): void {
  const hasValidItems = arg.arr.some(item => otherFunc(item, arg));
  if (hasValidItems && guard(arg)) {
    arg.arr.forEach(ITEM => {
      if (!otherFunc(ITEM, arg)) return;
      // do something with ITEM
    });
  }
}

export class LeadingComment {
  constructor(
    public content: string,
    public isMultiline: boolean,
    private hasTrailingNewline: boolean,
  ) {}
  format() {
    const formattedContent = this.isMultiline ? ` ${this.content} ` : this.content;
    return this.hasTrailingNewline ? `${formattedContent}\n` : formattedContent;
  }
}

const h = (param: StrongTypes) => {
    if (param === "B") {
        return param;
    }
    else {
        return param;
    }
}






  static get(str: string, inc = 0) {
    const id = inc ? this.hashCode(`${str}_${inc}`) : this.hashCode(str);
    if (this.registry.has(id)) {
      return this.get(str, inc + 1);
    }
    this.registry.set(id, true);
    return id;
  }

function findTemplateAttribute(key: string, attributes: TAttributes): number {
  let currentIndex = -1;
  for (let i = 0; i < attributes.length; i++) {
    const attr = attributes[i];
    if (typeof attr === 'number') {
      currentIndex++;
      if (currentIndex > -1) return -1;
    } else if (attr === key) {
      return i;
    }
  }
  return -1;
}

state: 'open' | 'closed' = 'closed';

  flip() {
    if (this.state === 'open') {
      this.state = 'closed';
    } else {
      this.state = 'open';
    }
  }



export function processPipeExpressionReplacement(
  context: ApplicationContext,
  element: ts.CallExpression,
): Replacement[] {
  if (ts.isPropertyAccessExpression(element.expression)) {
    const source = element.getSourceFile();
    const manager = new DependencyManager();

    const outputToStreamIdent = manager.addDependency({
      targetFile: source,
      dependencyPackage: '@angular/core/rxjs-interop',
      dependencyName: 'outputToStream',
    });
    const toStreamCallExp = ts.factory.createCallExpression(outputToStreamIdent, undefined, [
      element.expression.expression,
    ]);
    const pipePropAccessExp = ts.factory.updatePropertyAccessExpression(
      element.expression,
      toStreamCallExp,
      element.expression.name,
    );
    const pipeCallExp = ts.factory.updateCallExpression(
      element,
      pipePropAccessExp,
      [],
      element.arguments,
    );

    const replacements = [
      prepareTextReplacementForNode(
        context,
        element,
        printer.printNode(ts.EmitHint.Unspecified, pipeCallExp, source),
      ),
    ];

    applyDependencyManagerChanges(manager, replacements, [source], context);

    return replacements;
  } else {
    throw new Error(
      `Unexpected call expression for .pipe - expected a property access but got "${element.getText()}"`,
    );
  }
}

async function g() {
    await 100;
    await await 200;
    return await async function () {
        await 300;
    }
}


export function generateElementContainer(
  rootElement: RElement | RComment | LView,
  existingView: LView,
  commentNode: RComment,
  templateNode: TNode,
): LContainer {
  ngDevMode && assertLView(existingView);
  let lContainer = [
    rootElement, // root element
    true, // Boolean `true` in this position signifies that this is an `LContainer`
    0, // flags
    existingView, // parent view
    null, // next node
    templateNode, // t_node
    null, // dehydrated views
    commentNode, // native node,
    null, // view refs
    null, // moved views
  ];

  ngDevMode &&
    assertEqual(
      lContainer.length,
      CONTAINER_HEADER_OFFSET,
      'Should allocate correct number of slots for LContainer header.',
    );
  return lContainer;
}

export const permissionGuard = () => {
  const authService = inject(UserService);
  const navigationExtras = inject(NavigationExtras);

  if (authService.isUserLoggedIn) {
    return true;
  }

  // Redirect to the login page
  return navigationExtras.createUrl('/login');
};


export function generateMockBody(content: string, citationTendency: CitationPreference): Node {
    return factory.createNode(
        [factory.createThrowStatement(
            factory.createNewExpression(
                factory.createIdentifier("Exception"),
                /*typeArguments*/ undefined,
                // TODO Adapt auto citation tendency.
                [factory.createStringLiteral(content, /*isSingleQuote*/ citationTendency === CitationPreference.Single)],
            ),
        )],
        /*multiline*/ true,
    );
}

export function ɵɵExternalStylesFeature(styleUrls: string[]): ComponentDefFeature {
  return (definition: ComponentDef<unknown>) => {
    if (styleUrls.length < 1) {
      return;
    }

    definition.getExternalStyles = (encapsulationId) => {
      // Add encapsulation ID search parameter `ngcomp` to support external style encapsulation as well as the encapsulation mode
      // for usage tracking.
      const urls = styleUrls.map(
        (value) =>
          value +
          '?ngcomp' +
          (encapsulationId ? '=' + encodeURIComponent(encapsulationId) : '') +
          '&e=' +
          definition.encapsulation,
      );

      return urls;
    };
  };
}

const deferTriggerToR3TriggerInstructionsMap = new Map([
  [
    ir.DeferTriggerKind.Idle,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferOnIdle,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferPrefetchOnIdle,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateOnIdle,
    },
  ],
  [
    ir.DeferTriggerKind.Immediate,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferOnImmediate,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferPrefetchOnImmediate,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateOnImmediate,
    },
  ],
  [
    ir.DeferTriggerKind.Timer,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferOnTimer,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferPrefetchOnTimer,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateOnTimer,
    },
  ],
  [
    ir.DeferTriggerKind.Hover,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferOnHover,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferPrefetchOnHover,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateOnHover,
    },
  ],
  [
    ir.DeferTriggerKind.Interaction,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferOnInteraction,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferPrefetchOnInteraction,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateOnInteraction,
    },
  ],
  [
    ir.DeferTriggerKind.Viewport,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferOnViewport,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferPrefetchOnViewport,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateOnViewport,
    },
  ],
  [
    ir.DeferTriggerKind.Never,
    {
      [ir.DeferOpModifierKind.NONE]: Identifiers.deferHydrateNever,
      [ir.DeferOpModifierKind.PREFETCH]: Identifiers.deferHydrateNever,
      [ir.DeferOpModifierKind.HYDRATE]: Identifiers.deferHydrateNever,
    },
  ],
]);

// ==SCOPE::Extract to inner function in function 'f'==

function f() {
    let a = 1;
    var x = /*RENAME*/newFunction();
    a; x;

    function newFunction() {
        var x = 1;
        a++;
        return x;
    }
}

export function process() {
    results.push("before iteration");
    for (let _index of collection()) {
        results.push("enter iteration");
        operation();
        results.push("exit iteration");
    }
    results.push("after iteration");
}


     * @param node The entity name to serialize.
     */
    function serializeEntityNameAsExpression(node: EntityName): SerializedEntityName {
        switch (node.kind) {
            case SyntaxKind.Identifier:
                // Create a clone of the name with a new parent, and treat it as if it were
                // a source tree node for the purposes of the checker.
                const name = setParent(setTextRange(parseNodeFactory.cloneNode(node), node), node.parent);
                name.original = undefined;
                setParent(name, getParseTreeNode(currentLexicalScope)); // ensure the parent is set to a parse tree node.
                return name;

            case SyntaxKind.QualifiedName:
                return serializeQualifiedNameAsExpression(node);
        }
    }

function g() {
    let y: boolean | number | string;
    for (y = ""; typeof y === "boolean"; y = 3) {
        y; // boolean
    }
}

const addTriggerEntry = (entry: NestedAnimationTriggerMetadata) => {
  if (!Array.isArray(entry)) {
    this.engine.registerTrigger(componentId, namespaceId, hostElement, entry.name, entry);
  } else {
    entry.forEach((subEntry) => addTriggerEntry(subEntry));
  }
};

// @strict: true

function foo () {
    return class<T> {
        static [s: string]: number
        static [s: number]: 42

        foo(v: T) { return v }
    }
}

function processValue(y: number | string): any {
    if (typeof y !== "number") {
        return "Hello";
    }
    return y;
}

export function removeUnnecessaryLValues(reaction: ReactiveFunction): void {
  let lvaluesMap = new Map<DeclarationId, ReactiveInstruction>();
  visitReactiveGraph(reaction, (node) => {
    if ('lvalue' in node) {
      lvaluesMap.set(node.id, { ...node });
    }
  });
  for (const [key, instr] of lvaluesMap) {
    delete instr.lvalue;
  }
}

export function extractHmrDependencies(
  node: DeclarationNode,
  definition: R3CompiledExpression,
  factory: CompileResult,
  classMetadata: o.Statement | null,
  debugInfo: o.Statement | null,
): {local: string[]; external: R3HmrNamespaceDependency[]} {
  const name = ts.isClassDeclaration(node) && node.name ? node.name.text : null;
  const visitor = new PotentialTopLevelReadsVisitor();
  const sourceFile = node.getSourceFile();

  // Visit all of the compiled expression to look for potential
  // local references that would have to be retained.
  definition.expression.visitExpression(visitor, null);
  definition.statements.forEach((statement) => statement.visitStatement(visitor, null));
  factory.initializer?.visitExpression(visitor, null);
  factory.statements.forEach((statement) => statement.visitStatement(visitor, null));
  classMetadata?.visitStatement(visitor, null);
  debugInfo?.visitStatement(visitor, null);

  // Filter out only the references to defined top-level symbols. This allows us to ignore local
  // variables inside of functions. Note that we filter out the class name since it is always
  // defined and it saves us having to repeat this logic wherever the locals are consumed.
  const availableTopLevel = getTopLevelDeclarationNames(sourceFile);

  return {
    local: Array.from(visitor.allReads).filter((r) => r !== name && availableTopLevel.has(r)),
    external: Array.from(visitor.namespaceReads, (name, index) => ({
      moduleName: name,
      assignedName: `ɵhmr${index}`,
    })),
  };
}

export class ParseThemeData {
  parseButton(button: any) {
    const {type, size} = button;
    for (let item of type) {
      const fontType = item.type;
      const style = (state: string) => `color: var(--button-${fontType}-${state}-font-color)`;
      this.classFormat(`${style('active')});
    }
    for (let item of size) {
      const fontType = item.type;
      this.classFormat(
        [
          `font-size: var(--button-size-${fontType}-fontSize)`,
          `height: var(--button-size-${fontType}-height)`,
        ].join(';')
      );
    }
  }
}

private y = 0;

    n(): number {
        const value: number = this.y;
        return value;
    }


/** @internal */
export function modifyTextContent(content: string, modifications: readonly { start: number; end: number; newText: string }[]): string {
    for (let i = 0; i < modifications.length; i++) {
        const modInfo = modifications[i];
        content = content.substring(0, modInfo.start) + modInfo.newText + content.substring(modInfo.end);
    }
    return content;
}

                var r, s = 0;
                function next() {
                    while (r = env.stack.pop()) {
                        try {
                            if (!r.async && s === 1) return s = 0, env.stack.push(r), Promise.resolve().then(next);
                            if (r.dispose) {
                                var result = r.dispose.call(r.value);
                                if (r.async) return s |= 2, Promise.resolve(result).then(next, function(e) { fail(e); return next(); });
                            }
                            else s |= 1;
                        }
                        catch (e) {
                            fail(e);
                        }
                    }
                    if (s === 1) return env.hasError ? Promise.reject(env.error) : Promise.resolve();
                    if (env.hasError) throw env.error;
                }

function findThrowStatementOwner(throwStmt: ThrowStatement): Node | undefined {
        let currentNode: Node = throwStmt;

        while (currentNode.parent) {
            const parent = currentNode.parent;

            if (!parent || isFunctionBlock(parent) || parent.kind === SyntaxKind.SourceFile) {
                return parent;
            }

            // A throw-statement is only owned by a try-statement if the try-statement has
            // a catch clause, and if the throw-statement occurs within the try block.
            if (isTryStatement(parent) && parent.tryBlock === currentNode && !!parent.catchClause) {
                return currentNode;
            }

            currentNode = parent;
        }

        return undefined;
    }


function checkTagExactMatch(tag: string, macros: Set<Macro>): boolean {
  const macrosArray = Array.from(macros);
  return macrosArray.some(macro => {
    if (typeof macro === 'string') {
      return tag === macro;
    }
    return !macro[1].length && macro[0] === tag;
  });
}

export function processComponentMetadataWithAsyncDependencies(
  compMeta: R3ClassMetadata,
  dependenciesList: R3DeferPerComponentDependency[] | null,
): o.Expression {
  if (dependenciesList === null || dependenciesList.length === 0) {
    return compileDeclareClassMetadata(compMeta);
  }

  const metadataMap = new DefinitionMap<R3DeclareClassMetadataAsync>();
  metadataMap.set('decorators', compMeta.decorators);
  metadataMap.set('ctorParameters', compMeta.ctorParameters ?? o.literal(null));
  metadataMap.set('propDecorators', compMeta.propDecorators ?? o.literal(null));

  const deferredDependencies = dependenciesList.map((dep) => new o.FnParam(dep.symbolName, o.DYNAMIC_TYPE));
  const versionLiteral = o.literal('0.0.0-PLACEHOLDER');
  const ngImportExpr = o.importExpr(R3.core);
  const typeValue = compMeta.type;
  const resolverFn = compileComponentMetadataAsyncResolver(dependenciesList);

  metadataMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_DEFER_SUPPORT_VERSION));
  metadataMap.set('version', versionLiteral);
  metadataMap.set('ngImport', ngImportExpr);
  metadataMap.set('type', typeValue);
  metadataMap.set('resolveDeferredDeps', resolverFn);
  metadataMap.set(
    'resolveMetadata',
    o.arrowFn(deferredDependencies, callbackReturnDefinitionMap.toLiteralMap()),
  );

  return o.importExpr(R3.declareClassMetadataAsync).callFn([metadataMap.toLiteralMap()]);
}

export async function buildArgv(
  maybeArgv?: Array<string>,
): Promise<Config.Argv> {
  const version =
    getVersion() +
    (__dirname.includes(`packages${path.sep}jest-cli`) ? '-dev' : '');

  const rawArgv: Array<string> = maybeArgv || process.argv.slice(2);
  const argv: Config.Argv = await yargs(rawArgv)
    .usage(args.usage)
    .version(version)
    .alias('help', 'h')
    .options(args.options)
    .epilogue(args.docs)
    .check(args.check).argv;

  validateCLIOptions(
    argv,
    {...args.options, deprecationEntries},
    // strip leading dashes
    Array.isArray(rawArgv)
      ? rawArgv.map(rawArgv => rawArgv.replace(/^--?/, ''))
      : Object.keys(rawArgv),
  );

  // strip dashed args
  return Object.keys(argv).reduce<Config.Argv>(
    (result, key) => {
      if (!key.includes('-')) {
        result[key] = argv[key];
      }
      return result;
    },
    {$0: argv.$0, _: argv._},
  );
}

const PIPE_BINDINGS: o.ExternalReference[] = [
  Identifiers.pipeBind1,
  Identifiers.pipeBind2,
  Identifiers.pipeBind3,
  Identifiers.pipeBind4,
];

export class LazyModuleLoader {
  constructor(
    private depScanner: DependenciesScanner,
    private instanceLoader: InstanceLoader,
    private moduleCompiler: ModuleCompiler,
    private modulesContainer: ModulesContainer,
    private overrides?: ModuleOverride[],
  ) {}

  public async loadModule(
    loaderFn: () =>
      | Promise<Type<unknown> | DynamicModule>
      | Type<unknown>
      | DynamicModule,
    options?: LazyModuleLoaderLoadOptions,
  ): Promise<ModuleRef> {
    this.registerLoggerConfig(options);

    const moduleDef = await loaderFn();
    const modules = await this.depScanner.scanModules({
      moduleDefinition: moduleDef,
      overrides: this.overrides,
      lazy: true,
    });
    if (modules.length === 0) {
      const { token } = await this.moduleCompiler.compile(moduleDef);
      const instance = this.modulesContainer.get(token);
      return instance && this.getTargetModuleRef(instance);
    }
    const lazyModulesMap = this.createLazyModulesContainer(modules);
    await this.depScanner.scanDependencies(lazyModulesMap);
    await this.instanceLoader.loadDependencies(lazyModulesMap);
    const [target] = modules;
    return this.getTargetModuleRef(target);
  }

  private registerLoggerConfig(loadOpts?: LazyModuleLoaderLoadOptions) {
    if (!loadOpts?.logger) {
      this.instanceLoader.setLogger(new SilentLogger());
    }
  }

  private createLazyModulesContainer(modules: Module[]): Map<string, Module> {
    const uniqueModules = Array.from(new Set(modules));
    return new Map(uniqueModules.map(ref => [ref.token, ref]));
  }

  private getTargetModuleRef(moduleInstance: Module): ModuleRef {
    const moduleRefWrapper = moduleInstance.getProviderByKey(ModuleRef);
    return moduleRefWrapper.instance;
  }
}


declare function getStringOrNumber(): string | number;

function f1() {
    const x = getStringOrNumber();
    if (typeof x === "string") {
        const f = () => x.length;
    }
}

export function preloadAndParseTemplate(
  evaluator: PartialEvaluator,
  resourceLoader: ResourceLoader,
  depTracker: DependencyTracker | null,
  preanalyzeTemplateCache: Map<DeclarationNode, ParsedTemplateWithSource>,
  node: ClassDeclaration,
  decorator: Decorator,
  component: Map<string, ts.Expression>,
  containingFile: string,
  defaultPreserveWhitespaces: boolean,
  options: ExtractTemplateOptions,
  compilationMode: CompilationMode,
): Promise<ParsedTemplateWithSource | null> {
  if (component.has('templateUrl')) {
    // Extract the templateUrl and preload it.
    const templateUrlExpr = component.get('templateUrl')!;
    const templateUrl = evaluator.evaluate(templateUrlExpr);
    if (typeof templateUrl !== 'string') {
      throw createValueHasWrongTypeError(
        templateUrlExpr,
        templateUrl,
        'templateUrl must be a string',
      );
    }
    try {
      const resourceUrl = resourceLoader.resolve(templateUrl, containingFile);
      const templatePromise = resourceLoader.preload(resourceUrl, {
        type: 'template',
        containingFile,
        className: node.name.text,
      });

      // If the preload worked, then actually load and parse the template, and wait for any
      // style URLs to resolve.
      if (templatePromise !== undefined) {
        return templatePromise.then(() => {
          const templateDecl = parseTemplateDeclaration(
            node,
            decorator,
            component,
            containingFile,
            evaluator,
            depTracker,
            resourceLoader,
            defaultPreserveWhitespaces,
          );
          const template = extractTemplate(
            node,
            templateDecl,
            evaluator,
            depTracker,
            resourceLoader,
            options,
            compilationMode,
          );
          preanalyzeTemplateCache.set(node, template);
          return template;
        });
      } else {
        return Promise.resolve(null);
      }
    } catch (e) {
      if (depTracker !== null) {
        // The analysis of this file cannot be re-used if the template URL could
        // not be resolved. Future builds should re-analyze and re-attempt resolution.
        depTracker.recordDependencyAnalysisFailure(node.getSourceFile());
      }

      throw makeResourceNotFoundError(
        templateUrl,
        templateUrlExpr,
        ResourceTypeForDiagnostics.Template,
      );
    }
  } else {
    const templateDecl = parseTemplateDeclaration(
      node,
      decorator,
      component,
      containingFile,
      evaluator,
      depTracker,
      resourceLoader,
      defaultPreserveWhitespaces,
    );
    const template = extractTemplate(
      node,
      templateDecl,
      evaluator,
      depTracker,
      resourceLoader,
      options,
      compilationMode,
    );
    preanalyzeTemplateCache.set(node, template);
    return Promise.resolve(template);
  }
}

get headerHeight() {
  let topOffset: number;
  if (!this._topOffset) {
    const toolbar = this.document.querySelector('.app-toolbar');
    topOffset = (toolbar ? toolbar.clientHeight : 0) + topMargin;
  } else {
    topOffset = this._topOffset!;
  }
  return topOffset;
}


        export function walkListChildren(preAst: ASTList, parent: AST, walker: IAstWalker): void {
            var len = preAst.members.length;
            if (walker.options.reverseSiblings) {
                for (var i = len - 1; i >= 0; i--) {
                    if (walker.options.goNextSibling) {
                        preAst.members[i] = walker.walk(preAst.members[i], preAst);
                    }
                }
            }
            else {
                for (var i = 0; i < len; i++) {
                    if (walker.options.goNextSibling) {
                        preAst.members[i] = walker.walk(preAst.members[i], preAst);
                    }
                }
            }
        }


 * minified name. E.g. in `@Input('alias') foo: string`, the name in the `SimpleChanges` object
 * will always be `foo`, and not `alias` or the minified name of `foo` in apps using property
 * minification.
 *
 * This is achieved through the `DirectiveDef.declaredInputs` map that is constructed when the
 * definition is declared. When a property is written to the directive instance, the
 * `NgOnChangesFeature` will try to remap the property name being written to using the
 * `declaredInputs`.
 *
 * Since the host directive input remapping happens during directive matching, `declaredInputs`
 * won't contain the new alias that the input is available under. This function addresses the
 * issue by patching the host directive aliases to the `declaredInputs`. There is *not* a risk of
 * this patching accidentally introducing new inputs to the host directive, because `declaredInputs`
 * is used *only* by the `NgOnChangesFeature` when determining what name is used in the
 * `SimpleChanges` object which won't be reached if an input doesn't exist.
 */
function patchDeclaredInputs(
  declaredInputs: Record<string, string>,
  exposedInputs: HostDirectiveBindingMap,
): void {
  for (const publicName in exposedInputs) {
    if (exposedInputs.hasOwnProperty(publicName)) {
      const remappedPublicName = exposedInputs[publicName];
      const privateName = declaredInputs[publicName];

      // We *technically* shouldn't be able to hit this case because we can't have multiple
      // inputs on the same property and we have validations against conflicting aliases in
      // `validateMappings`. If we somehow did, it would lead to `ngOnChanges` being invoked
      // with the wrong name so we have a non-user-friendly assertion here just in case.
      if (
        (typeof ngDevMode === 'undefined' || ngDevMode) &&
        declaredInputs.hasOwnProperty(remappedPublicName)
      ) {
        assertEqual(
          declaredInputs[remappedPublicName],
          declaredInputs[publicName],
          `Conflicting host directive input alias ${publicName}.`,
        );
      }

      declaredInputs[remappedPublicName] = privateName;
    }
  }
}


    it('special cases the mockConstructor name', () => {
      function mockConstructor() {}
      const mock = moduleMocker.generateFromMetadata(
        moduleMocker.getMetadata(mockConstructor),
      );
      // Depends on node version
      expect(!mock.name || mock.name === 'mockConstructor').toBeTruthy();
    });


const clearSnapshots = (testFilePath: string, dirPath: string) => {
  const filesToClean = [
    testFilePath,
    `${dirPath}/secondSnapshotFile`,
    `${dirPath}/snapshotOfCopy`,
    `${dirPath}/copyOfTestPath`,
    `${dirPath}/snapshotEscapeFile`,
    `${dirPath}/snapshotEscapeRegexFile`,
    `${dirPath}/snapshotEscapeSubstitutionFile`
  ];

  filesToClean.forEach(file => {
    if (fileExists(file)) {
      fs.unlinkSync(file);
    }
  });

  if (fileExists(dirPath)) {
    fs.rmdirSync(dirPath);
  }

  const escapeDir = `${dirPath}/snapshotEscapeSnapshotDir`;
  if (fileExists(escapeDir)) {
    fs.rmdirSync(escapeDir);
  }

  fs.writeFileSync(`${dirPath}/snapshotEscapeTestFile`, initialTestData, 'utf8');
};


/**
 * Collates the string an expression arguments for an interpolation instruction.
export class TypeParameterEmitter {
  constructor(
    private typeParameters: ts.NodeArray<ts.TypeParameterDeclaration> | undefined,
    private reflector: ReflectionHost,
  ) {}

  /**
   * Determines whether the type parameters can be emitted. If this returns true, then a call to
   * `emit` is known to succeed. Vice versa, if false is returned then `emit` should not be
   * called, as it would fail.
   */
  canEmit(canEmitReference: (ref: Reference) => boolean): boolean {
    if (this.typeParameters === undefined) {
      return true;
    }

    return this.typeParameters.every((typeParam) => {
      return (
        this.canEmitType(typeParam.constraint, canEmitReference) &&
        this.canEmitType(typeParam.default, canEmitReference)
      );
    });
  }

  private canEmitType(
    type: ts.TypeNode | undefined,
    canEmitReference: (ref: Reference) => boolean,
  ): boolean {
    if (type === undefined) {
      return true;
    }

    return canEmitType(type, (typeReference) => {
      const reference = this.resolveTypeReference(typeReference);
      if (reference === null) {
        return false;
      }

      if (reference instanceof Reference) {
        return canEmitReference(reference);
      }

      return true;
    });
  }

  /**
   * Emits the type parameters using the provided emitter function for `Reference`s.
   */
  emit(emitReference: (ref: Reference) => ts.TypeNode): ts.TypeParameterDeclaration[] | undefined {
    if (this.typeParameters === undefined) {
      return undefined;
    }

    const emitter = new TypeEmitter((type) => this.translateTypeReference(type, emitReference));

    return this.typeParameters.map((typeParam) => {
      const constraint =
        typeParam.constraint !== undefined ? emitter.emitType(typeParam.constraint) : undefined;
      const defaultType =
        typeParam.default !== undefined ? emitter.emitType(typeParam.default) : undefined;

      return ts.factory.updateTypeParameterDeclaration(
        typeParam,
        typeParam.modifiers,
        typeParam.name,
        constraint,
        defaultType,
      );
    });
  }

  private resolveTypeReference(
    type: ts.TypeReferenceNode,
  ): Reference | ts.TypeReferenceNode | null {
    const target = ts.isIdentifier(type.typeName) ? type.typeName : type.typeName.right;
    const declaration = this.reflector.getDeclarationOfIdentifier(target);

    // If no declaration could be resolved or does not have a `ts.Declaration`, the type cannot be
    // resolved.
    if (declaration === null || declaration.node === null) {
      return null;
    }

    // If the declaration corresponds with a local type parameter, the type reference can be used
    // as is.
    if (this.isLocalTypeParameter(declaration.node)) {
      return type;
    }

    let owningModule: OwningModule | null = null;
    if (typeof declaration.viaModule === 'string') {
      owningModule = {
        specifier: declaration.viaModule,
        resolutionContext: type.getSourceFile().fileName,
      };
    }

    return new Reference(
      declaration.node,
      declaration.viaModule === AmbientImport ? AmbientImport : owningModule,
    );
  }

  private translateTypeReference(
    type: ts.TypeReferenceNode,
    emitReference: (ref: Reference) => ts.TypeNode | null,
  ): ts.TypeReferenceNode | null {
    const reference = this.resolveTypeReference(type);
    if (!(reference instanceof Reference)) {
      return reference;
    }

    const typeNode = emitReference(reference);
    if (typeNode === null) {
      return null;
    }

    if (!ts.isTypeReferenceNode(typeNode)) {
      throw new Error(
        `Expected TypeReferenceNode for emitted reference, got ${ts.SyntaxKind[typeNode.kind]}.`,
      );
    }
    return typeNode;
  }

  private isLocalTypeParameter(decl: DeclarationNode): boolean {
    // Checking for local type parameters only occurs during resolution of type parameters, so it is
    // guaranteed that type parameters are present.
    return this.typeParameters!.some((param) => param === decl);
  }
}

function call<OpT extends ir.CreateOp | ir.UpdateOp>(
  instruction: o.ExternalReference,
  args: o.Expression[],
  sourceSpan: ParseSourceSpan | null,
): OpT {
  const expr = o.importExpr(instruction).callFn(args, sourceSpan);
  return ir.createStatementOp(new o.ExpressionStatement(expr, sourceSpan)) as OpT;
}


/**
 * Describes a specific flavor of instruction used to represent variadic instructions, which
 * have some number of variants for specific argument counts.
 */
interface VariadicInstructionConfig {
  constant: o.ExternalReference[];
  variable: o.ExternalReference | null;
  mapping: (argCount: number) => number;
}

/**
 * `InterpolationConfig` for the `textInterpolate` instruction.
 */
const TEXT_INTERPOLATE_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.textInterpolate,
    Identifiers.textInterpolate1,
    Identifiers.textInterpolate2,
    Identifiers.textInterpolate3,
    Identifiers.textInterpolate4,
    Identifiers.textInterpolate5,
    Identifiers.textInterpolate6,
    Identifiers.textInterpolate7,
    Identifiers.textInterpolate8,
  ],
  variable: Identifiers.textInterpolateV,
  mapping: (n) => {
    if (n % 2 === 0) {
      throw new Error(`Expected odd number of arguments`);
    }
    return (n - 1) / 2;
  },
};

/**
 * `InterpolationConfig` for the `propertyInterpolate` instruction.
 */
const PROPERTY_INTERPOLATE_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.propertyInterpolate,
    Identifiers.propertyInterpolate1,
    Identifiers.propertyInterpolate2,
    Identifiers.propertyInterpolate3,
    Identifiers.propertyInterpolate4,
    Identifiers.propertyInterpolate5,
    Identifiers.propertyInterpolate6,
    Identifiers.propertyInterpolate7,
    Identifiers.propertyInterpolate8,
  ],
  variable: Identifiers.propertyInterpolateV,
  mapping: (n) => {
    if (n % 2 === 0) {
      throw new Error(`Expected odd number of arguments`);
    }
    return (n - 1) / 2;
  },
};

/**
 * `InterpolationConfig` for the `stylePropInterpolate` instruction.
 */
const STYLE_PROP_INTERPOLATE_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.styleProp,
    Identifiers.stylePropInterpolate1,
    Identifiers.stylePropInterpolate2,
    Identifiers.stylePropInterpolate3,
    Identifiers.stylePropInterpolate4,
    Identifiers.stylePropInterpolate5,
    Identifiers.stylePropInterpolate6,
    Identifiers.stylePropInterpolate7,
    Identifiers.stylePropInterpolate8,
  ],
  variable: Identifiers.stylePropInterpolateV,
  mapping: (n) => {
    if (n % 2 === 0) {
      throw new Error(`Expected odd number of arguments`);
    }
    return (n - 1) / 2;
  },
};

/**
 * `InterpolationConfig` for the `attributeInterpolate` instruction.
 */
const ATTRIBUTE_INTERPOLATE_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.attribute,
    Identifiers.attributeInterpolate1,
    Identifiers.attributeInterpolate2,
    Identifiers.attributeInterpolate3,
    Identifiers.attributeInterpolate4,
    Identifiers.attributeInterpolate5,
    Identifiers.attributeInterpolate6,
    Identifiers.attributeInterpolate7,
    Identifiers.attributeInterpolate8,
  ],
  variable: Identifiers.attributeInterpolateV,
  mapping: (n) => {
    if (n % 2 === 0) {
      throw new Error(`Expected odd number of arguments`);
    }
    return (n - 1) / 2;
  },
};

/**
 * `InterpolationConfig` for the `styleMapInterpolate` instruction.
 */
const STYLE_MAP_INTERPOLATE_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.styleMap,
    Identifiers.styleMapInterpolate1,
    Identifiers.styleMapInterpolate2,
    Identifiers.styleMapInterpolate3,
    Identifiers.styleMapInterpolate4,
    Identifiers.styleMapInterpolate5,
    Identifiers.styleMapInterpolate6,
    Identifiers.styleMapInterpolate7,
    Identifiers.styleMapInterpolate8,
  ],
  variable: Identifiers.styleMapInterpolateV,
  mapping: (n) => {
    if (n % 2 === 0) {
      throw new Error(`Expected odd number of arguments`);
    }
    return (n - 1) / 2;
  },
};

/**
 * `InterpolationConfig` for the `classMapInterpolate` instruction.
 */
const CLASS_MAP_INTERPOLATE_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.classMap,
    Identifiers.classMapInterpolate1,
    Identifiers.classMapInterpolate2,
    Identifiers.classMapInterpolate3,
    Identifiers.classMapInterpolate4,
    Identifiers.classMapInterpolate5,
    Identifiers.classMapInterpolate6,
    Identifiers.classMapInterpolate7,
    Identifiers.classMapInterpolate8,
  ],
  variable: Identifiers.classMapInterpolateV,
  mapping: (n) => {
    if (n % 2 === 0) {
      throw new Error(`Expected odd number of arguments`);
    }
    return (n - 1) / 2;
  },
};

const PURE_FUNCTION_CONFIG: VariadicInstructionConfig = {
  constant: [
    Identifiers.pureFunction0,
    Identifiers.pureFunction1,
    Identifiers.pureFunction2,
    Identifiers.pureFunction3,
    Identifiers.pureFunction4,
    Identifiers.pureFunction5,
    Identifiers.pureFunction6,
    Identifiers.pureFunction7,
    Identifiers.pureFunction8,
  ],
  variable: Identifiers.pureFunctionV,
  mapping: (n) => n,
};

function callVariadicInstructionExpr(
  config: VariadicInstructionConfig,
  baseArgs: o.Expression[],
  interpolationArgs: o.Expression[],
  extraArgs: o.Expression[],
  sourceSpan: ParseSourceSpan | null,
): o.Expression {
  const n = config.mapping(interpolationArgs.length);
  if (n < config.constant.length) {
    // Constant calling pattern.
    return o
      .importExpr(config.constant[n])
      .callFn([...baseArgs, ...interpolationArgs, ...extraArgs], sourceSpan);
  } else if (config.variable !== null) {
    // Variable calling pattern.
    return o
      .importExpr(config.variable)
      .callFn([...baseArgs, o.literalArr(interpolationArgs), ...extraArgs], sourceSpan);
  } else {
    throw new Error(`AssertionError: unable to call variadic function`);
  }
}

function callVariadicInstruction(
  config: VariadicInstructionConfig,
  baseArgs: o.Expression[],
  interpolationArgs: o.Expression[],
  extraArgs: o.Expression[],
  sourceSpan: ParseSourceSpan | null,
): ir.UpdateOp {
  return ir.createStatementOp(
    callVariadicInstructionExpr(
      config,
      baseArgs,
      interpolationArgs,
      extraArgs,
      sourceSpan,
    ).toStmt(),
  );
}
