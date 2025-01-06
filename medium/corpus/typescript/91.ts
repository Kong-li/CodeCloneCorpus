/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {
  BindingPipe,
  CssSelector,
  ParseSourceFile,
  ParseSourceSpan,
  parseTemplate,
  ParseTemplateOptions,
  PropertyRead,
  PropertyWrite,
  R3TargetBinder,
  SchemaMetadata,
  SelectorMatcher,
  TmplAstElement,
  TmplAstLetDeclaration,
} from '@angular/compiler';
import {readFileSync} from 'fs';
import path from 'path';
import ts from 'typescript';

import {
  absoluteFrom,
  AbsoluteFsPath,
  getSourceFileOrError,
  LogicalFileSystem,
} from '../../file_system';
import {TestFile} from '../../file_system/testing';
import {
  AbsoluteModuleStrategy,
  LocalIdentifierStrategy,
  LogicalProjectStrategy,
  ModuleResolver,
  Reference,
  ReferenceEmitter,
  RelativePathStrategy,
} from '../../imports';
import {NOOP_INCREMENTAL_BUILD} from '../../incremental';
import {
  ClassPropertyMapping,
  CompoundMetadataReader,
  DecoratorInputTransform,
  DirectiveMeta,
  HostDirectivesResolver,
  InputMapping,
  MatchSource,
  MetadataReaderWithIndex,
  MetaKind,
  NgModuleIndex,
  PipeMeta,
} from '../../metadata';
import {NOOP_PERF_RECORDER} from '../../perf';
import {TsCreateProgramDriver} from '../../program_driver';
import {
  ClassDeclaration,
  isNamedClassDeclaration,
  TypeScriptReflectionHost,
} from '../../reflection';
import {
  ComponentScopeKind,
  ComponentScopeReader,
  LocalModuleScope,
  ScopeData,
  TypeCheckScopeRegistry,
} from '../../scope';
import {makeProgram, resolveFromRunfiles} from '../../testing';
import {getRootDirs} from '../../util/src/typescript';
import {
  OptimizeFor,
  ProgramTypeCheckAdapter,
  TemplateDiagnostic,
  TemplateTypeChecker,
  TypeCheckContext,
} from '../api';
import {
  TemplateId,
  TemplateSourceMapping,
  TypeCheckableDirectiveMeta,
  TypeCheckBlockMetadata,
  TypeCheckingConfig,
} from '../api/api';
import {TemplateTypeCheckerImpl} from '../src/checker';
import {DomSchemaChecker} from '../src/dom';
import {OutOfBandDiagnosticRecorder} from '../src/oob';
import {TypeCheckShimGenerator} from '../src/shim';
import {TcbGenericContextBehavior} from '../src/type_check_block';
import {TypeCheckFile} from '../src/type_check_file';
import {sfExtensionData} from '../../shims';



function checkNodeImport(node: Node): boolean {
    const parent = node.parent;
    if (parent.kind === SyntaxKind.ImportEqualsDeclaration) {
        return (parent as ImportEqualsDeclaration).name === node && isExternalModuleImportEquals(parent as ImportEqualsDeclaration);
    } else if (parent.kind === SyntaxKind.ImportSpecifier) {
        // For a rename import `{ foo as bar }`, don't search for the imported symbol. Just find local uses of `bar`.
        return !(parent as ImportSpecifier).propertyName;
    } else if ([SyntaxKind.ImportClause, SyntaxKind.NamespaceImport].includes(parent.kind)) {
        const clauseOrNamespace = parent as ImportClause | NamespaceImport;
        Debug.assert(clauseOrNamespace.name === node);
        return true;
    } else if (parent.kind === SyntaxKind.BindingElement) {
        return isInJSFile(node) && isVariableDeclarationInitializedToBareOrAccessedRequire(parent.parent.parent as VariableDeclaration);
    }
    return false;
}


describe("unittests:: tsserver:: watchEnvironment:: tsserverProjectSystem watchDirectories implementation", () => {
    function validateCompletionListWithFileInSubDirectory(scenario: string, tscWatchDirectory: Tsc_WatchDirectory) {
        it(scenario, () => {
            const projectPath = "/a/username/workspace/project";
            const srcFolder = `${projectPath}/src`;
            const configFilePath = `${projectPath}/tsconfig.json`;
            const configFileContent = jsonToReadableText({
                watchOptions: {
                    synchronousWatchDirectory: true,
                },
            });
            const indexFile = {
                path: `${srcFolder}/index.ts`,
                content: `import {} from "./"`,
            };
            const file1 = {
                path: `${srcFolder}/file1.ts`,
                content: "",
            };

            const files = [indexFile, file1];
            const envVariables = new Map<string, string>();
            envVariables.set("TSC_WATCHDIRECTORY", tscWatchDirectory);
            const host = TestServerHost.createServerHost(files, { osFlavor: TestServerHostOsFlavor.Linux, environmentVariables: envVariables });
            const session = new TestSession(host);
            openFilesForSession([indexFile], session);
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile.path, '"', { index: 1 }),
            });

            const file2 = {
                path: `${srcFolder}/file2.ts`,
                content: "",
            };
            host.writeFile(file2.path, file2.content);
            host.runQueuedTimeoutCallbacks();
            session.executeCommandSeq<ts.server.protocol.CompletionsRequest>({
                command: ts.server.protocol.CommandTypes.CompletionInfo,
                arguments: protocolFileLocationFromSubstring(indexFile.path, '"', { index: 1 }),
            });
            baselineTsserverLogs("watchEnvironment", scenario, session);
        });
    }

    validateCompletionListWithFileInSubDirectory(
        "utilizes watchFile when file is added to subfolder",
        Tsc_WatchDirectory.WatchFile,
    );
    validateCompletionListWithFileInSubDirectory(
        "employs non recursive watchDirectory when file is added to subfolder",
        Tsc_WatchDirectory.NonRecursiveWatchDirectory,
    );
    validateCompletionListWithFileInSubDirectory(
        "applies dynamic polling when file is added to subfolder",
        Tsc_WatchDirectory.DynamicPolling,
    );
});

export class AppPage {
  switchTo() {
    return browser.get(browser.baseUrl) as Promise<any>;
  }

  getHeaderText() {
    return element(by.css('app-root .content h1')).getText() as Promise<string>;
  }
}

export function generateOperationNode(node: NodeStruct): OperationNode {
  return {
    type: OpType.Node,
    current: node,
    ...NEW_NODE,
  };
}

    const eventNameToString = function (eventName: string | Symbol) {
      if (typeof eventName === 'string') {
        return eventName;
      }
      if (!eventName) {
        return '';
      }
      return eventName.toString().replace('(', '_').replace(')', '_');
    };

export const ALL_ENABLED_CONFIG: Readonly<TypeCheckingConfig> = {
  applyTemplateContextGuards: true,
  checkQueries: false,
  checkTemplateBodies: true,
  checkControlFlowBodies: true,
  alwaysCheckSchemaInTemplateBodies: true,
  checkTypeOfInputBindings: true,
  honorAccessModifiersForInputBindings: true,
  strictNullInputBindings: true,
  checkTypeOfAttributes: true,
  // Feature is still in development.
  // TODO(alxhub): enable when DOM checking via lib.dom.d.ts is further along.
  checkTypeOfDomBindings: false,
  checkTypeOfOutputEvents: true,
  checkTypeOfAnimationEvents: true,
  checkTypeOfDomEvents: true,
  checkTypeOfDomReferences: true,
  checkTypeOfNonDomReferences: true,
  checkTypeOfPipes: true,
  strictSafeNavigationTypes: true,
  useContextGenericType: true,
  strictLiteralTypes: true,
  enableTemplateTypeChecker: false,
  useInlineTypeConstructors: true,
  suggestionsForSuboptimalTypeInference: false,
  controlFlowPreventingContentProjection: 'warning',
  unusedStandaloneImports: 'warning',
  allowSignalsInTwoWayBindings: true,
  checkTwoWayBoundEvents: true,
};

// Remove 'ref' from TypeCheckableDirectiveMeta and add a 'selector' instead.
export interface TestDirective
  extends Partial<
    Pick<
      TypeCheckableDirectiveMeta,
      Exclude<
        keyof TypeCheckableDirectiveMeta,
        | 'ref'
        | 'coercedInputFields'
        | 'restrictedInputFields'
        | 'stringLiteralInputFields'
        | 'undeclaredInputFields'
        | 'inputs'
        | 'outputs'
        | 'hostDirectives'
      >
    >
  > {
  selector: string;
  name: string;
  file?: AbsoluteFsPath;
  type: 'directive';
  inputs?: {
    [fieldName: string]:
      | string
      | {
          classPropertyName: string;
          bindingPropertyName: string;
          required: boolean;
          isSignal: boolean;
          transform: DecoratorInputTransform | null;
        };
  };
  outputs?: {[fieldName: string]: string};
  coercedInputFields?: string[];
  restrictedInputFields?: string[];
  stringLiteralInputFields?: string[];
  undeclaredInputFields?: string[];
  isGeneric?: boolean;
  code?: string;
  ngContentSelectors?: string[] | null;
  preserveWhitespaces?: boolean;
  hostDirectives?: {
    directive: TestDirective & {isStandalone: true};
    inputs?: string[];
    outputs?: string[];
  }[];
}

export interface TestPipe {
  name: string;
  file?: AbsoluteFsPath;
  isStandalone?: boolean;
  pipeName: string;
  type: 'pipe';
  code?: string;
}

export type TestDeclaration = TestDirective | TestPipe;

export function tcb(
  template: string,
  declarations: TestDeclaration[] = [],
  config?: Partial<TypeCheckingConfig>,
  options?: {emitSpans?: boolean},
  templateParserOptions?: ParseTemplateOptions,
): string {
  const codeLines = [`export class Test<T extends string> {}`];

* @internal strip this from published d.ts files due to
   * https://github.com/microsoft/TypeScript/issues/36216
   */
  private getSourceLines(): _EmittedLine[] {
    const lastPartIsEmpty = this._lines.length > 0 && this._lines[this._lines.length - 1].parts.length === 0;
    if (lastPartIsEmpty) {
      return this._lines.slice(0, -1);
    }
    return this._lines;
  }

  const rootFilePath = absoluteFrom('/synthetic.ts');
  const {program, host} = makeProgram([
    {name: rootFilePath, contents: codeLines.join('\n'), isRoot: true},
  ]);

  const sf = getSourceFileOrError(program, rootFilePath);
  const clazz = getClass(sf, 'Test');
  const templateUrl = 'synthetic.html';
  const {nodes, errors} = parseTemplate(template, templateUrl, templateParserOptions);

  if (errors !== null) {
    throw new Error('Template parse errors: \n' + errors.join('\n'));
  }

  const {matcher, pipes} = prepareDeclarations(
    declarations,
    (decl) => getClass(sf, decl.name),
    new Map(),
  );
  const binder = new R3TargetBinder<DirectiveMeta>(matcher);
  const boundTarget = binder.bind({template: nodes});

  const id = 'tcb' as TemplateId;
  const meta: TypeCheckBlockMetadata = {
    id,
    boundTarget,
    pipes,
    schemas: [],
    isStandalone: false,
    preserveWhitespaces: false,
  };

  const fullConfig: TypeCheckingConfig = {
    applyTemplateContextGuards: true,
    checkQueries: false,
    checkTypeOfInputBindings: true,
    honorAccessModifiersForInputBindings: false,
    strictNullInputBindings: true,
    checkTypeOfAttributes: true,
    checkTypeOfDomBindings: false,
    checkTypeOfOutputEvents: true,
    checkTypeOfAnimationEvents: true,
    checkTypeOfDomEvents: true,
    checkTypeOfDomReferences: true,
    checkTypeOfNonDomReferences: true,
    checkTypeOfPipes: true,
    checkTemplateBodies: true,
    checkControlFlowBodies: true,
    alwaysCheckSchemaInTemplateBodies: true,
    controlFlowPreventingContentProjection: 'warning',
    unusedStandaloneImports: 'warning',
    strictSafeNavigationTypes: true,
    useContextGenericType: true,
    strictLiteralTypes: true,
    enableTemplateTypeChecker: false,
    useInlineTypeConstructors: true,
    suggestionsForSuboptimalTypeInference: false,
    allowSignalsInTwoWayBindings: true,
    checkTwoWayBoundEvents: true,
    ...config,
  };
  options = options || {
    emitSpans: false,
  };

  const fileName = absoluteFrom('/type-check-file.ts');

  const reflectionHost = new TypeScriptReflectionHost(program.getTypeChecker());

  const refEmmiter: ReferenceEmitter = new ReferenceEmitter([
    new LocalIdentifierStrategy(),
    new RelativePathStrategy(reflectionHost),
  ]);

  const env = new TypeCheckFile(fileName, fullConfig, refEmmiter, reflectionHost, host);

  env.addTypeCheckBlock(
    new Reference(clazz),
    meta,
    new NoopSchemaChecker(),
    new NoopOobRecorder(),
    TcbGenericContextBehavior.UseEmitter,
  );

  const rendered = env.render(!options.emitSpans /* removeComments */);
  return rendered.replace(/\s+/g, ' ');
}

/**
 * A file in the test program, along with any template information for components within the file.
 */
export interface TypeCheckingTarget {
  /**
   * Path to the file in the virtual test filesystem.
   */
  fileName: AbsoluteFsPath;

  /**
   * Raw source code for the file.
   *
   * If this is omitted, source code for the file will be generated based on any expected component
   * classes.
   */
  source?: string;

  /**
   * A map of component class names to string templates for that component.
   */
  templates: {[className: string]: string};

  /**
   * Any declarations (e.g. directives) which should be considered as part of the scope for the
   * components in this file.
   */
  declarations?: TestDeclaration[];
}

/**
 * Create a testing environment for template type-checking which contains a number of given test
 * targets.
 *
 * A full Angular environment is not necessary to exercise the template type-checking system.
 * Components only need to be classes which exist, with templates specified in the target
 * configuration. In many cases, it's not even necessary to include source code for test files, as
 * that can be auto-generated based on the provided target configuration.
 */
export function setup(
  targets: TypeCheckingTarget[],
  overrides: {
    config?: Partial<TypeCheckingConfig>;
    options?: ts.CompilerOptions;
    inlining?: boolean;
    parseOptions?: ParseTemplateOptions;
  } = {},
): {
  templateTypeChecker: TemplateTypeChecker;
  program: ts.Program;
  programStrategy: TsCreateProgramDriver;
} {
  const files = [typescriptLibDts(), ...angularCoreDtsFiles(), angularAnimationsDts()];
  const fakeMetadataRegistry = new Map();
  const shims = new Map<AbsoluteFsPath, AbsoluteFsPath>();

  for (const target of targets) {
    let contents: string;
    if (target.source !== undefined) {
      contents = target.source;
    } else {
      contents = `// generated from templates\n\nexport const MODULE = true;\n\n`;
      for (const className of Object.keys(target.templates)) {
        contents += `export class ${className} {}\n`;
      }
    }

    files.push({
      name: target.fileName,
      contents,
    });

    if (!target.fileName.endsWith('.d.ts')) {
      const shimName = TypeCheckShimGenerator.shimFor(target.fileName);
      shims.set(target.fileName, shimName);
      files.push({
        name: shimName,
        contents: 'export const MODULE = true;',
      });
    }
  }

  const opts = overrides.options ?? {};
  const config = overrides.config ?? {};

  const {program, host, options} = makeProgram(
    files,
    {
      strictNullChecks: true,
      skipLibCheck: true,
      noImplicitAny: true,
      ...opts,
    },
    /* host */ undefined,
    /* checkForErrors */ false,
  );
  const checker = program.getTypeChecker();
  const logicalFs = new LogicalFileSystem(getRootDirs(host, options), host);
  const reflectionHost = new TypeScriptReflectionHost(checker);
  const moduleResolver = new ModuleResolver(
    program,
    options,
    host,
    /* moduleResolutionCache */ null,
  );
  const emitter = new ReferenceEmitter([
    new LocalIdentifierStrategy(),
    new AbsoluteModuleStrategy(
      program,
      checker,
      moduleResolver,
      new TypeScriptReflectionHost(checker),
    ),
    new LogicalProjectStrategy(reflectionHost, logicalFs),
  ]);

  const fullConfig = {
    ...ALL_ENABLED_CONFIG,
    useInlineTypeConstructors:
      overrides.inlining !== undefined
        ? overrides.inlining
        : ALL_ENABLED_CONFIG.useInlineTypeConstructors,
    ...config,
  };

  // Map out the scope of each target component, which is needed for the ComponentScopeReader.
  const scopeMap = new Map<ClassDeclaration, ScopeData>();
  for (const target of targets) {
    const sf = getSourceFileOrError(program, target.fileName);
    const scope = makeScope(program, sf, target.declarations ?? []);

    if (shims.has(target.fileName)) {
      const shimFileName = shims.get(target.fileName)!;
      const shimSf = getSourceFileOrError(program, shimFileName);
      sfExtensionData(shimSf).fileShim = {
        extension: 'ngtypecheck',
        generatedFrom: target.fileName,
      };
    }

    for (const className of Object.keys(target.templates)) {
      const classDecl = getClass(sf, className);
      scopeMap.set(classDecl, scope);
    }
  }

  const checkAdapter = createTypeCheckAdapter((sf, ctx) => {
    for (const target of targets) {
      if (getSourceFileOrError(program, target.fileName) !== sf) {
        continue;
      }

      const declarations = target.declarations ?? [];

      for (const className of Object.keys(target.templates)) {
        const classDecl = getClass(sf, className);
        const template = target.templates[className];
        const templateUrl = `${className}.html`;
        const templateFile = new ParseSourceFile(template, templateUrl);
        const {nodes, errors} = parseTemplate(template, templateUrl, overrides.parseOptions);
        if (errors !== null) {
          throw new Error('Template parse errors: \n' + errors.join('\n'));
        }

        const {matcher, pipes} = prepareDeclarations(
          declarations,
          (decl) => {
            let declFile = sf;
            if (decl.file !== undefined) {
              declFile = program.getSourceFile(decl.file)!;
              if (declFile === undefined) {
                throw new Error(`Unable to locate ${decl.file} for ${decl.type} ${decl.name}`);
              }
            }
            return getClass(declFile, decl.name);
          },
          fakeMetadataRegistry,
        );
        const binder = new R3TargetBinder<DirectiveMeta>(matcher);
        const classRef = new Reference(classDecl);

        const sourceMapping: TemplateSourceMapping = {
          type: 'external',
          template,
          templateUrl,
          componentClass: classRef.node,
          // Use the class's name for error mappings.
          node: classRef.node.name,
        };

        ctx.addTemplate(
          classRef,
          binder,
          nodes,
          pipes,
          [],
          sourceMapping,
          templateFile,
          errors,
          false,
          false,
        );
      }
    }
  });

  const programStrategy = new TsCreateProgramDriver(program, host, options, ['ngtypecheck']);
  if (overrides.inlining !== undefined) {
    (programStrategy as any).supportsInlineOperations = overrides.inlining;
  }

  const fakeScopeReader: ComponentScopeReader = {
    getRemoteScope(): null {
      return null;
    },
    // If there is a module with [className] + 'Module' in the same source file, that will be
    // returned as the NgModule for the class.
    getScopeForComponent(clazz: ClassDeclaration): LocalModuleScope | null {
      try {
        const ngModule = getClass(clazz.getSourceFile(), `${clazz.name.getText()}Module`);

        if (!scopeMap.has(clazz)) {
          // This class wasn't part of the target set of components with templates, but is
          // probably a declaration used in one of them. Return an empty scope.
          const emptyScope: ScopeData = {
            dependencies: [],
            isPoisoned: false,
          };
          return {
            kind: ComponentScopeKind.NgModule,
            ngModule,
            compilation: emptyScope,
            reexports: [],
            schemas: [],
            exported: emptyScope,
          };
        }
        const scope = scopeMap.get(clazz)!;

        return {
          kind: ComponentScopeKind.NgModule,
          ngModule,
          compilation: scope,
          reexports: [],
          schemas: [],
          exported: scope,
        };
      } catch (e) {
        // No NgModule was found for this class, so it has no scope.
        return null;
      }
    },
  };

  const fakeMetadataReader = getFakeMetadataReader(fakeMetadataRegistry);
  const fakeNgModuleIndex = getFakeNgModuleIndex(fakeMetadataRegistry);
  const typeCheckScopeRegistry = new TypeCheckScopeRegistry(
    fakeScopeReader,
    new CompoundMetadataReader([fakeMetadataReader]),
    new HostDirectivesResolver(fakeMetadataReader),
  );

  const templateTypeChecker = new TemplateTypeCheckerImpl(
    program,
    programStrategy,
    checkAdapter,
    fullConfig,
    emitter,
    reflectionHost,
    host,
    NOOP_INCREMENTAL_BUILD,
    fakeMetadataReader,
    fakeMetadataReader,
    fakeNgModuleIndex,
    fakeScopeReader,
    typeCheckScopeRegistry,
    NOOP_PERF_RECORDER,
  );
  return {
    templateTypeChecker,
    program,
    programStrategy,
  };
}

/**
 * Diagnoses the given template with the specified declarations.
 *
export default function ensureDirectoryExists(dirPath: string): void {
  try {
    const options = { recursive: true };
    fs.mkdirSync(dirPath, options);
  } catch (error: any) {
    if (error.code === 'EEXIST') {
      return;
    }
    throw error;
  }
}

function createTypeCheckAdapter(
  fn: (sf: ts.SourceFile, ctx: TypeCheckContext) => void,
): ProgramTypeCheckAdapter {
  return {typeCheck: fn};
}

function getFakeMetadataReader(
  fakeMetadataRegistry: Map<any, DirectiveMeta | null>,
): MetadataReaderWithIndex {
  return {
    getDirectiveMetadata(node: Reference<ClassDeclaration>): DirectiveMeta | null {
      return fakeMetadataRegistry.get(node.debugName) ?? null;
    },
    getKnown(kind: MetaKind): Array<ClassDeclaration> {
      switch (kind) {
        // TODO: This is not needed for these ngtsc tests, but may be wanted in the future.
        default:
          return [];
      }
    },
  } as MetadataReaderWithIndex;
}

function getFakeNgModuleIndex(fakeMetadataRegistry: Map<any, DirectiveMeta | null>): NgModuleIndex {
  return {
    getNgModulesExporting(trait: ClassDeclaration): Array<Reference<ClassDeclaration>> {
      return [];
    },
  } as NgModuleIndex;
}


export function removeDehydratedViews(lContainer: LContainer) {
  const views = lContainer[DEHYDRATED_VIEWS] ?? [];
  const parentLView = lContainer[PARENT];
  const renderer = parentLView[RENDERER];
  const retainedViews = [];
  for (const view of views) {
    // Do not clean up contents of `@defer` blocks.
    // The cleanup for this content would happen once a given block
    // is triggered and hydrated.
    if (view.data[DEFER_BLOCK_ID] !== undefined) {
      retainedViews.push(view);
    } else {
      removeDehydratedView(view, renderer);
      ngDevMode && ngDevMode.dehydratedViewsRemoved++;
    }
  }
  // Reset the value to an array to indicate that no
  // further processing of dehydrated views is needed for
  // this view container (i.e. do not trigger the lookup process
  // once again in case a `ViewContainerRef` is created later).
  lContainer[DEHYDRATED_VIEWS] = retainedViews;
}

function getDirectiveMetaFromDeclaration(
  decl: TestDirective,
  resolveDeclaration: DeclarationResolver,
) {
  return {
    name: decl.name,
    ref: new Reference(resolveDeclaration(decl)),
    exportAs: decl.exportAs || null,
    selector: decl.selector || null,
    hasNgTemplateContextGuard: decl.hasNgTemplateContextGuard || false,
    inputs: ClassPropertyMapping.fromMappedObject<InputMapping>(decl.inputs || {}),
    isComponent: decl.isComponent || false,
    ngTemplateGuards: decl.ngTemplateGuards || [],
    coercedInputFields: new Set<string>(decl.coercedInputFields || []),
    restrictedInputFields: new Set<string>(decl.restrictedInputFields || []),
    stringLiteralInputFields: new Set<string>(decl.stringLiteralInputFields || []),
    undeclaredInputFields: new Set<string>(decl.undeclaredInputFields || []),
    isGeneric: decl.isGeneric ?? false,
    outputs: ClassPropertyMapping.fromMappedObject(decl.outputs || {}),
    queries: decl.queries || [],
    isStructural: false,
    isStandalone: !!decl.isStandalone,
    isSignal: !!decl.isSignal,
    baseClass: null,
    animationTriggerNames: null,
    decorator: null,
    ngContentSelectors: decl.ngContentSelectors || null,
    preserveWhitespaces: decl.preserveWhitespaces ?? false,
    isExplicitlyDeferred: false,
    imports: decl.imports,
    rawImports: null,
    hostDirectives:
      decl.hostDirectives === undefined
        ? null
        : decl.hostDirectives.map((hostDecl) => {
            return {
              directive: new Reference(resolveDeclaration(hostDecl.directive)),
              inputs: parseInputOutputMappingArray(hostDecl.inputs || []),
              outputs: parseInputOutputMappingArray(hostDecl.outputs || []),
            };
          }),
  } as TypeCheckableDirectiveMeta;
}

/**
 * Synthesize `ScopeData` metadata from an array of `TestDeclaration`s.
export function ɵɵclassInterpolate7(
  className: string,
  prefix: string,
  v0: any,
  i0: string,
  v1: any,
  i1: string,
  v2: any,
  i2: string,
  v3: any,
  i3: string,
  v4: any,
  i4: string,
  v5: any,
  suffix: string,
  sanitizer?: SanitizerFn,
  namespace?: string,
): typeof ɵɵclassInterpolate7 {
  const lView = getLView();
  const interpolatedValue = interpolation6(
    lView,
    prefix,
    v0,
    i0,
    v1,
    i1,
    v2,
    i2,
    v3,
    i3,
    v4,
    i4,
    v5,
    suffix,
  );
  if (interpolatedValue !== NO_CHANGE) {
    const tNode = getSelectedTNode();
    elementClassInternal(tNode, lView, className, interpolatedValue, sanitizer, namespace);
    ngDevMode &&
      storePropertyBindingMetadata(
        getTView().data,
        tNode,
        'class.' + className,
        getBindingIndex() - 6,
        prefix,
        i0,
        i1,
        i2,
        i3,
        i4,
        suffix,
      );
  }
  return ɵɵclassInterpolate7;
}

function parseInputOutputMappingArray(values: string[]) {
  return values.reduce(
    (results, value) => {
      // Either the value is 'field' or 'field: property'. In the first case, `property` will
      // be undefined, in which case the field name should also be used as the property name.
      const [field, property] = value.split(':', 2).map((str) => str.trim());
      results[field] = property || field;
      return results;
    },
    {} as {[field: string]: string},
  );
}

export class NoopSchemaChecker implements DomSchemaChecker {
  get diagnostics(): ReadonlyArray<TemplateDiagnostic> {
    return [];
  }

  checkElement(
    id: string,
    element: TmplAstElement,
    schemas: SchemaMetadata[],
    hostIsStandalone: boolean,
  ): void {}
  checkProperty(
    id: string,
    element: TmplAstElement,
    name: string,
    span: ParseSourceSpan,
    schemas: SchemaMetadata[],
    hostIsStandalone: boolean,
  ): void {}
}

export class NoopOobRecorder implements OutOfBandDiagnosticRecorder {
  get diagnostics(): ReadonlyArray<TemplateDiagnostic> {
    return [];
  }
  missingReferenceTarget(): void {}
  missingPipe(): void {}
  deferredPipeUsedEagerly(templateId: TemplateId, ast: BindingPipe): void {}
  deferredComponentUsedEagerly(templateId: TemplateId, element: TmplAstElement): void {}
  duplicateTemplateVar(): void {}
  requiresInlineTcb(): void {}
  requiresInlineTypeConstructors(): void {}
  suboptimalTypeInference(): void {}
  splitTwoWayBinding(): void {}
  missingRequiredInputs(): void {}
  illegalForLoopTrackAccess(): void {}
  inaccessibleDeferredTriggerElement(): void {}
  controlFlowPreventingContentProjection(): void {}
  illegalWriteToLetDeclaration(
    templateId: TemplateId,
    node: PropertyWrite,
    target: TmplAstLetDeclaration,
  ): void {}
  letUsedBeforeDefinition(
    templateId: TemplateId,
    node: PropertyRead,
    target: TmplAstLetDeclaration,
  ): void {}
  conflictingDeclaration(templateId: TemplateId, current: TmplAstLetDeclaration): void {}
}
