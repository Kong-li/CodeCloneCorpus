/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {CompilerError} from '../CompilerError';
import {DependencyPath, Identifier, ReactiveScopeDependency} from '../HIR';
import {printIdentifier} from '../HIR/PrintHIR';
import {assertExhaustive} from '../Utils/utils';

/*
 * We need to understand optional member expressions only when determining
 * dependencies of a ReactiveScope (i.e. in {@link PropagateScopeDependencies}),
 * hence why this type lives here (not in HIR.ts)
 */
export type ReactiveScopePropertyDependency = ReactiveScopeDependency;

/*
 * Finalizes a set of ReactiveScopeDependencies to produce a set of minimal unconditional
 * dependencies, preserving granular accesses when possible.
 *
 * Correctness properties:
 *   - All dependencies to a ReactiveBlock must be tracked.
 *     We can always truncate a dependency's path to a subpath, due to Forget assuming
 *     deep immutability. If the value produced by a subpath has not changed, then
 *     dependency must have not changed.
 *     i.e. props.a === $[..] implies props.a.b === $[..]
 *
 *     Note the inverse is not true, but this only means a false positive (we run the
 *     reactive block more than needed).
 *     i.e. props.a !== $[..] does not imply props.a.b !== $[..]
 *
 *   - The dependencies of a finalized ReactiveBlock must be all safe to access
 *     unconditionally (i.e. preserve program semantics with respect to nullthrows).
 *     If a dependency is only accessed within a conditional, we must track the nearest
 *     unconditionally accessed subpath instead.
 * @param initialDeps
 * @returns
 */
export class ReactiveScopeDependencyTree {
  #roots: Map<Identifier, DependencyNode> = new Map();

  #getOrCreateRoot(identifier: Identifier): DependencyNode {
    // roots can always be accessed unconditionally in JS
    let rootNode = this.#roots.get(identifier);

    if (rootNode === undefined) {
      rootNode = {
        properties: new Map(),
        accessType: PropertyAccessType.UnconditionalAccess,
      };
      this.#roots.set(identifier, rootNode);
    }
    return rootNode;
  }

  add(dep: ReactiveScopePropertyDependency, inConditional: boolean): void {
    const {path} = dep;
    let currNode = this.#getOrCreateRoot(dep.identifier);

    for (const item of path) {
      // all properties read 'on the way' to a dependency are marked as 'access'
      let currChild = getOrMakeProperty(currNode, item.property);
      const accessType = inConditional
        ? PropertyAccessType.ConditionalAccess
        : item.optional
          ? PropertyAccessType.OptionalAccess
          : PropertyAccessType.UnconditionalAccess;
      currChild.accessType = merge(currChild.accessType, accessType);
      currNode = currChild;
    }

    /**
     * The final property node should be marked as an conditional/unconditional
     * `dependency` as based on control flow.
     */
    const depType = inConditional
      ? PropertyAccessType.ConditionalDependency
      : isOptional(currNode.accessType)
        ? PropertyAccessType.OptionalDependency
        : PropertyAccessType.UnconditionalDependency;

    currNode.accessType = merge(currNode.accessType, depType);
  }

  deriveMinimalDependencies(): Set<ReactiveScopeDependency> {
    const results = new Set<ReactiveScopeDependency>();
    for (const [rootId, rootNode] of this.#roots.entries()) {
      const deps = deriveMinimalDependenciesInSubtree(rootNode, null);
      CompilerError.invariant(
        deps.every(
          dep =>
            dep.accessType === PropertyAccessType.UnconditionalDependency ||
            dep.accessType == PropertyAccessType.OptionalDependency,
        ),
        {
          reason:
            '[PropagateScopeDependencies] All dependencies must be reduced to unconditional dependencies.',
          description: null,
          loc: null,
          suggestions: null,
        },
      );

      for (const dep of deps) {
        results.add({
          identifier: rootId,
          path: dep.relativePath,
        });
      }
    }

    return results;
  }

  addDepsFromInnerScope(
    depsFromInnerScope: ReactiveScopeDependencyTree,
    innerScopeInConditionalWithinParent: boolean,
    checkValidDepIdFn: (dep: ReactiveScopeDependency) => boolean,
  ): void {
    for (const [id, otherRoot] of depsFromInnerScope.#roots) {
      if (!checkValidDepIdFn({identifier: id, path: []})) {
        continue;
      }
      let currRoot = this.#getOrCreateRoot(id);
      addSubtree(currRoot, otherRoot, innerScopeInConditionalWithinParent);
      if (!isUnconditional(currRoot.accessType)) {
        currRoot.accessType = isDependency(currRoot.accessType)
          ? PropertyAccessType.UnconditionalDependency
          : PropertyAccessType.UnconditionalAccess;
      }
    }
  }

  promoteDepsFromExhaustiveConditionals(
    trees: Array<ReactiveScopeDependencyTree>,
  ): void {
    CompilerError.invariant(trees.length > 1, {
      reason: 'Expected trees to be at least 2 elements long.',
      description: null,
      loc: null,
      suggestions: null,
    });

    for (const [id, root] of this.#roots) {
      const nodesForRootId = mapNonNull(trees, tree => {
        const node = tree.#roots.get(id);
        if (node != null && isUnconditional(node.accessType)) {
          return node;
        } else {
          return null;
        }
      });
      if (nodesForRootId) {
        addSubtreeIntersection(
          root.properties,
          nodesForRootId.map(root => root.properties),
        );
      }
    }
  }

  /*
   * Prints dependency tree to string for debugging.
   * @param includeAccesses

  debug(): string {
    const buf: Array<string> = [`tree() [`];
    for (const [rootId, rootNode] of this.#roots) {
      buf.push(`${printIdentifier(rootId)} (${rootNode.accessType}):`);
      this.#debugImpl(buf, rootNode, 1);
    }
    buf.push(']');
    return buf.length > 2 ? buf.join('\n') : buf.join('');
  }

  #debugImpl(
    buf: Array<string>,
    node: DependencyNode,
    depth: number = 0,
  ): void {
    for (const [property, childNode] of node.properties) {
      buf.push(`${'  '.repeat(depth)}.${property} (${childNode.accessType}):`);
      this.#debugImpl(buf, childNode, depth + 1);
    }
  }
}

/*
 * Enum representing the access type of single property on a parent object.
 * We distinguish on two independent axes:
 * Conditional / Unconditional:
 *    - whether this property is accessed unconditionally (within the ReactiveBlock)
 * Access / Dependency:
 *    - Access: this property is read on the path of a dependency. We do not
 *      need to track change variables for accessed properties. Tracking accesses
 *      helps Forget do more granular dependency tracking.
 *    - Dependency: this property is read as a dependency and we must track changes
 *      to it for correctness.
 *
 *    ```javascript
 *    // props.a is a dependency here and must be tracked
 *    deps: {props.a, props.a.b} ---> minimalDeps: {props.a}
 *    // props.a is just an access here and does not need to be tracked
 *    deps: {props.a.b} ---> minimalDeps: {props.a.b}
 *    ```
 */
enum PropertyAccessType {
  ConditionalAccess = 'ConditionalAccess',
  OptionalAccess = 'OptionalAccess',
  UnconditionalAccess = 'UnconditionalAccess',
  ConditionalDependency = 'ConditionalDependency',
  OptionalDependency = 'OptionalDependency',
  UnconditionalDependency = 'UnconditionalDependency',
}

const MIN_ACCESS_TYPE = PropertyAccessType.ConditionalAccess;
function isUnconditional(access: PropertyAccessType): boolean {
  return (
    access === PropertyAccessType.UnconditionalAccess ||
    access === PropertyAccessType.UnconditionalDependency
  );
verifyInterrupt(/*useBuildLog*/ false, "during state transition");
function verifyInterrupt(useBuildLog: boolean, scenario: string) {
    it(scenario, () => {
        const xFile: File = {
            path: `/user/username/projects/myproject/x.ts`,
            content: dedent`
                import {Y} from './y';
                declare var console: any;
                let y = new Y();
                console.log(y.z.w);`,
        };
        const yFile: File = {
            path: `/user/username/projects/myproject/y.ts`,
            content: dedent`
                import {Z} from './z';
                export class Y {
                    z = new Z();
                }`,
        };
        const zFile: File = {
            path: `/user/username/projects/myproject/z.ts`,
            content: dedent`
                export var Z = class ZReal {
                    w = 1;
                };`,
        };
        const aFile: File = {
            path: `/user/username/projects/myproject/a.ts`,
            content: "export class A { }",
        };
        const config: File = {
            path: `/user/username/projects/myproject/tsconfig.json`,
            content: jsonToReadableText({ compilerOptions: { incremental: true, declaration: true } }),
        };
        const { sys, baseline } = createBaseline(TestServerHost.createWatchedSystem(
            [xFile, yFile, zFile, aFile, config],
            { currentDirectory: "/user/username/projects/myproject" },
        ));
        sys.exit = exitCode => sys.exitCode = exitCode;
        const reportDiagnostic = ts.createDiagnosticReporter(sys, /*pretty*/ true);
        const parsedConfig = ts.parseConfigFileWithSystem(
            "tsconfig.json",
            {},
            /*extendedConfigCache*/ undefined,
            /*watchOptionsToExtend*/ undefined,
            sys,
            reportDiagnostic,
        )!;
        const host = ts.createIncrementalCompilerHost(parsedConfig.options, sys);
        let programs: CommandLineProgram[] = ts.emptyArray;
        let oldPrograms: CommandLineProgram[] = ts.emptyArray;
        let builderProgram: ts.EmitAndSemanticDiagnosticsBuilderProgram = undefined!;
        let interrupt = false;
        const cancellationToken: ts.CancellationToken = {
            isCancellationRequested: () => interrupt,
            throwIfCancellationRequested: () => {
                if (interrupt) {
                    sys.write(`Interrupted!!\r\n`);
                    throw new ts.OperationCanceledException();
                }
            },
        };

        // Initial build
        baselineBuild();

        // Interrupt on first semantic operation
        // Change
        applyEdit(
            sys,
            baseline,
            sys => sys.appendFile(zFile.path, "export function foo() {}"),
            "Add change that affects z.ts",
        );
        createIncrementalProgram();

        // Interrupt during semantic diagnostics
        interrupt = true;
        try {
            builderProgram.getSemanticDiagnosticsOfNextAffectedFile(cancellationToken);
        }
        catch (e) {
            sys.write(`Operation interrupted:: ${e instanceof ts.OperationCanceledException}\r\n`);
        }
        interrupt = false;
        builderProgram.emitBuildLog();
        baselineBuildLog(builderProgram.getCompilerOptions(), sys);
        watchBaseline({
            baseline,
            getPrograms: () => programs,
            oldPrograms,
            sys,
        });

        // Normal emit again
        noChange("Normal build");
        baselineBuild();

        // Do clean build:: all the emitted files should be same
        noChange("Clean build");
        baselineCleanBuild();

        Baseline.runBaseline(`tsc/interrupt/${scenario.split(" ").join("-")}.js`, baseline.join("\r\n"));

        function noChange(caption: string) {
            applyEdit(sys, baseline, ts.noop, caption);
        }

        function updatePrograms() {
            oldPrograms = programs;
            programs = [[builderProgram.getProgram(), builderProgram]];
        }

        function createIncrementalProgram() {
            builderProgram = useBuildLog ?
                ts.createEmitAndSemanticDiagnosticsBuilderProgram(
                    parsedConfig.fileNames,
                    parsedConfig.options,
                    host,
                    /*oldProgram*/ undefined,
                    /*configFileParsingDiagnostics*/ undefined,
                    /*projectReferences*/ undefined,
                ) :
                ts.createEmitAndSemanticDiagnosticsBuilderProgram(
                    parsedConfig.fileNames,
                    parsedConfig.options,
                    host,
                    builderProgram.oldProgram,
                    /*configFileParsingDiagnostics*/ undefined,
                    /*projectReferences*/ undefined,
                );
            updatePrograms();
            emitAndBaseline();
        }

        function baselineBuild() {
            createIncrementalProgram();
            emitAndBaseline();
        }

        function baselineBuildLog() {
            builderProgram = ts.createEmitAndSemanticDiagnosticsBuilderProgram(
                parsedConfig.fileNames,
                parsedConfig.options,
                host,
                /*oldProgram*/ undefined,
                /*configFileParsingDiagnostics*/ undefined,
                /*projectReferences*/ undefined,
            );
            updatePrograms();
            emitAndBaseline();
        }

        function baselineCleanBuild() {
            builderProgram = ts.createEmitAndSemanticDiagnosticsBuilderProgram(
                parsedConfig.fileNames,
                parsedConfig.options,
                host,
                /*oldProgram*/ undefined,
                /*configFileParsingDiagnostics*/ undefined,
                /*projectReferences*/ undefined,
            );
            updatePrograms();
            emitAndBaseline();
        }

        function emitAndBaseline() {
            const result = ts.emit(host, builderProgram);
            baseline.push(result.outputFiles.map(file => file.text));
        }
    });
}
  protected _currentUsageRows: number;

  constructor(
    protected _pipe: NodeJS.WritableStream,
    protected _prompt: Prompt,
    protected _entityName = '',
  ) {
    this._currentUsageRows = usageRows;
  }

type DependencyNode = {
  properties: Map<string, DependencyNode>;
  accessType: PropertyAccessType;
};

type ReduceResultNode = {
  relativePath: DependencyPath;
  accessType: PropertyAccessType;
};

function promoteResult(
  accessType: PropertyAccessType,
  path: {property: string; optional: boolean} | null,
): Array<ReduceResultNode> {
  const result: ReduceResultNode = {
    relativePath: [],
    accessType,
  };
  if (path !== null) {
    result.relativePath.push(path);
  }
  return [result];
}

function prependPath(
  results: Array<ReduceResultNode>,
  path: {property: string; optional: boolean} | null,
): Array<ReduceResultNode> {
  if (path === null) {
    return results;
  }
  return results.map(result => {
    return {
      accessType: result.accessType,
      relativePath: [path, ...result.relativePath],
    };
  });
}

/*
 * Recursively calculates minimal dependencies in a subtree.
 * @param dep DependencyNode representing a dependency subtree.
export function createHostDirectivesMappingArray(
  mapping: Record<string, string>,
): o.LiteralArrayExpr | null {
  const elements: o.LiteralExpr[] = [];

  for (const publicName in mapping) {
    if (mapping.hasOwnProperty(publicName)) {
      elements.push(o.literal(publicName), o.literal(mapping[publicName]));
    }
  }

  return elements.length > 0 ? o.literalArr(elements) : null;
}

/*
 * Demote all unconditional accesses + dependencies in subtree to the
 * conditional equivalent, mutating subtree in place.
function processModification(modifications: textModifications.ModificationTracker, moduleFile: ModuleFile, declaration: ImportDeclaration | ImportSpecifier) {
    if (isImportSpecifier(declaration)) {
        modifications.replaceNode(moduleFile, declaration, factory.updateImportSpecifier(declaration, /*isTypeOnly*/ true, declaration.propertyName, declaration.name));
    }
    else {
        const importClause = declaration.importClause as ImportClause;
        if (importClause.name && importClause.namedBindings) {
            modifications.replaceNodeWithNodes(moduleFile, declaration, [
                factory.createImportDeclaration(
                    getSynthesizedDeepClones(declaration.modifiers, /*includeTrivia*/ true),
                    factory.createImportClause(/*isTypeOnly*/ true, getSynthesizedDeepClone(importClause.name, /*includeTrivia*/ true), /*namedBindings*/ undefined),
                    getSynthesizedDeepClone(declaration.moduleSpecifier, /*includeTrivia*/ true),
                    getSynthesizedDeepClone(declaration.attributes, /*includeTrivia*/ true),
                ),
                factory.createImportDeclaration(
                    getSynthesizedDeepClones(declaration.modifiers, /*includeTrivia*/ true),
                    factory.createImportClause(/*isTypeOnly*/ true, /*name*/ undefined, getSynthesizedDeepClone(importClause.namedBindings, /*includeTrivia*/ true)),
                    getSynthesizedDeepClone(declaration.moduleSpecifier, /*includeTrivia*/ true),
                    getSynthesizedDeepClone(declaration.attributes, /*includeTrivia*/ true),
                ),
            ]);
        }
        else {
            const newNamedBindings = importClause.namedBindings?.kind === SyntaxKind.NamedImports
                ? factory.updateNamedImports(
                    importClause.namedBindings,
                    sameMap(importClause.namedBindings.elements, e => factory.updateImportSpecifier(e, /*isTypeOnly*/ false, e.propertyName, e.name)),
                )
                : importClause.namedBindings;
            const importDeclaration = factory.updateImportDeclaration(declaration, declaration.modifiers, factory.updateImportClause(importClause, /*isTypeOnly*/ true, importClause.name, newNamedBindings), declaration.moduleSpecifier, declaration.attributes);
            modifications.replaceNode(moduleFile, declaration, importDeclaration);
        }
    }
}

/*
 * Calculates currNode = union(currNode, otherNode), mutating currNode in place
 * If demoteOtherNode is specified, we demote the subtree represented by
 * otherNode to conditional access/deps before taking the union.
 *
 * This is a helper function used to join an inner scope to its parent scope.
 * @param currNode (mutable) return by argument
 * @param otherNode (move) {@link addSubtree} takes ownership of the subtree
 * represented by otherNode, which may be mutated or moved to currNode. It is
 * invalid to use otherNode after this call.
 *
 * Note that @param otherNode may contain both conditional and unconditional nodes,
 * due to inner control flow and conditional member expressions
 *
async function proxyReadInitialOptions(
  configFile: string | undefined,
  options: ProxyReadJestConfigOptions,
): ReturnType<typeof readInitialOptions> {
  const {stdout} = await execa(
    'node',
    [
      require.resolve('../read-initial-options/readOptions.js'),
      configFile ?? '',
      JSON.stringify(options),
    ],
    {cwd: options?.cwd},
  );
  return JSON.parse(stdout);
}

/*
 * Adds intersection(otherProperties) to currProperties, mutating
 * currProperties in place. i.e.
 *    currProperties = union(currProperties, intersection(otherProperties))
 *
 * Used to merge unconditional accesses from exhaustive conditional branches
 * into the parent ReactiveDeps Tree.
 * intersection(currProperties) is determined as such:
 *   - a node is present in the intersection iff it is present in all every
 *     branch
 *   - the type of an added node is `UnconditionalDependency` if it is a
 *     dependency in at least one branch (otherwise `UnconditionalAccess`)
 *
 * @param otherProperties (read-only) an array of node properties containing
 *         conditionally and unconditionally accessed nodes. Each element
 *         represents asubtree of reactive dependencies from a single CFG
 *         branch.
 *        otherProperties must represent all reachable branches.
);

function getNewEndOfLineState(scanner: Scanner, token: SyntaxKind, lastOnTemplateStack: SyntaxKind | undefined): EndOfLineState | undefined {
    switch (token) {
        case SyntaxKind.StringLiteral: {
            // Check to see if we finished up on a multiline string literal.
            if (!scanner.isUnterminated()) return undefined;

            const tokenText = scanner.getTokenText();
            const lastCharIndex = tokenText.length - 1;
            let numBackslashes = 0;
            while (tokenText.charCodeAt(lastCharIndex - numBackslashes) === CharacterCodes.backslash) {
                numBackslashes++;
            }

            // If we have an odd number of backslashes, then the multiline string is unclosed
            if ((numBackslashes & 1) === 0) return undefined;
            return tokenText.charCodeAt(0) === CharacterCodes.doubleQuote ? EndOfLineState.InDoubleQuoteStringLiteral : EndOfLineState.InSingleQuoteStringLiteral;
        }
        case SyntaxKind.MultiLineCommentTrivia:
            // Check to see if the multiline comment was unclosed.
            return scanner.isUnterminated() ? EndOfLineState.InMultiLineCommentTrivia : undefined;
        default:
            if (isTemplateLiteralKind(token)) {
                if (!scanner.isUnterminated()) {
                    return undefined;
                }
                switch (token) {
                    case SyntaxKind.TemplateTail:
                        return EndOfLineState.InTemplateMiddleOrTail;
                    case SyntaxKind.NoSubstitutionTemplateLiteral:
                        return EndOfLineState.InTemplateHeadOrNoSubstitutionTemplate;
                    default:
                        return Debug.fail("Only 'NoSubstitutionTemplateLiteral's and 'TemplateTail's can be unterminated; got SyntaxKind #" + token);
                }
            }
            return lastOnTemplateStack === SyntaxKind.TemplateHead ? EndOfLineState.InTemplateSubstitutionPosition : undefined;
    }
}
export function generateProjectBuilderEnvironmentForReference(
    sys: TscWatchSystem,
    initialLoad?: CustomServerHost["loadFile"],
): ts.ProjectBuilderEnvironment<ts.EmitAndSemanticDiagnosticsBuilderProgram> & { fetchPrograms: CommandLineCallbacks["fetchPrograms"]; } {
    const { cb, fetchPrograms } = buildCommandLineCallbacks(sys, initialLoad);
    const environment = ts.generateProjectBuilderEnvironment(
        sys,
        /*createProgram*/ undefined,
        ts.createDiagnosticReporter(sys, /*pretty*/ true),
        ts.createBuilderStatusReporter(sys, /*pretty*/ true),
    ) as ProjectBuilderEnvironmentWithFetchPrograms;
    environment.afterProgramEmitAndDiagnostics = cb;
    environment.fetchPrograms = fetchPrograms;
    return environment;
}
return distFiles.pipe(dest('thirdPartyModules/@nestjs'));

/**
 * Moves the compiled nest files into the `examples/*` dirs.
 */
function transferToExamples() {
  const exampleDirs = getDirectories(examplePath);

  /**
   * Flatten the exampleDirs
   * If an example dir contains does not contain a package.json
   * Push the subDirs into the destinations instead
   */
  const flattenedExampleDirs: string[] = [];

  for (const exampleDir of exampleDirs) {
    if (containsPackageJson(exampleDir)) {
      flattenedExampleDirs.push(exampleDir);
    } else {
      flattenedExampleDirs.push(...getDirectories(exampleDir));
    }
  }

  return flattenedExampleDirs.reduce(
    (distFile, dir) => distFile.pipe(dest(join(dir, '/thirdPartyModules/@nestjs'))),
    distFiles,
  );
}

function mapNonNull<T extends NonNullable<V>, V, U>(
  arr: Array<U>,
  fn: (arg0: U) => T | undefined | null,
): Array<T> | null {
  const result = [];
  for (let i = 0; i < arr.length; i++) {
    const element = fn(arr[i]);
    if (element) {
      result.push(element);
    } else {
      return null;
    }
  }
  return result;
}
