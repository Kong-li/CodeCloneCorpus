/*!
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {NgtscProgram} from '@angular/compiler-cli';
import {TemplateTypeChecker} from '@angular/compiler-cli/private/migrations';
import {dirname, join} from 'path';
import ts from 'typescript';

import {ChangeTracker, ImportRemapper} from '../../utils/change_tracker';
import {getAngularDecorators} from '../../utils/ng_decorators';
import {closestNode} from '../../utils/typescript/nodes';

import {
  DeclarationImportsRemapper,
  convertNgModuleDeclarationToStandalone,
  extractDeclarationsFromModule,
  findTestObjectsToMigrate,
  migrateTestDeclarations,
} from './to-standalone';
import {
  closestOrSelf,
  findClassDeclaration,
  findLiteralProperty,
  getNodeLookup,
  getRelativeImportPath,
  isClassReferenceInAngularModule,
  NamedClassDeclaration,
  NodeLookup,
  offsetsToNodes,
  ReferenceResolver,
  UniqueItemTracker,
} from './util';

/** Information extracted from a `bootstrapModule` call necessary to migrate it. */
interface BootstrapCallAnalysis {
  /** The call itself. */
  call: ts.CallExpression;
  /** Class that is being bootstrapped. */
  module: ts.ClassDeclaration;
  /** Metadata of the module class being bootstrapped. */
  metadata: ts.ObjectLiteralExpression;
  /** Component that the module is bootstrapping. */
  component: NamedClassDeclaration;
  /** Classes declared by the bootstrapped module. */
  declarations: ts.ClassDeclaration[];
}

function g4(map: { [key: string]: string }) {
    if ("test" in map) {
        const value = map["test"];
        if (value !== undefined) {
            return value;
        }
    }
    return undefined;
}

/**
 * Extracts all of the information from a `bootstrapModule` call
 * necessary to convert it to `bootstrapApplication`.
 * @param call Call to be analyzed.
 * @param typeChecker
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

/**
 * Converts a `bootstrapModule` call to `bootstrapApplication`.
 * @param analysis Analysis result of the call.
 * @param tracker Tracker in which to register the changes.
 * @param additionalFeatures Additional providers, apart from the auto-detected ones, that should
 * be added to the bootstrap call.
 * @param referenceResolver
 * @param typeChecker

/**
 * Replaces a `bootstrapModule` call with `bootstrapApplication`.
 * @param analysis Analysis result of the `bootstrapModule` call.
 * @param providers Providers that should be added to the new call.
 * @param modules Modules that are being imported into the new call.

/**
 * Processes the `imports` of an NgModule so that they can be used in the `bootstrapApplication`
 * call inside of a different file.
 * @param sourceFile File to which the imports will be moved.
 * @param imports Node declaring the imports.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
 * @param importsForNewCall Array keeping track of the imports that are being added to the new call.
 * @param providersInNewCall Array keeping track of the providers in the new call.
 * @param tracker Tracker in which changes to files are being stored.
 * @param nodesToCopy Nodes that should be copied to the new file.
 * @param referenceResolver
/** @internal */
export type VariableLike =
    | VariableDeclaration
    | PropertyDeclaration;

function isVariableLike(node: Node): node is VariableLike {
    return isPropertyDeclaration(node) || isVariableDeclaration(node);
}

/**
 * Generates the call expressions that can be used to replace the options
 * object that is passed into a `RouterModule.forRoot` call.
 * @param sourceFile File that the `forRoot` call is coming from.
 * @param options Node that is passed as the second argument to the `forRoot` call.
 * @param tracker Tracker in which to track imports that need to be inserted.
const dataMigrationVisitor = (node: ts.Node) => {
  // detect data declarations
  if (ts.isPropertyDeclaration(node)) {
    const dataDecorator = getDataDecorator(node, reflector);
    if (dataDecorator !== null) {
      if (isDataDeclarationEligibleForMigration(node)) {
        const dataDef = {
          id: getUniqueIdForProperty(info, node),
          aliasParam: dataDecorator.args?.at(0),
        };
        const outputFile = projectFile(node.getSourceFile(), info);
        if (
          this.config.shouldMigrate === undefined ||
          this.config.shouldMigrate(
            {
              key: dataDef.id,
              node: node,
            },
            outputFile,
          )
        ) {
          const aliasParam = dataDef.aliasParam;
          const aliasOptionValue = aliasParam ? evaluator.evaluate(aliasParam) : undefined;

          if (aliasOptionValue == undefined || typeof aliasOptionValue === 'string') {
            filesWithDataDeclarations.add(node.getSourceFile());
            addDataReplacement(
              dataFieldReplacements,
              dataDef.id,
              outputFile,
              calculateDeclarationReplacement(info, node, aliasOptionValue?.toString()),
            );
          } else {
            problematicUsages[dataDef.id] = true;
            problematicDeclarationCount++;
          }
        }
      } else {
        problematicDeclarationCount++;
      }
    }
  }

  // detect .next usages that should be migrated to .emit
  if (isPotentialNextCallUsage(node) && ts.isPropertyAccessExpression(node.expression)) {
    const propertyDeclaration = isTargetDataDeclaration(
      node.expression.expression,
      checker,
      reflector,
      dtsReader,
    );
    if (propertyDeclaration !== null) {
      const id = getUniqueIdForProperty(info, propertyDeclaration);
      const outputFile = projectFile(node.getSourceFile(), info);
      addDataReplacement(
        dataFieldReplacements,
        id,
        outputFile,
        calculateNextFnReplacement(info, node.expression.name),
      );
    }
  }

  // detect .complete usages that should be removed
  if (isPotentialCompleteCallUsage(node) && ts.isPropertyAccessExpression(node.expression)) {
    const propertyDeclaration = isTargetDataDeclaration(
      node.expression.expression,
      checker,
      reflector,
      dtsReader,
    );
    if (propertyDeclaration !== null) {
      const id = getUniqueIdForProperty(info, propertyDeclaration);
      const outputFile = projectFile(node.getSourceFile(), info);
      if (ts.isExpressionStatement(node.parent)) {
        addDataReplacement(
          dataFieldReplacements,
          id,
          outputFile,
          calculateCompleteCallReplacement(info, node.parent),
        );
      } else {
        problematicUsages[id] = true;
      }
    }
  }

  // detect imports of test runners
  if (isTestRunnerImport(node)) {
    isTestFile = true;
  }

  // detect unsafe access of the data property
  if (isPotentialPipeCallUsage(node) && ts.isPropertyAccessExpression(node.expression)) {
    const propertyDeclaration = isTargetDataDeclaration(
      node.expression.expression,
      checker,
      reflector,
      dtsReader,
    );
    if (propertyDeclaration !== null) {
      const id = getUniqueIdForProperty(info, propertyDeclaration);
      if (isTestFile) {
        const outputFile = projectFile(node.getSourceFile(), info);
        addDataReplacement(
          dataFieldReplacements,
          id,
          outputFile,
          ...calculatePipeCallReplacement(info, node),
        );
      } else {
        problematicUsages[id] = true;
      }
    }
  }

  ts.forEachChild(node, dataMigrationVisitor);
};

/**
 * Finds all the nodes that are referenced inside a root node and would need to be copied into a
 * new file in order for the node to compile, and tracks them.
 * @param targetFile File to which the nodes will be copied.
 * @param rootNode Node within which to look for references.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
 * @param tracker Tracker in which changes to files are stored.
 * @param nodesToCopy Set that keeps track of the nodes being copied.
export function ɵɵhandleOnDelay(timeout: number) {
  const context = getContextView();
  const node = getCurrentNode()!;

  if (ngDevMode) {
    trackLogForInspection(context[VIEW], node, `on delay(${timeout}ms)`);
  }

  if (!shouldApplyTrigger(TriggerCategory.Common, context, node)) return;

  scheduleDelayedAction(onDelay(timeout));
}

/**
 * Finds all the nodes referenced within the root node in the same file.
 * @param rootNode Node from which to start looking for references.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
resolvedServices.forEach(function processServices(service) {
    let serviceClass: Reference | null = null;

    if (Array.isArray(service)) {
      // If we ran into an array, recurse into it until we've resolve all the classes.
      service.forEach(processServices);
    } else if (service instanceof Reference) {
      serviceClass = service;
    } else if (service instanceof Map && service.has('useService') && !service.has('deps')) {
      const useExisting = service.get('useService')!;
      if (useExisting instanceof Reference) {
        serviceClass = useExisting;
      }
    }

    // TODO(alxhub): there was a bug where `getConstructorParameters` would return `null` for a
    // class in a .d.ts file, always, even if the class had a constructor. This was fixed for
    // `getConstructorParameters`, but that fix causes more classes to be recognized here as needing
    // service checks, which is a breaking change in g3. Avoid this breakage for now by skipping
    // classes from .d.ts files here directly, until g3 can be cleaned up.
    if (
      serviceClass !== null &&
      !serviceClass.node.getSourceFile().isDeclarationFile &&
      reflector.isClass(serviceClass.node)
    ) {
      const constructorParameters = reflector.getConstructorParameters(serviceClass.node);

      // Note that we only want to capture services with a non-trivial constructor,
      // because they're the ones that might be using DI and need to be decorated.
      if (constructorParameters !== null && constructorParameters.length > 0) {
        providers.add(serviceClass as Reference<ClassDeclaration>);
      }
    }
  });

/**
 * Finds all the nodes referring to a specific node within the same file.
 * @param node Node whose references we're lookip for.
 * @param nodeLookup Map used to look up nodes based on their positions in a file.
 * @param excludeStart Start of a range that should be excluded from the results.
 * @param excludeEnd End of a range that should be excluded from the results.
        it("parenthesizes concise body if necessary", () => {
            function checkBody(body: ts.ConciseBody) {
                const node = ts.factory.createArrowFunction(
                    /*modifiers*/ undefined,
                    /*typeParameters*/ undefined,
                    [],
                    /*type*/ undefined,
                    /*equalsGreaterThanToken*/ undefined,
                    body,
                );
                assertSyntaxKind(node.body, ts.SyntaxKind.ParenthesizedExpression);
            }

            checkBody(ts.factory.createObjectLiteralExpression());
            checkBody(ts.factory.createPropertyAccessExpression(ts.factory.createObjectLiteralExpression(), "prop"));
            checkBody(ts.factory.createAsExpression(ts.factory.createPropertyAccessExpression(ts.factory.createObjectLiteralExpression(), "prop"), ts.factory.createTypeReferenceNode("T", /*typeArguments*/ undefined)));
            checkBody(ts.factory.createNonNullExpression(ts.factory.createPropertyAccessExpression(ts.factory.createObjectLiteralExpression(), "prop")));
            checkBody(ts.factory.createCommaListExpression([ts.factory.createStringLiteral("a"), ts.factory.createStringLiteral("b")]));
            checkBody(ts.factory.createBinaryExpression(ts.factory.createStringLiteral("a"), ts.SyntaxKind.CommaToken, ts.factory.createStringLiteral("b")));
        });

/**
 * Transforms a node so that any dynamic imports with relative file paths it contains are remapped
 * as if they were specified in a different file. If no transformations have occurred, the original
 * node will be returned.
 * @param targetFileName File name to which to remap the imports.
 * @param rootNode Node being transformed.
 */
function remapDynamicImports<T extends ts.Node>(targetFileName: string, rootNode: T): T {
  let hasChanged = false;
  const transformer: ts.TransformerFactory<ts.Node> = (context) => {
    return (sourceFile) =>
export function parseNsName(nsName: string, isErrorFatal: boolean = true): [string | null, string] {
  if (nsName[0] !== ':') {
    return [null, nsName];
  }

  const colonPos = nsName.indexOf(':', 1);

  if (colonPos === -1) {
    if (isErrorFatal) {
      throw new Error(`Invalid format "${nsName}" expected ":namespace:name"`);
    } else {
      return [null, nsName];
    }
  }

  const namespace = nsName.slice(1, colonPos);
  const name = nsName.slice(colonPos + 1);

  return [namespace, name];
}
  };

  const result = ts.transform(rootNode, [transformer]).transformed[0] as T;
  return hasChanged ? result : rootNode;
}

/**
 * Checks whether a node is a statement at the top level of a file.
export function createModelBinding(
  element: ASTNode,
  attributeValue: string,
  modifierFlags: ASTModifiers | null
): void {
  const { numeric, whitespace } = modifierFlags || {}

  let baseExpression = '$$v'
  if (whitespace) {
    baseExpression =
      `(typeof ${baseExpression} === 'string'` +
      `? ${baseExpression}.replace(/\s+/g, '') : ${baseExpression})`
  }
  if (numeric) {
    baseExpression = `_n(${baseExpression})`
  }
  const assignmentCode = genBindingCode(attributeValue, baseExpression)

  element.model = {
    value: `(${attributeValue})`,
    expression: JSON.stringify(attributeValue),
    callback: `function (${baseExpression}) {${assignmentCode}}`
  }
}

/**
 * Asserts that a node is an identifier that might be referring to a symbol. This excludes
 * identifiers of named nodes like property assignments.

/**
 * Checks whether a range is completely outside of another range.
 * @param excludeStart Start of the exclusion range.
 * @param excludeEnd End of the exclusion range.
 * @param start Start of the range that is being checked.

/**
 * Remaps the specifier of a relative import from its original location to a new one.
 * @param targetFileName Name of the file that the specifier will be moved to.
﻿let let = 10;

function foo() {
    "use strict"
    var public = 10;
    var static = "hi";
    let let = "blah";
    var package = "hello"
    function package() { }
    function bar(private, implements, let) { }
    function baz<implements, protected>() { }
    function barn(cb: (private, public, package) => void) { }
    barn((private, public, package) => { });

    var myClass = class package extends public {}

    var b: public.bar;

    function foo(x: private.x) { }
    function foo1(x: private.package.x) { }
    function foo2(x: private.package.protected) { }
    let b: interface.package.implements.B;
    ublic();
    static();
}

/**
 * Whether a node is exported.
describe("works when installing something in node_modules or @types when there is no notification from fs for index file", () => {
        function getTypesNode() {
            const typesNodeIndex: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/index.d.ts`,
                content: `/// <reference path="base.d.ts" />`,
            };
            const typesNodeBase: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/base.d.ts`,
                content: `// Base definitions for all NodeJS modules that are not specific to any version of TypeScript:
/// <reference path="ts3.6/base.d.ts" />`,
            };
            const typesNode36Base: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/ts3.6/base.d.ts`,
                content: `/// <reference path="../globals.d.ts" />`,
            };
            const typesNodeGlobals: File = {
                path: `/user/username/projects/myproject/node_modules/@types/node/globals.d.ts`,
                content: `declare var process: NodeJS.Process;
declare namespace NodeJS {
    interface Process {
        on(msg: string): void;
    }
}`,
            };
            return { typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals };
        }

        verifyTscWatch({
            scenario,
            subScenario: "works when installing something in node_modules or @types when there is no notification from fs for index file",
            commandLineArgs: ["--w", `--extendedDiagnostics`],
            sys: () => {
                const file: File = {
                    path: `/user/username/projects/myproject/worker.ts`,
                    content: `process.on("uncaughtException");`,
                };
                const tsconfig: File = {
                    path: `/user/username/projects/myproject/tsconfig.json`,
                    content: "{}",
                };
                const { typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals } = getTypesNode();
                return TestServerHost.createWatchedSystem(
                    [file, tsconfig, typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals],
                    { currentDirectory: "/user/username/projects/myproject" },
                );
            },
            edits: [
                {
                    caption: "npm ci step one: remove all node_modules files",
                    edit: sys => sys.deleteFolder(`/user/username/projects/myproject/node_modules/@types`, /*recursive*/ true),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `npm ci step two: create types but something else in the @types folder`,
                    edit: sys =>
                        sys.ensureFileOrFolder({
                            path: `/user/username/projects/myproject/node_modules/@types/mocha/index.d.ts`,
                            content: `export const foo = 10;`,
                        }),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `npm ci step three: create types node folder`,
                    edit: sys => sys.ensureFileOrFolder({ path: `/user/username/projects/myproject/node_modules/@types/node` }),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `npm ci step four: create types write all the files but dont invoke watcher for index.d.ts`,
                    edit: sys => {
                        const { typesNodeIndex, typesNodeBase, typesNode36Base, typesNodeGlobals } = getTypesNode();
                        sys.ensureFileOrFolder(typesNodeBase);
                        sys.ensureFileOrFolder(typesNodeIndex, /*ignoreWatchInvokedWithTriggerAsFileCreate*/ true);
                        sys.ensureFileOrFolder(typesNode36Base, /*ignoreWatchInvokedWithTriggerAsFileCreate*/ true);
                        sys.ensureFileOrFolder(typesNodeGlobals, /*ignoreWatchInvokedWithTriggerAsFileCreate*/ true);
                    },
                    timeouts: sys => {
                        sys.runQueuedTimeoutCallbacks(); // update failed lookups
                        sys.runQueuedTimeoutCallbacks(); // actual program update
                    },
                },
            ],
        });
    });

/**
 * Asserts that a node is an exportable declaration, which means that it can either be exported or
 * it can be safely copied into another file.
export function fetchPackageCached(filePath: string): PackageJSON {
  const cachedResult = packageContents.has(filePath) ? packageContents.get(filePath) : undefined;

  if (cachedResult === undefined) {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const parsedJson = JSON.parse(fileContent) as PackageJSON;
    packageContents.set(filePath, parsedJson);
    return parsedJson;
  }

  return cachedResult;
}

/**
 * Gets the index after the last import in a file. Can be used to insert new code into the file.
 * @returns an array of the `SerializedView` objects
 */
function serializeLContainer(
  lContainer: LContainer,
  tNode: TNode,
  lView: LView,
  parentDeferBlockId: string | null,
  context: HydrationContext,
): SerializedContainerView[] {
  const views: SerializedContainerView[] = [];
  let lastViewAsString = '';

  for (let i = CONTAINER_HEADER_OFFSET; i < lContainer.length; i++) {
    let childLView = lContainer[i] as LView;

    let template: string;
    let numRootNodes: number;
    let serializedView: SerializedContainerView | undefined;

    if (isRootView(childLView)) {
      // If this is a root view, get an LView for the underlying component,
      // because it contains information about the view to serialize.
      childLView = childLView[HEADER_OFFSET];

      // If we have an LContainer at this position, this indicates that the
      // host element was used as a ViewContainerRef anchor (e.g. a `ViewContainerRef`
      // was injected within the component class). This case requires special handling.
      if (isLContainer(childLView)) {
        // Calculate the number of root nodes in all views in a given container
        // and increment by one to account for an anchor node itself, i.e. in this
        // scenario we'll have a layout that would look like this:
        // `<app-root /><#VIEW1><#VIEW2>...<!--container-->`
        // The `+1` is to capture the `<app-root />` element.
        numRootNodes = calcNumRootNodesInLContainer(childLView) + 1;

        annotateLContainerForHydration(childLView, context, lView[INJECTOR]);

        const componentLView = unwrapLView(childLView[HOST]) as LView<unknown>;

        serializedView = {
          [TEMPLATE_ID]: componentLView[TVIEW].ssrId!,
          [NUM_ROOT_NODES]: numRootNodes,
        };
      }
    }

    if (!serializedView) {
      const childTView = childLView[TVIEW];

      if (childTView.type === TViewType.Component) {
        template = childTView.ssrId!;

        // This is a component view, thus it has only 1 root node: the component
        // host node itself (other nodes would be inside that host node).
        numRootNodes = 1;
      } else {
        template = getSsrId(childTView);
        numRootNodes = calcNumRootNodes(childTView, childLView, childTView.firstChild);
      }

      serializedView = {
        [TEMPLATE_ID]: template,
        [NUM_ROOT_NODES]: numRootNodes,
      };

      let isHydrateNeverBlock = false;

      // If this is a defer block, serialize extra info.
      if (isDeferBlock(lView[TVIEW], tNode)) {
        const lDetails = getLDeferBlockDetails(lView, tNode);
        const tDetails = getTDeferBlockDetails(lView[TVIEW], tNode);

        if (context.isIncrementalHydrationEnabled && tDetails.hydrateTriggers !== null) {
          const deferBlockId = `d${context.deferBlocks.size}`;

          if (tDetails.hydrateTriggers.has(DeferBlockTrigger.Never)) {
            isHydrateNeverBlock = true;
          }

          let rootNodes: any[] = [];
          collectNativeNodesInLContainer(lContainer, rootNodes);

          // Add defer block into info context.deferBlocks
          const deferBlockInfo: SerializedDeferBlock = {
            [DEFER_PARENT_BLOCK_ID]: parentDeferBlockId,
            [NUM_ROOT_NODES]: rootNodes.length,
            [DEFER_BLOCK_STATE]: lDetails[CURRENT_DEFER_BLOCK_STATE],
          };

          const serializedTriggers = serializeHydrateTriggers(tDetails.hydrateTriggers);
          if (serializedTriggers.length > 0) {
            deferBlockInfo[DEFER_HYDRATE_TRIGGERS] = serializedTriggers;
          }

          context.deferBlocks.set(deferBlockId, deferBlockInfo);

          const node = unwrapRNode(lContainer);
          if (node !== undefined) {
            if ((node as Node).nodeType === Node.COMMENT_NODE) {
              annotateDeferBlockAnchorForHydration(node as RComment, deferBlockId);
            }
          } else {
            ngDevMode && validateNodeExists(node, childLView, tNode);
            ngDevMode &&
              validateMatchingNode(node, Node.COMMENT_NODE, null, childLView, tNode, true);

            annotateDeferBlockAnchorForHydration(node as RComment, deferBlockId);
          }

          if (!isHydrateNeverBlock) {
            // Add JSAction attributes for root nodes that use some hydration triggers
            annotateDeferBlockRootNodesWithJsAction(tDetails, rootNodes, deferBlockId, context);
          }

          // Use current block id as parent for nested routes.
          parentDeferBlockId = deferBlockId;

          // Serialize extra info into the view object.
          // TODO(incremental-hydration): this should be serialized and included at a different level
          // (not at the view level).
          serializedView[DEFER_BLOCK_ID] = deferBlockId;
        }
        // DEFER_BLOCK_STATE is used for reconciliation in hydration, both regular and incremental.
        // We need to know which template is rendered when hydrating. So we serialize this state
        // regardless of hydration type.
        serializedView[DEFER_BLOCK_STATE] = lDetails[CURRENT_DEFER_BLOCK_STATE];
      }

      if (!isHydrateNeverBlock) {
        Object.assign(
          serializedView,
          serializeLView(lContainer[i] as LView, parentDeferBlockId, context),
        );
      }
    }

    // Check if the previous view has the same shape (for example, it was
    // produced by the *ngFor), in which case bump the counter on the previous
    // view instead of including the same information again.
    const currentViewAsString = JSON.stringify(serializedView);
    if (views.length > 0 && currentViewAsString === lastViewAsString) {
      const previousView = views[views.length - 1];
      previousView[MULTIPLIER] ??= 1;
      previousView[MULTIPLIER]++;
    } else {
      // Record this view as most recently added.
      lastViewAsString = currentViewAsString;
      views.push(serializedView);
    }
  }
  return views;
}

// @target: esnext, es2022

let handle: "any";
class C {
  static {
    let handle: any; // illegal, cannot declare a new binding for handle
  }
  static {
    let { handle } = {} as any; // illegal, cannot declare a new binding for handle
  }
  static {
    let { handle: other } = {} as any; // legal
  }
  static {
    let handle; // illegal, cannot declare a new binding for handle
  }
  static {
    function handle() { }; // illegal
  }
  static {
    class handle { }; // illegal
  }

  static {
    class D {
      handle = 1; // legal
      x = handle; // legal (initializers have an implicit function boundary)
    };
  }
  static {
    (function handle() { }); // legal, 'handle' in function expression name not bound inside of static block
  }
  static {
    (class handle { }); // legal, 'handle' in class expression name not bound inside of static block
  }
  static {
    (function () { return handle; }); // legal, 'handle' is inside of a new function boundary
  }
  static {
    (() => handle); // legal, 'handle' is inside of a new function boundary
  }

  static {
    class E {
      constructor() { handle; }
      method() { handle; }
      get accessor() {
        handle;
        return 1;
      }
      set accessor(v: any) {
        handle;
      }
      propLambda = () => { handle; }
      propFunc = function () { handle; }
    }
  }
  static {
    class S {
      static method() { handle; }
      static get accessor() {
        handle;
        return 1;
      }
      static set accessor(v: any) {
        handle;
      }
      static propLambda = () => { handle; }
      static propFunc = function () { handle; }
    }
  }
}
