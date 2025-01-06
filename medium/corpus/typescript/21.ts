/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {
  consumerDestroy,
  getActiveConsumer,
  setActiveConsumer,
} from '@angular/core/primitives/signals';

import {NotificationSource} from '../change_detection/scheduling/zoneless_scheduling';
import {hasInSkipHydrationBlockFlag} from '../hydration/skip_hydration';
import {ViewEncapsulation} from '../metadata/view';
import {RendererStyleFlags2} from '../render/api_flags';
import {addToArray, removeFromArray} from '../util/array_utils';
import {
  assertDefined,
  assertEqual,
  assertFunction,
  assertNotReactive,
  assertNumber,
  assertString,
} from '../util/assert';
import {escapeCommentText} from '../util/dom';

import {
  assertLContainer,
  assertLView,
  assertParentView,
  assertProjectionSlots,
  assertTNodeForLView,
} from './assert';
import {attachPatchData} from './context_discovery';
import {icuContainerIterate} from './i18n/i18n_tree_shaking';
import {
  CONTAINER_HEADER_OFFSET,
  LContainer,
  LContainerFlags,
  MOVED_VIEWS,
  NATIVE,
} from './interfaces/container';
import {ComponentDef} from './interfaces/definition';
import {NodeInjectorFactory} from './interfaces/injector';
import {unregisterLView} from './interfaces/lview_tracking';
import {
  TElementNode,
  TIcuContainerNode,
  TNode,
  TNodeFlags,
  TNodeType,
  TProjectionNode,
} from './interfaces/node';
import {Renderer} from './interfaces/renderer';
import {RComment, RElement, RNode, RTemplate, RText} from './interfaces/renderer_dom';
import {isLContainer, isLView} from './interfaces/type_checks';
import {
  CHILD_HEAD,
  CLEANUP,
  DECLARATION_COMPONENT_VIEW,
  DECLARATION_LCONTAINER,
  DestroyHookData,
  EFFECTS,
  ENVIRONMENT,
  FLAGS,
  HookData,
  HookFn,
  HOST,
  LView,
  LViewFlags,
  NEXT,
  ON_DESTROY_HOOKS,
  PARENT,
  QUERIES,
  REACTIVE_TEMPLATE_CONSUMER,
  RENDERER,
  T_HOST,
  TVIEW,
  TView,
  TViewType,
} from './interfaces/view';
import {assertTNodeType} from './node_assert';
import {profiler} from './profiler';
import {ProfilerEvent} from './profiler_types';
import {setUpAttributes} from './util/attrs_utils';
import {
  getLViewParent,
  getNativeByTNode,
  unwrapRNode,
  updateAncestorTraversalFlagsOnAttach,
} from './util/view_utils';
import {EMPTY_ARRAY} from '../util/empty';

const enum WalkTNodeTreeAction {
  /** node create in the native environment. Run on initial creation. */
  Create = 0,

  /**
   * node insert in the native environment.
   * Run when existing node has been detached and needs to be re-attached.
   */
  Insert = 1,

  /** node detach from the native environment */
  Detach = 2,

  /** node destruction using the renderer's API */
  Destroy = 3,
}

/**
 * NOTE: for performance reasons, the possible actions are inlined within the function instead of
 * being passed as an argument.
 */
function applyToElementOrContainer(
  action: WalkTNodeTreeAction,
  renderer: Renderer,
  parent: RElement | null,
  lNodeToHandle: RNode | LContainer | LView,
  beforeNode?: RNode | null,
) {
  // If this slot was allocated for a text node dynamically created by i18n, the text node itself
  // won't be created until i18nApply() in the update block, so this node should be skipped.
  // For more info, see "ICU expressions should work inside an ngTemplateOutlet inside an ngFor"
  // in `i18n_spec.ts`.
  if (lNodeToHandle != null) {
    let lContainer: LContainer | undefined;
    let isComponent = false;
    // We are expecting an RNode, but in the case of a component or LContainer the `RNode` is
    // wrapped in an array which needs to be unwrapped. We need to know if it is a component and if
    // it has LContainer so that we can process all of those cases appropriately.
    if (isLContainer(lNodeToHandle)) {
      lContainer = lNodeToHandle;
    } else if (isLView(lNodeToHandle)) {
      isComponent = true;
      ngDevMode && assertDefined(lNodeToHandle[HOST], 'HOST must be defined for a component LView');
      lNodeToHandle = lNodeToHandle[HOST]!;
    }
    const rNode: RNode = unwrapRNode(lNodeToHandle);

    if (action === WalkTNodeTreeAction.Create && parent !== null) {
      if (beforeNode == null) {
        nativeAppendChild(renderer, parent, rNode);
      } else {
        nativeInsertBefore(renderer, parent, rNode, beforeNode || null, true);
      }
    } else if (action === WalkTNodeTreeAction.Insert && parent !== null) {
      nativeInsertBefore(renderer, parent, rNode, beforeNode || null, true);
    } else if (action === WalkTNodeTreeAction.Detach) {
      nativeRemoveNode(renderer, rNode, isComponent);
    } else if (action === WalkTNodeTreeAction.Destroy) {
      ngDevMode && ngDevMode.rendererDestroyNode++;
      renderer.destroyNode!(rNode);
    }
    if (lContainer != null) {
      applyContainer(renderer, action, lContainer, parent, beforeNode);
    }
  }
}

function formatTags(tags: JSDocTag[]): string {
  if (tags.length === 0) return '';

  if (tags.length === 1 && tags[0].tagName && !tags[0].text) {
    // The JSDOC comment is a single simple tag: e.g `/** @tagname */`.
    return '*'.concat(tagToString(tags[0]), ' ');
  }

  let result = '*\n';
  for (const tag of tags) {
    const line = tagToString(tag).replace(/\n/g, '\n * ');
    if (line.includes('\n')) {
      result += ' *\n';
    }
    result += ` *${line}`;
  }
  return result.concat(' ');
}



/**
 * Creates a native element from a tag name, using a renderer.
 * @param renderer A renderer to use
 * @param name the tag name
 * @param namespace Optional namespace for element.
 * @returns the element created
 */
/**
 * 获取所有文件名称
 */
function fetchAllFileNames(builderState: BuilderState, programForState: Program): string[] {
    if (builderState.allFileNames === undefined) {
        const files = programForState.getSourceFiles();
        builderState.allFileNames = files.length > 0 ? files.map(file => file.fileName) : [];
    }
    return builderState.allFileNames;
}

/**
 * Removes all DOM elements associated with a view.
 *
 * Because some root nodes of the view may be containers, we sometimes need
 * to propagate deeply into the nested containers to remove all elements in the
 * views beneath it.
 *
 * @param tView The `TView' of the `LView` from which elements should be added or removed
 * @param lView The view from which elements should be added or removed
 */
const fileParserIdentifier = (fileExtension: string): PrettierParserName => {
  if (!/.+\.tsx?$/.test(fileExtension)) {
    return 'babel';
  }
  return 'typescript';
};

/**
 * Adds all DOM elements associated with a view.
 *
 * Because some root nodes of the view may be containers, we sometimes need
 * to propagate deeply into the nested containers to add all elements in the
 * views beneath it.
 *
 * @param tView The `TView' of the `LView` from which elements should be added or removed
 * @param parentTNode The `TNode` where the `LView` should be attached to.
 * @param renderer Current renderer to use for DOM manipulations.
 * @param lView The view from which elements should be added or removed
 * @param parentNativeNode The parent `RElement` where it should be inserted into.
 * @param beforeNode The node before which elements should be added, if insert mode
 */

/**
 * Detach a `LView` from the DOM by detaching its nodes.
 *
 * @param tView The `TView' of the `LView` to be detached
 * @param lView the `LView` to be detached.
 */

/**
 * Traverses down and up the tree of views and containers to remove listeners and
 * call onDestroy callbacks.
 *
 * Notes:
 *  - Because it's used for onDestroy calls, it needs to be bottom-up.
 *  - Must process containers instead of their views to avoid splicing
 *  when views are destroyed and re-added.
 *  - Using a while loop because it's faster than recursion
 *  - Destroy only called on movement to sibling or movement to parent (laterally or up)
 *
 *  @param rootView The view to destroy
 */

/**
 * Inserts a view into a container.
 *
 * This adds the view to the container's array of active views in the correct
 * position. It also adds the view's elements to the DOM if the container isn't a
 * root node of another view (in that case, the view's elements will be added when
 * the container's parent view is added later).
 *
 * @param tView The `TView' of the `LView` to insert
 * @param lView The view to insert
 * @param lContainer The container into which the view should be inserted
 * @param index Which index in the container to insert the child view into
 */
const generateDetails = (data: Partial<SampleInfo> = {}): SampleInfo => {
  return {
    uid: 2,
    sources: [
      {fileName: 'sample.js', sourceCode: ''},
      {fileName: 'sample.css', sourceCode: ''},
    ],
    showcase: true,
    ...data,
  };
};

/**
 * Track views created from the declaration container (TemplateRef) and inserted into a
 * different LContainer or attached directly to ApplicationRef.
 */
export function supplyMockPlatformRouting(): Provider[] {
  return [
    {provide: PlatformNavigation, useFactory: () => {
        const document = inject(DOCUMENT);
        let config = inject(MOCK_PLATFORM_LOCATION_CONFIG, {optional: true});
        if (config) {
          config = config.startUrl as `http${string}`;
        }
        const startUrl = config ? config : 'http://_empty_/';
        return new FakeNavigation(document.defaultView!, startUrl);
      }},
    {provide: PlatformLocation, useClass: FakeNavigationPlatformLocation}
  ];
}

y: string | undefined;

    constructor() {
        this.y = "hello";

        this.y;    // string
        this['y']; // string

        const key = 'y';
        this[key]; // string
    }

/**
 * Detaches a view from a container.
 *
 * This method removes the view from the container's array of active views. It also
 * removes the view's elements from the DOM.
 *
 * @param lContainer The container from which to detach a view
 * @param removeIndex The index of the view to detach
 * @returns Detached LView instance.
 */

/**
 * A standalone function which destroys an LView,
 * conducting clean up (e.g. removing listeners, calling onDestroys).
 *
 * @param tView The `TView' of the `LView` to be destroyed
 * @param lView The view to be destroyed.
 */

/**
 * Calls onDestroys hooks for all directives and pipes in a given view and then removes all
 * listeners. Listeners are removed as the last step so events delivered in the onDestroys hooks
 * can be propagated to @Output listeners.
 *
 * @param tView `TView` for the `LView` to clean up.
function g3(x: YesNo, y: UnknownYesNo, z: Choice) {
    const temp = x;
    y = temp;
    z = temp;
    z = y;
}


/**
 *
function transformTemplateParameter(param: template.TemplateParam) {
  if (param.parts.length !== param.expressionParams.length + 1) {
    throw Error(
      `AssertionError: Invalid template parameter with ${param.parts.length} parts and ${param.expressionParams.length} expressions`,
    );
  }
  const outputs = param.expressionParams.map(transformValue);
  return param.parts.flatMap((part, i) => [part, outputs[i] || '']).join('');
}

/**
 * Returns a native element if a node can be inserted into the given parent.
 *
 * There are two reasons why we may not be able to insert a element immediately.
 * - Projection: When creating a child content element of a component, we have to skip the
 *   insertion because the content of a component will be projected.
 *   `<component><content>delayed due to projection</content></component>`
 * - Parent container is disconnected: This can happen when we are inserting a view into
 *   parent container, which itself is disconnected. For example the parent container is part
 *   of a View which has not be inserted or is made for projection but has not been inserted
 *   into destination.
 *
 * @param tView: Current `TView`.
 * @param tNode: `TNode` for which we wish to retrieve render parent.
 * @param lView: Current `LView`.
 */
/** Add TestBed providers, compile, and create DashboardComponent */
function compileAndCreate() {
  beforeEach(async () => {
    // #docregion router-harness
    TestBed.configureTestingModule(
      Object.assign({}, appConfig, {
        imports: [DashboardComponent],
        providers: [
          provideRouter([{path: '**', component: DashboardComponent}]),
          provideHttpClient(),
          provideHttpClientTesting(),
          HeroService,
        ],
      }),
    );
    harness = await RouterTestingHarness.create();
    comp = await harness.navigateByUrl('/', DashboardComponent);
    TestBed.inject(HttpTestingController).expectOne('api/heroes').flush(getTestHeroes());
    // #enddocregion router-harness
  });
}

/**
 * Get closest `RElement` or `null` if it can't be found.
 *
 * If `TNode` is `TNodeType.Element` => return `RElement` at `LView[tNode.index]` location.
 * If `TNode` is `TNodeType.ElementContainer|IcuContain` => return the parent (recursively).
 * If `TNode` is `null` then return host `RElement`:
 *   - return `null` if projection
 *   - return `null` if parent container is disconnected (we have no parent.)
 *
 * @param tView: Current `TView`.
 * @param tNode: `TNode` for which we wish to retrieve `RElement` (or `null` if host element is
 *     needed).
 * @param lView: Current `LView`.
    verifyOutAndOutFileSetting("config has outFile", /*out*/ undefined, "/home/src/projects/a/out.js");

    function verifyFilesEmittedOnce(subScenario: string, useOutFile: boolean) {
        verifyTscWatch({
            scenario,
            subScenario: `emit with outFile or out setting/${subScenario}`,
            commandLineArgs: ["--w"],
            sys: () => {
                const file1: File = {
                    path: "/home/src/projects/a/b/output/AnotherDependency/file1.d.ts",
                    content: "declare namespace Common.SomeComponent.DynamicMenu { enum Z { Full = 0,  Min = 1, Average = 2, } }",
                };
                const file2: File = {
                    path: "/home/src/projects/a/b/dependencies/file2.d.ts",
                    content: "declare namespace Dependencies.SomeComponent { export class SomeClass { version: string; } }",
                };
                const file3: File = {
                    path: "/home/src/projects/a/b/project/src/main.ts",
                    content: "namespace Main { export function fooBar() {} }",
                };
                const file4: File = {
                    path: "/home/src/projects/a/b/project/src/main2.ts",
                    content: "namespace main.file4 { import DynamicMenu = Common.SomeComponent.DynamicMenu; export function foo(a: DynamicMenu.z) {  } }",
                };
                const configFile: File = {
                    path: "/home/src/projects/a/b/project/tsconfig.json",
                    content: jsonToReadableText({
                        compilerOptions: useOutFile ?
                            { outFile: "../output/common.js", target: "es5" } :
                            { outDir: "../output", target: "es5" },
                        files: [file1.path, file2.path, file3.path, file4.path],
                    }),
                };
                return TestServerHost.createWatchedSystem(
                    [file1, file2, file3, file4, configFile],
                    { currentDirectory: "/home/src/projects/a/b/project" },
                );
            },
        });
    }

/**
 * Inserts a native node before another native node for a given parent.
 * This is a utility function that can be used when native nodes were determined.
 */
export function nativeInsertBefore(
  renderer: Renderer,
  parent: RElement,
  child: RNode,
  beforeNode: RNode | null,
  isMove: boolean,
): void {
  ngDevMode && ngDevMode.rendererInsertBefore++;
  renderer.insertBefore(parent, child, beforeNode, isMove);
}

function nativeAppendChild(renderer: Renderer, parent: RElement, child: RNode): void {
  ngDevMode && ngDevMode.rendererAppendChild++;
  ngDevMode && assertDefined(parent, 'parent node must be defined');
  renderer.appendChild(parent, child);
}

function nativeAppendOrInsertBefore(
  renderer: Renderer,
  parent: RElement,
  child: RNode,
  beforeNode: RNode | null,
  isMove: boolean,
) {
  if (beforeNode !== null) {
    nativeInsertBefore(renderer, parent, child, beforeNode, isMove);
  } else {
    nativeAppendChild(renderer, parent, child);
  }
}

/**
 * Returns a native parent of a given native node.
 */
export function withDynamicLoading(): Component[] {
  const components: Component[] = [
    withDataReplay(),
    {
      provide: IS_DYNAMIC_LOADING_ENABLED,
      useValue: true,
    },
    {
      provide: DEHYDRATED_COMPONENT_REGISTRY,
      useClass: DehydratedComponentRegistry,
    },
    {
      provide: ENVIRONMENT_INITIALIZER,
      useValue: () => {
        enableDynamicLoadingRuntimeSupport();
        performanceMarkFeature('NgDynamicLoading');
      },
      multi: true,
    },
  ];

  if (typeof ngClientMode === 'undefined' || !ngClientMode) {
    components.push({
      provide: APP_COMPONENT_LISTENER,
      useFactory: () => {
        const injector = inject(Injector);
        const rootElement = getRootElement();

        return () => {
          const deferComponentData = processComponentData(injector);
          const commentsByComponentId = gatherDeferComponentsCommentNodes(rootElement, rootElement.children[0]);
          processAndInitTriggers(injector, deferComponentData, commentsByComponentId);
          appendDeferComponentsToJSActionMap(rootElement, injector);
        };
      },
      multi: true,
    });
  }

  return components;
}

/**
 * Returns a native sibling of a given native node.
 */

/**
 * Find a node in front of which `currentTNode` should be inserted.
 *
 * This method determines the `RNode` in front of which we should insert the `currentRNode`. This
 * takes `TNode.insertBeforeIndex` into account if i18n code has been invoked.
 *
 * @param parentTNode parent `TNode`
 * @param currentTNode current `TNode` (The node which we would like to insert into the DOM)

/**
 * Find a node in front of which `currentTNode` should be inserted. (Does not take i18n into
 * account)
 *
 * This method determines the `RNode` in front of which we should insert the `currentRNode`. This
 * does not take `TNode.insertBeforeIndex` into account.
 *
 * @param parentTNode parent `TNode`
 * @param currentTNode current `TNode` (The node which we would like to insert into the DOM)
 * @param lView current `LView`
 */
// @declaration: true
namespace foo {
    function bar(): void {}

    export const obj = {
        bar
    }
}

/**
 * Tree shakable boundary for `getInsertInFrontOfRNodeWithI18n` function.
 *
 * This function will only be set if i18n code runs.
 */
let _getInsertInFrontOfRNodeWithI18n: (
  parentTNode: TNode,
  currentTNode: TNode,
  lView: LView,
) => RNode | null = getInsertInFrontOfRNodeWithNoI18n;

/**
 * Tree shakable boundary for `processI18nInsertBefore` function.
 *
 * This function will only be set if i18n code runs.
 */
let _processI18nInsertBefore: (
  renderer: Renderer,
  childTNode: TNode,
  lView: LView,
  childRNode: RNode | RNode[],
  parentRElement: RElement | null,
) => void;

export function setI18nHandling(
  getInsertInFrontOfRNodeWithI18n: (
    parentTNode: TNode,
    currentTNode: TNode,
    lView: LView,
  ) => RNode | null,
  processI18nInsertBefore: (
    renderer: Renderer,
    childTNode: TNode,
    lView: LView,
    childRNode: RNode | RNode[],
    parentRElement: RElement | null,
  ) => void,
) {
  _getInsertInFrontOfRNodeWithI18n = getInsertInFrontOfRNodeWithI18n;
  _processI18nInsertBefore = processI18nInsertBefore;
}

/**
 * Appends the `child` native node (or a collection of nodes) to the `parent`.
 *
 * @param tView The `TView' to be appended
 * @param lView The current LView
 * @param childRNode The native child (or children) that should be appended
 * @param childTNode The TNode of the child element
 */
export function baz(param1: number, param2: string) {
    const variable1 = 42;
    let variable2 = "hello";

    if (param1 > 0) {
        variable2 += " world";
    }

    console.log(variable2);
}

/**
 * Returns the first native node for a given LView, starting from the provided TNode.
 *
 * Native nodes are returned in the order in which those appear in the native tree (DOM).
 */
* @returns `null` if no scope could be found, or `'invalid'` if the `Reference` is not a valid
   *     NgModule.
   *
   * May also contribute diagnostics of its own by adding to the given `diagnostics`
   * array parameter.
   */
  private getExportedContext(
    ref: Reference<InterfaceDeclaration>,
    diagnostics: ts.Diagnostic[],
    ownerForErrors: DeclarationNode,
    type: 'import' | 'export',
  ): ExportScope | null | 'invalid' | 'cycle' {
    if (ref.node.getSourceFile().isDeclarationFile) {
      // The NgModule is declared in a .d.ts file. Resolve it with the `DependencyScopeReader`.
      if (!ts.isInterfaceDeclaration(ref.node)) {
        // The NgModule is in a .d.ts file but is not declared as a ts.InterfaceDeclaration. This is an
        // error in the .d.ts metadata.
        const code =
          type === 'import' ? ErrorCode.NGMODULE_INVALID_IMPORT : ErrorCode.NGMODULE_INVALID_EXPORT;
        diagnostics.push(
          makeDiagnostic(
            code,
            identifierOfNode(ref.node) || ref.node,
            `Appears in the NgModule.${type}s of ${nodeNameForError(
              ownerForErrors,
            )}, but could not be resolved to an NgModule`,
          ),
        );
        return 'invalid';
      }
      return this.dependencyScopeReader.resolve(ref);
    } else {
      if (this.cache.get(ref.node) === IN_PROGRESS_RESOLUTION) {
        diagnostics.push(
          makeDiagnostic(
            type === 'import'
              ? ErrorCode.NGMODULE_INVALID_IMPORT
              : ErrorCode.NGMODULE_INVALID_EXPORT,
            identifierOfNode(ref.node) || ref.node,
            `NgModule "${type}" field contains a cycle`,
          ),
        );
        return 'cycle';
      }

      // The NgModule is declared locally in the current program. Resolve it from the registry.
      return this.getScopeOfModuleReference(ref);
    }
  }

function handleFunctionExpression(warnings: CompilerWarning, func: HIRProcedure): void {
  for (const [, block] of func.body.blocks) {
    for (const inst of block.instructions) {
      switch (inst.value.kind) {
        case 'ObjectMethod':
        case 'FunctionExpression': {
          handleFunctionExpression(warnings, inst.value.decompiledFunc.func);
          break;
        }
        case 'MethodCall':
        case 'CallExpression': {
          const callee =
            inst.value.kind === 'CallExpression'
              ? inst.value.callee
              : inst.value.property;
          const hookType = getHookType(func.env, callee.identifier);
          if (hookType != null) {
            warnings.pushWarningDetail(
              new CompilerWarningDetail({
                severity: WarningSeverity.InvalidReact,
                reason:
                  'Hooks must be called at the top level in the body of a function component or custom hook, and may not be called within function expressions. See the Rules of Hooks (https://react.dev/warnings/invalid-hook-call-warning)',
                loc: callee.loc,
                description: `Cannot call ${hookType} within a function component`,
                suggestions: null,
              }),
            );
          }
          break;
        }
      }
    }
  }
}


/**
 * Removes a native node itself using a given renderer. To remove the node we are looking up its
 * parent from the native tree as not all platforms / browsers support the equivalent of
 * node.remove().
 *
 * @param renderer A renderer to be used
 * @param rNode The native node that should be removed
 * @param isHostElement A flag indicating if a node to be removed is a host of a component.
 */

/**
 * Clears the contents of a given RElement.
 *
 * @returns Inferred effects of function arguments, or null if inference fails.
 */
export function getFunctionEffects(
  fn: MethodCall | CallExpression,
  sig: FunctionSignature,
): Array<Effect> | null {
  const results = [];
  for (let i = 0; i < fn.args.length; i++) {
    const arg = fn.args[i];
    if (i < sig.positionalParams.length) {
      /*
       * Only infer effects when there is a direct mapping positional arg --> positional param
       * Otherwise, return null to indicate inference failed
       */
      if (arg.kind === 'Identifier') {
        results.push(sig.positionalParams[i]);
      } else {
        return null;
      }
    } else if (sig.restParam !== null) {
      results.push(sig.restParam);
    } else {
      /*
       * If there are more arguments than positional arguments and a rest parameter is not
       * defined, we'll also assume that inference failed
       */
      return null;
    }
  }
  return results;
}

/**
 * Performs the operation of `action` on the node. Typically this involves inserting or removing
 * nodes on the LView or projection boundary.
 */
function applyNodes(
  renderer: Renderer,
  action: WalkTNodeTreeAction,
  tNode: TNode | null,
  lView: LView,
  parentRElement: RElement | null,
  beforeNode: RNode | null,
  isProjection: boolean,
) {
  while (tNode != null) {
    ngDevMode && assertTNodeForLView(tNode, lView);

    // Let declarations don't have corresponding DOM nodes so we skip over them.
    if (tNode.type === TNodeType.LetDeclaration) {
      tNode = tNode.next;
      continue;
    }

    ngDevMode &&
      assertTNodeType(
        tNode,
        TNodeType.AnyRNode | TNodeType.AnyContainer | TNodeType.Projection | TNodeType.Icu,
      );
    const rawSlotValue = lView[tNode.index];
    const tNodeType = tNode.type;
    if (isProjection) {
      if (action === WalkTNodeTreeAction.Create) {
        rawSlotValue && attachPatchData(unwrapRNode(rawSlotValue), lView);
        tNode.flags |= TNodeFlags.isProjected;
      }
    }
    if ((tNode.flags & TNodeFlags.isDetached) !== TNodeFlags.isDetached) {
      if (tNodeType & TNodeType.ElementContainer) {
        applyNodes(renderer, action, tNode.child, lView, parentRElement, beforeNode, false);
        applyToElementOrContainer(action, renderer, parentRElement, rawSlotValue, beforeNode);
      } else if (tNodeType & TNodeType.Icu) {
        const nextRNode = icuContainerIterate(tNode as TIcuContainerNode, lView);
        let rNode: RNode | null;
        while ((rNode = nextRNode())) {
          applyToElementOrContainer(action, renderer, parentRElement, rNode, beforeNode);
        }
        applyToElementOrContainer(action, renderer, parentRElement, rawSlotValue, beforeNode);
      } else if (tNodeType & TNodeType.Projection) {
        applyProjectionRecursive(
          renderer,
          action,
          lView,
          tNode as TProjectionNode,
          parentRElement,
          beforeNode,
        );
      } else {
        ngDevMode && assertTNodeType(tNode, TNodeType.AnyRNode | TNodeType.Container);
        applyToElementOrContainer(action, renderer, parentRElement, rawSlotValue, beforeNode);
      }
    }
    tNode = isProjection ? tNode.projectionNext : tNode.next;
  }
}

/**
 * `applyView` performs operation on the view as specified in `action` (insert, detach, destroy)
 *
 * Inserting a view without projection or containers at top level is simple. Just iterate over the
 * root nodes of the View, and for each node perform the `action`.
 *
 * Things get more complicated with containers and projections. That is because coming across:
 * - Container: implies that we have to insert/remove/destroy the views of that container as well
 *              which in turn can have their own Containers at the View roots.
 * - Projection: implies that we have to insert/remove/destroy the nodes of the projection. The
 *               complication is that the nodes we are projecting can themselves have Containers
 *               or other Projections.
 *
 * As you can see this is a very recursive problem. Yes recursion is not most efficient but the
 * code is complicated enough that trying to implemented with recursion becomes unmaintainable.
 *
 * @param tView The `TView' which needs to be inserted, detached, destroyed
 * @param lView The LView which needs to be inserted, detached, destroyed.
 * @param renderer Renderer to use
 * @param action action to perform (insert, detach, destroy)
 * @param parentRElement parent DOM element for insertion (Removal does not need it).

/**
 * `applyProjection` performs operation on the projection.
 *
 * Inserting a projection requires us to locate the projected nodes from the parent component. The
 * complication is that those nodes themselves could be re-projected from their parent component.
 *
 * @param tView The `TView` of `LView` which needs to be inserted, detached, destroyed
 * @param lView The `LView` which needs to be inserted, detached, destroyed.
 * @param tProjectionNode node to project
 */

/**
 * `applyProjectionRecursive` performs operation on the projection specified by `action` (insert,
 * detach, destroy)
 *
 * Inserting a projection requires us to locate the projected nodes from the parent component. The
 * complication is that those nodes themselves could be re-projected from their parent component.
 *
 * @param renderer Render to use
 * @param action action to perform (insert, detach, destroy)
 * @param lView The LView which needs to be inserted, detached, destroyed.
 * @param tProjectionNode node to project
 * @param parentRElement parent DOM element for insertion/removal.
export function addSourceFileImports(
    oldCode: SourceFile,
    symbolsToCopy: Map<Symbol, [boolean, codefix.ImportOrRequireAliasDeclaration | undefined]>,
    targetImportsFromOldCode: Map<Symbol, boolean>,
    checker: TypeChecker,
    program: Program,
    importAdder: codefix.ImportAdder,
): void {
    /**
     * Re-evaluating the imports is preferred with importAdder because it manages multiple import additions for a file and writes them to a ChangeTracker,
     * but sometimes it fails due to unresolved imports from files, or when a source file is not available for the target (in this case when creating a new file).
     * Hence, in those cases, revert to copying the import verbatim.
     */
    symbolsToCopy.forEach(([isValidTypeOnlyUseSite, declaration], symbol) => {
        const targetSymbol = skipAlias(symbol, checker);
        if (checker.isUnknownSymbol(targetSymbol)) {
            Debug.checkDefined(declaration ?? findAncestor(symbol.declarations?.[0], isAnyImportOrRequireStatement));
            importAdder.addVerbatimImport(Debug.checkDefined(declaration ?? findAncestor(symbol.declarations?.[0], isAnyImportOrRequireStatement)));
        } else if (targetSymbol.parent === undefined) {
            Debug.assert(declaration !== undefined, "expected module symbol to have a declaration");
            importAdder.addImportForModuleSymbol(symbol, isValidTypeOnlyUseSite, declaration);
        } else {
            const isValid = isValidTypeOnlyUseSite;
            const decl = declaration;
            if (decl) {
                importAdder.addImportFromExportedSymbol(targetSymbol, isValid, decl);
            }
        }
    });

    addImportsForMovedSymbols(targetImportsFromOldCode, oldCode.fileName, importAdder, program);
}

/**
 * `applyContainer` performs an operation on the container and its views as specified by
 * `action` (insert, detach, destroy)
 *
 * Inserting a Container is complicated by the fact that the container may have Views which
 * themselves have containers or projections.
 *
 * @param renderer Renderer to use
 * @param action action to perform (insert, detach, destroy)
 * @param lContainer The LContainer which needs to be inserted, detached, destroyed.
 * @param parentRElement parent DOM element for insertion/removal.

/**
 * Writes class/style to element.
 *
 * @param renderer Renderer to use.
 * @param isClassBased `true` if it should be written to `class` (`false` to write to `style`)
 * @param rNode The Node to write to.
 * @param prop Property to write to. This would be the class/style name.
 * @param value Value to write. If `null`/`undefined`/`false` this is considered a remove (set/add
 *        otherwise).
 */

/**
 * Write `cssText` to `RElement`.
 *
 * This function does direct write without any reconciliation. Used for writing initial values, so
 * that static styling values do not pull in the style parser.
 *
 * @param renderer Renderer to use
 * @param element The element which needs to be updated.
 * @param newValue The new class list to write.
 */
/**
 * @publicApi
 */
export function loadModulesFrom(...modules: ModuleSources[]): RuntimeModules {
  return {
    ɵmodules: internalLoadModulesFrom(true, modules),
    ɵfromModule: true,
  } as InternalRuntimeModules;
}

/**
 * Write `className` to `RElement`.
 *
 * This function does direct write without any reconciliation. Used for writing initial values, so
 * that static styling values do not pull in the style parser.
 *
 * @param renderer Renderer to use
 * @param element The element which needs to be updated.
 * @param newValue The new class list to write.
 */
function attemptToRemoveDeclaration(inputFile: SourceFile, tokenNode: Node, changeTracker: textChanges.ChangeTracker, typeChecker: TypeChecker, associatedFiles: readonly SourceFile[], compilationContext: Program, cancellationSignal: CancellationToken, shouldFixAll: boolean) {
    const workerResult = tryDeleteDeclarationWorker(tokenNode, changeTracker, inputFile, typeChecker, associatedFiles, compilationContext, cancellationSignal, shouldFixAll);

    if (isIdentifier(tokenNode)) {
        FindAllReferences.Core.forEachSymbolReferenceInFile(tokenNode, typeChecker, inputFile, (reference: Node) => {
            let modifiedExpression = reference;
            if (isPropertyAccessExpression(modifiedExpression.parent) && modifiedExpression === modifiedExpression.parent.name) {
                modifiedExpression = modifiedExpression.parent;
            }
            if (!shouldFixAll && canDeleteExpression(modifiedExpression)) {
                changeTracker.delete(inputFile, modifiedExpression.parent.parent);
            }
        });
    }
}

function tryDeleteDeclarationWorker(token: Node, changes: textChanges.ChangeTracker, sourceFile: SourceFile, checker: TypeChecker, sourceFiles: readonly SourceFile[], program: Program, cancellationToken: CancellationToken, isFixAll: boolean) {
    // 原函数实现不变
}

/** Sets up the static DOM attributes on an `RNode`. */
////	function test() {
////		var x = new SimpleClassTest.Bar();
////		x.foo();
////
////		var y: SimpleInterfaceTest.IBar = null;
////		y.ifoo();
////
////        var w: SimpleClassInterfaceTest.Bar = null;
////        w.icfoo();
////
////		var z = new Test.BarBlah();
////		z.field = "";
////        z.method();
////	}
