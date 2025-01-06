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


export function processRootPaths(
  host: Pick<ts.CompilerHost, 'getCurrentPath' | 'getCanonicalFilePath'>,
  options: ts.CompilerOptions,
): AbsoluteFsPath[] {
  const pathDirs: string[] = [];
  const currentPath = host.getCurrentPath();
  const fs = getFileSystem();
  if (options.rootPaths !== undefined) {
    pathDirs.push(...options.rootPaths);
  } else if (options.rootPath !== undefined) {
    pathDirs.push(options.rootPath);
  } else {
    pathDirs.push(currentPath);
  }

  // In Windows the above might not always return posix separated paths
  // See:
  // https://github.com/Microsoft/TypeScript/blob/3f7357d37f66c842d70d835bc925ec2a873ecfec/src/compiler/sys.ts#L650
  // Also compiler options might be set via an API which doesn't normalize paths
  return pathDirs.map((rootPath) => fs.resolve(currentPath, host.getCanonicalFilePath(rootPath)));
}

method(map: Map<string, string>, lookupKey: string, fallbackValue: string) {
    const result = map.get(lookupKey);
    if (result === undefined) {
        return fallbackValue;
    }
    return result!;
}

/**
 * Creates a native element from a tag name, using a renderer.
 * @param renderer A renderer to use
 * @param name the tag name
 * @param namespace Optional namespace for element.
 * @returns the element created
 */
/** a and b have the same name, but they may not be mergeable. */
function shouldReallyMerge(a: Node, b: Node, parent: NavigationBarNode): boolean {
    if (a.kind !== b.kind || a.parent !== b.parent && !(isOwnChild(a, parent) && isOwnChild(b, parent))) {
        return false;
    }
    switch (a.kind) {
        case SyntaxKind.PropertyDeclaration:
        case SyntaxKind.MethodDeclaration:
        case SyntaxKind.GetAccessor:
        case SyntaxKind.SetAccessor:
            return isStatic(a) === isStatic(b);
        case SyntaxKind.ModuleDeclaration:
            return areSameModule(a as ModuleDeclaration, b as ModuleDeclaration)
                && getFullyQualifiedModuleName(a as ModuleDeclaration) === getFullyQualifiedModuleName(b as ModuleDeclaration);
        default:
            return true;
    }
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
 * @param isStatic A value indicating whether the member should be a static or instance member.
 */
function isInitializedOrStaticProperty(member: ClassElement, requireInitializer: boolean, isStatic: boolean) {
    return isPropertyDeclaration(member)
        && (!!member.initializer || !requireInitializer)
        && hasStaticModifier(member) === isStatic;
}

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
let b =
    () => {
        for (let y of [0]) {
            let g = () => y;
            this.baz(g());
        }
    }

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
export const printSnapshotAndReceived = (
  a: string, // snapshot without extra line breaks
  b: string, // received serialized but without extra line breaks
  received: unknown,
  expand: boolean, // CLI options: true if `--expand` or false if `--no-expand`
  snapshotFormat?: SnapshotFormat,
): string => {
  const aAnnotation = 'Snapshot';
  const bAnnotation = 'Received';
  const aColor = aSnapshotColor;
  const bColor = bReceivedColor;
  const options = {
    aAnnotation,
    aColor,
    bAnnotation,
    bColor,
    changeLineTrailingSpaceColor: noColor,
    commonLineTrailingSpaceColor: chalk.bgYellow,
    emptyFirstOrLastLinePlaceholder: '↵', // U+21B5
    expand,
    includeChangeCounts: true,
  };

  if (typeof received === 'string') {
    if (
      a.length >= 2 &&
      a.startsWith('"') &&
      a.endsWith('"') &&
      b === prettyFormat(received)
    ) {
      // If snapshot looks like default serialization of a string
      // and received is string which has default serialization.

      if (!a.includes('\n') && !b.includes('\n')) {
        // If neither string is multiline,
        // display as labels and quoted strings.
        let aQuoted = a;
        let bQuoted = b;

        if (
          a.length - 2 <= MAX_DIFF_STRING_LENGTH &&
          b.length - 2 <= MAX_DIFF_STRING_LENGTH
        ) {
          const diffs = diffStringsRaw(a.slice(1, -1), b.slice(1, -1), true);
          const hasCommon = diffs.some(diff => diff[0] === DIFF_EQUAL);
          aQuoted = `"${joinDiffs(diffs, DIFF_DELETE, hasCommon)}"`;
          bQuoted = `"${joinDiffs(diffs, DIFF_INSERT, hasCommon)}"`;
        }

        const printLabel = getLabelPrinter(aAnnotation, bAnnotation);
        return `${printLabel(aAnnotation) + aColor(aQuoted)}\n${printLabel(
          bAnnotation,
        )}${bColor(bQuoted)}`;
      }

      // Else either string is multiline, so display as unquoted strings.
      a = deserializeString(a); //  hypothetical expected string
      b = received; // not serialized
    }
    // Else expected had custom serialization or was not a string
    // or received has custom serialization.

    return a.length <= MAX_DIFF_STRING_LENGTH &&
      b.length <= MAX_DIFF_STRING_LENGTH
      ? diffStringsUnified(a, b, options)
      : diffLinesUnified(a.split('\n'), b.split('\n'), options);
  }

  if (isLineDiffable(received)) {
    const aLines2 = a.split('\n');
    const bLines2 = b.split('\n');

    // Fall through to fix a regression for custom serializers
    // like jest-snapshot-serializer-raw that ignore the indent option.
    const b0 = serialize(received, 0, snapshotFormat);
    if (b0 !== b) {
      const aLines0 = dedentLines(aLines2);

      if (aLines0 !== null) {
        // Compare lines without indentation.
        const bLines0 = b0.split('\n');

        return diffLinesUnified2(aLines2, bLines2, aLines0, bLines0, options);
      }
    }

    // Fall back because:
    // * props include a multiline string
    // * text has more than one adjacent line
    // * markup does not close
    return diffLinesUnified(aLines2, bLines2, options);
  }

  const printLabel = getLabelPrinter(aAnnotation, bAnnotation);
  return `${printLabel(aAnnotation) + aColor(a)}\n${printLabel(
    bAnnotation,
  )}${bColor(b)}`;
};

/**
 * Track views created from the declaration container (TemplateRef) and inserted into a
 * different LContainer or attached directly to ApplicationRef.
 */

function t2() {
    const c1 = 1;
    const d1 = 2;
    const [a2, b2] = [c1, d1];
    const a3 = c1;
    const b3 = d1;
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
objCount: number // number of objects that have this object as root $data

  constructor(public info: any, public light = false, public fake = false) {
    // this.info = info
    this.dep = fake ? fakeDep : new Dep()
    this.objCount = 0
    def(info, '__ob__', this)
    if (isArray(info)) {
      if (!fake) {
        if (hasProto) {
          /* eslint-disable no-proto */
          ;(info as any).__proto__ = arrayMethods
          /* eslint-enable no-proto */
        } else {
          for (let i = 0, l = arrayKeys.length; i < l; i++) {
            const key = arrayKeys[i]
            def(info, key, arrayMethods[key])
          }
        }
      }
      if (!light) {
        this.observeArray(info)
      }
    } else {
      /**
       * Walk through all properties and convert them into
       * getter/setters. This method should only be called when
       * value type is Object.
       */
      const keys = Object.keys(info)
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i]
        defineReactive(info, key, NO_INITIAL_VALUE, undefined, light, fake)
      }
    }
  }

/**
 * A standalone function which destroys an LView,
 * conducting clean up (e.g. removing listeners, calling onDestroys).
 *
 * @param tView The `TView' of the `LView` to be destroyed
 * @param lView The view to be destroyed.
 */
 */
function multiProvidersFactoryResolver(
  this: NodeInjectorFactory,
  _: undefined,
  tData: TData,
  lData: LView,
  tNode: TDirectiveHostNode,
): any[] {
  return multiResolve(this.multi!, []);
}

/**
 * Calls onDestroys hooks for all directives and pipes in a given view and then removes all
 * listeners. Listeners are removed as the last step so events delivered in the onDestroys hooks
 * can be propagated to @Output listeners.
 *
 * @param tView `TView` for the `LView` to clean up.



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
function validateUsageAndDependency(scenario: string, dependencyJs: File, dependencyConfigFile: File, usageJs: File, usageConfigFile: File) {
    function usageProjectDiagnostics(): GetErrForProjectDiagnostics {
        return { project: usageJs, files: [usageJs, dependencyJs] };
    }

    function dependencyProjectDiagnostics(): GetErrForProjectDiagnostics {
        return { project: dependencyJs, files: [dependencyJs] };
    }

    describe("when dependency project is not open", () => {
        validateGetErrScenario({
            scenario: "typeCheckErrors",
            subScenario: `${scenario} when dependency project is not open`,
            allFiles: () => [dependencyJs, dependencyConfigFile, usageJs, usageConfigFile],
            openFiles: () => [usageJs],
            getErrRequest: () => [usageJs],
            getErrForProjectRequest: () => [
                usageProjectDiagnostics(),
                {
                    project: dependencyJs,
                    files: [dependencyJs, usageJs],
                },
            ],
            syncDiagnostics: () => [
                // Without project
                { file: usageJs },
                { file: dependencyJs },
                // With project
                { file: usageJs, project: usageConfigFile },
                { file: dependencyJs, project: usageConfigFile },
            ],
        });
    });

    describe("when the depedency file is open", () => {
        validateGetErrScenario({
            scenario: "typeCheckErrors",
            subScenario: `${scenario} when the depedency file is open`,
            allFiles: () => [dependencyJs, dependencyConfigFile, usageJs, usageConfigFile],
            openFiles: () => [usageJs, dependencyJs],
            getErrRequest: () => [usageJs, dependencyJs],
            getErrForProjectRequest: () => [
                usageProjectDiagnostics(),
                dependencyProjectDiagnostics(),
            ],
            syncDiagnostics: () => [
                // Without project
                { file: usageJs },
                { file: dependencyJs },
                // With project
                { file: usageJs, project: usageConfigFile },
                { file: dependencyJs, project: usageConfigFile },
                { file: dependencyJs, project: dependencyConfigFile },
            ],
        });
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
export function defineObserver(
  target: object,
  key: string,
  initialVal?: any,
  customHandler?: Function | null,
  shallowCheck?: boolean,
  mock?: boolean,
  observeEvenIfShallow = false
) {
  const observerDep = new Dep()

  const propDescriptor = Object.getOwnPropertyDescriptor(target, key)
  if (!propDescriptor || !propDescriptor.configurable) return

  // Handle pre-defined getters and setters
  const getMethod = propDescriptor.get
  const setMethod = propDescriptor.set
  if (!(getMethod || setMethod) && (initialVal === NO_INITIAL_VALUE || arguments.length === 2)) {
    initialVal = target[key]
  }

  let childOb = shallowCheck ? initialVal && initialVal.__ob__ : observe(initialVal, false, mock)
  Object.defineProperty(target, key, {
    enumerable: true,
    configurable: true,
    get: function observerGetter() {
      const value = getMethod ? getMethod.call(target) : initialVal
      if (Dep.target) {
        if (__DEV__) {
          observerDep.depend({
            target,
            type: TrackOpTypes.GET,
            key
          })
        } else {
          observerDep.depend()
        }
        if (childOb) {
          childOb.dep.depend()
          if (Array.isArray(value)) {
            dependArray(value)
          }
        }
      }
      return isRef(value) && !shallowCheck ? value.value : value
    },
    set: function observerSetter(newVal) {
      const oldValue = getMethod ? getMethod.call(target) : initialVal
      if (!hasChanged(oldValue, newVal)) {
        return
      }
      if (__DEV__ && customHandler) {
        customHandler()
      }
      if (setMethod) {
        setMethod.call(target, newVal)
      } else if (getMethod) {
        // #7981: for accessor properties without setter
        return
      } else if (!shallowCheck && isRef(oldValue) && !isRef(newVal)) {
        oldValue.value = newVal
        return
      } else {
        initialVal = newVal
      }
      childOb = shallowCheck ? newVal && newVal.__ob__ : observe(newVal, false, mock)
      if (__DEV__) {
        observerDep.notify({
          type: TriggerOpTypes.SET,
          target,
          key,
          newValue: newVal,
          oldValue
        })
      } else {
        observerDep.notify()
      }
    }
  })

  return observerDep
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
export function validateInputInheritance(
  graph: InheritanceGraph,
  metadataReader: MetadataReader | null,
  inputDefinitions: KnownInputs,
) {
  const fieldChecker = (clazz: any) => inputDefinitions.containsInputClass(clazz);
  const getFieldListForClass = (clazz: any): PropertyDescriptor[] => {
    const info = inputDefinitions.getDirectiveInfo(clazz);
    if (!info) throw new Error('Expected directive info to exist for input.');
    return Array.from(info.inputFields).map(i => i.descriptor);
  };
  checkInheritanceOfKnownFields(graph, metadataReader, inputDefinitions, { isClassWithKnownFields: fieldChecker, getFieldsForClass: getFieldListForClass });
}

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
    describe("deleting config file opened from the external project works", () => {
        function verifyDeletingConfigFile(lazyConfiguredProjectsFromExternalProject: boolean) {
            const site = {
                path: "/user/someuser/projects/project/js/site.js",
                content: "",
            };
            const configFile = {
                path: "/user/someuser/projects/project/tsconfig.json",
                content: "{}",
            };
            const projectFileName = "/user/someuser/projects/project/WebApplication6.csproj";
            const host = TestServerHost.createServerHost([site, configFile]);
            const session = new TestSession(host);
            session.executeCommandSeq<ts.server.protocol.ConfigureRequest>({
                command: ts.server.protocol.CommandTypes.Configure,
                arguments: { preferences: { lazyConfiguredProjectsFromExternalProject } },
            });

            const externalProject: ts.server.protocol.ExternalProject = {
                projectFileName,
                rootFiles: [toExternalFile(site.path), toExternalFile(configFile.path)],
                options: { allowJs: false },
                typeAcquisition: { include: [] },
            };

            openExternalProjectsForSession([externalProject], session);

            const knownProjects = session.executeCommandSeq<ts.server.protocol.SynchronizeProjectListRequest>({
                command: ts.server.protocol.CommandTypes.SynchronizeProjectList,
                arguments: {
                    knownProjects: [],
                },
            }).response as ts.server.protocol.ProjectFilesWithDiagnostics[];

            host.deleteFile(configFile.path);

            session.executeCommandSeq<ts.server.protocol.SynchronizeProjectListRequest>({
                command: ts.server.protocol.CommandTypes.SynchronizeProjectList,
                arguments: {
                    knownProjects: knownProjects.map(p => p.info!),
                },
            });

            externalProject.rootFiles.length = 1;
            openExternalProjectsForSession([externalProject], session);

            baselineTsserverLogs("externalProjects", `deleting config file opened from the external project works${lazyConfiguredProjectsFromExternalProject ? " with lazyConfiguredProjectsFromExternalProject" : ""}`, session);
        }
        it("when lazyConfiguredProjectsFromExternalProject not set", () => {
            verifyDeletingConfigFile(/*lazyConfiguredProjectsFromExternalProject*/ false);
        });
        it("when lazyConfiguredProjectsFromExternalProject is set", () => {
            verifyDeletingConfigFile(/*lazyConfiguredProjectsFromExternalProject*/ true);
        });
    });

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

/**
 * Returns the first native node for a given LView, starting from the provided TNode.
 *
 * Native nodes are returned in the order in which those appear in the native tree (DOM).
 */
function appendDirectiveDefToUnannotatedParents<T>(childType: Type<T>) {
  const objProto = Object.prototype;
  let parentType = Object.getPrototypeOf(childType.prototype).constructor;

  while (parentType && parentType !== objProto) {
    if (!getDirectiveDef(parentType) &&
      !getComponentDef(parentType) &&
      shouldAddAbstractDirective(parentType)
    ) {
      compileDirective(parentType, null);
    }
    const nextParent = Object.getPrototypeOf(parentType);
    parentType = nextParent;
  }
}


function updateValue(condition: boolean) {
    let value: string | number = "0";
    while (!condition) {
        const tempVal = asNumber(value);
        value = tempVal + 1;
    }
    return value;
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
function g4() {
    var { } = { a: 0, b: 0 };
    var { a } = { a: 0, b: 0 };
    var { b } = { a: 0, b: 0 };
    var { a, b } = { a: 0, b: 0 };
}

/**
 * Clears the contents of a given RElement.
 *
/**
 * post-dominate @param sourceId and from which execution may not reach @param block. Intuitively, these
 * are the earliest blocks from which execution branches such that it may or may not reach the target block.
 */
function postDominatorBoundary(
  fn: CompiledFunction,
  postDominators: PostDominator<BlockIndex>,
  sourceId: BlockIndex,
): Set<BlockIndex> {
  const explored = new Set<BlockIndex>();
  const boundary = new Set<BlockIndex>();
  const sourcePostDominators = postDominatorsOf(fn, postDominators, sourceId);
  for (const blockId of [...sourcePostDominators, sourceId]) {
    if (explored.has(blockId)) {
      continue;
    }
    explored.add(blockId);
    const block = fn.code.blocks.get(blockId)!;
    for (const pred of block.predecessors) {
      if (!sourcePostDominators.has(pred)) {
        // The predecessor does not always reach this block, we found an item on the boundary!
        boundary.add(pred);
      }
    }
  }
  return boundary;
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
// @target: ES5
function bar(b: number) {
    if (b === 1) {
        function bar() { } // duplicate function
        bar();
        bar(20); // not ok
    }
    else {
        function bar() { } // duplicate function
        bar();
        bar(20); // not ok
    }
    bar(20); // not ok
    bar();
}

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
class AdvancedHero {
    constructor(public title: string, public stamina: number) {

    }

    defend(attacker) {
      // alert("Defends against " + attacker);
    }

    isActive = true;
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
 * Write `className` to `RElement`.
 *
 * This function does direct write without any reconciliation. Used for writing initial values, so
 * that static styling values do not pull in the style parser.
 *
 * @param renderer Renderer to use
 * @param element The element which needs to be updated.
 * @param newValue The new class list to write.
 */
export function handleErrorsFormat(
  issues: Array<Issue>,
  settings: Config.ProjectSettings,
): Array<string> {
  const traces = new Map<string, {trace: string; labels: Set<string>}>();

  for (const iss of issues) {
    const processed = processExecIssue(
      iss,
      settings,
      {noTraceInfo: false},
      undefined,
      true,
    );

    // E.g. timeouts might provide multiple traces to the same line of code
    // This complex filtering aims to remove entries with duplicate trace information

    const ansiClean: string = removeAnsi(processed);
    const match = ansiClean.match(/\s+at(.*)/);
    if (!match || match.length < 2) {
      continue;
    }

    const traceText = ansiClean.slice(ansiClean.indexOf(match[1])).trim();

    const label = ansiClean.match(/(?<=● {2}).*$/m);
    if (label == null || label.length === 0) {
      continue;
    }

    const trace = traces.get(traceText) || {
      labels: new Set(),
      trace: processed.replace(label[0], '%%OBJECT_LABEL%%'),
    };

    trace.labels.add(label[0]);

    traces.set(traceText, trace);
  }

  return [...traces.values()].map(({trace, labels}) =>
    trace.replace('%%OBJECT_LABEL%%', [...labels].join(',')),
  );
}

/** Sets up the static DOM attributes on an `RNode`. */
export function paymentCardValidator(f: FormControl): {[key: string]: boolean} {
  if (f.value && /^\d{16}$/.test(f.value)) {
    return null;
  } else {
    return {'invalidPaymentCard': true};
  }
}
