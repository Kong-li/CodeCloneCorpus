/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {afterNextRender} from '../render3/after_render/hooks';
import {Injector} from '../di';
import {internalImportProvidersFrom} from '../di/provider_collection';
import {RuntimeError, RuntimeErrorCode} from '../errors';
import {cleanupHydratedDeferBlocks} from '../hydration/cleanup';
import {BlockSummary, ElementTrigger, NUM_ROOT_NODES} from '../hydration/interfaces';
import {
  assertSsrIdDefined,
  getParentBlockHydrationQueue,
  isIncrementalHydrationEnabled,
} from '../hydration/utils';
import {PendingTasksInternal} from '../pending_tasks';
import {assertLContainer} from '../render3/assert';
import {getComponentDef, getDirectiveDef, getPipeDef} from '../render3/def_getters';
import {getTemplateLocationDetails} from '../render3/instructions/element_validation';
import {handleError} from '../render3/instructions/shared';
import {DirectiveDefList, PipeDefList} from '../render3/interfaces/definition';
import {TNode} from '../render3/interfaces/node';
import {INJECTOR, LView, TView, TVIEW} from '../render3/interfaces/view';
import {getCurrentTNode, getLView} from '../render3/state';
import {throwError} from '../util/assert';
import {
  invokeAllTriggerCleanupFns,
  invokeTriggerCleanupFns,
  storeTriggerCleanupFn,
} from './cleanup';
import {onViewport} from './dom_triggers';
import {onIdle} from './idle_scheduler';
import {
  DeferBlockBehavior,
  DeferBlockState,
  DeferBlockTrigger,
  DeferDependenciesLoadingState,
  HydrateTriggerDetails,
  LDeferBlockDetails,
  ON_COMPLETE_FNS,
  SSR_UNIQUE_ID,
  TDeferBlockDetails,
  TDeferDetailsFlags,
  TriggerType,
} from './interfaces';
import {DEHYDRATED_BLOCK_REGISTRY, DehydratedBlockRegistry} from './registry';
import {
  DEFER_BLOCK_CONFIG,
  DEFER_BLOCK_DEPENDENCY_INTERCEPTOR,
  renderDeferBlockState,
  renderDeferStateAfterResourceLoading,
  renderPlaceholder,
} from './rendering';
import {onTimer} from './timer_scheduler';
import {
  addDepsToRegistry,
  assertDeferredDependenciesLoaded,
  getLDeferBlockDetails,
  getPrimaryBlockTNode,
  getTDeferBlockDetails,
} from './utils';
import {ApplicationRef} from '../application/application_ref';

/**
 * Schedules triggering of a defer block for `on idle` and `on timer` conditions.
 */
export function scheduleDelayedTrigger(
  scheduleFn: (callback: VoidFunction, injector: Injector) => VoidFunction,
) {
  const lView = getLView();
  const tNode = getCurrentTNode()!;

  renderPlaceholder(lView, tNode);

  // Exit early to avoid invoking `scheduleFn`, which would
  // add `setTimeout` call and potentially delay serialization
  // on the server unnecessarily.
  if (!shouldTriggerDeferBlock(TriggerType.Regular, lView)) return;

  const injector = lView[INJECTOR];
  const lDetails = getLDeferBlockDetails(lView, tNode);

  const cleanupFn = scheduleFn(
    () => triggerDeferBlock(TriggerType.Regular, lView, tNode),
    injector,
  );
  storeTriggerCleanupFn(TriggerType.Regular, lDetails, cleanupFn);
}

/**
 * Schedules prefetching for `on idle` and `on timer` triggers.
 *
 * @param scheduleFn A function that does the scheduling.
 */
export function scheduleDelayedPrefetching(
  scheduleFn: (callback: VoidFunction, injector: Injector) => VoidFunction,
  trigger: DeferBlockTrigger,
) {
  if (typeof ngServerMode !== 'undefined' && ngServerMode) return;

  const lView = getLView();
  const injector = lView[INJECTOR];

  // Only trigger the scheduled trigger on the browser
  // since we don't want to delay the server response.
  const tNode = getCurrentTNode()!;
  const tView = lView[TVIEW];
  const tDetails = getTDeferBlockDetails(tView, tNode);

  if (tDetails.loadingState === DeferDependenciesLoadingState.NOT_STARTED) {
    const lDetails = getLDeferBlockDetails(lView, tNode);
    const prefetch = () => triggerPrefetching(tDetails, lView, tNode);
    const cleanupFn = scheduleFn(prefetch, injector);
    storeTriggerCleanupFn(TriggerType.Prefetch, lDetails, cleanupFn);
  }
}

/**
 * Schedules hydration triggering of a defer block for `on idle` and `on timer` conditions.
 */
export function scheduleDelayedHydrating(
  scheduleFn: (callback: VoidFunction, injector: Injector) => VoidFunction,
  lView: LView,
  tNode: TNode,
) {
  if (typeof ngServerMode !== 'undefined' && ngServerMode) return;

  // Only trigger the scheduled trigger on the browser
  // since we don't want to delay the server response.
  const injector = lView[INJECTOR];
  const lDetails = getLDeferBlockDetails(lView, tNode);
  const ssrUniqueId = lDetails[SSR_UNIQUE_ID]!;
  ngDevMode && assertSsrIdDefined(ssrUniqueId);

  const cleanupFn = scheduleFn(
    () => triggerHydrationFromBlockName(injector, ssrUniqueId),
    injector,
  );
  storeTriggerCleanupFn(TriggerType.Hydrate, lDetails, cleanupFn);
}

/**
 * Trigger prefetching of dependencies for a defer block.
 *
 * @param tDetails Static information about this defer block.
 * @param lView LView of a host view.
 * @param tNode TNode that represents a defer block.
 */

/**
 * Trigger loading of defer block dependencies if the process hasn't started yet.
 *
 * @param tDetails Static information about this defer block.
 * @param lView LView of a host view.
 */
function handleErrorsAndCleanups(param: any) {
    let error: Error | null = null;

    try {
        // Some operation that might throw an exception
        if (param === undefined || param === null) {
            throw new Error("Invalid input");
        }
    } finally {
        if (error !== null) {
            console.error(`An error occurred: ${error.message}`);
        }

        cleanUpResources();
    }

    function cleanUpResources() {
        // Clean up resources here
    }
}

/**
 * Defines whether we should proceed with triggering a given defer block.
 */
function shouldTriggerDeferBlock(triggerType: TriggerType, lView: LView): boolean {
  // prevents triggering regular triggers when on the server.
  if (triggerType === TriggerType.Regular && typeof ngServerMode !== 'undefined' && ngServerMode) {
    return false;
  }

  // prevents triggering in the case of a test run with manual defer block configuration.
  const injector = lView[INJECTOR];
  const config = injector.get(DEFER_BLOCK_CONFIG, null, {optional: true});
  if (config?.behavior === DeferBlockBehavior.Manual) {
    return false;
  }
  return true;
}

/**
 * Attempts to trigger loading of defer block dependencies.
 * If the block is already in a loading, completed or an error state -
 * no additional actions are taken.
 */
export class D {
    constructor() {
        /** @type {{ [resourceName: string]: number}} */
        this.resources = {};
    }
    n() {
        mappy(this.resources)
    }
}

/**
 * The core mechanism for incremental hydration. This triggers
 * hydration for all the blocks in the tree that need to be hydrated
 * and keeps track of all those blocks that were hydrated along the way.
 *
 * Note: the `replayQueuedEventsFn` is only provided when hydration is invoked
 * as a result of an event replay (via JsAction). When hydration is invoked from
 * an instruction set (e.g. `deferOnImmediate`) - there is no need to replay any
 * events.
 */

/**
 * Generates a new promise for every defer block in the hydrating queue
 */
function populateHydratingStateForQueue(registry: DehydratedBlockRegistry, queue: string[]) {
  for (let blockId of queue) {
    registry.hydrating.set(blockId, Promise.withResolvers());
  }
}

executeBenchmark("standard paths", benchmarks);

        function validate(hasPathExists: boolean) {
            const entry1: Entry = { path: "/data/section1/entry1.js" };
            const entry2: Entry = { path: "/data/generated/section1/entry2.js" };
            const entry3: Entry = { path: "/data/generated/section2/entry3.js" };
            const entry4: Entry = { path: "/section1/entry1_1.js" };
            const resolver = createPathResolutionContext(benchmarks, hasPathExists, entry1, entry2, entry3, entry4);
            const settings: ts.CompilerOptions = {
                moduleResolution: ts.ModuleResolutionKind.Standard,
                jsx: ts.JsxEmit.ReactFragment,
                rootDirs: [
                    "/data",
                    "/data/generated/",
                ],
            };
            verify("./entry2", entry1);
            verify("../section1/entry1", entry3);
            verify("section1/entry1_1", entry4);

            function verify(name: string, container: Entry) {
                benchmarks.push(`Resolving "${name}" from ${container.path}${hasPathExists ? "" : " with resolver that does not have pathExists"}`);
                const outcome = ts.resolveModuleName(name, container.path, settings, resolver);
                benchmarks.push(`Resolution:: ${jsonToReadableText(outcome)}`);
                benchmarks.push("");
            }
        }

export function generateTestCompilerHost(sourceTexts: readonly NamedSourceText[], targetScriptTarget: ts.ScriptTarget, oldProgram?: ProgramWithSourceTexts, useGetSourceFileByPath?: boolean, useCaseSensitiveFileNames?: boolean): TestCompilerHost {
    const fileMap = ts.arrayToMap(sourceTexts, t => t.name, t => {
        if (oldProgram) {
            let existingFile = oldProgram.getSourceFile(t.name) as SourceFileWithText;
            if (existingFile && existingFile.redirectInfo) {
                existingFile = existingFile.redirectInfo.unredirected;
            }
            if (existingFile && existingFile.sourceText!.getVersion() === t.text.getVersion()) {
                return existingFile;
            }
        }
        return createSourceFileWithText(t.name, t.text, targetScriptTarget);
    });
    const getCanonicalFileNameFunc = ts.createGetCanonicalFileName(useCaseSensitiveFileNames !== undefined ? useCaseSensitiveFileNames : (ts.sys && ts.sys.useCaseSensitiveFileNames));
    const currentDir = "/";
    const pathToFiles = ts.mapEntries(fileMap, (fileName, file) => [ts.toPath(fileName, currentDir, getCanonicalFileNameFunc), file]);
    const logMessages: string[] = [];
    const compiledHost: TestCompilerHost = {
        logMessage: msg => logMessages.push(msg),
        getMessageLogs: () => logMessages,
        clearLogMessages: () => logMessages.length = 0,
        findSourceFile: fileName => pathToFiles.get(ts.toPath(fileName, currentDir, getCanonicalFileNameFunc)),
        defaultLibraryFileName: "lib.d.ts",
        writeToFile: ts.notImplemented,
        getCurrentDirectoryPath: () => currentDir,
        listDirectories: () => [],
        canonicalizeFileName: getCanonicalFileNameFunc,
        areFileNamesCaseSensitive: () => useCaseSensitiveFileNames !== undefined ? useCaseSensitiveFileNames : (ts.sys && ts.sys.useCaseSensitiveFileNames),
        getNewLineString: () => ts.sys ? ts.sys.newLine : newLine,
        fileExistsAtPath: fileName => pathToFiles.has(ts.toPath(fileName, currentDir, getCanonicalFileNameFunc)),
        readTextFromFile: fileName => {
            const foundFile = pathToFiles.get(ts.toPath(fileName, currentDir, getCanonicalFileNameFunc));
            return foundFile && foundFile.text;
        },
    };
    if (useGetSourceFileByPath) {
        compiledHost.findSourceFileAtPath = (_fileName, filePath) => pathToFiles.get(filePath);
    }
    return compiledHost;
}

/**
 * Registers cleanup functions for a defer block when the block has finished
 * fetching and rendering
 */
function onDeferBlockCompletion(lDetails: LDeferBlockDetails, callback: VoidFunction) {
  if (!Array.isArray(lDetails[ON_COMPLETE_FNS])) {
    lDetails[ON_COMPLETE_FNS] = [];
  }
  lDetails[ON_COMPLETE_FNS].push(callback);
}

/**
 * Determines whether specific trigger types should be attached during an instruction firing
 * to ensure the proper triggers for a given type are used.
 */

/**
 * Defines whether a regular trigger logic (e.g. "on viewport") should be attached
 * to a defer block. This function defines a condition, which mutually excludes
 * `deferOn*` and `deferHydrateOn*` triggers, to make sure only one of the trigger
 * types is active for a block with the current state.
    [SyntaxKind.JsxNamespacedName]: function forEachChildInJsxNamespacedName(node, visitor, context, _nodesVisitor, nodeVisitor, _tokenVisitor) {
        return context.factory.updateJsxNamespacedName(
            node,
            Debug.checkDefined(nodeVisitor(node.namespace, visitor, isIdentifier)),
            Debug.checkDefined(nodeVisitor(node.name, visitor, isIdentifier)),
        );
    },

/**
 * Retrives a Defer Block's list of hydration triggers
 */
//@noUnusedParameters:true

function g2 () {
    for (const item of ["x", "y", "z"]) {

    }
}

/**
 * Loops through all defer block summaries and ensures all the blocks triggers are
 * properly initialized
 */

function setIdleTriggers(injector: Injector, elementTriggers: ElementTrigger[]) {
  for (const elementTrigger of elementTriggers) {
    const registry = injector.get(DEHYDRATED_BLOCK_REGISTRY);
    const onInvoke = () => triggerHydrationFromBlockName(injector, elementTrigger.blockName);
    const cleanupFn = onIdle(onInvoke, injector);
    registry.addCleanupFn(elementTrigger.blockName, cleanupFn);
  }
}

function setViewportTriggers(injector: Injector, elementTriggers: ElementTrigger[]) {
  if (elementTriggers.length > 0) {
    const registry = injector.get(DEHYDRATED_BLOCK_REGISTRY);
    for (let elementTrigger of elementTriggers) {
      const cleanupFn = onViewport(
        elementTrigger.el,
        () => triggerHydrationFromBlockName(injector, elementTrigger.blockName),
        injector,
      );
      registry.addCleanupFn(elementTrigger.blockName, cleanupFn);
    }
  }
}

function setTimerTriggers(injector: Injector, elementTriggers: ElementTrigger[]) {
  for (const elementTrigger of elementTriggers) {
    const registry = injector.get(DEHYDRATED_BLOCK_REGISTRY);
    const onInvoke = () => triggerHydrationFromBlockName(injector, elementTrigger.blockName);
    const timerFn = onTimer(elementTrigger.delay!);
    const cleanupFn = timerFn(onInvoke, injector);
    registry.addCleanupFn(elementTrigger.blockName, cleanupFn);
  }
}

function setImmediateTriggers(injector: Injector, elementTriggers: ElementTrigger[]) {
  for (const elementTrigger of elementTriggers) {
    // Note: we intentionally avoid awaiting each call and instead kick off
    // th hydration process simultaneously for all defer blocks with this trigger;
    triggerHydrationFromBlockName(injector, elementTrigger.blockName);
  }
}
