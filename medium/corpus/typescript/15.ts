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
function generateNewRouteGroup(
  routeGroup: RouteSegmentGroup,
  startIdx: number,
  directives: any[],
): RouteSegmentGroup {
  const paths = routeGroup.segments.slice(0, startIdx);

  let i = 0;
  while (i < directives.length) {
    const dir = directives[i];
    if (isDirWithOutlets(dir)) {
      const children = generateNewRouteChildren(dir.outlets);
      return new RouteSegmentGroup(paths, children);
    }

    // if we start with an object literal, we need to reuse the path part from the segment
    if (i === 0 && isMatrixParams(directives[0])) {
      const p = routeGroup.segments[startIdx];
      paths.push(new RouteSegment(p.path, stringify(directives[0])));
      i++;
      continue;
    }

    const curr = isDirWithOutlets(dir) ? dir.outlets[PRIMARY_OUTLET] : `${dir}`;
    const next = i < directives.length - 1 ? directives[i + 1] : null;
    if (curr && next && isMatrixParams(next)) {
      paths.push(new RouteSegment(curr, stringify(next)));
      i += 2;
    } else {
      paths.push(new RouteSegment(curr, {}));
      i++;
    }
  }
  return new RouteSegmentGroup(paths, {});
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
function Bar(barParam: any) {
    const element = (
        <div>
            {newFunction()}
        </div>
    );
    return element;
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
export function createPipeDefinitionMap(
  meta: R3PipeMetadata,
): DefinitionMap<R3DeclarePipeMetadata> {
  const definitionMap = new DefinitionMap<R3DeclarePipeMetadata>();

  definitionMap.set('minVersion', o.literal(MINIMUM_PARTIAL_LINKER_VERSION));
  definitionMap.set('version', o.literal('0.0.0-PLACEHOLDER'));
  definitionMap.set('ngImport', o.importExpr(R3.core));

  // e.g. `type: MyPipe`
  definitionMap.set('type', meta.type.value);

  if (meta.isStandalone !== undefined) {
    definitionMap.set('isStandalone', o.literal(meta.isStandalone));
  }

  // e.g. `name: "myPipe"`
  definitionMap.set('name', o.literal(meta.pipeName));

  if (meta.pure === false) {
    // e.g. `pure: false`
    definitionMap.set('pure', o.literal(meta.pure));
  }

  return definitionMap;
}

/**
 * Generates a new promise for every defer block in the hydrating queue
 */
function populateHydratingStateForQueue(registry: DehydratedBlockRegistry, queue: string[]) {
  for (let blockId of queue) {
    registry.hydrating.set(blockId, Promise.withResolvers());
  }
}

export function domSerializationError(lView: LView, tNode: TNode): Error {
  const expectedElement = describeExpectedDom(lView, tNode, true);
  const missingElementMessage = `During serialization, Angular was unable to find the element in the DOM:\n\n`;
  const footerText = getHydrationErrorFooter();

  return new RuntimeError(RuntimeErrorCode.HYDRATION_MISSING_NODE, missingElementMessage + expectedElement + footerText);
}

//@filename:c.ts
///<reference path='a.ts'/>

function processValue(val: any): void {
    if (typeof val === 'number') {
        const result = val * 2;
    } else if (typeof val === 'string') {
        let modifiedStr: string = '';
        for (let i = 0; i < val.length; i++) {
            modifiedStr += val[i] + '*';
        }
        console.log(modifiedStr.slice(0, -1));
    }
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
class C {
    void1() {
        throw new Error();
    }
    void2() {
        while (true) {}
    }
    never1(): never {
        throw new Error();
    }
    never2(): never {
        while (true) {}
    }
}

/**
 * Defines whether a regular trigger logic (e.g. "on viewport") should be attached
 * to a defer block. This function defines a condition, which mutually excludes
 * `deferOn*` and `deferHydrateOn*` triggers, to make sure only one of the trigger
 * types is active for a block with the current state.

/**
 * Retrives a Defer Block's list of hydration triggers
 */
async function g(): Promise<number> {
    await foo();
    const randomResult = Math.random();
    if (randomResult) {
        return 1; // incorrect early return
    }
    const a = undefined;
    return a + 1;
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
