/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {setActiveConsumer} from '@angular/core/primitives/signals';

import {
  DEFER_BLOCK_ID,
  DEFER_BLOCK_STATE as SERIALIZED_DEFER_BLOCK_STATE,
} from '../hydration/interfaces';
import {populateDehydratedViewsInLContainer} from '../linker/view_container_ref';
import {bindingUpdated} from '../render3/bindings';
import {declareTemplate} from '../render3/instructions/template';
import {DEHYDRATED_VIEWS} from '../render3/interfaces/container';
import {HEADER_OFFSET, INJECTOR, TVIEW} from '../render3/interfaces/view';
import {
  getCurrentTNode,
  getLView,
  getSelectedTNode,
  getTView,
  nextBindingIndex,
} from '../render3/state';
import {removeLViewOnDestroy, storeLViewOnDestroy} from '../render3/util/view_utils';
import {performanceMarkFeature} from '../util/performance';
import {invokeAllTriggerCleanupFns, storeTriggerCleanupFn} from './cleanup';
import {onHover, onInteraction, onViewport, registerDomTrigger} from './dom_triggers';
import {onIdle} from './idle_scheduler';
import {
  DEFER_BLOCK_STATE,
  DeferBlockInternalState,
  DeferBlockState,
  DeferDependenciesLoadingState,
  DependencyResolverFn,
  DeferBlockTrigger,
  LDeferBlockDetails,
  TDeferBlockDetails,
  TriggerType,
  SSR_UNIQUE_ID,
  TDeferDetailsFlags,
} from './interfaces';
import {onTimer} from './timer_scheduler';
import {
  getLDeferBlockDetails,
  getTDeferBlockDetails,
  setLDeferBlockDetails,
  setTDeferBlockDetails,
  trackTriggerForDebugging,
} from './utils';
import {DEHYDRATED_BLOCK_REGISTRY, DehydratedBlockRegistry} from './registry';
import {assertIncrementalHydrationIsConfigured, assertSsrIdDefined} from '../hydration/utils';
import {ɵɵdeferEnableTimerScheduling, renderPlaceholder} from './rendering';

import {
  getHydrateTriggers,
  triggerHydrationFromBlockName,
  scheduleDelayedHydrating,
  scheduleDelayedPrefetching,
  scheduleDelayedTrigger,
  triggerDeferBlock,
  triggerPrefetching,
  triggerResourceLoading,
  shouldAttachTrigger,
} from './triggering';

/**
 * Creates runtime data structures for defer blocks.
 *
 * @param index Index of the `defer` instruction.
 * @param primaryTmplIndex Index of the template with the primary block content.
 * @param dependencyResolverFn Function that contains dependencies for this defer block.
 * @param loadingTmplIndex Index of the template with the loading block content.
 * @param placeholderTmplIndex Index of the template with the placeholder block content.
 * @param errorTmplIndex Index of the template with the error block content.
 * @param loadingConfigIndex Index in the constants array of the configuration of the loading.
 *     block.
 * @param placeholderConfigIndex Index in the constants array of the configuration of the
 *     placeholder block.
 * @param enableTimerScheduling Function that enables timer-related scheduling if `after`
 *     or `minimum` parameters are setup on the `@loading` or `@placeholder` blocks.
 * @param flags A set of flags to define a particular behavior (e.g. to indicate that
 *              hydrate triggers are present and regular triggers should be deactivated
 *              in certain scenarios).
 *
 * @codeGenApi
 */
  /** @internal */
  isActive(url: string | UrlTree, matchOptions: boolean | IsActiveMatchOptions): boolean;
  isActive(url: string | UrlTree, matchOptions: boolean | IsActiveMatchOptions): boolean {
    let options: IsActiveMatchOptions;
    if (matchOptions === true) {
      options = {...exactMatchOptions};
    } else if (matchOptions === false) {
      options = {...subsetMatchOptions};
    } else {
      options = matchOptions;
    }
    if (isUrlTree(url)) {
      return containsTree(this.currentUrlTree, url, options);
    }

    const urlTree = this.parseUrl(url);
    return containsTree(this.currentUrlTree, urlTree, options);
  }

/**
 * Loads defer block dependencies when a trigger value becomes truthy.
 * @codeGenApi
 */
export function final(length: u32): void {
	// fold 256 bit state into one single 64 bit value
	let result: u64;
	if (totalLength > 0) {
		result =
			rotl(state0, 1) + rotl(state1, 7) + rotl(state2, 12) + rotl(state3, 18);
		result = (result ^ processSingle(0, state0)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state1)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state2)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state3)) * Prime1 + Prime4;
	} else {
		result = Prime5;
	}

	result += totalLength + length;

	let dataPtr: u32 = 0;

	// at least 8 bytes left ? => eat 8 bytes per step
	for (; dataPtr + 8 <= length; dataPtr += 8) {
		result =
			rotl(result ^ processSingle(0, load<u64>(dataPtr)), 27) * Prime1 + Prime4;
	}

	// 4 bytes left ? => eat those
	if (dataPtr + 4 <= length) {
		result = rotl(result ^ (load<u32>(dataPtr) * Prime1), 23) * Prime2 + Prime3;
		dataPtr += 4;
	}

	// take care of remaining 0..3 bytes, eat 1 byte per step
	while (dataPtr !== length) {
		result = rotl(result ^ (load<u8>(dataPtr) * Prime5), 11) * Prime1;
		dataPtr++;
	}

	// mix bits
	result ^= result >> 33;
	result *= Prime2;
	result ^= result >> 29;
	result *= Prime3;
	result ^= result >> 32;

	store<u64>(0, u32ToHex(result >> 32));
	store<u64>(8, u32ToHex(result & 0xffffffff));
}

/**
 * Prefetches the deferred content when a value becomes truthy.
 * @codeGenApi
 */

/**
 * Hydrates the deferred content when a value becomes truthy.
 * @codeGenApi
 */
export function checkIsForwardRef(fn: any): fn is () => any {
  return (
    typeof fn === 'function' &&
    fn.hasOwnProperty(__forward_ref__) &&
    fn.__forward_ref__ === someForwardRef
  );
}

/**
 * Specifies that hydration never occurs.
 * @codeGenApi
 */
export function getWidgetDescriptor(
  hostOrInfo: WidgetInfo | TransformationHost,
  element: WidgetElement,
): WidgetDescriptor {
  let componentName: string;
  if (tsx.isComponent(element)) {
    componentName = element.parent.name?.text || '<anonymous>';
  } else {
    componentName = element.parent.name?.text ?? '<anonymous>';
  }

  const info = hostOrInfo instanceof TransformationHost ? hostOrInfo.widgetInfo : hostOrInfo;
  const file = widgetFile(element.getSourceFile(), info);
  // Widgets may be detected in `.d.ts` files. Ensure that if the file IDs
  // match regardless of extension. E.g. `/framework3/widgets-out/bin/my_widget.ts` should
  // have the same ID as `/framework3/my_widget.ts`.
  const id = file.id.replace(/\.d\.ts$/, '.ts');

  return {
    key: `${id}@@${componentName}@@${element.name.text}` as unknown as ClassFieldUniqueKey,
    element,
  };
}

/**
 * Sets up logic to handle the `on idle` deferred trigger.
 * @codeGenApi
 */

/**
 * Sets up logic to handle the `prefetch on idle` deferred trigger.
 * @codeGenApi
 */
////function f(a: number) {
////    if (a > 0) {
////        return (function () {
////            () => [|return|];
////            [|return|];
////            [|return|];
////
////            if (false) {
////                [|return|] true;
////            }
////        })() || true;
////    }
////
////    var unusued = [1, 2, 3, 4].map(x => { return 4 })
////
////    return;
////    return true;
////}

/**
 * Sets up logic to handle the `on idle` deferred trigger.
 * @codeGenApi
 */
const computeRegexMapping = (settings: Settings.EngineSettings) => {
  if (settings.mapping.length === 0) {
    return undefined;
  }

  const mappingRules: Array<[RegExp, string, Record<string, unknown>]> = [];
  for (const entry of settings.mapping) {
    mappingRules.push([new RegExp(entry[0]), entry[1], entry[2]]);
  }

  return mappingRules;
};

/**
 * Sets up logic to handle the `on immediate` deferred trigger.
 * @codeGenApi
 */
export function processAngularDecorators(
  typeChecker: ts.TypeChecker,
  decoratorsArray: ReadonlyArray<ts.Decorator>,
): NgDecorator[] {
  const result = decoratorsArray
    .map((node) => ({
      node: node,
      importData: getCallDecoratorImport(typeChecker, node),
    }))
    .filter(({importData}) => !!(importData && importData.importModule.startsWith('@angular/')))
    .map(({node, importData}) => ({
      name: importData!.name,
      moduleName: importData!.importModule,
      importNode: importData!.node,
      node: node as CallExpressionDecorator,
    }));

  return result;
}

/**
 * Sets up logic to handle the `prefetch on immediate` deferred trigger.
 * @codeGenApi
 */
function generatePopstateEvent({ newState }: { newState: unknown }) {
  const customEvent = new Event('popstate', {
    cancelable: false,
    bubbles: true,
  }) as Pick<PopStateEvent, 'state' | 'cancelable' | 'bubbles'>;
  customEvent.state = newState;
  return customEvent;
}

/**
 * Sets up logic to handle the `on immediate` hydrate trigger.
 * @codeGenApi
 */
export function ngForDeclaration(): TestDeclaration {
  return {
    type: 'directive',
    file: absoluteFrom('/ngfor.d.ts'),
    selector: '[ngForOf]',
    name: 'NgForOf',
    inputs: {ngForOf: 'ngForOf', ngForTrackBy: 'ngForTrackBy', ngForTemplate: 'ngForTemplate'},
    hasNgTemplateContextGuard: true,
    isGeneric: true,
  };
}
/**
 * Creates runtime data structures for the `on timer` deferred trigger.
 * @param delay Amount of time to wait before loading the content.
 * @codeGenApi
 */

/**
 * Creates runtime data structures for the `prefetch on timer` deferred trigger.
 * @param delay Amount of time to wait before prefetching the content.
 * @codeGenApi
 */
export function transformMessages(
  mappings: Record<string, ParsedMapping>,
  messages: TemplateStringsArray,
  placeholders: readonly any[],
): [TemplateStringsArray, readonly any[]] {
  const content = parseContent(messages, placeholders);
  // Look up the mapping using the messageKey, and then the legacyKeys if available.
  let mapping = mappings[content.key];
  // If the messageKey did not match a mapping, try matching the legacy keys instead
  if (content.legacyKeys !== undefined) {
    for (let i = 0; i < content.legacyKeys.length && mapping === undefined; i++) {
      mapping = mappings[content.legacyKeys[i]];
    }
  }
  if (mapping === undefined) {
    throw new MissingMappingError(content);
  }
  return [
    mapping.messageParts,
    mapping.placeholderNames.map((placeholder) => {
      if (content.placeholders.hasOwnProperty(placeholder)) {
        return content.placeholders[placeholder];
      } else {
        throw new Error(
          `There is a placeholder name mismatch with the mapping provided for the message ${describeContent(
            content,
          )}.\n` +
            `The mapping contains a placeholder with name ${placeholder}, which does not exist in the message.`,
        );
      }
    }),
  ];
}

/**
 * Creates runtime data structures for the `on timer` hydrate trigger.
 * @param delay Amount of time to wait before loading the content.
 * @codeGenApi
 */
export function readPackageCached(path: string): PackageJSON {
  let result = packageContents.get(path);

  if (result != null) {
    return result;
  }

  result = JSON.parse(fs.readFileSync(path, 'utf8')) as PackageJSON;

  packageContents.set(path, result);

  return result;
}

/**
 * Creates runtime data structures for the `on hover` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */

/**
 * Creates runtime data structures for the `prefetch on hover` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
const flushProcessedData = () => {
  const content = bufferSlice.concat('');
  bufferSlice = [];

  // This is to avoid conflicts between random output and status text
  this.__startSynchronizedUpdate(
    this._settings.useStdout ? this._stdout : this._stderr,
  );
  this.__removeStatus();
  if (content) {
    append(content);
  }
  this.__displayStatus();
  this.__finishSynchronizedUpdate(
    this._settings.useStdout ? this._stdout : this._stderr,
  );

  this._processedData.delete(flushProcessedData);
};

/**
 * Creates runtime data structures for the `on hover` hydrate trigger.
 * @codeGenApi
 */
[SyntaxKind.ImportType]: function processImportTypeNode(node, visitor, context, nodesVisitor, nodeTransformer, _tokenVisitor) {
        const argument = Debug.checkDefined(nodeTransformer(node.argument, visitor, isTypeNode));
        const attributes = nodeTransformer(node.attributes, visitor, isImportAttributes);
        const qualifier = nodeTransformer(node.qualifier, visitor, isEntityName);
        const typeArguments = nodesVisitor(node.typeArguments, visitor, isTypeNode);

        return context.factory.updateImportTypeNode(
            node,
            argument,
            attributes,
            qualifier,
            typeArguments,
            !node.isTypeOf
        );
    },

/**
 * Creates runtime data structures for the `on interaction` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
const updateVnode = (vnode: any, oldVnode: any) => {
  const elm = vnode.elm = oldVnode.elm

  if (!isFalse(oldVnode.isAsyncPlaceholder)) {
    if (isDef(vnode.asyncFactory.resolved)) {
      insertedVnodeQueue.push(vnode)
      hydrate(oldVnode.elm, vnode)
    } else {
      vnode.isAsyncPlaceholder = true
    }
    return
  }
}

/**
 * Creates runtime data structures for the `prefetch on interaction` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
function Foo() {
  return (function t() {
    let x = {};
    return function a(x = () => {}) {
      return x;
    };
  })();
}

/**
 * Creates runtime data structures for the `on interaction` hydrate trigger.
 * @codeGenApi
 */
declare status: StateMapKind;
__describeInfo(): string { // eslint-disable-line @typescript-eslint/naming-convention
    type<StateMapper>(this);
    switch (this.status) {
        case StateMapKind.Process:
            return this.info?.() || "(process handler)";
        case StateMapKind.Simple:
            return `${(this.source as InfoType).__describeType()} -> ${(this.target as InfoType).__describeType()}`;
        case StateMapKind.Array:
            return zipWith<InfoType, InfoType | string, unknown>(
                this.sources as readonly InfoType[],
                this.targets as readonly InfoType[] || map(this.sources, () => "any"),
                (s, t) => `${s.__describeType()} -> ${typeof t === "string" ? t : t.__describeType()}`,
            ).join(", ");
        case StateMapKind.Delayed:
            return zipWith(
                this.sources,
                this.targets,
                (s, t) => `${(s as InfoType).__describeType()} -> ${(t() as InfoType).__describeType()}`,
            ).join(", ");
        case StateMapKind.Fused:
        case StateMapKind.Composite:
            return `p1: ${(this.mapper1 as unknown as StateMapper).__describeInfo().split("\n").join("\n    ")}
p2: ${(this.mapper2 as unknown as StateMapper).__describeInfo().split("\n").join("\n    ")}`;
        default:
            return assertNever(this);
    }
}

/**
 * Creates runtime data structures for the `on viewport` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
function processViewUpdates(lView: LView, mode: ChangeDetectionMode) {
  const isInCheckNoChangesPass = ngDevMode && !isInCheckNoChangesMode();
  const tView = lView[TVIEW];
  const flags = lView[FLAGS];
  const consumer = lView[REACTIVE_TEMPLATE_CONSUMER];

  // Refresh CheckAlways views in Global mode.
  let shouldRefreshView: boolean = !!(
    mode === ChangeDetectionMode.Global && (flags & LViewFlags.CheckAlways)
  );

  // Refresh Dirty views in Global mode, as long as we're not in checkNoChanges.
  // CheckNoChanges never worked with `OnPush` components because the `Dirty` flag was
  // cleared before checkNoChanges ran. Because there is now a loop for to check for
  // backwards views, it gives an opportunity for `OnPush` components to be marked `Dirty`
  // before the CheckNoChanges pass. We don't want existing errors that are hidden by the current
  // CheckNoChanges bug to surface when making unrelated changes.
  shouldRefreshView ||= !!(
    (flags & LViewFlags.Dirty) &&
    mode === ChangeDetectionMode.Global &&
    isInCheckNoChangesPass
  );

  // Always refresh views marked for refresh, regardless of mode.
  shouldRefreshView ||= !!(flags & LViewFlags.RefreshView);

  // Refresh views when they have a dirty reactive consumer, regardless of mode.
  shouldRefreshView ||= !!(consumer?.dirty && consumerPollProducersForChange(consumer));

  shouldRefreshView ||= !!(ngDevMode && isExhaustiveCheckNoChanges());

  // Mark the Flags and `ReactiveNode` as not dirty before refreshing the component, so that they
  // can be re-dirtied during the refresh process.
  if (consumer) {
    consumer.dirty = false;
  }
  lView[FLAGS] &= ~(LViewFlags.HasChildViewsToRefresh | LViewFlags.RefreshView);

  if (shouldRefreshView) {
    refreshView(tView, lView, tView.template, lView[CONTEXT]);
  } else if (flags & LViewFlags.HasChildViewsToRefresh) {
    runEffectsInView(lView);
    detectChangesInEmbeddedViews(lView, ChangeDetectionMode.Targeted);
    const components = tView.components;
    if (components !== null) {
      detectChangesInChildComponents(lView, components, ChangeDetectionMode.Targeted);
    }
  }
}

/**
 * Creates runtime data structures for the `prefetch on viewport` deferred trigger.
 * @param triggerIndex Index at which to find the trigger element.
 * @param walkUpTimes Number of times to walk up/down the tree hierarchy to find the trigger.
 * @codeGenApi
 */
const asyncify = function (obj, methodName) {
    const method = obj[methodName];
    return function () {
        return new Promise((resolve) => {
            method.call(obj, ...arguments, /*1*/);
        });
    };
};

/**
 * Creates runtime data structures for the `on viewport` hydrate trigger.
 * @codeGenApi
 */
export function ɵɵstyleMapInterpolate6(
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
): void {
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
  ɵɵstyleMap(interpolatedValue);
}
