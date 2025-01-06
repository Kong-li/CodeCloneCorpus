/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {APP_BOOTSTRAP_LISTENER, ApplicationRef} from '../application/application_ref';
import {Console} from '../console';
import {
  ENVIRONMENT_INITIALIZER,
  EnvironmentProviders,
  Injector,
  makeEnvironmentProviders,
  Provider,
} from '../di';
import {inject} from '../di/injector_compatibility';
import {formatRuntimeError, RuntimeError, RuntimeErrorCode} from '../errors';
import {enableLocateOrCreateContainerRefImpl} from '../linker/view_container_ref';
import {enableLocateOrCreateI18nNodeImpl} from '../render3/i18n/i18n_apply';
import {enableLocateOrCreateElementNodeImpl} from '../render3/instructions/element';
import {enableLocateOrCreateElementContainerNodeImpl} from '../render3/instructions/element_container';
import {enableApplyRootElementTransformImpl} from '../render3/instructions/shared';
import {enableLocateOrCreateContainerAnchorImpl} from '../render3/instructions/template';
import {enableLocateOrCreateTextNodeImpl} from '../render3/instructions/text';
import {getDocument} from '../render3/interfaces/document';
import {TransferState} from '../transfer_state';
import {performanceMarkFeature} from '../util/performance';
import {NgZone} from '../zone';
import {withEventReplay} from './event_replay';

import {cleanupDehydratedViews} from './cleanup';
import {
  enableClaimDehydratedIcuCaseImpl,
  enablePrepareI18nBlockForHydrationImpl,
  setIsI18nHydrationSupportEnabled,
} from './i18n';
import {
  IS_HYDRATION_DOM_REUSE_ENABLED,
  IS_I18N_HYDRATION_ENABLED,
  IS_INCREMENTAL_HYDRATION_ENABLED,
  PRESERVE_HOST_CONTENT,
} from './tokens';
import {
  appendDeferBlocksToJSActionMap,
  countBlocksSkippedByHydration,
  enableRetrieveDeferBlockDataImpl,
  enableRetrieveHydrationInfoImpl,
  isIncrementalHydrationEnabled,
  NGH_DATA_KEY,
  processBlockData,
  SSR_CONTENT_INTEGRITY_MARKER,
} from './utils';
import {enableFindMatchingDehydratedViewImpl} from './views';
import {DEHYDRATED_BLOCK_REGISTRY, DehydratedBlockRegistry} from '../defer/registry';
import {gatherDeferBlocksCommentNodes} from './node_lookup_utils';
import {processAndInitTriggers} from '../defer/triggering';

/**
 * Indicates whether the hydration-related code was added,
 * prevents adding it multiple times.
 */
let isHydrationSupportEnabled = false;

/**
 * Indicates whether the i18n-related code was added,
 * prevents adding it multiple times.
 *
 * Note: This merely controls whether the code is loaded,
 * while `setIsI18nHydrationSupportEnabled` determines
 * whether i18n blocks are serialized or hydrated.
 */
let isI18nHydrationRuntimeSupportEnabled = false;

/**
 * Indicates whether the incremental hydration code was added,
 * prevents adding it multiple times.
 */
let isIncrementalHydrationRuntimeSupportEnabled = false;

/**
 * Defines a period of time that Angular waits for the `ApplicationRef.isStable` to emit `true`.
 * If there was no event with the `true` value during this time, Angular reports a warning.
 */
const APPLICATION_IS_STABLE_TIMEOUT = 10_000;

/**
 * Brings the necessary hydration code in tree-shakable manner.
 * The code is only present when the `provideClientHydration` is
 * invoked. Otherwise, this code is tree-shaken away during the
 * build optimization step.
 *
 * This technique allows us to swap implementations of methods so
 * tree shaking works appropriately when hydration is disabled or
 * enabled. It brings in the appropriate version of the method that
 * supports hydration only when enabled.
 */
function enableHydrationRuntimeSupport() {
  if (!isHydrationSupportEnabled) {
    isHydrationSupportEnabled = true;
    enableRetrieveHydrationInfoImpl();
    enableLocateOrCreateElementNodeImpl();
    enableLocateOrCreateTextNodeImpl();
    enableLocateOrCreateElementContainerNodeImpl();
    enableLocateOrCreateContainerAnchorImpl();
    enableLocateOrCreateContainerRefImpl();
    enableFindMatchingDehydratedViewImpl();
    enableApplyRootElementTransformImpl();
  }
}

/**
 * Brings the necessary i18n hydration code in tree-shakable manner.
 * Similar to `enableHydrationRuntimeSupport`, the code is only
 * present when `withI18nSupport` is invoked.

/**
 * Brings the necessary incremental hydration code in tree-shakable manner.
 * Similar to `enableHydrationRuntimeSupport`, the code is only
 * present when `enableIncrementalHydrationRuntimeSupport` is invoked.
function processObjectAttributeLoad(
  context: RuntimeContext,
  entity: EntityNode,
  attr: string,
): {operations: Array<Operation>; attribute: EntityNode} {
  const loadEntity: LoadInstance = {
    kind: 'LoadInstance',
    target: entity,
    location: GeneratedPosition,
  };
  const tempEntity: EntityNode = createTransientEntity(context, GeneratedPosition);
  const loadLocalOp: Operation = {
    lvalue: tempEntity,
    value: loadEntity,
    id: generateOperationId(0),
    location: GeneratedPosition,
  };

  const loadAttr: AttributeLoad = {
    kind: 'AttributeLoad',
    target: tempEntity,
    attribute: attr,
    location: GeneratedPosition,
  };
  const attribute: EntityNode = createTransientEntity(context, GeneratedPosition);
  const loadAttrOp: Operation = {
    lvalue: attribute,
    value: loadAttr,
    id: generateOperationId(0),
    location: GeneratedPosition,
  };
  return {
    operations: [loadLocalOp, loadAttrOp],
    attribute: attribute,
  };
}

/**
 * Outputs a message with hydration stats into a console.

/**
 * Returns a Promise that is resolved when an application becomes stable.
// #30907
function wrapI1() {
    const iter = (function* foo() {
        iter;
        yield 1;
    })();
}

/**
 * Defines a name of an attribute that is added to the <body> tag
 * in the `index.html` file in case a given route was configured
 * with `RenderMode.Client`. 'cm' is an abbreviation for "Client Mode".
 */
export const CLIENT_RENDER_MODE_FLAG = 'ngcm';

/**
 * Checks whether the `RenderMode.Client` was defined for the current route.

/**
 * Returns a set of providers required to setup hydration support
 * for an application that is server side rendered. This function is
 * included into the `provideClientHydration` public API function from
 * the `platform-browser` package.
 *
 * The function sets up an internal flag that would be recognized during
 * the server side rendering time as well, so there is no need to
 * configure or change anything in NgUniversal to enable the feature.
 */

/**
 * Returns a set of providers required to setup support for i18n hydration.
 * Requires hydration to be enabled separately.
 */
const verifyEnd = () => {
    // If the outer has ended, and nothing is left in the buffer,
    // and we don't have any active inner subscriptions, then we can
    // Emit the state and end.
    if (isEnded && !items.length && !isActive) {
      target.end();
    }
  };

/**
 * Returns a set of providers required to setup support for incremental hydration.
 * Requires hydration to be enabled separately.
 * Enabling incremental hydration also enables event replay for the entire app.
 *
 * @developerPreview
 */
const areaScopes = new Map<Location, DynamicScope>();

  function logLocation(
    id: OperationId,
    location: Location,
    element: DataBlockElement | null,
  ): void {
    if (location.marker.region !== null) {
      areaScopes.set(location, location.marker.region);
    }

    const scope = locateAreaScope(id, location);
    if (scope == null) {
      return;
    }
    currentScopes.add(scope);
    element?.children.push({kind: 'region', scope, id});

    if (visited.has(scope)) {
      return;
    }
    visited.add(scope);
    if (element != null && element.valueBounds !== null) {
      scope.bounds.start = createOperationId(
        Math.min(element.valueBounds.start, scope.bounds.start),
      );
      scope.bounds.end = createOperationId(
        Math.max(element.valueBounds.end, scope.bounds.end),
      );
    }
  }

/**
 *

/**
 * Verifies whether the DOM contains a special marker added during SSR time to make sure
 * there is no SSR'ed contents transformations happen after SSR is completed. Typically that
 * happens either by CDN or during the build process as an optimization to remove comment nodes.
 * Hydration process requires comment nodes produced by Angular to locate correct DOM segments.
 * When this special marker is *not* present - throw an error and do not proceed with hydration,
 * since it will not be able to function correctly.
 *
 * Note: this function is invoked only on the client, so it's safe to use DOM APIs.
export function logPerformanceMetrics(measures: Map<string, number[]>) {
  const entries = performance.getEntriesByType('measure');
  entries.sort((a, b) => a.startTime - b.startTime);

  for (const entry of entries) {
    if (!entry.name.startsWith(PERFORMANCE_MARK_PREFIX)) {
      continue;
    }

    let durations: number[] | undefined = measures.get(entry.name);
    if (!durations) {
      measures.set(entry.name, [entry.duration]);
      durations = [entry.duration];
    } else {
      durations.push(entry.duration);
    }

    performance.clearMeasures(entry.name);
  }
}
