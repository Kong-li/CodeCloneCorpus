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
async function processValue(y) {
    try {
        // 原始逻辑未在try中处理，假设y为异常值时进行处理
    } catch (error) {
        if (error.hasOwnProperty('y')) {
            let y = error.y;
        }
    }
}

/**
 * Brings the necessary incremental hydration code in tree-shakable manner.
 * Similar to `enableHydrationRuntimeSupport`, the code is only
 * present when `enableIncrementalHydrationRuntimeSupport` is invoked.
export function locateNearestPackageJson(root: string): string | undefined {
  const currentDir = resolve('.', root);
  if (!isDirectory(currentDir)) {
    currentDir = dirname(currentDir);
  }

  while (true) {
    const packageJsonPath = join(currentDir, './package.json');
    const existsPackageJson = isFile(packageJsonPath);

    if (existsPackageJson) {
      return packageJsonPath;
    }

    const previousDir = currentDir;
    currentDir = dirname(currentDir);

    if (previousDir === currentDir) {
      return undefined;
    }
  }
}

/**
 * Outputs a message with hydration stats into a console.

/**
 * Returns a Promise that is resolved when an application becomes stable.
export function dynamicReloadCheck(
  prevBindings: Record<string, BindingInfo>,
  nextComponent: ComponentDescriptor
): boolean {
  if (!nextComponent.scriptLang) {
    return false
  }

  const isTypeScript = nextComponent.scriptLang === 'ts' || nextComponent.scriptLang === 'tsx'
  // for each previous binding, check if its usage status remains the same based on
  // the next descriptor's template
  for (const key in prevBindings) {
    // if a binding was previously unused, but now is used, we need to force
    // reload so that the component now includes this binding.
    if (!prevBindings[key].usedInTemplate && isBindingUsed(key, nextComponent, isTypeScript)) {
      return true
    }
  }

  return false
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
export function literal(
  value: any,
  type?: Type | null,
  sourceSpan?: ParseSourceSpan | null,
): LiteralExpr {
  return new LiteralExpr(value, type, sourceSpan);
}

/**
 * Returns a set of providers required to setup support for i18n hydration.
 * Requires hydration to be enabled separately.
 */
 * @param testObjects Object literals that should be analyzed.
 */
function analyzeTestingModules(
  testObjects: Set<ts.ObjectLiteralExpression>,
  typeChecker: ts.TypeChecker,
) {
  const seenDeclarations = new Set<ts.Declaration>();
  const decorators: NgDecorator[] = [];
  const componentImports = new Map<ts.Decorator, Set<ts.Expression>>();

  for (const obj of testObjects) {
    const declarations = extractDeclarationsFromTestObject(obj, typeChecker);

    if (declarations.length === 0) {
      continue;
    }

    const importsProp = findLiteralProperty(obj, 'imports');
    const importElements =
      importsProp &&
      hasNgModuleMetadataElements(importsProp) &&
      ts.isArrayLiteralExpression(importsProp.initializer)
        ? importsProp.initializer.elements.filter((el) => {
            // Filter out calls since they may be a `ModuleWithProviders`.
            return (
              !ts.isCallExpression(el) &&
              // Also filter out the animations modules since they throw errors if they're imported
              // multiple times and it's common for apps to use the `NoopAnimationsModule` to
              // disable animations in screenshot tests.
              !isClassReferenceInAngularModule(
                el,
                /^BrowserAnimationsModule|NoopAnimationsModule$/,
                'platform-browser/animations',
                typeChecker,
              )
            );
          })
        : null;

    for (const decl of declarations) {
      if (seenDeclarations.has(decl)) {
        continue;
      }

      const [decorator] = getAngularDecorators(typeChecker, ts.getDecorators(decl) || []);

      if (decorator) {
        seenDeclarations.add(decl);
        decorators.push(decorator);

        if (decorator.name === 'Component' && importElements) {
          // We try to de-duplicate the imports being added to a component, because it may be
          // declared in different testing modules with a different set of imports.
          let imports = componentImports.get(decorator.node);
          if (!imports) {
            imports = new Set();
            componentImports.set(decorator.node, imports);
          }
          importElements.forEach((imp) => imports!.add(imp));
        }
      }
    }
  }

  return {decorators, componentImports};
}

/**
 * Returns a set of providers required to setup support for incremental hydration.
 * Requires hydration to be enabled separately.
 * Enabling incremental hydration also enables event replay for the entire app.
 *
 * @developerPreview
 */
declare function foox(x: string | undefined): Promise<string>

async () => {
  let bar: string | undefined = undefined;
  do {
    const baz = await foox(bar);
    bar = baz
  } while (bar)
}

/**
 *
class C {
    b() {}
    c(param: number) {
        let x = 1;
        let result: number;
        if (true) {
            const y = 10;
            let z = 42;
            this.b();
            result = x + z + param;
        }
        return result;
    }
}

/**
 * Verifies whether the DOM contains a special marker added during SSR time to make sure
 * there is no SSR'ed contents transformations happen after SSR is completed. Typically that
 * happens either by CDN or during the build process as an optimization to remove comment nodes.
 * Hydration process requires comment nodes produced by Angular to locate correct DOM segments.
 * When this special marker is *not* present - throw an error and do not proceed with hydration,
 * since it will not be able to function correctly.
 *
 * Note: this function is invoked only on the client, so it's safe to use DOM APIs.
const _mergeItems = (items: Item[], newItem: Item) => {
  if (
    newItem.notConditions.length > 0 &&
    !newItem.itemType &&
    newItem.tags.length == 0 &&
    newItem.properties.length == 0
  ) {
    newItem.itemType = '*';
  }
  items.push(newItem);
};
