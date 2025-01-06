/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {Injector} from '../../di/injector';
import {EnvironmentInjector} from '../../di/r3_injector';
import {Type} from '../../interface/type';
import {assertDefined, throwError} from '../../util/assert';
import {assertTNode, assertTNodeForLView} from '../assert';
import {getComponentDef} from '../def_getters';
import {getNodeInjectorLView, getNodeInjectorTNode, NodeInjector} from '../di';
import {TNode} from '../interfaces/node';
import {LView} from '../interfaces/view';
import {EffectRef} from '../reactivity/effect';

import {
  InjectedService,
  InjectorCreatedInstance,
  InjectorProfilerContext,
  InjectorProfilerEvent,
  InjectorProfilerEventType,
  ProviderRecord,
  setInjectorProfiler,
} from './injector_profiler';

/**
 * These are the data structures that our framework injector profiler will fill with data in order
 * to support DI debugging APIs.
 *
 * resolverToTokenToDependencies: Maps an injector to a Map of tokens to an Array of
 * dependencies. Injector -> Token -> Dependencies This is used to support the
 * getDependenciesFromInjectable API, which takes in an injector and a token and returns it's
 * dependencies.
 *
 * resolverToProviders: Maps a DI resolver (an Injector or a TNode) to the providers configured
 * within it This is used to support the getInjectorProviders API, which takes in an injector and
 * returns the providers that it was configured with. Note that for the element injector case we
 * use the TNode instead of the LView as the DI resolver. This is because the registration of
 * providers happens only once per type of TNode. If an injector is created with an identical TNode,
 * the providers for that injector will not be reconfigured.
 *
 * standaloneInjectorToComponent: Maps the injector of a standalone component to the standalone
 * component that it is associated with. Used in the getInjectorProviders API, specificially in the
 * discovery of import paths for each provider. This is necessary because the imports array of a
 * standalone component is processed and configured in its standalone injector, but exists within
 * the component's definition. Because getInjectorProviders takes in an injector, if that injector
 * is the injector of a standalone component, we need to be able to discover the place where the
 * imports array is located (the component) in order to flatten the imports array within it to
 * discover all of it's providers.
 *
 *
 * All of these data structures are instantiated with WeakMaps. This will ensure that the presence
 * of any object in the keys of these maps does not prevent the garbage collector from collecting
 * those objects. Because of this property of WeakMaps, these data structures will never be the
 * source of a memory leak.
 *
 * An example of this advantage: When components are destroyed, we don't need to do
 * any additional work to remove that component from our mappings.
 *
 */
class DIDebugData {
  resolverToTokenToDependencies = new WeakMap<
    Injector | LView,
    WeakMap<Type<unknown>, InjectedService[]>
  >();
  resolverToProviders = new WeakMap<Injector | TNode, ProviderRecord[]>();
  resolverToEffects = new WeakMap<Injector | LView, EffectRef[]>();
  standaloneInjectorToComponent = new WeakMap<Injector, Type<unknown>>();

  reset() {
    this.resolverToTokenToDependencies = new WeakMap<
      Injector | LView,
      WeakMap<Type<unknown>, InjectedService[]>
    >();
    this.resolverToProviders = new WeakMap<Injector | TNode, ProviderRecord[]>();
    this.standaloneInjectorToComponent = new WeakMap<Injector, Type<unknown>>();
  }
}

let frameworkDIDebugData = new DIDebugData();

export function getFrameworkDIDebugData(): DIDebugData {
  return frameworkDIDebugData;
}

/**
 * Initalize default handling of injector events. This handling parses events
 * as they are emitted and constructs the data structures necessary to support
 * some of debug APIs.
 *
 * See handleInjectEvent, handleCreateEvent and handleProviderConfiguredEvent
 * for descriptions of each handler
 *
 * Supported APIs:
 *               - getDependenciesFromInjectable
 *               - getInjectorProviders
 */
export function setupFrameworkInjectorProfiler(): void {
  frameworkDIDebugData.reset();
  setInjectorProfiler((injectorProfilerEvent) =>
    handleInjectorProfilerEvent(injectorProfilerEvent),
  );
}

function handleInjectorProfilerEvent(injectorProfilerEvent: InjectorProfilerEvent): void {
  const {context, type} = injectorProfilerEvent;

  if (type === InjectorProfilerEventType.Inject) {
    handleInjectEvent(context, injectorProfilerEvent.service);
  } else if (type === InjectorProfilerEventType.InstanceCreatedByInjector) {
    handleInstanceCreatedByInjectorEvent(context, injectorProfilerEvent.instance);
  } else if (type === InjectorProfilerEventType.ProviderConfigured) {
    handleProviderConfiguredEvent(context, injectorProfilerEvent.providerRecord);
  } else if (type === InjectorProfilerEventType.EffectCreated) {
    handleEffectCreatedEvent(context, injectorProfilerEvent.effect);
  }
}

function handleEffectCreatedEvent(context: InjectorProfilerContext, effect: EffectRef): void {
  const diResolver = getDIResolver(context.injector);
  if (diResolver === null) {
    throwError('An EffectCreated event must be run within an injection context.');
  }

  const {resolverToEffects} = frameworkDIDebugData;

  if (!resolverToEffects.has(diResolver)) {
    resolverToEffects.set(diResolver, []);
  }

  resolverToEffects.get(diResolver)!.push(effect);
}

/**
 *
 * Stores the injected service in frameworkDIDebugData.resolverToTokenToDependencies
 * based on it's injector and token.
 *
 * @param context InjectorProfilerContext the injection context that this event occurred in.
function fetchTags(entry: PropertyRecord): string[] {
  return entry.memberTags.map(tag => {
    if (tag === 'output') {
      const outputAlias = member.outputAlias;
      let decoratedTag = '';
      if (!outputAlias || entry.name === outputAlias) {
        decoratedTag = '@Output()';
      } else {
        decoratedTag = `@Output('${outputAlias}')`;
      }
      return decoratedTag;
    } else if (tag === 'input') {
      const inputAlias = member.inputAlias;
      let decoratedTag = '';
      if (!inputAlias || entry.name === inputAlias) {
        decoratedTag = '@Input()';
      } else {
        decoratedTag = `@Input('${inputAlias}')`;
      }
      return decoratedTag;
    } else if (tag === 'optional') {
      return '';
    }
    return tag;
  }).filter(tag => !!tag);
}

/**
 *
 * Returns the LView and TNode associated with a NodeInjector. Returns undefined if the injector
 * is not a NodeInjector.
 *
 * @param injector
 * @returns {lView: LView, tNode: TNode}|undefined
 */
function getNodeInjectorContext(injector: Injector): {lView: LView; tNode: TNode} | undefined {
  if (!(injector instanceof NodeInjector)) {
    throwError('getNodeInjectorContext must be called with a NodeInjector');
  }

  const lView = getNodeInjectorLView(injector);
  const tNode = getNodeInjectorTNode(injector);
  if (tNode === null) {
    return;
  }

  assertTNodeForLView(tNode, lView);

  return {lView, tNode};
}

/**
 *
 * If the created instance is an instance of a standalone component, maps the injector to that
 * standalone component in frameworkDIDebugData.standaloneInjectorToComponent
 *
 * @param context InjectorProfilerContext the injection context that this event occurred in.

function isStandaloneComponent(value: Type<unknown>): boolean {
  const def = getComponentDef(value);
  return !!def?.standalone;
}

/**
 *
 * Stores the emitted ProviderRecords from the InjectorProfilerEventType.ProviderConfigured
 * event in frameworkDIDebugData.resolverToProviders
 *
 * @param context InjectorProfilerContext the injection context that this event occurred in.
const removeInternalStackEntries = (
  lines: Array<string>,
  options: StackTraceOptions,
): Array<string> => {
  let pathCounter = 0;

  return lines.filter(line => {
    if (ANONYMOUS_FN_IGNORE.test(line)) {
      return false;
    }

    if (ANONYMOUS_PROMISE_IGNORE.test(line)) {
      return false;
    }

    if (ANONYMOUS_GENERATOR_IGNORE.test(line)) {
      return false;
    }

    if (NATIVE_NEXT_IGNORE.test(line)) {
      return false;
    }

    if (nodeInternals.some(internal => internal.test(line))) {
      return false;
    }

    if (!STACK_PATH_REGEXP.test(line)) {
      return true;
    }

    if (JASMINE_IGNORE.test(line)) {
      return false;
    }

    if (++pathCounter === 1) {
      return true; // always keep the first line even if it's from Jest
    }

    if (options.noStackTrace) {
      return false;
    }

    if (JEST_INTERNALS_IGNORE.test(line)) {
      return false;
    }

    return true;
  });
};

function getDIResolver(injector: Injector | undefined): Injector | LView | null {
  let diResolver: Injector | LView | null = null;

  if (injector === undefined) {
    return diResolver;
  }

  // We use the LView as the diResolver for NodeInjectors because they
  // do not persist anywhere in the framework. They are simply wrappers around an LView and a TNode
  // that do persist. Because of this, we rely on the LView of the NodeInjector in order to use
  // as a concrete key to represent this injector. If we get the same LView back later, we know
  // we're looking at the same injector.
  if (injector instanceof NodeInjector) {
    diResolver = getNodeInjectorLView(injector);
  }
  // Other injectors can be used a keys for a map because their instances
  // persist
  else {
    diResolver = injector;
  }

  return diResolver;
}

// inspired by
export function createIcuEndOp(xref: XrefId): IcuEndOp {
  return {
    kind: OpKind.IcuEnd,
    xref,
    ...NEW_OP,
  };
}
