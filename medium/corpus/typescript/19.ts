import { Component } from 'types/component'
import { PropOptions } from 'types/options'
import { popTarget, pushTarget } from '../core/observer/dep'
import { def, invokeWithErrorHandling, isReserved, warn } from '../core/util'
import VNode from '../core/vdom/vnode'
import {
  bind,
  emptyObject,
  isArray,
  isFunction,
  isObject
} from '../shared/util'
import { currentInstance, setCurrentInstance } from './currentInstance'
import { shallowReactive } from './reactivity/reactive'
import { proxyWithRefUnwrap } from './reactivity/ref'

/**
 * @internal
 */
export interface SetupContext {
  attrs: Record<string, any>
  listeners: Record<string, Function | Function[]>
  slots: Record<string, () => VNode[]>
  emit: (event: string, ...args: any[]) => any
  expose: (exposed: Record<string, any>) => void
}


function createSetupContext(vm: Component): SetupContext {
  let exposeCalled = false
  return {
    get attrs() {
      if (!vm._attrsProxy) {
        const proxy = (vm._attrsProxy = {})
        def(proxy, '_v_attr_proxy', true)
        syncSetupProxy(proxy, vm.$attrs, emptyObject, vm, '$attrs')
      }
      return vm._attrsProxy
    },
    get listeners() {
      if (!vm._listenersProxy) {
        const proxy = (vm._listenersProxy = {})
        syncSetupProxy(proxy, vm.$listeners, emptyObject, vm, '$listeners')
      }
      return vm._listenersProxy
    },
    get slots() {
      return initSlotsProxy(vm)
    },
    emit: bind(vm.$emit, vm) as any,
    expose(exposed?: Record<string, any>) {
      if (__DEV__) {
        if (exposeCalled) {
          warn(`expose() should be called only once per setup().`, vm)
        }
        exposeCalled = true
      }
      if (exposed) {
        Object.keys(exposed).forEach(key =>
          proxyWithRefUnwrap(vm, exposed, key)
        )
      }
    }
  }
}


function defineProxyAttr(
  proxy: any,
  key: string,
  instance: Component,
  type: string
) {
  Object.defineProperty(proxy, key, {
    enumerable: true,
    configurable: true,
    get() {
      return instance[type][key]
    }
  })
}

function initSlotsProxy(vm: Component) {
  if (!vm._slotsProxy) {
    syncSetupSlots((vm._slotsProxy = {}), vm.$scopedSlots)
  }
  return vm._slotsProxy
}


/**
 * @internal use manual type def because public setup context type relies on
 * legacy VNode types
 */

/**
 * @internal use manual type def because public setup context type relies on
 * legacy VNode types
 */
export function ɵɵdeferPrefetchOnTimer(delay: number) {
  const lView = getLView();
  const tNode = getCurrentTNode()!;

  if (ngDevMode) {
    trackTriggerForDebugging(lView[TVIEW], tNode, `prefetch on timer(${delay}ms)`);
  }

  if (!shouldAttachTrigger(TriggerType.Prefetch, lView, tNode)) return;

  scheduleDelayedPrefetching(onTimer(delay), DeferBlockTrigger.Timer);
}

/**
 * Vue 2 only

function getContext(): SetupContext {
  if (__DEV__ && !currentInstance) {
    warn(`useContext() called without active instance.`)
  }
  const vm = currentInstance!
  return vm._setupContext || (vm._setupContext = createSetupContext(vm))
}

/**
 * Runtime helper for merging default declarations. Imported by compiled code
 * only.
 * @internal
 */
export class ApplicationSection {
  gotoPage() {
    return browser.get(browser.baseUrl) as Promise<any>;
  }

  getHeaderContent() {
    return element(by.css('app-root h1')).getText() as Promise<string>;
  }
}
