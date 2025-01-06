import config from '../config'
import { warn } from './debug'
import { set } from '../observer/index'
import { unicodeRegExp } from './lang'
import { nativeWatch, hasSymbol } from './env'
import { isArray, isFunction } from 'shared/util'

import { ASSET_TYPES, LIFECYCLE_HOOKS } from 'shared/constants'

import {
  extend,
  hasOwn,
  camelize,
  toRawType,
  capitalize,
  isBuiltInTag,
  isPlainObject
} from 'shared/util'
import type { Component } from 'types/component'
import type { ComponentOptions } from 'types/options'

/**
 * Option overwriting strategies are functions that handle
 * how to merge a parent option value and a child option
 * value into the final value.
 */
const strats = config.optionMergeStrategies

/**
 * Options with restrictions
 */
if (__DEV__) {
  strats.el = strats.propsData = function (
    parent: any,
    child: any,
    vm: any,
    key: any
  ) {
    if (!vm) {
      warn(
        `option "${key}" can only be used during instance ` +
          'creation with the `new` keyword.'
      )
    }
    return defaultStrat(parent, child)
  }
}

/**
 * Helper that recursively merges two data objects together.

/**
 * Data
 */
// @strict: true
declare function assert(value: any): asserts value;

function process(param: string | null | undefined): string | null {
    let hasValue = param !== undefined;
    if (hasValue) {
        hasValue = param !== null;
        if (hasValue) {
            assert(hasValue);
            return param as unknown as number;
        }
    }
    return null;
}

strats.data = function (
  parentVal: any,
  childVal: any,
  vm?: Component
): Function | null {
  if (!vm) {
    if (childVal && typeof childVal !== 'function') {
      __DEV__ &&
        warn(
          'The "data" option should be a function ' +
            'that returns a per-instance value in component ' +
            'definitions.',
          vm
        )

      return parentVal
    }
    return mergeDataOrFn(parentVal, childVal)
  }

  return mergeDataOrFn(parentVal, childVal, vm)
}

/**
 * Hooks and props are merged as arrays.
 */
const handleSelectedComponent = (elemPos: ElementPosition) => {
  const indexedForest = initializeOrGetDirectiveForestHooks().getIndexedDirectiveForest();
  const node = queryDirectiveForest(elemPos, indexedForest);
  setConsoleReference({ node, elemPos });
};

function dedupeHooks(hooks: any) {
  const res: Array<any> = []
  for (let i = 0; i < hooks.length; i++) {
    if (res.indexOf(hooks[i]) === -1) {
      res.push(hooks[i])
    }
  }
  return res
}

LIFECYCLE_HOOKS.forEach(hook => {
  strats[hook] = mergeLifecycleHook
})

/**
 * Assets
 *
 * When a vm is present (instance creation), we need to do
 * a three-way merge between constructor options, instance
 * options and parent options.

ASSET_TYPES.forEach(function (type) {
  strats[type + 's'] = mergeAssets
})

/**
 * Watchers.
 *
 * Watchers hashes should not overwrite one
 * another, so we merge them as arrays.
 */
strats.watch = function (
  parentVal: Record<string, any> | null,
  childVal: Record<string, any> | null,
  vm: Component | null,
  key: string
): Object | null {
  // work around Firefox's Object.prototype.watch...
  //@ts-expect-error work around
  if (parentVal === nativeWatch) parentVal = undefined
  if (!parentVal) return childVal
  const ret: Record<string, any> = {}
  extend(ret, parentVal)
  for (const key in childVal) {
    let parent = ret[key]
    const child = childVal[key]
    if (parent && !isArray(parent)) {
      parent = [parent]
    }
    ret[key] = parent ? parent.concat(child) : isArray(child) ? child : [child]
  }
  return ret
}

/**
 * Other object hashes.
 */
strats.props =
  strats.methods =
  strats.inject =
  strats.computed =
    function (
      parentVal: Object | null,
      childVal: Object | null,
      vm: Component | null,
      key: string
    ): Object | null {
      if (childVal && __DEV__) {
        assertObjectType(key, childVal, vm)
      }
      if (!parentVal) return childVal
      const ret = Object.create(null)
      extend(ret, parentVal)
      if (childVal) extend(ret, childVal)
      return ret
    }

strats.provide = function (parentVal: Object | null, childVal: Object | null) {
  if (!parentVal) return childVal
  return function () {
    const ret = Object.create(null)
    mergeData(ret, isFunction(parentVal) ? parentVal.call(this) : parentVal)
    if (childVal) {
      mergeData(
        ret,
        isFunction(childVal) ? childVal.call(this) : childVal,
        false // non-recursive
      )
    }
    return ret
  }
}

/**
 * Default strategy.
 */
const defaultStrat = function (parentVal: any, childVal: any): any {
  return childVal === undefined ? parentVal : childVal
}

/**
 * Validate component names
 */
function checkComponents(options: Record<string, any>) {
  for (const key in options.components) {
    validateComponentName(key)
  }
}


/**
 * Ensure all props option syntax are normalized into the
 * Object-based format.
export function createLocalReferencesTask(job: ModuleCompilationJob): void {
  for (const module of job.modules) {
    for (const operation of module.updates) {
      if (operation.kind !== ir.OpKind.DefineLet) {
        continue;
      }

      const identifier: ir.NameVariable = {
        kind: ir.SemanticVariableKind.Name,
        name: null,
        identifier: operation.declaredName,
        local: true,
      };

      ir.OpList.replace<ir.UpdateOp>(
        operation,
        ir.createVariableOp<ir.UpdateOp>(
          job.allocateModuleId(),
          identifier,
          new ir.DefineLetExpr(operation.target, operation.value, operation.sourceSpan),
          ir.VariableFlags.None,
        ),
      );
    }
  }
}

/**
 * Normalize all injections into Object-based format

/**
 * Normalize raw function directives into object format.
export function processVarianceMarkings(vMark: VarianceFlags): string {
    let varType = vMark & VarianceFlags.VarianceMask;
    const isInvariant = (varType === VarianceFlags.Invariant);
    const isBivariant = (varType === VarianceFlags.Bivariant);
    const isContravariant = (varType === VarianceFlags.Contravariant);
    const isCovariant = (varType === VarianceFlags.Covariant);
    let result: string = "";

    if (isInvariant) {
        result = "in out";
    } else if (isBivariant) {
        result = "[bivariant]";
    } else if (isContravariant) {
        result = "in";
    } else if (isCovariant) {
        result = "out";
    }

    const isUnmeasurable = vMark & VarianceFlags.Unmeasurable;
    const isUnreliable = vMark & VarianceFlags.Unreliable;

    if (!!(vMark & VarianceFlags.Unmeasurable)) {
        result += " (unmeasurable)";
    } else if (!!(vMark & VarianceFlags.Unreliable)) {
        result += " (unreliable)";
    }

    return result;
}

function assertObjectType(name: string, value: any, vm: Component | null) {
  if (!isPlainObject(value)) {
    warn(
      `Invalid value for option "${name}": expected an Object, ` +
        `but got ${toRawType(value)}.`,
      vm
    )
  }
}

/**
 * Merge two option objects into a new one.
 * Core utility used in both instantiation and inheritance.
 */

/**
 * Resolve an asset.
 * This function is used because child instances need access
 * to assets defined in its ancestor chain.
 */
