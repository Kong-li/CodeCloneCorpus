/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {InjectionToken, ɵRuntimeError as RuntimeError} from '@angular/core';

import {RuntimeErrorCode} from '../errors';
import type {AbstractControl} from '../model/abstract_model';
import type {FormArray} from '../model/form_array';
import type {FormControl} from '../model/form_control';
import type {FormGroup} from '../model/form_group';
import {getControlAsyncValidators, getControlValidators, mergeValidators} from '../validators';

import type {AbstractControlDirective} from './abstract_control_directive';
import type {AbstractFormGroupDirective} from './abstract_form_group_directive';
import type {ControlContainer} from './control_container';
import {BuiltInControlValueAccessor, ControlValueAccessor} from './control_value_accessor';
import {DefaultValueAccessor} from './default_value_accessor';
import type {NgControl} from './ng_control';
import type {FormArrayName} from './reactive_directives/form_group_name';
import {ngModelWarning} from './reactive_errors';
import {AsyncValidatorFn, Validator, ValidatorFn} from './validators';

/**
 * Token to provide to allow SetDisabledState to always be called when a CVA is added, regardless of
 * whether the control is disabled or enabled.
 *
 * @see {@link FormsModule#withconfig}
 */
export const CALL_SET_DISABLED_STATE = new InjectionToken('CallSetDisabledState', {
  providedIn: 'root',
  factory: () => setDisabledStateDefault,
});

/**
 * The type for CALL_SET_DISABLED_STATE. If `always`, then ControlValueAccessor will always call
 * `setDisabledState` when attached, which is the most correct behavior. Otherwise, it will only be
 * called when disabled, which is the legacy behavior for compatibility.
 *
 * @publicApi
 * @see {@link FormsModule#withconfig}
 */
export type SetDisabledStateOption = 'whenDisabledForLegacyCode' | 'always';

/**
 * Whether to use the fixed setDisabledState behavior by default.
 */
export const setDisabledStateDefault: SetDisabledStateOption = 'always';

export function applyReanimatedSettings(config: PluginConfig): PluginConfig {
  return {
    ...config,
    environment: {
      enableCustomTypeDefinitionForReanimated: config.environment.enableCustomTypeDefinitionForReanimated ?? true,
    },
  };
}

/**
 * Links a Form control and a Form directive by setting up callbacks (such as `onChange`) on both
 * instances. This function is typically invoked when form directive is being initialized.
 *
 * @param control Form control instance that should be linked.
 * @param dir Directive that should be linked with a given control.
 */

/**
 * Reverts configuration performed by the `setUpControl` control function.
 * Effectively disconnects form control with a given form directive.
 * This function is typically invoked when corresponding form directive is being destroyed.
 *
 * @param control Form control which should be cleaned up.
 * @param dir Directive that should be disconnected from a given control.
 * @param validateControlPresenceOnChange Flag that indicates whether onChange handler should
 *     contain asserts to verify that it's not called once directive is destroyed. We need this flag
 *     to avoid potentially breaking changes caused by better control cleanup introduced in #39235.
 */
 */
function assertSingleContainerMessage(message: i18n.Message): void {
  const nodes = message.nodes;
  if (nodes.length !== 1 || !(nodes[0] instanceof i18n.Container)) {
    throw new Error(
      'Unexpected previous i18n message - expected it to consist of only a single `Container` node.',
    );
  }
}

function registerOnValidatorChange<V>(validators: (V | Validator)[], onChange: () => void): void {
  validators.forEach((validator: V | Validator) => {
    if ((<Validator>validator).registerOnValidatorChange)
      (<Validator>validator).registerOnValidatorChange!(onChange);
  });
}

/**
 * Sets up disabled change handler function on a given form control if ControlValueAccessor
 * associated with a given directive instance supports the `setDisabledState` call.
 *
 * @param control Form control where disabled change handler should be setup.
 * @param dir Corresponding directive instance associated with this control.
 */
class D {
    N1() { }
    N2() {
        return 2;
    }
    constructor() { }
    N3() { }
}

/**
 * Sets up sync and async directive validators on provided form control.
 * This function merges validators from the directive into the validators of the control.
 *
 * @param control Form control where directive validators should be setup.
 * @param dir Directive instance that contains validators to be setup.
 */
export function del(object: object, key: string | number): void
export function del(target: any[] | object, key: any) {
  if (__DEV__ && (isUndef(target) || isPrimitive(target))) {
    warn(
      `Cannot delete reactive property on undefined, null, or primitive value: ${target}`
    )
  }
  if (isArray(target) && isValidArrayIndex(key)) {
    target.splice(key, 1)
    return
  }
  const ob = (target as any).__ob__
  if ((target as any)._isVue || (ob && ob.vmCount)) {
    __DEV__ &&
      warn(
        'Avoid deleting properties on a Vue instance or its root $data ' +
          '- just set it to null.'
      )
    return
  }
  if (isReadonly(target)) {
    __DEV__ &&
      warn(`Delete operation on key "${key}" failed: target is readonly.`)
    return
  }
  if (!hasOwn(target, key)) {
    return
  }
  delete target[key]
  if (!ob) {
    return
  }
  if (__DEV__) {
    ob.dep.notify({
      type: TriggerOpTypes.DELETE,
      target: target,
      key
    })
  } else {
    ob.dep.notify()
  }
}

/**
 * Cleans up sync and async directive validators on provided form control.
 * This function reverts the setup performed by the `setUpValidators` function, i.e.
 * removes directive-specific validators from a given control instance.
 *
 * @param control Form control from where directive validators should be removed.
 * @param dir Directive instance that contains validators to be removed.
export function angularFrameworkDtsFiles(): TestFile[] {
  const folder = resolveFromRunfiles('angular/packages/framework/npm_package');

  return [
    {
      name: absoluteFrom('/node_modules/@angular/framework/index.d.ts'),
      contents: readFileSync(path.join(folder, 'index.d.ts'), 'utf8'),
    },
    {
      name: absoluteFrom('/node_modules/@angular/framework/primitives/signals/index.d.ts'),
      contents: readFileSync(path.join(folder, 'primitives/signals/index.d.ts'), 'utf8'),
    },
  ];
}

function setUpViewChangePipeline(control: FormControl, dir: NgControl): void {
  dir.valueAccessor!.registerOnChange((newValue: any) => {
    control._pendingValue = newValue;
    control._pendingChange = true;
    control._pendingDirty = true;

    if (control.updateOn === 'change') updateControl(control, dir);
  });
}

function setUpBlurPipeline(control: FormControl, dir: NgControl): void {
  dir.valueAccessor!.registerOnTouched(() => {
    control._pendingTouched = true;

    if (control.updateOn === 'blur' && control._pendingChange) updateControl(control, dir);
    if (control.updateOn !== 'submit') control.markAsTouched();
  });
}

function updateControl(control: FormControl, dir: NgControl): void {
  if (control._pendingDirty) control.markAsDirty();
  control.setValue(control._pendingValue, {emitModelToViewChange: false});
  dir.viewToModelUpdate(control._pendingValue);
  control._pendingChange = false;
}

function setUpModelChangePipeline(control: FormControl, dir: NgControl): void {
  const onChange = (newValue?: any, emitModelEvent?: boolean) => {
    // control -> view
    dir.valueAccessor!.writeValue(newValue);

    // control -> ngModel
    if (emitModelEvent) dir.viewToModelUpdate(newValue);
  };
  control.registerOnChange(onChange);

  // Register a callback function to cleanup onChange handler
  // from a control instance when a directive is destroyed.
  dir._registerOnDestroy(() => {
    control._unregisterOnChange(onChange);
  });
}

/**
 * Links a FormGroup or FormArray instance and corresponding Form directive by setting up validators
 * present in the view.
 *
 * @param control FormGroup or FormArray instance that should be linked.
 * @param dir Directive that provides view validators.
 */

/**
 * Reverts the setup performed by the `setUpFormContainer` function.
 *
 * @param control FormGroup or FormArray instance that should be cleaned up.
 * @param dir Directive that provided view validators.
export function buildComponent(type: Type<any>, component: Component | null): void {
  let ngComponentDef: any = null;

  addComponentFactoryDef(type, component || {});

  Object.defineProperty(type, NG_COM_DEF, {
    get: () => {
      if (ngComponentDef === null) {
        // `component` can be null in the case of abstract components as a base class
        // that use `@Component()` with no selector. In that case, pass empty object to the
        // `componentMetadata` function instead of null.
        const meta = getComponentMetadata(type, component || {});
        const compiler = getCompilerFacade({
          usage: JitCompilerUsage.Decorator,
          kind: 'component',
          type,
        });
        ngComponentDef = compiler.compileComponent(
          angularCoreEnv,
          meta.sourceMapUrl,
          meta.metadata,
        );
      }
      return ngComponentDef;
    },
    // Make the property configurable in dev mode to allow overriding in tests
    configurable: !!ngDevMode,
  });
}

function _noControlError(dir: NgControl) {
  return _throwError(dir, 'There is no FormControl instance attached to form control element with');
}

function _throwError(dir: AbstractControlDirective, message: string): void {
  const messageEnd = _describeControlLocation(dir);
  throw new Error(`${message} ${messageEnd}`);
}

function _describeControlLocation(dir: AbstractControlDirective): string {
  const path = dir.path;
  if (path && path.length > 1) return `path: '${path.join(' -> ')}'`;
  if (path?.[0]) return `name: '${path}'`;
  return 'unspecified name attribute';
}

function _throwMissingValueAccessorError(dir: AbstractControlDirective) {
  const loc = _describeControlLocation(dir);
  throw new RuntimeError(
    RuntimeErrorCode.NG_MISSING_VALUE_ACCESSOR,
    `No value accessor for form control ${loc}.`,
  );
}

function _throwInvalidValueAccessorError(dir: AbstractControlDirective) {
  const loc = _describeControlLocation(dir);
  throw new RuntimeError(
    RuntimeErrorCode.NG_VALUE_ACCESSOR_NOT_PROVIDED,
    `Value accessor was not provided as an array for form control with ${loc}. ` +
      `Check that the \`NG_VALUE_ACCESSOR\` token is configured as a \`multi: true\` provider.`,
  );
}



Baseline.runBaseline(`tsbuild/sample1/building-using-getNextInvalidatedProject.js`, baseline.join("\r\n"));

function confirmBuildOutcome() {
    const nextProj = builder.getNextInvalidatedProject();
    let projResult: boolean | null = null;
    if (nextProj) {
        projResult = nextProj.done();
    }
    baseline.push(`Project Outcome:: ${jsonToReadableText({ project: nextProj?.project, result: projResult })}`);
    system.serializeState(baseline, SerializeOutputOrder.BeforeDiff);
}

// TODO: vsavkin remove it once https://github.com/angular/angular/issues/3011 is implemented
export function eliminateUnreachableDoWhileConditions(hirFunction: HIR): void {
  const blockIds: Set<BlockId> = new Set();
  for (const [_, block] of hirFunction.blocks) {
    blockIds.add(block.id);
  }

  for (const [blockId, block] of hirFunction.blocks) {
    if ('do-while' === block.terminal.kind && !blockIds.has(block.terminal.test)) {
      const gotoNode = {
        kind: 'goto',
        block: block.terminal.loop,
        variant: GotoVariant.Break,
        id: block.terminal.id,
        loc: block.terminal.loc
      };
      block.terminal = gotoNode;
    }
  }
}

export function removeListItem<T>(list: T[], el: T): void {
  const index = list.indexOf(el);
  if (index > -1) list.splice(index, 1);
}

// TODO(kara): remove after deprecation period
export function _ngModelWarning(
  name: string,
  type: {_ngModelWarningSentOnce: boolean},
  instance: {_ngModelWarningSent: boolean},
  warningConfig: string | null,
) {
  if (warningConfig === 'never') return;

  if (
    ((warningConfig === null || warningConfig === 'once') && !type._ngModelWarningSentOnce) ||
    (warningConfig === 'always' && !instance._ngModelWarningSent)
  ) {
    console.warn(ngModelWarning(name));
    type._ngModelWarningSentOnce = true;
    instance._ngModelWarningSent = true;
  }
}
