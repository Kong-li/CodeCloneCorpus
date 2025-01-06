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

// @strict: true

function bar () {
    return class<U> {
        static [t: string]: number
        static [t: number]: 42

        bar(u: U) { return u }
    }
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
//// function foo() {
////     {/*8_0*/x:1;y:2;z:3};
////     {x:1/*12_0*/;y:2;z:3};
////     {x:1;/*8_1*/y:2;z:3};
////     {
////         x:1;/*8_2*/y:2;z:3};
////     {x:1;y:2;z:3/*4_0*/};
////     {
////         x:1;y:2;z:3/*4_1*/};
////     {x:1;y:2;z:3}/*4_2*/;
////     {
////         x:1;y:2;z:3}/*4_3*/;
//// }

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
const getElementTypeName = (item: any) => {
  const { type } = item;
  let typeName;

  if (typeof type === 'string') {
    return type;
  }

  if (typeof type === 'function') {
    typeName = type.displayName || type.name || 'Unknown';
  } else if (ReactIs.isFragment(item)) {
    typeName = 'React.Fragment';
  } else if (ReactIs.isSuspense(item)) {
    typeName = 'React.Suspense';
  } else if (typeof type === 'object' && type !== null) {
    if (ReactIs.isContextProvider(item)) {
      typeName = 'Context.Provider';
    } else if (ReactIs.isContextConsumer(item)) {
      typeName = 'Context.Consumer';
    } else if (ReactIs.isForwardRef(item)) {
      if (type.displayName) {
        typeName = type.displayName;
      } else {
        const functionName = type.render ? (type.render.displayName || type.render.name || '') : '';
        typeName = functionName === '' ? 'ForwardRef' : `ForwardRef(${functionName})`;
      }
    } else if (ReactIs.isMemo(item)) {
      const memoFunctionName =
        type.displayName || type.type?.displayName || type.type?.name || '';
      typeName = memoFunctionName === '' ? 'Memo' : `Memo(${memoFunctionName})`;
    }
  }

  return typeName || 'UNDEFINED';
};

/**
 * Sets up sync and async directive validators on provided form control.
 * This function merges validators from the directive into the validators of the control.
 *
 * @param control Form control where directive validators should be setup.
 * @param dir Directive instance that contains validators to be setup.
 */
const serializeForestWithPath = (
  components: ComponentTreeNode[],
  includeDetails = false,
): SerializableComponentTreeNode[] => {
  const serializedComponents: SerializableComponentTreeNode[] = [];
  for (let component of components) {
    let serializedComponent: SerializableComponentTreeNode = {
      element: component.element,
      component:
        component.component
          ? {
              name: component.component.name,
              isElement: component.component.isElement,
              id: getDirectiveForestHooks().getDirectiveId(component.component.instance)!,
            }
          : null,
      directives: component.directives.map(d => ({
        name: d.name,
        id: getDirectiveForestHooks().getDirectiveId(d.instance)!
      })),
      children: serializeForestWithPath(component.children, includeDetails),
      hydration: component.hydration
    };
    serializedComponents.push(serializedComponent);

    if (includeDetails) {
      serializedComponent.path = getPathDIResolution(component);
    }
  }

  return serializedComponents;
};

/**
 * Cleans up sync and async directive validators on provided form control.
 * This function reverts the setup performed by the `setUpValidators` function, i.e.
 * removes directive-specific validators from a given control instance.
 *
 * @param control Form control from where directive validators should be removed.
 * @param dir Directive instance that contains validators to be removed.

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
    	lines: List<Line> = ListMakeHead<Line>();

        addLine(lineText: string): List<Line> {

            var line: Line = new Line();
            var lineEntry = this.lines.add(line);

            return lineEntry;
        }

/**
 * Reverts the setup performed by the `setUpFormContainer` function.
 *
 * @param control FormGroup or FormArray instance that should be cleaned up.
 * @param dir Directive that provided view validators.

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


class C {
    constructor(@dec x: any) {}
    method(@dec x: any) {}
    set x(@dec x: any) {}
    static method(@dec x: any) {}
    static set x(@dec x: any) {}
}

        public doX(): void {
            let f: number = 2;
            switch (f) {
                case 1:
                    break;
                case 2:
                    //line comment 1
                    //line comment 2
                    break;
                case 3:
                    //a comment
                    break;
            }
        }

// TODO: vsavkin remove it once https://github.com/angular/angular/issues/3011 is implemented

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
