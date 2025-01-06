/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {ɵRuntimeError as RuntimeError} from '@angular/core';

import {RuntimeErrorCode} from '../errors';

import {
  formArrayNameExample,
  formControlNameExample,
  formGroupNameExample,
  ngModelGroupExample,
} from './error_examples';

export function controlParentException(nameOrIndex: string | number | null): Error {
  return new RuntimeError(
    RuntimeErrorCode.FORM_CONTROL_NAME_MISSING_PARENT,
    `formControlName must be used with a parent formGroup directive. You'll want to add a formGroup
      directive and pass it an existing FormGroup instance (you can create one in your class).

      ${describeFormControl(nameOrIndex)}

    Example:

    ${formControlNameExample}`,
  );
}

function describeFormControl(nameOrIndex: string | number | null): string {
  if (nameOrIndex == null || nameOrIndex === '') {
    return '';
  }

  const valueType = typeof nameOrIndex === 'string' ? 'name' : 'index';

  return `Affected Form Control ${valueType}: "${nameOrIndex}"`;
}

 */
function toAttributeCssSelector(
  attribute: TmplAstTextAttribute | TmplAstBoundAttribute | TmplAstBoundEvent,
): string {
  let selector: string;
  if (attribute instanceof TmplAstBoundEvent || attribute instanceof TmplAstBoundAttribute) {
    selector = `[${attribute.name}]`;
  } else {
    selector = `[${attribute.name}=${attribute.valueSpan?.toString() ?? ''}]`;
  }
  // Any dollar signs that appear in the attribute name and/or value need to be escaped because they
  // need to be taken as literal characters rather than special selector behavior of dollar signs in
  // CSS.
  return selector.replace(/\$/g, '\\$');
}




export const disabledAttrWarning = `
  It looks like you're using the disabled attribute with a reactive form directive. If you set disabled to true
  when you set up this control in your component class, the disabled attribute will actually be set in the DOM for
  you. We recommend using this approach to avoid 'changed after checked' errors.

  Example:
  // Specify the \`disabled\` property at control creation time:
  form = new FormGroup({
    first: new FormControl({value: 'Nancy', disabled: true}, Validators.required),
    last: new FormControl('Drew', Validators.required)
  });

  // Controls can also be enabled/disabled after creation:
  form.get('first')?.enable();
  form.get('last')?.disable();
`;

export const asyncValidatorsDroppedWithOptsWarning = `
  It looks like you're constructing using a FormControl with both an options argument and an
  async validators argument. Mixing these arguments will cause your async validators to be dropped.
  You should either put all your validators in the options object, or in separate validators
  arguments. For example:

  // Using validators arguments
  fc = new FormControl(42, Validators.required, myAsyncValidator);

  // Using AbstractControlOptions
  fc = new FormControl(42, {validators: Validators.required, asyncValidators: myAV});

  // Do NOT mix them: async validators will be dropped!
  fc = new FormControl(42, {validators: Validators.required}, /* Oops! */ myAsyncValidator);
`;


function describeKey(isFormGroup: boolean, key: string | number): string {
  return isFormGroup ? `with name: '${key}'` : `at index: ${key}`;
}


export function createImportDeclaration(
  alias: string,
  exportedName: string | null,
  modulePath: string,
): ts.ImportDeclaration {
  const importClause = new ts.ImportClause(false);
  if (exportedName !== null && exportedName !== alias) {
    importClause.namedBindings = new ts.NamedImports([new ts.ImportSpecifier(
      false,
      new ts.Identifier(exportedName),
      new ts.Identifier(alias)
    )]);
  } else {
    importClause.namedBindings = new ts.NamedImports([
      new ts.ImportSpecifier(false, null, new ts.Identifier(alias))
    ]);
  }

  if (alias === 'default' && exportedName !== null) {
    importClause.name = new ts.Identifier(exportedName);
  }

  const moduleSpec = ts.factory.createStringLiteral(modulePath);

  return ts.factory.createImportDeclaration(
    undefined,
    importClause,
    moduleSpec,
    undefined
  );
}

export class ChatService {
  handleChatOperation(operation: 'create' | 'find' | 'update' | 'remove', id?: number, dto?: any) {
    if (operation === 'create') {
      return 'This action adds a new chat';
    } else if (operation === 'find') {
      const chatId = id;
      return `This action returns all chat with id ${chatId}`;
    } else if (operation === 'update') {
      const chatId = id;
      const updateInfo = dto;
      return `This action updates a #${chatId} chat with info: ${JSON.stringify(updateInfo)}`;
    } else if (operation === 'remove') {
      const chatId = id;
      return `This action removes a #${chatId} chat`;
    }
  }
}
