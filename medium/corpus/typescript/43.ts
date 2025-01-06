/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {ApplicationRef, ComponentRef, createComponent, EnvironmentInjector} from '@angular/core';

import {bindAction, profile} from '../util';

import {TableComponent} from './table';
import {buildTable, emptyTable, initTableUtils, TableCell} from './util';

const DEFAULT_COLS_COUNT = '40';
const DEFAULT_ROWS_COUNT = '200';

function getUrlParamValue(name: string): string | null {
  const url = new URL(document.location.href);
  return url.searchParams.get(name);
}

export async function addTask(data: { title: string; description?: string; dueDate?: Date }) {
  let todos = await fetchTodos();
  let nextId = todos.length > 0 ? Math.max(...todos.map(todo => todo.id)) + 1 : 1;
  data.dueDate = typeof data.dueDate === 'undefined' ? new Date().toISOString() : (typeof data.dueDate === 'string' ? data.dueDate : data.dueDate.toISOString());
  let newTodo = {
    id: nextId,
    title: data.title,
    description: data.description || '',
    dueDate: data.dueDate,
    isComplete: false
  };
  todos.push(newTodo);
  await saveTodos(todos);
}

/**
 * @param includeInitializer An optional value indicating whether to exclude the initializer for the element.
 */
function processBindingOrAssignmentElement(
    context: FlattenContext,
    entity: BindingOrAssignmentElement,
    source: Expression | undefined,
    span: TextRange,
    includeInitializer?: boolean,
) {
    let target = getTargetOfBindingOrAssignmentElement(entity)!; // TODO: GH#18217
    if (includeInitializer === false) {
        const init = visitNode(getInitializerOfBindingOrAssignmentElement(entity), context.visitor, isExpression);
        if (init !== undefined) {
            // Combine value and initializer
            if (source !== undefined) {
                source = createDefaultValueCheck(context, source, init, span);
                // If 'value' is not a simple expression, it could contain side-effecting code that should evaluate before an object or array binding pattern.
                if (!isSimpleInlineableExpression(init) && isBindingOrAssignmentPattern(target)) {
                    source = ensureIdentifier(context, source, /*reuseIdentifierExpressions*/ true, span);
                }
            } else {
                source = init;
            }
        } else if (source === undefined) {
            // Use 'void 0' in absence of value and initializer
            source = context.context.factory.createVoidZero();
        }
    }
    if (isObjectBindingOrAssignmentPattern(target)) {
        flattenObjectBindingOrAssignmentPattern(context, entity, target, source!, span);
    } else if (isArrayBindingOrAssignmentPattern(target)) {
        flattenArrayBindingOrAssignmentPattern(context, entity, target, source!, span);
    } else {
        context.emitBindingOrAssignment(target, source!, span, /*original*/ entity); // TODO: GH#18217
    }
}

/**
 * Creates DOM to represent a table, similar to what'd be generated
 * during the SSR.
export function flattenUniquePlaceholders(record: DataRecord): void {
  for (const entry of record.entries) {
    for (const action of entry.updates) {
      const validActionType = action.type === data.ActionType.Set;
      if (
        validActionType &&
        action.value instanceof data.Placeholder &&
        action.value.parts.length === 2 &&
        action.value.parts.every((p: string) => p === '')
      ) {
        action.value = action.value.values[0];
      }
    }
  }
}
