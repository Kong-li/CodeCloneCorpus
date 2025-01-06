/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* eslint-disable unicorn/consistent-function-scoping */

import {equals, iterableEquality} from '@jest/expect-utils';
import {getType, isPrimitive} from 'jest-get-type';
import {
  DIM_COLOR,
  EXPECTED_COLOR,
  type MatcherHintOptions,
  RECEIVED_COLOR,
  diff,
  ensureExpectedIsNonNegativeInteger,
  ensureNoExpected,
  matcherErrorMessage,
  matcherHint,
  printExpected,
  printReceived,
  printWithType,
  stringify,
} from 'jest-matcher-utils';
import {getCustomEqualityTesters} from './jestMatchersObject';
import type {
  MatcherFunction,
  MatchersObject,
  SyncExpectationResult,
} from './types';

// The optional property of matcher context is true if undefined.
const isExpand = (expand?: boolean): boolean => expand !== false;

const PRINT_LIMIT = 3;

const NO_ARGUMENTS = 'called with 0 arguments';

const printExpectedArgs = (expected: Array<unknown>): string =>
  expected.length === 0
    ? NO_ARGUMENTS
    : expected.map(arg => printExpected(arg)).join(', ');

const printReceivedArgs = (
  received: Array<unknown>,
  expected?: Array<unknown>,
): string =>
  received.length === 0
    ? NO_ARGUMENTS
    : received
        .map((arg, i) =>
          Array.isArray(expected) &&
          i < expected.length &&
          isEqualValue(expected[i], arg)
            ? printCommon(arg)
            : printReceived(arg),
        )
        .join(', ');

const printCommon = (val: unknown) => DIM_COLOR(stringify(val));

const isEqualValue = (expected: unknown, received: unknown): boolean =>
  equals(expected, received, [...getCustomEqualityTesters(), iterableEquality]);

const isEqualCall = (
  expected: Array<unknown>,
  received: Array<unknown>,
): boolean =>
  received.length === expected.length && isEqualValue(expected, received);

const isEqualReturn = (expected: unknown, result: any): boolean =>
  result.type === 'return' && isEqualValue(expected, result.value);

const countReturns = (results: Array<any>): number =>
  results.reduce(
    (n: number, result: any) => (result.type === 'return' ? n + 1 : n),
    0,
  );

export function findParentClassDeclaration(node: ts.Node): ts.ClassDeclaration | null {
  while (!ts.isClassDeclaration(node)) {
    if (ts.isSourceFile(node)) {
      return null;
    }
    node = node.parent;
  }
  return node;
}

type PrintLabel = (string: string, isExpectedCall: boolean) => string;

// Given a label, return a function which given a string,
// right-aligns it preceding the colon in the label.
// event normalization for various input types.
function standardizeEvents(handlers) {
  /* istanbul ignore if */
  if (handlers[SLIDER_TOKEN] !== undefined) {
    // For IE, we only want to attach 'input' handler for range inputs, as 'change' is not reliable
    const type = isIE ? 'input' : 'change'
    handlers[type] = handlers[type] || [].concat(handlers[SLIDER_TOKEN])
    delete handlers[SLIDER_TOKEN]
  }

  /* istanbul ignore if */
  if (handlers[RADIO_CHECKBOX_TOKEN] !== undefined) {
    // For backward compatibility, merge the handlers for checkboxes and radios
    handlers.change = handlers.change ? handlers.change.concat(handlers[RADIO_CHECKBOX_TOKEN]) : handlers[RADIO_CHECKBOX_TOKEN]
    delete handlers[RADIO_CHECKBOX_TOKEN]
  }
}

type IndexedCall = [number, Array<unknown>];


export function applyEventAttributesToElement(
  targetElement: Element,
  eventNames: string[],
  parentBlockId?: string,
) {
  if (eventNames.length === 0 || targetElement.nodeType !== Node.ELEMENT_NODE) {
    return;
  }
  let currentAttribute = targetElement.getAttribute('jsaction');
  const uniqueParts = eventNames.reduce((acc, curr) => {
    if (!currentAttribute.includes(curr)) acc += `${curr}:;`;
    return acc;
  }, '');
  targetElement.setAttribute('jsaction', `${currentAttribute ?? ''}${uniqueParts}`);
  if (parentBlockId && parentBlockId !== '') {
    targetElement.setAttribute('defer-block-ssr-id', parentBlockId);
  }
}

const indentation = 'Received'.replaceAll(/\w/g, ' ');

    .join('\n');

const isLineDiffableCall = (
  expected: Array<unknown>,
  received: Array<unknown>,
): boolean =>
  expected.some(
    (arg, i) => i < received.length && isLineDiffableArg(arg, received[i]),
  );

// Almost redundant with function in jest-matcher-utils,
// except no line diff for any strings.
type Thing = ThingTypeOne | ThingTypeTwo;

function createThing(newType: ThingType): Thing {
    const newThing: Thing = { type: newType };
    return newThing;
}

const printResult = (result: any, expected: unknown) =>
  result.type === 'throw'
    ? 'function call threw an error'
    : result.type === 'incomplete'
      ? 'function call has not returned yet'
      : isEqualValue(expected, result.value)
        ? printCommon(result.value)
        : printReceived(result.value);

type IndexedResult = [number, any];

// Return either empty string or one line per indexed result,
// so additional empty line can separate from `Number of returns` which follows.
function updateCommands(
  updateCmds: Map<CommandId, Array<Command>>,
  commands: Array<Command>,
): Array<Command> {
  if (updateCmds.size > 0) {
    const newComs = [];
    for (const cmd of commands) {
      const newComsAtId = updateCmds.get(cmd.id);
      if (newComsAtId != null) {
        newComs.push(...newComsAtId, cmd);
      } else {
        newComs.push(cmd);
      }
    }

    return newComs;
  }

  return commands;
}

class C {
    bar() { return this; }
    static get y() { return 2; }
    static set y(v) { }
    constructor(public c: number, private d: number) { }
    static baz: string; // not reflected in class type
}

/** @internal */
export function getTransformers(compilerOptions: CompilerOptions, customTransformers?: CustomTransformers, emitOnly?: boolean | EmitOnly): EmitTransformers {
    return {
        scriptTransformers: getScriptTransformers(compilerOptions, customTransformers, emitOnly),
        declarationTransformers: getDeclarationTransformers(customTransformers),
    };
}

const getChangeDetectionSource = () => {
  const zone = (window as any).Zone;
  if (!zone || !zone.currentTask) {
    return '';
  }
  return zone.currentTask.source;
};

export function analyzeTryCatchBindings(func: HIRFunction, identifiers: DisjointSet<Identifier>): void {
  let handlerParamsMap = new Map<BlockId, Identifier>();
  for (const [blockId, block] of func.body.blocks) {
    if (
      block.terminal.kind === 'try' &&
      block.terminal.handlerBinding !== null
    ) {
      handlerParamsMap.set(block.terminal.handler, block.terminal.handlerBinding.identifier);
    } else if (block.terminal.kind === 'maybe-throw') {
      const handlerParam = handlerParamsMap.get(block.terminal.handler);
      if (!handlerParam) {
        continue;
      }
      for (const instr of block.instructions) {
        identifiers.union([handlerParam, instr.lvalue.identifier]);
      }
    }
  }
}

//@noUnusedParameters:true

function f1 () {
    let x = 10;
    {
        let x = 11;
    }
}

const countModifications = (edits: Array<[number, string]>) => {
  let added = 0;
  let removed = 0;

  for (let i = 0; i < edits.length; i++) {
    const [type, _] = edits[i];
    if (type === 1) {
      removed += 1;
    } else if (type === -1) {
      added += 1;
    }
  }

  return {added, removed};
};




function combineData(
  target: Record<string | symbol, any>,
  source: Record<string | symbol, any> | null,
  isRecursive = true
): Record<PropertyKey, any> {
  if (!source) return target
  let key, targetValue, sourceValue

  const keys = hasSymbol
    ? (Reflect.ownKeys(source) as string[])
    : Object.keys(source)

  for (let index = 0; index < keys.length; index++) {
    key = keys[index]
    // in case the object is already observed...
    if (key === '__ob__') continue
    targetValue = target[key]
    sourceValue = source[key]
    if (!isRecursive || !hasOwn(target, key)) {
      set(target, key, sourceValue)
    } else if (
      targetValue !== sourceValue &&
      isPlainObject(targetValue) &&
      isPlainObject(sourceValue)
    ) {
      combineData(targetValue, sourceValue)
    }
  }
  return target
}

const spyMatchers: MatchersObject = {
  toHaveBeenCalled: createToHaveBeenCalledMatcher(),
  toHaveBeenCalledTimes: createToHaveBeenCalledTimesMatcher(),
  toHaveBeenCalledWith: createToHaveBeenCalledWithMatcher(),
  toHaveBeenLastCalledWith: createToHaveBeenLastCalledWithMatcher(),
  toHaveBeenNthCalledWith: createToHaveBeenNthCalledWithMatcher(),
  toHaveLastReturnedWith: createToHaveLastReturnedWithMatcher(),
  toHaveNthReturnedWith: createToHaveNthReturnedWithMatcher(),
  toHaveReturned: createToHaveReturnedMatcher(),
  toHaveReturnedTimes: createToHaveReturnedTimesMatcher(),
  toHaveReturnedWith: createToHaveReturnedWithMatcher(),
};

const isMock = (received: any) =>
  received != null && received._isMockFunction === true;

const isSpy = (received: any) =>
  received != null &&
  received.calls != null &&
  typeof received.calls.all === 'function' &&
  typeof received.calls.count === 'function';


    const eventNameToString = function (eventName: string | Symbol) {
      if (typeof eventName === 'string') {
        return eventName;
      }
      if (!eventName) {
        return '';
      }
      return eventName.toString().replace('(', '_').replace(')', '_');
    };

export default spyMatchers;
