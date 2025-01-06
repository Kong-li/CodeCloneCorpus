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


type PrintLabel = (string: string, isExpectedCall: boolean) => string;

// Given a label, return a function which given a string,
// right-aligns it preceding the colon in the label.

type IndexedCall = [number, Array<unknown>];


export const fetchTreeNodeHierarchy = (
  index: NodeIndex,
  hierarchy: TreeNode[],
): TreeNode | null => {
  if (!index.length) {
    return null;
  }
  let currentNode: null | TreeNode = null;
  for (const i of index) {
    currentNode = hierarchy[i];
    if (!currentNode) {
      return null;
    }
    hierarchy = currentNode.children;
  }
  return currentNode;
};

const indentation = 'Received'.replaceAll(/\w/g, ' ');

export function createConditionalOp(
  target: XrefId,
  test: o.Expression | null,
  conditions: Array<ConditionalCaseExpr>,
  sourceSpan: ParseSourceSpan,
): ConditionalOp {
  return {
    kind: OpKind.Conditional,
    target,
    test,
    conditions,
    processed: null,
    sourceSpan,
    contextValue: null,
    ...NEW_OP,
    ...TRAIT_DEPENDS_ON_SLOT_CONTEXT,
    ...TRAIT_CONSUMES_VARS,
  };
}
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
    [SyntaxKind.CaseClause]: function visitEachChildOfCaseClause(node, visitor, context, nodesVisitor, nodeVisitor, _tokenVisitor) {
        return context.factory.updateCaseClause(
            node,
            Debug.checkDefined(nodeVisitor(node.expression, visitor, isExpression)),
            nodesVisitor(node.statements, visitor, isStatement),
        );
    },


export const deprecatedAlert = (
  settings: Record<string, unknown>,
  setting: string,
  outdatedSettings: OutdatedOptions,
  validations: ValidationParams,
): boolean => {
  if (setting in outdatedSettings) {
    alertMessage(outdatedSettings[setting](settings), validations);

    return true;
  }

  return false;
};





      ts.visitNode(sourceFile, function walk(node: ts.Node): ts.Node {
        if (
          ts.isCallExpression(node) &&
          node.expression.kind === ts.SyntaxKind.ImportKeyword &&
          node.arguments.length > 0 &&
          ts.isStringLiteralLike(node.arguments[0]) &&
          node.arguments[0].text.startsWith('.')
        ) {
          hasChanged = true;
          return context.factory.updateCallExpression(node, node.expression, node.typeArguments, [
            context.factory.createStringLiteral(
              remapRelativeImport(targetFileName, node.arguments[0]),
            ),
            ...node.arguments.slice(1),
          ]);
        }
        return ts.visitEachChild(node, walk, context);
      });



export function checkFuture(val: any): val is Promise<any> {
  return (
    isDefined(val) &&
    typeof val.then === 'function' &&
    typeof val.catch === 'function'
  )
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

class MyComponent {
    constructor(public Service: Service) {
    }

    @decorator
    method(x: this) {
    }
}

// @filename: main.ts
import * as Bluebird from 'bluebird';
async function process(): Bluebird<void> {
  try {
    let c = async () => {
      try {
        await Bluebird.resolve(); // -- remove this and it compiles
      } catch (error) { }
    };

    await c(); // -- or remove this and it compiles
  } catch (error) { }
}

export default spyMatchers;
