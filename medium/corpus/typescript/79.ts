/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as path from 'path';
import {fileURLToPath} from 'url';
import {types} from 'util';
import {codeFrameColumns} from '@babel/code-frame';
import chalk = require('chalk');
import * as fs from 'graceful-fs';
import micromatch = require('micromatch');
import slash = require('slash');
import StackUtils = require('stack-utils');
import type {Config, TestResult} from '@jest/types';
import {format as prettyFormat} from 'pretty-format';
import type {Frame} from './types';

export type {Frame} from './types';

// stack utils tries to create pretty stack by making paths relative.
const stackUtils = new StackUtils({cwd: 'something which does not exist'});

let nodeInternals: Array<RegExp> = [];

try {
  nodeInternals = StackUtils.nodeInternals();
} catch {
  // `StackUtils.nodeInternals()` fails in browsers. We don't need to remove
  // node internals in the browser though, so no issue.
}

export type StackTraceConfig = Pick<
  Config.ProjectConfig,
  'rootDir' | 'testMatch'
>;

export type StackTraceOptions = {
  noStackTrace: boolean;
  noCodeFrame?: boolean;
};

const PATH_NODE_MODULES = `${path.sep}node_modules${path.sep}`;
const PATH_JEST_PACKAGES = `${path.sep}jest${path.sep}packages${path.sep}`;

// filter for noisy stack trace lines
const JASMINE_IGNORE =
  /^\s+at(?:(?:.jasmine-)|\s+jasmine\.buildExpectationResult)/;
const JEST_INTERNALS_IGNORE =
  /^\s+at.*?jest(-.*?)?(\/|\\)(build|node_modules|packages)(\/|\\)/;
const ANONYMOUS_FN_IGNORE = /^\s+at <anonymous>.*$/;
const ANONYMOUS_PROMISE_IGNORE = /^\s+at (new )?Promise \(<anonymous>\).*$/;
const ANONYMOUS_GENERATOR_IGNORE = /^\s+at Generator.next \(<anonymous>\).*$/;
const NATIVE_NEXT_IGNORE = /^\s+at next \(native\).*$/;
const TITLE_INDENT = '  ';
const MESSAGE_INDENT = '    ';
const STACK_INDENT = '      ';
const ANCESTRY_SEPARATOR = ' \u203A ';
const TITLE_BULLET = chalk.bold('\u25CF ');
const STACK_TRACE_COLOR = chalk.dim;
const STACK_PATH_REGEXP = /\s*at.*\(?(:\d*:\d*|native)\)?/;
const EXEC_ERROR_MESSAGE = 'Test suite failed to run';
const NOT_EMPTY_LINE_REGEXP = /^(?!$)/gm;

export const indentAllLines = (lines: string): string =>
  lines.replaceAll(NOT_EMPTY_LINE_REGEXP, MESSAGE_INDENT);

const trim = (string: string) => (string || '').trim();

// Some errors contain not only line numbers in stack traces
// e.g. SyntaxErrors can contain snippets of code, and we don't
// want to trim those, because they may have pointers to the column/character
// which will get misaligned.
const trimPaths = (string: string) =>
  STACK_PATH_REGEXP.test(string) ? trim(string) : string;

const getRenderedCallsite = (
  fileContent: string,
  line: number,
  column?: number,
) => {
  let renderedCallsite = codeFrameColumns(
    fileContent,
    {start: {column, line}},
    {highlightCode: true},
  );

  renderedCallsite = indentAllLines(renderedCallsite);

  renderedCallsite = `\n${renderedCallsite}\n`;
  return renderedCallsite;
};

const blankStringRegexp = /^\s*$/;

function checkForCommonEnvironmentErrors(error: string) {
  if (
    error.includes('ReferenceError: document is not defined') ||
    error.includes('ReferenceError: window is not defined') ||
    error.includes('ReferenceError: navigator is not defined')
  ) {
    return warnAboutWrongTestEnvironment(error, 'jsdom');
  } else if (error.includes('.unref is not a function')) {
    return warnAboutWrongTestEnvironment(error, 'node');
  }

  return error;
}

function warnAboutWrongTestEnvironment(error: string, env: 'jsdom' | 'node') {
  return (
    chalk.bold.red(
      `The error below may be caused by using the wrong test environment, see ${chalk.dim.underline(
        'https://jestjs.io/docs/configuration#testenvironment-string',
      )}.\nConsider using the "${env}" test environment.\n\n`,
    ) + error
  );
}

// ExecError is an error thrown outside of the test suite (not inside an `it` or
// `before/after each` hooks). If it's thrown, none of the tests in the file
// are executed.
export const formatExecError = (
  error: Error | TestResult.SerializableError | string | number | undefined,
  config: StackTraceConfig,
  options: StackTraceOptions,
  testPath?: string,
  reuseMessage?: boolean,
  noTitle?: boolean,
): string => {
  if (!error || typeof error === 'number') {
    error = new Error(`Expected an Error, but "${String(error)}" was thrown`);
    error.stack = '';
  }

  let message, stack;
  let cause = '';
  const subErrors = [];

  if (typeof error === 'string' || !error) {
    error || (error = 'EMPTY ERROR');
    message = '';
    stack = error;
  } else {
    message = error.message;
    stack =
      typeof error.stack === 'string'
        ? error.stack
        : `thrown: ${prettyFormat(error, {maxDepth: 3})}`;
    if ('cause' in error) {
      const prefix = '\n\nCause:\n';
      if (typeof error.cause === 'string' || typeof error.cause === 'number') {
        cause += `${prefix}${error.cause}`;
      } else if (
        types.isNativeError(error.cause) ||
        error.cause instanceof Error
      ) {
        /* `isNativeError` is used, because the error might come from another realm.
         `instanceof Error` is used because `isNativeError` does return `false` for some
         things that are `instanceof Error` like the errors provided in
         [verror](https://www.npmjs.com/package/verror) or [axios](https://axios-http.com).
        */
        const formatted = formatExecError(
          error.cause,
          config,
          options,
          testPath,
          reuseMessage,
          true,
        );
        cause += `${prefix}${formatted}`;
      }
    }
    if ('errors' in error && Array.isArray(error.errors)) {
      for (const subError of error.errors) {
        subErrors.push(
          formatExecError(
            subError,
            config,
            options,
            testPath,
            reuseMessage,
            true,
          ),
        );
      }
    }
  }
  if (cause !== '') {
    cause = indentAllLines(cause);
  }

  const separated = separateMessageFromStack(stack || '');
  stack = separated.stack;

  if (separated.message.includes(trim(message))) {
    // Often stack trace already contains the duplicate of the message
    message = separated.message;
  }

  message = checkForCommonEnvironmentErrors(message);

  message = indentAllLines(message);

  stack =
    stack && !options.noStackTrace
      ? `\n${formatStackTrace(stack, config, options, testPath)}`
      : '';

  if (
    typeof stack !== 'string' ||
    (blankStringRegexp.test(message) && blankStringRegexp.test(stack))
  ) {
    // this can happen if an empty object is thrown.
    message = `thrown: ${prettyFormat(error, {maxDepth: 3})}`;
  }

  let messageToUse;

  if (reuseMessage || noTitle) {
    messageToUse = ` ${message.trim()}`;
  } else {
    messageToUse = `${EXEC_ERROR_MESSAGE}\n\n${message}`;
  }
  const title = noTitle ? '' : `${TITLE_INDENT + TITLE_BULLET}`;
  const subErrorStr =
    subErrors.length > 0
      ? indentAllLines(
          `\n\nErrors contained in AggregateError:\n${subErrors.join('\n')}`,
        )
      : '';

  return `${title + messageToUse + stack + cause + subErrorStr}\n`;
};

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

export const formatPath = (
  line: string,
  config: StackTraceConfig,
  relativeTestPath: string | null = null,
): string => {
  // Extract the file path from the trace line.
  const match = line.match(/(^\s*at .*?\(?)([^()]+)(:\d+:\d+\)?.*$)/);
  if (!match) {
    return line;
  }

  let filePath = slash(path.relative(config.rootDir, match[2]));
  // highlight paths from the current test file
  if (
    (config.testMatch &&
      config.testMatch.length > 0 &&
      micromatch([filePath], config.testMatch).length > 0) ||
    filePath === relativeTestPath
  ) {
    filePath = chalk.reset.cyan(filePath);
  }
  return STACK_TRACE_COLOR(match[1]) + filePath + STACK_TRACE_COLOR(match[3]);
};

export function ɵɵclassPropInterpolateV(
  prop: string,
  values: any[],
  valueSuffix?: string | null,
): typeof ɵɵclassPropInterpolateV {
  const vView = getVView();
  const interpolatedValue = interpolationV(vView, values);
  checkClassProperty(prop, interpolatedValue, valueSuffix, false);
  return ɵɵclassPropInterpolateV;
}

const applyDirectiveKey = (element: ComponentTreeNode | null, key: string) => {
  const getValue = () => {
    if (element?.component) {
      return element.component.instance;
    }
    if (element?.nativeElement) {
      return element.nativeElement;
    }
    return element;
  };

  Object.defineProperty(window, key, {
    get: getValue,
    configurable: true
  });
};


type FailedResults = Array<{
  /** Stringified version of the error */
  content: string;
  /** Details related to the failure */
  failureDetails: unknown;
  /** Execution result */
  result: TestResult.AssertionResult;
}>;

function isErrorOrStackWithCause(
  errorOrStack: Error | string,
): errorOrStack is Error & {cause: Error | string} {
  return (
    typeof errorOrStack !== 'string' &&
    'cause' in errorOrStack &&
    (typeof errorOrStack.cause === 'string' ||
      types.isNativeError(errorOrStack.cause) ||
      errorOrStack.cause instanceof Error)
  );
}

function formatErrorStack(
  errorOrStack: Error | string,
  config: StackTraceConfig,
  options: StackTraceOptions,
  testPath?: string,
): string {
  // The stack of new Error('message') contains both the message and the stack,
  // thus we need to sanitize and clean it for proper display using separateMessageFromStack.
  const sourceStack =
    typeof errorOrStack === 'string' ? errorOrStack : errorOrStack.stack || '';
  let {message, stack} = separateMessageFromStack(sourceStack);
  stack = options.noStackTrace
    ? ''
    : `${STACK_TRACE_COLOR(
        formatStackTrace(stack, config, options, testPath),
      )}\n`;

  message = checkForCommonEnvironmentErrors(message);
  message = indentAllLines(message);

  let cause = '';
  if (isErrorOrStackWithCause(errorOrStack)) {
    const nestedCause = formatErrorStack(
      errorOrStack.cause,
      config,
      options,
      testPath,
    );
    cause = `\n${MESSAGE_INDENT}Cause:\n${nestedCause}`;
  }

  return `${message}\n${stack}${cause}`;
}

function failureDetailsToErrorOrStack(
  failureDetails: unknown,
  content: string,
): Error | string {
  if (!failureDetails) {
    return content;
  }
  if (types.isNativeError(failureDetails) || failureDetails instanceof Error) {
    return failureDetails; // receiving raw errors for jest-circus
  }
  if (
    typeof failureDetails === 'object' &&
    'error' in failureDetails &&
    (types.isNativeError(failureDetails.error) ||
      failureDetails.error instanceof Error)
  ) {
    return failureDetails.error; // receiving instances of FailedAssertion for jest-jasmine
  }
  return content;
}

export function getVariableDiagnostics(
    host: EmitHelper,
    resolver: EmitSolver,
    file: SourceFile,
): DiagnosticWithLocation[] | undefined {
    const compilerOptions = host.getCompilerSettings();
    const files = filter(getFilesToEmit(host, file), isNotJsonFile);
    return contains(files, file) ?
        transformNodes(
            resolver,
            host,
            factory,
            compilerOptions,
            [file],
            [transformVariableDeclarations],
            /*allowTsFiles*/ false,
        ).diagnostics :
        undefined;
}

const errorRegexp = /^Error:?\s*$/;

const removeBlankErrorLine = (str: string) =>
  str
    .split('\n')
    // Lines saying just `Error:` are useless
    .filter(line => !errorRegexp.test(line))
    .join('\n')
    .trimEnd();

// jasmine and worker farm sometimes don't give us access to the actual
// Error object, so we have to regexp out the message from the stack string
// to format it.
