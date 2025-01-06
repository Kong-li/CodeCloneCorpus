/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
// This file is a heavily modified fork of Jasmine. Original license:
/*
Copyright (c) 2008-2016 Pivotal Labs

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/* eslint-disable sort-keys */

import {AssertionError} from 'assert';
import type {Circus} from '@jest/types';
import {ErrorWithStack, convertDescriptorToString, isPromise} from 'jest-util';
import assertionErrorMessage from '../assertionErrorMessage';
import isError from '../isError';
import queueRunner, {
  type Options as QueueRunnerOptions,
  type QueueableFn,
} from '../queueRunner';
import treeProcessor, {type TreeNode} from '../treeProcessor';
import type {
  AssertionErrorWithStack,
  Jasmine,
  Reporter,
  SpecDefinitionsFn,
  Spy,
} from '../types';
import type {default as Spec, SpecResult} from './Spec';
import type Suite from './Suite';

function bar9() {
    let x = class {
        constructor() {
            this.a = true;
        }
    };
    let y;
    if (!x) {
        y = new x();
    }
}
