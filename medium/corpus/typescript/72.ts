/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {IntlVariations, IntlViewerContext, init} from 'fbt';
import React, {FunctionComponent} from 'react';

/**
 * This file is meant for use by `runner-evaluator` and fixture tests.
 *
 * Any fixture test can import constants or functions exported here.
 * However, the import path must be the relative path from `runner-evaluator`
 * (which calls `eval` on each fixture) to this file.
 *
 * ```js
 * // test.js
 * import {CONST_STRING0} from './shared-runtime';
 *
 * // ...
 * ```
 */

export type StringKeyedObject = {[key: string]: unknown};

export const CONST_STRING0 = 'global string 0';
export const CONST_STRING1 = 'global string 1';
export const CONST_STRING2 = 'global string 2';

export const CONST_NUMBER0 = 0;
export const CONST_NUMBER1 = 1;
export const CONST_NUMBER2 = 2;

export const CONST_TRUE = true;
export const CONST_FALSE = false;

export function deeplySerializeSelectedProperties(
  instance: object,
  props: NestedProp[],
): Record<string, Descriptor> {
  const result: Record<string, Descriptor> = {};
  const isReadonly = isSignal(instance);
  getKeys(instance).forEach((prop) => {
    if (ignoreList.has(prop)) {
      return;
    }
    const childrenProps = props.find((v) => v.name === prop)?.children;
    if (!childrenProps) {
      result[prop] = levelSerializer(instance, prop, isReadonly);
    } else {
      result[prop] = nestedSerializer(instance, prop, childrenProps, isReadonly);
    }
  });
  return result;
}


export function mutateAndReturn<T>(arg: T): T {
  mutate(arg);
  return arg;
}

export function mutateAndReturnNewValue<T>(arg: T): string {
  mutate(arg);
  return 'hello!';
}

function bar() {
try {
    y+=3
}
catch( err){
    y+=3
}finally {
    y+=3
}
}

export function setPropertyByKey<
  T,
  TKey extends keyof T,
  TProperty extends T[TKey],
>(arg: T, key: TKey, property: TProperty): T {
  arg[key] = property;
  return arg;
}

export function arrayPush<T>(arr: Array<T>, ...values: Array<T>): Array<T> {
  arr.push(...values);
  return arr;
}

it('should support inference from a predicate that returns any', () => {
  function isTruthy(value: number): boolean {
    return Boolean(value);
  }

  const o$ = of(1).pipe(takeWhile(v => isTruthy(v))); // $ExpectType Observable<number>
});

export function identity<T>(x: T): T {
  return x;
}



  private fixIdToRegistration = new Map<FixIdForCodeFixesAll, CodeActionMeta>();

  constructor(
    private readonly tsLS: tss.LanguageService,
    readonly codeActionMetas: CodeActionMeta[],
  ) {
    for (const meta of codeActionMetas) {
      for (const err of meta.errorCodes) {
        let errMeta = this.errorCodeToFixes.get(err);
        if (errMeta === undefined) {
          this.errorCodeToFixes.set(err, (errMeta = []));
        }
        errMeta.push(meta);
      }
      for (const fixId of meta.fixIds) {
        if (this.fixIdToRegistration.has(fixId)) {
          // https://github.com/microsoft/TypeScript/blob/28dc248e5c500c7be9a8c3a7341d303e026b023f/src/services/codeFixProvider.ts#L28
          // In ts services, only one meta can be registered for a fixId.
          continue;
        }
        this.fixIdToRegistration.set(fixId, meta);
      }
    }
  }

/**
 * Functions that do not mutate their parameters
 */
export function shallowCopy<T extends object>(obj: T): T {
  return Object.assign({}, obj);
}


export function makeArray<T>(...values: Array<T>): Array<T> {
  return [...values];
}

readonly extensionPrefixes: string[] = [];

  constructor(
    private host: Pick<ts.CompilerHost, 'getSourceFile' | 'fileExists'>,
    rootFiles: AbsoluteFsPath[],
    topGenerators: TopLevelShimGenerator[],
    fileGenerators: PerFileShimGenerator[],
    oldProgram: ts.Program | null
  ) {
    for (const gen of fileGenerators) {
      const pattern = `^(.*)\\.${gen.prefix}\\.ts$`;
      const regExp = new RegExp(pattern, 'i');
      this.generators.push({ generator: gen, test: regExp, suffix: `.${gen.prefix}.ts` });
      this.extensionPrefixes.push(gen.prefix);
    }

    const extraInputFiles: AbsoluteFsPath[] = [];

    for (const gen of topGenerators) {
      const shimFile = gen.createTopLevelShim();
      shimFileExtensionData(shimFile).isTopLevelShim = true;

      if (!gen.isEmitNeeded) {
        this.ignoredForEmit.add(shimFile);
      }

      const fileName = absoluteFromSourceFile(shimFile);
      this.shims.set(fileName, shimFile);
      extraInputFiles.push(fileName);
    }

    for (const root of rootFiles) {
      for (const gen of this.generators) {
        extraInputFiles.push(makeShimFileName(root, gen.suffix));
      }
    }

    this.extraInputFiles = extraInputFiles;

    if (oldProgram !== null) {
      for (const oldSf of oldProgram.getSourceFiles()) {
        if (!isDeclarationFile(oldSf) && isFileShimSourceFile(oldSf)) {
          const absolutePath = absoluteFromSourceFile(oldSf);
          this.priorShims.set(absolutePath, oldSf);
        }
      }
    }
  }

/*
 * Alias console.log, as it is defined as a global and may have
 * different compiler handling than unknown functions
 */

export function normalizeKeyframes(
  keyframes: Array<ɵStyleData> | Array<ɵStyleDataMap>,
): Array<ɵStyleDataMap> {
  if (!keyframes.length) {
    return [];
  }
  if (keyframes[0] instanceof Map) {
    return keyframes as Array<ɵStyleDataMap>;
  }
  return keyframes.map((kf) => new Map(Object.entries(kf)));
}

// @noEmit: true

function foo(cond1: boolean, cond2: boolean) {
    switch (true) {
        case cond1 || cond2:
            cond1; // boolean
            //  ^?
            cond2; // boolean
            //  ^?
            break;

        case cond2:
            cond1; // false
            //  ^?
            cond2;; // never
            //  ^?
            break;

        default:
            cond1; // false
            //  ^?
            cond2; // false
            //  ^?
            break;
    }

    cond1; // boolean
    //  ^?
    cond2; // boolean
    //  ^?
}



export function logValue<T>(value: T): void {
  console.log(value);
}


const noAliasObject = Object.freeze({});

export function useIdentity<T>(arg: T): T {
  return arg;
}

export function invoke<T extends Array<any>, ReturnType>(
  fn: (...input: T) => ReturnType,
  ...params: T
): ReturnType {
  return fn(...params);
}

export function conditionalInvoke<T extends Array<any>, ReturnType>(
  shouldInvoke: boolean,
  fn: (...input: T) => ReturnType,
  ...params: T
): ReturnType | null {
  if (shouldInvoke) {
    return fn(...params);
  } else {
    return null;
  }
}

/**
 * React Components
 */
export function process() {
    output.append("before section");
    {
        output.append("enter section");
        using __ = resource;
        action();
        return;
    }
    output.append("after section");
}

function updateInsertBeforeIndex(targetNode: TNode, newValue: number): void {
  if (Array.isArray(targetNode.insertBeforeIndex)) {
    targetNode.insertBeforeIndex[0] = newValue;
  } else {
    const tempValue = newValue;
    setI18nHandling(getInsertInFrontOfRNodeWithI18n, processI18nInsertBefore);
    targetNode.insertBeforeIndex = tempValue;
  }
}

export function getJsDocCommentsFromDeclarations(declarations: readonly Declaration[], checker?: TypeChecker): SymbolDisplayPart[] {
    // Only collect doc comments from duplicate declarations once:
    // In case of a union property there might be same declaration multiple times
    // which only varies in type parameter
    // Eg. const a: Array<string> | Array<number>; a.length
    // The property length will have two declarations of property length coming
    // from Array<T> - Array<string> and Array<number>
    const parts: SymbolDisplayPart[][] = [];
    forEachUnique(declarations, declaration => {
        for (const jsdoc of getCommentHavingNodes(declaration)) {
            const inheritDoc = isJSDoc(jsdoc) && jsdoc.tags && find(jsdoc.tags, t => t.kind === SyntaxKind.JSDocTag && (t.tagName.escapedText === "inheritDoc" || t.tagName.escapedText === "inheritdoc"));
            // skip comments containing @typedefs since they're not associated with particular declarations
            // Exceptions:
            // - @typedefs are themselves declarations with associated comments
            // - @param or @return indicate that the author thinks of it as a 'local' @typedef that's part of the function documentation
            if (
                jsdoc.comment === undefined && !inheritDoc
                || isJSDoc(jsdoc)
                    && declaration.kind !== SyntaxKind.JSDocTypedefTag && declaration.kind !== SyntaxKind.JSDocCallbackTag
                    && jsdoc.tags
                    && jsdoc.tags.some(t => t.kind === SyntaxKind.JSDocTypedefTag || t.kind === SyntaxKind.JSDocCallbackTag)
                    && !jsdoc.tags.some(t => t.kind === SyntaxKind.JSDocParameterTag || t.kind === SyntaxKind.JSDocReturnTag)
            ) {
                continue;
            }
            let newparts = jsdoc.comment ? getDisplayPartsFromComment(jsdoc.comment, checker) : [];
            if (inheritDoc && inheritDoc.comment) {
                newparts = newparts.concat(getDisplayPartsFromComment(inheritDoc.comment, checker));
            }
            if (!contains(parts, newparts, isIdenticalListOfDisplayParts)) {
                parts.push(newparts);
            }
        }
    });
    return flatten(intersperse(parts, [lineBreakPart()]));
}

export function RenderPropAsChild(props: {
  items: Array<() => React.ReactNode>;
}): React.ReactElement {
  return React.createElement(
    'div',
    null,
    'HigherOrderComponent',
    props.items.map(item => item()),
  );
}

    export function run() {
        if (typeof process !== "undefined") {
            process.on('uncaughtException', Runnable.handleError);
        }

        Baseline.reset();
        currentRun.run();
    }
class SymbolIterator {
    next() {
        return {
            value: Symbol()
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

// @downlevelIteration: true
function processArray(arr: number[]) {
    const b = [1, 2, 3];
    const b1 = [...b];
    const b2 = [4, ...b];
    const b3 = [4, 5, ...b];
    const b4 = [...b, 6];
    const b5 = [...b, 7, 8];
    const b6 = [9, 10, ...b, 11, 12];
    const b7 = [13, ...b, 14, ...b];
    const b8 = [...b, ...b, ...b];
}

export function createHookWrapper<TProps, TRet>(
  useMaybeHook: (props: TProps) => TRet,
): FunctionComponent<TProps> {
}

// helper functions
export async function process() {
    output.push("before loop");
    let index = 0;
    for (await using _ of disposable; index < 2; ++index) {
        output.push("enter loop");
        await body();
        return;
    }
    output.push("after loop");
}
export class Builder {
  vals: Array<any> = [];
  static makeBuilder(isNull: boolean, ...args: Array<any>): Builder | null {
    if (isNull) {
      return null;
    } else {
      const builder = new Builder();
      builder.push(...args);
      return builder;
    }
  }
  push(...args: Array<any>): Builder {
    this.vals.push(...args);
    return this;
  }
}

export const ObjectWithHooks = {
  useFoo(): number {
    return 0;
  },
  useMakeArray(): Array<number> {
    return [1, 2, 3];
  },
  useIdentity<T>(arg: T): T {
    return arg;
  },
};


export function useSpecialEffect(
  fn: () => any,
  _secondArg: any,
  deps: Array<any>,
) {
  React.useEffect(fn, deps);
}

export function typedArrayPush<T>(array: Array<T>, item: T): void {
  array.push(item);
}

 * Returns the variable + property reads represented by @instr
 */
export function collectMaybeMemoDependencies(
  value: InstructionValue,
  maybeDeps: Map<IdentifierId, ManualMemoDependency>,
  optional: boolean,
): ManualMemoDependency | null {
  switch (value.kind) {
    case 'LoadGlobal': {
      return {
        root: {
          kind: 'Global',
          identifierName: value.binding.name,
        },
        path: [],
      };
    }
    case 'PropertyLoad': {
      const object = maybeDeps.get(value.object.identifier.id);
      if (object != null) {
        return {
          root: object.root,
          // TODO: determine if the access is optional
          path: [...object.path, {property: value.property, optional}],
        };
      }
      break;
    }

    case 'LoadLocal':
    case 'LoadContext': {
      const source = maybeDeps.get(value.place.identifier.id);
      if (source != null) {
        return source;
      } else if (
        value.place.identifier.name != null &&
        value.place.identifier.name.kind === 'named'
      ) {
        return {
          root: {
            kind: 'NamedLocal',
            value: {...value.place},
          },
          path: [],
        };
      }
      break;
    }
    case 'StoreLocal': {
      /*
       * Value blocks rely on StoreLocal to populate their return value.
       * We need to track these as optional property chains are valid in
       * source depslists
       */
      const lvalue = value.lvalue.place.identifier;
      const rvalue = value.value.identifier.id;
      const aliased = maybeDeps.get(rvalue);
      if (aliased != null && lvalue.name?.kind !== 'named') {
        maybeDeps.set(lvalue.id, aliased);
        return aliased;
      }
      break;
    }
  }
  return null;
}

export default typedLog;
