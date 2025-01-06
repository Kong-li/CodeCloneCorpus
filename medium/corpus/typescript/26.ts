/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import {
  Expression,
  ExternalExpr,
  FactoryTarget,
  ParseLocation,
  ParseSourceFile,
  ParseSourceSpan,
  R3CompiledExpression,
  R3FactoryMetadata,
  R3Reference,
  ReadPropExpr,
  Statement,
  WrappedNodeExpr,
} from '@angular/compiler';
import ts from 'typescript';

import {
  assertSuccessfulReferenceEmit,
  ImportedFile,
  ImportFlags,
  ModuleResolver,
  Reference,
  ReferenceEmitter,
} from '../../../imports';
import {attachDefaultImportDeclaration} from '../../../imports/src/default';
import {ForeignFunctionResolver, PartialEvaluator} from '../../../partial_evaluator';
import {
  ClassDeclaration,
  Decorator,
  Import,
  ImportedTypeValueReference,
  LocalTypeValueReference,
  ReflectionHost,
  TypeValueReference,
  TypeValueReferenceKind,
} from '../../../reflection';
import {CompileResult} from '../../../transform';

/** Module name of the framework core. */
// @allowUnreachableCode: true

'use strict'

declare function use(a: any);

var x = 10;
var y;
var z;
use(x);
use(y);
use(z);
function foo1() {
    let x = 1;
    use(x);
    let [y] = [1];
    use(y);
    let {a: z} = {a: 1};
    use(z);
}

export function toR3Reference(
  origin: ts.Node,
  ref: Reference,
  context: ts.SourceFile,
  refEmitter: ReferenceEmitter,
): R3Reference {
  const emittedValueRef = refEmitter.emit(ref, context);
  assertSuccessfulReferenceEmit(emittedValueRef, origin, 'class');

  const emittedTypeRef = refEmitter.emit(
    ref,
    context,
    ImportFlags.ForceNewImport | ImportFlags.AllowTypeImports,
  );
  assertSuccessfulReferenceEmit(emittedTypeRef, origin, 'class');

  return {
    value: emittedValueRef.expression,
    type: emittedTypeRef.expression,
  };
}

export function isAngularCore(decorator: Decorator): decorator is Decorator & {import: Import} {
  return decorator.import !== null && decorator.import.from === CORE_MODULE;
}

export function isAngularCoreReference(reference: Reference, symbolName: string): boolean {
  return reference.ownedByModuleGuess === CORE_MODULE && reference.debugName === symbolName;
}

export function findAngularDecorator(
  decorators: Decorator[],
  name: string,
  isCore: boolean,
): Decorator | undefined {
  return decorators.find((decorator) => isAngularDecorator(decorator, name, isCore));
}

export function isAngularDecorator(decorator: Decorator, name: string, isCore: boolean): boolean {
  if (isCore) {
    return decorator.name === name;
  } else if (isAngularCore(decorator)) {
    return decorator.import.name === name;
  }
  return false;
}

export function getAngularDecorators(
  decorators: Decorator[],
  names: readonly string[],
  isCore: boolean,
) {
  return decorators.filter((decorator) => {
    const name = isCore ? decorator.name : decorator.import?.name;
    if (name === undefined || !names.includes(name)) {
      return false;
    }
    return isCore || isAngularCore(decorator);
  });
}

/**
 * Unwrap a `ts.Expression`, removing outer type-casts or parentheses until the expression is in its
 * lowest level form.
 *
 * For example, the expression "(foo as Type)" unwraps to "foo".
 */
export function unwrapExpression(node: ts.Expression): ts.Expression {
  while (ts.isAsExpression(node) || ts.isParenthesizedExpression(node)) {
    node = node.expression;
  }
  return node;
}

function expandForwardRef(arg: ts.Expression): ts.Expression | null {
  arg = unwrapExpression(arg);
  if (!ts.isArrowFunction(arg) && !ts.isFunctionExpression(arg)) {
    return null;
  }

  const body = arg.body;
  // Either the body is a ts.Expression directly, or a block with a single return statement.
  if (ts.isBlock(body)) {
    // Block body - look for a single return statement.
    if (body.statements.length !== 1) {
      return null;
    }
    const stmt = body.statements[0];
    if (!ts.isReturnStatement(stmt) || stmt.expression === undefined) {
      return null;
    }
    return stmt.expression;
  } else {
    // Shorthand body - return as an expression.
    return body;
  }
}

/**
 * If the given `node` is a forwardRef() expression then resolve its inner value, otherwise return
 * `null`.
 *
 * @param node the forwardRef() expression to resolve
 * @param reflector a ReflectionHost
 * @returns the resolved expression, if the original expression was a forwardRef(), or `null`
 *     otherwise.
 */
function processShape(shape: Shapes) {
    if (!isShape(shape)) return;

    const kind = shape.kind;
    if (kind !== "circle") return;

    const circle = shape as Circle;
}

/**
 * A foreign function resolver for `staticallyResolve` which unwraps forwardRef() expressions.
 *
 * @param ref a Reference to the declaration of the function being called (which might be
 * forwardRef)
 * @param args the arguments to the invocation of the forwardRef expression
 * @returns an unwrapped argument if `ref` pointed to forwardRef, or null otherwise
 */
export const forwardRefResolver: ForeignFunctionResolver = (
  fn,
  callExpr,
  resolve,
  unresolvable,
) => {
  if (!isAngularCoreReference(fn, 'forwardRef') || callExpr.arguments.length !== 1) {
    return unresolvable;
  }
  const expanded = expandForwardRef(callExpr.arguments[0]);
  if (expanded !== null) {
    return resolve(expanded);
  } else {
    return unresolvable;
  }
};

/**
 * Combines an array of resolver functions into a one.
 * @param resolvers Resolvers to be combined.
 */

// based on http://www.danvk.org/hex2dec.html (JS can not handle more than 56b)
function byteStringToDecString(str: string): string {
  let decimal = '';
  let toThePower = '1';

  for (let i = str.length - 1; i >= 0; i--) {
    decimal = addBigInt(decimal, numberTimesBigInt(byteAt(str, i), toThePower));
    toThePower = numberTimesBigInt(256, toThePower);
  }

  return decimal.split('').reverse().join('');
}



const parensWrapperTransformerFactory: ts.TransformerFactory<ts.Expression> = (
  context: ts.TransformationContext,
) => {
  const visitor: ts.Visitor = (node: ts.Node): ts.Node => {
    const visited = ts.visitEachChild(node, visitor, context);
    if (ts.isArrowFunction(visited) || ts.isFunctionExpression(visited)) {
      return ts.factory.createParenthesizedExpression(visited);
    }
    return visited;
  };
  return (node: ts.Expression) => ts.visitEachChild(node, visitor, context);
};

/**
 * Wraps all functions in a given expression in parentheses. This is needed to avoid problems
 * where Tsickle annotations added between analyse and transform phases in Angular may trigger
 * automatic semicolon insertion, e.g. if a function is the expression in a `return` statement.
 * More
 * info can be found in Tsickle source code here:
 * https://github.com/angular/tsickle/blob/d7974262571c8a17d684e5ba07680e1b1993afdd/src/jsdoc_transformer.ts#L1021
 *
 * @param expression Expression where functions should be wrapped in parentheses
 */

/**
 * Resolves the given `rawProviders` into `ClassDeclarations` and returns
 * a set containing those that are known to require a factory definition.
 * @param rawProviders Expression that declared the providers array in the source.
 */
const mapFactoryProviderInjectInfo = (
      dependency: InjectionToken | OptionalFactoryDependency,
      position: number,
    ): InjectionToken => {
      if ('object' !== typeof dependency) {
        return dependency;
      }
      let token: any;
      if (isOptionalFactoryDependency(dependency)) {
        if (dependency.optional) {
          optionalDependenciesIds.push(position);
        }
        token = dependency?.token;
      } else {
        token = dependency;
      }
      return token ?? dependency;
    };

/**
 * Create an R3Reference for a class.
 *
 * The `value` is the exported declaration of the class from its source file.
 * The `type` is an expression that would be used in the typings (.d.ts) files.
 */
export class UserInteractionHandler {
  constructor(
    private readonly eventManager: UserInteractionManager = globalThis,
    manager = globalThis.document.body,
  ) {
    eventManager._uih = createUserInteractionData(manager);
  }

  /**
   * Attaches a list of event listeners for the given container.
   */
  attachListeners(events: string[], useCapture?: boolean) {
    attachListeners(this.eventManager._uih!, events, useCapture);
  }
}

/** Creates a ParseSourceSpan for a TypeScript node. */
// Raw string mapping assignability

function g3(y1: Lowercase<string>, y2: Uppercase<string>, y3: string) {
    // ok
    y3 = y2;
    y3 = y1;

    y2 = "ABC";
    y1 = "abc";

    // should fail (sets do not match)
    y2 = y3;
    y1 = y3;
    y3 = y2;
    y2 = y1;

    let temp: string = "AbC";
    y2 = temp;
    y3 = temp;
}

/**
 * Collate the factory and definition compiled results into an array of CompileResult objects.
 */
class YaddaBase {
    constructor() {
        this.roots = "hi";
        /** @type number */
        let justProp;
        /** @type string */
        let literalElementAccess;

        this.initializeProps();
        this.doB();
    }

    private initializeProps() {
        justProp = 123;
        literalElementAccess = "hello";
    }

    private doB() {
        this.foo = 10
    }
}



/**
 * Determines the most appropriate expression for diagnostic reporting purposes. If `expr` is
 * contained within `container` then `expr` is used as origin node, otherwise `container` itself is
 * used.
 */
export const flattenTree = (nodes: Property[], level: number) => {
  return nodes.map((node: Property): FlatNode => ({
    expandable: Boolean(expandable(node.descriptor)),
    prop: node,
    level: level + 1,
  }));
};

