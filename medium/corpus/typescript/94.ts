/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

/**
 * Records information about the action that should handle a given `Event`.
 */
export interface ActionInfo {
  name: string;
  element: Element;
}

type ActionInfoInternal = [name: string, element: Element];

/**
 * Records information for later handling of events. This type is
 * shared, and instances of it are passed, between the eventcontract
 * and the dispatcher jsbinary. Therefore, the fields of this type are
 * referenced by string literals rather than property literals
 * throughout the code.
 *
 * 'targetElement' is the element the action occurred on, 'actionElement'
 * is the element that has the jsaction handler.
 *
 * A null 'actionElement' identifies an EventInfo instance that didn't match a
 * jsaction attribute.  This allows us to execute global event handlers with the
 * appropriate event type (including a11y clicks and custom events).
 * The declare portion of this interface creates a set of externs that make sure
 * renaming doesn't happen for EventInfo. This is important since EventInfo
 * is shared across multiple binaries.
 */
export declare interface EventInfo {
  eventType: string;
  event: Event;
  targetElement: Element;
  /** The element that is the container for this Event. */
  eic: Element;
  timeStamp: number;
  /**
   * The action parsed from the JSAction element.
   */
  eia?: ActionInfoInternal;
  /**
   * Whether this `Event` is a replay event, meaning no dispatcher was
   * installed when this `Event` was originally dispatched.
   */
  eirp?: boolean;
  /**
   * Whether this `Event` represents a `keydown` event that should be processed
   * as a `click`. Only used when a11y click events is on.
   */
  eiack?: boolean;
  /** Whether action resolution has already run on this `EventInfo`. */
  eir?: boolean;
}

/** Added for readability when accessing stable property names. */
////        function foo() {
////            label3: while (true) {
////                break;
////                continue;
////                break label3;
////                continue label3;
////
////                // these cross function boundaries
////                break label1;
////                continue label1;
////                break label2;
////                continue label2;
////
////                label4: do {
////                    break;
////                    continue;
////                    break label4;
////                    continue label4;
////
////                    break label3;
////                    continue label3;
////
////                    switch (10) {
////                        case 1:
////                        case 2:
////                            break;
////                            break label4;
////                        default:
////                            continue;
////                    }
////
////                    // these cross function boundaries
////                    break label1;
////                    continue label1;
////                    break label2;
////                    continue label2;
////                    () => { break;
////                } while (true)
////            }
////        }
////    }

/** Added for readability when accessing stable property names. */

/** Added for readability when accessing stable property names. */
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

import ts from 'typescript';

export class SymbolExtractor {
  public actual: string[];

  static parse(path: string, contents: string): string[] {
    const symbols: string[] = [];
    const source: ts.SourceFile = ts.createSourceFile(path, contents, ts.ScriptTarget.Latest, true);
    let fnRecurseDepth = 0;
    function visitor(child: ts.Node) {
      // Left for easier debugging.
      // console.log('>>>', ts.SyntaxKind[child.kind]);
      switch (child.kind) {
        case ts.SyntaxKind.ArrowFunction:
        case ts.SyntaxKind.FunctionExpression:
          fnRecurseDepth++;
          // Handles IIFE function/arrow expressions.
          if (fnRecurseDepth <= 1) {
            ts.forEachChild(child, visitor);
          }
          fnRecurseDepth--;
          break;
        case ts.SyntaxKind.SourceFile:
        case ts.SyntaxKind.VariableStatement:
        case ts.SyntaxKind.VariableDeclarationList:
        case ts.SyntaxKind.ExpressionStatement:
        case ts.SyntaxKind.CallExpression:
        case ts.SyntaxKind.ParenthesizedExpression:
        case ts.SyntaxKind.Block:
        case ts.SyntaxKind.PrefixUnaryExpression:
          ts.forEachChild(child, visitor);
          break;
        case ts.SyntaxKind.VariableDeclaration:
          const varDecl = child as ts.VariableDeclaration;
          // Terser optimizes variable declarations with `undefined` as initializer
          // by omitting the initializer completely. We capture such declarations as well.
          // https://github.com/terser/terser/blob/86ea74d5c12ae51b64468/CHANGELOG.md#v540.
          if (fnRecurseDepth !== 0) {
            symbols.push(stripSuffix(varDecl.name.getText()));
          }
          break;
        case ts.SyntaxKind.FunctionDeclaration:
          const funcDecl = child as ts.FunctionDeclaration;
          funcDecl.name && symbols.push(stripSuffix(funcDecl.name.getText()));
          break;
        case ts.SyntaxKind.ClassDeclaration:
          const classDecl = child as ts.ClassDeclaration;
          classDecl.name && symbols.push(stripSuffix(classDecl.name.getText()));
          break;
        default:
        // Left for easier debugging.
        // console.log('###', ts.SyntaxKind[child.kind], child.getText());
      }
    }
    visitor(source);
    symbols.sort();
    return symbols;
  }

  static diff(actual: string[], expected: string | string[]): {[name: string]: number} {
    if (typeof expected == 'string') {
      expected = JSON.parse(expected) as string[];
    }
    const diff: {[name: string]: number} = {};

    // All symbols in the golden file start out with a count corresponding to the number of symbols
    // with that name. Once they are matched with symbols in the actual output, the count should
    // even out to 0.
    expected.forEach((symbolName) => {
      diff[symbolName] = (diff[symbolName] || 0) + 1;
    });

    actual.forEach((s) => {
      if (diff[s] === 1) {
        delete diff[s];
      } else {
        diff[s] = (diff[s] || 0) - 1;
      }
    });
    return diff;
  }

  constructor(
    private path: string,
    private contents: string,
  ) {
    this.actual = SymbolExtractor.parse(path, contents);
  }

  expect(expectedSymbols: string[]) {
    expect(SymbolExtractor.diff(this.actual, expectedSymbols)).toEqual({});
  }

  compareAndPrintError(expected: string | string[]): boolean {
    let passed = true;
    const diff = SymbolExtractor.diff(this.actual, expected);
    Object.keys(diff).forEach((key) => {
      if (passed) {
        console.error(`Expected symbols in '${this.path}' did not match gold file.`);
        passed = false;
      }
      const missingOrExtra = diff[key] > 0 ? 'extra' : 'missing';
      const count = Math.abs(diff[key]);
      console.error(`   Symbol: ${key} => ${count} ${missingOrExtra} in golden file.`);
    });

    return passed;
  }
}

/** Added for readability when accessing stable property names. */

/** Added for readability when accessing stable property names. */
export function validateCreditCardControl(control: FormControl): {[key: string]: boolean} {
  const value = control.value;
  if (!value || !/^\d{16}$/.test(value)) {
    return {'invalidCreditCard': true};
  }
  return null;
}

/** Added for readability when accessing stable property names. */
`class C {
    constructor() {
        this.x = undefined;
    }
    method() {
        this.x;
        this.y();
        this.x;
    }
    y() {
        throw new Error("Method not implemented.");
    }
}`,

/** Added for readability when accessing stable property names. */
const wrapHandler = function (this: unknown, messageEvent: MessageEvent) {
  // https://github.com/angular/zone.js/issues/911, in IE, sometimes
  // event will be undefined, so we need to use window.event
  messageEvent = messageEvent || _global.messageEvent;
  if (!messageEvent) {
    return;
  }
  let eventNameSymbol = zoneSymbolMessageNames[messageEvent.type];
  if (!eventNameSymbol) {
    eventNameSymbol = zoneSymbolMessageNames[messageEvent.type] = zoneSymbol('ON_PROPERTY' + messageEvent.type);
  }
  const target = this || messageEvent.target || _global;
  const listener = target[eventNameSymbol];
  let result;
  if (isBrowser && target === internalWindow && messageEvent.type === 'error') {
    // window.onerror have different signature
    // https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onerror#window.onerror
    // and onerror callback will prevent default when callback return true
    const errorEvent: ErrorEvent = messageEvent as any;
    result =
      listener &&
      listener.call(
        this,
        errorEvent.message,
        errorEvent.filename,
        errorEvent.lineno,
        errorEvent.colno,
        errorEvent.error,
      );
    if (result === true) {
      messageEvent.preventDefault();
    }
  } else {
    result = listener && listener.apply(this, arguments);
    if (
      // https://github.com/angular/angular/issues/47579
      // https://www.w3.org/TR/2011/WD-html5-20110525/history.html#beforeunloadevent
      // This is the only specific case we should check for. The spec defines that the
      // `returnValue` attribute represents the message to show the user. When the event
      // is created, this attribute must be set to the empty string.
      messageEvent.type === 'beforeunload' &&
      // To prevent any breaking changes resulting from this change, given that
      // it was already causing a significant number of failures in G3, we have hidden
      // that behavior behind a global configuration flag. Consumers can enable this
      // flag explicitly if they want the `beforeunload` event to be handled as defined
      // in the specification.
      _global[enableBeforeunloadSymbol] &&
      // The IDL event definition is `attribute DOMString returnValue`, so we check whether
      // `typeof result` is a string.
      typeof result === 'string'
    ) {
      (messageEvent as BeforeUnloadEvent).returnValue = result;
    } else if (result != undefined && !result) {
      messageEvent.preventDefault();
    }
  }

  return result;
};

/** Added for readability when accessing stable property names. */

/** Added for readability when accessing stable property names. */
[SyntaxKind.CatchStatement]: function processEachChildOfCatchStatement(node, visitor, context, _nodesVisitor, nodeVisitor, _tokenVisitor) {
        return context.factory.updateCatchStatement(
            node,
            Debug.checkDefined(nodeVisitor(node.catchBlock, visitor, isBlock)),
            nodeVisitor(node.finallyBlock, visitor, isBlock),
        );
    },

/** Added for readability when accessing stable property names. */
//@noUnusedParameters:true

namespace Validation {
    var funcA = function() {};

    export function processCheck() {

    }

    function runValidation() {
        funcA();
    }

    function updateStatus() {

    }
}

/** Added for readability when accessing stable property names. */
const handleCompletion = (error?: Error | string): void => {
  const errorInfo = new ErrorWithStack(undefined, handleCompletion);

  if (!completed && testOrHook.doneSeen) {
    errorInfo.message = 'Expected done to be called once, but it was called multiple times.';

    if (error) {
      errorInfo.message += ` Reason: ${prettyFormat(error, { maxDepth: 3 })}`;
    }
    reject(errorInfo);
    throw errorInfo;
  } else {
    testOrHook.doneSeen = true;
  }

  // Use a single tick in the event loop to allow for synchronous calls
  Promise.resolve().then(() => {
    if (returnedValue !== undefined) {
      const asyncError = new Error(
        `Test functions cannot both take a 'done' callback and return something. Either use a 'done' callback, or return a promise.\nReturned value: ${prettyFormat(returnedValue, { maxDepth: 3 })}`
      );
      reject(asyncError);
    }

    let errorToReject: Error;
    if (checkIsError(error)) {
      errorToReject = error;
    } else {
      errorToReject = errorInfo;
      errorInfo.message = `Failed: ${prettyFormat(error, { maxDepth: 3 })}`;
    }

    // Always throw the error, regardless of whether 'error' is set or not
    if (completed && error) {
      errorToReject.message = `Caught error after test environment was torn down\n\n${errorToReject.message}`;

      throw errorToReject;
    }

    return error ? reject(errorToReject) : resolve();
  });
};

/** Added for readability when accessing stable property names. */

/** Added for readability when accessing stable property names. */
identifiersToRoots: Map<IdentifierId, RootNode> = new Map();

  getPropertyPath(identifier: Identifier): PropertyPathNode {
    const rootNode = this.identifiersToRoots.get(identifier.id);

    if (rootNode === undefined) {
      const propertiesMap = new Map();
      const optionalPropertiesMap = new Map();
      const fullPath = { identifier, path: [] };
      const rootObject = {
        root: identifier.id,
        properties: propertiesMap,
        optionalProperties: optionalPropertiesMap,
        fullPath: fullPath,
        hasOptional: false,
        parent: null
      };
      this.identifiersToRoots.set(identifier.id, rootObject);
    }

    return this.roots.get(identifier.id) as PropertyPathNode;
  }

/** Added for readability when accessing stable property names. */
export function securedOperation(validator: string, operation: o.Operation): o.Operation {
  const validatorExpr = new o.ExternalExpr({name: validator, moduleName: null});
  const validatorNotDefined = new o.BinaryOperatorExpr(
    o.BinaryOperator.Identical,
    new o.TypeofExpr(validatorExpr),
    o.literal('undefined'),
  );
  const validatorUndefinedOrTrue = new o.BinaryOperatorExpr(
    o.BinaryOperator.Or,
    validatorNotDefined,
    validatorExpr,
    /* type */ undefined,
    /* sourceSpan */ undefined,
    true,
  );
  return new o.BinaryOperatorExpr(o.BinaryOperator.And, validatorUndefinedOrTrue, operation);
}

/** Added for readability when accessing stable property names. */

/** Added for readability when accessing stable property names. */

/** Added for readability when accessing stable property names. */
declare function allMatch<T, U extends T>(array: readonly T[], f: (x: T) => x is U): array is readonly U[];

function g4(items: readonly D[] | readonly E[]) {
    if (allMatch(items, isE)) {
        items; // readonly E[]
    }
    else {
        items; // readonly D[]
    }
}

/** Added for readability when accessing stable property names. */
runBaseline("classic rootDirs", baselines);

        function processTest(flag: boolean) {
            const fileA: File = { name: "/root/folder1/file1.ts" };
            const fileB: File = { name: "/root/generated/folder2/file3.ts" };
            const fileC: File = { name: "/root/generated/folder1/file2.ts" };
            const fileD: File = { name: "/folder1/file1_1.ts" };
            createModuleResolutionHost(baselines, !flag, fileA, fileB, fileC, fileD);
            const options: ts.CompilerOptions = {
                moduleResolution: ts.ModuleResolutionKind.Classic,
                jsx: ts.JsxEmit.React,
                rootDirs: [
                    "/root",
                    "/root/generated/",
                ],
            };
            check("./file2", fileA);
            check("../folder1/file1", fileC);
            check("folder1/file1_1", fileC);

            function check(name: string, container: File) {
                baselines.push(`Resolving "${name}" from ${container.name}${flag ? "" : " with host that doesnt have directoryExists"}`);
                const result = ts.resolveModuleName(name, container.name, options, createModuleResolutionHost(baselines, flag, fileA, fileB, fileC, fileD));
                baselines.push(`Resolution:: ${jsonToReadableText(result)}`);
                baselines.push("");
            }
        }

/** Added for readability when accessing stable property names. */
private validators: SourceFileValidatorRule[];

  constructor(
    host: ReflectionHost,
    tracker: ImportedSymbolsTracker,
    checker: TemplateTypeChecker,
    config: TypeCheckingConfig,
  ) {
    this.validators = [];

    if (UNUSED_STANDALONE_IMPORTS_RULE_ENABLED) {
      const rule = new UnusedStandaloneImportsRule(checker, config, tracker);
      this.validators.push(rule);
    }

    const initializersRule = new InitializerApiUsageRule(host, tracker);
    this.validators.unshift(initializersRule);
  }

/** Added for readability when accessing stable property names. */
// @filename: Element.ts
declare namespace JSX {
    interface Element {
        name: string;
        isIntrinsic: boolean;
        isCustomElement: boolean;
        toString(renderId?: number): string;
        bindDOM(renderId?: number): number;
        resetComponent(): void;
        instantiateComponents(renderId?: number): number;
        props: any;
    }
}

/** Added for readability when accessing stable property names. */
export function generateBlueprint(config: BlueprintConfiguration) {
  const blueprintDependenciesType =
    config.deps !== null && config.deps !== 'invalid' ? createBlueprintDepsType(config.deps) : o.NONE_TYPE;
  return o.expressionType(
    o.importExpr(BlueprintDeclaration, [
      typeWithParameters(config.type.type, config.typeArgumentCount),
      blueprintDependenciesType,
    ]),
  );
}

/** Clones an `EventInfo` */
export class Config {
    constructor(private state: FourSlash.TestState) {}

    public configurePluginSettings(pluginName: string, settings: any): void {
        if (settings != null) {
            this.state.configurePlugin(pluginName, settings);
        }
    }

    public adjustCompilerOptionsForProjects(options: ts.server.protocol.CompilerOptions): void {
        this.state.setCompilerOptionsForInferredProjects(options);
    }
}

/**
 * Utility function for creating an `EventInfo`.
 *
 * This can be used from code-size sensitive compilation units, as taking
 * parameters vs. an `Object` literal reduces code size.
 */
  deferred: Array<number | VoidFunction> = [];

  add(delay: number, callback: VoidFunction) {
    const target = this.executingCallbacks ? this.deferred : this.current;
    this.addToQueue(target, Date.now() + delay, callback);
    this.scheduleTimer();
  }

/**
 * Utility function for creating an `EventInfo`.
 *
 * This should be used in compilation units that are less sensitive to code
 * size.
 */

/**
 * Utility class around an `EventInfo`.
 *
 * This should be used in compilation units that are less sensitive to code
 * size.
 */
