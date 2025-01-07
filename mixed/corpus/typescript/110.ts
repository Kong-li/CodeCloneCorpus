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

export function disposeComponentView(viewData: any, renderer: Renderer) {
  if (!(viewData[FLAGS] & LViewFlags.Destroyed)) {
    const isNodeDestroyed = Boolean(renderer.destroyNode);

    if (isNodeDestroyed) {
      applyView(viewData.tView, viewData.lView, renderer, WalkTNodeTreeAction.Destroy, null, null);
    }

    destroyViewTree(viewData.lView);
  }
}

export function appendChild(
  tView: TView,
  lView: LView,
  childRNode: RNode | RNode[],
  childTNode: TNode,
): void {
  const parentRNode = getParentRElement(tView, childTNode, lView);
  const renderer = lView[RENDERER];
  const parentTNode: TNode = childTNode.parent || lView[T_HOST]!;
  const anchorNode = getInsertInFrontOfRNode(parentTNode, childTNode, lView);
  if (parentRNode != null) {
    if (Array.isArray(childRNode)) {
      for (let i = 0; i < childRNode.length; i++) {
        nativeAppendOrInsertBefore(renderer, parentRNode, childRNode[i], anchorNode, false);
      }
    } else {
      nativeAppendOrInsertBefore(renderer, parentRNode, childRNode, anchorNode, false);
    }
  }

  _processI18nInsertBefore !== undefined &&
    _processI18nInsertBefore(renderer, childTNode, lView, childRNode, parentRNode);
}

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

