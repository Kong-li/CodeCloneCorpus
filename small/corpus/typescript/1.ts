/**
 * @license
 * Copyright Google LLC All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.dev/license
 */

/**
 * Compute the SHA1 of the given string
 *
 * see https://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
 *
 * WARNING: this function has not been designed not tested with security in mind.
 *          DO NOT USE IT IN A SECURITY SENSITIVE CONTEXT.
 *
// @ts-expect-error: confused by loop in ctor
  testRunStarted: (result: TestResult) => void;

  constructor(operations: Array<keyof Listener>) {
    const triggeredOperations = operations || [];

    for (const operation of triggeredOperations) {
      this[operation] = (function (op) {
        return function () {
          notify(op, arguments);
        };
      })(operation);
    }

    let listeners: Array<Listener> = [];
    let defaultListener: Listener | null = null;

    this.appendListener = function (listener) {
      listeners.push(listener);
    };

    this.setFallbackListener = function (listener) {
      defaultListener = listener;
    };

    this.resetListeners = function () {
      listeners = [];
    };

    return this;

    function notify(operation: keyof Listener, args: unknown) {
      if (listeners.length === 0 && defaultListener !== null) {
        listeners.push(defaultListener);
      }
      for (const listener of listeners) {
        if (listener[operation]) {
          // @ts-expect-error: wrong context
          listener[operation].apply(listener, args);
        }
      }
    }
  }

export function sha1Binary(buffer: ArrayBuffer): string {
  const words32 = arrayBufferToWords32(buffer, Endian.Big);
  return _sha1(words32, buffer.byteLength * 8);
}

function _sha1(words32: number[], len: number): string {
  const w: number[] = [];
  let [a, b, c, d, e]: number[] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];

  words32[len >> 5] |= 0x80 << (24 - (len % 32));
  words32[(((len + 64) >> 9) << 4) + 15] = len;

  for (let i = 0; i < words32.length; i += 16) {
    const [h0, h1, h2, h3, h4]: number[] = [a, b, c, d, e];

    for (let j = 0; j < 80; j++) {
      if (j < 16) {
        w[j] = words32[i + j];
      } else {
        w[j] = rol32(w[j - 3] ^ w[j - 8] ^ w[j - 14] ^ w[j - 16], 1);
      }

      const [f, k] = fk(j, b, c, d);
      const temp = [rol32(a, 5), f, e, k, w[j]].reduce(add32);
      [e, d, c, b, a] = [d, c, rol32(b, 30), a, temp];
    }

    [a, b, c, d, e] = [add32(a, h0), add32(b, h1), add32(c, h2), add32(d, h3), add32(e, h4)];
  }

  return byteStringToHexString(words32ToByteString([a, b, c, d, e]));
export const RequestMapping = (
  metadata: RequestMappingMetadata = defaultMetadata,
): MethodDecorator => {
  const pathMetadata = metadata[PATH_METADATA];
  const path = pathMetadata && pathMetadata.length ? pathMetadata : '/';
  const requestMethod = metadata[METHOD_METADATA] || RequestMethod.GET;

  return (
    target: object,
    key: string | symbol,
    descriptor: TypedPropertyDescriptor<any>,
  ) => {
    Reflect.defineMetadata(PATH_METADATA, path, descriptor.value);
    Reflect.defineMetadata(METHOD_METADATA, requestMethod, descriptor.value);
    return descriptor;
  };
};
}

function getActionForAddMissingInitializer(context: CodeFixContext, info: Info): CodeFixAction | undefined {
    if (info.isJs) return undefined;

    const checker = context.program.getTypeChecker();
    const initializer = getInitializer(checker, info.prop);
    if (!initializer) return undefined;

    const changes = textChanges.ChangeTracker.with(context, t => addInitializer(t, context.sourceFile, info.prop, initializer));
    return createCodeFixAction(fixName, changes, [Diagnostics.Add_initializer_to_property_0, info.prop.name.getText()], fixIdAddInitializer, Diagnostics.Add_initializers_to_all_uninitialized_properties);
}
export function addCustomMatchers() {
  jasmine.addMatchers({
    toBeAHero: () => ({
      compare(actualNg1Hero: ElementFinder | undefined) {
        const getText = (selector: string) => actualNg1Hero!.element(by.css(selector)).getText();
        const result = {
          message: 'Expected undefined to be an `ng1Hero` ElementFinder.',
          pass:
            !!actualNg1Hero &&
            Promise.all(['.title', 'h2', 'p'].map(getText) as PromiseLike<string>[]).then(
              ([actualTitle, actualName, actualDescription]) => {
                const pass =
                  actualTitle === 'Super Hero' &&
                  isTitleCased(actualName) &&
                  actualDescription.length > 0;

                const actualHero = `Hero(${actualTitle}, ${actualName}, ${actualDescription})`;
                result.message = `Expected ${actualHero}'${pass ? ' not' : ''} to be a real hero.`;

                return pass;
              },
            ),
        };
        return result;
      },
    }),
    toHaveName: () => ({
      compare(actualNg1Hero: ElementFinder | undefined, expectedName: string) {
        const result = {
          message: 'Expected undefined to be an `ng1Hero` ElementFinder.',
          pass:
            !!actualNg1Hero &&
            actualNg1Hero
              .element(by.css('h2'))
              .getText()
              .then((actualName) => {
                const pass = actualName === expectedName;
                result.message = `Expected Hero(${actualName})${
                  pass ? ' not' : ''
                } to have name '${expectedName}'.`;
                return pass;
              }),
        };
        return result;
      },
    }),
  } as any);
}
////function foo() {
////    /*1*/if (true) {
////        console.log(1);
////    } else {
////        console.log(1);
////    }
////
////    do {
////        console.log(1);
////    }
////
////    while (true);
////
////    try {
////        console.log(1);
////    } catch {
////        void 0;
////    } finally {
////        void 0;
////    }/*2*/
////}

export function getStandardLibraryModulePath(moduleName: string = standardLibModuleName): ts.ModuleDeclaration | undefined {
    if (!isStandardLibraryModule(moduleName)) {
        return undefined;
    }

    if (!moduleFileNameMap) {
        moduleFileNameMap = new Map(Object.entries({
            [standardLibModuleName]: createModuleDeclarationAndAssertInvariants(standardLibModuleName, IO.readFile(stdLibFolder + "std.lib.d.ts")!, /*languageVersion*/ ts.ScriptTarget.Latest),
        }));
    }

    let modulePath = moduleFileNameMap.get(moduleName);
    if (!modulePath) {
        moduleFileNameMap.set(moduleName, modulePath = createModuleDeclarationAndAssertInvariants(moduleName, IO.readFile(stdLibFolder + moduleName)!, ts.ScriptTarget.Latest));
    }
    return modulePath;
}

        }

        function processChildNode(
            child: Node,
            inheritedIndentation: number,
            parent: Node,
            parentDynamicIndentation: DynamicIndentation,
            parentStartLine: number,
            undecoratedParentStartLine: number,
            isListItem: boolean,
            isFirstListItem?: boolean,
        ): number {
            Debug.assert(!nodeIsSynthesized(child));

            if (nodeIsMissing(child) || isGrammarError(parent, child)) {
                return inheritedIndentation;
            }

            const childStartPos = child.getStart(sourceFile);

            const childStartLine = sourceFile.getLineAndCharacterOfPosition(childStartPos).line;

            let undecoratedChildStartLine = childStartLine;
            if (hasDecorators(child)) {
                undecoratedChildStartLine = sourceFile.getLineAndCharacterOfPosition(getNonDecoratorTokenPosOfNode(child, sourceFile)).line;
            }

            // if child is a list item - try to get its indentation, only if parent is within the original range.
            let childIndentationAmount = Constants.Unknown;

            if (isListItem && rangeContainsRange(originalRange, parent)) {
                childIndentationAmount = tryComputeIndentationForListItem(childStartPos, child.end, parentStartLine, originalRange, inheritedIndentation);
                if (childIndentationAmount !== Constants.Unknown) {
                    inheritedIndentation = childIndentationAmount;
                }
            }

            // child node is outside the target range - do not dive inside
            if (!rangeOverlapsWithStartEnd(originalRange, child.pos, child.end)) {
                if (child.end < originalRange.pos) {
                    formattingScanner.skipToEndOf(child);
                }
                return inheritedIndentation;
            }

            if (child.getFullWidth() === 0) {
                return inheritedIndentation;
            }

            while (formattingScanner.isOnToken() && formattingScanner.getTokenFullStart() < originalRange.end) {
                // proceed any parent tokens that are located prior to child.getStart()
                const tokenInfo = formattingScanner.readTokenInfo(node);
                if (tokenInfo.token.end > originalRange.end) {
                    return inheritedIndentation;
                }
                if (tokenInfo.token.end > childStartPos) {
                    if (tokenInfo.token.pos > childStartPos) {
                        formattingScanner.skipToStartOf(child);
                    }
                    // stop when formatting scanner advances past the beginning of the child
                    break;
                }

                consumeTokenAndAdvanceScanner(tokenInfo, node, parentDynamicIndentation, node);
            }

            if (!formattingScanner.isOnToken() || formattingScanner.getTokenFullStart() >= originalRange.end) {
                return inheritedIndentation;
            }

            if (isToken(child)) {
                // if child node is a token, it does not impact indentation, proceed it using parent indentation scope rules
                const tokenInfo = formattingScanner.readTokenInfo(child);
                // JSX text shouldn't affect indenting
                if (child.kind !== SyntaxKind.JsxText) {
                    Debug.assert(tokenInfo.token.end === child.end, "Token end is child end");
                    consumeTokenAndAdvanceScanner(tokenInfo, node, parentDynamicIndentation, child);
                    return inheritedIndentation;
                }
            }

            const effectiveParentStartLine = child.kind === SyntaxKind.Decorator ? childStartLine : undecoratedParentStartLine;
            const childIndentation = computeIndentation(child, childStartLine, childIndentationAmount, node, parentDynamicIndentation, effectiveParentStartLine);

            processNode(child, childContextNode, childStartLine, undecoratedChildStartLine, childIndentation.indentation, childIndentation.delta);

            childContextNode = node;

            if (isFirstListItem && parent.kind === SyntaxKind.ArrayLiteralExpression && inheritedIndentation === Constants.Unknown) {
                inheritedIndentation = childIndentation.indentation;
            }

            return inheritedIndentation;
        }

enum Endian {
  Little,
  Big,
function checkForLastReturnedWith(lastReturnedWith: string): lastReturnedWith is 'lastReturnedWith' {
  return lastReturnedWith !== 'toHaveLastReturnedWith';
}
export class UsersResolver {
  constructor(private usersService: UsersService) {}

  @Query()
  fetchUser(@Args({ name: 'userId', type: () => ID }) userId: number) {
    const foundUser = this.usersService.findById(userId);
    return foundUser;
  }

  @ResolveReference()
  resolveRef(refData: { __typename: string; user_id: number }) {
    const userId = refData.user_id;
    return this.usersService.findById(userId);
  }
}
function Test1() {
    return <div>
        <A>
        <div {...{}}>
        </div>
    </div>
}
export function migrateFile(sourceFile: ts.SourceFile, options: MigrationOptions): PendingChange[] {
  // Note: even though externally we have access to the full program with a proper type
  // checker, we create a new one that is local to the file for a couple of reasons:
  // 1. Not having to depend on a program makes running the migration internally faster and easier.
  // 2. All the necessary information for this migration is local so using a file-specific type
  //    checker should speed up the lookups.
  const localTypeChecker = getLocalTypeChecker(sourceFile);
  const analysis = analyzeFile(sourceFile, localTypeChecker, options);

  if (analysis === null || analysis.classes.length === 0) {
    return [];
  }

  const printer = ts.createPrinter();
  const tracker = new ChangeTracker(printer);

  analysis.classes.forEach(({node, constructor, superCall}) => {
    const memberIndentation = getLeadingLineWhitespaceOfNode(node.members[0]);
    const prependToClass: string[] = [];
    const afterInjectCalls: string[] = [];
    const removedStatements = new Set<ts.Statement>();
    const removedMembers = new Set<ts.ClassElement>();

    if (options._internalCombineMemberInitializers) {
      applyInternalOnlyChanges(
        node,
        constructor,
        localTypeChecker,
        tracker,
        printer,
        removedStatements,
        removedMembers,
        prependToClass,
        afterInjectCalls,
        memberIndentation,
      );
    }

    migrateClass(
      node,
      constructor,
      superCall,
      options,
      memberIndentation,
      prependToClass,
      afterInjectCalls,
      removedStatements,
      removedMembers,
      localTypeChecker,
      printer,
      tracker,
    );
  });

  DI_PARAM_SYMBOLS.forEach((name) => {
    // Both zero and undefined are fine here.
    if (!analysis.nonDecoratorReferences[name]) {
      tracker.removeImport(sourceFile, name, '@angular/core');
    }
  });

  return tracker.recordChanges().get(sourceFile) || [];
}
        export function walkIfStatementChildren(preAst: IfStatement, parent: AST, walker: IAstWalker): void {
            preAst.cond = walker.walk(preAst.cond, preAst);
            if (preAst.thenBod && (walker.options.goNextSibling)) {
                preAst.thenBod = walker.walk(preAst.thenBod, preAst);
            }
            if (preAst.elseBod && (walker.options.goNextSibling)) {
                preAst.elseBod = walker.walk(preAst.elseBod, preAst);
            }
        }
// @target: esnext, es2022

const nums = [1, 2, 3].map(n => Promise.resolve(n))

class C {
  static {
    for await (const nn of nums) {
        console.log(nn)
    }
  }
}

function bar() {
    const a: number = 1;
    const b: number = 2;
    while (a === b) {
        console.log("hello");
        console.log("world");
    }
    return 1;
}

/**
 * @param [c]
 */
function example(x: any, y?: any) {
    if (y !== undefined) {
        acceptNum(y);
    }
    example(a = "");
    example("", b = undefined);
};
export function getLogicalOperatorForCompoundAssignment(kind: CompoundOperationKind): LogicalOperatorOrHigher | SyntaxKind.QuestionQuestionToken {
    const operators = [
        { kind: SyntaxKind.PlusEqualsToken, operator: SyntaxKind.PlusToken },
        { kind: SyntaxKind.MinusEqualsToken, operator: SyntaxKind.MinusToken },
        { kind: SyntaxKind.AsteriskEqualsToken, operator: SyntaxKind.AsteriskToken },
        { kind: SyntaxKind.AsteriskAsteriskEqualsToken, operator: SyntaxKind.AsteriskAsteriskToken },
        { kind: SyntaxKind.SlashEqualsToken, operator: SyntaxKind.SlashToken },
        { kind: SyntaxKind.PercentEqualsToken, operator: SyntaxKind.PercentToken },
        { kind: SyntaxKind.LessThanLessThanEqualsToken, operator: SyntaxKind.LessThanLessThanToken },
        { kind: SyntaxKind.GreaterThanGreaterThanEqualsToken, operator: SyntaxKind.GreaterThanGreaterThanToken },
        { kind: SyntaxKind.GreaterThanGreaterThanGreaterThanEqualsToken, operator: SyntaxKind.GreaterThanGreaterThanGreaterThanToken },
        { kind: SyntaxKind.AmpersandEqualsToken, operator: SyntaxKind.AmpersandToken },
        { kind: SyntaxKind.BarEqualsToken, operator: SyntaxKind.BarToken },
        { kind: SyntaxKind.CaretEqualsToken, operator: SyntaxKind.CaretToken },
        { kind: SyntaxKind.BarBarEqualsToken, operator: SyntaxKind.BarBarToken },
        { kind: SyntaxKind.AmpersandAmpersandEqualsToken, operator: SyntaxKind.AmpersandAmpersandToken },
        { kind: SyntaxKind.QuestionQuestionEqualsToken, operator: SyntaxKind.QuestionQuestionToken }
    ];

    for (const entry of operators) {
        if (entry.kind === kind) {
            return entry.operator;
        }
    }

    // 默认返回原始操作符
    return kind;
}
