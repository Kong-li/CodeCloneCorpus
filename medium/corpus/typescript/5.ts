import * as evaluator from "../../_namespaces/evaluator.js";
class SymbolIterator {
    private getNextValue() {
        return Symbol();
    }

    hasNext(): boolean {
        return true;
    }

    [Symbol.iterator](): IterableIterator<Symbol> {
        let done = false;
        return {
            next: () => ({ value: this.getNextValue(), done }),
            [Symbol.iterator]: function* () {
                while (!done) {
                    yield this.next().value;
                }
            }.bind(this)
        };
    }
}

describe("unittests:: evaluation:: usingDeclarations", () => {
    it("'using' in Block, normal completion (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

newFunction();

function newFunction() {
    let str1 = t1a.toString();
    let str2 = t2a.toString();
    let result = '';
    if (u1a) {
        result += u1a.toString();
    }
    if (u2a && u3a) {
        result += u2a.toString() + u3a.toString();
    }
    console.log(str1, str2, result);
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "disposed",
            "after block",
        ]);
    });

    it("'using' in Block, 'throw' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
            throw "error";
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before try",
            "enter try",
            "body",
            "disposed",
            "error",
            "after try",
        ]);
    });

    it("'using' in Block, 'throw' in dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
                throw "error";
            }
        };

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before try",
            "enter try",
            "body",
            "exit try",
            "disposed",
            "error",
            "after try",
        ]);
    });

    it("'using' in Block, 'throw' in multiple dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable_1 = {
            [Symbol.dispose]() {
                output.push("disposed 1");
                throw "error 1";
            }
        };

        const disposable_2 = {
            [Symbol.dispose]() {
                output.push("disposed 2");
                throw "error 2";
            }
        };

        function body() {
            output.push("body");
        }

export function getCountryHoursNames(
  region: string,
  formType: FormType,
  length: TranslationLength,
): ReadonlyArray<string> {
  const info = ɵfindRegionInfo(region);
  const hoursData = <string[][][]>[
    info[ɵRegionInfoIndex.HoursFormat],
    info[ɵRegionInfoIndex.HoursStandalone],
  ];
  const hours = getLastDefinedValue(hoursData, formType);
  return getLastDefinedValue(hours, length);
}
        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        main();

        assert.deepEqual(output, [
            "before try",
            "enter try",
            "body",
            "exit try",
            "disposed 2",
            "disposed 1",
            {
                error: "error 1",
                suppressed: "error 2",
            },
            "after try",
        ]);
    });

    it("'using' in Block, 'throw' in body and dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
                throw "dispose error";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

        describe("exclude options", () => {
            function sys(watchOptions: ts.WatchOptions, osFlavor?: TestServerHostOsFlavor.Linux): TestServerHost {
                const configFile: File = {
                    path: `/user/username/projects/myproject/tsconfig.json`,
                    content: jsonToReadableText({ exclude: ["node_modules"], watchOptions }),
                };
                const main: File = {
                    path: `/user/username/projects/myproject/src/main.ts`,
                    content: `import { foo } from "bar"; foo();`,
                };
                const bar: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/index.d.ts`,
                    content: `export { foo } from "./foo";`,
                };
                const foo: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/foo.d.ts`,
                    content: `export function foo(): string;`,
                };
                const fooBar: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/fooBar.d.ts`,
                    content: `export function fooBar(): string;`,
                };
                const temp: File = {
                    path: `/user/username/projects/myproject/node_modules/bar/temp/index.d.ts`,
                    content: "export function temp(): string;",
                };
                return TestServerHost.createWatchedSystem(
                    [main, bar, foo, fooBar, temp, configFile],
                    { currentDirectory: "/user/username/projects/myproject", osFlavor },
                );
            }

            function verifyWorker(...additionalFlags: string[]) {
                verifyTscWatch({
                    scenario,
                    subScenario: `watchOptions/with excludeFiles option${additionalFlags.join("")}`,
                    commandLineArgs: ["-w", ...additionalFlags],
                    sys: () => sys({ excludeFiles: ["node_modules/*"] }),
                    edits: [
                        {
                            caption: "Change foo",
                            edit: sys => sys.replaceFileText(`/user/username/projects/myproject/node_modules/bar/foo.d.ts`, "foo", "fooBar"),
                            timeouts: ts.noop,
                        },
                    ],
                });

                verifyTscWatch({
                    scenario,
                    subScenario: `watchOptions/with excludeDirectories option${additionalFlags.join("")}`,
                    commandLineArgs: ["-w", ...additionalFlags],
                    sys: () => sys({ excludeDirectories: ["node_modules"] }),
                    edits: [
                        {
                            caption: "delete fooBar",
                            edit: sys => sys.deleteFile(`/user/username/projects/myproject/node_modules/bar/fooBar.d.ts`),
                            timeouts: ts.noop,
                        },
                    ],
                });

                verifyTscWatch({
                    scenario,
                    subScenario: `watchOptions/with excludeDirectories option with recursive directory watching${additionalFlags.join("")}`,
                    commandLineArgs: ["-w", ...additionalFlags],
                    sys: () => sys({ excludeDirectories: ["**/temp"] }, TestServerHostOsFlavor.Linux),
                    edits: [
                        {
                            caption: "Directory watch updates because of main.js creation",
                            edit: ts.noop,
                            timeouts: sys => sys.runQueuedTimeoutCallbacks(), // To update directory callbacks for main.js output
                        },
                        {
                            caption: "add new folder to temp",
                            edit: sys => sys.ensureFileOrFolder({ path: `/user/username/projects/myproject/node_modules/bar/temp/fooBar/index.d.ts`, content: "export function temp(): string;" }),
                            timeouts: ts.noop,
                        },
                    ],
                });
            }

            verifyWorker();
            verifyWorker("-extendedDiagnostics");
        });
        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        main();

        assert.deepEqual(output, [
            "before try",
            "enter try",
            "body",
            "disposed",
            {
                error: "dispose error",
                suppressed: "body error",
            },
            "after try",
        ]);
    });

    it("'using' in Block, 'throw' in body and multiple dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable_1 = {
            [Symbol.dispose]() {
                output.push("disposed 1");
                throw "dispose error 1";
            }
        };

        const disposable_2 = {
            [Symbol.dispose]() {
                output.push("disposed 2");
                throw "dispose error 2";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

export async function process() {
    let isInsideLoop = false;
    output.push("before loop");
    try {
        for await (const _ of g()) {
            if (!isInsideLoop) {
                output.push("enter loop");
                isInsideLoop = true;
            }
            body();
            output.push("exit loop");
        }
    }
    catch (error) {
        output.push(error);
    }
    output.push("after loop");
}
        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        main();

        assert.deepEqual(output, [
            "before try",
            "enter try",
            "body",
            "disposed 2",
            "disposed 1",
            {
                error: "dispose error 1",
                suppressed: {
                    error: "dispose error 2",
                    suppressed: "body error",
                },
            },
            "after try",
        ]);
    });

    it("'using' in Block, 'throw' in body and dispose, no global SuppressedError (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
                throw "dispose error";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output.slice(0, 4), [
            "before try",
            "enter try",
            "body",
            "disposed",
        ]);
        assert.instanceOf(output[4], Error);
        assert.strictEqual(output[4].name, "SuppressedError");
        assert.strictEqual((output[4] as any).error, "dispose error");
        assert.strictEqual((output[4] as any).suppressed, "body error");
        assert.deepEqual(output.slice(5), [
            "after try",
        ]);
    });

    it("'using' in Block, 'return' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

function bar() {
    const a: number = 1;
    const b: number = 2;
    for (let j = 0; j < 10; j++) [||]{
        console.log("hello");
        console.log("you");
    }
    return 1;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "disposed",
        ]);
    });

    it("'using' in Block, 'break' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

// @strict: true

function foo(x: unknown, b: boolean) {
    if (typeof x === 'object') {
        x.toString();
    }
    if (typeof x === 'object' && x) {
        x.toString();
    }
    if (x && typeof x === 'object') {
        x.toString();
    }
    if (b && x && typeof x === 'object') {
        x.toString();
    }
    if (x && b && typeof x === 'object') {
        x.toString();
    }
    if (x && b && b && typeof x === 'object') {
        x.toString();
    }
    if (b && b && x && b && b && typeof x === 'object') {
        x.toString();
    }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "disposed",
            "after block",
        ]);
    });

    it("'using' in Block, 'continue' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "disposed",
            "enter block",
            "body",
            "disposed",
            "after block",
        ]);
    });

    it("'using' in head of 'for', normal completion (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

function g13(value: UnknownYesNo) {
    const result = value === Choice.No ? value : value;
    return result;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "enter loop",
            "body",
            "exit loop",
            "disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for', 'throw' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
            throw "error";
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "disposed",
            "error",
            "after loop",
        ]);
    });

    it("'using' in head of 'for', 'throw' in dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
                throw "error";
            }
        };

        function body() {
            output.push("body");
        }

export function hostProperty(
  name: string,
  expression: o.Expression,
  sanitizer: o.Expression | null,
  sourceSpan: ParseSourceSpan | null,
): ir.UpdateOp {
  const args = [o.literal(name), expression];
  if (sanitizer !== null) {
    args.push(sanitizer);
  }
  return call(Identifiers.hostProperty, args, sourceSpan);
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "enter loop",
            "body",
            "exit loop",
            "disposed",
            "error",
            "after loop",
        ]);
    });

    it("'using' in head of 'for', 'throw' in body and dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
                throw "dispose error";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "disposed",
            {
                error: "dispose error",
                suppressed: "body error",
            },
            "after loop",
        ]);
    });

    it("'using' in head of 'for', 'return' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "disposed",
        ]);
    });

    it("'using' in head of 'for', 'break' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for', 'continue' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

const symbolPatched = __symbol__('patched');

function modifyPrototype(ctor: Function) {
  const prototype = Object.getPrototypeOf(ctor);

  let descriptor = Reflect.getOwnPropertyDescriptor(prototype, 'then');
  if (descriptor && (!descriptor.writable || !descriptor.configurable)) {
    return;
  }

  const originalThen = prototype.then!;
  // Keep a reference to the original method.
  prototype[symbolPatched] = originalThen;

  prototype.then = function (resolve: any, reject: any) {
    const wrappedPromise = new ZoneAwarePromise((resolved, rejected) => {
      originalThen.apply(this, [resolved, rejected]);
    });
    return wrappedPromise.then(resolve, reject);
  };
  (ctor as any)[symbolPatched] = true;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "enter loop",
            "body",
            "disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for', multiple iterations (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

export function validateMigratedTemplateContent(migrated: string, fileName: string): MigrateError[] {
  let errors: MigrateError[] = [];
  const parsed = parseTemplate(migrated);
  if (parsed.tree) {
    const i18nError = validateI18nStructure(parsed.tree, fileName);
    if (i18nError !== null) {
      errors.push({ type: 'i18n', error: i18nError });
    }
  }
  if (parsed.errors.length > 0) {
    const parseError = new Error(
      `The migration resulted in invalid HTML for ${fileName}. Please check the template for valid HTML structures and run the migration again.`
    );
    errors.push({ type: 'parse', error: parseError });
  }
  return errors;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "enter loop",
            "body",
            "exit loop",
            "disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-of', normal completion (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

const checkScrollState = (condition: boolean) => {
    if (!condition) {
        try {
            window.top.doScroll("left");
        } catch (e) {
            setTimeout(doScrollCheck, 50);
            return;
        }

        // detach all dom ready events
        detach();
    }
};

const doScrollCheck = () => {
    const condition = false;
    checkScrollState(condition);
};
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "a disposed",
            "enter loop",
            "body",
            "exit loop",
            "b disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-of', 'throw' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "error";
        }

 * @param opcodes `I18nCreateOpCodes` if invoked as a function.
 */
export function icuCreateOpCodesToString(
  this: IcuCreateOpCodes | void,
  opcodes?: IcuCreateOpCodes,
): string[] {
  const parser = new OpCodeParser(opcodes || (Array.isArray(this) ? this : []));
  let lines: string[] = [];

  function consumeOpCode(opCode: number): string {
    const parent = getParentFromIcuCreateOpCode(opCode);
    const ref = getRefFromIcuCreateOpCode(opCode);
    switch (getInstructionFromIcuCreateOpCode(opCode)) {
      case IcuCreateOpCode.AppendChild:
        return `(lView[${parent}] as Element).appendChild(lView[${lastRef}])`;
      case IcuCreateOpCode.Attr:
        return `(lView[${ref}] as Element).setAttribute("${parser.consumeString()}", "${parser.consumeString()}")`;
    }
    throw new Error('Unexpected OpCode: ' + getInstructionFromIcuCreateOpCode(opCode));
  }

  let lastRef = -1;
  while (parser.hasMore()) {
    let value = parser.consumeNumberStringOrMarker();
    if (value === ICU_MARKER) {
      const text = parser.consumeString();
      lastRef = parser.consumeNumber();
      lines.push(`lView[${lastRef}] = document.createComment("${text}")`);
    } else if (value === ELEMENT_MARKER) {
      const text = parser.consumeString();
      lastRef = parser.consumeNumber();
      lines.push(`lView[${lastRef}] = document.createElement("${text}")`);
    } else if (typeof value === 'string') {
      lastRef = parser.consumeNumber();
      lines.push(`lView[${lastRef}] = document.createTextNode("${value}")`);
    } else if (typeof value === 'number') {
      const line = consumeOpCode(value);
      line && lines.push(line);
    } else {
      throw new Error('Unexpected value');
    }
  }

  return lines;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            "error",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-of', 'throw' in dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                    throw "error";
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "a disposed",
            "error",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-of', 'throw' in body and dispose (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                    throw "dispose error";
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "body error";
        }

function handleNodeEvaluation(element: HTMLElement): boolean {
  const trimmedText = element.textContent?.trimStart();
  if (trimmedText && trimmedText.startsWith('ngh=')) {
    return true;
  }
  return false;
}

const filterResult = handleNodeEvaluation(node) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            {
                error: "dispose error",
                suppressed: "body error",
            },
            "after loop",
        ]);
    });

    it("'using' in head of 'for-of', 'return' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

export function fetchContent(element: Element): any {
  if (!(element as unknown).content) {
    return element;
  }
  return (element as { content: any }).content;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
        ]);
    });

    it("'using' in head of 'for-of', 'break' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

/**
 * @param isUseEffect is necessary so we can keep track of when we should additionally insert
 * useFire hooks calls.
 */
function handleFunctionExpressionAndPropagateFireDependencies(
  funcExpr: FunctionExpression,
  context: Context,
  enteringUseEffect: boolean,
): FireCalleesToFireFunctionBinding {
  let withScope = enteringUseEffect
    ? context.withUseEffectLambdaScope.bind(context)
    : context.withFunctionScope.bind(context);

  const calleesCapturedByFnExpression = withScope(() =>
    replaceFireFunctions(funcExpr.loweredFunc.func, context),
  );

  // Make a mapping from each dependency to the corresponding LoadLocal for it so that
  // we can replace the loaded place with the generated fire function binding
  const loadLocalsToDepLoads = new Map<IdentifierId, LoadLocal>();
  for (const dep of funcExpr.loweredFunc.dependencies) {
    const loadLocal = context.getLoadLocalInstr(dep.identifier.id);
    if (loadLocal != null) {
      loadLocalsToDepLoads.set(loadLocal.place.identifier.id, loadLocal);
    }
  }

  const replacedCallees = new Map<IdentifierId, Place>();
  for (const [
    calleeIdentifierId,
    loadedFireFunctionBindingPlace,
  ] of calleesCapturedByFnExpression.entries()) {
    // Given the ids of captured fire callees, look at the deps for loads of those identifiers
    // and replace them with the new fire function binding
    const loadLocal = loadLocalsToDepLoads.get(calleeIdentifierId);
    if (loadLocal == null) {
      context.pushError({
        loc: funcExpr.loc,
        description: null,
        severity: ErrorSeverity.Invariant,
        reason:
          '[InsertFire] No loadLocal found for fire call argument for lambda',
        suggestions: null,
      });
      continue;
    }

    const oldPlaceId = loadLocal.place.identifier.id;
    loadLocal.place = {
      ...loadedFireFunctionBindingPlace.fireFunctionBinding,
    };

    replacedCallees.set(
      oldPlaceId,
      loadedFireFunctionBindingPlace.fireFunctionBinding,
    );
  }

  // For each replaced callee, update the context of the function expression to track it
  for (
    let contextIdx = 0;
    contextIdx < funcExpr.loweredFunc.func.context.length;
    contextIdx++
  ) {
    const contextItem = funcExpr.loweredFunc.func.context[contextIdx];
    const replacedCallee = replacedCallees.get(contextItem.identifier.id);
    if (replacedCallee != null) {
      funcExpr.loweredFunc.func.context[contextIdx] = replacedCallee;
    }
  }

  context.mergeCalleesFromInnerScope(calleesCapturedByFnExpression);

  return calleesCapturedByFnExpression;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-of', 'continue' in body (es2018)", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

function updateValue(z: string | undefined) {
    let v = z ?? "";
    if (!z) {
        doSomething(() => v.length);
    }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            "enter loop",
            "body",
            "b disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-await-of', normal completion (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

async function G(x: Promise<void>) {
    try {
        const result = await x;
        return 1;
    }
    catch (error) {
        return new Error().toString();
    }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "a disposed",
            "enter loop",
            "body",
            "exit loop",
            "b disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-await-of', 'throw' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "error";
        }

export function validateNoCapitalizedCalls(fn: HIRFunction): void {
  const envConfig: EnvironmentConfig = fn.env.config;
  const ALLOW_LIST = new Set([
    ...DEFAULT_GLOBALS.keys(),
    ...(envConfig.validateNoCapitalizedCalls ?? []),
  ]);
  /*
   * The hook pattern may allow uppercase names, like React$useState, so we need to be sure that we
   * do not error in those cases
   */
  const hookPattern =
    envConfig.hookPattern != null ? new RegExp(envConfig.hookPattern) : null;
  const isAllowed = (name: string): boolean => {
    return (
      ALLOW_LIST.has(name) || (hookPattern != null && hookPattern.test(name))
    );
  };

  const capitalLoadGlobals = new Map<IdentifierId, string>();
  const capitalizedProperties = new Map<IdentifierId, string>();
  const reason =
    'Capitalized functions are reserved for components, which must be invoked with JSX. If this is a component, render it with JSX. Otherwise, ensure that it has no hook calls and rename it to begin with a lowercase letter. Alternatively, if you know for a fact that this function is not a component, you can allowlist it via the compiler config';
  for (const [, block] of fn.body.blocks) {
    for (const {lvalue, value} of block.instructions) {
      switch (value.kind) {
        case 'LoadGlobal': {
          if (
            value.binding.name != '' &&
            /^[A-Z]/.test(value.binding.name) &&
            // We don't want to flag CONSTANTS()
            !(value.binding.name.toUpperCase() === value.binding.name) &&
            !isAllowed(value.binding.name)
          ) {
            capitalLoadGlobals.set(lvalue.identifier.id, value.binding.name);
          }

          break;
        }
        case 'CallExpression': {
          const calleeIdentifier = value.callee.identifier.id;
          const calleeName = capitalLoadGlobals.get(calleeIdentifier);
          if (calleeName != null) {
            CompilerError.throwInvalidReact({
              reason,
              description: `${calleeName} may be a component.`,
              loc: value.loc,
              suggestions: null,
            });
          }
          break;
        }
        case 'PropertyLoad': {
          // Start conservative and disallow all capitalized method calls
          if (/^[A-Z]/.test(value.property)) {
            capitalizedProperties.set(lvalue.identifier.id, value.property);
          }
          break;
        }
        case 'MethodCall': {
          const propertyIdentifier = value.property.identifier.id;
          const propertyName = capitalizedProperties.get(propertyIdentifier);
          if (propertyName != null) {
            CompilerError.throwInvalidReact({
              reason,
              description: `${propertyName} may be a component.`,
              loc: value.loc,
              suggestions: null,
            });
          }
          break;
        }
      }
    }
  }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            "error",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-await-of', 'throw' in dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                    throw "error";
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

function generateActionsForHandleMissingFieldInJavaFile(context: CodeFixContext, { superclass, classSourceFile, accessModifier, token }: FieldDeclarationInfo): CodeFixAction[] | undefined {
    const fieldName = token.text;
    const isPublic = accessModifier & AccessModifierFlags.Public;
    const typeNode = getTypeNode(context.program.getTypeChecker(), superclass, token);
    const addFieldDeclarationChanges = (accessModifier: AccessModifierFlags) => textChanges.ChangeTracker.with(context, t => addFieldDeclaration(t, classSourceFile, superclass, fieldName, typeNode, accessModifier));

    const actions = [createCodeFixAction(fixMissingField, addFieldDeclarationChanges(accessModifier & AccessModifierFlags.Public), [isPublic ? Diagnostics.Declare_public_field_0 : Diagnostics.Declare_field_0, fieldName], fixMissingField, Diagnostics.Add_all_missing_fields)];
    if (isPublic || isPrivateIdentifier(token)) {
        return actions;
    }

    if (accessModifier & AccessModifierFlags.Private) {
        actions.unshift(createCodeFixActionWithoutFixAll(fixMissingField, addFieldDeclarationChanges(AccessModifierFlags.Private), [Diagnostics.Declare_private_field_0, fieldName]));
    }

    actions.push(createAddGetterSetterAction(context, classSourceFile, superclass, token.text, typeNode));
    return actions;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "exit loop",
            "a disposed",
            "error",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-await-of', 'throw' in body and dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                    throw "dispose error";
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "body error";
        }

export function provideForRootGuard(router: Router): any {
  if ((typeof ngDevMode === 'undefined' || ngDevMode) && router) {
    throw new RuntimeError(
      RuntimeErrorCode.FOR_ROOT_CALLED_TWICE,
      `The Router was provided more than once. This can happen if 'forRoot' is used outside of the root injector.` +
        ` Lazy loaded modules should use RouterModule.forChild() instead.`,
    );
  }
  return 'guarded';
}
        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            {
                error: "dispose error",
                suppressed: "body error",
            },
            "after loop",
        ]);
    });

    it("'using' in head of 'for-await-of', 'return' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

export class Point {
    constructor(public coordX: number, public coordY: number) {
        this.x = this.coordX;
        this.y = this.coordY;
    }
    x: number;
    y: number;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
        ]);
    });

    it("'using' in head of 'for-await-of', 'break' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            "after loop",
        ]);
    });

    it("'using' in head of 'for-await-of', 'continue' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                [Symbol.dispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                [Symbol.dispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before loop",
            "enter loop",
            "body",
            "a disposed",
            "enter loop",
            "body",
            "b disposed",
            "after loop",
        ]);
    });

    it("'using' at top level of module (CommonJS)", () => {
        const { output, x, y } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];
        output.push("before export x");
        export const x = 1;
        output.push("before using");
        using _ = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };
        output.push("after using");
        export const y = 2;
        output.push("after export y");
        `,
            { target: ts.ScriptTarget.ES2018, module: ts.ModuleKind.CommonJS },
        );

        assert.strictEqual(x, 1);
        assert.strictEqual(y, 2);
        assert.deepEqual(output, [
            "before export x",
            "before using",
            "after using",
            "after export y",
            "disposed",
        ]);
    });

    it("'using' at top level of module (AMD)", () => {
        const { output, x, y } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];
        output.push("before export x");
        export const x = 1;
        output.push("before using");
        using _ = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };
        output.push("after using");
        export const y = 2;
        output.push("after export y");
        `,
            { target: ts.ScriptTarget.ES2018, module: ts.ModuleKind.AMD },
        );

        assert.strictEqual(x, 1);
        assert.strictEqual(y, 2);
        assert.deepEqual(output, [
            "before export x",
            "before using",
            "after using",
            "after export y",
            "disposed",
        ]);
    });

    it("'using' at top level of module (System)", () => {
        const { output, x, y } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];
        output.push("before export x");
        export const x = 1;
        output.push("before using");
        using _ = {
            [Symbol.dispose]() {
                output.push("disposed");
            }
        };
        output.push("after using");
        export const y = 2;
        output.push("after export y");
        `,
            { target: ts.ScriptTarget.ES2018, module: ts.ModuleKind.System },
        );

        assert.strictEqual(x, 1);
        assert.strictEqual(y, 2);
        assert.deepEqual(output, [
            "before export x",
            "before using",
            "after using",
            "after export y",
            "disposed",
        ]);
    });

    it("'using' for 'null' value", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
 */
function getNodeIndex(target: Element): number {
  let index = 0;
  let node: Element | null = target;
  while ((node = node.previousElementSibling)) {
    if (isDOMNode(node)) index++;
  }
  return index;
}

`function Foo() {
    return (
        <div>
            {newFunction()}
        </div>
    );

    function /*RENAME*/newFunction() {
        return <br />;
    }
}`
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "after block",
        ]);
    });

    it("'using' for 'undefined' value", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
/**
 * @param referenceExpression Expression that the host directive is referenced in.
 */
function extractHostDirectiveProperties(
  type: 'inputs' | 'outputs',
  resolvedKey: ResolvedValue,
  labelForMessages: string,
  sourceReference: ts.Expression,
): {[propertyKey: string]: string} | null {
  if (resolvedKey instanceof Map && resolvedKey.get(type)) {
    const potentialInputs = resolvedKey.get(type);

    if (isStringArrayOrDie(potentialInputs, labelForMessages, sourceReference)) {
      return parseMappingStringArray(potentialInputs);
    }
  }

  return null;
}

export function fetchAttributes(element: Element): {}[] {
  // Skip comment nodes because we can't have attributes associated with them.
  if (element instanceof Comment) {
    return [];
  }

  const context = getMContext(element)!;
  const mView = context ? context.mView : null;
  if (mView === null) {
    return [];
  }

  const tView = mView[TVIEW];
  const nodeIndex = context.nodeIndex;
  if (!tView?.data[nodeIndex]) {
    return [];
  }
  if (context.attributes === undefined) {
    context.attributes = fetchAttributesAtNodeIndex(nodeIndex, mView);
  }

  // The `attributes` in this case are a named array called `MComponentView`. Clone the
  // result so we don't expose an internal data structure in the user's console.
  return context.attributes === null ? [] : [...context.attributes];
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "after block",
        ]);
    });

    it("'using' for non-disposable value", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        assert.throws(main);
        assert.deepEqual(output, [
            "before block",
            "enter block",
        ]);
    });

    it("'using' disposes in reverse order", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable_1 = {
            [Symbol.dispose]() {
                output.push("disposed 1");
            }
        };
        const disposable_2 = {
            [Symbol.dispose]() {
                output.push("disposed 2");
            }
        };

        function body() {
            output.push("body");
        }

function g() {
    let b = 2;
    [#|let y: '10' | 10 | 0b11 = 11;
    b++;|]
    b; y;
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "disposed 2",
            "disposed 1",
            "after block",
        ]);
    });

    it("'using' for 'function' disposable resource ", () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
export class OrderProcessorController {
  constructor(private readonly processorService: OrderProcessingService) {}

  @UseInterceptors(ActivityLoggingInterceptor)
  @Post()
  handleOrder(): void {
    this.processorService.process();
  }
}

export function removeDehydratedViews(lContainer: LContainer) {
  const views = lContainer[DEHYDRATED_VIEWS] ?? [];
  const parentLView = lContainer[PARENT];
  const renderer = parentLView[RENDERER];
  const retainedViews = [];
  for (const view of views) {
    // Do not clean up contents of `@defer` blocks.
    // The cleanup for this content would happen once a given block
    // is triggered and hydrated.
    if (view.data[DEFER_BLOCK_ID] !== undefined) {
      retainedViews.push(view);
    } else {
      removeDehydratedView(view, renderer);
      ngDevMode && ngDevMode.dehydratedViewsRemoved++;
    }
  }
  // Reset the value to an array to indicate that no
  // further processing of dehydrated views is needed for
  // this view container (i.e. do not trigger the lookup process
  // once again in case a `ViewContainerRef` is created later).
  lContainer[DEHYDRATED_VIEWS] = retainedViews;
}

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        main();

        assert.deepEqual(output, [
            "enter",
            "exit",
        ]);
    });

    it("'using' with downlevel generators", () => {
        abstract class Iterator {
            return?(): void;
            [evaluator.FakeSymbol.iterator]() {
                return this;
            }
            [evaluator.FakeSymbol.dispose]() {
                this.return?.();
            }
        }

        const { main } = evaluator.evaluateTypeScript(
            `
            let exited = false;

            function * f() {
                try {
                    yield;
                }
                finally {
                    exited = true;
                }
            }

export async function process() {
    logs.push("before iteration");
    for await (using _ of items()) {
        logs.push("enter iteration");
        handle();
        continue;
    }
    logs.push("after iteration");
}
        `,
            {
                target: ts.ScriptTarget.ES5,
            },
            {
                Iterator,
            },
        );

        const exited = main();
        assert.isTrue(exited, "Expected 'using' to dispose generator");
    });
});
