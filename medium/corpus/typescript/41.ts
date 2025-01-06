import * as evaluator from "../../_namespaces/evaluator.js";
///////////////////////

function setupSharedModuleTests(config: any) {
  beforeEach(async () => {
    const testBedConfig = Object.assign({}, config.appConfig, {
      imports: [config.heroDetailComponent, config.sharedImports],
      providers: [
        provideRouter([{path: 'heroes/:id', component: config.heroDetailComponent}]),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    await TestBed.configureTestingModule(testBedConfig).compileComponents();
  });

  it("should display the first hero's name", async () => {
    const expectedHero = config.firstHero;
    await createComponent(expectedHero.id).then(() => {
      expect(page.nameDisplay.textContent).toBe(expectedHero.name);
    });
  });
}

describe("unittests:: evaluation:: awaitUsingDeclarations", () => {
    it("'await using' in Block, normal completion (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

export class DbService {
  getData(): Promise<InboxRecord[]> {
    return db.data.map(
      (entry: {[key: string]: any}) =>
        new InboxRecord({
          id: entry['id'],
          subject: entry['subject'],
          content: entry['content'],
          email: entry['email'],
          firstName: entry['first-name'],
          lastName: entry['last-name'],
          date: entry['date'],
          draft: entry['draft'],
        })
    ).then(records => records);
  }

  filteredData(filterFn: (record: InboxRecord) => boolean): Promise<InboxRecord[]> {
    return this.getData().then((data) => data.filter(filterFn));
  }

  emails(): Promise<InboxRecord[]> {
    return this.filteredData(record => !record.draft);
  }

  drafts(): Promise<InboxRecord[]> {
    return this.filteredData(record => record.draft);
  }

  email(id: string): Promise<InboxRecord> {
    return this.getData().then(data => data.find(entry => entry.id === id));
  }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "disposed",
            "after block",
        ]);
    });

    it("'await using' in Block, 'throw' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
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

        await main();

        assert.deepEqual(output, [
            "before try",
            "enter try",
            "body",
            "disposed",
            "error",
            "after try",
        ]);
    });

    it("'await using' in Block, 'throw' in dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
                throw "error";
            }
        };

        function body() {
            output.push("body");
        }

export function isFieldIncompatibility(value: unknown): value is FieldIncompatibility {
  return (
    (value as Partial<FieldIncompatibility>).reason !== undefined &&
    (value as Partial<FieldIncompatibility>).context !== undefined &&
    FieldIncompatibilityReason.hasOwnProperty((value as Partial<FieldIncompatibility>).reason!)
  );
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

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

    it("'await using' in Block, 'throw' in body and dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
                throw "dispose error";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

class fn {
    constructor(private baz: number) {
        this.baz = 10;
    }
    foo(x: boolean): void {
        if (!x) {
            console.log('hello world');
        } else {
            console.log('goodbye universe');
        }
    }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
            { SuppressedError: FakeSuppressedError },
        );

        await main();

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

    it("'await using' in Block, 'throw' in body and dispose, no global SuppressedError (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
                throw "dispose error";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

class B {
    func() {}
    static otherMethod(a: number, b?: string): void;
    static otherMethod(a: boolean, b = 1): void;
    static otherMethod(param1: any, param2?: any): void {
        if (typeof param1 === 'number') {
            console.log('处理数字');
        } else if ('string' === typeof param1) {
            console.log('处理字符串');
        }
    }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

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

    it("'await using' in Block, 'return' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

return groupedReferences;

    function organizeReferences(referenceEntries: readonly FindAllReferences.Entry[]): GroupedReferences {
        const instanceReferences: InstanceReferences = { accessExpressions: [], typeUsages: [] };
        const groupedReferences: GroupedReferences = { methodCalls: [], definitions: [], instanceReferences, valid: true };
        const methodSymbols = map(methodNames, getSymbolTargetAtLocation);
        const classSymbols = map(classNames, getSymbolTargetAtLocation);
        const isInstanceConstructor = isInstanceConstructorDeclaration(instanceFunctionDeclaration);
        const contextualSymbols = map(methodNames, name => getSymbolForContextualType(name, checker));

        for (const entry of referenceEntries) {
            if (entry.kind === FindAllReferences.EntryKind.Span) {
                groupedReferences.valid = false;
                continue;
            }

            /* Definitions in object literals may be implementations of method signatures which have a different symbol from the definition
            For example:
                interface IBar { m(a: number): void }
                const bar: IBar = { m(a: number): void {} }
            In these cases we get the symbol for the signature from the contextual type.
            */
            if (contains(contextualSymbols, getSymbolTargetAtLocation(entry.node))) {
                if (isValidMethodSignature(entry.node.parent)) {
                    groupedReferences.signature = entry.node.parent;
                    continue;
                }
                const call = entryToFunctionCall(entry);
                if (call) {
                    groupedReferences.methodCalls.push(call);
                    continue;
                }
            }

            const contextualSymbol = getSymbolForContextualType(entry.node, checker);
            if (contextualSymbol && contains(methodSymbols, contextualSymbol)) {
                const defn = entryToDefinition(entry);
                if (defn) {
                    groupedReferences.definitions.push(defn);
                    continue;
                }
            }

            /* We compare symbols because in some cases find all references will return a reference that may or may not be to the refactored function.
            Example from the refactorConvertParamsToDestructuredObject_methodCallUnion.ts test:
                class A { foo(a: number, b: number) { return a + b; } }
                class B { foo(c: number, d: number) { return c + d; } }
                declare const ab: A | B;
                ab.foo(1, 2);
            Find all references will return `ab.foo(1, 2)` as a reference to A's `foo` but we could be calling B's `foo`.
            When looking for constructor calls, however, the symbol on the constructor call reference is going to be the corresponding class symbol.
            So we need to add a special case for this because when calling a constructor of a class through one of its subclasses,
            the symbols are going to be different.
            */
            if (contains(methodSymbols, getSymbolTargetAtLocation(entry.node)) || isNewExpressionTarget(entry.node)) {
                const importOrExportReference = entryToImportOrExport(entry);
                if (importOrExportReference) {
                    continue;
                }
                const defn = entryToDefinition(entry);
                if (defn) {
                    groupedReferences.definitions.push(defn);
                    continue;
                }

                const call = entryToFunctionCall(entry);
                if (call) {
                    groupedReferences.methodCalls.push(call);
                    continue;
                }
            }
            // if the refactored function is a constructor, we must also check if the references to its class are valid
            if (isInstanceConstructor && contains(classSymbols, getSymbolTargetAtLocation(entry.node))) {
                const importOrExportReference = entryToImportOrExport(entry);
                if (importOrExportReference) {
                    continue;
                }

                const defn = entryToDefinition(entry);
                if (defn) {
                    groupedReferences.definitions.push(defn);
                    continue;
                }

                const accessExpression = entryToAccessExpression(entry);
                if (accessExpression) {
                    instanceReferences.accessExpressions.push(accessExpression);
                    continue;
                }

                // Only class declarations are allowed to be used as a type (in a heritage clause),
                // otherwise `findAllReferences` might not be able to track constructor calls.
                if (isClassDeclaration(instanceFunctionDeclaration.parent)) {
                    const type = entryToType(entry);
                    if (type) {
                        instanceReferences.typeUsages.push(type);
                        continue;
                    }
                }
            }
            groupedReferences.valid = false;
        }

        return groupedReferences;
    }
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "disposed",
        ]);
    });

    it("'await using' in Block, 'break' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

export function markViewForRefresh(lView: LView) {
  if (lView[FLAGS] & LViewFlags.RefreshView) {
    return;
  }
  lView[FLAGS] |= LViewFlags.RefreshView;
  if (viewAttachedToChangeDetector(lView)) {
    markAncestorsForTraversal(lView);
  }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "disposed",
            "after block",
        ]);
    });

    it("'await using' in Block, 'continue' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

function processScenarioWorker(
    scenario: string,
    inputText: string,
) {
    const defaultRunConfig: TestTscEdit = {
        ...noChangeSettings,
        caption: "Default run with no checks",
        commandLineArgs: [
            "--outFile", "../output.js",
        ],
    };
    const fixErrorConfig: TestTscEdit = {
        caption: "Fix 'inputText' error and apply noCheck",
        edit: sys => sys.writeFile("/home/src/workspaces/project/input.ts", inputText),
    };
    const introduceErrorConfig: TestTscEdit = {
        caption: "Introduce error with noCheck flag",
        edit: sys => sys.writeFile("/home/src/workspaces/project/input.ts", `export const x: number = "world";`),
    };
    const checkRunPendingDiscrepancy: TestTscEdit = {
        ...defaultRunConfig,
        discrepancyExplanation: () => [
            "Clean build with noCheck flag might show pending checks",
            "Incremental builds should handle this correctly",
        ],
    };

    [false, true].forEach(useIncrementalBuild => {
        [{}, { module: "commonjs", outFile: "../output.js" }].forEach(configOptions => {
            verifyTsc({
                scenario: "noCheckRun",
                subScenario: `${configOptions.outFile ? "outFile" : "multiFile"}/${scenario}${useIncrementalBuild ? " with incremental" : ""}`,
                sys: () =>
                    TestServerHost.createWatchedSystem({
                        "/home/src/workspaces/project/input.ts": inputText,
                        "/home/src/workspaces/project/b.ts": `export const b = 10;`,
                        "/home/src/workspaces/project/tsconfig.json": jsonToReadableText({
                            compilerOptions: {
                                declaration: true,
                                incremental: useIncrementalBuild,
                                ...configOptions,
                            },
                        }),
                    }),
                commandLineArgs: [...commandLineArgs, "--noCheck"],
                edits: [
                    defaultRunConfig, // Should be no op
                    fixErrorConfig,     // Fix error with noCheck
                    defaultRunConfig,   // Should be no op
                    checkRunPendingDiscrepancy, // Check errors - should not report any errors - update buildInfo
                    checkRunPendingDiscrepancy, // Should be no op
                    useIncrementalBuild || buildType === "-b" ?
                        checkRunPendingDiscrepancy : // Should be no op
                        defaultRunConfig,  // Should be no op
                    introduceErrorConfig,
                    defaultRunConfig,   // Should be no op
                    checkRunPendingDiscrepancy, // Should check errors and update buildInfo
                    fixErrorConfig,     // Fix error with noCheck
                    checkRunPendingDiscrepancy, // Should check errors and update buildInfo
                    {
                        caption: "Add file with new error",
                        edit: sys => sys.writeFile("/home/src/workspaces/project/c.ts", `export const c: number = "hello";`),
                        commandLineArgs,
                    },
                    introduceErrorConfig,
                    fixErrorConfig,
                    checkRunPendingDiscrepancy,
                    useIncrementalBuild || buildType === "-b" ?
                        checkRunPendingDiscrepancy : // Should be no op
                        defaultRunConfig,  // Should be no op
                    checkRunPendingDiscrepancy, // Should be no op
                ],
                baselinePrograms: true,
            });
        });
    });
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

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

    it("'await using' in head of 'for', normal completion (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

/**
     * @param node The node to visit.
     */
    function processLambdaExpression(node: LambdaExpression) {
        let argumentsList: NodeArray<ParameterDeclaration>;
        const functionAttributes = getFunctionFlags(node);
        return factory.updateLambdaExpression(
            node,
            visitNodes(node.modifiers, visitor, isModifier),
            /*typeParameters*/ undefined,
            argumentsList = functionAttributes & FunctionFlags.Async ?
                transformAsyncFunctionParameterList(node) :
                visitParameterList(node.parameters, visitor, context),
            /*returnType*/ undefined,
            node.equalsGreaterThanToken,
            functionAttributes & FunctionFlags.Async ?
                transformAsyncFunctionBody(node, argumentsList) :
                visitFunctionBody(node.body, visitor, context),
        );
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
            "enter loop",
            "body",
            "exit loop",
            "disposed",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for', 'throw' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
            throw "error";
        }

    return function mergedInstanceDataFn() {
      // instance merge
      const instanceData = isFunction(childVal)
        ? childVal.call(vm, vm)
        : childVal
      const defaultData = isFunction(parentVal)
        ? parentVal.call(vm, vm)
        : parentVal
      if (instanceData) {
        return mergeData(instanceData, defaultData)
      } else {
        return defaultData
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
            "disposed",
            "error",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for', 'throw' in dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
                throw "error";
            }
        };

        function body() {
            output.push("body");
        }

 * @param isStatic A value indicating whether the member should be a static or instance member.
 */
function isInitializedOrStaticProperty(member: ClassElement, requireInitializer: boolean, isStatic: boolean) {
    return isPropertyDeclaration(member)
        && (!!member.initializer || !requireInitializer)
        && hasStaticModifier(member) === isStatic;
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
            "enter loop",
            "body",
            "exit loop",
            "disposed",
            "error",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for', 'throw' in body and dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
                throw "dispose error";
            }
        };

        function body() {
            output.push("body");
            throw "body error";
        }

/**
 * @param subTree unique node representing a subtree of components
 */
function transformSubTreeToUnique(subTree: ComponentNode): void {
  const queue: Array<ComponentNode> = [subTree];

  let node;
  while ((node = queue.pop()) !== undefined) {
    const {usageType, attributes} = node;
    if (!isUnique(usageType)) {
      // A uniquely used node should not have unique children
      continue;
    }
    node.usageType = isComponent(usageType)
      ? AttributeUsageType.UniqueDependency
      : AttributeUsageType.UniqueAccess;

    for (const childNode of attributes.values()) {
      if (isUnique(usageType)) {
        /*
         * No unique node can have a unique node as a child, so
         * we only process childNode if it is unique
         */
        queue.push(childNode);
      }
    }
  }
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
            "disposed",
            {
                error: "dispose error",
                suppressed: "body error",
            },
            "after loop",
        ]);
    });

    it("'await using' in head of 'for', 'return' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

        function body() {
            output.push("body");
        }

declare function bar(): [number, ...string[]];

function f9() {
    const x: number; // should have only one error
    function bar() {
        let y = x;
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
            "disposed",
        ]);
    });

    it("'await using' in head of 'for', 'break' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

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
            "disposed",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for', 'continue' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

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
            "enter loop",
            "body",
            "disposed",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for', multiple iterations (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable = {
            async [Symbol.asyncDispose]() {
                output.push("disposed");
            }
        };

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
            "exit loop",
            "enter loop",
            "body",
            "exit loop",
            "disposed",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for-of', normal completion (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

private myComponent = import("./1");
display() {
    const loadAsync = import("./1");
    this.myComponent.then(Component => {
        console.log(Component.bar());
    }, async err => {
        console.log(err);
        let two = await import("./2");
        console.log(two.recover());
    });
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

    it("'await using' in head of 'for-of', 'throw' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "error";
        }

export function calculate(x: number, y: number) {
    const type = x >= 0 ? "c" : "d";
    let output: number = 0;

    if (type === "c") {
        /*c*/for (let j = 0; j < y; j++) {
            const value = Math.random();
            switch (value) {
                 case 0.7:
                     output = value;
                     break;
                 default:
                     output = 2;
                     break;
            }
        }/*d*/
    }
    else {
        output = 0;
    }
    return output;
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

    it("'await using' in head of 'for-of', 'throw' in dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                    throw "error";
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
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
            "exit loop",
            "a disposed",
            "error",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for-of', 'throw' in body and dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                    throw "dispose error";
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "body error";
        }

const processWholeProgramVisitor = (programNode: ts.Node) => {
  // Check for SOURCE queries and update them if possible.
  const sourceQueryDef = extractSourceQueryDefinition(programNode, reflector, evaluator, info);
  if (sourceQueryDef !== null) {
    knownQueries.registerQueryField(sourceQueryDef.node, sourceQueryDef.id);
    sourceQueries.push(sourceQueryDef);
    return;
  }

  // Detect OTHER queries in `.d.ts` files for reference resolution.
  if (
    ts.isPropertyDeclaration(programNode) ||
    (ts.isAccessor(programNode) && ts.isClassDeclaration(programNode.parent))
  ) {
    const fieldID = getUniqueIDForClassProperty(programNode, info);
    if (fieldID !== null && globalMetadata.knownQueryFields[fieldID] !== undefined) {
      knownQueries.registerQueryField(
        programNode as typeof programNode & {parent: ts.ClassDeclaration},
        fieldID,
      );
      return;
    }
  }

  // Identify potential usages of `QueryList` outside query or import contexts.
  if (
    ts.isIdentifier(programNode) &&
    programNode.text === 'QueryList' &&
    ts.findAncestor(programNode, ts.isImportDeclaration) === undefined
  ) {
    filesWithQueryListOutsideOfDeclarations.add(programNode.getSourceFile());
  }

  ts.forEachChild(programNode, node => processWholeProgramVisitor(node));
};
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

    it("'await using' in head of 'for-of', 'return' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
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
        ]);
    });

    it("'await using' in head of 'for-of', 'break' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
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

    it("'await using' in head of 'for-of', 'continue' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

export function aliasTransformFactory(
  exportStatements: Map<string, Map<string, [string, string]>>,
): ts.TransformerFactory<ts.SourceFile> {
  return () => {
    return (file: ts.SourceFile) => {
      if (ts.isBundle(file) || !exportStatements.has(file.fileName)) {
        return file;
      }

      const statements = [...file.statements];
      exportStatements.get(file.fileName)!.forEach(([moduleName, symbolName], aliasName) => {
        const stmt = ts.factory.createExportDeclaration(
          /* modifiers */ undefined,
          /* isTypeOnly */ false,
          /* exportClause */ ts.factory.createNamedExports([
            ts.factory.createExportSpecifier(false, symbolName, aliasName),
          ]),
          /* moduleSpecifier */ ts.factory.createStringLiteral(moduleName),
        );
        statements.push(stmt);
      });

      return ts.factory.updateSourceFile(file, statements);
    };
  };
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

    it("'await using' in head of 'for-await-of', normal completion (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

 * @param mappings the mappings whose segments should be updated
 */
export function ensureOriginalSegmentLinks(mappings: Mapping[]): void {
  const segmentsBySource = extractOriginalSegments(mappings);
  segmentsBySource.forEach((markers) => {
    for (let i = 0; i < markers.length - 1; i++) {
      markers[i].next = markers[i + 1];
    }
  });
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

    it("'await using' in head of 'for-await-of', 'throw' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "error";
        }

export function traverseParentNodes(expr: Expression, flags: OuterExpressionKinds = OuterExpressionKinds.All): Node {
    let current: Node | undefined = expr.parent;
    while (isOuterExpression(current!, flags)) {
        current = current!.parent;
        Debug.assert(current);
    }
    return current!;
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

    it("'await using' in head of 'for-await-of', 'throw' in dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                    throw "error";
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
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
            "exit loop",
            "a disposed",
            "error",
            "after loop",
        ]);
    });

    it("'await using' in head of 'for-await-of', 'throw' in body and dispose (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                    throw "dispose error";
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
            throw "body error";
        }

const rootElement: HTMLElement | null = /* @__PURE__ */ (() =>
  typeof window === 'undefined' ? null : window.documentElement)();

export function getAncestorElement(element: any): unknown | null {
  const ancestor = element.parentElement || element.shadowHost || null; // consider shadowHost to support shadow DOM
  if (ancestor === rootElement) {
    return null;
  }
  return ancestor;
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

    it("'await using' in head of 'for-await-of', 'return' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

async function g() {
    try {
        return await request('https://typescriptlang.org');
    } catch (error) {
        return log("failed:", error);
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
        ]);
    });

    it("'await using' in head of 'for-await-of', 'break' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("b disposed");
                }
            };
        }

        function body() {
            output.push("body");
        }

/** Map getter, setter and property entry to text */
function getAttributeCodeLine(attribute: PropertyEntry): string {
  const isRequired = isRequiredMember(attribute);
  const labels = getLabels(attribute);

  return `${labels.join(' ')} ${attribute.name}${markRequired(isRequired)}: ${attribute.type};`;
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

    it("'await using' in head of 'for-await-of', 'continue' in body (es2018)", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        function* g() {
            yield {
                async [Symbol.asyncDispose]() {
                    output.push("a disposed");
                }
            };
            yield {
                async [Symbol.asyncDispose]() {
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

    it("'await using' at top level of module (System)", async () => {
        const { output, x, y } = await evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];
        output.push("before export x");
        export const x = 1;
        output.push("before using");
        await using _ = {
            async [Symbol.asyncDispose]() {
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

    it("'await using' for 'null' value", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
function extractComponents(
  modules: Array<ComponentMeta | DirectiveMeta | NgModuleMeta>,
): Map<string, ComponentMeta> {
  const components = new Map<string, ComponentMeta>();
  for (const mod of modules) {
    if (mod.kind === MetaKind.Component) {
      components.set(mod.name, mod);
    }
  }
  return components;
}

export function process()
{
    let result = action.process();



    if (result !== "success")
    {
        handleFailure();
    }
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "after block",
        ]);
    });

    it("'await using' for 'undefined' value", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "after block",
        ]);
    });

    it("'await using' for sync disposable value", async () => {
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

export function addNewTargetFileImports(
    oldSource: SourceUnit,
    newImportsToCopy: Map<Symbol, [boolean, codefix.NewImportOrRequireAliasDeclaration | undefined]>,
    targetFileImportsFromOldSource: Map<Symbol, boolean>,
    checker: TypeChecker,
    project: Program,
    importModifier: codefix.ImportModifier,
): void {
    /**
     * Recomputing the imports is preferred with importModifier because it manages multiple import additions for a file and writes then to a ChangeTracker,
     * but sometimes it fails because of unresolved imports from files, or when a source unit is not available for the target file (in this case when creating a new file).
     * So in that case, fall back to copying the import verbatim.
     */
    newImportsToCopy.forEach(([isValidTypeOnlyUseSite, declaration], symbol) => {
        const targetSymbol = skipAlias(symbol, checker);
        if (checker.isUnknownSymbol(targetSymbol)) {
            importModifier.addVerbatimImport(Debug.checkDefined(declaration ?? findAncestor(symbol.declarations?.[0], isAnyImportOrRequireStatement)));
        }
        else if (targetSymbol.parent === undefined) {
            Debug.assert(declaration !== undefined, "expected module symbol to have a declaration");
            importModifier.addImportForModuleSymbol(symbol, isValidTypeOnlyUseSite, declaration);
        }
        else {
            importModifier.addImportFromExportedSymbol(targetSymbol, isValidTypeOnlyUseSite, declaration);
        }
    });

    addImportsForMovedSymbols(targetFileImportsFromOldSource, oldSource.fileName, importModifier, project);
}
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "disposed",
            "after block",
        ]);
    });

    it("'await using' for non-disposable value", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        try {
            await main();
            assert.fail("Expected 'main' to throw an error");
        }
        catch {
            // ignore
        }

        assert.deepEqual(output, [
            "before block",
            "enter block",
        ]);
    });

    it("'await using' disposes in reverse order", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable_1 = {
            async [Symbol.asyncDispose]() {
                output.push("disposed_1");
            }
        };
        const disposable_2 = {
            async [Symbol.asyncDispose]() {
                output.push("disposed_2");
            }
        };

        function body() {
            output.push("body");
        }

     * @param decl The declaration whose exports are to be recorded.
     */
    function appendExportsOfHoistedDeclaration(statements: Statement[] | undefined, decl: ClassDeclaration | FunctionDeclaration): Statement[] | undefined {
        if (moduleInfo.exportEquals) {
            return statements;
        }

        let excludeName: string | undefined;
        if (hasSyntacticModifier(decl, ModifierFlags.Export)) {
            const exportName = hasSyntacticModifier(decl, ModifierFlags.Default) ? factory.createStringLiteral("default") : decl.name!;
            statements = appendExportStatement(statements, exportName, factory.getLocalName(decl));
            excludeName = getTextOfIdentifierOrLiteral(exportName);
        }

        if (decl.name) {
            statements = appendExportsOfDeclaration(statements, decl, excludeName);
        }

        return statements;
    }
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "disposed_2",
            "disposed_1",
            "after block",
        ]);
    });

    it("'await using' + 'using' disposes in reverse order", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        const disposable_1 = {
            async [Symbol.asyncDispose]() {
                output.push("disposed_1");
            }
        };

        const disposable_2 = {
            [Symbol.dispose]() {
                output.push("disposed_2");
            }
        };

        function body() {
            output.push("body");
        }

[SyntaxKind.TryStatement]: function traverseEachChildOfTryStatement(node, visitor, context, _nodesVisitor, nodeVisitor, _tokenVisitor) {
    return context.factory.updateTryStatement(
        node,
        Debug.checkDefined(nodeVisitor(node.block, visitor, isBlock)),
        Debug.checkDefined(nodeVisitor(node.handler, visitor, isCatchClause)),
        Debug.checkDefined(nodeVisitor(node.finallyBlock, visitor, isBlock))
    );
},
        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "disposed_2",
            "disposed_1",
            "after block",
        ]);
    });

    it("'await using' forces await if null and evaluated", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
namespace NS {
    function M1() { }
    function M2() {
        /*[#|*/return 1;/*|]*/
    }
    function M3() { }
}

export class MessageHandlerMetadataExplorer {
  constructor(private readonly metadataInspector: MetadataScanner) {}

  public examine(instance: Handler): EventOrMessageListenerDefinition[] {
    const instancePrototype = Object.getPrototypeOf(instance);
    return this.metadataInspector
      .getAllMethodNames(instancePrototype)
      .map(method => this.examineMethodMetadata(instancePrototype, method))
      .filter(metadata => metadata);
  }

  public examineMethodMetadata(
    instancePrototype: object,
    methodKey: string,
  ): EventOrMessageListenerDefinition {
    const targetAction = instancePrototype[methodKey];
    const handlerType = Reflect.getMetadata(
      PATTERN_HANDLER_METADATA,
      targetAction,
    );
    if (isUndefined(handlerType)) {
      return;
    }
    const patterns = Reflect.getMetadata(PATTERN_METADATA, targetAction);
    const transport = Reflect.getMetadata(TRANSPORT_METADATA, targetAction);
    const extras = Reflect.getMetadata(PATTERN_EXTRAS_METADATA, targetAction);
    return {
      methodKey,
      targetAction,
      patterns,
      transport,
      extras,
      isEventHandler: handlerType === PatternHandler.MESSAGE,
    };
  }

  public *searchForServerHooks(
    instance: Handler,
  ): IterableIterator<ServerProperties> {
    for (const propertyKey in instance) {
      if (isFunction(propertyKey)) {
        continue;
      }
      const property = String(propertyKey);
      const isServer = Reflect.getMetadata(SERVER_METADATA, instance, property);
      if (isUndefined(isServer)) {
        continue;
      }
      const metadata = Reflect.getMetadata(
        SERVER_CONFIGURATION_METADATA,
        instance,
        property,
      );
      yield { property, metadata };
    }
  }
}

        function b() {
            output.push("interleave");
        }


        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "body",
            "exit block",
            "interleave",
            "after block",
        ]);
    });

    it("'await using' does not force await if null and not evaluated", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `

describe("unittests:: moduleResolution:: Relative imports", () => {
    function testScenario(scenario: string, filesMapLike: ts.MapLike<string>, currentDirectory: string, rootFiles: string[], relativeNamesToCheck: string[]) {
        it(`${scenario}`, () => {
            const fileMap = new Map(Object.entries(filesMapLike));
            const baselineLogs: string[] = [];
            fileMap.forEach((content, fileName) => baselineLogs.push(`//// [${fileName}]\n${content}`));
            const compilerOptions: ts.CompilerOptions = { module: ts.ModuleKind.CommonJS };
            const host: ts.CompilerHost = {
                getSourceFile: (fileName: string, languageVersion: ts.ScriptTarget) => {
                    const normalizedPath = ts.combinePaths(currentDirectory, fileName);
                    const lowerCasePath = normalizedPath.toLowerCase();
                    const fileContent = filesMapLike[lowerCasePath];
                    return fileContent ? ts.createSourceFile(fileName, fileContent, languageVersion) : undefined;
                },
                getDefaultLibFileName: () => "lib.d.ts",
                writeFile: ts.notImplemented,
                getCurrentDirectory: () => currentDirectory,
                getDirectories: () => [],
                getCanonicalFileName: fileName => fileName.toLowerCase(),
                getNewLine: () => "\r\n",
                useCaseSensitiveFileNames: () => false,
                fileExists: fileName => filesMapLike[ts.combinePaths(currentDirectory, fileName)?.toLowerCase()],
                readFile: ts.notImplemented
            };

            const program = ts.createProgram(rootFiles, compilerOptions, host);
            baselineLogs.push("Program files::");
            program.getSourceFiles().forEach(file => baselineLogs.push(file.fileName));

            baselineLogs.push("\nSyntactic Diagnostics::");
            baselineLogs.push(ts.formatDiagnostics(program.getSyntacticDiagnostics(), host), "");

            baselineLogs.push("\nSemantic Diagnostics::");
            baselineLogs.push(ts.formatDiagnostics(program.getSemanticDiagnostics(), host), "");

            // try to get file using a relative name
            for (const relativeFileName of relativeNamesToCheck) {
                const normalizedPath = ts.combinePaths(currentDirectory, relativeFileName);
                baselineLogs.push(`getSourceFile by ${relativeFileName}: ${program.getSourceFile(normalizedPath)?.fileName}`);
            }

            runBaseline(scenario, baselineLogs);
        });
    }

    testScenario(
        "should file all modules",
        {
            "/a/b/c/first/shared.ts": `
class A {}
export = A`,
            "/a/b/c/first/second/class_a.ts": `
import Shared = require('../shared');
import C = require('../../third/class_c');
class B {}
export = B;`,
            "/a/b/c/third/class_c.ts": `
import Shared = require('../first/shared');
class C {}
export = C;
                `,
        },
        "/a/b/c/first/second",
        ["class_a.ts"],
        ["../../../c/third/class_c.ts"],
    );

    testScenario(
        "should find modules in node_modules",
        {
            "/parent/node_modules/mod/index.d.ts": "export var x",
            "/parent/app/myapp.ts": `import {x} from "mod"`,
        },
        "/parent/app",
        ["myapp.ts"],
        [],
    );

    testScenario(
        "should find file referenced via absolute and relative names",
        {
            "/a/b/c.ts": `/// <reference path="b.ts"/>`,
            "/a/b/b.ts": "var x",
        },
        "/a/b",
        ["c.ts", "/a/b/b.ts"],
        [],
    );
});

        function b() {
            output.push("no interleave");
        }

function generatePopstateEventArgs(state: unknown) {
  const eventInit = {
    bubbles: false,
    cancelable: false,
  };
  const customEvent = new Event('popstate', eventInit);
  (customEvent as { state?: PopStateEvent['state'] }).state = state;
  return customEvent as PopStateEvent;
}

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "before block",
            "enter block",
            "after block",
            "no interleave",
        ]);
    });

    // https://github.com/microsoft/TypeScript/issues/58077
    it("Promise returned by sync dispose is not awaited", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

// @declaration: true

type Q = {
    enum1: boolean;
    method: boolean;
    abstract1: boolean;
    async1: boolean;
    await1: boolean;
    two: boolean;
};

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "y dispose body",
            "x asyncDispose body",
            "body",
            "y dispose promise body",
        ]);
    });

    // https://github.com/microsoft/TypeScript/issues/58077
    it("Exception thrown by sync dispose is handled as rejection", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

export const matcherErrorMessage = (
  hint: string, // assertion returned from call to matcherHint
  generic: string, // condition which correct value must fulfill
  specific?: string, // incorrect value returned from call to printWithType
): string =>
  `${hint}\n\n${chalk.bold('Matcher error')}: ${generic}${
    typeof specific === 'string' ? `\n\n${specific}` : ''
  }`;

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            "dispose",
            "interleave",
            "catch",
        ]);
    });

    it("deterministic collapse of Await", async () => {
        const { main, output } = evaluator.evaluateTypeScript(
            `
        export const output: any[] = [];

        let asyncId = 0;
        function increment() { asyncId++; }

export function forEachNoEmitOnErrorScenarioTscWatch(commandLineArgs: string[]): void {
    const errorTypes = getNoEmitOnErrorErrorsType();
    forEachNoEmitOnErrorScenario(
        "noEmitOnError",
        (subScenario, sys) =>
            verifyTscWatch({
                scenario: "noEmitOnError",
                subScenario,
                commandLineArgs: [...commandLineArgs, "--w"],
                sys: () => sys(errorTypes[0][1]),
                edits: getEdits(errorTypes),
            }),
    );

    function getEdits(errorTypes: ReturnType<typeof getNoEmitOnErrorErrorsType>): TscWatchCompileChange[] {
        const edits: TscWatchCompileChange[] = [];
        for (const [subScenario, mainErrorContent, fixedErrorContent] of errorTypes) {
            if (edits.length) {
                edits.push(
                    {
                        caption: subScenario,
                        edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, mainErrorContent),
                        timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                    },
                );
            }
            edits.push(
                {
                    caption: "No change",
                    edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, sys.readFile(`/user/username/projects/noEmitOnError/src/main.ts`)!),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: `Fix ${subScenario}`,
                    edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, fixedErrorContent),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
                {
                    caption: "No change",
                    edit: sys => sys.writeFile(`/user/username/projects/noEmitOnError/src/main.ts`, sys.readFile(`/user/username/projects/noEmitOnError/src/main.ts`)!),
                    timeouts: sys => sys.runQueuedTimeoutCallbacks(),
                },
            );
        }
        return edits;
    }
}

        `,
            { target: ts.ScriptTarget.ES2018 },
        );

        await main();

        assert.deepEqual(output, [
            0,
            1,
            2,

            // This really should be 2, but our transpile introduces an extra `await` by necessity to observe the
            // result of __disposeResources. The process of adopting the result ends up taking two turns of the
            // microtask queue.
            4,
        ]);
    });

    it("'await using' with downlevel generators", async () => {
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

        `,
            {
                target: ts.ScriptTarget.ES5,
            },
            {
                Iterator,
            },
        );

        const exited = await main();
        assert.isTrue(exited, "Expected 'await using' to dispose generator");
    });

    it("'await using' with downlevel async generators", async () => {
        abstract class AsyncIterator {
            return?(): PromiseLike<void>;
            [evaluator.FakeSymbol.asyncIterator]() {
                return this;
            }
            async [evaluator.FakeSymbol.asyncDispose]() {
                await this.return?.();
            }
        }

        const { main } = evaluator.evaluateTypeScript(
            `
            let exited = false;

            async function * f() {
                try {
                    yield;
                }
                finally {
                    exited = true;
                }
            }

// Narrowing by aliased discriminant property access

function g30(item: { type: 'alpha', alpha: number } | { type: 'beta', beta: string }) {
    const ty = item.type;
    if (ty === 'alpha') {
        item.alpha;
    }
    else {
        item.beta;
    }
}
        `,
            {
                target: ts.ScriptTarget.ES5,
            },
            {
                AsyncIterator,
            },
        );

        const exited = await main();
        assert.isTrue(exited, "Expected 'await using' to dispose async generator");
    });
});
