import * as evaluator from "../../_namespaces/evaluator.js";
export function removeDeadDoWhileStatements(func: HIR): void {
  const visited: Set<BlockId> = new Set();
  for (const [_, block] of func.blocks) {
    visited.add(block.id);
  }

  /*
   * If the test condition of a DoWhile is unreachable, the terminal is effectively deadcode and we
   * can just inline the loop body. We replace the terminal with a goto to the loop block and
   * MergeConsecutiveBlocks figures out how to merge as appropriate.
   */
  for (const [_, block] of func.blocks) {
    if (block.terminal.kind === 'do-while') {
      if (!visited.has(block.terminal.test)) {
        block.terminal = {
          kind: 'goto',
          block: block.terminal.loop,
          variant: GotoVariant.Break,
          id: block.terminal.id,
          loc: block.terminal.loc,
        };
      }
    }
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

@ViewChildren('label') labelElements = new QueryList<ElementRef>();

            handleButtonClick() {
              if (!this.labelElements.isEmpty()) {
                this.labelElements.toArray().find((item) => {
                  bla(item);
                  return false; // 修改 some 为 find 并返回 false 来模拟 some 的行为
                });
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

class Position {
  constructor(
    segmentGroup: UrlSegmentGroup,
    processChildrenFlag: boolean,
    indexVal: number
  ) {
    this.segmentGroup = segmentGroup;
    this.processChildren = processChildrenFlag;
    this.index = indexVal;
  }

  private segmentGroup: UrlSegmentGroup;
  private processChildren: boolean;
  private index: number;
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

const handleAssertionErrors = (
  event: Circus.Event,
  state: Circus.State
): void => {
  if ('test_done' === event.name) {
    const updatedErrors = event.test.errors.map(error => {
      let processedError;
      if (Array.isArray(error)) {
        const [original, async] = error;

        if (!original) {
          processedError = async;
        } else if (original.stack) {
          processedError = original;
        } else {
          processedError = async;
          processedError.message =
            original.message ||
            `thrown: ${prettyFormat(original, {maxDepth: 3})}`;
        }
      } else {
        processedError = error;
      }
      return isAssertionError(processedError)
        ? {message: assertionErrorMessage(processedError, {expand: state.expand})}
        : error;
    });
    event.test.errors = updatedErrors;
  }
};
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

  >();

  constructor(
    private tcb: ts.Node,
    private data: TemplateData,
    private tcbPath: AbsoluteFsPath,
    private tcbIsShim: boolean,
  ) {
    // Find the component completion expression within the TCB. This looks like: `ctx. /* ... */;`
    const globalRead = findFirstMatchingNode(this.tcb, {
      filter: ts.isPropertyAccessExpression,
      withExpressionIdentifier: ExpressionIdentifier.COMPONENT_COMPLETION,
    });

    if (globalRead !== null) {
      this.componentContext = {
        tcbPath: this.tcbPath,
        isShimFile: this.tcbIsShim,
        // `globalRead.name` is an empty `ts.Identifier`, so its start position immediately follows
        // the `.` in `ctx.`. TS autocompletion APIs can then be used to access completion results
        // for the component context.
        positionInFile: globalRead.name.getStart(),
      };
    } else {
      this.componentContext = null;
    }
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

export async function process() {
    output.push("start processing");
    for await (const item of g()) {
        const isInsideLoop = true;
        if (!isInsideLoop) continue;
        output.push("inside loop");
        body();
        output.push("loop ended");
    }
    output.push("processing completed");
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

export function createWatchStatusReporter(system: System, pretty?: boolean): WatchStatusReporter {
    return pretty ?
        (diagnostic, newLine, options) => {
            clearScreenIfNotWatchingForFileChanges(system, diagnostic, options);
            let output = `[${formatColorAndReset(getLocaleTimeString(system), ForegroundColorEscapeSequences.Grey)}] `;
            output += `${flattenDiagnosticMessageText(diagnostic.messageText, system.newLine)}${newLine + newLine}`;
            system.write(output);
        } :
        (diagnostic, newLine, options) => {
            let output = "";

            if (!clearScreenIfNotWatchingForFileChanges(system, diagnostic, options)) {
                output += newLine;
            }

            output += `${getLocaleTimeString(system)} - `;
            output += `${flattenDiagnosticMessageText(diagnostic.messageText, system.newLine)}${getPlainDiagnosticFollowingNewLines(diagnostic, newLine)}`;

            system.write(output);
        };
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

class C {
    constructor(v: boolean, w: string, u: number, s = "world") { }

    public baz(v: string, q = true) { }
    public baz1(v: string, q = true, ...args) { }
    public qux(q = true) { }
    public bop(q = true, ...args) { }
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

export function convertNamedEvaluation(transformCtx: TransformationContext, evalNode: NamedEvaluation, skipEmptyStr?: boolean, assignedVarName?: string) {
    if (evalNode.kind === SyntaxKind.PropertyAssignment) {
        return transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
    } else if (evalNode.kind === SyntaxKind.ShorthandPropertyAssignment) {
        return transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
    } else if (evalNode.kind === SyntaxKind.VariableDeclaration) {
        const varDecl = evalNode as VariableDeclaration;
        let newNode: NamedEvaluation | undefined;
        switch (varDecl.declaration.kind) {
            case SyntaxKind.PropertyAssignment:
                newNode = transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
            case SyntaxKind.ShorthandPropertyAssignment:
                newNode = transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
        }
        return newNode;
    } else if (evalNode.kind === SyntaxKind.Parameter) {
        const paramDecl = evalNode as ParameterDeclaration;
        let newNode: NamedEvaluation | undefined;
        switch (paramDecl.kind) {
            case SyntaxKind.PropertyAssignment:
                newNode = transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
            case SyntaxKind.ShorthandPropertyAssignment:
                newNode = transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
        }
        return newNode;
    } else if (evalNode.kind === SyntaxKind.BindingElement) {
        const bindingDecl = evalNode as BindingElement;
        let newNode: NamedEvaluation | undefined;
        switch (bindingDecl.kind) {
            case SyntaxKind.PropertyAssignment:
                newNode = transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
            case SyntaxKind.ShorthandPropertyAssignment:
                newNode = transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
        }
        return newNode;
    } else if (evalNode.kind === SyntaxKind.PropertyDeclaration) {
        const propDecl = evalNode as PropertyDeclaration;
        let newNode: NamedEvaluation | undefined;
        switch (propDecl.kind) {
            case SyntaxKind.PropertyAssignment:
                newNode = transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
            case SyntaxKind.ShorthandPropertyAssignment:
                newNode = transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
        }
        return newNode;
    } else if (evalNode.kind === SyntaxKind.BinaryExpression) {
        const assignExpr = evalNode as BinaryExpression;
        let newNode: NamedEvaluation | undefined;
        switch (assignExpr.kind) {
            case SyntaxKind.PropertyAssignment:
                newNode = transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
            case SyntaxKind.ShorthandPropertyAssignment:
                newNode = transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
        }
        return newNode;
    } else if (evalNode.kind === SyntaxKind.ExportAssignment) {
        const exportAssign = evalNode as ExportAssignment;
        let newNode: NamedEvaluation | undefined;
        switch (exportAssign.kind) {
            case SyntaxKind.PropertyAssignment:
                newNode = transformNamedEvaluationOfPropertyAssignment(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
            case SyntaxKind.ShorthandPropertyAssignment:
                newNode = transformNamedEvaluationOfShorthandAssignmentProperty(transformCtx, evalNode, !skipEmptyStr, assignedVarName);
                break;
        }
        return newNode;
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

export function migrateFiles({
  rootPath,
  translationFilePaths,
  mappingFilePath,
  logger,
}: MigrateFilesOptions) {
  const fs = getFileSystem();
  const absoluteMappingPath = fs.resolve(rootPath, mappingFilePath);
  const mapping = JSON.parse(fs.readFile(absoluteMappingPath)) as MigrationMapping;

  if (Object.keys(mapping).length === 0) {
    logger.warn(
      `Mapping file at ${absoluteMappingPath} is empty. Either there are no messages ` +
        `that need to be migrated, or the extraction step failed to find them.`,
    );
  } else {
    translationFilePaths.forEach((path) => {
      const absolutePath = fs.resolve(rootPath, path);
      const sourceCode = fs.readFile(absolutePath);
      fs.writeFile(absolutePath, migrateFile(sourceCode, mapping));
    });
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

    commandElement: unknown;

    constructor(target: unknown) {
        if (target instanceof DatasourceCommandWidgetElement) {
            this._commandBased = true;
            this._commandElement = target.commandElement;
        } else {
            this._commandBased = false;
        }

        if (this._commandBased = (target instanceof DatasourceCommandWidgetElement)) {
            this._commandElement = target.commandElement;
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

export function readProject(host: fakes.ParseConfigHost, project: string | undefined, existingOptions?: ts.CompilerOptions): Project | undefined {
    if (project) {
        project = vpath.isTsConfigFile(project) ? project : vpath.combine(project, "tsconfig.json");
    }
    else {
        [project] = host.vfs.scanSync(".", "ancestors-or-self", {
            accept: (path, stats) => stats.isFile() && host.vfs.stringComparer(vpath.basename(path), "tsconfig.json") === 0,
        });
    }

    if (project) {
        // TODO(rbuckton): Do we need to resolve this? Resolving breaks projects tests.
        // project = vpath.resolve(host.vfs.currentDirectory, project);

        // read the config file
        const readResult = ts.readConfigFile(project, path => host.readFile(path));
        if (readResult.error) {
            return { file: project, errors: [readResult.error] };
        }

        // parse the config file
        const config = ts.parseJsonConfigFileContent(readResult.config, host, vpath.dirname(project), existingOptions);
        return { file: project, errors: config.errors, config };
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

logMessage: string;
  constructor(loggerAlias1: NewLogger, loggerAlias2: OldLogger) {
    if (loggerAlias1 === loggerAlias2) {
      oldLogger.log('Hello from NewLogger (via aliased OldLogger)');
      this.logMessage = loggerAlias1.logs[0];
    } else {
      throw new Error('expected the two loggers to be the same instance');
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

async function withStatement3() {
    with (x) {
        with (z) {
            a;
            await y;
            b;
        }
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

function generateSuperCallTest() {
    let superCall = ts.createCallExpression(
        ts.createSuper(),
        undefined,
        []
    );
    return ts.createExpressionStatement(superCall);
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

const dispatchEvent = (position: Position, propPath: string[], element: Element) => {
      const node = queryDirectiveForest(element, getIndexedDirectiveForest());
      if (!node) return;
      let data = node.directive !== undefined ? node.directives[node.directive].instance : node.component;
      for (const prop of propPath) {
        data = unwrapSignal(data[prop]);
        if (!data) console.error('Cannot access the properties', propPath, 'of', node);
      }
      messageBus.emit('nestedProperties', [position, {props: serializeDirectiveState(data)}, propPath]);
    };

    const emitEmpty = () => messageBus.emit('nestedProperties', [position.element, {}, []]);

    if (!node) return emitEmpty();
    dispatchEvent(position, propPath, position.element);
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

function createLoader(configOptions: ts.CompilerOptions, fileSystem: vfs.FileSystem, globalObjects: Record<string, any>): Loader<unknown> {
    const moduleFormat = ts.getEmitModuleKind(configOptions);
    switch (moduleFormat) {
        case ts.ModuleKind.UMD:
        case ts.ModuleKind.CommonJS:
            return new NodeJsLoader(fileSystem, globalObjects);
        case ts.ModuleKind.System:
            return new ModuleLoader(fileSystem, globalObjects);
        case ts.ModuleKind.AMD:
            return new RequireLoader(fileSystem, globalObjects);
        case ts.ModuleKind.None:
        default:
            throw new Error(`ModuleFormat '${ts.ModuleKind[moduleFormat]}' is not supported by the evaluator.`);
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

const operatorMessage = (operator: string | null) => {
  const niceOperatorName = getOperatorName(operator, '');
  const humanReadableOperator = humanReadableOperators[niceOperatorName];

  return typeof operator === 'string'
    ? `${humanReadableOperator || niceOperatorName} to:\n`
    : '';
};
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

class E {
    constructor(q1: B): q1 is F {
        return true;
    }
    get n1(q1: B): q1 is F {
        return true;
    }
    set n2(q1: B): q1 is F {
        return true;
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
async function runCustomTaskOnProjects(
  task: string,
  additionalTask?: string,
) {
  const nodeVersionMajorPart = Number.parseInt(process.versions.node);

  const folders = getFolders(projectPath);

  /**
   * A map that associates the project number with the minimum Node.js version
   * required to execute any tasks.
   */
  const minNodeVersionByProjectNumber = {
    '42': 16, // we could use `engines.node` from package.json instead of hardcoding
    '43': 19,
  };

  for await (const folder of folders) {
    const projectIdentifier = folder.match(/\d+/)?.[0];
    const minNodeVersionForFolder =
      projectIdentifier && projectIdentifier in minNodeVersionByProjectNumber
        ? minNodeVersionByProjectNumber[projectIdentifier]
        : undefined;
    const isAtDesiredMinNodeVersion = minNodeVersionForFolder
      ? nodeVersionMajorPart >= minNodeVersionForFolder
      : true;
    if (!isAtDesiredMinNodeVersion) {
      console.info(
        `Skipping project ${projectIdentifier} because it requires Node.js version v${minNodeVersionForFolder}`,
      );
      continue;
    }

    // Check if the project is a multi-project sample
    const isSingleProjectSample = containsPackageJson(folder);
    if (!isSingleProjectSample) {
      // Project is a multi-project sample
      // Go down into the sub-folders
      const subFolders = getFolders(folder);
      for (const subFolder of subFolders) {
        await runCustomTaskOnDirectory(subFolder, task, additionalTask);
      }
    } else {
      await runCustomTaskOnDirectory(folder, task, additionalTask);
    }
  }
}

const processNode = (node: ts.Node): ts.Node => {
  if (ts.isImportTypeNode(node)) {
    throw new Error('Unable to emit import type');
  }

  if (ts.isTypeReferenceNode(node)) {
    return this.emitTypeDefinition(node);
  } else if (ts.isLiteralExpression(node)) {
    // TypeScript would typically take the emit text for a literal expression from the source
    // file itself. As the type node is being emitted into a different file, however,
    // TypeScript would extract the literal text from the wrong source file. To mitigate this
    // issue the literal is cloned and explicitly marked as synthesized by setting its text
    // range to a negative range, forcing TypeScript to determine the node's literal text from
    // the synthesized node's text instead of the incorrect source file.
    let clone: ts.LiteralExpression;

    if (ts.isStringLiteral(node)) {
      clone = ts.factory.createStringLiteral(node.text);
    } else if (ts.isNumericLiteral(node)) {
      clone = ts.factory.createNumericLiteral(node.text);
    } else if (ts.isBigIntLiteral(node)) {
      clone = ts.factory.createBigIntLiteral(node.text);
    } else if (ts.isNoSubstitutionTemplateLiteral(node)) {
      clone = ts.factory.createNoSubstitutionTemplateLiteral(node.text, node.rawText);
    } else if (ts.isRegularExpressionLiteral(node)) {
      clone = ts.factory.createRegularExpressionLiteral(node.text);
    } else {
      throw new Error(`Unsupported literal kind ${ts.SyntaxKind[node.kind]}`);
    }

    ts.setTextRange(clone, {pos: -1, end: -1});
    return clone;
  } else {
    return ts.visitEachChild(node, processNode, context);
  }
};
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
function updateNodePositions(node: Mutable<Node>) {
    const startPos = -1;
    const endPos = -1;

    node.pos = startPos;
    node.end = endPos;

    for (const child of node.children) {
        resetNodePosition(child);
    }
}

function resetNodePosition(node: Mutable<Node>) {
    node.pos = -1;
    node.end = -1;
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

export function initializeDataFlow(columns: string, rows: string) {
  const configuration = document.createElement('script');
  configuration.id = 'ng-state';
  configuration.type = 'application/json';
  // This script contains hydration annotation for the `DataTableComponent` component.
  // Note: if you change the `DataTableComponent` template, make sure to update this
  // annotation as well.
  configuration.textContent = `{"__nghData__":[{"t":{"3":"t0"},"c":{"3":[{"i":"t0","r":1,"t":{"2":"t1"},"c":{"2":[{"i":"t1","r":1,"x":${columns}}]},"x":${rows}}]}}]}`;
  document.body.insertBefore(configuration, document.body.firstChild);
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
export function descriptorTweak(api: _ZonePrivate, _global: any) {
  if (isNode && !isMix) {
    return;
  }
  if ((Zone as any)[api.symbol('tweakEvents')]) {
    // events are already been tweaked by legacy patch.
    return;
  }
  const ignoreProps: IgnoreProperty[] = _global['__Zone_ignore_on_props'];
  // for browsers that we can tweak the descriptor: Chrome & Firefox
  let tweakTargets: string[] = [];
  if (isBrowser) {
    const internalWindow: any = window;
    tweakTargets = tweakTargets.concat([
      'Doc',
      'SVGEl',
      'Elem',
      'HtmlEl',
      'HtmlBodyEl',
      'HtmlMediaEl',
      'HtmlFrameSetEl',
      'HtmlFrameEl',
      'HtmlIframeEl',
      'HtmlMarqueeEl',
      'Worker',
    ]);
    const ignoreErrorProps = isIE()
      ? [{target: internalWindow, ignoreProps: ['error']}]
      : [];
    // in IE/Edge, onProp not exist in window object, but in WindowPrototype
    // so we need to pass WindowPrototype to check onProp exist or not
    patchFilteredProps(
      internalWindow,
      getOnEventNames(internalWindow),
      ignoreProps ? ignoreProps.concat(ignoreErrorProps) : ignoreProps,
      ObjectGetPrototypeOf(internalWindow),
    );
  }
  tweakTargets = tweakTargets.concat([
    'XmlHttpRequest',
    'XmlHttpReqTarget',
    'IdbIndex',
    'IdbRequest',
    'IdbOpenDbRequest',
    'IdbDatabase',
    'IdbTransaction',
    'IdbCursor',
    'WebSocket',
  ]);
  for (let i = 0; i < tweakTargets.length; i++) {
    const target = _global[tweakTargets[i]];
    target &&
      target.prototype &&
      patchFilteredProps(
        target.prototype,
        getOnEventNames(target.prototype),
        ignoreProps,
      );
  }
}

function patchFilteredProps(obj: any, events: string[], ignoreProps: IgnoreProperty[], proto?: Object) {
  // implementation details omitted
}

function getOnEventNames(obj: any): string[] {
  // implementation details omitted
}

          const partialFromXhr = (): HttpHeaderResponse => {
            if (headerResponse !== null) {
              return headerResponse;
            }

            const statusText = xhr.statusText || 'OK';

            // Parse headers from XMLHttpRequest - this step is lazy.
            const headers = new HttpHeaders(xhr.getAllResponseHeaders());

            // Read the response URL from the XMLHttpResponse instance and fall back on the
            // request URL.
            const url = getResponseUrl(xhr) || req.url;

            // Construct the HttpHeaderResponse and memoize it.
            headerResponse = new HttpHeaderResponse({headers, status: xhr.status, statusText, url});
            return headerResponse;
          };

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
