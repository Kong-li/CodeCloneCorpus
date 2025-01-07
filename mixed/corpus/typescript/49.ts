function combineData(
  target: Record<string | symbol, any>,
  source: Record<string | symbol, any> | null,
  isRecursive = true
): Record<PropertyKey, any> {
  if (!source) return target
  let key, targetValue, sourceValue

  const keys = hasSymbol
    ? (Reflect.ownKeys(source) as string[])
    : Object.keys(source)

  for (let index = 0; index < keys.length; index++) {
    key = keys[index]
    // in case the object is already observed...
    if (key === '__ob__') continue
    targetValue = target[key]
    sourceValue = source[key]
    if (!isRecursive || !hasOwn(target, key)) {
      set(target, key, sourceValue)
    } else if (
      targetValue !== sourceValue &&
      isPlainObject(targetValue) &&
      isPlainObject(sourceValue)
    ) {
      combineData(targetValue, sourceValue)
    }
  }
  return target
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

* @param node The node to visit.
    */
    function traverseMethodDeclaration(node: MethodDeclaration): VisitResult<Statement> {
        let argumentsList: NodeArray<ParameterDeclaration>;
        const preservedLocalVariables = localVariablesContext;
        localVariablesContext = undefined;
        const methodFlags = getFunctionFlags(node);
        const updated = factory.updateMethodDeclaration(
            node,
            visitNodes(node.modifiers, visitor, isModifierLike),
            node.asteriskToken,
            node.name,
            /*typeParameters*/ undefined,
            argumentsList = methodFlags & MethodFlags.Async ?
                transformAsyncFunctionParameterList(node) :
                visitParameterList(node.parameters, visitor, context),
            /*returnType*/ undefined,
            methodFlags & MethodFlags.Async ?
                transformAsyncFunctionBody(node, argumentsList) :
                visitFunctionBody(node.body, visitor, context),
        );
        localVariablesContext = preservedLocalVariables;
        return updated;
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

     * @param node The node to visit.
     */
    function visitFunctionExpression(node: FunctionExpression): Expression {
        let parameters: NodeArray<ParameterDeclaration>;
        const savedLexicalArgumentsBinding = lexicalArgumentsBinding;
        lexicalArgumentsBinding = undefined;
        const functionFlags = getFunctionFlags(node);
        const updated = factory.updateFunctionExpression(
            node,
            visitNodes(node.modifiers, visitor, isModifier),
            node.asteriskToken,
            node.name,
            /*typeParameters*/ undefined,
            parameters = functionFlags & FunctionFlags.Async ?
                transformAsyncFunctionParameterList(node) :
                visitParameterList(node.parameters, visitor, context),
            /*type*/ undefined,
            functionFlags & FunctionFlags.Async ?
                transformAsyncFunctionBody(node, parameters) :
                visitFunctionBody(node.body, visitor, context),
        );
        lexicalArgumentsBinding = savedLexicalArgumentsBinding;
        return updated;
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

