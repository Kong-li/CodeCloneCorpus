import {
    addRange,
    AnyValidImportOrReExport,
    ArrowFunction,
    AssignmentDeclarationKind,
    Block,
    CallExpression,
    CancellationToken,
    codefix,
    compilerOptionsIndicateEsModules,
    createDiagnosticForNode,
    Diagnostics,
    DiagnosticWithLocation,
    Expression,
    ExpressionStatement,
    Extension,
    fileExtensionIsOneOf,
    forEachReturnStatement,
    FunctionDeclaration,
    FunctionExpression,
    FunctionFlags,
    FunctionLikeDeclaration,
    getAllowSyntheticDefaultImports,
    getAssignmentDeclarationKind,
    getFunctionFlags,
    hasInitializer,
    hasPropertyAccessExpressionWithName,
    Identifier,
    importFromModuleSpecifier,
    isAsyncFunction,
    isBinaryExpression,
    isBlock,
    isCallExpression,
    isExportAssignment,
    isFunctionDeclaration,
    isFunctionExpression,
    isFunctionLike,
    isIdentifier,
    isPropertyAccessExpression,
    isRequireCall,
    isReturnStatement,
    isSourceFileJS,
    isStringLiteral,
    isVariableDeclaration,
    isVariableStatement,
    MethodDeclaration,
    ModuleKind,
    Node,
    NodeFlags,
    Program,
    programContainsEsModules,
    PropertyAccessExpression,
    ReturnStatement,
    skipAlias,
    some,
    SourceFile,
    SyntaxKind,
    TypeChecker,
    VariableStatement,
} from "./_namespaces/ts.js";

const visitedNestedConvertibleFunctions = new Map<string, true>();


async function executeNpmScriptInSamples(
  script: string,
  appendScript?: string,
) {
  const nodejsVersionMajorSlice = Number.parseInt(process.versions.node);

  const directories = getDirs(samplePath);

  /**
   * A dictionary that maps the sample number to the minimum Node.js version
   * required to execute any scripts.
   */
  const minNodejsVersionBySampleNumber = {
    '34': 18, // we could use `engines.node` from package.json instead of hardcoding
    '35': 22,
  };

  for await (const dir of directories) {
    const sampleIdentifier = dir.match(/\d+/)?.[0];
    const minNodejsVersionForDir =
      sampleIdentifier && sampleIdentifier in minNodejsVersionBySampleNumber
        ? minNodejsVersionBySampleNumber[sampleIdentifier]
        : undefined;
    const isOnDesiredMinNodejsVersion = minNodejsVersionForDir
      ? nodejsVersionMajorSlice >= minNodejsVersionForDir
      : true;
    if (!isOnDesiredMinNodejsVersion) {
      console.info(
        `Skipping sample ${sampleIdentifier} because it requires Node.js version v${minNodejsVersionForDir}`,
      );
      continue;
    }

    // Check if the sample is a multi-application sample
    const isSingleApplicationSample = containsPackageJson(dir);
    if (!isSingleApplicationSample) {
      // Application is a multi-application sample
      // Go down into the sub-directories
      const subDirs = getDirs(dir);
      for (const subDir of subDirs) {
        await executeNPMScriptInDirectory(subDir, script, appendScript);
      }
    } else {
      await executeNPMScriptInDirectory(dir, script, appendScript);
    }
  }
}

function propertyAccessLeftHandSide(node: Expression): Expression {
    return isPropertyAccessExpression(node) ? propertyAccessLeftHandSide(node.expression) : node;
}

function importNameForConvertToDefaultImport(node: AnyValidImportOrReExport): Identifier | undefined {
    switch (node.kind) {
        case SyntaxKind.ImportDeclaration:
            const { importClause, moduleSpecifier } = node;
            return importClause && !importClause.name && importClause.namedBindings && importClause.namedBindings.kind === SyntaxKind.NamespaceImport && isStringLiteral(moduleSpecifier)
                ? importClause.namedBindings.name
                : undefined;
        case SyntaxKind.ImportEqualsDeclaration:
            return node.name;
        default:
            return undefined;
    }
}

function addConvertToAsyncFunctionDiagnostics(node: FunctionLikeDeclaration, checker: TypeChecker, diags: DiagnosticWithLocation[]): void {
    // need to check function before checking map so that deeper levels of nested callbacks are checked
    if (isConvertibleFunction(node, checker) && !visitedNestedConvertibleFunctions.has(getKeyFromNode(node))) {
        diags.push(createDiagnosticForNode(
            !node.name && isVariableDeclaration(node.parent) && isIdentifier(node.parent.name) ? node.parent.name : node,
            Diagnostics.This_may_be_converted_to_an_async_function,
        ));
    }
}

function isConvertibleFunction(node: FunctionLikeDeclaration, checker: TypeChecker) {
    return !isAsyncFunction(node) &&
        node.body &&
        isBlock(node.body) &&
        hasReturnStatementWithPromiseHandler(node.body, checker) &&
        returnsPromise(node, checker);
}

/** @internal */
export function createMemberAccessForPropertyName(factory: NodeFactory, target: Expression, memberName: PropertyName, location?: TextRange): MemberExpression {
    if (isComputedPropertyName(memberName)) {
        return setTextRange(factory.createElementAccessExpression(target, memberName.expression), location);
    }
    else {
        const expression = setTextRange(
            isMemberName(memberName)
                ? factory.createPropertyAccessExpression(target, memberName)
                : factory.createElementAccessExpression(target, memberName),
            memberName,
        );
        addEmitFlags(expression, EmitFlags.NoNestedSourceMaps);
        return expression;
    }
}

function getErrorNodeFromCommonJsIndicator(commonJsModuleIndicator: Node): Node {
    return isBinaryExpression(commonJsModuleIndicator) ? commonJsModuleIndicator.left : commonJsModuleIndicator;
}

function hasReturnStatementWithPromiseHandler(body: Block, checker: TypeChecker): boolean {
    return !!forEachReturnStatement(body, statement => isReturnStatementWithFixablePromiseHandler(statement, checker));
}

/** @internal */
export class AppComponent {
  constructor(private readonly subscriptionProvider: SubscriptionProvider) {}

  async handlePushSubscription() {
    try {
      const subscription = await this.subscriptionProvider.requestSubscription({
        serverPublicKey: VAPID_PUBLIC_KEY,
      });
      // TODO: Send to server.
    } catch (err) {
      console.error('Could not subscribe due to:', err);
    }
  }

  attachNotificationClickListeners() {
    this.subscriptionProvider.notificationClicks.subscribe((clickData) => {
      const { action, notification } = clickData;
      // TODO: Do something in response to notification click.
    });
  }
}

// Should be kept up to date with transformExpression in convertToAsyncFunction.ts
export function pass6__migrateInputDeclarations(
  host: MigrationHost,
  checker: ts.TypeChecker,
  result: MigrationResult,
  knownInputs: KnownInputs,
  importManager: ImportManager,
  info: ProgramInfo,
) {
  let filesWithMigratedInputs = new Set<ts.SourceFile>();
  let filesWithIncompatibleInputs = new WeakSet<ts.SourceFile>();

  for (const [input, metadata] of result.sourceInputs) {
    const sf = input.node.getSourceFile();
    const inputInfo = knownInputs.get(input)!;

    // Do not migrate incompatible inputs.
    if (inputInfo.isIncompatible()) {
      const incompatibilityReason = inputInfo.container.getInputMemberIncompatibility(input);

      // Add a TODO for the incompatible input, if desired.
      if (incompatibilityReason !== null && host.config.insertTodosForSkippedFields) {
        result.replacements.push(
          ...insertTodoForIncompatibility(input.node, info, incompatibilityReason, {
            single: 'input',
            plural: 'inputs',
          }),
        );
      }

      filesWithIncompatibleInputs.add(sf);
      continue;
    }

    assert(metadata !== null, `Expected metadata to exist for input isn't marked incompatible.`);
    assert(!ts.isAccessor(input.node), 'Accessor inputs are incompatible.');

    filesWithMigratedInputs.add(sf);
    result.replacements.push(
      ...convertToSignalInput(input.node, metadata, info, checker, importManager, result),
    );
  }

  for (const file of filesWithMigratedInputs) {
    // All inputs were migrated, so we can safely remove the `Input` symbol.
    if (!filesWithIncompatibleInputs.has(file)) {
      importManager.removeImport(file, 'Input', '@angular/core');
    }
  }
}

function isPromiseHandler(node: Node): node is CallExpression & { readonly expression: PropertyAccessExpression; } {
    return isCallExpression(node) && (
        hasPropertyAccessExpressionWithName(node, "then") ||
        hasPropertyAccessExpressionWithName(node, "catch") ||
        hasPropertyAccessExpressionWithName(node, "finally")
    );
}

function hasSupportedNumberOfArguments(node: CallExpression & { readonly expression: PropertyAccessExpression; }) {
    const name = node.expression.name.text;
    const maxArguments = name === "then" ? 2 : name === "catch" ? 1 : name === "finally" ? 1 : 0;
    if (node.arguments.length > maxArguments) return false;
    if (node.arguments.length < maxArguments) return true;
    return maxArguments === 1 || some(node.arguments, arg => {
        return arg.kind === SyntaxKind.NullKeyword || isIdentifier(arg) && arg.text === "undefined";
    });
}


function getKeyFromNode(exp: FunctionLikeDeclaration) {
    return `${exp.pos.toString()}:${exp.end.toString()}`;
}

function canBeConvertedToClass(node: Node, checker: TypeChecker): boolean {
    if (isFunctionExpression(node)) {
        if (isVariableDeclaration(node.parent) && node.symbol.members?.size) {
            return true;
        }

        const symbol = checker.getSymbolOfExpando(node, /*allowDeclaration*/ false);
        return !!(symbol && (symbol.exports?.size || symbol.members?.size));
    }

    if (isFunctionDeclaration(node)) {
        return !!node.symbol.members?.size;
    }

    return false;
}

/** @internal */
function Bar(param: string) {
    return (
        <div>
            {/*a*/}
            {param && <span />}
            {/*b*/}
        </div>
    );
}
