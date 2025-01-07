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

