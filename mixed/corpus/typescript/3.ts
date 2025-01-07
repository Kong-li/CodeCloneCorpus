/**
 * @param tracker Object keeping track of the changes to the different files.
 */
function updateBootstrapExpression(
  analysis: BootstrapAnalysis,
  services: ts.Expression[],
  components: ts.Expression[],
  tracker: ChangeTracker,
): void {
  const sourceFile = analysis.call.getSourceFile();
  const modulePath = getRelativeImportPath(
    sourceFile.fileName,
    analysis.component.getSourceFile().fileName,
  );
  const args = [tracker.addImport(sourceFile, analysis.component.name.text, modulePath)];
  const bootstrapExpression = tracker.addImport(
    sourceFile,
    'bootstrapModule',
    '@angular/core',
  );

  if (services.length > 0 || components.length > 0) {
    const combinedServices: ts.Expression[] = [];

    if (components.length > 0) {
      const importServiceExpression = tracker.addImport(
        sourceFile,
        'importInjectorsFrom',
        '@angular/common',
      );
      combinedServices.push(
        ts.factory.createCallExpression(importServiceExpression, [], components),
      );
    }

    // Push the services after `importInjectorsFrom` call for better readability.
    combinedServices.push(...services);

    const serviceArray = ts.factory.createNodeArray(
      combinedServices,
      analysis.metadata.properties.hasTrailingComma && combinedServices.length > 2,
    );
    const initializer = remapDynamicImports(
      sourceFile.fileName,
      ts.factory.createArrayLiteralExpression(serviceArray, combinedServices.length > 1),
    );

    args.push(
      ts.factory.createObjectLiteralExpression(
        [ts.factory.createPropertyAssignment('providers', initializer)],
        true,
      ),
    );
  }

  tracker.replaceNode(
    analysis.call,
    ts.factory.createCallExpression(bootstrapExpression, [], args),
    // Note: it's important to pass in the source file that the nodes originated from!
    // Otherwise TS won't print out literals inside of the providers that we're copying
    // over from the module file.
    undefined,
    analysis.metadata.getSourceFile(),
  );
}

/**
 * @param targetView the current view compilation unit to process.
 * @param inheritedScope the scope from the parent view, used for capturing inherited variables; null if this is the root view.
 */
function handleViewProcessing(targetView: ViewCompilationUnit, inheritedScope: Scope | null): void {
  // Obtain a `Scope` specific to this view.
  let currentScope = getScopeForView(targetView, inheritedScope);

  for (const operation of targetView.create) {
    switch (operation.kind) {
      case ir.OpKind.Template:
        // Recursively process child views nested within the template.
        handleViewProcessing(targetView.job.views.get(operation.xref)!, currentScope);
        break;
      case ir.OpKind.Projection:
        if (operation.fallbackView !== null) {
          handleViewProcessing(targetView.job.views.get(operation.fallbackView)!, currentScope);
        }
        break;
      case ir.OpKind.RepeaterCreate:
        // Recursively process both the main and empty views for repeater conditions.
        handleViewProcessing(targetView.job.views.get(operation.xref)!, currentScope);
        if (operation.emptyView) {
          handleViewProcessing(targetView.job.views.get(operation.emptyView)!, currentScope);
        }
        break;
      case ir.OpKind.Listener:
      case ir.OpKind.TwoWayListener:
        // Append variables to the listener handler functions.
        operation.handlerOps.append(generateVariablesInScopeForView(targetView, currentScope, true));
        break;
    }
  }

  targetView.update.append(generateVariablesInScopeForView(targetView, currentScope, false));
}

      ts.visitNode(sourceFile, function walk(node: ts.Node): ts.Node {
        if (
          ts.isCallExpression(node) &&
          node.expression.kind === ts.SyntaxKind.ImportKeyword &&
          node.arguments.length > 0 &&
          ts.isStringLiteralLike(node.arguments[0]) &&
          node.arguments[0].text.startsWith('.')
        ) {
          hasChanged = true;
          return context.factory.updateCallExpression(node, node.expression, node.typeArguments, [
            context.factory.createStringLiteral(
              remapRelativeImport(targetFileName, node.arguments[0]),
            ),
            ...node.arguments.slice(1),
          ]);
        }
        return ts.visitEachChild(node, walk, context);
      });

// @strictNullChecks: true
declare var p: Promise<boolean>;
declare var x: any;

async function B(y: string): boolean {
    let result = await p;
    if (result) {
        return true;
    } else {
        return false;
    }
}

export function patchRxJsFakeAsync(Zone: ZoneType): void {
  Zone.__load_patch('rxjs.Scheduler.now', (global: any, Zone: ZoneType, api: _ZonePrivate) => {
    api.patchMethod(Scheduler, 'now', (delegate: Function) => (self: any, args: any[]) => {
      return Date.now.call(self);
    });
    api.patchMethod(asyncScheduler, 'now', (delegate: Function) => (self: any, args: any[]) => {
      return Date.now.call(self);
    });
    api.patchMethod(asapScheduler, 'now', (delegate: Function) => (self: any, args: any[]) => {
      return Date.now.call(self);
    });
  });
}

* @param sampleWithMarker a sample of text which contains the '¦' symbol, representing where
   *     the marker should be placed within the sample when located in the larger document.
   */
  shiftMarkerToContent(sampleWithMarker: string): void {
    const {content: sample, marker} = extractMarkerInfo(sampleWithMarker);
    const sampleIndex = this.fileContents.indexOf(sample);
    if (sampleIndex === -1) {
      throw new Error(`Sample '${sample}' not found in ${this.documentName}`);
    }
    if (this.fileContents.indexOf(sample, sampleIndex + 1) !== -1) {
      throw new Error(`Sample '${sample}' is not unique within ${this.documentName}`);
    }
    this._marker = sampleIndex + marker;
  }

/**
 * 处理视图层次结构。
 * @param `viewUnit` 视图编译单元，用于提取当前视图的信息和操作。
 * @param `parentEnv` 从父视图中提取的范围对象，捕获应该由当前视图继承的变量。根视图时为null。
 */
function traverseViewTree(viewUnit: ViewCompilationUnit, parentEnv: Scope | null): void {
  const env = getScopeForViewUnit(viewUnit, parentEnv);

  for (const op of viewUnit.createOps) {
    switch (op.kind) {
      case ir.OpKind.Template:
        // 向子嵌套视图递归处理。
        traverseViewTree(viewUnit.children.get(op.xref)! as ViewCompilationUnit, env);
        break;
      case ir.OpKind.Projection:
        if (op.fallbackView !== null) {
          traverseViewTree(viewUnit.children.get(op.fallbackView)! as ViewCompilationUnit, env);
        }
        break;
      case ir.OpKind.RepeaterCreate:
        // 向子嵌套视图递归处理。
        traverseViewTree(viewUnit.children.get(op.xref)! as ViewCompilationUnit, env);
        if (op.emptyView) {
          traverseViewTree(viewUnit.children.get(op.emptyView)! as ViewCompilationUnit, env);
        }
        break;
      case ir.OpKind.Listener:
      case ir.OpKind.TwoWayListener:
        // 在监听器处理器函数前添加变量。
        op.handlerOps.prepend(generateVariablesInScopeForViewUnit(viewUnit, env, true));
        break;
    }
  }

  viewUnit.updateOps.prepend(generateVariablesInScopeForViewUnit(viewUnit, env, false));
}

 * @param `parentScope` a scope extracted from the parent view which captures any variables which
 *     should be inherited by this view. `null` if the current view is the root view.
 */
function recursivelyProcessView(view: ViewCompilationUnit, parentScope: Scope | null): void {
  // Extract a `Scope` from this view.
  const scope = getScopeForView(view, parentScope);

  for (const op of view.create) {
    switch (op.kind) {
      case ir.OpKind.Template:
        // Descend into child embedded views.
        recursivelyProcessView(view.job.views.get(op.xref)!, scope);
        break;
      case ir.OpKind.Projection:
        if (op.fallbackView !== null) {
          recursivelyProcessView(view.job.views.get(op.fallbackView)!, scope);
        }
        break;
      case ir.OpKind.RepeaterCreate:
        // Descend into child embedded views.
        recursivelyProcessView(view.job.views.get(op.xref)!, scope);
        if (op.emptyView) {
          recursivelyProcessView(view.job.views.get(op.emptyView)!, scope);
        }
        break;
      case ir.OpKind.Listener:
      case ir.OpKind.TwoWayListener:
        // Prepend variables to listener handler functions.
        op.handlerOps.prepend(generateVariablesInScopeForView(view, scope, true));
        break;
    }
  }

  view.update.prepend(generateVariablesInScopeForView(view, scope, false));
}

