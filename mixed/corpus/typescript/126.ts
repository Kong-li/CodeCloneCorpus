function hasIC(values: any[], item: string) {
    let lowerItem = item.toLowerCase();
    if (!values.length || !lowerItem) {
        return false;
    }
    for (let i = 0, len = values.length; i < len; ++i) {
        if (lowerItem === values[i].toLowerCase()) {
            return true;
        }
    }
    return false;
}

export function generateWsDecoratorFactory(
  decoratorType: WsParamtype,
): (...transformPipes: (Type<PipeTransform> | PipeTransform)[]) => MethodDecorator {
  return (...transformPipes: (Type<PipeTransform> | PipeTransform)[]) =>
    (target, methodName, descriptor) => {
      const existingParams =
        Reflect.getMetadata(PARAMETERS_METADATA, target.constructor, methodName) || {};
      Reflect.defineMetadata(
        PARAMETERS_METADATA,
        assignMetadata(existingParams, decoratorType, descriptor.value ? 0 : -1, undefined, ...transformPipes),
        target.constructor,
        methodName,
      );
    };
}

/**
 * @returns an array of `ts.Diagnostic`s representing errors when visible classes are not exported properly.
 */
export function validateExportedClasses(
  entryPoint: ts.SourceFile,
  checker: ts.TypeChecker,
  refGraph: ReferenceGraph,
): ts.Diagnostic[] {
  const diagnostics: ts.Diagnostic[] = [];

  // Firstly, compute the exports of the entry point. These are all the Exported classes.
  const topLevelExports = new Set<DeclarationNode>();

  // Do this via `ts.TypeChecker.getExportsOfModule`.
  const moduleSymbol = checker.getSymbolAtLocation(entryPoint);
  if (moduleSymbol === undefined) {
    throw new Error(`Internal error: failed to get symbol for entrypoint`);
  }
  const exportedSymbols = checker.getExportsOfModule(moduleSymbol);

  // Loop through the exported symbols, de-alias if needed, and add them to `topLevelExports`.
  // TODO(alxhub): use proper iteration when build.sh is removed. (#27762)
  for (const symbol of exportedSymbols) {
    if ((symbol.flags & ts.SymbolFlags.Alias) !== 0) {
      const aliasedSymbol = checker.getAliasedSymbol(symbol);
      if (aliasedSymbol.valueDeclaration !== undefined) {
        topLevelExports.add(aliasedSymbol.valueDeclaration);
      }
    } else if (symbol.valueDeclaration !== undefined) {
      topLevelExports.add(symbol.valueDeclaration);
    }
  }

  // Next, go through each exported class and expand it to the set of classes it makes Visible,
  // using the `ReferenceGraph`. For each Visible class, verify that it's also Exported, and queue
  // an error if it isn't. `checkedSet` ensures only one error is queued per class.
  const checkedSet = new Set<DeclarationNode>();

  // Loop through each Exported class.
  for (const mainExport of topLevelExports) {
    // Loop through each class made Visible by the Exported class.
    refGraph.transitiveReferencesOf(mainExport).forEach((transitiveReference) => {
      // Skip classes which have already been checked.
      if (checkedSet.has(transitiveReference)) {
        return;
      }
      checkedSet.add(transitiveReference);

      // Verify that the Visible class is also Exported.
      if (!topLevelExports.has(transitiveReference)) {
        const descriptor = getDescriptorOfDeclaration(transitiveReference);
        const name = getNameOfDeclaration(transitiveReference);

        // Construct the path of visibility, from `mainExport` to `transitiveReference`.
        let visibleVia = 'NgModule exports';
        const transitivePath = refGraph.pathFrom(mainExport, transitiveReference);
        if (transitivePath !== null) {
          visibleVia = transitivePath.map((seg) => getNameOfDeclaration(seg)).join(' -> ');
        }

        const diagnostic: ts.Diagnostic = {
          category: ts.DiagnosticCategory.Error,
          code: ngErrorCode(ErrorCode.SYMBOL_NOT_EXPORTED),
          file: transitiveReference.getSourceFile(),
          ...getPosOfDeclaration(transitiveReference),
          messageText: `Unsupported private ${descriptor} ${name}. This ${descriptor} is visible to consumers via ${visibleVia}, but is not exported from the top-level library entrypoint.`,
        };

        diagnostics.push(diagnostic);
      }
    });
  }

  return diagnostics;
}

