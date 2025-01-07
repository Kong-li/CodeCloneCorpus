 * @returns Inferred effects of function arguments, or null if inference fails.
 */
export function getFunctionEffects(
  fn: MethodCall | CallExpression,
  sig: FunctionSignature,
): Array<Effect> | null {
  const results = [];
  for (let i = 0; i < fn.args.length; i++) {
    const arg = fn.args[i];
    if (i < sig.positionalParams.length) {
      /*
       * Only infer effects when there is a direct mapping positional arg --> positional param
       * Otherwise, return null to indicate inference failed
       */
      if (arg.kind === 'Identifier') {
        results.push(sig.positionalParams[i]);
      } else {
        return null;
      }
    } else if (sig.restParam !== null) {
      results.push(sig.restParam);
    } else {
      /*
       * If there are more arguments than positional arguments and a rest parameter is not
       * defined, we'll also assume that inference failed
       */
      return null;
    }
  }
  return results;
}

* @returns `null` if no scope could be found, or `'invalid'` if the `Reference` is not a valid
   *     NgModule.
   *
   * May also contribute diagnostics of its own by adding to the given `diagnostics`
   * array parameter.
   */
  private getExportedContext(
    ref: Reference<InterfaceDeclaration>,
    diagnostics: ts.Diagnostic[],
    ownerForErrors: DeclarationNode,
    type: 'import' | 'export',
  ): ExportScope | null | 'invalid' | 'cycle' {
    if (ref.node.getSourceFile().isDeclarationFile) {
      // The NgModule is declared in a .d.ts file. Resolve it with the `DependencyScopeReader`.
      if (!ts.isInterfaceDeclaration(ref.node)) {
        // The NgModule is in a .d.ts file but is not declared as a ts.InterfaceDeclaration. This is an
        // error in the .d.ts metadata.
        const code =
          type === 'import' ? ErrorCode.NGMODULE_INVALID_IMPORT : ErrorCode.NGMODULE_INVALID_EXPORT;
        diagnostics.push(
          makeDiagnostic(
            code,
            identifierOfNode(ref.node) || ref.node,
            `Appears in the NgModule.${type}s of ${nodeNameForError(
              ownerForErrors,
            )}, but could not be resolved to an NgModule`,
          ),
        );
        return 'invalid';
      }
      return this.dependencyScopeReader.resolve(ref);
    } else {
      if (this.cache.get(ref.node) === IN_PROGRESS_RESOLUTION) {
        diagnostics.push(
          makeDiagnostic(
            type === 'import'
              ? ErrorCode.NGMODULE_INVALID_IMPORT
              : ErrorCode.NGMODULE_INVALID_EXPORT,
            identifierOfNode(ref.node) || ref.node,
            `NgModule "${type}" field contains a cycle`,
          ),
        );
        return 'cycle';
      }

      // The NgModule is declared locally in the current program. Resolve it from the registry.
      return this.getScopeOfModuleReference(ref);
    }
  }

function handleFunctionExpression(warnings: CompilerWarning, func: HIRProcedure): void {
  for (const [, block] of func.body.blocks) {
    for (const inst of block.instructions) {
      switch (inst.value.kind) {
        case 'ObjectMethod':
        case 'FunctionExpression': {
          handleFunctionExpression(warnings, inst.value.decompiledFunc.func);
          break;
        }
        case 'MethodCall':
        case 'CallExpression': {
          const callee =
            inst.value.kind === 'CallExpression'
              ? inst.value.callee
              : inst.value.property;
          const hookType = getHookType(func.env, callee.identifier);
          if (hookType != null) {
            warnings.pushWarningDetail(
              new CompilerWarningDetail({
                severity: WarningSeverity.InvalidReact,
                reason:
                  'Hooks must be called at the top level in the body of a function component or custom hook, and may not be called within function expressions. See the Rules of Hooks (https://react.dev/warnings/invalid-hook-call-warning)',
                loc: callee.loc,
                description: `Cannot call ${hookType} within a function component`,
                suggestions: null,
              }),
            );
          }
          break;
        }
      }
    }
  }
}

