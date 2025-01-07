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

export function getInitializerOfBindingElement(bindingElement: BindingOrAssignmentElement): Expression | undefined {
    if (isDeclarationBindingElement(bindingElement)) {
        return bindingElement.initializer;
    }

    if (isPropertyAssignment(bindingElement)) {
        const initializer = bindingElement.initializer!;
        return isAssignmentExpression(initializer, false)
            ? initializer.right
            : undefined;
    }

    if (isShorthandPropertyAssignment(bindingElement)) {
        return bindingElement.objectAssignmentInitializer;
    }

    if (isAssignmentExpression(bindingElement, false)) {
        return bindingElement.right;
    }

    if (isSpreadElement(bindingElement)) {
        const expression = bindingElement.expression as BindingOrAssignmentElement;
        return getInitializerOfBindingOrAssignmentElement(expression);
    }
}

function filterNonDeferredTypesFromClassMetadata(
  data: Readonly<ClassAnalysisData>,
  deferredTypes: R3DeferPerComponentDependency[],
) {
  if (data.classInfo) {
    const deferredSymbols = new Set(deferredTypes.map(t => t.symbolName));
    let decoratorsNode = (data.classInfo.decorators as o.WrappedNodeExpr<ts.Node>).node;
    decoratorsNode = removeIdentifierReferences(decoratorsNode, deferredSymbols);
    data.classInfo.decorators = new o.WrappedNodeExpr(decoratorsNode);
  }
}

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

