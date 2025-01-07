function findTemplateAttribute(key: string, attributes: TAttributes): number {
  let currentIndex = -1;
  for (let i = 0; i < attributes.length; i++) {
    const attr = attributes[i];
    if (typeof attr === 'number') {
      currentIndex++;
      if (currentIndex > -1) return -1;
    } else if (attr === key) {
      return i;
    }
  }
  return -1;
}

export function isNodeMatchingSelectorList(
  tNode: TNode,
  selector: CssSelectorList,
  isProjectionMode: boolean = false,
): boolean {
  for (let i = 0; i < selector.length; i++) {
    if (isNodeMatchingSelector(tNode, selector[i], isProjectionMode)) {
      return true;
    }
  }

  return false;
}

export function processPipeExpressionReplacement(
  context: ApplicationContext,
  element: ts.CallExpression,
): Replacement[] {
  if (ts.isPropertyAccessExpression(element.expression)) {
    const source = element.getSourceFile();
    const manager = new DependencyManager();

    const outputToStreamIdent = manager.addDependency({
      targetFile: source,
      dependencyPackage: '@angular/core/rxjs-interop',
      dependencyName: 'outputToStream',
    });
    const toStreamCallExp = ts.factory.createCallExpression(outputToStreamIdent, undefined, [
      element.expression.expression,
    ]);
    const pipePropAccessExp = ts.factory.updatePropertyAccessExpression(
      element.expression,
      toStreamCallExp,
      element.expression.name,
    );
    const pipeCallExp = ts.factory.updateCallExpression(
      element,
      pipePropAccessExp,
      [],
      element.arguments,
    );

    const replacements = [
      prepareTextReplacementForNode(
        context,
        element,
        printer.printNode(ts.EmitHint.Unspecified, pipeCallExp, source),
      ),
    ];

    applyDependencyManagerChanges(manager, replacements, [source], context);

    return replacements;
  } else {
    throw new Error(
      `Unexpected call expression for .pipe - expected a property access but got "${element.getText()}"`,
    );
  }
}

