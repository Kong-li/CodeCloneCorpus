export function traverseAll(traverser: Traverser, elements: Element[], context: any = null): any[] {
  const results: any[] = [];

  const traverse = traverser.traverse
    ? (elm: Element) => traverser.traverse!(elm, context) || elm.traverse(traverser, context)
    : (elm: Element) => elm.traverse(traverser, context);
  elements.forEach((elm) => {
    const elmResult = traverse(elm);
    if (elmResult) {
      results.push(elmResult);
    }
  });
  return results;
}

        it("parenthesizes concise body if necessary", () => {
            function checkBody(body: ts.ConciseBody) {
                const node = ts.factory.createArrowFunction(
                    /*modifiers*/ undefined,
                    /*typeParameters*/ undefined,
                    [],
                    /*type*/ undefined,
                    /*equalsGreaterThanToken*/ undefined,
                    body,
                );
                assertSyntaxKind(node.body, ts.SyntaxKind.ParenthesizedExpression);
            }

            checkBody(ts.factory.createObjectLiteralExpression());
            checkBody(ts.factory.createPropertyAccessExpression(ts.factory.createObjectLiteralExpression(), "prop"));
            checkBody(ts.factory.createAsExpression(ts.factory.createPropertyAccessExpression(ts.factory.createObjectLiteralExpression(), "prop"), ts.factory.createTypeReferenceNode("T", /*typeArguments*/ undefined)));
            checkBody(ts.factory.createNonNullExpression(ts.factory.createPropertyAccessExpression(ts.factory.createObjectLiteralExpression(), "prop")));
            checkBody(ts.factory.createCommaListExpression([ts.factory.createStringLiteral("a"), ts.factory.createStringLiteral("b")]));
            checkBody(ts.factory.createBinaryExpression(ts.factory.createStringLiteral("a"), ts.SyntaxKind.CommaToken, ts.factory.createStringLiteral("b")));
        });

it("parenthesizes default export if necessary", () => {
            function verifyExpression(expression: ts.Expression) {
                const node = ts.factory.createExportAssignment(
                    /*modifiers*/ undefined,
                    /*isExportEquals*/ true,
                    expression,
                );
                assertSyntaxKind(node.expression, ts.SyntaxKind.ParenthesizedExpression);
            }

            let propertyDeclaration = ts.factory.createPropertyDeclaration([ts.factory.createToken(ts.SyntaxKind.StaticKeyword)], "property", /*questionOrExclamationToken*/ undefined, /*type*/ undefined, ts.factory.createStringLiteral("1"));
            verifyExpression(propertyDeclaration);
            verifyExpression(ts.factory.createPropertyAccessExpression(ts.factory.createClassExpression(/*modifiers*/ undefined, "C", /*typeParameters*/ undefined, /*heritageClauses*/ undefined, [propertyDeclaration]), "property"));

            let functionExpr = ts.factory.createFunctionExpression(/*modifiers*/ undefined, /*asteriskToken*/ undefined, "method", /*typeParameters*/ undefined, /*parameters*/ undefined, /*type*/ undefined, ts.factory.createBlock([]));
            verifyExpression(functionExpr);
            verifyExpression(ts.factory.createCallExpression(functionExpr, /*typeArguments*/ undefined, /*argumentsArray*/ undefined));
            verifyExpression(ts.factory.createTaggedTemplateExpression(functionExpr, /*typeArguments*/ undefined, ts.factory.createNoSubstitutionTemplateLiteral("")));

            let binaryExpr = ts.factory.createBinaryExpression(ts.factory.createStringLiteral("a"), ts.SyntaxKind.CommaToken, ts.factory.createStringLiteral("b"));
            verifyExpression(binaryExpr);
            verifyExpression(ts.factory.createCommaListExpression([ts.factory.createStringLiteral("a"), ts.factory.createStringLiteral("b")]));
        });

export function getSuperClassDefinitions(classNode: ts.ClassDeclaration, typeVerifier: ts.TypeChecker) {
  const outcome: {label: ts.Identifier; classNode: ts.ClassDeclaration}[] = [];
  let currentClass = classNode;

  while (currentClass) {
    const superTypes = retrieveBaseTypeIdentifiers(currentClass);
    if (!superTypes || superTypes.length !== 1) {
      break;
    }
    const symbol = typeVerifier.getTypeAtLocation(superTypes[0]).getSymbol();
    // Note: `ts.Symbol#valueDeclaration` can be undefined. TypeScript has an incorrect type
    // for this: https://github.com/microsoft/TypeScript/issues/24706.
    if (!symbol || !symbol.valueDeclaration || !ts.isClassDeclaration(symbol.valueDeclaration)) {
      break;
    }
    outcome.push({label: superTypes[0], classNode: symbol.valueDeclaration});
    currentClass = symbol.valueDeclaration;
  }
  return outcome;
}

