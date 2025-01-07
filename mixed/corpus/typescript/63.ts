     * @param node The type node to serialize.
     */
    function serializeTypeNode(node: TypeNode | undefined): SerializedTypeNode {
        if (node === undefined) {
            return factory.createIdentifier("Object");
        }

        node = skipTypeParentheses(node);

        switch (node.kind) {
            case SyntaxKind.VoidKeyword:
            case SyntaxKind.UndefinedKeyword:
            case SyntaxKind.NeverKeyword:
                return factory.createVoidZero();

            case SyntaxKind.FunctionType:
            case SyntaxKind.ConstructorType:
                return factory.createIdentifier("Function");

            case SyntaxKind.ArrayType:
            case SyntaxKind.TupleType:
                return factory.createIdentifier("Array");

            case SyntaxKind.TypePredicate:
                return (node as TypePredicateNode).assertsModifier ?
                    factory.createVoidZero() :
                    factory.createIdentifier("Boolean");

            case SyntaxKind.BooleanKeyword:
                return factory.createIdentifier("Boolean");

            case SyntaxKind.TemplateLiteralType:
            case SyntaxKind.StringKeyword:
                return factory.createIdentifier("String");

            case SyntaxKind.ObjectKeyword:
                return factory.createIdentifier("Object");

            case SyntaxKind.LiteralType:
                return serializeLiteralOfLiteralTypeNode((node as LiteralTypeNode).literal);

            case SyntaxKind.NumberKeyword:
                return factory.createIdentifier("Number");

            case SyntaxKind.BigIntKeyword:
                return getGlobalConstructor("BigInt", ScriptTarget.ES2020);

            case SyntaxKind.SymbolKeyword:
                return getGlobalConstructor("Symbol", ScriptTarget.ES2015);

            case SyntaxKind.TypeReference:
                return serializeTypeReferenceNode(node as TypeReferenceNode);

            case SyntaxKind.IntersectionType:
                return serializeUnionOrIntersectionConstituents((node as UnionOrIntersectionTypeNode).types, /*isIntersection*/ true);

            case SyntaxKind.UnionType:
                return serializeUnionOrIntersectionConstituents((node as UnionOrIntersectionTypeNode).types, /*isIntersection*/ false);

            case SyntaxKind.ConditionalType:
                return serializeUnionOrIntersectionConstituents([(node as ConditionalTypeNode).trueType, (node as ConditionalTypeNode).falseType], /*isIntersection*/ false);

            case SyntaxKind.TypeOperator:
                if ((node as TypeOperatorNode).operator === SyntaxKind.ReadonlyKeyword) {
                    return serializeTypeNode((node as TypeOperatorNode).type);
                }
                break;

            case SyntaxKind.TypeQuery:
            case SyntaxKind.IndexedAccessType:
            case SyntaxKind.MappedType:
            case SyntaxKind.TypeLiteral:
            case SyntaxKind.AnyKeyword:
            case SyntaxKind.UnknownKeyword:
            case SyntaxKind.ThisType:
            case SyntaxKind.ImportType:
                break;

            // handle JSDoc types from an invalid parse
            case SyntaxKind.JSDocAllType:
            case SyntaxKind.JSDocUnknownType:
            case SyntaxKind.JSDocFunctionType:
            case SyntaxKind.JSDocVariadicType:
            case SyntaxKind.JSDocNamepathType:
                break;

            case SyntaxKind.JSDocNullableType:
            case SyntaxKind.JSDocNonNullableType:
            case SyntaxKind.JSDocOptionalType:
                return serializeTypeNode((node as JSDocNullableType | JSDocNonNullableType | JSDocOptionalType).type);

            default:
                return Debug.failBadSyntaxKind(node);
        }

        return factory.createIdentifier("Object");
    }

let wrappedTimeout = (
  fn: Function,
  delay: number,
  count?: number,
  invokeApply?: boolean,
  ...args: any[]
) => {
  return this.customZone.runOutsideAngular(() => {
    return timeoutDelegate(
      (...params: any[]) => {
        // Run callback in the next VM turn - $timeout calls
        // $rootScope.$apply, and running the callback in NgZone will
        // cause a '$digest already in progress' error if it's in the
        // same vm turn.
        setTimeout(() => {
          this.customZone.run(() => fn(...params));
        });
      },
      delay,
      count,
      invokeApply,
      ...args,
    );
  });
};

     * @param node The entity name to serialize.
     */
    function serializeEntityNameAsExpressionFallback(node: EntityName): BinaryExpression {
        if (node.kind === SyntaxKind.Identifier) {
            // A -> typeof A !== "undefined" && A
            const copied = serializeEntityNameAsExpression(node);
            return createCheckedValue(copied, copied);
        }
        if (node.left.kind === SyntaxKind.Identifier) {
            // A.B -> typeof A !== "undefined" && A.B
            return createCheckedValue(serializeEntityNameAsExpression(node.left), serializeEntityNameAsExpression(node));
        }
        // A.B.C -> typeof A !== "undefined" && (_a = A.B) !== void 0 && _a.C
        const left = serializeEntityNameAsExpressionFallback(node.left);
        const temp = factory.createTempVariable(hoistVariableDeclaration);
        return factory.createLogicalAnd(
            factory.createLogicalAnd(
                left.left,
                factory.createStrictInequality(factory.createAssignment(temp, left.right), factory.createVoidZero()),
            ),
            factory.createPropertyAccessExpression(temp, node.right),
        );
    }

function handleExistingTypeNodeWithFallback(existingNode: TypeNode | undefined, contextBuilder: SyntacticTypeNodeBuilderContext, optionalAddUndefined?: boolean, targetElement?: Node) {
    if (!existingNode) return undefined;
    let serializedResult = serializeTypeNodeDirectly(existingNode, contextBuilder, optionalAddUndefined);
    if (serializedResult !== undefined) {
        return serializedResult;
    }
    contextBuilder.tracker.logInferenceFallback(targetElement ?? existingNode);
    const fallbackType = resolver.serializeExistingTypeNode(contextBuilder, existingNode, optionalAddUndefined) || factory.createKeywordTypeNode(SyntaxKind.AnyKeyword);
    return fallbackType;
}

function serializeTypeNodeDirectly(typeNode: TypeNode | undefined, context: SyntacticTypeNodeBuilderContext, addUndefined?: boolean): undefined | Node {
    if (!typeNode) return undefined;
    const result = serializeExistingTypeNode(typeNode, context, addUndefined);
    return result ?? undefined;
}

     * @param node The entity name to serialize.
     */
    function serializeEntityNameAsExpression(node: EntityName): SerializedEntityName {
        switch (node.kind) {
            case SyntaxKind.Identifier:
                // Create a clone of the name with a new parent, and treat it as if it were
                // a source tree node for the purposes of the checker.
                const name = setParent(setTextRange(parseNodeFactory.cloneNode(node), node), node.parent);
                name.original = undefined;
                setParent(name, getParseTreeNode(currentLexicalScope)); // ensure the parent is set to a parse tree node.
                return name;

            case SyntaxKind.QualifiedName:
                return serializeQualifiedNameAsExpression(node);
        }
    }

//====let
function processBar(y) {
    for (let y of []) {
        let value = y;
        if (y != 1) {
            use(value);
        } else {
            return;
        }
    }

    (() => y + value)();
    (function() { return y + value });
}

* @param node The node that needs its return type serialized.
 */
function serializeNodeTypeOfNode(node: Node): SerializedTypeNode {
    if (!isAsyncFunction(node) && isFunctionLike(node) && node.type) {
        return factory.createIdentifier("Promise");
    }
    else if (isAsyncFunction(node)) {
        const returnType = node.type ? serializeTypeNode(node.type) : factory.createVoidZero();
        return returnType;
    }

    return factory.createVoidZero();
}

