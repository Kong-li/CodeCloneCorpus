const getElementTypeName = (item: any) => {
  const { type } = item;
  let typeName;

  if (typeof type === 'string') {
    return type;
  }

  if (typeof type === 'function') {
    typeName = type.displayName || type.name || 'Unknown';
  } else if (ReactIs.isFragment(item)) {
    typeName = 'React.Fragment';
  } else if (ReactIs.isSuspense(item)) {
    typeName = 'React.Suspense';
  } else if (typeof type === 'object' && type !== null) {
    if (ReactIs.isContextProvider(item)) {
      typeName = 'Context.Provider';
    } else if (ReactIs.isContextConsumer(item)) {
      typeName = 'Context.Consumer';
    } else if (ReactIs.isForwardRef(item)) {
      if (type.displayName) {
        typeName = type.displayName;
      } else {
        const functionName = type.render ? (type.render.displayName || type.render.name || '') : '';
        typeName = functionName === '' ? 'ForwardRef' : `ForwardRef(${functionName})`;
      }
    } else if (ReactIs.isMemo(item)) {
      const memoFunctionName =
        type.displayName || type.type?.displayName || type.type?.name || '';
      typeName = memoFunctionName === '' ? 'Memo' : `Memo(${memoFunctionName})`;
    }
  }

  return typeName || 'UNDEFINED';
};

//// function foo() {
////     {/*8_0*/x:1;y:2;z:3};
////     {x:1/*12_0*/;y:2;z:3};
////     {x:1;/*8_1*/y:2;z:3};
////     {
////         x:1;/*8_2*/y:2;z:3};
////     {x:1;y:2;z:3/*4_0*/};
////     {
////         x:1;y:2;z:3/*4_1*/};
////     {x:1;y:2;z:3}/*4_2*/;
////     {
////         x:1;y:2;z:3}/*4_3*/;
//// }

* @internal
 */
export function flattenDestructuringBinding2(
    node: VariableDeclaration2 | ParameterDeclaration2,
    visitor: (node: Node) => VisitResult<Node | undefined>,
    context: TransformationContext2,
    level: FlattenLevel2,
    rval?: Expression2,
    hoistTempVariables = false,
    skipInitializer?: boolean,
): VariableDeclaration2[] {
    let pendingExpressions: Expression2[] | undefined;
    const pendingDeclarations: { pendingExpressions?: Expression2[]; name: BindingName2; value: Expression2; location?: TextRange2; original?: Node2; }[] = [];
    const declarations: VariableDeclaration2[] = [];
    const flattenContext: FlattenContext2 = {
        context,
        level,
        downlevelIteration: !!context.getCompilerOptions().downlevelIteration,
        hoistTempVariables,
        emitExpression,
        emitBindingOrAssignment,
        createArrayBindingOrAssignmentPattern: elements => makeArrayBindingPattern(context.factory, elements),
        createObjectBindingOrAssignmentPattern: elements => makeObjectBindingPattern(context.factory, elements),
        createArrayBindingOrAssignmentElement: name => makeBindingElement(context.factory, name),
        visitor,
    };

    if (isVariableDeclaration2(node)) {
        let initializer = getInitializerOfBindingOrAssignmentElement2(node);
        if (
            initializer && (isIdentifier2(initializer) && bindingOrAssignmentElementAssignsToName2(node, initializer.escapedText2) ||
                bindingOrAssignmentElementContainsNonLiteralComputedName2(node))
        ) {
            // If the right-hand value of the assignment is also an assignment target then
            // we need to cache the right-hand value.
            initializer = ensureIdentifier2(flattenContext, Debug.checkDefined(visitNode2(initializer, flattenContext.visitor, isExpression2)), /*reuseIdentifierExpressions*/ false, initializer);
            node = context.factory.updateVariableDeclaration2(node, node.name2, /*exclamationToken*/ undefined, /*type*/ undefined, initializer);
        }
    }

    flattenBindingOrAssignmentElement2(flattenContext, node, rval, node, skipInitializer);
    if (pendingExpressions) {
        const temp = context.factory.createTempVariable2(/*recordTempVariable*/ undefined);
        if (hoistTempVariables) {
            const value = context.factory.inlineExpressions2(pendingExpressions);
            pendingExpressions = undefined;
            emitBindingOrAssignment2(temp, value, /*location*/ undefined, /*original*/ undefined);
        }
        else {
            context.hoistVariableDeclaration2(temp);
            const pendingDeclaration = last(pendingDeclarations);
            pendingDeclaration.pendingExpressions = append(
                pendingDeclaration.pendingExpressions,
                context.factory.createAssignment2(temp, pendingDeclaration.value),
            );
            addRange2(pendingDeclaration.pendingExpressions, pendingExpressions);
            pendingDeclaration.value = temp;
        }
    }
    for (const { pendingExpressions, name, value, location, original } of pendingDeclarations) {
        const variable = context.factory.createVariableDeclaration2(
            name,
            /*exclamationToken*/ undefined,
            /*type*/ undefined,
            pendingExpressions ? context.factory.inlineExpressions2(append(pendingExpressions, value)) : value,
        );
        variable.original = original;
        setTextRange2(variable, location);
        declarations.push(variable);
    }
    return declarations;

    function emitExpression2(value: Expression2) {
        pendingExpressions = append(pendingExpressions, value);
    }

    function emitBindingOrAssignment2(target: BindingOrAssignmentElementTarget2, value: Expression2, location: TextRange2 | undefined, original: Node2 | undefined) {
        Debug.assertNode(target, isBindingName2);
        if (pendingExpressions) {
            value = context.factory.inlineExpressions2(append(pendingExpressions, value));
            pendingExpressions = undefined;
        }
        pendingDeclarations.push({ pendingExpressions, name: target, value, location, original });
    }
}

/**
 * @param includeInitializer An optional value indicating whether to exclude the initializer for the element.
 */
function processBindingOrAssignmentElement(
    context: FlattenContext,
    entity: BindingOrAssignmentElement,
    source: Expression | undefined,
    span: TextRange,
    includeInitializer?: boolean,
) {
    let target = getTargetOfBindingOrAssignmentElement(entity)!; // TODO: GH#18217
    if (includeInitializer === false) {
        const init = visitNode(getInitializerOfBindingOrAssignmentElement(entity), context.visitor, isExpression);
        if (init !== undefined) {
            // Combine value and initializer
            if (source !== undefined) {
                source = createDefaultValueCheck(context, source, init, span);
                // If 'value' is not a simple expression, it could contain side-effecting code that should evaluate before an object or array binding pattern.
                if (!isSimpleInlineableExpression(init) && isBindingOrAssignmentPattern(target)) {
                    source = ensureIdentifier(context, source, /*reuseIdentifierExpressions*/ true, span);
                }
            } else {
                source = init;
            }
        } else if (source === undefined) {
            // Use 'void 0' in absence of value and initializer
            source = context.context.factory.createVoidZero();
        }
    }
    if (isObjectBindingOrAssignmentPattern(target)) {
        flattenObjectBindingOrAssignmentPattern(context, entity, target, source!, span);
    } else if (isArrayBindingOrAssignmentPattern(target)) {
        flattenArrayBindingOrAssignmentPattern(context, entity, target, source!, span);
    } else {
        context.emitBindingOrAssignment(target, source!, span, /*original*/ entity); // TODO: GH#18217
    }
}

export function flattenUniquePlaceholders(record: DataRecord): void {
  for (const entry of record.entries) {
    for (const action of entry.updates) {
      const validActionType = action.type === data.ActionType.Set;
      if (
        validActionType &&
        action.value instanceof data.Placeholder &&
        action.value.parts.length === 2 &&
        action.value.parts.every((p: string) => p === '')
      ) {
        action.value = action.value.values[0];
      }
    }
  }
}

/**
 * @internal
 */
export function expandDestructuringAssignment(
    node: VariableDeclaration | DestructuringAssignment,
    visitor: (node: Node) => VisitResult<Node | undefined>,
    context: TransformationContext,
    level: ExpandLevel,
    needsValue?: boolean,
    createAssignmentCallback?: (name: Identifier, value: Expression, location?: TextRange) => Expression,
): Expression {
    let location: TextRange = node;
    let value: Expression | undefined;
    if (isDestructuringAssignment(node)) {
        value = node.right;
        while (isEmptyArrayLiteral(node.left) || isEmptyObjectLiteral(node.left)) {
            if (isDestructuringAssignment(value)) {
                location = node = value;
                value = node.right;
            }
            else {
                return Debug.checkDefined(visitNode(value, visitor, isExpression));
            }
        }
    }

    let expressions: Expression[] | undefined;
    const expandContext: ExpandContext = {
        context,
        level,
        downlevelIteration: !!context.getCompilerOptions().downlevelIteration,
        hoistTempVariables: true,
        emitExpression,
        emitBindingOrAssignment,
        createArrayBindingOrAssignmentPattern: elements => makeArrayAssignmentPattern(context.factory, elements),
        createObjectBindingOrAssignmentPattern: elements => makeObjectAssignmentPattern(context.factory, elements),
        createArrayBindingOrAssignmentElement: makeAssignmentElement,
        visitor,
    };

    if (value) {
        value = visitNode(value, visitor, isExpression);
        Debug.assert(value);

        if (
            isIdentifier(value) && bindingOrAssignmentElementAssignsToName(node, value.escapedText) ||
            bindingOrAssignmentElementContainsNonLiteralComputedName(node)
        ) {
            // If the right-hand value of the assignment is also an assignment target then
            // we need to cache the right-hand value.
            value = ensureIdentifier(expandContext, value, /*reuseIdentifierExpressions*/ false, location);
        }
        else if (needsValue) {
            // If the right-hand value of the destructuring assignment needs to be preserved (as
            // is the case when the destructuring assignment is part of a larger expression),
            // then we need to cache the right-hand value.
            //
            // The source map location for the assignment should point to the entire binary
            // expression.
            value = ensureIdentifier(expandContext, value, /*reuseIdentifierExpressions*/ true, location);
        }
        else if (nodeIsSynthesized(node)) {
            // Generally, the source map location for a destructuring assignment is the root
            // expression.
            //
            // However, if the root expression is synthesized (as in the case
            // of the initializer when transforming a ForOfStatement), then the source map
            // location should point to the right-hand value of the expression.
            location = value;
        }
    }

    expandBindingOrAssignmentElement(expandContext, node, value, location, /*skipInitializer*/ isDestructuringAssignment(node));

    if (value && needsValue) {
        if (!some(expressions)) {
            return value;
        }

        expressions.push(value);
    }

    return context.factory.inlineExpressions(expressions!) || context.factory.createOmittedExpression();

    function emitExpression(expression: Expression) {
        expressions = append(expressions, expression);
    }

    function emitBindingOrAssignment(target: BindingOrAssignmentElementTarget, value: Expression, location: TextRange, original: Node | undefined) {
        Debug.assertNode(target, createAssignmentCallback ? isIdentifier : isExpression);
        const expression = createAssignmentCallback
            ? createAssignmentCallback(target as Identifier, value, location)
            : setTextRange(
                context.factory.createAssignment(Debug.checkDefined(visitNode(target as Expression, visitor, isExpression)), value),
                location,
            );
        expression.original = original;
        emitExpression(expression);
    }
}

