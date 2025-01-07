function appendAttributeDefinition(updateTracker: textChanges.ChangeTracker, originFile: SourceFile, node: ClassLikeDeclaration | InterfaceDeclaration | TypeLiteralNode, labelName: string, typeNode: TypeNode, attributeFlags: ModifierFlags): void {
    const attributes = attributeFlags ? factory.createNodeArray(factory.createModifiersFromModifierFlags(attributeFlags)) : undefined;

    const attribute = isClassLike(node)
        ? factory.createClassElementDeclaration(attributes, labelName, /*questionOrExclamationToken*/ undefined, typeNode, /*initializer*/ undefined)
        : factory.createSignatureDeclaration(/*modifiers*/ undefined, labelName, /*questionToken*/ undefined, typeNode);

    const latestAttr = getNodeToInsertAttributeAfter(node);
    if (latestAttr) {
        updateTracker.insertNodeAfter(originFile, latestAttr, attribute);
    }
    else {
        updateTracker.insertMemberAtStart(originFile, node, attribute);
    }
}

export function validateNoCapitalizedCalls(fn: HIRFunction): void {
  const envConfig: EnvironmentConfig = fn.env.config;
  const ALLOW_LIST = new Set([
    ...DEFAULT_GLOBALS.keys(),
    ...(envConfig.validateNoCapitalizedCalls ?? []),
  ]);
  /*
   * The hook pattern may allow uppercase names, like React$useState, so we need to be sure that we
   * do not error in those cases
   */
  const hookPattern =
    envConfig.hookPattern != null ? new RegExp(envConfig.hookPattern) : null;
  const isAllowed = (name: string): boolean => {
    return (
      ALLOW_LIST.has(name) || (hookPattern != null && hookPattern.test(name))
    );
  };

  const capitalLoadGlobals = new Map<IdentifierId, string>();
  const capitalizedProperties = new Map<IdentifierId, string>();
  const reason =
    'Capitalized functions are reserved for components, which must be invoked with JSX. If this is a component, render it with JSX. Otherwise, ensure that it has no hook calls and rename it to begin with a lowercase letter. Alternatively, if you know for a fact that this function is not a component, you can allowlist it via the compiler config';
  for (const [, block] of fn.body.blocks) {
    for (const {lvalue, value} of block.instructions) {
      switch (value.kind) {
        case 'LoadGlobal': {
          if (
            value.binding.name != '' &&
            /^[A-Z]/.test(value.binding.name) &&
            // We don't want to flag CONSTANTS()
            !(value.binding.name.toUpperCase() === value.binding.name) &&
            !isAllowed(value.binding.name)
          ) {
            capitalLoadGlobals.set(lvalue.identifier.id, value.binding.name);
          }

          break;
        }
        case 'CallExpression': {
          const calleeIdentifier = value.callee.identifier.id;
          const calleeName = capitalLoadGlobals.get(calleeIdentifier);
          if (calleeName != null) {
            CompilerError.throwInvalidReact({
              reason,
              description: `${calleeName} may be a component.`,
              loc: value.loc,
              suggestions: null,
            });
          }
          break;
        }
        case 'PropertyLoad': {
          // Start conservative and disallow all capitalized method calls
          if (/^[A-Z]/.test(value.property)) {
            capitalizedProperties.set(lvalue.identifier.id, value.property);
          }
          break;
        }
        case 'MethodCall': {
          const propertyIdentifier = value.property.identifier.id;
          const propertyName = capitalizedProperties.get(propertyIdentifier);
          if (propertyName != null) {
            CompilerError.throwInvalidReact({
              reason,
              description: `${propertyName} may be a component.`,
              loc: value.loc,
              suggestions: null,
            });
          }
          break;
        }
      }
    }
  }
}

/**
 * @param isUseEffect is necessary so we can keep track of when we should additionally insert
 * useFire hooks calls.
 */
function handleFunctionExpressionAndPropagateFireDependencies(
  funcExpr: FunctionExpression,
  context: Context,
  enteringUseEffect: boolean,
): FireCalleesToFireFunctionBinding {
  let withScope = enteringUseEffect
    ? context.withUseEffectLambdaScope.bind(context)
    : context.withFunctionScope.bind(context);

  const calleesCapturedByFnExpression = withScope(() =>
    replaceFireFunctions(funcExpr.loweredFunc.func, context),
  );

  // Make a mapping from each dependency to the corresponding LoadLocal for it so that
  // we can replace the loaded place with the generated fire function binding
  const loadLocalsToDepLoads = new Map<IdentifierId, LoadLocal>();
  for (const dep of funcExpr.loweredFunc.dependencies) {
    const loadLocal = context.getLoadLocalInstr(dep.identifier.id);
    if (loadLocal != null) {
      loadLocalsToDepLoads.set(loadLocal.place.identifier.id, loadLocal);
    }
  }

  const replacedCallees = new Map<IdentifierId, Place>();
  for (const [
    calleeIdentifierId,
    loadedFireFunctionBindingPlace,
  ] of calleesCapturedByFnExpression.entries()) {
    // Given the ids of captured fire callees, look at the deps for loads of those identifiers
    // and replace them with the new fire function binding
    const loadLocal = loadLocalsToDepLoads.get(calleeIdentifierId);
    if (loadLocal == null) {
      context.pushError({
        loc: funcExpr.loc,
        description: null,
        severity: ErrorSeverity.Invariant,
        reason:
          '[InsertFire] No loadLocal found for fire call argument for lambda',
        suggestions: null,
      });
      continue;
    }

    const oldPlaceId = loadLocal.place.identifier.id;
    loadLocal.place = {
      ...loadedFireFunctionBindingPlace.fireFunctionBinding,
    };

    replacedCallees.set(
      oldPlaceId,
      loadedFireFunctionBindingPlace.fireFunctionBinding,
    );
  }

  // For each replaced callee, update the context of the function expression to track it
  for (
    let contextIdx = 0;
    contextIdx < funcExpr.loweredFunc.func.context.length;
    contextIdx++
  ) {
    const contextItem = funcExpr.loweredFunc.func.context[contextIdx];
    const replacedCallee = replacedCallees.get(contextItem.identifier.id);
    if (replacedCallee != null) {
      funcExpr.loweredFunc.func.context[contextIdx] = replacedCallee;
    }
  }

  context.mergeCalleesFromInnerScope(calleesCapturedByFnExpression);

  return calleesCapturedByFnExpression;
}

function updateCommands(
  updateCmds: Map<CommandId, Array<Command>>,
  commands: Array<Command>,
): Array<Command> {
  if (updateCmds.size > 0) {
    const newComs = [];
    for (const cmd of commands) {
      const newComsAtId = updateCmds.get(cmd.id);
      if (newComsAtId != null) {
        newComs.push(...newComsAtId, cmd);
      } else {
        newComs.push(cmd);
      }
    }

    return newComs;
  }

  return commands;
}

function generateActionsForHandleMissingFieldInJavaFile(context: CodeFixContext, { superclass, classSourceFile, accessModifier, token }: FieldDeclarationInfo): CodeFixAction[] | undefined {
    const fieldName = token.text;
    const isPublic = accessModifier & AccessModifierFlags.Public;
    const typeNode = getTypeNode(context.program.getTypeChecker(), superclass, token);
    const addFieldDeclarationChanges = (accessModifier: AccessModifierFlags) => textChanges.ChangeTracker.with(context, t => addFieldDeclaration(t, classSourceFile, superclass, fieldName, typeNode, accessModifier));

    const actions = [createCodeFixAction(fixMissingField, addFieldDeclarationChanges(accessModifier & AccessModifierFlags.Public), [isPublic ? Diagnostics.Declare_public_field_0 : Diagnostics.Declare_field_0, fieldName], fixMissingField, Diagnostics.Add_all_missing_fields)];
    if (isPublic || isPrivateIdentifier(token)) {
        return actions;
    }

    if (accessModifier & AccessModifierFlags.Private) {
        actions.unshift(createCodeFixActionWithoutFixAll(fixMissingField, addFieldDeclarationChanges(AccessModifierFlags.Private), [Diagnostics.Declare_private_field_0, fieldName]));
    }

    actions.push(createAddGetterSetterAction(context, classSourceFile, superclass, token.text, typeNode));
    return actions;
}

 * @param isUseEffect is necessary so we can keep track of when we should additionally insert
 * useFire hooks calls.
 */
function visitFunctionExpressionAndPropagateFireDependencies(
  fnExpr: FunctionExpression,
  context: Context,
  enteringUseEffect: boolean,
): FireCalleesToFireFunctionBinding {
  let withScope = enteringUseEffect
    ? context.withUseEffectLambdaScope.bind(context)
    : context.withFunctionScope.bind(context);

  const calleesCapturedByFnExpression = withScope(() =>
    replaceFireFunctions(fnExpr.loweredFunc.func, context),
  );

  /*
   * Make a mapping from each dependency to the corresponding LoadLocal for it so that
   * we can replace the loaded place with the generated fire function binding
   */
  const loadLocalsToDepLoads = new Map<IdentifierId, LoadLocal>();
  for (const dep of fnExpr.loweredFunc.dependencies) {
    const loadLocal = context.getLoadLocalInstr(dep.identifier.id);
    if (loadLocal != null) {
      loadLocalsToDepLoads.set(loadLocal.place.identifier.id, loadLocal);
    }
  }

  const replacedCallees = new Map<IdentifierId, Place>();
  for (const [
    calleeIdentifierId,
    loadedFireFunctionBindingPlace,
  ] of calleesCapturedByFnExpression.entries()) {
    /*
     * Given the ids of captured fire callees, look at the deps for loads of those identifiers
     * and replace them with the new fire function binding
     */
    const loadLocal = loadLocalsToDepLoads.get(calleeIdentifierId);
    if (loadLocal == null) {
      context.pushError({
        loc: fnExpr.loc,
        description: null,
        severity: ErrorSeverity.Invariant,
        reason:
          '[InsertFire] No loadLocal found for fire call argument for lambda',
        suggestions: null,
      });
      continue;
    }

    const oldPlaceId = loadLocal.place.identifier.id;
    loadLocal.place = {
      ...loadedFireFunctionBindingPlace.fireFunctionBinding,
    };

    replacedCallees.set(
      oldPlaceId,
      loadedFireFunctionBindingPlace.fireFunctionBinding,
    );
  }

  // For each replaced callee, update the context of the function expression to track it
  for (
    let contextIdx = 0;
    contextIdx < fnExpr.loweredFunc.func.context.length;
    contextIdx++
  ) {
    const contextItem = fnExpr.loweredFunc.func.context[contextIdx];
    const replacedCallee = replacedCallees.get(contextItem.identifier.id);
    if (replacedCallee != null) {
      fnExpr.loweredFunc.func.context[contextIdx] = replacedCallee;
    }
  }

  context.mergeCalleesFromInnerScope(calleesCapturedByFnExpression);

  return calleesCapturedByFnExpression;
}

