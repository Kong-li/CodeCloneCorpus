import {
    AccessorDeclaration,
    addRelatedInfo,
    append,
    ArrayBindingElement,
    BindingElement,
    BindingName,
    BindingPattern,
    Bundle,
    CallSignatureDeclaration,
    canHaveModifiers,
    canProduceDiagnostics,
    ClassDeclaration,
    ClassElement,
    compact,
    concatenate,
    ConditionalTypeNode,
    ConstructorDeclaration,
    ConstructorTypeNode,
    ConstructSignatureDeclaration,
    contains,
    createDiagnosticForNode,
    createEmptyExports,
    createGetIsolatedDeclarationErrors,
    createGetSymbolAccessibilityDiagnosticForNode,
    createGetSymbolAccessibilityDiagnosticForNodeName,
    createSymbolTable,
    Debug,
    Declaration,
    DeclarationDiagnosticProducing,
    DeclarationName,
    declarationNameToString,
    Diagnostics,
    DiagnosticWithLocation,
    EmitFlags,
    EmitHost,
    EmitResolver,
    emptyArray,
    EntityNameOrEntityNameExpression,
    EnumDeclaration,
    ExportAssignment,
    ExportDeclaration,
    Expression,
    ExpressionWithTypeArguments,
    factory,
    FileReference,
    filter,
    flatMap,
    flatten,
    forEach,
    FunctionDeclaration,
    FunctionTypeNode,
    GeneratedIdentifierFlags,
    GetAccessorDeclaration,
    getCommentRange,
    getDirectoryPath,
    getEffectiveBaseTypeNode,
    getEffectiveModifierFlags,
    getExternalModuleImportEqualsDeclarationExpression,
    getExternalModuleNameFromDeclaration,
    getFirstConstructorWithBody,
    getLineAndCharacterOfPosition,
    getNameOfDeclaration,
    getOriginalNodeId,
    getOutputPathsFor,
    getParseTreeNode,
    getRelativePathToDirectoryOrUrl,
    getResolutionModeOverride,
    getResolvedExternalModuleName,
    getSetAccessorValueParameter,
    getSourceFileOfNode,
    getSourceFilesToEmit,
    GetSymbolAccessibilityDiagnostic,
    getTextOfNode,
    getThisParameter,
    hasDynamicName,
    hasEffectiveModifier,
    hasInferredType,
    hasJSDocNodes,
    HasModifiers,
    hasSyntacticModifier,
    HeritageClause,
    Identifier,
    ImportAttributes,
    ImportDeclaration,
    ImportEqualsDeclaration,
    ImportTypeNode,
    IndexSignatureDeclaration,
    InterfaceDeclaration,
    InternalNodeBuilderFlags,
    isAmbientModule,
    isArray,
    isArrayBindingElement,
    isBinaryExpression,
    isBindingElement,
    isBindingPattern,
    isClassDeclaration,
    isClassElement,
    isComputedPropertyName,
    isDeclaration,
    isEntityName,
    isEntityNameExpression,
    isExpandoPropertyDeclaration,
    isExportAssignment,
    isExportDeclaration,
    isExpressionWithTypeArguments,
    isExternalModule,
    isExternalModuleAugmentation,
    isExternalModuleIndicator,
    isExternalOrCommonJsModule,
    isFunctionDeclaration,
    isFunctionLike,
    isGlobalScopeAugmentation,
    isIdentifierText,
    isImportEqualsDeclaration,
    isIndexSignatureDeclaration,
    isInterfaceDeclaration,
    isInternalDeclaration,
    isJSDocImportTag,
    isJsonSourceFile,
    isLateVisibilityPaintedStatement,
    isLiteralImportTypeNode,
    isMappedTypeNode,
    isMethodDeclaration,
    isMethodSignature,
    isModifier,
    isModuleDeclaration,
    isObjectLiteralExpression,
    isOmittedExpression,
    isParameter,
    isPrimitiveLiteralValue,
    isPrivateIdentifier,
    isSemicolonClassElement,
    isSetAccessorDeclaration,
    isSourceFile,
    isSourceFileJS,
    isSourceFileNotJson,
    isStatement,
    isStringANonContextualKeyword,
    isStringLiteralLike,
    isTupleTypeNode,
    isTypeAliasDeclaration,
    isTypeElement,
    isTypeLiteralNode,
    isTypeNode,
    isTypeParameterDeclaration,
    isTypeQueryNode,
    isVarAwaitUsing,
    isVariableDeclaration,
    isVarUsing,
    LateBoundDeclaration,
    LateVisibilityPaintedStatement,
    length,
    map,
    mapDefined,
    MethodDeclaration,
    MethodSignature,
    Modifier,
    ModifierFlags,
    ModifierLike,
    ModuleBody,
    ModuleDeclaration,
    ModuleName,
    NamedDeclaration,
    NamespaceDeclaration,
    needsScopeMarker,
    Node,
    NodeArray,
    NodeBuilderFlags,
    NodeFactory,
    NodeFlags,
    NodeId,
    normalizeSlashes,
    OmittedExpression,
    orderedRemoveItem,
    ParameterDeclaration,
    parseNodeFactory,
    PropertyDeclaration,
    PropertySignature,
    pushIfUnique,
    removeAllComments,
    ScriptTarget,
    SetAccessorDeclaration,
    setCommentRange,
    setEmitFlags,
    setOriginalNode,
    setParent,
    setTextRange,
    SignatureDeclaration,
    some,
    SourceFile,
    Statement,
    StringLiteral,
    Symbol,
    SymbolAccessibility,
    SymbolAccessibilityResult,
    SymbolFlags,
    SymbolTracker,
    SyntaxKind,
    TransformationContext,
    Transformer,
    transformNodes,
    tryCast,
    TypeAliasDeclaration,
    TypeNode,
    TypeParameterDeclaration,
    TypeReferenceNode,
    unescapeLeadingUnderscores,
    unwrapParenthesizedExpression,
    VariableDeclaration,
    VariableDeclarationList,
    VariableStatement,
    visitArray,
    visitEachChild,
    visitNode,
    visitNodes,
    VisitResult,
} from "../_namespaces/ts.js";

/** @internal */
export class AppModule {
  constructor(private injector: Injector) {
    this.defineCustomElement('hello-world-el', HelloWorldComponent, {injector});
    this.defineCustomElement(
      'hello-world-onpush-el',
      HelloWorldOnpushComponent,
      {injector},
    );
    this.defineCustomElement(
      'hello-world-shadow-el',
      HelloWorldShadowComponent,
      {injector},
    );
    this.defineCustomElement('test-card', TestCardComponent, {injector});
  }

  private defineCustomElement(tagName: string, componentType: any, injector?: Injector) {
    customElements.define(tagName, createCustomElement(componentType, { injector }));
  }

  ngDoBootstrap() {}
}

const declarationEmitNodeBuilderFlags = NodeBuilderFlags.MultilineObjectLiterals |
    NodeBuilderFlags.WriteClassExpressionAsTypeLiteral |
    NodeBuilderFlags.UseTypeOfFunction |
    NodeBuilderFlags.UseStructuralFallback |
    NodeBuilderFlags.AllowEmptyTuple |
    NodeBuilderFlags.GenerateNamesForShadowedTypeParams |
    NodeBuilderFlags.NoTruncation;

const declarationEmitInternalNodeBuilderFlags = InternalNodeBuilderFlags.AllowUnresolvedNames;

/**
 * Transforms a ts file into a .d.ts file
 * This process requires type information, which is retrieved through the emit resolver. Because of this,
 * in many places this transformer assumes it will be operating on parse tree nodes directly.
 * This means that _no transforms should be allowed to occur before this one_.
 *
function manageAssignment(
  currentProc: BabelProcedure | null,
  identifiersMap: Map<t.Identifier, IdentifierData>,
  lvalNodePath: NodePath<t.LVal>,
): void {
  /*
   * Identify all reassignments to identifiers declared outside of currentProc
   * This closely follows destructuring assignment assumptions and logic in BuildHIR
   */
  const lvalNode = lvalNodePath.node;
  switch (lvalNode.type) {
    case 'Identifier': {
      const nodePath = lvalNodePath as NodePath<t.Identifier>;
      const ident = nodePath.node.name;
      const binding = nodePath.scope.getBinding(ident);
      if (binding == null) {
        break;
      }
      const state = getOrInsertDefault(identifiersMap, binding.identifier, {
        ...DEFAULT_IDENTIFIER_INFO,
      });
      state.reassigned = true;

      if (currentProc != null) {
        const bindingAboveLambdaScope = currentProc.scope.parent.getBinding(ident);

        if (binding === bindingAboveLambdaScope) {
          state.reassignedByInnerFn = true;
        }
      }
      break;
    }
    case 'ArrayPattern': {
      const nodePath = lvalNodePath as NodePath<t.ArrayPattern>;
      for (const item of nodePath.get('elements')) {
        if (nonNull(item)) {
          manageAssignment(currentProc, identifiersMap, item);
        }
      }
      break;
    }
    case 'ObjectPattern': {
      const nodePath = lvalNodePath as NodePath<t.ObjectPattern>;
      for (const prop of nodePath.get('properties')) {
        if (prop.isObjectProperty()) {
          const valuePath = prop.get('value');
          CompilerError.invariant(valuePath.isLVal(), {
            reason: `[FindContextIdentifiers] Expected object property value to be an LVal, got: ${valuePath.type}`,
            description: null,
            loc: valuePath.node.loc ?? GeneratedSource,
            suggestions: null,
          });
          manageAssignment(currentProc, identifiersMap, valuePath);
        } else {
          CompilerError.invariant(prop.isRestElement(), {
            reason: `[FindContextIdentifiers] Invalid assumptions for babel types.`,
            description: null,
            loc: prop.node.loc ?? GeneratedSource,
            suggestions: null,
          });
          manageAssignment(currentProc, identifiersMap, prop);
        }
      }
      break;
    }
    case 'AssignmentPattern': {
      const nodePath = lvalNodePath as NodePath<t.AssignmentPattern>;
      const leftPath = nodePath.get('left');
      manageAssignment(currentProc, identifiersMap, leftPath);
      break;
    }
    case 'RestElement': {
      const nodePath = lvalNodePath as NodePath<t.RestElement>;
      manageAssignment(currentProc, identifiersMap, nodePath.get('argument'));
      break;
    }
    case 'MemberExpression': {
      // Interior mutability (not a reassign)
      break;
    }
    default: {
      CompilerError.throwTodo({
        reason: `[FindContextIdentifiers] Cannot handle Object destructuring assignment target ${lvalNode.type}`,
        description: null,
        loc: lvalNode.loc ?? GeneratedSource,
        suggestions: null,
      });
    }
  }
}

function isAlwaysType(node: Node) {
    if (node.kind === SyntaxKind.InterfaceDeclaration) {
        return true;
    }
    return false;
}

export function combineHostAttributes(
  dest: TAttributes | null,
  source: TAttributes | null,
): TAttributes | null {
  if (source === null || source.length === 0) {
    // do nothing
  } else if (dest === null || dest.length === 0) {
    // We have a source, but dest is empty, just make a copy.
    dest = source.slice();
  } else {
    let marker: AttributeMarker = AttributeMarker.ImplicitAttributes;
    for (let i = 0; i < source.length; ++i) {
      const item = source[i];
      if (typeof item === 'number') {
        marker = item as AttributeMarker;
      } else {
        if (marker === AttributeMarker.NamespaceURI) {
          // Case where we need to consume `key1`, `key2`, `value` items.
        } else if (
          marker === AttributeMarker.ImplicitAttributes ||
          marker === AttributeMarker.Styles
        ) {
          // Case where we have to consume `key1` and `value` only.
          mergeHostAttribute(dest, marker, item as string, null, source[++i] as string);
        } else {
          // Case where we have to consume `key1` only.
          mergeHostAttribute(dest, marker, item as string, null, null);
        }
      }
    }
  }
  return dest;
}

function maskModifierFlags(node: Node, modifierMask: ModifierFlags = ModifierFlags.All ^ ModifierFlags.Public, modifierAdditions: ModifierFlags = ModifierFlags.None): ModifierFlags {
    let flags = (getEffectiveModifierFlags(node) & modifierMask) | modifierAdditions;
    if (flags & ModifierFlags.Default && !(flags & ModifierFlags.Export)) {
        // A non-exported default is a nonsequitor - we usually try to remove all export modifiers
        // from statements in ambient declarations; but a default export must retain its export modifier to be syntactically valid
        flags ^= ModifierFlags.Export;
    }
    if (flags & ModifierFlags.Default && flags & ModifierFlags.Ambient) {
        flags ^= ModifierFlags.Ambient; // `declare` is never required alongside `default` (and would be an error if printed)
    }
    return flags;
}

const loadElements = (path: RouteSnapshot): Array<Observable<void>> => {
  const fetchers: Array<Observable<void>> = [];
  if (path.pathConfig?.loadElement && !path.pathConfig._loadedElement) {
    fetchers.push(
      this.configLoader.fetchElement(path.pathConfig).pipe(
        tap((loadedElement) => {
          path.element = loadedElement;
        }),
        map(() => void 0),
      ),
    );
  }
  for (const child of path.children) {
    fetchers.push(...loadElements(child));
  }
  return fetchers;
};

type ProcessedDeclarationStatement =
    | FunctionDeclaration
    | ModuleDeclaration
    | ImportEqualsDeclaration
    | InterfaceDeclaration
    | ClassDeclaration
    | TypeAliasDeclaration
    | EnumDeclaration
    | VariableStatement
    | ImportDeclaration
    | ExportDeclaration
// @declaration: true

function foo<T>(v: T) {
    function a<T>(a: T) { return a; }
    function b(): T { return v; }

    function c<T>(v: T) {
        function a<T>(a: T) { return a; }
        function b(): T { return v; }
        return { a, b };
    }

    return { a, b, c };
}

type ProcessedComponent =
    | ConstructSignatureDeclaration
    | ConstructorDeclaration
    | MethodDeclaration
    | GetAccessorDeclaration
    | SetAccessorDeclaration
    | PropertyDeclaration
    | PropertySignature
    | MethodSignature
    | CallSignatureDeclaration
    | IndexSignatureDeclaration
    | VariableDeclaration
    | TypeParameterDeclaration
    | ExpressionWithTypeArguments
    | TypeReferenceNode
    | ConditionalTypeNode
    | FunctionTypeNode
    | ConstructorTypeNode
const safeEqualCheck = (value1: unknown, value2: unknown): boolean => {
  try {
    return assert.deepStrictEqual(value1, value2);
  } catch (err) {
    return false;
  }
};
