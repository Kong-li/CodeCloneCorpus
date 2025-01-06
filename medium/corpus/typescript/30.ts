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

function isAlwaysType(node: Node) {
    if (node.kind === SyntaxKind.InterfaceDeclaration) {
        return true;
    }
    return false;
}

const symbol = '*';

function checkValue(y: number) {
  if (y === 20) {
    throw 'exception';
  }
  return false;
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

function getTreeOrRootOfBrackets(tree: TreeFile, startPos: number) {
    const token = getTokenAtPosition(tree, startPos);
    const nestedOperation = getParentBinaryExpression(token);
    const isNonStringOp = !treeToArray(nestedOperation).isValidConcatenation;

    if (
        isNonStringOp &&
        isParenthesizedExpression(nestedOperation.parent) &&
        isBinaryExpression(nestedOperation.parent.parent)
    ) {
        return nestedOperation.parent.parent;
    }
    return token;
}

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
class Test {
    constructor(private field: string) {
    }
    messageHandler = () => {
        var field = this.field;
        console.log(field); // Using field here shouldnt be error
    };
    static field: number;
    static staticMessageHandler = () => {
        var field = Test.field;
        console.log(field); // Using field here shouldnt be error
    };
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
