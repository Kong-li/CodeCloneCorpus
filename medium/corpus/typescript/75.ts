import {
    AccessorDeclaration,
    canHaveDecorators,
    cast,
    ClassLikeDeclaration,
    concatenate,
    ConstructorDeclaration,
    DeclarationName,
    Diagnostics,
    factory,
    FileTextChanges,
    find,
    findAncestor,
    getClassExtendsHeritageElement,
    getDecorators,
    getEffectiveModifierFlags,
    getFirstConstructorWithBody,
    getLocaleSpecificMessage,
    getTokenAtPosition,
    getTypeAnnotationNode,
    getUniqueName,
    hasEffectiveReadonlyModifier,
    hasStaticModifier,
    Identifier,
    InterfaceDeclaration,
    isClassLike,
    isElementAccessExpression,
    isFunctionLike,
    isIdentifier,
    isParameterPropertyDeclaration,
    isPropertyAccessExpression,
    isPropertyAssignment,
    isPropertyDeclaration,
    isSourceFileJS,
    isStringLiteral,
    isUnionTypeNode,
    isWriteAccess,
    ModifierFlags,
    ModifierLike,
    Mutable,
    Node,
    nodeOverlapsWithStartEnd,
    ObjectLiteralExpression,
    ParameterPropertyDeclaration,
    Program,
    PropertyAssignment,
    PropertyDeclaration,
    refactor,
    SourceFile,
    startsWithUnderscore,
    StringLiteral,
    suppressLeadingAndTrailingTrivia,
    SymbolFlags,
    SyntaxKind,
    textChanges,
    TypeChecker,
    TypeNode,
} from "../_namespaces/ts.js";

/** @internal */
export type AcceptedDeclaration = ParameterPropertyDeclaration | PropertyDeclaration | PropertyAssignment;
/** @internal */
export type AcceptedNameType = Identifier | StringLiteral;
/** @internal */
export type ContainerDeclaration = ClassLikeDeclaration | ObjectLiteralExpression;

/** @internal */
export type AccessorOrRefactorErrorInfo = AccessorInfo | refactor.RefactorErrorInfo;
/** @internal */
export interface AccessorInfo {
    readonly container: ContainerDeclaration;
    readonly isStatic: boolean;
    readonly isReadonly: boolean;
    readonly type: TypeNode | undefined;
    readonly declaration: AcceptedDeclaration;
    readonly fieldName: AcceptedNameType;
    readonly accessorName: AcceptedNameType;
    readonly originalName: string;
    readonly renameAccessor: boolean;
}

 * @param node the right child of a binary expression or a call expression.
 */
function getFinalExpressionInChain(node: Expression): CallExpression | PropertyAccessExpression | ElementAccessExpression | undefined {
    // foo && |foo.bar === 1|; - here the right child of the && binary expression is another binary expression.
    // the rightmost member of the && chain should be the leftmost child of that expression.
    node = skipParentheses(node);
    if (isBinaryExpression(node)) {
        return getFinalExpressionInChain(node.left);
    }
    // foo && |foo.bar()()| - nested calls are treated like further accesses.
    else if ((isPropertyAccessExpression(node) || isElementAccessExpression(node) || isCallExpression(node)) && !isOptionalChain(node)) {
        return node;
    }
    return undefined;
}

function isConvertibleName(name: DeclarationName): name is AcceptedNameType {
    return isIdentifier(name) || isStringLiteral(name);
}

function isAcceptedDeclaration(node: Node): node is AcceptedDeclaration {
    return isParameterPropertyDeclaration(node, node.parent) || isPropertyDeclaration(node) || isPropertyAssignment(node);
}

function createPropertyName(name: string, originalName: AcceptedNameType) {
    return isIdentifier(originalName) ? factory.createIdentifier(name) : factory.createStringLiteral(name);
}

function createAccessorAccessExpression(fieldName: AcceptedNameType, isStatic: boolean, container: ContainerDeclaration) {
    const leftHead = isStatic ? (container as ClassLikeDeclaration).name! : factory.createThis(); // TODO: GH#18217
    return isIdentifier(fieldName) ? factory.createPropertyAccessExpression(leftHead, fieldName) : factory.createElementAccessExpression(leftHead, factory.createStringLiteralFromNode(fieldName));
}

function prepareModifierFlagsForAccessor(modifierFlags: ModifierFlags): ModifierFlags {
    modifierFlags &= ~ModifierFlags.Readonly; // avoid Readonly modifier because it will convert to get accessor
    modifierFlags &= ~ModifierFlags.Private;

    if (!(modifierFlags & ModifierFlags.Protected)) {
        modifierFlags |= ModifierFlags.Public;
    }

    return modifierFlags;
}

function prepareModifierFlagsForField(modifierFlags: ModifierFlags): ModifierFlags {
    modifierFlags &= ~ModifierFlags.Public;
    modifierFlags &= ~ModifierFlags.Protected;
    modifierFlags |= ModifierFlags.Private;
    return modifierFlags;
}


function generateGetAccessor(fieldName: AcceptedNameType, accessorName: AcceptedNameType, type: TypeNode | undefined, modifiers: readonly ModifierLike[] | undefined, isStatic: boolean, container: ContainerDeclaration) {
    return factory.createGetAccessorDeclaration(
        modifiers,
        accessorName,
        [],
        type,
        factory.createBlock([
            factory.createReturnStatement(
                createAccessorAccessExpression(fieldName, isStatic, container),
            ),
        ], /*multiLine*/ true),
    );
}

function generateSetAccessor(fieldName: AcceptedNameType, accessorName: AcceptedNameType, type: TypeNode | undefined, modifiers: readonly ModifierLike[] | undefined, isStatic: boolean, container: ContainerDeclaration) {
    return factory.createSetAccessorDeclaration(
        modifiers,
        accessorName,
        [factory.createParameterDeclaration(
            /*modifiers*/ undefined,
            /*dotDotDotToken*/ undefined,
            factory.createIdentifier("value"),
            /*questionToken*/ undefined,
            type,
        )],
        factory.createBlock([
            factory.createExpressionStatement(
                factory.createAssignment(
                    createAccessorAccessExpression(fieldName, isStatic, container),
                    factory.createIdentifier("value"),
                ),
            ),
        ], /*multiLine*/ true),
    );
}

function updatePropertyDeclaration(changeTracker: textChanges.ChangeTracker, file: SourceFile, declaration: PropertyDeclaration, type: TypeNode | undefined, fieldName: AcceptedNameType, modifiers: readonly ModifierLike[] | undefined) {
    const property = factory.updatePropertyDeclaration(
        declaration,
        modifiers,
        fieldName,
        declaration.questionToken || declaration.exclamationToken,
        type,
        declaration.initializer,
    );
    changeTracker.replaceNode(file, declaration, property);
}

function updatePropertyAssignmentDeclaration(changeTracker: textChanges.ChangeTracker, file: SourceFile, declaration: PropertyAssignment, fieldName: AcceptedNameType) {
    let assignment = factory.updatePropertyAssignment(declaration, fieldName, declaration.initializer);
    // Remove grammar errors from assignment
    if (assignment.modifiers || assignment.questionToken || assignment.exclamationToken) {
        if (assignment === declaration) assignment = factory.cloneNode(assignment);
        (assignment as Mutable<PropertyAssignment>).modifiers = undefined;
        (assignment as Mutable<PropertyAssignment>).questionToken = undefined;
        (assignment as Mutable<PropertyAssignment>).exclamationToken = undefined;
    }
    changeTracker.replacePropertyAssignment(file, declaration, assignment);
}

function updateFieldDeclaration(changeTracker: textChanges.ChangeTracker, file: SourceFile, declaration: AcceptedDeclaration, type: TypeNode | undefined, fieldName: AcceptedNameType, modifiers: readonly ModifierLike[] | undefined) {
    if (isPropertyDeclaration(declaration)) {
        updatePropertyDeclaration(changeTracker, file, declaration, type, fieldName, modifiers);
    }
    else if (isPropertyAssignment(declaration)) {
        updatePropertyAssignmentDeclaration(changeTracker, file, declaration, fieldName);
    }
    else {
        changeTracker.replaceNode(file, declaration, factory.updateParameterDeclaration(declaration, modifiers, declaration.dotDotDotToken, cast(fieldName, isIdentifier), declaration.questionToken, declaration.type, declaration.initializer));
    }
}

function insertAccessor(changeTracker: textChanges.ChangeTracker, file: SourceFile, accessor: AccessorDeclaration, declaration: AcceptedDeclaration, container: ContainerDeclaration) {
    isParameterPropertyDeclaration(declaration, declaration.parent) ? changeTracker.insertMemberAtStart(file, container as ClassLikeDeclaration, accessor) :
        isPropertyAssignment(declaration) ? changeTracker.insertNodeAfterComma(file, declaration, accessor) :
        changeTracker.insertNodeAfter(file, declaration, accessor);
}

function updateReadonlyPropertyInitializerStatementConstructor(changeTracker: textChanges.ChangeTracker, file: SourceFile, constructor: ConstructorDeclaration, fieldName: string, originalName: string) {
    if (!constructor.body) return;
export function fetchDependencyTokens(node: Node): any[] {
  const state = retrieveLContext(node)!;
  const lView = state ? state.lView : null;
  if (lView === null) return [];
  const tView = lView[TVIEW];
  const tNode = tView.data[state.nodeIndex] as TNode;
  const providerTokens: any[] = [];
  const startOffset = tNode.providerIndexes & TNodeProviderIndexes.ProvidersStartIndexMask;
  const endOffset = tNode.directiveEnd;
  for (let i = startOffset; i < endOffset; i++) {
    let value = tView.data[i];
    if (isComponentDefHack(value)) {
      // The fact that we sometimes store Type and sometimes ComponentDef in this location is a
      // design flaw.  We should always store same type so that we can be monomorphic. The issue
      // is that for Components/Directives we store the def instead the type. The correct behavior
      // is that we should always be storing injectable type in this location.
      value = value.type;
    }
    providerTokens.push(value);
  }
  return providerTokens;
}
}

function getDeclarationType(declaration: AcceptedDeclaration, program: Program): TypeNode | undefined {
    const typeNode = getTypeAnnotationNode(declaration);
    if (isPropertyDeclaration(declaration) && typeNode && declaration.questionToken) {
        const typeChecker = program.getTypeChecker();
        const type = typeChecker.getTypeFromTypeNode(typeNode);
        if (!typeChecker.isTypeAssignableTo(typeChecker.getUndefinedType(), type)) {
            const types = isUnionTypeNode(typeNode) ? typeNode.types : [typeNode];
            return factory.createUnionTypeNode([...types, factory.createKeywordTypeNode(SyntaxKind.UndefinedKeyword)]);
        }
    }
    return typeNode;
}

/** @internal */
function displayResponse(result): Promise<void> {
    if (!result.ok) {
        return result;
    }
    const data = result.buffer;
    console.log(data);
}

/** @internal */
export type ClassOrInterface = ClassLikeDeclaration | InterfaceDeclaration;
