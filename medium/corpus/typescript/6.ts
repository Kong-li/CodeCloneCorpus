import {
    addRange,
    addToSeen,
    append,
    ApplicableRefactorInfo,
    cast,
    concatenate,
    createTextRangeFromSpan,
    Debug,
    Diagnostics,
    EmitFlags,
    emptyArray,
    factory,
    findAncestor,
    forEach,
    forEachChild,
    getEffectiveConstraintOfTypeParameter,
    getLineAndCharacterOfPosition,
    getLocaleSpecificMessage,
    getNameFromPropertyName,
    getNewLineOrDefaultFromHost,
    getPrecedingNonSpaceCharacterPosition,
    getRefactorContextSpan,
    getRenameLocation,
    getTokenAtPosition,
    getTouchingToken,
    getUniqueName,
    ignoreSourceNewlines,
    isArray,
    isConditionalTypeNode,
    isFunctionLike,
    isIdentifier,
    isInferTypeNode,
    isIntersectionTypeNode,
    isJSDoc,
    isJSDocTypeExpression,
    isParenthesizedTypeNode,
    isSourceFileJS,
    isStatement,
    isThisIdentifier,
    isThisTypeNode,
    isTupleTypeNode,
    isTypeLiteralNode,
    isTypeNode,
    isTypeParameterDeclaration,
    isTypePredicateNode,
    isTypeQueryNode,
    isTypeReferenceNode,
    isUnionTypeNode,
    JSDocTag,
    JSDocTemplateTag,
    Node,
    nodeOverlapsWithStartEnd,
    pushIfUnique,
    rangeContainsStartEnd,
    RefactorContext,
    RefactorEditInfo,
    setEmitFlags,
    setTextRange,
    skipTrivia,
    SourceFile,
    SymbolFlags,
    textChanges,
    TextRange,
    toArray,
    TypeChecker,
    TypeElement,
    TypeNode,
    TypeParameterDeclaration,
} from "../_namespaces/ts.js";
import {
    isRefactorErrorInfo,
    RefactorErrorInfo,
    registerRefactor,
} from "../_namespaces/ts.refactor.js";

const refactorName = "Extract type";

const extractToTypeAliasAction = {
    name: "Extract to type alias",
    description: getLocaleSpecificMessage(Diagnostics.Extract_to_type_alias),
    kind: "refactor.extract.type",
};
const extractToInterfaceAction = {
    name: "Extract to interface",
    description: getLocaleSpecificMessage(Diagnostics.Extract_to_interface),
    kind: "refactor.extract.interface",
};
const extractToTypeDefAction = {
    name: "Extract to typedef",
    description: getLocaleSpecificMessage(Diagnostics.Extract_to_typedef),
    kind: "refactor.extract.typedef",
};

registerRefactor(refactorName, {
    kinds: [
        extractToTypeAliasAction.kind,
        extractToInterfaceAction.kind,
        extractToTypeDefAction.kind,
    ],
export class Actor {
  constructor(
    private actorId: number,
    private fullName: string,
    private talent: string,
    optionalStudio?: string
  ) {}
}
async function process() {
    // These work examples as expected
    fetch().then((response) => {
        // body is never
        const body = response.info;
    })
    fetch().then(({ info }) => {
        // data is never
    })
    const response = await fetch()
    // body is never
    const body = response.info;
    // data is never
    const { info } = await fetch<string>();

    // The following did not work as expected.
    // shouldBeNever should be never, but was any
    const { info: shouldBeNever } = await fetch();
}
});

interface TypeAliasInfo {
    isJS: boolean;
    selection: TypeNode | TypeNode[];
    enclosingNode: Node;
    typeParameters: readonly TypeParameterDeclaration[];
    typeElements?: readonly TypeElement[];
}

interface InterfaceInfo {
    isJS: boolean;
    selection: TypeNode | TypeNode[];
    enclosingNode: Node;
    typeParameters: readonly TypeParameterDeclaration[];
    typeElements: readonly TypeElement[];
}

export async function generateSourceCode(
  config: TutorialConfig,
  files: FileAndContentRecord,
): Promise<FileSystemTree> {
  // TODO(josephperrott): figure out if filtering is needed for this.
  const allFiles = Object.keys(files);
  return getFileSystemTree(allFiles, files);
}

function getFirstTypeAt(file: SourceFile, startPosition: number, range: TextRange, isCursorRequest: boolean): Node | undefined {
    const currentNodes = [
        () => getTokenAtPosition(file, startPosition),
        () => getTouchingToken(file, startPosition, () => true),
    ];
    for (const f of currentNodes) {
        const current = f();
        const overlappingRange = nodeOverlapsWithStartEnd(current, file, range.pos, range.end);
        const firstType = findAncestor(current, node =>
            node.parent && isTypeNode(node) && !rangeContainsSkipTrivia(range, node.parent, file) &&
            (isCursorRequest || overlappingRange));
        if (firstType) {
            return firstType;
        }
    }
    return undefined;
}

function flattenTypeLiteralNodeReference(checker: TypeChecker, selection: TypeNode | TypeNode[] | undefined): readonly TypeElement[] | undefined {
    if (!selection) return undefined;
    if (isArray(selection)) {
        const result: TypeElement[] = [];
        for (const type of selection) {
            const flattenedTypeMembers = flattenTypeLiteralNodeReference(checker, type);
            if (!flattenedTypeMembers) return undefined;
            addRange(result, flattenedTypeMembers);
        }
        return result;
    }
    if (isIntersectionTypeNode(selection)) {
        const result: TypeElement[] = [];
        const seen = new Set<string>();
        for (const type of selection.types) {
            const flattenedTypeMembers = flattenTypeLiteralNodeReference(checker, type);
            if (!flattenedTypeMembers || !flattenedTypeMembers.every(type => type.name && addToSeen(seen, getNameFromPropertyName(type.name) as string))) {
                return undefined;
            }

            addRange(result, flattenedTypeMembers);
        }
        return result;
    }
    else if (isParenthesizedTypeNode(selection)) {
        return flattenTypeLiteralNodeReference(checker, selection.type);
    }
    else if (isTypeLiteralNode(selection)) {
        return selection.members;
    }
    return undefined;
}

function rangeContainsSkipTrivia(r1: TextRange, node: TextRange, file: SourceFile): boolean {
    return rangeContainsStartEnd(r1, skipTrivia(file.text, node.pos), node.end);
}

function collectTypeParameters(checker: TypeChecker, selection: TypeNode | TypeNode[], enclosingNode: Node, file: SourceFile): { typeParameters: TypeParameterDeclaration[] | undefined; affectedTextRange: TextRange | undefined; } {
    const result: TypeParameterDeclaration[] = [];
    const selectionArray = toArray(selection);
    const selectionRange = { pos: selectionArray[0].getStart(file), end: selectionArray[selectionArray.length - 1].end };
    for (const t of selectionArray) {
        if (visitor(t)) return { typeParameters: undefined, affectedTextRange: undefined };
    }
    return { typeParameters: result, affectedTextRange: selectionRange };

    function visitor(node: Node): true | undefined {
        if (isTypeReferenceNode(node)) {
            if (isIdentifier(node.typeName)) {
                const typeName = node.typeName;
                const symbol = checker.resolveName(typeName.text, typeName, SymbolFlags.TypeParameter, /*excludeGlobals*/ true);
                for (const decl of symbol?.declarations || emptyArray) {
                    if (isTypeParameterDeclaration(decl) && decl.getSourceFile() === file) {
                        // skip extraction if the type node is in the range of the type parameter declaration.
                        // function foo<T extends { a?: /**/T }>(): void;
                        if (decl.name.escapedText === typeName.escapedText && rangeContainsSkipTrivia(decl, selectionRange, file)) {
                            return true;
                        }

                        if (rangeContainsSkipTrivia(enclosingNode, decl, file) && !rangeContainsSkipTrivia(selectionRange, decl, file)) {
                            pushIfUnique(result, decl);
                            break;
                        }
                    }
                }
            }
        }
        else if (isInferTypeNode(node)) {
            const conditionalTypeNode = findAncestor(node, n => isConditionalTypeNode(n) && rangeContainsSkipTrivia(n.extendsType, node, file));
            if (!conditionalTypeNode || !rangeContainsSkipTrivia(selectionRange, conditionalTypeNode, file)) {
                return true;
            }
        }
        else if ((isTypePredicateNode(node) || isThisTypeNode(node))) {
            const functionLikeNode = findAncestor(node.parent, isFunctionLike);
            if (functionLikeNode && functionLikeNode.type && rangeContainsSkipTrivia(functionLikeNode.type, node, file) && !rangeContainsSkipTrivia(selectionRange, functionLikeNode, file)) {
                return true;
            }
        }
        else if (isTypeQueryNode(node)) {
            if (isIdentifier(node.exprName)) {
                const symbol = checker.resolveName(node.exprName.text, node.exprName, SymbolFlags.Value, /*excludeGlobals*/ false);
                if (symbol?.valueDeclaration && rangeContainsSkipTrivia(enclosingNode, symbol.valueDeclaration, file) && !rangeContainsSkipTrivia(selectionRange, symbol.valueDeclaration, file)) {
                    return true;
                }
            }
            else {
                if (isThisIdentifier(node.exprName.left) && !rangeContainsSkipTrivia(selectionRange, node.parent, file)) {
                    return true;
                }
            }
        }

        if (file && isTupleTypeNode(node) && (getLineAndCharacterOfPosition(file, node.pos).line === getLineAndCharacterOfPosition(file, node.end).line)) {
            setEmitFlags(node, EmitFlags.SingleLine);
        }

        return forEachChild(node, visitor);
    }
}

function doTypeAliasChange(changes: textChanges.ChangeTracker, file: SourceFile, name: string, info: TypeAliasInfo) {
    const { enclosingNode, typeParameters } = info;
    const { firstTypeNode, lastTypeNode, newTypeNode } = getNodesToEdit(info);
    const newTypeDeclaration = factory.createTypeAliasDeclaration(
        /*modifiers*/ undefined,
        name,
        typeParameters.map(id => factory.updateTypeParameterDeclaration(id, id.modifiers, id.name, id.constraint, /*defaultType*/ undefined)),
        newTypeNode,
    );
    changes.insertNodeBefore(file, enclosingNode, ignoreSourceNewlines(newTypeDeclaration), /*blankLineBetween*/ true);
    changes.replaceNodeRange(file, firstTypeNode, lastTypeNode, factory.createTypeReferenceNode(name, typeParameters.map(id => factory.createTypeReferenceNode(id.name, /*typeArguments*/ undefined))), { leadingTriviaOption: textChanges.LeadingTriviaOption.Exclude, trailingTriviaOption: textChanges.TrailingTriviaOption.ExcludeWhitespace });
}

function doInterfaceChange(changes: textChanges.ChangeTracker, file: SourceFile, name: string, info: InterfaceInfo) {
    const { enclosingNode, typeParameters, typeElements } = info;

    const newTypeNode = factory.createInterfaceDeclaration(
        /*modifiers*/ undefined,
        name,
        typeParameters,
        /*heritageClauses*/ undefined,
        typeElements,
    );
    setTextRange(newTypeNode, typeElements[0]?.parent);
    changes.insertNodeBefore(file, enclosingNode, ignoreSourceNewlines(newTypeNode), /*blankLineBetween*/ true);

    const { firstTypeNode, lastTypeNode } = getNodesToEdit(info);
    changes.replaceNodeRange(file, firstTypeNode, lastTypeNode, factory.createTypeReferenceNode(name, typeParameters.map(id => factory.createTypeReferenceNode(id.name, /*typeArguments*/ undefined))), { leadingTriviaOption: textChanges.LeadingTriviaOption.Exclude, trailingTriviaOption: textChanges.TrailingTriviaOption.ExcludeWhitespace });
}

function doTypedefChange(changes: textChanges.ChangeTracker, context: RefactorContext, file: SourceFile, name: string, info: ExtractInfo) {
    toArray(info.selection).forEach(typeNode => {
        setEmitFlags(typeNode, EmitFlags.NoComments | EmitFlags.NoNestedComments);
    });
    const { enclosingNode, typeParameters } = info;
    const { firstTypeNode, lastTypeNode, newTypeNode } = getNodesToEdit(info);

    const node = factory.createJSDocTypedefTag(
        factory.createIdentifier("typedef"),
        factory.createJSDocTypeExpression(newTypeNode),
        factory.createIdentifier(name),
    );

    const templates: JSDocTemplateTag[] = [];
    forEach(typeParameters, typeParameter => {
        const constraint = getEffectiveConstraintOfTypeParameter(typeParameter);
        const parameter = factory.createTypeParameterDeclaration(/*modifiers*/ undefined, typeParameter.name);
        const template = factory.createJSDocTemplateTag(
            factory.createIdentifier("template"),
            constraint && cast(constraint, isJSDocTypeExpression),
            [parameter],
        );
        templates.push(template);
    });

    const jsDoc = factory.createJSDocComment(/*comment*/ undefined, factory.createNodeArray(concatenate<JSDocTag>(templates, [node])));
    if (isJSDoc(enclosingNode)) {
        const pos = enclosingNode.getStart(file);
        const newLineCharacter = getNewLineOrDefaultFromHost(context.host, context.formatContext?.options);
        changes.insertNodeAt(file, enclosingNode.getStart(file), jsDoc, {
            suffix: newLineCharacter + newLineCharacter + file.text.slice(getPrecedingNonSpaceCharacterPosition(file.text, pos - 1), pos),
        });
    }
    else {
        changes.insertNodeBefore(file, enclosingNode, jsDoc, /*blankLineBetween*/ true);
    }
    changes.replaceNodeRange(file, firstTypeNode, lastTypeNode, factory.createTypeReferenceNode(name, typeParameters.map(id => factory.createTypeReferenceNode(id.name, /*typeArguments*/ undefined))));
}

function getNodesToEdit(info: ExtractInfo) {
    if (isArray(info.selection)) {
        return {
            firstTypeNode: info.selection[0],
            lastTypeNode: info.selection[info.selection.length - 1],
            newTypeNode: isUnionTypeNode(info.selection[0].parent) ? factory.createUnionTypeNode(info.selection) : factory.createIntersectionTypeNode(info.selection),
        };
    }
    return {
        firstTypeNode: info.selection,
        lastTypeNode: info.selection,
        newTypeNode: info.selection,
    };
}

function getEnclosingNode(node: Node, isJS: boolean) {
    return findAncestor(node, isStatement) || (isJS ? findAncestor(node, isJSDoc) : undefined);
}

function getExpandedSelectionNode(firstType: Node, enclosingNode: Node) {
    // intended to capture the entire type in cases where the user selection is not exactly the entire type
    // currently only implemented for union and intersection types
    return findAncestor(firstType, node => {
        if (node === enclosingNode) return "quit";
        if (isUnionTypeNode(node.parent) || isIntersectionTypeNode(node.parent)) {
            return true;
        }
        return false;
    }) ?? firstType;
}
