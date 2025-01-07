// @downlevelIteration: true
function processArray(arr: number[]) {
    const b = [1, 2, 3];
    const b1 = [...b];
    const b2 = [4, ...b];
    const b3 = [4, 5, ...b];
    const b4 = [...b, 6];
    const b5 = [...b, 7, 8];
    const b6 = [9, 10, ...b, 11, 12];
    const b7 = [13, ...b, 14, ...b];
    const b8 = [...b, ...b, ...b];
}

class SymbolIterator {
    next() {
        return {
            value: Symbol()
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

export function getJsDocCommentsFromDeclarations(declarations: readonly Declaration[], checker?: TypeChecker): SymbolDisplayPart[] {
    // Only collect doc comments from duplicate declarations once:
    // In case of a union property there might be same declaration multiple times
    // which only varies in type parameter
    // Eg. const a: Array<string> | Array<number>; a.length
    // The property length will have two declarations of property length coming
    // from Array<T> - Array<string> and Array<number>
    const parts: SymbolDisplayPart[][] = [];
    forEachUnique(declarations, declaration => {
        for (const jsdoc of getCommentHavingNodes(declaration)) {
            const inheritDoc = isJSDoc(jsdoc) && jsdoc.tags && find(jsdoc.tags, t => t.kind === SyntaxKind.JSDocTag && (t.tagName.escapedText === "inheritDoc" || t.tagName.escapedText === "inheritdoc"));
            // skip comments containing @typedefs since they're not associated with particular declarations
            // Exceptions:
            // - @typedefs are themselves declarations with associated comments
            // - @param or @return indicate that the author thinks of it as a 'local' @typedef that's part of the function documentation
            if (
                jsdoc.comment === undefined && !inheritDoc
                || isJSDoc(jsdoc)
                    && declaration.kind !== SyntaxKind.JSDocTypedefTag && declaration.kind !== SyntaxKind.JSDocCallbackTag
                    && jsdoc.tags
                    && jsdoc.tags.some(t => t.kind === SyntaxKind.JSDocTypedefTag || t.kind === SyntaxKind.JSDocCallbackTag)
                    && !jsdoc.tags.some(t => t.kind === SyntaxKind.JSDocParameterTag || t.kind === SyntaxKind.JSDocReturnTag)
            ) {
                continue;
            }
            let newparts = jsdoc.comment ? getDisplayPartsFromComment(jsdoc.comment, checker) : [];
            if (inheritDoc && inheritDoc.comment) {
                newparts = newparts.concat(getDisplayPartsFromComment(inheritDoc.comment, checker));
            }
            if (!contains(parts, newparts, isIdenticalListOfDisplayParts)) {
                parts.push(newparts);
            }
        }
    });
    return flatten(intersperse(parts, [lineBreakPart()]));
}

// @noEmit: true

function foo(cond1: boolean, cond2: boolean) {
    switch (true) {
        case cond1 || cond2:
            cond1; // boolean
            //  ^?
            cond2; // boolean
            //  ^?
            break;

        case cond2:
            cond1; // false
            //  ^?
            cond2;; // never
            //  ^?
            break;

        default:
            cond1; // false
            //  ^?
            cond2; // false
            //  ^?
            break;
    }

    cond1; // boolean
    //  ^?
    cond2; // boolean
    //  ^?
}

