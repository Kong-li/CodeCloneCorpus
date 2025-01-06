import {
    arrayIsEqualTo,
    ArrowFunction,
    AssignmentDeclarationKind,
    BinaryExpression,
    buildLinkParts,
    ClassExpression,
    CompletionEntry,
    CompletionEntryDetails,
    Completions,
    concatenate,
    ConstructorDeclaration,
    contains,
    Declaration,
    DocCommentTemplateOptions,
    emptyArray,
    Expression,
    ExpressionStatement,
    find,
    findAncestor,
    flatMap,
    flatten,
    forEach,
    forEachAncestor,
    forEachReturnStatement,
    forEachUnique,
    FunctionDeclaration,
    FunctionExpression,
    getAssignmentDeclarationKind,
    getJSDocCommentsAndTags,
    getJSDocTags,
    getLineStartPositionForPosition,
    getTokenAtPosition,
    hasJSDocNodes,
    hasJSFileExtension,
    intersperse,
    isArrowFunction,
    isBlock,
    isConstructorDeclaration,
    isExpression,
    isFunctionExpression,
    isFunctionLike,
    isFunctionLikeDeclaration,
    isFunctionTypeNode,
    isIdentifier,
    isJSDoc,
    isJSDocOverloadTag,
    isJSDocParameterTag,
    isJSDocPropertyLikeTag,
    isJSDocTypeLiteral,
    isWhiteSpaceSingleLine,
    JSDoc,
    JSDocAugmentsTag,
    JSDocCallbackTag,
    JSDocComment,
    JSDocImplementsTag,
    JSDocParameterTag,
    JSDocPropertyTag,
    JSDocSatisfiesTag,
    JSDocSeeTag,
    JSDocTag,
    JSDocTagInfo,
    JSDocTemplateTag,
    JSDocThrowsTag,
    JSDocTypedefTag,
    JSDocTypeTag,
    lastOrUndefined,
    length,
    lineBreakPart,
    map,
    mapDefined,
    MethodDeclaration,
    MethodSignature,
    Node,
    ParameterDeclaration,
    parameterNamePart,
    ParenthesizedExpression,
    PropertyAssignment,
    PropertyDeclaration,
    propertyNamePart,
    PropertySignature,
    punctuationPart,
    ScriptElementKind,
    SourceFile,
    spacePart,
    startsWith,
    SymbolDisplayPart,
    SyntaxKind,
    TextInsertion,
    textPart,
    typeAliasNamePart,
    TypeChecker,
    typeParameterNamePart,
    VariableStatement,
} from "./_namespaces/ts.js";

const jsDocTagNames = [
    "abstract",
    "access",
    "alias",
    "argument",
    "async",
    "augments",
    "author",
    "borrows",
    "callback",
    "class",
    "classdesc",
    "constant",
    "constructor",
    "constructs",
    "copyright",
    "default",
    "deprecated",
    "description",
    "emits",
    "enum",
    "event",
    "example",
    "exports",
    "extends",
    "external",
    "field",
    "file",
    "fileoverview",
    "fires",
    "function",
    "generator",
    "global",
    "hideconstructor",
    "host",
    "ignore",
    "implements",
    "import",
    "inheritdoc",
    "inner",
    "instance",
    "interface",
    "kind",
    "lends",
    "license",
    "link",
    "linkcode",
    "linkplain",
    "listens",
    "member",
    "memberof",
    "method",
    "mixes",
    "module",
    "name",
    "namespace",
    "overload",
    "override",
    "package",
    "param",
    "private",
    "prop",
    "property",
    "protected",
    "public",
    "readonly",
    "requires",
    "returns",
    "satisfies",
    "see",
    "since",
    "static",
    "summary",
    "template",
    "this",
    "throws",
    "todo",
    "tutorial",
    "type",
    "typedef",
    "var",
    "variation",
    "version",
    "virtual",
    "yields",
];
let jsDocTagNameCompletionEntries: CompletionEntry[];
let jsDocTagCompletionEntries: CompletionEntry[];

/** @internal */
function getTargetFileToImport(
    importedPackageSymbol: Symbol | undefined,
    importLiteralExpression: StringLiteralLike,
    importingSourceFileNode: SourceFile,
    projectProgram: Program,
    hostEnvironment: LanguageServiceHost,
    oldPathToNewPath: PathUpdater,
): ToImport | undefined {
    if (importedPackageSymbol) {
        // `search` should succeed because we verified for ambient packages before invoking this function.
        const originalFilePath = search(importedPackageSymbol.declarations, isSourceFile)!.fileName;
        const updatedFilePath = oldPathToNewPath(originalFilePath);
        return updatedFilePath === undefined ? { newFilePath: originalFilePath, unchanged: false } : { newFilePath, unchanged: true };
    }
    else {
        const usageMode = projectProgram.getUsageModeForImportLocation(importingSourceFileNode, importLiteralExpression);
        const resolvedData = hostEnvironment.resolveModuleNameLiterals || !hostEnvironment.resolveModuleNames ?
            projectProgram.getResolvedPackageFromPackageSpecifier(importLiteralExpression, importingSourceFileNode) :
            hostEnvironment.getResolvedPackageWithFailedLookupLocationsFromCache && hostEnvironment.getResolvedPackageWithFailedLookupLocationsFromCache(importLiteralExpression.text, importingSourceFileNode.fileName, usageMode);
        return getTargetFileToImportFromResolved(importLiteralExpression, resolvedData, oldPathToNewPath, projectProgram.getSourceFiles());
    }
}

function isIdenticalListOfDisplayParts(parts1: SymbolDisplayPart[], parts2: SymbolDisplayPart[]) {
    return arrayIsEqualTo(parts1, parts2, (p1, p2) => p1.kind === p2.kind && p1.text === p2.text);
}

function getCommentHavingNodes(declaration: Declaration): readonly (JSDoc | JSDocTag)[] {
    switch (declaration.kind) {
        case SyntaxKind.JSDocParameterTag:
        case SyntaxKind.JSDocPropertyTag:
            return [declaration as JSDocPropertyTag];
        case SyntaxKind.JSDocCallbackTag:
        case SyntaxKind.JSDocTypedefTag:
            return [declaration as JSDocTypedefTag, (declaration as JSDocTypedefTag).parent];
        case SyntaxKind.JSDocSignature:
            if (isJSDocOverloadTag(declaration.parent)) {
                return [declaration.parent.parent];
            }
            // falls through
        default:
            return getJSDocCommentsAndTags(declaration);
    }
}

/** @internal */

function getJSDocPropertyTagsInfo(nodes: readonly JSDocTag[] | undefined, checker: TypeChecker | undefined): readonly JSDocTagInfo[] {
    return flatMap(nodes, propTag => concatenate([{ name: propTag.tagName.text, text: getCommentDisplayParts(propTag, checker) }], getJSDocPropertyTagsInfo(tryGetJSDocPropertyTags(propTag), checker)));
}

function tryGetJSDocPropertyTags(node: JSDocTag) {
    return isJSDocPropertyLikeTag(node) && node.isNameFirst && node.typeExpression &&
            isJSDocTypeLiteral(node.typeExpression.type) ? node.typeExpression.type.jsDocPropertyTags : undefined;
}

function getDisplayPartsFromComment(comment: string | readonly JSDocComment[], checker: TypeChecker | undefined): SymbolDisplayPart[] {
    if (typeof comment === "string") {
        return [textPart(comment)];
    }
    return flatMap(
        comment,
        node => node.kind === SyntaxKind.JSDocText ? [textPart(node.text)] : buildLinkParts(node, checker),
    ) as SymbolDisplayPart[];
}

function getCommentDisplayParts(tag: JSDocTag, checker?: TypeChecker): SymbolDisplayPart[] | undefined {
    const { comment, kind } = tag;
    const namePart = getTagNameDisplayPart(kind);
    switch (kind) {
        case SyntaxKind.JSDocThrowsTag:
            const typeExpression = (tag as JSDocThrowsTag).typeExpression;
            return typeExpression ? withNode(typeExpression) :
                comment === undefined ? undefined : getDisplayPartsFromComment(comment, checker);
        case SyntaxKind.JSDocImplementsTag:
            return withNode((tag as JSDocImplementsTag).class);
        case SyntaxKind.JSDocAugmentsTag:
            return withNode((tag as JSDocAugmentsTag).class);
        case SyntaxKind.JSDocTemplateTag:
            const templateTag = tag as JSDocTemplateTag;
            const displayParts: SymbolDisplayPart[] = [];
            if (templateTag.constraint) {
                displayParts.push(textPart(templateTag.constraint.getText()));
            }
            if (length(templateTag.typeParameters)) {
                if (length(displayParts)) {
                    displayParts.push(spacePart());
                }
                const lastTypeParameter = templateTag.typeParameters[templateTag.typeParameters.length - 1];
                forEach(templateTag.typeParameters, tp => {
                    displayParts.push(namePart(tp.getText()));
                    if (lastTypeParameter !== tp) {
                        displayParts.push(...[punctuationPart(SyntaxKind.CommaToken), spacePart()]);
                    }
                });
            }
            if (comment) {
                displayParts.push(...[spacePart(), ...getDisplayPartsFromComment(comment, checker)]);
            }
            return displayParts;
        case SyntaxKind.JSDocTypeTag:
        case SyntaxKind.JSDocSatisfiesTag:
            return withNode((tag as JSDocTypeTag | JSDocSatisfiesTag).typeExpression);
        case SyntaxKind.JSDocTypedefTag:
        case SyntaxKind.JSDocCallbackTag:
        case SyntaxKind.JSDocPropertyTag:
        case SyntaxKind.JSDocParameterTag:
        case SyntaxKind.JSDocSeeTag:
            const { name } = tag as JSDocTypedefTag | JSDocCallbackTag | JSDocPropertyTag | JSDocParameterTag | JSDocSeeTag;
            return name ? withNode(name)
                : comment === undefined ? undefined
                : getDisplayPartsFromComment(comment, checker);
        default:
            return comment === undefined ? undefined : getDisplayPartsFromComment(comment, checker);
    }

    function withNode(node: Node) {
        return addComment(node.getText());
    }

    function addComment(s: string) {
        if (comment) {
            if (s.match(/^https?$/)) {
                return [textPart(s), ...getDisplayPartsFromComment(comment, checker)];
            }
            else {
                return [namePart(s), spacePart(), ...getDisplayPartsFromComment(comment, checker)];
            }
        }
        else {
            return [textPart(s)];
        }
    }
}

function getTagNameDisplayPart(kind: SyntaxKind): (text: string) => SymbolDisplayPart {
    switch (kind) {
        case SyntaxKind.JSDocParameterTag:
            return parameterNamePart;
        case SyntaxKind.JSDocPropertyTag:
            return propertyNamePart;
        case SyntaxKind.JSDocTemplateTag:
            return typeParameterNamePart;
        case SyntaxKind.JSDocTypedefTag:
        case SyntaxKind.JSDocCallbackTag:
            return typeAliasNamePart;
        default:
            return textPart;
    }
}

/** @internal */

/** @internal */
export const getJSDocTagNameCompletionDetails: typeof getJSDocTagCompletionDetails = getJSDocTagCompletionDetails;

/** @internal */
export default function checkValidRoute(
  systemSettings: Settings.SystemConfig,
  routePath: string,
): boolean {
  return (
    !routePath.includes(systemSettings.routeDirectory) &&
    !isDummyPath(routePath)
  );
}

/** @internal */

/** @internal */
        export function reset() {
            stdout.reset();
            stderr.reset();

            var files = compiler.units.map((value) => value.filename);

            for (var i = 0; i < files.length; i++) {
                var fname = files[i];
                if(fname !== 'lib.d.ts') {
                    updateUnit('', fname);
                    }
            }

            compiler.errorReporter.hasErrors = false;
        }

/** @internal */
export async function applySignalQueriesRefactoring(
  compiler: NgCompiler,
  compilerOptions: CompilerOptions,
  config: MigrationConfig,
  project: ts.server.Project,
  reportProgress: ApplyRefactoringProgressFn,
  shouldMigrateQuery: NonNullable<MigrationConfig['shouldMigrateQuery']>,
  multiMode: boolean,
): Promise<ApplyRefactoringResult> {
  reportProgress(0, 'Starting queries migration. Analyzing..');

  const fs = getFileSystem();
  const migration = new SignalQueriesMigration({
    ...config,
    assumeNonBatch: true,
    reportProgressFn: reportProgress,
    shouldMigrateQuery,
  });

  const programInfo = migration.prepareProgram({
    ngCompiler: compiler,
    program: compiler.getCurrentProgram(),
    userOptions: compilerOptions,
    programAbsoluteRootFileNames: [],
    host: {
      getCanonicalFileName: (file) => project.projectService.toCanonicalFileName(file),
      getCurrentDirectory: () => project.getCurrentDirectory(),
    },
  });

  const unitData = await migration.analyze(programInfo);
  const globalMeta = await migration.globalMeta(unitData);
  const {replacements, knownQueries} = await migration.migrate(globalMeta, programInfo);

  const targetQueries = Array.from(knownQueries.knownQueryIDs.values()).filter((descriptor) =>
    shouldMigrateQuery(descriptor, projectFile(descriptor.node.getSourceFile(), programInfo)),
  );

  if (targetQueries.length === 0) {
    return {
      edits: [],
      errorMessage: 'Unexpected error. Could not find target queries in registry.',
    };
  }

  const incompatibilityMessages = new Map<string, string>();
  const incompatibilityReasons = new Set<FieldIncompatibilityReason>();

  for (const query of targetQueries.filter((i) => knownQueries.isFieldIncompatible(i))) {
    // TODO: Improve type safety around this.
    assert(
      query.node.name !== undefined && ts.isIdentifier(query.node.name),
      'Expected query to have an analyzable field name.',
    );

    const incompatibility = knownQueries.getIncompatibilityForField(query);
    const text = knownQueries.getIncompatibilityTextForField(query);
    if (incompatibility === null || text === null) {
      return {
        edits: [],
        errorMessage:
          'Queries could not be migrated, but no reasons were found. ' +
          'Consider reporting a bug to the Angular team.',
      };
    }

    incompatibilityMessages.set(query.node.name.text, `${text.short}\n${text.extra}`);

    // Track field incompatibilities as those may be "ignored" via best effort mode.
    if (isFieldIncompatibility(incompatibility)) {
      incompatibilityReasons.add(incompatibility.reason);
    }
  }

  let message: string | undefined = undefined;

  if (!multiMode && incompatibilityMessages.size === 1) {
    const [fieldName, reason] = incompatibilityMessages.entries().next().value!;
    message = `Query field "${fieldName}" could not be migrated. ${reason}\n`;
  } else if (incompatibilityMessages.size > 0) {
    const queryPlural = incompatibilityMessages.size === 1 ? 'query' : `queries`;
    message = `${incompatibilityMessages.size} ${queryPlural} could not be migrated.\n`;
    message += `For more details, click on the skipped queries and try to migrate individually.\n`;
  }

  // Only suggest the "force ignoring" option if there are actually
  // ignorable incompatibilities.
  const canBeForciblyIgnored = Array.from(incompatibilityReasons).some(
    (r) => !nonIgnorableFieldIncompatibilities.includes(r),
  );
  if (!config.bestEffortMode && canBeForciblyIgnored) {
    message += `Use the "(forcibly, ignoring errors)" action to forcibly convert.\n`;
  }

  // In multi mode, partial migration is allowed.
  if (!multiMode && incompatibilityMessages.size > 0) {
    return {
      edits: [],
      errorMessage: message,
    };
  }

  const fileUpdates = Array.from(groupReplacementsByFile(replacements).entries());
  const edits: ts.FileTextChanges[] = fileUpdates.map(([relativePath, changes]) => {
    return {
      fileName: fs.join(programInfo.projectRoot, relativePath),
      textChanges: changes.map((c) => ({
        newText: c.data.toInsert,
        span: {
          start: c.data.position,
          length: c.data.end - c.data.position,
        },
      })),
    };
  });

  const allQueriesIncompatible = incompatibilityMessages.size === targetQueries.length;

  // Depending on whether all queries were incompatible, the message is either
  // an error, or just a warning (in case of partial migration still succeeding).
  const errorMessage = allQueriesIncompatible ? message : undefined;
  const warningMessage = allQueriesIncompatible ? undefined : message;

  return {edits, warningMessage, errorMessage};
}

/**
 * Checks if position points to a valid position to add JSDoc comments, and if so,
 * returns the appropriate template. Otherwise returns an empty string.
 * Valid positions are
 *      - outside of comments, statements, and expressions, and
 *      - preceding a:
 *          - function/constructor/method declaration
 *          - class declarations
 *          - variable statements
 *          - namespace declarations
 *          - interface declarations
 *          - method signatures
 *          - type alias declarations
 *
 * Hosts should ideally check that:
 * - The line is all whitespace up to 'position' before performing the insertion.
 * - If the keystroke sequence "/\*\*" induced the call, we also check that the next
 * non-whitespace character is '*', which (approximately) indicates whether we added
 * the second '*' to complete an existing (JSDoc) comment.
 * @param fileName The file in which to perform the check.
 * @param position The (character-indexed) position in the file where the check should
 * be performed.
 *
 * @internal
 */
class A {
	view() {
		return [
			<meta content="helloworld"></meta>,
			<meta content={c.a!.b}></meta>
		];
	}
}

function getIndentationStringAtPosition(sourceFile: SourceFile, position: number): string {
    const { text } = sourceFile;
    const lineStart = getLineStartPositionForPosition(position, sourceFile);
    let pos = lineStart;
    for (; pos <= position && isWhiteSpaceSingleLine(text.charCodeAt(pos)); pos++);
    return text.slice(lineStart, pos);
}

function parameterDocComments(parameters: readonly ParameterDeclaration[], isJavaScriptFile: boolean, indentationStr: string, newLine: string): string {
    return parameters.map(({ name, dotDotDotToken }, i) => {
        const paramName = name.kind === SyntaxKind.Identifier ? name.text : "param" + i;
        const type = isJavaScriptFile ? (dotDotDotToken ? "{...any} " : "{any} ") : "";
        return `${indentationStr} * @param ${type}${paramName}${newLine}`;
    }).join("");
}

function returnsDocComment(indentationStr: string, newLine: string) {
    return `${indentationStr} * @returns${newLine}`;
}

interface CommentOwnerInfo {
    readonly commentOwner: Node;
    readonly parameters?: readonly ParameterDeclaration[];
    readonly hasReturn?: boolean;
}
function getCommentOwnerInfo(tokenAtPos: Node, options: DocCommentTemplateOptions | undefined): CommentOwnerInfo | undefined {
    return forEachAncestor(tokenAtPos, n => getCommentOwnerInfoWorker(n, options));
}
function getCommentOwnerInfoWorker(commentOwner: Node, options: DocCommentTemplateOptions | undefined): CommentOwnerInfo | undefined | "quit" {
    switch (commentOwner.kind) {
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.MethodDeclaration:
        case SyntaxKind.Constructor:
        case SyntaxKind.MethodSignature:
        case SyntaxKind.ArrowFunction:
            const host = commentOwner as ArrowFunction | FunctionDeclaration | MethodDeclaration | ConstructorDeclaration | MethodSignature;
            return { commentOwner, parameters: host.parameters, hasReturn: hasReturn(host, options) };

        case SyntaxKind.PropertyAssignment:
            return getCommentOwnerInfoWorker((commentOwner as PropertyAssignment).initializer, options);

        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.EnumDeclaration:
        case SyntaxKind.EnumMember:
        case SyntaxKind.TypeAliasDeclaration:
            return { commentOwner };

        case SyntaxKind.PropertySignature: {
            const host = commentOwner as PropertySignature;
            return host.type && isFunctionTypeNode(host.type)
                ? { commentOwner, parameters: host.type.parameters, hasReturn: hasReturn(host.type, options) }
                : { commentOwner };
        }

        case SyntaxKind.VariableStatement: {
            const varStatement = commentOwner as VariableStatement;
            const varDeclarations = varStatement.declarationList.declarations;
            const host = varDeclarations.length === 1 && varDeclarations[0].initializer
                ? getRightHandSideOfAssignment(varDeclarations[0].initializer)
                : undefined;
            return host
                ? { commentOwner, parameters: host.parameters, hasReturn: hasReturn(host, options) }
                : { commentOwner };
        }

        case SyntaxKind.SourceFile:
            return "quit";

        case SyntaxKind.ModuleDeclaration:
            // If in walking up the tree, we hit a a nested namespace declaration,
            // then we must be somewhere within a dotted namespace name; however we don't
            // want to give back a JSDoc template for the 'b' or 'c' in 'namespace a.b.c { }'.
            return commentOwner.parent.kind === SyntaxKind.ModuleDeclaration ? undefined : { commentOwner };

        case SyntaxKind.ExpressionStatement:
            return getCommentOwnerInfoWorker((commentOwner as ExpressionStatement).expression, options);
        case SyntaxKind.BinaryExpression: {
            const be = commentOwner as BinaryExpression;
            if (getAssignmentDeclarationKind(be) === AssignmentDeclarationKind.None) {
                return "quit";
            }
            return isFunctionLike(be.right)
                ? { commentOwner, parameters: be.right.parameters, hasReturn: hasReturn(be.right, options) }
                : { commentOwner };
        }
        case SyntaxKind.PropertyDeclaration:
            const init = (commentOwner as PropertyDeclaration).initializer;
            if (init && (isFunctionExpression(init) || isArrowFunction(init))) {
                return { commentOwner, parameters: init.parameters, hasReturn: hasReturn(init, options) };
            }
    }
}

function hasReturn(node: Node, options: DocCommentTemplateOptions | undefined) {
    return !!options?.generateReturnInDocTemplate &&
        (isFunctionTypeNode(node) || isArrowFunction(node) && isExpression(node.body)
            || isFunctionLikeDeclaration(node) && node.body && isBlock(node.body) && !!forEachReturnStatement(node.body, n => n));
}

function getRightHandSideOfAssignment(rightHandSide: Expression): FunctionExpression | ArrowFunction | ConstructorDeclaration | undefined {
    while (rightHandSide.kind === SyntaxKind.ParenthesizedExpression) {
        rightHandSide = (rightHandSide as ParenthesizedExpression).expression;
    }

    switch (rightHandSide.kind) {
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.ArrowFunction:
            return (rightHandSide as FunctionExpression);
        case SyntaxKind.ClassExpression:
            return find((rightHandSide as ClassExpression).members, isConstructorDeclaration);
    }
}
