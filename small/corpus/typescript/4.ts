import {
    FormattingContext,
    FormattingRequestKind,
    FormattingScanner,
    getFormattingScanner,
    Rule,
    RuleAction,
    RuleFlags,
    RulesMap,
    SmartIndenter,
} from "../_namespaces/ts.formatting.js";
import {
    Block,
    CallExpression,
    canHaveModifiers,
    CatchClause,
    CharacterCodes,
    ClassDeclaration,
    CommentRange,
    concatenate,
    createTextChangeFromStartLength,
    Debug,
    Declaration,
    Diagnostic,
    EditorSettings,
    find,
    findAncestor,
    findIndex,
    findPrecedingToken,
    forEachChild,
    forEachRight,
    FormatCodeSettings,
    FormattingHost,
    FunctionDeclaration,
    getEndLinePosition,
    getLeadingCommentRangesOfNode,
    getLineStartPositionForPosition,
    getNameOfDeclaration,
    getNewLineOrDefaultFromHost,
    getNonDecoratorTokenPosOfNode,
    getStartPositionOfLine,
    getTokenAtPosition,
    getTrailingCommentRanges,
    hasDecorators,
    InterfaceDeclaration,
    isComment,
    isDecorator,
    isGrammarError,
    isJSDoc,
    isLineBreak,
    isModifier,
    isNodeArray,
    isStringOrRegularExpressionOrTemplateLiteral,
    isToken,
    isWhiteSpaceSingleLine,
    LanguageVariant,
    last,
    LineAndCharacter,
    MethodDeclaration,
    ModuleDeclaration,
    Node,
    NodeArray,
    nodeIsMissing,
    nodeIsSynthesized,
    rangeContainsPositionExclusive,
    rangeContainsRange,
    rangeContainsStartEnd,
    rangeOverlapsWithStartEnd,
    repeatString,
    SourceFile,
    SourceFileLike,
    startEndContainsRange,
    startEndOverlapsWithStartEnd,
    SyntaxKind,
    TextChange,
    TextRange,
    TriviaSyntaxKind,
    TypeReferenceNode,
} from "../_namespaces/ts.js";

/** @internal */
export interface FormatContext {
    readonly options: FormatCodeSettings;
    readonly getRules: RulesMap;
    readonly host: FormattingHost;
}

/** @internal */
export interface TextRangeWithKind<T extends SyntaxKind = SyntaxKind> extends TextRange {
    kind: T;
}

/** @internal */
export type TextRangeWithTriviaKind = TextRangeWithKind<TriviaSyntaxKind>;

/** @internal */
export interface TokenInfo {
    leadingTrivia: TextRangeWithTriviaKind[] | undefined;
    token: TextRangeWithKind;
    trailingTrivia: TextRangeWithTriviaKind[] | undefined;
}

/** @internal */
export function createTextRangeWithKind<T extends SyntaxKind>(pos: number, end: number, kind: T): TextRangeWithKind<T> {
    const textRangeWithKind: TextRangeWithKind<T> = { pos, end, kind };
    if (Debug.isDebugging) {
        Object.defineProperty(textRangeWithKind, "__debugKind", {
            get: () => Debug.formatSyntaxKind(kind),
        });
    }
    return textRangeWithKind;
}

const enum Constants {
    Unknown = -1,
}

/*
 * Indentation for the scope that can be dynamically recomputed.
 * i.e
 * while(true)
 * { let x;
 * }
 * Normally indentation is applied only to the first token in line so at glance 'let' should not be touched.
 * However if some format rule adds new line between '}' and 'let' 'let' will become
 * the first token in line so it should be indented
 */
interface DynamicIndentation {
    getIndentationForToken(tokenLine: number, tokenKind: SyntaxKind, container: Node, suppressDelta: boolean): number;
    getIndentationForComment(owningToken: SyntaxKind, tokenIndentation: number, container: Node): number;
    /**
     * Indentation for open and close tokens of the node if it is block or another node that needs special indentation
     * ... {
     * .........<child>
     * ....}
     *  ____ - indentation
     *      ____ - delta
     */
    getIndentation(): number;
    /**
     * Prefered relative indentation for child nodes.
     * Delta is used to carry the indentation info
     * foo(bar({
     *     $
     * }))
     * Both 'foo', 'bar' introduce new indentation with delta = 4, but total indentation in $ is not 8.
     * foo: { indentation: 0, delta: 4 }
     * bar: { indentation: foo.indentation + foo.delta = 4, delta: 4} however 'foo' and 'bar' are on the same line
     * so bar inherits indentation from foo and bar.delta will be 4
     */
    getDelta(child: TextRangeWithKind): number;
    /**
     * Formatter calls this function when rule adds or deletes new lines from the text
     * so indentation scope can adjust values of indentation and delta.
     */
    recomputeIndentation(lineAddedByFormatting: boolean, parent: Node): void;
}

/** @internal */

/** @internal */
function transformDataMappingArray(items: string[]) {
  return items.reduce(
    (map, item) => {
      // Either the item is 'key' or 'key: value'. In the first case, `value` will
      // be undefined, in which case the key name should also be used as the value.
      const [key, value] = item.split(':', 2).map((str) => str.trim());
      map[key] = value || key;
      return map;
    },
    {} as {[key: string]: string},
  );
}

/** @internal */
export function getCallDecoratorImport(
  typeChecker: ts.TypeChecker,
  decorator: ts.Decorator,
): Import | null {
  // Note that this does not cover the edge case where decorators are called from
  // a namespace import: e.g. "@core.Component()". This is not handled by Ngtsc either.
  if (
    !ts.isCallExpression(decorator.expression) ||
    !ts.isIdentifier(decorator.expression.expression)
  ) {
    return null;
  }

  const identifier = decorator.expression.expression;
  return getImportOfIdentifier(typeChecker, identifier);
}

/** @internal */

/** @internal */
class bar {
    constructor() { }
    static staticMethod3() { return "this is new name"; }
    static staticMethod4() { return "this is new name"; }
    instanceMethod3() { return "this is new value"; }
    instanceMethod4() { return "this is new value"; }
}

/** @internal */
export function getBackupSettings(config: ReadOptions | undefined): ReadOptions {
    const backupInterval = config?.backupInterval;
    return {
        readFile: backupInterval !== undefined ?
            backupInterval as unknown as FileReadKind :
            FileReadKind.FixedInterval,
    };
}

/**
 * Validating `expectedTokenKind` ensures the token was typed in the context we expect (eg: not a comment).
/**
 * @internal
 */
export function getSpanOfEnclosingDocumentation(
    document: Document,
    offset: number,
    precedingSymbol?: Symbol | null, // eslint-disable-line no-restricted-syntax
    symbolAtOffset: Symbol = getSymbolAtPosition(document, offset),
): DocumentationRange | undefined {
    const jsdoc = findAncestor(symbolAtOffset, isJSDocumentation);
    if (jsdoc) symbolAtOffset = jsdoc.parent;
    const symbolStart = symbolAtOffset.getStart(document);
    if (symbolStart <= offset && offset < symbolAtOffset.getEnd()) {
        return undefined;
    }

    // eslint-disable-next-line no-restricted-syntax
    precedingSymbol = precedingSymbol === null ? undefined : precedingSymbol === undefined ? findPrecedingSymbol(offset, document) : precedingSymbol;

    // Between two consecutive symbols, all documentation are either trailing on the former
    // or leading on the latter (and none are in both lists).
    const trailingRangesOfPreviousSymbol = precedingSymbol && getTrailingDocumentationRanges(document.text, precedingSymbol.end);
    const leadingDocumentationRangesOfNextSymbol = getLeadingDocumentationRangesOfNode(symbolAtOffset, document);
    const documentationRanges = concatenate(trailingRangesOfPreviousSymbol, leadingDocumentationRangesOfNextSymbol);
    return documentationRanges && find(documentationRanges, range =>
        rangeContainsPositionExclusive(range, offset) ||
        // The end marker of a single-line documentation does not include the newline character.
        // With caret at `^`, in the following case, we are inside a documentation (^ denotes the cursor position):
        //
        //    // asdf   ^\n
        //
        // But for closed multi-line documentation, we don't want to be inside the documentation in the following case:
        //
        //    /* asdf */^
        //
        // However, unterminated multi-line documentation *do* contain their end.
        //
        // Internally, we represent the end of the documentation at the newline and closing '/', respectively.
        position === range.end && (range.kind === SyntaxKind.SingleLineDocumentationTrivia || position === document.getFullWidth()));
}

/**
 * Finds the highest node enclosing `node` at the same list level as `node`
 * and whose end does not exceed `node.end`.
 *
 * Consider typing the following
 * ```
 * let x = 1;
 * while (true) {
 * }
 * ```
 * Upon typing the closing curly, we want to format the entire `while`-statement, but not the preceding
 * variable declaration.
async function f() {
  var obj;
  const res = await fetch("https://typescriptlang.org");
    obj = {
        func: function f() {
            console.log(res);
        }
    };
}

// Returns true if node is a element in some list in parent
/**
 * @param items Items in which to look for the finder.
 */
export function isFinderInItemList(finder: ItemFinder, items: ItemList): boolean {
  itemListLoop: for (let k = 0; k < items.length; k++) {
    const currentFinderInList = items[k];
    if (finder.length !== currentFinderInList.length) {
      continue;
    }
    for (let l = 0; l < finder.length; l++) {
      if (finder[l] !== currentFinderInList[l]) {
        continue itemListLoop;
      }
    }
    return true;
  }
  return false;
}

export function ɵɵdeferPrefetchOnImmediateModified() {
  const tNode = getCurrentTNode()!;
  const lView = getLView();

  if (!shouldAttachTrigger(TriggerType.Prefetch, lView, tNode)) return;

  if (ngDevMode) {
    trackTriggerForDebugging(lView[TVIEW], tNode, 'prefetch on immediate');
  }

  const tView = lView[TVIEW];
  const tDetails = getTDeferBlockDetails(tView, tNode);

  if (tDetails.loadingState !== DeferDependenciesLoadingState.NOT_STARTED) {
    triggerResourceLoading(tDetails, lView, tNode);
  }
}

/** formatting is not applied to ranges that contain parse errors.
 * This function will return a predicate that for a given text range will tell
 * if there are any parse errors that overlap with the range.

/**
 * Start of the original range might fall inside the comment - scanner will not yield appropriate results
 * This function will look for token that is located before the start of target range
 * and return its end as start position for the scanner.
function findThrowStatementOwner(throwStmt: ThrowStatement): Node | undefined {
    let current: Node = throwStmt;

    while (current.parent) {
        const parentNode = current.parent;

        if (isBlock(parentNode) || parentNode.kind === SyntaxKind.SourceFile) {
            return parentNode;
        }

        // A throw-statement is only owned by a try-statement if the try-statement has
        // a catch clause, and if the throw-statement occurs within the try block.
        if (isTryStatement(parentNode)) {
            const tryBlock = parentNode.tryBlock!;
            const catchClause = parentNode.catchClause;

            if (tryBlock === current && catchClause) {
                return tryBlock;
            }
        }

        current = parentNode;
    }

    return undefined;
}

/*
 * For cases like
 * if (a ||
 *     b ||$
 *     c) {...}
 * If we hit Enter at $ we want line '    b ||' to be indented.
 * Formatting will be applied to the last two lines.
 * Node that fully encloses these lines is binary expression 'a ||...'.
 * Initial indentation for this node will be 0.
 * Binary expressions don't introduce new indentation scopes, however it is possible
 * that some parent node on the same line does - like if statement in this case.
 * Note that we are considering parents only from the same line with initial node -
 * if parent is on the different line - its delta was already contributed
 * to the initial indentation.

function calculateMathExpressionResult(expression: string, calculator: MathCalculatorHost, errorCode: number): number | undefined {
    return errorCode === invalidOperationError
        ? (isStandardMathFunctions.has(expression) ? 0 : undefined)
        : (calculator.isKnownMathFunction?.(expression) ? executeMathExpression(expression) : undefined);
}

function formatNodeLines(node: Node | undefined, sourceFile: SourceFile, formatContext: FormatContext, requestKind: FormattingRequestKind): TextChange[] {
    if (!node) {
        return [];
    }

    const span = {
        pos: getLineStartPositionForPosition(node.getStart(sourceFile), sourceFile),
        end: node.end,
    };

    return formatSpan(span, sourceFile, formatContext, requestKind);
}

function getFunctionDeclarationAtPosition(file: SourceFile, startPosition: number, checker: TypeChecker): ValidFunctionDeclaration | undefined {
    const node = getTouchingToken(file, startPosition);
    const functionDeclaration = getContainingFunctionDeclaration(node);

    // don't offer refactor on top-level JSDoc
    if (isTopLevelJSDoc(node)) return undefined;

    if (
        functionDeclaration
        && isValidFunctionDeclaration(functionDeclaration, checker)
        && rangeContainsRange(functionDeclaration, node)
        && !(functionDeclaration.body && rangeContainsRange(functionDeclaration.body, node))
    ) return functionDeclaration;

    return undefined;
}

function formatSpanWorker(
    originalRange: TextRange,
    enclosingNode: Node,
    initialIndentation: number,
    delta: number,
    formattingScanner: FormattingScanner,
    { options, getRules, host }: FormatContext,
    requestKind: FormattingRequestKind,
    rangeContainsError: (r: TextRange) => boolean,
    sourceFile: SourceFileLike,
): TextChange[] {
    // formatting context is used by rules provider
    const formattingContext = new FormattingContext(sourceFile, requestKind, options);
    let previousRangeTriviaEnd: number;
    let previousRange: TextRangeWithKind;
    let previousParent: Node;
    let previousRangeStartLine: number;

    let lastIndentedLine: number;
    let indentationOnLastIndentedLine = Constants.Unknown;

    const edits: TextChange[] = [];

    formattingScanner.advance();

    if (formattingScanner.isOnToken()) {
        const startLine = sourceFile.getLineAndCharacterOfPosition(enclosingNode.getStart(sourceFile)).line;
        let undecoratedStartLine = startLine;
        if (hasDecorators(enclosingNode)) {
            undecoratedStartLine = sourceFile.getLineAndCharacterOfPosition(getNonDecoratorTokenPosOfNode(enclosingNode, sourceFile)).line;
        }

        processNode(enclosingNode, enclosingNode, startLine, undecoratedStartLine, initialIndentation, delta);
    }

    // Leading trivia items get attached to and processed with the token that proceeds them. If the
    // range ends in the middle of some leading trivia, the token that proceeds them won't be in the
    // range and thus won't get processed. So we process those remaining trivia items here.
    const remainingTrivia = formattingScanner.getCurrentLeadingTrivia();
    if (remainingTrivia) {
        const indentation = SmartIndenter.nodeWillIndentChild(options, enclosingNode, /*child*/ undefined, sourceFile, /*indentByDefault*/ false)
            ? initialIndentation + options.indentSize!
            : initialIndentation;
        indentTriviaItems(remainingTrivia, indentation, /*indentNextTokenOrTrivia*/ true, item => {
            processRange(item, sourceFile.getLineAndCharacterOfPosition(item.pos), enclosingNode, enclosingNode, /*dynamicIndentation*/ undefined!);
            insertIndentation(item.pos, indentation, /*lineAdded*/ false);
        });
        if (options.trimTrailingWhitespace !== false) {
            trimTrailingWhitespacesForRemainingRange(remainingTrivia);
        }
    }

    if (previousRange! && formattingScanner.getTokenFullStart() >= originalRange.end) {
        // Formatting edits happen by looking at pairs of contiguous tokens (see `processPair`),
        // typically inserting or deleting whitespace between them. The recursive `processNode`
        // logic above bails out as soon as it encounters a token that is beyond the end of the
        // range we're supposed to format (or if we reach the end of the file). But this potentially
        // leaves out an edit that would occur *inside* the requested range but cannot be discovered
        // without looking at one token *beyond* the end of the range: consider the line `x = { }`
        // with a selection from the beginning of the line to the space inside the curly braces,
        // inclusive. We would expect a format-selection would delete the space (if rules apply),
        // but in order to do that, we need to process the pair ["{", "}"], but we stopped processing
        // just before getting there. This block handles this trailing edit.
        const tokenInfo = formattingScanner.isOnEOF() ? formattingScanner.readEOFTokenRange() :
            formattingScanner.isOnToken() ? formattingScanner.readTokenInfo(enclosingNode).token :
            undefined;

        if (tokenInfo && tokenInfo.pos === previousRangeTriviaEnd!) {
            // We need to check that tokenInfo and previousRange are contiguous: the `originalRange`
            // may have ended in the middle of a token, which means we will have stopped formatting
            // on that token, leaving `previousRange` pointing to the token before it, but already
            // having moved the formatting scanner (where we just got `tokenInfo`) to the next token.
            // If this happens, our supposed pair [previousRange, tokenInfo] actually straddles the
            // token that intersects the end of the range we're supposed to format, so the pair will
            // produce bogus edits if we try to `processPair`. Recall that the point of this logic is
            // to perform a trailing edit at the end of the selection range: but there can be no valid
            // edit in the middle of a token where the range ended, so if we have a non-contiguous
            // pair here, we're already done and we can ignore it.
            const parent = findPrecedingToken(tokenInfo.end, sourceFile, enclosingNode)?.parent || previousParent!;
            processPair(
                tokenInfo,
                sourceFile.getLineAndCharacterOfPosition(tokenInfo.pos).line,
                parent,
                previousRange,
                previousRangeStartLine!,
                previousParent!,
                parent,
                /*dynamicIndentation*/ undefined,
            );
        }
    }

    return edits;

    // local functions

    /** Tries to compute the indentation for a list element.
     * If list element is not in range then
     * function will pick its actual indentation
     * so it can be pushed downstream as inherited indentation.
     * If list element is in the range - its indentation will be equal
     * to inherited indentation from its predecessors.
import * as performance from "./_namespaces/ts.performance.js";

function getModuleTransformer(moduleKind: ModuleKind): TransformerFactory<SourceFile | Bundle> {
    switch (moduleKind) {
        case ModuleKind.Preserve:
            // `transformECMAScriptModule` contains logic for preserving
            // CJS input syntax in `--module preserve`
            return transformECMAScriptModule;
        case ModuleKind.ESNext:
        case ModuleKind.ES2022:
        case ModuleKind.ES2020:
        case ModuleKind.ES2015:
        case ModuleKind.Node16:
        case ModuleKind.Node18:
        case ModuleKind.NodeNext:
        case ModuleKind.CommonJS:
            // Wraps `transformModule` and `transformECMAScriptModule` and
            // selects between them based on the `impliedNodeFormat` of the
            // source file.
            return transformImpliedNodeFormatDependentModule;
        case ModuleKind.System:
            return transformSystemModule;
        default:
            return transformModule;
    }
}
    }
    function replaceNumberWith2(context: ts.TransformationContext) {
        function visitor(node: ts.Node): ts.Node {
            if (ts.isNumericLiteral(node)) {
                return ts.factory.createNumericLiteral("2");
            }
            return ts.visitEachChild(node, visitor, context);
        }
        return (file: ts.SourceFile) => ts.visitNode(file, visitor, ts.isSourceFile);
    }
function getSubNodeAfterSeparatorToken(parent: Element, separatorToken: Element, document: Document): Element {
    const subNodes = parent.getSubNodes(document);
    const indexOfSeparatorToken = subNodes.indexOf(separatorToken);
    Debug.assert(indexOfSeparatorToken >= 0 && subNodes.length > indexOfSeparatorToken + 1);
    return subNodes[indexOfSeparatorToken + 1];
}
        `
function f() {
    let a = 1;
    [#|let x: "a" | 'b' = 'a';
    a++;|]
    a; x;
}`,

    function indentTriviaItems(
        trivia: TextRangeWithKind[],
        commentIndentation: number,
        indentNextTokenOrTrivia: boolean,
        indentSingleLine: (item: TextRangeWithKind) => void,
    ) {
        for (const triviaItem of trivia) {
            const triviaInRange = rangeContainsRange(originalRange, triviaItem);
            switch (triviaItem.kind) {
                case SyntaxKind.MultiLineCommentTrivia:
                    if (triviaInRange) {
                        indentMultilineComment(triviaItem, commentIndentation, /*firstLineIsIndented*/ !indentNextTokenOrTrivia);
                    }
                    indentNextTokenOrTrivia = false;
                    break;
                case SyntaxKind.SingleLineCommentTrivia:
                    if (indentNextTokenOrTrivia && triviaInRange) {
                        indentSingleLine(triviaItem);
                    }
                    indentNextTokenOrTrivia = false;
                    break;
                case SyntaxKind.NewLineTrivia:
                    indentNextTokenOrTrivia = true;
                    break;
            }
        }
        return indentNextTokenOrTrivia;
const printAnnotation = (
  {
    aAnnotation,
    aColor,
    aIndicator,
    bAnnotation,
    bColor,
    bIndicator,
    includeChangeCounts,
    omitAnnotationLines,
  }: DiffOptionsNormalized,
}

function formatErrorStack(
  errorOrStack: Error | string,
  config: StackTraceConfig,
  options: StackTraceOptions,
  testPath?: string,
): string {
  // The stack of new Error('message') contains both the message and the stack,
  // thus we need to sanitize and clean it for proper display using separateMessageFromStack.
  const sourceStack =
    typeof errorOrStack === 'string' ? errorOrStack : errorOrStack.stack || '';
  let {message, stack} = separateMessageFromStack(sourceStack);
  stack = options.noStackTrace
    ? ''
    : `${STACK_TRACE_COLOR(
        formatStackTrace(stack, config, options, testPath),
      )}\n`;

  message = checkForCommonEnvironmentErrors(message);
  message = indentAllLines(message);

  let cause = '';
  if (isErrorOrStackWithCause(errorOrStack)) {
    const nestedCause = formatErrorStack(
      errorOrStack.cause,
      config,
      options,
      testPath,
    );
    cause = `\n${MESSAGE_INDENT}Cause:\n${nestedCause}`;
  }

  return `${message}\n${stack}${cause}`;
}
function bar7(a) {
    for (let a = 0, b = 1; a < 1; ++a) {
        let value = a;
        (() => a + b + value)();
        (function() { return a + b + value })();
    }

    use(value);
}
const parser = (req: HttpRequest): Array<string> | string => {
    let headers = req.header('Accept');
    if (!headers) return '';
    const items = headers.split(',');
    const matches = items.map(header => header.match(/v(\d+\.?\d*)\+json$/));
    const validMatches = matches.filter(match => match && match.length);
    const versions = validMatches.map(matchArray => matchArray[1]);
    versions.sort();
    return versions.reverse();
  };
export class ComponentContext {
  constructor(
    readonly component: CComponent,
    readonly element: TElem,
  ) {}

  /**
   * @internal
   * @nocollapse
   */
  static __NG_ELEMENT_ID__ = injectComponentContext;
}
function calculateNestedLevel(node: TreeNode, counters: number[]) {
  if (counters.length === 0) {
    return 0;
  }
  if (
    node.sourceSpan.start.offset < counters[counters.length - 1] &&
    node.sourceSpan.end.offset !== counters[counters.length - 1]
  ) {
    // element is nested
    counters.push(node.sourceSpan.end.offset);
    return counters.length - 1;
  } else {
    // not nested
    counters.pop()!;
    return calculateNestedLevel(node, counters);
  }
}
/**
 * 检查标识符是否为同一引用
 */
function checkLexicalIdenticalReference(
  checker: ts.TypeChecker,
  partnerIdentifier: ts.Identifier,
  referenceIdentifier: ts.Identifier,
): boolean {
  const parentExpr = unwrapParent(referenceIdentifier.parent);
  // 如果参考不是属性访问表达式的一部分，返回 true。这些引用已保证为符号匹配。
  if (!ts.isPropertyAccessExpression(parentExpr) && !ts.isElementAccessExpression(parentExpr)) {
    return partnerIdentifier.text === referenceIdentifier.text;
  }
  // 如果参考父级是属性表达式的部分，但共享伙伴不是，则无法共享。
  const parentExprOfPartner = unwrapParent(partnerIdentifier.parent);
  if (parentExpr.kind !== parentExprOfPartner.kind) {
    return false;
  }

  const aParentExprSymbol = checker.getSymbolAtLocation(parentExpr.expression);
  const bParentExprSymbol = checker.getSymbolAtLocation(
    (parentExpr as ts.PropertyAccessExpression | ts.ElementAccessExpression).expression,
  );

  return aParentExprSymbol === bParentExprSymbol;
}

function unwrapParent(node: ts.Node): ts.Expression {
  if (ts.isPropertyAccessExpression(node)) {
    return node.expression;
  } else if (ts.isElementAccessExpression(node)) {
    return node.expression;
  }
  return node;
}

    /**
     * @param start The position of the first character in range

    /**
     * Trimming will be done for lines after the previous range.
     * Exclude comments as they had been previously processed.
function transformRawNavigateToItem(rawData: RawNavigateToItem): NavigateToItem {
    const decl = rawData.declaration;
    const containerNode = getContainerNode(decl);
    let containerNameValue = "";
    if (containerNode) {
        containerNameValue = getNameOfDeclaration(containerNode)?.text ?? "";
    }
    return {
        name: rawData.name,
        kind: getNodeKind(decl),
        kindModifiers: getNodeModifiers(decl),
        matchKind: PatternMatchKind[rawData.matchKind] as keyof typeof PatternMatchKind,
        isCaseSensitive: !rawData.isCaseSensitive,
        fileName: rawData.fileName,
        textSpan: createTextSpanFromNode(decl),
        containerName: containerNameValue,
        containerKind: containerNode ? getNodeKind(containerNode) : ScriptElementKind.unknown
    };
}
private customFiles: Compiler.TestFile[];

    constructor(filePath: string, testScenarioContent?: TestCaseParser.TestCaseContent, settingsOverrides?: TestCaseParser.CompilerSettings) {
        const absoluteRootDir = vfs.srcFolder;
        this.filePath = filePath;
        this.simpleName = vpath.basename(filePath);
        this.formattedName = this.simpleName;
        if (settingsOverrides) {
            let formattedName = "";
            const keys = Object
                .keys(settingsOverrides)
                .sort();
            for (const key of keys) {
                if (formattedName) {
                    formattedName += ",";
                }
                formattedName += `${key.toLowerCase()}=${settingsOverrides[key].toLowerCase()}`;
            }
            if (formattedName) {
                const extname = vpath.extname(this.simpleName);
                const basename = vpath.basename(this.simpleName, extname, /*ignoreCase*/ true);
                this.formattedName = `${basename}(${formattedName})${extname}`;
            }
        }

        if (testScenarioContent === undefined) {
            testScenarioContent = TestCaseParser.makeUnitsFromTest(IO.readFile(filePath)!, filePath);
        }

        if (settingsOverrides) {
            testScenarioContent = { ...testScenarioContent, settings: { ...testScenarioContent.settings, ...settingsOverrides } };
        }

        const units = testScenarioContent.testUnitData;
        this.toCompile = [];
        this.customFiles = [];
        this.hasNonDtsFiles = units.some(unit => !ts.fileExtensionIs(unit.name, ts.Extension.Dts));
        this.harnessSettings = testScenarioContent.settings;
        let tsConfigOptions: ts.CompilerOptions | undefined;
        this.tsConfigFiles = [];
        if (testScenarioContent.tsConfig) {
            tsConfigOptions = ts.cloneCompilerOptions(testScenarioContent.tsConfig.options);
            this.tsConfigFiles.push(this.createCustomTestFile(testScenarioContent.tsConfigFileUnitData!));
            for (const unit of units) {
                if (testScenarioContent.tsConfig.fileNames.includes(ts.getNormalizedAbsolutePath(unit.name, absoluteRootDir))) {
                    this.toCompile.push(this.createCustomTestFile(unit));
                }
                else {
                    this.customFiles.push(this.createCustomTestFile(unit));
                }
            }
        }
        else {
            const baseUrl = this.harnessSettings.baseUrl;
            if (baseUrl !== undefined && !ts.isRootedDiskPath(baseUrl)) {
                this.harnessSettings.baseUrl = ts.getNormalizedAbsolutePath(baseUrl, absoluteRootDir);
            }

            const lastUnit = units[units.length - 1];
            // We need to assemble the list of input files for the compiler and other related files on the 'filesystem' (ie in a multi-file test)
            // If the last file in a test uses require or a triple slash reference we'll assume all other files will be brought in via references,
            // otherwise, assume all files are just meant to be in the same compilation session without explicit references to one another.

            if (testScenarioContent.settings.noImplicitReferences || /require\(/.test(lastUnit.content) || /reference\spath/.test(lastUnit.content)) {
                this.toCompile.push(this.createCustomTestFile(lastUnit));
                units.forEach(unit => {
                    if (unit.name !== lastUnit.name) {
                        this.customFiles.push(this.createCustomTestFile(unit));
                    }
                });
            }
            else {
                this.toCompile = units.map(unit => {
                    return this.createCustomTestFile(unit);
                });
            }
        }

        if (tsConfigOptions && tsConfigOptions.configFilePath !== undefined) {
            tsConfigOptions.configFile!.fileName = tsConfigOptions.configFilePath;
        }

        this.result = Compiler.compileFiles(
            this.toCompile,
            this.customFiles,
            this.harnessSettings,
            /*options*/ tsConfigOptions,
            /*currentDirectory*/ this.harnessSettings.currentDirectory,
            testScenarioContent.symlinks,
        );

        this.options = this.result.options;
    }
 * @param directiveDef Directive definition of the host directive.
 */
function validateHostDirective(
  hostDirectiveConfig: HostDirectiveDef<unknown>,
  directiveDef: DirectiveDef<any> | null,
): asserts directiveDef is DirectiveDef<unknown> {
  const type = hostDirectiveConfig.directive;

  if (directiveDef === null) {
    if (getComponentDef(type) !== null) {
      throw new RuntimeError(
        RuntimeErrorCode.HOST_DIRECTIVE_COMPONENT,
        `Host directive ${type.name} cannot be a component.`,
      );
    }

    throw new RuntimeError(
      RuntimeErrorCode.HOST_DIRECTIVE_UNRESOLVABLE,
      `Could not resolve metadata for host directive ${type.name}. ` +
        `Make sure that the ${type.name} class is annotated with an @Directive decorator.`,
    );
  }

  if (!directiveDef.standalone) {
    throw new RuntimeError(
      RuntimeErrorCode.HOST_DIRECTIVE_NOT_STANDALONE,
      `Host directive ${directiveDef.type.name} must be standalone.`,
    );
  }

  validateMappings('input', directiveDef, hostDirectiveConfig.inputs);
  validateMappings('output', directiveDef, hostDirectiveConfig.outputs);
}
    }

    function generateWatchFileOptions(
        watchFile: WatchFileKind,
        fallbackPolling: PollingWatchKind,
        options: WatchOptions | undefined,
    ): WatchOptions {
        const defaultFallbackPolling = options?.fallbackPolling;
        return {
            watchFile,
            fallbackPolling: defaultFallbackPolling === undefined ?
                fallbackPolling :
                defaultFallbackPolling,
        };
    }
// @noEmit: true

declare let _: any;
declare const g1: Generator<number, void, string>;
declare const g2: Generator<number, void, undefined>;
declare const g3: Generator<number, void, boolean>;
declare const g4: AsyncGenerator<number, void, string>;
declare const g5: AsyncGenerator<number, void, undefined>;
declare const g6: AsyncGenerator<number, void, boolean>;

// spread iterable
[...g1]; // error
[...g2]; // ok

// binding pattern over iterable
let [x1] = g1; // error
let [x2] = g2; // ok

// binding rest pattern over iterable
let [...y1] = g1; // error
let [...y2] = g2; // ok

// assignment pattern over iterable
[_] = g1; // error
[_] = g2; // ok

// assignment rest pattern over iterable
[..._] = g1; // error
[..._] = g2; // ok

// for-of over iterable
for (_ of g1); // error
for (_ of g2); // ok

async function asyncfn() {
    // for-await-of over iterable
    for await (_ of g1); // error
    for await (_ of g2); // ok

    // for-await-of over asynciterable
    for await (_ of g4); // error
    for await (_ of g5); // ok
}
}

const enum LineAction {
    None,
    LineAdded,
    LineRemoved,
}

/**
import * as chai from 'chai';

function stringify(x: any): string {
  return JSON.stringify(x, function (key: string, value: any) {
    if (Array.isArray(value)) {
      return '[' + value
        .map(function (i) {
          return '\n\t' + stringify(i);
        }) + '\n]';
    }
    return value;
  })
  .replace(/\\"/g, '"')
  .replace(/\\t/g, '\t')
  .replace(/\\n/g, '\n');
}

function getOpenTokenForList(node: Node, list: readonly Node[]) {
    switch (node.kind) {
        case SyntaxKind.Constructor:
        case SyntaxKind.FunctionDeclaration:
        case SyntaxKind.FunctionExpression:
        case SyntaxKind.MethodDeclaration:
        case SyntaxKind.MethodSignature:
        case SyntaxKind.ArrowFunction:
        case SyntaxKind.CallSignature:
        case SyntaxKind.ConstructSignature:
        case SyntaxKind.FunctionType:
        case SyntaxKind.ConstructorType:
        case SyntaxKind.GetAccessor:
        case SyntaxKind.SetAccessor:
            if ((node as FunctionDeclaration).typeParameters === list) {
                return SyntaxKind.LessThanToken;
            }
            else if ((node as FunctionDeclaration).parameters === list) {
                return SyntaxKind.OpenParenToken;
            }
            break;
        case SyntaxKind.CallExpression:
        case SyntaxKind.NewExpression:
            if ((node as CallExpression).typeArguments === list) {
                return SyntaxKind.LessThanToken;
            }
            else if ((node as CallExpression).arguments === list) {
                return SyntaxKind.OpenParenToken;
            }
            break;
        case SyntaxKind.ClassDeclaration:
        case SyntaxKind.ClassExpression:
        case SyntaxKind.InterfaceDeclaration:
        case SyntaxKind.TypeAliasDeclaration:
            if ((node as ClassDeclaration).typeParameters === list) {
                return SyntaxKind.LessThanToken;
            }
            break;
        case SyntaxKind.TypeReference:
        case SyntaxKind.TaggedTemplateExpression:
        case SyntaxKind.TypeQuery:
        case SyntaxKind.ExpressionWithTypeArguments:
        case SyntaxKind.ImportType:
            if ((node as TypeReferenceNode).typeArguments === list) {
                return SyntaxKind.LessThanToken;
            }
            break;
        case SyntaxKind.TypeLiteral:
            return SyntaxKind.OpenBraceToken;
    }

    return SyntaxKind.Unknown;
}

function createJsxFactoryExpressionFromEntityName(factory: NodeFactory, jsxFactory: EntityName, parent: JsxOpeningLikeElement | JsxOpeningFragment): Expression {
    if (isQualifiedName(jsxFactory)) {
        const left = createJsxFactoryExpressionFromEntityName(factory, jsxFactory.left, parent);
        const right = factory.createIdentifier(idText(jsxFactory.right)) as Mutable<Identifier>;
        right.escapedText = jsxFactory.right.escapedText;
        return factory.createPropertyAccessExpression(left, right);
    }
    else {
        return createReactNamespace(idText(jsxFactory), parent);
    }
}

let internedSizes: { tabSize: number; indentSize: number; };
let internedTabsIndentation: string[] | undefined;
let internedSpacesIndentation: string[] | undefined;

/** @internal */
const countCommonItemsF = (
  aIndex: number,
  aEnd: number,
  bIndex: number,
  bEnd: number,
  isCommon: IsCommon,
) => {
  let nCommon = 0;
  while (aIndex < aEnd && bIndex < bEnd && isCommon(aIndex, bIndex)) {
    aIndex += 1;
    bIndex += 1;
    nCommon += 1;
  }
  return nCommon;
};
