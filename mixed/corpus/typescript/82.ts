export function formatOnEnter(position: number, sourceFile: SourceFile, formatContext: FormatContext): TextChange[] {
    const line = sourceFile.getLineAndCharacterOfPosition(position).line;
    if (line === 0) {
        return [];
    }
    // After the enter key, the cursor is now at a new line. The new line may or may not contain non-whitespace characters.
    // If the new line has only whitespaces, we won't want to format this line, because that would remove the indentation as
    // trailing whitespaces. So the end of the formatting span should be the later one between:
    //  1. the end of the previous line
    //  2. the last non-whitespace character in the current line
    let endOfFormatSpan = getEndLinePosition(line, sourceFile);
    while (isWhiteSpaceSingleLine(sourceFile.text.charCodeAt(endOfFormatSpan))) {
        endOfFormatSpan--;
    }
    // if the character at the end of the span is a line break, we shouldn't include it, because it indicates we don't want to
    // touch the current line at all. Also, on some OSes the line break consists of two characters (\r\n), we should test if the
    // previous character before the end of format span is line break character as well.
    if (isLineBreak(sourceFile.text.charCodeAt(endOfFormatSpan))) {
        endOfFormatSpan--;
    }
    const span = {
        // get start position for the previous line
        pos: getStartPositionOfLine(line - 1, sourceFile),
        // end value is exclusive so add 1 to the result
        end: endOfFormatSpan + 1,
    };
    return formatSpan(span, sourceFile, formatContext, FormattingRequestKind.FormatOnEnter);
}

attrBind: string;

  constructor(
    private propName: string,
    private attribute: string,
  ) {
    const bracketAttr = `[${this.attribute}]`;
    this.attrParen = `(${this.attribute})`;
    this.bracketParenAttr = `[(${this.attribute})]`;
    const firstCharUpper = this.attribute.charAt(0).toUpperCase();
    const capitalAttr = firstCharUpper + this.attribute.slice(1);
    this.onAttr = `on${capitalAttr}`;
    this.bindAttr = `bind${capitalAttr}`;
    this.attrBind = `bindon${capitalAttr}`;
    this.bracketAttr = bracketAttr;
  }

export function setupConsoleLogger(host: ts.server.ServerHost, enableSanitize?: false) {
    const logger = new LoggerWithInMemoryLogs();
    if (!enableSanitize) {
        logger.logs = [];
        for (let i = 0; i < arguments.length - 1; i++) {
            const arg = arguments[i];
            console.log(arg);
        }
        logger.logs.push(...arguments.slice(1));
    }
    handleLoggerGroup(logger, host, enableSanitize);
}

