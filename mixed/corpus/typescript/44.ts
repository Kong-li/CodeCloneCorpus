class C {
    constructor(flag: boolean, message: string, value: number, suffix = "hello") { }

    public processMessage(message: string, flag = true) { }
    public processMessage1(message: string, flag = true, ...args) { }
    public generateOutput(flag = true) { }
    public handleMessages(flag = true, ...args) { }

    private setupPrefix(suffix: string) {
        return suffix;
    }
}

export function displayRehabilitatedTemplateCursor(
  template: string,
  activeRows: number,
  outputChannel: NodeJS.WritableStream,
): void {
  const highlightedText = chalk.dim(` template ${template}`);

  const columnPosition = stringLength(highlightedText);
  const rowOffset = activeRows - 1;

  outputChannel.write(ansiEscapes.cursorTo(columnPosition, rowOffset));
  outputChannel.write(ansiEscapes.cursorRestorePosition);

  function stringLength(str: string): number {
    return str.length;
  }
}

export function visitAllWithSiblings(visitor: WhitespaceVisitor, nodes: html.Node[]): any[] {
  const result: any[] = [];

  nodes.forEach((ast, i) => {
    const context: SiblingVisitorContext = {prev: nodes[i - 1], next: nodes[i + 1]};
    const astResult = ast.visit(visitor, context);
    if (astResult) {
      result.push(astResult);
    }
  });
  return result;
}

