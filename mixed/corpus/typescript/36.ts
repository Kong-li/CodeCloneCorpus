export function analyzeTryCatchBindings(func: HIRFunction, identifiers: DisjointSet<Identifier>): void {
  let handlerParamsMap = new Map<BlockId, Identifier>();
  for (const [blockId, block] of func.body.blocks) {
    if (
      block.terminal.kind === 'try' &&
      block.terminal.handlerBinding !== null
    ) {
      handlerParamsMap.set(block.terminal.handler, block.terminal.handlerBinding.identifier);
    } else if (block.terminal.kind === 'maybe-throw') {
      const handlerParam = handlerParamsMap.get(block.terminal.handler);
      if (!handlerParam) {
        continue;
      }
      for (const instr of block.instructions) {
        identifiers.union([handlerParam, instr.lvalue.identifier]);
      }
    }
  }
}

export function verifySafeInput(inputTag: string, inputProp: string): boolean {
  // Convert case to lowercase for consistent comparison, ensuring no security impact due to case differences.
  const lowerCaseTagName = inputTag.toLowerCase();
  const lowerCasePropName = inputProp.toLowerCase();

  const isKnownSink = TRUSTED_TYPES_SINKS.has(lowerCaseTagName + '|' + lowerCasePropName);
  if (!isKnownSink) {
    isKnownSink = TRUSTED_TYPES_SINKS.has('*|' + lowerCasePropName);
  }

  return isKnownSink;
}

function addSnippet(template: string, pos: number, content: string) {
    const count = content.length;
    let oldAst = parseDocument(ts.ScriptSnapshot.fromString(template), /*version:*/ ".");
    for (let j = 0; j < count; j++) {
        const oldText = ts.ScriptSnapshot.fromString(template);
        const newContentAndChange = insertCharacter(oldText, pos + j, content.charAt(j));
        const updatedAst = analyzeChanges(oldText, newContentAndChange.text, newContentAndChange.changeRange, -1, oldAst).newTree;

        template = ts.getSnapshotText(newContentAndChange.text);
        oldAst = updatedAst;
    }
}

//@noUnusedParameters:true

class A {
    public a: number;

    public method(this: this): number {
        return this.a;
    }

    public method2(this: A): number {
        return this.a;
    }

    public method3(this: this): number {
        var fn = () => this.a;
        return fn();
    }

    public method4(this: A): number {
        var fn = () => this.a;
        return fn();
    }

    static staticMethod(this: A): number {
        return this.a;
    }
}

export function processTryCatchBindings(
  func: HIRFunction,
  labels: DisjointSet<Identifier>,
): void {
  let handlerParamsMap = new Map<BlockId, Identifier>();
  for (const [blockId, block] of func.body.blocks) {
    if (
      block.terminal.kind === 'try' &&
      block.terminal.handlerBinding !== null
    ) {
      handlerParamsMap.set(
        block.terminal.handler,
        block.terminal.handlerBinding.identifier,
      );
    } else if (block.terminal.kind === 'maybe-throw') {
      const maybeHandlerParam = handlerParamsMap.get(block.terminal.handler);
      if (!maybeHandlerParam) {
        continue;
      }
      for (const instruction of block.instructions) {
        labels.union([instruction.lvalue.identifier, maybeHandlerParam]);
      }
    }
  }
}

