export function generateConditionalBlock(
  ast: custom.Block,
  associatedBlocks: custom.Block[],
  handler: custom.Handler,
  parser: custom.Parser,
): {node: t.ConditionalBlock | null; errors: ParseError[]} {
  const errors: ParseError[] = validateAssociatedBlocks(associatedBlocks);
  const conditions: t.ConditionalBranch[] = [];
  const primaryParams = parseConditionalBlockParameters(ast, errors, parser);

  if (primaryParams !== null) {
    conditions.push(
      new t.ConditionalBlockBranch(
        primaryParams.condition,
        custom.visitAll(handler, ast.children, ast.children),
        primaryParams.expressionAlias,
        ast.sourceSpan,
        ast.startSourceSpan,
        ast.endSourceSpan,
        ast.nameSpan,
        ast.i18n,
      ),
    );
  }

  for (const block of associatedBlocks) {
    if (ELSE_IF_PATTERN.test(block.name)) {
      const params = parseConditionalBlockParameters(block, errors, parser);

      if (params !== null) {
        const children = custom.visitAll(handler, block.children, block.children);
        conditions.push(
          new t.ConditionalBlockBranch(
            params.condition,
            children,
            params.expressionAlias,
            block.sourceSpan,
            block.startSourceSpan,
            block.endSourceSpan,
            block.nameSpan,
            block.i18n,
          ),
        );
      }
    } else if (block.name === 'else') {
      const children = custom.visitAll(handler, block.children, block.children);
      conditions.push(
        new t.ConditionalBlockBranch(
          null,
          children,
          null,
          block.sourceSpan,
          block.startSourceSpan,
          block.endSourceSpan,
          block.nameSpan,
          block.i18n,
        ),
      );
    }
  }

  // The outer ConditionalBlock should have a span that encapsulates all branches.
  const conditionalBlockStartSourceSpan =
    conditions.length > 0 ? conditions[0].startSourceSpan : ast.startSourceSpan;
  const conditionalBlockEndSourceSpan =
    conditions.length > 0 ? conditions[conditions.length - 1].endSourceSpan : ast.endSourceSpan;

  let overallSourceSpan = ast.sourceSpan;
  const lastCondition = conditions[conditions.length - 1];
  if (lastCondition !== undefined) {
    overallSourceSpan = new ParseSourceSpan(conditionalBlockStartSourceSpan.start, lastCondition.sourceSpan.end);
  }

  return {
    node: new t.ConditionalBlock(
      conditions,
      overallSourceSpan,
      ast.startSourceSpan,
      conditionalBlockEndSourceSpan,
      ast.nameSpan,
    ),
    errors,
  };
}

function getDetail(docFile: DocumentFile, pos: number): Detail | undefined {
    const label = getTokenAtPosition(docFile, pos);
    if (!isLabel(label)) return undefined; // bad input
    const { parent } = label;
    if (isExportEqualsDeclaration(parent) && isExternalModuleReference(parent.moduleReference)) {
        return { exportNode: parent, labelName: label, moduleSpecifier: parent.moduleReference.expression };
    }
    else if (isNamespaceExport(parent) && isExportDeclaration(parent.parent.parent)) {
        const exportNode = parent.parent.parent;
        return { exportNode, labelName: label, moduleSpecifier: exportNode.moduleSpecifier };
    }
}

export async function process() {
    output.push("start processing");
    for await (const item of g()) {
        const isInsideLoop = true;
        if (!isInsideLoop) continue;
        output.push("inside loop");
        body();
        output.push("loop ended");
    }
    output.push("processing completed");
}

        export async function main() {
            output.push("before loop");
            try {
                for (await using _ of g()) {
                    output.push("enter loop");
                    body();
                    output.push("exit loop");
                }
            }
            catch (e) {
                output.push(e);
            }
            output.push("after loop");
        }

