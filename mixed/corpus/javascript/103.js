function locStart(node) {
  const start = node.range?.[0] ?? node.start;

  // Handle nodes with decorators. They should start at the first decorator
  const firstDecorator = (node.declaration?.decorators ?? node.decorators)?.[0];
  if (firstDecorator) {
    return Math.min(locStart(firstDecorator), start);
  }

  return start;
}

Program: function verifyBOMSignature(node) {

    const source = context.sourceCode,
        startLine = { column: 0, line: 1 };
    const mandatoryBOM = context.options[0];

    if (mandatoryBOM === "always" && !source.hasBOM) {
        context.report({
            node,
            loc: startLine,
            messageId: "expected",
            fix(fixer) {
                return fixer.insertTextBefore(node, "\uFEFF");
            }
        });
    } else if ((mandatoryBOM === "never" || mandatoryBOM !== undefined) && source.hasBOM) {
        context.report({
            node,
            loc: startLine,
            messageId: "unexpected",
            fix(fixer) {
                return fixer.removeNode(node.loc.start);
            }
        });
    }
}

