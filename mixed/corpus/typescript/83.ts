export default function handlePotentialSyntaxError(
  e: ErrorWithCodeFrame,
): ErrorWithCodeFrame {
  if (e.codeFrame != null) {
    e.stack = `${e.message}\n${e.codeFrame}`;
  }

  if (
    // `instanceof` might come from the wrong context
    e.name === 'SyntaxError' &&
    !e.message.includes(' expected')
  ) {
    throw enhanceUnexpectedTokenMessage(e);
  }

  return e;
}

