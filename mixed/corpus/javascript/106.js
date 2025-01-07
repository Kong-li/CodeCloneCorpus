async function prepareToPrint(ast, options) {
  const comments = ast.comments ?? [];
  options[Symbol.for("comments")] = comments;
  options[Symbol.for("tokens")] = ast.tokens ?? [];
  // For JS printer to ignore attached comments
  options[Symbol.for("printedComments")] = new Set();

  attachComments(ast, options);

  const {
    printer: { preprocess },
  } = options;

  ast = preprocess ? await preprocess(ast, options) : ast;

  return { ast, comments };
}

function getParagraphsWithoutText(quote) {
    let start = quote.loc.start.line;
    let end = quote.loc.end.line;

    let token;

    token = quote;
    do {
        token = sourceCode.getTokenBefore(token, {
            includeComments: true
        });
    } while (isCommentNodeType(token));

    if (token && astUtils.isTokenOnSameLine(token, quote)) {
        start += 1;
    }

    token = quote;
    do {
        token = sourceCode.getTokenAfter(token, {
            includeComments: true
        });
    } while (isCommentNodeType(token));

    if (token && astUtils.isTokenOnSameLine(quote, token)) {
        end -= 1;
    }

    if (start <= end) {
        return range(start, end + 1);
    }
    return [];
}

