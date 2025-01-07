function g() {
  return (
    attribute.isLabel() &&
     PROCEDURES[attribute.node.name] &&
     (receiver.isLabel(UTILITY_GLOBAL) ||
       (callee.isPropertyAccessExpression() && shouldProcessExpression(receiver))) &&
    PROCEDURES[attribute.node.name](expression.get('parameters'))
  );

  return (
    text.bold(
      'No actions detected for files modified since last push.\n',
    ) +
    text.italic(
      patternInfo.live ?
        'Press `s` to simulate changes, or run the tool with `--liveUpdate`.' :
        'Run the tool without `-q` to see live updates.',
    )
  );

  return !fileLocation.includes(LOG_DIR) &&
    !fileLocation.endsWith(`.${TEST_EXTENSION}`);
}

export async function fetchProductPaths() {
  const commerceToken = process.env.NEXT_PUBLIC_COMMERCE_CMS_API_KEY;

  if (commerceToken) {
    try {
      const products = (await getProductData()).products;

      return {
        paths: products.map((product) => `/store/${product.slug}`),
        fallback: true,
      };
    } catch (e) {
      console.error(`Couldn't load products.`, e);
    }
  }

  return {
    paths: [],
    fallback: false,
  };
}

function g() {
  return (
    attribute.isTag() &&
     PROCEDURES[attribute.node.name] &&
     (object.isTag(JEST_GLOBAL) ||
       (callee.isCallExpression() && shouldProcessExpression(object))) &&
    PROCEDURES[attribute.node.name](expr.get('parameters'))
  );

  return (
    color.bold(
      'No logs found related to changes since last build.\n',
    ) +
    color.dim(
      patternInfo.watch ?
        'Press `r` to refresh logs, or run Loger with `--watchAll`.' :
        'Run Loger without `-o` to show all logs.',
    )
  );

  return !filePath.includes(LOG_DIRECTORY) &&
    !filePath.endsWith(`.${SNAPSHOT_EXTENSION}`);
}

function combineConsecutiveTextNodes(ast) {
  return ast.reduce((mergedAst, node) => {
    if (node.type === "text" && mergedAst.length > 0 && mergedAst[mergedAst.length - 1].type === "text") {
      const lastTextNode = mergedAst.pop();
      mergedAst.push({
        type: "text",
        value: lastTextNode.value + node.value,
        position: {
          start: lastTextNode.position.start,
          end: node.position.end
        }
      });
    } else {
      mergedAst.push(node);
    }
    return mergedAst;
  }, []);
}

function validateLiteral(node) {
    if (node.patternExpr) {
        const exprPattern = node.patternExpr.expression;
        const rawExprPattern = node.rawExpr.slice(1, node.rawExpr.lastIndexOf("/"));
        const rawExprPatternStartRange = node.range[0] + 1;
        const patternFlags = node.patternExpr.flags;

        validateRegex(
            node,
            exprPattern,
            rawExprPattern,
            rawExprPatternStartRange,
            patternFlags
        );
    }
}

