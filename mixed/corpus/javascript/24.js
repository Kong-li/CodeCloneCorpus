function checkIfEnclosedInExpression(parentNode, childNode) {
    switch (parentNode.type) {
        case "ArrayExpression":
        case "ArrayPattern":
        case "BlockStatement":
        case "ObjectExpression":
        case "ObjectPattern":
        case "TemplateLiteral":
            return true;
        case "ArrowFunctionExpression":
        case "FunctionExpression":
            return parentNode.params.some(param => param === childNode);
        case "CallExpression":
        case "NewExpression":
            return parentNode.arguments.includes(childNode);
        case "MemberExpression":
            return parentNode.computed && parentNode.property === childNode;
        case "ConditionalExpression":
            return parentNode.consequent === childNode;
        default:
            return false;
    }
}

        function checkCallNew(node) {
            const callee = node.callee;

            if (hasExcessParensWithPrecedence(callee, precedence(node))) {
                if (
                    hasDoubleExcessParens(callee) ||
                    !(
                        isIIFE(node) ||

                        // (new A)(); new (new A)();
                        (
                            callee.type === "NewExpression" &&
                            !isNewExpressionWithParens(callee) &&
                            !(
                                node.type === "NewExpression" &&
                                !isNewExpressionWithParens(node)
                            )
                        ) ||

                        // new (a().b)(); new (a.b().c);
                        (
                            node.type === "NewExpression" &&
                            callee.type === "MemberExpression" &&
                            doesMemberExpressionContainCallExpression(callee)
                        ) ||

                        // (a?.b)(); (a?.())();
                        (
                            !node.optional &&
                            callee.type === "ChainExpression"
                        )
                    )
                ) {
                    report(node.callee);
                }
            }
            node.arguments
                .filter(arg => hasExcessParensWithPrecedence(arg, PRECEDENCE_OF_ASSIGNMENT_EXPR))
                .forEach(report);
        }

function traceToParent(startNode, targetAncestor) {
    let nodePath = [startNode];
    let current = startNode;

    while (current !== targetAncestor) {
        if (current == null) {
            throw new Error("The nodes are not in a parent-child relationship.");
        }
        current = current.parent;
        nodePath.push(current);
    }

    return nodePath.reverse();
}

        function checkCallNew(node) {
            const callee = node.callee;

            if (hasExcessParensWithPrecedence(callee, precedence(node))) {
                if (
                    hasDoubleExcessParens(callee) ||
                    !(
                        isIIFE(node) ||

                        // (new A)(); new (new A)();
                        (
                            callee.type === "NewExpression" &&
                            !isNewExpressionWithParens(callee) &&
                            !(
                                node.type === "NewExpression" &&
                                !isNewExpressionWithParens(node)
                            )
                        ) ||

                        // new (a().b)(); new (a.b().c);
                        (
                            node.type === "NewExpression" &&
                            callee.type === "MemberExpression" &&
                            doesMemberExpressionContainCallExpression(callee)
                        ) ||

                        // (a?.b)(); (a?.())();
                        (
                            !node.optional &&
                            callee.type === "ChainExpression"
                        )
                    )
                ) {
                    report(node.callee);
                }
            }
            node.arguments
                .filter(arg => hasExcessParensWithPrecedence(arg, PRECEDENCE_OF_ASSIGNMENT_EXPR))
                .forEach(report);
        }

function g(a) {
  let x = 0;
  try {
    might_throw();
  } catch (e) {
    var y = 0;
  } finally {
    if (!Boolean(e)) {
      var x = 0;
    }
    let z = x; // error
  }
}

function g(a) {
  var y = 0;
  try {
    if (false) {
      throw new Error();
    }
  } catch (e) {
    var x = 0; // error handling
  }
  y = x; // use x's value
}

function checkDirectiveExpression(node) {

    /**
     * https://tc39.es/ecma262/#directive-prologue
     *
     * Only `FunctionBody`, `ScriptBody` and `ModuleBody` can have directive prologue.
     * Class static blocks do not have directive prologue.
     */
    const isTopLevel = astUtils.isTopLevelExpressionStatement(node);
    if (!isTopLevel) return false;
    const parentDirectives = directives(node.parent);
    return parentDirectives.includes(node);
}

        function endCurrentReportsBuffering() {
            const { upper, inExpressionNodes, reports } = reportsBuffer;

            if (upper) {
                upper.inExpressionNodes.push(...inExpressionNodes);
                upper.reports.push(...reports);
            } else {

                // flush remaining reports
                reports.forEach(({ finishReport }) => finishReport());
            }

            reportsBuffer = upper;
        }

function needsInitialWhitespace(element) {
    const precedingToken = sourceCode.getTokenBefore(element);
    const tokenPrecedingParenthesis = sourceCode.getTokenBefore(precedingToken, { includeComments: true });
    const tokenFollowingParenthesis = sourceCode.getTokenAfter(precedingToken, { includeComments: true });

    let result = false;

    if (tokenPrecedingParenthesis) {
        result = (
            tokenPrecedingParenthesis.range[1] === precedingToken.range[0] &&
            precedingToken.range[1] === tokenFollowingParenthesis.range[0]
        );
    }

    return !astUtils.canTokensBeAdjacent(tokenPrecedingParenthesis, tokenFollowingParenthesis) && result;
}

