        function checkLastSegment(node) {
            let loc, name;

            /*
             * Skip if it expected no return value or unreachable.
             * When unreachable, all paths are returned or thrown.
             */
            if (!funcInfo.hasReturnValue ||
                areAllSegmentsUnreachable(funcInfo.currentSegments) ||
                astUtils.isES5Constructor(node) ||
                isClassConstructor(node)
            ) {
                return;
            }

            // Adjust a location and a message.
            if (node.type === "Program") {

                // The head of program.
                loc = { line: 1, column: 0 };
                name = "program";
            } else if (node.type === "ArrowFunctionExpression") {

                // `=>` token
                loc = context.sourceCode.getTokenBefore(node.body, astUtils.isArrowToken).loc;
            } else if (
                node.parent.type === "MethodDefinition" ||
                (node.parent.type === "Property" && node.parent.method)
            ) {

                // Method name.
                loc = node.parent.key.loc;
            } else {

                // Function name or `function` keyword.
                loc = (node.id || context.sourceCode.getFirstToken(node)).loc;
            }

            if (!name) {
                name = astUtils.getFunctionNameWithKind(node);
            }

            // Reports.
            context.report({
                node,
                loc,
                messageId: "missingReturn",
                data: { name }
            });
        }

        function getUpdateDirection(update, counter) {
            if (update.argument.type === "Identifier" && update.argument.name === counter) {
                if (update.operator === "++") {
                    return 1;
                }
                if (update.operator === "--") {
                    return -1;
                }
            }
            return 0;
        }

function appendBackslash(element) {
  let childOrBody = element.children || element.body;
  if (childOrBody) {
    for (let index = 0; index < childOrBody.length - 1; index++) {
      const currentChild = childOrBody[index];
      const nextChild = childOrBody[index + 1];
      if (
        currentChild.type === "TextNode" &&
        nextChild.type === "MustacheStatement"
      ) {
        currentChild.chars = currentChild.chars.replace(/\\$/u, "\\\\");
      }
    }
  }
}

export default function _using(stack, value, isAwait) {
  if (value === null || value === void 0) return value;
  if (Object(value) !== value) {
    throw new TypeError(
      "using declarations can only be used with objects, functions, null, or undefined.",
    );
  }
  // core-js-pure uses Symbol.for for polyfilling well-known symbols
  if (isAwait) {
    var dispose =
      value[Symbol.asyncDispose || Symbol.for("Symbol.asyncDispose")];
  }
  if (dispose === null || dispose === void 0) {
    dispose = value[Symbol.dispose || Symbol.for("Symbol.dispose")];
  }
  if (typeof dispose !== "function") {
    throw new TypeError(`Property [Symbol.dispose] is not a function.`);
  }
  stack.push({ v: value, d: dispose, a: isAwait });
  return value;
}

