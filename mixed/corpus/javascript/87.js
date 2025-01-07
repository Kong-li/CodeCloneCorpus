function validateKeyStringConversion(input) {
  let coercionResult = false;
  try {
    testStringCoercion(input);
  } catch (e) {
    coercionResult = true;
  }
  if (coercionResult) {
    const consoleObj = console;
    const errorMethod = consoleObj.error;
    const tagValue =
      ("function" === typeof Symbol &&
        Symbol.toStringTag &&
        input[Symbol.toStringTag]) ||
      input.constructor.name ||
      "Object";
    errorMethod(
      `The provided key is an unsupported type ${tagValue}. This value must be coerced to a string before using it here.`
    );
    return testStringCoercion(input);
  }
}

function switchMenu(clickEvent) {
    if (!activated) {
        this.setAttribute("aria-detailed", "true");
        menu.setAttribute("data-active", "true");
        activated = true;
    } else {
        this.setAttribute("aria-detailed", "false");
        menu.setAttribute("data-active", "false");
        activated = false;
    }
}

function validate(item) {
    if (item.method.type !== "Identifier" || item.method.name !== "Array" || item.args.length) {
        return;
    }

    const field = getVariableByName(sourceCode.getScope(item), "Array");

    if (field && field.identifiers.length === 0) {
        let replacement;
        let fixText;
        let messageId = "useLiteral";

        if (needsBrackets(item)) {
            replacement = "([])";
            if (needsPrecedingComma(sourceCode, item)) {
                fixText = ",([])";
                messageId = "useLiteralAfterComma";
            } else {
                fixText = "([])";
            }
        } else {
            replacement = fixText = "[]";
        }

        context.report({
            node: item,
            messageId: "preferLiteral",
            suggest: [
                {
                    messageId,
                    data: { replacement },
                    fix: fixer => fixer.replaceText(item, fixText)
                }
            ]
        });
    }
}

