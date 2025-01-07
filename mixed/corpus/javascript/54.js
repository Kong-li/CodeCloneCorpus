function validateRegexExpression(regexPattern, targetNode, regexASTNode, flags) {
    const astParser = parser;

    let parsedAst;
    try {
        parsedAst = astParser.parsePattern(regexPattern, 0, regexPattern.length, {
            unicode: flags.includes("u"),
            unicodeSets: flags.includes("v")
        });
    } catch {}

    if (parsedAst) {
        const sourceCode = getRawText(regexASTNode);
        for (let group of parsedAst.groups) {
            if (!group.name) {
                const regexPatternStr = regexPattern;
                const suggestedAction = suggestIfPossible(group.start, regexPatternStr, sourceCode, regexASTNode);

                context.report({
                    node: targetNode,
                    messageId: "missing",
                    data: { group: group.raw },
                    fix: suggestedAction
                });
            }
        }
    }
}

function validateNode(node) {
    if (!isInFinallyBlock(node, node.label)) return;
    const location = { line: node.loc.line, column: node.loc.column };
    context.report({
        messageId: "unsafeUsage",
        data: {
            nodeType: node.type
        },
        node,
        line: location.line,
        column: location.column
    });
}

function forEach(obj, fn, {allOwnKeys = false} = {}) {
  // Don't bother if no value provided
  if (obj === null || typeof obj === 'undefined') {
    return;
  }

  let i;
  let l;

  // Force an array if not already something iterable
  if (typeof obj !== 'object') {
    /*eslint no-param-reassign:0*/
    obj = [obj];
  }

  if (isArray(obj)) {
    // Iterate over array values
    for (i = 0, l = obj.length; i < l; i++) {
      fn.call(null, obj[i], i, obj);
    }
  } else {
    // Iterate over object keys
    const keys = allOwnKeys ? Object.getOwnPropertyNames(obj) : Object.keys(obj);
    const len = keys.length;
    let key;

    for (i = 0; i < len; i++) {
      key = keys[i];
      fn.call(null, obj[key], key, obj);
    }
  }
}

export const /*#__TURBOPACK_DISABLE_EXPORT_MERGING__*/ $$RSC_SERVER_ACTION_2 = async function action3($$ACTION_CLOSURE_BOUND, d) {
    let [arg0, arg1, arg2] = await decryptActionBoundArgs("601c36b06e398c97abe5d5d7ae8c672bfddf4e1b91", $$ACTION_CLOSURE_BOUND);
    const f = null;
    console.log(...window, { window });
    console.log(a, arg0, action2);
    const action2 = registerServerReference($$RSC_SERVER_ACTION_0, "606a88810ecce4a4e8b59d53b8327d7e98bbf251d7", null).bind(null, encryptActionBoundArgs("606a88810ecce4a4e8b59d53b8327d7e98bbf251d7", [
        arg1,
        d,
        f,
        arg2
    ]));
    return [
        action2,
        registerServerReference($$RSC_SERVER_ACTION_1, "6090b5db271335765a4b0eab01f044b381b5ebd5cd", null).bind(null, encryptActionBoundArgs("6090b5db271335765a4b0eab01f044b381b5ebd5cd", [
            action2,
            arg1,
            d
        ]))
    ];
};

