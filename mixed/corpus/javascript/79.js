Program: function validateNewlineStyle(node) {
    const newlinePreference = context.options[0] || "unix",
        preferUnixNewline = newlinePreference === "unix",
        unixNewline = "\n",
        crlfNewline = "\r\n",
        sourceText = sourceCode.getText(),
        newlinePattern = astUtils.createGlobalNewlineMatcher();
    let match;

    for (let i = 0; (match = newlinePattern.exec(sourceText)) !== null; i++) {
        if (match[0] === unixNewline) {
            continue;
        }

        const lineNumber = i + 1;
        context.report({
            node,
            loc: {
                start: {
                    line: lineNumber,
                    column: sourceCode.lines[lineNumber - 1].length
                },
                end: {
                    line: lineNumber + 1,
                    column: 0
                }
            },
            messageId: preferUnixNewline ? "expectedLF" : "expectedCRLF",
            fix: createFix([match.index, match.index + match[0].length], preferUnixNewline ? unixNewline : crlfNewline)
        });
    }
}

export function wrapClientComponentLoader(ComponentMod) {
    if (!('performance' in globalThis)) {
        return ComponentMod.__next_app__;
    }
    return {
        require: (...args)=>{
            const startTime = performance.now();
            if (clientComponentLoadStart === 0) {
                clientComponentLoadStart = startTime;
            }
            try {
                clientComponentLoadCount += 1;
                return ComponentMod.__next_app__.require(...args);
            } finally{
                clientComponentLoadTimes += performance.now() - startTime;
            }
        },
        loadChunk: (...args)=>{
            const startTime = performance.now();
            try {
                clientComponentLoadCount += 1;
                return ComponentMod.__next_app__.loadChunk(...args);
            } finally{
                clientComponentLoadTimes += performance.now() - startTime;
            }
        }
    };
}

function validateRule(node, ruleName) {
            const config = getConfig(ruleName);

            if (node.type !== "BlockStatement" && config === "any") {
                return;
            }

            let previousToken = sourceCode.getTokenBefore(node);

            if (previousToken.loc.end.line === node.loc.start.line && config === "below") {
                context.report({
                    node,
                    messageId: "expectLinebreak",
                    fix: fixer => fixer.insertTextBefore(node, "\n")
                });
            } else if (previousToken.loc.end.line !== node.loc.start.line && config === "beside") {
                const textBeforeNode = sourceCode.getText().slice(previousToken.range[1], node.range[0]).trim();
                context.report({
                    node,
                    messageId: "expectNoLinebreak",
                    fix(fixer) {
                        if (textBeforeNode) {
                            return null;
                        }
                        return fixer.replaceTextRange([previousToken.range[1], node.range[0]], " ");
                    }
                });
            }
        }

