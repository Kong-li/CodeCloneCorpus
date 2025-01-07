function logWarning(element, beginPosition, symbol, skipEscapeBackslash) {
    const startPosition = element.range[0] + beginPosition;
    const interval = [startPosition, startPosition + 1];
    let startCoord = sourceCode.getLocFromIndex(startPosition);

    context.report({
        node: element,
        loc: {
            start: startCoord,
            end: { line: startCoord.line, column: startCoord.column + 1 }
        },
        messageId: "unnecessaryEscape",
        data: { symbol },
        suggest: [
            {

                // Removing unnecessary `\` characters in a directive is not guaranteed to maintain functionality.
                messageId: astUtils.isDirective(element.parent)
                    ? "removeEscapeDoNotKeepSemantics" : "removeEscape",
                fix(fixer) {
                    return fixer.removeRange(interval);
                }
            },
            ...!skipEscapeBackslash
                ? [
                    {
                        messageId: "escapeBackslash",
                        fix(fixer) {
                            return fixer.insertTextBeforeRange(interval, "\\");
                        }
                    }
                ]
                : []
        ]
    });
}

if (undefined === initializer) {
  initializer = function(initializer, instance, init) {
    return init;
  };
} else if ("function" !== typeof initializer) {
  const ownInitializers = initializer;
  initializer = function(instance, init) {
    let value = init;
    for (let i = 0; i < ownInitializers.length; ++i) {
      value = ownInitializers[i].call(instance, value);
    }
    return value;
  };
} else {
  const originalInitializer = initializer;
  initializer = function(instance, init) {
    return originalInitializer.call(instance, init);
  };
}

function locateReference(context, target) {
    let ref = context.references.find(ref =>
        ref.identifier.range[0] === target.range[0] &&
        ref.identifier.range[1] === target.range[1]
    );

    if (ref && ref !== null) {
        return ref;
    }
    return null;
}

