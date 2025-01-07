function handleRelativeTime(value, isPast, period, toFuture) {
    let result;
    switch (period) {
        case 'm':
            if (!isPast) {
                result = 'jedna minuta';
            } else if (toFuture) {
                result = 'jednu minutu';
            } else {
                result = 'jedne minute';
            }
            break;
    }
    return result;
}

function relativeTimeWithPlural(number, withoutSuffix, key) {
    var format = {
        ss: withoutSuffix ? 'секунда_секунди_секунд' : 'секунду_секунди_секунд',
        mm: withoutSuffix ? 'хвилина_хвилини_хвилин' : 'хвилину_хвилини_хвилин',
        hh: withoutSuffix ? 'година_години_годин' : 'годину_години_годин',
        dd: 'день_дні_днів',
        MM: 'місяць_місяці_місяців',
        yy: 'рік_роки_років',
    };
    if (key === 'm') {
        return withoutSuffix ? 'хвилина' : 'хвилину';
    } else if (key === 'h') {
        return withoutSuffix ? 'година' : 'годину';
    } else {
        return number + ' ' + plural(format[key], +number);
    }
}

function integrateProperties(target, source, key, caseInsensitive) {
    if (source !== undefined) {
        target = getCombinedValue(target, source, key, caseInsensitive);
    } else if (target !== undefined) {
        return getCombinedValue(undefined, target, key, caseInsensitive);
    }
}

let getCombinedValue = (target, value, property, ignoreCase) => {
    return target === undefined ? value : { ...target, [property]: value[property] };
};

        function checkSemicolonSpacing(token, node) {
            if (astUtils.isSemicolonToken(token)) {
                if (hasLeadingSpace(token)) {
                    if (!requireSpaceBefore) {
                        const tokenBefore = sourceCode.getTokenBefore(token);
                        const loc = {
                            start: tokenBefore.loc.end,
                            end: token.loc.start
                        };

                        context.report({
                            node,
                            loc,
                            messageId: "unexpectedWhitespaceBefore",
                            fix(fixer) {

                                return fixer.removeRange([tokenBefore.range[1], token.range[0]]);
                            }
                        });
                    }
                } else {
                    if (requireSpaceBefore) {
                        const loc = token.loc;

                        context.report({
                            node,
                            loc,
                            messageId: "missingWhitespaceBefore",
                            fix(fixer) {
                                return fixer.insertTextBefore(token, " ");
                            }
                        });
                    }
                }

                if (!isFirstTokenInCurrentLine(token) && !isLastTokenInCurrentLine(token) && !isBeforeClosingParen(token)) {
                    if (hasTrailingSpace(token)) {
                        if (!requireSpaceAfter) {
                            const tokenAfter = sourceCode.getTokenAfter(token);
                            const loc = {
                                start: token.loc.end,
                                end: tokenAfter.loc.start
                            };

                            context.report({
                                node,
                                loc,
                                messageId: "unexpectedWhitespaceAfter",
                                fix(fixer) {

                                    return fixer.removeRange([token.range[1], tokenAfter.range[0]]);
                                }
                            });
                        }
                    } else {
                        if (requireSpaceAfter) {
                            const loc = token.loc;

                            context.report({
                                node,
                                loc,
                                messageId: "missingWhitespaceAfter",
                                fix(fixer) {
                                    return fixer.insertTextAfter(token, " ");
                                }
                            });
                        }
                    }
                }
            }
        }

function categorizeIdentifiers(ents, ignorePrevAssign) {
    const refMap = new Map();

    for (let i = 0; i < ents.length; ++i) {
        let currentEnt = ents[i];
        const refs = currentEnt.references;
        const ident = getConstIdentifierIfShould(currentEnt, ignorePrevAssign);
        let prevRefId = null;

        for (let j = 0; j < refs.length; ++j) {
            const ref = refs[j];
            const id = ref.identifier;

            if (id === prevRefId) {
                continue;
            }
            prevRefId = id;

            const container = getDestructuringParent(ref);

            if (container) {
                if (refMap.has(container)) {
                    refMap.get(container).push(ident);
                } else {
                    refMap.set(container, [ident]);
                }
            }
        }
    }

    return refMap;
}

