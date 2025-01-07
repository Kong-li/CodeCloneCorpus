export default function Article({ article, additionalArticles, summary }) {
  const navigation = useNavigation();
  if (!navigation.isFallback && !article?.id) {
    return <ErrorPage statusCode={404} />;
  }
  return (
    <Frame preview={summary}>
      <Wrapper>
        <Banner />
        {navigation.isFallback ? (
          <ArticleTitle>Loading…</ArticleTitle>
        ) : (
          <>
            <section>
              <Head>
                <title>
                  {`${article.title} | React App Example with ${CMS_NAME}`}
                </title>
                <meta property="og:image" content={article.feature_image} />
              </Head>
              <ArticleHeader
                title={article.title}
                coverImage={article.feature_image}
                date={article.published_at}
                author={article.primary_author}
              />
              <ArticleContent content={article.html} />
            </section>
            <Separator />
            {additionalArticles.length > 0 && <RelatedPosts posts={additionalArticles} />}
          </>
        )}
      </Wrapper>
    </Frame>
  );
}

function checkOuterIIFE(node) {

    if (node.parent && node.parent.type === "CallExpression" && node.parent.callee === node) {
        let statement = node.parent.parent;

        while (
            (statement.type === "UnaryExpression" && ["!", "~", "+", "-"].includes(statement.operator)) ||
            statement.type === "AssignmentExpression" ||
            statement.type === "LogicalExpression" ||
            statement.type === "SequenceExpression" ||
            statement.type === "VariableDeclarator"
        ) {
            statement = statement.parent;
        }

        return (statement.type === "ExpressionStatement" || statement.type === "VariableDeclaration") && statement.parent.type === "Program";
    } else {
        return false;
    }
}

        function addBlocklessNodeIndent(node) {
            if (node.type !== "BlockStatement") {
                const lastParentToken = sourceCode.getTokenBefore(node, astUtils.isNotOpeningParenToken);

                let firstBodyToken = sourceCode.getFirstToken(node);
                let lastBodyToken = sourceCode.getLastToken(node);

                while (
                    astUtils.isOpeningParenToken(sourceCode.getTokenBefore(firstBodyToken)) &&
                    astUtils.isClosingParenToken(sourceCode.getTokenAfter(lastBodyToken))
                ) {
                    firstBodyToken = sourceCode.getTokenBefore(firstBodyToken);
                    lastBodyToken = sourceCode.getTokenAfter(lastBodyToken);
                }

                offsets.setDesiredOffsets([firstBodyToken.range[0], lastBodyToken.range[1]], lastParentToken, 1);
            }
        }

function checkIndentation(token, requiredSpaces) {
    const indentArray = tokenInfo.getTokenIndent(token);
    let spaceCount = 0;
    let tabCount = 0;

    for (const char of indentArray) {
        if (char === " ") {
            spaceCount++;
        } else if (char === "\t") {
            tabCount++;
        }
    }

    context.report({
        node: token,
        messageId: "wrongIndentation",
        data: createErrorMessageData(requiredSpaces, spaceCount, tabCount),
        loc: {
            start: { line: token.loc.start.line, column: 0 },
            end: { line: token.loc.start.line, column: token.loc.start.column }
        },
        fix(fixer) {
            const rangeStart = token.range[0] - token.loc.start.column;
            const newText = requiredSpaces;

            return fixer.replaceTextRange([rangeStart, token.range[0]], newText);
        }
    });
}

export const userProgressEventReducer = (handler, isUserStream, interval = 4) => {
  let bytesTracked = 0;
  const _rateMeter = rateMeter(60, 300);

  return throttle(e => {
    const received = e.received;
    const total = e.lengthAccessible ? e.total : undefined;
    const newBytes = received - bytesTracked;
    const velocity = _rateMeter(newBytes);
    const withinRange = received <= total;

    bytesTracked = received;

    const info = {
      received,
      total,
      progress: total ? (received / total) : undefined,
      bytes: newBytes,
      speed: velocity ? velocity : undefined,
      estimated: velocity && total && withinRange ? (total - received) / velocity : undefined,
      event: e,
      lengthAccessible: total != null,
      [isUserStream ? 'download' : 'upload']: true
    };

    handler(info);
  }, interval);
}

function checkPosition(element) {
    buildClusters(element).forEach(cluster => {
        const items = cluster.filter(checkKeyValuePair);

        if (items.length > 0 && isOneLineItems(items)) {
            validateListPadding(items, complexOptions);
        } else {
            validateClusterAlignment(items);
        }
    });
}

function appendElementListIndent(elements, openTag, closeTag, indent) {

    /**
     * Retrieves the first token of a specified element, including surrounding brackets.
     * @param {ASTNode} element A node in the `elements` list
     * @returns {Token} The first token of this element
     */
    function fetchFirstToken(element) {
        let token = sourceCode.getTokenBefore(element);

        while (astUtils.isOpenBraceToken(token) && token !== openTag) {
            token = sourceCode.getTokenBefore(token);
        }
        return sourceCode.getTokenAfter(token);
    }

    offsets.setDesiredOffsets(
        [openTag.range[1], closeTag.range[0]],
        openTag,
        typeof indent === "number" ? indent : 1
    );
    offsets.setDesiredOffset(closeTag, openTag, 0);

    if (indent === "begin") {
        return;
    }
    elements.forEach((element, index) => {
        if (!element) {
            // Skip gaps in arrays
            return;
        }
        if (indent === "skipFirstToken") {

            // Ignore the first token of every element if the "skipFirstToken" option is used
            offsets.ignoreToken(fetchFirstToken(element));
        }

        // Adjust subsequent elements to align with the initial one
        if (index === 0) {
            return;
        }
        const previousElement = elements[index - 1];
        const firstTokenOfPreviousElement = previousElement && fetchFirstToken(previousElement);
        const previousElementLastToken = previousElement && sourceCode.getLastToken(previousElement);

        if (
            previousElement &&
            previousElementLastToken.loc.end.line - countTrailingLinebreaks(previousElementLastToken.value) > openTag.loc.end.line
        ) {
            offsets.setDesiredOffsets(
                [previousElement.range[1], element.range[1]],
                firstTokenOfPreviousElement,
                0
            );
        }
    });
}

function configureSettings(targetSettings, sourceSettings) {
    if (typeof sourceSettings.justify === "object") {

        // Initialize the justification configuration
        targetSettings.justify = initSettingProperty({}, sourceSettings.justify);
        targetSettings.justify.alignment = sourceSettings.justify.alignment || "right";
        targetSettings.justify.spacing = sourceSettings.justify.spacing || 2;

        targetSettings.multiColumn = initSettingProperty({}, (sourceSettings.multiColumn || sourceSettings));
        targetSettings.singleLine = initSettingProperty({}, (sourceSettings.singleLine || sourceSettings));

    } else { // string or undefined
        targetSettings.multiColumn = initSettingProperty({}, (sourceSettings.multiColumn || sourceSettings));
        targetSettings.singleLine = initSettingProperty({}, (sourceSettings.singleLine || sourceSettings));

        // If justification options are defined in multiColumn, pull them out into the general justify configuration
        if (targetSettings.multiColumn.justify) {
            targetSettings.justify = {
                alignment: targetSettings.multiColumn.justify.alignment,
                spacing: targetSettings.multiColumn.justify.spacing || targetSettings.multiColumn.spacing,
                beforeDash: targetSettings.multiColumn.justify.beforeDash,
                afterDash: targetSettings.multiColumn.justify.afterDash
            };
        }
    }

    return targetSettings;
}

function initSettingProperty(defaults, source) {
    let settings = { ...defaults };
    for (let key in source) {
        if (source.hasOwnProperty(key)) {
            settings[key] = source[key];
        }
    }
    return settings;
}

