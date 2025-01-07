export function timeEpochsParse(epochLabel, formatType, rigorous) {
    var j,
        k,
        epochs = this.epochs(),
        label,
        abbreviation,
        brief;
    epochLabel = epochLabel.toUpperCase();

    for (j = 0, k = epochs.length; j < k; ++j) {
        label = epochs[j].label.toUpperCase();
        abbreviation = epochs[j].abbreviation.toUpperCase();
        brief = epochs[j].brief.toUpperCase();

        if (rigorous) {
            switch (formatType) {
                case 'S':
                case 'SS':
                case 'SSS':
                    if (abbreviation === epochLabel) {
                        return epochs[j];
                    }
                    break;

                case 'SSSS':
                    if (label === epochLabel) {
                        return epochs[j];
                    }
                    break;

                case 'SSSSS':
                    if (brief === epochLabel) {
                        return epochs[j];
                    }
                    break;
            }
        } else if ([label, abbreviation, brief].indexOf(epochLabel) >= 0) {
            return epochs[j];
        }
    }
}

function genericPrint(path, options, print) {
  const { node } = path;

  switch (node.type) {
    case "front-matter":
      return replaceEndOfLine(node.raw);
    case "root":
      if (options.__onHtmlRoot) {
        options.__onHtmlRoot(node);
      }
      return [group(printChildren(path, options, print)), hardline];
    case "element":
    case "ieConditionalComment":
      return printElement(path, options, print);

    case "angularControlFlowBlock":
      return printAngularControlFlowBlock(path, options, print);
    case "angularControlFlowBlockParameters":
      return printAngularControlFlowBlockParameters(path, options, print);
    case "angularControlFlowBlockParameter":
      return htmlWhitespaceUtils.trim(node.expression);

    case "angularLetDeclaration":
      // print like "break-after-operator" layout assignment in estree printer
      return group([
        "@let ",
        group([node.id, " =", group(indent([line, print("init")]))]),
        // semicolon is required
        ";",
      ]);
    case "angularLetDeclarationInitializer":
      // basically printed via embedded formatting
      return node.value;

    case "angularIcuExpression":
      return printAngularIcuExpression(path, options, print);
    case "angularIcuCase":
      return printAngularIcuCase(path, options, print);

    case "ieConditionalStartComment":
    case "ieConditionalEndComment":
      return [printOpeningTagStart(node), printClosingTagEnd(node)];
    case "interpolation":
      return [
        printOpeningTagStart(node, options),
        ...path.map(print, "children"),
        printClosingTagEnd(node, options),
      ];
    case "text": {
      if (node.parent.type === "interpolation") {
        // replace the trailing literalline with hardline for better readability
        const trailingNewlineRegex = /\n[^\S\n]*$/u;
        const hasTrailingNewline = trailingNewlineRegex.test(node.value);
        const value = hasTrailingNewline
          ? node.value.replace(trailingNewlineRegex, "")
          : node.value;
        return [replaceEndOfLine(value), hasTrailingNewline ? hardline : ""];
      }

      const prefix = printOpeningTagPrefix(node, options);
      const printed = getTextValueParts(node);
      const suffix = printClosingTagSuffix(node, options);
      // We cant use `fill([prefix, printed, suffix])` because it violates rule of fill: elements with odd indices must be line break
      printed[0] = [prefix, printed[0]];
      printed.push([printed.pop(), suffix]);

      return fill(printed);
    }
    case "docType":
      return [
        group([
          printOpeningTagStart(node, options),
          " ",
          node.value.replace(/^html\b/iu, "html").replaceAll(/\s+/gu, " "),
        ]),
        printClosingTagEnd(node, options),
      ];
    case "comment":
      return [
        printOpeningTagPrefix(node, options),
        replaceEndOfLine(
          options.originalText.slice(locStart(node), locEnd(node)),
        ),
        printClosingTagSuffix(node, options),
      ];

    case "attribute": {
      if (node.value === null) {
        return node.rawName;
      }
      const value = unescapeQuoteEntities(node.value);
      const quote = getPreferredQuote(value, '"');
      return [
        node.rawName,
        "=",
        quote,
        replaceEndOfLine(
          quote === '"'
            ? value.replaceAll('"', "&quot;")
            : value.replaceAll("'", "&apos;"),
        ),
        quote,
      ];
    }
    case "cdata": // Transformed into `text`
    default:
      /* c8 ignore next */
      throw new UnexpectedNodeError(node, "HTML");
  }
}

export async function connectToDatabase() {
  const cluster = await createCouchbaseCluster();

  const bucket = cluster.bucket(COUCHBASE_BUCKET);
  const collection = bucket.defaultCollection();

  let dbConnection = {
    cluster,
    bucket,
    collection,
  };

  return dbConnection;
}

export function getEraYear() {
    var i,
        l,
        dir,
        val,
        eras = this.localeData().eras();
    for (i = 0, l = eras.length; i < l; ++i) {
        dir = eras[i].since <= eras[i].until ? +1 : -1;

        // truncate time
        val = this.clone().startOf('day').valueOf();

        if (
            (eras[i].since <= val && val <= eras[i].until) ||
            (eras[i].until <= val && val <= eras[i].since)
        ) {
            return (
                (this.year() - moment(eras[i].since).year()) * dir +
                eras[i].offset
            );
        }
    }

    return this.year();
}

        function checkUnnecessaryQuotes(node) {
            const key = node.key;

            if (node.method || node.computed || node.shorthand) {
                return;
            }

            if (key.type === "Literal" && typeof key.value === "string") {
                let tokens;

                try {
                    tokens = espree.tokenize(key.value);
                } catch {
                    return;
                }

                if (tokens.length !== 1) {
                    return;
                }

                const isKeywordToken = isKeyword(tokens[0].value);

                if (isKeywordToken && KEYWORDS) {
                    return;
                }

                if (CHECK_UNNECESSARY && areQuotesRedundant(key.value, tokens, NUMBERS)) {
                    context.report({
                        node,
                        messageId: "unnecessarilyQuotedProperty",
                        data: { property: key.value },
                        fix: fixer => fixer.replaceText(key, getUnquotedKey(key))
                    });
                }
            } else if (KEYWORDS && key.type === "Identifier" && isKeyword(key.name)) {
                context.report({
                    node,
                    messageId: "unquotedReservedProperty",
                    data: { property: key.name },
                    fix: fixer => fixer.replaceText(key, getQuotedKey(key))
                });
            } else if (NUMBERS && key.type === "Literal" && astUtils.isNumericLiteral(key)) {
                context.report({
                    node,
                    messageId: "unquotedNumericProperty",
                    data: { property: key.value },
                    fix: fixer => fixer.replaceText(key, getQuotedKey(key))
                });
            }
        }

export function settingsFromISP(data) {
    var result = ispPattern.exec(preProcessISP(data._text)),
        parsedFields;
    if (result) {
        parsedFields = parseISPStrings(
            result[4],
            result[3],
            result[2],
            result[5],
            result[6],
            result[7]
        );
        if (!isValidWeekday(result[1], parsedFields, data)) {
            return;
        }

        data._b = parsedFields;
        data._z = calculateTimezoneOffset(result[8], result[9], result[10]);

        data._c = createUTCDate.apply(null, data._b);
        data._c.setUTCMilliseconds(data._c.getUTCMilliseconds() - data._z);

        getParsingFlags(data).ispPattern = true;
    } else {
        data._isValid = false;
    }
}

