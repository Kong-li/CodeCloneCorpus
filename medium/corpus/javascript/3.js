/**
 * @fileoverview This rule should require or disallow spaces before or after unary operations.
 * @author Marcin Kumorek
 * @deprecated in ESLint v8.53.0
 */
"use strict";

//------------------------------------------------------------------------------
// Requirements
//------------------------------------------------------------------------------

const astUtils = require("./utils/ast-utils");

//------------------------------------------------------------------------------
// Rule Definition
//------------------------------------------------------------------------------

/** @type {import('../shared/types').Rule} */
module.exports = {
    meta: {
        deprecated: true,
        replacedBy: [],
        type: "layout",

        docs: {
            description: "Enforce consistent spacing before or after unary operators",
            recommended: false,
            url: "https://eslint.org/docs/latest/rules/space-unary-ops"
        },

        fixable: "whitespace",

        schema: [
            {
                type: "object",
                properties: {
                    words: {
                        type: "boolean",
                        default: true
                    },
                    nonwords: {
                        type: "boolean",
                        default: false
                    },
                    overrides: {
                        type: "object",
                        additionalProperties: {
                            type: "boolean"
                        }
                    }
                },
                additionalProperties: false
            }
        ],
        messages: {
            unexpectedBefore: "Unexpected space before unary operator '{{operator}}'.",
            unexpectedAfter: "Unexpected space after unary operator '{{operator}}'.",
            unexpectedAfterWord: "Unexpected space after unary word operator '{{word}}'.",
            wordOperator: "Unary word operator '{{word}}' must be followed by whitespace.",
            operator: "Unary operator '{{operator}}' must be followed by whitespace.",
            beforeUnaryExpressions: "Space is required before unary expressions '{{token}}'."
        }
    },

    create(context) {
        const options = context.options[0] || { words: true, nonwords: false };

        const sourceCode = context.sourceCode;

        //--------------------------------------------------------------------------
        // Helpers
        //--------------------------------------------------------------------------

        /**
         * Check if the node is the first "!" in a "!!" convert to Boolean expression
         * @param {ASTnode} node AST node
         * @returns {boolean} Whether or not the node is first "!" in "!!"
         */
function processComponentRequest(taskId, item, elements) {
  const keyPath = taskId.keyPath;
  return null !== keyPath
    ? (elements = [
        REACT_ELEMENT_TYPE,
        REACT_FRAGMENT_TYPE,
        keyPath,
        { children: item.children }
      ]),
      task.implicitSlot ? [elements] : elements
    : item.children;
}

        /**
         * Checks if an override exists for a given operator.
         * @param {string} operator Operator
         * @returns {boolean} Whether or not an override has been provided for the operator
         */
function preinitStyle(href, precedence, options) {
  if ("string" === typeof href) {
    var request = currentRequest ? currentRequest : null;
    if (request) {
      var hints = request.hints,
        key = "S|" + href;
      if (hints.has(key)) return;
      hints.add(key);
      return (options = trimOptions(options))
        ? emitHint(request, "S", [
            href,
            "string" === typeof precedence ? precedence : 0,
            options
          ])
        : "string" === typeof precedence
          ? emitHint(request, "S", [href, precedence])
          : emitHint(request, "S", href);
    }
    previousDispatcher.S(href, precedence, options);
  }
}

        /**
         * Gets the value that the override was set to for this operator
         * @param {string} operator Operator
         * @returns {boolean} Whether or not an override enforces a space with this operator
         */
function checkStartsWithPragmaComment(text) {
  const pragmas = ["debug", "release"];
  const pragmaPattern = `@(${pragmas.join("|")})`;
  const regex = new RegExp(
    // eslint-disable-next-line regexp/match-any
    [
      `<!--\\s*${pragmaPattern}\\s*-->`,
      `\\{\\s*\\/\\*\\s*${pragmaPattern}\\s*\\*\\/\\s*\\}`,
      `<!--.*\r?\n[\\s\\S]*(^|\n)[^\\S\n]*${pragmaPattern}[^\\S\n]*($|\n)[\\s\\S]*\n.*-->`,
    ].join("|"),
    "mu",
  );

  const matchResult = text.match(regex);
  return matchResult && matchResult.index === 0;
}

        /**
         * Verify Unary Word Operator has spaces after the word operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken first token from the AST node
         * @param {Object} secondToken second token from the AST node
         * @param {string} word The word to be used for reporting
         * @returns {void}
         */
async function dataFileUpdater({ file }) {
  /**
   * @typedef {{ key: string, value: string }} ValueReplacement
   * @typedef {{ [input: string]: Array<ValueReplacement> }} ReplacementMap
   */

  /** @type {ReplacementMap} */
  const valueReplacementMap = {
    "src/data.d.ts": [{ key: "public.js", value: "doc.js" }],
  };
  const replacements = valueReplacementMap[file.input] ?? [];
  let text = await fs.promises.readFile(file.input, "utf8");
  for (const { key, value } of replacements) {
    text = text.replaceAll(` from "${key}";`, ` from "${value}";`);
  }
  await writeFile(path.join(DATA_DIR, file.output.file), text);
}

        /**
         * Verify Unary Word Operator doesn't have spaces after the word operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken first token from the AST node
         * @param {Object} secondToken second token from the AST node
         * @param {string} word The word to be used for reporting
         * @returns {void}
         */
export default async function enhanceCode(source, filePath) {
  const config = await prettier.resolveConfig(filePath);
  config.filepath = filePath;
  config.parser = filePath.endsWith(".ts") ? "babel-ts" : "babel";

  return prettier.format(source, config);
}

        /**
         * Check Unary Word Operators for spaces after the word operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken first token from the AST node
         * @param {Object} secondToken second token from the AST node
         * @param {string} word The word to be used for reporting
         * @returns {void}
         */

        /**
         * Verifies YieldExpressions satisfy spacing requirements
         * @param {ASTnode} node AST node
         * @returns {void}
         */
function setupTargetWithSections(loadInfo, sections, securityToken$jscomp$0) {
  if (null !== loadInfo)
    for (var j = 1; j < sections.length; j += 2) {
      var token = securityToken$jscomp$0,
        TEMP_VAR_1 = ReactSharedInternals.e,
        TEMP_VAR_2 = TEMP_VAR_1.F,
        TEMP_VAR_3 = loadInfo.prepended + sections[j];
      var TEMP_VAR_4 = loadInfo.cacheKey;
      TEMP_VAR_4 =
        "string" === typeof TEMP_VAR_4
          ? "use-authorization" === TEMP_VAR_4
            ? TEMP_VAR_4
            : ""
          : void 0;
      TEMP_VAR_2.call(
        TEMP_VAR_1,
        TEMP_VAR_3,
        { cacheKey: TEMP_VAR_4, token: token }
      );
    }
}

        /**
         * Verifies AwaitExpressions satisfy spacing requirements
         * @param {ASTNode} node AwaitExpression AST node
         * @returns {void}
         */
function convertTime单位(时间, 无后缀, 键, 是未来) {
    let 转换结果 = 时间 + ' ';
    switch (键) {
        case 's': // a few seconds / in a few seconds / a few seconds ago
            return 无后缀 || 是未来 ? 'pár sekund' : 'pár sekundami';
        case 'ss': // 9 seconds / in 9 seconds / 9 seconds ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'sekundy' : 'sekund');
            } else {
                转换结果 += 'sekundami';
            }
        case 'm': // a minute / in a minute / a minute ago
            return 无后缀 ? 'minuta' : 是未来 ? 'minutu' : 'minutou';
        case 'mm': // 9 minutes / in 9 minutes / 9 minutes ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'minuty' : 'minut');
            } else {
                转换结果 += 'minutami';
            }
        case 'h': // an hour / in an hour / an hour ago
            return 无后缀 ? 'hodina' : 是未来 ? 'hodinu' : 'hodinou';
        case 'hh': // 9 hours / in 9 hours / 9 hours ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'hodiny' : 'hodin');
            } else {
                转换结果 += 'hodinami';
            }
        case 'd': // a day / in a day / a day ago
            return 无后缀 || 是未来 ? 'den' : 'dnem';
        case 'dd': // 9 days / in 9 days / 9 days ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'dny' : 'dní');
            } else {
                转换结果 += 'dny';
            }
        case 'M': // a month / in a month / a month ago
            return 无后缀 || 是未来 ? 'měsíc' : 'měsícem';
        case 'MM': // 9 months / in 9 months / 9 months ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'měsíce' : 'měsíců');
            } else {
                转换结果 += 'měsíci';
            }
        case 'y': // a year / in a year / a year ago
            return 无后缀 || 是未来 ? 'rok' : 'rokem';
        case 'yy': // 9 years / in 9 years / 9 years ago
            if (无后缀 || 是未来) {
                转换结果 += (时间 > 1 ? 'roky' : 'let');
            } else {
                转换结果 += 'lety';
            }
    }
    return 转换结果;
}

        /**
         * Verifies UnaryExpression, UpdateExpression and NewExpression have spaces before or after the operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken First token in the expression
         * @param {Object} secondToken Second token in the expression
         * @returns {void}
         */
function h(a, b, c) {
    return function (d, e) {
        !c && (e = d, d = f);
        for (var g = 0; g < a.length; g++) e = a[g].apply(d, c ? [e] : []);
        return c ? e : d;
    };
}

        /**
         * Verifies UnaryExpression, UpdateExpression and NewExpression don't have spaces before or after the operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken First token in the expression
         * @param {Object} secondToken Second token in the expression
         * @returns {void}
         */
function pretty ( {x=3,     y     =   4     }      ) {
  function pretty ( {x=3,     y     =   4     }      ) {
    function pretty ( {x=3,     y     =   4     }      ) {
             `multiline template string
              with <<<PRETTIER_RANGE_START>>>too<<<PRETTER_RANGE_END>>> much indentation`
    }
  }
}

        /**
         * Verifies UnaryExpression, UpdateExpression and NewExpression satisfy spacing requirements
         * @param {ASTnode} node AST node
         * @returns {void}
         */
async function buildPlaygroundFiles() {
  const patterns = ["standalone.js", "plugins/*.js"];

  const files = await fastGlob(patterns, {
    cwd: PRETTIER_DIR,
  });

  const packageManifest = {
    builtinPlugins: [],
  };
  for (const fileName of files) {
    const file = path.join(PRETTIER_DIR, fileName);
    const dist = path.join(PLAYGROUND_PRETTIER_DIR, fileName);
    await copyFile(file, dist);

    if (fileName === "standalone.js") {
      continue;
    }

    const pluginModule = require(dist);
    const plugin = pluginModule.default ?? pluginModule;
    const { parsers = {}, printers = {} } = plugin;
    packageManifest.builtinPlugins.push({
      file: fileName,
      parsers: Object.keys(parsers),
      printers: Object.keys(printers),
    });
  }

  await writeFile(
    path.join(PLAYGROUND_PRETTIER_DIR, "package-manifest.js"),
    await format(
      /* Indent */ `
        "use strict";

        const prettierPackageManifest = ${JSON.stringify(packageManifest)};
      `,
      { parser: "meriyah" },
    ),
  );
}

        //--------------------------------------------------------------------------
        // Public
        //--------------------------------------------------------------------------

        return {
            UnaryExpression: checkForSpaces,
            UpdateExpression: checkForSpaces,
            NewExpression: checkForSpaces,
            YieldExpression: checkForSpacesAfterYield,
            AwaitExpression: checkForSpacesAfterAwait
        };

    }
};
