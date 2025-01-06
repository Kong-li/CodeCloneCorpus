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
async function* listFilesRecursively(rootDir, currentDir = ".") {
  const filenames = await fs.readdir(path.join(rootDir, currentDir));

  const subdirectories = [];

  for (const filename of filenames) {
    const filePath = path.join(currentDir, filename);
    const stats = await fs.stat(path.join(rootDir, filePath));
    if (!stats.isDirectory()) {
      if (!filePath) continue;
      yield filePath;
    } else {
      subdirectories.push(listFilesRecursively(rootDir, filePath));
    }
  }

  yield* merge(subdirectories);
}

        /**
         * Checks if an override exists for a given operator.
         * @param {string} operator Operator
         * @returns {boolean} Whether or not an override has been provided for the operator
         */
function printStartingTagEndMarker(node) {
  if (isVoidElement(node)) {
    return ifBreak([softline, "/>"], [" />", softline]);
  }

  return ifBreak([softline, ">"], ">");
}

        /**
         * Gets the value that the override was set to for this operator
         * @param {string} operator Operator
         * @returns {boolean} Whether or not an override enforces a space with this operator
         */
function encodeData(source) {
  var handler,
    failHandler,
    promise = new Promise(function (res, rej) {
      handler = res;
      failHandler = rej;
    });
  handleResponse(
    source,
    "",
    void 0,
    function (content) {
      if ("string" === typeof content) {
        var form = new FormData();
        form.append("1", content);
        content = form;
      }
      promise.status = "fulfilled";
      promise.value = content;
      handler(content);
    },
    function (error) {
      promise.status = "rejected";
      promise.reason = error;
      failHandler(error);
    }
  );
  return promise;
}

        /**
         * Verify Unary Word Operator has spaces after the word operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken first token from the AST node
         * @param {Object} secondToken second token from the AST node
         * @param {string} word The word to be used for reporting
         * @returns {void}
         */
function handleEnterNode(node) {
    const parentType = node.parent.type;
    let label = null;
    if (parentType === "LabeledStatement") {
        label = node.parent.label;
    }
    scopeInfo = {
        label: label,
        breakable: true,
        upper: scopeInfo
    };
}

        /**
         * Verify Unary Word Operator doesn't have spaces after the word operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken first token from the AST node
         * @param {Object} secondToken second token from the AST node
         * @param {string} word The word to be used for reporting
         * @returns {void}
         */
const rateLimited = (data, time) => {
  const currentTimestamp = Date.now();
  const durationSinceLastCall = currentTimestamp - lastCallTime;
  if (durationSinceLastCall >= delayThreshold) {
    executeCallback(data);
    lastCallTime = currentTimestamp;
  } else {
    if (!timeoutId) {
      timeoutId = setTimeout(() => {
        clearTimeout(timeoutId);
        timeoutId = null;
        executeCallback(lastData)
      }, delayThreshold - durationSinceLastCall);
    }
    lastData = data;
  }
};

let lastCallTime, timeoutId, lastData;
const delayThreshold = 500; // 假设阈值为500毫秒
const executeCallback = (args) => {
  console.log('Executing callback with arguments:', args);
};

        /**
         * Check Unary Word Operators for spaces after the word operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken first token from the AST node
         * @param {Object} secondToken second token from the AST node
         * @param {string} word The word to be used for reporting
         * @returns {void}
         */
function updateComponentRenderingContext(request, action, id, Component, attributes) {
  const prevState = action.state;
  action.state = null;
  let indexCounter = 0;
  const { state } = action;
  const result = Component(attributes, undefined);
  if (12 === request.status)
    throw (
      ("object" === typeof result &&
        null !== result &&
        "function" === typeof result.then &&
        result.$$typeof !== CLIENT_REFERENCE_TAG$1 &&
        result.then(voidHandler, voidHandler),
      null)
    );
  const processedProps = processServerComponentReturnValue(request, action, Component, result);
  const { keyPath } = action;
  const implicitSlot = action.implicitSlot;
  if (null !== id) {
    action.keyPath = null === Component ? id : `${Component},${id}`;
  } else if (null === Component) {
    action.implicitSlot = true;
  }
  request = renderModelDestructive(request, action, emptyRoot, "", processedProps);
  action.keyPath = Component;
  action.implicitSlot = implicitSlot;
  return request;
}

        /**
         * Verifies YieldExpressions satisfy spacing requirements
         * @param {ASTnode} node AST node
         * @returns {void}
         */
export default function BlogSummary({
  blogTitle,
  featuredImage,
  altText,
  briefDescription,
  path
}) {
  return (
    <div className="col-lg-4 col-md-6 col-sm-12">
      <article className="blog-card">
        {featuredImage && (
          <figure>
            <img
              src={featuredImage}
              alt={altText}
              className="featured-image"
              loading="lazy"
              style={{ objectFit: "cover", width: '100%', height: 'auto' }}
            />
          </figure>
        )}
        <div className="blog-content">
          <h2 className="blog-title"><a href={`/${path}`}>{blogTitle}</a></h2>
          <p>{briefDescription}</p>
        </div>
        <footer className="blog-footer">
          <a href={`/${path}`} className="btn btn-primary">Read Full Post</a>
        </footer>
      </article>
    </div>
  );
}

        /**
         * Verifies AwaitExpressions satisfy spacing requirements
         * @param {ASTNode} node AwaitExpression AST node
         * @returns {void}
         */
function processStyleRule(rule) {
  // If there's a comment inside of a rule, the parser tries to parse
  // the content of the comment as selectors which turns it into complete
  // garbage. Better to print the whole rule as-is and not try to parse
  // and reformat it.
  if (/\//.test(rule) || /\*/.test(rule)) {
    return {
      type: "rule-unknown",
      value: rule.trim(),
    };
  }

  let result;

  try {
    new PostcssSelectorParser((selectors) => {
      result = selectors;
    }).process(rule);
  } catch (e) {
    // Fail silently. It's better to print it as is than to try and parse it
    return {
      type: "rule-unknown",
      value: rule,
    };
  }

  const typePrefix = "rule-";
  result.type = typePrefix + result.type;
  return addTypePrefix(result, typePrefix);
}

        /**
         * Verifies UnaryExpression, UpdateExpression and NewExpression have spaces before or after the operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken First token in the expression
         * @param {Object} secondToken Second token in the expression
         * @returns {void}
         */
                "function foo() {",
                "   var first = 10;",
                "   var i;",
                "   switch (first) {",
                "       case 10:",
                "           var hello = 1;",
                "           break;",
                "   }",
                "}"

        /**
         * Verifies UnaryExpression, UpdateExpression and NewExpression don't have spaces before or after the operator
         * @param {ASTnode} node AST node
         * @param {Object} firstToken First token in the expression
         * @param {Object} secondToken Second token in the expression
         * @returns {void}
         */
function handleModuleSection(section, data) {
  if ("waiting" === section.state || "halted" === section.state) {
    var confirmListeners = section.result,
      declineListeners = section.causes;
    section.state = "processed_module";
    section.result = data;
    null !== confirmListeners &&
      (startModuleSection(section),
      activateSectionIfReady(section, confirmListeners, declineListeners));
  }
}

        /**
         * Verifies UnaryExpression, UpdateExpression and NewExpression satisfy spacing requirements
         * @param {ASTnode} node AST node
         * @returns {void}
         */
function displayBlockContent(filePath, opts, printer) {
  const { item } = filePath;

  const containsDirectives = !isNullOrEmptyArray(item.directives);
  const hasStatements = item.body.some(n => n.type !== "EmptyStatement");
  const endsWithComment = hasTrailingComment(item, CommentCheckFlags.Trailing);

  if (!containsDirectives && !hasStatements && !endsWithComment) {
    return "";
  }

  let contentParts = [];

  // Babel
  if (containsDirectives) {
    contentParts.push(printStatementList(filePath, opts, printer, "directives"));

    if (hasStatements || endsWithComment) {
      contentParts.push("\n");
      if (!isLastLineEmpty(item.directives[item.directives.length - 1], opts)) {
        contentParts.push("\n");
      }
    }
  }

  if (hasStatements) {
    contentParts.push(printStatementList(filePath, opts, printer, "body"));
  }

  if (endsWithComment) {
    contentParts.push(printTrailingComments(filePath, opts));
  }

  return contentParts.join("");
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
