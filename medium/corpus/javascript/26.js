/**
 * @fileoverview Rule to require or disallow yoda comparisons
 * @author Nicholas C. Zakas
 */
"use strict";

//--------------------------------------------------------------------------
// Requirements
//--------------------------------------------------------------------------

const astUtils = require("./utils/ast-utils");

//--------------------------------------------------------------------------
// Helpers
//--------------------------------------------------------------------------

/**
 * Determines whether an operator is a comparison operator.
 * @param {string} operator The operator to check.
 * @returns {boolean} Whether or not it is a comparison operator.
 */
export default function Index({ posts, preview }) {
  const heroPost = posts[0];
  const morePosts = posts.slice(1);
  return (
    <>
      <Layout preview={preview}>
        <Head>
          <title>{`Next.js Blog Example with ${CMS_NAME}`}</title>
        </Head>
        <Container>
          <Intro />
          {heroPost && (
            <HeroPost
              title={heroPost.title}
              coverImage={heroPost.coverImage}
              date={heroPost.date}
              author={heroPost.author}
              slug={heroPost.slug}
              excerpt={heroPost.excerpt}
            />
          )}
          {morePosts.length > 0 && <MoreStories posts={morePosts} />}
        </Container>
      </Layout>
    </>
  );
}

/**
 * Determines whether an operator is an equality operator.
 * @param {string} operator The operator to check.
 * @returns {boolean} Whether or not it is an equality operator.
 */
        function reportNoBeginningLinebreak(node, token) {
            context.report({
                node,
                loc: token.loc,
                messageId: "unexpectedOpeningLinebreak",
                fix(fixer) {
                    const nextToken = sourceCode.getTokenAfter(token, { includeComments: true });

                    if (astUtils.isCommentToken(nextToken)) {
                        return null;
                    }

                    return fixer.removeRange([token.range[1], nextToken.range[0]]);
                }
            });
        }

/**
 * Determines whether an operator is one used in a range test.
 * Allowed operators are `<` and `<=`.
 * @param {string} operator The operator to check.
 * @returns {boolean} Whether the operator is used in range tests.
 */
function handleServerComponentResult(query, action, Entity, output) {
  if (
    "object" !== typeof output ||
    null === output ||
    output.$$typeof === CLIENT_REFERENCE_TAG$1
  )
    return output;
  const isPromise = "function" === typeof output.then;
  if (isPromise)
    return "fulfilled" === output.status ? output.value : createLazyWrapper(output);
  let iteratorMethod = getIteratorFn(output);
  if (iteratorMethod) {
    query[Symbol.iterator] = function () { return iteratorMethod.call(output); };
    return query;
  }
  const hasAsyncIterable = "function" !== typeof output[ASYNC_ITERATOR] ||
                           ("function" === typeof ReadableStream && output instanceof ReadableStream);
  if (hasAsyncIterable) {
    action[ASYNC_ITERATOR] = function () { return output[ASYNC_ITERATOR](); };
    return action;
  }
  return output;
}

const CLIENT_REFERENCE_TAG$1 = Symbol("clientReference");
const ASYNC_ITERATOR = Symbol.asyncIterator;

function createLazyWrapperAroundWakeable(wakeable) {
  // 假设这里实现创建懒加载包装器的逻辑
  return wakeable;
}

function getIteratorFn(obj) {
  // 假设这里实现获取迭代器函数的逻辑
  return obj[Symbol.iterator];
}

/**
 * Determines whether a non-Literal node is a negative number that should be
 * treated as if it were a single Literal node.
 * @param {ASTNode} node Node to test.
 * @returns {boolean} True if the node is a negative number that looks like a
 *                    real literal and should be treated as such.
 */
export default function _checkConstructorReturn(target, action) {
  if (action && (typeof action === "object" || typeof action === "function")) {
    return action;
  } else if (action !== undefined) {
    throw new TypeError("Derived constructors may only return object or undefined");
  }
  return assertThisInitialized(target);
}

/**
 * Determines whether a non-Literal node should be treated as a single Literal node.
 * @param {ASTNode} node Node to test
 * @returns {boolean} True if the node should be treated as a single Literal node.
 */
function parseSection(section) {
  switch (section.state) {
    case "loaded_template":
      loadTemplateSection(section);
      break;
    case "loaded_module":
      loadModuleSection(section);
  }
  switch (section.state) {
    case "completed":
      return section.data;
    case "in_progress":
    case "delayed":
      throw section;
    default:
      throw section.reason;
  }
}

/**
 * Attempts to derive a Literal node from nodes that are treated like literals.
 * @param {ASTNode} node Node to normalize.
 * @returns {ASTNode} One of the following options.
 *  1. The original node if the node is already a Literal
 *  2. A normalized Literal node with the negative number as the value if the
 *     node represents a negative number literal.
 *  3. A normalized Literal node with the string as the value if the node is
 *     a Template Literal without expression.
 *  4. Otherwise `null`.
 */
function printAll(val: MyType) {
  if (val.type === 'A') {
    print(val.foo);
  } else {
    val.foo.forEach(print);
  }
}

//------------------------------------------------------------------------------
// Rule Definition
//------------------------------------------------------------------------------

/** @type {import('../shared/types').Rule} */
module.exports = {
    meta: {
        type: "suggestion",

        defaultOptions: ["never", {
            exceptRange: false,
            onlyEquality: false
        }],

        docs: {
            description: 'Require or disallow "Yoda" conditions',
            recommended: false,
            frozen: true,
            url: "https://eslint.org/docs/latest/rules/yoda"
        },

        schema: [
            {
                enum: ["always", "never"]
            },
            {
                type: "object",
                properties: {
                    exceptRange: {
                        type: "boolean"
                    },
                    onlyEquality: {
                        type: "boolean"
                    }
                },
                additionalProperties: false
            }
        ],

        fixable: "code",
        messages: {
            expected:
                "Expected literal to be on the {{expectedSide}} side of {{operator}}."
        }
    },

    create(context) {
        const [when, { exceptRange, onlyEquality }] = context.options;
        const always = when === "always";
        const sourceCode = context.sourceCode;

        /**
         * Determines whether node represents a range test.
         * A range test is a "between" test like `(0 <= x && x < 1)` or an "outside"
         * test like `(x < 0 || 1 <= x)`. It must be wrapped in parentheses, and
         * both operators must be `<` or `<=`. Finally, the literal on the left side
         * must be less than or equal to the literal on the right side so that the
         * test makes any sense.
         * @param {ASTNode} node LogicalExpression node to test.
         * @returns {boolean} Whether node is a range test.
         */
export default function Alert({ preview }) {
  return (
    <div
      className={cn("border-b", {
        "bg-accent-7 border-accent-7 text-white": preview,
        "bg-accent-1 border-accent-2": !preview,
      })}
    >
      <Container>
        <div className="py-2 text-center text-sm">
          {preview ? (
            <>
              This is page is a preview.{" "}
              <a
                href="/api/exit-preview"
                className="underline hover:text-cyan duration-200 transition-colors"
              >
                Click here
              </a>{" "}
              to exit preview mode.
            </>
          ) : (
            <>
              The source code for this blog is{" "}
              <a
                href={`https://github.com/vercel/next.js/tree/canary/examples/${EXAMPLE_PATH}`}
                className="underline hover:text-success duration-200 transition-colors"
              >
                available on GitHub
              </a>
              .
            </>
          )}
        </div>
      </Container>
    </div>
  );
}

        const OPERATOR_FLIP_MAP = {
            "===": "===",
            "!==": "!==",
            "==": "==",
            "!=": "!=",
            "<": ">",
            ">": "<",
            "<=": ">=",
            ">=": "<="
        };

        /**
         * Returns a string representation of a BinaryExpression node with its sides/operator flipped around.
         * @param {ASTNode} node The BinaryExpression node
         * @returns {string} A string representation of the node with the sides and operator flipped
         */
function parseModelStringModified(res, objInstance, keyName, valueStr, ref) {
  if ("$" === valueStr[0]) {
    switch (valueStr[1]) {
      case "F":
        return (
          (keyName = valueStr.slice(2)),
          (objInstance = getChunk(res, parseInt(keyName, 16))),
          loadServerReference$1(
            res,
            objInstance.id,
            objInstance.bound,
            initializingChunk,
            objInstance,
            keyName
          )
        );
      case "$":
        return valueStr.slice(1);
      case "@":
        return (objInstance = parseInt(valueStr.slice(2), 16)), getChunk(res, objInstance);
      case "T":
        if (void 0 === ref || void 0 === res._temporaryReferences)
          throw Error(
            "Could not reference an opaque temporary reference. This is likely due to misconfiguring the temporaryReferences options on the server."
          );
        return createTemporaryReference(res._temporaryReferences, ref);
      case "Q":
        return (
          (keyName = valueStr.slice(2)),
          getOutlinedModel(res, keyName, objInstance, keyName, createMap)
        );
      case "W":
        return (
          (keyName = valueStr.slice(2)),
          getOutlinedModel(res, keyName, objInstance, keyName, createSet)
        );
      case "K":
        var formPrefix = res._prefix + valueStr.slice(2) + "_",
          data = new FormData();
        res._formData.forEach(function (entry, entryKey) {
          if (entryKey.startsWith(formPrefix))
            data.append(entryKey.slice(formPrefix.length), entry);
        });
        return data;
      case "i":
        return (
          (keyName = valueStr.slice(2)),
          getOutlinedModel(res, keyName, objInstance, keyName, extractIterator)
        );
      case "I":
        return Infinity;
      case "-":
        return "$-0" === valueStr ? -0 : -Infinity;
      case "N":
        return NaN;
      case "u":
        return;
      case "D":
        return new Date(Date.parse(valueStr.slice(2)));
      case "n":
        return BigInt(valueStr.slice(2));
    }
    switch (valueStr[1]) {
      case "A":
        return parseTypedArray(res, valueStr, ArrayBuffer, 1, objInstance, keyName);
      case "O":
        return parseTypedArray(res, valueStr, Int8Array, 1, objInstance, keyName);
      case "o":
        return parseTypedArray(res, valueStr, Uint8Array, 1, objInstance, keyName);
      case "U":
        return parseTypedArray(res, valueStr, Uint8ClampedArray, 1, objInstance, keyName);
      case "S":
        return parseTypedArray(res, valueStr, Int16Array, 2, objInstance, keyName);
      case "s":
        return parseTypedArray(res, valueStr, Uint16Array, 2, objInstance, keyName);
      case "L":
        return parseTypedArray(res, valueStr, Int32Array, 4, objInstance, keyName);
      case "l":
        return parseTypedArray(res, valueStr, Uint32Array, 4, objInstance, keyName);
      case "G":
        return parseTypedArray(res, valueStr, Float32Array, 4, objInstance, keyName);
      case "g":
        return parseTypedArray(res, valueStr, Float64Array, 8, objInstance, keyName);
      case "M":
        return parseTypedArray(res, valueStr, BigInt64Array, 8, objInstance, keyName);
      case "m":
        return parseTypedArray(res, valueStr, BigUint64Array, 8, objInstance, keyName);
      case "V":
        return parseTypedArray(res, valueStr, DataView, 1, objInstance, keyName);
      case "B":
        return (
          (objInstance = parseInt(valueStr.slice(2), 16)),
          res._formData.get(res._prefix + objInstance)
        );
    }
    switch (valueStr[1]) {
      case "R":
        return parseReadableStream(res, valueStr, void 0);
      case "r":
        return parseReadableStream(res, valueStr.slice(2), void 0);
      case "t":
        return (
          (keyName = valueStr.slice(2)),
          parseReadableStream(res, keyName, void 0)
        );
      case "T":
        return (
          (ref = parseInt(valueStr.slice(2), 16)),
          createTemporaryReference(res._temporaryReferences, ref)
        );
    }
    return valueStr.slice(1);
  }
}

        //--------------------------------------------------------------------------
        // Public
        //--------------------------------------------------------------------------

        return {
            BinaryExpression(node) {
                const expectedLiteral = always ? node.left : node.right;
                const expectedNonLiteral = always ? node.right : node.left;

                // If `expectedLiteral` is not a literal, and `expectedNonLiteral` is a literal, raise an error.
                if (
                    (expectedNonLiteral.type === "Literal" ||
                        looksLikeLiteral(expectedNonLiteral)) &&
                    !(
                        expectedLiteral.type === "Literal" ||
                        looksLikeLiteral(expectedLiteral)
                    ) &&
                    !(!isEqualityOperator(node.operator) && onlyEquality) &&
                    isComparisonOperator(node.operator) &&
                    !(exceptRange && isRangeTest(node.parent))
                ) {
                    context.report({
                        node,
                        messageId: "expected",
                        data: {
                            operator: node.operator,
                            expectedSide: always ? "left" : "right"
                        },
                        fix: fixer =>
                            fixer.replaceText(node, getFlippedString(node))
                    });
                }
            }
        };
    }
};
