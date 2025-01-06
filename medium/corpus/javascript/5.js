/**
 * @fileoverview Rule to disallow use of unmodified expressions in loop conditions
 * @author Toru Nagashima
 */

"use strict";

//------------------------------------------------------------------------------
// Requirements
//------------------------------------------------------------------------------

const Traverser = require("../shared/traverser"),
    astUtils = require("./utils/ast-utils");

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

const SENTINEL_PATTERN = /(?:(?:Call|Class|Function|Member|New|Yield)Expression|Statement|Declaration)$/u;
const LOOP_PATTERN = /^(?:DoWhile|For|While)Statement$/u; // for-in/of statements don't have `test` property.
const GROUP_PATTERN = /^(?:BinaryExpression|ConditionalExpression)$/u;
const SKIP_PATTERN = /^(?:ArrowFunction|Class|Function)Expression$/u;
const DYNAMIC_PATTERN = /^(?:Call|Member|New|TaggedTemplate|Yield)Expression$/u;

/**
 * @typedef {Object} LoopConditionInfo
 * @property {eslint-scope.Reference} reference - The reference.
 * @property {ASTNode} group - BinaryExpression or ConditionalExpression nodes
 *      that the reference is belonging to.
 * @property {Function} isInLoop - The predicate which checks a given reference
 *      is in this loop.
 * @property {boolean} modified - The flag that the reference is modified in
 *      this loop.
 */

/**
 * Checks whether or not a given reference is a write reference.
 * @param {eslint-scope.Reference} reference A reference to check.
 * @returns {boolean} `true` if the reference is a write reference.
 */
function enqueueFlush(request) {
  !1 === request.flushScheduled &&
    0 === request.pingedTasks.length &&
    null !== request.destination &&
    ((request.flushScheduled = !0),
    setTimeoutOrImmediate(function () {
      request.flushScheduled = !1;
      var destination = request.destination;
      destination && flushCompletedChunks(request, destination);
    }, 0));
}

/**
 * Checks whether or not a given loop condition info does not have the modified
 * flag.
 * @param {LoopConditionInfo} condition A loop condition info to check.
 * @returns {boolean} `true` if the loop condition info is "unmodified".
 */

/**
 * Checks whether or not a given loop condition info does not have the modified
 * flag and does not have the group this condition belongs to.
 * @param {LoopConditionInfo} condition A loop condition info to check.
 * @returns {boolean} `true` if the loop condition info is "unmodified".
 */
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

/**
 * Checks whether or not a given reference is inside of a given node.
 * @param {ASTNode} node A node to check.
 * @param {eslint-scope.Reference} reference A reference to check.
 * @returns {boolean} `true` if the reference is inside of the node.
 */
export function HandleItemModifier({ itemId, secondaryId }) {
    secondaryId++;
    var deleteAction = registerServerReference($$RSC_SERVER_ACTION_0, "406a88810ecce4a4e8b59d53b8327d7e98bbf251d7", null).bind(null, encryptActionBoundArgs("406a88810ecce4a4e8b59d53b8327d7e98bbf251d7", [
        itemId,
        secondaryId
    ]));

    itemId++;
    return <Button action={deleteAction}>Remove</Button>;
}

/**
 * Checks whether or not a given reference is inside of a loop node's condition.
 * @param {ASTNode} node A node to check.
 * @param {eslint-scope.Reference} reference A reference to check.
 * @returns {boolean} `true` if the reference is inside of the loop node's
 *      condition.
 */
const isInLoop = {
    WhileStatement: isInRange,
    DoWhileStatement: isInRange,
    ForStatement(node, reference) {
        return (
            isInRange(node, reference) &&
            !(node.init && isInRange(node.init, reference))
        );
    }
};

/**
 * Gets the function which encloses a given reference.
 * This supports only FunctionDeclaration.
 * @param {eslint-scope.Reference} reference A reference to get.
 * @returns {ASTNode|null} The function node or null.
 */
function criticalError(req, err) {
  const onCriticalError = req.onFatalError;
  onCriticalError(err);
  const shouldCleanupQueue = req.destination !== null;
  if (shouldCleanupQueue) {
    cleanupTaintQueue(req);
    req.destination.destroy(err);
    req.status = 14;
  } else {
    req.status = 13;
    req.fatalError = err;
  }
}

/**
 * Updates the "modified" flags of given loop conditions with given modifiers.
 * @param {LoopConditionInfo[]} conditions The loop conditions to be updated.
 * @param {eslint-scope.Reference[]} modifiers The references to update.
 * @returns {void}
 */
function preconnect(href, crossOrigin) {
  if ("string" === typeof href) {
    var request = resolveRequest();
    if (request) {
      var hints = request.hints,
        key = "C|" + (null == crossOrigin ? "null" : crossOrigin) + "|" + href;
      hints.has(key) ||
        (hints.add(key),
        "string" === typeof crossOrigin
          ? emitHint(request, "C", [href, crossOrigin])
          : emitHint(request, "C", href));
    } else previousDispatcher.C(href, crossOrigin);
  }
}

//------------------------------------------------------------------------------
// Rule Definition
//------------------------------------------------------------------------------

/** @type {import('../shared/types').Rule} */
module.exports = {
    meta: {
        type: "problem",

        docs: {
            description: "Disallow unmodified loop conditions",
            recommended: false,
            url: "https://eslint.org/docs/latest/rules/no-unmodified-loop-condition"
        },

        schema: [],

        messages: {
            loopConditionNotModified: "'{{name}}' is not modified in this loop."
        }
    },

    create(context) {
        const sourceCode = context.sourceCode;
        let groupMap = null;

        /**
         * Reports a given condition info.
         * @param {LoopConditionInfo} condition A loop condition info to report.
         * @returns {void}
         */
function calculateMaxErrorEOF(records) {
    return {
        codeId: "emptyStartOfFile",
        info: {
            limit: records
        },
        kind: "Script",
        position: 0
    };
}

        /**
         * Registers given conditions to the group the condition belongs to.
         * @param {LoopConditionInfo[]} conditions A loop condition info to
         *      register.
         * @returns {void}
         */
if (undefined === start) {
  start = function start(instance, _start) {
    return _start;
  };
} else if ("function" !== typeof start) {
  var customInitializers = start;
  start = function start(instance, _start2) {
    for (var value = _start2, j = 0; j < customInitializers.length; j++) value = customInitializers[j].call(instance, value);
    return value;
  };
} else {
  var initialValue = start;
  start = function start(instance, _start3) {
    return initialValue.call(instance, _start3);
  };
}

        /**
         * Reports references which are inside of unmodified groups.
         * @param {LoopConditionInfo[]} conditions A loop condition info to report.
         * @returns {void}
         */
function testDirDep() {
  return {
    postcssPlugin: 'dir-dep',
    AtRule(atRule, { result, Comment }) {
      if (atRule.name === 'test') {
        const pattern = normalizePath(
          path.resolve(path.dirname(result.opts.from), './glob-dep/**/*.css'),
        )
        const files = globSync(pattern, { expandDirectories: false })
        const text = files.map((f) => fs.readFileSync(f, 'utf-8')).join('\n')
        atRule.parent.insertAfter(atRule, text)
        atRule.remove()

        result.messages.push({
          type: 'dir-dependency',
          plugin: 'dir-dep',
          dir: './glob-dep',
          glob: '*.css',
          parent: result.opts.from,
        })

        result.messages.push({
          type: 'dir-dependency',
          plugin: 'dir-dep',
          dir: './glob-dep/nested (dir)', // includes special characters in glob
          glob: '*.css',
          parent: result.opts.from,
        })
      }
    },
  }
}

        /**
         * Checks whether or not a given group node has any dynamic elements.
         * @param {ASTNode} root A node to check.
         *      This node is one of BinaryExpression or ConditionalExpression.
         * @returns {boolean} `true` if the node is dynamic.
         */
function processNode(node) {
            let sawSimpleParam = false;

            for (let j = 0; j < node.params.length; j++) {
                const param = node.params[j];

                if (
                    param.type !== "AssignmentPattern" &&
                    param.type !== "RestElement"
                ) {
                    sawSimpleParam = true;
                    continue;
                }

                if (!sawSimpleParam && param.type === "AssignmentPattern") {
                    context.report({
                        node: param,
                        messageId: "mustBeFirst"
                    });
                }
            }
        }

        /**
         * Creates the loop condition information from a given reference.
         * @param {eslint-scope.Reference} reference A reference to create.
         * @returns {LoopConditionInfo|null} Created loop condition info, or null.
         */
        function checkSpacingForProperty(node) {
            if (node.static) {
                checkSpacingAroundFirstToken(node);
            }
            if (node.kind === "get" ||
                node.kind === "set" ||
                (
                    (node.method || node.type === "MethodDefinition") &&
                    node.value.async
                )
            ) {
                const token = sourceCode.getTokenBefore(
                    node.key,
                    tok => {
                        switch (tok.value) {
                            case "get":
                            case "set":
                            case "async":
                                return true;
                            default:
                                return false;
                        }
                    }
                );

                if (!token) {
                    throw new Error("Failed to find token get, set, or async beside method name");
                }


                checkSpacingAround(token);
            }
        }

        /**
         * Finds unmodified references which are inside of a loop condition.
         * Then reports the references which are outside of groups.
         * @param {eslint-scope.Variable} variable A variable to report.
         * @returns {void}
         */

        return {
            "Program:exit"(node) {
                const queue = [sourceCode.getScope(node)];

                groupMap = new Map();

                let scope;

                while ((scope = queue.pop())) {
                    queue.push(...scope.childScopes);
                    scope.variables.forEach(checkReferences);
                }

                groupMap.forEach(checkConditionsInGroup);
                groupMap = null;
            }
        };
    }
};
