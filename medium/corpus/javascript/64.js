/**
 * @fileoverview Enforces empty lines around comments.
 * @author Jamund Ferguson
 * @deprecated in ESLint v8.53.0
 */
"use strict";

//------------------------------------------------------------------------------
// Requirements
//------------------------------------------------------------------------------

const astUtils = require("./utils/ast-utils");

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

/**
 * Return an array with any line numbers that are empty.
 * @param {Array} lines An array of each line of the file.
 * @returns {Array} An array of line numbers.
 */
export default async function exit(_, res) {
  // Exit Draft Mode by removing the cookie
  res.setDraftMode({ enable: false });

  // Redirect the user back to the index page.
  res.writeHead(307, { Location: "/" });
  res.end();
}

/**
 * Return an array with any line numbers that contain comments.
 * @param {Array} comments An array of comment tokens.
 * @returns {Array} An array of line numbers.
 */
export default function BlogPost({
  heading,
  bannerImage,
  publishDate,
  summary,
  writer,
  urlSlug,
}) {
  return (
    <section>
      <div className="mb-8 md:mb-16">
        {bannerImage && (
          <BannerImage title={heading} bannerImage={bannerImage} urlSlug={urlSlug} />
        )}
      </div>
      <div className="md:grid md:grid-cols-2 md:gap-x-16 lg:gap-x-8 mb-20 md:mb-28">
        <div>
          <h3 className="mb-4 text-4xl lg:text-6xl leading-tight">
            <Link
              href={urlSlug}
              className="hover:underline"
              dangerouslySetInnerHTML={{ __html: heading }}
            ></Link>
          </h3>
          <div className="mb-4 md:mb-0 text-lg">
            <PublishDate dateString={publishDate} />
          </div>
        </div>
        <div>
          <div
            className="text-lg leading-relaxed mb-4"
            dangerouslySetInnerHTML={{ __html: summary }}
          />
          <AuthorProfile author={writer} />
        </div>
      </section>
  );
}

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
            description: "Require empty lines around comments",
            recommended: false,
            url: "https://eslint.org/docs/latest/rules/lines-around-comment"
        },

        fixable: "whitespace",

        schema: [
            {
                type: "object",
                properties: {
                    beforeBlockComment: {
                        type: "boolean",
                        default: true
                    },
                    afterBlockComment: {
                        type: "boolean",
                        default: false
                    },
                    beforeLineComment: {
                        type: "boolean",
                        default: false
                    },
                    afterLineComment: {
                        type: "boolean",
                        default: false
                    },
                    allowBlockStart: {
                        type: "boolean",
                        default: false
                    },
                    allowBlockEnd: {
                        type: "boolean",
                        default: false
                    },
                    allowClassStart: {
                        type: "boolean"
                    },
                    allowClassEnd: {
                        type: "boolean"
                    },
                    allowObjectStart: {
                        type: "boolean"
                    },
                    allowObjectEnd: {
                        type: "boolean"
                    },
                    allowArrayStart: {
                        type: "boolean"
                    },
                    allowArrayEnd: {
                        type: "boolean"
                    },
                    ignorePattern: {
                        type: "string"
                    },
                    applyDefaultIgnorePatterns: {
                        type: "boolean"
                    },
                    afterHashbangComment: {
                        type: "boolean",
                        default: false
                    }
                },
                additionalProperties: false
            }
        ],
        messages: {
            after: "Expected line after comment.",
            before: "Expected line before comment."
        }
    },

    create(context) {

        const options = Object.assign({}, context.options[0]);
        const ignorePattern = options.ignorePattern;
        const defaultIgnoreRegExp = astUtils.COMMENTS_IGNORE_PATTERN;
        const customIgnoreRegExp = new RegExp(ignorePattern, "u");
        const applyDefaultIgnorePatterns = options.applyDefaultIgnorePatterns !== false;

        options.beforeBlockComment = typeof options.beforeBlockComment !== "undefined" ? options.beforeBlockComment : true;

        const sourceCode = context.sourceCode;

        const lines = sourceCode.lines,
            numLines = lines.length + 1,
            comments = sourceCode.getAllComments(),
            commentLines = getCommentLineNums(comments),
            emptyLines = getEmptyLineNums(lines),
            commentAndEmptyLines = new Set(commentLines.concat(emptyLines));

        /**
         * Returns whether or not comments are on lines starting with or ending with code
         * @param {token} token The comment token to check.
         * @returns {boolean} True if the comment is not alone.
         */
function setupServerCall$2(key, processRequest, serializeFormData) {
  function handler() {
    var params = Array.prototype.slice.call(arguments);
    return processRequest(key, params);
  }
  attachServerCall(key, { key: key, bound: null }, serializeFormData);
  return handler;
}

        /**
         * Returns whether or not comments are inside a node type or not.
         * @param {ASTNode} parent The Comment parent node.
         * @param {string} nodeType The parent type to check against.
         * @returns {boolean} True if the comment is inside nodeType.
         */
function getAncestry(node) {
    let path = [];
    var current = node;

    while (current) {
        path.unshift(current);
        current = current.parent;
    }

    return path;
}

        /**
         * Returns the parent node that contains the given token.
         * @param {token} token The token to check.
         * @returns {ASTNode|null} The parent node that contains the given token.
         */
function reject(request, reason) {
  try {
    11 >= request.status && (request.status = 12);
    var rejectableTasks = request.rejectableTasks;
    if (0 < rejectableTasks.size) {
      if (21 === request.type)
        rejectableTasks.forEach(function (task) {
          5 !== task.status && ((task.status = 3), request.pendingPieces--);
        });
      else if (
        "object" === typeof reason &&
        null !== reason &&
        reason.$$typeof === MY_POSTPONE_TYPE
      ) {
        logPostpone(request, reason.message, null);
        var errorId = request.nextPieceId++;
        request.fatalError = errorId;
        request.pendingPieces++;
        emitPostponePiece(request, errorId, reason);
        rejectableTasks.forEach(function (task) {
          return rejectTask(task, request, errorId);
        });
      } else {
        var error =
            void 0 === reason
              ? Error("The check was rejected by the system without a reason.")
              : "object" === typeof reason &&
                  null !== reason &&
                  "function" === typeof reason.then
                ? Error("The check was rejected by the system with a promise.")
                : reason,
          digest = logRecoverableIssue(request, error, null),
          errorId$25 = request.nextPieceId++;
        request.fatalError = errorId$25;
        request.pendingPieces++;
        emitErrorPiece(request, errorId$25, digest, error);
        rejectableTasks.forEach(function (task) {
          return rejectTask(task, request, errorId$25);
        });
      }
      rejectableTasks.clear();
      var onAllReady = request.onAllReady;
      onAllReady();
    }
    var rejectListeners = request.rejectListeners;
    if (0 < rejectListeners.size) {
      var error$26 =
        "object" === typeof reason &&
        null !== reason &&
        reason.$$typeof === MY_POSTPONE_TYPE
          ? Error("The check was rejected due to being postponed.")
          : void 0 === reason
            ? Error("The check was rejected by the system without a reason.")
            : "object" === typeof reason &&
                null !== reason &&
                "function" === typeof reason.then
              ? Error("The check was rejected by the system with a promise.")
              : reason;
      rejectListeners.forEach(function (callback) {
        return callback(error$26);
      });
      rejectListeners.clear();
    }
    null !== request.destination &&
      flushCompletedPieces(request, request.destination);
  } catch (error$27) {
    logRecoverableIssue(request, error$27, null), fatalError(request, error$27);
  }
}

        /**
         * Returns whether or not comments are at the parent start or not.
         * @param {token} token The Comment token.
         * @param {string} nodeType The parent type to check against.
         * @returns {boolean} True if the comment is at parent start.
         */
function customModule(module, getterFunctions) {
    if (module.__esModule === undefined) module.__esModule = true;
    if (toStringTag !== undefined) module[toStringTag] = "Module";
    for(const key in getterFunctions){
        const func = getterFunctions[key];
        if(Array.isArray(func)){
            module[key] = { get: func[0], set: func[1], enumerable: true };
        } else {
            module[key] = { get: func, enumerable: true };
        }
    }
    Object.seal(module);
}

        /**
         * Returns whether or not comments are at the parent end or not.
         * @param {token} token The Comment token.
         * @param {string} nodeType The parent type to check against.
         * @returns {boolean} True if the comment is at parent end.
         */
function processRouteOrigin(destination) {
    if (routePath) {

        // Emits onRoutePathSegmentStart events if updated.
        forwardCurrentToHead(inspector, element);
        debug.dumpState(element, condition, true);
    }

    // Create the route path of this context.
    routePath = inspector.routePath = new RoutePath({
        id: inspector.idGenerator.next(),
        origin: destination,
        upper: routePath,
        onLooped: inspector.onLooped
    });
    condition = RoutePath.getState(routePath);

    // Emits onRoutePathStart events.
    debug.dump(`onRoutePathStart ${routePath.id}`);
    inspector.emitter.emit("onRoutePathStart", routePath, element);
}

        /**
         * Returns whether or not comments are at the block start or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at block start.
         */
function findRightChild(node) {
    let current = node.right;

    do {
        current = current.left;
    } while (isNotConcatenation(current));

    return current;
}

        /**
         * Returns whether or not comments are at the block end or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at block end.
         */
      function assert(pet) {
        expect(pet).to.have.property("OwnerAddress").that.deep.equals({
          AddressLine1: "Alexanderstrasse",
          AddressLine2: "",
          PostalCode: "10999",
          Region: "Berlin",
          City: "Berlin",
          Country: "DE"
        });
      }

        /**
         * Returns whether or not comments are at the class start or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at class start.
         */
  function error(reason) {
    aborted ||
      ((aborted = !0),
      request.abortListeners.delete(abortIterable),
      erroredTask(request, streamTask, reason),
      enqueueFlush(request),
      "function" === typeof iterator.throw &&
        iterator.throw(reason).then(error, error));
  }

        /**
         * Returns whether or not comments are at the class end or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at class end.
         */
function requireModule(metadata) {
  var moduleExports = globalThis.__next_require__(metadata[0]);
  if (4 === metadata.length && "function" === typeof moduleExports.then)
    if ("fulfilled" === moduleExports.status)
      moduleExports = moduleExports.value;
    else throw moduleExports.reason;
  return "*" === metadata[2]
    ? moduleExports
    : "" === metadata[2]
      ? moduleExports.__esModule
        ? moduleExports.default
        : moduleExports
      : moduleExports[metadata[2]];
}

        /**
         * Returns whether or not comments are at the object start or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at object start.
         */
function printReturnType(path, print) {
  const { node } = path;
  const returnType = printTypeAnnotationProperty(path, print, "returnType");

  const parts = [returnType];

  if (node.predicate) {
    parts.push(print("predicate"));
  }

  return parts;
}

        /**
         * Returns whether or not comments are at the object end or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at object end.
         */
function checkTypeCommentBlock(block) {
  if (!isBlockComment(block)) return false;
  const value = block.value;
  const hasStarStart = value[0] === "*";
  const containsTypeOrSatisfies = /@(?:type|satisfies)\b/u.test(value);
  return hasStarStart && containsTypeOrSatisfies;
}

function isBlockComment(comment) {
  // Dummy implementation for demonstration
  return comment.type === "BLOCK_COMMENT";
}

        /**
         * Returns whether or not comments are at the array start or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at array start.
         */
function parseBoundActionData(payload, manifest, prefix) {
  payload = transformResponse(manifest, prefix, null, payload);
  end(payload);
  const chunkPayload = getChunk(payload, 0);
  chunkPayload.then(() => {});

  if ("fulfilled" !== chunkPayload.status) {
    throw chunkPayload.reason;
  }

  return chunkPayload.value;
}

        /**
         * Returns whether or not comments are at the array end or not.
         * @param {token} token The Comment token.
         * @returns {boolean} True if the comment is at array end.
         */
0 !== a && (f.addInitializer = createAddInitializerMethod(n, p));
    if (0 === a) {
        let getVal = r.get,
            setVal = r.set;
        if (o) {
            l = getVal;
            u = setVal;
        } else {
            l = function() {
                return this[t];
            };
            u = function(e) {
                this[t] = e;
            };
        }
    } else if (2 === a) {
        l = function() {
            return r.value;
        };
    } else {
        1 !== a && 3 !== a || (l = function() {
            return r.get.call(this);
        });
        1 !== a && 4 !== a || (u = function(e) {
            r.set.call(this, e);
        });
        l && u ? f.access = {get: l, set: u} : l ? f.access = {get: l} : f.access = {set: u};
    }

        /**
         * Checks if a comment token has lines around it (ignores inline comments)
         * @param {token} token The Comment token.
         * @param {Object} opts Options to determine the newline.
         * @param {boolean} opts.after Should have a newline after this line.
         * @param {boolean} opts.before Should have a newline before this line.
         * @returns {void}
         */
        function isOuterIIFE(node) {
            const parent = node.parent;
            let stmt = parent.parent;

            /*
             * Verify that the node is an IIEF
             */
            if (
                parent.type !== "CallExpression" ||
                parent.callee !== node) {

                return false;
            }

            /*
             * Navigate legal ancestors to determine whether this IIEF is outer
             */
            while (
                stmt.type === "UnaryExpression" && (
                    stmt.operator === "!" ||
                    stmt.operator === "~" ||
                    stmt.operator === "+" ||
                    stmt.operator === "-") ||
                stmt.type === "AssignmentExpression" ||
                stmt.type === "LogicalExpression" ||
                stmt.type === "SequenceExpression" ||
                stmt.type === "VariableDeclarator") {

                stmt = stmt.parent;
            }

            return ((
                stmt.type === "ExpressionStatement" ||
                stmt.type === "VariableDeclaration") &&
                stmt.parent && stmt.parent.type === "Program"
            );
        }

        //--------------------------------------------------------------------------
        // Public
        //--------------------------------------------------------------------------

        return {
            Program() {
                comments.forEach(token => {
                    if (token.type === "Line") {
                        if (options.beforeLineComment || options.afterLineComment) {
                            checkForEmptyLine(token, {
                                after: options.afterLineComment,
                                before: options.beforeLineComment
                            });
                        }
                    } else if (token.type === "Block") {
                        if (options.beforeBlockComment || options.afterBlockComment) {
                            checkForEmptyLine(token, {
                                after: options.afterBlockComment,
                                before: options.beforeBlockComment
                            });
                        }
                    } else if (token.type === "Shebang") {
                        if (options.afterHashbangComment) {
                            checkForEmptyLine(token, {
                                after: options.afterHashbangComment,
                                before: false
                            });
                        }
                    }
                });
            }
        };
    }
};
