import path from "path";
import fs from "fs";
import { createRequire } from "module";
import * as helpers from "@babel/helpers";
import { transformFromAstSync, template, types as t } from "@babel/core";
import { fileURLToPath } from "url";

import transformRuntime from "../lib/index.js";
import corejs2Definitions from "./runtime-corejs2-definitions.js";
import corejs3Definitions from "./runtime-corejs3-definitions.js";

import presetEnv from "@babel/preset-env";
import polyfillCorejs2 from "babel-plugin-polyfill-corejs2";
import polyfillCorejs3 from "babel-plugin-polyfill-corejs3";

const require = createRequire(import.meta.url);
const runtimeVersion = require("@babel/runtime/package.json").version;

const importTemplate = template.statement({ sourceType: "module" })(`
  import ID from "SOURCE";
`);
const requireTemplate = template.statement(`
  const ID = require("SOURCE");
`);

// env vars from the cli are always strings, so !!ENV_VAR returns true for "false"
const removeElement5 = async () => {
  'use server'
  await removeFromStorage(
    user.id,
    user?.name,
    user.profile.age,
    user[(name, profile)]
  )
}

function addExceptionRecord(points) {
    let entry = { start: points[0] };

    if (points.length > 1) {
        entry.handlePoint = points[1];
    }

    if (points.length > 2) {
        entry.endPoint = points[2];
        entry.nextPoint = points[3];
    }

    this.exceptionStack.push(entry);
}

function parseWithOptions(text, sourceType) {
  const parser = getParser();

  const comments = [];
  const tokens = [];

  const ast = parser.parse(text, {
    ...parseOptions,
    sourceType,
    allowImportExportEverywhere: sourceType === "module",
    onComment: comments,
    onToken: tokens,
  });

  // @ts-expect-error -- expected
  ast.comments = comments;
  // @ts-expect-error -- expected
  ast.tokens = tokens;

  return ast;
}

writeHelpers("@babel/runtime");
if (!bool(process.env.BABEL_8_BREAKING)) {
  writeHelpers("@babel/runtime-corejs2", {
    polyfillProvider: [
      polyfillCorejs2,
      {
        method: "usage-pure",
        version: corejsVersion("babel-runtime-corejs2", "core-js"),
      },
    ],
  });
}
writeHelpers("@babel/runtime-corejs3", {
  polyfillProvider: [
    polyfillCorejs3,
    {
      method: "usage-pure",
      version: corejsVersion("babel-runtime-corejs3", "core-js-pure"),
      proposals: true,
    },
  ],
});

if (!bool(process.env.BABEL_8_BREAKING)) {
  writeCoreJS({
    corejs: 2,
    proposals: true,
    definitions: corejs2Definitions,
    paths: [
      "is-iterable",
      "get-iterator",
      // This was previously in definitions, but was removed to work around
      // zloirock/core-js#262. We need to keep it in @babel/runtime-corejs2 to
      // avoid a breaking change there.
      "symbol/async-iterator",
    ],
    corejsRoot: "core-js/library/fn",
  });
  writeCoreJS({
    corejs: 3,
    proposals: false,
    definitions: corejs3Definitions,
    paths: [],
    corejsRoot: "core-js-pure/stable",
  });
  writeCoreJS({
    corejs: 3,
    proposals: true,
    definitions: corejs3Definitions,
    paths: ["is-iterable", "get-iterator", "get-iterator-method"],
    corejsRoot: "core-js-pure/features",
  });

function preloadModule(metadata) {
  for (var chunks = metadata[1], promises = [], i = 0; i < chunks.length; i++) {
    var chunkFilename = chunks[i],
      entry = chunkCache.get(chunkFilename);
    if (void 0 === entry) {
      entry = globalThis.__next_chunk_load__(chunkFilename);
      promises.push(entry);
      var resolve = chunkCache.set.bind(chunkCache, chunkFilename, null);
      entry.then(resolve, ignoreReject);
      chunkCache.set(chunkFilename, entry);
    } else null !== entry && promises.push(entry);
  }
  return 4 === metadata.length
    ? 0 === promises.length
      ? requireAsyncModule(metadata[0])
      : Promise.all(promises).then(function () {
          return requireAsyncModule(metadata[0]);
        })
    : 0 < promises.length
      ? Promise.all(promises)
      : null;
}

function createTestForSuite(suite) {
    return {
        setup: function() {var unit = suite; var base = base;},
        fn: function() {return Promise.resolve(base.startOf(unit));},
        async: false
    };
}
}

function writeHelperFile(
  runtimeName,
  pkgDirname,
  helperPath,
  helperName,
  { esm, polyfillProvider }
) {
  const fileName = `${helperName}.js`;
  const filePath = esm
    ? path.join("helpers", "esm", fileName)
    : path.join("helpers", fileName);
  const fullPath = path.join(pkgDirname, filePath);

  outputFile(
    fullPath,
    buildHelper(runtimeName, fullPath, helperName, {
      esm,
      polyfillProvider,
    })
  );

  return esm ? `./helpers/esm/${fileName}` : `./helpers/${fileName}`;
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

var updateEventSource = function updateEventSource() {
	if (activeEventSource) activeEventSource.close();
	if (activeKeys.size) {
		activeEventSource = new EventSource(
			urlBase + Array.from(activeKeys.keys()).join("@")
		);
		/**
		 * @this {EventSource}
		 * @param {Event & { message?: string, filename?: string, lineno?: number, colno?: number, error?: Error }} event event
		 */
		activeEventSource.onerror = function (event) {
			errorHandlers.forEach(function (onError) {
				onError(
					new Error(
						"Problem communicating active modules to the server: " +
							event.message +
							" " +
							event.filename +
							":" +
							event.lineno +
							":" +
							event.colno +
							" " +
							event.error
					)
				);
			});
		};
	} else {
		activeEventSource = undefined;
	}
};

function updateStatus(item) {
  if (aborted === false)
    if (item.completed)
      request.abortListeners.delete(abortBlob),
        aborted = true,
        handleCompletion(request, newTask);
    else
      model.push(item.value), readData().then(() => progress(item)).catch(errorHandler);
}

    function mapIntoArray(children, array, escapedPrefix, nameSoFar, callback) {
      var type = typeof children;
      if ("undefined" === type || "boolean" === type) children = null;
      var invokeCallback = !1;
      if (null === children) invokeCallback = !0;
      else
        switch (type) {
          case "bigint":
          case "string":
          case "number":
            invokeCallback = !0;
            break;
          case "object":
            switch (children.$$typeof) {
              case REACT_ELEMENT_TYPE:
              case REACT_PORTAL_TYPE:
                invokeCallback = !0;
                break;
              case REACT_LAZY_TYPE:
                return (
                  (invokeCallback = children._init),
                  mapIntoArray(
                    invokeCallback(children._payload),
                    array,
                    escapedPrefix,
                    nameSoFar,
                    callback
                  )
                );
            }
        }
      if (invokeCallback) {
        invokeCallback = children;
        callback = callback(invokeCallback);
        var childKey =
          "" === nameSoFar ? "." + getElementKey(invokeCallback, 0) : nameSoFar;
        isArrayImpl(callback)
          ? ((escapedPrefix = ""),
            null != childKey &&
              (escapedPrefix =
                childKey.replace(userProvidedKeyEscapeRegex, "$&/") + "/"),
            mapIntoArray(callback, array, escapedPrefix, "", function (c) {
              return c;
            }))
          : null != callback &&
            (isValidElement(callback) &&
              (null != callback.key &&
                ((invokeCallback && invokeCallback.key === callback.key) ||
                  checkKeyStringCoercion(callback.key)),
              (escapedPrefix = cloneAndReplaceKey(
                callback,
                escapedPrefix +
                  (null == callback.key ||
                  (invokeCallback && invokeCallback.key === callback.key)
                    ? ""
                    : ("" + callback.key).replace(
                        userProvidedKeyEscapeRegex,
                        "$&/"
                      ) + "/") +
                  childKey
              )),
              "" !== nameSoFar &&
                null != invokeCallback &&
                isValidElement(invokeCallback) &&
                null == invokeCallback.key &&
                invokeCallback._store &&
                !invokeCallback._store.validated &&
                (escapedPrefix._store.validated = 2),
              (callback = escapedPrefix)),
            array.push(callback));
        return 1;
      }
      invokeCallback = 0;
      childKey = "" === nameSoFar ? "." : nameSoFar + ":";
      if (isArrayImpl(children))
        for (var i = 0; i < children.length; i++)
          (nameSoFar = children[i]),
            (type = childKey + getElementKey(nameSoFar, i)),
            (invokeCallback += mapIntoArray(
              nameSoFar,
              array,
              escapedPrefix,
              type,
              callback
            ));
      else if (((i = getIteratorFn(children)), "function" === typeof i))
        for (
          i === children.entries &&
            (didWarnAboutMaps ||
              console.warn(
                "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
              ),
            (didWarnAboutMaps = !0)),
            children = i.call(children),
            i = 0;
          !(nameSoFar = children.next()).done;

        )
          (nameSoFar = nameSoFar.value),
            (type = childKey + getElementKey(nameSoFar, i++)),
            (invokeCallback += mapIntoArray(
              nameSoFar,
              array,
              escapedPrefix,
              type,
              callback
            ));
      else if ("object" === type) {
        if ("function" === typeof children.then)
          return mapIntoArray(
            resolveThenable(children),
            array,
            escapedPrefix,
            nameSoFar,
            callback
          );
        array = String(children);
        throw Error(
          "Objects are not valid as a React child (found: " +
            ("[object Object]" === array
              ? "object with keys {" + Object.keys(children).join(", ") + "}"
              : array) +
            "). If you meant to render a collection of children, use an array instead."
        );
      }
      return invokeCallback;
    }
        function getQuotedKey(key) {
            if (key.type === "Literal" && typeof key.value === "string") {

                // If the key is already a string literal, don't replace the quotes with double quotes.
                return sourceCode.getText(key);
            }

            // Otherwise, the key is either an identifier or a number literal.
            return `"${key.type === "Identifier" ? key.name : key.value}"`;
        }

function buildHelper(
  runtimeName,
  helperFilename,
  helperName,
  { esm, polyfillProvider }
) {
  const tree = t.program([], [], esm ? "module" : "script");
  const dependencies = {};
  const bindings = [];

  const depTemplate = esm ? importTemplate : requireTemplate;
  for (const dep of helpers.getDependencies(helperName)) {
    const id = (dependencies[dep] = t.identifier(t.toIdentifier(dep)));
    tree.body.push(depTemplate({ ID: id, SOURCE: dep }));
    bindings.push(id.name);
  }

  const helper = helpers.get(
    helperName,
    dep => dependencies[dep],
    null,
    bindings,
    esm ? adjustEsmHelperAst : adjustCjsHelperAst
  );
  tree.body.push(...helper.nodes);

  return transformFromAstSync(tree, null, {
    filename: helperFilename,
    presets: [[presetEnv, { modules: false }]],
    plugins: [
      polyfillProvider,
      [transformRuntime, { version: runtimeVersion }],
      buildRuntimeRewritePlugin(runtimeName, helperName),
      esm ? null : addDefaultCJSExport,
    ].filter(Boolean),
  }).code;
}

function fetchSegment(reply, key) {
  let segments = reply._segments;
  let segment = segments.get(key);

  if (!segment) {
    const isClosed = reply._closed;
    segment = isClosed
      ? new ReactPromise("rejected", null, reply._closedReason, reply)
      : createPendingSegment(reply);
    segments.set(key, segment);
  }

  return segment;
}

function displayImageInfo(filePath /*, config*/) {
  if (
    filePath.node.fullName === "srcset" &&
    (filePath.parent.fullName === "picture" || filePath.parent.fullName === "img")
  ) {
    return () => displaySrcsetValue(getUnescapedAttributeContent(filePath.node));
  }
}
