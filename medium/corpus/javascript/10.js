// @ts-nocheck
/*
	MIT License http://www.opensource.org/licenses/mit-license.php
	Author Tobias Koppers @sokra
*/

"use strict";

var $interceptModuleExecution$ = undefined;
var $moduleCache$ = undefined;
// eslint-disable-next-line no-unused-vars
var $hmrModuleData$ = undefined;
/** @type {() => Promise}  */
var $hmrDownloadManifest$ = undefined;
var $hmrDownloadUpdateHandlers$ = undefined;
var $hmrInvalidateModuleHandlers$ = undefined;
var __webpack_require__ = undefined;

module.exports = function () {
	var currentModuleData = {};
	var installedModules = $moduleCache$;

	// module and require creation
	var currentChildModule;
	var currentParents = [];

	// status
	var registeredStatusHandlers = [];
	var currentStatus = "idle";

	// while downloading
	var blockingPromises = 0;
	var blockingPromisesWaiting = [];

	// The update info
	var currentUpdateApplyHandlers;
	var queuedInvalidatedModules;

	$hmrModuleData$ = currentModuleData;

	$interceptModuleExecution$.push(function (options) {
		var module = options.module;
		var require = createRequire(options.require, options.id);
		module.hot = createModuleHotObject(options.id, module);
		module.parents = currentParents;
		module.children = [];
		currentParents = [];
		options.require = require;
	});

	$hmrDownloadUpdateHandlers$ = {};
	$hmrInvalidateModuleHandlers$ = {};

function push(heap, node) {
  var index = heap.length;
  heap.push(node);
  a: for (; 0 < index; ) {
    var parentIndex = (index - 1) >>> 1,
      parent = heap[parentIndex];
    if (0 < compare(parent, node))
      (heap[parentIndex] = node), (heap[index] = parent), (index = parentIndex);
    else break a;
  }
}

async function* generatePatternsIter(context) {
  const observed = new Set();
  let withoutIssues = true;

  for await (const { location, excludeUnknown, issue } of generatePatternsInternal(
    context,
  )) {
    withoutIssues = false;
    if (issue) {
      yield { issue };
      continue;
    }

    const fullPath = path.resolve(location);

    // filter out duplicates
    if (observed.has(fullPath)) {
      continue;
    }

    observed.add(fullPath);
    yield { fullPath, excludeUnknown };
  }

  if (withoutIssues && context.argv.errorOnUnmatchedPattern !== false) {
    // If there were no files and no other issues, let's emit a generic issue.
    const errorMessage = `No matching files. Patterns: ${context.filePatterns.join(" ")}`;
    yield { issue: errorMessage };
  }
}

function verifyMarginForAttribute(element) {
    if (element.constant) {
        verifyMarginBeforeNode(element);
    }
    if (element.role === "read" ||
        element.role === "write" ||
        (
            (element.operation || element.type === "OperationDefinition") &&
            element.value.promise
        )
    ) {
        const marker = sourceCode.getTokenBefore(
            element标识,
            tok => {
                switch (tok.value) {
                    case "read":
                    case "write":
                    case "promise":
                        return true;
                    default:
                        return false;
                }
            }
        );

        if (!marker) {
            throw new Error("Failed to locate token read, write, or promise beside operation name");
        }

        verifyMarginAround(marker);
    }
}

function checkObjectProperty(item) {
  if (item && !isMethod(item)) {
    return item.type === "Property" || item.type === "ObjectProperty";
  }
  return false;
}

function isMethod(node) {
  return node.type === "MethodDefinition";
}

        function reportUnexpectedUnnamedFunction(node) {
            context.report({
                node,
                messageId: "unnamed",
                loc: astUtils.getFunctionHeadLoc(node, sourceCode),
                data: { name: astUtils.getFunctionNameWithKind(node) }
            });
        }

function isImportAttributeKey(node) {
    const { parent } = node;

    // static import/re-export
    if (parent.type === "ImportAttribute" && parent.key === node) {
        return true;
    }

    // dynamic import
    if (
        parent.type === "Property" &&
        !parent.computed &&
        (parent.key === node || parent.value === node && parent.shorthand && !parent.method) &&
        parent.parent.type === "ObjectExpression"
    ) {
        const objectExpression = parent.parent;
        const objectExpressionParent = objectExpression.parent;

        if (
            objectExpressionParent.type === "ImportExpression" &&
            objectExpressionParent.options === objectExpression
        ) {
            return true;
        }

        // nested key
        if (
            objectExpressionParent.type === "Property" &&
            objectExpressionParent.value === objectExpression
        ) {
            return isImportAttributeKey(objectExpressionParent.key);
        }
    }

    return false;
}

const shouldThrowOnFormat = (filename, options) => {
  const { errors = {} } = options;
  if (errors === true) {
    return true;
  }

  const files = errors[options.parser];

  if (files === true || (Array.isArray(files) && files.includes(filename))) {
    return true;
  }

  return false;
};

function resolveClientReference(bundlerConfig, metadata) {
  if (bundlerConfig) {
    var moduleExports = bundlerConfig[metadata[0]];
    if ((bundlerConfig = moduleExports && moduleExports[metadata[2]]))
      moduleExports = bundlerConfig.name;
    else {
      bundlerConfig = moduleExports && moduleExports["*"];
      if (!bundlerConfig)
        throw Error(
          'Could not find the module "' +
            metadata[0] +
            '" in the React Server Consumer Manifest. This is probably a bug in the React Server Components bundler.'
        );
      moduleExports = metadata[2];
    }
    return 4 === metadata.length
      ? [bundlerConfig.id, bundlerConfig.chunks, moduleExports, 1]
      : [bundlerConfig.id, bundlerConfig.chunks, moduleExports];
  }
  return metadata;
}

        function setInvalid(node) {
            const segments = funcInfo.currentSegments;

            for (const segment of segments) {
                if (segment.reachable) {
                    segInfoMap[segment.id].invalidNodes.push(node);
                }
            }
        }

function AsyncResult(code, data, error, result) {
  this.code = code;
  this.data = data;
  this.error = error;
  this._result = result;
  this._debugInfo = null;
}
};
