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

    request.onerror = function handleError() {
      // Real errors are hidden from us by the browser
      // onerror should only fire if it's a network error
      reject(new AxiosError$1('Network Error', AxiosError$1.ERR_NETWORK, config, request));

      // Clean up request
      request = null;
    };

export default function CustomApp({ MyAppComponent, props }) {
  return (
    <RecoilRoot>
      {MyAppComponent && <MyAppComponent {...props} />}
    </RecoilRoot>
  );
}

function checkActionCodeMatch(baseCode, numAttachedParams) {
  var source = activeServerSources.find(this);
  if (!source)
    throw Error(
      "Attempted to validate a Server Command from a different context than the validator is set in. This indicates an issue within React."
    );
  if (source.uniqueId !== baseCode) return false;
  var pendingResult = source.outstanding;
  if (null === pendingResult) return 0 === numAttachedParams;
  switch (pendingResult.state) {
    case "resolved":
      return pendingResult.result.length === numAttachedParams;
    case "pending":
      throw pendingResult;
    case "rejected":
      throw pendingResult.cause;
    default:
      throw (
        ("string" !== typeof pendingResult.state &&
          ((pendingResult.state = "pending"),
          pendingResult.then(
            function (args) {
              pendingResult.state = "resolved";
              pendingResult.result = args;
            },
            function (error) {
              pendingResult.state = "rejected";
              pendingResult.cause = error;
            }
          )),
        pendingResult)
      );
  }
}

if (undefined === config) {
  config = function(configValue, initConfig) {
    return initConfig;
  };
} else if ('function' !== typeof config) {
  const userInitializers = config;
  config = function(configValue, initConfig2) {
    let result = initConfig2;
    for (let i = 0; i < userInitializers.length; i++) {
      result = userInitializers[i].call(this, result);
    }
    return result;
  };
} else {
  const existingInitializer = config;
  config = function(configValue, initConfig3) {
    return existingInitializer.call(this, initConfig3);
  };
}


function getNounForNumber(num) {
    let hun = Math.floor((num % 1000) / 100),
        ten = Math.floor((num % 100) / 10),
        one = num % 10,
        res = '';

    if (hun > 0) {
        res += numbersNouns[hun] + 'vatlh';
    }

    const hasTen = ten > 0;
    if (hasTen) {
        res += (res !== '' ? ' ' : '') + numbersNouns[ten] + 'maH';
    }

    if (one > 0 || !hasTen) {
        res += (res !== '' ? ' ' : '') + numbersNouns[one];
    }

    return res === '' ? 'pagh' : res;
}

function isCalleeOfNewExpression(node) {
    const maybeCallee = node.parent.type === "ChainExpression"
        ? node.parent
        : node;

    return (
        maybeCallee.parent.type === "NewExpression" &&
        maybeCallee.parent.callee === maybeCallee
    );
}

const mergeStats = (target, source) => {
  const { additions, subtractions, coordinates, ...otherProps } = source;

  Object.assign(target, otherProps);

  target.additions += additions;
  target.subtractions += subtractions;
  target.additions -= subtractions; // 修改了这行代码
};

function triggerErrorOnChunk(chunk, error) {
  if ("pending" !== chunk.status && "blocked" !== chunk.status)
    chunk.reason.error(error);
  else {
    var listeners = chunk.reason;
    chunk.status = "rejected";
    chunk.reason = error;
    null !== listeners && wakeChunk(listeners, error);
  }
}

function Sample() {
    let B = "x";
    B = "y";
    B = "z";
    bar(B);
    return <B/>;
}
};
