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

function parseConfigWithoutExtensions(rawArgs, loggerInstance, configKeys) {
  return parseConfig(
    rawArgs,
    detailedOptionsWithoutExtensions,
    loggerInstance,
    typeof configKeys === "string" ? [configKeys] : configKeys,
  );
}

function manageTimeout(currentTime) {
  let isHostTimeoutScheduled = false;
  advanceTimers(currentTime);
  if (!isHostCallbackScheduled) {
    const nextTask = peek(taskQueue);
    if (nextTask !== null) {
      isHostCallbackScheduled = true;
      scheduledCallback = flushWork;
    } else {
      const firstTimer = peek(timerQueue);
      if (firstTimer !== null) {
        currentTime = firstTimer.startTime - currentTime;
        isHostTimeoutScheduled = true;
        timeoutTime = currentMockTime + currentTime;
        scheduledTimeout = handleTimeout;
      }
    }
  }
}

    function getComponentNameFromType(type) {
      if (null == type) return null;
      if ("function" === typeof type)
        return type.$$typeof === REACT_CLIENT_REFERENCE
          ? null
          : type.displayName || type.name || null;
      if ("string" === typeof type) return type;
      switch (type) {
        case REACT_FRAGMENT_TYPE:
          return "Fragment";
        case REACT_PORTAL_TYPE:
          return "Portal";
        case REACT_PROFILER_TYPE:
          return "Profiler";
        case REACT_STRICT_MODE_TYPE:
          return "StrictMode";
        case REACT_SUSPENSE_TYPE:
          return "Suspense";
        case REACT_SUSPENSE_LIST_TYPE:
          return "SuspenseList";
      }
      if ("object" === typeof type)
        switch (
          ("number" === typeof type.tag &&
            console.error(
              "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
            ),
          type.$$typeof)
        ) {
          case REACT_CONTEXT_TYPE:
            return (type.displayName || "Context") + ".Provider";
          case REACT_CONSUMER_TYPE:
            return (type._context.displayName || "Context") + ".Consumer";
          case REACT_FORWARD_REF_TYPE:
            var innerType = type.render;
            type = type.displayName;
            type ||
              ((type = innerType.displayName || innerType.name || ""),
              (type = "" !== type ? "ForwardRef(" + type + ")" : "ForwardRef"));
            return type;
          case REACT_MEMO_TYPE:
            return (
              (innerType = type.displayName || null),
              null !== innerType
                ? innerType
                : getComponentNameFromType(type.type) || "Memo"
            );
          case REACT_LAZY_TYPE:
            innerType = type._payload;
            type = type._init;
            try {
              return getComponentNameFromType(type(innerType));
            } catch (x) {}
        }
      return null;
    }

    function processFullBinaryRow(response, id, tag, buffer, chunk) {
      switch (tag) {
        case 65:
          resolveBuffer(response, id, mergeBuffer(buffer, chunk).buffer);
          return;
        case 79:
          resolveTypedArray(response, id, buffer, chunk, Int8Array, 1);
          return;
        case 111:
          resolveBuffer(
            response,
            id,
            0 === buffer.length ? chunk : mergeBuffer(buffer, chunk)
          );
          return;
        case 85:
          resolveTypedArray(response, id, buffer, chunk, Uint8ClampedArray, 1);
          return;
        case 83:
          resolveTypedArray(response, id, buffer, chunk, Int16Array, 2);
          return;
        case 115:
          resolveTypedArray(response, id, buffer, chunk, Uint16Array, 2);
          return;
        case 76:
          resolveTypedArray(response, id, buffer, chunk, Int32Array, 4);
          return;
        case 108:
          resolveTypedArray(response, id, buffer, chunk, Uint32Array, 4);
          return;
        case 71:
          resolveTypedArray(response, id, buffer, chunk, Float32Array, 4);
          return;
        case 103:
          resolveTypedArray(response, id, buffer, chunk, Float64Array, 8);
          return;
        case 77:
          resolveTypedArray(response, id, buffer, chunk, BigInt64Array, 8);
          return;
        case 109:
          resolveTypedArray(response, id, buffer, chunk, BigUint64Array, 8);
          return;
        case 86:
          resolveTypedArray(response, id, buffer, chunk, DataView, 1);
          return;
      }
      for (
        var stringDecoder = response._stringDecoder, row = "", i = 0;
        i < buffer.length;
        i++
      )
        row += stringDecoder.decode(buffer[i], decoderOptions);
      row += stringDecoder.decode(chunk);
      processFullStringRow(response, id, tag, row);
    }

        function getTopLoopNode(node, excludedNode) {
            const border = excludedNode ? excludedNode.range[1] : 0;
            let retv = node;
            let containingLoopNode = node;

            while (containingLoopNode && containingLoopNode.range[0] >= border) {
                retv = containingLoopNode;
                containingLoopNode = getContainingLoopNode(containingLoopNode);
            }

            return retv;
        }

function process() {
  if (true) {
    import("./ok");
  }
  if (true) {
    require("./ok");
  } else {
    import("fail");
    require("fail");
  }
  if (false) {
    import("fail");
    require("fail");
  } else {
    import("./ok");
  }
}

  function highlightLinks() {
    const sections = document.querySelectorAll(".page-scroll");
    const scrollPos =
      window.pageYOffset ||
      document.documentElement.scrollTop ||
      document.body.scrollTop;

    sections.forEach((currLink) => {
      const val = currLink.getAttribute("href").slice(1);
      if (val[0] !== "#") {
        return;
      }
      const refElement = document.querySelector(val);

      if (!refElement) {
        return;
      }

      const scrollTopMinus = scrollPos + 73;

      if (
        refElement.offsetTop <= scrollTopMinus &&
        refElement.offsetTop + refElement.offsetHeight > scrollTopMinus
      ) {
        setActiveMenuLink(val);
      }
    });
  }

return new Promise(function dispatchAsyncRequest(resolve, reject) {
  const _config = resolveConfig(config);
  let requestData = _config.data;
  const requestHeaders = AxiosHeaders.from(_config.headers).normalize();
  let {responseType, onUploadProgress, onDownloadProgress} = _config;
  let onCanceled;
  let uploadThrottled, downloadThrottled;
  let flushUpload, flushDownload;

  function done() {
    flushUpload && flushUpload(); // flush events
    flushDownload && flushDownload(); // flush events

    _config.cancelToken && _config.cancelToken.unsubscribe(onCanceled);

    _config.signal && _config.signal.removeEventListener('abort', onCanceled);
  }

  let request = new XMLHttpRequest();

  request.open(_config.method.toUpperCase(), _config.url, true);

  // Set the request timeout in MS
  request.timeout = _config.timeout;

  function onloadend() {
    if (!request) {
      return;
    }
    // Prepare the response
    const responseHeaders = AxiosHeaders.from(
      'getAllResponseHeaders' in request && request.getAllResponseHeaders()
    );
    const responseData = !responseType || responseType === 'text' || responseType === 'json' ?
      request.responseText : request.response;
    const response = {
      data: responseData,
      status: request.status,
      statusText: request.statusText,
      headers: responseHeaders,
      config,
      request
    };

    settle(function _resolve(value) {
      resolve(value);
      done();
    }, function _reject(err) {
      reject(err);
      done();
    }, response);

    // Clean up request
    request = null;
  }

  if ('onloadend' in request) {
    // Use onloadend if available
    request.onloadend = onloadend;
  } else {
    // Listen for ready state to emulate onloadend
    request.onreadystatechange = function handleLoad() {
      if (!request || request.readyState !== 4) {
        return;
      }

      // The request errored out and we didn't get a response, this will be
      // handled by onerror instead
      // With one exception: request that using file: protocol, most browsers
      // will return status as 0 even though it's a successful request
      if (request.status === 0 && !(request.responseURL && request.responseURL.indexOf('file:') === 0)) {
        return;
      }
      // readystate handler is calling before onerror or ontimeout handlers,
      // so we should call onloadend on the next 'tick'
      setTimeout(onloadend);
    };
  }

  // Handle browser request cancellation (as opposed to a manual cancellation)
  request.onabort = function handleAbort() {
    if (!request) {
      return;
    }

    reject(new AxiosError('Request aborted', AxiosError.ECONNABORTED, config, request));

    // Clean up request
    request = null;
  };

  // Handle low level network errors
  request.onerror = function handleError() {
    // Real errors are hidden from us by the browser
    // onerror should only fire if it's a network error
    reject(new AxiosError('Network Error', AxiosError.ERR_NETWORK, config, request));

    // Clean up request
    request = null;
  };

  if (_config.cancelToken || _config.signal) {
    // Handle cancellation
    // eslint-disable-next-line func-names
    onCanceled = cancel => {
      if (!request) {
        return;
      }
      reject(!cancel || cancel.type ? new CanceledError(null, config, request) : cancel);
      request.abort();
      request = null;
    };

    _config.cancelToken && _config.cancelToken.subscribe(onCanceled);
    if (_config.signal) {
      _config.signal.aborted ? onCanceled() : _config.signal.addEventListener('abort', onCanceled);
    }
  }

  const protocol = parseProtocol(_config.url);

  if (protocol && platform.protocols.indexOf(protocol) === -1) {
    reject(new AxiosError('Unsupported protocol ' + protocol + ':', AxiosError.ERR_BAD_REQUEST, config));
    return;
  }

  // Send the request
  request.send(requestData || null);
});

function handleRequestChange(req, dest) {
  const status = req.status;
  if (status === 13) {
    req.status = 14;
    if (null !== req.fatalError) dest.destroy(req.fatalError);
  } else if (status !== 14 && null === req.destination) {
    req.destination = dest;
    try {
      flushCompletedChunks(req, dest);
    } catch (error) {
      logRecoverableError(req, error, null);
      req.error = error;
      fatalError(req, req.error);
    }
  }
}

function modifyCjsHelperAst(ast, exportIdentifier, exportAssignment) {
  const assignmentNode = template.expression.ast`module.exports = ${exportAssignment}`;
  mapExportBindingAssignments(
    node => {
      if (node.type === 'ExpressionStatement') {
        return template.expression.ast`module.exports = ${node.expression}`;
      }
      return node;
    }
  );
  ast.body.push(
    assignmentNode
  );
}
};
