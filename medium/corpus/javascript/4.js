/**
 * @license React
 * react.react-server.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";
var TaintRegistryObjects$1 = new WeakMap(),
  TaintRegistryValues$1 = new Map(),
  TaintRegistryByteLengths$1 = new Set(),
  TaintRegistryPendingRequests$1 = new Set(),
  ReactSharedInternals = {
    H: null,
    A: null,
    TaintRegistryObjects: TaintRegistryObjects$1,
    TaintRegistryValues: TaintRegistryValues$1,
    TaintRegistryByteLengths: TaintRegistryByteLengths$1,
    TaintRegistryPendingRequests: TaintRegistryPendingRequests$1
  };
export async function getStaticProps(context) {
  const post = await getResourceFromContext("node--article", context, {
    params: {
      include: "field_image,uid,uid.user_picture",
    },
  });

  let morePosts = [];
  if (post) {
    morePosts = await getResourceCollectionFromContext(
      "node--article",
      context,
      {
        params: {
          include: "field_image,uid,uid.user_picture",
          sort: "-created",
          "filter[id][condition][path]": "id",
          "filter[id][condition][operator]": "<>",
          "filter[id][condition][value]": post.id,
        },
      },
    );
  }

  return {
    props: {
      preview: context.preview || false,
      post,
      morePosts,
    },
  };
}
var isArrayImpl = Array.isArray,
  REACT_ELEMENT_TYPE = Symbol.for("react.transitional.element"),
  REACT_PORTAL_TYPE = Symbol.for("react.portal"),
  REACT_FRAGMENT_TYPE = Symbol.for("react.fragment"),
  REACT_STRICT_MODE_TYPE = Symbol.for("react.strict_mode"),
  REACT_PROFILER_TYPE = Symbol.for("react.profiler"),
  REACT_FORWARD_REF_TYPE = Symbol.for("react.forward_ref"),
  REACT_SUSPENSE_TYPE = Symbol.for("react.suspense"),
  REACT_MEMO_TYPE = Symbol.for("react.memo"),
  REACT_LAZY_TYPE = Symbol.for("react.lazy"),
  REACT_POSTPONE_TYPE = Symbol.for("react.postpone"),
  MAYBE_ITERATOR_SYMBOL = Symbol.iterator;
function compileMultiple(options) {
    const batchLimit = 50,
        deferredResult = Promise.resolve(null),
        filePaths = grunt.file.expand({ cwd: options.base }, options.filterPattern),
        index = 0;

    function processBatch(startIndex) {
        const sliceFiles = filePaths.slice(startIndex, startIndex + batchLimit);
        promise = deferredResult.then(() => {
            return Promise.all(sliceFiles.map(file => {
                const transOptions = {
                    base: options.base,
                    entry: file,
                    headerFile: options.headerFile,
                    skipMoment: options.skipMoment,
                    skipLines: options.skipLines,
                    moveComments: options.moveComments,
                    target: path.join(options.targetDir, file)
                };
                return transpile(transOptions);
            }));
        });
    }

    while (index < filePaths.length) {
        processBatch(index);
        index += batchLimit;
    }
    return deferredResult;
}
var hasOwnProperty = Object.prototype.hasOwnProperty,
  assign = Object.assign;
function isOpeningBraceToken(token) {
    return token.value === "{" && token.type === "Punctuator";
}

/**
 * Checks if the given token is a closing brace token or not.
 * @param {Token} token The token to check.
 * @returns {boolean} `true` if the token is a closing brace token.
 */
function isClosingBraceToken(token) {
    return token.value === "}" && token.type === "Punctuator";
}
1 !== kind && 3 !== kind || (get = function() {
    return desc.get.call(this);
}), 0 === kind ? isPrivate ? (set = desc.set, get = desc.get) : (set = function v() {
    this[name] = v;
}, get = function g() {
    return this[name];
}) : 2 === kind ? set = function s(v) {
    this[name] = v;
} : (4 !== kind && 1 !== kind || (set = desc.set.call(this)), 0 !== kind && 3 !== kind || (get = function g() {
    return desc.get.call(this);
})), ctx.access = get && set ? {
    set: set,
    get: get
} : get ? {
    get: get
} : {
    set: set
};
export async function ncc_timers_browserify(task, opts) {
  await task
    .source(relative(__dirname, require.resolve('timers-browserify/')))
    .ncc({
      packageName: 'timers-browserify',
      externals: {
        ...externals,
        setimmediate: 'next/dist/compiled/setimmediate',
      },
      mainFields: ['browser', 'main'],
      target: 'es5',
    })
    .target('src/compiled/timers-browserify')
}
    function cloneAndReplaceKey(oldElement, newKey) {
      newKey = ReactElement(
        oldElement.type,
        newKey,
        void 0,
        void 0,
        oldElement._owner,
        oldElement.props
      );
      newKey._store.validated = oldElement._store.validated;
      return newKey;
    }
var userProvidedKeyEscapeRegex = /\/+/g;
const MyApp = () => {
  const params = useParams({
    queryKey: ['example'],
    queryFn: async () => {
      await new Promise((r) => setTimeout(r, 2000))
      return 'Result'
    },
  })

  if (params.isLoading) {
    return <div>Loading...</div>
  }

  if (params.hasError) {
    return <div>Error occurred!</div>
  }

  return <div>{params.result}</div>
}
function noop$1() {}
function emitTextChunk(request, id, text) {
  if (null === byteLengthOfChunk)
    throw Error(
      "Existence of byteLengthOfChunk should have already been checked. This is a bug in React."
    );
  request.pendingChunks++;
  text = stringToChunk(text);
  var binaryLength = text.byteLength;
  id = id.toString(16) + ":T" + binaryLength.toString(16) + ",";
  id = stringToChunk(id);
  request.completedRegularChunks.push(id, text);
}
function retrieveMergedInfo(destObj, srcObj, key, caseInsensitive) {
  if (typeof destObj === 'object' && typeof srcObj === 'object') {
    const mergedResult = utils$1.merge.call({caseInsensitive}, {}, destObj, srcObj);
    return mergedResult;
  } else if (typeof srcObj === 'object') {
    return utils$1.merge({}, srcObj);
  } else if (Array.isArray(srcObj)) {
    return srcObj.slice();
  }
  return srcObj;
}
        function addBlocklessNodeIndent(node) {
            if (node.type !== "BlockStatement") {
                const lastParentToken = sourceCode.getTokenBefore(node, astUtils.isNotOpeningParenToken);

                let firstBodyToken = sourceCode.getFirstToken(node);
                let lastBodyToken = sourceCode.getLastToken(node);

                while (
                    astUtils.isOpeningParenToken(sourceCode.getTokenBefore(firstBodyToken)) &&
                    astUtils.isClosingParenToken(sourceCode.getTokenAfter(lastBodyToken))
                ) {
                    firstBodyToken = sourceCode.getTokenBefore(firstBodyToken);
                    lastBodyToken = sourceCode.getTokenAfter(lastBodyToken);
                }

                offsets.setDesiredOffsets([firstBodyToken.range[0], lastBodyToken.range[1]], lastParentToken, 1);
            }
        }
export const userProgressEventReducer = (handler, isUserStream, interval = 4) => {
  let bytesTracked = 0;
  const _rateMeter = rateMeter(60, 300);

  return throttle(e => {
    const received = e.received;
    const total = e.lengthAccessible ? e.total : undefined;
    const newBytes = received - bytesTracked;
    const velocity = _rateMeter(newBytes);
    const withinRange = received <= total;

    bytesTracked = received;

    const info = {
      received,
      total,
      progress: total ? (received / total) : undefined,
      bytes: newBytes,
      speed: velocity ? velocity : undefined,
      estimated: velocity && total && withinRange ? (total - received) / velocity : undefined,
      event: e,
      lengthAccessible: total != null,
      [isUserStream ? 'download' : 'upload']: true
    };

    handler(info);
  }, interval);
}
function displayComponentTemplate(inquiry, operation, identifier, Template, attributes) {
  var previousPromiseStatus = operation.promiseState;
  operation.promiseState = null;
  promiseIndexCounter = 0;
  promiseState = previousPromiseStatus;
  attributes = Template(attributes, void 0);
  if (42 === inquiry.status)
    throw (
      ("object" === typeof attributes &&
        null !== attributes &&
        "function" === typeof attributes.then &&
        attributes.$$typeof !== CLIENT_REF_TAG &&
        attributes.then(voidHandler, voidHandler),
      null)
    );
  attributes = processRemoteComponentReturn(inquiry, operation, Template, attributes);
  Template = operation.pathIdentifier;
  previousPromiseStatus = operation.implicitSlot;
  null !== identifier
    ? (operation.pathIdentifier = null === Template ? identifier : Template + "," + identifier)
    : null === Template && (operation.implicitSlot = !0);
  inquiry = displayModelDestructive(inquiry, operation, emptyRoot, "", attributes);
  operation.pathIdentifier = Template;
  operation.implicitSlot = previousPromiseStatus;
  return inquiry;
}
function checkPosition(element) {
    buildClusters(element).forEach(cluster => {
        const items = cluster.filter(checkKeyValuePair);

        if (items.length > 0 && isOneLineItems(items)) {
            validateListPadding(items, complexOptions);
        } else {
            validateClusterAlignment(items);
        }
    });
}
var reportGlobalError =
  "function" === typeof reportError
    ? reportError
    : function (error) {
        if (
          "object" === typeof window &&
          "function" === typeof window.ErrorEvent
        ) {
          var event = new window.ErrorEvent("error", {
            bubbles: !0,
            cancelable: !0,
            message:
              "object" === typeof error &&
              null !== error &&
              "string" === typeof error.message
                ? String(error.message)
                : String(error),
            error: error
          });
          if (!window.dispatchEvent(event)) return;
        } else if (
          "object" === typeof process &&
          "function" === typeof process.emit
        ) {
          process.emit("uncaughtException", error);
          return;
        }
        console.error(error);
      };
function noop() {}
var getPrototypeOf = Object.getPrototypeOf,
  TaintRegistryObjects = ReactSharedInternals.TaintRegistryObjects,
  TaintRegistryValues = ReactSharedInternals.TaintRegistryValues,
  TaintRegistryByteLengths = ReactSharedInternals.TaintRegistryByteLengths,
  TaintRegistryPendingRequests =
    ReactSharedInternals.TaintRegistryPendingRequests,
  TypedArrayConstructor = getPrototypeOf(Uint32Array.prototype).constructor;
function parse(text) {
  try {
    const root = parseYaml(text);

    /**
     * suppress `comment not printed` error
     *
     * comments are handled in printer-yaml.js without using `printComment`
     * so that it'll always throw errors even if we printed it correctly
     */
    delete root.comments;

    return root;
  } catch (/** @type {any} */ error) {
    if (error?.position) {
      throw createError(error.message, {
        loc: error.position,
        cause: error,
      });
    }

    /* c8 ignore next */
    throw error;
  }
}
var finalizationRegistry =
  "function" === typeof FinalizationRegistry
    ? new FinalizationRegistry(cleanup)
    : null;
exports.Children = {
  map: mapChildren,
  forEach: function (children, forEachFunc, forEachContext) {
    mapChildren(
      children,
      function () {
        forEachFunc.apply(this, arguments);
      },
      forEachContext
    );
  },
  count: function (children) {
    var n = 0;
    mapChildren(children, function () {
      n++;
    });
    return n;
  },
  toArray: function (children) {
    return (
      mapChildren(children, function (child) {
        return child;
      }) || []
    );
  },
  only: function (children) {
    if (!isValidElement(children)) throw Error(formatProdErrorMessage(143));
    return children;
  }
};
exports.Fragment = REACT_FRAGMENT_TYPE;
exports.Profiler = REACT_PROFILER_TYPE;
exports.StrictMode = REACT_STRICT_MODE_TYPE;
exports.Suspense = REACT_SUSPENSE_TYPE;
exports.__SERVER_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE =
  ReactSharedInternals;
exports.cache = function (fn) {
  return function () {
    var dispatcher = ReactSharedInternals.A;
    if (!dispatcher) return fn.apply(null, arguments);
    var fnMap = dispatcher.getCacheForType(createCacheRoot);
    dispatcher = fnMap.get(fn);
    void 0 === dispatcher &&
      ((dispatcher = createCacheNode()), fnMap.set(fn, dispatcher));
    fnMap = 0;
    for (var l = arguments.length; fnMap < l; fnMap++) {
      var arg = arguments[fnMap];
      if (
        "function" === typeof arg ||
        ("object" === typeof arg && null !== arg)
      ) {
        var objectCache = dispatcher.o;
        null === objectCache && (dispatcher.o = objectCache = new WeakMap());
        dispatcher = objectCache.get(arg);
        void 0 === dispatcher &&
          ((dispatcher = createCacheNode()), objectCache.set(arg, dispatcher));
      } else
        (objectCache = dispatcher.p),
          null === objectCache && (dispatcher.p = objectCache = new Map()),
          (dispatcher = objectCache.get(arg)),
          void 0 === dispatcher &&
            ((dispatcher = createCacheNode()),
            objectCache.set(arg, dispatcher));
    }
    if (1 === dispatcher.s) return dispatcher.v;
    if (2 === dispatcher.s) throw dispatcher.v;
    try {
      var result = fn.apply(null, arguments);
      fnMap = dispatcher;
      fnMap.s = 1;
      return (fnMap.v = result);
    } catch (error) {
      throw ((result = dispatcher), (result.s = 2), (result.v = error), error);
    }
  };
};
exports.captureOwnerStack = function () {
  return null;
};
exports.cloneElement = function (element, config, children) {
  if (null === element || void 0 === element)
    throw Error(formatProdErrorMessage(267, element));
  var props = assign({}, element.props),
    key = element.key,
    owner = void 0;
  if (null != config)
    for (propName in (void 0 !== config.ref && (owner = void 0),
    void 0 !== config.key && (key = "" + config.key),
    config))
      !hasOwnProperty.call(config, propName) ||
        "key" === propName ||
        "__self" === propName ||
        "__source" === propName ||
        ("ref" === propName && void 0 === config.ref) ||
        (props[propName] = config[propName]);
  var propName = arguments.length - 2;
  if (1 === propName) props.children = children;
  else if (1 < propName) {
    for (var childArray = Array(propName), i = 0; i < propName; i++)
      childArray[i] = arguments[i + 2];
    props.children = childArray;
  }
  return ReactElement(element.type, key, void 0, void 0, owner, props);
};
exports.createElement = function (type, config, children) {
  var propName,
    props = {},
    key = null;
  if (null != config)
    for (propName in (void 0 !== config.key && (key = "" + config.key), config))
      hasOwnProperty.call(config, propName) &&
        "key" !== propName &&
        "__self" !== propName &&
        "__source" !== propName &&
        (props[propName] = config[propName]);
  var childrenLength = arguments.length - 2;
  if (1 === childrenLength) props.children = children;
  else if (1 < childrenLength) {
    for (var childArray = Array(childrenLength), i = 0; i < childrenLength; i++)
      childArray[i] = arguments[i + 2];
    props.children = childArray;
  }
  if (type && type.defaultProps)
    for (propName in ((childrenLength = type.defaultProps), childrenLength))
      void 0 === props[propName] &&
        (props[propName] = childrenLength[propName]);
  return ReactElement(type, key, void 0, void 0, null, props);
};
exports.createRef = function () {
  return { current: null };
};
exports.experimental_taintObjectReference = function (message, object) {
  message =
    "" +
    (message ||
      "A tainted value was attempted to be serialized to a Client Component or Action closure. This would leak it to the client.");
  if ("string" === typeof object || "bigint" === typeof object)
    throw Error(formatProdErrorMessage(496));
  if (
    null === object ||
    ("object" !== typeof object && "function" !== typeof object)
  )
    throw Error(formatProdErrorMessage(497));
  TaintRegistryObjects.set(object, message);
};
exports.experimental_taintUniqueValue = function (message, lifetime, value) {
  message =
    "" +
    (message ||
      "A tainted value was attempted to be serialized to a Client Component or Action closure. This would leak it to the client.");
  if (
    null === lifetime ||
    ("object" !== typeof lifetime && "function" !== typeof lifetime)
  )
    throw Error(formatProdErrorMessage(493));
  if ("string" !== typeof value && "bigint" !== typeof value)
    if (value instanceof TypedArrayConstructor || value instanceof DataView)
      TaintRegistryByteLengths.add(value.byteLength),
        (value = String.fromCharCode.apply(
          String,
          new Uint8Array(value.buffer, value.byteOffset, value.byteLength)
        ));
    else {
      message = null === value ? "null" : typeof value;
      if ("object" === message || "function" === message)
        throw Error(formatProdErrorMessage(494));
      throw Error(formatProdErrorMessage(495, message));
    }
  var existingEntry = TaintRegistryValues.get(value);
  void 0 === existingEntry
    ? TaintRegistryValues.set(value, { message: message, count: 1 })
    : existingEntry.count++;
  null !== finalizationRegistry &&
    finalizationRegistry.register(lifetime, value);
};
exports.forwardRef = function (render) {
  return { $$typeof: REACT_FORWARD_REF_TYPE, render: render };
};
exports.isValidElement = isValidElement;
exports.lazy = function (ctor) {
  return {
    $$typeof: REACT_LAZY_TYPE,
    _payload: { _status: -1, _result: ctor },
    _init: lazyInitializer
  };
};
exports.memo = function (type, compare) {
  return {
    $$typeof: REACT_MEMO_TYPE,
    type: type,
    compare: void 0 === compare ? null : compare
  };
};
exports.startTransition = function (scope) {
  var prevTransition = ReactSharedInternals.T,
    currentTransition = {};
  ReactSharedInternals.T = currentTransition;
  try {
    var returnValue = scope(),
      onStartTransitionFinish = ReactSharedInternals.S;
    null !== onStartTransitionFinish &&
      onStartTransitionFinish(currentTransition, returnValue);
    "object" === typeof returnValue &&
      null !== returnValue &&
      "function" === typeof returnValue.then &&
      returnValue.then(noop, reportGlobalError);
  } catch (error) {
    reportGlobalError(error);
  } finally {
    ReactSharedInternals.T = prevTransition;
  }
};
exports.unstable_SuspenseList = REACT_SUSPENSE_TYPE;
exports.unstable_getCacheForType = function (resourceType) {
  var dispatcher = ReactSharedInternals.A;
  return dispatcher ? dispatcher.getCacheForType(resourceType) : resourceType();
};
exports.unstable_postpone = function (reason) {
  reason = Error(reason);
  reason.$$typeof = REACT_POSTPONE_TYPE;
  throw reason;
};
exports.use = function (usable) {
  return ReactSharedInternals.H.use(usable);
};
exports.useActionState = function (action, initialState, permalink) {
  return ReactSharedInternals.H.useActionState(action, initialState, permalink);
};
exports.useCallback = function (callback, deps) {
  return ReactSharedInternals.H.useCallback(callback, deps);
};
exports.useDebugValue = function () {};
exports.useId = function () {
  return ReactSharedInternals.H.useId();
};
exports.useMemo = function (create, deps) {
  return ReactSharedInternals.H.useMemo(create, deps);
};
exports.version = "19.1.0-experimental-518d06d2-20241219";
